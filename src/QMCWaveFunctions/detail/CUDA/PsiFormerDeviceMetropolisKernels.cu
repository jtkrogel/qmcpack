//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceMetropolisKernels.h"
#include "PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include <cmath>

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Preserve the first failing logical index without writing accepted state.
__device__ void fail(std::uint64_t* status, std::size_t index, unsigned code)
{
  atomicMin(reinterpret_cast<unsigned long long*>(status),
            static_cast<unsigned long long>((std::uint64_t(index) << 8) | code));
}

/// Apply the drift-free species-mass diffusion law to one packed coordinate.
__global__ void propose(std::size_t count, std::size_t ne, const double* accepted, const double* gaussian,
    const double* inverse_mass, double tau, double* proposed, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const double variance = tau * inverse_mass[(index / 3) % ne];
  const double result = accepted[index] + sqrt(variance) * gaussian[index];
  proposed[index] = result;
  if (!device_math::isFiniteBinary64(variance) || variance <= 0 || !device_math::isFiniteBinary64(result))
    fail(status, index, 1);
}

/// Treat exact nodes per walker; other invalid VALUE outputs are fatal admission errors.
__global__ void stage(std::size_t count, std::size_t offset, bool seed, const double* phase,
    const double* logabs, const double* accepted_log, double* staged_phase, double* staged_log,
    double* log_acceptance, std::uint32_t* mask, std::uint64_t* status)
{
  const auto local = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= count)
    return;
  const auto walker = offset + local;
  const double sign = phase[local], value = logabs[local];
  staged_phase[walker] = sign;
  staged_log[walker] = value;
  mask[walker] = 0;
  log_acceptance[walker] = -INFINITY;
  // Inspect the canonical node representation even under finite-math flags.
  union Binary64
  {
    double floating;
    std::uint64_t integer;
  } encoded{value};
  if (sign == 0 && encoded.integer == UINT64_C(0xfff0000000000000))
  {
    if (seed)
      fail(status, walker, 2);
    return;
  }
  if ((sign != 1 && sign != -1) || !device_math::isFiniteBinary64(value))
  {
    fail(status, walker, 3);
    return;
  }
  if (!seed)
  {
    const double acceptance = 2 * (value - accepted_log[walker]);
    log_acceptance[walker] = acceptance;
    // Native collective VMC consumes uniforms only for finite log acceptance.
    if (device_math::isFiniteBinary64(acceptance))
      mask[walker] = METROPOLIS_ELIGIBLE;
  }
}

/// Complete next accepted state entirely in staging before the owner's pointer swap.
__global__ void resolve(std::size_t population, std::size_t ne, const double* accepted,
    double* proposed, const double* accepted_phase, const double* accepted_log,
    double* staged_phase, double* staged_log, const double* uniforms, const double* log_acceptance,
    std::uint32_t* mask, std::uint64_t* status)
{
  const auto walker = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (walker >= population)
    return;
  bool take = false;
  if (mask[walker] & METROPOLIS_ELIGIBLE)
  {
    const double u = uniforms[walker];
    if (!device_math::isFiniteBinary64(u) || u < 0 || u >= 1)
    {
      fail(status, walker, 4);
      return;
    }
    // Zero is a valid native uniform. Avoid dependence on device log(0) handling.
    take = u == 0 || metropolisLogComparison(log(u), log_acceptance[walker]);
  }
  mask[walker] = take ? METROPOLIS_ACCEPTED : 0;
  if (!take)
  {
    for (std::size_t coordinate = 0; coordinate < 3 * ne; ++coordinate)
      proposed[walker * 3 * ne + coordinate] = accepted[walker * 3 * ne + coordinate];
    staged_phase[walker] = accepted_phase[walker];
    staged_log[walker] = accepted_log[walker];
  }
}

/// Validate launch extents independently of the owner's checked byte admission.
unsigned blocks(std::size_t count)
{
  if (!count || (count - 1) / 128 >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("Device Metropolis launch extent is invalid");
  return static_cast<unsigned>((count - 1) / 128 + 1);
}

/// Return the current vendor's non-synchronizing launch status.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Launch checked coordinate staging, never an accepted-state update.
Error launchMetropolisProposal(Stream stream, std::size_t population, std::size_t electrons,
    const double* accepted, const double* gaussian, const double* inverse_mass, double tau,
    double* proposed, std::uint64_t* status)
{
  if (!population || !electrons || population > std::numeric_limits<std::size_t>::max() / 3 / electrons ||
      !accepted || !gaussian || !inverse_mass || !proposed || !status)
    throw std::invalid_argument("Metropolis proposal has invalid extent or device storage");
  const auto count = population * electrons * 3;
  propose<<<blocks(count), 128, 0, stream>>>(count, electrons, accepted, gaussian, inverse_mass, tau, proposed, status);
  return launched();
}

/// Launch immediate VALUE consumption with explicit full-population destination offset.
Error launchMetropolisStageValues(Stream stream, std::size_t count, std::size_t offset, bool seed,
    const double* phase, const double* logabs, const double* accepted_log,
    double* staged_phase, double* staged_log, double* log_acceptance, std::uint32_t* mask, std::uint64_t* status)
{
  if (offset > std::numeric_limits<std::size_t>::max() - count || !phase || !logabs || !accepted_log ||
      !staged_phase || !staged_log || !log_acceptance || !mask || !status)
    throw std::invalid_argument("Metropolis VALUE staging has invalid extent or device storage");
  stage<<<blocks(count), 128, 0, stream>>>(count, offset, seed, phase, logabs, accepted_log,
      staged_phase, staged_log, log_acceptance, mask, status);
  return launched();
}

/// Launch transactional rejection merge into the already allocated proposal source.
Error launchMetropolisResolve(Stream stream, std::size_t population, std::size_t electrons,
    const double* accepted, double* proposed, const double* accepted_phase, const double* accepted_log,
    double* staged_phase, double* staged_log, const double* uniforms, const double* log_acceptance,
    std::uint32_t* mask, std::uint64_t* status)
{
  if (!population || !electrons || population > std::numeric_limits<std::size_t>::max() / 3 / electrons ||
      !accepted || !proposed || !accepted_phase || !accepted_log || !staged_phase || !staged_log ||
      !uniforms || !log_acceptance || !mask || !status)
    throw std::invalid_argument("Metropolis resolution has invalid extent or device storage");
  resolve<<<blocks(population), 128, 0, stream>>>(population, electrons, accepted, proposed,
      accepted_phase, accepted_log, staged_phase, staged_log, uniforms, log_acceptance, mask, status);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
