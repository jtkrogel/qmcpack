//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceAdamKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Pending-only recurrence: publication is a later noexcept host pointer swap.
__global__ void candidate(std::size_t count, const AdamBlock* blocks, std::size_t block_count,
    const double* parameters, double* gradient_candidate, const double* first, const double* second,
    double* next_first, double* next_second, double beta1, double beta2, double correction1,
    double correction2, double epsilon, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  // Locate this lane's compact schema block without a per-parameter rate tensor.
  std::size_t low = 0, high = block_count;
  while (low + 1 < high)
  {
    const auto middle = low + (high - low) / 2;
    if (blocks[middle].begin <= index)
      low = middle;
    else
      high = middle;
  }
  const auto block = blocks[low];
  // Calculate only speculative state; frozen lanes preserve parameters/zero moments.
  double m = 0, v = 0, result = parameters[index];
  if (block.trainable)
  {
    const double g = gradient_candidate[index];
    m = beta1 * first[index] + (1.0 - beta1) * g;
    v = beta2 * second[index] + (1.0 - beta2) * g * g;
    const double direction = (m / correction1) / (sqrt(v / correction2) + epsilon);
    result -= block.learning_rate * direction;
    if (!device_math::isFiniteBinary64(direction))
      atomicMin(reinterpret_cast<unsigned long long*>(status),
                static_cast<unsigned long long>((std::uint64_t(index) << 8) | 1));
  }
  // Publish scratch, then record first finite/status failure before host admission.
  next_first[index] = m;
  next_second[index] = v;
  gradient_candidate[index] = result;
  if (!device_math::isFiniteBinary64(m) || !device_math::isFiniteBinary64(v) ||
      !device_math::isFiniteBinary64(result) || v < 0)
    atomicMin(reinterpret_cast<unsigned long long*>(status),
              static_cast<unsigned long long>((std::uint64_t(index) << 8) | 2));
}
} // namespace

/// Launch only admitted extents; host status admission precedes candidate exposure.
Error launchAdamCandidate(Stream stream, std::size_t count, const AdamBlock* blocks,
    std::size_t block_count, const double* parameters, double* gradient_candidate,
    const double* first, const double* second, double* next_first, double* next_second,
    double beta1, double beta2, double correction1, double correction2, double epsilon,
    std::uint64_t* status)
{
  if (!count || !block_count || (count - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()) ||
      !blocks || !parameters || !gradient_candidate || !first || !second || !next_first || !next_second || !status)
    throw std::invalid_argument("Device Adam launch extent/storage is invalid");
  candidate<<<static_cast<unsigned>((count - 1) / 128 + 1), 128, 0, stream>>>(count, blocks, block_count,
      parameters, gradient_candidate, first, second, next_first, next_second,
      beta1, beta2, correction1, correction2, epsilon, status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace qmcplusplus::psiformer::device
