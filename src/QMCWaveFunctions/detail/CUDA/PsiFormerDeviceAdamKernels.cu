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
/// Reduce one completed 128-lane block without squaring large finite values.
__device__ void stablePartial(const double* values, double* partials)
{
  double scale = 0, sum = 1;
  for (unsigned lane = 0; lane < 128; ++lane)
  {
    const double value = fabs(values[lane]);
    if (value != 0)
    {
      if (scale < value)
      {
        const double ratio = scale / value;
        sum = 1 + sum * ratio * ratio;
        scale = value;
      }
      else
      {
        const double ratio = value / scale;
        sum += ratio * ratio;
      }
    }
  }
  partials[2 * blockIdx.x] = scale;
  partials[2 * blockIdx.x + 1] = sum;
}

/// One bounded partial per block avoids squaring large finite gradients directly.
__global__ void energyGradient(std::size_t count, const double* channels, const double* parameters,
    const double* image_parameters, double weight, double energy, double* gradient,
    double* partials, std::uint64_t* status)
{
  __shared__ double values[128];
  const auto index = std::size_t(blockIdx.x) * 128 + threadIdx.x;
  double g = 0;
  if (index < count)
  {
    g = 2.0 * (channels[2 * index] + channels[2 * count + 2 * index] -
        energy * channels[4 * count + 2 * index]) / weight;
    gradient[index] = g;
    if (!device_math::isFiniteBinary64(g) || !device_math::isFiniteBinary64(parameters[index]) ||
        parameters[index] != image_parameters[index])
      atomicMin(reinterpret_cast<unsigned long long*>(status),
                static_cast<unsigned long long>((std::uint64_t(index) << 8) | 3));
  }
  values[threadIdx.x] = g;
  __syncthreads();
  if (threadIdx.x == 0)
    stablePartial(values, partials);
}

/// Pending-only recurrence: publication is a later noexcept host pointer swap.
__global__ void candidate(std::size_t count, const AdamBlock* blocks, std::size_t block_count,
    const double* parameters, double* gradient_candidate, const double* first, const double* second,
    double* next_first, double* next_second, double beta1, double beta2, double correction1,
    double correction2, double epsilon, std::uint64_t* status, double* displacement_partials)
{
  __shared__ double displacements[128];
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  double displacement = 0;
  if (index < count)
  {
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
    displacement = result - parameters[index];
  }
  if (displacement_partials)
  {
    displacements[threadIdx.x] = displacement;
    __syncthreads();
    if (threadIdx.x == 0)
      stablePartial(displacements, displacement_partials);
  }
}
} // namespace

/// Extents are admitted by the owning Adam preparation before this private seam.
Error launchAdamEnergyGradient(Stream stream, std::size_t count, const double* channels,
    const double* parameters, const double* image_parameters, double weight, double energy,
    double* gradient, double* norm_partials, std::uint64_t* status)
{
  energyGradient<<<static_cast<unsigned>((count - 1) / 128 + 1), 128, 0, stream>>>(
      count, channels, parameters, image_parameters, weight, energy, gradient, norm_partials, status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

/// Launch only admitted extents; host status admission precedes candidate exposure.
Error launchAdamCandidate(Stream stream, std::size_t count, const AdamBlock* blocks,
    std::size_t block_count, const double* parameters, double* gradient_candidate,
    const double* first, const double* second, double* next_first, double* next_second,
    double beta1, double beta2, double correction1, double correction2, double epsilon,
    std::uint64_t* status, double* displacement_partials)
{
  if (!count || !block_count || (count - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()) ||
      !blocks || !parameters || !gradient_candidate || !first || !second || !next_first || !next_second || !status)
    throw std::invalid_argument("Device Adam launch extent/storage is invalid");
  candidate<<<static_cast<unsigned>((count - 1) / 128 + 1), 128, 0, stream>>>(count, blocks, block_count,
      parameters, gradient_candidate, first, second, next_first, next_second,
      beta1, beta2, correction1, correction2, epsilon, status, displacement_partials);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace qmcplusplus::psiformer::device
