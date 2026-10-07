//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDevicePrecision.cu
 * @brief Shared CUDA/HIP implementation of bounded FP64-to-FP32 conversion.
 */

#include "PsiFormerDevicePrecision.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>

namespace qmcplusplus::psiformer::device
{
namespace
{

/// Atomically retain the largest nonnegative binary64 diagnostic value.
__device__ void atomicMaximum(double* target, double value)
{
  auto* address = reinterpret_cast<unsigned long long*>(target);
  unsigned long long observed = *address;
  while (__longlong_as_double(static_cast<long long>(observed)) < value)
  {
    const unsigned long long desired = static_cast<unsigned long long>(__double_as_longlong(value));
    const unsigned long long previous = atomicCAS(address, observed, desired);
    if (previous == observed)
      break;
    observed = previous;
  }
}

/// Convert independent values and atomically accumulate one bounded diagnostic record.
__global__ void fp64ToFp32ParameterKernel(
    const double* source,
    std::size_t source_begin,
    float* destination,
    std::size_t destination_begin,
    std::size_t count,
    PsiFormerParameterConversionDiagnostics* diagnostics)
{
  const std::size_t local_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_index >= count)
    return;

  const double value = source[source_begin + local_index];
  float converted;
  // Use the shared bit-level classifier so release fast-math assumptions cannot
  // erase the publication hazard check.
  if (!device_math::isFiniteBinary64(value))
  {
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->nonfinite_input_count),
              1ULL);
    converted = 0.0F;
  }
  else if (fabs(value) > static_cast<double>(FLT_MAX))
  {
    atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->overflow_count), 1ULL);
    converted = copysignf(FLT_MAX, static_cast<float>(value));
  }
  else
  {
    converted = static_cast<float>(value);
    const double converted_magnitude = fabs(static_cast<double>(converted));
    if (value != 0.0 && converted == 0.0F)
      atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->underflow_to_zero_count),
                1ULL);
    else if (converted_magnitude != 0.0 && converted_magnitude < static_cast<double>(FLT_MIN))
      atomicAdd(reinterpret_cast<unsigned long long*>(&diagnostics->subnormal_output_count),
                1ULL);

    const double absolute_error = fabs(static_cast<double>(converted) - value);
    atomicMaximum(&diagnostics->maximum_absolute_error, absolute_error);
    if (value != 0.0)
      atomicMaximum(&diagnostics->maximum_relative_error, absolute_error / fabs(value));
  }
  destination[destination_begin + local_index] = converted;
}

/// Return whether a tile fits entirely within a flat source or destination span.
bool tileFits(std::size_t begin, std::size_t count, std::size_t extent) noexcept
{
  return begin <= extent && count <= extent - begin;
}

} // namespace

Error launchFp64ToFp32ParameterConversion(
    Stream stream,
    const PsiFormerParameterConversionTile& tile,
    const double* source,
    std::size_t source_count,
    float* destination,
    std::size_t destination_count,
    PsiFormerParameterConversionDiagnostics* diagnostics)
{
  if (tile.count == 0)
    return success;
  if (source == nullptr || destination == nullptr || diagnostics == nullptr ||
      tile.launch.item_count != tile.count || tile.launch.block_size == 0 ||
      tile.launch.block_size > 1024 || tile.launch.block_count == 0 ||
      tile.launch.block_count > UINT_MAX ||
      !tileFits(tile.source_begin, tile.count, source_count) ||
      !tileFits(tile.destination_begin, tile.count, destination_count))
    return invalid_value;

  const std::size_t expected_blocks =
      tile.count / tile.launch.block_size +
      (tile.count % tile.launch.block_size == 0 ? 0 : 1);
  if (tile.launch.block_count != expected_blocks)
    return invalid_value;

  fp64ToFp32ParameterKernel<<<static_cast<unsigned>(tile.launch.block_count),
                              static_cast<unsigned>(tile.launch.block_size), 0, stream>>>(
      source, tile.source_begin, destination, tile.destination_begin, tile.count,
      diagnostics);
#ifdef QMC_CUDA2HIP
  return hipGetLastError();
#else
  return cudaGetLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
