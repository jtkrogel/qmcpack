//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ADAM_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ADAM_KERNELS_H
#include "PsiFormerDeviceRuntime.h"
#include <cstddef>
#include <cstdint>

namespace qmcplusplus::psiformer::device
{
/// Form all ENERGY gradients and scaled-sum-of-squares partials, including frozen lanes.
Error launchAdamEnergyGradient(Stream stream, std::size_t count, const double* channels,
    const double* parameters, const double* image_parameters, double weight, double energy,
    double* gradient, double* norm_partials, std::uint64_t* status);

/// Physical score covariance from ordered EO/O sums, with no response channel.
Error launchAdamScoreCovarianceGradient(Stream stream, std::size_t count, const double* channels,
    const double* parameters, const double* image_parameters, double weight, double energy,
    double* gradient, double* norm_partials, std::uint64_t* status);

/// Compact immutable schema blocks; no per-parameter rate/trainability tensor.
struct AdamBlock
{
  std::uint64_t begin, count;
  double learning_rate;
  std::uint64_t trainable;
};
static_assert(sizeof(AdamBlock) == 32);

/// Form pending moments and overwrite gradient scratch with a raw Adam candidate.
Error launchAdamCandidate(Stream stream, std::size_t count, const AdamBlock* blocks,
    std::size_t block_count, const double* parameters, double* gradient_candidate,
    const double* first, const double* second, double* next_first, double* next_second,
    double beta1, double beta2, double correction1, double correction2, double epsilon,
    std::uint64_t* status, double* displacement_partials = nullptr);
} // namespace qmcplusplus::psiformer::device
#endif
