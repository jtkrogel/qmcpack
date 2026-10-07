//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceKernels.h
 * @brief CUDA/HIP launch wrappers for the first PsiFormer device math kernels.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_KERNELS_H

#include "PsiFormerDeviceRuntime.h"

#include <cstddef>

namespace qmcplusplus::psiformer::device
{

Error launchPairFeatures(Stream stream,
                         const double* displacements,
                         const double* complementary_displacements,
                         const double* radii,
                         std::size_t pair_count,
                         bool periodic,
                         double* features);

Error launchTanhJets(Stream stream,
                     const double* values,
                     const double* first,
                     const double* second,
                     std::size_t element_count,
                     double* output_values,
                     double* output_first,
                     double* output_second);

Error launchResidualJets(Stream stream,
                         const double* left_values,
                         const double* left_first,
                         const double* left_second,
                         const double* right_values,
                         const double* right_first,
                         const double* right_second,
                         std::size_t element_count,
                         double* output_values,
                         double* output_first,
                         double* output_second);

Error launchEnvelopeContributions(Stream stream,
                                  const double* distances,
                                  const double* pi,
                                  const double* zeta,
                                  std::size_t element_count,
                                  double* output);

Error launchCuspPairs(Stream stream,
                      const double* distances,
                      const double* alpha,
                      const double* cusp_factors,
                      std::size_t pair_count,
                      double* output);

/** Compile-time foundation reduction. Runtime tuning and acceptance are deferred. */
Error launchSerialSum(Stream stream, const double* input, std::size_t element_count, double* output);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_KERNELS_H
