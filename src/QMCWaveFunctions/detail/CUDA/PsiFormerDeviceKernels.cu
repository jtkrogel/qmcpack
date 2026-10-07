//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceKernels.cu
 * @brief Shared CUDA/HIP wrappers for PsiFormer feature and elementwise kernels.
 */

#include "PsiFormerDeviceKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int block_size = 128;

__global__ void pairFeaturesKernel(const double* displacements,
                                   const double* complementary_displacements,
                                   const double* radii,
                                   std::size_t pair_count,
                                   bool periodic,
                                   double* features)
{
  const std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count)
    return;
  const std::size_t width = periodic ? 7 : 4;
  device_math::assemblePairFeatures(displacements + 3 * pair,
                                    periodic ? complementary_displacements + 3 * pair : nullptr,
                                    radii[pair], periodic, features + width * pair);
}

__global__ void tanhJetsKernel(const double* values,
                               const double* first,
                               const double* second,
                               std::size_t element_count,
                               double* output_values,
                               double* output_first,
                               double* output_second)
{
  const std::size_t element = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= element_count)
    return;
  const device_math::ScalarJet<double> output =
      device_math::tanhJet(device_math::ScalarJet<double>{values[element], first[element], second[element]});
  output_values[element]  = output.value;
  output_first[element]   = output.first;
  output_second[element]  = output.second;
}

__global__ void residualJetsKernel(const double* left_values,
                                   const double* left_first,
                                   const double* left_second,
                                   const double* right_values,
                                   const double* right_first,
                                   const double* right_second,
                                   std::size_t element_count,
                                   double* output_values,
                                   double* output_first,
                                   double* output_second)
{
  const std::size_t element = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= element_count)
    return;
  const device_math::ScalarJet<double> output = device_math::addJets(
      device_math::ScalarJet<double>{left_values[element], left_first[element], left_second[element]},
      device_math::ScalarJet<double>{right_values[element], right_first[element], right_second[element]});
  output_values[element] = output.value;
  output_first[element]  = output.first;
  output_second[element] = output.second;
}

__global__ void envelopeContributionsKernel(const double* distances,
                                            const double* pi,
                                            const double* zeta,
                                            std::size_t element_count,
                                            double* output)
{
  const std::size_t element = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element < element_count)
    output[element] = device_math::envelopeContribution(distances[element], pi[element], zeta[element]);
}

__global__ void cuspPairsKernel(const double* distances,
                                const double* alpha,
                                const double* cusp_factors,
                                std::size_t pair_count,
                                double* output)
{
  const std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair < pair_count)
    output[pair] = device_math::cuspPair(distances[pair], alpha[pair], cusp_factors[pair]);
}

__global__ void serialSumKernel(const double* input, std::size_t element_count, double* output)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    double sum = 0;
    for (std::size_t element = 0; element < element_count; ++element)
      sum += input[element];
    output[0] = sum;
  }
}

inline unsigned int blockCount(std::size_t element_count)
{
  return static_cast<unsigned int>((element_count + block_size - 1) / block_size);
}

} // namespace

Error launchPairFeatures(Stream stream,
                         const double* displacements,
                         const double* complementary_displacements,
                         const double* radii,
                         std::size_t pair_count,
                         bool periodic,
                         double* features)
{
  if (pair_count == 0)
    return success;
  pairFeaturesKernel<<<blockCount(pair_count), block_size, 0, stream>>>(
      displacements, complementary_displacements, radii, pair_count, periodic, features);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchTanhJets(Stream stream,
                     const double* values,
                     const double* first,
                     const double* second,
                     std::size_t element_count,
                     double* output_values,
                     double* output_first,
                     double* output_second)
{
  if (element_count == 0)
    return success;
  tanhJetsKernel<<<blockCount(element_count), block_size, 0, stream>>>(
      values, first, second, element_count, output_values, output_first, output_second);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

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
                         double* output_second)
{
  if (element_count == 0)
    return success;
  residualJetsKernel<<<blockCount(element_count), block_size, 0, stream>>>(
      left_values, left_first, left_second, right_values, right_first, right_second,
      element_count, output_values, output_first, output_second);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchEnvelopeContributions(Stream stream,
                                  const double* distances,
                                  const double* pi,
                                  const double* zeta,
                                  std::size_t element_count,
                                  double* output)
{
  if (element_count == 0)
    return success;
  envelopeContributionsKernel<<<blockCount(element_count), block_size, 0, stream>>>(
      distances, pi, zeta, element_count, output);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchCuspPairs(Stream stream,
                      const double* distances,
                      const double* alpha,
                      const double* cusp_factors,
                      std::size_t pair_count,
                      double* output)
{
  if (pair_count == 0)
    return success;
  cuspPairsKernel<<<blockCount(pair_count), block_size, 0, stream>>>(
      distances, alpha, cusp_factors, pair_count, output);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error launchSerialSum(Stream stream, const double* input, std::size_t element_count, double* output)
{
  serialSumKernel<<<1, 1, 0, stream>>>(input, element_count, output);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

} // namespace qmcplusplus::psiformer::device
