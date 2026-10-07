//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatial.cu
 * @brief Shared CUDA/HIP spatial-jet kernels without warp-size assumptions.
 */

#include "PsiFormerDeviceSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetKernels.h"

#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int spatial_block_size = 128;

__global__ void softmaxJetRowsKernel(SoftmaxJetRowLayout layout,
                                     double* jets,
                                     device_math::JetMathStatus* row_status)
{
  const std::size_t batch_row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t batch_row_count =
      layout.jets.configuration_count * layout.row_count;
  if (batch_row >= batch_row_count)
    return;
  const std::size_t configuration = batch_row / layout.row_count;
  const std::size_t row           = batch_row % layout.row_count;
  const std::size_t group         = row / layout.rows_per_group;
  const std::size_t row_in_group  = row % layout.rows_per_group;
  const std::size_t row_offset = configuration * layout.jets.configuration_stride +
      group * layout.row_group_stride + row_in_group * layout.row_stride;
  double* laplacian = layout.jets.laplacian_lanes == 0
      ? nullptr
      : jets + row_offset +
          (1 + layout.jets.gradient_lanes) * layout.jets.plane_stride;
  row_status[batch_row] = device_math::softmaxJetRowInPlace(
      jets + row_offset, jets + row_offset + layout.jets.plane_stride,
      laplacian, layout.row_width, layout.jets.plane_stride,
      layout.jets.gradient_lanes, layout.jets.laplacian_lanes);
}

__global__ void attentionLogitJetsKernel(SpatialAttentionJetLayout layout,
                                         const double* query,
                                         const double* key,
                                         double* attention)
{
  const std::size_t entry = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t entries_per_configuration =
      layout.heads * layout.rows * layout.rows;
  const std::size_t entry_count =
      layout.features.configuration_count * entries_per_configuration;
  if (entry >= entry_count)
    return;
  const std::size_t configuration = entry / entries_per_configuration;
  const std::size_t remainder     = entry % entries_per_configuration;
  const std::size_t head          = remainder / (layout.rows * layout.rows);
  const std::size_t row_pair      = remainder % (layout.rows * layout.rows);
  spatial_jet::attentionLogitJetElement(
      layout, query, key, configuration, head, row_pair / layout.rows,
      row_pair % layout.rows, attention);
}

__global__ void attentionContextJetsKernel(SpatialAttentionJetLayout layout,
                                           const double* attention,
                                           const double* value,
                                           double* context)
{
  const std::size_t entry = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t feature_width = layout.heads * layout.head_width;
  const std::size_t entries_per_configuration = layout.rows * feature_width;
  const std::size_t entry_count =
      layout.features.configuration_count * entries_per_configuration;
  if (entry >= entry_count)
    return;
  const std::size_t configuration = entry / entries_per_configuration;
  const std::size_t remainder     = entry % entries_per_configuration;
  const std::size_t output_row    = remainder / feature_width;
  const std::size_t output_feature = remainder % feature_width;
  spatial_jet::attentionContextJetElement(
      layout, attention, value, configuration, output_row,
      output_feature / layout.head_width, output_feature % layout.head_width,
      context);
}

__global__ void openFeatureJetsKernel(OpenFeatureJetLayout layout,
                                      const double* positions,
                                      const double* nuclei,
                                      const std::size_t* active_electrons,
                                      double* output,
                                      OpenSpatialStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.output.configuration_count)
    status[configuration] = open_spatial::buildOpenFeatureJetsForConfiguration(
        layout, positions, nuclei, configuration, active_electrons[configuration], output);
}

__global__ void biasTanhJetsKernel(SpatialElementwiseJetLayout layout,
                                   const double* input,
                                   const double* bias,
                                   double* output,
                                   OpenSpatialStatus* status)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t per_configuration = layout.row_count * layout.row_width;
  const std::size_t count = layout.jets.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t within = logical % per_configuration;
  const std::size_t element = (within / layout.row_width) * layout.row_stride +
      within % layout.row_width;
  status[logical] = open_spatial::biasTanhJetElement(
      layout, input, bias, configuration, element, output);
}

__global__ void residualSpatialJetsKernel(SpatialElementwiseJetLayout layout,
                                          const double* left,
                                          const double* right,
                                          double* output,
                                          OpenSpatialStatus* status)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t per_configuration = layout.row_count * layout.row_width;
  const std::size_t count = layout.jets.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t within = logical % per_configuration;
  const std::size_t element = (within / layout.row_width) * layout.row_stride +
      within % layout.row_width;
  status[logical] = open_spatial::residualJetElement(
      layout, left, right, configuration, element, output);
}

__global__ void openCuspJetsKernel(OpenCuspJetLayout layout,
                                   const double* positions,
                                   const std::size_t* active_electrons,
                                   double same_spin_alpha,
                                   double opposite_spin_alpha,
                                   double* output,
                                   OpenSpatialStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.output.configuration_count)
    status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
        layout, positions, configuration, active_electrons[configuration],
        same_spin_alpha, opposite_spin_alpha, output);
}

__global__ void finalSpatialCombinationKernel(
    FinalSpatialJetLayout layout,
    const device_determinant::CombinationMetadata* determinant_metadata,
    const device_determinant::DerivativeStatus* determinant_status,
    const double* determinant_gradient,
    const double* determinant_laplacian_log,
    const double* cusp,
    double* phase,
    double* output,
    double* laplacian_ratio,
    OpenSpatialStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.output.configuration_count)
    status[configuration] = open_spatial::combineFinalSpatialForConfiguration(
        layout, determinant_metadata, determinant_status, determinant_gradient,
        determinant_laplacian_log, cusp, configuration, phase, output,
        laplacian_ratio);
}

__global__ void openOrbitalJetsKernel(OpenOrbitalJetLayout layout,
                                      const double* features,
                                      const double* positions,
                                      const double* nuclei,
                                      const double* backflow_up,
                                      const double* backflow_down,
                                      const double* pi_up,
                                      const double* pi_down,
                                      const double* zeta_up,
                                      const double* zeta_down,
                                      const std::size_t* active_electrons,
                                      double* output,
                                      OpenSpatialStatus* status)
{
  const std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t electron_count = layout.features.electron_count;
  const std::size_t per_configuration =
      layout.determinant_count * electron_count * electron_count;
  const std::size_t count = layout.features.configuration_count * per_configuration;
  if (logical >= count)
    return;
  const std::size_t configuration = logical / per_configuration;
  const std::size_t remainder = logical % per_configuration;
  const std::size_t determinant = remainder / (electron_count * electron_count);
  const std::size_t matrix_element = remainder % (electron_count * electron_count);
  status[logical] = open_orbital::buildElement(
      layout, features, positions, nuclei, backflow_up, backflow_down,
      pi_up, pi_down, zeta_up, zeta_down, configuration,
      active_electrons[configuration], determinant,
      matrix_element / electron_count, matrix_element % electron_count, output);
}

} // namespace

Error launchSoftmaxJetRows(Stream stream,
                           const SoftmaxJetRowLayout& layout,
                           double* jets,
                           device_math::JetMathStatus* row_status)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 &&
      layout.row_width == 0)
    return success;
  validateSoftmaxJetRowLayout(layout);
  const std::size_t batch_row_count = softmaxJetBatchRowCount(layout);
  if (batch_row_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer softmax jet grid exceeds the CUDA/HIP x dimension");
  if (!jets || !row_status)
    throw std::invalid_argument("PsiFormer softmax jet device storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (batch_row_count + spatial_block_size - 1) / spatial_block_size);
  softmaxJetRowsKernel<<<block_count, spatial_block_size, 0, stream>>>(
      layout, jets, row_status);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

void denseJetsForward(AcceleratorBlasHandle& handle,
                      const SpatialDenseJetLayout& layout,
                      const double* source,
                      const double* weight,
                      double* target)
{
  validateSpatialDenseJetLayout(layout);
  if (!source || !weight || !target)
    throw std::invalid_argument("PsiFormer spatial dense storage is null");
  const DenseForwardLayout plane_layout = makeDenseForwardLayout(
      layout.rows, layout.input_width, layout.output_width,
      layout.source_row_stride, layout.weight_row_stride,
      layout.target_row_stride);
  for (std::size_t configuration = 0;
       configuration < layout.source.configuration_count; ++configuration)
    for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
      denseForward(
          handle, plane_layout,
          source + configuration * layout.source.configuration_stride +
              plane * layout.source.plane_stride,
          weight,
          target + configuration * layout.target.configuration_stride +
              plane * layout.target.plane_stride);
}

void projectQkvJetsForward(AcceleratorBlasHandle& handle,
                           const SpatialDenseJetLayout& layout,
                           const double* source,
                           const double* query_weight,
                           const double* key_weight,
                           const double* value_weight,
                           double* query,
                           double* key,
                           double* value)
{
  validateSpatialDenseJetLayout(layout);
  if (!source || !query_weight || !key_weight || !value_weight ||
      !query || !key || !value)
    throw std::invalid_argument("PsiFormer spatial QKV storage is null");
  const DenseForwardLayout plane_layout = makeDenseForwardLayout(
      layout.rows, layout.input_width, layout.output_width,
      layout.source_row_stride, layout.weight_row_stride,
      layout.target_row_stride);
  for (std::size_t configuration = 0;
       configuration < layout.source.configuration_count; ++configuration)
    for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
      projectQkvForward(
          handle, plane_layout,
          source + configuration * layout.source.configuration_stride +
              plane * layout.source.plane_stride,
          query_weight, key_weight, value_weight,
          query + configuration * layout.target.configuration_stride +
              plane * layout.target.plane_stride,
          key + configuration * layout.target.configuration_stride +
              plane * layout.target.plane_stride,
          value + configuration * layout.target.configuration_stride +
              plane * layout.target.plane_stride);
}

Error launchAttentionJetWeights(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    double* attention,
    device_math::JetMathStatus* row_status)
{
  if (layout.features.configuration_count == 0 && layout.rows == 0 &&
      layout.heads == 0 && layout.head_width == 0)
    return success;
  validateSpatialAttentionJetLayout(layout);
  const std::size_t entries_per_configuration = spatial_detail::checkedProduct(
      layout.heads,
      spatial_detail::checkedProduct(
          layout.rows, layout.rows,
          "PsiFormer attention jet row-pair count overflow"),
      "PsiFormer attention jet entry count overflow");
  const std::size_t entry_count = spatial_detail::checkedProduct(
      layout.features.configuration_count, entries_per_configuration,
      "PsiFormer attention jet batch extent overflow");
  if (entry_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer attention jet grid exceeds the CUDA/HIP x dimension");
  if (!query || !key || !attention || !row_status)
    throw std::invalid_argument("PsiFormer attention jet storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (entry_count + spatial_block_size - 1) / spatial_block_size);
  attentionLogitJetsKernel<<<block_count, spatial_block_size, 0, stream>>>(
      layout, query, key, attention);
#ifdef QMC_CUDA2HIP
  const Error logit_error = hipPeekAtLastError();
#else
  const Error logit_error = cudaPeekAtLastError();
#endif
  if (logit_error != success)
    return logit_error;
  return launchSoftmaxJetRows(
      stream, makeAttentionSoftmaxJetRowLayout(layout), attention, row_status);
}

Error launchAttentionContextJets(Stream stream,
                                 const SpatialAttentionJetLayout& layout,
                                 const double* attention,
                                 const double* value,
                                 double* context)
{
  if (layout.features.configuration_count == 0 && layout.rows == 0 &&
      layout.heads == 0 && layout.head_width == 0)
    return success;
  validateSpatialAttentionJetLayout(layout);
  const std::size_t feature_width = spatial_detail::checkedProduct(
      layout.heads, layout.head_width,
      "PsiFormer attention context feature width overflow");
  const std::size_t entry_count = spatial_detail::checkedProduct(
      layout.features.configuration_count,
      spatial_detail::checkedProduct(
          layout.rows, feature_width,
          "PsiFormer attention context configuration extent overflow"),
      "PsiFormer attention context batch extent overflow");
  if (entry_count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("PsiFormer attention context grid exceeds the CUDA/HIP x dimension");
  if (!attention || !value || !context)
    throw std::invalid_argument("PsiFormer attention context storage is null");

  const unsigned int block_count = static_cast<unsigned int>(
      (entry_count + spatial_block_size - 1) / spatial_block_size);
  attentionContextJetsKernel<<<block_count, spatial_block_size, 0, stream>>>(
      layout, attention, value, context);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

namespace
{
unsigned int checkedSpatialBlockCount(std::size_t count, const char* description)
{
  if (count > std::numeric_limits<unsigned int>::max())
    throw std::length_error(description);
  return static_cast<unsigned int>((count + spatial_block_size - 1) / spatial_block_size);
}

Error peekSpatialLaunchError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

Error launchOpenFeatureJets(Stream stream, const OpenFeatureJetLayout& layout,
                            const double* positions, const double* nuclei,
                            const std::size_t* active_electrons, double* output,
                            OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.nucleus_count == 0)
    return success;
  validateOpenFeatureJetLayout(layout);
  if (!positions || !nuclei || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer open feature device storage is null");
  const unsigned int blocks = checkedSpatialBlockCount(
      layout.output.configuration_count, "PsiFormer open feature grid exceeds device x dimension");
  openFeatureJetsKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, nuclei, active_electrons, output, status);
  return peekSpatialLaunchError();
}

Error launchBiasTanhJets(Stream stream, const SpatialElementwiseJetLayout& layout,
                         const double* input, const double* bias, double* output,
                         OpenSpatialStatus* status)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 && layout.row_width == 0)
    return success;
  validateSpatialElementwiseJetLayout(layout);
  if (!input || !bias || !output || !status)
    throw std::invalid_argument("PsiFormer tanh jet device storage is null");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.jets.configuration_count,
      spatial_detail::checkedProduct(layout.row_count, layout.row_width,
                                     "PsiFormer tanh jet extent overflow"),
      "PsiFormer tanh jet batch extent overflow");
  const unsigned int blocks = checkedSpatialBlockCount(
      count, "PsiFormer tanh jet grid exceeds device x dimension");
  biasTanhJetsKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, input, bias, output, status);
  return peekSpatialLaunchError();
}

Error launchResidualSpatialJets(Stream stream, const SpatialElementwiseJetLayout& layout,
                                const double* left, const double* right, double* output,
                                OpenSpatialStatus* status)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 && layout.row_width == 0)
    return success;
  validateSpatialElementwiseJetLayout(layout);
  if (!left || !right || !output || !status)
    throw std::invalid_argument("PsiFormer residual jet device storage is null");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.jets.configuration_count,
      spatial_detail::checkedProduct(layout.row_count, layout.row_width,
                                     "PsiFormer residual jet extent overflow"),
      "PsiFormer residual jet batch extent overflow");
  const unsigned int blocks = checkedSpatialBlockCount(
      count, "PsiFormer residual jet grid exceeds device x dimension");
  residualSpatialJetsKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, left, right, output, status);
  return peekSpatialLaunchError();
}

Error launchOpenCuspJets(Stream stream, const OpenCuspJetLayout& layout,
                         const double* positions, const std::size_t* active_electrons,
                         double same_spin_alpha, double opposite_spin_alpha,
                         double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.output.electron_count == 0)
    return success;
  validateOpenCuspJetLayout(layout);
  if (!positions || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cusp jet device storage is null");
  const unsigned int blocks = checkedSpatialBlockCount(
      layout.output.configuration_count, "PsiFormer cusp jet grid exceeds device x dimension");
  openCuspJetsKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
      output, status);
  return peekSpatialLaunchError();
}

Error launchFinalSpatialCombination(
    Stream stream, const FinalSpatialJetLayout& layout,
    const device_determinant::CombinationMetadata* determinant_metadata,
    const device_determinant::DerivativeStatus* determinant_status,
    const double* determinant_gradient, const double* determinant_laplacian_log,
    const double* cusp, double* phase, double* output, double* laplacian_ratio,
    OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.output.electron_count == 0)
    return success;
  validateFinalSpatialJetLayout(layout);
  if (!determinant_metadata || !determinant_status || !determinant_gradient ||
      !cusp || !phase || !output || !status ||
      (layout.output.laplacian_lanes != 0 &&
       (!determinant_laplacian_log || !laplacian_ratio)))
    throw std::invalid_argument("PsiFormer final spatial device storage is null");
  const unsigned int blocks = checkedSpatialBlockCount(
      layout.output.configuration_count, "PsiFormer final spatial grid exceeds device x dimension");
  finalSpatialCombinationKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, determinant_metadata, determinant_status, determinant_gradient,
      determinant_laplacian_log, cusp, phase, output, laplacian_ratio, status);
  return peekSpatialLaunchError();
}

Error launchOpenOrbitalJets(Stream stream, const OpenOrbitalJetLayout& layout,
                            const double* features, const double* positions,
                            const double* nuclei, const double* backflow_up,
                            const double* backflow_down, const double* pi_up,
                            const double* pi_down, const double* zeta_up,
                            const double* zeta_down,
                            const std::size_t* active_electrons, double* output,
                            OpenSpatialStatus* status)
{
  if (layout.features.configuration_count == 0 &&
      layout.determinant_count == 0 && layout.features.electron_count == 0)
    return success;
  validateOpenOrbitalJetLayout(layout);
  const bool need_up = layout.spin_up_count != 0;
  const bool need_down = layout.spin_up_count != layout.features.electron_count;
  if (!features || !positions || !nuclei || !active_electrons || !output || !status ||
      (need_up && (!backflow_up || !pi_up || !zeta_up)) ||
      (need_down && (!backflow_down || !pi_down || !zeta_down)))
    throw std::invalid_argument("PsiFormer open orbital device storage is null");
  const std::size_t count = openOrbitalElementCount(layout);
  const unsigned int blocks = checkedSpatialBlockCount(
      count, "PsiFormer open orbital grid exceeds device x dimension");
  openOrbitalJetsKernel<<<blocks, spatial_block_size, 0, stream>>>(
      layout, features, positions, nuclei, backflow_up, backflow_down,
      pi_up, pi_down, zeta_up, zeta_down, active_electrons, output, status);
  return peekSpatialLaunchError();
}

} // namespace qmcplusplus::psiformer::device
