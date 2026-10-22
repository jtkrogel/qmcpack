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
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalProjection.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicPairAttention.h"

#include <limits>
#include <cstdlib>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int spatial_block_size = 128;

__global__ void scatterSpatialTargetsKernel(SpatialJetLayout source, SpatialJetLayout target,
    const double* source_phase, const double* source_jets, const double* source_ratios,
    double* target_phase, double* target_jets, double* target_ratios)
{
  const auto configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < source.configuration_count)
    open_spatial::scatterSpatialTargetsForConfiguration(source, target, configuration,
        source_phase, source_jets, source_ratios, target_phase, target_jets, target_ratios);
}

/// Finalize independent value requests without derivative storage or traces.
__global__ void finalValueCombinationKernel(std::size_t configuration_count,
    const device_determinant::CombinationMetadata* metadata,
    const double* cusp, double* phase, double* logabs, OpenSpatialStatus* status)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < configuration_count)
    status[configuration] = open_spatial::combineFinalValue(
        metadata[configuration], cusp[configuration], phase[configuration], logabs[configuration]);
}

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

// An 8x8 output tile reuses each loaded Q/K feature across eight products.
// Jet planes are independent blocks; padded shared rows bound storage to 5760 bytes.
__global__ void attentionLogitJetsTiledKernel(SpatialAttentionJetLayout layout,
    const double* query, const double* key, double* attention)
{
  constexpr std::size_t tile = 8;
  __shared__ double q[5][tile][tile + 1];
  __shared__ double k[5][tile][tile + 1];
  const std::size_t x = threadIdx.x;
  const std::size_t y = threadIdx.y;
  const std::size_t tiles = (layout.rows + tile - 1) / tile;
  std::size_t index = blockIdx.x;
  const std::size_t key_base = (index % tiles) * tile;
  index /= tiles;
  const std::size_t query_base = (index % tiles) * tile;
  index /= tiles;
  const std::size_t head = index % layout.heads;
  const std::size_t configuration = index / layout.heads;
  const std::size_t plane = blockIdx.y;
  const bool gradient = plane > 0 && plane <= layout.features.gradient_lanes;
  const bool laplacian = plane > layout.features.gradient_lanes;
  const std::size_t electron = laplacian ? plane - 1 - layout.features.gradient_lanes : 0;
  // Slot zero holds values, slot one the requested first derivative or trace.
  // A trace additionally needs its three first derivatives in slots two to four.
  const std::size_t planes = laplacian ? 5 : gradient ? 2 : 1;
  const std::size_t configuration_offset = configuration * layout.features.configuration_stride;
  const double scale = 1.0 / ::sqrt(static_cast<double>(layout.head_width));
  double result = 0.0;
  for (std::size_t feature_base = 0; feature_base < layout.head_width; feature_base += tile)
  {
    const std::size_t feature = feature_base + x;
    // Neighboring threads stage adjacent features from each query/key row.
    for (std::size_t p = 0; p < planes; ++p)
    {
      const std::size_t source_plane = p == 0 ? 0 : p == 1 ? plane : 1 + 3 * electron + p - 2;
      const std::size_t offset = configuration_offset + source_plane * layout.features.plane_stride +
          head * layout.head_width + feature;
      q[p][y][x] = query_base + y < layout.rows && feature < layout.head_width
          ? query[offset + (query_base + y) * layout.feature_row_stride] : 0.0;
      k[p][y][x] = key_base + y < layout.rows && feature < layout.head_width
          ? key[offset + (key_base + y) * layout.feature_row_stride] : 0.0;
    }
    __syncthreads();
    const std::size_t remaining = layout.head_width - feature_base;
    const std::size_t width = remaining < tile ? remaining : tile;
    for (std::size_t f = 0; f < width; ++f)
    {
      if (laplacian)
      {
        double gradient_dot = 0.0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_dot += q[2 + dimension][y][f] * k[2 + dimension][x][f];
        result += scale * (q[1][y][f] * k[0][x][f] + 2.0 * gradient_dot +
                           q[0][y][f] * k[1][x][f]);
      }
      else if (gradient)
        result += scale * (q[1][y][f] * k[0][x][f] + q[0][y][f] * k[1][x][f]);
      else
        result += scale * q[0][y][f] * k[0][x][f];
    }
    __syncthreads();
  }
  // Partial-row threads participate in every barrier before suppressing stores.
  if (query_base + y < layout.rows && key_base + x < layout.rows)
    attention[configuration * layout.attention.configuration_stride +
        plane * layout.attention.plane_stride + head * layout.attention_head_stride +
        (query_base + y) * layout.attention_row_stride + key_base + x] = result;
}

/// Each entry owns the learned pair addition before normalization and tape capture.
__global__ void periodicPairAttentionJetsKernel(SpatialAttentionJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, const double* weights,
    const double* bias, const std::size_t* active_electrons, std::size_t spin_up_count,
    double* attention)
{
  const auto entry = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto per_configuration = layout.heads * layout.rows * layout.rows;
  if (entry >= layout.features.configuration_count * per_configuration)
    return;
  const auto configuration = entry / per_configuration;
  const auto remainder = entry % per_configuration;
  const auto head = remainder / (layout.rows * layout.rows);
  const auto query = (remainder / layout.rows) % layout.rows;
  const auto key = remainder % layout.rows;
  const auto active = layout.attention.mode == SpatialJetMode::ACTIVE ? active_electrons[configuration] : 0;
  if (!periodic_geometry::addLogComplementAttentionJet(layout, cell,
      positions + configuration * 3 * layout.rows, spin_up_count, weights, bias,
      configuration, head, query, key, active, attention))
  {
    union { unsigned long long integer; double floating; } invalid{0x7ff8000000000000ULL};
    const auto element = head * layout.attention_head_stride + query * layout.attention_row_stride + key;
    attention[layout.attention.uncheckedValueOffset(configuration, element)] = invalid.floating;
  }
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

/// Specialize the immutable boundary policy without changing open arithmetic.
template<bool periodic>
__global__ void boundaryFeatureJetsKernel(OpenFeatureJetLayout layout,
                                      const double* positions,
                                      const double* nuclei,
                                      const std::size_t* active_electrons,
                                      double* output,
                                      OpenSpatialStatus* status,
                                      periodic_geometry::Cell cell)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.output.configuration_count)
    status[configuration] = open_spatial::buildOpenFeatureJetsForConfiguration(
        layout, positions, nuclei, configuration, active_electrons[configuration], output,
        periodic ? &cell : nullptr);
}

/// Write one OPEN EI pair with the canonical scalar expressions and disjoint output columns.
__device__ bool writeOpenFeaturePair(OpenFeatureJetLayout layout, const double* positions,
    const double* nuclei, std::size_t configuration, std::size_t active_electron,
    std::size_t electron, std::size_t nucleus, double* output)
{
  const double* electron_position = positions + configuration * layout.position_configuration_stride +
      electron * layout.position_row_stride;
  const double* nucleus_position = nuclei + nucleus * layout.nucleus_row_stride;
  double displacement[3];
  double radius_squared = 0.0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    if (!device_math::isFiniteBinary64(electron_position[dimension]) ||
        !device_math::isFiniteBinary64(nucleus_position[dimension]))
      return false;
    displacement[dimension] = electron_position[dimension] - nucleus_position[dimension];
    radius_squared += displacement[dimension] * displacement[dimension];
  }
  const double radius = ::sqrt(radius_squared);
  if (!(radius > 0.0) && layout.output.mode != SpatialJetMode::VALUE_ONLY)
    return false;
  const device_math::SoftenedRadial<double> radial = device_math::softenedRadial(radius);
  const std::size_t base = electron * layout.feature_width + 4 * nucleus;
  output[layout.output.uncheckedValueOffset(configuration, base)] = radial.log1p_radius;
  for (std::size_t component = 0; component < 3; ++component)
    output[layout.output.uncheckedValueOffset(configuration, base + 1 + component)] =
        displacement[component] * radial.log1p_over_radius;

  const bool differentiated = (layout.output.mode == SpatialJetMode::FULL_VGL &&
      layout.output.containsTargetElectron(electron)) ||
      (layout.output.mode == SpatialJetMode::ACTIVE && electron == active_electron);
  if (differentiated)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t lane = layout.output.mode == SpatialJetMode::FULL_VGL
          ? 3 * (electron - layout.output.target_electron_begin) + dimension : dimension;
      const double direction = displacement[dimension] / radius;
      output[layout.output.uncheckedGradientOffset(configuration, lane, base)] =
          radial.log1p_first * direction;
      for (std::size_t component = 0; component < 3; ++component)
        output[layout.output.uncheckedGradientOffset(configuration, lane, base + 1 + component)] =
            (component == dimension ? radial.log1p_over_radius : 0.0) +
            displacement[component] * radial.log1p_over_radius_first * direction;
    }
  if (layout.output.mode == SpatialJetMode::FULL_VGL && differentiated)
  {
    output[layout.output.uncheckedLaplacianOffset(configuration,
        electron - layout.output.target_electron_begin, base)] =
        radial.log1p_second + 2.0 * radial.log1p_first / radius;
    for (std::size_t component = 0; component < 3; ++component)
      output[layout.output.uncheckedLaplacianOffset(configuration,
          electron - layout.output.target_electron_begin, base + 1 + component)] =
          2.0 * radial.log1p_over_radius_first * displacement[component] / radius +
          displacement[component] *
              (radial.log1p_over_radius_second + 2.0 * radial.log1p_over_radius_first / radius);
  }
  return true;
}

/** One block per configuration; failure-only device replay preserves exact scalar diagnostics.
 * All numerical writes are pair-disjoint. Shared control never scales with the model.
 */
__global__ void openFeatureJetsCooperativeKernel(OpenFeatureJetLayout layout,
    const double* positions, const double* nuclei, const std::size_t* active_electrons,
    std::size_t logical_elements, std::size_t pairs, double* output, OpenSpatialStatus* status)
{
  const std::size_t configuration = blockIdx.x;
  const std::size_t active = active_electrons[configuration];
  __shared__ unsigned int failed;
  if (threadIdx.x == 0)
    failed = 0;
  // Zero only logical elements; padded strides remain caller-owned and untouched.
  for (std::size_t logical = threadIdx.x; logical < logical_elements; logical += blockDim.x)
    output[layout.output.uncheckedPlaneOffset(configuration,
        logical / layout.output.element_count, logical % layout.output.element_count)] = 0.0;
  __syncthreads();
  if (active >= layout.output.electron_count)
  {
    if (threadIdx.x == 0)
      status[configuration] = open_spatial::buildOpenFeatureJetsForConfiguration(
          layout, positions, nuclei, configuration, active, output);
    return; // Uniform branch after the cooperative clear.
  }
  for (std::size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x)
    if (!writeOpenFeaturePair(layout, positions, nuclei, configuration, active,
        pair / layout.nucleus_count, pair % layout.nucleus_count, output))
      atomicOr(&failed, 1U);
  // Exactly one thread writes each electron's spin value; derivative planes stay zero.
  for (std::size_t electron = threadIdx.x; electron < layout.output.electron_count; electron += blockDim.x)
    output[layout.output.uncheckedValueOffset(configuration,
        electron * layout.feature_width + layout.feature_width - 1)] =
        electron < layout.spin_up_count ? 1.0 : -1.0;
  __syncthreads();
  for (std::size_t logical = threadIdx.x; logical < logical_elements; logical += blockDim.x)
    if (!device_math::isFiniteBinary64(output[layout.output.uncheckedPlaneOffset(configuration,
        logical / layout.output.element_count, logical % layout.output.element_count)]))
      atomicOr(&failed, 1U);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    // Reconstruct even partial failed scratch in scalar order. In particular a
    // later pair's geometric failure precedes the final computed-finite scan.
    status[configuration] = failed ? open_spatial::buildOpenFeatureJetsForConfiguration(
        layout, positions, nuclei, configuration, active, output) : OpenSpatialStatus::REGULAR;
  }
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

/** Add bias only to retained value planes; coordinate derivatives stay unchanged. */
__global__ void retainedPreactivationBiasKernel(SpatialElementwiseJetLayout layout,
    const double* bias, double* retained)
{
  const auto logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto per_configuration = layout.row_count * layout.row_width;
  if (logical >= layout.jets.configuration_count * per_configuration)
    return;
  const auto configuration = logical / per_configuration;
  const auto within = logical % per_configuration;
  const auto element = (within / layout.row_width) * layout.row_stride + within % layout.row_width;
  retained[layout.jets.uncheckedValueOffset(configuration, element)] += bias[within % layout.row_width];
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

/// Dispatch one configuration's radial cusp using the selected real geometry.
template<bool periodic>
__global__ void boundaryCuspJetsKernel(OpenCuspJetLayout layout,
                                   const double* positions,
                                   const std::size_t* active_electrons,
                                   double same_spin_alpha,
                                   double opposite_spin_alpha,
                                   double* output,
                                   OpenSpatialStatus* status,
                                   periodic_geometry::Cell cell)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.output.configuration_count)
    status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
        layout, positions, configuration, active_electrons[configuration],
        same_spin_alpha, opposite_spin_alpha, output, periodic ? &cell : nullptr);
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

/// Preserve the same backflow-envelope product graph for both boundary policies.
template<bool periodic>
__global__ void boundaryOrbitalJetsKernel(OpenOrbitalJetLayout layout,
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
                                      OpenSpatialStatus* status,
                                      double* retained_backflow,
                                      double* retained_envelope,
                                      periodic_geometry::Cell cell,
                                      const double* projected_backflow)
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
      matrix_element / electron_count, matrix_element % electron_count, output,
      retained_backflow, retained_envelope, periodic ? &cell : nullptr, projected_backflow);
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
    device_math::JetMathStatus* row_status,
    double* retained_logits,
    const PeriodicPairAttentionInputs* pair_attention)
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
  if (retained_logits && (retained_logits == attention || retained_logits == query || retained_logits == key))
    throw std::invalid_argument("Retained logit jets require a distinct destination");
  if (pair_attention)
  {
    if (!pair_attention->cell || !pair_attention->positions || !pair_attention->weights || !pair_attention->bias ||
        pair_attention->spin_up_count == 0 || pair_attention->spin_up_count >= layout.rows ||
        (layout.attention.mode == SpatialJetMode::ACTIVE && !pair_attention->active_electrons))
      throw std::invalid_argument("Periodic pair-attention inputs require complete geometry and parameters");
    periodic_geometry::validateCell(*pair_attention->cell);
  }

  const unsigned int block_count = static_cast<unsigned int>(
      (entry_count + spatial_block_size - 1) / spatial_block_size);
  const char* tiled_option = std::getenv("PSIFORMER_GPU_ATTENTION_JET_TILED");
  bool launched_tiled = false;
  if (tiled_option && tiled_option[0] == '1' && tiled_option[1] == '\0')
  {
    const std::size_t tiles = (layout.rows + 7) / 8;
    const std::size_t tile_count = spatial_detail::checkedProduct(
        spatial_detail::checkedProduct(layout.features.configuration_count, layout.heads,
            "PsiFormer attention tile batch overflow"),
        spatial_detail::checkedProduct(tiles, tiles, "PsiFormer attention tile count overflow"),
        "PsiFormer attention tile grid overflow");
    const std::size_t planes = layout.features.plane_count;
    if (tile_count <= std::numeric_limits<int>::max() && planes <= 65535)
    {
      attentionLogitJetsTiledKernel<<<dim3(static_cast<unsigned int>(tile_count),
          static_cast<unsigned int>(planes)), dim3(8, 8), 0, stream>>>(layout, query, key, attention);
      launched_tiled = true;
    }
  }
  if (!launched_tiled)
    attentionLogitJetsKernel<<<block_count, spatial_block_size, 0, stream>>>(
        layout, query, key, attention);
#ifdef QMC_CUDA2HIP
  const Error logit_error = hipPeekAtLastError();
#else
  const Error logit_error = cudaPeekAtLastError();
#endif
  if (logit_error != success)
    return logit_error;
  if (pair_attention)
  {
    periodicPairAttentionJetsKernel<<<block_count, spatial_block_size, 0, stream>>>(layout,
        *pair_attention->cell, pair_attention->positions, pair_attention->weights, pair_attention->bias,
        pair_attention->active_electrons, pair_attention->spin_up_count, attention);
#ifdef QMC_CUDA2HIP
    const auto pair_error = hipPeekAtLastError();
#else
    const auto pair_error = cudaPeekAtLastError();
#endif
    if (pair_error != success)
      return pair_error;
  }
  if (retained_logits)
  {
    // Copy before in-place softmax on the same ordered stream; lifted reverse
    // needs raw logit planes, not reconstructed logarithms of probabilities.
    const auto bytes = spatial_detail::checkedProduct(spatialJetSpanElements(layout.attention),
        sizeof(double), "Retained attention logit bytes overflow");
#ifdef QMC_CUDA2HIP
    const auto copy_error = hipMemcpyAsync(retained_logits, attention, bytes, hipMemcpyDeviceToDevice, stream);
#else
    const auto copy_error = cudaMemcpyAsync(retained_logits, attention, bytes, cudaMemcpyDeviceToDevice, stream);
#endif
    if (copy_error != success)
      return copy_error;
  }
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
  boundaryFeatureJetsKernel<false><<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, nuclei, active_electrons, output, status, {});
  return peekSpatialLaunchError();
}

Error launchOpenFeatureJetsCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const double* positions, const double* nuclei, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.nucleus_count == 0)
    return success;
  validateOpenFeatureJetLayout(layout);
  if (!positions || !nuclei || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cooperative open feature storage is null");
  if (layout.output.electron_count < 64)
    return launchOpenFeatureJets(stream, layout, positions, nuclei, active_electrons, output, status);
  if (layout.output.configuration_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative feature grid exceeds portable device x dimension");
  const auto logical_elements = spatial_detail::checkedProduct(layout.output.plane_count,
      layout.output.element_count, "PsiFormer cooperative feature logical extent overflow");
  const auto pairs = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.nucleus_count, "PsiFormer cooperative feature pair extent overflow");
  openFeatureJetsCooperativeKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, positions, nuclei, active_electrons,
          logical_elements, pairs, output, status);
  return peekSpatialLaunchError();
}

Error launchPeriodicFeatureJets(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.nucleus_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenFeatureJetLayout(layout, true);
  if (!positions || !nuclei || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer periodic feature device storage is null");
  const auto blocks = checkedSpatialBlockCount(layout.output.configuration_count,
      "PsiFormer periodic feature grid exceeds device x dimension");
  boundaryFeatureJetsKernel<true><<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, nuclei, active_electrons, output, status, cell);
  return peekSpatialLaunchError();
}

Error launchBiasTanhJets(Stream stream, const SpatialElementwiseJetLayout& layout,
                         const double* input, const double* bias, double* output,
                         OpenSpatialStatus* status,
                         double* retained_preactivation)
{
  if (layout.jets.configuration_count == 0 && layout.row_count == 0 && layout.row_width == 0)
    return success;
  validateSpatialElementwiseJetLayout(layout);
  if (retained_preactivation && (retained_preactivation == input || retained_preactivation == output ||
                                retained_preactivation == bias))
    throw std::invalid_argument("Retained preactivation jets require a distinct destination");
  if (!input || !bias || !output || !status)
    throw std::invalid_argument("PsiFormer tanh jet device storage is null");
  const std::size_t count = spatial_detail::checkedProduct(
      layout.jets.configuration_count,
      spatial_detail::checkedProduct(layout.row_count, layout.row_width,
                                     "PsiFormer tanh jet extent overflow"),
      "PsiFormer tanh jet batch extent overflow");
  const unsigned int blocks = checkedSpatialBlockCount(
      count, "PsiFormer tanh jet grid exceeds device x dimension");
  if (retained_preactivation)
  {
    // Snapshot in a separate ordered operation before any in-place nonlinear
    // writes. This avoids coupling capture loads to the activation's aliasing.
    const auto bytes = spatial_detail::checkedProduct(spatialJetSpanElements(layout.jets),
        sizeof(double), "Retained preactivation bytes overflow");
#ifdef QMC_CUDA2HIP
    const auto copy_error = hipMemcpyAsync(retained_preactivation, input, bytes, hipMemcpyDeviceToDevice, stream);
#else
    const auto copy_error = cudaMemcpyAsync(retained_preactivation, input, bytes, cudaMemcpyDeviceToDevice, stream);
#endif
    if (copy_error != success)
      return copy_error;
    retainedPreactivationBiasKernel<<<blocks, spatial_block_size, 0, stream>>>(layout, bias, retained_preactivation);
    const auto bias_error = peekSpatialLaunchError();
    if (bias_error != success)
      return bias_error;
  }
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
  boundaryCuspJetsKernel<false><<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
      output, status, {});
  return peekSpatialLaunchError();
}

Error launchPeriodicCuspJets(Stream stream, const OpenCuspJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.output.electron_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenCuspJetLayout(layout);
  if (!positions || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer periodic cusp device storage is null");
  const auto blocks = checkedSpatialBlockCount(layout.output.configuration_count,
      "PsiFormer periodic cusp grid exceeds device x dimension");
  boundaryCuspJetsKernel<true><<<blocks, spatial_block_size, 0, stream>>>(
      layout, positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
      output, status, cell);
  return peekSpatialLaunchError();
}

/// Launch the checked one-plane signed-log finalization path.
Error launchFinalValueCombination(Stream stream, std::size_t configuration_count,
    const device_determinant::CombinationMetadata* metadata,
    const double* cusp, double* phase, double* logabs, OpenSpatialStatus* status)
{
  if (configuration_count == 0)
    return success;
  if (!metadata || !cusp || !phase || !logabs || !status)
    throw std::invalid_argument("PsiFormer final value device storage is null");
  const unsigned int blocks = checkedSpatialBlockCount(
      configuration_count, "PsiFormer final value grid exceeds device x dimension");
  finalValueCombinationKernel<<<blocks, spatial_block_size, 0, stream>>>(
      configuration_count, metadata, cusp, phase, logabs, status);
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

// Reuse the retained factor allocation: each GEMM writes its logical tile only.
void projectOpenOrbitalJets(AcceleratorBlasHandle& handle, const OpenOrbitalJetLayout& layout,
    const double* features, const double* backflow_up, const double* backflow_down,
    double* projected_backflow)
{
  validateOpenOrbitalJetLayout(layout);
  if (!features || !projected_backflow || projected_backflow == features ||
      projected_backflow == backflow_up || projected_backflow == backflow_down ||
      (layout.spin_up_count != 0 && !backflow_up) ||
      (layout.spin_up_count != layout.features.electron_count && !backflow_down))
    throw std::invalid_argument("PsiFormer orbital projection requires distinct valid storage");

  forEachOpenOrbitalProjection(layout, [&](const OpenOrbitalProjectionSlice& slice) {
    const double* weight = slice.spin_up ? backflow_up : backflow_down;
    denseForward(handle, slice.dense, features + slice.source_offset,
        weight + slice.weight_offset, projected_backflow + slice.target_offset);
  });
}

Error launchOpenOrbitalJets(Stream stream, const OpenOrbitalJetLayout& layout,
                            const double* features, const double* positions,
                            const double* nuclei, const double* backflow_up,
                            const double* backflow_down, const double* pi_up,
                            const double* pi_down, const double* zeta_up,
                            const double* zeta_down,
                            const std::size_t* active_electrons, double* output,
                            OpenSpatialStatus* status,
                            double* retained_backflow,
                            double* retained_envelope,
                            const double* projected_backflow)
{
  if (layout.features.configuration_count == 0 &&
      layout.determinant_count == 0 && layout.features.electron_count == 0)
    return success;
  validateOpenOrbitalJetLayout(layout);
  if ((retained_backflow == nullptr) != (retained_envelope == nullptr) ||
      (retained_backflow && (retained_backflow == retained_envelope ||
                            retained_backflow == output || retained_envelope == output)))
    throw std::invalid_argument("Retained orbital factors require distinct paired destinations");
  if (projected_backflow && (projected_backflow == output || projected_backflow == retained_envelope))
    throw std::invalid_argument("Projected backflow must be distinct from product and envelope storage");
  const bool need_up = layout.spin_up_count != 0;
  const bool need_down = layout.spin_up_count != layout.features.electron_count;
  if (!features || !positions || !nuclei || !active_electrons || !output || !status ||
      (need_up && (!backflow_up || !pi_up || !zeta_up)) ||
      (need_down && (!backflow_down || !pi_down || !zeta_down)))
    throw std::invalid_argument("PsiFormer open orbital device storage is null");
  const std::size_t count = openOrbitalElementCount(layout);
  const unsigned int blocks = checkedSpatialBlockCount(
      count, "PsiFormer open orbital grid exceeds device x dimension");
  boundaryOrbitalJetsKernel<false><<<blocks, spatial_block_size, 0, stream>>>(
      layout, features, positions, nuclei, backflow_up, backflow_down,
      pi_up, pi_down, zeta_up, zeta_down, active_electrons, output, status,
      retained_backflow, retained_envelope, {}, projected_backflow);
  return peekSpatialLaunchError();
}

Error launchPeriodicOrbitalJets(Stream stream, const OpenOrbitalJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* features,
    const double* positions, const double* nuclei, const double* backflow_up,
    const double* backflow_down, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status, double* retained_backflow, double* retained_envelope,
    const double* projected_backflow)
{
  if (layout.features.configuration_count == 0 &&
      layout.determinant_count == 0 && layout.features.electron_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenOrbitalJetLayout(layout);
  if ((retained_backflow == nullptr) != (retained_envelope == nullptr) ||
      (retained_backflow && (retained_backflow == retained_envelope ||
                            retained_backflow == output || retained_envelope == output)))
    throw std::invalid_argument("Retained orbital factors require distinct paired destinations");
  if (projected_backflow && (projected_backflow == output || projected_backflow == retained_envelope))
    throw std::invalid_argument("Projected backflow must be distinct from product and envelope storage");
  const bool need_up = layout.spin_up_count != 0;
  const bool need_down = layout.spin_up_count != layout.features.electron_count;
  if (!features || !positions || !nuclei || !active_electrons || !output || !status ||
      (need_up && (!backflow_up || !pi_up || !zeta_up)) ||
      (need_down && (!backflow_down || !pi_down || !zeta_down)))
    throw std::invalid_argument("PsiFormer periodic orbital device storage is null");
  const auto blocks = checkedSpatialBlockCount(openOrbitalElementCount(layout),
      "PsiFormer periodic orbital grid exceeds device x dimension");
  boundaryOrbitalJetsKernel<true><<<blocks, spatial_block_size, 0, stream>>>(
      layout, features, positions, nuclei, backflow_up, backflow_down,
      pi_up, pi_down, zeta_up, zeta_down, active_electrons, output, status,
      retained_backflow, retained_envelope, cell, projected_backflow);
  return peekSpatialLaunchError();
}

Error launchScatterSpatialTargets(Stream stream, const SpatialJetLayout& source, const SpatialJetLayout& target,
    const double* source_phase, const double* source_jets, const double* source_ratios,
    double* target_phase, double* target_jets, double* target_ratios)
{
  validateSpatialTargetScatter(source, target);
  if (!source_phase || !source_jets || !source_ratios || !target_phase || !target_jets || !target_ratios)
    throw std::invalid_argument("PsiFormer target scatter storage is null");
  for (const auto* input : {source_phase, source_jets, source_ratios})
    for (const auto* output : {target_phase, target_jets, target_ratios})
      if (input == output)
        throw std::invalid_argument("PsiFormer target scatter requires separate output storage");
  if (target_phase == target_jets || target_phase == target_ratios || target_jets == target_ratios)
    throw std::invalid_argument("PsiFormer target scatter output destinations alias");
  const auto blocks = checkedSpatialBlockCount(source.configuration_count,
      "PsiFormer target scatter grid exceeds device x dimension");
  scatterSpatialTargetsKernel<<<blocks, spatial_block_size, 0, stream>>>(source, target,
      source_phase, source_jets, source_ratios, target_phase, target_jets, target_ratios);
  return peekSpatialLaunchError();
}

} // namespace qmcplusplus::psiformer::device
