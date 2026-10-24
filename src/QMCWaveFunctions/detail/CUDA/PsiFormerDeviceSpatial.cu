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
#include "PsiFormerDeviceOrbitalProjection.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"
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

/// Exact periodic branch of the scalar feature builder, restricted to FULL jets.
/// There is no inter-pair reduction: every pair owns seven complete columns.
__device__ bool writePeriodicFeatureJetPair(OpenFeatureJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, std::size_t configuration,
    std::size_t electron, std::size_t nucleus, double* output)
{
  periodic_geometry::Pair pair;
  if (!periodic_geometry::evaluatePair(cell,
      positions + configuration * layout.position_configuration_stride + electron * layout.position_row_stride,
      nuclei + nucleus * layout.nucleus_row_stride, pair, true))
    return false;
  if (pair.radius == 0)
    return false;
  const auto radial = device_math::softenedRadial(pair.radius);
  const std::size_t base = electron * layout.feature_width + 7 * nucleus;
  output[layout.output.uncheckedValueOffset(configuration, base)] = radial.log1p_radius;
  const bool differentiated = layout.output.containsTargetElectron(electron);
  for (std::size_t dimension = 0; dimension < 3 && differentiated; ++dimension)
  {
    const auto lane = 3 * (electron - layout.output.target_electron_begin) + dimension;
    output[layout.output.uncheckedGradientOffset(configuration, lane, base)] =
        radial.log1p_first * pair.radius_gradient[dimension];
  }
  if (differentiated)
    output[layout.output.uncheckedLaplacianOffset(configuration,
        electron - layout.output.target_electron_begin, base)] =
        radial.log1p_second * pair.radius_gradient_norm_squared + radial.log1p_first * pair.radius_laplacian;

  for (std::size_t vector = 0; vector < 2; ++vector)
    for (std::size_t component = 0; component < 3; ++component)
    {
      const double value = vector == 0 ? pair.displacement[component] : pair.complementary[component];
      const double* jacobian = vector == 0 ? pair.jacobian[component] : pair.complementary_jacobian[component];
      const double laplacian = vector == 0 ? pair.laplacian[component] : pair.complementary_laplacian[component];
      const auto element = base + 1 + 3 * vector + component;
      output[layout.output.uncheckedValueOffset(configuration, element)] = value * radial.log1p_over_radius;
      double gradient_dot_radius = 0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        gradient_dot_radius += jacobian[dimension] * pair.radius_gradient[dimension];
        if (differentiated)
        {
          const auto lane = 3 * (electron - layout.output.target_electron_begin) + dimension;
          output[layout.output.uncheckedGradientOffset(configuration, lane, element)] =
              jacobian[dimension] * radial.log1p_over_radius +
              value * radial.log1p_over_radius_first * pair.radius_gradient[dimension];
        }
      }
      if (differentiated)
        output[layout.output.uncheckedLaplacianOffset(configuration,
            electron - layout.output.target_electron_begin, element)] =
            laplacian * radial.log1p_over_radius + 2 * radial.log1p_over_radius_first * gradient_dot_radius +
            value * (radial.log1p_over_radius_second * pair.radius_gradient_norm_squared +
                       radial.log1p_over_radius_first * pair.radius_laplacian);
    }
  return true;
}

/// Cooperative clear/pairs/scan; failure replay restores scalar partial output and precedence.
__global__ void periodicFeatureJetsCooperativeKernel(OpenFeatureJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, std::size_t logical_elements, std::size_t pairs,
    double* output, OpenSpatialStatus* status)
{
  const std::size_t configuration = blockIdx.x;
  const std::size_t active = active_electrons[configuration];
  __shared__ unsigned int failed;
  if (threadIdx.x == 0)
    failed = 0;
  // Initialize logical lanes only: padding belongs to the caller.
  for (std::size_t logical = threadIdx.x; logical < logical_elements; logical += blockDim.x)
    output[layout.output.uncheckedPlaneOffset(configuration,
        logical / layout.output.element_count, logical % layout.output.element_count)] = 0.0;
  __syncthreads();
  if (active >= layout.output.electron_count)
  {
    if (threadIdx.x == 0)
      status[configuration] = open_spatial::buildOpenFeatureJetsForConfiguration(
          layout, positions, nuclei, configuration, active, output, &cell);
    return; // Block-uniform: active is configuration-owned.
  }
  // No pair shares feature columns or performs a cross-pair numerical reduction.
  for (std::size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x)
    if (!writePeriodicFeatureJetPair(layout, cell, positions, nuclei, configuration,
        pair / layout.nucleus_count, pair % layout.nucleus_count, output))
      atomicOr(&failed, 1U);
  // Spin values have one writer; all spin derivatives retain the cleared zero.
  for (std::size_t electron = threadIdx.x; electron < layout.output.electron_count; electron += blockDim.x)
    output[layout.output.uncheckedValueOffset(configuration,
        electron * layout.feature_width + layout.feature_width - 1)] =
        electron < layout.spin_up_count ? 1.0 : -1.0;
  __syncthreads();
  // Match the canonical final finite scan only after every numerical writer ends.
  for (std::size_t logical = threadIdx.x; logical < logical_elements; logical += blockDim.x)
    if (!device_math::isFiniteBinary64(output[layout.output.uncheckedPlaneOffset(configuration,
        logical / layout.output.element_count, logical % layout.output.element_count)]))
      atomicOr(&failed, 1U);
  __syncthreads();
  // A failed configuration is reconstructed in original scalar order, including
  // its exact partial output and first-error/coalescence precedence.
  if (threadIdx.x == 0)
    status[configuration] = failed ? open_spatial::buildOpenFeatureJetsForConfiguration(
        layout, positions, nuclei, configuration, active, output, &cell) : OpenSpatialStatus::REGULAR;
}

/// Canonical seven-column torus VALUE expressions; each pair owns disjoint outputs.
__device__ bool writePeriodicFeatureValuePair(OpenFeatureJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, std::size_t configuration,
    std::size_t electron, std::size_t nucleus, double* output)
{
  periodic_geometry::Pair pair;
  if (!periodic_geometry::evaluatePair(cell,
      positions + configuration * layout.position_configuration_stride + electron * layout.position_row_stride,
      nuclei + nucleus * layout.nucleus_row_stride, pair, true))
    return false;
  const auto radial = device_math::softenedRadial(pair.radius);
  const auto base = electron * layout.feature_width + 7 * nucleus;
  output[layout.output.uncheckedValueOffset(configuration, base)] = radial.log1p_radius;
  for (std::size_t vector = 0; vector < 2; ++vector)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double value = vector == 0 ? pair.displacement[dimension] : pair.complementary[dimension];
      output[layout.output.uncheckedValueOffset(configuration, base + 1 + 3 * vector + dimension)] =
          value * radial.log1p_over_radius;
    }
  return true;
}

/// One block/configuration; canonical GPU replay reconstructs all failed scratch.
__global__ void periodicFeatureValuesCooperativeKernel(OpenFeatureJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, std::size_t logical_elements, std::size_t pairs,
    double* output, OpenSpatialStatus* status)
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
          layout, positions, nuclei, configuration, active, output, &cell);
    return; // Uniform branch after the cooperative clear.
  }
  for (std::size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x)
    if (!writePeriodicFeatureValuePair(layout, cell, positions, nuclei, configuration,
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
        layout, positions, nuclei, configuration, active, output, &cell) : OpenSpatialStatus::REGULAR;
  }
}


/// Independent VALUE pair tiles write only their own seven feature columns.
__global__ void periodicFeatureValueTilesKernel(OpenFeatureJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, std::size_t pairs, std::size_t tiles,
    std::uint32_t* flags, double* output)
{
  const auto configuration = std::size_t(blockIdx.x) / tiles;
  const auto tile = std::size_t(blockIdx.x) % tiles;
  const auto pair = tile * blockDim.x + threadIdx.x;
  __shared__ unsigned int failed;
  if (threadIdx.x == 0)
    failed = active_electrons[configuration] >= layout.output.electron_count;
  __syncthreads();
  if (pair < pairs && active_electrons[configuration] < layout.output.electron_count &&
      !writePeriodicFeatureValuePair(layout, cell, positions, nuclei, configuration,
          pair / layout.nucleus_count, pair % layout.nucleus_count, output))
    atomicOr(&failed, 1U);
  __syncthreads();
  if (threadIdx.x == 0)
    flags[blockIdx.x] = failed;
}

/// Publish spin/finiteness only after every pair tile, or reconstruct scalar failure scratch.
__global__ void periodicFeatureValueTilesFinishKernel(OpenFeatureJetLayout layout,
    periodic_geometry::Cell cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, std::size_t tiles, const std::uint32_t* flags,
    double* output, OpenSpatialStatus* status)
{
  const auto configuration = std::size_t(blockIdx.x);
  __shared__ unsigned int failed;
  if (threadIdx.x == 0)
    failed = 0;
  __syncthreads();
  for (std::size_t tile = threadIdx.x; tile < tiles; tile += blockDim.x)
    if (flags[configuration * tiles + tile])
      atomicOr(&failed, 1U);
  __syncthreads();
  if (!failed)
  {
    for (std::size_t electron = threadIdx.x; electron < layout.output.electron_count; electron += blockDim.x)
      output[layout.output.uncheckedValueOffset(configuration,
          electron * layout.feature_width + layout.feature_width - 1)] =
          electron < layout.spin_up_count ? 1.0 : -1.0;
    __syncthreads();
    for (std::size_t logical = threadIdx.x; logical < layout.output.element_count; logical += blockDim.x)
      if (!device_math::isFiniteBinary64(output[layout.output.uncheckedValueOffset(configuration, logical)]))
        atomicOr(&failed, 1U);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    status[configuration] = failed ? open_spatial::buildOpenFeatureJetsForConfiguration(
        layout, positions, nuclei, configuration, active_electrons[configuration], output, &cell) :
        OpenSpatialStatus::REGULAR;
}

/// Materialize unchanged periodic pair expressions; no partial sums or early public errors.
__global__ void periodicCuspValueTilesKernel(OpenCuspJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const std::size_t* active_electrons, double same_spin_alpha,
    double opposite_spin_alpha, std::size_t cells, std::size_t tiles,
    double* terms, std::uint32_t* flags)
{
  const auto configuration = std::size_t(blockIdx.x) / tiles;
  const auto tile = std::size_t(blockIdx.x) % tiles;
  const auto linear = tile * blockDim.x + threadIdx.x;
  const auto ne = layout.output.electron_count;
  const double* coordinates = positions + configuration * layout.position_configuration_stride;
  __shared__ unsigned int failed;
  if (threadIdx.x == 0)
    failed = active_electrons[configuration] >= ne ||
        !device_math::isFiniteBinary64(same_spin_alpha) ||
        !device_math::isFiniteBinary64(opposite_spin_alpha);
  __syncthreads();
  if (linear < cells)
  {
    terms[configuration * cells + linear] = 0.0;
    const auto first = linear / ne, second = linear % ne;
    if (first < second)
    {
      double radius_squared = 0.0;
      bool finite_input = true;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double first_value = coordinates[first * layout.position_row_stride + dimension];
        const double second_value = coordinates[second * layout.position_row_stride + dimension];
        if (!device_math::isFiniteBinary64(first_value) || !device_math::isFiniteBinary64(second_value))
          finite_input = false;
        const double displacement = first_value - second_value;
        radius_squared += displacement * displacement;
      }
      double radius = ::sqrt(radius_squared);
      periodic_geometry::Pair pair;
      if (!periodic_geometry::evaluatePair(cell,
          coordinates + first * layout.position_row_stride,
          coordinates + second * layout.position_row_stride, pair))
        finite_input = false;
      else
        radius = pair.radius;
      const bool same_spin = (first < layout.spin_up_count) == (second < layout.spin_up_count);
      const double alpha = same_spin ? same_spin_alpha : opposite_spin_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      const double denominator = alpha + radius;
      const double term = factor * alpha * alpha / denominator;
      if (!finite_input || !device_math::isFiniteBinary64(term))
        atomicOr(&failed, 1U);
      else
        terms[configuration * cells + linear] = term;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    flags[blockIdx.x] = failed;
}

/// A single ordered subtraction per configuration preserves every canonical rounding step.
__global__ void periodicCuspValueTilesFinishKernel(OpenCuspJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const std::size_t* active_electrons, double same_spin_alpha,
    double opposite_spin_alpha, std::size_t cells, std::size_t tiles,
    const double* terms, const std::uint32_t* flags, double* output, OpenSpatialStatus* status)
{
  const auto configuration = std::size_t(blockIdx.x);
  bool failed = false;
  for (std::size_t tile = 0; tile < tiles; ++tile)
    failed = failed || flags[configuration * tiles + tile];
  open_spatial::clearConfiguration(layout.output, configuration, output);
  double value = 0.0;
  if (!failed)
    for (std::size_t first = 0; first < layout.output.electron_count; ++first)
      for (std::size_t second = first + 1; second < layout.output.electron_count; ++second)
        value -= terms[configuration * cells + first * layout.output.electron_count + second];
  if (failed || !device_math::isFiniteBinary64(value))
    status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
        layout, positions, configuration, active_electrons[configuration],
        same_spin_alpha, opposite_spin_alpha, output, &cell);
  else
  {
    output[layout.output.uncheckedValueOffset(configuration, 0)] = value;
    status[configuration] = OpenSpatialStatus::REGULAR;
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

/** Compute bounded OPEN VALUE pair chunks cooperatively, then preserve the scalar sum order.
 * No global pair tape: padding and failure diagnostics remain owned by the canonical leaf.
 */
__global__ void openCuspValuesCooperativeKernel(OpenCuspJetLayout layout,
    const double* positions, const std::size_t* active_electrons,
    double same_spin_alpha, double opposite_spin_alpha, std::size_t cells,
    double* output, OpenSpatialStatus* status)
{
  constexpr std::size_t chunk_capacity = 512;
  __shared__ double terms[chunk_capacity];
  __shared__ unsigned int failed;
  const std::size_t configuration = blockIdx.x;
  const auto ne = layout.output.electron_count;
  const auto active = active_electrons[configuration];
  const double* coordinates = positions + configuration * layout.position_configuration_stride;
  if (threadIdx.x == 0)
  {
    open_spatial::clearConfiguration(layout.output, configuration, output);
    failed = active >= ne || !device_math::isFiniteBinary64(same_spin_alpha) ||
        !device_math::isFiniteBinary64(opposite_spin_alpha);
  }
  __syncthreads();
  if (failed)
  {
    if (threadIdx.x == 0)
      status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
          layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output);
    return;
  }
  double value = 0.0;
  for (std::size_t begin = 0; begin < cells;)
  {
    const auto span = cells - begin < chunk_capacity ? cells - begin : chunk_capacity;
    for (std::size_t local = threadIdx.x; local < chunk_capacity; local += blockDim.x)
    {
      terms[local] = 0.0;
      if (local >= span)
        continue;
      const auto linear = begin + local;
      const auto first = linear / ne, second = linear % ne;
      if (first >= second)
        continue;
      double radius_squared = 0.0;
      bool finite_input = true;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double first_value = coordinates[first * layout.position_row_stride + dimension];
        const double second_value = coordinates[second * layout.position_row_stride + dimension];
        if (!device_math::isFiniteBinary64(first_value) || !device_math::isFiniteBinary64(second_value))
          finite_input = false;
        const double displacement = first_value - second_value;
        radius_squared += displacement * displacement;
      }
      const double radius = ::sqrt(radius_squared);
      const bool same_spin = (first < layout.spin_up_count) == (second < layout.spin_up_count);
      const double alpha = same_spin ? same_spin_alpha : opposite_spin_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      const double denominator = alpha + radius;
      const double term = factor * alpha * alpha / denominator;
      if (!finite_input || !device_math::isFiniteBinary64(term))
        atomicOr(&failed, 1U);
      else
        terms[local] = term;
    }
    __syncthreads();
    if (failed)
    {
      if (threadIdx.x == 0)
        status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
            layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output);
      return;
    }
    if (threadIdx.x == 0)
    {
      auto first = begin / ne, second = begin % ne;
      for (std::size_t local = 0; local < span; ++local)
      {
        if (first < second)
          value -= terms[local];
        if (++second == ne)
        {
          second = 0;
          ++first;
        }
      }
    }
    __syncthreads();
    // Advance by the valid span: even the largest representable extent cannot wrap.
    begin += span;
  }
  if (threadIdx.x == 0)
  {
    if (device_math::isFiniteBinary64(value))
    {
      output[layout.output.uncheckedValueOffset(configuration, 0)] = value;
      status[configuration] = OpenSpatialStatus::REGULAR;
    }
    else
      status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
          layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output);
  }
}

/// Periodic pair math, with the same bounded tape and exact scalar subtraction order.
__global__ void periodicCuspValuesCooperativeKernel(OpenCuspJetLayout layout,
    const double* positions, const std::size_t* active_electrons, periodic_geometry::Cell cell,
    double same_spin_alpha, double opposite_spin_alpha, std::size_t cells,
    double* output, OpenSpatialStatus* status)
{
  constexpr std::size_t chunk_capacity = 512;
  __shared__ double terms[chunk_capacity];
  __shared__ unsigned int failed;
  const std::size_t configuration = blockIdx.x;
  const auto ne = layout.output.electron_count;
  const auto active = active_electrons[configuration];
  const double* coordinates = positions + configuration * layout.position_configuration_stride;
  if (threadIdx.x == 0)
  {
    open_spatial::clearConfiguration(layout.output, configuration, output);
    failed = active >= ne || !device_math::isFiniteBinary64(same_spin_alpha) ||
        !device_math::isFiniteBinary64(opposite_spin_alpha);
  }
  __syncthreads();
  if (failed)
  {
    if (threadIdx.x == 0)
      status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
          layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output, &cell);
    return;
  }
  double value = 0.0;
  for (std::size_t begin = 0; begin < cells;)
  {
    const auto span = cells - begin < chunk_capacity ? cells - begin : chunk_capacity;
    for (std::size_t local = threadIdx.x; local < chunk_capacity; local += blockDim.x)
    {
      terms[local] = 0.0;
      if (local >= span)
        continue;
      const auto linear = begin + local;
      const auto first = linear / ne, second = linear % ne;
      if (first >= second)
        continue;
      double radius_squared = 0.0;
      bool finite_input = true;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double first_value = coordinates[first * layout.position_row_stride + dimension];
        const double second_value = coordinates[second * layout.position_row_stride + dimension];
        if (!device_math::isFiniteBinary64(first_value) || !device_math::isFiniteBinary64(second_value))
          finite_input = false;
        const double displacement = first_value - second_value;
        radius_squared += displacement * displacement;
      }
      double radius = ::sqrt(radius_squared);
      periodic_geometry::Pair pair;
      if (!periodic_geometry::evaluatePair(cell,
          coordinates + first * layout.position_row_stride,
          coordinates + second * layout.position_row_stride, pair))
        finite_input = false;
      else
        radius = pair.radius;
      const bool same_spin = (first < layout.spin_up_count) == (second < layout.spin_up_count);
      const double alpha = same_spin ? same_spin_alpha : opposite_spin_alpha;
      const double factor = same_spin ? 0.25 : 0.5;
      const double denominator = alpha + radius;
      const double term = factor * alpha * alpha / denominator;
      if (!finite_input || !device_math::isFiniteBinary64(term))
        atomicOr(&failed, 1U);
      else
        terms[local] = term;
    }
    __syncthreads();
    if (failed)
    {
      if (threadIdx.x == 0)
        status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
            layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output, &cell);
      return;
    }
    if (threadIdx.x == 0)
    {
      auto first = begin / ne, second = begin % ne;
      for (std::size_t local = 0; local < span; ++local)
      {
        if (first < second)
          value -= terms[local];
        if (++second == ne)
        {
          second = 0;
          ++first;
        }
      }
    }
    __syncthreads();
    // Advance by the valid span: even the largest representable extent cannot wrap.
    begin += span;
  }
  if (threadIdx.x == 0)
  {
    if (device_math::isFiniteBinary64(value))
    {
      output[layout.output.uncheckedValueOffset(configuration, 0)] = value;
      status[configuration] = OpenSpatialStatus::REGULAR;
    }
    else
      status[configuration] = open_spatial::buildOpenCuspJetsForConfiguration(
          layout, positions, configuration, active, same_spin_alpha, opposite_spin_alpha, output, &cell);
  }
}

template<bool periodic>
__global__ void gatherCuspTargetsKernel(OpenCuspJetLayout complete, OpenCuspJetLayout target,
    const double* positions, const std::size_t* active_electrons,
    double same_spin_alpha, double opposite_spin_alpha, const double* cached,
    const OpenSpatialStatus* cached_status, double* output, OpenSpatialStatus* status,
    periodic_geometry::Cell cell)
{
  const std::size_t configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < target.output.configuration_count)
    status[configuration] = open_spatial::gatherCuspTargetsForConfiguration(
        complete, target, positions, configuration, active_electrons[configuration],
        same_spin_alpha, opposite_spin_alpha, cached, cached_status[configuration],
        output, periodic ? &cell : nullptr);
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
                                      const double* projected_backflow,
                                      SparseEnvelopeAnchorView anchor)
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
  const std::size_t row = matrix_element / electron_count;
  const std::size_t column = matrix_element % electron_count;
  const std::size_t bank_element = determinant * electron_count * electron_count + matrix_element;
  bool reuse = periodic && anchor.ready && anchor.references && anchor.changed_electrons &&
      anchor.envelope && anchor.positions && anchor.references[configuration] == 0 &&
      anchor.changed_electrons[configuration] < electron_count &&
      row != anchor.changed_electrons[configuration];
  // IEEE bits deliberately distinguish signed zero. A mismatch takes the old
  // envelope path; malformed descriptors cannot change physical diagnostics.
  const double* position = positions + configuration * layout.position_configuration_stride +
      row * layout.position_row_stride;
  for (std::size_t dimension = 0; reuse && dimension < 3; ++dimension)
  {
    union Bits { double value; unsigned long long bits; } current{position[dimension]},
        original{anchor.positions[row * 3 + dimension]};
    reuse = current.bits == original.bits;
  }
  const bool capture = periodic && anchor.capture && configuration == 0 && anchor.envelope &&
      anchor.positions && anchor.references && anchor.changed_electrons &&
      anchor.references[0] == 0 && anchor.changed_electrons[0] == electron_count;
  if (capture && determinant == 0 && column == 0)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      anchor.positions[row * 3 + dimension] = position[dimension];
  status[logical] = open_orbital::buildElement(
      layout, features, positions, nuclei, backflow_up, backflow_down,
      pi_up, pi_down, zeta_up, zeta_down, configuration,
      active_electrons[configuration], determinant,
      row, column, output, retained_backflow, retained_envelope, periodic ? &cell : nullptr, projected_backflow,
      false, nullptr, reuse ? anchor.envelope + bank_element : nullptr,
      capture ? anchor.envelope + bank_element : nullptr);
  if (reuse && determinant == 0 && column == 0 && anchor.counters &&
      status[logical] == OpenSpatialStatus::REGULAR)
  {
    auto previous = atomicCAS(anchor.counters, 0ULL, 0ULL);
    while (previous != ~0ULL)
    {
      const auto observed = atomicCAS(anchor.counters, previous, previous + 1ULL);
      if (observed == previous)
        break;
      previous = observed;
    }
  }
}

/// Share native feature/weight loads while retaining each increasing-feature sum.
__global__ void openOrbitalValuesTiledKernel(OpenOrbitalJetLayout layout,
    const double* features, const double* positions, const double* nuclei,
    const double* backflow_up, const double* backflow_down,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status,
    std::size_t tiles)
{
  constexpr unsigned int tile = 8;
  __shared__ double feature_tile[tile][tile];
  __shared__ double up_weight_tile[tile][tile];
  __shared__ double down_weight_tile[tile][tile];
  __shared__ unsigned int hazard;
  const auto x = threadIdx.x, y = threadIdx.y;
  std::size_t index = blockIdx.x;
  const std::size_t column = (index % tiles) * tile + x;
  index /= tiles;
  const std::size_t row = (index % tiles) * tile + y;
  index /= tiles;
  const std::size_t determinant = index % layout.determinant_count;
  const std::size_t configuration = index / layout.determinant_count;
  const std::size_t ne = layout.features.electron_count;
  if (x == 0 && y == 0)
    hazard = active_electrons[configuration] >= ne;
  __syncthreads();
  double value = 0.0;
  double absolute_sum = 0.0;
  for (std::size_t begin = 0; begin < layout.feature_width;)
  {
    const std::size_t width = layout.feature_width - begin < tile ? layout.feature_width - begin : tile;
    double feature = 0.0, up = 0.0, down = 0.0;
    if (row < ne && x < width)
      feature = features[layout.features.uncheckedValueOffset(
          configuration, row * layout.feature_row_stride + begin + x)];
    if (column < ne && y < width)
    {
      const auto weight = open_orbital::weightOffset(layout, begin + y, determinant, column);
      // Row tiles may cross the spin boundary; never assume one shared spin bank.
      if (layout.spin_up_count) up = backflow_up[weight];
      if (layout.spin_up_count != ne) down = backflow_down[weight];
    }
    feature_tile[y][x] = feature;
    up_weight_tile[y][x] = up;
    down_weight_tile[y][x] = down;
    if (!device_math::isFiniteBinary64(feature) || !device_math::isFiniteBinary64(up) ||
        !device_math::isFiniteBinary64(down))
      atomicExch(&hazard, 1u);
    __syncthreads();
    if (row < ne && column < ne)
      for (std::size_t feature_index = 0; feature_index < width; ++feature_index)
      {
        const double feature_value = feature_tile[y][feature_index];
        const double weight = row < layout.spin_up_count
            ? up_weight_tile[feature_index][x] : down_weight_tile[feature_index][x];
        const double product = feature_value * weight;
        value += product;
        absolute_sum += ::fabs(product);
        if (feature_value != 0.0 && weight != 0.0 &&
            ::fabs(product) < 0x1p-1022)
          atomicExch(&hazard, 1u);
      }
    __syncthreads();
    begin += width;
  }
  // Reassociation/FMA can distinguish a tiny cancellation residue from native signed zero.
  // Replay those sensitive sums instead of relaxing the scalar signed-zero contract.
  if (!device_math::isFiniteBinary64(value) || !device_math::isFiniteBinary64(absolute_sum) ||
      ::fabs(value) <= 8.0 * 0x1p-52 * layout.feature_width * absolute_sum)
    atomicExch(&hazard, 1u);
  __syncthreads();
  if (row < ne && column < ne)
  {
    const auto logical = configuration * layout.determinant_count * ne * ne +
        determinant * ne * ne + row * ne + column;
    if (hazard)
      status[logical] = open_orbital::buildElement(layout, features, positions, nuclei,
          backflow_up, backflow_down, pi_up, pi_down, zeta_up, zeta_down,
          configuration, active_electrons[configuration], determinant, row, column, output);
    else
      status[logical] = open_orbital::buildElement(layout, features, positions, nuclei,
          backflow_up, backflow_down, pi_up, pi_down, zeta_up, zeta_down,
          configuration, active_electrons[configuration], determinant, row, column,
          output, nullptr, nullptr, nullptr, nullptr, false, &value);
  }
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

/// Explicit FULL/target alternative; validates extents before the owner-free launch.
Error launchPeriodicFeatureJetsCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.nucleus_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenFeatureJetLayout(layout, true);
  if (!positions || !nuclei || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cooperative periodic feature jet storage is null");
  if (layout.output.mode != SpatialJetMode::FULL_VGL || layout.output.electron_count < 64)
    return launchPeriodicFeatureJets(stream, layout, cell, positions, nuclei, active_electrons, output, status);
  if (layout.output.configuration_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative periodic feature jet grid exceeds portable device x dimension");
  const auto logical_elements = spatial_detail::checkedProduct(layout.output.plane_count,
      layout.output.element_count, "PsiFormer cooperative periodic feature jet logical extent overflow");
  const auto pairs = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.nucleus_count, "PsiFormer cooperative periodic feature jet pair extent overflow");
  periodicFeatureJetsCooperativeKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, cell, positions, nuclei, active_electrons,
          logical_elements, pairs, output, status);
  return peekSpatialLaunchError();
}

Error launchPeriodicFeatureValuesCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.nucleus_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenFeatureJetLayout(layout, true);
  if (!positions || !nuclei || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cooperative periodic feature storage is null");
  if (layout.output.mode != SpatialJetMode::VALUE_ONLY || layout.output.electron_count < 64)
    return launchPeriodicFeatureJets(stream, layout, cell, positions, nuclei, active_electrons, output, status);
  if (layout.output.configuration_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative periodic feature grid exceeds portable device x dimension");
  const auto logical_elements = spatial_detail::checkedProduct(layout.output.plane_count,
      layout.output.element_count, "PsiFormer cooperative periodic feature logical extent overflow");
  const auto pairs = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.nucleus_count, "PsiFormer cooperative periodic feature pair extent overflow");
  periodicFeatureValuesCooperativeKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, cell, positions, nuclei, active_electrons,
          logical_elements, pairs, output, status);
  return peekSpatialLaunchError();
}


/// Launch independent VALUE feature tiles and publish only after all tiles finish.
Error launchPeriodicFeatureValuesTiled(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, std::uint32_t* flags, std::size_t flag_capacity,
    double* output, OpenSpatialStatus* status)
{
  periodic_geometry::validateCell(cell);
  validateOpenFeatureJetLayout(layout, true);
  if (layout.output.mode != SpatialJetMode::VALUE_ONLY || layout.output.electron_count < 64 ||
      !layout.output.configuration_count || !positions || !nuclei || !active_electrons ||
      !flags || !output || !status)
    throw std::invalid_argument("Tiled periodic features require ordinary VALUE storage");
  const auto pairs = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.nucleus_count, "Tiled periodic feature pair extent overflow");
  const auto tiles = pairs / spatial_block_size + (pairs % spatial_block_size != 0);
  const auto blocks = spatial_detail::checkedProduct(layout.output.configuration_count, tiles,
      "Tiled periodic feature grid overflow");
  if (flag_capacity < blocks)
    throw std::length_error("Tiled periodic feature flags exceed admitted capacity");
  if (blocks > std::size_t(std::numeric_limits<int>::max()) ||
      layout.output.configuration_count > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("Tiled periodic feature grid exceeds portable dimension");
  periodicFeatureValueTilesKernel<<<static_cast<unsigned int>(blocks), spatial_block_size, 0, stream>>>(
      layout, cell, positions, nuclei, active_electrons, pairs, tiles, flags, output);
  auto error = peekSpatialLaunchError();
  if (error != success)
    return error;
  periodicFeatureValueTilesFinishKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, cell, positions, nuclei, active_electrons,
          tiles, flags, output, status);
  return peekSpatialLaunchError();
}

/// Build a global term tape, then retain the canonical per-configuration subtraction order.
Error launchPeriodicCuspValuesTiled(Stream stream, const OpenCuspJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    double* terms, std::size_t term_capacity, std::uint32_t* flags, std::size_t flag_capacity,
    double* output, OpenSpatialStatus* status)
{
  periodic_geometry::validateCell(cell);
  validateOpenCuspJetLayout(layout);
  if (layout.output.mode != SpatialJetMode::VALUE_ONLY || layout.output.electron_count < 64 ||
      !layout.output.configuration_count || !positions || !active_electrons || !terms ||
      !flags || !output || !status)
    throw std::invalid_argument("Tiled periodic cusp requires ordinary VALUE storage");
  const auto cells = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.output.electron_count, "Tiled periodic cusp pair extent overflow");
  const auto tiles = cells / spatial_block_size + (cells % spatial_block_size != 0);
  const auto blocks = spatial_detail::checkedProduct(layout.output.configuration_count, tiles,
      "Tiled periodic cusp grid overflow");
  const auto count = spatial_detail::checkedProduct(layout.output.configuration_count, cells,
      "Tiled periodic cusp tape extent overflow");
  if (term_capacity < count || flag_capacity < blocks)
    throw std::length_error("Tiled periodic cusp exceeds admitted tape capacity");
  if (blocks > std::size_t(std::numeric_limits<int>::max()) ||
      layout.output.configuration_count > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("Tiled periodic cusp grid exceeds portable dimension");
  periodicCuspValueTilesKernel<<<static_cast<unsigned int>(blocks), spatial_block_size, 0, stream>>>(
      layout, cell, positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
      cells, tiles, terms, flags);
  auto error = peekSpatialLaunchError();
  if (error != success)
    return error;
  periodicCuspValueTilesFinishKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      1, 0, stream>>>(layout, cell, positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
          cells, tiles, terms, flags, output, status);
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

Error launchOpenCuspValuesCooperative(Stream stream, const OpenCuspJetLayout& layout,
    const double* positions, const std::size_t* active_electrons,
    double same_spin_alpha, double opposite_spin_alpha, double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.output.electron_count == 0)
    return success;
  validateOpenCuspJetLayout(layout);
  if (!positions || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cooperative cusp device storage is null");
  if (layout.output.mode != SpatialJetMode::VALUE_ONLY || layout.output.electron_count < 64)
    return launchOpenCuspJets(stream, layout, positions, active_electrons,
        same_spin_alpha, opposite_spin_alpha, output, status);
  if (layout.output.configuration_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative cusp grid exceeds portable device x dimension");
  const auto cells = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.output.electron_count, "PsiFormer cooperative cusp pair extent overflow");
  openCuspValuesCooperativeKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, positions, active_electrons,
          same_spin_alpha, opposite_spin_alpha, cells, output, status);
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

Error launchPeriodicCuspValuesCooperative(Stream stream, const OpenCuspJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const std::size_t* active_electrons,
    double same_spin_alpha, double opposite_spin_alpha, double* output, OpenSpatialStatus* status)
{
  if (layout.output.configuration_count == 0 && layout.output.electron_count == 0)
    return success;
  periodic_geometry::validateCell(cell);
  validateOpenCuspJetLayout(layout);
  if (!positions || !active_electrons || !output || !status)
    throw std::invalid_argument("PsiFormer cooperative periodic cusp device storage is null");
  if (layout.output.mode != SpatialJetMode::VALUE_ONLY || layout.output.electron_count < 64)
    return launchPeriodicCuspJets(stream, layout, cell, positions, active_electrons,
        same_spin_alpha, opposite_spin_alpha, output, status);
  if (layout.output.configuration_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PsiFormer cooperative periodic cusp grid exceeds portable device x dimension");
  const auto cells = spatial_detail::checkedProduct(layout.output.electron_count,
      layout.output.electron_count, "PsiFormer cooperative periodic cusp pair extent overflow");
  periodicCuspValuesCooperativeKernel<<<static_cast<unsigned int>(layout.output.configuration_count),
      spatial_block_size, 0, stream>>>(layout, positions, active_electrons, cell,
          same_spin_alpha, opposite_spin_alpha, cells, output, status);
  return peekSpatialLaunchError();
}

Error launchGatherCuspTargets(Stream stream, const OpenCuspJetLayout& complete,
    const OpenCuspJetLayout& target, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    const double* cached, const OpenSpatialStatus* cached_status,
    double* output, OpenSpatialStatus* status, const periodic_geometry::Cell* cell)
{
  validateOpenCuspJetLayout(complete);
  validateOpenCuspJetLayout(target);
  if (complete.output.mode != SpatialJetMode::FULL_VGL || target.output.mode != SpatialJetMode::FULL_VGL ||
      complete.output.target_electron_begin != 0 ||
      complete.output.laplacian_lanes != complete.output.electron_count ||
      complete.output.electron_count != target.output.electron_count ||
      complete.output.configuration_count != target.output.configuration_count ||
      complete.spin_up_count != target.spin_up_count ||
      complete.position_row_stride != target.position_row_stride ||
      complete.position_configuration_stride != target.position_configuration_stride)
    throw std::invalid_argument("PsiFormer cusp cache/target layouts are incompatible");
  if (!positions || !active_electrons || !cached || !cached_status || !output || !status ||
      cached == output || cached_status == status)
    throw std::invalid_argument("PsiFormer cusp cache gather storage is null or aliases");
  if (cell) periodic_geometry::validateCell(*cell);
  const auto blocks = checkedSpatialBlockCount(target.output.configuration_count,
      "PsiFormer cusp gather grid exceeds device x dimension");
  if (cell)
    gatherCuspTargetsKernel<true><<<blocks, spatial_block_size, 0, stream>>>(complete, target,
        positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
        cached, cached_status, output, status, *cell);
  else
    gatherCuspTargetsKernel<false><<<blocks, spatial_block_size, 0, stream>>>(complete, target,
        positions, active_electrons, same_spin_alpha, opposite_spin_alpha,
        cached, cached_status, output, status, {});
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

/// Materialize only pointer metadata; all numerical inputs/outputs keep their old layout.
__global__ void orbitalProjectionPointersKernel(OpenOrbitalJetLayout layout, std::size_t count,
    const double* features, const double* up, const double* down, double* output,
    const double** weights, const double** source, double** destinations)
{
  auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const auto destination = index;
  const auto determinant = index % layout.determinant_count;
  index /= layout.determinant_count;
  const auto plane = index % layout.features.plane_count;
  index /= layout.features.plane_count;
  const auto configuration = index % layout.features.configuration_count;
  const auto spin = index / layout.features.configuration_count;
  const bool is_up = layout.spin_up_count != 0 && spin == 0;
  const auto first = is_up ? 0 : layout.spin_up_count;
  weights[destination] = (is_up ? up : down) + determinant * layout.backflow_determinant_stride;
  source[destination] = features + layout.features.uncheckedPlaneOffset(configuration, plane,
      first * layout.feature_row_stride);
  destinations[destination] = output + layout.orbitals.uncheckedPlaneOffset(configuration, plane,
      open_orbital::outputElement(layout, determinant, first, 0));
}

/// Refresh one complete stage before its stream-ordered grouped GEMMs.
Error launchOrbitalProjectionPointers(Stream stream, const OpenOrbitalJetLayout& layout,
    const double* features, const double* up, const double* down, double* output,
    const double** weights, const double** source, double** destinations)
{
  validateOpenOrbitalJetLayout(layout);
  const auto bytes = orbitalProjectionPointerBytes(layout.features.configuration_count,
      layout.features.plane_count, layout.determinant_count,
      (layout.spin_up_count != 0) + (layout.spin_up_count != layout.features.electron_count));
  const auto count = bytes / (sizeof(const double*) * 2 + sizeof(double*));
  const auto blocks = checkedSpatialBlockCount(count, "Orbital projection pointer grid exceeds device x dimension");
  if (!features || !output || !weights || !source || !destinations ||
      (layout.spin_up_count && !up) || (layout.spin_up_count != layout.features.electron_count && !down))
    throw std::invalid_argument("Orbital projection pointer refresh has null admitted storage");
  orbitalProjectionPointersKernel<<<blocks, spatial_block_size, 0, stream>>>(layout, count,
      features, up, down, output, weights, source, destinations);
  return peekSpatialLaunchError();
}

/// Validate ordering without touching stream state or introducing a fence.
void validateOrbitalProjectionStream(const AcceleratorBlasHandle& handle, Stream stream)
{
  if (handle.h_stream != stream)
    throw std::invalid_argument("Orbital projection pointers and BLAS require one ordered stream");
}

/// Preserve the same FP64 logical matrix per determinant using documented pointer batching.
void projectOrbitalDeterminantsBatched(AcceleratorBlasHandle& handle, const DenseForwardLayout& layout,
    std::size_t determinants, const double* const* weights, const double* const* source,
    double* const* output)
{
  validateDenseForwardLayout(layout);
  const auto checked = [](std::size_t n) {
    if (!n || n > std::size_t(std::numeric_limits<int>::max()))
      throw std::overflow_error("Orbital projection batched BLAS extent exceeds int");
    return static_cast<int>(n);
  };
  compute::BLAS::gemm_batched(handle, 'N', 'N', checked(layout.output_width), checked(layout.rows),
      checked(layout.input_width), 1.0, weights, checked(layout.weight_row_stride),
      source, checked(layout.source_row_stride), 0.0, output, checked(layout.target_row_stride), checked(determinants));
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
      retained_backflow, retained_envelope, {}, projected_backflow, {});
  return peekSpatialLaunchError();
}

Error launchOpenOrbitalValuesTiled(Stream stream, const OpenOrbitalJetLayout& layout,
    const double* features, const double* positions, const double* nuclei,
    const double* backflow_up, const double* backflow_down,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status)
{
  if (layout.features.mode != SpatialJetMode::VALUE_ONLY || layout.features.electron_count < 64)
    return launchOpenOrbitalJets(stream, layout, features, positions, nuclei, backflow_up, backflow_down,
        pi_up, pi_down, zeta_up, zeta_down, active_electrons, output, status);
  validateOpenOrbitalJetLayout(layout);
  const auto ne = layout.features.electron_count;
  if (!features || !positions || !nuclei || !active_electrons || !output || !status ||
      (layout.spin_up_count && (!backflow_up || !pi_up || !zeta_up)) ||
      (layout.spin_up_count != ne && (!backflow_down || !pi_down || !zeta_down)))
    throw std::invalid_argument("PsiFormer tiled orbital value device storage is null");
  const auto tiles = ne / 8 + (ne % 8 != 0);
  const auto count = spatial_detail::checkedProduct(layout.features.configuration_count,
      spatial_detail::checkedProduct(layout.determinant_count,
          spatial_detail::checkedProduct(tiles, tiles, "PsiFormer orbital tile grid overflow"),
          "PsiFormer orbital tile grid overflow"), "PsiFormer orbital tile grid overflow");
  if (count > std::numeric_limits<int>::max())
    return launchOpenOrbitalJets(stream, layout, features, positions, nuclei, backflow_up, backflow_down,
        pi_up, pi_down, zeta_up, zeta_down, active_electrons, output, status);
  openOrbitalValuesTiledKernel<<<static_cast<unsigned int>(count), dim3(8, 8), 0, stream>>>(layout,
      features, positions, nuclei, backflow_up, backflow_down, pi_up, pi_down, zeta_up, zeta_down,
      active_electrons, output, status, tiles);
  return peekSpatialLaunchError();
}

Error launchPeriodicOrbitalJets(Stream stream, const OpenOrbitalJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* features,
    const double* positions, const double* nuclei, const double* backflow_up,
    const double* backflow_down, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status, double* retained_backflow, double* retained_envelope,
    const double* projected_backflow, SparseEnvelopeAnchorView anchor)
{
  // Invalid optional views never weaken the old leaf's physical diagnostics.
  if (layout.features.mode != SpatialJetMode::VALUE_ONLY || retained_backflow || retained_envelope ||
      projected_backflow || !anchor.references || !anchor.changed_electrons || !anchor.positions ||
      !anchor.envelope || anchor.envelope == output || anchor.positions == positions ||
      anchor.envelope == positions || anchor.positions == output)
    anchor = {};
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
      retained_backflow, retained_envelope, cell, projected_backflow, anchor);
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
