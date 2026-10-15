//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceRawFeatureReverseValidation.h
 * @brief Focused seeded-trunk input-adjoint and storage qualification.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RAW_FEATURE_REVERSE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RAW_FEATURE_REVERSE_VALIDATION_H

#include "PsiFormerDeviceNetworkReverse.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFeatureCoordinateReverse.h"

#include <limits>

namespace qmcplusplus::psiformer::device
{

/** Check the input VJP in a physical coordinate direction using independent
 * central differences of the existing final-feature forward objective. This
 * seeds only the neural trunk; it is not a complete wavefunction gradient.
 */
inline void validateRawFeatureNetworkReverse(std::shared_ptr<const PreparedDeviceModel> image,
    const PreparedDeviceInference& forward, compute::Queue<PlatformKind::CUDA>& queue,
    AcceleratorBlasHandle& blas, const std::vector<double>& positions, const std::vector<double>& nuclei,
    const std::vector<double>& seed)
{
  const auto& shape = image->plan().modelShape();
  const auto count = forward.valueTapeBatchCount(), ne = shape.electrons();
  const auto raw_width = image->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
  const auto capacity = count + 1;
  PreparedDeviceNetworkReverse raw(image, capacity, queue, blas, NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY);
  const auto bytes = raw.deviceWorkspaceBytes();
  const auto raw_capacity = capacity * ne * raw_width;
  const auto chunks = raw_capacity / 4096 + (raw_capacity % 4096 != 0);
  const auto expected_bytes = (10 * capacity * ne * shape.feature_dimension +
      capacity * shape.attention_heads * ne * ne + raw_capacity) * sizeof(double) +
      chunks * sizeof(DerivativeFiniteStatus);
  if (raw.parameterCount() != 0 || bytes != expected_bytes)
    throw std::runtime_error("Raw-only reverse retained unexpected numerical storage or a P destination");

  // A failed request must revoke the previous view, including a wrong-mode call.
  bool unavailable = false;
  try { (void)raw.rawFeatureAdjoints(); }
  catch (const std::logic_error&) { unavailable = true; }
  if (!unavailable)
    throw std::runtime_error("Raw-only reverse exposed an incomplete view");
  const auto view = raw.reverseRawFeatures(forward, seed);
  if (view.layout.configuration_count != count || view.layout.rows != ne ||
      view.layout.width != raw_width || view.layout.configuration_stride != ne * raw_width ||
      view.model_version != image->version() || view.evaluation_generation != forward.evaluationGeneration())
    throw std::runtime_error("Raw-only reverse published a capacity-sized or incompatible live view");
  std::vector<double> raw_bar(count * ne * raw_width);
  raw.downloadRawFeatureAdjoints(raw_bar);
  bool wrong_mode = false, stale = false;
  try { (void)raw.reverse(forward, seed); }
  catch (const std::logic_error&) { wrong_mode = true; }
  try { (void)raw.rawFeatureAdjoints(); }
  catch (const std::logic_error&) { stale = true; }
  if (!wrong_mode || !stale)
    throw std::runtime_error("Wrong-mode reverse exposed stale raw cotangents");
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "raw reverse validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "raw reverse validation device");
#endif
  const auto before_reuse = deviceMemoryMetrics(ordinal).snapshot();
  raw.reverseRawFeatures(forward, seed);
  const auto after_reuse = deviceMemoryMetrics(ordinal).snapshot();
  raw.downloadRawFeatureAdjoints(raw_bar);
  if (raw.deviceWorkspaceBytes() != bytes || before_reuse.allocation_calls != after_reuse.allocation_calls)
    throw std::runtime_error("Raw-only reverse changed prepared storage during reuse");

  const auto* cell = image->periodicCell();
  const auto layout = cell
      ? makePeriodicFeatureJetLayout(count, ne, shape.nuclei, shape.spin_up_electrons, SpatialJetMode::VALUE_ONLY)
      : makeOpenFeatureJetLayout(count, ne, shape.nuclei, shape.spin_up_electrons, SpatialJetMode::VALUE_ONLY);
  if (layout.feature_width != raw_width || positions.size() != count * ne * 3 || nuclei.size() != shape.nuclei * 3)
    throw std::invalid_argument("Raw-only validation geometry extent mismatch");
  double contraction = 0;
  auto minus = positions, plus = positions;
  constexpr double delta = 1e-5;
  for (std::size_t configuration = 0; configuration < count; ++configuration)
    for (std::size_t electron = 0; electron < ne; ++electron)
    {
      double coordinate[3]{};
      if (feature_coordinate::coordinateAdjointForElectron(layout, positions.data(), nuclei.data(),
          raw_bar.data(), configuration, electron, coordinate, cell) != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Raw-only geometry transpose rejected a regular fixture");
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const auto index = 3 * (configuration * ne + electron) + dimension;
        const double direction = 0.03 * (static_cast<int>(index % 7) - 3);
        minus[index] -= delta * direction;
        plus[index] += delta * direction;
        contraction += coordinate[dimension] * direction;
      }
    }

  double objectives[2]{};
  PreparedDeviceInference shifted(image, count, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  std::vector<double> features(seed.size());
  for (std::size_t side = 0; side < 2; ++side)
  {
    shifted.evaluate(side == 0 ? minus : plus);
    shifted.downloadValueFinalFeatures(features);
    for (std::size_t index = 0; index < seed.size(); ++index)
      objectives[side] += features[index] * seed[index];
  }
  const double difference = (objectives[1] - objectives[0]) / (2 * delta);
  if (!device_math::isFiniteBinary64(contraction) || !device_math::isFiniteBinary64(difference) ||
      ::fabs(contraction - difference) > 2e-7 * (1 + ::fabs(difference)))
    throw std::runtime_error("Raw-only seeded trunk VJP differs from coordinate directional finite difference");

  // A nonfinite device seed must be rejected by the selected-output finite scan.
  DeviceBuffer<double> device_seed(seed.size());
  auto invalid_seed = seed;
  invalid_seed[0] = std::numeric_limits<double>::quiet_NaN();
  device_seed.upload(invalid_seed.data(), invalid_seed.size());
  bool nonfinite = false;
  try { (void)raw.reverseRawFeatures(forward, {}, device_seed.data()); }
  catch (const std::runtime_error&) { nonfinite = true; }
  stale = false;
  try { (void)raw.rawFeatureAdjoints(); }
  catch (const std::logic_error&) { stale = true; }
  if (!nonfinite || !stale)
    throw std::runtime_error("Nonfinite raw reverse exposed a completed view");
  raw.reverseRawFeatures(forward, seed);
}

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_RAW_FEATURE_REVERSE_VALIDATION_H
