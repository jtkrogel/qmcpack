//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceCoordinateGradientValidation.h
 * @brief Focused all-coordinate real-core reverse qualification.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_VALIDATION_H

#include "PsiFormerDeviceCoordinateGradient.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace qmcplusplus::psiformer::device
{

/** Check all coordinates against independent CPU full VGL, a scalar directional
 * difference, and failure/rebind/publication invariants on one compact fixture.
 * Only the validation translation unit includes the native CPU engine.
 */
inline void validateCoordinateGradient(std::shared_ptr<const PreparedDeviceModel> image,
    pf::PsiFormer& cpu_model, const std::vector<double>& positions,
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, bool spot = false)
{
  const auto& s = image->plan().modelShape();
  const auto ne = s.electrons(), count = positions.size() / (3 * ne);
  if (count == 0 || positions.size() != count * 3 * ne)
    throw std::invalid_argument("Coordinate validation requires complete configurations");
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "coordinate validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "coordinate validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  PreparedDeviceInference forward(image, count, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  const auto before_prepare = metrics.snapshot();
  PreparedDeviceCoordinateGradient gradient(image, count, queue, blas);
  const auto after_prepare = metrics.snapshot();
  const auto bytes = gradient.deviceWorkspaceBytes(), host_bytes = gradient.hostWorkspaceBytes();
  const auto raw_width = image->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0];
  const auto chunks = [](std::size_t elements) { return elements / 4096 + (elements % 4096 != 0); };
  const auto rows = count * ne;
  const auto expected_bytes = sizeof(double) * (10 * rows * s.feature_dimension +
      count * s.attention_heads * ne * ne + rows * raw_width + 4 * count * s.determinants * ne * ne +
      rows * s.feature_dimension + rows * 3 + count) +
      sizeof(DerivativeFiniteStatus) * (chunks(rows * raw_width) + chunks(rows * 3)) +
      sizeof(device_determinant::DerivativeStatus) * count + sizeof(OpenSpatialStatus) * rows;
  const auto expected_host_bytes = sizeof(double) * count +
      sizeof(device_determinant::FactorizationMetadata) * count * s.determinants +
      sizeof(device_determinant::DerivativeStatus) * count + sizeof(OpenSpatialStatus) * rows +
      sizeof(DerivativeFiniteStatus) * (chunks(rows * raw_width) + chunks(rows * 3));
  if (bytes != expected_bytes || host_bytes != expected_host_bytes ||
      after_prepare.live_bytes - before_prepare.live_bytes != bytes)
    throw std::runtime_error("Coordinate reverse storage differs from exact owner allocation accounting");
  pf::DirectValueExecutor cpu_value(cpu_model, image->plan());
  pf::DirectSpatialExecutor cpu_spatial(cpu_model, cpu_value, image->plan());
  auto cpu_workspace = cpu_spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  std::vector<double> result(rows * 3);
  double max_abs_gradient_error = 0, max_relative_gradient_error = 0;

  // Repeated logical prefixes reuse every forward/reverse allocation, while all
  // Cartesian entries remain checked against the existing CPU spatial executor.
  const auto prefixes = spot ? std::vector<std::size_t>{count}
                            : std::vector<std::size_t>{count, 1, std::min<std::size_t>(2, count), count};
  for (const auto live : prefixes)
  {
    std::vector<double> prefix(positions.begin(), positions.begin() + live * ne * 3);
    result.resize(live * ne * 3);
    const auto before = metrics.snapshot();
    forward.evaluate(prefix);
    const auto view = gradient.reverse(forward);
    const auto after = metrics.snapshot();
    if (after.allocation_calls != before.allocation_calls || gradient.deviceWorkspaceBytes() != bytes ||
        gradient.hostWorkspaceBytes() != host_bytes || gradient.borrowedForwardDeviceBytes() != forward.deviceWorkspaceBytes() ||
        gradient.borrowedForwardHostBytes() != forward.hostWorkspaceBytes() ||
        view.model_image != image.get() || view.model_version != image->version() ||
        view.evaluation_generation != forward.evaluationGeneration() || view.configuration_count != live)
      throw std::runtime_error("Coordinate reverse reuse, provenance or borrowed-owner accounting changed");
    gradient.downloadGradient(result);
    for (std::size_t sample = 0; sample < live; ++sample)
    {
      cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(prefix.data() + sample * ne * 3, ne));
      const auto reference = cpu_spatial.evaluateFull(*cpu_workspace);
      for (std::size_t coordinate = 0; coordinate < ne * 3; ++coordinate)
      {
        const double error = ::fabs(result[sample * ne * 3 + coordinate] - reference.gradient[coordinate]);
        max_abs_gradient_error = std::max(max_abs_gradient_error, error);
        max_relative_gradient_error = std::max(max_relative_gradient_error,
            error / std::max(::fabs(reference.gradient[coordinate]), 1e-12));
        if (error >
            2e-7 * (1 + ::fabs(reference.gradient[coordinate])))
          throw std::runtime_error("Coordinate reverse differs from independent full-VGL CPU gradient");
      }
    }
  }

  const auto report = [&]() {
    std::cout << "Coordinate gradient capacity: B=" << count << " predicted_reverse_device_bytes=" << expected_bytes
              << " actual_reverse_device_bytes=" << bytes << " reverse_host_bytes=" << host_bytes
              << " borrowed_forward_device_bytes=" << gradient.borrowedForwardDeviceBytes()
              << " borrowed_forward_host_bytes=" << gradient.borrowedForwardHostBytes()
              << " max_abs_gradient_error=" << max_abs_gradient_error
              << " max_relative_gradient_error=" << max_relative_gradient_error
              << " relative_denominator_floor=1e-12\n";
  };
  if (spot)
  {
    // Time only the warmed value-forward plus exact gradient reverse operation;
    // no determinant/parameter response or unrelated kinetic suite is evaluated.
    const auto before = metrics.snapshot();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < 3; ++repeat)
    {
      forward.evaluate(positions);
      gradient.reverse(forward);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Warmed coordinate spot operation allocated device storage");
    report();
    std::cout << "Coordinate gradient spot: electrons=" << ne << " configurations=" << count
              << " warmed_value_forward_plus_reverse_seconds_per_configuration=" << elapsed / (3 * count)
              << " parameter_response_vectors=0 warm_device_allocations=0\n";
    return;
  }

  const auto stale_rejected = [&]() {
    try { (void)gradient.view(); }
    catch (const std::logic_error&) { return true; }
    return false;
  };
  auto minus = positions, plus = positions;
  constexpr double delta = 1e-5;
  double contraction = 0;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate)
  {
    const double direction = 0.03 * (static_cast<int>(coordinate % 7) - 3);
    minus[coordinate] -= delta * direction;
    plus[coordinate] += delta * direction;
    contraction += result[coordinate] * direction;
  }
  double objectives[2]{};
  for (std::size_t side = 0; side < 2; ++side)
  {
    forward.evaluate(side == 0 ? minus : plus);
    if (!stale_rejected())
      throw std::runtime_error("Forward reevaluation exposed a stale coordinate gradient");
    std::vector<double> phase(count), values(count), ratios;
    forward.downloadResults(phase, values, ratios);
    for (const double logabs : values)
      objectives[side] += logabs;
  }
  const auto difference = (objectives[1] - objectives[0]) / (2 * delta);
  if (!device_math::isFiniteBinary64(contraction) || !device_math::isFiniteBinary64(difference) ||
      ::fabs(contraction - difference) > 2e-6 * (1 + ::fabs(difference)))
    throw std::runtime_error("Complete coordinate gradient differs from log-amplitude directional difference");

  // A failed forward and a late geometry failure each revoke the whole result;
  // the next ordinary request must recover without allocating another graph.
  forward.evaluate(positions);
  gradient.reverse(forward);
  bool failed_forward = false;
  auto invalid = positions;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  try { forward.evaluate(invalid); }
  catch (const std::exception&) { failed_forward = true; }
  if (!failed_forward || !stale_rejected())
    throw std::runtime_error("Failed forward exposed a stale coordinate gradient");
  auto coalesced = positions;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    coalesced[(count - 1) * ne * 3 + dimension] = cpu_model.cfg.nuclei.x[dimension];
  bool failed_geometry = false;
  try { forward.evaluate(coalesced); gradient.reverse(forward); }
  catch (const std::exception&) { failed_geometry = true; }
  if (!failed_geometry || !stale_rejected())
    throw std::runtime_error("Late coordinate geometry failure published a partial result");
  forward.evaluate(positions);
  gradient.reverse(forward);

  // Weight successors share metadata but invalidate every borrowed tape/output.
  const auto canonical = cpu_model.p.flat_values();
  auto successor = image->makeUpdatedImage(canonical, image->version() + 1);
  gradient.rebindModel(successor);
  bool old_tape = false;
  try { gradient.reverse(forward); }
  catch (const std::invalid_argument&) { old_tape = true; }
  if (!old_tape || !stale_rejected())
    throw std::runtime_error("Coordinate rebind admitted an old image tape");
  forward.rebindModel(successor);
  forward.evaluate(positions);
  gradient.reverse(forward);

  // One singular channel with a finite total, then a total node, must both fail
  // under the existing all-channel inverse-admission convention.
  auto parameters = canonical;
  const auto& weight = image->plan().parameter(ne == s.spin_down_electrons
      ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT);
  for (std::size_t feature = 0; feature < s.feature_dimension; ++feature)
    for (std::size_t orbital = 0; orbital < ne; ++orbital)
      parameters[weight.begin + feature * s.determinants * ne + orbital] = 0;
  for (std::size_t failure = 0; failure < 2; ++failure)
  {
    if (failure)
      std::fill(parameters.begin() + weight.begin, parameters.begin() + weight.end, 0.0);
    successor = successor->makeUpdatedImage(parameters, successor->version() + 1);
    gradient.rebindModel(successor);
    forward.rebindModel(successor);
    bool rejected = false;
    try { forward.evaluate(positions); gradient.reverse(forward); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected || !stale_rejected())
      throw std::runtime_error("Coordinate gradient admitted a singular channel or total node");
  }
  successor = successor->makeUpdatedImage(canonical, successor->version() + 1);
  gradient.rebindModel(successor);
  forward.rebindModel(successor);
  forward.evaluate(positions);
  gradient.reverse(forward);
  if (gradient.deviceWorkspaceBytes() != bytes || gradient.hostWorkspaceBytes() != host_bytes)
    throw std::runtime_error("Coordinate gradient recovery changed prepared storage");
  report();
}

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_VALIDATION_H
