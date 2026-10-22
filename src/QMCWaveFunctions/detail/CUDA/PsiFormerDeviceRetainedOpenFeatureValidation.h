//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_OPEN_FEATURE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_OPEN_FEATURE_VALIDATION_H

#include "PsiFormerDeviceTargetKinetic.h"
#include "PsiFormerDeviceScoreReverse.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>

namespace qmcplusplus::psiformer::device
{
/// Inspect existing private retained tapes without adding a production accessor.
struct RetainedOpenFeatureValidationAccess
{
  /// Restore exact caller configuration after positive and negative admission tests.
  struct RestoreSelector
  {
    std::string name;
    std::optional<std::string> value;

    explicit RestoreSelector(const char* key) : name(key)
    {
      if (const char* current = std::getenv(key))
        value = current;
    }

    ~RestoreSelector()
    {
      if (value)
        setenv(name.c_str(), value->c_str(), 1);
      else
        unsetenv(name.c_str());
    }
  };

  /// Compare IEEE representations, including signed zero, only in defined lanes.
  static void exact(const std::vector<double>& actual, const std::vector<double>& expected)
  {
    if (actual.size() != expected.size() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)) != 0)
      throw std::runtime_error("Retained OPEN feature admission changed a defined tape/response lane");
    for (double value : actual)
      if (!device_math::isFiniteBinary64(value))
        throw std::runtime_error("Retained OPEN feature fixture produced a nonfinite lane");
  }

  /// Materialize defined raw, neural, determinant, root and status lanes immediately.
  static std::vector<double> tape(const PreparedDeviceInference& graph)
  {
    std::vector<double> result;
    const auto append = [&](const DeviceBuffer<double>& buffer, std::size_t count) {
      const auto begin = result.size();
      result.resize(begin + count);
      buffer.download(result.data() + begin, count);
    };
    append(graph.raw_, spatialJetSpanElements(graph.features_.output));
    append(graph.a_, spatialJetSpanElements(graph.dense_.source));
    append(graph.output_, spatialJetSpanElements(graph.final_.output));
    append(graph.phase_, graph.batch_);
    append(graph.normalized_, graph.batch_ * graph.shape_.determinants);
    append(graph.inverse_, graph.batch_ * graph.matrices_per_configuration_);
    if (graph.mode_ == SpatialJetMode::FULL_VGL)
    {
      append(graph.det_gradient_, 3 * graph.final_.output.laplacian_lanes);
      append(graph.det_lap_log_, graph.final_.output.laplacian_lanes);
    }
    std::vector<device_determinant::FactorizationMetadata> factors(graph.batch_ * graph.shape_.determinants);
    graph.factors_.download(factors.data(), factors.size());
    for (const auto& entry : factors)
    {
      result.insert(result.end(), {entry.phase, entry.log_abs, entry.matrix_scale,
          entry.minimum_scaled_pivot, entry.maximum_scaled_pivot,
          double(entry.status), double(entry.inverse_available)});
      if (entry.status != device_determinant::FactorizationStatus::REGULAR || !entry.inverse_available)
        throw std::runtime_error("Retained OPEN feature fixture lacks a regular inverse");
    }
    std::vector<OpenSpatialStatus> status(graph.batch_);
    graph.open_status_.download(status.data(), status.size());
    for (auto entry : status)
      result.push_back(double(entry));
    return result;
  }

  /// Build a small architectural graph, with localized envelopes for a regular head.
  static std::shared_ptr<const PreparedDeviceModel> fixture(std::size_t ne,
      std::vector<double>& positions)
  {
    const ModelShape shape{ne / 2, ne - ne / 2, ne, 1, 64, 4, 1};
    std::vector<ParameterLayoutInput> layouts;
    const auto append = [&](std::string module, std::string name, std::vector<std::size_t> dimensions) {
      std::size_t count = 1;
      for (auto dimension : dimensions)
        count *= dimension;
      const auto begin = layouts.empty() ? 0 : layouts.back().end;
      layouts.push_back({std::move(module), std::move(name), std::move(dimensions), begin, begin + count});
    };
    const std::string prefix = "neural_network_wave_function/~/";
    append(prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
    append(prefix + "electronic_cusp_asymptotic", "same_alpha", {});
    for (const auto* name : {"pi_down", "pi_up", "zetas_down", "zetas_up"})
      append(prefix + "exponential_envelopes", name, {ne, ne});
    for (const auto* backflow : {"Backflow", "Backflow_1"})
      append(prefix + "omni_net/~/" + backflow + "/~/mlp/linear_0", "w", {64, ne});
    append(prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w", {4 * ne + 1, 64});
    const auto base = prefix + "omni_net/~/electron_gnn/~/electron_gnn_layer/~/node_attention_electron_update_feature/";
    for (const auto* linear : {"linear_0", "linear_1"})
    {
      append(base + "mlp/" + linear, "b", {64});
      append(base + "mlp/" + linear, "w", {64, 64});
    }
    for (const auto* projection : {"key", "linear", "query", "value"})
      append(base + "multi_head_attention/" + projection, "w", {64, 64});
    const PsiFormerExecutionPlan plan(shape, layouts);
    std::vector<double> parameters(plan.parameterCount()), nuclei(3 * ne);
    std::uint64_t state = 17;
    for (double& value : parameters)
    {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      const auto random = state * UINT64_C(2685821657736338717);
      value = 0.2 * (double(random >> 11) / 9007199254740992.0 - 0.5);
    }
    for (const auto role : {ParameterRole::ENVELOPE_PI_UP, ParameterRole::ENVELOPE_PI_DOWN,
         ParameterRole::ENVELOPE_ZETA_UP, ParameterRole::ENVELOPE_ZETA_DOWN})
    {
      const auto& tensor = plan.parameter(role);
      const bool pi = role == ParameterRole::ENVELOPE_PI_UP || role == ParameterRole::ENVELOPE_PI_DOWN;
      for (std::size_t column = 0; column < ne; ++column)
        for (std::size_t nucleus = 0; nucleus < ne; ++nucleus)
          parameters[tensor.begin + column * ne + nucleus] = pi ? double(column == nucleus) : 2.0;
    }
    parameters[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] = 0.6;
    parameters[plan.parameter(ParameterRole::CUSP_SAME_ALPHA).begin] = 0.4;
    positions.resize(3 * ne);
    for (std::size_t electron = 0; electron < ne; ++electron)
    {
      nuclei[3 * electron] = 4.0 * (electron % 4);
      nuclei[3 * electron + 1] = 4.0 * ((electron / 4) % 4);
      nuclei[3 * electron + 2] = 4.0 * (electron / 16);
      for (std::size_t axis = 0; axis < 3; ++axis)
        positions[3 * electron + axis] = nuclei[3 * electron + axis] + 0.17 + 0.06 * axis;
    }
    return std::make_shared<PreparedDeviceModel>(plan, parameters, nuclei, 17);
  }

  /// Fail before reverse work if the fixture's actual matrix is ill-conditioned.
  static void qualifyMatrix(const PreparedDeviceInference& graph)
  {
    const auto n = graph.ne_;
    std::vector<double> matrix(n * n), inverse(n * n);
    graph.matrices_.download(matrix.data(), matrix.size());
    graph.inverse_.download(inverse.data(), inverse.size());
    double matrix_norm = 0, inverse_norm = 0, residual = 0;
    for (std::size_t row = 0; row < n; ++row)
    {
      double a_sum = 0, b_sum = 0;
      for (std::size_t column = 0; column < n; ++column)
      {
        a_sum += std::abs(matrix[row * n + column]);
        b_sum += std::abs(inverse[row * n + column]);
        double product = 0;
        for (std::size_t k = 0; k < n; ++k)
          product += matrix[row * n + k] * inverse[k * n + column];
        residual = std::max(residual, std::abs(product - double(row == column)));
      }
      matrix_norm = std::max(matrix_norm, a_sum);
      inverse_norm = std::max(inverse_norm, b_sum);
    }
    const auto condition = matrix_norm * inverse_norm;
    if (!device_math::isFiniteBinary64(condition) || condition > 1e6 || residual > 1e-8)
      throw std::runtime_error("Retained OPEN feature fixture conditioning preflight failed");
    std::cout << "Retained feature fixture Ne=" << n << " infinity_condition=" << condition
              << " inverse_residual=" << residual << '\n';
  }

  /// Compare retained score and compact physical-target products without FD sweeps.
  static void run(compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
  {
    RestoreSelector restore("PSIFORMER_GPU_RETAINED_OPEN_FEATURES");
    RestoreSelector ordinary("PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE");
    setenv("PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE", "1", 1);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "retained feature validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "retained feature validation device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    constexpr std::size_t trace_bytes = 1024 * 1024;
    for (std::size_t ne : {std::size_t(64), std::size_t(65)})
    {
      std::vector<double> positions;
      const auto image = fixture(ne, positions);
      const auto parameters = image->plan().parameterCount();
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "malformed", 1);
      const auto before_invalid = metrics.snapshot();
      bool rejected = false;
      try
      {
        PreparedDeviceInference invalid(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      }
      catch (const std::invalid_argument&)
      {
        rejected = true;
      }
      if (!rejected || metrics.snapshot().allocation_calls != before_invalid.allocation_calls)
        throw std::runtime_error("Malformed retained feature selector allocated a graph");

      // Existing ordinary flag1 must not implicitly admit the retained graph.
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "0", 1);
      PreparedDeviceInference reference(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "1", 1);
      PreparedDeviceInference selected(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "0", 1);
      if (reference.usesCooperativeOpenFeatures() || !selected.usesCooperativeOpenFeatures())
        throw std::runtime_error("Retained feature selection was not independent and frozen");
      reference.evaluate(positions);
      selected.evaluate(positions);
      exact(tape(selected), tape(reference));
      qualifyMatrix(reference); // Before score or compact response work.
      PreparedDeviceScoreReverse reverse(image, 1, queue, blas);
      std::vector<double> expected(parameters), actual(parameters);
      for (double weight : {0.7, -0.25, 0.0})
      {
        reverse.reverse(reference, {weight});
        reverse.downloadResponse(expected);
        reverse.reverse(selected, {weight});
        reverse.downloadResponse(actual);
        exact(actual, expected);
      }
      const auto before_warm = metrics.snapshot();
      selected.evaluate(positions);
      reverse.reverse(selected, {-0.25});
      reverse.downloadResponse(actual);
      if (metrics.snapshot().allocation_calls != before_warm.allocation_calls ||
          metrics.snapshot().live_bytes != before_warm.live_bytes)
        throw std::runtime_error("Retained score warmed request allocated");

      // Inspect every compact K3 tape immediately; Ne64/65 exercise ragged tails.
      const auto cost = estimateRetainedTargetForwardCost(image->plan(), 3);
      const TargetVglPreparation preparation{3, cost.device_workspace_bytes, cost.charged_scalar_work_units};
      const auto selected_cost = estimateRetainedTargetForwardCost(image->plan(), 3, trace_bytes);
      const TargetVglPreparation selected_preparation{3, selected_cost.device_workspace_bytes,
          selected_cost.charged_scalar_work_units, trace_bytes};
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "0", 1);
      PreparedDeviceInference compact_reference(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
          false, true, InferenceStatusPolicy::SYNCHRONOUS_DEBUG, preparation, true);
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "1", 1);
      PreparedDeviceInference compact_selected(image, 1, queue, blas, SpatialJetMode::FULL_VGL,
          false, true, InferenceStatusPolicy::SYNCHRONOUS_DEBUG, selected_preparation, true);
      if (compact_reference.usesCooperativeOpenFeatures() || !compact_selected.usesCooperativeOpenFeatures() ||
          compact_selected.deviceWorkspaceBytes() - compact_reference.deviceWorkspaceBytes() !=
              selected_cost.device_workspace_bytes - cost.device_workspace_bytes ||
          compact_selected.hostWorkspaceBytes() != compact_reference.hostWorkspaceBytes())
        throw std::runtime_error("Retained compact feature dispatch changed admission or owners");
      std::vector<std::vector<double>> target_reference;
      compact_reference.evaluateRetainedTargets(positions, [&](const auto&) {
        target_reference.push_back(tape(compact_reference));
      });
      std::size_t tile = 0;
      compact_selected.evaluateRetainedTargets(positions, [&](const auto&) {
        exact(tape(compact_selected), target_reference.at(tile++));
      });
      if (tile != (ne + 2) / 3)
        throw std::runtime_error("Retained compact feature target coverage changed");
      const auto compact_warm = metrics.snapshot();
      tile = 0;
      compact_selected.evaluateRetainedTargets(positions, [&](const auto&) {
        exact(tape(compact_selected), target_reference.at(tile++));
      });
      if (metrics.snapshot().allocation_calls != compact_warm.allocation_calls ||
          metrics.snapshot().live_bytes != compact_warm.live_bytes)
        throw std::runtime_error("Retained compact feature warmed tape allocated");

      // Exercise the unchanged public complete compact kinetic-product owner.
      const auto kinetic_cost = estimateTargetKineticCost(image->plan(), 3, 1);
      const TargetKineticPreparation kinetic_preparation{3, kinetic_cost.device_workspace_bytes,
          kinetic_cost.charged_scalar_work_units, 1};
      const auto selected_kinetic_cost = estimateTargetKineticCost(image->plan(), 3, 1, trace_bytes);
      const TargetKineticPreparation selected_kinetic_preparation{3, selected_kinetic_cost.device_workspace_bytes,
          selected_kinetic_cost.charged_scalar_work_units, 1, trace_bytes};
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "0", 1);
      PreparedDeviceTargetKinetic kinetic_reference(image, kinetic_preparation, queue, blas);
      setenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES", "1", 1);
      PreparedDeviceTargetKinetic kinetic_selected(image, selected_kinetic_preparation, queue, blas);
      kinetic_reference.evaluate(positions, {-0.25});
      kinetic_reference.downloadResponse(expected);
      kinetic_selected.evaluate(positions, {-0.25});
      kinetic_selected.downloadResponse(actual);
      exact(actual, expected);
      const auto response_warm = metrics.snapshot();
      kinetic_selected.evaluate(positions, {-0.25});
      kinetic_selected.downloadResponse(actual);
      exact(actual, expected);
      if (metrics.snapshot().allocation_calls != response_warm.allocation_calls ||
          metrics.snapshot().live_bytes != response_warm.live_bytes)
        throw std::runtime_error("Retained compact feature warmed response allocated");

      auto bad_positions = positions;
      bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
      bool failed = false, unavailable = false;
      try { kinetic_selected.evaluate(bad_positions, {-0.25}); }
      catch (const std::exception&) { failed = true; }
      try { kinetic_selected.downloadResponse(actual); }
      catch (const std::logic_error&) { unavailable = true; }
      if (!failed || !unavailable)
        throw std::runtime_error("Retained feature failure exposed a kinetic response");
      kinetic_selected.evaluate(positions, {-0.25});
      kinetic_selected.downloadResponse(actual);
      exact(actual, expected);
      selected.excludeCooperativeOpenFeaturesForMixedRequest();
      if (selected.usesCooperativeOpenFeatures())
        throw std::runtime_error("Retained feature mixed exclusion was lost");
      std::cout << "PASS retained OPEN feature Ne=" << ne
                << " score/raw/tape/status/compact_K3_ragged_response exact canonical_trace_1MiB frozen_selector zero_warm_allocation failure_recovery\n";
    }
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
