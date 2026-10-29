//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_TARGET_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_TARGET_VALIDATION_H

#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"
#include "PsiFormerDeviceMixedComplexSpatialValidation.h"
#include "PsiFormerDeviceBatchBackend.h"
#include <cstdlib>
#include <fstream>
#include <optional>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace qmcplusplus::psiformer::device::mixed_target_validation
{

/// Keep each changed-seam assertion explicit in the focused actual-device entry point.
inline void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

/// Read the private unbuffered interposer receipt, binding actual injections to one scalar case.
inline std::size_t tailInjectionCount(const char* label)
{
  const char* path = std::getenv("T10_TARGET_FAULT_RECEIPT");
  require(path && path[0] == '/', "Late target fault lacks its exclusive receipt path");
  std::ifstream receipt(path);
  require(bool(receipt), "Late target fault receipt was not created by the interposer");
  const std::string prefix = std::string("T10_TARGET_FAULT_INJECT case=") + label + ' ';
  std::size_t count = 0;
  std::string line;
  while (std::getline(receipt, line))
    if (line.compare(0, prefix.size(), prefix) == 0) ++count;
  return count;
}

/// Compare complete same-recipe results, including signed-zero output bits.
template<class T>
inline bool identical(const std::vector<T>& first, const std::vector<T>& second)
{
  return first.size() == second.size() &&
      (first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0);
}

/// Reuse the existing mixed-core physical bounds for every live configuration.
inline void near(double actual, double reference, double tolerance)
{
  require(device_math::isFiniteBinary64(actual) && device_math::isFiniteBinary64(reference) &&
      std::abs(actual - reference) <= tolerance * (1 + std::abs(reference)),
      "Mixed target physical result exceeds its existing core bound");
}

/// Hold a complete physical REAL result rather than a partially scattered slab.
struct RealFields
{
  std::vector<double> phases, jets, ratios;
};

/// Download the full public layout, whose extent is independent of selected K.
template<class Graph>
inline RealFields realFields(Graph& graph, std::size_t batch, std::size_t n)
{
  RealFields result{std::vector<double>(batch),
      std::vector<double>(spatialJetSpanElements(graph.resultLayout())), std::vector<double>(batch * n)};
  graph.downloadResults(result.phases, result.jets, result.ratios);
  return result;
}

/// A repeated REAL request must preserve every physical output bit.
inline bool identical(const RealFields& first, const RealFields& second)
{
  return identical(first.phases, second.phases) && identical(first.jets, second.jets) &&
      identical(first.ratios, second.ratios);
}

/// Check REAL log/G/L and kinetic fields for all B, including a non-dividing K tail.
inline void realAccuracy(const RealFields& actual, const RealFields& reference,
    const SpatialJetLayout& layout, std::size_t batch, std::size_t n)
{
  for (std::size_t b = 0; b < batch; ++b)
  {
    near(actual.phases[b], reference.phases[b], 0);
    near(actual.jets[layout.uncheckedValueOffset(b, 0)], reference.jets[layout.uncheckedValueOffset(b, 0)], 5e-5);
    double kinetic = 0, reference_kinetic = 0;
    for (std::size_t e = 0; e < n; ++e)
    {
      for (std::size_t axis = 0; axis < 3; ++axis)
        near(actual.jets[layout.uncheckedGradientOffset(b, 3 * e + axis, 0)],
            reference.jets[layout.uncheckedGradientOffset(b, 3 * e + axis, 0)], 5e-4);
      near(actual.jets[layout.uncheckedLaplacianOffset(b, e, 0)],
          reference.jets[layout.uncheckedLaplacianOffset(b, e, 0)], 1e-3);
      near(actual.ratios[b * n + e], reference.ratios[b * n + e], 1e-3);
      kinetic -= .5 * actual.ratios[b * n + e];
      reference_kinetic -= .5 * reference.ratios[b * n + e];
    }
    near(kinetic, reference_kinetic, 1e-3);
  }
}

/// Preserve paired-complex amplitude and every G/L/ratio component in separate owned outputs.
struct ComplexFields
{
  std::vector<ComplexValueResult> amplitude;
  std::vector<cuDoubleComplex> gradient, lap_log, lap_ratio;
};

/// Copy only the current complete paired publication ticket.
template<class Graph>
inline ComplexFields complexFields(Graph& graph, std::size_t batch, std::size_t n)
{
  ComplexFields result{std::vector<ComplexValueResult>(batch), std::vector<cuDoubleComplex>(3 * batch * n),
      std::vector<cuDoubleComplex>(batch * n), std::vector<cuDoubleComplex>(batch * n)};
  graph.download(graph.resultView(), result.amplitude, result.gradient, result.lap_log, result.lap_ratio);
  return result;
}

/// All modeled paired result fields must repeat exactly for one frozen recipe.
inline bool identical(const ComplexFields& first, const ComplexFields& second)
{
  return identical(first.amplitude, second.amplitude) && identical(first.gradient, second.gradient) &&
      identical(first.lap_log, second.lap_log) && identical(first.lap_ratio, second.lap_ratio);
}

/// Check actual complex magnitude rather than treating paired channels as independent REAL errors.
inline void complexAccuracy(const ComplexFields& actual, const ComplexFields& reference,
    std::size_t batch, std::size_t n)
{
  const auto complex = [](cuDoubleComplex value) { return std::complex<double>(cuCreal(value), cuCimag(value)); };
  const auto near_complex = [&](cuDoubleComplex a, cuDoubleComplex b, double tolerance) {
    const auto x = complex(a), y = complex(b);
    require(device_math::isFiniteBinary64(x.real()) && device_math::isFiniteBinary64(x.imag()) &&
        device_math::isFiniteBinary64(y.real()) && device_math::isFiniteBinary64(y.imag()) &&
        std::abs(x - y) <= tolerance * (1 + std::abs(y)), "Mixed target complex field exceeds its existing core bound");
  };
  for (std::size_t b = 0; b < batch; ++b)
  {
    near(actual.amplitude[b].log_abs, reference.amplitude[b].log_abs, 5e-5);
    near_complex(actual.amplitude[b].phase, reference.amplitude[b].phase, 5e-5);
    std::complex<double> kinetic = 0, reference_kinetic = 0;
    for (std::size_t e = 0; e < n; ++e)
    {
      for (std::size_t axis = 0; axis < 3; ++axis)
        near_complex(actual.gradient[3 * (b * n + e) + axis], reference.gradient[3 * (b * n + e) + axis], 5e-4);
      near_complex(actual.lap_log[b * n + e], reference.lap_log[b * n + e], 1e-3);
      near_complex(actual.lap_ratio[b * n + e], reference.lap_ratio[b * n + e], 1e-3);
      kinetic -= .5 * complex(actual.lap_ratio[b * n + e]);
      reference_kinetic -= .5 * complex(reference.lap_ratio[b * n + e]);
    }
    require(std::abs(kinetic - reference_kinetic) <= 1e-3 * (1 + std::abs(reference_kinetic)),
        "Mixed target complex kinetic exceeds its existing core bound");
  }
}

/// Query a complete-owner target cap before allocating either REAL numerical route.
inline TargetVglPreparation realPreparation(const PsiFormerExecutionPlan& plan, std::size_t batch, std::size_t k)
{
  TargetVglPreparation preparation{};
  preparation.reuse_cusp_state = false;
  if (k)
  {
    preparation.max_targets = k;
    preparation.scalar_work_budget = mixedTargetRequestScalarWork(
        estimateTargetVglCost(plan, batch, k).charged_scalar_work_units);
    preparation.device_workspace_budget_bytes = PreparedDeviceMixedInference::estimateTargetDeviceWorkspaceBytes(
        plan, batch, preparation);
  }
  return preparation;
}

/// Query one private paired target graph/cast cap; its immutable cache is charged once separately.
inline TargetVglPreparation complexPreparation(const PsiFormerExecutionPlan& plan, std::size_t batch,
    std::size_t k, std::size_t concurrent)
{
  TargetVglPreparation preparation{};
  preparation.reuse_cusp_state = false;
  if (k)
  {
    preparation.max_targets = k;
    preparation.scalar_work_budget = mixedTargetRequestScalarWork(
        estimateHeadFreeTargetPrefixCost(plan, k, batch).charged_scalar_work_units);
    // Pure cost query uses the existing private2GiB finite ceiling; construction below uses exact bytes.
    preparation.device_workspace_budget_bytes = std::size_t(2) * 1024 * 1024 * 1024;
    preparation.device_workspace_budget_bytes = PreparedDeviceMixedComplexSpatial::estimatePrivateDeviceBytes(
        plan, batch, concurrent, preparation);
  }
  return preparation;
}

/// Verify an invalid complete-owner cap cannot allocate even the first private graph.
template<class Construct>
inline void capRejectedWithoutAllocation(DeviceMemoryMetrics& metrics, Construct construct)
{
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { construct(); } catch (const std::length_error&) { rejected = true; }
  require(rejected && metrics.snapshot().allocation_calls == before.allocation_calls,
      "Mixed target cap-minus-one was not rejected before allocation");
}

/// Ordinary FULL and selected K share the same tracked warmed-allocation contract.
template<class Evaluate>
inline void warmed(DeviceMemoryMetrics& metrics, compute::Queue<PlatformKind::CUDA>& queue, Evaluate evaluate)
{
  queue.sync();
  const auto before = metrics.snapshot();
  evaluate();
  evaluate();
  queue.sync();
  const auto after = metrics.snapshot();
  require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls &&
      after.live_bytes == before.live_bytes, "Warmed mixed FULL/target changed tracked DeviceBuffer storage");
}

/// Parse an exact diagnostic token; missing, malformed or overflowing counters fail the fixture.
inline std::uint64_t diagnosticCount(const std::string& reason, const char* key)
{
  const std::string token = std::string(key) + '=';
  const auto position = reason.find(token);
  require(position != std::string::npos && (position == 0 || reason[position - 1] == ' '),
      "Condition-only rejection is missing a bounded diagnostic token");
  const auto begin = position + token.size();
  auto end = begin;
  while (end < reason.size() && reason[end] >= '0' && reason[end] <= '9') ++end;
  require(end > begin && (end == reason.size() || reason[end] == ' '),
      "Condition-only rejection contains a malformed diagnostic counter");
  return std::stoull(reason.substr(begin, end - begin));
}

/// Require one new spatial hazard per live configuration and no other numerical hazard.
inline bool conditionOnly(const std::string& reason, std::size_t batch)
{
  if (diagnosticCount(reason, "ill_conditioned_spatial") != batch) return false;
  for (const char* key : {"execution_nonfinite", "invalid_softmax", "small_determinant_pivot",
       "severe_cancellation", "extreme_ecp_ratio", "conversion_nonfinite", "conversion_overflow",
       "conversion_subnormal", "conversion_underflow_to_zero"})
    if (diagnosticCount(reason, key) != 0) return false;
  return true;
}

/// Isolate the existing FULL spatial condition screen with one deterministic common-column scale.
inline void guardedNegative(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& values, const std::vector<double>& positions, std::size_t batch,
    std::size_t k, bool paired, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
    DeviceMemoryMetrics& metrics)
{
  const auto& plan = image->plan();
  const auto n = plan.modelShape().electrons(), d = plan.modelShape().determinants;
  auto negative = values;
  for (const auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    for (const auto component : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
    {
      if (!paired && component == HeadTensorComponent::IMAG) continue;
      const auto& tensor = plan.parameter(role, NO_ATTENTION_BLOCK, component);
      for (std::size_t row = 0; row < tensor.shape[0]; ++row)
        for (std::size_t determinant = 0; determinant < d; ++determinant)
          negative[tensor.begin + row * d * n + determinant * n] *= .02;
    }
  auto negative_image = image->makeUpdatedImage(negative, image->version() + 1);
  if (!paired)
  {
    const auto preparation = realPreparation(plan, batch, k);
    PreparedDeviceInference canonical(negative_image, batch, queue, blas, SpatialJetMode::FULL_VGL,
        false, false, configuredInferenceStatusPolicy(), preparation);
    canonical.evaluate(positions);
    const auto expected = realFields(canonical, batch, n);
    PreparedDeviceMixedInference owner(negative_image, batch, queue, blas, SpatialJetMode::FULL_VGL, true, preparation);
    owner.evaluate(positions);
    require(owner.usedFp64Retry() && conditionOnly(owner.retryReason(), batch) &&
        identical(realFields(owner, batch, n), expected), "REAL target condition-only whole-request replay failed");
    warmed(metrics, queue, [&] {
      owner.evaluate(positions);
      require(owner.usedFp64Retry() && identical(realFields(owner, batch, n), expected),
          "Warmed REAL target replay changed canonical physical bits");
    });
    PreparedDeviceMixedInference strict(negative_image, batch, queue, blas, SpatialJetMode::FULL_VGL, false, preparation);
    bool rejected = false;
    try { strict.evaluate(positions); } catch (const MixedPrecisionRetryRequired&) { rejected = true; }
    require(rejected && !strict.usedFp64Retry(), "Strict REAL target did not propagate its numerical rejection");
    rejected = false;
    try { (void)realFields(strict, batch, n); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Strict REAL target exposed a partial physical publication");
  }
  else
  {
    constexpr std::size_t concurrent = 4;
    const auto preparation = complexPreparation(plan, batch, k, concurrent);
    const auto budget = PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, batch, concurrent, preparation);
    auto canonical = k ? PreparedDeviceComplexSpatial::prepare(negative_image, queue, blas, preparation, batch, concurrent)
                       : PreparedDeviceComplexSpatial::prepare(negative_image, batch, queue, blas,
                           estimateComplexSpatialCost(plan, batch, concurrent).device_bytes, concurrent);
    canonical->evaluate(positions);
    const auto expected = complexFields(*canonical, batch, n);
    PreparedDeviceMixedComplexSpatial owner(negative_image, batch, queue, blas, budget, true, concurrent, {}, preparation);
    owner.evaluate(positions);
    require(owner.usedFp64Retry() && conditionOnly(owner.retryReason(), batch) &&
        identical(complexFields(owner, batch, n), expected), "Paired target condition-only whole-request replay failed");
    warmed(metrics, queue, [&] {
      owner.evaluate(positions);
      require(owner.usedFp64Retry() && identical(complexFields(owner, batch, n), expected),
          "Warmed paired target replay changed canonical physical bits");
    });
    const auto stale = owner.resultView();
    bool rejected = false;
    try { owner.evaluate(positions, true); } catch (const MixedPrecisionRetryRequired&) { rejected = true; }
    require(rejected && !owner.usedFp64Retry(), "Strict paired target did not propagate without local replay");
    rejected = false;
    try { (void)owner.resultView(); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Strict paired target exposed a partial physical publication");
    ComplexFields output{std::vector<ComplexValueResult>(batch), std::vector<cuDoubleComplex>(3 * batch * n),
        std::vector<cuDoubleComplex>(batch * n), std::vector<cuDoubleComplex>(batch * n)};
    rejected = false;
    try { owner.download(stale, output.amplitude, output.gradient, output.lap_log, output.lap_ratio); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Strict paired target retained an earlier canonical ticket");
  }
  std::cout << "PASS mixed target guarded paired=" << paired << " K=" << k << " B=" << batch
            << " spatial_count=" << batch << " old_hazards=0 conversion_hazards=0"
            << " complete_canonical_replay_bits=exact strict_no_publication=1 warm_DeviceBuffer_alloc_free_delta=0\n";
}

/// Rebind the native REAL one-graph seam with two changed leaves and the same frozen target policy.
inline void realRebind(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& values, const std::vector<double>& positions, std::size_t batch,
    TargetVglPreparation preparation, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto& plan = image->plan();
  const auto n = plan.modelShape().electrons();
  PreparedDeviceInference graph(image, batch, queue, blas, SpatialJetMode::FULL_VGL,
      false, false, configuredInferenceStatusPolicy(), preparation);
  auto cache = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue);
  auto adapter = std::make_shared<PreparedDeviceMixedValueDense>(cache, graph.mixedDenseRowCapacity(), queue, blas);
  graph.setMixedDense(adapter);
  std::string retry_reason;
  (void)graph.evaluateWithFp64Retry(positions, retry_reason);
  auto next = values;
  next[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += .001;
  next[plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin] += .001;
  auto successor = image->makeUpdatedImage(next, image->version() + 1);
  auto successor_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(successor, queue);
  graph.validateModelRebindWithMixedParameters(successor, successor_cache);
  graph.adoptPreparedModelWithMixedParameters(successor, successor_cache);
  bool revoked = false;
  try { (void)realFields(graph, batch, n); } catch (const std::logic_error&) { revoked = true; }
  require(revoked, "REAL target rebind retained predecessor physical output");
  (void)graph.evaluateWithFp64Retry(positions, retry_reason);
  PreparedDeviceInference fresh(successor, batch, queue, blas, SpatialJetMode::FULL_VGL,
      false, false, configuredInferenceStatusPolicy(), preparation);
  auto fresh_adapter = std::make_shared<PreparedDeviceMixedValueDense>(successor_cache,
      fresh.mixedDenseRowCapacity(), queue, blas);
  fresh.setMixedDense(fresh_adapter);
  (void)fresh.evaluateWithFp64Retry(positions, retry_reason);
  require(identical(realFields(graph, batch, n), realFields(fresh, batch, n)),
      "REAL target rebind retained stale converted parameters or scratch state");
}

/// Exercise the actual REAL backend's selected-K preparation, strict epoch and publication/rebind path.
inline void realNativeBinding(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& values, const std::vector<double>& positions)
{
  const auto& plan = image->plan();
  const auto n = plan.modelShape().electrons();
  DeviceExecutionPreparation preparation{};
  preparation.vgl = realPreparation(plan, 1, 1);
  PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, preparation, false);
  auto workspace = backend.prepare(1);
  const auto copy = [&](const pf::DirectBatchTileResultView& view, std::size_t version) {
    require(view.parameter_version == version, "REAL native target published a stale parameter version");
    require(view.size == 1 && view.gradient && view.lap_log && view.lap_ratio,
        "REAL native target did not publish complete physical fields");
    std::vector<double> fields{view.sign[0], view.logabs[0]};
    for (std::size_t lane = 0; lane < 3 * n; ++lane) fields.push_back(view.gradient[lane * view.gradient_lane_stride]);
    for (std::size_t lane = 0; lane < n; ++lane)
    {
      fields.push_back(view.lap_log[lane * view.lap_log_lane_stride]);
      fields.push_back(view.lap_ratio[lane]);
    }
    for (double value : fields) require(device_math::isFiniteBinary64(value), "REAL native target published a nonfinite field");
    return fields;
  };
  const auto epoch = backend.enterTrainingEpoch(image->version(), PsiFormerGpuTrainingEpochMode::STRICT_MIXED);
  (void)copy(backend.evaluate(*workspace, pf::DirectBatchMode::FULL_VGL, positions.data(), 1, image->version()), image->version());
  auto next = values;
  next[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += .001;
  next[plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin] += .001;
  bool committed = false, rejected = false;
  try { backend.publishParameters(next, image->version() + 1, [&] { committed = true; }); }
  catch (const std::logic_error&) { rejected = true; }
  require(rejected && !committed, "REAL target native publication bypassed its active strict epoch");
  backend.leaveTrainingEpoch(epoch);
  backend.publishParameters(next, image->version() + 1, [&] { committed = true; });
  require(committed, "REAL target native idle publication did not commit");
  const auto after = copy(backend.evaluate(*workspace, pf::DirectBatchMode::FULL_VGL,
      positions.data(), 1, image->version() + 1), image->version() + 1);
  auto successor = image->makeUpdatedImage(next, image->version() + 1);
  PreparedDeviceBatchBackend fresh(successor, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, preparation, false);
  auto fresh_workspace = fresh.prepare(1);
  const auto expected = copy(fresh.evaluate(*fresh_workspace, pf::DirectBatchMode::FULL_VGL,
      positions.data(), 1, successor->version()), successor->version());
  require(identical(after, expected), "Native REAL target rebind differs from fresh same-image owner bits");
  std::cout << "PASS native REAL mixed selected-K binding K=1 strict_epoch=1 idle_publication=1 fresh_rebind_bits=exact\n";
}

/// Run changed REAL and paired seams on the existing localized-head fixture, not a searched model.
inline void validate()
{
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "mixed target validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "mixed target validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const char* saved_label = std::getenv("T10_TARGET_FAULT_CASE");
  const std::optional<std::string> previous_label = saved_label ? std::optional<std::string>(saved_label) : std::nullopt;
  device_determinant::CombinationMetadata combination{};
  combination.status = device_determinant::CombinationStatus::REGULAR;
  for (bool mixed : {false, true})
  {
    bool rejected = false;
    try { admitTargetVglBase(1., .001, 1., 0., combination, mixed); }
    catch (const MixedPrecisionRetryRequired&) { require(mixed, "FP64 base bound changed exception class"); rejected = true; }
    catch (const std::runtime_error&) { require(!mixed, "Mixed base bound lost typed retry identity"); rejected = true; }
    require(rejected, "Cross-slab base disagreement did not preserve the existing bound");
  }
  bool work_overflow = false;
  try { (void)mixedTargetRequestScalarWork(std::numeric_limits<std::size_t>::max() / 2 + 1); }
  catch (const std::length_error&) { work_overflow = true; }
  require(work_overflow, "Mixed complete-request retry work overflow did not throw");
  const ModelShape shape{2, 2, 4, 2, 8, 1, 1};
  const std::vector<double> nuclei{0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2};
  const std::vector<double> first{.1, .2, .3, 2.1, .2, .3, .1, 2.2, .3, .1, .2, 2.3};
  for (bool paired : {false, true})
  {
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.periodic_axes = {true, true, true};
    environment.lattice_vectors = {{{8., 0., 0.}, {0., 8., 0.}, {0., 0., 8.}}};
    if (paired)
    {
      environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
      environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
    }
    const auto parameters = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
    const auto values = parameters.flat_values();
    std::shared_ptr<const PreparedDeviceModel> image = paired
        ? PreparedDeviceModel::preparePairedComplexValue(plan, values, nuclei)
        : std::make_shared<PreparedDeviceModel>(plan, values, nuclei);
    for (const auto& request : {std::pair<std::size_t, std::size_t>{0, 1}, {1, 1}, {3, 2}})
    {
      const auto k = request.first, batch = request.second;
      auto positions = first;
      if (batch == 2)
      {
        positions.insert(positions.end(), first.begin(), first.end());
        positions[12] += .031;
      }
      // The private interposer arm must prove actual late-tail retry; no production code reads this switch.
      const char* fault_switch = std::getenv("T10_TARGET_TAIL_FAULT");
      const bool fault_run = fault_switch && std::string_view(fault_switch) == "1";
      const bool late_fault = k == 3 && fault_run;
      const char* fault_label = paired ? "paired_K3_B2_C4" : "real_K3_B2";
      if (fault_run) setenv("T10_TARGET_FAULT_CASE", late_fault ? fault_label : "control_FULL_or_K1", 1);
      const auto before_initial_injections = late_fault ? tailInjectionCount(fault_label) : 0;
      bool used_request_retry = false;
      std::string request_retry_reason;
      if (!paired)
      {
        const auto preparation = realPreparation(plan, batch, k);
        if (k)
        {
          auto short_cap = preparation;
          --short_cap.device_workspace_budget_bytes;
          capRejectedWithoutAllocation(metrics, [&] {
            PreparedDeviceMixedInference rejected(image, batch, queue, blas, SpatialJetMode::FULL_VGL, false, short_cap);
          });
          auto short_work = preparation;
          --short_work.scalar_work_budget;
          capRejectedWithoutAllocation(metrics, [&] {
            PreparedDeviceMixedInference rejected(image, batch, queue, blas, SpatialJetMode::FULL_VGL, true, short_work);
          });
          if (k == 1)
          {
            auto single_attempt = preparation;
            single_attempt.scalar_work_budget /= 2;
            capRejectedWithoutAllocation(metrics, [&] {
              PreparedDeviceMixedInference rejected(image, batch, queue, blas, SpatialJetMode::FULL_VGL, true, single_attempt);
            });
          }
        }
        PreparedDeviceMixedInference owner(image, batch, queue, blas, SpatialJetMode::FULL_VGL, k == 3, preparation);
        PreparedDeviceInference reference(image, batch, queue, blas, SpatialJetMode::FULL_VGL,
            false, false, configuredInferenceStatusPolicy(), preparation);
        reference.evaluate(positions);
        const auto expected = realFields(reference, batch, shape.electrons());
        owner.evaluate(positions);
        used_request_retry = owner.usedFp64Retry();
        request_retry_reason = owner.retryReason();
        require((k == 3 || !used_request_retry) && owner.statistics().projections > 0,
            "Stable REAL K1/FULL did not execute actual mixed arithmetic");
        const auto accepted = realFields(owner, batch, shape.electrons());
        if (late_fault) require(used_request_retry && tailInjectionCount(fault_label) == before_initial_injections + 1,
            "REAL tail fault arm lacks exactly one actual per-case late injection/retry");
        if (used_request_retry) require(identical(accepted, expected), "REAL tail retry changed canonical same-K bits");
        realAccuracy(accepted, expected, owner.resultLayout(), batch, shape.electrons());
        warmed(metrics, queue, [&] {
          const auto before_injections = late_fault ? tailInjectionCount(fault_label) : 0;
          owner.evaluate(positions);
          if (late_fault) require(tailInjectionCount(fault_label) == before_injections + 1,
              "Warmed REAL tail request did not receive its actual per-case injection");
          require(owner.usedFp64Retry() == used_request_retry && identical(realFields(owner, batch, shape.electrons()), accepted),
              "Warmed REAL target changed its admitted recipe or result bits");
        });
        if (late_fault)
        {
          PreparedDeviceMixedInference strict(image, batch, queue, blas, SpatialJetMode::FULL_VGL, false, preparation);
          const auto before_strict_injections = tailInjectionCount(fault_label);
          bool rejected = false;
          try { strict.evaluate(positions); }
          catch (const MixedPrecisionRetryRequired&) { rejected = true; }
          catch (const InferenceStatusError& error) { if (error.stage() == 0) throw; rejected = true; }
          require(rejected && !strict.usedFp64Retry() &&
              tailInjectionCount(fault_label) == before_strict_injections + 1,
              "Late REAL strict failure lacks an actual injection or silently replayed");
          rejected = false;
          try { (void)realFields(strict, batch, shape.electrons()); }
          catch (const std::logic_error&) { rejected = true; }
          require(rejected, "Late REAL strict failure exposed a partial physical result");
          std::cout << "PASS late REAL target strict_no_publication=1 same_K_whole_replay_bits=exact\n";
        }
        if (k == 3) realRebind(image, values, positions, batch, preparation, queue, blas);
      }
      else
      {
        constexpr std::size_t concurrent = 4;
        const auto preparation = complexPreparation(plan, batch, k, concurrent);
        const auto budget = PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, batch, concurrent, preparation);
        capRejectedWithoutAllocation(metrics, [&] {
          PreparedDeviceMixedComplexSpatial rejected(image, batch, queue, blas, budget - 1, false, concurrent, {}, preparation);
        });
        if (k)
        {
          auto short_work = preparation;
          --short_work.scalar_work_budget;
          capRejectedWithoutAllocation(metrics, [&] {
            PreparedDeviceMixedComplexSpatial rejected(image, batch, queue, blas, budget, true, concurrent, {}, short_work);
          });
          if (k == 1)
          {
            auto single_attempt = preparation;
            single_attempt.scalar_work_budget /= 2;
            capRejectedWithoutAllocation(metrics, [&] {
              PreparedDeviceMixedComplexSpatial rejected(image, batch, queue, blas, budget, true, concurrent, {}, single_attempt);
            });
          }
        }
        PreparedDeviceMixedComplexSpatial owner(image, batch, queue, blas, budget, k == 3, concurrent, {}, preparation);
        auto reference = k ? PreparedDeviceComplexSpatial::prepare(image, queue, blas, preparation, batch, concurrent)
                           : PreparedDeviceComplexSpatial::prepare(image, batch, queue, blas,
                               estimateComplexSpatialCost(plan, batch, concurrent).device_bytes, concurrent);
        reference->evaluate(positions);
        const auto expected = complexFields(*reference, batch, shape.electrons());
        owner.evaluate(positions, k != 3);
        used_request_retry = owner.usedFp64Retry();
        request_retry_reason = owner.retryReason();
        require((k == 3 || !used_request_retry) && owner.statistics().projections > 0,
            "Stable paired K1/FULL did not execute actual mixed arithmetic");
        const auto accepted = complexFields(owner, batch, shape.electrons());
        if (late_fault) require(used_request_retry && tailInjectionCount(fault_label) == before_initial_injections + 1,
            "Paired tail fault arm lacks exactly one actual per-case late injection/retry");
        if (used_request_retry) require(identical(accepted, expected), "Paired tail retry changed canonical same-K/C bits");
        complexAccuracy(accepted, expected, batch, shape.electrons());
        warmed(metrics, queue, [&] {
          const auto before_injections = late_fault ? tailInjectionCount(fault_label) : 0;
          owner.evaluate(positions, k != 3);
          if (late_fault) require(tailInjectionCount(fault_label) == before_injections + 1,
              "Warmed paired tail request did not receive its actual per-case injection");
          require(owner.usedFp64Retry() == used_request_retry && identical(complexFields(owner, batch, shape.electrons()), accepted),
              "Warmed paired target changed its admitted recipe or result bits");
        });
        if (late_fault)
        {
          const auto prior = owner.resultView();
          const auto before_strict_injections = tailInjectionCount(fault_label);
          bool rejected = false;
          try { owner.evaluate(positions, true); }
          catch (const MixedPrecisionRetryRequired&) { rejected = true; }
          catch (const MixedComplexValueFailure& error)
          {
            if (error.stage() == ComplexValueFailureStage::PHYSICAL_ENVELOPE) throw;
            rejected = true;
          }
          require(rejected && !owner.usedFp64Retry() &&
              tailInjectionCount(fault_label) == before_strict_injections + 1,
              "Late paired strict failure lacks an actual injection or silently replayed");
          rejected = false;
          try { (void)owner.resultView(); } catch (const std::logic_error&) { rejected = true; }
          require(rejected, "Late paired strict failure exposed a partial physical result");
          ComplexFields discarded{std::vector<ComplexValueResult>(batch),
              std::vector<cuDoubleComplex>(3 * batch * shape.electrons()),
              std::vector<cuDoubleComplex>(batch * shape.electrons()), std::vector<cuDoubleComplex>(batch * shape.electrons())};
          rejected = false;
          try { owner.download(prior, discarded.amplitude, discarded.gradient, discarded.lap_log, discarded.lap_ratio); }
          catch (const std::logic_error&) { rejected = true; }
          require(rejected, "Late paired strict failure retained the earlier publication ticket");
          owner.evaluate(positions);
          std::cout << "PASS late paired target strict_no_publication=1 stale_ticket_revoked=1 same_K_C_whole_replay_bits=exact\n";
        }
        auto successor_values = values;
        successor_values[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += .001;
        successor_values[plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK,
            HeadTensorComponent::REAL).begin] += .001;
        auto successor = image->makeUpdatedImage(successor_values, 1);
        const auto stale = owner.resultView();
        owner.rebindModel(successor, owner.successorWorkspaceBytes());
        bool revoked = false;
        try { (void)owner.resultView(); } catch (const std::logic_error&) { revoked = true; }
        require(revoked, "Paired target rebind retained predecessor publication");
        owner.evaluate(positions, k != 3);
        PreparedDeviceMixedComplexSpatial fresh(successor, batch, queue, blas, budget, k == 3, concurrent, {}, preparation);
        fresh.evaluate(positions, k != 3);
        require(identical(complexFields(owner, batch, shape.electrons()), complexFields(fresh, batch, shape.electrons())),
            "Paired target rebind retained stale converted weights or graph state");
        ComplexFields stale_fields{std::vector<ComplexValueResult>(batch), std::vector<cuDoubleComplex>(3 * batch * shape.electrons()),
            std::vector<cuDoubleComplex>(batch * shape.electrons()), std::vector<cuDoubleComplex>(batch * shape.electrons())};
        revoked = false;
        try { owner.download(stale, stale_fields.amplitude, stale_fields.gradient, stale_fields.lap_log, stale_fields.lap_ratio); }
        catch (const std::logic_error&) { revoked = true; }
        require(revoked, "Paired target accepted a stale predecessor ticket");
      }
      if (k != 1) guardedNegative(image, values, positions, batch, k, paired, queue, blas, metrics);
      std::cout << "PASS mixed target stable paired=" << paired << " K=" << k << " B=" << batch
                << " actual_mixed=" << !used_request_retry << " whole_request_FP64_retry=" << used_request_retry
                << " tail=" << bool(k && shape.electrons() % k)
                << " warm_DeviceBuffer_alloc_free_delta=0 no_opaque_vendor_claim=1 retry_reason="
                << request_retry_reason << '\n';
    }
    if (!paired) realNativeBinding(image, values, first);
    bool overflow = false;
    try
    {
      if (paired)
        (void)PreparedDeviceMixedComplexSpatial::maximumRows(plan, std::numeric_limits<std::size_t>::max(),
            complexPreparation(plan, 1, 1, 4));
      else
        (void)PreparedDeviceMixedInference::estimateTargetDeviceWorkspaceBytes(plan,
            std::numeric_limits<std::size_t>::max(), realPreparation(plan, 1, 1));
    }
    catch (const std::length_error&) { overflow = true; }
    require(overflow, "Mixed selected-K row/cost overflow did not throw before allocation");
  }
  if (previous_label) setenv("T10_TARGET_FAULT_CASE", previous_label->c_str(), 1);
  else unsetenv("T10_TARGET_FAULT_CASE");
}

} // namespace qmcplusplus::psiformer::device::mixed_target_validation

/// Invoke only the changed mixed FULL/selected-K seams from the focused hardware target.
inline void validateMixedTargetStreaming()
{
  qmcplusplus::psiformer::device::mixed_target_validation::validate();
}

#endif
