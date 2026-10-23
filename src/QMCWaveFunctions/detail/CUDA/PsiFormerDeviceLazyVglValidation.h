//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_LAZY_VGL_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_LAZY_VGL_VALIDATION_H

#include "PsiFormerDeviceBatchBackend.h"

namespace qmcplusplus::psiformer::device
{
/// Exercise released VALUE ownership, transactional FULL admission and warmed reuse.
inline void validateLazyVglPreparation(const std::shared_ptr<const PreparedDeviceModel>& image,
                                      const std::vector<double>& positions)
{
  const auto& environment = image->plan().environment();
  if (!supportsLazyRealInferenceEnvironment(environment))
    return;
  const auto ne = image->plan().modelShape().electrons();
  const auto batch = positions.size() / (3 * ne);
  if (!batch || positions.size() != batch * 3 * ne)
    throw std::invalid_argument("Lazy VGL validation requires complete configurations");
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "Lazy VGL validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "Lazy VGL validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  DeviceExecutionPreparation preparation;
  preparation.vgl.max_targets = 1;
  preparation.vgl.reuse_cusp_state = false;
  const auto cost = estimatePreparedTargetVglCost(image->plan(), batch, preparation.vgl);
  preparation.vgl.device_workspace_budget_bytes = cost.device_workspace_bytes;
  preparation.vgl.scalar_work_budget = cost.charged_scalar_work_units;
  using Mode = pf::DirectBatchMode;
  using Backend = PreparedDeviceBatchBackend;
  const auto version = image->version();
  const auto initial = metrics.snapshot().live_bytes;
  if (environment.boundary == BoundaryCondition::PERIODIC)
  {
    for (int fault = 0; fault < 6; ++fault)
    {
      auto invalid = environment;
      if (fault == 0) invalid.periodic_profile = std::make_shared<PeriodicProfileMetadata>();
      if (fault == 1) invalid.periodic_axes[1] = false;
      if (fault == 2) invalid.fixed_nuclei = false;
      if (fault == 3) invalid.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
      if (fault == 4) invalid.parameter_scalar_domain = ScalarDomain::COMPLEX;
      if (fault == 5) invalid.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
      if (supportsLazyRealInferenceEnvironment(invalid))
        throw std::runtime_error("Lazy Gamma admitted unsupported metadata");
    }
    bool rejected_twist = false, rejected_mixed = false;
    try { Backend rejected(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation, true, true); }
    catch (const std::invalid_argument&) { rejected_twist = true; }
    try { Backend rejected(image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, {}, true); }
    catch (const std::invalid_argument&) { rejected_mixed = true; }
    if (!rejected_twist || !rejected_mixed || metrics.snapshot().live_bytes != initial)
      throw std::runtime_error("Lazy Gamma negative admission allocated or missed twist/mixed");
  }
  Backend eager(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation, false);
  auto eager_workspace = eager.prepare(batch);
  const auto eager_bytes = metrics.snapshot().live_bytes - initial;
  Backend lazy(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation, true);
  const auto before_lazy = metrics.snapshot().live_bytes;
  auto workspace = lazy.prepare(batch);
  const auto lazy_bytes = metrics.snapshot().live_bytes - before_lazy;
  if (eager_bytes - lazy_bytes != cost.device_workspace_bytes)
    throw std::runtime_error("Lazy VALUE preparation did not omit exactly the FULL owner");

  // Materialize views immediately: the next call may replace their HOST staging.
  const auto collect = [&](const pf::DirectBatchTileResultView& view, bool full) {
    std::vector<double> result;
    for (std::size_t b = 0; b < batch; ++b)
    {
      result.push_back(view.sign[b]);
      result.push_back(view.logabs[b * view.value_stride]);
      if (!full) continue;
      for (std::size_t lane = 0; lane < 3 * ne; ++lane)
        result.push_back(view.gradient[b * view.gradient_configuration_stride + lane * view.gradient_lane_stride]);
      for (std::size_t electron = 0; electron < ne; ++electron)
      {
        result.push_back(view.lap_log[b * view.lap_log_configuration_stride + electron * view.lap_log_lane_stride]);
        result.push_back(view.lap_ratio[b * view.lap_ratio_configuration_stride + electron]);
      }
    }
    return result;
  };
  const auto evaluate = [&](const Backend& backend, pf::DirectBatchTileBackendWorkspace& scratch,
                            Mode mode, std::size_t model_version) {
    return collect(backend.evaluate(scratch, mode, positions.data(), batch, model_version), mode != Mode::VALUE_ONLY);
  };
  const auto expected_value = evaluate(eager, *eager_workspace, Mode::VALUE_ONLY, version);
  const auto expected_full = evaluate(eager, *eager_workspace, Mode::FULL_VGL, version);
  const auto before_value = metrics.snapshot();
  if (evaluate(lazy, *workspace, Mode::VALUE_ONLY, version) != expected_value ||
      metrics.snapshot().allocation_calls != before_value.allocation_calls)
    throw std::runtime_error("Lazy VALUE evaluation allocated or changed values");
  const auto before_upgrade = metrics.snapshot();
  if (evaluate(lazy, *workspace, Mode::FULL_VGL, version) != expected_full)
    throw std::runtime_error("Lazy FULL upgrade changed results");
  const auto after_upgrade = metrics.snapshot();
  if (after_upgrade.live_bytes - before_upgrade.live_bytes != cost.device_workspace_bytes ||
      after_upgrade.free_calls != before_upgrade.free_calls)
    throw std::runtime_error("Lazy FULL upgrade replaced VALUE ownership");
  evaluate(lazy, *workspace, Mode::VALUE_ONLY, version);
  evaluate(lazy, *workspace, Mode::FULL_VGL, version);
  if (metrics.snapshot().allocation_calls != after_upgrade.allocation_calls)
    throw std::runtime_error("Warmed lazy graph allocated");

  auto retry = lazy.prepare(batch);
  const auto before_failure = metrics.snapshot();
  bool rejected = false;
  {
    // Reserve admission capacity without a vendor allocation: deterministic failure, not OOM.
    const auto& policy = configuredDeviceMemoryAdmission();
    const auto maximum = policy.limit_bytes ? *policy.limit_bytes - policy.external_reserve_bytes
                                           : std::numeric_limits<std::size_t>::max();
    auto reservation = metrics.reserveAllocation(maximum - before_failure.live_bytes);
    try
    {
      evaluate(lazy, *retry, Mode::FULL_VGL, version);
    }
    catch (const std::length_error&)
    {
      rejected = true;
    }
    catch (const std::overflow_error&)
    {
      rejected = true;
    }
  }
  if (!rejected || metrics.snapshot().allocation_calls != before_failure.allocation_calls ||
      metrics.snapshot().live_bytes != before_failure.live_bytes ||
      evaluate(lazy, *retry, Mode::VALUE_ONLY, version) != expected_value ||
      evaluate(lazy, *retry, Mode::FULL_VGL, version) != expected_full)
    throw std::runtime_error("Failed lazy FULL admission damaged VALUE ownership or retry");

  auto recovered = lazy.prepare(batch);
  auto invalid_positions = positions;
  invalid_positions.front() = std::numeric_limits<double>::quiet_NaN();
  const auto before_recovery = metrics.snapshot().live_bytes;
  bool failed_value = false;
  try
  {
    lazy.evaluate(*recovered, Mode::VALUE_ONLY, invalid_positions.data(), batch, version);
  }
  catch (const std::exception&)
  {
    failed_value = true;
  }
  if (!failed_value || metrics.snapshot().live_bytes != before_recovery ||
      evaluate(lazy, *recovered, Mode::VALUE_ONLY, version) != expected_value)
    throw std::runtime_error("Lazy VALUE failure recovery allocated FULL storage or changed retry");

  auto rebound = lazy.prepare(batch);
  std::vector<double> parameters(image->plan().parameterCount());
  image->downloadCanonical(parameters);
  lazy.publishParameters(parameters, version + 1, [] {});
  const auto before_rebind = metrics.snapshot();
  if (evaluate(lazy, *rebound, Mode::VALUE_ONLY, version + 1) != expected_value ||
      metrics.snapshot().allocation_calls != before_rebind.allocation_calls ||
      evaluate(lazy, *rebound, Mode::FULL_VGL, version + 1) != expected_full)
    throw std::runtime_error("Lazy VALUE-only rebind or FULL upgrade changed results");
  bool stale_rejected = false;
  try
  {
    evaluate(lazy, *rebound, Mode::VALUE_ONLY, version);
  }
  catch (const std::invalid_argument&)
  {
    stale_rejected = true;
  }
  if (!stale_rejected) throw std::runtime_error("Lazy rebind accepted a stale model version");
  for (int unsupported = 0; unsupported < 3; ++unsupported)
  {
    auto rejected_preparation = preparation;
    if (unsupported == 0) rejected_preparation.active_inference_crowds = 2;
    if (unsupported == 1) rejected_preparation.metropolis_workspace_budget_bytes = 1;
    if (unsupported == 2) rejected_preparation.metropolis_drift_budget_bytes = 1;
    const auto before_rejection = metrics.snapshot().allocation_calls;
    bool rejected_profile = false;
    try
    {
      Backend invalid(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, rejected_preparation, true);
    }
    catch (const std::exception&)
    {
      rejected_profile = true;
    }
    if (!rejected_profile || metrics.snapshot().allocation_calls != before_rejection)
      throw std::runtime_error("Unsupported lazy profile was not rejected before allocation");
  }
  std::cout << "PASS lazy FULL preparation: exact VALUE/FULL, owner bytes, warm reuse, failed upgrade/recovery, rebind, profile guards\n";
}
}
#endif
