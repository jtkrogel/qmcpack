//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_VALUE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_COMPLEX_VALUE_VALIDATION_H
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "PsiFormerDeviceMixedComplexValue.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include <cstring>

/// Diagnostic collection must leave the canonical factor result and overwritten matrix bit-exact.
inline void validateMixedComplexPivotDiagnostics()
{
  using namespace qmcplusplus::psiformer::device;
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  constexpr std::size_t channels = 4, n = 3;
  std::vector<cuDoubleComplex> input(channels * n * n, make_cuDoubleComplex(0, 0));
  for (std::size_t c = 0; c < channels; ++c)
    for (std::size_t i = 0; i < n; ++i)
      input[(c * n + i) * n + i] = make_cuDoubleComplex(1. + .25 * i, .125 * (i + 1));
  input[(1 * n + 2) * n + 2] = make_cuDoubleComplex(0, 0);
  input[(2 * n + 2) * n + 2] = make_cuDoubleComplex(1e-7, 1e-8);
  for (std::size_t i = 3 * n * n; i < input.size(); ++i)
    input[i] = make_cuDoubleComplex(cuCreal(input[i]) * 1e100, cuCimag(input[i]) * 1e100);
  DeviceBuffer<cuDoubleComplex> old_matrix(input.size()), diagnostic_matrix(input.size());
  DeviceBuffer<ComplexValueChannel> old_channels(channels), diagnostic_channels(channels);
  DeviceBuffer<double> pivots(channels);
  DeviceBuffer<std::uint64_t> old_status(1), diagnostic_status(1);
  const std::uint64_t clear = UINT64_MAX;
  old_matrix.upload(input.data(), input.size());
  diagnostic_matrix.upload(input.data(), input.size());
  old_status.upload(&clear, 1);
  diagnostic_status.upload(&clear, 1);
  checkRuntime(launchComplexValueFactor(queue.getNative(), channels, n, old_matrix.data(),
      old_channels.data(), old_status.data()), "default paired diagnostic oracle");
  checkRuntime(launchComplexValueFactorWithDiagnostics(queue.getNative(), channels, n, diagnostic_matrix.data(),
      diagnostic_channels.data(), diagnostic_status.data(), pivots.data()), "paired diagnostic collection");
  queue.sync();
  std::vector<cuDoubleComplex> a(input.size()), b(input.size());
  std::vector<ComplexValueChannel> ca(channels), cb(channels);
  std::vector<double> p(channels);
  std::uint64_t sa, sb;
  old_matrix.download(a.data(), a.size()); diagnostic_matrix.download(b.data(), b.size());
  old_channels.download(ca.data(), ca.size()); diagnostic_channels.download(cb.data(), cb.size());
  old_status.download(&sa, 1); diagnostic_status.download(&sb, 1); pivots.download(p.data(), p.size());
  if (sa != sb || sa != clear || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) ||
      std::memcmp(ca.data(), cb.data(), ca.size() * sizeof(ca[0])) ||
      !(p[0] > 1e-3) || p[1] != 0 || !(p[2] > 0 && p[2] < 1e-3) || std::abs(p[3] - p[0]) > 1e-15)
    throw std::runtime_error("Paired scaled-pivot diagnostics changed canonical LU or its dimensionless evidence");
}

/// Small canonical builder arguments are supplied by the existing numerical validator.
inline void validateMixedComplexValue(pf::PsiFormer& model,
    const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  constexpr std::size_t capacity = 3;
  const auto n = plan.modelShape().electrons();
  if (model.cfg.electrons.x.size() != capacity * 3 * n)
    throw std::invalid_argument("Mixed paired fixture needs exactly three complete configurations");
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, model.p.flat_values(), model.cfg.nuclei.x);
  const auto budget = PreparedDeviceMixedComplexValue::estimateDeviceBytes(plan, capacity);
  int ordinal;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "mixed paired validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "mixed paired validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { PreparedDeviceMixedComplexValue denied(image, capacity, queue, blas, budget - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Mixed paired cost-minus-one allocated numeric storage");
  PreparedDeviceMixedComplexValue mixed(image, capacity, queue, blas, budget);
  if (metrics.snapshot().live_bytes - before.live_bytes != budget)
    throw std::runtime_error("Mixed paired exact owner cost differs from live device bytes");
  if (metrics.snapshot().peak_bytes > std::max(before.peak_bytes, before.live_bytes + budget))
    throw std::runtime_error("Mixed paired cold parameter conversion exceeded its admitted numeric peak");
  auto fp64 = PreparedDeviceComplexValue::prepare(image, capacity, queue, blas,
      estimateComplexValueCost(plan, capacity).device_bytes);
  pf::DirectComplexValueExecutor oracle(model, plan);
  auto workspace = oracle.makeWorkspace();
  const auto warm = metrics.snapshot();
  std::size_t actual_mixed = 0, actual_replays = 0, comparisons = 0;
  double maximum_ratio_error = 0, maximum_phase_error = 0, maximum_log_error = 0;
  for (std::size_t live : {3u, 1u, 3u})
  {
    const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + live * 3 * n);
    mixed.evaluate(positions);
    fp64->evaluate(positions);
    actual_mixed += !mixed.usedFp64Retry();
    actual_replays += mixed.usedFp64Retry();
    std::vector<ComplexValueResult> actual(live), reference(live);
    mixed.download(mixed.resultView(), actual); fp64->download(fp64->resultView(), reference);
    for (std::size_t i = 0; i < live; ++i)
    {
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + i * 3 * n, n));
      const auto expected = oracle.evaluate(*workspace);
      const auto reference_phase = std::complex<double>(cuCreal(reference[i].phase), cuCimag(reference[i].phase));
      if (expected.amplitude.isZero() || reference[i].status ||
          std::abs(reference[i].log_abs - expected.amplitude.log_abs) > 2e-10 ||
          std::abs(reference_phase - expected.amplitude.phase) > 2e-10)
        throw std::runtime_error("Mixed paired VALUE FP64 shadow disagrees with complete CPU complex oracle");
      // A normalized complex amplitude ratio avoids phase branch-cut comparisons.
      const auto ratio = std::exp(actual[i].log_abs - reference[i].log_abs) *
          std::complex<double>(cuCreal(actual[i].phase), cuCimag(actual[i].phase)) *
          std::conj(reference_phase);
      maximum_ratio_error = std::max(maximum_ratio_error, std::abs(ratio - std::complex<double>(1, 0)));
      maximum_phase_error = std::max(maximum_phase_error,
          std::abs(std::complex<double>(cuCreal(actual[i].phase), cuCimag(actual[i].phase)) - reference_phase));
      maximum_log_error = std::max(maximum_log_error, std::abs(actual[i].log_abs - reference[i].log_abs));
      if (actual[i].status || reference[i].status || !std::isfinite(std::abs(ratio)) ||
          std::abs(ratio - std::complex<double>(1, 0)) > 5e-5)
        throw std::runtime_error("Mixed paired VALUE exceeds the proposed focused normalized-ratio contract");
      if (mixed.usedFp64Retry() && std::memcmp(&actual[i], &reference[i], sizeof(actual[i])))
        throw std::runtime_error("Mixed paired replay spliced or changed complete FP64 result fields");
      ++comparisons;
    }
  }
  if (!actual_mixed || metrics.snapshot().allocation_calls != warm.allocation_calls ||
      metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Mixed paired VALUE never selected mixed arithmetic or allocated on its warm path");
  const auto stale = mixed.resultView();
  auto bad = model.cfg.electrons.x;
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  rejected = false;
  try { mixed.evaluate(bad); } catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Mixed paired VALUE hid invalid geometry behind replay");
  rejected = false;
  std::vector<ComplexValueResult> results(capacity);
  try { mixed.download(stale, results); } catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Failed mixed paired attempt retained a stale publication");
  mixed.evaluate(model.cfg.electrons.x);
  const auto old_image = mixed.resultView();
  auto updated = model.p.flat_values();
  updated[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += 1. / 64.;
  const auto active_head = plan.modelShape().spin_up_electrons ?
      ParameterRole::BACKFLOW_UP_WEIGHT : ParameterRole::BACKFLOW_DOWN_WEIGHT;
  updated[plan.parameter(active_head, NO_ATTENTION_BLOCK, HeadTensorComponent::REAL).begin] += 1. / 32.;
  auto successor = image->makeUpdatedImage(updated, 1);
  rejected = false;
  const auto before_successor = metrics.snapshot();
  try { mixed.rebindModel(successor, mixed.successorWorkspaceBytes() - 1); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || metrics.snapshot().allocation_calls != before_successor.allocation_calls)
    throw std::runtime_error("Mixed paired pending cache cost-minus-one allocated storage");
  mixed.download(old_image, results);
  mixed.rebindModel(successor, mixed.successorWorkspaceBytes());
  if (metrics.snapshot().peak_bytes >
      std::max(before_successor.peak_bytes, before_successor.live_bytes + mixed.successorWorkspaceBytes()))
    throw std::runtime_error("Mixed paired successor conversion exceeded its pending diagnostic peak");
  rejected = false;
  try { mixed.download(old_image, results); } catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Mixed paired successor retained an old-image publication");
  mixed.evaluate(model.cfg.electrons.x);
  mixed.download(mixed.resultView(), results);
  PreparedDeviceMixedComplexValue fresh_mixed(successor, capacity, queue, blas, budget);
  auto fresh_fp64 = PreparedDeviceComplexValue::prepare(successor, capacity, queue, blas,
      estimateComplexValueCost(plan, capacity).device_bytes);
  fresh_mixed.evaluate(model.cfg.electrons.x);
  fresh_fp64->evaluate(model.cfg.electrons.x);
  std::vector<ComplexValueResult> fresh(capacity), exact(capacity);
  fresh_mixed.download(fresh_mixed.resultView(), fresh); fresh_fp64->download(fresh_fp64->resultView(), exact);
  if (mixed.usedFp64Retry() != fresh_mixed.usedFp64Retry() ||
      std::memcmp(results.data(), fresh.data(), capacity * sizeof(results[0])))
    throw std::runtime_error("Mixed paired rebind retained stale converted weights or FP64 head values");
  model.p.set_flat_values(updated);
  pf::DirectComplexValueExecutor successor_oracle(model, plan);
  auto successor_workspace = successor_oracle.makeWorkspace();
  for (std::size_t i = 0; i < capacity; ++i)
  {
    successor_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data() + i * 3 * n, n));
    const auto expected = successor_oracle.evaluate(*successor_workspace);
    const auto phase = std::complex<double>(cuCreal(exact[i].phase), cuCimag(exact[i].phase));
    const auto mixed_phase = std::complex<double>(cuCreal(results[i].phase), cuCimag(results[i].phase));
    const auto error = std::abs(std::exp(results[i].log_abs - exact[i].log_abs) * mixed_phase * std::conj(phase) -
                                std::complex<double>(1, 0));
    if (expected.amplitude.isZero() || exact[i].status || results[i].status ||
        std::abs(exact[i].log_abs - expected.amplitude.log_abs) > 2e-10 ||
        std::abs(phase - expected.amplitude.phase) > 2e-10 || !(error <= 5e-5))
      throw std::runtime_error("Mixed paired successor differs from fresh FP64/CPU complete model");
    maximum_ratio_error = std::max(maximum_ratio_error, error);
    maximum_phase_error = std::max(maximum_phase_error, std::abs(mixed_phase - phase));
    maximum_log_error = std::max(maximum_log_error, std::abs(results[i].log_abs - exact[i].log_abs));
  }
  std::cout << "PASS mixed paired VALUE draft: normalized_ratio_comparisons=" << comparisons
            << " actual_mixed_requests=" << actual_mixed
            << " actual_FP64_replays=" << actual_replays
            << " maximum_ratio_error=" << maximum_ratio_error
            << " maximum_phase_error=" << maximum_phase_error << " maximum_log_error=" << maximum_log_error
            << " exact_device_bytes=" << budget << " warm_allocations=0 stale_failure_rebind=1\n";
}

/// Typed conversion, activation, singular-channel and cancellation hazards replay the entire same FP64 graph.
inline void validateMixedComplexValueReplays(pf::PsiFormer& model,
    const qmcplusplus::psiformer::PsiFormerExecutionPlan& plan)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  const auto n = plan.modelShape().electrons(), d = plan.modelShape().determinants;
  if (d < 2 || d % 2)
    throw std::invalid_argument("Mixed paired replay fixture needs an even multi-determinant head");
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  const auto original = model.p.flat_values();
  const auto positions = std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * n);
  for (const char* kind : {"conversion", "activation", "pivot", "cancellation"})
  {
    auto values = original;
    const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    if (!std::strcmp(kind, "conversion"))
      values[embedding.begin] = 1e-300;
    else if (!std::strcmp(kind, "activation"))
    {
      const auto cell = periodic_geometry::makeCell(plan.environment().lattice_vectors,
                                                    plan.environment().periodic_axes);
      const auto layout = makePeriodicFeatureJetLayout(1, n, model.cfg.nuclei.x.size() / 3,
          plan.modelShape().spin_up_electrons, SpatialJetMode::VALUE_ONLY);
      std::vector<double> raw(layout.output.configuration_stride);
      if (open_spatial::buildOpenFeatureJetsForConfiguration(layout, positions.data(), model.cfg.nuclei.x.data(),
              0, 0, raw.data(), &cell) != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Activation fixture raw features are not admissible");
      double absolute_sum = 0;
      for (std::size_t row = 0; row < layout.feature_width; ++row)
        absolute_sum += std::abs(raw[row]);
      if (!(absolute_sum > 2.0))
        throw std::runtime_error("Activation fixture lacks its analytical overflow margin");
      // Every coefficient is an exactly promoted finite float. For electron zero,
      // signs match its canonical raw row: dot >= .75*FLT_MAX*sum(abs(raw)) > 1.5*FLT_MAX.
      // This executes embedding output zero and tests activation, not weight conversion.
      const double coefficient = static_cast<double>(.75f * std::numeric_limits<float>::max());
      for (std::size_t row = 0; row < layout.feature_width; ++row)
        values[embedding.begin + row * embedding.shape[1]] = std::copysign(coefficient, raw[row]);
      std::cout << "Activation fixture analytical evidence: raw_absolute_sum=" << absolute_sum
                << " coefficient=" << coefficient << " lower_bound_FLT_MAX=1.5\n";
    }
    else
      for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
        for (auto component : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
        {
          const auto& weight = plan.parameter(role, NO_ATTENTION_BLOCK, component);
          for (std::size_t row = 0; row < plan.modelShape().feature_dimension; ++row)
            for (std::size_t determinant = 0; determinant < d; ++determinant)
              for (std::size_t column = 0; column < n; ++column)
              {
                const auto target = weight.begin + row * d * n + determinant * n + column;
                if (!std::strcmp(kind, "pivot"))
                {
                  if (!determinant) values[target] = 0;
                }
                else
                  values[target] = original[weight.begin + row * d * n + column] *
                      (determinant % 2 && column == 0 ? -1. : 1.);
              }
        }
    auto image = PreparedDeviceModel::preparePairedComplexValue(plan, values, model.cfg.nuclei.x);
    PreparedDeviceMixedComplexValue mixed(image, 1, queue, blas,
        PreparedDeviceMixedComplexValue::estimateDeviceBytes(plan, 1));
    auto reference = PreparedDeviceComplexValue::prepare(image, 1, queue, blas,
        estimateComplexValueCost(plan, 1).device_bytes);
    mixed.evaluate(positions);
    reference->evaluate(positions);
    std::vector<ComplexValueResult> a(1), b(1);
    mixed.download(mixed.resultView(), a); reference->download(reference->resultView(), b);
    std::cout << "Mixed paired replay evidence: kind=" << kind << " used_FP64_replay=" << mixed.usedFp64Retry()
              << " full_fields_equal=" << (std::memcmp(a.data(), b.data(), sizeof(a[0])) == 0)
              << " actual_status=" << a[0].status << " reference_status=" << b[0].status
              << " actual_log=" << a[0].log_abs << " reference_log=" << b[0].log_abs
              << " actual_phase=" << cuCreal(a[0].phase) << ',' << cuCimag(a[0].phase)
              << " reference_phase=" << cuCreal(b[0].phase) << ',' << cuCimag(b[0].phase)
              << " reason=" << mixed.retryReason() << '\n';
    if (!mixed.usedFp64Retry() || mixed.retryReason().empty() || std::memcmp(a.data(), b.data(), sizeof(a[0])))
      throw std::runtime_error(std::string("Mixed paired typed whole-graph replay failed: ") + kind);
    if (!std::strcmp(kind, "cancellation") && a[0].status != 1)
      throw std::runtime_error("Mixed paired exact cancellation did not preserve the FP64 node");
    std::cout << "PASS mixed paired complete replay: kind=" << kind << " FP64_fields_bits=exact reason="
              << mixed.retryReason() << '\n';
  }
}
/// Permanent bounded entry; factories reuse the canonical validator's existing physical fixture.
template<class ParameterFactory, class ConfigurationFactory>
inline void validateMixedComplexValueNumerical(ParameterFactory parameters, ConfigurationFactory configurations)
{
  using namespace qmcplusplus::psiformer;
  std::cout.precision(17);
  validateMixedComplexPivotDiagnostics();
  for (std::size_t up : {2u, 1u, 0u})
  {
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{4.0, .4, -.2}, {.7, 5.0, .3}, {-.1, .8, 6.0}}};
    environment.periodic_axes = {true, true, true};
    environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
    if (!up)
    {
      bool rejected = false;
      try
      {
        (void)PsiFormerExecutionPlan::fromParameters(parameters(environment), ModelShape{0, 4, 2, 4, 16, 2, 2}, environment);
      }
      catch (const std::invalid_argument& error)
      {
        rejected = std::string(error.what()) == "PsiFormer execution plan dimensions must be positive";
      }
      if (!rejected) throw std::runtime_error("Empty-spin fixture bypassed existing plan admission");
      std::cout << "PASS empty-spin negative admission; no mixed owner or capability widening\n";
      continue;
    }
    auto configuration = configurations(environment);
    configuration.nup = up;
    configuration.ndown = 4 - up;
    pf::PsiFormer model(parameters(environment), std::move(configuration), 4, 16, 2, 2);
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, ModelShape{up, 4 - up, 2, 4, 16, 2, 2}, environment);
    auto values = model.p.flat_values();
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      for (std::size_t i = imaginary.begin; i < imaginary.end; ++i)
        values[i] = .09 * std::sin(double(i - imaginary.begin + 1));
    }
    model.p.set_flat_values(values);
    validateMixedComplexValue(model, plan);
    if (up == 2) validateMixedComplexValueReplays(model, plan);
  }
  std::cout << "PASS paired mixed VALUE numerical fixture; no outer-twist/XML/FULL admission\n";
}
#endif
