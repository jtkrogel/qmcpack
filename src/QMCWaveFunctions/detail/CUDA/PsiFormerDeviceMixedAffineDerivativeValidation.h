//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_AFFINE_DERIVATIVE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_AFFINE_DERIVATIVE_VALIDATION_H

#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"
#include "PsiFormerDeviceComplexScore.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "PsiFormerDeviceBatchBackend.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"
#include <cstring>
#include <limits>

namespace qmcplusplus::psiformer::device
{
/// Independently verify transpose/leading-dimension mapping and untouched padding.
inline void validateMixedAffinePaddedPrimitives(compute::Queue<PlatformKind::CUDA>& queue,
    AcceleratorBlasHandle& blas)
{
  const DenseForwardLayout layout{4, 3, 2, 5, 4, 3};
  std::vector<float> x(20, 31.f), a(12, 31.f), w(12, 31.f), input(20, 31.f), weight(12, 31.f);
  for (std::size_t r = 0; r < 4; ++r)
  {
    for (std::size_t i = 0; i < 3; ++i) x[r * 5 + i] = .125f * (r + 1) * (i + 1);
    for (std::size_t j = 0; j < 2; ++j) a[r * 3 + j] = -.0625f * (r + 1) * (j + 1);
  }
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j) w[i * 4 + j] = .03125f * (i + 1) * (j + 1);
  DeviceBuffer<float> gx(x.size()), ga(a.size()), gw(w.size()), gi(input.size()), gt(weight.size());
  gx.upload(x.data(), x.size()); ga.upload(a.data(), a.size()); gw.upload(w.data(), w.size());
  gi.upload(input.data(), input.size()); gt.upload(weight.data(), weight.size());
#ifdef QMC_CUDA2HIP
  constexpr auto backend = PsiFormerAcceleratorBackend::HIP;
#else
  constexpr auto backend = PsiFormerAcceleratorBackend::CUDA;
#endif
  const auto mode = makePsiFormerBlasMathModePlan(PsiFormerBackendMathMode::FP32_STRICT, backend);
  denseInputAdjointFp32(blas, mode, layout, ga.data(), gw.data(), gi.data());
  denseWeightAdjointFp32(blas, mode, layout, gx.data(), ga.data(), gt.data());
  queue.sync(); gi.download(input.data(), input.size()); gt.download(weight.data(), weight.size());
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t i = 0; i < 5; ++i)
    {
      float expected = i < 3 ? 0.f : 31.f;
      if (i < 3) for (std::size_t j = 0; j < 2; ++j) expected += a[r * 3 + j] * w[i * 4 + j];
      if (input[r * 5 + i] != expected) throw std::runtime_error("Mixed input VJP stride or padding mismatch");
    }
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
    {
      float expected = j < 2 ? 0.f : 31.f;
      if (j < 2) for (std::size_t r = 0; r < 4; ++r) expected += x[r * 5 + i] * a[r * 3 + j];
      if (weight[i * 4 + j] != expected) throw std::runtime_error("Mixed weight VJP stride or padding mismatch");
    }
  std::cout << "PASS mixed affine padded input/weight VJP exact_binary_fixture=1 untouched_padding=1\n";
}

/// Exercise real native recipe ownership, epoch guards and warm cache publication.
inline void validateMixedRealProductBinding(const std::shared_ptr<const PreparedDeviceModel>& image,
    const std::vector<double>& positions, const std::vector<double>& mixed_reference,
    const std::vector<double>& fp64_reference, const std::vector<double>& unchanged)
{
  const auto require = [](bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
  };
  const auto compare = [&](const std::vector<double>& actual, const std::vector<double>& expected, double tolerance) {
    require(actual.size() == expected.size(), "Real mixed product score extent changed");
    long double error = 0, norm = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
      require(device_math::isFiniteBinary64(actual[i]), "Real mixed product score is nonfinite");
      error += (actual[i] - expected[i]) * (actual[i] - expected[i]);
      norm += expected[i] * expected[i];
    }
    require(std::sqrt(error) <= tolerance * std::max(1.L, std::sqrt(norm)),
        "Real product recipe differs from its independent score reference");
  };
  using Mode = PsiFormerGpuTrainingEpochMode;
  PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, {}, false);
  bool rejected = false;
  try { (void)backend.prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Mixed real product preparation silently admitted kinetic responses");
  auto sessions = backend.prepareMixedTrainingProductSessions(1);
  auto& mixed = *sessions.first;
  auto& fallback = *sessions.second;
  require(mixed.usesMixedAffineProducts() && !fallback.usesMixedAffineProducts(),
      "Real product recipes did not retain distinct arithmetic owners");
  require(mixed.deviceWorkspaceBytes() == estimateMixedScoreTileStorage(image->plan(), 1).device_bytes &&
      fallback.deviceWorkspaceBytes() == estimateScoreTileStorage(image->plan(), 1).device_bytes,
      "Real product recipe capacities differ from their prepared inventory");

  auto physical = backend.prepare(1);
  const auto generation = backend.trainingEpochGeneration();
  const auto strict = backend.enterTrainingEpoch(image->version(), Mode::STRICT_MIXED);
  require(strict == generation + 1 && backend.trainingEpochMode() == Mode::STRICT_MIXED,
      "Real strict mixed epoch did not publish its generation");
  rejected = false;
  try { (void)backend.enterTrainingEpoch(image->version(), Mode::FP64_REPLAY); }
  catch (const std::logic_error&) { rejected = true; }
  require(rejected, "Real arithmetic epochs permitted overlapping selection");
  compare(mixed.score(positions.data(), image->version()), mixed_reference, 2e-6);
  std::vector<double> direction(mixed_reference.size());
  long double dot = 0;
  for (std::size_t i = 0; i < direction.size(); ++i)
  { direction[i] = .01 * std::sin(.19 * (i + 1)); dot += direction[i] * mixed_reference[i]; }
  require(std::abs(mixed.scoreJvp(positions.data(), direction, image->version()) - dot) <=
      2e-5L * (1 + std::abs(dot)), "Real bound mixed JVP violates its declared arithmetic envelope");
  require(mixed.mixedReverseStatistics() && mixed.mixedReverseStatistics()->completed != 0 &&
      mixed.mixedReverseStatistics()->fp32_gemms != 0 && mixed.mixedJvpStatistics() &&
      mixed.mixedJvpStatistics()->completed != 0 && mixed.mixedJvpStatistics()->fp32_gemms != 0 &&
      !fallback.mixedReverseStatistics(), "Real product binding reported policy without actual mixed arithmetic");
  const auto primal = backend.evaluate(*physical, pf::DirectBatchMode::VALUE_ONLY,
      positions.data(), 1, image->version());
  require(primal.size == 1 && device_math::isFiniteBinary64(primal.logabs[0]),
      "Strict mixed physical primal did not complete");
  const double canonical_scalar_sign = primal.sign[0], canonical_scalar_log = primal.logabs[0];
  bool committed = false;
  rejected = false;
  try { backend.publishParameters(unchanged, image->version() + 1, [&] { committed = true; }); }
  catch (const std::logic_error&) { rejected = true; }
  require(rejected && !committed, "Real epoch allowed parameter publication before closing");
  backend.leaveTrainingEpoch(strict);

  const auto replay = backend.enterTrainingEpoch(image->version(), Mode::FP64_REPLAY);
  compare(fallback.score(positions.data(), image->version()), fp64_reference, 2e-11);
  const auto fp64_primal = backend.evaluate(*physical, pf::DirectBatchMode::VALUE_ONLY,
      positions.data(), 1, image->version());
  require(fp64_primal.size == 1 && device_math::isFiniteBinary64(fp64_primal.logabs[0]),
      "Explicit FP64 physical replay did not complete");
  require(std::memcmp(&canonical_scalar_sign, fp64_primal.sign, sizeof(double)) == 0 &&
      std::memcmp(&canonical_scalar_log, fp64_primal.logabs, sizeof(double)) == 0,
      "Native mixed policy scalar VALUE differs from explicit FP64 scalar arithmetic");
  backend.leaveTrainingEpoch(replay);
  require(backend.trainingEpochMode() == Mode::INFERENCE &&
      backend.trainingEpochGeneration() == generation + 4,
      "Real arithmetic epoch did not restore ordinary inference");

  auto successor = unchanged;
  successor.front() += .0001;
  backend.publishParameters(successor, image->version() + 1, [&] { committed = true; });
  require(committed, "Real idle publication did not commit the native image");
  const auto warm_mixed = mixed.score(positions.data(), image->version() + 1);
  const auto warm_fp64 = fallback.score(positions.data(), image->version() + 1);
  auto fresh_image = image->makeUpdatedImage(successor, image->version() + 1);
  PreparedDeviceBatchBackend fresh(fresh_image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, {}, false);
  auto fresh_sessions = fresh.prepareMixedTrainingProductSessions(1);
  compare(warm_mixed, fresh_sessions.first->score(positions.data(), image->version() + 1), 2e-6);
  compare(warm_fp64, fresh_sessions.second->score(positions.data(), image->version() + 1), 2e-11);
  // A finite FP64 master may fail the explicit FP32 conversion contract.
  // Scalar VALUE is canonical FP64; strict mixed FULL must preserve a typed numerical rejection without replay.
  auto out_of_range = unchanged;
  const auto& embedding = image->plan().parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
  out_of_range[embedding.begin] = 2. * static_cast<double>(std::numeric_limits<float>::max());
  auto rejected_image = image->makeUpdatedImage(out_of_range, image->version() + 1);
  PreparedDeviceBatchBackend rejected_backend(rejected_image, {},
      PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, {}, false);
  auto rejected_physical = rejected_backend.prepare(1);
  const auto rejected_epoch = rejected_backend.enterTrainingEpoch(rejected_image->version(), Mode::STRICT_MIXED);
  bool typed_rejection = false;
  try
  {
    (void)rejected_backend.evaluate(*rejected_physical, pf::DirectBatchMode::FULL_VGL,
        positions.data(), 1, rejected_image->version());
  }
  catch (const MixedPrecisionRetryRequired&) { typed_rejection = true; }
  rejected_backend.leaveTrainingEpoch(rejected_epoch);
  require(typed_rejection, "Strict real FULL erased its typed precision rejection or silently replayed");
  std::cout << "PASS real mixed native product binding strict/replay=1 distinct_owners=1 warm_publication=1"
            << " typed_FULL_rejection=1 canonical_scalar_FP64_bits=1\n";
}

/// Independent rounded-operand oracle; no production dense helper computes the reference.
inline void validateMixedAffineDerivatives()
{
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  validateMixedAffinePaddedPrimitives(queue, blas);
  const auto rounded = [](double x) { return static_cast<double>(static_cast<float>(x)); };
  const auto require = [](bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
  };
  const auto near = [&](double actual, long double expected, long double scale) {
    require(device_math::isFiniteBinary64(actual) &&
        std::abs(static_cast<long double>(actual) - expected) <= 3e-6L * (1 + scale),
        "Mixed affine derivative differs from independent rounded-operand oracle");
  };
  for (const bool paired : {false, true})
  {
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{8., 0., 0.}, {0., 8., 0.}, {0., 0., 8.}}};
    environment.periodic_axes = {true, true, true};
    if (paired)
    {
      environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
      environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    }
    const ModelShape shape{2, 2, 4, 1, 4, 1, 1};
    const auto p = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(p, shape, environment);
    auto theta = p.flat_values();
    for (const auto& tensor : plan.parameterTensors())
      if (tensor.role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT ||
          tensor.role == ParameterRole::ATTENTION_QUERY_WEIGHT || tensor.role == ParameterRole::ATTENTION_KEY_WEIGHT ||
          tensor.role == ParameterRole::ATTENTION_VALUE_WEIGHT || tensor.role == ParameterRole::ATTENTION_OUTPUT_WEIGHT ||
          tensor.role == ParameterRole::UPDATE_HIDDEN_WEIGHT || tensor.role == ParameterRole::UPDATE_OUTPUT_WEIGHT)
        for (std::size_t i = tensor.begin; i < tensor.end; ++i)
          theta[i] = .013 * std::sin(.37 * double(i + 1));
    const auto& embedding_weight = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    theta[embedding_weight.begin + (embedding_weight.shape[0] - 1) * shape.feature_dimension] = .5;
    const std::vector<double> nuclei{0., 0., 0., 2., 0., 0., 0., 2., 0., 0., 0., 2.};
    const std::shared_ptr<const PreparedDeviceModel> image = paired
        ? PreparedDeviceModel::preparePairedComplexValue(plan, theta, nuclei, 0)
        : std::make_shared<PreparedDeviceModel>(plan, theta, nuclei, 0);
    auto cache = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue,
        paired ? MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD : MixedTrunkHeadPolicy::REAL_AMPLITUDE);
    PreparedDeviceMixedAffineDerivatives adapter(cache, 5, queue, blas);
    require(adapter.deviceWorkspaceBytes() == PreparedDeviceMixedAffineDerivatives::estimateDeviceWorkspaceBytes(plan, 5),
        "Mixed derivative planned and actual bytes differ");
    require(adapter.hostWorkspaceBytes() == sizeof(PsiFormerDeviceNumericalDiagnostics),
        "Mixed derivative host numeric diagnostics charge differs");
    // Exercise the actual retained-primal/score composition, independently of affine leaves.
    const std::vector<double> positions{.1, .2, .3, 2.1, .2, .3, .1, 2.2, .3, .1, .2, 2.3};
    pf::ConfigData config(pf::Tensor({1, 4, 3}, positions), pf::Tensor({4, 3}, nuclei),
        pf::Tensor({4}, {1., 1., 1., 1.}), 2, 2, 4);
    pf::PsiFormer model(p, config, 1, 4, 1, 1);
    model.p.set_flat_values(theta);
    const auto compare_score = [&](const std::vector<std::complex<double>>& actual,
                                   const std::vector<std::complex<double>>& expected) {
      long double error = 0, reference = 0;
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        require(device_math::isFiniteBinary64(actual[i].real()) && device_math::isFiniteBinary64(actual[i].imag()),
            "Mixed composed score is nonfinite");
        error += std::norm(actual[i] - expected[i]); reference += std::norm(expected[i]);
      }
      require(std::sqrt(error) <= 2e-4L * std::max(1.L, std::sqrt(reference)),
          "Mixed composed score exceeds its deterministic CPU FP64 error envelope");
    };
    if (paired)
    {
      const auto cost = estimateMixedComplexScoreCost(plan);
      bool underbudget = false;
      try { (void)PreparedDeviceComplexScore::prepareMixed(cache, queue, blas, cost.device_bytes - 1); }
      catch (const std::length_error&) { underbudget = true; }
      require(underbudget, "Mixed complex score admitted an insufficient exact owner budget");
      auto composed = PreparedDeviceComplexScore::prepareMixed(cache, queue, blas, cost.device_bytes);
      pf::DirectComplexScoreExecutor cpu(model, plan);
      auto workspace = cpu.makeWorkspace();
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), shape.electrons()));
      composed->evaluate(positions);
      std::vector<cuDoubleComplex> gpu_score(plan.parameterCount()); ComplexValueResult amplitude;
      composed->download(composed->resultView(), gpu_score, amplitude);
      const auto expected = cpu.evaluate(*workspace);
      std::vector<std::complex<double>> actual(gpu_score.size()), reference(gpu_score.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
      { actual[i] = {cuCreal(gpu_score[i]), cuCimag(gpu_score[i])}; reference[i] = expected.parameter_score[i]; }
      compare_score(actual, reference);

      // Preflight every mutable leaf before adopting a prepared successor.
      const auto stale = composed->resultView();
      auto next_theta = theta;
      next_theta.front() += .0001;
      const auto next_image = image->makeUpdatedImage(next_theta, 1);
      const auto next_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(next_image, queue,
          MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD);
      composed->validateMixedParameterRebind(next_cache);
      composed->adoptPreparedMixedParameters(next_cache);
      bool stale_rejected = false;
      try { composed->download(stale, gpu_score, amplitude); } catch (const std::invalid_argument&) { stale_rejected = true; }
      require(stale_rejected, "Mixed complex score exposed a predecessor-generation view");
      composed->evaluate(positions); composed->download(composed->resultView(), gpu_score, amplitude);
      auto fresh = PreparedDeviceComplexScore::prepareMixed(next_cache, queue, blas, cost.device_bytes);
      fresh->evaluate(positions); std::vector<cuDoubleComplex> fresh_score(gpu_score.size());
      fresh->download(fresh->resultView(), fresh_score, amplitude);
      for (std::size_t i = 0; i < actual.size(); ++i)
      { actual[i] = {cuCreal(gpu_score[i]), cuCimag(gpu_score[i])}; reference[i] = {cuCreal(fresh_score[i]), cuCimag(fresh_score[i])}; }
      compare_score(actual, reference);
    }
    else
    {
      auto primal = std::make_shared<PreparedDeviceMixedValueDense>(cache, 4, queue, blas);
      PreparedDeviceInference graph(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      graph.setMixedRetainedValueDense(primal);
      auto reverse_arithmetic = std::make_shared<PreparedDeviceMixedAffineDerivatives>(cache, 4, queue, blas);
      auto tangent_arithmetic = std::make_shared<PreparedDeviceMixedAffineDerivatives>(cache, 4, queue, blas);
      PreparedDeviceScoreReverse reverse(image, 1, queue, blas, reverse_arithmetic);
      PreparedDeviceScoreJvp tangent(image, 1, queue, blas, tangent_arithmetic);
      graph.evaluate(positions);
      require(graph.valueTapeMixedParameters().get() == cache.get(), "Mixed retained primal lost its compute-cache identity");
      PreparedDeviceScoreReverse wrong_arithmetic(image, 1, queue, blas);
      bool mismatch = false;
      try { wrong_arithmetic.reverse(graph, {1.}); } catch (const std::invalid_argument&) { mismatch = true; }
      require(mismatch, "FP64 reverse silently consumed a mixed primal tape");
      reverse.reverse(graph, {1.}); std::vector<double> score(plan.parameterCount()); reverse.downloadResponse(score);
      pf::DirectScoreExecutor cpu(model, plan);
      auto workspace = cpu.makeWorkspace();
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), shape.electrons()));
      const auto expected = cpu.evaluate(*workspace);
      std::vector<std::complex<double>> actual(score.size()), reference(score.size());
      for (std::size_t i = 0; i < score.size(); ++i) { actual[i] = score[i]; reference[i] = expected.parameter_score[i]; }
      compare_score(actual, reference);
      std::vector<double> fp64_reference(score.size());
      for (std::size_t i = 0; i < score.size(); ++i) fp64_reference[i] = expected.parameter_score[i];
      validateMixedRealProductBinding(image, positions, score, fp64_reference, theta);
      std::vector<double> direction(score.size()), product(1);
      long double reverse_dot = 0;
      for (std::size_t i = 0; i < score.size(); ++i)
      { direction[i] = .01 * std::sin(.19 * (i + 1)); reverse_dot += direction[i] * score[i]; }
      tangent.evaluate(graph, direction); tangent.downloadProducts(product);
      require(std::abs(product[0] - reverse_dot) <= 2e-5L * (1 + std::abs(reverse_dot)),
          "Mixed composed score VJP/JVP duality exceeds its FP32 arithmetic envelope");
      auto next_theta = theta;
      next_theta.front() += .0001;
      const auto next_image = image->makeUpdatedImage(next_theta, 1);
      const auto next_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(next_image, queue);
      graph.validateModelRebindWithMixedParameters(next_image, next_cache);
      reverse.validateModelRebindWithMixedParameters(next_image, next_cache);
      tangent.validateModelRebindWithMixedParameters(next_image, next_cache);
      graph.adoptPreparedModelWithMixedParameters(next_image, next_cache);
      reverse.adoptPreparedModelWithMixedParameters(next_image, next_cache);
      tangent.adoptPreparedModelWithMixedParameters(next_image, next_cache);
      graph.evaluate(positions);
      reverse.reverse(graph, {1.});
      reverse.downloadResponse(score);
      auto fresh_primal = std::make_shared<PreparedDeviceMixedValueDense>(next_cache, 4, queue, blas);
      PreparedDeviceInference fresh_graph(next_image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      fresh_graph.setMixedRetainedValueDense(fresh_primal); fresh_graph.evaluate(positions);
      auto fresh_arithmetic = std::make_shared<PreparedDeviceMixedAffineDerivatives>(next_cache, 4, queue, blas);
      PreparedDeviceScoreReverse fresh_reverse(next_image, 1, queue, blas, fresh_arithmetic);
      fresh_reverse.reverse(fresh_graph, {1.}); std::vector<double> fresh_score(score.size()); fresh_reverse.downloadResponse(fresh_score);
      for (std::size_t i = 0; i < score.size(); ++i) { actual[i] = score[i]; reference[i] = fresh_score[i]; }
      compare_score(actual, reference);
    }
    std::cout << "PASS mixed retained score composition paired=" << paired
              << " CPU_FP64_error_envelope=2e-4 fresh_after_update=1 same_cache=1\n";

    // Independent affine leaves also exercise ragged row counts and failed epochs.
    std::size_t cases = 0;
    for (const auto role : {ParameterRole::ELECTRON_EMBEDDING_WEIGHT, ParameterRole::ATTENTION_QUERY_WEIGHT})
    {
      const auto& tensor = plan.parameter(role, role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT ? NO_ATTENTION_BLOCK : 0);
      const auto ni = tensor.shape[0], no = tensor.shape[1];
      for (const std::size_t rows : {3, 5, 1})
      {
        DenseForwardLayout layout{rows, ni, no, ni, no, no};
        std::vector<double> x(rows * ni), a(rows * no), dx(x.size()), dw(ni * no);
        for (std::size_t i = 0; i < x.size(); ++i)
        { x[i] = .2 * std::sin(.17 * (i + 1)); dx[i] = .07 * std::cos(.23 * (i + 1)); }
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = .11 * std::cos(.31 * (i + 1));
        for (std::size_t i = 0; i < dw.size(); ++i) dw[i] = .03 * std::sin(.41 * (i + 1));
        DeviceBuffer<double> gx(x.size()), ga(a.size()), gdx(dx.size()), gdw(dw.size());
        DeviceBuffer<double> input(x.size()), weight(dw.size()), tangent(a.size());
        gx.upload(x.data(), x.size()); ga.upload(a.data(), a.size());
        gdx.upload(dx.data(), dx.size()); gdw.upload(dw.data(), dw.size());
        std::vector<double> input_result(x.size(), .125), weight_result(dw.size(), -.25), tangent_result(a.size());
        input.upload(input_result.data(), input_result.size()); weight.upload(weight_result.data(), weight_result.size());
        const auto* canonical_weight = image->parameters() + tensor.begin;
        adapter.beginEvaluation(image);
        const auto observed = adapter.accumulateWeightAdjoint(layout, gx.data(), ga.data(), canonical_weight, weight.data(), true);
        std::vector<double> observed_x(x.size()), observed_a(a.size());
        queue.sync();
#ifdef QMC_CUDA2HIP
        checkRuntime(hipMemcpy(observed_x.data(), observed.source, checkedDeviceBytes<double>(x.size()), hipMemcpyDeviceToHost), "mixed observation source");
        checkRuntime(hipMemcpy(observed_a.data(), observed.adjoint, checkedDeviceBytes<double>(a.size()), hipMemcpyDeviceToHost), "mixed observation adjoint");
#else
        checkRuntime(cudaMemcpy(observed_x.data(), observed.source, checkedDeviceBytes<double>(x.size()), cudaMemcpyDeviceToHost), "mixed observation source");
        checkRuntime(cudaMemcpy(observed_a.data(), observed.adjoint, checkedDeviceBytes<double>(a.size()), cudaMemcpyDeviceToHost), "mixed observation adjoint");
#endif
        for (std::size_t i = 0; i < x.size(); ++i)
          require(observed_x[i] == rounded(x[i]), "Mixed observation does not expose consumed FP32 source");
        for (std::size_t i = 0; i < a.size(); ++i)
          require(observed_a[i] == rounded(a[i]), "Mixed observation does not expose consumed FP32 adjoint");
        adapter.accumulateInputAdjoint(layout, ga.data(), canonical_weight, input.data());
        adapter.projectDirection(layout, gx.data(), gdx.data(), canonical_weight, gdw.data(), tangent.data());
        adapter.finishEvaluation();
        input.download(input_result.data(), input_result.size()); weight.download(weight_result.data(), weight_result.size());
        tangent.download(tangent_result.data(), tangent_result.size());
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t i = 0; i < ni; ++i)
          {
            long double sum = .125L, scale = 0;
            for (std::size_t j = 0; j < no; ++j)
            { const long double v = rounded(a[r * no + j]) * rounded(theta[tensor.begin + i * no + j]); sum += v; scale += std::abs(v); }
            near(input_result[r * ni + i], sum, scale);
          }
        for (std::size_t i = 0; i < ni; ++i)
          for (std::size_t j = 0; j < no; ++j)
          {
            long double sum = -.25L, scale = 0;
            for (std::size_t r = 0; r < rows; ++r)
            { const long double v = rounded(x[r * ni + i]) * rounded(a[r * no + j]); sum += v; scale += std::abs(v); }
            near(weight_result[i * no + j], sum, scale);
          }
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t j = 0; j < no; ++j)
          {
            long double sum = 0, scale = 0;
            for (std::size_t i = 0; i < ni; ++i)
            {
              const long double v = rounded(dx[r * ni + i]) * rounded(theta[tensor.begin + i * no + j]) +
                  rounded(x[r * ni + i]) * rounded(dw[i * no + j]);
              sum += v; scale += std::abs(v);
            }
            near(tangent_result[r * no + j], sum, scale);
          }
        // Bias derivatives remain the unrounded FP64 island, separately from q(A) factor rows.
        DeviceBuffer<double> bias(no);
        std::vector<double> bias_result(no, 0.0); bias.upload(bias_result.data(), no);
        checkRuntime(launchDenseBiasAdjoint(queue.getNative(), layout, ga.data(), bias.data()), "mixed FP64 bias island");
        queue.sync(); bias.download(bias_result.data(), no);
        for (std::size_t j = 0; j < no; ++j)
        {
          double expected = 0, rounded_metric_row = 0;
          for (std::size_t r = 0; r < rows; ++r)
          { expected += a[r * no + j]; rounded_metric_row += rounded(a[r * no + j]); }
          require(bias_result[j] == expected, "Mixed bias derivative lost its FP64 island");
          require(std::abs(expected - rounded_metric_row) <= rows * 1e-7,
              "Mixed homogeneous bias metric exceeds its declared cast error");
        }
        // Resolving epsilon ladder for the rounded primal; no infinitesimal derivative claim.
        long double algorithmic = 0, derivative_scale = 0;
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t j = 0; j < no; ++j)
            for (std::size_t i = 0; i < ni; ++i)
            {
              const long double v = rounded(a[r * no + j]) *
                  (rounded(dx[r * ni + i]) * rounded(theta[tensor.begin + i * no + j]) +
                   rounded(x[r * ni + i]) * rounded(dw[i * no + j]));
              algorithmic += v; derivative_scale += std::abs(v);
            }
        const auto rounded_primal = [&](double epsilon) {
          long double result = 0;
          for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < no; ++j)
            {
              float sum = 0.f;
              for (std::size_t i = 0; i < ni; ++i)
                sum += static_cast<float>(x[r * ni + i] + epsilon * dx[r * ni + i]) *
                    static_cast<float>(theta[tensor.begin + i * no + j] + epsilon * dw[i * no + j]);
              result += rounded(a[r * no + j]) * static_cast<double>(sum);
            }
          return result;
        };
        for (const double epsilon : {.25, .125, .0625})
        {
          const auto plus = rounded_primal(epsilon), minus = rounded_primal(-epsilon);
          const auto finite_difference = (plus - minus) / (2 * epsilon);
          const auto floor = 32 * std::numeric_limits<float>::epsilon() *
              (std::abs(plus) + std::abs(minus) + derivative_scale) / epsilon + 1e-9L;
          require(std::abs(finite_difference - algorithmic) <= floor,
              "Mixed affine algorithmic adjoint exceeds the FP32-resolving epsilon error floor");
        }
        // The fixed-geometry embedding tangent omits dX, rather than reusing stale scratch.
        adapter.beginEvaluation(image);
        adapter.projectDirection(layout, gx.data(), nullptr, canonical_weight, gdw.data(), tangent.data());
        adapter.finishEvaluation(); tangent.download(tangent_result.data(), tangent_result.size());
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t j = 0; j < no; ++j)
          {
            long double sum = 0, scale = 0;
            for (std::size_t i = 0; i < ni; ++i)
            { const long double v = rounded(x[r * ni + i]) * rounded(dw[i * no + j]); sum += v; scale += std::abs(v); }
            near(tangent_result[r * no + j], sum, scale);
          }
        for (const double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max()})
        {
          auto invalid = a; invalid.back() = bad; ga.upload(invalid.data(), invalid.size());
          adapter.beginEvaluation(image);
          adapter.accumulateInputAdjoint(layout, ga.data(), canonical_weight, input.data());
          bool rejected = false;
          try { adapter.finishEvaluation(); } catch (const MixedPrecisionRetryRequired&) { rejected = true; }
          require(rejected, "Mixed derivative failed to demand complete FP64 replay");
        }
        ga.upload(a.data(), a.size());
        adapter.beginEvaluation(image);
        adapter.accumulateInputAdjoint(layout, ga.data(), canonical_weight, input.data());
        adapter.finishEvaluation();
        ++cases;
      }
    }
    auto successor = image->makeUpdatedImage(theta, 1);
    bool rejected = false;
    try { adapter.beginEvaluation(successor); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Mixed derivative admitted a foreign generation");
    auto successor_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(successor, queue,
        paired ? MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD : MixedTrunkHeadPolicy::REAL_AMPLITUDE);
    adapter.beginEvaluation(image); rejected = false;
    try { adapter.rebindParameters(successor_cache); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Mixed derivative rebound during an open evaluation");
    adapter.cancelEvaluation(); adapter.rebindParameters(successor_cache);
    adapter.beginEvaluation(successor); adapter.finishEvaluation();
    const auto& statistics = adapter.statistics();
    require(statistics.attempts == statistics.completed + statistics.rejected &&
        statistics.fp32_gemms > 0 && statistics.cast_elements > 0 && statistics.promoted_elements > 0 &&
        statistics.logicalTransferBytes() == (statistics.cast_elements + statistics.promoted_elements) * (sizeof(double) + sizeof(float)),
        "Mixed derivative logical work counters are inconsistent");
    std::cout << "PASS mixed affine derivatives scope=affine_leaves paired=" << paired << " cases=" << cases
              << " CPU_rounded_operand_VJP_JVP=1 consumed_observations=1 overflow_recovery=1 generations=1 bytes="
              << adapter.deviceWorkspaceBytes() << '\n';
  }
}
} // namespace qmcplusplus::psiformer::device
#endif
