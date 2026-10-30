//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_CANONICAL_HEAD_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_CANONICAL_HEAD_VALIDATION_H
#include "PsiFormerDeviceMixedTargetValidation.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceBatchBackend.h"
#include <limits>

namespace qmcplusplus::psiformer::device::canonical_head_validation
{
using mixed_target_validation::require;
using mixed_target_validation::identical;

/// Copy a callback's transient device rows into test-owned HOST evidence, without retaining borrowed pointers.
template<class T>
inline std::vector<T> hostCopy(const T* source, std::size_t count, compute::Queue<PlatformKind::CUDA>& queue)
{
  std::vector<T> result(count);
#ifdef QMC_CUDA2HIP
  checkRuntime(hipMemcpyAsync(result.data(), source, checkedDeviceBytes<T>(count), hipMemcpyDeviceToHost,
      queue.getNative()), "canonical validation D2H");
#else
  checkRuntime(cudaMemcpyAsync(result.data(), source, checkedDeviceBytes<T>(count), cudaMemcpyDeviceToHost,
      queue.getNative()), "canonical validation D2H");
#endif
  queue.sync();
  return result;
}

/// Observe actual KFAC callback tensors under the same seed route as the score gradient.
struct Observations final : DeviceAffineObservationSink
{
  struct Entry
  {
    ParameterRole role;
    std::size_t layer, rows, input, output;
    std::vector<double> activations, sensitivities;
  };
  compute::Queue<PlatformKind::CUDA>& queue;
  std::vector<Entry> entries;

  /// Retain only test evidence; production observation storage remains unchanged.
  explicit Observations(compute::Queue<PlatformKind::CUDA>& q) : queue(q) {}

  /// Consume callback rows before graph scratch can be reused.
  void observe(ParameterRole role, std::size_t layer, const double* activations, const double* sensitivities,
      std::size_t rows, std::size_t input, std::size_t output) override
  {
    Entry entry{role, layer, rows, input, output,
        hostCopy(activations, rows * input, queue), hostCopy(sensitivities, rows * output, queue)};
    for (const auto* values : {&entry.activations, &entry.sensitivities})
      for (double value : *values) require(device_math::isFiniteBinary64(value), "Canonical affine observation is nonfinite");
    entries.push_back(std::move(entry));
  }
};

/// Require exact FP64 observation identity, including callback order and extents.
inline bool equal(const Observations& first, const Observations& second)
{
  if (first.entries.size() != second.entries.size()) return false;
  for (std::size_t i = 0; i < first.entries.size(); ++i)
  {
    const auto& a = first.entries[i]; const auto& b = second.entries[i];
    if (a.role != b.role || a.layer != b.layer || a.rows != b.rows || a.input != b.input || a.output != b.output ||
        !identical(a.activations, b.activations) || !identical(a.sensitivities, b.sensitivities)) return false;
  }
  return true;
}

/// Exercise the new REAL source-domain seam on the already qualified localized Ne4 fixture.
inline void validate()
{
  const ModelShape shape{2, 2, 4, 2, 8, 1, 1};
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.periodic_axes = {true, true, true};
  environment.lattice_vectors = {{{8., 0., 0.}, {0., 8., 0.}, {0., 0., 8.}}};
  const auto parameters = mixed_periodic_validation::parameters(environment, shape);
  const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
  const auto values = parameters.flat_values();
  const std::vector<double> nuclei{0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2};
  const std::vector<double> first{.1, .2, .3, 2.1, .2, .3, .1, 2.2, .3, .1, .2, 2.3};
  auto image = std::make_shared<PreparedDeviceModel>(plan, values, nuclei, 0);
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "canonical validation current device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "canonical validation current device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  const auto cost = PreparedDeviceCanonicalRealHead::required(plan, 2);
  mixed_target_validation::capRejectedWithoutAllocation(metrics, [&] {
    (void)PreparedDeviceCanonicalRealHead::prepare(image, 2, queue, blas, cost.device_bytes - 1);
  });
  {
    auto producer = PreparedDeviceCanonicalRealHead::prepare(image, 2, queue, blas, cost.device_bytes);
    require(producer->deviceWorkspaceBytes() == cost.device_bytes && producer->hostWorkspaceBytes() == cost.host_bytes,
        "Canonical REAL owner differs from pure complete inventory");
    auto cache = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue);
    for (const auto request : {std::pair<std::size_t, std::size_t>{0, 1}, {1, 1}, {3, 2}})
    {
      const auto k = request.first, batch = request.second;
      auto positions = first;
      if (batch == 2) { positions.insert(positions.end(), first.begin(), first.end()); positions[12] += .031; }
      const auto preparation = mixed_target_validation::realPreparation(plan, batch, k);
      PreparedDeviceInference reference(image, batch, queue, blas, SpatialJetMode::FULL_VGL, false, false,
          configuredInferenceStatusPolicy(), preparation);
      PreparedDeviceInference mixed(image, batch, queue, blas, SpatialJetMode::FULL_VGL, false, false,
          configuredInferenceStatusPolicy(), preparation);
      auto arithmetic = std::make_shared<PreparedDeviceMixedValueDense>(cache, mixed.mixedDenseRowCapacity(), queue, blas);
      mixed.setMixedDense(arithmetic);
      auto head = producer->evaluate(positions);
      if (k == 0)
      {
        const int foreign_queue_identity = 0;
        bool refused = false;
        try { head.validate(image, 0, &foreign_queue_identity, batch, 4, 2); }
        catch (const std::invalid_argument&) { refused = true; }
        require(refused, "Canonical head admitted an unrelated queue identity");
      }
      reference.evaluateDeviceCoordinates(head.coordinates());
      mixed.evaluateCanonicalHead(head);
      const auto actual = mixed_target_validation::realFields(mixed, batch, 4);
      const auto expected = mixed_target_validation::realFields(reference, batch, 4);
      mixed_target_validation::realAccuracy(actual, expected, mixed.resultLayout(), batch, 4);
      const auto canonical_phase = hostCopy(head.phases(), batch, queue);
      const auto canonical_log = hostCopy(head.logValues(), batch, queue);
      require(identical(actual.phases, canonical_phase), "Canonical FULL changed its authoritative phase");
      for (std::size_t c = 0; c < batch; ++c)
        require(std::memcmp(&actual.jets[mixed.resultLayout().uncheckedValueOffset(c, 0)], &canonical_log[c], sizeof(double)) == 0,
            "Canonical FULL changed its authoritative physical log amplitude");
      require(arithmetic->statistics().projections != 0, "Canonical FULL did not execute actual mixed jet projections");
      const auto old = head;
      mixed_target_validation::warmed(metrics, queue, [&] {
        head = producer->evaluate(positions); mixed.evaluateCanonicalHead(head);
      });
      bool rejected = false;
      try { mixed.evaluateCanonicalHead(old); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "Stale canonical head was admitted");
      rejected = false;
      try { (void)mixed_target_validation::realFields(mixed, batch, 4); } catch (const std::logic_error&) { rejected = true; }
      require(rejected, "Failed canonical replacement retained a result publication");
      std::cout << "PASS REAL canonical FULL K=" << k << " B=" << batch
                << " same_primal_bits=1 actual_mixed=1 stale_no_publication=1 warm_DeviceBuffer_alloc_free_delta=0\n";
    }
    {
      auto head = producer->evaluate(first);
      PreparedDeviceInference reference(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      PreparedDeviceInference mixed(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      auto primal = std::make_shared<PreparedDeviceMixedValueDense>(cache, 4, queue, blas);
      auto derivative = std::make_shared<PreparedDeviceMixedAffineDerivatives>(cache, 4, queue, blas);
      mixed.setMixedRetainedValueDense(primal);
      PreparedDeviceScoreReverse ref_reverse(image, 1, queue, blas), reverse(image, 1, queue, blas, derivative);
      std::vector<double> expected(plan.parameterCount()), identity(expected.size()), gradient(expected.size()), observed(expected.size());
      reference.evaluateDeviceCoordinates(head.coordinates());
      Observations ordinary(queue), canonical(queue), mixed_observations(queue);
      ref_reverse.observeAffine(reference, ordinary); ref_reverse.downloadResponse(expected);
      reference.evaluateRetainedCanonicalHead(head);
      ref_reverse.observeAffineCanonicalHead(reference, head, canonical); ref_reverse.downloadResponse(identity);
      require(identical(expected, identity) && equal(ordinary, canonical),
          "Canonical FP64 score/KFAC callback seed identity differs");
      mixed.evaluateRetainedCanonicalHead(head);
      reverse.reverseCanonicalHead(mixed, head, {1.}); reverse.downloadResponse(gradient);
      reverse.observeAffineCanonicalHead(mixed, head, mixed_observations); reverse.downloadResponse(observed);
      require(identical(gradient, observed) && !mixed_observations.entries.empty(),
          "Canonical mixed gradient and affine observations use different score seeds");
      long double error2 = 0, reference2 = 0;
      for (std::size_t i = 0; i < expected.size(); ++i)
      {
        require(std::abs(gradient[i] - expected[i]) <= 5e-4 + 1e-3 * std::abs(expected[i]),
            "Canonical mixed score exceeds the unchanged component bound");
        const long double delta = static_cast<long double>(gradient[i]) - expected[i];
        error2 += delta * delta;
        reference2 += static_cast<long double>(expected[i]) * expected[i];
      }
      require(error2 <= 1e-6L * reference2, "Canonical mixed score exceeds the unchanged relative L2 bound");
      bool rejected = false;
      try { (void)mixed.valueHeadTape(); } catch (const std::logic_error&) { rejected = true; }
      require(rejected, "Canonical tape exposed unused ordinary coefficient storage");
      rejected = false;
      try { reverse.reverse(mixed, {1.}); } catch (const std::logic_error&) { rejected = true; }
      require(rejected, "Legacy score accepted a canonical tape without explicit head provenance");
      auto foreign = image->prepareCoordinates(1, checkedDeviceBytes<double>(12)); foreign->upload(first);
      mixed.evaluateDeviceCoordinates(foreign->view(0, 1));
      rejected = false;
      try { reverse.reverseCanonicalHead(mixed, head, {1.}); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "Canonical score accepted a different R source with identical coordinates");
      rejected = false;
      try { reverse.downloadResponse(observed); } catch (const std::logic_error&) { rejected = true; }
      require(rejected, "Failed canonical score validation retained a response publication");
      mixed_target_validation::warmed(metrics, queue, [&] {
        head = producer->evaluate(first); mixed.evaluateRetainedCanonicalHead(head);
        reverse.reverseCanonicalHead(mixed, head, {1.}); reverse.downloadResponse(gradient);
      });
      require(primal->statistics().projections && derivative->statistics().fp32_gemms && derivative->statistics().completed,
          "Canonical score lacks actual mixed primal/reverse execution");
      std::cout << "PASS REAL canonical score canonical_FP64_identity_bits=exact KFAC_observations=exact"
                   " actual_mixed_primal_reverse=1 implicit_head_refused=1 warm_DeviceBuffer_alloc_free_delta=0\n";
    }
    {
      // Same immutable image, one ordinary configuration and one near same-spin coalescence.
      // The tiny finite displacement creates derivative sensitivity without changing the policy threshold.
      auto sensitive = first;
      sensitive[3] = first[0] + 1e-5; sensitive[4] = first[1]; sensitive[5] = first[2];
      for (bool reversed : {false, true})
      {
        auto population = reversed ? sensitive : first;
        const auto& tail = reversed ? first : sensitive;
        population.insert(population.end(), tail.begin(), tail.end());
        const auto head = producer->evaluate(population);
        require(head.derivativeSensitive(reversed ? 0 : 1) && !head.derivativeSensitive(reversed ? 1 : 0),
            "Heterogeneous fixture failed to bind one sensitive and one ordinary head");
        const auto member = head.subview(1, 1);
        require(member.coordinates().sameSourceRange(head.coordinates().subview(1, 1)),
            "Canonical subview changed its source interval");
        require(member.subview(0, 1).coordinates().sameSourceRange(member.coordinates()),
            "Nested canonical subview changed its source interval");
        for (auto range : {std::pair<std::size_t, std::size_t>{0, 0}, {2, 1}, {1, 2}})
        {
          bool refused = false;
          try { (void)head.subview(range.first, range.second); } catch (const std::invalid_argument&) { refused = true; }
          require(refused, "Canonical subview admitted an empty or out-of-range interval");
          refused = false;
          try { (void)head.coordinates().subview(range.first, range.second); }
          catch (const std::invalid_argument&) { refused = true; }
          require(refused, "Coordinate subview admitted an empty or out-of-range interval");
        }
        {
          PreparedDeviceInference too_small(image, 1, queue, blas, SpatialJetMode::FULL_VGL);
          auto arithmetic = std::make_shared<PreparedDeviceMixedValueDense>(cache,
              too_small.mixedDenseRowCapacity(), queue, blas);
          too_small.setMixedDense(arithmetic);
          bool refused = false;
          try { too_small.evaluateCanonicalHead(head); } catch (const std::invalid_argument&) { refused = true; }
          require(refused && !too_small.sensitiveDerivativeConfigurations() && !too_small.ordinaryDerivativeConfigurations(),
              "Heterogeneous FULL exceeded output capacity before singleton scatter");
        }
        for (std::size_t k : {std::size_t(0), std::size_t(3)})
        {
          const auto preparation = mixed_target_validation::realPreparation(plan, 2, k);
          PreparedDeviceInference full(image, 2, queue, blas, SpatialJetMode::FULL_VGL, false, false,
              configuredInferenceStatusPolicy(), preparation);
          auto arithmetic = std::make_shared<PreparedDeviceMixedValueDense>(cache, full.mixedDenseRowCapacity(), queue, blas);
          full.setMixedDense(arithmetic);
          full.evaluateCanonicalHead(head);
          const auto actual = mixed_target_validation::realFields(full, 2, 4);
          require(full.sensitiveDerivativeConfigurations() == 1 && full.ordinaryDerivativeConfigurations() == 1 &&
              full.sensitiveFp64AffineProjections() && full.ordinaryFp32AffineProjections(),
              "Heterogeneous FULL promoted an ordinary member or lost selected arithmetic receipts");
          const auto tiles = k ? (4 / k + (4 % k != 0)) : 1;
          const auto expected_affines = tiles * (1 + 6 * shape.attention_blocks);
          require(full.sensitiveFp64AffineProjections() == expected_affines &&
              full.ordinaryFp32AffineProjections() == expected_affines,
              "Heterogeneous FULL counters omitted a complete or tail target tile");
          for (std::size_t c = 0; c < 2; ++c)
          {
            const auto one = head.subview(c, 1);
            PreparedDeviceInference singleton(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, false,
                configuredInferenceStatusPolicy(), mixed_target_validation::realPreparation(plan, 1, k));
            auto single_arithmetic = std::make_shared<PreparedDeviceMixedValueDense>(cache,
                singleton.mixedDenseRowCapacity(), queue, blas);
            singleton.setMixedDense(single_arithmetic); singleton.evaluateCanonicalHead(one);
            const auto expected = mixed_target_validation::realFields(singleton, 1, 4);
            const auto stride = full.resultLayout().configuration_stride;
            require(actual.phases[c] == expected.phases[0] &&
                identical(std::vector<double>(actual.jets.begin() + c * stride, actual.jets.begin() + (c + 1) * stride), expected.jets) &&
                identical(std::vector<double>(actual.ratios.begin() + c * 4, actual.ratios.begin() + (c + 1) * 4), expected.ratios),
                "Heterogeneous FULL changed singleton arithmetic or result order");
          }
        }
        {
          // The native factory intentionally caps this four-nucleus fixture at one.
          // Exercise the existing explicitly admitted numerical owner without widening that policy.
          std::shared_ptr<const PreparedDeviceModel> active_image = image;
          std::shared_ptr<const PreparedDeviceMixedDenseParameters> active_cache = cache;
          PsiFormerAcceleratorSharedModel owner(PreparedDeviceCrowdHooks::backend(),
              makePsiFormerDeviceLayout(plan).fingerprint, image->version());
          std::mutex control;
          PreparedDeviceProductSession session(active_image, owner, control, 1, ProductSessionProfile::SCORE_ONLY,
              2, {}, true, false, &active_cache);
          const auto descriptor = estimateMixedScoreTileStorage(plan, 2);
          require(session.scoreCapacity() == 2 && session.deviceWorkspaceBytes() == descriptor.device_bytes &&
              session.hostWorkspaceBytes() >= descriptor.host_numeric_bytes,
              "Explicit heterogeneous score owner lost its admitted capacity or cold inventory");
          session.prepareScoreContractions(1, image->version());
          const std::vector<double> weights{.25, -.75};
          std::vector<std::complex<double>> batched(values.size()), serial(values.size());
          session.beginScoreContraction(1, image->version());
          session.accumulateWeightedScoreTile(population.data(), weights.data(), 2, image->version());
          session.finishScoreContraction(image->version());
          session.downloadScoreContraction(0, batched.data(), batched.size(), image->version());
          session.beginScoreContraction(1, image->version());
          for (std::size_t c = 0; c < 2; ++c)
            session.accumulateWeightedScoreTile(population.data() + 12 * c, weights.data() + c, 1, image->version());
          session.finishScoreContraction(image->version());
          session.downloadScoreContraction(0, serial.data(), serial.size(), image->version());
          require(identical(batched, serial), "Heterogeneous weighted tile changed singleton fold order or sample weights");
          std::vector<double> direction(values.size());
          for (std::size_t i = 0; i < direction.size(); ++i) direction[i] = .01 * std::sin(.19 * (i + 1));
          session.prepareScoreDirection(direction, image->version());
          const auto products = session.scoreJvpPreparedTile(population.data(), 2, image->version());
          for (std::size_t c = 0; c < 2; ++c)
          {
            const auto score = session.score(population.data() + c * 12, image->version());
            long double dot = 0;
            for (std::size_t i = 0; i < direction.size(); ++i) dot += direction[i] * score[i];
            require(std::abs(products[c] - dot) <= 2e-5L * (1 + std::abs(dot)),
                "Heterogeneous canonical JVP/VJP lost arithmetic or configuration correspondence");
          }
          require(session.ordinaryDerivativeConfigurations() && session.ordinaryFp32PrimalAffines() &&
              session.sensitiveDerivativeConfigurations() && session.sensitiveFp64PrimalAffines() &&
              session.sensitiveFp64ReverseAffines() && session.sensitiveFp64JvpAffines() &&
              session.mixedReverseStatistics()->fp32_gemms && session.mixedJvpStatistics()->fp32_gemms,
              "Heterogeneous score products lost ordinary FP32 or selected FP64 launch receipts");
          const auto selected_before = session.sensitiveDerivativeConfigurations();
          const auto ordinary_before = session.ordinaryDerivativeConfigurations();
          const std::vector<double> zero_weights{0., 0.};
          session.beginScoreContraction(1, image->version());
          session.accumulateWeightedScoreTile(population.data(), zero_weights.data(), 2, image->version());
          session.finishScoreContraction(image->version());
          require(session.sensitiveDerivativeConfigurations() == selected_before + 1 &&
              session.ordinaryDerivativeConfigurations() == ordinary_before + 1,
              "Zero score weights skipped configuration arithmetic or head validation");
          session.beginScoreContraction(1, image->version());
          const std::vector<double> bad_weights{1., std::numeric_limits<double>::quiet_NaN()};
          bool failed = false;
          try { session.accumulateWeightedScoreTile(population.data(), bad_weights.data(), 2, image->version()); }
          catch (const std::invalid_argument&) { failed = true; }
          require(failed, "Sensitive weighted tile admitted a nonfinite late member");
          failed = false;
          try { session.downloadScoreContraction(0, serial.data(), serial.size(), image->version()); }
          catch (const std::logic_error&) { failed = true; }
          require(failed, "Failed weighted tile exposed partial contraction publication");
        }
        const auto stale_member = head.subview(0, 1);
        (void)producer->evaluate(first);
        bool refused = false;
        try { (void)stale_member.derivativeSensitive(0); } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "Stale canonical subview retained sensitivity authority");
      }
      std::cout << "PASS REAL sensitive_population both_orders=exact ordinary_FP32=1 selected_FP64=1"
                   " FULL_K0_K3=exact subview_bounds_stale=reject\n";
    }
    auto external = image->prepareCoordinates(2, checkedDeviceBytes<double>(24));
    auto two = first; two.insert(two.end(), first.begin(), first.end()); external->upload(two);
    require(!external->view(0, 1).sameSourceRange(external->view(1, 1)), "Coordinate range comparison ignored interval begin");
    bool rejected = false;
    const auto old = external->view(0, 1);
    try { external->copyFrom(old, queue); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Coordinate import admitted self-copy");
    rejected = false;
    try { (void)producer->evaluateDeviceCoordinates(old); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Failed coordinate replacement did not revoke its old view");
    auto head = producer->evaluate(first);
    auto successor = image->makeUpdatedImage(values, 1);
    rejected = false;
    try { producer->validateModelRebind(successor); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Canonical producer rebound with a live issued borrow");
    head = {};
    producer->rebindModel(successor);
    head = producer->evaluate(first);
    auto fresh = PreparedDeviceCanonicalRealHead::prepare(successor, 2, queue, blas, cost.device_bytes);
    const auto fresh_head = fresh->evaluate(first);
    require(identical(hostCopy(head.inverses(), 32, queue), hostCopy(fresh_head.inverses(), 32, queue)),
        "Canonical model rebind differs from a fresh inverse owner");
    rejected = false;
    try { head.validate(image, 0, &queue, 1, 4, 2); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Canonical head admitted a foreign model image/version");
    std::cout << "PASS REAL canonical provenance cap_minus_one=1 coordinate_generation=1 rebind_fresh_inverse_bits=exact"
                 " foreign_image_refused=1\n";
    head = {};
    auto unsafe_values = values;
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& parameter = plan.parameter(role);
      for (std::size_t row = 0; row < shape.feature_dimension; ++row)
        for (std::size_t d = 0; d < shape.determinants; ++d)
          unsafe_values[parameter.begin + row * shape.determinants * 4 + d * 4] *= 1e-6;
    }
    // A column scaling can make a pivot small without corrupting its FP64 inverse.
    auto low_pivot_image = successor->makeUpdatedImage(unsafe_values, 2);
    producer->rebindModel(low_pivot_image);
    head = producer->evaluate(first);
    const auto low_factors = hostCopy(head.factorizations(), shape.determinants, queue);
    require(std::any_of(low_factors.begin(), low_factors.end(), [](const auto& f) {
      return f.minimum_scaled_pivot < 1e-3;
    }), "Canonical low-pivot positive did not exercise a small pivot");
    std::cout << "PASS REAL canonical accurate_low_pivot inverse_certificate=valid mixed_jet_accuracy_claim=0\n";
    // A certified sensitive head selects an explicit FP64 derivative island before mixed work.
    {
      auto low_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(low_pivot_image, queue);
      require(low_cache->usable(), "Low-pivot retained regression has no usable mixed cache");
      auto primal = std::make_shared<PreparedDeviceMixedValueDense>(low_cache, 4, queue, blas);
      auto derivative = std::make_shared<PreparedDeviceMixedAffineDerivatives>(low_cache, 4, queue, blas);
      PreparedDeviceInference retained(low_pivot_image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      PreparedDeviceScoreReverse reverse(low_pivot_image, 1, queue, blas, derivative);
      retained.setMixedRetainedValueDense(primal);
      retained.evaluateRetainedCanonicalHead(head);
      reverse.reverseCanonicalHead(retained, head, {1.});
      std::vector<double> response(values.size());
      reverse.downloadResponse(response);
      require(response.size() == values.size() && std::all_of(response.begin(), response.end(), [](double value) {
        return device_math::isFiniteBinary64(value);
      }), "Certified low-pivot retained SCORE did not publish a complete finite response");
      require(!primal->statistics().projections && !derivative->statistics().fp32_gemms &&
          retained.valueTapeDerivativeIsland() && !retained.valueTapeMixedParameters() &&
          retained.sensitiveDerivativeConfigurations() == 1 && retained.sensitiveFp64AffineProjections() &&
          reverse.sensitiveFp64AffineProjections(),
          "Sensitive retained SCORE did not report truthful FP64 primal/reverse arithmetic");
      PreparedDeviceInference reference(low_pivot_image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      PreparedDeviceScoreReverse reference_reverse(low_pivot_image, 1, queue, blas);
      reference.evaluateRetainedCanonicalHead(head);
      reference_reverse.reverseCanonicalHead(reference, head, {1.});
      std::vector<double> expected(response.size()); reference_reverse.downloadResponse(expected);
      require(identical(expected, response), "Selected sensitive score differs from same-head FP64 reference");
      PreparedDeviceScoreJvp jvp(low_pivot_image, 1, queue, blas, derivative);
      std::vector<double> direction(response.size()), product(1);
      long double dot = 0;
      for (std::size_t i = 0; i < direction.size(); ++i)
      { direction[i] = .01 * std::sin(.19 * (i + 1)); dot += direction[i] * response[i]; }
      jvp.prepareDirection(direction); jvp.evaluatePrepared(retained, &head); jvp.downloadProducts(product);
      require(jvp.sensitiveFp64AffineProjections() && !derivative->statistics().fp32_gemms &&
          std::abs(product[0] - dot) <= 2e-10L * (1 + std::abs(dot)),
          "Sensitive canonical JVP/VJP disagree or report hidden FP32 arithmetic");
    }
    // The same approximate model must retain the original raw FP32-produced-head pivot guard.
    {
      auto low_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(low_pivot_image, queue);
      auto primal = std::make_shared<PreparedDeviceMixedValueDense>(low_cache, 4, queue, blas);
      PreparedDeviceInference raw(low_pivot_image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      raw.setMixedRetainedValueDense(primal);
      rejected = false;
      try { raw.evaluate(first); }
      catch (const MixedPrecisionRetryRequired& error)
      {
        rejected = mixed_target_validation::diagnosticCount(error.what(), "small_determinant_pivot") != 0;
      }
      require(rejected, "Raw FP32-produced low-pivot head lost its original rejection");
    }
    std::cout << "PASS REAL canonical low_pivot_retained_SCORE declared_FP64_derivative_island=1 actual_FP32_GEMMs=0"
                 " certified_FP64_head=accepted raw_FP32_head=typed_reject FP64_score_bits=exact JVP_VJP=consistent\n";
    head = {};
    // A zero orbital column has no inverse and must never mint a head borrow.
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& parameter = plan.parameter(role);
      for (std::size_t row = 0; row < shape.feature_dimension; ++row)
        for (std::size_t d = 0; d < shape.determinants; ++d)
          unsafe_values[parameter.begin + row * shape.determinants * 4 + d * 4] = 0;
    }
    auto unsafe_image = low_pivot_image->makeUpdatedImage(unsafe_values, 3);
    producer->rebindModel(unsafe_image);
    rejected = false;
    try { (void)producer->evaluate(first); }
    catch (const MixedPrecisionRetryRequired& error)
    {
      rejected = mixed_target_validation::diagnosticCount(error.what(), "invalid_factor") != 0 ||
          mixed_target_validation::diagnosticCount(error.what(), "uncertified_fp64_inverse") != 0;
    }
    catch (const InferenceStatusError& error) { rejected = error.stage() != 0; }
    require(rejected, "Canonical actual-used-head inverse safeguard did not reject a noninvertible column");
    rejected = false;
    try { (void)producer->view(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Failed canonical head guard minted a usable view");
    std::cout << "PASS REAL canonical actual_head_guard typed_inverse_rejection=1 no_head_publication=1\n";

  }
  {
    const auto start = metrics.snapshot();
    const auto one = PreparedDeviceCanonicalRealHead::required(plan, 1);
    auto temporary = PreparedDeviceCanonicalRealHead::prepare(image, 1, queue, blas, one.device_bytes);
    auto retired = temporary->evaluate(first);
    temporary.reset();
    require(metrics.snapshot().live_bytes == start.live_bytes + one.device_bytes,
        "Retired producer freed storage still retained by an issued view");
    bool rejected = false;
    try { retired.validate(image, 0, &queue, 1, 4, 2); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Retired canonical producer left a usable borrow");
    retired = {};
    require(metrics.snapshot().live_bytes == start.live_bytes, "Last canonical borrow retained its device owner");
  }
  queue.sync();
  require(metrics.snapshot().live_bytes == before.live_bytes && metrics.snapshot().failed_frees == before.failed_frees,
      "Canonical REAL validation retained numeric owners after retirement");
}
} // namespace qmcplusplus::psiformer::device::canonical_head_validation

/// Run the changed REAL canonical source-domain and observation seams in the focused GPU executable.
inline void validateRealCanonicalHeads()
{
  qmcplusplus::psiformer::device::canonical_head_validation::validate();
}
#endif
