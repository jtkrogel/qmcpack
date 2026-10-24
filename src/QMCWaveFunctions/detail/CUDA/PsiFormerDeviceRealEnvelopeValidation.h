//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDeviceRealEnvelopeValidation.h
 * @brief Bounded REAL request-local envelope ownership and exact-value oracles.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_REAL_ENVELOPE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_REAL_ENVELOPE_VALIDATION_H

#include "PsiFormerDeviceBatchBackend.h"
#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Reuse the existing well-conditioned Ne64 fixture, never a new production model.
struct RealEnvelopeValidationAccess
{
  static void run(compute::Queue<PlatformKind::CUDA>& queue)
  {
    const char* key = "PSIFORMER_GPU_REAL_VIRTUAL_ENVELOPE_REUSE";
    std::optional<std::string> previous;
    if (const char* value = std::getenv(key)) previous = value;
    struct Restore
    {
      const char* key;
      const std::optional<std::string>& previous;
      ~Restore() { if (previous) setenv(key, previous->c_str(), 1); else unsetenv(key); }
    } restore{key, previous};
    unsetenv(key);
    if (configuredRealVirtualEnvelopeReuse())
      throw std::runtime_error("REAL envelope default is not OFF");
    setenv(key, "invalid", 1);
    bool invalid = false;
    try { configuredRealVirtualEnvelopeReuse(); }
    catch (const std::invalid_argument&) { invalid = true; }
    if (!invalid) throw std::runtime_error("REAL envelope accepted invalid selector");

    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const ModelShape shape{32, 32, 64, 1, 64, 4, 1};
    auto parameters = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
    const auto ne = shape.electrons();
    std::vector<double> nuclei(3 * ne), base(3 * ne);
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto grid = d == 0 ? e % 4 : d == 1 ? (e / 4) % 4 : e / 16;
        nuclei[3 * e + d] = 1. + 6. * grid;
        base[3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d;
      }
    auto image = std::make_shared<PreparedDeviceModel>(plan, parameters.flat_values(), nuclei, 0);
    AcceleratorBlasHandle blas(queue);
    PreparedDeviceInference scalar(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
    PreparedDeviceInference candidate(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
    if (!candidate.admitsSparseEnvelopeAnchor())
      throw std::runtime_error("Ne64 legacy REAL did not admit envelope owner");
    const auto baseline = candidate.deviceWorkspaceBytes();
    candidate.prepareSparseEnvelopeAnchor();
    if (candidate.deviceWorkspaceBytes() - baseline != candidate.sparseEnvelopeDeviceBytes())
      throw std::runtime_error("REAL envelope prepared bytes differ from exact cold cost");
    PreparedDeviceInference retained(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
    if (retained.admitsSparseEnvelopeAnchor())
      throw std::runtime_error("Retained score admitted ordinary envelope owner");
    // This fixture exercises compact live B3/B2/B1 outputs, unlike the B1-only
    // mixed-geometry fixture. Size destinations from the actual live layout.
    const auto download = [](PreparedDeviceInference& graph) {
      mixed_periodic_validation::Output result;
      const auto& layout = graph.resultLayout();
      result.phases.resize(layout.configuration_count);
      result.jets.resize(spatialJetSpanElements(layout));
      result.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
      graph.downloadResults(result.phases, result.jets, result.ratios);
      return result;
    };
    const auto compare = [&](const std::vector<double>& positions, const std::vector<std::size_t>& refs,
        const std::vector<std::size_t>& moved, std::size_t begin, std::size_t generation, bool last) {
      scalar.evaluate(positions);
      candidate.beginSparseEnvelopeTile(&candidate, generation, 2, begin, refs.data(), moved.data(), refs.size());
      try
      {
        candidate.evaluate(positions);
        candidate.endSparseEnvelopeTile(true, last);
      }
      catch (...) { candidate.endSparseEnvelopeTile(false, true); throw; }
      const auto a = download(scalar);
      const auto b = download(candidate);
      mixed_periodic_validation::identical(a, b);
    };
    std::vector<double> first;
    for (std::size_t b = 0; b < 3; ++b)
    {
      first.insert(first.end(), base.begin(), base.end());
      if (b == 1) first[b * 3 * ne + 2] += .03;
      if (b == 2) first[b * 3 * ne + 3 * 4] += .02;
    }
    compare(first, {0, 1, 0}, {ne, ne, 4}, 0, 1, false);
    unsigned long long reused[4]{};
    candidate.sparse_envelope_->counters.download(reused, 4);
    if (reused[0] != 0 || !candidate.sparse_envelope_->ready)
      throw std::runtime_error("Capture tile reused an unadmitted bank");

    auto tail = std::vector<double>(first.begin(), first.begin() + 6 * ne);
    tail[3 * (ne - 1)] += .04;
    tail[3 * ne + 3 * 5] += .02;
    const auto before = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
    compare(tail, {0, 1}, {ne - 1, 5}, 3, 1, false);
    candidate.sparse_envelope_->counters.download(reused, 4);
    if (reused[0] != ne - 1)
      throw std::runtime_error("REAL envelope did not select only unchanged reference-zero rows");
    auto short_tail = base;
    short_tail[0] += .01;
    short_tail[3] += .02; // Descriptor-unmentioned row must fail its bit proof.
    compare(short_tail, {0}, {0}, 5, 1, false);
    candidate.sparse_envelope_->counters.download(reused, 4);
    if (reused[0] != 2 * ne - 3)
      throw std::runtime_error("REAL envelope reused an unequal row position");
    auto broken = tail;
    broken[3 * ne + 2] = std::numeric_limits<double>::quiet_NaN();
    const std::size_t failed_refs[]{0, 1}, failed_moves[]{ne - 1, 5};
    candidate.beginSparseEnvelopeTile(&candidate, 1, 2, 3, failed_refs, failed_moves, 2);
    bool failed = false;
    try { candidate.evaluate(broken); }
    catch (const std::runtime_error&) { failed = true; }
    candidate.endSparseEnvelopeTile(false, true);
    if (!failed || candidate.sparse_envelope_->ready || candidate.sparse_envelope_->tile.references)
      throw std::runtime_error("Nonzero-walker failure retained an envelope request");
    compare(first, {0, 1, 0}, {ne, ne, 4}, 0, 2, false);
    const auto warm_after = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
    if (warm_after.allocation_calls != before.allocation_calls || warm_after.live_bytes != before.live_bytes)
      throw std::runtime_error("Warmed REAL envelope calls allocated scratch");
    auto successor = image->makeUpdatedImage(parameters.flat_values(), 1);
    const auto rebind_before = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
    scalar.rebindModel(successor);
    candidate.rebindModel(successor);
    if (candidate.sparse_envelope_->ready)
      throw std::runtime_error("Model rebind retained a prior envelope bank");
    compare(tail, {0, 1}, {ne - 1, 5}, 3, 2, true); // No matching new-image base: canonical.
    const auto after = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
    if (after.allocation_calls != rebind_before.allocation_calls || after.live_bytes != rebind_before.live_bytes)
      throw std::runtime_error("REAL envelope rebind allocated scratch");

    setenv(key, "1", 1);
    DeviceExecutionPreparation preparation;
    preparation.vgl = {1, std::numeric_limits<std::size_t>::max(),
                          std::numeric_limits<std::size_t>::max(), 0, false};
    PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation, true);
    setenv(key, "0", 1);
    if (!backend.sparseValueHintsRequested())
      throw std::runtime_error("Backend reread changed envelope selector");
    auto context = backend.prepare(3);
    auto* typed = dynamic_cast<PreparedDeviceBatchBackend::Context*>(context.get());
    const auto extra = typed->hooks.values_->sparseEnvelopeDeviceBytes();
    const auto future = estimatePreparedTargetVglCost(plan, 3, preparation.vgl).device_workspace_bytes;
    const auto policy = configuredDeviceMemoryAdmission();
    auto& metrics = deviceMemoryMetrics(candidate.positions_.deviceOrdinal());
    const auto live = metrics.snapshot().live_bytes;
    if (!policy.limit_bytes || live + future + extra >= *policy.limit_bytes - policy.external_reserve_bytes)
      throw std::runtime_error("REAL envelope validator needs baseline plus optional owner budget");
    {
      const auto hold = *policy.limit_bytes - policy.external_reserve_bytes - live - future - extra + 1;
      auto denied_optional = metrics.reserveAllocation(hold);
      if (backend.prepareSparseValueReuse(*context) || typed->hooks.values_->sparse_envelope_)
        throw std::runtime_error("Baseline-fit/anchor-minus-one did not select canonical fallback");
    }
    auto fitting = backend.prepare(3);
    if (!backend.prepareSparseValueReuse(*fitting))
      throw std::runtime_error("Admitted REAL envelope owner did not prepare");
    std::cout << "PASS real_virtual_envelope Ne64 exact_bits R2 anchor0 B3_B2_B1 row_bit_miss failure rebind warm0 cost_minus1 cold_flag retained_reject\n";
    runPairGeometry(queue);
  }

  /// Exercise the production selector/owners independently of the orbital component.
  static void runPairGeometry(compute::Queue<PlatformKind::CUDA>& queue)
  {
    const char* key = "PSIFORMER_GPU_PERIODIC_VALUE_PAIR_GEOMETRY";
    const std::optional<std::string> previous = std::getenv(key)
        ? std::optional<std::string>(std::getenv(key)) : std::nullopt;
    struct Restore
    {
      const char* key;
      const std::optional<std::string>& previous;
      ~Restore()
      {
        if (previous)
          setenv(key, previous->c_str(), 1);
        else
          unsetenv(key);
      }
    } restore{key, previous};
    unsetenv(key);
    if (configuredPeriodicValuePairGeometry())
      throw std::runtime_error("Periodic VALUE pair default is not OFF");
    setenv(key, "invalid", 1);
    bool invalid = false;
    try { configuredPeriodicValuePairGeometry(); }
    catch (const std::invalid_argument&) { invalid = true; }
    if (!invalid)
      throw std::runtime_error("Periodic VALUE pair selector accepted invalid input");

    // Use the existing well-conditioned model; no production physics is replaced.
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const ModelShape shape{32, 32, 64, 1, 64, 4, 1};
    auto parameters = mixed_periodic_validation::parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
    const auto ne = shape.electrons();
    std::vector<double> nuclei(3 * ne), positions(4 * 3 * ne);
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto grid = d == 0 ? e % 4 : d == 1 ? (e / 4) % 4 : e / 16;
        nuclei[3 * e + d] = 1. + 6. * grid;
        for (std::size_t b = 0; b < 4; ++b)
          positions[b * 3 * ne + 3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d + .001 * b;
      }
    auto image = std::make_shared<PreparedDeviceModel>(plan, parameters.flat_values(), nuclei, 0);
    AcceleratorBlasHandle blas(queue);
    PreparedDeviceInference scalar(image, 4, queue, blas, SpatialJetMode::VALUE_ONLY);
    PreparedDeviceInference candidate(image, 4, queue, blas, SpatialJetMode::VALUE_ONLY);
    PreparedDeviceInference retained(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
    const auto baseline = candidate.deviceWorkspaceBytes();
    const auto bytes = estimatePeriodicValuePairCost(4, ne, shape.nuclei).device_bytes;
    candidate.preparePeriodicValuePairs();
    retained.preparePeriodicValuePairs();
    if (!candidate.hasPeriodicValuePairs() || retained.hasPeriodicValuePairs() ||
        candidate.deviceWorkspaceBytes() - baseline != bytes)
      throw std::runtime_error("Periodic VALUE pair eligibility/capacity bytes differ");
    const auto download = [](PreparedDeviceInference& graph) {
      mixed_periodic_validation::Output result;
      const auto& layout = graph.resultLayout();
      result.phases.resize(layout.configuration_count);
      result.jets.resize(spatialJetSpanElements(layout));
      result.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
      graph.downloadResults(result.phases, result.jets, result.ratios);
      return result;
    };
    const auto compare = [&](std::size_t live) {
      const std::vector<double> prefix(positions.begin(), positions.begin() + live * 3 * ne);
      scalar.evaluate(prefix);
      candidate.evaluate(prefix);
      mixed_periodic_validation::identical(download(scalar), download(candidate));
    };
    compare(4);
    auto successor = image->makeUpdatedImage(parameters.flat_values(), 1);
    auto& metrics = deviceMemoryMetrics(candidate.positions_.deviceOrdinal());
    const auto warm_before = metrics.snapshot();
    compare(2);
    compare(1);
    compare(4);
    scalar.rebindModel(successor);
    candidate.rebindModel(successor);
    compare(2);
    const auto warm_after = metrics.snapshot();
    if (warm_after.allocation_calls != warm_before.allocation_calls ||
        warm_after.live_bytes != warm_before.live_bytes)
      throw std::runtime_error("Periodic VALUE pair tails/rebind allocated warm scratch");

    // The backend must charge BOTH simultaneous banks plus future ordinary FULL.
    DeviceExecutionPreparation preparation;
    preparation.vgl = {1, std::numeric_limits<std::size_t>::max(),
                          std::numeric_limits<std::size_t>::max(), 0, false};
    setenv(key, "1", 1);
    PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation, true);
    setenv(key, "0", 1);
    if (!backend.periodicValuePairsRequested())
      throw std::runtime_error("Periodic VALUE pair backend reread its cold flag");
    auto ordinary = backend.prepare(4);
    auto virtuals = backend.prepare(16);
    auto* first = dynamic_cast<PreparedDeviceBatchBackend::Context*>(ordinary.get());
    auto* second = dynamic_cast<PreparedDeviceBatchBackend::Context*>(virtuals.get());
    first->hooks.values_->prepareSparseEnvelopeAnchor();
    second->hooks.values_->prepareSparseEnvelopeAnchor();
    const auto* first_anchor = first->hooks.values_->sparse_envelope_.get();
    const auto* second_anchor = second->hooks.values_->sparse_envelope_.get();
    const auto first_baseline_bytes = first->hooks.values_->deviceWorkspaceBytes();
    const auto second_baseline_bytes = second->hooks.values_->deviceWorkspaceBytes();
    const auto extra = first->hooks.values_->periodicValuePairDeviceBytes() +
        second->hooks.values_->periodicValuePairDeviceBytes();
    if (extra != estimatePeriodicValuePairCost(20, ne, shape.nuclei).device_bytes)
      throw std::runtime_error("Periodic VALUE pair combined owner cost differs");
    const auto future = estimatePreparedTargetVglCost(plan, 4, preparation.vgl).device_workspace_bytes;
    const auto policy = configuredDeviceMemoryAdmission();
    const auto live = metrics.snapshot().live_bytes;
    if (!policy.limit_bytes || live + future + extra >= *policy.limit_bytes - policy.external_reserve_bytes)
      throw std::runtime_error("Periodic VALUE pair validator lacks admitted baseline budget");
    {
      const auto hold = *policy.limit_bytes - policy.external_reserve_bytes - live - future - extra + 1;
      auto denied_optional = metrics.reserveAllocation(hold);
      if (backend.preparePeriodicValuePairs(*ordinary, virtuals.get()) ||
          first->hooks.values_->hasPeriodicValuePairs() || second->hooks.values_->hasPeriodicValuePairs())
        throw std::runtime_error("Periodic VALUE pair aggregate minus-one did not roll back both banks");
      if (first->hooks.values_->sparse_envelope_.get() != first_anchor ||
          second->hooks.values_->sparse_envelope_.get() != second_anchor ||
          first->hooks.values_->deviceWorkspaceBytes() != first_baseline_bytes ||
          second->hooks.values_->deviceWorkspaceBytes() != second_baseline_bytes)
        throw std::runtime_error("Periodic VALUE pair rollback changed a pre-existing anchor owner");
    }
    auto fitting_first = backend.prepare(4);
    auto fitting_second = backend.prepare(16);
    if (!backend.preparePeriodicValuePairs(*fitting_first, fitting_second.get()))
      throw std::runtime_error("Periodic VALUE pair admitted owners did not prepare");
    std::cout << "PASS real_periodic_pair_geometry Ne64 exact_bits B4_B2_B1_B4 rebind warm0 retained_reject combined_B4_B16_minus1 cold_flag\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
