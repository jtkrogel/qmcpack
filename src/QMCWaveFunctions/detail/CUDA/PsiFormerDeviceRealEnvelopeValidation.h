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

  /// Compare the ordinary molecular/Gamma owner at the shared projection boundary.
  static void runProjection(compute::Queue<PlatformKind::CUDA>& queue, bool pair_geometry = false)
  {
    const char* key = "PSIFORMER_GPU_ECP_VALUE_PROJECTED_BACKFLOW";
    const std::optional<std::string> previous = std::getenv(key)
        ? std::optional<std::string>(std::getenv(key)) : std::nullopt;
    struct Restore
    {
      const char* key;
      const std::optional<std::string>& previous;
      ~Restore() { if (previous) setenv(key, previous->c_str(), 1); else unsetenv(key); }
    } restore{key, previous};
    std::size_t comparisons = 0;
    double maximum = 0.;
    for (const bool periodic : {false, true})
      for (const std::size_t determinants : {16, 64})
      {
        ExecutionEnvironment environment;
        if (periodic)
        {
          environment.boundary = BoundaryCondition::PERIODIC;
          environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
          environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
          environment.periodic_axes = {true, true, true};
        }
        const ModelShape shape{26, 18, 44, determinants, 64, 4, 1};
        auto parameters = mixed_periodic_validation::parameters(environment, shape);
        const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
        constexpr std::size_t ne = 44;
        std::vector<double> nuclei(3 * ne), positions(3 * ne * 3);
        for (std::size_t e = 0; e < ne; ++e)
          for (std::size_t d = 0; d < 3; ++d)
          {
            const auto grid = d == 0 ? e % 4 : d == 1 ? (e / 4) % 4 : e / 16;
            nuclei[3 * e + d] = 1. + 6. * grid;
            for (std::size_t b = 0; b < 3; ++b)
              positions[b * 3 * ne + 3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d + .001 * b;
          }
        auto image = std::make_shared<PreparedDeviceModel>(plan, parameters.flat_values(), nuclei, 0);
        AcceleratorBlasHandle blas(queue);
        PreparedDeviceInference scalar(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
        PreparedDeviceInference candidate(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
        std::unique_ptr<PreparedDeviceInference> grouped_reference;
        if (periodic && pair_geometry)
        {
          candidate.preparePeriodicValuePairs();
          if (!candidate.hasPeriodicValuePairs())
            throw std::runtime_error("N44 periodic pair owner did not prepare");
          grouped_reference = std::make_unique<PreparedDeviceInference>(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
        }
        setenv(key, "1", 1);
        PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE,
            DeviceExecutionPreparation{}, true);
        if (!backend.sparseValueHintsRequested())
          throw std::runtime_error("REAL ECP backend did not admit Gamma projection hints");
        setenv(key, "0", 1);
        const auto compare = [&](std::size_t live, bool translate) {
          std::vector<double> input(positions.begin(), positions.begin() + live * 3 * ne);
          if (translate)
            for (std::size_t d = 0; d < 3; ++d)
              input[d] += environment.lattice_vectors[0][d];
          scalar.evaluate(input);
          const std::size_t refs[]{0, 1, 0}, moved[]{ne, ne, 4};
          setenv(key, "1", 1);
          candidate.beginSparseEnvelopeTile(&candidate, 1, 2, 0, refs, moved, live);
          if (!candidate.ecp_projected_backflow_active_)
            throw std::runtime_error("REAL ECP projection did not activate");
          try { candidate.evaluate(input); candidate.endSparseEnvelopeTile(true, true); }
          catch (...) { candidate.endSparseEnvelopeTile(false, true); throw; }
          if (candidate.ecp_projected_backflow_active_)
            throw std::runtime_error("REAL ECP projection escaped tile lifetime");
          if (grouped_reference)
          {
            grouped_reference->beginSparseEnvelopeTile(grouped_reference.get(), 1, 2, 0, refs, moved, live);
            try { grouped_reference->evaluate(input); grouped_reference->endSparseEnvelopeTile(true, true); }
            catch (...) { grouped_reference->endSparseEnvelopeTile(false, true); throw; }
            const auto count = spatialJetSpanElements(candidate.orbitals_.orbitals);
            std::vector<double> baseline(count), banked(count);
            grouped_reference->orbitalStorage().download(baseline.data(), count);
            candidate.orbitalStorage().download(banked.data(), count);
            if (std::memcmp(baseline.data(), banked.data(), count * sizeof(double)))
              throw std::runtime_error("Grouped REAL Pair bank changed canonical orbital bits");
          }
          const auto count = spatialJetSpanElements(candidate.orbitals_.orbitals);
          std::vector<double> a(count), b(count);
          scalar.orbitalStorage().download(a.data(), count);
          candidate.orbitalStorage().download(b.data(), count);
          for (std::size_t i = 0; i < count; ++i)
          {
            const auto error = std::abs(a[i] - b[i]);
            maximum = std::max(maximum, error);
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || error > 2e-12 * (1 + std::abs(a[i])))
              throw std::runtime_error("REAL ECP projected orbitals differ from scalar contraction");
            ++comparisons;
          }
          const auto download = [](PreparedDeviceInference& graph) {
            mixed_periodic_validation::Output result;
            const auto& layout = graph.resultLayout();
            result.phases.resize(layout.configuration_count);
            result.jets.resize(spatialJetSpanElements(layout));
            result.ratios.resize(layout.configuration_count * layout.laplacian_lanes);
            graph.downloadResults(result.phases, result.jets, result.ratios);
            return result;
          };
          const auto first = download(scalar), second = download(candidate);
          for (std::size_t i = 0; i < first.phases.size(); ++i)
            if (first.phases[i] != second.phases[i] ||
                !std::isfinite(second.jets[i]) || std::abs(first.jets[i] - second.jets[i]) > 2e-10)
              throw std::runtime_error("REAL ECP projection changed sign or amplitude");
          setenv(key, "0", 1);
        };
        compare(3, false); // Cold pointer preparation precedes warm-allocation proof.
        const auto before = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
        for (const std::size_t live : {1, 2, 3}) compare(live, periodic);
        const auto after = deviceMemoryMetrics(candidate.positions_.deviceOrdinal()).snapshot();
        if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls)
          throw std::runtime_error("Warm REAL ECP projection allocated scratch");
        auto changed = parameters.flat_values();
        for (const auto& tensor : plan.parameterTensors())
          if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
            changed[tensor.begin] += .0001;
        auto successor = image->makeUpdatedImage(changed, 1);
        scalar.rebindModel(successor);
        candidate.rebindModel(successor);
        if (grouped_reference) grouped_reference->rebindModel(successor);
        compare(2, false);
        const std::size_t refs[]{0}, moved[]{ne};
        setenv(key, "1", 1);
        candidate.beginSparseEnvelopeTile(nullptr, 1, 1, 0, refs, moved, 1);
        if (candidate.ecp_projected_backflow_active_)
          throw std::runtime_error("REAL ECP projection admitted absent request owner");
        candidate.endSparseEnvelopeTile(false, true);
        // Physical failures revoke the result even when projection was selected.
        candidate.beginSparseEnvelopeTile(&candidate, 2, 1, 0, refs, moved, 1);
        auto broken = std::vector<double>(positions.begin(), positions.begin() + 3 * ne);
        broken.back() = std::numeric_limits<double>::quiet_NaN();
        bool failed = false;
        try { candidate.evaluate(broken); }
        catch (const std::runtime_error&) { failed = true; }
        candidate.endSparseEnvelopeTile(false, true);
        if (!failed || candidate.result_valid_ || candidate.ecp_projected_backflow_active_)
          throw std::runtime_error("REAL ECP projection exposed failed physical outputs");
        setenv(key, "0", 1);
        compare(1, false); // Recovery must not inherit an invalid projection/request.
      }
    std::cout << "PASS real_ecp_projection N44 spin26_18 D16_D64 molecule_Gamma liveB3_1_2"
              << " pair_geometry=" << pair_geometry
              << " translated_images rebind request_lifetime physical_failure recovery warm0 comparisons=" << comparisons
              << " max_orbital_error=" << maximum << "\n";
  }

  /// Exercise the production selector/owners independently of the orbital component.
  static void runPairGeometry(compute::Queue<PlatformKind::CUDA>& queue, std::size_t electrons = 64)
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
    const ModelShape shape{electrons == 44 ? 26u : 32u, electrons == 44 ? 18u : 32u,
                           electrons, 1, 64, 4, 1};
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
    std::cout << "PASS real_periodic_pair_geometry Ne=" << ne << " exact_bits B4_B2_B1_B4 rebind warm0 retained_reject combined_B4_B16_minus1 cold_flag\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
