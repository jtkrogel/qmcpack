//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexWFImpl.h
 * @brief PRIVATE SINGLE-TU IMPLEMENTATION: include only inside PsiFormerWF.cpp.
 * PsiFormerNative.h defines non-inline symbols. This header is deliberately not a
 * second compiled native owner, nor a public numerical header. The sole consumer
 * supplies existing geometry/materialization/provenance helpers in qmcplusplus.
 */
#ifndef QMC_PSIFORMER_COMPLEX_WF_IMPL_OWNER_TU
#error "PsiFormerComplexWFImpl.h belongs only to the PsiFormerWF.cpp native owner"
#endif
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_IMPL_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_IMPL_H
#ifdef QMC_COMPLEX

/// Clone-local scalar preparation over one versioned, address-stable paired model.
struct PsiFormerComplexWF::Impl
{
  using C = psiformer::determinant::Complex;
  using Amplitude = psiformer::determinant::ComplexLogAmplitude;

#ifdef ENABLE_CUDA
  /** One exclusive numerical lane, owned by the model or an opt-in crowd resource.
   * Numeric owners share one immutable image. No borrowed GPU output escapes a call.
   */
  struct GpuLane
  {
    /// Freeze the opt-in numerical tile width once when creating a model family.
    static std::size_t configuredCapacity(const char* name = "PSIFORMER_GPU_COMPLEX_TILE_CAPACITY")
    {
      const char* value = std::getenv(name);
      if (!value)
        return 1;
      std::size_t result = 0;
      if (!*value)
        throw std::invalid_argument(std::string(name) + " must be a positive finite decimal integer");
      for (const char* digit = value; *digit; ++digit)
      {
        if (*digit < '0' || *digit > '9' ||
            result > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
          throw std::invalid_argument(std::string(name) + " must be a positive finite decimal integer");
        result = 10 * result + (*digit - '0');
      }
      if (!result || result == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(name) + " must be a positive finite decimal integer");
      return result;
    }

    /// Paired inference admits explicit forward VGL only; the legacy REAL selector is not applicable.
    static psiformer::device::TargetVglPreparation configuredPreparation()
    {
      const auto selection = psiformer::device::parseDeviceTargetSelection(
          std::getenv("PSIFORMER_GPU_TARGET_SELECTION"));
      if (selection.first || selection.second)
        throw std::invalid_argument("Complex GPU inference requires explicit VGL targets, not automatic selection");
      for (const char* name : {"PSIFORMER_GPU_VGL_PARALLEL_TRACE_BYTES", "PSIFORMER_GPU_KINETIC_TARGETS", "PSIFORMER_GPU_LIFTED_CHANNELS",
          "PSIFORMER_GPU_RETAINED_COORDINATE_BYTES", "PSIFORMER_GPU_METROPOLIS_BYTES",
          "PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES", "PSIFORMER_GPU_ADAM_BYTES",
          "PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES",
          "PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE"})
        if (std::getenv(name))
          throw std::invalid_argument(std::string("Complex GPU inference does not support ") + name);
      // Keep the existing decimal-positive convention, but reject SIZE_MAX as an unbounded paired cap.
      const auto positive = [](const char* name) -> std::size_t {
        const char* value = std::getenv(name);
        if (!value)
          return 0;
        std::size_t result = 0;
        if (!*value)
          throw std::invalid_argument(std::string(name) + " requires a positive finite decimal integer");
        for (const char* digit = value; *digit; ++digit)
        {
          if (*digit < '0' || *digit > '9' ||
              result > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
            throw std::invalid_argument(std::string(name) + " requires a positive finite decimal integer");
          result = result * 10 + (*digit - '0');
        }
        if (!result || result == std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(std::string(name) + " requires a positive finite decimal integer");
        return result;
      };
      const auto targets = positive("PSIFORMER_GPU_VGL_TARGETS");
      const auto bytes = positive("PSIFORMER_GPU_TARGET_WORKSPACE_BYTES");
      const auto work = positive("PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET");
      if (targets ? (!bytes || !work) : (bytes || work))
        throw std::invalid_argument("Complex GPU VGL targets require explicit K and both byte/work limits");
      return {targets, bytes, work};
    }

    /// Check the complete owner sum before allocating the image or either graph.
    static std::unique_ptr<GpuLane> prepare(const psiformer::PsiFormerExecutionPlan& plan,
        const std::vector<double>& parameters, const std::vector<double>& nuclei, std::size_t version,
        std::size_t limit, std::size_t capacity)
    {
      psiformer::device::reportCurrentDeviceMemoryAdmission(app_log());
      return prepareFrozen(plan, parameters, nuclei, version, limit, configuredPreparation(), capacity);
    }

    /// Successors preserve the admitted family profile without reading mutable process controls again.
    static std::unique_ptr<GpuLane> prepareFrozen(const psiformer::PsiFormerExecutionPlan& plan,
        const std::vector<double>& parameters, const std::vector<double>& nuclei, std::size_t version,
        std::size_t limit, psiformer::device::TargetVglPreparation preparation, std::size_t capacity = 1,
        std::shared_ptr<const psiformer::device::PreparedDeviceModel> shared_image = {})
    {
      using namespace psiformer::device;
      if (!limit || limit == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU inference requires a positive finite shared-lane byte cap");
      if (!capacity || capacity == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU tile capacity must be positive and finite");
      if (shared_image)
      {
        shared_image->validateCurrentDevice();
        if (shared_image->version() != version || shared_image->plan().parameterCount() != plan.parameterCount())
          throw std::invalid_argument("Complex crowd shared image/version differs from its admitted family");
      }
      if (capacity > 1 && preparation.max_targets)
        throw std::invalid_argument("Dense complex GPU tiles do not support the B1 target-K profile");
      if (capacity > 1 && !configuredDeviceMemoryAdmission().limit_bytes)
        throw std::invalid_argument("Dense complex GPU tiles require native aggregate device admission");
      const auto value = estimateComplexValueCost(plan, capacity);
      const auto spatial = preparation.max_targets
          ? estimateComplexSpatialTargetCost(plan, preparation.max_targets) : estimateComplexSpatialCost(plan, capacity);
      const auto work = preparation.max_targets
          ? estimateHeadFreeTargetPrefixCost(plan, preparation.max_targets).charged_scalar_work_units : 0;
      if (preparation.max_targets && (spatial.device_bytes > preparation.device_workspace_budget_bytes ||
                                     work > preparation.scalar_work_budget))
        throw std::length_error("Complex GPU target VGL exceeds its explicit owner byte/work limits");
      const auto bytes = psiformer::spatial_detail::checkedAdd(value.device_bytes, spatial.device_bytes,
                                                             "Complex GPU lane bytes overflow");
      if (bytes > limit)
        throw std::length_error("Complex GPU VALUE+spatial lane exceeds its explicit device byte cap");
      constexpr auto staging_per_electron = 3 * sizeof(double) + 5 * sizeof(cuDoubleComplex) + 4 * sizeof(C);
      const auto host = psiformer::spatial_detail::checkedAdd(
          psiformer::spatial_detail::checkedAdd(value.host_bytes, spatial.host_bytes, "Complex GPU host bytes overflow"),
          psiformer::spatial_detail::checkedAdd(
              psiformer::spatial_detail::checkedProduct(sizeof(ComplexValueResult), capacity, "Complex GPU amplitude staging overflow"),
              psiformer::spatial_detail::checkedProduct(staging_per_electron,
                  psiformer::spatial_detail::checkedProduct(capacity, plan.modelShape().electrons(), "Complex GPU staging overflow"),
                  "Complex GPU staging overflow"),
              "Complex GPU staging overflow"), "Complex GPU host bytes overflow");
      if (capacity > 1 || shared_image)
        app_log() << "  Complex PsiFormer dense GPU preflight: tile_capacity=" << capacity
                  << ", value_device_bytes=" << value.device_bytes << ", full_vgl_device_bytes=" << spatial.device_bytes
                  << ", host_numeric_bytes=" << host << "; before_allocation=yes; lane_serialized=yes"
                  << ", independent_resource_lane=" << bool(shared_image)
                  << "; image_and_vendor_bytes_excluded=yes" << std::endl;
      return std::unique_ptr<GpuLane>(new GpuLane(plan, parameters, nuclei, version, value, spatial,
                                                preparation, work, bytes, host, capacity, std::move(shared_image)));
    }

    /// Prepare fixed host staging and two qualified numeric owners, never a CPU tape.
    GpuLane(const psiformer::PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
        const std::vector<double>& nuclei, std::size_t version,
        psiformer::device::ComplexValueCost value_cost, psiformer::device::ComplexSpatialCost spatial_cost,
        psiformer::device::TargetVglPreparation preparation, std::size_t work, std::size_t bytes, std::size_t host,
        std::size_t capacity, std::shared_ptr<const psiformer::device::PreparedDeviceModel> shared_image)
        : capacity(capacity), electrons(plan.modelShape().electrons()), blas(queue),
          image(shared_image ? std::move(shared_image) :
              psiformer::device::PreparedDeviceModel::preparePairedComplexValue(plan, parameters, nuclei, version)),
          value(psiformer::device::PreparedDeviceComplexValue::prepare(image, capacity, queue, blas, value_cost.device_bytes)),
          full(preparation.max_targets
              ? psiformer::device::PreparedDeviceComplexSpatial::prepare(image, queue, blas, preparation)
              : psiformer::device::PreparedDeviceComplexSpatial::prepare(image, capacity, queue, blas, spatial_cost.device_bytes)),
          positions(3 * electrons * capacity), amplitude(capacity), gradient(positions.size()),
          laplacian(electrons * capacity), ratio(laplacian.size()), converted(4 * laplacian.size()),
          device_bytes(bytes), host_bytes(host), preparation(preparation)
    {
      const auto actual_host = value_cost.host_bytes + spatial_cost.host_bytes +
              sizeof(double) * positions.capacity() + sizeof(psiformer::device::ComplexValueResult) * amplitude.capacity() +
              sizeof(cuDoubleComplex) * (gradient.capacity() + laplacian.capacity() + ratio.capacity()) +
              sizeof(C) * converted.capacity();
      if (actual_host != host_bytes)
        throw std::logic_error("Complex GPU host capacities differ from checked preparation");
      if (value->cost().device_bytes + full->cost().device_bytes != device_bytes)
        throw std::logic_error("Complex GPU owner capacities differ from checked preparation");
      app_log() << "  Complex PsiFormer GPU inference prepared: mode="
                << (preparation.max_targets ? "EXPLICIT_TARGET" : "FULL") << ", K=" << preparation.max_targets
                << ", tile_capacity=" << capacity
                << ", spatial_owner_device_bytes=" << spatial_cost.device_bytes << ", shared_device_bytes=" << bytes
                << ", shared_host_numeric_bytes=" << host << ", request_scalar_work=" << work
                << "; shared_image_and_vendor_bytes_excluded=yes" << std::endl;
    }

    /// Catch an idle resource up under model read and its exclusive execution lock.
    void catchUp(const std::shared_ptr<const psiformer::device::PreparedDeviceModel>& current)
    {
      current->validateCurrentDevice();
      if (image == current)
        return;
      value->validateModelRebind(current);
      full->validateModelRebind(current);
      queue.sync();
      auto retired = image; // Prevent graph adoption from retiring the last old image.
      value->adoptPreparedModel(current);
      full->adoptPreparedModel(current);
      image = current;
    }

    /// Shrink only live prefixes; all capacities were admitted at the cold boundary.
    void resizeLive(std::size_t count)
    {
      if (!count || count > capacity)
        throw std::invalid_argument("Complex GPU live tile exceeds its frozen capacity");
      positions.resize(3 * electrons * count);
      amplitude.resize(count);
      gradient.resize(3 * electrons * count);
      laplacian.resize(electrons * count);
      ratio.resize(electrons * count);
      converted.resize(4 * electrons * count);
    }

    const std::size_t capacity, electrons;
    compute::Queue<PlatformKind::CUDA> queue;
    psiformer::device::AcceleratorBlasHandle blas;
    std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexValue> value;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexSpatial> full;
    std::vector<double> positions;
    std::vector<psiformer::device::ComplexValueResult> amplitude;
    std::vector<cuDoubleComplex> gradient, laplacian, ratio;
    std::vector<C> converted;
    const std::size_t device_bytes, host_bytes;
    const psiformer::device::TargetVglPreparation preparation;
    std::size_t dense_value_calls = 0, dense_vgl_calls = 0, dense_ratio_grad_calls = 0, dense_eval_grad_calls = 0,
        dense_collective_calls = 0;
    std::mutex mutex;
  };
#endif

  /// A weak family reference shares scratch without extending its lifetime or creating a cycle.
  struct DerivativeLane;
#ifdef ENABLE_CUDA
  struct GpuDerivativeLane;
#endif

  /// Stable shared ownership keeps metadata/executors fixed while canonical values mutate under mutex.
  struct Model
  {
    /// CPU construction never reads GPU controls; successors reuse this frozen width.
    static std::size_t requestedCapacity(std::size_t gpu_limit,
        const char* name = "PSIFORMER_GPU_COMPLEX_TILE_CAPACITY")
    {
#ifdef ENABLE_CUDA
      return gpu_limit ? GpuLane::configuredCapacity(name) : 1;
#else
      return 1;
#endif
    }

    /// Preserve paired metadata rather than recreating a default-real environment.
    static psiformer::ExecutionEnvironment environmentFor(const ParticleSet& electrons)
    {
      auto environment = makeExecutionEnvironment(electrons);
      if (environment.boundary != psiformer::BoundaryCondition::PERIODIC || electrons.isSpinor())
        throw std::invalid_argument("Complex PsiFormer inference requires nonspinor periodic bulk electrons");
      environment.head_representation = psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX;
      environment.amplitude_scalar_domain = psiformer::ScalarDomain::COMPLEX;
      return environment;
    }

    /// Retain initializer provenance and construct only qualified complex inference owners.
    Model(const std::string& component_name, InitializedNativeModel initialized,
          const ParticleSet& electrons, const ParticleSet& ions, std::array<double, 3> reduced, std::size_t gpu_limit
#ifdef ENABLE_CUDA
          , std::shared_ptr<GpuLane> prepared = {}
#endif
          )
        : model(std::move(initialized.model)), profile(std::move(initialized.profile)), seed(initialized.seed),
          environment(environmentFor(electrons)), reduced_twist(reduced),
          cartesian_twist(makeTwistPhase(electrons, reduced).cartesian),
          plan(psiformer::PsiFormerExecutionPlan::fromParameters(model.p,
              {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
               model.dim, model.heads, model.blocks}, environment)), value(model, plan), spatial(model, plan),
          identity(persistentModelIdentity(model, "internal", profile, seed, environment,
                                           reduced_twist, cartesian_twist)), source(&ions),
          schema(makeStructuredParameterSchema(component_name, model.p)), gpu_limit(gpu_limit),
          gpu_capacity(
#ifdef ENABLE_CUDA
              prepared ? prepared->capacity :
#endif
              requestedCapacity(gpu_limit)),
          gpu_crowds(
#ifdef ENABLE_CUDA
              prepared ? 1 :
#endif
              requestedCapacity(gpu_limit, "PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS"))
#ifdef ENABLE_CUDA
          , inference_concurrency(gpu_crowds)
#endif
    {
      if (gpu_limit)
      {
#ifdef ENABLE_CUDA
        if (gpu_crowds > 1 && !psiformer::device::configuredDeviceMemoryAdmission().limit_bytes)
          throw std::invalid_argument("Concurrent complex inference requires native aggregate device admission");
        gpu = prepared ? std::move(prepared) :
            GpuLane::prepare(plan, model.p.flat_values(), model.cfg.nuclei.x, model.p.version(), gpu_limit, gpu_capacity);
#else
        throw std::invalid_argument("Complex GPU inference is not compiled");
#endif
      }
    }

    pf::PsiFormer model;
    const std::string profile;
    const std::uint64_t seed;
    const psiformer::ExecutionEnvironment environment;
    const std::array<double, 3> reduced_twist, cartesian_twist;
    const psiformer::PsiFormerExecutionPlan plan;
    const pf::DirectComplexValueExecutor value;
    const pf::DirectComplexSpatialExecutor spatial;
    const std::uint64_t identity;
    const ParticleSet* const source;
    const std::shared_ptr<const wftrain::StructuredParameterSchema> schema;
    const std::size_t gpu_limit;
    const std::size_t gpu_capacity, gpu_crowds;
    std::atomic<std::size_t> acquired_teams{0};
#ifdef ENABLE_CUDA
    psiformer::PsiFormerInferenceConcurrency inference_concurrency;
    std::shared_ptr<GpuLane> gpu;
#endif
    mutable std::shared_mutex mutex;
    std::atomic<std::size_t> pending_proposals{0};
    std::mutex derivative_factory_mutex;
    std::weak_ptr<DerivativeLane> derivative_lane;
#ifdef ENABLE_CUDA
    std::weak_ptr<GpuDerivativeLane> gpu_derivative_lane;
#endif
  };

#ifdef ENABLE_CUDA
  /** Retain final model/lane authority and both images until commit/discard.
   * Declaration order releases both locks before candidate/retired resources.
   */
  struct PreparedGpuPublication final : PreparedGpuParameterPublication
  {
    /// Move fully resolved native leaves and final authority into one owning action.
    PreparedGpuPublication(std::shared_ptr<Model> model, std::shared_ptr<GpuLane> lane,
        std::shared_ptr<const psiformer::device::PreparedDeviceModel> next,
        pf::Parameters::PreparedFlatValues native, std::unique_lock<std::shared_mutex> authority,
        std::unique_lock<std::mutex> execution)
        : state(std::move(model)), generation(std::move(lane)), successor(std::move(next)),
          retired_image(generation->image), native_candidate(std::move(native)),
          write(std::move(authority)), gpu(std::move(execution)) {}

    /// Fixed native copies and pointer adoption only; retirement occurs after unlock.
    std::size_t commit() noexcept override
    {
      if (committed || !write.owns_lock() || !gpu.owns_lock())
        std::terminate();
      const auto version = native_candidate.commit();
      generation->value->adoptPreparedModel(successor);
      generation->full->adoptPreparedModel(successor);
      generation->image = successor;
      retired_derivative.swap(state->gpu_derivative_lane);
      committed = true;
      gpu.unlock();
      write.unlock();
      return version;
    }

    std::shared_ptr<Model> state;
    std::shared_ptr<GpuLane> generation;
    std::shared_ptr<const psiformer::device::PreparedDeviceModel> successor, retired_image;
    pf::Parameters::PreparedFlatValues native_candidate;
    std::weak_ptr<GpuDerivativeLane> retired_derivative;
    std::unique_lock<std::shared_mutex> write;
    std::unique_lock<std::mutex> gpu;
    bool committed = false;
  };
#endif

  /** Transaction staging for one acquired crowd; default mode borrows the model lane.
   * Concurrent resources own independent scratch but share the exact immutable image.
   */
  struct MultiWalkerResource final : Resource
  {
    /// Anchor an empty transaction resource to its immutable model family.
    explicit MultiWalkerResource(std::shared_ptr<Model> state)
        : Resource("PsiFormerComplexMultiWalkerResource"), state(std::move(state)) {}

    /// Clone only the model anchor; each acquired crowd prepares independent HOST staging.
    std::unique_ptr<Resource> makeClone() const override
    { return std::make_unique<MultiWalkerResource>(state); }

    /// Reject legacy real-adapter preparation metadata rather than silently ignoring it.
    void validateBatchResourcePreparation(const BatchResourcePreparationContext& context) const override
    {
      if (context.plan)
        throw std::invalid_argument("Dense complex GPU resources do not admit a legacy batch_memory plan");
    }

    /// Log attempted logical HOST ownership before any cold vector growth can fail.
    void prepareTeam(const RefVectorWithLeader<WaveFunctionComponent>& list)
    {
      const auto n = state->model.ne;
      const auto elements = psiformer::spatial_detail::checkedProduct(list.size(), n, "Complex crowd staging overflow");
      const bool growth = list.size() > components.capacity() || list.size() > amplitude.capacity() ||
          list.size() > logs.capacity() || list.size() > ratios.capacity() || list.size() > keys.capacity() ||
          elements > gradient.capacity() || elements > laplacian.capacity();
      if (growth)
      {
        constexpr auto per_walker = sizeof(PsiFormerComplexWF*) + sizeof(Amplitude) + sizeof(LogValue) +
            sizeof(PsiValue) + sizeof(std::uint64_t);
        auto requested = pf::checkedStorageProduct(list.size(), per_walker, "Complex crowd HOST request overflow");
        pf::addStorageBytes(requested, pf::checkedStorageProduct(elements, sizeof(GradType) + sizeof(ValueType),
            "Complex crowd HOST request overflow"), "Complex crowd HOST request overflow");
        app_log() << "  Complex PsiFormer crowd HOST preparation: walkers=" << list.size() << ", Ne=" << n
                  << ", attempted_logical_bytes=" << requested << ", retained_capacity_bytes=" << hostNumericBytes()
                  << "; before_allocation=yes; aggregate_HOST_bound=no; allocator_growth_and_metadata_excluded=yes" << std::endl;
      }
      components.resize(list.size());
      amplitude.resize(list.size());
      logs.resize(list.size());
      ratios.resize(list.size());
      keys.resize(list.size());
      gradient.resize(elements);
      laplacian.resize(elements);
      for (std::size_t i = 0; i < list.size(); ++i)
        components[i] = &static_cast<PsiFormerComplexWF&>(list[i]);
    }

    /// Count retained logical numeric HOST capacities without claiming a process-memory bound.
    std::size_t hostNumericBytes() const
    {
      std::size_t bytes = 0;
      const auto add = [&](std::size_t count, std::size_t width) {
        pf::addStorageBytes(bytes, pf::checkedStorageProduct(count, width, "Complex crowd HOST bytes overflow"),
                            "Complex crowd HOST bytes overflow");
      };
      add(components.capacity(), sizeof(PsiFormerComplexWF*));
      add(amplitude.capacity(), sizeof(Amplitude));
      add(logs.capacity(), sizeof(LogValue));
      add(ratios.capacity(), sizeof(PsiValue));
      add(keys.capacity(), sizeof(std::uint64_t));
      add(gradient.capacity(), sizeof(GradType));
      add(laplacian.capacity(), sizeof(ValueType));
      return bytes;
    }

    std::shared_ptr<Model> state;
#ifdef ENABLE_CUDA
    /// Independent opt-in scratch survives loan returns; all bytes remain globally charged.
    std::shared_ptr<GpuLane> gpu;
#endif
    std::vector<PsiFormerComplexWF*> components;
    std::vector<Amplitude> amplitude;
    std::vector<LogValue> logs;
    std::vector<PsiValue> ratios;
    std::vector<std::uint64_t> keys;
    std::vector<GradType> gradient;
    std::vector<ValueType> laplacian;
    std::size_t reported_host_bytes = 0;
    bool collective_pending = false;
    std::size_t collective_version = 0;
    std::uint64_t collective_descriptor = 0;
  };

  /// Fixed ions are part of the captured derivative problem, not an unchecked caller convention.
  static void validateDerivativeSource(const Model& s)
  {
    validateRuntimeLattice(*s.source, s.environment, "Complex PsiFormer derivative source");
    if (s.source->getTotalNum() != static_cast<int>(s.model.cfg.nuclei.shape[0]) ||
        s.source->R.size() != s.model.cfg.nuclei.shape[0] || s.source->GroupID.size() != s.source->R.size())
      throw std::invalid_argument("Complex PsiFormer derivative fixed-source extent mismatch");
    const auto& species = s.source->getSpeciesSet();
    const int charge = species.findAttribute("charge");
    if (charge < 0)
      throw std::invalid_argument("Complex PsiFormer derivative fixed-source charges are missing");
    for (int atom = 0; atom < s.source->getTotalNum(); ++atom)
    {
      if (s.source->GroupID[atom] < 0 || s.source->GroupID[atom] >= s.source->groups() ||
          !psiformer::determinant::isFiniteReal(species(charge, s.source->GroupID[atom])) ||
          species(charge, s.source->GroupID[atom]) != s.model.cfg.charges.x[atom])
        throw std::invalid_argument("Complex PsiFormer derivative fixed-source charge changed");
      for (int d = 0; d < 3; ++d)
        if (!psiformer::determinant::isFiniteReal(s.source->R[atom][d]) ||
            s.source->R[atom][d] != s.model.cfg.nuclei.x[3 * atom + d])
          throw std::invalid_argument("Complex PsiFormer derivative fixed-source configuration changed");
    }
  }

#ifdef ENABLE_CUDA
  /// One weakly shared immutable-image derivative lane; ownership keeps borrowed queue/BLAS alive.
  struct GpuDerivativeLane
  {
    using Products = psiformer::device::PreparedDeviceComplexProducts;

    /// Aggregate R(S) permits S>maxB; a single operator is separately restricted to B<=maxB.
    static PsiFormerComplexWF::GpuDerivativeRequirements required(const Model& state, std::size_t samples,
        std::size_t maximum_samples, std::size_t targets, std::size_t chunk)
    {
      const auto cost = psiformer::device::estimateComplexProductsCost(state.plan, targets, maximum_samples, 3, chunk,
          psiformer::device::ComplexProductsHostOutput::CHUNKED);
      const char* message = "Complex GPU derivative bytes overflow";
      std::size_t host = cost.host_bytes;
      pf::addStorageBytes(host, pf::checkedStorageBytes<C>(maximum_samples, message), message);
      pf::addStorageBytes(host, pf::checkedStorageBytes<double>(pf::checkedStorageProduct(4,
          pf::checkedStorageProduct(samples, state.model.ne, message), message), message), message);
      auto total = host;
      pf::addStorageBytes(total, cost.device_bytes, message); // Generic diagnostics sum both domains.
      return {cost.device_bytes, host, cost.matrix_work};
    }

    /// Allocate only after both domains and maximum request work have been admitted.
    GpuDerivativeLane(std::shared_ptr<Model> model, std::size_t maximum_samples, std::size_t targets,
        std::size_t chunk, const PsiFormerComplexWF::GpuDerivativeRequirements& budget)
        : state(std::move(model)), generation(state->gpu), maximum_samples(maximum_samples), targets(targets), chunk(chunk),
          products(Products::prepare(generation->image, targets, maximum_samples, 3, chunk,
              generation->queue, generation->blas, budget.device_bytes,
              budget.host_bytes - maximum_samples * sizeof(C), budget.matrix_work,
              psiformer::device::ComplexProductsHostOutput::CHUNKED)),
          zeros(maximum_samples), channels(3)
    {}

    const std::shared_ptr<Model> state;
    /// Declare before Products: borrowers die before their exact queue/BLAS/image generation.
    const std::shared_ptr<GpuLane> generation;
    const std::size_t maximum_samples, targets, chunk;
    const std::unique_ptr<Products> products;
    const std::vector<C> zeros;
    std::vector<Products::Channel> channels;
    std::mutex execution_mutex;
  };

  /// Immutable HOST samples with no CPU tapes; numerical output remains leased through sink callbacks.
  struct GpuDerivativeStream final : wftrain::StreamingDerivativeOperator
  {
    /// Preserve captured version/schema and bounded snapshot ownership.
    GpuDerivativeStream(std::shared_ptr<GpuDerivativeLane> scratch, std::size_t ordinal,
        std::size_t offset, std::size_t samples)
        : lane(std::move(scratch)), version(lane->state->model.p.version()), ordinal(ordinal), offset(offset),
          samples(samples), chunks(*lane->state->schema, version, lane->chunk),
          positions(3 * samples * lane->state->model.ne), inverse_masses(samples * lane->state->model.ne)
    {}

    /// Numerical full-score/device-dot JVP and both adjoints retain REAL tangents/HOST complex outputs.
    wftrain::StreamingDerivativeCapabilities capabilities() const noexcept override
    {
      using namespace wftrain;
      StreamingDerivativeCapabilities c;
      c.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
          derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) | derivativeProductBit(DerivativeProduct::SCORE_JVP);
      c.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) | derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
      c.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
      c.maximum_vjp_channels = 3;
      c.maximum_parameter_chunk_size = chunks.maximumChunkSize();
      c.maximum_sample_tile_size = 1;
      c.fixed_parameter_scratch_vectors = 10; // Three retained complex-P HOST outputs became bounded chunks.
      c.block_streaming = true;
      return c;
    }

    /// Schema storage is shared with the immutable image's model family.
    const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override { return *lane->state->schema; }

    /// Requests must use the captured runtime version, not a checkpoint's historical version.
    std::size_t parameterVersion() const noexcept override { return version; }

    /// Preserve the actual caller batch identity.
    std::size_t batchOrdinal() const noexcept override { return ordinal; }

    /// Preserve the immutable sample interval origin.
    std::size_t sampleOffset() const noexcept override { return offset; }

    /// Every prepared operator has a nonempty bounded sample snapshot.
    std::size_t sampleCount() const noexcept override { return samples; }

    /// Emit canonical block-respecting chunks, never a B by P matrix.
    const wftrain::ParameterChunkPlan& parameterChunkPlan() const noexcept override { return chunks; }

    /// Sum both domains; aggregate callers deduplicate the shared identity and charge snapshots separately.
    wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
    {
      wftrain::StreamingDerivativeStorageDiagnostics d;
      const auto& cost = lane->products->cost();
      d.parameter_count = cost.parameters;
      d.sample_count = samples;
      d.real_parameter_vectors = 4;
      d.complex_parameter_vectors = 5;
      d.parameter_scratch_bytes = 112 * cost.parameters + 32 * cost.chunk;
      d.shared_workspace_bytes = cost.device_bytes + cost.host_bytes + lane->zeros.capacity() * sizeof(C);
      d.evaluator_workspace_bytes = d.shared_workspace_bytes - d.parameter_scratch_bytes;
      d.sample_position_bytes = positions.capacity() * sizeof(double);
      d.sample_auxiliary_bytes = inverse_masses.capacity() * sizeof(double);
      d.retained_numeric_bytes = d.shared_workspace_bytes + d.sample_position_bytes + d.sample_auxiliary_bytes;
      d.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(lane.get());
      d.allocation_generation = 1;
      d.storage_fingerprint = lane->products->storageFingerprint();
      for (const auto* storage : {&positions, &inverse_masses})
        psiformer::determinant::detail::mixStorage(d.storage_fingerprint, *storage);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, lane->zeros);
      return d;
    }

    /// Contract the GPU canonical score with one REAL direction; only O(B) scalar products return.
    void evaluateScoreJVP(const wftrain::StructuredParameterVectorConstView& direction,
        wftrain::SampleProductSink& sink) const override
    {
      NativeTimerScope timer(phase_timers.score_jvp);
      std::unique_lock execution(lane->execution_mutex);
      const C* output;
      {
        std::shared_lock read(lane->state->mutex);
        if (version != lane->state->model.p.version() || lane->generation != lane->state->gpu ||
            lane->state->pending_proposals.load(std::memory_order_relaxed))
          throw std::logic_error("Complex GPU score JVP snapshot is stale or family has a pending move");
        std::unique_lock gpu(lane->generation->mutex);
        validateDerivativeSource(*lane->state);
        {
          NativeTimerScope host_timer(phase_timers.gpu_call);
          lane->products->evaluateScoreJVP(positions, direction.values().data(), direction.values().size());
        }
        const auto result = lane->products->scoreJvpView();
        lane->products->validate(result);
        output = result.data;
      }
      // Callback code never runs under model/GPU locks; the numerical lane remains leased.
      for (std::size_t sample = 0; sample < samples; ++sample)
      {
        const wftrain::SampleProductTileDescriptor tile{lane->state->schema->providerId(), lane->state->schema->fingerprint(),
            version, ordinal, offset + sample, 1, sample};
        sink.add({tile, {output + sample, 1}});
      }
    }

    /// Lease execution before model read; release GPU/model locks BEFORE any caller callback.
    void evaluateVJPs(wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> requested,
        wftrain::DerivativeAdjoint adjoint, wftrain::ParameterReductionSink& sink) const override
    {
      NativeTimerScope timer(phase_timers.parameter_vjp);
      using namespace wftrain;
      std::unique_lock execution(lane->execution_mutex);
      GpuDerivativeLane::Products::CompletedResult result;
      {
        std::shared_lock read(lane->state->mutex);
        if (version != lane->state->model.p.version() || lane->generation != lane->state->gpu ||
            lane->state->pending_proposals.load(std::memory_order_relaxed))
          throw std::logic_error("Complex GPU derivative snapshot is stale or family has a pending move");
        std::unique_lock gpu(lane->generation->mutex);
        validateDerivativeSource(*lane->state);
        for (auto& channel : lane->channels)
          channel = {GpuDerivativeLane::Products::Role::SCORE, lane->zeros.data(), samples};
        for (std::size_t c = 0; c < requested.size(); ++c)
          lane->channels[c] = {requested[c].product == DerivativeProduct::SCORE_VJP ?
              GpuDerivativeLane::Products::Role::SCORE : GpuDerivativeLane::Products::Role::KINETIC,
              requested[c].coefficients.values.data(), samples};
        {
          NativeTimerScope host_timer(phase_timers.gpu_call);
          lane->products->evaluate(positions, inverse_masses, lane->state->cartesian_twist, lane->channels,
              adjoint == DerivativeAdjoint::HERMITIAN);
        }
        result = lane->products->completedResult();
      }
      // All DEVICE channels were finite-admitted; only requested outputs cross D2H.
      for (const auto& chunk : chunks.chunks())
        for (std::size_t c = 0; c < requested.size(); ++c)
        {
          const C* output;
          {
            // The leased old owner/image remain valid even if a prior callback published.
            // Never acquire Model authority here, or hold this lane lock through a callback.
            std::unique_lock gpu(lane->generation->mutex);
            output = lane->products->downloadCompletedChunk(result, c, chunk.parameter_offset, chunk.count);
          }
          sink.add(c, {chunk, {output, chunk.count}});
        }
    }

    psiformer::PsiFormerTimers phase_timers;
    const std::shared_ptr<GpuDerivativeLane> lane;
    const std::size_t version, ordinal, offset, samples;
    const wftrain::ParameterChunkPlan chunks;
    std::vector<double> positions, inverse_masses;
  };
#endif

  /// One leased score tape, one complete kinetic tape, and three fixed complex contractions.
  struct DerivativeLane
  {
    /// Derive the same checked shape consumed by both qualified numerical owners.
    static pf::PsiFormerStorageShape shape(const Model& s)
    {
      return {s.model.ne, s.model.cfg.nuclei.shape[0], s.model.ndet, s.model.dim, s.model.heads,
          s.plan.parameter(psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0],
          s.model.blocks, s.model.p.size()};
    }

    /// Include both executor ion copies and all channel buffers before any numerical allocation.
    static std::size_t required(const Model& s, std::size_t samples)
    {
      const auto dimensions = shape(s);
      const char* message = "Complex PsiFormer derivative bytes overflow";
      std::size_t bytes = pf::DirectComplexScoreWorkspace::storageRequirement(dimensions);
      pf::addStorageBytes(bytes, pf::DirectComplexKineticWorkspace::storageRequirement(dimensions), message);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<C>(
          pf::checkedStorageProduct(3, dimensions.parameter_count, message), message), message);
      std::size_t doubles = pf::checkedStorageProduct(6, dimensions.nuclei, message);
      pf::addStorageBytes(doubles, pf::checkedStorageProduct(4,
          pf::checkedStorageProduct(samples, dimensions.electrons, message), message), message);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(doubles, message), message);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<C>(samples, message), message);
      return bytes;
    }

    /// Only the already-admitted factory may allocate the lane; there is no probe preparation.
    explicit DerivativeLane(const Model& s)
        : score(s.model, s.plan), kinetic(s.model, s.plan, s.cartesian_twist),
          score_work(score.makeWorkspace()), kinetic_work(kinetic.makeWorkspace(kinetic.requiredStorageBytes()))
    {
      for (auto& channel : channels)
        channel.resize(s.model.p.size());
    }

    /// Count capacities, including executor metadata that is outside the two workspaces.
    std::size_t bytes() const
    {
      std::size_t result = score_work->vectorStorageBytes();
      pf::addStorageBytes(result, kinetic_work->vectorStorageBytes(), "derivative lane bytes overflow");
      for (const auto bytes : {score.numericMetadataStorageBytes(), kinetic.numericMetadataStorageBytes()})
        pf::addStorageBytes(result, bytes, "derivative lane bytes overflow");
      for (const auto& channel : channels)
        pf::addStorageBytes(result, pf::checkedStorageBytes<C>(channel.capacity(), "channel bytes overflow"),
                            "derivative lane bytes overflow");
      return result;
    }

    /// Track all persistent numerical allocations, not evaluation generations.
    std::size_t fingerprint() const noexcept
    {
      std::size_t result = score_work->storageFingerprint() ^ kinetic_work->storageFingerprint();
      result ^= score.numericMetadataStorageFingerprint() ^ kinetic.numericMetadataStorageFingerprint();
      for (const auto& channel : channels)
        psiformer::determinant::detail::mixStorage(result, channel);
      return result;
    }

    const pf::DirectComplexScoreExecutor score;
    const pf::DirectComplexKineticExecutor kinetic;
    std::unique_ptr<pf::DirectComplexScoreWorkspace> score_work;
    std::unique_ptr<pf::DirectComplexKineticWorkspace> kinetic_work;
    std::array<std::vector<C>, 3> channels;
    std::mutex execution_mutex;
  };

  /// Orbital pretraining needs no kinetic workspace or three derivative-channel P buffers.
  struct OrbitalMSELane
  {
    static std::size_t required(const Model& s)
    {
      auto bytes = pf::DirectComplexScoreWorkspace::storageRequirement(DerivativeLane::shape(s));
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(pf::checkedStorageProduct(
          3, s.model.cfg.nuclei.shape[0], "orbital metadata overflow"), "orbital metadata overflow"),
          "orbital owner overflow");
      return bytes;
    }
    explicit OrbitalMSELane(const Model& s) : score(s.model, s.plan), work(score.makeWorkspace()) {}
    std::size_t bytes() const { return work->vectorStorageBytes() + score.numericMetadataStorageBytes(); }
    const pf::DirectComplexScoreExecutor score;
    std::unique_ptr<pf::DirectComplexScoreWorkspace> work;
    std::mutex execution_mutex;
  };

  /// Immutable sample snapshot; operators in one model family lease a single numerical lane.
  struct DerivativeStream final : wftrain::StreamingDerivativeOperator
  {
    /// Metadata and sample buffers are allocated only after the outer factory's exact cap check.
    DerivativeStream(std::shared_ptr<Model> model, std::shared_ptr<DerivativeLane> scratch,
        std::size_t ordinal, std::size_t offset, std::size_t samples, std::size_t chunk)
        : state(std::move(model)), lane(std::move(scratch)), version(state->model.p.version()),
          ordinal(ordinal), offset(offset), samples(samples), chunks(*state->schema, version, chunk),
          positions(3 * samples * state->model.ne), inverse_masses(samples * state->model.ne), sample_products(samples)
    {}

    /// Real tangents have complex score products; both adjoints cover score AND complete kinetic VJPs.
    wftrain::StreamingDerivativeCapabilities capabilities() const noexcept override
    {
      using namespace wftrain;
      StreamingDerivativeCapabilities c;
      c.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
          derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) | derivativeProductBit(DerivativeProduct::SCORE_JVP);
      c.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
          derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
      c.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
      c.maximum_vjp_channels = 3;
      c.maximum_parameter_chunk_size = chunks.maximumChunkSize();
      c.maximum_sample_tile_size = 1;
      c.fixed_parameter_scratch_vectors = 7;
      c.block_streaming = true;
      return c;
    }

    /// Keep schema identity valid for the complete operator lifetime.
    const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override { return *state->schema; }

    /// Bind requests to the snapshot version even after later model publication.
    std::size_t parameterVersion() const noexcept override { return version; }

    /// Preserve the caller's ordered batch identity.
    std::size_t batchOrdinal() const noexcept override { return ordinal; }

    /// Preserve the caller's sample interval origin.
    std::size_t sampleOffset() const noexcept override { return offset; }

    /// Empty batches retain the honestly charged lane but emit no parameter chunks.
    std::size_t sampleCount() const noexcept override { return samples; }

    /// Return the immutable canonical, block-respecting partition.
    const wftrain::ParameterChunkPlan& parameterChunkPlan() const noexcept override { return chunks; }

    /// Count each lane once in aggregate owners and report every per-operator numeric buffer.
    wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
    {
      wftrain::StreamingDerivativeStorageDiagnostics d;
      d.parameter_count = state->schema->parameterCount();
      d.sample_count = samples;
      d.real_parameter_vectors = 2;
      d.complex_parameter_vectors = 5;
      d.parameter_scratch_bytes = d.parameter_count * (2 * sizeof(double) + 5 * sizeof(C));
      d.shared_workspace_bytes = lane->bytes();
      d.evaluator_workspace_bytes = d.shared_workspace_bytes - d.parameter_scratch_bytes;
      d.sample_position_bytes = positions.capacity() * sizeof(double);
      d.sample_auxiliary_bytes = inverse_masses.capacity() * sizeof(double);
      d.sample_product_bytes = sample_products.capacity() * sizeof(C);
      d.retained_numeric_bytes = d.shared_workspace_bytes + d.sample_position_bytes + d.sample_auxiliary_bytes +
          d.sample_product_bytes;
      d.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(lane.get());
      d.allocation_generation = 1;
      d.storage_fingerprint = lane->fingerprint();
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, positions);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, inverse_masses);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, sample_products);
      return d;
    }

    /// Traverse each sample once per requested qualified owner, never once per parameter chunk.
    void evaluateVJPs(wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> requested,
        wftrain::DerivativeAdjoint adjoint, wftrain::ParameterReductionSink& sink) const override
    {
      NativeTimerScope timer(phase_timers.parameter_vjp);
      using namespace wftrain;
      std::unique_lock execution(lane->execution_mutex);
      {
        std::shared_lock read(state->mutex);
        if (version != state->model.p.version() || state->pending_proposals.load(std::memory_order_relaxed))
          throw std::logic_error("Complex PsiFormer derivative snapshot is stale or family has a pending move");
        bool need_score = false, need_kinetic = false;
        for (std::size_t channel = 0; channel < requested.size(); ++channel)
        {
          std::fill(lane->channels[channel].begin(), lane->channels[channel].end(), C{});
          need_score |= requested[channel].product == DerivativeProduct::SCORE_VJP;
          need_kinetic |= requested[channel].product == DerivativeProduct::LOCAL_ENERGY_VJP;
        }
        const auto n = state->model.ne, p = state->model.p.size();
        for (std::size_t sample = 0; sample < samples; ++sample)
        {
          const auto coordinates = pf::GeometryPositionView::interleaved(positions.data() + 3 * sample * n, n);
          const C* score = nullptr;
          const C* kinetic = nullptr;
          if (need_score)
          {
            NativeTimerScope numerical_timer(phase_timers.cpu_reverse);
            lane->score_work->setPositions(coordinates);
            score = lane->score.evaluate(*lane->score_work).parameter_score;
          }
          if (need_kinetic)
          {
            NativeTimerScope numerical_timer(phase_timers.cpu_kinetic);
            lane->kinetic_work->setPositions(coordinates);
            kinetic = lane->kinetic.evaluate(*lane->kinetic_work, inverse_masses.data() + sample * n, n).parameter_response;
          }
          for (std::size_t channel = 0; channel < requested.size(); ++channel)
          {
            const auto* response = requested[channel].product == DerivativeProduct::SCORE_VJP ? score : kinetic;
            const C coefficient = requested[channel].coefficients.values[sample];
            for (std::size_t parameter = 0; parameter < p; ++parameter)
              lane->channels[channel][parameter] += coefficient *
                  (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(response[parameter]) : response[parameter]);
          }
        }
        for (std::size_t channel = 0; channel < requested.size(); ++channel)
          for (const C value : lane->channels[channel])
            if (!psiformer::determinant::isFiniteComplex(value))
              throw std::runtime_error("Complex PsiFormer derivative contraction is nonfinite");
      }
      // Caller-controlled sink callbacks never run under the model lock; lane storage remains leased.
      if (samples)
        for (const auto& chunk : chunks.chunks())
          for (std::size_t channel = 0; channel < requested.size(); ++channel)
            sink.add(channel, {chunk, {lane->channels[channel].data() + chunk.parameter_offset, chunk.count}});
    }

    /// Contract qualified complex scores with a validated REAL tangent on one version-coherent transaction.
    void evaluateScoreJVP(const wftrain::StructuredParameterVectorConstView& direction,
        wftrain::SampleProductSink& sink) const override
    {
      NativeTimerScope timer(phase_timers.score_jvp);
      std::unique_lock execution(lane->execution_mutex);
      {
        std::shared_lock read(state->mutex);
        if (version != state->model.p.version() || state->pending_proposals.load(std::memory_order_relaxed))
          throw std::logic_error("Complex PsiFormer score JVP snapshot is stale or family has a pending move");
        const auto n = state->model.ne;
        for (std::size_t sample = 0; sample < samples; ++sample)
        {
          lane->score_work->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 3 * sample * n, n));
          const auto score = lane->score.evaluate(*lane->score_work);
          C product{};
          for (std::size_t parameter = 0; parameter < score.parameter_count; ++parameter)
            product += score.parameter_score[parameter] * direction.values()[parameter].real();
          if (!psiformer::determinant::isFiniteComplex(product))
            throw std::runtime_error("Complex PsiFormer score JVP is nonfinite");
          sample_products[sample] = product;
        }
      }
      // Complete all numerical work before callbacks; only the lane lease spans emission.
      for (std::size_t sample = 0; sample < samples; ++sample)
      {
        const wftrain::SampleProductTileDescriptor tile{state->schema->providerId(), state->schema->fingerprint(),
            version, ordinal, offset + sample, 1, sample};
        sink.add({tile, {sample_products.data() + sample, 1}});
      }
    }

    psiformer::PsiFormerTimers phase_timers;
    const std::shared_ptr<Model> state;
    const std::shared_ptr<DerivativeLane> lane;
    const std::size_t version, ordinal, offset, samples;
    const wftrain::ParameterChunkPlan chunks;
    std::vector<double> positions, inverse_masses;
    mutable std::vector<C> sample_products;
  };

  /// Override initializer values before parameter-dependent identity or device image construction.
  static InitializedNativeModel materialize(psiformer::InitializedPsiFormerParameters initialized,
      const ParticleSet& electrons, const ParticleSet& ions, const std::string& component_name,
      const std::string& parameter_checkpoint)
  {
    if (initialized.profile != psiformer::PSIFORMER_COMPLEX_PERIODIC_V1)
      throw std::invalid_argument("Complex PsiFormer inference requires explicit paired initialization");
    validateMatchingLattices(electrons, ions);
    auto native = makeInitializedNativeModel(std::move(initialized), electrons, ions);
    if (!parameter_checkpoint.empty())
    {
      const auto schema = makeStructuredParameterSchema(component_name, native.model.p);
      const auto saved = wftrain::TrainingCheckpoint::readModelParameters(parameter_checkpoint, *schema);
      native.model.p.set_flat_values(saved.values);
      app_log() << "Complex PsiFormer PARAMETER-ONLY WARMSTART: root=" << parameter_checkpoint
                << " provider=" << schema->providerId() << " schema=" << saved.schema_fingerprint
                << " saved_version=" << saved.version << " runtime_version=" << native.model.p.version()
                << "; CURRENT XML physics applies; optimizer/progress/walkers/RNG NOT restored; "
                << "temporary host parameter I/O excluded from numeric-owner caps" << std::endl;
    }
    return native;
  }

  /// Admit clone G/L before the shared GPU image can be constructed.
  static std::size_t admitGpuClone(const ParticleSet& electrons, std::size_t limit, std::size_t gpu_limit)
  {
    if (gpu_limit && (!limit || limit == std::numeric_limits<std::size_t>::max() ||
        gpu_limit == std::numeric_limits<std::size_t>::max()))
      throw std::invalid_argument("Complex GPU inference requires positive finite clone and shared-lane caps");
    if (gpu_limit && pf::checkedStorageSum(
        pf::checkedStorageBytes<GradType>(electrons.getTotalNum(), "Complex clone G overflow"),
        pf::checkedStorageBytes<ValueType>(electrons.getTotalNum(), "Complex clone L overflow"),
        "Complex clone G/L overflow") > limit)
      throw std::length_error("Complex GPU clone G/L exceeds its cap before device preparation");
    return gpu_limit;
  }

  /// Prepare only the selected CPU or GPU numerical profile.
  Impl(const std::string& component_name, psiformer::InitializedPsiFormerParameters initialized,
       const ParticleSet& electrons, const ParticleSet& ions, std::size_t limit, std::array<double, 3> reduced,
       std::size_t gpu_limit, const std::string& parameter_checkpoint)
      : Impl(std::make_shared<Model>(component_name,
                                   materialize(std::move(initialized), electrons, ions, component_name, parameter_checkpoint),
                                   electrons, ions, reduced, admitGpuClone(electrons, limit, gpu_limit)),
             electrons, limit)
  {}

  /// Check the full requirement before allocation; shared clones enter under their caller's read lease.
  Impl(std::shared_ptr<Model> shared, const ParticleSet& electrons, std::size_t limit)
      : state(std::move(shared)), particles(&electrons), numeric_limit(limit)
  {
    using namespace psiformer;
    using pf::addStorageBytes;
    using pf::checkedStorageBytes;
    validate(electrons);
    if (!limit || limit == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Complex PsiFormer requires a positive finite per-clone numeric byte cap");
    const auto& model = state->model;
    const pf::PsiFormerStorageShape shape{model.ne, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads,
        state->plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], model.blocks, model.p.size()};
    if (!state->gpu_limit)
    {
      numeric_required = pf::DirectComplexValueWorkspace::storageRequirement(shape);
      for (auto mode : {pf::DirectSpatialMode::FULL_VGL, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT})
        addStorageBytes(numeric_required, pf::DirectComplexSpatialWorkspace::storageRequirement(shape, mode),
                        "complex clone bytes overflow");
    }
    addStorageBytes(numeric_required, checkedStorageBytes<GradType>(model.ne, "complex own gradient bytes overflow"),
                    "complex clone bytes overflow");
    addStorageBytes(numeric_required, checkedStorageBytes<ValueType>(model.ne, "complex own Laplacian bytes overflow"),
                    "complex clone bytes overflow");
    if (numeric_required > numeric_limit)
      throw std::length_error("Complex PsiFormer full clone numeric requirement exceeds its per-clone byte cap");
    if (!state->gpu_limit)
    {
      value = state->value.makeWorkspace();
      full = state->spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
      active = state->spatial.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
    }
    own_gradient.resize(model.ne);
    own_laplacian.resize(model.ne);
    if (numericStorage().allocated_bytes != numeric_required)
      throw std::logic_error("Complex PsiFormer numeric capacity differs from its admitted exact requirement");
  }

  /// Sum live capacities and allocation identities, excluding immutable shared model storage.
  NumericStorage numericStorage() const
  {
    using namespace psiformer;
    using pf::addStorageBytes;
    using pf::checkedStorageBytes;
    std::size_t bytes = 0;
    if (value)
    {
      bytes = value->vectorStorageBytes();
      addStorageBytes(bytes, full->vectorStorageBytes(), "complex clone bytes overflow");
      addStorageBytes(bytes, active->vectorStorageBytes(), "complex clone bytes overflow");
    }
    addStorageBytes(bytes, checkedStorageBytes<GradType>(own_gradient.capacity(), "complex own gradient bytes overflow"),
                    "complex clone bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<ValueType>(own_laplacian.capacity(), "complex own Laplacian bytes overflow"),
                    "complex clone bytes overflow");
    std::size_t fingerprint = 0;
    if (value)
    {
      fingerprint = value->storageFingerprint();
      for (auto next : {full->storageFingerprint(), active->storageFingerprint()})
        fingerprint = (fingerprint ^ next) * 1099511628211ULL;
    }
    determinant::detail::mixStorage(fingerprint, own_gradient);
    determinant::detail::mixStorage(fingerprint, own_laplacian);
    NumericStorage result{numeric_required, bytes, numeric_limit, fingerprint};
#ifdef ENABLE_CUDA
    if (state->gpu)
    {
      result.shared_gpu_bytes = state->gpu->device_bytes;
      result.shared_gpu_host_bytes = state->gpu->host_bytes;
      result.shared_image_bytes = state->gpu->image->deviceStorageBytes();
      result.shared_image_identity = state->gpu->image.get();
    }
#endif
    return result;
  }

  /// Release exactly this clone's family proposal count under its caller's read transaction.
  void discardPending() noexcept
  {
    if (pending)
    {
      pending = false;
      state->pending_proposals.fetch_sub(1, std::memory_order_relaxed);
    }
    collective_pending = false;
  }

  /// Mixed entrypoints must not revoke a collectively retained FULL transaction.
  void requireNoCollective() const
  {
    if (collective_pending)
      throw std::logic_error("Complex PsiFormer requires collective resolution before mixed PBYP/evaluate/persistence calls");
  }

  /// Revoke all accepted/proposed publications without reallocating scratch.
  void invalidate() noexcept
  {
    discardPending();
    accepted_valid = own_full_valid = false;
  }

  /// Persistence and energy observation never consume an unresolved particle proposal.
  void requireIdle(const ParticleSet& p) const
  {
    validate(p);
    if (pending || p.getActivePtcl() >= 0)
      throw std::logic_error("Complex PsiFormer persistence/observation requires no pending move");
  }

  /// Own spatial products are usable only at the accepted configuration and parameter version.
  bool fullMatches(const ParticleSet& p) const
  {
    return accepted_valid && own_full_valid && accepted_key == configurationIdentity(p) &&
        accepted_version == state->model.p.version();
  }

  /// Scalar-only schema: eight limb-pair integers, eleven physical scalars, G/L and checksum limbs.
  std::size_t recordScalars() const
  {
    return pf::checkedStorageSum(29, pf::checkedStorageProduct(8, state->model.ne,
        "complex record extent overflow"), "complex record extent overflow");
  }

  /// Check PooledMemory's independently allocated scalar region before any get/put.
  void requireRecordExtent(WFBufferType& buffer) const
  {
    static_assert(sizeof(*buffer.Scalar_ptr) == sizeof(double));
    const auto base = reinterpret_cast<std::uintptr_t>(buffer.data());
    const auto scalar = reinterpret_cast<std::uintptr_t>(buffer.Scalar_ptr);
    if (!buffer.Scalar_ptr || scalar < base || scalar - base > buffer.size() ||
        buffer.current() > scalar - base)
      throw std::out_of_range("Complex PsiFormer walker scalar region is not allocated");
    const auto available = (buffer.size() - (scalar - base)) / sizeof(*buffer.Scalar_ptr);
    if (buffer.current_scalar() > available || recordScalars() > available - buffer.current_scalar())
      throw std::out_of_range("Complex PsiFormer walker record exceeds the allocated scalar region");
  }

  /// Identify exact stored binary64 contents, excluding the final checksum itself.
  std::uint64_t recordFingerprint(const WFBufferType& buffer, std::size_t start) const
  {
    std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
    for (std::size_t i = 0; i < recordScalars() - 2; ++i)
      mixPersistentDouble(hash, buffer.Scalar_ptr[start + i]);
    return hash;
  }

  /// Write only a coherent FULL record, never caller totals or partial accepted jets.
  void putRecord(WFBufferType& buffer, const LogValue& physical_log) const
  {
    requireRecordExtent(buffer);
    const auto start = buffer.current_scalar();
    for (auto integer : std::array<std::uint64_t, 8>{record_magic, 1, state->identity,
        accepted_version, accepted_key, state->model.ne, recordScalars(), 1})
      putPersistentInteger(buffer, integer);
    auto put = [&buffer](double scalar) { buffer.put(scalar); };
    for (double scalar : {accepted.log_abs, accepted.phase.real(), accepted.phase.imag(),
                          physical_log.real(), physical_log.imag()})
      put(scalar);
    for (const auto* twist : {&state->reduced_twist, &state->cartesian_twist})
      for (double scalar : *twist)
        put(scalar);
    for (std::size_t e = 0; e < state->model.ne; ++e)
    {
      for (int d = 0; d < 3; ++d)
        for (double scalar : {own_gradient[e][d].real(), own_gradient[e][d].imag()})
          put(scalar);
      for (double scalar : {own_laplacian[e].real(), own_laplacian[e].imag()})
        put(scalar);
    }
    putPersistentInteger(buffer, recordFingerprint(buffer, start));
  }

  /// Read only after checksum and provenance admission; publish no accepted state here.
  bool getRecord(const ParticleSet& p, WFBufferType& buffer, Amplitude& amplitude, LogValue& physical_log)
  {
    requireRecordExtent(buffer);
    const auto start = buffer.current_scalar();
    const auto end = start + recordScalars();
    bool zero = true;
    for (std::size_t i = start; i < end; ++i)
      zero = zero && psiformer::determinant::isFiniteReal(buffer.Scalar_ptr[i]) && buffer.Scalar_ptr[i] == 0;
    if (zero)
    {
      buffer.rewind(buffer.current(), end); // fresh driver sentinel is consumed, never accepted
      return false;
    }
    // Admit content integrity before decoding provenance or mutating any owned products.
    buffer.rewind(buffer.current(), end - 2);
    const auto checksum = getPersistentInteger(buffer, "complex checksum");
    if (checksum != recordFingerprint(buffer, start))
      throw std::runtime_error("Complex PsiFormer walker record checksum mismatch");
    buffer.rewind(buffer.current(), start);
    for (auto expected : std::array<std::uint64_t, 8>{record_magic, 1, state->identity,
        state->model.p.version(), configurationIdentity(p), state->model.ne, recordScalars(), 1})
      if (getPersistentInteger(buffer, "complex provenance") != expected)
        throw std::runtime_error("Complex PsiFormer walker provenance or FULL extent mismatch");
    auto get = [&buffer]() { double scalar; buffer.get(scalar); return scalar; };
    amplitude.log_abs = get();
    const double phase_real = get(), phase_imaginary = get();
    amplitude.phase = {phase_real, phase_imaginary};
    const double log_real = get(), log_imaginary = get();
    physical_log = {log_real, log_imaginary};
    for (const auto* twist : {&state->reduced_twist, &state->cartesian_twist})
      for (double expected : *twist)
      {
        const double stored = get();
        if (!psiformer::determinant::isFiniteReal(stored) || stored != expected)
          throw std::runtime_error("Complex PsiFormer walker twist mismatch");
      }
    if (!psiformer::determinant::isFiniteReal(amplitude.log_abs) ||
        !isFiniteWavefunctionValue(amplitude.phase) || std::abs(std::abs(amplitude.phase) - 1) > 1e-12 ||
        !isFiniteWavefunctionValue(physical_log) || physical_log.real() != amplitude.log_abs ||
        std::abs(std::exp(C{0, physical_log.imag() - physicalLog(amplitude, p).imag()}) - C{1}) > 1e-12)
      throw std::runtime_error("Complex PsiFormer walker amplitude is not finite/coherent");
    for (std::size_t e = 0; e < state->model.ne; ++e)
    {
      for (int d = 0; d < 3; ++d)
      {
        const double real = get(), imaginary = get();
        own_gradient[e][d] = makePsiFormerScalar<ValueType>(real, imaginary);
        if (!isFiniteWavefunctionValue(own_gradient[e][d]))
          throw std::runtime_error("Complex PsiFormer walker gradient is nonfinite");
      }
      const double real = get(), imaginary = get();
      own_laplacian[e] = makePsiFormerScalar<ValueType>(real, imaginary);
      if (!isFiniteWavefunctionValue(own_laplacian[e]))
        throw std::runtime_error("Complex PsiFormer walker Laplacian is nonfinite");
    }
    buffer.rewind(buffer.current(), end);
    return true;
  }

  /// Bind scalar requests to the target's identity, shape and immutable feature cell.
  void validate(const ParticleSet& p, int electron = -1, bool proposal = false) const
  {
    if (&p != particles || p.getTotalNum() != static_cast<int>(state->model.ne) ||
        p.groups() != 2 || p.groupsize(0) != static_cast<int>(state->model.cfg.nup) ||
        p.groupsize(1) != static_cast<int>(state->model.cfg.ndown))
      throw std::invalid_argument("Complex PsiFormer scalar target identity or shape mismatch");
    validateRuntimeLattice(p, state->environment, "Complex PsiFormer inference");
    if (electron < -1 || (electron >= 0 && static_cast<std::size_t>(electron) >= state->model.ne) ||
        (proposal && (electron < 0 || p.getActivePtcl() != electron)))
      throw std::invalid_argument("Complex PsiFormer scalar active particle mismatch");
  }

  /// Copy positions through checked setters, optionally replacing one row by an absolute virtual position.
  template<class Workspace>
  void positions(Workspace& workspace, const ParticleSet& p, int replacement = -1,
      const ParticleSet::PosType* absolute = nullptr) const
  {
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension,
            replacement == electron ? (absolute ? (*absolute)[dimension] : p.activeR(electron)[dimension])
                                    : p.R[electron][dimension]);
  }

  /// Capacity is immutable even when publication replaces the generation pointer.
  bool denseEnabled() const noexcept { return state->gpu_capacity > 1; }

  /// B1/target inference also needs native resources when independent crowds are enabled.
  bool crowdResourcesEnabled() const noexcept { return denseEnabled() || state->gpu_crowds > 1; }

#ifdef ENABLE_CUDA
  /// Resolve the exact native crowd's lane; scalar non-resource callers keep the original owner.
  std::shared_ptr<GpuLane> selectedGpuLane() const
  {
    if (acquired_leader)
    {
      const auto& leader = *acquired_leader->impl_;
      if (leader.multiwalker && leader.multiwalker.getResource().gpu)
        return leader.multiwalker.getResource().gpu;
    }
    return state->gpu;
  }

  /// The caller holds Model read and this lane; reject drift before invalidating any borrowed epoch.
  void bindGpuLane(GpuLane& lane) const
  {
    const auto& current = state->gpu->image;
    if (current->version() != state->model.p.version())
      throw std::logic_error("Complex crowd current image/native parameter version differs");
    lane.catchUp(current);
  }

  /// The caller holds Model read; charge dynamic numeric copies, not shared image or descriptor metadata.
  PsiFormerComplexWF::GpuParameterPublicationRequirements candidateIsolationRequirements() const
  {
    if (!state->gpu || state->gpu_crowds != 1)
      throw std::logic_error("Private complex GPU isolation requires a one-lane GPU family");
    requireNoCollective();
    requireIdle(*particles);
    if (acquired_leader || multiwalker || state->acquired_teams.load() || state->pending_proposals.load())
      throw std::logic_error("Private complex GPU isolation requires an idle unacquired family");
    validateDerivativeSource(*state);
    state->gpu->image->validateCurrentDevice();
    if (state->gpu->image->version() != state->model.p.version())
      throw std::logic_error("Private complex GPU isolation image/native version mismatch");
    const char* message = "Private complex GPU HOST ownership overflow";
    const auto& cfg = state->model.cfg;
    auto host = pf::checkedStorageBytes<double>(pf::checkedStorageProduct(2, state->model.p.size(), message), message);
    for (const auto* values : {&cfg.electrons.x, &cfg.nuclei.x, &cfg.charges.x})
      pf::addStorageBytes(host, pf::checkedStorageBytes<double>(values->size(), message), message);
    // VALUE trunk plus spatial metadata/trunk each own one immutable nucleus vector.
    pf::addStorageBytes(host, pf::checkedStorageBytes<double>(
        pf::checkedStorageProduct(3, cfg.nuclei.x.size(), message), message), message);
    pf::addStorageBytes(host, numeric_required, message); // GPU clone's G/L only, never CPU workspace.
    pf::addStorageBytes(host, state->gpu->host_bytes, message);
    return {state->gpu->device_bytes, host};
  }
#endif

  /// Match the exact native loan before reading or mutating transaction scratch.
  MultiWalkerResource& requireDenseTeam(const RefVectorWithLeader<WaveFunctionComponent>& components,
      const RefVectorWithLeader<ParticleSet>& targets, int electron = -1, bool collective_resolution = false)
  {
    const auto* leader = dynamic_cast<const PsiFormerComplexWF*>(&components.getLeader());
    if (!leader || leader->impl_.get() != this || !multiwalker || components.empty() ||
        components.size() != targets.size() || &components[0] != leader)
      throw std::logic_error("Dense complex GPU call requires its acquired native leader/team");
    auto& resource = multiwalker.getResource();
    if (resource.state != state || resource.components.size() != components.size())
      throw std::invalid_argument("Dense complex GPU crowd has foreign resource ownership or extent");
    if (resource.collective_pending && !collective_resolution)
      throw std::logic_error("Dense complex GPU staging is frozen until collective resolution");
    for (std::size_t i = 0; i < components.size(); ++i)
    {
      auto* component = dynamic_cast<PsiFormerComplexWF*>(&components[i]);
      if (!component || component != resource.components[i] || component->impl_->state != state ||
          component->impl_->acquired_leader != leader || component->impl_->acquired_lane != i)
        throw std::invalid_argument("Dense complex GPU call changed its acquired model or lane order");
      component->impl_->validate(targets[i], electron, electron >= 0);
    }
    return resource;
  }

  /// Run actual dense numerical prefixes; no caller/cache state changes here.
  void stageDense(MultiWalkerResource& resource, const RefVectorWithLeader<ParticleSet>& targets,
      bool spatial, int replacement = -1, bool accepted_gradient = false,
      const MCMultiParticleMoves<CoordsType::POS>* collective = nullptr)
  {
#ifdef ENABLE_CUDA
    auto permit = state->inference_concurrency.acquire();
    auto& lane = resource.gpu ? *resource.gpu : *state->gpu;
    std::unique_lock lease(lane.mutex);
    bindGpuLane(lane);
    if (lane.image->version() != state->model.p.version() || lane.capacity != state->gpu_capacity)
      throw std::logic_error("Dense complex GPU image version/capacity is stale");
    const auto n = state->model.ne;
    std::size_t dense_tiles = 0, tail = 0;
    for (std::size_t begin = 0; begin < targets.size();)
    {
      const auto count = std::min(lane.capacity, targets.size() - begin);
      lane.resizeLive(count);
      for (std::size_t local = 0; local < count; ++local)
      {
        std::optional<MCMultiParticleMoves<CoordsType::POS>::Slice> slice;
        if (collective)
          slice.emplace(collective->slice(begin + local));
        std::size_t selected = 0;
        for (std::size_t e = 0; e < n; ++e)
        {
          const bool replaced = slice && selected < slice->size() &&
              static_cast<std::size_t>(slice->particleIndex(selected)) == e;
          const auto& position = replaced ? slice->proposedPosition(selected++) :
              (replacement == static_cast<int>(e) ? targets[begin + local].activeR(e) : targets[begin + local].R[e]);
          for (int d = 0; d < 3; ++d)
            lane.positions[(local * n + e) * 3 + d] = position[d];
        }
      }
      {
        NativeTimerScope timer(phase_timers.gpu_evaluate);
        if (spatial)
          lane.full->evaluate(lane.positions);
        else
          lane.value->evaluate(lane.positions);
      }
      {
        NativeTimerScope timer(phase_timers.gpu_download);
        if (spatial)
          lane.full->download(lane.full->resultView(), lane.amplitude, lane.gradient, lane.laplacian, lane.ratio);
        else
          lane.value->download(lane.value->resultView(), lane.amplitude);
      }
      for (std::size_t local = 0; local < count; ++local)
      {
        const auto walker = begin + local;
        const auto& component = *resource.components[walker]->impl_;
        const auto& amplitude = lane.amplitude[local];
        resource.amplitude[walker] = {{amplitude.phase.x, amplitude.phase.y}, amplitude.log_abs};
        resource.logs[walker] = collective
            ? component.physicalLog(resource.amplitude[walker], targets[walker], collective->slice(walker))
            : component.physicalLog(resource.amplitude[walker], targets[walker], replacement);
        resource.keys[walker] = collective ? configurationIdentity(targets[walker], collective->slice(walker))
                                          : configurationIdentity(targets[walker], replacement);
        if (spatial)
          for (std::size_t e = 0; e < n; ++e)
          {
            for (int d = 0; d < 3; ++d)
            {
              const auto& gradient = lane.gradient[(local * n + e) * 3 + d];
              resource.gradient[walker * n + e][d] = component.physicalGradient({gradient.x, gradient.y}, d);
            }
            const auto& laplacian = lane.laplacian[local * n + e];
            const auto converted = makePsiFormerScalar<ValueType>(laplacian.x, laplacian.y);
            if (!isFiniteWavefunctionValue(converted))
              throw std::overflow_error("Dense complex GPU Laplacian conversion overflow");
            resource.laplacian[walker * n + e] = converted;
          }
      }
      dense_tiles += count > 1;
      tail = count;
      begin += count;
    }
    const bool proposed_gradient = spatial && replacement >= 0;
    auto& calls = collective ? lane.dense_collective_calls : (spatial ? (proposed_gradient ? lane.dense_ratio_grad_calls :
        (accepted_gradient ? lane.dense_eval_grad_calls : lane.dense_vgl_calls)) : lane.dense_value_calls);
    if (dense_tiles && calls++ == 0)
      app_log() << "  Complex PsiFormer numerical dense tile completed: operation="
                << (collective ? "COLLECTIVE_FULL" : (spatial ? (proposed_gradient ? "RATIO_GRAD_FULL" :
                    (accepted_gradient ? "ACCEPTED_GRAD_FULL" : "FULL_VGL")) : "VALUE"))
                << ", walkers=" << targets.size() << ", tile_capacity=" << lane.capacity
                << ", dense_tiles=" << dense_tiles << ", last_tile_count=" << tail
                << ", parameter_version=" << lane.image->version() << ", maximum_active_inference_crowds=" << state->gpu_crowds
                << "; per_lane_serialized=yes; CPU_fallback=no" << std::endl;
#else
    throw std::logic_error("Dense complex GPU inference is not compiled");
#endif
  }

  /// Common core view keeps a GPU operation locked until its scalar caller publishes copies.
  struct CoreResult : pf::DirectComplexSpatialResult
  {
#ifdef ENABLE_CUDA
    std::shared_ptr<GpuLane> owner;
    std::optional<psiformer::PsiFormerInferenceConcurrency::Lease> permit;
#endif
    std::unique_lock<std::mutex> lease;
  };

  /// Route VALUE/FULL/ACTIVE without duplicating physical twist, ratio or cache algebra.
  CoreResult core(const ParticleSet& p, bool spatial, int replacement = -1, int electron = -1,
      const ParticleSet::PosType* absolute = nullptr)
  {
    NativeTimerScope timer(spatial ? (electron < 0 ? phase_timers.vgl : phase_timers.gradient) : phase_timers.value);
    CoreResult result;
#ifdef ENABLE_CUDA
    if (state->gpu)
    {
      result.permit.emplace(state->inference_concurrency.acquire());
      result.owner = selectedGpuLane();
      auto& lane = *result.owner;
      result.lease = std::unique_lock(lane.mutex);
      bindGpuLane(lane);
      lane.resizeLive(1);
      for (int e = 0; e < p.getTotalNum(); ++e)
        for (int d = 0; d < 3; ++d)
          lane.positions[e * 3 + d] = replacement == e ? (absolute ? (*absolute)[d] : p.activeR(e)[d]) : p.R[e][d];
      if (spatial)
      {
        {
          NativeTimerScope host_timer(phase_timers.gpu_evaluate);
          lane.full->evaluate(lane.positions);
        }
        {
          NativeTimerScope host_timer(phase_timers.gpu_download);
          lane.full->download(lane.full->resultView(), lane.amplitude, lane.gradient, lane.laplacian, lane.ratio);
        }
        for (std::size_t i = 0; i < lane.gradient.size(); ++i)
          lane.converted[i] = {lane.gradient[i].x, lane.gradient[i].y};
        for (std::size_t i = 0; i < lane.laplacian.size(); ++i)
          lane.converted[lane.gradient.size() + i] = {lane.laplacian[i].x, lane.laplacian[i].y};
        result.gradient = lane.converted.data() + (electron < 0 ? 0 : 3 * electron);
        result.laplacian_log = lane.converted.data() + lane.gradient.size();
      }
      else
      {
        {
          NativeTimerScope host_timer(phase_timers.gpu_evaluate);
          lane.value->evaluate(lane.positions);
        }
        {
          NativeTimerScope host_timer(phase_timers.gpu_download);
          lane.value->download(lane.value->resultView(), lane.amplitude);
        }
      }
      result.amplitude = {{lane.amplitude[0].phase.x, lane.amplitude[0].phase.y}, lane.amplitude[0].log_abs};
      result.parameter_version = lane.image->version();
      return result;
    }
#endif
    if (spatial)
    {
      NativeTimerScope numerical_timer(phase_timers.cpu_spatial);
      auto& workspace = electron < 0 ? *full : *active;
      positions(workspace, p, replacement, absolute);
      static_cast<pf::DirectComplexSpatialResult&>(result) = electron < 0
          ? state->spatial.evaluateFull(workspace) : state->spatial.evaluateActive(workspace, electron);
    }
    else
    {
      positions(*value, p, replacement, absolute);
      const auto evaluated = state->value.evaluate(*value);
      result.amplitude = evaluated.amplitude;
      result.parameter_version = evaluated.parameter_version;
    }
    return result;
  }

  /// Apply the sole common outer phase at the adapter boundary, never inside an orbital head.
  double angle(const ParticleSet& p, int replacement = -1) const
  {
    double result = 0;
    if (state->reduced_twist != std::array<double, 3>{})
      for (int electron = 0; electron < p.getTotalNum(); ++electron)
        for (int dimension = 0; dimension < 3; ++dimension)
          result += state->cartesian_twist[dimension] *
              (replacement == electron ? p.activeR(electron)[dimension] : p.R[electron][dimension]);
    return result;
  }

  /// Canonical zeros have no phase; nonzero core phases remain fully complex.
  LogValue physicalLog(const Amplitude& amplitude, const ParticleSet& p, int replacement = -1) const
  {
    if (amplitude.isZero())
      return {amplitude.log_abs, 0};
    const LogValue result{amplitude.log_abs, std::arg(amplitude.phase) + angle(p, replacement)};
    if (!isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer physical logarithm overflow");
    return result;
  }

  /// Apply the moved-coordinate outer phase once for the complete selected configuration.
  LogValue physicalLog(const Amplitude& amplitude, const ParticleSet& p,
      const MCMultiParticleMoves<CoordsType::POS>::Slice& moves) const
  {
    double phase = angle(p);
    for (std::size_t selected = 0; selected < moves.size(); ++selected)
      for (int d = 0; d < 3; ++d)
        phase += state->cartesian_twist[d] *
            (moves.proposedPosition(selected)[d] - p.R[moves.particleIndex(selected)][d]);
    const LogValue result{amplitude.log_abs, std::arg(amplitude.phase) + phase};
    if (amplitude.isZero() || !isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer collective logarithm is undefined or nonfinite");
    return result;
  }

  /// Add i*k to logarithmic gradients; logarithmic Laplacians receive no outer-phase term.
  ValueType physicalGradient(C core, std::size_t dimension) const
  {
    const auto result = makePsiFormerScalar<ValueType>(core.real(), core.imag() + state->cartesian_twist[dimension]);
    if (!isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer physical gradient overflow");
    return result;
  }

  /// Validate the accepted reference before computing a complex ratio from unit core phases.
  PsiValue formRatio(const Amplitude& proposed, const ParticleSet& p, int electron,
      const ParticleSet::PosType* absolute = nullptr) const
  {
    if (!accepted_valid || accepted_key != configurationIdentity(p) ||
        accepted_version != state->model.p.version() || accepted.isZero())
      throw std::logic_error("Complex PsiFormer ratio requires a current accepted amplitude");
    C result{};
    if (!proposed.isZero())
    {
      double delta = 0;
      for (int dimension = 0; dimension < 3; ++dimension)
        delta += state->cartesian_twist[dimension] *
            ((absolute ? (*absolute)[dimension] : p.activeR(electron)[dimension]) - p.R[electron][dimension]);
      result = proposed.phase / accepted.phase * std::exp(proposed.log_abs - accepted.log_abs) *
          C{std::cos(delta), std::sin(delta)};
    }
    const PsiValue converted = makePsiFormerScalar<PsiValue>(result.real(), result.imag());
    if (!isFiniteWavefunctionValue(converted))
      throw std::overflow_error("Complex PsiFormer physical ratio overflow");
    return converted;
  }

  /// Publish a proposal only after all numerical and caller-output validation succeeds.
  void cache(const Amplitude& amplitude, const LogValue& log, const ParticleSet& p, int electron) noexcept
  {
    proposed = amplitude;
    proposed_log = log;
    proposed_particle = electron;
    proposed_key = configurationIdentity(p, electron);
    proposed_version = state->model.p.version();
    if (!pending)
      state->pending_proposals.fetch_add(1, std::memory_order_relaxed);
    pending = true;
  }

  psiformer::PsiFormerTimers phase_timers;
  std::shared_ptr<Model> state;
  std::shared_ptr<OrbitalMSELane> orbital_mse;
  const ParticleSet* particles;
  std::unique_ptr<pf::DirectComplexValueWorkspace> value;
  std::unique_ptr<pf::DirectComplexSpatialWorkspace> full, active;
  std::vector<GradType> own_gradient;
  std::vector<ValueType> own_laplacian;
  ResourceHandle<MultiWalkerResource> multiwalker;
  const PsiFormerComplexWF* acquired_leader = nullptr;
  std::size_t acquired_lane = 0, resource_cursor = 0;
  const ResourceCollection* resource_collection = nullptr;
  std::size_t numeric_required = 0, numeric_limit;
  Amplitude accepted, proposed;
  LogValue proposed_log{};
  std::uint64_t accepted_key = 0, proposed_key = 0;
  std::size_t accepted_version = 0, proposed_version = 0;
  int proposed_particle = -1;
  bool accepted_valid = false, pending = false, own_full_valid = false, collective_pending = false;
  static constexpr std::uint64_t record_magic = UINT64_C(0x5053435746303031); // PSCWF001
};

/// Construct the explicit paired CPU or strictly capped shared-GPU inference profile.
PsiFormerComplexWF::PsiFormerComplexWF(std::string name, psiformer::InitializedPsiFormerParameters initialized,
    const ParticleSet& electrons, const ParticleSet& ions, std::size_t limit, std::array<double, 3> reduced,
    std::size_t gpu_limit, const std::string& parameter_checkpoint)
    : WaveFunctionComponent(name),
      impl_(std::make_unique<Impl>(name, std::move(initialized), electrons, ions, limit, reduced,
                                 gpu_limit, parameter_checkpoint))
{
  if (!parameter_checkpoint.empty())
    app_log() << "Complex PsiFormer imported model identity=" << impl_->state->identity << std::endl;
}

/// A clone shares versioned model ownership and starts with invalid evaluation caches.
PsiFormerComplexWF::PsiFormerComplexWF(std::string name, std::unique_ptr<Impl> impl)
    : WaveFunctionComponent(name), impl_(std::move(impl))
{}

/// Expose honest per-clone numeric admission rather than the legacy batch-budget policy.
PsiFormerComplexWF::NumericStorage PsiFormerComplexWF::numericStorage() const
{
  std::shared_lock read(impl_->state->mutex);
  return impl_->numericStorage();
}

/// Share the immutable boundary convention without exposing model or numerical storage.
std::array<double, 3> PsiFormerComplexWF::reducedTwist() const { return impl_->state->reduced_twist; }

/// Channel metadata is immutable throughout the model family's lifetime.
std::size_t PsiFormerComplexWF::orbitalMSEChannelCount() const noexcept { return impl_->state->model.ndet; }

/// This explicit owner excludes target/caller P output and all ordinary inference owners.
PsiFormerComplexWF::OrbitalMSEStorage PsiFormerComplexWF::orbitalMSEStorage() const
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex orbital-MSE is CPU-only");
  std::lock_guard factory(state->derivative_factory_mutex);
  const auto& lane = impl_->orbital_mse;
  return {Impl::OrbitalMSELane::required(*state), lane ? lane->bytes() : 0,
      lane ? lane->work->storageFingerprint() ^ lane->score.numericMetadataStorageFingerprint() : 0};
}

/// Admit the exact score-only allocation before construction; repeat preparation is a no-op.
void PsiFormerComplexWF::prepareOrbitalMSE(std::size_t maximum_bytes)
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex orbital-MSE is CPU-only");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto required = Impl::OrbitalMSELane::required(*state);
  if (!maximum_bytes || maximum_bytes == std::numeric_limits<std::size_t>::max() || required > maximum_bytes)
    throw std::length_error("Complex orbital-MSE numeric cap exceeded before preparation");
  if (state->pending_proposals.load(std::memory_order_relaxed))
    throw std::logic_error("Complex orbital-MSE requires an idle model family");
  std::lock_guard factory(state->derivative_factory_mutex);
  if (!impl_->orbital_mse)
    impl_->orbital_mse = std::make_shared<Impl::OrbitalMSELane>(*state);
}

/// Read one stable model version and use only prepared scratch, never accepted caches.
double PsiFormerComplexWF::evaluateOrbitalMSE(const ParticleSet& particles,
    const std::vector<std::complex<double>>& target, std::vector<double>& gradient,
    std::size_t expected_version) const
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex orbital-MSE is CPU-only");
  if (state->model.p.version() != expected_version)
    throw std::runtime_error("Complex orbital-MSE parameter version is stale");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto n = state->model.ne, up = state->model.cfg.nup;
  if (state->pending_proposals.load(std::memory_order_relaxed) || particles.getActivePtcl() >= 0 ||
      particles.isSpinor() || particles.getTotalNum() != n || particles.groups() != 2 ||
      particles.groupsize(0) != up || particles.groupsize(1) != n - up || particles.R.size() != n || particles.GroupID.size() != n)
    throw std::invalid_argument("Complex orbital-MSE sample/proposal/spin mismatch");
  validateRuntimeLattice(particles, state->environment, "Complex orbital-MSE sample");
  for (std::size_t row = 0; row < n; ++row)
    for (int d = 0; d < 3; ++d)
      if (particles.GroupID[row] != (row < up ? 0 : 1) || !psiformer::determinant::isFiniteReal(particles.R[row][d]))
        throw std::invalid_argument("Complex orbital-MSE coordinates/spin order invalid");
  std::shared_ptr<Impl::OrbitalMSELane> lane;
  {
    std::lock_guard factory(state->derivative_factory_mutex);
    lane = impl_->orbital_mse;
  }
  if (!lane)
    throw std::logic_error("Complex orbital-MSE must be explicitly prepared");
  std::lock_guard execution(lane->execution_mutex);
  impl_->positions(*lane->work, particles);
  return lane->score.evaluateOrbitalMSE(*lane->work,
      {target.data(), target.size(), gradient.data(), gradient.size()}).loss;
}

/// Bind restart to fixed physical semantics, never mutable parameters or pointer identities.
std::string PsiFormerComplexWF::trainingConfigurationFingerprint() const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& s = *impl_->state;
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
#ifdef ENABLE_CUDA
  if (s.gpu)
  {
    mixPersistentString(hash, "gpu_complex_pathwise_coulomb_v1");
    mixPersistentInteger(hash, s.gpu->preparation.max_targets); // Frozen inference FULL(0)/target-K profile.
  }
  else
#endif
    mixPersistentString(hash, "cpu_complex_pathwise_coulomb_v1");
  mixPersistentString(hash, s.profile);
  mixPersistentInteger(hash, s.seed);
  mixPersistentString(hash, s.schema->fingerprint());
  mixPersistentInteger(hash, static_cast<std::uint64_t>(s.environment.geometry_feature_policy));
  for (bool periodic : s.environment.periodic_axes)
    mixPersistentInteger(hash, periodic);
  for (const auto& row : s.environment.lattice_vectors)
    for (double value : row)
      mixPersistentDouble(hash, value);
  mixPersistentInteger(hash, s.model.cfg.nup);
  mixPersistentInteger(hash, s.model.cfg.ndown);
  mixPersistentInteger(hash, s.model.ndet);
  mixPersistentInteger(hash, s.model.dim);
  mixPersistentInteger(hash, s.model.heads);
  mixPersistentInteger(hash, s.model.blocks);
  for (const auto* values : {&s.model.cfg.nuclei.x, &s.model.cfg.charges.x})
    for (double value : *values)
      mixPersistentDouble(hash, value);
  for (const auto& twist : {s.reduced_twist, s.cartesian_twist})
    for (double value : twist)
      mixPersistentDouble(hash, value);
  for (double mass : impl_->particles->get_mass_by_group())
    mixPersistentDouble(hash, mass);
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Preflight an independent derivative budget without constructing either retained tape.
std::size_t PsiFormerComplexWF::streamingDerivativeRequiredBytes(std::size_t samples) const
{
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex GPU inference does not expose CPU derivative storage");
  std::shared_lock read(impl_->state->mutex);
  return Impl::DerivativeLane::required(*impl_->state, samples);
}

/// Validate every concrete binding before allocation; unbudgeted discovery remains deliberately closed.
std::unique_ptr<wftrain::StreamingDerivativeOperator> PsiFormerComplexWF::makeBudgetedStreamingDerivativeOperator(
    const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& particles, std::size_t ordinal, std::size_t offset,
    std::size_t chunk, std::size_t maximum_bytes) const
{
  NativeTimerScope timer(impl_->phase_timers.preparation);
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex GPU inference does not prepare CPU derivative lanes");
  const auto state = impl_->state;
  std::shared_lock read(state->mutex);
  const auto samples = particles.size(), n = state->model.ne;
  const auto required = Impl::DerivativeLane::required(*state, samples);
  if (!maximum_bytes || maximum_bytes == std::numeric_limits<std::size_t>::max() || required > maximum_bytes)
    throw std::length_error("Complex PsiFormer derivative numeric budget exceeded before preparation");
  if (&components.getLeader() != this || &particles.getLeader() != impl_->particles ||
      components.size() != samples || !chunk || offset > std::numeric_limits<std::size_t>::max() - samples)
    throw std::invalid_argument("Complex PsiFormer derivative leader, extent or chunk mismatch");
  if (state->pending_proposals.load(std::memory_order_relaxed))
    throw std::logic_error("Complex PsiFormer derivative preparation requires an idle model family");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  // No clone, schema, physical mass or coordinate failure may follow numerical lane allocation.
  for (std::size_t sample = 0; sample < samples; ++sample)
  {
    const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[sample]);
    if (!component || component->impl_->state != state ||
        &component->parameterSchema() != state->schema.get())
      throw std::invalid_argument("Complex PsiFormer derivative clone/model/schema mismatch");
    component->impl_->requireIdle(particles[sample]);
    const auto& p = particles[sample];
    const auto& masses = p.get_mass_by_group();
    if (masses.size() != static_cast<std::size_t>(p.groups()))
      throw std::invalid_argument("Complex PsiFormer derivative target mass extent mismatch");
    for (int group = 0; group < p.groups(); ++group)
      if (!psiformer::determinant::isFiniteReal(masses[group]) || masses[group] <= 0 ||
          !psiformer::determinant::isFiniteReal(1.0 / masses[group]))
        throw std::invalid_argument("Complex PsiFormer derivative masses must have positive finite inverses");
    for (std::size_t electron = 0; electron < n; ++electron)
      for (int d = 0; d < 3; ++d)
        if (!psiformer::determinant::isFiniteReal(p.R[electron][d]))
          throw std::invalid_argument("Complex PsiFormer derivative sample coordinate is nonfinite");
  }
  // Factory serialization never waits on a lane execution lease while holding the model read lock.
  std::lock_guard preparation(state->derivative_factory_mutex);
  auto lane = state->derivative_lane.lock();
  if (!lane)
    lane = std::make_shared<Impl::DerivativeLane>(*state);
  auto result = std::make_unique<Impl::DerivativeStream>(state, lane, ordinal, offset, samples, chunk);
  for (std::size_t sample = 0; sample < samples; ++sample)
  {
    const auto& p = particles[sample];
    const auto& masses = p.get_mass_by_group();
    for (int group = 0; group < p.groups(); ++group)
      for (int electron = p.first(group); electron < p.last(group); ++electron)
        result->inverse_masses[sample * n + electron] = 1.0 / masses[group];
    for (std::size_t electron = 0; electron < n; ++electron)
      for (int d = 0; d < 3; ++d)
        result->positions[3 * (sample * n + electron) + d] = p.R[electron][d];
  }
  if (result->storageDiagnostics().retained_numeric_bytes != required)
    throw std::logic_error("Complex PsiFormer derivative numeric capacity differs from exact admission");
  state->derivative_lane = lane;
  return result;
}

/// Checked additional two-domain ownership; generic GPU discovery stays closed.
PsiFormerComplexWF::GpuDerivativeRequirements PsiFormerComplexWF::gpuDerivativeRequirements(
    std::size_t samples, std::size_t maximum_samples, std::size_t targets, std::size_t chunk) const
{
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    std::shared_lock read(impl_->state->mutex);
    // Dense inference owns separate graphs; reverse products remain B1 per sample.
    // Multiple inference crowds still cannot share the serialized training lane.
    if (impl_->state->gpu_crowds > 1)
      throw std::logic_error("Concurrent complex inference does not support GPU derivatives or training");
    return Impl::GpuDerivativeLane::required(*impl_->state, samples, maximum_samples, targets, chunk);
  }
#endif
  throw std::logic_error("Explicit GPU derivatives require an immutable GPU inference profile");
}

/// Admit the actual SOLE TrialWF batch before allocating any derivative numerical ownership.
std::unique_ptr<wftrain::StreamingDerivativeOperator> PsiFormerComplexWF::makeBudgetedGpuDerivativeOperator(
    const RefVectorWithLeader<TrialWaveFunction>& wavefunctions,
    const RefVectorWithLeader<ParticleSet>& particles, std::size_t ordinal, std::size_t offset,
    std::size_t maximum_samples, std::size_t targets, std::size_t chunk,
    std::size_t device_cap, std::size_t host_cap, std::size_t work_cap) const
{
  NativeTimerScope timer(impl_->phase_timers.preparation);
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    const auto state = impl_->state;
    {
      std::shared_lock read(state->mutex);
      // Product capacity is independent of the dense inference graph capacity.
      if (state->gpu_crowds > 1)
        throw std::logic_error("Concurrent complex inference does not support GPU derivatives or training");
    }
    const auto samples = particles.size(), n = state->model.ne;
    const auto required = Impl::GpuDerivativeLane::required(*state, samples, maximum_samples, targets, chunk);
    for (const auto cap : {device_cap, host_cap, work_cap})
      if (!cap || cap == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU derivatives require positive finite device/host/work caps");
    if (required.device_bytes > device_cap || required.host_bytes > host_cap || required.matrix_work > work_cap)
      throw std::length_error("Complex GPU derivatives exceed device/host/work admission before allocation");
    if (!samples || samples > maximum_samples || wavefunctions.size() != samples ||
        &particles.getLeader() != impl_->particles || offset > std::numeric_limits<std::size_t>::max() - samples ||
        wavefunctions.getLeader().getOrbitals().size() != 1 ||
        wavefunctions.getLeader().getOrbitals()[0].get() != this)
      throw std::invalid_argument("Complex GPU derivative sole TrialWF leader/extent mismatch");
    // Match inference: model read BEFORE its current generation GPU mutex, never wait for an execution lease here.
    std::shared_lock read(state->mutex);
    std::unique_lock gpu(state->gpu->mutex);
    if (state->pending_proposals.load(std::memory_order_relaxed))
      throw std::logic_error("Complex GPU derivative preparation requires an idle model family");
    Impl::validateDerivativeSource(*state);
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
      const auto& orbitals = wavefunctions[sample].getOrbitals();
      const auto* component = orbitals.size() == 1 ? dynamic_cast<const PsiFormerComplexWF*>(orbitals[0].get()) : nullptr;
      if (!component || component->impl_->state != state || &component->parameterSchema() != state->schema.get())
        throw std::invalid_argument("Complex GPU derivatives require actual sole same-family TrialWF clones");
      component->impl_->requireIdle(particles[sample]);
      const auto& p = particles[sample];
      if (p.isSpinor() || p.R.size() != n || p.spins.size() != n || p.GroupID.size() != n)
        throw std::invalid_argument("Complex GPU derivatives require nonspinor physical arrays of extent Ne");
      const auto& masses = p.get_mass_by_group();
      if (masses.size() != static_cast<std::size_t>(p.groups()))
        throw std::invalid_argument("Complex GPU derivative mass extent mismatch");
      for (const auto mass : masses)
        if (!psiformer::determinant::isFiniteReal(mass) || mass <= 0 ||
            !psiformer::determinant::isFiniteReal(1.0 / mass))
          throw std::invalid_argument("Complex GPU derivatives require positive finite physical masses");
      for (std::size_t electron = 0; electron < n; ++electron)
      {
        if (!psiformer::determinant::isFiniteReal(p.spins[electron]) ||
            p.GroupID[electron] != (electron < state->model.cfg.nup ? 0 : 1))
          throw std::invalid_argument("Complex GPU derivative physical spin/group data are invalid");
        for (int d = 0; d < 3; ++d)
          if (!psiformer::determinant::isFiniteReal(p.R[electron][d]))
            throw std::invalid_argument("Complex GPU derivative sample coordinate is nonfinite");
      }
    }
    std::lock_guard preparation(state->derivative_factory_mutex);
    auto lane = state->gpu_derivative_lane.lock();
    if (lane && lane->generation != state->gpu)
      lane.reset();
    if (lane && (lane->maximum_samples != maximum_samples || lane->targets != targets || lane->chunk != chunk))
      throw std::invalid_argument("Complex GPU derivative live lane profile mismatch");
    if (!lane)
      lane = std::make_shared<Impl::GpuDerivativeLane>(state, maximum_samples, targets, chunk,
          Impl::GpuDerivativeLane::required(*state, 0, maximum_samples, targets, chunk));
    auto result = std::make_unique<Impl::GpuDerivativeStream>(lane, ordinal, offset, samples);
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
      const auto& p = particles[sample];
      for (int group = 0; group < p.groups(); ++group)
        for (int electron = p.first(group); electron < p.last(group); ++electron)
          result->inverse_masses[sample * n + electron] = 1.0 / p.get_mass_by_group()[group];
      for (std::size_t electron = 0; electron < n; ++electron)
        for (int d = 0; d < 3; ++d)
          result->positions[3 * (sample * n + electron) + d] = p.R[electron][d];
    }
    if (result->storageDiagnostics().retained_numeric_bytes != required.device_bytes + required.host_bytes)
      throw std::logic_error("Complex GPU derivative numeric capacity differs from exact admission");
    state->gpu_derivative_lane = lane;
    return result;
  }
#endif
  throw std::logic_error("Explicit GPU derivatives require an immutable GPU inference profile");
}

/// Destroy native numerical state only in the sole owner TU.
PsiFormerComplexWF::~PsiFormerComplexWF()
{
  std::shared_lock read(impl_->state->mutex);
  impl_->discardPending();
}

/// Report the actual complete determinant expansion.
void PsiFormerComplexWF::exportModel(const std::string& path) const
{
  std::shared_lock lock(impl_->state->mutex);
  const auto& state = *impl_->state;
  writeStandaloneModel(path, state.model, state.environment, state.profile, state.seed,
                       "all_electron", state.reduced_twist);
}

bool PsiFormerComplexWF::isMultiDet() const
{
  std::shared_lock read(impl_->state->mutex);
  return impl_->state->model.ndet > 1;
}

/// The profile is immutable for the complete shared model lifetime.
bool PsiFormerComplexWF::gpuInferenceEnabled() const noexcept { return impl_->state->gpu_limit != 0; }

/// Immutable schema ownership is stable across every clone and publication; no read lock is needed.
const wftrain::StructuredParameterSchema& PsiFormerComplexWF::parameterSchema() const noexcept
{
  return *impl_->state->schema;
}

/// Immutable typed roles select canonical weights, never names masquerading as policy.
std::vector<std::string> PsiFormerComplexWF::neuralWeightBlocks() const
{
  using Role = psiformer::ParameterRole;
  std::vector<std::string> result;
  for (const auto& tensor : impl_->state->plan.parameterTensors())
    switch (tensor.role)
    {
    case Role::ELECTRON_EMBEDDING_WEIGHT:
    case Role::ATTENTION_QUERY_WEIGHT:
    case Role::ATTENTION_KEY_WEIGHT:
    case Role::ATTENTION_VALUE_WEIGHT:
    case Role::ATTENTION_OUTPUT_WEIGHT:
    case Role::UPDATE_HIDDEN_WEIGHT:
    case Role::UPDATE_OUTPUT_WEIGHT:
    case Role::BACKFLOW_UP_WEIGHT:
    case Role::BACKFLOW_DOWN_WEIGHT:
      result.push_back(tensor.module + "/" + tensor.name);
      break;
    default: break;
    }
  return result;
}

/// Copy canonical values and their exact version in one read transaction.
wftrain::StructuredParameterSnapshot PsiFormerComplexWF::snapshotParameters() const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& parameters = impl_->state->model.p;
  return {parameterSchema().fingerprint(), parameters.version(), parameters.flat_values()};
}

/// Prepare independent CPU ownership completely before replacing this clone's state.
void PsiFormerComplexWF::isolateParameterState()
{
  // A GPU clone requires a separately admitted private image/workspace contract.
  // Never hide a CPU fallback behind the original immutable GPU profile.
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex GPU candidate isolation is not qualified");
  const auto original = impl_->state;
  std::unique_ptr<Impl> independent;
  {
    std::shared_lock read(original->mutex);
    impl_->requireNoCollective();
    impl_->requireIdle(*impl_->particles);
    if (impl_->acquired_leader || impl_->multiwalker ||
        original->acquired_teams.load(std::memory_order_relaxed) != 0)
      throw std::logic_error("Complex candidate isolation requires no acquired resource team");
    if (original->pending_proposals.load(std::memory_order_relaxed) != 0)
      throw std::logic_error("Complex candidate isolation requires no family pending proposals");
    Impl::validateDerivativeSource(*original);

    // Copying Parameters itself would share graph NodePtrs with the live family.
    // Reconstruct leaves, preserving the exact current canonical version instead.
    const auto& model = original->model;
    pf::Parameters parameters(model.p.flat_values(), model.p.layouts);
    parameters.parameter_version = model.p.version();
    pf::PsiFormer detached(std::move(parameters), model.cfg, model.ndet,
                           model.dim, model.heads, model.blocks);
    auto state = std::make_shared<Impl::Model>(getName(),
        InitializedNativeModel{std::move(detached), original->profile, original->seed},
        *impl_->particles, *original->source, original->reduced_twist, 0);
    independent = std::make_unique<Impl>(std::move(state), *impl_->particles, impl_->numeric_limit);
  }
  // Fresh executors/workspaces bind only detached leaves; accepted state and weak
  // derivative lanes start empty. Retire this clone's old owners after unlocking.
  impl_.swap(independent);
}

/// Report the complete additional private numeric peak before any canonical/configuration copy.
PsiFormerComplexWF::GpuParameterPublicationRequirements
PsiFormerComplexWF::gpuCandidateIsolationRequirements() const
{
#ifdef ENABLE_CUDA
  std::shared_lock read(impl_->state->mutex);
  return impl_->candidateIsolationRequirements();
#else
  throw std::logic_error("Private complex GPU isolation is not compiled");
#endif
}

/// Prepare private GPU scratch outside source authority, then adopt only an exact still-current capture.
void PsiFormerComplexWF::isolateBudgetedGpuParameterState(std::size_t device_cap, std::size_t host_cap)
{
#ifdef ENABLE_CUDA
  for (auto cap : {device_cap, host_cap})
    if (!cap || cap == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Private complex GPU isolation requires positive finite additional caps");
  const auto original = impl_->state;
  std::shared_ptr<Impl::GpuLane> source_lane;
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
  std::vector<double> values;
  std::vector<pf::Layout> layouts;
  std::optional<pf::ConfigData> configuration;
  GpuParameterPublicationRequirements required{};
  std::size_t version;
  {
    std::shared_lock read(original->mutex);
    required = impl_->candidateIsolationRequirements();
    if (required.device_bytes > device_cap || required.host_bytes > host_cap)
      throw std::length_error("Private complex GPU isolation exceeds additional owner caps before allocation");
    app_log() << "  Complex private GPU isolation: attempted_device_bytes=" << required.device_bytes
              << ", attempted_host_numeric_bytes=" << required.host_bytes
              << "; before_allocation=yes; shared_image_counted_once=yes; excludes_metadata_vendor_caller=yes"
              << "; later_successor_publication_separately_admitted=yes" << std::endl;
    source_lane = original->gpu;
    image = source_lane->image;
    version = original->model.p.version();
    values = original->model.p.flat_values();
    layouts = original->model.p.layouts;
    configuration.emplace(original->model.cfg);
  }
  // Moves retain one canonical vector; fresh named leaves add exactly one P copy.
  pf::Parameters parameters(std::move(values), std::move(layouts));
  parameters.parameter_version = version;
  auto lane = std::shared_ptr<Impl::GpuLane>(Impl::GpuLane::prepareFrozen(original->plan,
      parameters.flat_values(), configuration->nuclei.x, version, original->gpu_limit,
      source_lane->preparation, source_lane->capacity, image));
  pf::PsiFormer model(std::move(parameters), std::move(*configuration), original->model.ndet,
                      original->model.dim, original->model.heads, original->model.blocks);
  auto state = std::make_shared<Impl::Model>(getName(),
      InitializedNativeModel{std::move(model), original->profile, original->seed},
      *impl_->particles, *original->source, original->reduced_twist, original->gpu_limit, std::move(lane));
  auto independent = std::make_unique<Impl>(std::move(state), *impl_->particles, impl_->numeric_limit);
  {
    std::shared_lock read(original->mutex);
    if (impl_->state != original || original->gpu != source_lane || source_lane->image != image ||
        original->model.p.version() != version)
      throw std::runtime_error("Private complex GPU isolation lost its exact source generation");
    impl_->candidateIsolationRequirements(); // Recheck current device, fixed ions and idle ownership.
    impl_.swap(independent); // No runtime, allocation, validation or retirement in adoption.
  }
  // Both failed preparation and retired clone owners are destroyed outside Model read.
#else
  throw std::logic_error("Private complex GPU isolation is not compiled");
#endif
}

/// Validate every candidate invariant before the same-size native copy synchronizes graph leaves.
std::size_t PsiFormerComplexWF::publishParameters(const wftrain::StructuredParameterSnapshot& candidate,
                                                std::size_t expected_version)
{
  NativeTimerScope timer(impl_->phase_timers.publication);
  if (gpuInferenceEnabled())
    throw std::logic_error("Complex GPU inference image is immutable; parameter publication is unqualified");
  if (candidate.schema_fingerprint != parameterSchema().fingerprint() ||
      candidate.values.size() != parameterSchema().parameterCount() || candidate.version != expected_version)
    throw std::invalid_argument("Complex PsiFormer publication schema, extent or source version mismatch");
  if (expected_version == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Complex PsiFormer parameter version exhausted");
  for (double value : candidate.values)
    if (!psiformer::determinant::isFiniteReal(value))
      throw std::invalid_argument("Complex PsiFormer publication values must be finite");

  std::unique_lock write(impl_->state->mutex);
  auto& parameters = impl_->state->model.p;
  if (parameters.version() != expected_version)
    throw std::runtime_error("Complex PsiFormer publication rejected a stale parameter version");
  if (impl_->state->pending_proposals.load(std::memory_order_relaxed) != 0)
    throw std::logic_error("Complex PsiFormer publication requires no family pending proposals");
  // Native preflight allocates/resolves leaves BEFORE any value mutates; the final copies cannot throw.
  parameters.set_flat_values(candidate.values);
  impl_->invalidate();
  // Other clones reject old accepted/buffer versions lazily; no clone pointer registry is retained.
  return parameters.version();
}

/// Admit only a new P image and native P candidate; reuse metadata/inference scratch.
PsiFormerComplexWF::GpuParameterPublicationRequirements
PsiFormerComplexWF::gpuParameterPublicationRequirements() const
{
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    std::shared_lock read(impl_->state->mutex);
    const auto native_bytes = psiformer::spatial_detail::checkedProduct(
        sizeof(double), parameterSchema().parameterCount(), "Complex GPU native candidate bytes overflow");
    return {native_bytes, native_bytes};
  }
#endif
  throw std::logic_error("Explicit GPU publication requires a GPU inference profile");
}

/// Preserve synchronous expert publication through the same prepared/no-fail boundary.
std::size_t PsiFormerComplexWF::publishBudgetedGpuParameters(
    const wftrain::StructuredParameterSnapshot& candidate, std::size_t expected_version,
    std::size_t additional_device_cap, std::size_t additional_host_cap)
{
  auto prepared = prepareBudgetedGpuParameterPublication(candidate, expected_version,
      additional_device_cap, additional_host_cap);
  return prepared->commit();
}

/// Upload outside write; quiesce reused scratch and resolve native leaves under final authority.
std::unique_ptr<PsiFormerComplexWF::PreparedGpuParameterPublication>
PsiFormerComplexWF::prepareBudgetedGpuParameterPublication(
    const wftrain::StructuredParameterSnapshot& candidate, std::size_t expected_version,
    std::size_t additional_device_cap, std::size_t additional_host_cap)
{
  NativeTimerScope timer(impl_->phase_timers.publication);
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    const auto state = impl_->state;
    if (candidate.schema_fingerprint != parameterSchema().fingerprint() ||
        candidate.values.size() != parameterSchema().parameterCount() || candidate.version != expected_version)
      throw std::invalid_argument("Complex GPU publication schema, extent or source version mismatch");
    if (expected_version == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex GPU publication version exhausted");
    for (const auto cap : {additional_device_cap, additional_host_cap})
      if (!cap || cap == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU publication requires positive finite new-generation caps");
    for (double value : candidate.values)
      if (!psiformer::determinant::isFiniteReal(value))
        throw std::invalid_argument("Complex GPU publication values must be finite");

    // Capture the exact immutable image under read; prepare its upload outside write.
    std::shared_ptr<Impl::GpuLane> captured;
    std::shared_ptr<const psiformer::device::PreparedDeviceModel> source_image;
    GpuParameterPublicationRequirements required{};
    const auto native_bytes = psiformer::spatial_detail::checkedProduct(
        sizeof(double), candidate.values.size(), "Complex GPU native candidate bytes overflow");
    {
      std::shared_lock read(state->mutex);
      if (state->model.p.version() != expected_version)
        throw std::runtime_error("Complex GPU publication rejected a stale parameter version");
      if (state->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex GPU publication requires no family pending proposals");
      Impl::validateDerivativeSource(*state);
      captured = state->gpu;
      source_image = captured->image;
      source_image->validateCurrentDevice();
      if (source_image->version() != expected_version)
        throw std::logic_error("Complex GPU publication current image/native version mismatch");
      required = {native_bytes, native_bytes};
      if (required.device_bytes > additional_device_cap || required.host_bytes > additional_host_cap)
        throw std::length_error("Complex GPU publication exceeds new-generation caps before allocation");
    }

    app_log() << "  Complex GPU prepared publication: attempted_new_device_bytes=" << required.device_bytes
              << ", attempted_new_host_numeric_bytes=" << required.host_bytes
              << ", owned_native_candidate_bytes=" << native_bytes
              << "; reuses immutable metadata and VALUE/FULL workspace; excludes caller/image HOST, "
                 "resolved-leaf/lock metadata and retained owners" << std::endl;
    // Independent canonical bytes share the immutable family, not an old P image.
    const auto next_version = expected_version + 1;
    auto successor = source_image->makeUpdatedImage(candidate.values, next_version);

    std::unique_lock write(state->mutex);
    if (state->model.p.version() != expected_version || state->gpu != captured || captured->image != source_image)
      throw std::runtime_error("Complex GPU publication lost its captured version/generation");
    if (state->pending_proposals.load(std::memory_order_relaxed))
      throw std::logic_error("Complex GPU publication requires no family pending proposals");
    Impl::validateDerivativeSource(*state);
    std::unique_lock gpu(captured->mutex);
    captured->image->validateCurrentDevice();
    captured->queue.sync();
    captured->value->validateModelRebind(successor);
    captured->full->validateModelRebind(successor);
    auto native = state->model.p.prepare_flat_values(candidate.values, expected_version);
    // All clones, including this publisher, reject old accepted caches by version.
    return std::make_unique<Impl::PreparedGpuPublication>(state, captured, std::move(successor),
        std::move(native), std::move(write), std::move(gpu));
  }
#endif
  throw std::logic_error("Explicit GPU publication requires a GPU inference profile");
}

/// Register native staging for dense tiles or opt-in independent inference crowds.
void PsiFormerComplexWF::createResource(ResourceCollection& collection) const
{
  if (impl_->crowdResourcesEnabled())
    collection.addResource(std::make_unique<Impl::MultiWalkerResource>(impl_->state));
}

/// Transactionally acquire one compatible native crowd and preserve its exact lane order.
void PsiFormerComplexWF::acquireResource(ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& components) const
{
  if (!impl_->crowdResourcesEnabled())
    return;
  NativeTimerScope timer(impl_->phase_timers.resource);
  std::shared_lock read(impl_->state->mutex);
  if (&components.getLeader() != this || components.empty() || &components[0] != this || impl_->multiwalker)
    throw std::logic_error("Dense complex GPU acquisition requires its idle native leader");
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[i]);
    if (!component || component->impl_->state != impl_->state)
      throw std::invalid_argument("Dense complex GPU acquisition requires same-model component clones");
    for (std::size_t prior = 0; prior < i; ++prior)
      if (&components[i] == &components[prior])
        throw std::invalid_argument("Dense complex GPU acquisition contains a duplicate clone");
  }
  if (collection.getBatchResourcePreparationProvenance().state != BatchResourcePreparationState::UNPREPARED)
    throw std::invalid_argument("Dense complex GPU acquisition does not support legacy batch_memory resources");
  auto count = impl_->state->acquired_teams.load();
  do
  {
    if (count >= impl_->state->gpu_crowds)
      throw std::logic_error("Complex GPU acquisition exceeds its frozen crowd limit");
  } while (!impl_->state->acquired_teams.compare_exchange_weak(count, count + 1));
  const auto cursor = collection.getCursor();
  ResourceHandle<Impl::MultiWalkerResource> candidate;
  try
  {
    for (std::size_t i = 0; i < components.size(); ++i)
      if (static_cast<PsiFormerComplexWF&>(components[i]).impl_->acquired_leader)
        throw std::logic_error("Dense complex GPU clone already belongs to an acquired crowd");
    candidate = collection.lendResource<Impl::MultiWalkerResource>();
    if (candidate.getResource().state != impl_->state)
      throw std::invalid_argument("Dense complex GPU resource belongs to another model family");
    candidate.getResource().prepareTeam(components);
    auto& resource = candidate.getResource();
#ifdef ENABLE_CUDA
    if (impl_->state->gpu_crowds > 1 && !resource.gpu)
    {
      const auto& primary = *impl_->state->gpu;
      resource.gpu = Impl::GpuLane::prepareFrozen(impl_->state->plan,
          impl_->state->model.p.flat_values(), impl_->state->model.cfg.nuclei.x,
          impl_->state->model.p.version(), impl_->state->gpu_limit,
          primary.preparation, primary.capacity, primary.image);
      app_log() << "  Complex GPU independent crowd owner: maximum_active=" << impl_->state->gpu_crowds
                << ", extra_device_bytes=" << resource.gpu->device_bytes
                << ", extra_host_numeric_bytes=" << resource.gpu->host_bytes
                << "; exact_shared_image=yes; original_scalar_lane_retained=yes; dormant_bytes_remain_charged=yes" << std::endl;
    }
#endif
    const auto host_bytes = resource.hostNumericBytes();
    if (resource.reported_host_bytes != host_bytes)
    {
      app_log() << "  Complex PsiFormer crowd transaction: walkers=" << components.size()
                << ", retained_host_numeric_bytes=" << host_bytes
                << "; excludes clones, image, vendor and allocator metadata; aggregate_HOST_bound=no" << std::endl;
      resource.reported_host_bytes = host_bytes;
    }
  }
  catch (...)
  {
    if (candidate)
    {
      collection.rewind(cursor);
      collection.takebackResource(candidate);
      collection.rewind(cursor);
    }
    impl_->state->acquired_teams.fetch_sub(1);
    throw;
  }
  impl_->multiwalker = std::move(candidate);
  impl_->resource_collection = &collection;
  impl_->resource_cursor = cursor;
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    auto& clone = *static_cast<PsiFormerComplexWF&>(components[i]).impl_;
    clone.acquired_leader = this;
    clone.acquired_lane = i;
  }
}

/// Return the original resource loan before revoking the crowd's acquisition markers.
void PsiFormerComplexWF::releaseResource(ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& components) const
{
  if (!impl_->crowdResourcesEnabled())
    return;
  std::shared_lock read(impl_->state->mutex);
  if (&components.getLeader() != this || !impl_->multiwalker || impl_->resource_collection != &collection ||
      collection.getCursor() != impl_->resource_cursor)
    throw std::logic_error("Dense complex GPU release requires its original resource loan and cursor");
  auto& resource = impl_->multiwalker.getResource();
  if (components.size() != resource.components.size())
    throw std::invalid_argument("Dense complex GPU release changed the acquired crowd extent");
  for (std::size_t i = 0; i < components.size(); ++i)
    if (&components[i] != resource.components[i] || resource.components[i]->impl_->acquired_leader != this ||
        resource.components[i]->impl_->acquired_lane != i)
      throw std::invalid_argument("Dense complex GPU release changed its acquired lane order");
  // Valid-loan RAII cleanup revokes only component proposals, never ParticleSet coordinates.
  if (resource.collective_pending)
  {
    for (auto* component : resource.components)
      component->impl_->discardPending();
    resource.collective_pending = false;
    resource.collective_descriptor = 0;
  }
  collection.takebackResource(impl_->multiwalker);
  for (auto* component : resource.components)
  {
    component->impl_->acquired_leader = nullptr;
    component->impl_->acquired_lane = 0;
  }
  resource.components.clear();
  impl_->resource_collection = nullptr;
  impl_->resource_cursor = 0;
  impl_->state->acquired_teams.fetch_sub(1);
}

/// Collective support is explicit dense GPU inference only; CPU/defaultB1 stay closed.
bool PsiFormerComplexWF::supportsMultiParticleMoves() const noexcept
{
  return impl_->denseEnabled();
}

/// Retain pure FULL products until a matching collective resolution, with all-output atomicity.
void PsiFormerComplexWF::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& components, const RefVectorWithLeader<ParticleSet>& targets,
    const MCMultiParticleMoves<CoordsType::POS>& moves, std::vector<LogValue>& log_ratios,
    const RefVector<ParticleSet::ParticleGradient>& gradients,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacians) const
{
  if (!supportsMultiParticleMoves())
    throw std::logic_error("Complex collective proposals require opt-in dense GPU inference");
  NativeTimerScope timer(impl_->phase_timers.batch_vgl);
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets);
  moves.validateFor(targets);
  const auto n = impl_->state->model.ne, walkers = components.size();
  if (log_ratios.size() != walkers || gradients.size() != walkers || laplacians.size() != walkers)
    throw std::invalid_argument("Complex collective proposal output extents differ");
  const auto log_range = checkedMemoryRange(log_ratios.data(), walkers, "Complex collective log output range overflow");
  for (std::size_t i = 0; i < walkers; ++i)
  {
    const auto& clone = *resource.components[i]->impl_;
    if (clone.pending || targets[i].getActivePtcl() >= 0 || !clone.accepted_valid || clone.accepted.isZero() ||
        clone.accepted_version != impl_->state->model.p.version() || clone.accepted_key != configurationIdentity(targets[i]))
      throw std::logic_error("Complex collective proposal requires current idle accepted amplitudes");
    if (gradients[i].get().size() != n || laplacians[i].get().size() != n)
      throw std::invalid_argument("Complex collective proposal spatial output extent differs");
    const auto g = checkedMemoryRange(gradients[i].get().data(), n, "Complex collective gradient range overflow");
    const auto l = checkedMemoryRange(laplacians[i].get().data(), n, "Complex collective Laplacian range overflow");
    if (memoryRangesOverlap(g, l) || memoryRangesOverlap(g, log_range) || memoryRangesOverlap(l, log_range))
      throw std::invalid_argument("Complex collective proposal outputs overlap");
    for (std::size_t target = 0; target < walkers; ++target)
    {
      const auto accepted_g = checkedMemoryRange(targets[target].G.data(), targets[target].G.size(),
          "Complex accepted gradient range overflow");
      const auto accepted_l = checkedMemoryRange(targets[target].L.data(), targets[target].L.size(),
          "Complex accepted Laplacian range overflow");
      if (memoryRangesOverlap(g, accepted_g) || memoryRangesOverlap(g, accepted_l) ||
          memoryRangesOverlap(l, accepted_g) || memoryRangesOverlap(l, accepted_l) ||
          memoryRangesOverlap(log_range, accepted_g) || memoryRangesOverlap(log_range, accepted_l))
        throw std::invalid_argument("Complex collective outputs alias accepted ParticleSet products");
    }
    for (std::size_t prior = 0; prior < i; ++prior)
    {
      const auto pg = checkedMemoryRange(gradients[prior].get().data(), n, "Complex collective gradient range overflow");
      const auto pl = checkedMemoryRange(laplacians[prior].get().data(), n, "Complex collective Laplacian range overflow");
      if (memoryRangesOverlap(g, pg) || memoryRangesOverlap(g, pl) || memoryRangesOverlap(l, pg) || memoryRangesOverlap(l, pl))
        throw std::invalid_argument("Complex collective proposal walker outputs overlap");
    }
  }
  impl_->stageDense(resource, targets, true, -1, false, &moves);
  for (std::size_t i = 0; i < walkers; ++i)
  {
    if (!isFiniteWavefunctionValue(resource.logs[i] - resource.components[i]->log_value_))
      throw std::overflow_error("Complex collective physical log ratio overflow");
    for (std::size_t e = 0; e < n; ++e)
    {
      for (int d = 0; d < 3; ++d)
        if (!isFiniteWavefunctionValue(gradients[i].get()[e][d] + resource.gradient[i * n + e][d]))
          throw std::overflow_error("Complex collective gradient scatter overflow");
      if (!isFiniteWavefunctionValue(laplacians[i].get()[e] + resource.laplacian[i * n + e]))
        throw std::overflow_error("Complex collective Laplacian scatter overflow");
    }
  }
  // Every failure point precedes publication; retain the unmodified proposed G/L buffer.
  for (std::size_t i = 0; i < walkers; ++i)
  {
    auto& component = *resource.components[i];
    auto& clone = *component.impl_;
    log_ratios[i] = resource.logs[i] - component.log_value_;
    for (std::size_t e = 0; e < n; ++e)
    {
      gradients[i].get()[e] += resource.gradient[i * n + e];
      laplacians[i].get()[e] += resource.laplacian[i * n + e];
    }
    clone.proposed = resource.amplitude[i];
    clone.proposed_log = resource.logs[i];
    clone.proposed_key = resource.keys[i];
    clone.proposed_particle = -1;
    clone.proposed_version = impl_->state->model.p.version();
    clone.pending = clone.collective_pending = true;
    impl_->state->pending_proposals.fetch_add(1, std::memory_order_relaxed);
  }
  resource.collective_version = impl_->state->model.p.version();
  resource.collective_descriptor = moves.fingerprint();
  resource.collective_pending = true;
}

/// Validate the whole mask and every receipt before committing accepted FULL or discarding proposals.
void PsiFormerComplexWF::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& components, const RefVectorWithLeader<ParticleSet>& targets,
    const MCMultiParticleMoves<CoordsType::POS>& moves, const std::vector<bool>& accepted) const
{
  if (!supportsMultiParticleMoves())
    throw std::logic_error("Complex collective resolution requires opt-in dense GPU inference");
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets, -1, /*collective_resolution=*/true);
  moves.validateFor(targets);
  if (!resource.collective_pending || accepted.size() != components.size() ||
      resource.collective_descriptor != moves.fingerprint() || resource.collective_version != impl_->state->model.p.version())
    throw std::logic_error("Complex collective resolution has a stale or mismatched descriptor/mask/version");
  const auto n = impl_->state->model.ne;
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    const auto& clone = *resource.components[i]->impl_;
    if (!clone.pending || !clone.collective_pending || clone.proposed_particle != -1 ||
        clone.proposed_version != resource.collective_version || clone.proposed_key != resource.keys[i] ||
        clone.own_gradient.size() != n || clone.own_laplacian.size() != n ||
        (accepted[i] && clone.proposed_key != configurationIdentity(targets[i])))
      throw std::logic_error("Complex collective resolution has incoherent products or accepted coordinates");
  }
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    auto& component = *resource.components[i];
    auto& clone = *component.impl_;
    if (accepted[i])
    {
      for (std::size_t e = 0; e < n; ++e)
      {
        clone.own_gradient[e] = resource.gradient[i * n + e];
        clone.own_laplacian[e] = resource.laplacian[i * n + e];
      }
      clone.accepted = clone.proposed;
      clone.accepted_key = clone.proposed_key;
      clone.accepted_version = clone.proposed_version;
      clone.accepted_valid = clone.own_full_valid = true;
      component.log_value_ = clone.proposed_log;
    }
    clone.discardPending();
  }
  resource.collective_pending = false;
  resource.collective_descriptor = 0;
}

/// Validate every output and tile before the nonthrowing whole-crowd scatter.
void PsiFormerComplexWF::mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& targets, const RefVector<ParticleSet::ParticleGradient>& gradients,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacians) const
{
  if (!impl_->denseEnabled())
  {
    WaveFunctionComponent::mw_evaluateLog(components, targets, gradients, laplacians);
    return;
  }
  NativeTimerScope timer(impl_->phase_timers.batch_vgl);
  if (components.empty())
    return;
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets);
  const auto n = impl_->state->model.ne;
  if (gradients.size() != components.size() || laplacians.size() != components.size())
    throw std::invalid_argument("Dense complex GPU full scatter list sizes differ");
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    if (gradients[i].get().size() != n || laplacians[i].get().size() != n)
      throw std::invalid_argument("Dense complex GPU full scatter has a wrong electron extent");
    const auto g = checkedMemoryRange(gradients[i].get().data(), n, "Complex gradient output range overflow");
    const auto l = checkedMemoryRange(laplacians[i].get().data(), n, "Complex Laplacian output range overflow");
    if (memoryRangesOverlap(g, l))
      throw std::invalid_argument("Dense complex GPU gradient and Laplacian outputs overlap");
    for (std::size_t prior = 0; prior < i; ++prior)
    {
      const auto pg = checkedMemoryRange(gradients[prior].get().data(), n, "Complex gradient output range overflow");
      const auto pl = checkedMemoryRange(laplacians[prior].get().data(), n, "Complex Laplacian output range overflow");
      if (memoryRangesOverlap(g, pg) || memoryRangesOverlap(g, pl) || memoryRangesOverlap(l, pg) || memoryRangesOverlap(l, pl))
        throw std::invalid_argument("Dense complex GPU walker outputs overlap");
    }
  }
  impl_->stageDense(resource, targets, true);
  for (std::size_t i = 0; i < components.size(); ++i)
    for (std::size_t e = 0; e < n; ++e)
    {
      for (int d = 0; d < 3; ++d)
        if (!isFiniteWavefunctionValue(gradients[i].get()[e][d] + resource.gradient[i * n + e][d]))
          throw std::overflow_error("Dense complex GPU gradient scatter overflow");
      if (!isFiniteWavefunctionValue(laplacians[i].get()[e] + resource.laplacian[i * n + e]))
        throw std::overflow_error("Dense complex GPU Laplacian scatter overflow");
    }
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    auto& component = *resource.components[i];
    auto& clone = *component.impl_;
    clone.invalidate();
    for (std::size_t e = 0; e < n; ++e)
    {
      clone.own_gradient[e] = resource.gradient[i * n + e];
      clone.own_laplacian[e] = resource.laplacian[i * n + e];
      gradients[i].get()[e] += clone.own_gradient[e];
      laplacians[i].get()[e] += clone.own_laplacian[e];
    }
    clone.accepted = resource.amplitude[i];
    clone.accepted_key = resource.keys[i];
    clone.accepted_version = impl_->state->model.p.version();
    clone.accepted_valid = clone.own_full_valid = true;
    component.log_value_ = resource.logs[i];
  }
}

void PsiFormerComplexWF::mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& targets, const RefVector<ParticleSet::ParticleGradient>& gradients,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacians, bool fromscratch) const
{
  if (!impl_->denseEnabled())
    WaveFunctionComponent::mw_evaluateGL(components, targets, gradients, laplacians, fromscratch);
  else
    mw_evaluateLog(components, targets, gradients, laplacians);
}

/// Dense VALUE consumes active replacements; proposal caches commit only after all tails validate.
void PsiFormerComplexWF::mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& targets, int electron, std::vector<PsiValue>& ratios) const
{
  if (!impl_->denseEnabled())
  {
    WaveFunctionComponent::mw_calcRatio(components, targets, electron, ratios);
    return;
  }
  NativeTimerScope timer(impl_->phase_timers.batch_value);
  if (components.empty())
    return;
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets, electron);
  if (electron < 0 || ratios.size() != components.size())
    throw std::invalid_argument("Dense complex GPU ratio output/electron extent mismatch");
  impl_->stageDense(resource, targets, false, electron);
  for (std::size_t i = 0; i < components.size(); ++i)
    resource.ratios[i] = resource.components[i]->impl_->formRatio(resource.amplitude[i], targets[i], electron);
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    auto& clone = *resource.components[i]->impl_;
    clone.discardPending();
    clone.cache(resource.amplitude[i], resource.logs[i], targets[i], electron);
    ratios[i] = resource.ratios[i];
  }
}

/// Accepted gradients WRITE caller outputs; never publish FULL caches or overwrite ParticleSet G/L.
void PsiFormerComplexWF::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& targets, int electron, std::vector<GradType>& gradients) const
{
  if (!impl_->denseEnabled())
  {
    WaveFunctionComponent::mw_evalGrad(components, targets, electron, gradients);
    return;
  }
  NativeTimerScope timer(impl_->phase_timers.batch_vgl);
  if (components.empty())
    return;
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets);
  const auto n = impl_->state->model.ne;
  if (electron < 0 || static_cast<std::size_t>(electron) >= n || gradients.size() != components.size())
    throw std::invalid_argument("Dense complex GPU accepted gradient output/electron extent mismatch");
  impl_->stageDense(resource, targets, true, -1, /*accepted_gradient=*/true);
  // Every physical row and every tail has validated; only nonthrowing writes follow.
  for (std::size_t i = 0; i < components.size(); ++i)
    gradients[i] = resource.gradient[i * n + electron];
}

/// Dense proposed FULL products preserve every output and pending cache until all tails validate.
void PsiFormerComplexWF::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& targets, int electron, std::vector<PsiValue>& ratios,
    std::vector<GradType>& gradients) const
{
  if (!impl_->denseEnabled())
  {
    WaveFunctionComponent::mw_ratioGrad(components, targets, electron, ratios, gradients);
    return;
  }
  NativeTimerScope timer(impl_->phase_timers.batch_vgl);
  if (components.empty())
    return;
  std::shared_lock read(impl_->state->mutex);
  auto& resource = impl_->requireDenseTeam(components, targets, electron);
  if (electron < 0 || ratios.size() != components.size() || gradients.size() != components.size())
    throw std::invalid_argument("Dense complex GPU ratio/gradient output/electron extent mismatch");
  const auto r = checkedMemoryRange(ratios.data(), ratios.size(), "Complex ratio output range overflow");
  const auto g = checkedMemoryRange(gradients.data(), gradients.size(), "Complex proposed gradient output range overflow");
  if (memoryRangesOverlap(r, g))
    throw std::invalid_argument("Dense complex GPU ratio and proposed gradient outputs overlap");
  impl_->stageDense(resource, targets, true, electron);
  const auto n = impl_->state->model.ne;
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    resource.ratios[i] = resource.components[i]->impl_->formRatio(resource.amplitude[i], targets[i], electron);
    // Reuse the moved row of the FULL transaction buffer for additive output staging.
    auto& staged = resource.gradient[i * n + electron];
    for (int d = 0; d < 3; ++d)
    {
      staged[d] += gradients[i][d];
      if (!isFiniteWavefunctionValue(staged[d]))
        throw std::overflow_error("Dense complex GPU proposed gradient scatter overflow");
    }
  }
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    auto& clone = *resource.components[i]->impl_;
    clone.discardPending();
    clone.cache(resource.amplitude[i], resource.logs[i], targets[i], electron);
    ratios[i] = resource.ratios[i];
    gradients[i] = resource.gradient[i * n + electron];
  }
}

/// Validate complete scatter before publishing any accepted amplitude or caller G/L.
PsiFormerComplexWF::LogValue PsiFormerComplexWF::evaluateLog(const ParticleSet& p,
    ParticleSet::ParticleGradient& g, ParticleSet::ParticleLaplacian& l)
{
  std::shared_lock read(impl_->state->mutex);
  return evaluateLogLocked(p, g, l);
}

/// Perform FULL scatter only under the public caller's existing read transaction.
PsiFormerComplexWF::LogValue PsiFormerComplexWF::evaluateLogLocked(const ParticleSet& p,
    ParticleSet::ParticleGradient& g, ParticleSet::ParticleLaplacian& l)
{
  impl_->requireNoCollective();
  impl_->invalidate();
  impl_->validate(p);
  if (g.size() != static_cast<std::size_t>(p.getTotalNum()) || l.size() != g.size())
    throw std::invalid_argument("Complex PsiFormer full scatter shape mismatch");
  const auto result = impl_->core(p, true);
  const LogValue physical = impl_->physicalLog(result.amplitude, p);
  // Stage this component's own products separately from any caller accumulation.
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      impl_->own_gradient[electron][dimension] =
          impl_->physicalGradient(result.gradient[3 * electron + dimension], dimension);
    impl_->own_laplacian[electron] = makePsiFormerScalar<ValueType>(result.laplacian_log[electron].real(),
                                                                 result.laplacian_log[electron].imag());
  }
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      if (!isFiniteWavefunctionValue(g[electron][dimension] +
          impl_->own_gradient[electron][dimension]))
        throw std::overflow_error("Complex PsiFormer gradient scatter overflow");
    if (!isFiniteWavefunctionValue(l[electron] + impl_->own_laplacian[electron]))
      throw std::overflow_error("Complex PsiFormer Laplacian scatter overflow");
  }
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      g[electron][dimension] += impl_->own_gradient[electron][dimension];
    l[electron] += impl_->own_laplacian[electron];
  }
  impl_->accepted = result.amplitude;
  impl_->accepted_key = configurationIdentity(p);
  impl_->accepted_version = result.parameter_version;
  impl_->accepted_valid = true;
  impl_->own_full_valid = true;
  return log_value_ = physical;
}

/// VALUE ratio proposals never allocate a spatial or parameter tape.
PsiFormerComplexWF::PsiValue PsiFormerComplexWF::ratio(ParticleSet& p, int electron)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  impl_->discardPending();
  impl_->validate(p, electron, true);
  const auto result = impl_->core(p, false, electron);
  const PsiValue ratio = impl_->formRatio(result.amplitude, p, electron);
  const LogValue physical = impl_->physicalLog(result.amplitude, p, electron);
  impl_->cache(result.amplitude, physical, p, electron);
  return ratio;
}

/// Accepted ACTIVE jets do not overwrite accepted scalar or pending move state.
PsiFormerComplexWF::GradType PsiFormerComplexWF::evalGrad(ParticleSet& p, int electron)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  impl_->validate(p, electron);
  if (electron < 0)
    throw std::invalid_argument("Complex PsiFormer gradient electron is negative");
  const auto result = impl_->core(p, true, -1, electron);
  GradType gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
    gradient[dimension] = impl_->physicalGradient(result.gradient[dimension], dimension);
  return gradient;
}

/// Form finite ratio/gradient staging before the nonthrowing caller/proposal publication.
PsiFormerComplexWF::PsiValue PsiFormerComplexWF::ratioGrad(ParticleSet& p, int electron, GradType& gradient)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  impl_->discardPending();
  impl_->validate(p, electron, true);
  const auto result = impl_->core(p, true, electron, electron);
  const PsiValue ratio = impl_->formRatio(result.amplitude, p, electron);
  const LogValue physical = impl_->physicalLog(result.amplitude, p, electron);
  GradType staged = gradient;
  for (int dimension = 0; dimension < 3; ++dimension)
  {
    staged[dimension] += impl_->physicalGradient(result.gradient[dimension], dimension);
    if (!isFiniteWavefunctionValue(staged[dimension]))
      throw std::overflow_error("Complex PsiFormer proposed gradient scatter overflow");
  }
  gradient = staged;
  impl_->cache(result.amplitude, physical, p, electron);
  return ratio;
}

/// Commit cached amplitude only after proving ParticleSet active proposal provenance.
void PsiFormerComplexWF::acceptMove(ParticleSet& p, int electron, bool)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  const bool pending = impl_->pending;
  impl_->discardPending();
  impl_->validate(p, electron, true);
  if (!pending || !impl_->accepted_valid ||
      impl_->accepted_key != configurationIdentity(p) ||
      impl_->accepted_version != impl_->state->model.p.version() || electron != impl_->proposed_particle ||
      impl_->proposed_key != configurationIdentity(p, electron) ||
      impl_->proposed_version != impl_->state->model.p.version())
    throw std::logic_error("Complex PsiFormer accept does not match its cached proposal");
  impl_->accepted = impl_->proposed;
  impl_->own_full_valid = false; // VALUE/ACTIVE accepted state is never a FULL persistence/energy cache
  impl_->accepted_key = impl_->proposed_key;
  impl_->accepted_version = impl_->proposed_version;
  impl_->accepted_valid = true;
  log_value_ = impl_->proposed_log;
}

/// Reject mismatched restore requests without committing or evaluating any move.
void PsiFormerComplexWF::restore(int electron)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  const bool pending = impl_->pending;
  const int proposed_particle = impl_->proposed_particle;
  impl_->discardPending();
  if (pending && electron != proposed_particle)
    throw std::logic_error("Complex PsiFormer restore does not match its cached proposal");
}

/// Clone immutable provenance, but never accepted state, pending moves or numerical allocations.
std::unique_ptr<WaveFunctionComponent> PsiFormerComplexWF::makeClone(ParticleSet& particles) const
{
  std::shared_lock read(impl_->state->mutex);
  return std::unique_ptr<WaveFunctionComponent>(new PsiFormerComplexWF(getName(),
      std::make_unique<Impl>(impl_->state, particles, impl_->numeric_limit)));
}

/// Register scalar extents only; the standard caller allocates and zero-initializes the shared pool.
void PsiFormerComplexWF::registerData(ParticleSet& p, WFBufferType& buffer)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  try
  {
    impl_->requireIdle(p);
    pf::checkedStorageSum(buffer.current_scalar(), impl_->recordScalars(), "complex record cursor overflow");
    double placeholder = 0;
    for (std::size_t scalar = 0; scalar < impl_->recordScalars(); ++scalar)
      buffer.add(placeholder);
  }
  catch (...) { impl_->invalidate(); throw; }
}

/// Persist only own complete products, adding them to caller totals exactly once.
PsiFormerComplexWF::LogValue PsiFormerComplexWF::updateBuffer(ParticleSet& p, WFBufferType& buffer, bool from_scratch)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  try
  {
    impl_->requireIdle(p);
    impl_->requireRecordExtent(buffer);
    if (from_scratch || !impl_->fullMatches(p))
      evaluateLogLocked(p, p.G, p.L);
    else
    {
      if (p.G.size() != impl_->own_gradient.size() || p.L.size() != impl_->own_laplacian.size())
        throw std::invalid_argument("Complex PsiFormer walker scatter extent mismatch");
      // Validate every addition before publishing any caller product.
      for (std::size_t e = 0; e < impl_->own_gradient.size(); ++e)
      {
        for (int d = 0; d < 3; ++d)
          if (!isFiniteWavefunctionValue(p.G[e][d] + impl_->own_gradient[e][d]))
            throw std::overflow_error("Complex PsiFormer walker gradient scatter overflow");
        if (!isFiniteWavefunctionValue(p.L[e] + impl_->own_laplacian[e]))
          throw std::overflow_error("Complex PsiFormer walker Laplacian scatter overflow");
      }
      for (std::size_t e = 0; e < impl_->own_gradient.size(); ++e)
      {
        p.G[e] += impl_->own_gradient[e];
        p.L[e] += impl_->own_laplacian[e];
      }
    }
    impl_->putRecord(buffer, log_value_);
    return log_value_;
  }
  catch (...) { impl_->invalidate(); throw; }
}

/// Validate all record content before accepted publication; failures revoke every cache.
void PsiFormerComplexWF::copyFromBuffer(ParticleSet& p, WFBufferType& buffer)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  const bool pending = impl_->pending;
  impl_->invalidate();
  impl_->requireIdle(p);
  if (pending)
    throw std::logic_error("Complex PsiFormer cannot restore over a pending proposal");
  Impl::Amplitude amplitude;
  LogValue physical;
  if (!impl_->getRecord(p, buffer, amplitude, physical))
    return;
  impl_->accepted = amplitude;
  impl_->accepted_key = configurationIdentity(p);
  impl_->accepted_version = impl_->state->model.p.version();
  impl_->accepted_valid = impl_->own_full_valid = true;
  log_value_ = physical;
}

/// Observe the complete complex kinetic energy with actual masses and ordinary, not Hermitian, squares.
PsiFormerComplexWF::CoulombObservation PsiFormerComplexWF::observeCoulombEnergy(
    const ParticleSet& p, FullPrecRealType potential) const
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireIdle(p);
  if (!impl_->fullMatches(p))
    throw std::logic_error("Complex PsiFormer Coulomb observation requires current FULL own products");
  if (!psiformer::determinant::isFiniteReal(potential))
    throw std::invalid_argument("Complex PsiFormer Coulomb potential must be finite and from this sample");
  LogValue kinetic{};
  const auto& masses = p.get_mass_by_group();
  if (masses.size() != static_cast<std::size_t>(p.groups()))
    throw std::invalid_argument("Complex PsiFormer target mass extent mismatch");
  for (int group = 0; group < p.groups(); ++group)
  {
    const double mass = masses[group];
    if (!psiformer::determinant::isFiniteReal(mass) || mass <= 0)
      throw std::invalid_argument("Complex PsiFormer target mass must be positive and finite");
    const double inverse_mass = 1.0 / mass;
    for (int e = p.first(group); e < p.last(group); ++e)
    {
      LogValue ratio = impl_->own_laplacian[e];
      for (int d = 0; d < 3; ++d)
        ratio += LogValue(impl_->own_gradient[e][d]) * LogValue(impl_->own_gradient[e][d]);
      kinetic -= .5 * inverse_mass * ratio;
    }
  }
  const LogValue local_energy = kinetic + potential; // caller guarantees Coulomb V belongs to this sample
  if (!isFiniteWavefunctionValue(kinetic) || !isFiniteWavefunctionValue(local_energy))
    throw std::overflow_error("Complex PsiFormer Coulomb observation overflow");
  return {kinetic, local_energy, potential};
}

/// Expose the immutable fixed-source identity solely for final-system admission.
const ParticleSet& PsiFormerComplexWF::sourceParticles() const
{
  std::shared_lock read(impl_->state->mutex);
  return *impl_->state->source;
}

/// Each clone binds its own actual target independently of the shared parameter values.
const ParticleSet& PsiFormerComplexWF::targetParticles() const
{
  std::shared_lock read(impl_->state->mutex);
  return *impl_->particles;
}

/// Every unsupported path fails explicitly before touching caller products or legacy fallback storage.
void PsiFormerComplexWF::rejectUnsupported(const char* operation)
{
  throw std::logic_error(std::string("Complex PsiFormer CPU inference does not support ") + operation);
}

/// Return no fictitious source-force derivative.
PsiFormerComplexWF::GradType PsiFormerComplexWF::evalGradSource(ParticleSet&, ParticleSet&, int)
{
  rejectUnsupported("source gradients");
}

/// Mixed source/electron response remains unqualified.
PsiFormerComplexWF::GradType PsiFormerComplexWF::evalGradSource(ParticleSet&, ParticleSet&, int,
    TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>&,
    TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>&)
{
  rejectUnsupported("mixed source derivatives");
}

/// Hessian requests fail descriptively rather than abort.
void PsiFormerComplexWF::evaluateHessian(ParticleSet&, HessVector&)
{
  rejectUnsupported("Hessians");
}

/// A spinor request must not silently use the positional gradient.
PsiFormerComplexWF::GradType PsiFormerComplexWF::evalGradWithSpin(ParticleSet&, int, ComplexType&)
{
  rejectUnsupported("spin gradients");
}

/// A spinor request must not silently publish a positional proposal.
PsiFormerComplexWF::PsiValue PsiFormerComplexWF::ratioGradWithSpin(ParticleSet&, int, GradType&, ComplexType&)
{
  rejectUnsupported("spin proposal gradients");
}

/// All-to-one geometry cannot reuse one selected ParticleSet proposal.
void PsiFormerComplexWF::evaluateRatiosAlltoOne(ParticleSet&, std::vector<ValueType>&)
{
  rejectUnsupported("all-to-one ratios");
}

/// Evaluate pure wavefunction VALUE ratios without moving or publishing the accepted walker.
void PsiFormerComplexWF::evaluateRatios(const VirtualParticleSet& virtuals, std::vector<ValueType>& ratios)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  const auto& reference = virtuals.getRefPS();
  impl_->requireIdle(reference);
  impl_->validate(reference, virtuals.refPtcl);
  if (reference.isSpinor() || virtuals.isSpinor() || virtuals.refPtcl < 0 ||
      ratios.size() != virtuals.getTotalNum() || virtuals.R.size() != virtuals.getTotalNum())
    throw std::invalid_argument("Complex PsiFormer virtual VALUE shape/spin/reference mismatch");
  for (const auto& position : virtuals.R)
    for (int d = 0; d < 3; ++d)
      if (!psiformer::determinant::isFiniteReal(position[d]))
        throw std::invalid_argument("Complex PsiFormer virtual VALUE position is nonfinite");
  // Validate the current nonzero denominator before touching either numerical owner.
  impl_->formRatio(impl_->accepted, reference, virtuals.refPtcl, &reference.R[virtuals.refPtcl]);
  // The admitted clone budget also bounds this transient Q-vector; caller/model/shared GPU storage is excluded.
  if (ratios.size() > (impl_->numeric_limit - impl_->numeric_required) / sizeof(ValueType))
    throw std::length_error("Complex PsiFormer virtual VALUE staging exceeds clone numeric cap");
  std::vector<ValueType> staged(ratios.size());
  for (std::size_t q = 0; q < staged.size(); ++q)
  {
    const auto evaluated = impl_->core(reference, false, virtuals.refPtcl, -1, &virtuals.R[q]);
    staged[q] = impl_->formRatio(evaluated.amplitude, reference, virtuals.refPtcl, &virtuals.R[q]);
  }
  // No partial output is visible if a later virtual evaluation fails.
  std::copy(staged.begin(), staged.end(), ratios.begin());
}

/// Reuse the scalar numerical path without recursive Model locks, publishing only one exact version.
WaveFunctionComponent::EvaluationStamp PsiFormerComplexWF::mw_evaluateVirtualRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& particles,
    const RefVectorWithLeader<VirtualParticleSet>& scratch,
    const VirtualParticleBatch& batch, std::vector<ValueType>& ratios) const
{
  const auto state = impl_->state;
  std::size_t version;
  {
    std::shared_lock read(state->mutex);
    impl_->requireNoCollective();
    if (&components.getLeader() != this || components.size() != batch.walkerCount() ||
        particles.size() != components.size() || scratch.size() != components.size())
      throw std::invalid_argument("Complex virtual capture leader/list extent mismatch");
    batch.validateOutputExtent(ratios.size());
    batch.validateFor(particles);
    for (std::size_t sample = 0; sample < components.size(); ++sample)
    {
      const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[sample]);
      if (!component || component->impl_->state != state || component->impl_->particles != &particles[sample])
        throw std::invalid_argument("Complex virtual capture requires exact same-family reference clones");
      component->impl_->requireIdle(particles[sample]);
      component->impl_->validate(particles[sample], 0);
      component->impl_->formRatio(component->impl_->accepted, particles[sample], 0, &particles[sample].R[0]);
    }
    // Outer stage + base stage + segment output + scalar stage coexist at worst Q each.
    const auto bytes = pf::checkedStorageBytes<ValueType>(
        pf::checkedStorageProduct(4, batch.size(), "Complex virtual capture staging overflow"),
        "Complex virtual capture staging overflow");
    if (bytes > impl_->numeric_limit - impl_->numeric_required)
      throw std::length_error("Complex virtual capture nested staging exceeds clone numeric cap");
    app_log() << "Complex virtual capture before allocation: HOST_nested_staging_bytes=" << bytes
              << "; four_Q_complex_peak=yes; existing_clone_numeric_cap=yes" << std::endl;
    version = state->model.p.version();
  }
  std::vector<ValueType> staged(ratios.size());
  // Base validates distinct component/scratch objects, nonalias, spin and ragged descriptors.
  WaveFunctionComponent::mw_evaluateVirtualRatios(components, particles, scratch, batch, staged);
  {
    std::shared_lock read(state->mutex);
    if (impl_->state != state || state->model.p.version() != version ||
        state->pending_proposals.load(std::memory_order_relaxed))
      throw std::logic_error("Complex virtual capture family/version changed during VALUE traversal");
    impl_->requireNoCollective();
    for (std::size_t sample = 0; sample < components.size(); ++sample)
    {
      const auto& component = static_cast<const PsiFormerComplexWF&>(components[sample]);
      if (component.impl_->state != state)
        throw std::logic_error("Complex virtual capture clone changed family during VALUE traversal");
      component.impl_->requireIdle(particles[sample]);
    }
    ratios.swap(staged);
    return EvaluationStamp::versioned(state.get(), static_cast<std::uint64_t>(version));
  }
}

/// Internal score does not qualify virtual parameter response.
void PsiFormerComplexWF::evaluateDerivRatios(const VirtualParticleSet&, const OptVariables&,
    std::vector<ValueType>&, Matrix<ValueType>&)
{
  rejectUnsupported("virtual derivatives");
}

/// Reject before allocating the base compatibility derivative matrix.
void PsiFormerComplexWF::evaluateDerivRatiosWeighted(const VirtualParticleSet&, const OptVariables&,
    const std::vector<ValueType>&, ParameterDerivativeView)
{
  rejectUnsupported("weighted virtual derivatives");
}

/// Per-clone exact admission is not aggregate planned-memory qualification.
BatchMemoryContribution PsiFormerComplexWF::estimateBatchExecutionMemory(const BatchExecutionPlanningContext&) const
{
  rejectUnsupported("explicit batch memory planning");
}

/// Only the ordinary unplanned execution contract is admitted.
void PsiFormerComplexWF::validateBatchExecutionPlanBinding(const BatchExecutionParticipantPlan& plan) const
{
  if (plan)
    rejectUnsupported("batch plan binding");
}

/// Preparation may be a no-op only for an empty participant plan.
void PsiFormerComplexWF::prepareBatchExecutionClone(const BatchExecutionParticipantPlan& plan)
{
  if (plan)
    rejectUnsupported("planned clone preparation");
}

/// Public optimization requires complete complex kinetic response, not only a score.
void PsiFormerComplexWF::evaluateDerivatives(ParticleSet&, const OptVariables&,
    Vector<ValueType>&, Vector<ValueType>&)
{
  throw std::logic_error("Complex PsiFormer public optimization is not admitted");
}
#endif
#endif
