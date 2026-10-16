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
  /** One model-shared B=1 lane; lock through conversion and caller/cache publication.
   * Numeric owners share one immutable image. No borrowed GPU output escapes a call.
   */
  struct GpuLane
  {
    /// Paired inference admits explicit forward VGL only; the legacy REAL selector is not applicable.
    static psiformer::device::TargetVglPreparation configuredPreparation()
    {
      const auto selection = psiformer::device::parseDeviceTargetSelection(
          std::getenv("PSIFORMER_GPU_TARGET_SELECTION"));
      if (selection.first || selection.second)
        throw std::invalid_argument("Complex GPU inference requires explicit VGL targets, not automatic selection");
      for (const char* name : {"PSIFORMER_GPU_KINETIC_TARGETS", "PSIFORMER_GPU_LIFTED_CHANNELS",
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
        std::size_t limit)
    {
      using namespace psiformer::device;
      if (!limit || limit == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU inference requires a positive finite shared-lane byte cap");
      const auto preparation = configuredPreparation();
      const auto value = estimateComplexValueCost(plan, 1);
      const auto spatial = preparation.max_targets
          ? estimateComplexSpatialTargetCost(plan, preparation.max_targets) : estimateComplexSpatialCost(plan, 1);
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
          psiformer::spatial_detail::checkedAdd(sizeof(ComplexValueResult),
              psiformer::spatial_detail::checkedProduct(staging_per_electron, plan.modelShape().electrons(), "Complex GPU staging overflow"),
              "Complex GPU staging overflow"), "Complex GPU host bytes overflow");
      return std::unique_ptr<GpuLane>(new GpuLane(plan, parameters, nuclei, version, value, spatial,
                                                preparation, work, bytes, host));
    }

    /// Prepare fixed host staging and two qualified numeric owners, never a CPU tape.
    GpuLane(const psiformer::PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
        const std::vector<double>& nuclei, std::size_t version,
        psiformer::device::ComplexValueCost value_cost, psiformer::device::ComplexSpatialCost spatial_cost,
        psiformer::device::TargetVglPreparation preparation, std::size_t work, std::size_t bytes, std::size_t host)
        : blas(queue), image(psiformer::device::PreparedDeviceModel::preparePairedComplexValue(plan, parameters, nuclei, version)),
          value(psiformer::device::PreparedDeviceComplexValue::prepare(image, 1, queue, blas, value_cost.device_bytes)),
          full(preparation.max_targets
              ? psiformer::device::PreparedDeviceComplexSpatial::prepare(image, queue, blas, preparation)
              : psiformer::device::PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, spatial_cost.device_bytes)),
          positions(3 * plan.modelShape().electrons()), amplitude(1), gradient(positions.size()),
          laplacian(plan.modelShape().electrons()), ratio(laplacian.size()), converted(4 * laplacian.size()),
          device_bytes(bytes), host_bytes(host)
    {
      const auto actual_host = value_cost.host_bytes + spatial_cost.host_bytes +
              sizeof(double) * positions.capacity() + sizeof(psiformer::device::ComplexValueResult) * amplitude.capacity() +
              sizeof(cuDoubleComplex) * (gradient.capacity() + laplacian.capacity() + ratio.capacity()) +
              sizeof(C) * converted.capacity();
      if (actual_host != host_bytes)
        throw std::logic_error("Complex GPU host capacities differ from checked preparation");
      app_log() << "  Complex PsiFormer GPU inference prepared: mode="
                << (preparation.max_targets ? "EXPLICIT_TARGET" : "FULL") << ", K=" << preparation.max_targets
                << ", spatial_owner_device_bytes=" << spatial_cost.device_bytes << ", shared_device_bytes=" << bytes
                << ", shared_host_numeric_bytes=" << host << ", request_scalar_work=" << work
                << "; shared_image_and_vendor_bytes_excluded=yes" << std::endl;
    }

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
    std::mutex mutex;
  };
#endif

  /// A weak family reference shares scratch without extending its lifetime or creating a cycle.
  struct DerivativeLane;

  /// Stable shared ownership keeps metadata/executors fixed while canonical values mutate under mutex.
  struct Model
  {
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
          const ParticleSet& electrons, const ParticleSet& ions, std::array<double, 3> reduced, std::size_t gpu_limit)
        : model(std::move(initialized.model)), profile(std::move(initialized.profile)), seed(initialized.seed),
          environment(environmentFor(electrons)), reduced_twist(reduced),
          cartesian_twist(makeTwistPhase(electrons, reduced).cartesian),
          plan(psiformer::PsiFormerExecutionPlan::fromParameters(model.p,
              {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
               model.dim, model.heads, model.blocks}, environment)), value(model, plan), spatial(model, plan),
          identity(persistentModelIdentity(model, "internal", profile, seed, environment,
                                           reduced_twist, cartesian_twist)), source(&ions),
          schema(makeStructuredParameterSchema(component_name, model.p)), gpu_limit(gpu_limit)
    {
      if (gpu_limit)
      {
#ifdef ENABLE_CUDA
        gpu = GpuLane::prepare(plan, model.p.flat_values(), model.cfg.nuclei.x, model.p.version(), gpu_limit);
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
#ifdef ENABLE_CUDA
    std::unique_ptr<GpuLane> gpu;
#endif
    mutable std::shared_mutex mutex;
    std::atomic<std::size_t> pending_proposals{0};
    std::mutex derivative_factory_mutex;
    std::weak_ptr<DerivativeLane> derivative_lane;
  };

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
            lane->score_work->setPositions(coordinates);
            score = lane->score.evaluate(*lane->score_work).parameter_score;
          }
          if (need_kinetic)
          {
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

  /// Copy physical positions through checked double input setters, replacing only the active row.
  template<class Workspace>
  void positions(Workspace& workspace, const ParticleSet& p, int replacement = -1) const
  {
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension,
            replacement == electron ? p.activeR(electron)[dimension] : p.R[electron][dimension]);
  }

  /// Common core view keeps a GPU operation locked until its scalar caller publishes copies.
  struct CoreResult : pf::DirectComplexSpatialResult
  {
    std::unique_lock<std::mutex> lease;
  };

  /// Route VALUE/FULL/ACTIVE without duplicating physical twist, ratio or cache algebra.
  CoreResult core(const ParticleSet& p, bool spatial, int replacement = -1, int electron = -1)
  {
    CoreResult result;
#ifdef ENABLE_CUDA
    if (state->gpu)
    {
      auto& lane = *state->gpu;
      result.lease = std::unique_lock(lane.mutex);
      for (int e = 0; e < p.getTotalNum(); ++e)
        for (int d = 0; d < 3; ++d)
          lane.positions[e * 3 + d] = replacement == e ? p.activeR(e)[d] : p.R[e][d];
      if (spatial)
      {
        lane.full->evaluate(lane.positions);
        lane.full->download(lane.full->resultView(), lane.amplitude, lane.gradient, lane.laplacian, lane.ratio);
        for (std::size_t i = 0; i < lane.gradient.size(); ++i)
          lane.converted[i] = {lane.gradient[i].x, lane.gradient[i].y};
        for (std::size_t i = 0; i < lane.laplacian.size(); ++i)
          lane.converted[lane.gradient.size() + i] = {lane.laplacian[i].x, lane.laplacian[i].y};
        result.gradient = lane.converted.data() + (electron < 0 ? 0 : 3 * electron);
        result.laplacian_log = lane.converted.data() + lane.gradient.size();
      }
      else
      {
        lane.value->evaluate(lane.positions);
        lane.value->download(lane.value->resultView(), lane.amplitude);
      }
      result.amplitude = {{lane.amplitude[0].phase.x, lane.amplitude[0].phase.y}, lane.amplitude[0].log_abs};
      result.parameter_version = lane.image->version();
      return result;
    }
#endif
    if (spatial)
    {
      auto& workspace = electron < 0 ? *full : *active;
      positions(workspace, p, replacement);
      static_cast<pf::DirectComplexSpatialResult&>(result) = electron < 0
          ? state->spatial.evaluateFull(workspace) : state->spatial.evaluateActive(workspace, electron);
    }
    else
    {
      positions(*value, p, replacement);
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

  /// Add i*k to logarithmic gradients; logarithmic Laplacians receive no outer-phase term.
  ValueType physicalGradient(C core, std::size_t dimension) const
  {
    const auto result = makePsiFormerScalar<ValueType>(core.real(), core.imag() + state->cartesian_twist[dimension]);
    if (!isFiniteWavefunctionValue(result))
      throw std::overflow_error("Complex PsiFormer physical gradient overflow");
    return result;
  }

  /// Validate the accepted reference before computing a complex ratio from unit core phases.
  PsiValue formRatio(const Amplitude& proposed, const ParticleSet& p, int electron) const
  {
    if (!accepted_valid || accepted_key != configurationIdentity(p) ||
        accepted_version != state->model.p.version() || accepted.isZero())
      throw std::logic_error("Complex PsiFormer ratio requires a current accepted amplitude");
    C result{};
    if (!proposed.isZero())
    {
      double delta = 0;
      for (int dimension = 0; dimension < 3; ++dimension)
        delta += state->cartesian_twist[dimension] * (p.activeR(electron)[dimension] - p.R[electron][dimension]);
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

  std::shared_ptr<Model> state;
  const ParticleSet* particles;
  std::unique_ptr<pf::DirectComplexValueWorkspace> value;
  std::unique_ptr<pf::DirectComplexSpatialWorkspace> full, active;
  std::vector<GradType> own_gradient;
  std::vector<ValueType> own_laplacian;
  std::size_t numeric_required = 0, numeric_limit;
  Amplitude accepted, proposed;
  LogValue proposed_log{};
  std::uint64_t accepted_key = 0, proposed_key = 0;
  std::size_t accepted_version = 0, proposed_version = 0;
  int proposed_particle = -1;
  bool accepted_valid = false, pending = false, own_full_valid = false;
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

/// Bind restart to fixed physical semantics, never mutable parameters or pointer identities.
std::string PsiFormerComplexWF::trainingConfigurationFingerprint() const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& s = *impl_->state;
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
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
  validateRuntimeLattice(*state->source, state->environment, "Complex PsiFormer derivative source");
  if (state->source->getTotalNum() != static_cast<int>(state->model.cfg.nuclei.shape[0]))
    throw std::invalid_argument("Complex PsiFormer derivative fixed-source extent mismatch");
  const auto& species = state->source->getSpeciesSet();
  const int charge = species.findAttribute("charge");
  if (charge < 0)
    throw std::invalid_argument("Complex PsiFormer derivative fixed-source charges are missing");
  for (int atom = 0; atom < state->source->getTotalNum(); ++atom)
  {
    if (species(charge, state->source->GroupID[atom]) != state->model.cfg.charges.x[atom])
      throw std::invalid_argument("Complex PsiFormer derivative fixed-source charge changed");
    for (int d = 0; d < 3; ++d)
      if (state->source->R[atom][d] != state->model.cfg.nuclei.x[3 * atom + d])
        throw std::invalid_argument("Complex PsiFormer derivative fixed-source configuration changed");
  }
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

/// Destroy native numerical state only in the sole owner TU.
PsiFormerComplexWF::~PsiFormerComplexWF()
{
  std::shared_lock read(impl_->state->mutex);
  impl_->discardPending();
}

/// Report the actual complete determinant expansion.
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

/// Copy canonical values and their exact version in one read transaction.
wftrain::StructuredParameterSnapshot PsiFormerComplexWF::snapshotParameters() const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& parameters = impl_->state->model.p;
  return {parameterSchema().fingerprint(), parameters.version(), parameters.flat_values()};
}

/// Validate every candidate invariant before the same-size native copy synchronizes graph leaves.
std::size_t PsiFormerComplexWF::publishParameters(const wftrain::StructuredParameterSnapshot& candidate,
                                                std::size_t expected_version)
{
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

/// Virtual/ECP geometry has no public complex qualification.
void PsiFormerComplexWF::evaluateRatios(const VirtualParticleSet&, std::vector<ValueType>&)
{
  rejectUnsupported("virtual/ECP ratios");
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
