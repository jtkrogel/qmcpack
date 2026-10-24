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

  /** Family-wide authority for CPU scratch and collective products, including dormant resources.
   * This numerical budget excludes clone G/L, parameters and derivative owners.
   */
  struct CpuScratchBudget
  {
    explicit CpuScratchBudget(std::size_t cap) : limit(cap)
    {
      if (!cap || cap == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex shared CPU scratch requires a positive finite cap");
    }

    /// Reserve before allocation; the caller rolls back an unpublished candidate.
    void reserve(std::size_t bytes)
    {
      std::lock_guard lock(mutex);
      if (bytes > limit - retained)
        throw std::length_error("Complex shared CPU scratch exceeds its family numeric cap before allocation");
      retained += bytes;
    }

    /// Return only this owner's admitted bytes, including allocation rollback.
    void release(std::size_t bytes) noexcept
    {
      std::lock_guard lock(mutex);
      retained -= bytes;
    }

    const std::size_t limit;
    std::mutex mutex;
    std::size_t retained = 0, owners = 0, fingerprint = 0;
  };

  /** Exclusive lazy CPU VALUE/FULL/ACTIVE scratch with no accepted walker state.
   * A CoreResult retains this owner and its execution lock until output copies finish.
   */
  struct CpuLane
  {
    enum class Mode { VALUE, FULL_VGL, ACTIVE_ELECTRON_GRADIENT };

    explicit CpuLane(std::shared_ptr<CpuScratchBudget> authority) : budget(std::move(authority))
    {
      std::lock_guard lock(budget->mutex);
      ++budget->owners;
    }

    ~CpuLane()
    {
      // Release allocations before returning their reservation to another lane.
      value.reset();
      full.reset();
      active.reset();
      std::lock_guard lock(budget->mutex);
      budget->retained -= charged;
      --budget->owners;
      budget->fingerprint ^= fingerprint;
    }

    /// Publish only a complete, exactly sized workspace; failures preserve prior modes.
    template<class Workspace, class Factory>
    Workspace& prepare(std::unique_ptr<Workspace>& destination, std::size_t bytes, Factory factory, Mode mode)
    {
      if (!destination)
      {
        budget->reserve(bytes);
        try
        {
          auto candidate = factory();
          if (candidate->vectorStorageBytes() != bytes)
            throw std::logic_error("Complex shared CPU workspace differs from its admitted requirement");
          const auto next = (candidate->storageFingerprint() ^ bytes) * 1099511628211ULL;
          std::lock_guard lock(budget->mutex);
          destination = std::move(candidate);
          charged += bytes;
          fingerprint ^= next;
          budget->fingerprint ^= next;
        }
        catch (...)
        {
          budget->release(bytes);
          throw;
        }
        // No trace work on warm calls; a failed numerical preparation owns no event.
        // HostCapacity is nonthrowing and keeps optional I/O failure outside science.
        const auto index = static_cast<std::size_t>(mode);
        trace_capacity[index].snapshotAt(__FILE__, __LINE__, trace_names[index], bytes,
                                         reinterpret_cast<std::uintptr_t>(this));
      }
      return *destination;
    }

    std::shared_ptr<CpuScratchBudget> budget;
    std::mutex mutex;
    static constexpr std::array<const char*, 3> trace_names{
        "ComplexWF::shared_cpu_VALUE", "ComplexWF::shared_cpu_FULL_VGL",
        "ComplexWF::shared_cpu_ACTIVE_ELECTRON_GRADIENT"};
    std::array<psiformer::memory::HostCapacity, 3> trace_capacity;
    std::unique_ptr<pf::DirectComplexValueWorkspace> value;
    std::unique_ptr<pf::DirectComplexSpatialWorkspace> full, active;
    std::size_t charged = 0, fingerprint = 0;
  };

#ifdef ENABLE_CUDA
  /** One exclusive numerical lane, owned by the model or an opt-in crowd resource.
   * Numeric owners share one immutable image. No borrowed GPU output escapes a call.
   */
  struct GpuLane
  {
    /// Capture trace concurrency once; successor and crowd owners reuse this profile.
    static std::size_t configuredTraceTargets()
    {
      const char* option = std::getenv("PSIFORMER_GPU_COMPLEX_SPATIAL_TRACE_TARGETS");
      if (!option || std::string(option) == "0")
        return 0;
      const auto targets = configuredCapacity("PSIFORMER_GPU_COMPLEX_SPATIAL_TRACE_TARGETS");
      const char* cooperative = std::getenv("PSIFORMER_GPU_COOPERATIVE_SPATIAL_TRACES");
      if (targets > 8 || !cooperative || std::string(cooperative) != "1")
        throw std::invalid_argument("Complex spatial trace targets require capacity1..8 and cooperative spatial traces=1");
      return targets;
    }

    /// A separate VALUE-only width never multiplies the accepted FULL owner.
    static std::size_t configuredVirtualCapacity()
    {
      const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY");
      if (!value || std::string(value) == "0")
        return 0;
      return configuredCapacity("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY");
    }

    /// Capture request-local envelope reuse only for the separate ordinary virtual VALUE owner.
    static bool configuredVirtualEnvelopeReuse()
    {
      const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_VIRTUAL_ENVELOPE_REUSE");
      if (!value || std::string_view(value) == "0")
        return false;
      if (std::string_view(value) == "1")
        return true;
      throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_VIRTUAL_ENVELOPE_REUSE must be 0 or 1");
    }

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

    /// Capture the optional determinant grouping once for this prepared family.
    static bool configuredHeadDeterminantBatch()
    {
      const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_HEAD_DETERMINANT_BATCH");
      if (!value || std::string_view(value) == "0")
        return false;
      if (std::string_view(value) == "1")
        return true;
      throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_HEAD_DETERMINANT_BATCH must be 0 or 1");
    }

    /// Freeze the ordinary FP64 FULL inverse schedule with its prepared family.
    static bool configuredFullInverseColumns()
    {
      const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_FULL_INVERSE_COLUMNS");
      if (!value || std::string_view(value) == "0")
        return false;
      if (std::string_view(value) == "1")
        return true;
      throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_FULL_INVERSE_COLUMNS must be 0 or 1");
    }

    /// Freeze exact within-request target factors, never accepted-walker state.
    static bool configuredTargetFactorReuse()
    {
      const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_TARGET_FACTOR_REUSE");
      if (!value || std::string_view(value) == "0")
        return false;
      if (std::string_view(value) == "1")
        return true;
      throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_TARGET_FACTOR_REUSE must be 0 or 1");
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
      return prepareFrozen(plan, parameters, nuclei, version, limit, configuredPreparation(), capacity, {},
                           configuredVirtualCapacity(), configuredTraceTargets(), configuredVirtualEnvelopeReuse(),
                           configuredHeadDeterminantBatch(), configuredFullInverseColumns(), configuredTargetFactorReuse(),
                           psiformer::device::configuredPeriodicValueGeometryTiled());
    }

    /// Successors preserve the admitted family profile without reading mutable process controls again.
    static std::unique_ptr<GpuLane> prepareFrozen(const psiformer::PsiFormerExecutionPlan& plan,
        const std::vector<double>& parameters, const std::vector<double>& nuclei, std::size_t version,
        std::size_t limit, psiformer::device::TargetVglPreparation preparation, std::size_t capacity = 1,
        std::shared_ptr<const psiformer::device::PreparedDeviceModel> shared_image = {},
        std::size_t virtual_capacity = 0, std::size_t trace_targets = 0, bool virtual_envelope_reuse = false,
        bool head_determinant_batch = false, bool full_inverse_columns = false, bool target_factor_reuse = false,
        bool periodic_value_tiling = false)
    {
      using namespace psiformer::device;
      // Freeze one effective choice for every simultaneous ordinary VALUE owner.
      const bool tiled_requested = periodic_value_tiling;
      const auto& environment = plan.environment();
      periodic_value_tiling = periodic_value_tiling &&
          plan.modelShape().electrons() >= 64 && environment.fixed_nuclei &&
          !environment.periodic_profile &&
          environment.geometry_feature_policy == psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
          std::all_of(environment.periodic_axes.begin(), environment.periodic_axes.end(),
              [](bool axis) { return axis; });
      if (tiled_requested && !periodic_value_tiling)
        std::clog << "Periodic VALUE geometry tiling: effective=0 reason=ineligible_profile\n";
      const auto canonical_geometry = [&] {
        app_log() << "  Periodic VALUE geometry tiling: effective=0 reason=optional_admission"
                  << "; unchanged_baseline_owners=yes" << std::endl;
        return prepareFrozen(plan, parameters, nuclei, version, limit, preparation, capacity, shared_image,
            virtual_capacity, trace_targets, virtual_envelope_reuse, head_determinant_batch,
            full_inverse_columns, target_factor_reuse, false);
      };
      if (!limit || limit == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU inference requires a positive finite shared-lane byte cap");
      if (!capacity || capacity == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex GPU tile capacity must be positive and finite");
      if (virtual_envelope_reuse && !virtual_capacity)
        throw std::invalid_argument("Virtual envelope reuse requires the separate complex virtual tile owner");
      if (target_factor_reuse && !preparation.max_targets)
        throw std::invalid_argument("Request-local factor reuse requires explicit paired VGL targets");
      if (full_inverse_columns)
      {
        if (preparation.max_targets)
          throw std::invalid_argument("Parallel inverse columns require ordinary paired FULL, not target VGL");
        validateComplexInverseColumnsProfile(plan);
      }
      if (shared_image)
      {
        shared_image->validateCurrentDevice();
        if (shared_image->version() != version || shared_image->plan().parameterCount() != plan.parameterCount())
          throw std::invalid_argument("Complex crowd shared image/version differs from its admitted family");
      }
      if (capacity > 1 && !configuredDeviceMemoryAdmission().limit_bytes)
        throw std::invalid_argument("Dense complex GPU tiles require native aggregate device admission");
      const auto value = estimateComplexValueCost(plan, capacity, true, false, head_determinant_batch,
          periodic_value_tiling);
      const auto spatial = preparation.max_targets
          ? estimateComplexSpatialTargetCost(plan, preparation.max_targets, capacity, trace_targets,
                head_determinant_batch, target_factor_reuse, preparation.reuse_cusp_state)
          : estimateComplexSpatialCost(plan, capacity, trace_targets, head_determinant_batch);
      const auto work = preparation.max_targets
          ? estimateHeadFreeTargetPrefixCost(plan, preparation.max_targets, capacity,
                preparation.reuse_cusp_state).charged_scalar_work_units : 0;
      if (preparation.max_targets && (spatial.device_bytes > preparation.device_workspace_budget_bytes ||
                                     work > preparation.scalar_work_budget))
        throw std::length_error("Complex GPU target VGL exceeds its explicit owner byte/work limits");
      auto bytes = psiformer::spatial_detail::checkedAdd(value.device_bytes, spatial.device_bytes,
                                                             "Complex GPU lane bytes overflow");
      if (bytes > limit)
      {
        if (periodic_value_tiling)
          return canonical_geometry();
        throw std::length_error("Complex GPU VALUE+spatial lane exceeds its explicit device byte cap");
      }
      constexpr auto staging_per_electron = 3 * sizeof(double) + 5 * sizeof(cuDoubleComplex) + 4 * sizeof(C);
      auto host = psiformer::spatial_detail::checkedAdd(
          psiformer::spatial_detail::checkedAdd(value.host_bytes, spatial.host_bytes, "Complex GPU host bytes overflow"),
          psiformer::spatial_detail::checkedAdd(
              psiformer::spatial_detail::checkedProduct(sizeof(ComplexValueResult), capacity, "Complex GPU amplitude staging overflow"),
              psiformer::spatial_detail::checkedProduct(staging_per_electron,
                  psiformer::spatial_detail::checkedProduct(capacity, plan.modelShape().electrons(), "Complex GPU staging overflow"),
                  "Complex GPU staging overflow"),
              "Complex GPU staging overflow"), "Complex GPU host bytes overflow");
      if (virtual_capacity)
      {
        if (plan.environment().periodic_profile || !configuredDeviceMemoryAdmission().limit_bytes)
          throw std::invalid_argument("Complex virtual tiles require legacy periodic FP64 and aggregate device admission");
        const auto extra = estimateComplexValueCost(plan, virtual_capacity, true, virtual_envelope_reuse,
            head_determinant_batch, periodic_value_tiling);
        const auto add = [](std::size_t a, std::size_t b) {
          return psiformer::spatial_detail::checkedAdd(a, b, "Complex virtual tile bytes overflow");
        };
        const auto mul = [](std::size_t a, std::size_t b) {
          return psiformer::spatial_detail::checkedProduct(a, b, "Complex virtual tile bytes overflow");
        };
        bytes = add(bytes, extra.device_bytes);
        host = add(host, add(extra.host_bytes, mul(virtual_capacity,
            add(mul(3 * sizeof(double), plan.modelShape().electrons()),
                sizeof(ComplexValueResult) + sizeof(std::array<std::size_t, 3>)))));
        if (add(bytes, host) > limit)
        {
          if (periodic_value_tiling)
            return canonical_geometry();
          throw std::length_error("Complex virtual tile exceeds conservative combined DEVICE/HOST numeric bound");
        }
        app_log() << "  Complex virtual tile preflight: capacity=" << virtual_capacity
                  << ", extra_value_device_bytes=" << extra.device_bytes
                  << ", envelope_anchor_device_bytes=" << extra.envelope_anchor_bytes
                  << ", combined_numeric_bytes=" << add(bytes, host)
                  << "; existing_numeric_limit=yes; aggregate_HOST_fit=no; before_allocation=yes" << std::endl;
      }
      if (capacity > 1 || shared_image)
        app_log() << "  Complex PsiFormer dense GPU preflight: tile_capacity=" << capacity
                  << ", value_device_bytes=" << value.device_bytes << ", full_vgl_device_bytes=" << spatial.device_bytes
                  << ", host_numeric_bytes=" << host << "; before_allocation=yes; lane_serialized=yes"
                  << ", independent_resource_lane=" << bool(shared_image)
                  << "; image_and_vendor_bytes_excluded=yes" << std::endl;
      try
      {
        if (periodic_value_tiling)
        {
          int ordinal;
#ifdef QMC_CUDA2HIP
          checkRuntime(hipGetDevice(&ordinal), "tiled VALUE geometry device");
#else
          checkRuntime(cudaGetDevice(&ordinal), "tiled VALUE geometry device");
#endif
          // Charge both VALUE owners plus the unchanged spatial owner together.
          // Image/vendor ownership remains separately charged by existing buffers.
          { auto complete_graphs = deviceMemoryMetrics(ordinal).reserveAllocation(bytes); }
        }
        return std::unique_ptr<GpuLane>(new GpuLane(plan, parameters, nuclei, version, value, spatial,
            preparation, work, bytes, host, capacity, shared_image, virtual_capacity, trace_targets,
            virtual_envelope_reuse, head_determinant_batch, full_inverse_columns,
            target_factor_reuse, periodic_value_tiling));
      }
      catch (const std::length_error&)
      {
        if (!periodic_value_tiling) throw;
      }
      catch (const std::bad_alloc&)
      {
        if (!periodic_value_tiling) throw;
      }
      catch (const std::runtime_error& error)
      {
        if (!periodic_value_tiling || !recoverOptionalEnvelopeAllocation(error)) throw;
      }
      // A failed constructor has retired every unpublished owner before retry.
      return canonical_geometry();
    }

    /// Prepare fixed host staging and two qualified numeric owners, never a CPU tape.
    GpuLane(const psiformer::PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
        const std::vector<double>& nuclei, std::size_t version,
        psiformer::device::ComplexValueCost value_cost, psiformer::device::ComplexSpatialCost spatial_cost,
        psiformer::device::TargetVglPreparation preparation, std::size_t work, std::size_t bytes, std::size_t host,
        std::size_t capacity, std::shared_ptr<const psiformer::device::PreparedDeviceModel> shared_image,
        std::size_t virtual_capacity, std::size_t trace_targets, bool virtual_envelope_reuse,
        bool head_determinant_batch, bool full_inverse_columns, bool target_factor_reuse,
        bool periodic_value_tiling)
        : capacity(capacity), electrons(plan.modelShape().electrons()), virtual_capacity(virtual_capacity),
          trace_targets(trace_targets), virtual_envelope_reuse(virtual_envelope_reuse),
          head_determinant_batch(head_determinant_batch), full_inverse_columns(full_inverse_columns),
          target_factor_reuse(target_factor_reuse),
          periodic_value_tiling(periodic_value_tiling), blas(queue),
          image(shared_image ? std::move(shared_image) :
              (plan.environment().periodic_profile
                  ? psiformer::device::PreparedDeviceModel::preparePairedTwoCellProfile(plan, parameters, nuclei, version)
                  : psiformer::device::PreparedDeviceModel::preparePairedComplexValue(plan, parameters, nuclei, version))),
          value(psiformer::device::PreparedDeviceComplexValue::prepare(image, capacity, queue, blas,
              value_cost.device_bytes, false, head_determinant_batch, periodic_value_tiling)),
          full(preparation.max_targets
              ? psiformer::device::PreparedDeviceComplexSpatial::prepare(image, queue, blas, preparation, capacity,
                    trace_targets, head_determinant_batch, target_factor_reuse)
              : psiformer::device::PreparedDeviceComplexSpatial::prepare(image, capacity, queue, blas,
                    spatial_cost.device_bytes, trace_targets, head_determinant_batch, full_inverse_columns)),
          positions(3 * electrons * capacity), amplitude(capacity), gradient(positions.size()),
          laplacian(electrons * capacity), ratio(laplacian.size()), converted(4 * laplacian.size()),
          virtual_positions(3 * electrons * virtual_capacity), virtual_amplitude(virtual_capacity),
          virtual_indices(virtual_capacity),
          device_bytes(bytes), host_bytes(host), preparation(preparation)
    {
      if (virtual_capacity)
      {
        const auto extra = psiformer::device::estimateComplexValueCost(plan, virtual_capacity, true,
            virtual_envelope_reuse, head_determinant_batch, periodic_value_tiling);
        int ordinal;
#ifdef QMC_CUDA2HIP
        psiformer::device::checkRuntime(hipGetDevice(&ordinal), "complex virtual tile device");
#else
        psiformer::device::checkRuntime(cudaGetDevice(&ordinal), "complex virtual tile device");
#endif
        // Probe the complete addition, then release it before per-buffer reservations.
        // Concurrent model allocations can still cause clean RAII rollback.
        { auto probe = psiformer::device::deviceMemoryMetrics(ordinal).reserveAllocation(extra.device_bytes); }
        virtual_value = psiformer::device::PreparedDeviceComplexValue::prepare(
            image, virtual_capacity, queue, blas, extra.device_bytes, virtual_envelope_reuse,
            head_determinant_batch, periodic_value_tiling);
      }
      const auto actual_host = value_cost.host_bytes + spatial_cost.host_bytes +
              sizeof(double) * positions.capacity() + sizeof(psiformer::device::ComplexValueResult) * amplitude.capacity() +
              sizeof(cuDoubleComplex) * (gradient.capacity() + laplacian.capacity() + ratio.capacity()) +
              sizeof(C) * converted.capacity() +
              (virtual_value ? virtual_value->cost().host_bytes : 0) +
              sizeof(double) * virtual_positions.capacity() +
              sizeof(psiformer::device::ComplexValueResult) * virtual_amplitude.capacity() +
              sizeof(std::array<std::size_t, 3>) * virtual_indices.capacity();
      if (actual_host != host_bytes)
        throw std::logic_error("Complex GPU host capacities differ from checked preparation");
      if (value->cost().device_bytes + full->cost().device_bytes +
          (virtual_value ? virtual_value->cost().device_bytes : 0) != device_bytes)
        throw std::logic_error("Complex GPU owner capacities differ from checked preparation");
      app_log() << "  Complex PsiFormer GPU inference prepared: mode="
                << (preparation.max_targets ? "EXPLICIT_TARGET" : "FULL") << ", K=" << preparation.max_targets
                << ", tile_capacity=" << capacity
                << ", concurrent_trace_targets=" << trace_targets
                << ", virtual_envelope_reuse=" << virtual_envelope_reuse
                << ", head_determinant_batch=" << head_determinant_batch
                << ", full_inverse_columns=" << full_inverse_columns
                << ", target_factor_reuse=" << target_factor_reuse
                << ", trace_storage_fingerprint=" << full->traceStorageFingerprint()
                << ", spatial_owner_device_bytes=" << spatial_cost.device_bytes << ", shared_device_bytes=" << bytes
                << ", shared_host_numeric_bytes=" << host << ", request_scalar_work=" << work
                << ", cusp_state_reuse=" << (full ? full->usesCuspStateReuse() : false)
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
      if (virtual_value)
        virtual_value->validateModelRebind(current);
      queue.sync();
      auto retired = image; // Prevent graph adoption from retiring the last old image.
      value->adoptPreparedModel(current);
      full->adoptPreparedModel(current);
      if (virtual_value)
        virtual_value->adoptPreparedModel(current);
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

    const std::size_t capacity, electrons, virtual_capacity, trace_targets;
    const bool virtual_envelope_reuse;
    const bool head_determinant_batch;
    const bool full_inverse_columns;
    const bool target_factor_reuse;
    const bool periodic_value_tiling;
    compute::Queue<PlatformKind::CUDA> queue;
    psiformer::device::AcceleratorBlasHandle blas;
    std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexValue> value;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexValue> virtual_value;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexSpatial> full;
    std::vector<double> positions;
    std::vector<psiformer::device::ComplexValueResult> amplitude;
    std::vector<cuDoubleComplex> gradient, laplacian, ratio;
    std::vector<C> converted;
    std::vector<double> virtual_positions;
    std::vector<psiformer::device::ComplexValueResult> virtual_amplitude;
    std::vector<std::array<std::size_t, 3>> virtual_indices;
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
    static psiformer::ExecutionEnvironment environmentFor(const ParticleSet& electrons,
        const psiformer::ExecutionEnvironment& initialized)
    {
      auto environment = makeExecutionEnvironment(electrons);
      if (environment.boundary != psiformer::BoundaryCondition::PERIODIC || electrons.isSpinor())
        throw std::invalid_argument("Complex PsiFormer inference requires nonspinor periodic bulk electrons");
      environment.head_representation = psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX;
      environment.amplitude_scalar_domain = psiformer::ScalarDomain::COMPLEX;
      if (initialized.periodic_profile)
      {
        validateRuntimeLattice(electrons, initialized, "Complex PsiFormer periodic initialization");
        if (initialized.head_representation != environment.head_representation ||
            initialized.amplitude_scalar_domain != environment.amplitude_scalar_domain)
          throw std::invalid_argument("Complex PsiFormer periodic profile requires paired complex metadata");
        environment = initialized;
      }
      return environment;
    }

    /// Snapshot all physical ions independently of any reduced primitive feature motif.
    static std::vector<double> physicalSource(const ParticleSet& ions)
    {
      return physicalSourceSnapshot(ions);
    }

    /// Preserve ordered physical species labels independently of mutable ParticleSet metadata.
    static std::vector<std::string> physicalSpecies(const ParticleSet& ions)
    {
      return physicalSpeciesSnapshot(ions);
    }

    /// Preserve existing AE identities exactly; only explicit PP metadata adds a new distinction.
    static std::uint64_t identityFor(const pf::PsiFormer& model, const std::string& profile, std::uint64_t seed,
        const psiformer::ExecutionEnvironment& environment, const std::array<double, 3>& reduced,
        const std::array<double, 3>& cartesian, const std::string& kind, const ParticleSet& ions)
    {
      auto identity = persistentModelIdentity(model, "internal", profile, seed, environment, reduced, cartesian);
      if (kind != "all_electron") mixPersistentString(identity, kind);
      if (environment.periodic_profile)
      {
        for (double value : physicalSource(ions)) mixPersistentDouble(identity, value);
        for (int ion = 0; ion < ions.getTotalNum(); ++ion)
          mixPersistentString(identity, ions.getSpeciesSet().getSpeciesName(ions.GroupID[ion]));
      }
      return identity;
    }

    /// Retain initializer provenance and construct only qualified complex inference owners.
    Model(const std::string& component_name, InitializedNativeModel initialized,
          const ParticleSet& electrons, const ParticleSet& ions, std::array<double, 3> reduced, std::size_t gpu_limit,
          const std::string& kind = "all_electron", std::size_t cpu_limit = 0
#ifdef ENABLE_CUDA
          , std::shared_ptr<GpuLane> prepared = {}
#endif
          )
        : model(std::move(initialized.model)), profile(std::move(initialized.profile)), seed(initialized.seed),
          system_kind(kind), environment(environmentFor(electrons, initialized.environment)), reduced_twist(reduced),
          cartesian_twist(makeTwistPhase(electrons, reduced).cartesian),
          plan(psiformer::PsiFormerExecutionPlan::fromParameters(model.p,
              {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
               model.dim, model.heads, model.blocks}, environment)), value(model, plan), spatial(model, plan),
          identity(identityFor(model, profile, seed, environment, reduced_twist, cartesian_twist, system_kind, ions)), source(&ions),
          physical_source(physicalSource(ions)),
          physical_species(physicalSpecies(ions)),
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
      if (cpu_limit)
      {
        if (gpu_limit)
          throw std::invalid_argument("Complex shared CPU scratch cannot be combined with GPU inference");
        cpu_budget = std::make_shared<CpuScratchBudget>(cpu_limit);
        cpu = std::make_shared<CpuLane>(cpu_budget);
        const pf::PsiFormerStorageShape shape{model.ne, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads,
            plan.parameter(psiformer::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], model.blocks, model.p.size(),
            bool(environment.periodic_profile), occupiedMomentumCount(plan),
            environment.periodic_profile ? environment.periodic_profile->pair_feature_dimension : 8,
            fourierModeCount(plan)};
        cpu_requirements = {pf::DirectComplexValueWorkspace::storageRequirement(shape),
            pf::DirectComplexSpatialWorkspace::storageRequirement(shape, pf::DirectSpatialMode::FULL_VGL),
            pf::DirectComplexSpatialWorkspace::storageRequirement(shape, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT)};
      }
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
    const std::string system_kind;
    const psiformer::ExecutionEnvironment environment;
    const std::array<double, 3> reduced_twist, cartesian_twist;
    const psiformer::PsiFormerExecutionPlan plan;
    const pf::DirectComplexValueExecutor value;
    const pf::DirectComplexSpatialExecutor spatial;
    const std::uint64_t identity;
    const ParticleSet* const source;
    const std::vector<double> physical_source;
    const std::vector<std::string> physical_species;
    const std::shared_ptr<const wftrain::StructuredParameterSchema> schema;
    const std::size_t gpu_limit;
    const std::size_t gpu_capacity, gpu_crowds;
    std::shared_ptr<CpuScratchBudget> cpu_budget;
    std::shared_ptr<CpuLane> cpu;
    std::array<std::size_t, 3> cpu_requirements{};
    std::atomic<std::size_t> acquired_teams{0};
#ifdef ENABLE_CUDA
    psiformer::PsiFormerInferenceConcurrency inference_concurrency;
    std::shared_ptr<GpuLane> gpu;
    // Only budgeted GPU isolation establishes this lineage; publication never changes its source generation.
    std::weak_ptr<Model> private_origin;
    std::weak_ptr<const psiformer::device::PreparedDeviceModel> private_origin_image;
    std::size_t private_origin_version = 0;
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
      if (generation->virtual_value)
        generation->virtual_value->adoptPreparedModel(successor);
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

    /// Retire proposed CPU products before returning their family reservation.
    ~MultiWalkerResource() override
    {
      if (cpu_collective_bytes)
      {
        std::vector<Amplitude>().swap(amplitude);
        std::vector<LogValue>().swap(logs);
        std::vector<std::uint64_t>().swap(keys);
        std::vector<GradType>().swap(gradient);
        std::vector<ValueType>().swap(laplacian);
        std::lock_guard lock(state->cpu_budget->mutex);
        state->cpu_budget->retained -= cpu_collective_bytes;
        state->cpu_budget->fingerprint ^= cpu_collective_fingerprint;
      }
    }

    /// Prepare a whole candidate while old proposed products remain live and charged.
    void prepareCpuCollective(std::size_t walkers)
    {
      const char* message = "Complex CPU collective staging overflow";
      const auto elements = pf::checkedStorageProduct(walkers, state->model.ne, message);
      if (walkers <= amplitude.size() && elements <= gradient.size())
        return;
      auto bytes = pf::checkedStorageProduct(walkers,
          sizeof(Amplitude) + sizeof(LogValue) + sizeof(std::uint64_t), message);
      pf::addStorageBytes(bytes, pf::checkedStorageProduct(elements,
          sizeof(GradType) + sizeof(ValueType), message), message);
      state->cpu_budget->reserve(bytes);
      try
      {
        std::vector<Amplitude> next_amplitude(walkers);
        std::vector<LogValue> next_logs(walkers);
        std::vector<std::uint64_t> next_keys(walkers);
        std::vector<GradType> next_gradient(elements);
        std::vector<ValueType> next_laplacian(elements);
        const auto actual = next_amplitude.capacity() * sizeof(Amplitude) + next_logs.capacity() * sizeof(LogValue) +
            next_keys.capacity() * sizeof(std::uint64_t) + next_gradient.capacity() * sizeof(GradType) +
            next_laplacian.capacity() * sizeof(ValueType);
        if (actual != bytes)
          throw std::logic_error("Complex CPU collective capacity differs from its reservation");
        amplitude.swap(next_amplitude);
        logs.swap(next_logs);
        keys.swap(next_keys);
        gradient.swap(next_gradient);
        laplacian.swap(next_laplacian);
      }
      catch (...)
      {
        state->cpu_budget->release(bytes);
        throw;
      }
      // Local candidates (now holding old allocations) have been destroyed above.
      {
        std::lock_guard lock(state->cpu_budget->mutex);
        state->cpu_budget->retained -= cpu_collective_bytes;
        cpu_collective_bytes = bytes;
        state->cpu_budget->fingerprint ^= cpu_collective_fingerprint;
        cpu_collective_fingerprint = (reinterpret_cast<std::uintptr_t>(gradient.data()) ^ bytes) * 1099511628211ULL;
        state->cpu_budget->fingerprint ^= cpu_collective_fingerprint;
      }
      trace_cpu_collective.snapshotAt(__FILE__, __LINE__, "ComplexWF::shared_cpu_COLLECTIVE_PRODUCTS",
          bytes, reinterpret_cast<std::uintptr_t>(this));
    }

    /// Clone only the model anchor; each acquired crowd prepares independent HOST staging.
    std::unique_ptr<Resource> makeClone() const override
    { return std::make_unique<MultiWalkerResource>(state); }

    /// CPU scalar dispatch needs membership only, not dense GPU transaction staging.
    void prepareCpuTeam(const RefVectorWithLeader<WaveFunctionComponent>& list)
    {
      if (!cpu)
        cpu = std::make_shared<CpuLane>(state->cpu_budget);
      std::vector<PsiFormerComplexWF*> candidate;
      candidate.reserve(list.size());
      for (std::size_t i = 0; i < list.size(); ++i)
        candidate.push_back(&static_cast<PsiFormerComplexWF&>(list[i]));
      components.swap(candidate);
    }

    std::shared_ptr<CpuLane> cpu;

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
    std::size_t cpu_collective_bytes = 0, cpu_collective_fingerprint = 0;
    psiformer::memory::HostCapacity trace_cpu_collective;
    bool collective_pending = false;
    std::size_t collective_version = 0;
    std::uint64_t collective_descriptor = 0;
  };

  /// Fixed ions are part of the captured derivative problem, not an unchecked caller convention.
  static void validateDerivativeSource(const Model& s)
  {
    validateRuntimeLattice(*s.source, s.environment, "Complex PsiFormer derivative source");
    if (s.source->getTotalNum() != static_cast<int>(s.physical_source.size() / 5) ||
        s.source->R.size() != s.physical_source.size() / 5 || s.source->GroupID.size() != s.source->R.size())
      throw std::invalid_argument("Complex PsiFormer derivative fixed-source extent mismatch");
    const auto& species = s.source->getSpeciesSet();
    const int charge = species.findAttribute("charge");
    if (charge < 0)
      throw std::invalid_argument("Complex PsiFormer derivative fixed-source charges are missing");
    for (int atom = 0; atom < s.source->getTotalNum(); ++atom)
    {
      if (s.source->GroupID[atom] < 0 || s.source->GroupID[atom] >= s.source->groups() ||
          !psiformer::determinant::isFiniteReal(species(charge, s.source->GroupID[atom])) ||
          species(charge, s.source->GroupID[atom]) != s.physical_source[5 * atom + 3] ||
          s.source->GroupID[atom] != s.physical_source[5 * atom + 4] ||
          species.getSpeciesName(s.source->GroupID[atom]) != s.physical_species[atom])
        throw std::invalid_argument("Complex PsiFormer derivative fixed-source charge changed");
      for (int d = 0; d < 3; ++d)
        if (!psiformer::determinant::isFiniteReal(s.source->R[atom][d]) ||
            s.source->R[atom][d] != s.physical_source[5 * atom + d])
          throw std::invalid_argument("Complex PsiFormer derivative fixed-source configuration changed");
    }
  }

#ifdef ENABLE_CUDA
  /// One weakly shared immutable-image derivative lane; ownership keeps borrowed queue/BLAS alive.
  struct GpuDerivativeLane
  {
    using Products = psiformer::device::PreparedDeviceComplexProducts;

    /// Validate the public domain scope before estimating or allocating numerical owners.
    static auto numericalScope(GpuDerivativeScope scope)
    {
      using Scope = psiformer::device::ComplexProductsScope;
      switch (scope)
      {
      case GpuDerivativeScope::COMPLETE: return Scope::COMPLETE;
      case GpuDerivativeScope::SCORE_ECP_ONLY: return Scope::SCORE_ECP_ONLY;
      }
      throw std::invalid_argument("Complex GPU derivative preparation scope is invalid");
    }

    /// Aggregate R(S) permits S>maxB; a single operator is separately restricted to B<=maxB.
    static PsiFormerComplexWF::GpuDerivativeRequirements required(const Model& state, std::size_t samples,
        std::size_t maximum_samples, std::size_t targets, std::size_t chunk, GpuDerivativeScope scope)
    {
      const auto cost = psiformer::device::estimateComplexProductsCost(state.plan, targets, maximum_samples, 3, chunk,
          psiformer::device::ComplexProductsHostOutput::CHUNKED, numericalScope(scope));
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
        std::size_t chunk, const PsiFormerComplexWF::GpuDerivativeRequirements& budget, GpuDerivativeScope scope)
        : state(std::move(model)), generation(state->gpu), maximum_samples(maximum_samples), targets(targets), chunk(chunk),
          scope(scope),
          products(Products::prepare(generation->image, targets, maximum_samples, 3, chunk,
              generation->queue, generation->blas, budget.device_bytes,
              budget.host_bytes - maximum_samples * sizeof(C), std::max(std::size_t{1}, budget.matrix_work),
              psiformer::device::ComplexProductsHostOutput::CHUNKED, numericalScope(scope))),
          zeros(maximum_samples), channels(3)
    {}

    const std::shared_ptr<Model> state;
    /// Declare before Products: borrowers die before their exact queue/BLAS/image generation.
    const std::shared_ptr<GpuLane> generation;
    const std::size_t maximum_samples, targets, chunk;
    const GpuDerivativeScope scope;
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
        std::size_t offset, std::size_t samples, std::size_t host_cap)
        : lane(std::move(scratch)), version(lane->state->model.p.version()), ordinal(ordinal), offset(offset),
          samples(samples), host_cap(host_cap), chunks(*lane->state->schema, version, lane->chunk),
          positions(3 * samples * lane->state->model.ne), inverse_masses(samples * lane->state->model.ne)
    {}

    /// Numerical full-score/device-dot JVP and both adjoints retain REAL tangents/HOST complex outputs.
    wftrain::StreamingDerivativeCapabilities capabilities() const noexcept override
    {
      using namespace wftrain;
      StreamingDerivativeCapabilities c;
      c.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) | derivativeProductBit(DerivativeProduct::SCORE_JVP);
      c.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) | derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
      if (lane->scope == GpuDerivativeScope::COMPLETE)
      {
        c.product_mask |= derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
        c.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
      }
      c.maximum_vjp_channels = 3;
      c.maximum_parameter_chunk_size = chunks.maximumChunkSize();
      c.maximum_sample_tile_size = 1;
      // Conservative vector-count upper bound; exact mixed-width bytes are in storageDiagnostics().
      c.fixed_parameter_scratch_vectors = lane->scope == GpuDerivativeScope::COMPLETE ? 10 : 8;
      c.block_streaming = true;
      if (replay_bound.load(std::memory_order_acquire))
      {
        c.product_mask |= derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) |
            derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP);
        c.local_energy_term_mask |= localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
        c.maximum_vjp_channels = 3; // One complete energy channel plus SCORE EO/O, in any caller order.
      }
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
      const bool bound = replay_bound.load(std::memory_order_acquire);
      const auto& cost = lane->products->cost();
      d.parameter_count = cost.parameters;
      d.sample_count = samples;
      const bool complete = lane->scope == GpuDerivativeScope::COMPLETE;
      d.real_parameter_vectors = complete ? 4 : 3;
      d.complex_parameter_vectors = complete ? 5 : 4;
      d.parameter_scratch_bytes = (complete ? 112 : 88) * cost.parameters + 32 * cost.chunk;
      d.shared_workspace_bytes = cost.device_bytes + cost.host_bytes + lane->zeros.capacity() * sizeof(C);
      d.evaluator_workspace_bytes = d.shared_workspace_bytes - d.parameter_scratch_bytes;
      d.sample_position_bytes = positions.capacity() * sizeof(double);
      d.sample_auxiliary_bytes = inverse_masses.capacity() * sizeof(double);
      if (bound)
      {
        pf::addStorageBytes(d.sample_auxiliary_bytes, replay->storageDiagnostics().retained_numeric_bytes,
                            "Complex replay diagnostics overflow");
        pf::addStorageBytes(d.sample_auxiliary_bytes, replayHostBytes(), "Complex replay diagnostics overflow");
      }
      d.retained_numeric_bytes = d.shared_workspace_bytes + d.sample_position_bytes + d.sample_auxiliary_bytes;
      d.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(lane.get());
      d.allocation_generation = bound ? 2 : 1;
      d.storage_fingerprint = lane->products->storageFingerprint();
      for (const auto* storage : {&positions, &inverse_masses})
        psiformer::determinant::detail::mixStorage(d.storage_fingerprint, *storage);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, lane->zeros);
      if (bound)
      {
        d.storage_fingerprint ^= replay->storageDiagnostics().storage_fingerprint;
        for (const auto* storage : std::array<const std::vector<C>*, 2>{&complete_weights, &sample_coefficients})
          psiformer::determinant::detail::mixStorage(d.storage_fingerprint, *storage);
        psiformer::determinant::detail::mixStorage(d.storage_fingerprint, references);
      }
      return d;
    }

    /// Validate captured VALUE provenance and complete admission before adopting any replay storage.
    void bindNonLocalECPJVPReplay(std::unique_ptr<wftrain::NonLocalECPJVPReplay> incoming) override
    {
      std::unique_lock execution(lane->execution_mutex);
      std::shared_lock read(lane->state->mutex);
      requireCurrent();
      if (!incoming || replay || has_evaluated)
        throw std::logic_error("Complex replay requires a nonnull first binding before any derivative action");
      const auto& schema = parameterSchema();
      const auto points = incoming->points();
      const auto source = incoming->referencePositions();
      const auto diagnostics = incoming->storageDiagnostics();
      if (incoming->providerId() != schema.providerId() || incoming->schemaFingerprint() != schema.fingerprint() ||
          incoming->parameterVersion() != version || incoming->batchOrdinal() != ordinal ||
          incoming->sampleOffset() != offset || incoming->sampleCount() != samples ||
          incoming->electronCount() != lane->state->model.ne ||
          incoming->localEnergyTermMask() != wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP) ||
          !incoming->gridFingerprint() || source.size() != positions.size() ||
          !std::equal(source.begin(), source.end(), positions.begin()) ||
          diagnostics.point_count != points.size() || diagnostics.segment_count != incoming->segments().size() ||
          diagnostics.tile_count != incoming->tiles().size() ||
          diagnostics.evaluation_stamp_count != incoming->evaluationStamps().size() || !diagnostics.storage_fingerprint)
        throw std::invalid_argument("Complex replay source/grid/schema/reference diagnostics mismatch");
      if (samples > lane->maximum_samples || points.size() > lane->maximum_samples - samples)
        throw std::length_error("Complex replay B+Q exceeds the existing numerical request capacity");
      const char* message = "Complex replay HOST admission overflow";
      std::size_t extra = diagnostics.retained_numeric_bytes;
      pf::addStorageBytes(extra, pf::checkedStorageBytes<std::size_t>(points.size(), message), message);
      pf::addStorageBytes(extra, pf::checkedStorageBytes<C>(samples + points.size(), message), message);
      const auto current_host = storageDiagnostics().retained_numeric_bytes - lane->products->cost().device_bytes;
      if (extra > host_cap || current_host > host_cap - extra)
        throw std::length_error("Complex replay complete HOST storage exceeds factory admission");
      const auto finite = [](C x) { return psiformer::determinant::isFiniteReal(x.real()) &&
                                           psiformer::determinant::isFiniteReal(x.imag()); };
      for (const auto& point : points)
      {
        if (point.sample >= samples || point.electron >= lane->state->model.ne ||
            !finite(point.bare_weight) || point.bare_weight.imag() != 0 || !finite(point.complete_ratio) ||
            !finite(point.bare_weight * point.complete_ratio))
          throw std::invalid_argument("Complex replay point/complete coefficient is invalid");
        for (const auto x : point.replacement_position)
          if (!psiformer::determinant::isFiniteReal(x))
            throw std::invalid_argument("Complex replay replacement coordinate is nonfinite");
      }
      std::size_t covered_points = 0, covered_segments = 0, covered_stamps = 0;
      for (const auto& segment : incoming->segments())
      {
        if (segment.sample >= samples || segment.electron >= lane->state->model.ne ||
            segment.point_offset != covered_points || covered_points > points.size() ||
            segment.point_count > points.size() - covered_points)
          throw std::invalid_argument("Complex replay segment traversal is not exact");
        for (std::size_t q = 0; q < segment.point_count; ++q)
          if (points[covered_points + q].sample != segment.sample || points[covered_points + q].electron != segment.electron)
            throw std::invalid_argument("Complex replay segment/point reference mismatch");
        covered_points += segment.point_count;
      }
      if (covered_points != points.size()) throw std::invalid_argument("Complex replay segment coverage is incomplete");
      covered_points = 0;
      std::size_t tile_ordinal = 0;
      for (const auto& tile : incoming->tiles())
      {
        const auto stamps = incoming->evaluationStamps();
        if (tile.tile_ordinal != tile_ordinal++ || !tile.virtual_batch_fingerprint ||
            tile.point_offset != covered_points || tile.segment_offset != covered_segments ||
            covered_points > points.size() || tile.point_count > points.size() - covered_points ||
            covered_segments > incoming->segments().size() || tile.segment_count > incoming->segments().size() - covered_segments ||
            tile.stamp_offset != covered_stamps || covered_stamps > stamps.size() ||
            tile.stamp_count > stamps.size() - covered_stamps)
          throw std::invalid_argument("Complex replay tile traversal/provenance range mismatch");
        bool matching = false;
        for (std::size_t s = 0; s < tile.stamp_count; ++s)
          matching |= stamps[tile.stamp_offset + s].matches(lane->state.get(), version);
        if (!matching) throw std::invalid_argument("Complex replay has no exact paired VALUE evaluation stamp");
        std::size_t tile_points = 0;
        for (std::size_t s = 0; s < tile.segment_count; ++s)
          pf::addStorageBytes(tile_points, incoming->segments()[covered_segments + s].point_count, message);
        if (tile_points != tile.point_count)
          throw std::invalid_argument("Complex replay tile/segment point counts disagree");
        covered_points += tile.point_count; covered_segments += tile.segment_count; covered_stamps += tile.stamp_count;
      }
      if (covered_points != points.size() || covered_segments != incoming->segments().size() ||
          covered_stamps != incoming->evaluationStamps().size())
        throw std::invalid_argument("Complex replay tile coverage is incomplete");
      app_log() << "Complex replay before allocation: HOST_additional_bytes=" << extra
                << ", bases=" << samples << ", points=" << points.size() << "; DEVICE_additional_bytes=0; C1_only=yes" << std::endl;
      std::vector<std::size_t> refs(points.size());
      std::vector<C> weights(points.size()), coefficients(samples);
      for (std::size_t q = 0; q < points.size(); ++q)
      {
        const auto& point = points[q]; refs[q] = point.sample; weights[q] = point.bare_weight * point.complete_ratio;
      }
      std::size_t actual = diagnostics.retained_numeric_bytes;
      for (const auto* storage : {&weights, &coefficients})
        pf::addStorageBytes(actual, pf::checkedStorageBytes<C>(storage->capacity(), message), message);
      pf::addStorageBytes(actual, pf::checkedStorageBytes<std::size_t>(refs.capacity(), message), message);
      if (actual > host_cap || current_host > host_cap - actual)
        throw std::length_error("Complex replay concrete HOST capacities exceed factory admission");
      references.swap(refs);
      complete_weights.swap(weights); sample_coefficients.swap(coefficients); replay = std::move(incoming);
      replay_bound.store(true, std::memory_order_release);
    }

    /// Preserve exact capacity diagnostics after cold replay adoption, with no parameter-sized storage.
    std::size_t replayHostBytes() const
    {
      std::size_t bytes = 0;
      const char* message = "Complex replay HOST capacity overflow";
      for (const auto* storage : std::array<const std::vector<C>*, 2>{&complete_weights, &sample_coefficients})
        pf::addStorageBytes(bytes, pf::checkedStorageBytes<C>(storage->capacity(), message), message);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<std::size_t>(references.capacity(), message), message);
      return bytes;
    }

    /// Borrow immutable move metadata only; this accessor never locks or calls the model.
    auto replacementAccessor() const
    {
      return [this](std::size_t q) {
        const auto& point = replay->points()[q];
        return psiformer::device::ComplexVirtualReplacement{point.electron, point.replacement_position};
      };
    }

    /// Every action/binding starts under the same execution lease and exact Model read fence.
    void requireCurrent() const
    {
      if (version != lane->state->model.p.version() || lane->generation != lane->state->gpu ||
          lane->state->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex GPU derivative snapshot is stale or family has a pending move");
      validateDerivativeSource(*lane->state);
    }

    /// Contract the GPU canonical score with one REAL direction; only O(B) scalar products return.
    void evaluateScoreJVP(const wftrain::StructuredParameterVectorConstView& direction,
        wftrain::SampleProductSink& sink) const override
    {
      NativeTimerScope timer(phase_timers.score_jvp);
      std::unique_lock execution(lane->execution_mutex);
      has_evaluated = true;
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

    /// Map only the three implemented complete-response masks at the execution fence.
    static GpuDerivativeLane::Products::LocalTerms localTerms(std::uint32_t mask)
    {
      using namespace wftrain;
      using Terms = GpuDerivativeLane::Products::LocalTerms;
      if (mask == localEnergyTermBit(LocalEnergyTerm::KINETIC)) return Terms::KINETIC;
      if (mask == localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)) return Terms::NONLOCAL_ECP;
      if (mask == (localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)))
        return Terms::COMPLETE;
      throw std::invalid_argument("Complex captured response has unsupported local-energy terms");
    }

    /// The same immutable captured grid supplies complete kinetic/ECP directional responses.
    void evaluateLocalEnergyJVP(const wftrain::StructuredParameterVectorConstView& direction,
        std::uint32_t mask, wftrain::SampleProductSink& sink) const override
    {
      std::unique_lock execution(lane->execution_mutex);
      has_evaluated = true;
      const C* output;
      {
        std::shared_lock read(lane->state->mutex);
        requireCurrent();
        if (!replay) throw std::invalid_argument("Complex local-energy JVP requires bound replay");
        const auto terms = localTerms(mask);
        std::unique_lock gpu(lane->generation->mutex);
        lane->products->evaluateLocalEnergyJVPReplacements(positions, replacementAccessor(), references, complete_weights,
            inverse_masses, lane->state->cartesian_twist, terms, direction.values().data(), direction.values().size());
        const auto result = lane->products->localEnergyJvpView(); lane->products->validate(result); output = result.data;
      }
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
      has_evaluated = true;
      GpuDerivativeLane::Products::CompletedResult result;
      {
        std::shared_lock read(lane->state->mutex);
        std::size_t energy_channel = requested.size();
        if (replay)
        {
          if (requested.empty() || requested.size() > lane->channels.size())
            throw std::invalid_argument("Complex bound replay channel extent invalid");
          for (std::size_t c = 0; c < requested.size(); ++c)
            if (requested[c].product == DerivativeProduct::LOCAL_ENERGY_VJP)
            {
              if (energy_channel != requested.size())
                throw std::invalid_argument("Complex bound replay admits only one complete energy channel");
              localTerms(requested[c].local_energy_term_mask);
              energy_channel = c;
            }
            else if (requested[c].product != DerivativeProduct::SCORE_VJP)
              throw std::invalid_argument("Complex bound replay requires energy/SCORE channels");
        }
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
          if (replay && energy_channel != requested.size())
          {
            std::copy_n(requested[energy_channel].coefficients.values.data(), samples, sample_coefficients.data());
            lane->products->evaluateLocalEnergyVJPReplacements(positions, replacementAccessor(), references, complete_weights,
                sample_coefficients, inverse_masses, lane->state->cartesian_twist,
                localTerms(requested[energy_channel].local_energy_term_mask), adjoint == DerivativeAdjoint::HERMITIAN,
                requested.size() > 1 ? &lane->channels : nullptr, energy_channel);
          }
          else
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
    const std::size_t version, ordinal, offset, samples, host_cap;
    const wftrain::ParameterChunkPlan chunks;
    std::vector<double> positions, inverse_masses;
    std::shared_ptr<const wftrain::NonLocalECPJVPReplay> replay;
    std::atomic<bool> replay_bound{false};
    std::vector<std::size_t> references;
    std::vector<C> complete_weights;
    mutable std::vector<C> sample_coefficients;
    mutable bool has_evaluated = false;
  };

  /// One immutable O(B+Q) record; keeping the family alive also protects stamp identity reuse.
  struct GpuNonLocalReplay final : wftrain::NonLocalECPJVPReplay
  {
    const std::shared_ptr<Model> state;
    const std::shared_ptr<const std::size_t> capture_identity = std::make_shared<const std::size_t>(0);
    const std::size_t version, ordinal, offset, samples;
    std::uint64_t grid = 0;
    std::vector<double> reference_positions;
    std::vector<wftrain::NonLocalECPReplayPoint> point_records;
    std::vector<wftrain::NonLocalECPReplaySegment> segment_records;
    std::vector<wftrain::NonLocalECPReplayTile> tile_records;
    std::vector<WaveFunctionComponent::EvaluationStamp> stamp_records;
    wftrain::NonLocalECPReplayStorageDiagnostics admitted_storage;

    /// Pin the captured family and copy its accepted reference coordinates.
    explicit GpuNonLocalReplay(const GpuDerivativeStream& s)
        : state(s.lane->state), version(s.version), ordinal(s.ordinal), offset(s.offset), samples(s.samples),
          reference_positions(s.positions) {}

    /// Identify the structured provider that owns this immutable replay.
    std::string_view providerId() const noexcept override { return state->schema->providerId(); }
    /// Reject binding against a different canonical parameter layout.
    std::string_view schemaFingerprint() const noexcept override { return state->schema->fingerprint(); }
    /// Report the captured accepted parameter generation.
    std::size_t parameterVersion() const noexcept override { return version; }
    /// Preserve the producer's batch identity.
    std::size_t batchOrdinal() const noexcept override { return ordinal; }
    /// Preserve the producer's global sample origin.
    std::size_t sampleOffset() const noexcept override { return offset; }
    /// Report the accepted reference population.
    std::size_t sampleCount() const noexcept override { return samples; }
    /// Report the coordinate stride's electron extent.
    std::size_t electronCount() const noexcept override { return state->model.ne; }
    /// Advertise only the nonlocal response captured by this record.
    std::uint32_t localEnergyTermMask() const noexcept override
    { return wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP); }
    /// Identify the actual accepted quadrature traversal.
    std::uint64_t gridFingerprint() const noexcept override { return grid; }
    /// Borrow the immutable accepted base coordinates.
    wftrain::DerivativeArrayView<const double> referencePositions() const noexcept override
    { return {reference_positions.data(), reference_positions.size()}; }
    /// Borrow captured weighted virtual moves in traversal order.
    wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayPoint> points() const noexcept override
    { return {point_records.data(), point_records.size()}; }
    /// Borrow the captured reference-to-point mapping.
    wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplaySegment> segments() const noexcept override
    { return {segment_records.data(), segment_records.size()}; }
    /// Borrow complete traversal tile and stamp ranges.
    wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayTile> tiles() const noexcept override
    { return {tile_records.data(), tile_records.size()}; }
    /// Borrow actual versioned VALUE evaluation provenance.
    wftrain::DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp> evaluationStamps() const noexcept override
    { return {stamp_records.data(), stamp_records.size()}; }

    /// Count capacities rather than filled extents, including worst-case provenance reserved before RNG.
    wftrain::NonLocalECPReplayStorageDiagnostics measureStorageDiagnostics() const
    {
      const char* error = "Complex ECP replay storage overflow";
      wftrain::NonLocalECPReplayStorageDiagnostics d;
      d.point_count = point_records.size(); d.segment_count = segment_records.size();
      d.tile_count = tile_records.size(); d.evaluation_stamp_count = stamp_records.size();
      d.reference_position_bytes = pf::checkedStorageBytes<double>(reference_positions.capacity(), error);
      d.point_bytes = pf::checkedStorageBytes<wftrain::NonLocalECPReplayPoint>(point_records.capacity(), error);
      d.provenance_bytes = pf::checkedStorageBytes<wftrain::NonLocalECPReplaySegment>(segment_records.capacity(), error);
      pf::addStorageBytes(d.provenance_bytes, sizeof(std::size_t), error);
      pf::addStorageBytes(d.provenance_bytes,
          pf::checkedStorageBytes<wftrain::NonLocalECPReplayTile>(tile_records.capacity(), error), error);
      pf::addStorageBytes(d.provenance_bytes,
          pf::checkedStorageBytes<WaveFunctionComponent::EvaluationStamp>(stamp_records.capacity(), error), error);
      d.retained_numeric_bytes = d.reference_position_bytes;
      pf::addStorageBytes(d.retained_numeric_bytes, d.point_bytes, error);
      pf::addStorageBytes(d.retained_numeric_bytes, d.provenance_bytes, error);
      d.storage_fingerprint = 1469598103934665603ULL;
      for (const void* pointer : {static_cast<const void*>(reference_positions.data()),
          static_cast<const void*>(point_records.data()), static_cast<const void*>(segment_records.data()),
          static_cast<const void*>(tile_records.data()), static_cast<const void*>(stamp_records.data())})
        d.storage_fingerprint = (d.storage_fingerprint ^ reinterpret_cast<std::uintptr_t>(pointer)) * 1099511628211ULL;
      return d;
    }

    /// Capacities never grow after admission; the interface's noexcept query only updates filled extents.
    wftrain::NonLocalECPReplayStorageDiagnostics storageDiagnostics() const noexcept override
    {
      auto d = admitted_storage;
      d.point_count = point_records.size(); d.segment_count = segment_records.size();
      d.tile_count = tile_records.size(); d.evaluation_stamp_count = stamp_records.size();
      return d;
    }
  };

  /// Replay-only ordinary-locality capture; caller keeps its ParticleSets alive through end/abort.
  struct GpuNonLocalCapture final : wftrain::NonLocalECPDerivativeConsumer
  {
    const std::shared_ptr<GpuDerivativeLane> lane;
    std::unique_lock<std::mutex> execution;
    std::unique_ptr<GpuNonLocalReplay> record;
    std::vector<const ParticleSet*> particles;
    std::size_t expected = 0;
    enum class State { IDLE, ACTIVE, COMPLETE, POISONED } state = State::IDLE;

    /// One exact shape core serves both pre-MC metadata admission and the concrete operator factory.
    static GpuNonLocalReplayRequirements requiredShape(std::size_t samples, std::size_t electrons, std::size_t q)
    {
      const char* error = "Complex ECP capture admission overflow";
      if (!samples || !electrons) throw std::invalid_argument("Complex ECP capture shape is empty");
      const auto stride = pf::checkedStorageProduct(3, electrons, error);
      std::size_t bytes = pf::checkedStorageBytes<double>(pf::checkedStorageProduct(samples, stride, error), error);
      pf::addStorageBytes(bytes, sizeof(std::size_t), error);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<const ParticleSet*>(samples, error), error);
      for (auto size : {sizeof(wftrain::NonLocalECPReplayPoint), sizeof(wftrain::NonLocalECPReplaySegment),
          sizeof(wftrain::NonLocalECPReplayTile), sizeof(WaveFunctionComponent::EvaluationStamp)})
        pf::addStorageBytes(bytes, pf::checkedStorageProduct(q, size, error), error);
      std::size_t extra = bytes;
      pf::addStorageBytes(extra, pf::checkedStorageBytes<std::size_t>(q, error), error);
      pf::addStorageBytes(extra, pf::checkedStorageBytes<C>(pf::checkedStorageSum(samples, q, error), error), error);
      return {bytes, extra};
    }

    /// Admit capture AND its eventual bind packing, without allocating any storage.
    static std::size_t required(const GpuDerivativeStream& s)
    {
      const auto r = requiredShape(s.samples, s.lane->state->model.ne, s.lane->maximum_samples - s.samples);
      const auto current = s.storageDiagnostics().retained_numeric_bytes - s.lane->products->cost().device_bytes;
      if (r.bound_host_extra_bytes > s.host_cap || current > s.host_cap - r.bound_host_extra_bytes)
        throw std::length_error("Complex ECP capture plus eventual binding exceeds operator HOST admission");
      return r.capture_host_bytes;
    }

    /// Allocate only preadmitted capture capacities before the Hamiltonian traversal.
    GpuNonLocalCapture(const GpuDerivativeStream& s, const RefVectorWithLeader<ParticleSet>& targets,
        std::unique_lock<std::mutex>&& lease, std::size_t cap)
        : lane(s.lane), execution(std::move(lease)), record(std::make_unique<GpuNonLocalReplay>(s))
    {
      const auto q = lane->maximum_samples - s.samples;
      record->point_records.reserve(q); record->segment_records.reserve(q);
      record->tile_records.reserve(q); record->stamp_records.reserve(q);
      record->admitted_storage = record->measureStorageDiagnostics();
      particles.reserve(s.samples);
      for (auto& p : targets) particles.push_back(&p.get());
      if (storageDiagnostics().retained_numeric_bytes > cap)
        throw std::length_error("Complex ECP actual capture capacities exceed admission");
    }

    /// Share only a one-shot identity, never mutable replay storage, with the native traversal seal.
    std::shared_ptr<const void> nativeCaptureIdentity() const noexcept override
    { return record ? record->capture_identity : std::shared_ptr<const void>{}; }

    /// Recheck the accepted family and geometry before reading any live tile.
    void requireCurrent() const
    {
      std::shared_lock read(lane->state->mutex);
      if (!record || lane->state->model.p.version() != record->version || lane->state->gpu != lane->generation ||
          lane->state->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex ECP capture source is stale/pending");
      validateDerivativeSource(*lane->state);
      for (std::size_t b = 0; b < particles.size(); ++b)
      {
        const auto& p = *particles[b];
        if (p.getActivePtcl() >= 0 || p.R.size() != lane->state->model.ne)
          throw std::logic_error("Complex ECP capture reference is pending/invalid");
        for (std::size_t e = 0; e < p.R.size(); ++e)
          for (int d = 0; d < 3; ++d)
            if (p.R[e][d] != record->reference_positions[3 * (b * p.R.size() + e) + d])
              throw std::logic_error("Complex ECP capture reference coordinates changed");
      }
    }

    /// Reject unsupported or stale traversal metadata before consuming any grid data.
    void preflight(const wftrain::NonLocalECPDerivativeContext& c,
        wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
        const wftrain::ParameterReductionSink& sink) const override
    {
      if (state != State::IDLE || !record || channels.size() != 1 ||
          sink.state() != wftrain::DerivativeSinkState::IDLE || c.provider_id != record->providerId() ||
          c.schema_fingerprint != record->schemaFingerprint() || c.parameter_version != record->version ||
          c.batch_ordinal != record->ordinal || c.sample_offset != record->offset || c.sample_count != record->samples ||
          c.localization != wftrain::NonLocalECPLocalization::ORDINARY_LOCALITY || !c.uses_virtual_particles ||
          !c.scalar_relativistic || c.used_dense_derivative_fallback)
        throw std::invalid_argument("Complex ECP capture requires current scalar ordinary-locality replay-only C1");
      const auto& channel = channels[0]; const auto& v = channel.coefficients;
      if (channel.id.empty() || channel.product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
          channel.local_energy_term_mask != record->localEnergyTermMask() || v.provider_id != c.provider_id ||
          v.schema_fingerprint != c.schema_fingerprint || v.parameter_version != c.parameter_version ||
          v.batch_ordinal != c.batch_ordinal || v.sample_offset != c.sample_offset ||
          v.values.size() != c.sample_count || !v.values.data())
        throw std::invalid_argument("Complex ECP capture coefficient identity mismatch");
      for (auto value : v.values)
        if (!isFiniteWavefunctionValue(value)) throw std::invalid_argument("Complex ECP capture nonfinite coefficient");
      requireCurrent();
    }

    /// Admit the exact grid extent and enter active capture without allocating.
    void begin(const wftrain::NonLocalECPDerivativeContext& c,
        wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
        wftrain::ParameterReductionSink& sink) override
    {
      preflight(c, channels, sink);
      if (!c.grid_fingerprint || c.expected_point_count > record->point_records.capacity())
        throw std::length_error("Complex ECP traversal exceeds prepared capture capacity");
      expected = c.expected_point_count; record->grid = c.grid_fingerprint; state = State::ACTIVE;
    }

    /// Copy transient tile data only after its exact source stamp and bounds are admitted.
    void consume(const wftrain::NonLocalECPDerivativeTile& tile) override
    {
      try
      {
        requireCurrent();
        const auto& batch = tile.virtual_particles;
        if (state != State::ACTIVE || tile.grid_fingerprint != record->grid ||
            tile.tile_ordinal != record->tile_records.size() || batch.empty() || !batch.fingerprint() ||
            batch.walkerCount() != record->samples ||
            batch.size() > expected - record->point_records.size() || tile.bare_weights.size() != batch.size() ||
            tile.complete_ratios.size() != batch.size() || !tile.bare_weights.data() || !tile.complete_ratios.data() ||
            tile.evaluation_stamps.size() != 1 || !tile.evaluation_stamps.data() ||
            !tile.evaluation_stamps[0].matches(lane->state.get(), record->version))
          throw std::invalid_argument("Complex ECP tile lacks exact traversal/stamp coverage");
        const auto start = record->point_records.size(), segments = record->segment_records.size();
        for (std::size_t s = 0; s < batch.segmentCount(); ++s)
        {
          const auto slice = batch.slice(s);
          if (slice.empty() || slice.walkerId() < 0 || slice.electronId() < 0 ||
              std::size_t(slice.walkerId()) >= record->samples || std::size_t(slice.electronId()) >= record->electronCount())
            throw std::invalid_argument("Complex ECP tile reference is invalid");
          record->segment_records.push_back({start + slice.flatOffset(), slice.size(), std::size_t(slice.walkerId()),
              std::size_t(slice.electronId()), slice.isOnSphere(), std::size_t(slice.sourceCenterId())});
          for (std::size_t i = 0; i < slice.size(); ++i)
          {
            const auto q = slice.flatOffset() + i;
            const C bare = tile.bare_weights[q], ratio = tile.complete_ratios[q];
            const auto& p = slice.absolutePosition(i);
            if (bare.imag() != 0 || !isFiniteWavefunctionValue(bare) || !isFiniteWavefunctionValue(ratio) ||
                !isFiniteWavefunctionValue(bare * ratio))
              throw std::invalid_argument("Complex ECP tile weight/ratio is invalid");
            for (int d = 0; d < 3; ++d)
              if (!psiformer::determinant::isFiniteReal(p[d])) throw std::invalid_argument("Complex ECP point nonfinite");
            record->point_records.push_back({std::size_t(slice.walkerId()), std::size_t(slice.electronId()),
                {p[0], p[1], p[2]}, bare, ratio});
          }
        }
        record->tile_records.push_back({tile.tile_ordinal, start, batch.size(), segments,
            record->segment_records.size() - segments, record->stamp_records.size(), 1, batch.fingerprint()});
        record->stamp_records.push_back(tile.evaluation_stamps[0]);
      }
      catch (...) { abort(); throw; }
    }

    /// Only a complete current record can escape; release the execution lease before binding it.
    void end() override
    {
      try
      {
        requireCurrent();
        if (state != State::ACTIVE || record->point_records.size() != expected)
          throw std::logic_error("Complex ECP capture ended with incomplete coverage");
        for (std::size_t sample = 0; sample < record->samples; ++sample)
        {
          C sum{};
          for (const auto& point : record->point_records)
            if (point.sample == sample) sum += point.bare_weight * point.complete_ratio;
          if (!isFiniteWavefunctionValue(sum)) throw std::overflow_error("Complex captured ECP sum overflow");
        }
        state = State::COMPLETE; execution.unlock();
      }
      catch (...) { abort(); throw; }
    }

    /// Poison incomplete capture and release its execution lease.
    void abort() noexcept override
    { state = State::POISONED; if (execution.owns_lock()) execution.unlock(); }

    /// Transfer a completed immutable record exactly once.
    std::unique_ptr<wftrain::NonLocalECPJVPReplay> releaseJVPReplay() override
    {
      if (state != State::COMPLETE || !record) throw std::logic_error("Complex ECP replay is incomplete/already released");
      return std::move(record);
    }

    /// Report retained record and borrowed-target pointer capacities honestly.
    wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
    {
      wftrain::StreamingDerivativeStorageDiagnostics d;
      const auto pointer_bytes = pf::checkedStorageBytes<const ParticleSet*>(particles.capacity(),
          "Complex ECP capture pointer storage overflow");
      d.sample_auxiliary_bytes = d.retained_numeric_bytes = pointer_bytes;
      d.storage_fingerprint = reinterpret_cast<std::uintptr_t>(particles.data());
      d.allocation_generation = 1;
      if (!record) return d;
      const auto r = record->storageDiagnostics();
      d.parameter_count = lane->state->schema->parameterCount(); d.sample_count = record->samples;
      d.sample_position_bytes = r.reference_position_bytes;
      pf::addStorageBytes(d.sample_auxiliary_bytes, r.point_bytes, "Complex ECP capture storage overflow");
      pf::addStorageBytes(d.sample_auxiliary_bytes, r.provenance_bytes, "Complex ECP capture storage overflow");
      d.retained_numeric_bytes = r.retained_numeric_bytes;
      pf::addStorageBytes(d.retained_numeric_bytes, pointer_bytes, "Complex ECP capture storage overflow");
      d.storage_fingerprint = r.storage_fingerprint ^ reinterpret_cast<std::uintptr_t>(particles.data());
      d.allocation_generation = 1; return d;
    }
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
          s.model.blocks, s.model.p.size(), bool(s.environment.periodic_profile),
          psiformer::occupiedMomentumCount(s.plan), s.environment.periodic_profile ?
              s.environment.periodic_profile->pair_feature_dimension : 8, psiformer::fourierModeCount(s.plan)};
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
      pf::addStorageBytes(doubles, pf::checkedStorageProduct(6, dimensions.occupied_momentum_count, message), message);
      pf::addStorageBytes(doubles, pf::checkedStorageProduct(6, dimensions.fourier_mode_count, message), message);
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
      if (psiformer::memory::enabled())
      {
        try { trace_capacity.snapshotAt(__FILE__, __LINE__, "ComplexDerivative::shared_host_lane", bytes(), reinterpret_cast<std::uintptr_t>(this)); }
        catch (...) { psiformer::memory::emit({"snapshot_failed", 0, "ComplexDerivative::shared_host_lane"}); }
      }
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
    psiformer::memory::HostCapacity trace_capacity;
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
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<std::array<double, 3>>(
          pf::checkedStorageSum(psiformer::occupiedMomentumCount(s.plan), psiformer::fourierModeCount(s.plan),
              "orbital mode extent overflow"), "orbital momentum metadata overflow"), "orbital owner overflow");
      return bytes;
    }
    explicit OrbitalMSELane(const Model& s) : score(s.model, s.plan), work(score.makeWorkspace()) {}
    std::size_t bytes() const { return work->vectorStorageBytes() + score.numericMetadataStorageBytes(); }
    const pf::DirectComplexScoreExecutor score;
    std::unique_ptr<pf::DirectComplexScoreWorkspace> work;
    std::mutex execution_mutex;
  };

#ifdef ENABLE_CUDA
  /// Separately capped raw-orbital observation/VJP scratch, never an accepted inference cache.
  struct GpuOrbitalMSELane
  {
    static std::size_t required(const Model& s)
    {
      const auto cost = psiformer::device::estimateComplexOrbitalVJPCost(s.plan);
      auto bytes = cost.device_bytes;
      pf::addStorageBytes(bytes, cost.host_bytes, "GPU orbital owner overflow");
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(pf::checkedStorageProduct(3, s.model.ne,
          "orbital coordinates overflow"), "orbital coordinates overflow"),
          "GPU orbital owner overflow");
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<cuDoubleComplex>(cost.forward.matrices,
          "orbital matrices overflow"), "GPU orbital owner overflow");
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(s.plan.parameterCount(),
          "orbital response overflow"), "GPU orbital owner overflow");
      return bytes;
    }
    explicit GpuOrbitalMSELane(const Model& s)
        : blas(queue), image(s.gpu->image),
          owner(psiformer::device::PreparedDeviceComplexScore::prepareOrbitalVJP(image, queue, blas,
              psiformer::device::estimateComplexOrbitalVJPCost(s.plan).device_bytes)),
          positions(3 * s.model.ne), raw(owner->cost().forward.matrices), response(s.plan.parameterCount()),
          bytes(required(s)) {}
    void forward(const Model& s, const ParticleSet& p)
    {
      if (image != s.gpu->image)
      {
        owner->rebindModel(s.gpu->image);
        image = s.gpu->image;
      }
      if (image->version() != s.model.p.version())
        throw std::logic_error("Complex orbital image/native version mismatch");
      for (std::size_t row = 0; row < s.model.ne; ++row)
        for (int d = 0; d < 3; ++d) positions[3 * row + d] = p.R[row][d];
      owner->evaluateOrbitals(positions);
      owner->downloadOrbitals(owner->orbitalView(), raw);
    }
    std::size_t fingerprint() const noexcept
    {
      auto value = reinterpret_cast<std::uintptr_t>(owner.get()) ^ bytes;
      psiformer::determinant::detail::mixStorage(value, positions);
      psiformer::determinant::detail::mixStorage(value, raw);
      psiformer::determinant::detail::mixStorage(value, response);
      return value;
    }
    compute::Queue<PlatformKind::CUDA> queue;
    psiformer::device::AcceleratorBlasHandle blas;
    std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
    std::unique_ptr<psiformer::device::PreparedDeviceComplexScore> owner;
    std::vector<double> positions;
    std::vector<cuDoubleComplex> raw;
    std::vector<double> response;
    const std::size_t bytes;
    std::mutex execution_mutex;
  };
#endif

  /// Resolve each actual real affine tensor, including the two independent head leaves.
  static std::vector<wftrain::AffineParameterBlock> affineBlocks(const Model& s)
  {
    using Role = psiformer::ParameterRole;
    const auto schema_block = [&](const psiformer::ParameterTensorDescriptor& tensor) {
      const auto& blocks = s.schema->blocks();
      const auto found = std::find_if(blocks.begin(), blocks.end(), [&](const auto& block) {
        return block.id == tensor.module + "/" + tensor.name && block.offset == tensor.begin &&
            block.count == tensor.size() && block.shape == tensor.shape &&
            block.scalar_domain == wftrain::ParameterScalarDomain::REAL64;
      });
      if (found == blocks.end())
        throw std::logic_error("Complex affine tensor differs from the canonical schema");
      return static_cast<std::size_t>(found - blocks.begin());
    };
    std::vector<wftrain::AffineParameterBlock> result;
    for (const auto& tensor : s.plan.parameterTensors())
    {
      switch (tensor.role)
      {
      case Role::ELECTRON_EMBEDDING_WEIGHT:
      case Role::ATTENTION_QUERY_WEIGHT:
      case Role::ATTENTION_KEY_WEIGHT:
      case Role::ATTENTION_VALUE_WEIGHT:
      case Role::ATTENTION_OUTPUT_WEIGHT:
      case Role::UPDATE_HIDDEN_WEIGHT:
      case Role::UPDATE_OUTPUT_WEIGHT:
      case Role::PAIR_ATTENTION_WEIGHT:
      case Role::BACKFLOW_UP_WEIGHT:
      case Role::BACKFLOW_DOWN_WEIGHT: break;
      default: continue;
      }
      const auto weight = schema_block(tensor);
      auto bias = wftrain::NO_AFFINE_BIAS_BLOCK;
      if (tensor.role == Role::UPDATE_HIDDEN_WEIGHT || tensor.role == Role::UPDATE_OUTPUT_WEIGHT ||
          tensor.role == Role::PAIR_ATTENTION_WEIGHT)
      {
        const auto bias_role = tensor.role == Role::UPDATE_HIDDEN_WEIGHT
            ? Role::UPDATE_HIDDEN_BIAS : tensor.role == Role::UPDATE_OUTPUT_WEIGHT
            ? Role::UPDATE_OUTPUT_BIAS : Role::PAIR_ATTENTION_BIAS;
        bias = schema_block(s.plan.parameter(bias_role, tensor.attention_block));
        if (s.schema->blocks()[bias].trainable != s.schema->blocks()[weight].trainable)
          throw std::logic_error("Complex affine weight and bias trainability differ");
      }
      if (tensor.shape.size() != 2)
        throw std::logic_error("Complex affine weight must be a matrix");
      if (s.schema->blocks()[weight].trainable)
        result.push_back({s.schema->blocks()[weight].id, weight, bias, tensor.shape[0], tensor.shape[1]});
    }
    return result;
  }

#ifdef ENABLE_CUDA
  /// Bind the trainer's exact ordered registry, including independent imaginary heads.
  static auto deviceAffineDescriptors(const Model& s, const wftrain::KFACBlockRegistry& registry)
  {
    const auto canonical = affineBlocks(s);
    if (registry.parameterSchema().fingerprint() != s.schema->fingerprint() ||
        registry.affineBlocks().size() != canonical.size())
      throw std::invalid_argument("Complex DEVICE affine registry differs from the canonical model");
    std::vector<psiformer::device::DeviceAffineFactorDescriptor> result;
    result.reserve(canonical.size());
    for (std::size_t i = 0; i < canonical.size(); ++i)
    {
      const auto& block = registry.affineBlocks()[i];
      const auto& expected = canonical[i];
      if (block.id != expected.id || block.weight_block != expected.weight_block ||
          block.bias_block != expected.bias_block || block.input_width != expected.input_width ||
          block.output_width != expected.output_width)
        throw std::invalid_argument("Complex DEVICE affine block order or binding differs");
      const auto& weight = s.schema->blocks().at(block.weight_block);
      const auto& tensors = s.plan.parameterTensors();
      const auto tensor = std::find_if(tensors.begin(), tensors.end(), [&](const auto& candidate) {
        return candidate.begin == weight.offset && candidate.size() == weight.count &&
            candidate.module + "/" + candidate.name == weight.id && candidate.shape == weight.shape;
      });
      if (tensor == tensors.end())
        throw std::invalid_argument("Complex DEVICE affine tensor identity differs");
      result.push_back({tensor->role, tensor->attention_block, block.input_width, block.output_width,
          block.bias_block != wftrain::NO_KFAC_BIAS_BLOCK, tensor->head_component});
    }
    return result;
  }

  /// Exact DEVICE banks and HOST packed/import outputs, fallback channels and coordinates.
  static DeviceAffineStatisticsRequirements deviceAffineRequirements(const Model& s,
      const std::vector<psiformer::device::DeviceAffineFactorDescriptor>& blocks)
  {
    using Owner = psiformer::device::PreparedDeviceAffineStatistics;
    const char* error = "Complex DEVICE affine bytes overflow";
    auto host = Owner::requiredHostBytes(blocks);
    for (const auto& block : blocks)
    {
      const auto input = pf::checkedStorageSum(block.input_width, block.bias_augmented, error);
      const auto elements = pf::checkedStorageSum(pf::checkedStorageProduct(input, input, error),
          pf::checkedStorageProduct(block.output_width, block.output_width, error), error);
      pf::addStorageBytes(host, pf::checkedStorageBytes<double>(elements, error), error);
    }
    const auto host_elements = pf::checkedStorageSum(pf::checkedStorageProduct(2, s.schema->parameterCount(), error),
        pf::checkedStorageProduct(3, s.model.ne, error), error);
    pf::addStorageBytes(host, pf::checkedStorageBytes<double>(host_elements, error), error);
    return {Owner::requiredDeviceBytes(blocks), host};
  }
#endif

  /// One sample's channel pairing scratch; neither sample count nor parameter count is an axis.
  struct AffineLane
  {
    /// Bind a typed model tensor to its provider ordinal and first-channel scratch interval.
    struct Binding
    {
      psiformer::ParameterRole role;
      std::size_t attention_block, ordinal, response_offset;
      psiformer::HeadTensorComponent component;
    };

    /// Count one saved real sensitivity array per trunk affine boundary.
    static std::size_t responseElements(const Model& s)
    {
      const char* error = "Complex affine response extent overflow";
      const auto layers = pf::checkedStorageSum(1, pf::checkedStorageProduct(6, s.model.blocks, error), error);
      auto elements = pf::checkedStorageProduct(layers,
          pf::checkedStorageProduct(s.model.ne, s.model.dim, error), error);
      if (s.environment.periodic_profile)
        pf::addStorageBytes(elements, pf::checkedStorageProduct(s.model.blocks,
            pf::checkedStorageProduct(s.model.ne, pf::checkedStorageProduct(s.model.ne - 1,
                s.model.heads, error), error), error), error);
      return elements;
    }

    /// Charge the score owner and all channel-pairing buffers before numerical allocation.
    static std::size_t required(const Model& s)
    {
      const char* error = "Complex affine owner bytes overflow";
      auto elements = responseElements(s);
      pf::addStorageBytes(elements, pf::checkedStorageProduct(2,
          pf::checkedStorageProduct(s.model.ndet, pf::checkedStorageProduct(s.model.ne, s.model.ne, error), error),
          error), error);
      auto bytes = OrbitalMSELane::required(s);
      pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(elements, error), error);
      return bytes;
    }

    /// Prepare fixed-size buffers and resolve canonical affine bindings once per owner.
    explicit AffineLane(const Model& s)
        : score(s.model, s.plan), work(score.makeWorkspace()), blocks(affineBlocks(s)),
          real_responses(responseElements(s)),
          head_real(s.model.ndet * s.model.ne * s.model.ne), head_imaginary(head_real.size())
    {
      std::size_t offset = 0;
      for (std::size_t ordinal = 0; ordinal < blocks.size(); ++ordinal)
      {
        const auto& weight = s.schema->blocks()[blocks[ordinal].weight_block];
        const auto& tensors = s.plan.parameterTensors();
        const auto tensor = std::find_if(tensors.begin(), tensors.end(), [&](const auto& candidate) {
          return candidate.begin == weight.offset && candidate.size() == weight.count;
        });
        if (tensor == tensors.end())
          throw std::logic_error("Complex affine observation lacks a tensor binding");
        bindings.push_back({tensor->role, tensor->attention_block, ordinal, offset, tensor->head_component});
        if (tensor->role != psiformer::ParameterRole::BACKFLOW_UP_WEIGHT &&
            tensor->role != psiformer::ParameterRole::BACKFLOW_DOWN_WEIGHT)
          offset += (tensor->role == psiformer::ParameterRole::PAIR_ATTENTION_WEIGHT
              ? s.model.ne * (s.model.ne - 1) : s.model.ne) * blocks[ordinal].output_width;
      }
      if (offset > real_responses.size())
        throw std::logic_error("Complex affine response scratch is smaller than its bindings");
    }

    /// Report actual retained numeric capacities, including executor-owned ion metadata.
    std::size_t bytes() const
    {
      return work->vectorStorageBytes() + score.numericMetadataStorageBytes() + sizeof(double) *
          (real_responses.capacity() + head_real.capacity() + head_imaginary.capacity());
    }

    /// Pair the same typed boundary across the two existing real reverse passes.
    class ChannelSink final : public pf::DirectAffineObservationSink
    {
    public:
      /// Select which reverse channel this transient adapter consumes.
      ChannelSink(AffineLane& lane, wftrain::AffineScoreObserver& observer, bool imaginary)
          : lane_(lane), observer_(observer), imaginary_(imaginary) {}
      /// Save the amplitude response or pair it with the matching phase response.
      void observe(const pf::DirectAffineObservation& observation) override
      {
        const auto found = std::find_if(lane_.bindings.begin(), lane_.bindings.end(), [&](const auto& binding) {
          return binding.role == observation.role && binding.attention_block == observation.attention_block;
        });
        if (found == lane_.bindings.end())
          return;
        const auto& block = lane_.blocks[found->ordinal];
        if (observation.input_width != block.input_width || observation.output_width != block.output_width)
          throw std::logic_error("Complex affine observation widths differ from its registry");
        const auto count = observation.row_count * observation.output_width;
        if (found->response_offset > lane_.real_responses.size() ||
            count > lane_.real_responses.size() - found->response_offset)
          throw std::logic_error("Complex affine observation exceeds prepared response scratch");
        auto* real = lane_.real_responses.data() + found->response_offset;
        if (!imaginary_)
          std::copy_n(observation.sensitivities, count, real);
        else
          observer_.observe({found->ordinal, observation.activations, real, observation.row_count,
                              observation.sensitivities});
      }
    private:
      AffineLane& lane_;
      wftrain::AffineScoreObserver& observer_;
      bool imaginary_;
    };

    const pf::DirectComplexScoreExecutor score;
    std::unique_ptr<pf::DirectComplexScoreWorkspace> work;
    const std::vector<wftrain::AffineParameterBlock> blocks;
    std::vector<Binding> bindings;
    std::vector<double> real_responses, head_real, head_imaginary;
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
    {
      if (psiformer::memory::enabled())
      {
        const auto storage = storageDiagnostics();
        trace_capacity.snapshotAt(__FILE__, __LINE__, "ComplexDerivative::sample_positions_and_products",
            storage.sample_position_bytes + storage.sample_auxiliary_bytes + storage.sample_product_bytes,
            reinterpret_cast<std::uintptr_t>(lane.get()), ordinal, version);
      }
    }

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
    psiformer::memory::HostCapacity trace_capacity;
  };

  /// One native accepted-grid VJP; borrow the immutable accepted stream, never retain Q records.
  struct NonLocalVJPCapture final : wftrain::NonLocalECPDerivativeConsumer
  {
    enum class State { IDLE, ACTIVE, COMPLETE, POISONED };

    /// Admit only new fixed capture buffers; accepted coordinates and derivative scratch are borrowed.
    static std::size_t required(std::size_t samples, std::size_t electrons, bool gpu)
    {
      const char* message = "Complex inline ECP capture extent overflow";
      if (!samples || !electrons)
        throw std::invalid_argument("Complex inline ECP capture requires a nonempty accepted batch");
      auto bytes = pf::checkedStorageProduct(samples, 2 * sizeof(C) + sizeof(const ParticleSet*), message);
      if (!gpu)
        pf::addStorageBytes(bytes, pf::checkedStorageBytes<double>(pf::checkedStorageProduct(3, electrons, message), message), message);
      return bytes;
    }

    /// Reuse the accepted CPU stream's leased score tape and canonical contraction vector.
    NonLocalVJPCapture(const DerivativeStream& s, const RefVectorWithLeader<ParticleSet>& targets,
        std::unique_lock<std::mutex>&& lease)
        : NonLocalVJPCapture(s.state, s, s.positions, targets, std::move(lease), false)
    {
      cpu_lane = s.lane;
    }

#ifdef ENABLE_CUDA
    /// Reuse the accepted GPU stream's immutable image and device accumulator without a Q owner.
    NonLocalVJPCapture(const GpuDerivativeStream& s, const RefVectorWithLeader<ParticleSet>& targets,
        std::unique_lock<std::mutex>&& lease)
        : NonLocalVJPCapture(s.lane->state, s, s.positions, targets, std::move(lease), true)
    {
      gpu_lane = s.lane;
    }
#endif

    /// Allocate admitted B scalars/pointers and CPU coordinate scratch before any native grid draw.
    NonLocalVJPCapture(std::shared_ptr<Model> model, const wftrain::StreamingDerivativeOperator& source,
        const std::vector<double>& accepted_positions, const RefVectorWithLeader<ParticleSet>& targets,
        std::unique_lock<std::mutex>&& lease, bool gpu)
        : model(std::move(model)), source(source), positions(accepted_positions), execution(std::move(lease)),
          coefficients(source.sampleCount()), energies(source.sampleCount()),
          moved_positions(gpu ? 0 : 3 * this->model->model.ne)
    {
      particles.reserve(targets.size());
      for (auto& target : targets) particles.push_back(&target.get());
    }

    /// An abandoned live traversal cannot leave a publishable partial response.
    ~NonLocalVJPCapture() override
    {
      if (state == State::ACTIVE)
        abort();
    }

    /// Fence the exact accepted family, generation, coordinates and idle reference particles.
    void requireCurrent() const
    {
      std::shared_lock read(model->mutex);
      if (model->model.p.version() != source.parameterVersion() ||
          model->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex inline ECP source is stale or pending");
#ifdef ENABLE_CUDA
      if (gpu_lane && model->gpu != gpu_lane->generation)
        throw std::logic_error("Complex inline ECP device generation changed");
#endif
      validateDerivativeSource(*model);
      const auto n = model->model.ne;
      for (std::size_t b = 0; b < particles.size(); ++b)
      {
        const auto& p = *particles[b];
        if (p.getActivePtcl() >= 0 || p.R.size() != n)
          throw std::logic_error("Complex inline ECP reference is pending or invalid");
        for (std::size_t e = 0; e < n; ++e)
          for (int d = 0; d < 3; ++d)
            if (p.R[e][d] != positions[3 * (b * n + e) + d])
              throw std::logic_error("Complex inline ECP accepted reference changed");
      }
    }

    /// Reject foreign metadata or coefficients before the native Hamiltonian advances its RNG.
    void preflight(const wftrain::NonLocalECPDerivativeContext& c,
        wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
        const wftrain::ParameterReductionSink& sink) const override
    {
      using namespace wftrain;
      const auto& schema = source.parameterSchema();
      if (state != State::IDLE || channels.size() != 1 || sink.state() != DerivativeSinkState::IDLE ||
          c.provider_id != schema.providerId() || c.schema_fingerprint != schema.fingerprint() ||
          c.parameter_version != source.parameterVersion() || c.batch_ordinal != source.batchOrdinal() ||
          c.sample_offset != source.sampleOffset() || c.sample_count != source.sampleCount() ||
          c.localization != NonLocalECPLocalization::ORDINARY_LOCALITY || !c.uses_virtual_particles ||
          !c.scalar_relativistic || c.used_dense_derivative_fallback)
        throw std::invalid_argument("Complex inline ECP requires matching scalar ordinary-locality metadata");
      const auto& channel = channels[0];
      const auto& v = channel.coefficients;
      if (channel.id.empty() || channel.product != DerivativeProduct::LOCAL_ENERGY_VJP ||
          channel.local_energy_term_mask != localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP) ||
          v.provider_id != c.provider_id || v.schema_fingerprint != c.schema_fingerprint ||
          v.parameter_version != c.parameter_version || v.batch_ordinal != c.batch_ordinal ||
          v.sample_offset != c.sample_offset || v.values.size() != c.sample_count || !v.values.data())
        throw std::invalid_argument("Complex inline ECP coefficient identity differs");
      for (auto value : v.values)
        if (!isFiniteWavefunctionValue(value))
          throw std::invalid_argument("Complex inline ECP coefficient is nonfinite");
      requireCurrent();
    }

    /// Bind the actual accepted point count and activate one isolated transpose response channel.
    void begin(const wftrain::NonLocalECPDerivativeContext& c,
        wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
        wftrain::ParameterReductionSink& output) override
    {
      using namespace wftrain;
      preflight(c, channels, output);
      if (!c.grid_fingerprint)
        throw std::invalid_argument("Complex inline ECP requires accepted grid identity");
      expected = c.expected_point_count;
      grid = c.grid_fingerprint;
      std::copy(channels[0].coefficients.values.begin(), channels[0].coefficients.values.end(), coefficients.begin());
      descriptor = {c.provider_id, c.schema_fingerprint, c.parameter_version,
          derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP), DerivativeAdjoint::TRANSPOSE,
          ParameterScalarDomain::REAL64, ParameterScalarDomain::COMPLEX128, c.batch_ordinal, c.sample_offset,
          c.sample_count, ReductionDomain::CROWD_LOCAL, DerivativeExecutionDomain::HOST, 1,
          source.parameterChunkPlan().maximumChunkSize(), 1, localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
      sink = &output;
      try
      {
        sink->begin(descriptor, source.parameterChunkPlan(), channels);
#ifdef ENABLE_CUDA
        if (gpu_lane)
        {
          std::shared_lock read(model->mutex);
          std::unique_lock gpu(gpu_lane->generation->mutex);
          gpu_lane->products->beginStreamingNonlocalVJP(pf::checkedStorageSum(expected, source.sampleCount(),
              "Complex inline ECP score count overflow"));
        }
        else
#endif
          std::fill(cpu_lane->channels[0].begin(), cpu_lane->channels[0].end(), C{});
        state = State::ACTIVE;
      }
      catch (...) { abort(); throw; }
    }

    /// Callback inputs remain live only during this call; scores are contracted before returning.
    void consume(const wftrain::NonLocalECPDerivativeTile& tile) override
    {
      try
      {
        requireCurrent();
        const auto& batch = tile.virtual_particles;
        if (state != State::ACTIVE || tile.grid_fingerprint != grid || tile.tile_ordinal != tiles ||
            batch.empty() || !batch.fingerprint() || batch.walkerCount() != source.sampleCount() ||
            points > expected || batch.size() > expected - points ||
            tile.bare_weights.size() != batch.size() || tile.complete_ratios.size() != batch.size() ||
            !tile.bare_weights.data() || !tile.complete_ratios.data() || tile.evaluation_stamps.size() != 1 ||
            !tile.evaluation_stamps.data() || !tile.evaluation_stamps[0].matches(model.get(), source.parameterVersion()))
          throw std::invalid_argument("Complex inline ECP tile lacks exact grid, source or coverage");
        for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
        {
          const auto slice = batch.slice(segment);
          if (slice.empty() || slice.walkerId() < 0 || slice.electronId() < 0 ||
              std::size_t(slice.walkerId()) >= source.sampleCount() || std::size_t(slice.electronId()) >= model->model.ne)
            throw std::invalid_argument("Complex inline ECP segment reference is invalid");
          const auto sample = std::size_t(slice.walkerId());
          for (std::size_t local = 0; local < slice.size(); ++local)
          {
            const auto q = slice.flatOffset() + local;
            const C bare = tile.bare_weights[q], ratio = tile.complete_ratios[q];
            const auto weight = bare * ratio;
            if (bare.imag() != 0 || !isFiniteWavefunctionValue(bare) || !isFiniteWavefunctionValue(ratio) ||
                !isFiniteWavefunctionValue(weight) || !isFiniteWavefunctionValue(energies[sample] + weight))
              throw std::invalid_argument("Complex inline ECP complete weight is invalid");
            const auto& position = slice.absolutePosition(local);
            const std::array<double, 3> replacement{position[0], position[1], position[2]};
            fold(sample, std::size_t(slice.electronId()), &replacement, weight);
            energies[sample] += weight;
          }
        }
        points += batch.size();
        ++tiles;
      }
      catch (...) { abort(); throw; }
    }

    /// Add c*weight*O at one configuration, including zero-weight node validation.
    void fold(std::size_t sample, std::size_t electron, const std::array<double, 3>* replacement, C weight)
    {
      std::shared_lock read(model->mutex);
      if (model->model.p.version() != source.parameterVersion() || model->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex inline ECP source changed during score contraction");
#ifdef ENABLE_CUDA
      if (gpu_lane)
      {
        if (model->gpu != gpu_lane->generation)
          throw std::logic_error("Complex inline ECP image changed");
        std::unique_lock gpu(gpu_lane->generation->mutex);
        const psiformer::device::ComplexVirtualReplacement move{electron, replacement ? *replacement : std::array<double, 3>{}};
        gpu_lane->products->accumulateStreamingNonlocalScore(positions, sample, replacement ? &move : nullptr,
            coefficients[sample], weight);
        return;
      }
#endif
      const auto n = model->model.ne;
      std::copy_n(positions.data() + 3 * sample * n, 3 * n, moved_positions.data());
      if (replacement)
        std::copy(replacement->begin(), replacement->end(), moved_positions.begin() + 3 * electron);
      cpu_lane->score_work->setPositions(pf::GeometryPositionView::interleaved(moved_positions.data(), n));
      const auto score = cpu_lane->score.evaluate(*cpu_lane->score_work);
      const auto weighted = coefficients[sample] * weight;
      if (!isFiniteWavefunctionValue(weighted))
        throw std::overflow_error("Complex inline ECP weighted coefficient overflow");
      for (std::size_t parameter = 0; parameter < score.parameter_count; ++parameter)
        cpu_lane->channels[0][parameter] += weighted * score.parameter_score[parameter];
    }

    /// Subtract accepted scores and emit complete finite chunks outside model/device locks.
    void end() override
    {
      try
      {
        requireCurrent();
        if (state != State::ACTIVE || points != expected)
          throw std::logic_error("Complex inline ECP traversal is incomplete");
        for (std::size_t sample = 0; sample < source.sampleCount(); ++sample)
          fold(sample, 0, nullptr, -energies[sample]);
#ifdef ENABLE_CUDA
        if (gpu_lane)
        {
          psiformer::device::PreparedDeviceComplexProducts::CompletedResult receipt;
          {
            std::shared_lock read(model->mutex);
            if (model->model.p.version() != source.parameterVersion() || model->gpu != gpu_lane->generation ||
                model->pending_proposals.load(std::memory_order_relaxed))
              throw std::logic_error("Complex inline ECP source changed before publication");
            std::unique_lock gpu(gpu_lane->generation->mutex);
            gpu_lane->products->finishStreamingNonlocalVJP();
            receipt = gpu_lane->products->completedResult();
          }
          for (const auto& chunk : source.parameterChunkPlan().chunks())
          {
            const C* output;
            {
              std::unique_lock gpu(gpu_lane->generation->mutex);
              output = gpu_lane->products->downloadCompletedChunk(receipt, 0, chunk.parameter_offset, chunk.count);
            }
            sink->add(0, {chunk, {output, chunk.count}});
          }
        }
        else
#endif
        {
          for (auto value : cpu_lane->channels[0])
            if (!isFiniteWavefunctionValue(value))
              throw std::overflow_error("Complex inline ECP contraction is nonfinite");
          for (const auto& chunk : source.parameterChunkPlan().chunks())
            sink->add(0, {chunk, {cpu_lane->channels[0].data() + chunk.parameter_offset, chunk.count}});
        }
        sink->end();
        state = State::COMPLETE;
        execution.unlock();
      }
      catch (...) { abort(); throw; }
    }

    /// Poison the sink and release execution authority without exposing partial scalar/vector data.
    void abort() noexcept override
    {
#ifdef ENABLE_CUDA
      if (gpu_lane && execution.owns_lock())
        gpu_lane->products->abortStreamingNonlocalVJP();
#endif
      if (sink)
        sink->abort();
      state = State::POISONED;
      if (execution.owns_lock())
        execution.unlock();
    }

    /// Fixed-weight inline capture deliberately cannot serve future arbitrary-weight or JVP requests.
    std::unique_ptr<wftrain::NonLocalECPJVPReplay> releaseJVPReplay() override
    {
      throw std::logic_error("Complex inline ECP capture has no arbitrary-weight replay");
    }

    /// Preserve the full complex Vnl for complete native energy assembly after successful capture.
    wftrain::DerivativeArrayView<const C> nonlocalEnergies() const
    {
      if (state != State::COMPLETE)
        throw std::logic_error("Complex inline ECP energies are incomplete");
      return {energies.data(), energies.size()};
    }

    /// Report concrete fixed capacities separately from the already-admitted shared stream.
    wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
    {
      wftrain::StreamingDerivativeStorageDiagnostics d;
      d.sample_count = source.sampleCount();
      d.sample_auxiliary_bytes = (coefficients.capacity() + energies.capacity()) * sizeof(C) +
          particles.capacity() * sizeof(const ParticleSet*) + moved_positions.capacity() * sizeof(double);
      d.retained_numeric_bytes = d.sample_auxiliary_bytes;
      d.allocation_generation = 1;
      d.storage_fingerprint = 1469598103934665603ULL;
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, coefficients);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, energies);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, moved_positions);
      psiformer::determinant::detail::mixStorage(d.storage_fingerprint, particles);
      return d;
    }

    const std::shared_ptr<Model> model;
    const wftrain::StreamingDerivativeOperator& source;
    const std::vector<double>& positions;
    std::unique_lock<std::mutex> execution;
    std::shared_ptr<DerivativeLane> cpu_lane;
#ifdef ENABLE_CUDA
    std::shared_ptr<GpuDerivativeLane> gpu_lane;
#endif
    std::vector<C> coefficients, energies;
    std::vector<double> moved_positions;
    std::vector<const ParticleSet*> particles;
    wftrain::ParameterReductionSink* sink = nullptr;
    wftrain::DerivativeStreamDescriptor descriptor;
    State state = State::IDLE;
    std::uint64_t grid = 0;
    std::size_t expected = 0, points = 0, tiles = 0;
  };

  /// Override initializer values before parameter-dependent identity or device image construction.
  static InitializedNativeModel materialize(psiformer::InitializedPsiFormerParameters initialized,
      const ParticleSet& electrons, const ParticleSet& ions, const std::string& component_name,
      const std::string& parameter_checkpoint, const std::string& system_kind)
  {
    if (system_kind != "all_electron" && system_kind != "pseudopotential")
      throw std::invalid_argument("Complex PsiFormer requires an explicit AE or pseudopotential system kind");
    if (initialized.profile != psiformer::PSIFORMER_COMPLEX_PERIODIC_V1 &&
        initialized.profile != psiformer::PSIFORMER_PERIODIC_PAIR_V1)
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
       std::size_t gpu_limit, const std::string& parameter_checkpoint, const std::string& system_kind,
       std::size_t cpu_limit)
      : Impl(std::make_shared<Model>(component_name,
                                   materialize(std::move(initialized), electrons, ions, component_name, parameter_checkpoint, system_kind),
                                   electrons, ions, reduced, admitGpuClone(electrons, limit, gpu_limit), system_kind, cpu_limit),
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
        state->plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], model.blocks, model.p.size(),
        bool(state->environment.periodic_profile), occupiedMomentumCount(state->plan),
        state->environment.periodic_profile ? state->environment.periodic_profile->pair_feature_dimension : 8,
        fourierModeCount(state->plan)};
    if (!state->gpu_limit && !state->cpu_budget)
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
    if (!state->gpu_limit && !state->cpu_budget)
    {
      value = state->value.makeWorkspace();
      full = state->spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
      active = state->spatial.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
    }
    own_gradient.resize(model.ne);
    own_laplacian.resize(model.ne);
    if (numericStorage().allocated_bytes != numeric_required)
      throw std::logic_error("Complex PsiFormer numeric capacity differs from its admitted exact requirement");
    if (psiformer::memory::enabled())
      trace_clone_capacity.snapshotAt(__FILE__, __LINE__, "ComplexWF::clone_host_workspace", numericStorage().allocated_bytes,
          reinterpret_cast<std::uintptr_t>(this));
  }

  /// Sum live capacities and allocation identities, excluding immutable shared model storage.
  NumericStorage numericStorage() const
  {
    using namespace psiformer;
    using pf::addStorageBytes;
    using pf::checkedStorageBytes;
    std::size_t bytes = 0;
    if (value)
      addStorageBytes(bytes, value->vectorStorageBytes(), "complex clone bytes overflow");
    if (full)
      addStorageBytes(bytes, full->vectorStorageBytes(), "complex clone bytes overflow");
    if (active)
      addStorageBytes(bytes, active->vectorStorageBytes(), "complex clone bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<GradType>(own_gradient.capacity(), "complex own gradient bytes overflow"),
                    "complex clone bytes overflow");
    addStorageBytes(bytes, checkedStorageBytes<ValueType>(own_laplacian.capacity(), "complex own Laplacian bytes overflow"),
                    "complex clone bytes overflow");
    std::size_t fingerprint = 0;
    if (value)
      fingerprint = value->storageFingerprint();
    for (auto next : {full ? full->storageFingerprint() : 0, active ? active->storageFingerprint() : 0})
      if (next)
        fingerprint = (fingerprint ^ next) * 1099511628211ULL;
    determinant::detail::mixStorage(fingerprint, own_gradient);
    determinant::detail::mixStorage(fingerprint, own_laplacian);
    NumericStorage result{numeric_required, bytes, numeric_limit, fingerprint};
    if (state->cpu_budget)
    {
      std::lock_guard lock(state->cpu_budget->mutex);
      result.shared_cpu_bytes = state->cpu_budget->retained;
      result.shared_cpu_limit_bytes = state->cpu_budget->limit;
      result.shared_cpu_owners = state->cpu_budget->owners;
      result.shared_cpu_fingerprint = state->cpu_budget->fingerprint;
      result.shared_cpu_identity = state->cpu_budget.get();
      result.shared_cpu_value_bytes = state->cpu_requirements[0];
      result.shared_cpu_full_bytes = state->cpu_requirements[1];
      result.shared_cpu_active_bytes = state->cpu_requirements[2];
    }
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

  /// Validate the entire additive observation before changing any caller product.
  void validateOwnScatter(const ParticleSet::ParticleGradient& g, const ParticleSet::ParticleLaplacian& l) const
  {
    if (g.size() != own_gradient.size() || l.size() != own_laplacian.size())
      throw std::invalid_argument("Complex PsiFormer accepted FULL scatter extent mismatch");
    for (std::size_t e = 0; e < own_gradient.size(); ++e)
    {
      for (int d = 0; d < 3; ++d)
        if (!isFiniteWavefunctionValue(g[e][d] + own_gradient[e][d]))
          throw std::overflow_error("Complex PsiFormer accepted FULL gradient scatter overflow");
      if (!isFiniteWavefunctionValue(l[e] + own_laplacian[e]))
        throw std::overflow_error("Complex PsiFormer accepted FULL Laplacian scatter overflow");
    }
  }

  /// Publish only after every destination in this observation has passed validation.
  void scatterOwn(ParticleSet::ParticleGradient& g, ParticleSet::ParticleLaplacian& l) const noexcept
  {
    for (std::size_t e = 0; e < own_gradient.size(); ++e)
    {
      g[e] += own_gradient[e];
      l[e] += own_laplacian[e];
    }
  }

  /// Caller holds Model read authority; preserve ordinary complex squares and actual masses.
  LogValue observeOwnKinetic(const ParticleSet& p) const
  {
    requireIdle(p);
    if (!fullMatches(p)) throw std::logic_error("Complex energy observation requires current FULL own products");
    LogValue kinetic{};
    const auto& masses = p.get_mass_by_group();
    if (masses.size() != static_cast<std::size_t>(p.groups()))
      throw std::invalid_argument("Complex PsiFormer target mass extent mismatch");
    for (int group = 0; group < p.groups(); ++group)
    {
      const double mass = masses[group];
      if (!psiformer::determinant::isFiniteReal(mass) || mass <= 0)
        throw std::invalid_argument("Complex PsiFormer target mass must be positive and finite");
      for (int e = p.first(group); e < p.last(group); ++e)
      {
        LogValue ratio = own_laplacian[e];
        for (int d = 0; d < 3; ++d)
          ratio += LogValue(own_gradient[e][d]) * LogValue(own_gradient[e][d]);
        kinetic -= .5 * (1.0 / mass) * ratio;
      }
    }
    if (!isFiniteWavefunctionValue(kinetic)) throw std::overflow_error("Complex kinetic observation overflow");
    return kinetic;
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
      const ParticleSet::PosType* absolute = nullptr,
      const MCMultiParticleMoves<CoordsType::POS>::Slice* collective = nullptr) const
  {
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension,
            replacement == electron ? (absolute ? (*absolute)[dimension] : p.activeR(electron)[dimension])
                                    : p.R[electron][dimension]);
    if (collective)
      for (std::size_t selected = 0; selected < collective->size(); ++selected)
        for (int dimension = 0; dimension < 3; ++dimension)
          workspace.setPosition(collective->particleIndex(selected), dimension,
              collective->proposedPosition(selected)[dimension]);
  }

  /// Capacity is immutable even when publication replaces the generation pointer.
  bool denseEnabled() const noexcept { return state->gpu_capacity > 1; }

  /// Collective B1 and shared CPU inference also need an exact native resource loan.
  bool crowdResourcesEnabled() const noexcept { return bool(state->cpu_budget) || state->gpu_limit != 0; }

  /// Select one crowd's exclusive scratch, retaining the family fallback for scalar callers.
  std::shared_ptr<CpuLane> selectedCpuLane() const
  {
    if (acquired_leader)
    {
      const auto& leader = *acquired_leader->impl_;
      if (leader.multiwalker && leader.multiwalker.getResource().cpu)
        return leader.multiwalker.getResource().cpu;
    }
    return state->cpu;
  }

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
      throw std::logic_error("Complex native call requires its acquired leader/team");
    auto& resource = multiwalker.getResource();
    if (resource.state != state || resource.components.size() != components.size())
      throw std::invalid_argument("Complex native crowd has foreign resource ownership or extent");
    if (resource.collective_pending && !collective_resolution)
      throw std::logic_error("Complex native staging is frozen until collective resolution");
    for (std::size_t i = 0; i < components.size(); ++i)
    {
      auto* component = dynamic_cast<PsiFormerComplexWF*>(&components[i]);
      if (!component || component != resource.components[i] || component->impl_->state != state ||
          component->impl_->acquired_leader != leader || component->impl_->acquired_lane != i)
        throw std::invalid_argument("Complex native call changed its acquired model or lane order");
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
    std::shared_ptr<CpuLane> cpu_owner;
#ifdef ENABLE_CUDA
    std::shared_ptr<GpuLane> owner;
    std::optional<psiformer::PsiFormerInferenceConcurrency::Lease> permit;
#endif
    std::unique_lock<std::mutex> lease;
  };

  /// Route VALUE/FULL/ACTIVE without duplicating physical twist, ratio or cache algebra.
  CoreResult core(const ParticleSet& p, bool spatial, int replacement = -1, int electron = -1,
      const ParticleSet::PosType* absolute = nullptr,
      const MCMultiParticleMoves<CoordsType::POS>::Slice* collective = nullptr)
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
    if (state->cpu_budget)
    {
      result.cpu_owner = selectedCpuLane();
      result.lease = std::unique_lock(result.cpu_owner->mutex);
    }
    if (spatial)
    {
      NativeTimerScope numerical_timer(phase_timers.cpu_spatial);
      const auto mode = electron < 0 ? pf::DirectSpatialMode::FULL_VGL : pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT;
      auto& workspace = result.cpu_owner
          ? result.cpu_owner->prepare(electron < 0 ? result.cpu_owner->full : result.cpu_owner->active,
              state->cpu_requirements[electron < 0 ? 1 : 2], [&] { return state->spatial.makeWorkspace(mode); },
              electron < 0 ? CpuLane::Mode::FULL_VGL : CpuLane::Mode::ACTIVE_ELECTRON_GRADIENT)
          : (electron < 0 ? *full : *active);
      positions(workspace, p, replacement, absolute, collective);
      static_cast<pf::DirectComplexSpatialResult&>(result) = electron < 0
          ? state->spatial.evaluateFull(workspace) : state->spatial.evaluateActive(workspace, electron);
    }
    else
    {
      auto& workspace = result.cpu_owner
          ? result.cpu_owner->prepare(result.cpu_owner->value, state->cpu_requirements[0],
              [&] { return state->value.makeWorkspace(); }, CpuLane::Mode::VALUE) : *value;
      positions(workspace, p, replacement, absolute);
      const auto evaluated = state->value.evaluate(workspace);
      result.amplitude = evaluated.amplitude;
      result.parameter_version = evaluated.parameter_version;
    }
    return result;
  }

  /// Retain one base coordinate upload while immediately consuming independent GPU VALUE replacements.
  template<class ElectronAt, class PositionAt>
  bool streamVirtualValues(const ParticleSet& p, ElectronAt&& electron_at, PositionAt&& position_at,
      std::vector<ValueType>& staged)
  {
#ifdef ENABLE_CUDA
    if (state->gpu && staged.size() > 1)
    {
      NativeTimerScope timer(phase_timers.value);
      auto permit = state->inference_concurrency.acquire();
      const auto owner = selectedGpuLane();
      auto& lane = *owner;
      std::unique_lock lease(lane.mutex);
      bindGpuLane(lane);
      lane.resizeLive(1);
      for (int e = 0; e < p.getTotalNum(); ++e)
        for (int d = 0; d < 3; ++d)
          lane.positions[3 * e + d] = p.R[e][d];
      lane.value->evaluateReplacementStream(lane.positions, staged.size(),
          [&](std::size_t point) {
            const auto absolute = position_at(point);
            return psiformer::device::PreparedDeviceComplexValue::Replacement{
                static_cast<std::size_t>(electron_at(point)), {absolute[0], absolute[1], absolute[2]}};
          },
          [&](std::size_t point, const auto& view) {
            {
              NativeTimerScope download(phase_timers.gpu_download);
              lane.value->download(view, lane.amplitude);
            }
            const Amplitude amplitude{{lane.amplitude[0].phase.x, lane.amplitude[0].phase.y}, lane.amplitude[0].log_abs};
            const auto absolute = position_at(point);
            staged[point] = formRatio(amplitude, p, electron_at(point), &absolute);
          },
          [&](auto&& evaluate) {
            NativeTimerScope numerical(phase_timers.gpu_evaluate);
            evaluate();
          });
      return true;
    }
#endif
    return false;
  }

  /// Reuse one crowd FULL tape, copying each borrowed result before releasing its lease.
  void stageCpuCollective(MultiWalkerResource& resource, const RefVectorWithLeader<ParticleSet>& targets,
      const MCMultiParticleMoves<CoordsType::POS>& moves)
  {
    resource.prepareCpuCollective(targets.size());
    const auto n = state->model.ne;
    for (std::size_t walker = 0; walker < targets.size(); ++walker)
    {
      auto& clone = *resource.components[walker]->impl_;
      const auto slice = moves.slice(walker);
      const auto result = clone.core(targets[walker], true, -1, -1, nullptr, &slice);
      resource.amplitude[walker] = result.amplitude;
      resource.logs[walker] = clone.physicalLog(result.amplitude, targets[walker], slice);
      resource.keys[walker] = configurationIdentity(targets[walker], slice);
      for (std::size_t e = 0; e < n; ++e)
      {
        for (int d = 0; d < 3; ++d)
          resource.gradient[walker * n + e][d] = clone.physicalGradient(result.gradient[3 * e + d], d);
        const auto value = result.laplacian_log[e];
        const auto converted = makePsiFormerScalar<ValueType>(value.real(), value.imag());
        if (!isFiniteWavefunctionValue(converted))
          throw std::overflow_error("Complex CPU collective Laplacian conversion overflow");
        resource.laplacian[walker * n + e] = converted;
      }
    }
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
#ifdef ENABLE_CUDA
  std::shared_ptr<GpuOrbitalMSELane> gpu_orbital_mse;
#endif
  std::shared_ptr<AffineLane> affine_lane;
  const ParticleSet* particles;
  std::unique_ptr<pf::DirectComplexValueWorkspace> value;
  std::unique_ptr<pf::DirectComplexSpatialWorkspace> full, active;
  std::vector<GradType> own_gradient;
  psiformer::memory::HostCapacity trace_clone_capacity;
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

/// Keep the accepted traversal and both provenance owners alive without copying virtual configurations.
class PsiFormerComplexWF::FixedGridEnergyReference
{
  friend class PsiFormerComplexWF;
#ifdef ENABLE_CUDA
  std::shared_ptr<Impl::Model> source;
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
  std::shared_ptr<const Impl::GpuNonLocalReplay> replay;
  std::shared_ptr<const wftrain::NonLocalECPNativeCapture> native;
  std::vector<double> inverse_masses;
  std::vector<int> groups;
  std::size_t sample = 0;

  /// One checked core serves requested fresh-vector admission and actual retained-capacity verification.
  static FixedGridEnergyReferenceRequirements requirements(std::size_t mass_capacity, std::size_t group_capacity)
  {
    const char* error = "Fixed-grid energy reference HOST capacity overflow";
    auto bytes = pf::checkedStorageSum(sizeof(FixedGridEnergyReference),
        pf::checkedStorageBytes<double>(mass_capacity, error), error);
    pf::addStorageBytes(bytes, pf::checkedStorageBytes<int>(group_capacity, error), error);
    return {bytes, sizeof(Impl::GpuNonLocalReplay)};
  }

  /// Both callers hold Model read; require identical accepted physical reference and kinetic metadata.
  void validateParticles(const Impl& owner, const ParticleSet& p) const
  {
    owner.requireIdle(p);
    const auto& masses = p.get_mass_by_group();
    if (p.R.size() != groups.size() || masses.size() != static_cast<std::size_t>(p.groups()))
      throw std::invalid_argument("Fixed-grid candidate particle extent/masses differ");
    for (std::size_t e = 0; e < groups.size(); ++e)
    {
      const auto group = p.GroupID[e];
      if (group != groups[e] || group < 0 || static_cast<std::size_t>(group) >= masses.size() ||
          !psiformer::determinant::isFiniteReal(masses[group]) || masses[group] <= 0 ||
          1.0 / masses[group] != inverse_masses[e])
        throw std::invalid_argument("Fixed-grid candidate spin/mass metadata differs");
      for (int d = 0; d < 3; ++d)
        if (p.R[e][d] != replay->reference_positions[3 * (sample * groups.size() + e) + d])
          throw std::invalid_argument("Fixed-grid candidate reference coordinates differ");
    }
  }

  /// Source publication, private re-isolation, pending moves and stale immutable images fail closed.
  void validate(const Impl& original, const Impl& candidate) const
  {
    if (original.state != source || candidate.state == source || !source->gpu || !candidate.state->gpu ||
        source->gpu->image != image || source->model.p.version() != replay->version ||
        source->pending_proposals.load() || candidate.state->pending_proposals.load() ||
        candidate.state->private_origin.lock() != source || candidate.state->private_origin_image.lock() != image ||
        candidate.state->private_origin_version != replay->version ||
        candidate.state->gpu->image->version() != candidate.state->model.p.version() ||
        candidate.state->system_kind != source->system_kind || candidate.state->source != source->source ||
        candidate.state->schema->fingerprint() != source->schema->fingerprint() ||
        candidate.state->reduced_twist != source->reduced_twist)
      throw std::logic_error("Fixed-grid candidate requires its current private GPU isolation origin");
    Impl::validateDerivativeSource(*source); Impl::validateDerivativeSource(*candidate.state);
    image->validateCurrentDevice(); candidate.state->gpu->image->validateCurrentDevice();
    validateParticles(original, *original.particles);
  }
#endif
};

/// Construct the explicit paired CPU or strictly capped shared-GPU inference profile.
PsiFormerComplexWF::PsiFormerComplexWF(std::string name, psiformer::InitializedPsiFormerParameters initialized,
    const ParticleSet& electrons, const ParticleSet& ions, std::size_t limit, std::array<double, 3> reduced,
    std::size_t gpu_limit, const std::string& parameter_checkpoint, const std::string& system_kind,
    std::size_t cpu_limit)
    : WaveFunctionComponent(name),
      impl_(std::make_unique<Impl>(name, std::move(initialized), electrons, ions, limit, reduced,
                                 gpu_limit, parameter_checkpoint, system_kind, cpu_limit))
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
  std::lock_guard factory(state->derivative_factory_mutex);
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    const auto& lane = impl_->gpu_orbital_mse;
    return {Impl::GpuOrbitalMSELane::required(*state), lane ? lane->bytes : 0,
        lane ? lane->fingerprint() : 0};
  }
#endif
  const auto& lane = impl_->orbital_mse;
  return {Impl::OrbitalMSELane::required(*state), lane ? lane->bytes() : 0,
      lane ? lane->work->storageFingerprint() ^ lane->score.numericMetadataStorageFingerprint() : 0};
}

/// Admit the exact score-only allocation before construction; repeat preparation is a no-op.
void PsiFormerComplexWF::prepareOrbitalMSE(std::size_t maximum_bytes)
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto required =
#ifdef ENABLE_CUDA
      gpuInferenceEnabled() ? Impl::GpuOrbitalMSELane::required(*state) :
#endif
      Impl::OrbitalMSELane::required(*state);
  if (!maximum_bytes || maximum_bytes == std::numeric_limits<std::size_t>::max() || required > maximum_bytes)
    throw std::length_error("Complex orbital-MSE numeric cap exceeded before preparation");
  if (state->pending_proposals.load(std::memory_order_relaxed))
    throw std::logic_error("Complex orbital-MSE requires an idle model family");
  std::lock_guard factory(state->derivative_factory_mutex);
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    std::lock_guard execution(state->gpu->mutex);
    if (!impl_->gpu_orbital_mse)
      impl_->gpu_orbital_mse = std::make_shared<Impl::GpuOrbitalMSELane>(*state);
    return;
  }
#endif
  if (!impl_->orbital_mse)
    impl_->orbital_mse = std::make_shared<Impl::OrbitalMSELane>(*state);
}

/// CPU borrows VALUE scratch; GPU borrows its explicitly prepared raw-VJP owner.
void PsiFormerComplexWF::evaluateOrbitalValues(const ParticleSet& particles,
    std::vector<std::complex<double>>& orbitals, std::size_t expected_version) const
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (state->model.p.version() != expected_version)
    throw std::runtime_error("Complex orbital forward parameter version is stale");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto n = state->model.ne, up = state->model.cfg.nup;
  if (state->pending_proposals.load(std::memory_order_relaxed) || particles.getActivePtcl() >= 0 ||
      particles.isSpinor() || particles.getTotalNum() != n || particles.groups() != 2 ||
      particles.groupsize(0) != up || particles.groupsize(1) != n - up || particles.R.size() != n || particles.GroupID.size() != n)
    throw std::invalid_argument("Complex orbital forward sample/proposal/spin mismatch");
  validateRuntimeLattice(particles, state->environment, "Complex orbital forward sample");
  for (std::size_t row = 0; row < n; ++row)
    for (int d = 0; d < 3; ++d)
      if (particles.GroupID[row] != (row < up ? 0 : 1) || !psiformer::determinant::isFiniteReal(particles.R[row][d]))
        throw std::invalid_argument("Complex orbital forward coordinates/spin order invalid");
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    std::shared_ptr<Impl::GpuOrbitalMSELane> lane;
    { std::lock_guard factory(state->derivative_factory_mutex); lane = impl_->gpu_orbital_mse; }
    if (!lane) throw std::logic_error("Complex GPU orbital forward requires explicit orbital-MSE preparation");
    std::scoped_lock execution(lane->execution_mutex, state->gpu->mutex);
    lane->forward(*state, particles);
    orbitals.resize(lane->raw.size());
    for (std::size_t i = 0; i < orbitals.size(); ++i)
      orbitals[i] = {cuCreal(lane->raw[i]), cuCimag(lane->raw[i])};
    return;
  }
#endif
  auto owner = state->cpu_budget ? impl_->selectedCpuLane() : nullptr;
  std::unique_lock<std::mutex> execution;
  if (owner)
    execution = std::unique_lock(owner->mutex);
  auto& workspace = owner ? owner->prepare(owner->value, state->cpu_requirements[0],
      [&] { return state->value.makeWorkspace(); }, Impl::CpuLane::Mode::VALUE) : *impl_->value;
  impl_->positions(workspace, particles);
  const auto values = state->value.evaluateOrbitalValues(workspace);
  orbitals.assign(values.values, values.values + values.size);
}

/// Read one stable model version and use only prepared scratch, never accepted caches.
double PsiFormerComplexWF::evaluateOrbitalMSE(const ParticleSet& particles,
    const std::vector<std::complex<double>>& target, std::vector<double>& gradient,
    std::size_t expected_version, wftrain::OrbitalLossNormalization normalization) const
{
  auto state = impl_->state;
  std::shared_lock read(state->mutex);
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
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    using Normalization = wftrain::OrbitalLossNormalization;
    if (normalization != Normalization::SPIN_SECTOR_SUM && normalization != Normalization::FULL_MATRIX_MEAN)
      throw std::invalid_argument("Invalid complex orbital-MSE normalization");
    std::shared_ptr<Impl::GpuOrbitalMSELane> lane;
    { std::lock_guard factory(state->derivative_factory_mutex); lane = impl_->gpu_orbital_mse; }
    if (!lane) throw std::logic_error("Complex GPU orbital-MSE must be explicitly prepared");
    if (target.size() != lane->raw.size() || gradient.size() != lane->response.size())
      throw std::invalid_argument("Complex GPU orbital-MSE target/gradient extent mismatch");
    const auto output = reinterpret_cast<std::uintptr_t>(gradient.data());
    const auto overlaps = [&](const void* data, std::size_t bytes) {
      const auto input = reinterpret_cast<std::uintptr_t>(data);
      const auto output_bytes = pf::checkedStorageBytes<double>(gradient.size(), "orbital output bytes overflow");
      if (bytes > std::numeric_limits<std::uintptr_t>::max() - input ||
          output_bytes > std::numeric_limits<std::uintptr_t>::max() - output)
        throw std::invalid_argument("Complex GPU orbital-MSE address overflow");
      return bytes && output_bytes && output < input + bytes && input < output + output_bytes;
    };
    if (overlaps(target.data(), pf::checkedStorageBytes<Impl::C>(target.size(), "orbital target bytes overflow")) ||
        overlaps(state->model.p.flat_values().data(), pf::checkedStorageBytes<double>(gradient.size(), "orbital parameter bytes overflow")))
      throw std::invalid_argument("Complex GPU orbital-MSE output aliases target/parameters");
    for (const auto value : target)
      if (!isFiniteWavefunctionValue(value)) throw std::invalid_argument("Complex GPU orbital-MSE target is nonfinite");
    std::scoped_lock execution(lane->execution_mutex, state->gpu->mutex);
    lane->forward(*state, particles);
    double loss = 0;
    for (std::size_t i = 0; i < target.size(); ++i)
    {
      const auto row = (i / n) % n;
      const auto denominator = normalization == Normalization::FULL_MATRIX_MEAN ? n : (row < up ? up : n - up);
      const double scale = 1.0 / static_cast<double>(state->model.ndet * denominator * n);
      const Impl::C residual = Impl::C{cuCreal(lane->raw[i]), cuCimag(lane->raw[i])} - target[i];
      loss += scale * std::norm(residual);
      const auto seed = 2.0 * scale * std::conj(residual);
      lane->raw[i] = make_cuDoubleComplex(seed.real(), seed.imag());
    }
    if (!psiformer::determinant::isFiniteReal(loss)) throw std::overflow_error("Complex GPU orbital-MSE loss overflow");
    lane->owner->reverseOrbitals(lane->owner->orbitalView(), lane->raw);
    lane->owner->downloadOrbitalVJP(lane->owner->orbitalVJPView(), lane->response);
    for (double value : lane->response)
      if (!psiformer::determinant::isFiniteReal(value)) throw std::overflow_error("Complex GPU orbital-MSE response overflow");
    std::copy(lane->response.begin(), lane->response.end(), gradient.begin());
    return loss;
  }
#endif
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
      {target.data(), target.size(), gradient.data(), gradient.size(), normalization}).loss;
}

/// Report immutable system metadata without exposing a borrowed string or model.
bool PsiFormerComplexWF::isPseudopotentialSystem() const
{
  std::shared_lock read(impl_->state->mutex);
  return impl_->state->system_kind == "pseudopotential";
}

/// Validate the fixed physical source and selected Gamma geometry without copying learned parameters.
bool PsiFormerComplexWF::matchesPairedDevicePairTarget(const ParticleSet& electrons) const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& state = *impl_->state;
  if (!psiformer::matchesPairedDevicePairMetadata(state.environment) ||
      electrons.getLattice().getSuperCellEnum() != SUPERCELL_BULK)
    return false;
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!psiformer::determinant::isFiniteReal(electrons.getTwist()[axis]) ||
        electrons.getTwist()[axis] != 0 || state.reduced_twist[axis] != 0 || state.cartesian_twist[axis] != 0)
      return false;
  validateRuntimeLattice(electrons, state.environment, "Paired GPU selected target");
  Impl::validateDerivativeSource(state);
  return true;
}

/// Match immutable legacy metadata and the actual target boundary without parameter copies.
bool PsiFormerComplexWF::matchesLegacyTorusAllElectronTarget(const ParticleSet& electrons) const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& state = *impl_->state;
  return electrons.getLattice().SuperCellEnum == SUPERCELL_BULK &&
      state.system_kind == "all_electron" && !state.environment.periodic_profile &&
      state.environment.boundary == psiformer::BoundaryCondition::PERIODIC &&
      state.environment.geometry_feature_policy == psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
}

/// Match the fixed legacy torus target, including its outer twist and source bindings.
bool PsiFormerComplexWF::matchesLegacyTorusPseudopotentialTarget(const ParticleSet& electrons) const
{
  std::shared_lock read(impl_->state->mutex);
  const auto& state = *impl_->state;
  if (&electrons != impl_->particles || electrons.getLattice().SuperCellEnum != SUPERCELL_BULK ||
      state.system_kind != "pseudopotential" || state.environment.periodic_profile ||
      state.environment.boundary != psiformer::BoundaryCondition::PERIODIC ||
      state.environment.geometry_feature_policy != psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1)
    return false;
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!psiformer::determinant::isFiniteReal(electrons.getTwist()[axis]) ||
        electrons.getTwist()[axis] != state.reduced_twist[axis])
      return false;
  validateRuntimeLattice(electrons, state.environment, "Paired legacy PP target");
  Impl::validateDerivativeSource(state);
  return true;
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
  if (s.system_kind != "all_electron") mixPersistentString(hash, s.system_kind);
  mixPersistentInteger(hash, s.seed);
  mixPeriodicProfile(hash, s.environment);
  if (s.environment.periodic_profile)
  {
    for (double value : s.physical_source) mixPersistentDouble(hash, value);
    for (const auto& species : s.physical_species) mixPersistentString(hash, species);
  }
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

/// Bound only transient scalar/coordinate capture; shared derivative scratch is already admitted.
std::size_t PsiFormerComplexWF::nonLocalVJPCaptureRequiredBytes(std::size_t samples) const
{
  std::shared_lock read(impl_->state->mutex);
  return Impl::NonLocalVJPCapture::required(samples, impl_->state->model.ne, gpuInferenceEnabled());
}

/// Pin a fresh accepted derivative stream through one native, fixed-coefficient ECP traversal.
std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer> PsiFormerComplexWF::makeBudgetedNonLocalVJPConsumer(
    const RefVectorWithLeader<TrialWaveFunction>& wavefunctions, const RefVectorWithLeader<ParticleSet>& particles,
    const wftrain::StreamingDerivativeOperator& op, std::size_t host_cap) const
{
  const auto* cpu = dynamic_cast<const Impl::DerivativeStream*>(&op);
#ifdef ENABLE_CUDA
  const auto* gpu = dynamic_cast<const Impl::GpuDerivativeStream*>(&op);
  const bool device = gpu != nullptr;
  if ((cpu && cpu->state != impl_->state) || (gpu && gpu->lane->state != impl_->state) || (!cpu && !gpu))
    throw std::invalid_argument("Complex inline ECP requires its own accepted derivative stream");
  std::unique_lock execution(device ? gpu->lane->execution_mutex : cpu->lane->execution_mutex);
#else
  const bool device = false;
  if (!cpu || cpu->state != impl_->state)
    throw std::invalid_argument("Complex inline ECP requires its own CPU derivative stream");
  std::unique_lock execution(cpu->lane->execution_mutex);
#endif
  std::shared_lock read(impl_->state->mutex);
  const auto& positions = cpu ? cpu->positions
#ifdef ENABLE_CUDA
      : gpu->positions
#else
      : cpu->positions
#endif
      ;
  if (!host_cap || host_cap == std::numeric_limits<std::size_t>::max() ||
      Impl::NonLocalVJPCapture::required(op.sampleCount(), impl_->state->model.ne, device) > host_cap)
    throw std::length_error("Complex inline ECP capture exceeds finite HOST admission");
  if (op.parameterVersion() != impl_->state->model.p.version() ||
      impl_->state->pending_proposals.load(std::memory_order_relaxed) || particles.size() != op.sampleCount() ||
      wavefunctions.size() != op.sampleCount() || &particles.getLeader() != impl_->particles ||
      &particles[0] != &particles.getLeader() || &wavefunctions[0] != &wavefunctions.getLeader())
    throw std::invalid_argument("Complex inline ECP accepted stream or leader identity differs");
  Impl::validateDerivativeSource(*impl_->state);
#ifdef ENABLE_CUDA
  if (gpu && (gpu->lane->generation != impl_->state->gpu || gpu->replay))
    throw std::invalid_argument("Complex inline ECP requires an unbound current GPU generation");
#endif
  for (std::size_t b = 0; b < particles.size(); ++b)
  {
    const auto& orbitals = wavefunctions[b].getOrbitals();
    const auto* component = orbitals.size() == 1 ? dynamic_cast<const PsiFormerComplexWF*>(orbitals[0].get()) : nullptr;
    if (!component || component->impl_->state != impl_->state || particles[b].isSpinor())
      throw std::invalid_argument("Complex inline ECP requires sole same-family nonspinor clones");
    component->impl_->requireIdle(particles[b]);
    for (std::size_t previous = 0; previous < b; ++previous)
      if (&particles[previous] == &particles[b] || &wavefunctions[previous] == &wavefunctions[b])
        throw std::invalid_argument("Complex inline ECP native clone scratch aliases");
    for (std::size_t e = 0; e < particles[b].R.size(); ++e)
      for (int d = 0; d < 3; ++d)
        if (particles[b].R[e][d] != positions[3 * (b * particles[b].R.size() + e) + d])
          throw std::invalid_argument("Complex inline ECP accepted coordinates changed before preparation");
  }
  std::unique_ptr<Impl::NonLocalVJPCapture> capture;
#ifdef ENABLE_CUDA
  if (gpu) capture = std::make_unique<Impl::NonLocalVJPCapture>(*gpu, particles, std::move(execution));
  else
#endif
    capture = std::make_unique<Impl::NonLocalVJPCapture>(*cpu, particles, std::move(execution));
  if (capture->storageDiagnostics().retained_numeric_bytes > host_cap)
    throw std::length_error("Complex inline ECP actual capture capacities exceed HOST admission");
  return capture;
}

/// Borrow completed complex nonlocal energy from the same native response transaction.
wftrain::DerivativeArrayView<const wftrain::DerivativeValue> PsiFormerComplexWF::capturedNonLocalEnergies(
    const wftrain::NonLocalECPDerivativeConsumer& consumer) const
{
  const auto* capture = dynamic_cast<const Impl::NonLocalVJPCapture*>(&consumer);
  if (!capture || capture->model != impl_->state)
    throw std::invalid_argument("Complex inline ECP energy receipt belongs to another provider");
  return capture->nonlocalEnergies();
}

/// Metadata-only pre-MC ledger query; no dummy operator, allocation or GPU call.
PsiFormerComplexWF::GpuNonLocalReplayRequirements PsiFormerComplexWF::gpuNonLocalReplayRequirements(
    std::size_t samples, std::size_t maximum_points) const
{
#ifdef ENABLE_CUDA
  std::shared_lock read(impl_->state->mutex);
  if (!impl_->state->gpu || impl_->state->gpu_crowds != 1)
    throw std::logic_error("Complex ECP capture requires one active GPU inference owner");
  Impl::validateDerivativeSource(*impl_->state);
  return Impl::GpuNonLocalCapture::requiredShape(samples, impl_->state->model.ne, maximum_points);
#else
  throw std::logic_error("Complex ECP capture requires the explicit GPU component route");
#endif
}

/// Explicit HOST capture admission includes the eventual bound operator's worst-case packing.
std::size_t PsiFormerComplexWF::gpuNonLocalReplayRequiredBytes(const wftrain::StreamingDerivativeOperator& op) const
{
#ifdef ENABLE_CUDA
  const auto* s = dynamic_cast<const Impl::GpuDerivativeStream*>(&op);
  if (!s || s->lane->state != impl_->state) throw std::invalid_argument("Complex ECP capture requires its own GPU operator");
  std::unique_lock execution(s->lane->execution_mutex);
  std::shared_lock read(impl_->state->mutex);
  s->requireCurrent();
  if (s->replay || s->has_evaluated) throw std::logic_error("Complex ECP capture requires an unused unbound operator");
  return Impl::GpuNonLocalCapture::required(*s);
#else
  throw std::logic_error("Complex ECP capture requires the explicit GPU component route");
#endif
}

/// Prepare a replay-only transaction without exposing generic/unbudgeted GPU discovery.
std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer> PsiFormerComplexWF::makeBudgetedGpuNonLocalReplayConsumer(
    const RefVectorWithLeader<TrialWaveFunction>& wavefunctions, const RefVectorWithLeader<ParticleSet>& particles,
    const wftrain::StreamingDerivativeOperator& op, std::size_t host_cap) const
{
#ifdef ENABLE_CUDA
  const auto* s = dynamic_cast<const Impl::GpuDerivativeStream*>(&op);
  if (!s || s->lane->state != impl_->state || !host_cap || host_cap == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Complex ECP capture requires its own operator and finite HOST cap");
  std::unique_lock execution(s->lane->execution_mutex);
  std::shared_lock read(impl_->state->mutex);
  s->requireCurrent();
  if (s->replay || s->has_evaluated) throw std::logic_error("Complex ECP capture requires an unused unbound operator");
  const auto bytes = Impl::GpuNonLocalCapture::required(*s);
  if (bytes > host_cap) throw std::length_error("Complex ECP capture exceeds HOST admission before allocation");
  if (particles.size() != s->samples || wavefunctions.size() != s->samples ||
      &particles.getLeader() != impl_->particles || &particles[0] != &particles.getLeader() ||
      &wavefunctions[0] != &wavefunctions.getLeader())
    throw std::invalid_argument("Complex ECP capture leader/batch mismatch");
  for (std::size_t b = 0; b < particles.size(); ++b)
  {
    const auto& orbitals = wavefunctions[b].getOrbitals();
    const auto* c = orbitals.size() == 1 ? dynamic_cast<const PsiFormerComplexWF*>(orbitals[0].get()) : nullptr;
    if (!c || c->impl_->state != impl_->state || particles[b].isSpinor())
      throw std::invalid_argument("Complex ECP capture requires sole same-family nonspinor clones");
    c->impl_->requireIdle(particles[b]);
    for (std::size_t previous = 0; previous < b; ++previous)
      if (&particles[previous] == &particles[b] || &wavefunctions[previous] == &wavefunctions[b])
        throw std::invalid_argument("Complex ECP capture clone scratch aliases");
    for (std::size_t e = 0; e < particles[b].R.size(); ++e)
      for (int d = 0; d < 3; ++d)
        if (particles[b].R[e][d] != s->positions[3 * (b * particles[b].R.size() + e) + d])
          throw std::invalid_argument("Complex ECP capture reference changed before preparation");
  }
  app_log() << "Complex ECP replay-only preallocation HOST=" << bytes
            << " maximum_points=" << s->lane->maximum_samples - s->samples << " parameter_workspace=0" << std::endl;
  return std::make_unique<Impl::GpuNonLocalCapture>(*s, particles, std::move(execution), bytes);
#else
  throw std::logic_error("Complex ECP capture requires the explicit GPU component route");
#endif
}

/// Checked additional two-domain ownership; generic GPU discovery stays closed.
PsiFormerComplexWF::GpuDerivativeRequirements PsiFormerComplexWF::gpuDerivativeRequirements(
    std::size_t samples, std::size_t maximum_samples, std::size_t targets, std::size_t chunk, GpuDerivativeScope scope) const
{
#ifdef ENABLE_CUDA
  if (gpuInferenceEnabled())
  {
    std::shared_lock read(impl_->state->mutex);
    // Dense inference owns separate graphs; reverse products remain B1 per sample.
    // Multiple inference crowds still cannot share the serialized training lane.
    if (impl_->state->gpu_crowds > 1)
      throw std::logic_error("Concurrent complex inference does not support GPU derivatives or training");
    return Impl::GpuDerivativeLane::required(*impl_->state, samples, maximum_samples, targets, chunk, scope);
  }
#endif
  throw std::logic_error("Explicit GPU derivatives require an immutable GPU inference profile");
}

/// Admit the actual SOLE TrialWF batch before allocating any derivative numerical ownership.
std::unique_ptr<wftrain::StreamingDerivativeOperator> PsiFormerComplexWF::makeBudgetedGpuDerivativeOperator(
    const RefVectorWithLeader<TrialWaveFunction>& wavefunctions,
    const RefVectorWithLeader<ParticleSet>& particles, std::size_t ordinal, std::size_t offset,
    std::size_t maximum_samples, std::size_t targets, std::size_t chunk,
    std::size_t device_cap, std::size_t host_cap, std::size_t work_cap, GpuDerivativeScope scope) const
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
    const auto required = Impl::GpuDerivativeLane::required(*state, samples, maximum_samples, targets, chunk, scope);
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
    if (lane && (lane->maximum_samples != maximum_samples || lane->targets != targets || lane->chunk != chunk || lane->scope != scope))
      throw std::invalid_argument("Complex GPU derivative live lane profile mismatch");
    if (!lane)
      lane = std::make_shared<Impl::GpuDerivativeLane>(state, maximum_samples, targets, chunk,
          Impl::GpuDerivativeLane::required(*state, 0, maximum_samples, targets, chunk, scope), scope);
    auto result = std::make_unique<Impl::GpuDerivativeStream>(lane, ordinal, offset, samples, host_cap);
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
                       state.system_kind, state.reduced_twist, state.physical_source);
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

/// Expose exact affine metadata for the qualified CPU component.
std::vector<wftrain::AffineParameterBlock> PsiFormerComplexWF::affineParameterBlocks() const
{
  std::shared_lock read(impl_->state->mutex);
  return gpuInferenceEnabled() ? std::vector<wftrain::AffineParameterBlock>{} : Impl::affineBlocks(*impl_->state);
}

/// Expose affine roles only through the explicitly qualified hybrid profile.
std::vector<wftrain::AffineParameterBlock> PsiFormerComplexWF::hybridAffineParameterBlocks() const
{
  std::shared_lock read(impl_->state->mutex);
  if (!gpuInferenceEnabled() || (impl_->state->environment.periodic_profile &&
      !psiformer::matchesPairedDevicePairMetadata(impl_->state->environment)))
    throw std::logic_error("Hybrid affine metadata requires a qualified torus or selected paired GPU family");
  return Impl::affineBlocks(*impl_->state);
}

/// Explicit opt-in; unsupported requested profiles fail rather than replaying on HOST.
bool PsiFormerComplexWF::usesDeviceAffineStatistics() const
{
  const char* value = std::getenv("PSIFORMER_GPU_KFAC_AFFINE");
  if (!gpuInferenceEnabled() || !value || std::strcmp(value, "0") == 0)
    return false;
  if (std::strcmp(value, "1") != 0)
    throw std::invalid_argument("PSIFORMER_GPU_KFAC_AFFINE must be 0 or 1 for paired training");
#ifdef ENABLE_CUDA
  const auto& e = impl_->state->environment;
  if (e.periodic_profile || e.boundary != psiformer::BoundaryCondition::PERIODIC || !e.fixed_nuclei ||
      e.geometry_feature_policy != psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1 ||
      e.head_representation != psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX ||
      e.parameter_scalar_domain != psiformer::ScalarDomain::REAL ||
      e.compute_scalar_domain != psiformer::ScalarDomain::REAL ||
      e.amplitude_scalar_domain != psiformer::ScalarDomain::COMPLEX)
    throw std::invalid_argument("Paired DEVICE affine statistics require legacy FP64 torus metadata");
  return true;
#else
  throw std::logic_error("Paired DEVICE affine statistics are unavailable in this build");
#endif
}

PsiFormerComplexWF::DeviceAffineStatisticsRequirements PsiFormerComplexWF::deviceAffineStatisticsRequirements(
    const wftrain::KFACBlockRegistry& registry) const
{
#ifdef ENABLE_CUDA
  if (!usesDeviceAffineStatistics())
    throw std::logic_error("Paired DEVICE affine statistics are not enabled");
  std::shared_lock read(impl_->state->mutex);
  return Impl::deviceAffineRequirements(*impl_->state, Impl::deviceAffineDescriptors(*impl_->state, registry));
#else
  throw std::logic_error("Paired DEVICE affine statistics are unavailable in this build");
#endif
}

/// Collect raw DEVICE factors and HOST fallback samples without borrowing accepted caches.
void PsiFormerComplexWF::accumulateDeviceAffineStatistics(const std::vector<const ParticleSet*>& samples,
    const wftrain::KFACBlockRegistry& registry, std::size_t expected_version, bool gaussian,
    wftrain::DeviceAffineStatisticsSink& sink, std::size_t maximum_numeric_bytes) const
{
#ifdef ENABLE_CUDA
  if (!usesDeviceAffineStatistics() || samples.empty() || !maximum_numeric_bytes ||
      maximum_numeric_bytes == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Paired DEVICE affine collection requires samples and a finite numeric bound");
  const auto state = impl_->state;
  std::shared_ptr<Impl::GpuDerivativeLane> lane;
  {
    std::shared_lock read(state->mutex);
    std::lock_guard factory(state->derivative_factory_mutex);
    lane = state->gpu_derivative_lane.lock();
  }
  if (!lane)
    throw std::logic_error("Paired DEVICE affine collection requires an existing prepared score-products lane");
  // Match derivative streams: execution precedes model read and spans collection.
  // HOST sinks run outside the model and GPU locks.
  std::unique_lock execution(lane->execution_mutex);
  const auto current = [&] {
    if (impl_->state != state || lane->generation != state->gpu ||
        expected_version != state->model.p.version() || lane->scope != GpuDerivativeScope::SCORE_ECP_ONLY ||
        state->pending_proposals.load(std::memory_order_relaxed))
      throw std::logic_error("Paired DEVICE affine model/version/score scope is stale or busy");
    impl_->requireIdle(*impl_->particles);
    Impl::validateDerivativeSource(*state);
  };
  using Owner = psiformer::device::PreparedDeviceAffineStatistics;
  std::vector<psiformer::device::DeviceAffineFactorDescriptor> descriptors;
  std::unique_ptr<Owner> factors;
  const auto n = state->model.ne, up = state->model.cfg.nup;
  const auto p = state->schema->parameterCount();
  {
    std::shared_lock read(state->mutex);
    current();
    descriptors = Impl::deviceAffineDescriptors(*state, registry);
    const auto requirement = Impl::deviceAffineRequirements(*state, descriptors);
    if (pf::checkedStorageSum(requirement.device_bytes, requirement.host_bytes,
        "Complex DEVICE affine combined bound overflow") > maximum_numeric_bytes)
      throw std::length_error("Paired DEVICE affine collection exceeds its combined numeric bound");
    std::unique_lock gpu(lane->generation->mutex);
    factors = lane->products->prepareAffineStatistics(descriptors, gaussian,
        Owner::requiredDeviceBytes(descriptors), Owner::requiredHostBytes(descriptors), expected_version);
  }
  std::vector<double> positions(3 * n), real(p), imaginary(p);
  for (const auto* particles : samples)
  {
    if (!particles || particles->getActivePtcl() >= 0 || particles->R.getUnit() != PosUnit::Cartesian ||
        particles->isSpinor() || particles->getTotalNum() != n || particles->groups() != 2 ||
        particles->groupsize(0) != up || particles->groupsize(1) != n - up ||
        particles->R.size() != n || particles->GroupID.size() != n)
      throw std::invalid_argument("Paired DEVICE affine sample/proposal/spin mismatch");
    validateRuntimeLattice(*particles, state->environment, "Paired DEVICE affine sample");
    for (std::size_t electron = 0; electron < n; ++electron)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto coordinate = particles->R[electron][d];
        if (particles->GroupID[electron] != (electron < up ? 0 : 1) ||
            !psiformer::determinant::isFiniteReal(coordinate))
          throw std::invalid_argument("Paired DEVICE affine sample coordinate or spin order is invalid");
        positions[3 * electron + d] = coordinate;
      }
    sink.beginSample();
    try
    {
      {
        std::shared_lock read(state->mutex);
        current();
        std::unique_lock gpu(lane->generation->mutex);
        factors->beginSample(expected_version);
        lane->products->affineFallbackComplexScore(positions, *factors, real, imaginary);
      }
      sink.addComplexFallbackScore(real, imaginary);
      std::optional<Owner::SampleCommit> prepared;
      {
        std::shared_lock read(state->mutex);
        current();
        std::unique_lock gpu(lane->generation->mutex);
        prepared.emplace(factors->prepareSampleCommit());
      }
      sink.endSample();
      // No throwing operation may separate HOST acceptance and DEVICE publication.
      // A later source/version change rejects the disposable collection before import.
      prepared->commit();
    }
    catch (...)
    {
      sink.abortSample();
      factors->abortSample();
      throw;
    }
  }
  std::vector<psiformer::device::DeviceAffineFactorSums> downloaded(descriptors.size());
  for (std::size_t block = 0; block < descriptors.size(); ++block)
  {
    const auto input = descriptors[block].input_width + descriptors[block].bias_augmented;
    downloaded[block].activation_outer_sum.resize(input * input);
    downloaded[block].sensitivity_outer_sum.resize(descriptors[block].output_width * descriptors[block].output_width);
  }
  {
    std::shared_lock read(state->mutex);
    current();
    std::unique_lock gpu(lane->generation->mutex);
    factors->download(downloaded);
    current();
  }
  std::vector<wftrain::KFACFactorStatistics> imported;
  imported.reserve(descriptors.size());
  for (std::size_t block = 0; block < descriptors.size(); ++block)
    imported.push_back({descriptors[block].input_width + descriptors[block].bias_augmented,
        descriptors[block].output_width, downloaded[block].sample_weight_sum, downloaded[block].row_weight_sum,
        downloaded[block].row_count, std::move(downloaded[block].activation_outer_sum),
        std::move(downloaded[block].sensitivity_outer_sum)});
  sink.import(std::move(imported));
  app_log() << "Paired KFAC affine statistics: samples=" << factors->sampleCount()
            << " device_numeric_bytes=" << factors->deviceWorkspaceBytes()
            << " host_numeric_bytes=" << factors->hostWorkspaceBytes()
            << "; raw affine sums=DEVICE; separate score pass; fallback/reduction/solve/update=HOST" << std::endl;
#else
  throw std::logic_error("Paired DEVICE affine statistics are unavailable in this build");
#endif
}

/// Classify cusp and reciprocal-factor coefficients for exact per-coordinate diagonal statistics.
std::vector<std::size_t> PsiFormerComplexWF::genericKFACParameterBlocks() const
{
  std::shared_lock read(impl_->state->mutex);
  std::vector<std::size_t> result;
  const auto& state = *impl_->state;
  for (auto role : {psiformer::ParameterRole::CUSP_SAME_ALPHA, psiformer::ParameterRole::CUSP_OPPOSITE_ALPHA,
                   psiformer::ParameterRole::FOURIER_COEFFICIENT_UP, psiformer::ParameterRole::FOURIER_COEFFICIENT_DOWN})
    if (state.plan.hasParameter(role))
    {
      const auto offset = state.plan.parameter(role).begin;
      for (std::size_t block = 0; block < state.schema->blocks().size(); ++block)
        if (state.schema->blocks()[block].offset == offset && state.schema->blocks()[block].trainable)
          result.push_back(block);
    }
  return result;
}

/// Inspect required and prepared numeric storage under the model/factory locks.
PsiFormerComplexWF::AffineScoreStorage PsiFormerComplexWF::affineScoreStorage() const
{
  std::shared_lock read(impl_->state->mutex);
  std::lock_guard factory(impl_->state->derivative_factory_mutex);
  return {Impl::AffineLane::required(*impl_->state), impl_->affine_lane ? impl_->affine_lane->bytes() : 0};
}

/// Allocate the complete affine owner only after explicit idle-state and budget admission.
void PsiFormerComplexWF::prepareAffineScores(std::size_t maximum_bytes)
{
  prepareAffineScoresImpl(maximum_bytes, false);
}

/// Reuse the bounded CPU tape; this preparation allocates no device storage.
void PsiFormerComplexWF::prepareHybridAffineScores(std::size_t maximum_bytes)
{
  prepareAffineScoresImpl(maximum_bytes, true);
}

/// Require the correct entry point before admitting a reusable HOST owner.
void PsiFormerComplexWF::prepareAffineScoresImpl(std::size_t maximum_bytes, bool hybrid)
{
  NativeTimerScope timer(impl_->phase_timers.preparation);
  const auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (gpuInferenceEnabled() != hybrid)
    throw std::logic_error("Complex affine preparation requires its selected CPU or hybrid GPU family");
  if (hybrid && state->environment.periodic_profile &&
      !psiformer::matchesPairedDevicePairMetadata(state->environment))
    throw std::logic_error("Hybrid affine preparation requires the torus or selected paired GPU profile");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto required = Impl::AffineLane::required(*state);
  if (!maximum_bytes || maximum_bytes == std::numeric_limits<std::size_t>::max() || required > maximum_bytes)
    throw std::length_error("Complex affine numeric cap exceeded before preparation");
  if (state->pending_proposals.load(std::memory_order_relaxed))
    throw std::logic_error("Complex affine preparation requires an idle model family");
  std::lock_guard factory(state->derivative_factory_mutex);
  if (!impl_->affine_lane)
  {
    auto lane = std::make_shared<Impl::AffineLane>(*state);
    if (lane->bytes() != required)
      throw std::logic_error("Complex affine prepared storage differs from its admission");
    impl_->affine_lane = std::move(lane);
  }
}

/// Observe one retained sample without changing accepted inference caches or model parameters.
void PsiFormerComplexWF::observeComplexAffineScores(const ParticleSet& particles,
    wftrain::AffineScoreObserver& observer, std::vector<double>& real_score,
    std::vector<double>& imaginary_score) const
{
  observeComplexAffineScoresImpl(particles, observer, real_score, imaginary_score, false);
}

/// GPU publication and this HOST reverse share the model's parameter read authority.
void PsiFormerComplexWF::observeHybridComplexAffineScores(const ParticleSet& particles,
    wftrain::AffineScoreObserver& observer, std::vector<double>& real_score,
    std::vector<double>& imaginary_score) const
{
  observeComplexAffineScoresImpl(particles, observer, real_score, imaginary_score, true);
}

/// Run the common two-channel reverse without changing inference state or parameters.
void PsiFormerComplexWF::observeComplexAffineScoresImpl(const ParticleSet& particles,
    wftrain::AffineScoreObserver& observer, std::vector<double>& real_score,
    std::vector<double>& imaginary_score, bool hybrid) const
{
  NativeTimerScope timer(impl_->phase_timers.cpu_reverse);
  const auto state = impl_->state;
  std::shared_lock read(state->mutex);
  if (gpuInferenceEnabled() != hybrid)
    throw std::logic_error("Complex affine observations require their selected CPU or hybrid GPU family");
  impl_->requireIdle(*impl_->particles);
  Impl::validateDerivativeSource(*state);
  const auto n = state->model.ne, up = state->model.cfg.nup;
  if (state->pending_proposals.load(std::memory_order_relaxed) || particles.getActivePtcl() >= 0 ||
      particles.R.getUnit() != PosUnit::Cartesian || particles.isSpinor() || particles.getTotalNum() != n ||
      particles.groups() != 2 || particles.groupsize(0) != up || particles.groupsize(1) != n - up ||
      particles.R.size() != n || particles.GroupID.size() != n)
    throw std::invalid_argument("Complex affine sample/proposal/spin mismatch");
  validateRuntimeLattice(particles, state->environment, "Complex affine sample");
  for (std::size_t row = 0; row < n; ++row)
    for (int d = 0; d < 3; ++d)
      if (particles.GroupID[row] != (row < up ? 0 : 1) || !psiformer::determinant::isFiniteReal(particles.R[row][d]))
        throw std::invalid_argument("Complex affine coordinates/spin order invalid");
  std::shared_ptr<Impl::AffineLane> lane;
  {
    std::lock_guard factory(state->derivative_factory_mutex);
    lane = impl_->affine_lane;
  }
  if (!lane)
    throw std::logic_error("Complex affine scores must be explicitly prepared");
  std::lock_guard execution(lane->execution_mutex);
  Impl::AffineLane::ChannelSink real(*lane, observer, false), imaginary(*lane, observer, true);
  impl_->positions(*lane->work, particles);
  const auto result = lane->score.evaluate(*lane->work, &real, &imaginary);
  const auto channels = state->model.ndet * n, width = state->model.dim;
  for (std::size_t spin = 0; spin < 2; ++spin)
  {
    const std::size_t begin = spin ? up : 0, end = spin ? n : up;
    const auto role = spin ? psiformer::ParameterRole::BACKFLOW_DOWN_WEIGHT : psiformer::ParameterRole::BACKFLOW_UP_WEIGHT;
    for (std::size_t row = begin; row < end; ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        const auto sensitivity = result.head_sensitivities[(channel / n * n + row) * n + channel % n];
        const auto index = (row - begin) * channels + channel;
        lane->head_real[index] = sensitivity.real();
        lane->head_imaginary[index] = sensitivity.imag();
      }
    for (auto component : {psiformer::HeadTensorComponent::REAL, psiformer::HeadTensorComponent::IMAG})
    {
      const auto binding = std::find_if(lane->bindings.begin(), lane->bindings.end(), [&](const auto& candidate) {
        return candidate.role == role && candidate.component == component;
      });
      if (binding == lane->bindings.end())
        continue;
      if (component == psiformer::HeadTensorComponent::IMAG)
        for (std::size_t index = 0; index < (end - begin) * channels; ++index)
          lane->head_imaginary[index] = -lane->head_imaginary[index];
      observer.observe({binding->ordinal, result.head_activations + begin * width,
          component == psiformer::HeadTensorComponent::REAL ? lane->head_real.data() : lane->head_imaginary.data(),
          end - begin,
          component == psiformer::HeadTensorComponent::REAL ? lane->head_imaginary.data() : lane->head_real.data()});
    }
  }
  if (result.parameter_version != state->model.p.version() || result.parameter_count != state->schema->parameterCount())
    throw std::logic_error("Complex affine result has an inconsistent parameter identity");
  real_score.resize(result.parameter_count);
  imaginary_score.resize(result.parameter_count);
  for (std::size_t parameter = 0; parameter < result.parameter_count; ++parameter)
  {
    real_score[parameter] = result.parameter_score[parameter].real();
    imaginary_score[parameter] = result.parameter_score[parameter].imag();
  }
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
        InitializedNativeModel{std::move(detached), original->profile, original->seed, original->environment},
        *impl_->particles, *original->source, original->reduced_twist, 0, original->system_kind,
        original->cpu_budget ? original->cpu_budget->limit : 0);
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
      source_lane->preparation, source_lane->capacity, image, source_lane->virtual_capacity, source_lane->trace_targets,
      source_lane->virtual_envelope_reuse, source_lane->head_determinant_batch, source_lane->full_inverse_columns,
      source_lane->target_factor_reuse, source_lane->periodic_value_tiling));
  pf::PsiFormer model(std::move(parameters), std::move(*configuration), original->model.ndet,
                      original->model.dim, original->model.heads, original->model.blocks);
  auto state = std::make_shared<Impl::Model>(getName(),
      InitializedNativeModel{std::move(model), original->profile, original->seed, original->environment},
      *impl_->particles, *original->source, original->reduced_twist, original->gpu_limit, original->system_kind,
      0, std::move(lane));
  state->private_origin = original;
  state->private_origin_image = image;
  state->private_origin_version = version;
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
    if (captured->virtual_value)
      captured->virtual_value->validateModelRebind(successor);
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
    throw std::logic_error("Complex resource acquisition requires its idle native leader");
  for (std::size_t i = 0; i < components.size(); ++i)
  {
    const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[i]);
    if (!component || component->impl_->state != impl_->state)
      throw std::invalid_argument("Complex resource acquisition requires same-model component clones");
    if (impl_->state->cpu_budget)
      component->impl_->requireIdle(*component->impl_->particles);
    for (std::size_t prior = 0; prior < i; ++prior)
      if (&components[i] == &components[prior])
        throw std::invalid_argument("Complex resource acquisition contains a duplicate clone");
  }
  if (collection.getBatchResourcePreparationProvenance().state != BatchResourcePreparationState::UNPREPARED)
    throw std::invalid_argument("Complex resource acquisition does not support legacy batch_memory resources");
  auto count = impl_->state->acquired_teams.load();
  do
  {
    if (!impl_->state->cpu_budget && count >= impl_->state->gpu_crowds)
      throw std::logic_error("Complex GPU acquisition exceeds its frozen crowd limit");
  } while (!impl_->state->acquired_teams.compare_exchange_weak(count, count + 1));
  const auto cursor = collection.getCursor();
  ResourceHandle<Impl::MultiWalkerResource> candidate;
  try
  {
    for (std::size_t i = 0; i < components.size(); ++i)
      if (static_cast<PsiFormerComplexWF&>(components[i]).impl_->acquired_leader)
        throw std::logic_error("Complex clone already belongs to an acquired crowd");
    candidate = collection.lendResource<Impl::MultiWalkerResource>();
    if (candidate.getResource().state != impl_->state)
      throw std::invalid_argument("Complex resource belongs to another model family");
    if (impl_->state->cpu_budget)
      candidate.getResource().prepareCpuTeam(components);
    else
      candidate.getResource().prepareTeam(components);
    auto& resource = candidate.getResource();
#ifdef ENABLE_CUDA
    if (!impl_->state->cpu_budget && impl_->state->gpu_crowds > 1 && !resource.gpu)
    {
      const auto& primary = *impl_->state->gpu;
      resource.gpu = Impl::GpuLane::prepareFrozen(impl_->state->plan,
          impl_->state->model.p.flat_values(), impl_->state->model.cfg.nuclei.x,
          impl_->state->model.p.version(), impl_->state->gpu_limit,
          primary.preparation, primary.capacity, primary.image, primary.virtual_capacity, primary.trace_targets,
          primary.virtual_envelope_reuse, primary.head_determinant_batch, primary.full_inverse_columns,
          primary.target_factor_reuse, primary.periodic_value_tiling);
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
    throw std::logic_error("Complex resource release requires its original loan and cursor");
  auto& resource = impl_->multiwalker.getResource();
  if (components.size() != resource.components.size())
    throw std::invalid_argument("Complex resource release changed the acquired crowd extent");
  for (std::size_t i = 0; i < components.size(); ++i)
    if (&components[i] != resource.components[i] || resource.components[i]->impl_->acquired_leader != this ||
        resource.components[i]->impl_->acquired_lane != i)
      throw std::invalid_argument("Complex resource release changed its acquired lane order");
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

/// Shared CPU and native GPU families retain bounded complete-configuration products.
bool PsiFormerComplexWF::supportsMultiParticleMoves() const noexcept
{
  return bool(impl_->state->cpu_budget) || impl_->state->gpu_limit != 0;
}

/// Retain pure FULL products until a matching collective resolution, with all-output atomicity.
void PsiFormerComplexWF::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& components, const RefVectorWithLeader<ParticleSet>& targets,
    const MCMultiParticleMoves<CoordsType::POS>& moves, std::vector<LogValue>& log_ratios,
    const RefVector<ParticleSet::ParticleGradient>& gradients,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacians) const
{
  if (!supportsMultiParticleMoves())
    throw std::logic_error("Complex collective proposals require shared CPU or native GPU inference");
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
  if (impl_->state->cpu_budget)
    impl_->stageCpuCollective(resource, targets, moves);
  else
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
    throw std::logic_error("Complex collective resolution requires shared CPU or native GPU inference");
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
  if (!fromscratch && !components.empty())
  {
    std::shared_lock read(impl_->state->mutex);
    if (&components.getLeader() != this || targets.size() != components.size() ||
        gradients.size() != components.size() || laplacians.size() != components.size())
      throw std::invalid_argument("Complex accepted FULL observation leader/list mismatch");
    if (impl_->denseEnabled() || impl_->acquired_leader)
      impl_->requireDenseTeam(components, targets);
    Impl::validateDerivativeSource(*impl_->state);
    bool current = true;
    for (std::size_t i = 0; i < components.size(); ++i)
    {
      const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[i]);
      if (!component || component->impl_->state != impl_->state)
        throw std::invalid_argument("Complex accepted FULL observation requires one model family");
      const auto& clone = *component->impl_;
      clone.requireNoCollective();
      clone.validate(targets[i]);
      current = current && !clone.pending && targets[i].getActivePtcl() < 0 && clone.fullMatches(targets[i]);
      for (std::size_t prior = 0; prior < i; ++prior)
        if (&components[prior] == &components[i] || &targets[prior] == &targets[i])
          throw std::invalid_argument("Complex accepted FULL observation aliases walkers");
    }
    if (current)
    {
      for (std::size_t i = 0; i < components.size(); ++i)
      {
        const auto& clone = *static_cast<const PsiFormerComplexWF&>(components[i]).impl_;
        clone.validateOwnScatter(gradients[i], laplacians[i]);
        const auto g = checkedMemoryRange(gradients[i].get().data(), gradients[i].get().size(),
            "Complex accepted gradient output range overflow");
        const auto l = checkedMemoryRange(laplacians[i].get().data(), laplacians[i].get().size(),
            "Complex accepted Laplacian output range overflow");
        if (memoryRangesOverlap(g, l))
          throw std::invalid_argument("Complex accepted FULL observation outputs overlap");
        for (std::size_t prior = 0; prior < i; ++prior)
        {
          const auto pg = checkedMemoryRange(gradients[prior].get().data(), gradients[prior].get().size(),
              "Complex accepted gradient output range overflow");
          const auto pl = checkedMemoryRange(laplacians[prior].get().data(), laplacians[prior].get().size(),
              "Complex accepted Laplacian output range overflow");
          if (memoryRangesOverlap(g, pg) || memoryRangesOverlap(g, pl) ||
              memoryRangesOverlap(l, pg) || memoryRangesOverlap(l, pl))
            throw std::invalid_argument("Complex accepted FULL observation walker outputs overlap");
        }
      }
      for (std::size_t i = 0; i < components.size(); ++i)
        static_cast<const PsiFormerComplexWF&>(components[i]).impl_->scatterOwn(gradients[i], laplacians[i]);
      return;
    }
  }
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
PsiFormerComplexWF::LogValue PsiFormerComplexWF::evaluateGL(const ParticleSet& p,
    ParticleSet::ParticleGradient& g, ParticleSet::ParticleLaplacian& l, bool fromscratch)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  impl_->validate(p);
  Impl::validateDerivativeSource(*impl_->state);
  if (!fromscratch && !impl_->pending && p.getActivePtcl() < 0 && impl_->fullMatches(p))
  {
    impl_->validateOwnScatter(g, l);
    impl_->scatterOwn(g, l);
    return log_value_;
  }
  return evaluateLogLocked(p, g, l);
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
  if (!psiformer::determinant::isFiniteReal(potential))
    throw std::invalid_argument("Complex PsiFormer Coulomb potential must be finite and from this sample");
  const LogValue kinetic = impl_->observeOwnKinetic(p);
  const LogValue local_energy = kinetic + potential; // caller guarantees Coulomb V belongs to this sample
  if (!isFiniteWavefunctionValue(kinetic) || !isFiniteWavefunctionValue(local_energy))
    throw std::overflow_error("Complex PsiFormer Coulomb observation overflow");
  return {kinetic, local_energy, potential};
}

/// Observe only a bound accepted grid; REAL Hamiltonian properties remain untouched.
PsiFormerComplexWF::CapturedLocalEnergyObservation PsiFormerComplexWF::observeCapturedLocalEnergy(
    const ParticleSet& p, const wftrain::StreamingDerivativeOperator& op, std::size_t sample,
    FullPrecRealType potential) const
{
#ifdef ENABLE_CUDA
  const auto* stream = dynamic_cast<const Impl::GpuDerivativeStream*>(&op);
  if (!stream || stream->lane->state != impl_->state)
    throw std::invalid_argument("Complex captured observation requires its own GPU derivative family");
  std::unique_lock execution(stream->lane->execution_mutex);
  std::shared_lock read(impl_->state->mutex);
  stream->requireCurrent();
  if (!stream->replay || sample >= stream->samples || !psiformer::determinant::isFiniteReal(potential))
    throw std::invalid_argument("Complex captured observation requires a finite same-sample potential and bound grid");
  impl_->requireIdle(p);
  const auto& masses = p.get_mass_by_group();
  if (masses.size() != static_cast<std::size_t>(p.groups()))
    throw std::invalid_argument("Complex captured observation mass extent changed");
  for (int group = 0; group < p.groups(); ++group)
    for (int e = p.first(group); e < p.last(group); ++e)
      if (!psiformer::determinant::isFiniteReal(masses[group]) || masses[group] <= 0 ||
          1.0 / masses[group] != stream->inverse_masses[sample * p.R.size() + e])
        throw std::invalid_argument("Complex captured observation kinetic masses changed");
  for (std::size_t e = 0; e < p.R.size(); ++e)
    for (int d = 0; d < 3; ++d)
      if (p.R[e][d] != stream->positions[3 * (sample * p.R.size() + e) + d])
        throw std::invalid_argument("Complex captured observation reference coordinates changed");
  const auto kinetic = impl_->observeOwnKinetic(p);
  LogValue nonlocal{};
  for (std::size_t q = 0; q < stream->references.size(); ++q)
    if (stream->references[q] == sample)
    {
      nonlocal += stream->complete_weights[q];
      if (!isFiniteWavefunctionValue(nonlocal)) throw std::overflow_error("Complex captured energy overflow");
    }
  // Re(nonlocal) is ALREADY in the native same-sample local potential.
  const auto energy = kinetic + potential + LogValue(0, nonlocal.imag());
  if (!isFiniteWavefunctionValue(energy)) throw std::overflow_error("Complex captured local energy overflow");
  return {kinetic, nonlocal, energy, potential, stream->version, stream->replay->gridFingerprint()};
#else
  throw std::logic_error("Complex captured energy observation requires the explicit GPU component route");
#endif
}

/// Immutable owner-local query; native PP/replay arrays remain once-per-shared-owner charges in the caller ledger.
PsiFormerComplexWF::FixedGridEnergyReferenceRequirements PsiFormerComplexWF::fixedGridEnergyReferenceRequirements() const
{
#ifdef ENABLE_CUDA
  std::shared_lock read(impl_->state->mutex);
  if (!impl_->state->gpu) throw std::logic_error("Fixed-grid reference requirements need a GPU family");
  return FixedGridEnergyReference::requirements(impl_->state->model.ne, impl_->state->model.ne);
#else
  throw std::logic_error("Fixed-grid candidate energy requires the explicit GPU component route");
#endif
}

/// Seal only a bound, native-produced replay; copied scientific strings and synthetic grids cannot impersonate it.
std::shared_ptr<const PsiFormerComplexWF::FixedGridEnergyReference>
PsiFormerComplexWF::makeBudgetedFixedGridEnergyReference(const ParticleSet& p,
    const wftrain::StreamingDerivativeOperator& op, std::size_t sample,
    std::shared_ptr<const wftrain::NonLocalECPNativeCapture> native, std::size_t host_cap) const
{
#ifdef ENABLE_CUDA
  const auto* stream = dynamic_cast<const Impl::GpuDerivativeStream*>(&op);
  if (!stream || stream->lane->state != impl_->state || !native || !host_cap ||
      host_cap == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Fixed-grid energy reference requires its own captured GPU family and finite HOST cap");
  std::unique_lock execution(stream->lane->execution_mutex);
  std::shared_lock read(impl_->state->mutex);
  stream->requireCurrent();
  auto replay = std::dynamic_pointer_cast<const Impl::GpuNonLocalReplay>(stream->replay);
  if (!replay || sample >= stream->samples || native->captureIdentity() != replay->capture_identity ||
      native->gridFingerprint() != replay->grid || native->reportedPotentials().size() != stream->samples ||
      native->scientificDefinition().empty())
    throw std::invalid_argument("Fixed-grid energy reference requires an authoritative same-traversal native receipt");
  const char* error = "Fixed-grid energy reference HOST admission overflow";
  const auto required = FixedGridEnergyReference::requirements(impl_->state->model.ne, impl_->state->model.ne);
  auto bytes = pf::checkedStorageSum(required.new_reference_host_bytes, native->hostBytes(), error);
  pf::addStorageBytes(bytes, required.shared_replay_descriptor_bytes, error);
  pf::addStorageBytes(bytes, replay->storageDiagnostics().retained_numeric_bytes, error);
  if (bytes > host_cap) throw std::length_error("Fixed-grid energy reference exceeds HOST cap before allocation: requested=" +
      std::to_string(bytes) + ", cap=" + std::to_string(host_cap));
  impl_->observeOwnKinetic(p); // Require current accepted FULL products, not an arbitrary VALUE reference.
  auto result = std::make_shared<FixedGridEnergyReference>();
  result->source = impl_->state; result->image = impl_->state->gpu->image;
  result->replay = std::move(replay); result->native = std::move(native); result->sample = sample;
  result->inverse_masses.assign(stream->inverse_masses.begin() + sample * p.R.size(),
      stream->inverse_masses.begin() + (sample + 1) * p.R.size());
  result->groups.assign(p.GroupID.begin(), p.GroupID.end());
  result->validateParticles(*impl_, p);
  const auto measured = FixedGridEnergyReference::requirements(result->inverse_masses.capacity(), result->groups.capacity());
  auto actual = pf::checkedStorageSum(measured.new_reference_host_bytes, result->native->hostBytes(), error);
  pf::addStorageBytes(actual, measured.shared_replay_descriptor_bytes, error);
  pf::addStorageBytes(actual, result->replay->storageDiagnostics().retained_numeric_bytes, error);
  if (actual > host_cap) throw std::length_error("Fixed-grid energy reference actual capacities exceed HOST cap");
  app_log() << "Complex fixed-grid reference: HOST_retained_bytes=" << actual
            << "; shared_replay_and_PP_counted_once=yes; DEVICE_additional_bytes=0" << std::endl;
  return result;
#else
  throw std::logic_error("Fixed-grid candidate energy requires the explicit GPU component route");
#endif
}

/// Lock both authorities before the private lane; every point uses the same complete VALUE ratio as public VP.
PsiFormerComplexWF::FixedGridCandidateEnergy PsiFormerComplexWF::evaluateFixedGridCandidateEnergy(
    const std::shared_ptr<const FixedGridEnergyReference>& reference, const PsiFormerComplexWF& candidate,
    const ParticleSet& p) const
{
#ifdef ENABLE_CUDA
  if (!reference || candidate.impl_->state == impl_->state)
    throw std::invalid_argument("Fixed-grid candidate energy requires a sealed reference and distinct private family");
  std::shared_lock source_read(impl_->state->mutex, std::defer_lock);
  std::shared_lock candidate_read(candidate.impl_->state->mutex, std::defer_lock);
  std::lock(source_read, candidate_read);
  reference->validate(*impl_, *candidate.impl_);
  reference->validateParticles(*candidate.impl_, p);
  candidate.impl_->requireNoCollective();
  const auto kinetic = candidate.impl_->observeOwnKinetic(p);
  candidate.impl_->formRatio(candidate.impl_->accepted, p, 0, &p.R[0]); // Zero accepted denominator always rejects.
  LogValue source_nonlocal{}, trial_nonlocal{};
  for (const auto& point : reference->replay->point_records)
    if (point.sample == reference->sample)
    {
      ParticleSet::PosType absolute;
      for (int d = 0; d < 3; ++d) absolute[d] = point.replacement_position[d];
      const auto value = candidate.impl_->core(p, false, point.electron, -1, &absolute);
      const auto ratio = candidate.impl_->formRatio(value.amplitude, p, point.electron, &absolute);
      source_nonlocal += point.bare_weight * point.complete_ratio;
      trial_nonlocal += point.bare_weight * ratio; // Also evaluate bare0; a zero virtual numerator is valid.
      if (!isFiniteWavefunctionValue(source_nonlocal) || !isFiniteWavefunctionValue(trial_nonlocal))
        throw std::overflow_error("Fixed-grid candidate nonlocal energy overflow");
    }
  const auto energy = kinetic + (reference->native->reportedPotentials()[reference->sample] - source_nonlocal.real()) + trial_nonlocal;
  if (!isFiniteWavefunctionValue(energy)) throw std::overflow_error("Fixed-grid candidate complete energy overflow");
  FixedGridCandidateEnergy result;
  result.kinetic = kinetic; result.nonlocal = trial_nonlocal; result.local_energy = energy;
  result.source_version = reference->replay->version; result.candidate_version = candidate.impl_->state->model.p.version();
  result.grid_fingerprint = reference->replay->grid; result.reference = reference;
  result.candidate_state = candidate.impl_->state; result.candidate_image = candidate.impl_->state->gpu->image;
  return result;
#else
  throw std::logic_error("Fixed-grid candidate energy requires the explicit GPU component route");
#endif
}

/// No numerical call or allocation is needed to reject stale/foreign complete-energy receipts.
void PsiFormerComplexWF::validateFixedGridCandidateEnergy(const FixedGridCandidateEnergy& result,
    const PsiFormerComplexWF& candidate) const
{
#ifdef ENABLE_CUDA
  if (!result.reference || candidate.impl_->state == impl_->state)
    throw std::invalid_argument("Fixed-grid candidate energy receipt is empty or foreign");
  std::shared_lock source_read(impl_->state->mutex, std::defer_lock);
  std::shared_lock candidate_read(candidate.impl_->state->mutex, std::defer_lock);
  std::lock(source_read, candidate_read);
  result.reference->validate(*impl_, *candidate.impl_);
  result.reference->validateParticles(*candidate.impl_, *candidate.impl_->particles);
  if (result.candidate_state != candidate.impl_->state || result.candidate_image != candidate.impl_->state->gpu->image ||
      result.candidate_version != candidate.impl_->state->model.p.version() ||
      result.source_version != result.reference->replay->version || result.grid_fingerprint != result.reference->replay->grid)
    throw std::logic_error("Fixed-grid candidate energy receipt generation is stale/foreign");
#else
  throw std::logic_error("Fixed-grid candidate energy requires the explicit GPU component route");
#endif
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

/// Observe complete single-replacement ratios at one absolute position without publishing proposals.
void PsiFormerComplexWF::evaluateRatiosAlltoOne(ParticleSet& particles, std::vector<ValueType>& ratios)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  impl_->requireIdle(particles);
  Impl::validateDerivativeSource(*impl_->state);
  if (particles.isSpinor() || ratios.size() != static_cast<std::size_t>(particles.getTotalNum()))
    throw std::invalid_argument("Complex PsiFormer all-to-one VALUE shape/spin mismatch");
  const auto position = particles.getActivePos();
  for (int d = 0; d < 3; ++d)
    if (!psiformer::determinant::isFiniteReal(position[d]))
      throw std::invalid_argument("Complex PsiFormer all-to-one VALUE position is nonfinite");
  // Validate the denominator before using the shared numerical owner or publishing output.
  impl_->formRatio(impl_->accepted, particles, 0, &particles.R[0]);
  if (ratios.size() > (impl_->numeric_limit - impl_->numeric_required) / sizeof(ValueType))
    throw std::length_error("Complex PsiFormer all-to-one VALUE staging exceeds clone numeric cap");
  std::vector<ValueType> staged(ratios.size());
  if (!impl_->streamVirtualValues(particles, [](std::size_t point) { return static_cast<int>(point); },
      [&](std::size_t) { return position; }, staged))
    for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    {
      const auto evaluated = impl_->core(particles, false, electron, -1, &position);
      staged[electron] = impl_->formRatio(evaluated.amplitude, particles, electron, &position);
    }
  std::copy(staged.begin(), staged.end(), ratios.begin());
}

/// Evaluate pure wavefunction VALUE ratios without moving or publishing the accepted walker.
void PsiFormerComplexWF::evaluateRatios(const VirtualParticleSet& virtuals, std::vector<ValueType>& ratios)
{
  std::shared_lock read(impl_->state->mutex);
  impl_->requireNoCollective();
  const auto& reference = virtuals.getRefPS();
  impl_->requireIdle(reference);
  Impl::validateDerivativeSource(*impl_->state);
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
  if (!impl_->streamVirtualValues(reference, [&](std::size_t) { return virtuals.refPtcl; },
      [&](std::size_t point) { return virtuals.R[point]; }, staged))
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
#ifdef ENABLE_CUDA
  if (state->gpu && state->gpu->virtual_capacity)
  {
    std::shared_lock read(state->mutex);
    impl_->requireNoCollective();
    if (&components.getLeader() != this || components.size() != batch.walkerCount() ||
        particles.size() != components.size() || scratch.size() != components.size())
      throw std::invalid_argument("Complex virtual tile leader/list extent mismatch");
    batch.validateOutputExtent(ratios.size());
    batch.validateFor(particles, state->source->getTotalNum());
    for (const auto& position : batch.absolutePositions())
      for (int dimension = 0; dimension < 3; ++dimension)
        if (!psiformer::determinant::isFiniteReal(position[dimension]))
          throw std::invalid_argument("Complex virtual tile positions must be finite");
    for (std::size_t walker = 0; walker < components.size(); ++walker)
    {
      const auto* component = dynamic_cast<const PsiFormerComplexWF*>(&components[walker]);
      if (!component || component->impl_->state != state || component->impl_->particles != &particles[walker])
        throw std::invalid_argument("Complex virtual tile requires exact same-family reference clones");
      for (std::size_t other = 0; other < walker; ++other)
        if (&components[walker] == &components[other] || &scratch[walker] == &scratch[other])
          throw std::invalid_argument("Complex virtual tile requires distinct clones and scratch objects");
      for (std::size_t reference = 0; reference < components.size(); ++reference)
        if (static_cast<const ParticleSet*>(&scratch[walker]) == &particles[reference])
          throw std::invalid_argument("Complex virtual tile scratch aliases a reference walker");
      if (scratch[walker].isSpinor() != particles[walker].isSpinor())
        throw std::invalid_argument("Complex virtual tile scratch/reference spinor mismatch");
      component->impl_->requireIdle(particles[walker]);
      component->impl_->validate(particles[walker], 0);
      component->impl_->formRatio(component->impl_->accepted, particles[walker], 0, &particles[walker].R[0]);
    }
    const auto bytes = pf::checkedStorageBytes<ValueType>(
        pf::checkedStorageProduct(4, batch.size(), "Complex virtual tile staging overflow"),
        "Complex virtual tile staging overflow");
    if (bytes > impl_->numeric_limit - impl_->numeric_required)
      throw std::length_error("Complex virtual tile output exceeds clone numeric staging cap");
    version = state->model.p.version();
    auto permit = state->inference_concurrency.acquire();
    const auto owner = impl_->selectedGpuLane();
    auto& lane = *owner;
    std::unique_lock lease(lane.mutex);
    impl_->bindGpuLane(lane);
    if (!lane.virtual_value || lane.virtual_capacity != state->gpu->virtual_capacity)
      throw std::logic_error("Complex virtual tile owner differs from its frozen family");
    std::vector<ValueType> staged(ratios.size());
    std::size_t count = 0;
    using EnvelopeRequest = psiformer::device::PreparedDeviceComplexValue::EnvelopeRequest;
    std::optional<EnvelopeRequest> envelope_request;
    std::size_t envelope_walker = components.size();
    const auto end_envelope_request = [&] {
      lane.virtual_value->endVirtualEnvelopeBase();
      envelope_request.reset();
    };
    const auto flush = [&] {
      if (!count)
        return;
      if (lane.virtual_envelope_reuse)
      {
        const auto first_walker = lane.virtual_indices[0][1];
        std::size_t reusable = 0;
        for (std::size_t local = 0; local < count; ++local)
          reusable += lane.virtual_indices[local][1] == first_walker;
        if (reusable < 2)
          end_envelope_request();
        else if (!envelope_request || envelope_walker != first_walker)
        {
          end_envelope_request();
          envelope_walker = first_walker;
          const auto& reference = particles[first_walker];
          for (std::size_t electron = 0; electron < lane.electrons; ++electron)
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              lane.positions[3 * electron + dimension] = reference.R[electron][dimension];
          envelope_request = lane.virtual_value->beginVirtualEnvelopeBase(lane.positions.data(), 3 * lane.electrons);
        }
      }
      lane.virtual_positions.resize(3 * lane.electrons * count);
      lane.virtual_amplitude.resize(count);
      {
        NativeTimerScope numerical(impl_->phase_timers.gpu_evaluate);
        if (envelope_request)
          lane.virtual_value->evaluateVirtualTile(*envelope_request, lane.virtual_positions,
              [&](std::size_t local) {
                return lane.virtual_indices[local][1] == envelope_walker ? lane.virtual_indices[local][2] : lane.electrons;
              });
        else
          lane.virtual_value->evaluate(lane.virtual_positions);
      }
      {
        NativeTimerScope download(impl_->phase_timers.gpu_download);
        lane.virtual_value->download(lane.virtual_value->resultView(), lane.virtual_amplitude);
      }
      for (std::size_t local = 0; local < count; ++local)
      {
        const auto& index = lane.virtual_indices[local];
        const auto& result = lane.virtual_amplitude[local];
        const Impl::Amplitude amplitude{{result.phase.x, result.phase.y}, result.log_abs};
        const auto& component = static_cast<const PsiFormerComplexWF&>(components[index[1]]);
        const auto& position = batch.absolutePositions()[index[0]];
        staged[index[0]] = component.impl_->formRatio(amplitude, particles[index[1]], index[2], &position);
      }
      count = 0;
    };
    try
    {
      NativeTimerScope value_timer(impl_->phase_timers.value);
      for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
      {
        const auto slice = batch.slice(segment);
        for (std::size_t point = 0; point < slice.size(); ++point)
        {
          // Every packed configuration starts from its own immutable walker base.
          lane.virtual_positions.resize(3 * lane.electrons * lane.virtual_capacity);
          const auto offset = 3 * lane.electrons * count;
          const auto& reference = particles[slice.walkerId()];
          for (std::size_t electron = 0; electron < lane.electrons; ++electron)
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              lane.virtual_positions[offset + 3 * electron + dimension] = reference.R[electron][dimension];
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            lane.virtual_positions[offset + 3 * slice.electronId() + dimension] = slice.absolutePosition(point)[dimension];
          lane.virtual_indices[count++] = {slice.flatOffset() + point,
              static_cast<std::size_t>(slice.walkerId()), static_cast<std::size_t>(slice.electronId())};
          if (count == lane.virtual_capacity)
            flush();
        }
      }
      flush();
      end_envelope_request();
      Impl::validateDerivativeSource(*state);
      for (std::size_t walker = 0; walker < components.size(); ++walker)
      {
        const auto& component = static_cast<const PsiFormerComplexWF&>(components[walker]);
        component.impl_->requireIdle(particles[walker]);
        component.impl_->validate(particles[walker], 0);
        component.impl_->formRatio(component.impl_->accepted, particles[walker], 0, &particles[walker].R[0]);
      }
      if (state->model.p.version() != version || state->pending_proposals.load(std::memory_order_relaxed))
        throw std::logic_error("Complex virtual tile source/version changed before publication");
      ratios.swap(staged);
      return EvaluationStamp::versioned(state.get(), static_cast<std::uint64_t>(version));
    }
    catch (...)
    {
      end_envelope_request();
      lane.virtual_value->discardResults();
      throw;
    }
  }
#endif
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
