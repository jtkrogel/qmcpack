//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_H
#include "PsiFormerDeviceKineticTrunk.h"
#include "PsiFormerDeviceComplexValue.h"
#include "PsiFormerDeviceComplexSpatialKernels.h"
#include "PsiFormerDeviceComplexKineticKernels.h"
#include <cstdlib>
#include <string_view>

namespace qmcplusplus::psiformer::device
{
/// Capture one strict cold factor policy; the original scalar route remains the default.
inline bool configuredComplexKineticCooperativeLu()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_KINETIC_COOPERATIVE_LU");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_KINETIC_COOPERATIVE_LU must be 0 or 1");
  }();
  return enabled;
}

/// Capture the channel-workspace policy before enclosing allocation admission.
inline bool configuredComplexKineticChannelRoots()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_KINETIC_CHANNEL_ROOTS");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_KINETIC_CHANNEL_ROOTS must be 0 or 1");
  }();
  return enabled;
}

/// Bound existing qualified factor composition to fixed legacy-torus paired FP64 owners.
inline bool complexKineticCooperativeLuEligible(const PsiFormerExecutionPlan& plan)
{
  const auto& environment = plan.environment();
  const auto& shape = plan.modelShape();
  return environment.boundary == BoundaryCondition::PERIODIC && environment.fixed_nuclei &&
      environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX &&
      environment.parameter_scalar_domain == ScalarDomain::REAL && environment.compute_scalar_domain == ScalarDomain::REAL &&
      environment.amplitude_scalar_domain == ScalarDomain::COMPLEX &&
      environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 && !environment.periodic_profile &&
      environment.periodic_axes[0] && environment.periodic_axes[1] && environment.periodic_axes[2] &&
      shape.electrons() >= 64 && shape.electrons() <= 300 && shape.determinants >= 1 && shape.determinants <= 16;
}

/// Standalone max-K numeric inventory; image/vendor/object metadata are excluded.
struct ComplexKineticCost
{
  KineticTrunkCost trunk;
  std::size_t device_bytes, host_bytes, canonical_bytes, matrix_work, matrices, head_elements;
  std::size_t private_features, private_parameters, root_scratch;
  bool parallel_head, cooperative_factor, channel_roots;
};

/// Exact combined preflight: one real trunk/P, max-K mixed jets and complexP.
inline ComplexKineticCost estimateComplexKineticCost(const PsiFormerExecutionPlan& plan, std::size_t targets,
    bool parallel_requested = complexKineticHeadParallelRequested(),
    bool cooperative_requested = configuredComplexKineticCooperativeLu(),
    bool channel_roots_requested = configuredComplexKineticChannelRoots())
{
  complex_detail::baseCost(plan, 1, SpatialJetMode::VALUE_ONLY);
  const auto t = estimateKineticTrunkCost(plan, targets);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex kinetic extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex kinetic bytes overflow"); };
  const auto& s = plan.modelShape();
  const auto n = s.electrons(), d = s.determinants, p = plan.parameterCount(), nn = mul(n, n);
  const auto m = mul(d, nn), h = mul(add(1, mul(4, targets)), m), tiles = n / targets + (n % targets != 0);
  const bool channel_roots = channel_roots_requested && d >= 2 && complexKineticCooperativeLuEligible(plan);
  const auto root_scratch = mul(3, channel_roots ? m : nn);
  const auto largest = std::max(h, p);
  if (largest - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (largest - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Complex kinetic launch/status extent exceeds admission");
  auto bytes = add(t.device_bytes, add(mul(56, h), mul(16, m)));
  bytes = add(bytes, add(mul(16 + sizeof(std::size_t), mul(d, n)), mul(sizeof(cuDoubleComplex), root_scratch)));
  if (channel_roots)
    bytes = add(bytes, sizeof(std::uint64_t));
  bytes = add(bytes, add(mul(144, targets), add(mul(16, p), add(mul(8, n), add(mul(32, d), 96)))));
  // Only ordinary fixed legacy-torus FP64 owners enter this independent-output route.
  const auto& environment = plan.environment();
  const bool parallel_head = parallel_requested && n >= 64 && n <= 300 && d <= 16 &&
      !environment.periodic_profile && environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      environment.periodic_axes[0] && environment.periodic_axes[1] && environment.periodic_axes[2];
  const auto private_features = parallel_head ? mul(add(1, mul(4, targets)), mul(n, s.feature_dimension)) : 0;
  const auto private_parameters = parallel_head ? p : 0;
  if (parallel_head)
    bytes = add(bytes, add(mul(8, add(private_features, private_parameters)), sizeof(std::uint64_t)));
  // Matrix-contraction sites only, not neural/head FLOPs or a throughput estimate.
  const auto work = add(mul(14, mul(d, mul(nn, nn))), mul(4, mul(d, mul(tiles, mul(nn, n)))));
  return {t, bytes, add(t.host_bytes, sizeof(ComplexValueResult)), mul(24, p), work, m, h,
      private_features, private_parameters, root_scratch, parallel_head,
      cooperative_requested && complexKineticCooperativeLuEligible(plan), channel_roots};
}

/** B1 numerical dK/d(real parameters), with learned paired-head complex phase.
 * No additional per-orbital reciprocal/Bloch factor or public training route.
 * Caller queue/BLAS and exclusive image-publication/execution lease outlive the
 * owner. Finite unweighted root factors are required; no Wide-domain fallback.
 */
class PreparedDeviceComplexKinetic
{
public:
  /// Borrow only the complete canonical response; scalar/amplitude use explicit download.
  struct ResultView
  {
    const PreparedDeviceComplexKinetic* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, count;
    const cuDoubleComplex* data;
  };

  /// Reject both whole-request caps BEFORE any trunk or mixed numeric allocation.
  static std::unique_ptr<PreparedDeviceComplexKinetic> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t targets, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
      std::size_t budget, std::size_t matrix_work_budget,
      bool parallel_requested = complexKineticHeadParallelRequested(),
      bool cooperative_requested = configuredComplexKineticCooperativeLu(),
      bool channel_roots_requested = configuredComplexKineticChannelRoots())
  {
    if (!image)
      throw std::invalid_argument("Complex kinetic requires a paired numerical image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexKineticCost(image->plan(), targets, parallel_requested, cooperative_requested,
        channel_roots_requested);
    if (cost.device_bytes > budget || cost.matrix_work > matrix_work_budget)
      throw std::length_error("Complex kinetic exceeds its per-owner byte or matrix-work cap");
    return std::unique_ptr<PreparedDeviceComplexKinetic>(new PreparedDeviceComplexKinetic(image, targets, queue, blas, cost));
  }

  /// Expose the admitted immutable factor policy without borrowing numeric results.
  bool usesCooperativeFactorization() const { return cost_.cooperative_factor; }

  /// Expose the frozen channel policy without lending its private workspace.
  bool usesChannelRoots() const { return cost_.channel_roots; }

  /// Reuse each immediate K tape for BOTH objectives before releasing its ticket.
  void evaluate(const std::vector<double>& positions, const std::vector<double>& inverse_masses,
      const std::array<double, 3>& kappa)
  {
    beginRequest(positions, inverse_masses, kappa);
    std::size_t covered = 0, ordinal = 0;
    trunk_->withTargets(positions, [&](const auto& ticket) {
      if (ticket.image != image_.get() || ticket.ordinal != ordinal || ticket.features.target_electron_begin != covered)
        throw std::logic_error("Complex kinetic target/image sequence mismatch");
      for (std::size_t component = 0; component < 2; ++component)
      {
        trunk_->reverseSeeded(ticket, [&](const auto& current, double* feature, double* cusp, double* response) {
          seed(current, component, feature, cusp, response);
        }, true);
        finishPass(component);
      }
      covered += ticket.features.laplacian_lanes;
      ++ordinal;
    });
    if (covered != image_->plan().modelShape().electrons())
      throw std::logic_error("Complex kinetic target coverage is incomplete");
    ready_ = true;
  }

  /// Revoke products before atomically eligible same-family image replacement.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    invalidate();
    image_->validateSuccessor(successor);
    trunk_->rebindModel(successor);
    image_ = std::move(successor);
  }

  /// Expose no ticket/head/root/scratch or partially accumulated complexP.
  ResultView resultView() const
  {
    image_->validateCurrentDevice();
    if (!ready_)
      throw std::logic_error("Complex kinetic response unavailable");
    return {this, image_.get(), generation_, response_.capacity(), response_.data()};
  }

  /// Download canonical P only at the explicit host publication boundary.
  void download(const ResultView& view, std::vector<cuDoubleComplex>& response, cuDoubleComplex& kinetic,
      ComplexValueResult& amplitude) const
  {
    image_->validateCurrentDevice();
    if (!ready_ || view.owner != this || view.image != image_.get() || view.generation != generation_ ||
        view.count != response_.capacity() || view.data != response_.data() || response.size() != view.count)
      throw std::invalid_argument("Complex kinetic response is stale, foreign or wrong extent");
    response_.download(response.data(), response.size());
    energy_.download(&kinetic, 1);
    amplitude = first_amplitude_; // CORE+cusp, without the common outer phase.
  }

  /// Report numeric ownership separately from shared image/vendor/object metadata.
  const ComplexKineticCost& cost() const noexcept { return cost_; }

private:
  friend struct ComplexKineticOwnerValidationAccess;

  /// Allocate only the exact preflight inventory; one realP is already in trunk.
  PreparedDeviceComplexKinetic(std::shared_ptr<const PreparedDeviceModel> image, std::size_t targets,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, ComplexKineticCost cost)
      : cost_(cost), image_(image), queue_(queue), blas_(blas),
        trunk_(PreparedDeviceKineticTrunk::prepare(image, targets, cost.trunk.device_bytes, queue, blas)),
        layout_(makeOpenOrbitalJetLayout(1, image->plan().modelShape().determinants, image->plan().modelShape().electrons(),
            image->plan().modelShape().nuclei, image->plan().modelShape().feature_dimension,
            image->plan().modelShape().spin_up_electrons, SpatialJetMode::FULL_VGL)),
        envelope_(cost.head_elements, "ComplexKinetic::envelope_", __FILE__, __LINE__),
        projections_(2 * cost.head_elements, "ComplexKinetic::projections_", __FILE__, __LINE__),
        head_(cost.head_elements, "ComplexKinetic::head_", __FILE__, __LINE__),
        roots_(cost.head_elements, "ComplexKinetic::roots_", __FILE__, __LINE__),
        inverse_(cost.matrices, "ComplexKinetic::inverse_", __FILE__, __LINE__),
        solve_(image->plan().modelShape().determinants * image->plan().modelShape().electrons(), "ComplexKinetic::solve_", __FILE__, __LINE__),
        scratch_(cost.root_scratch, "ComplexKinetic::scratch_", __FILE__, __LINE__),
        outputs_(5 * targets, "ComplexKinetic::outputs_", __FILE__, __LINE__),
        coefficients_(4 * targets, "ComplexKinetic::coefficients_", __FILE__, __LINE__),
        response_(image->plan().parameterCount(), "ComplexKinetic::response_", __FILE__, __LINE__),
        masses_(image->plan().modelShape().electrons(), "ComplexKinetic::masses_", __FILE__, __LINE__),
        kappa_(3, "ComplexKinetic::kappa_", __FILE__, __LINE__),
        energy_(1, "ComplexKinetic::energy_", __FILE__, __LINE__),
        permutation_(solve_.capacity(), "ComplexKinetic::permutation_", __FILE__, __LINE__),
        channels_(image->plan().modelShape().determinants, "ComplexKinetic::channels_", __FILE__, __LINE__),
        amplitude_(1, "ComplexKinetic::amplitude_", __FILE__, __LINE__),
        status_(1, "ComplexKinetic::status_", __FILE__, __LINE__),
        private_features_(cost.private_features, "ComplexKinetic::private_features_", __FILE__, __LINE__),
        private_response_(cost.private_parameters, "ComplexKinetic::private_response_", __FILE__, __LINE__),
        private_status_(cost.parallel_head ? 1 : 0, "ComplexKinetic::private_status_", __FILE__, __LINE__),
        root_failure_(cost.channel_roots ? 1 : 0, "ComplexKinetic::root_failure_", __FILE__, __LINE__)
  {
    layout_.features = withSpatialTargetRange(layout_.features, 0, targets);
    layout_.orbitals = withSpatialTargetRange(layout_.orbitals, 0, targets);
    // Plane zero is exactly the existing D*N*N VALUE provider bank. The rest of
    // each max-K jet allocation remains untouched by factor restoration/replay.
    const auto n = image_->plan().modelShape().electrons();
    if (cost.cooperative_factor && (layout_.orbitals.element_count != cost.matrices ||
        layout_.orbitals.plane_stride != cost.matrices || layout_.orbital_row_stride != n ||
        layout_.orbital_determinant_stride != n * n ||
        layout_.orbitals.configuration_stride != cost.head_elements ||
        layout_.orbitals.plane_count * cost.matrices != cost.head_elements))
      throw std::logic_error("Cooperative kinetic factor requires exact packed VALUE prefixes");
    // Prepared trunk feature planes are packed; generic padded leaves charge their own full span.
    if (cost.parallel_head && (layout_.feature_row_stride != image->plan().modelShape().feature_dimension ||
        layout_.features.element_count != image->plan().modelShape().electrons() * image->plan().modelShape().feature_dimension ||
        cost.private_features != layout_.orbitals.plane_count * layout_.features.element_count))
      throw std::logic_error("Parallel kinetic private feature cost requires exact packed prepared planes");
    static_assert(sizeof(ComplexValueResult) == 48 && sizeof(ComplexValueChannel) == 32 && sizeof(std::uint64_t) == 8);
    auto actual = cost.trunk.device_bytes + 8 * (envelope_.capacity() + projections_.capacity() + masses_.capacity() + kappa_.capacity());
    for (const auto* b : {&head_, &roots_, &inverse_, &solve_, &scratch_, &outputs_, &coefficients_, &response_, &energy_})
      actual += sizeof(cuDoubleComplex) * b->capacity();
    actual += sizeof(std::size_t) * permutation_.capacity() + 32 * channels_.capacity() + 48 + 8;
    actual += 8 * (private_features_.capacity() + private_response_.capacity() + private_status_.capacity() +
        root_failure_.capacity());
    if (actual != cost.device_bytes || cost.canonical_bytes != 24 * response_.capacity())
      throw std::logic_error("Complex kinetic exact inventory/24P mismatch");
  }

  /// Revoke every old view even on a later malformed input or failed successor.
  void invalidate()
  {
    ready_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex kinetic generation exhausted");
    ++generation_;
  }

  /// Byte fills avoid aliasing complex storage; all writes use the borrowed queue.
  void fill(void* data, std::size_t bytes, int value)
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemsetAsync(data, value, bytes, queue_.getNative()), "complex kinetic fill");
#else
    checkRuntime(cudaMemsetAsync(data, value, bytes, queue_.getNative()), "complex kinetic fill");
#endif
  }

  /// Admit host extents/scalars before any request upload or numeric mutation.
  void beginRequest(const std::vector<double>& positions, const std::vector<double>& masses, const std::array<double, 3>& kappa)
  {
    invalidate();
    image_->validateCurrentDevice();
    if (positions.size() != 3 * masses_.capacity() || masses.size() != masses_.capacity())
      throw std::invalid_argument("Complex kinetic physical input extent mismatch");
    for (double x : positions)
      if (!device_math::isFiniteBinary64(x))
        throw std::invalid_argument("Complex kinetic nonfinite coordinate");
    for (double x : masses)
      if (!device_math::isFiniteBinary64(x) || x <= 0)
        throw std::invalid_argument("Complex kinetic inverse masses must be finite positive");
    for (double x : kappa)
      if (!device_math::isFiniteBinary64(x))
        throw std::invalid_argument("Complex kinetic nonfinite common kappa");
    fill(response_.data(), 16 * response_.capacity(), 0);
    fill(energy_.data(), 16, 0);
    fill(status_.data(), 8, 0xff);
    masses_.upload(masses.data(), masses.size());
    kappa_.upload(kappa.data(), 3);
  }

  /// Resolve current canonical intervals without retaining parameter pointers.
  const double* parameter(ParameterRole role, HeadTensorComponent part = HeadTensorComponent::REAL) const
  {
    return image_->parameters() + image_->plan().parameter(role, NO_ATTENTION_BLOCK, part).begin;
  }

  /// Fail closed at every phase; a node is not a valid inverse/response request.
  void admit()
  {
    queue_.sync();
    std::uint64_t status;
    status_.download(&status, 1);
    if (status != std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("Complex kinetic finite/inverse admission failed, status=" + std::to_string(status));
  }

  /// Build mixed jets/inverses/CORE traces/physical roots ONCE for this immediate K.
  template<class Ticket>
  void prepareTile(const Ticket& ticket, const double* zero_cusp)
  {
    const auto& shape = image_->plan().modelShape();
    const auto n = shape.electrons(), d = shape.determinants, k = ticket.features.laplacian_lanes;
    layout_.features = ticket.features;
    layout_.orbitals = withSpatialTargetRange(layout_.orbitals, ticket.features.target_electron_begin, k);
    const auto stream = queue_.getNative();
    projectOpenOrbitalJets(blas_, layout_, ticket.final_features, parameter(ParameterRole::BACKFLOW_UP_WEIGHT),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT), projections_.data());
    projectOpenOrbitalJets(blas_, layout_, ticket.final_features, parameter(ParameterRole::BACKFLOW_UP_WEIGHT, HeadTensorComponent::IMAG),
        parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT, HeadTensorComponent::IMAG), projections_.data() + cost_.head_elements);
    checkRuntime(launchComplexValueEnvelopes(stream, layout_, *image_->periodicCell(), ticket.positions, image_->nuclei(),
        parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN), parameter(ParameterRole::ENVELOPE_ZETA_UP),
        parameter(ParameterRole::ENVELOPE_ZETA_DOWN), envelope_.data(), status_.data()), "complex kinetic envelopes");
    checkRuntime(launchComplexSpatialHead(stream, layout_, envelope_.data(), projections_.data(),
        projections_.data() + cost_.head_elements, head_.data(), status_.data()), "complex kinetic paired head");
    admit();
    if (cost_.cooperative_factor)
      checkRuntime(launchComplexScoreCooperativeInverseColumns(stream, d, n, envelope_.data(), projections_.data(),
          projections_.data() + cost_.head_elements, head_.data(), channels_.data(), status_.data(), inverse_.data(),
          permutation_.data(), solve_.data()), "complex kinetic cooperative inverse");
    else
      checkRuntime(launchComplexValueFactor(stream, d, n, head_.data(), channels_.data(), status_.data(), inverse_.data(),
          permutation_.data(), solve_.data()), "complex kinetic inverse");
    admit();
    checkRuntime(launchComplexValueReduce(stream, 1, d, n, shape.spin_up_electrons, *image_->periodicCell(), ticket.positions,
        channels_.data(), ticket.cusp_jets, amplitude_.data(), status_.data(), ticket.cusp.configuration_stride), "complex kinetic amplitude");
    admit();
    ComplexValueResult current;
    amplitude_.download(&current, 1);
    if (!ticket.ordinal)
      first_amplitude_ = current;
    else if (std::abs(current.log_abs - first_amplitude_.log_abs) > 2e-11 ||
        std::hypot(current.phase.x - first_amplitude_.phase.x, current.phase.y - first_amplitude_.phase.y) > 2e-11)
      throw std::runtime_error("Complex kinetic target amplitude changed within request");
    checkRuntime(launchComplexSpatialTraces(stream, layout_, channels_.data(), amplitude_.data(), inverse_.data(), head_.data(),
        zero_cusp, scratch_.data(), outputs_.data(), outputs_.data() + 3 * k, outputs_.data() + 4 * k, status_.data()), "complex kinetic CORE traces");
    admit();
    checkRuntime(launchComplexKineticCoefficients(stream, ticket.cusp, outputs_.data(), outputs_.data() + 3 * k,
        ticket.cusp_jets, masses_.data(), kappa_.data(), coefficients_.data(), coefficients_.data() + 3 * k,
        energy_.data(), status_.data()), "complex kinetic physical roots");
    admit();
    if (cost_.channel_roots)
      checkRuntime(launchComplexKineticChannelRoots(stream, layout_, channels_.data(), amplitude_.data(), inverse_.data(), head_.data(),
          outputs_.data(), outputs_.data() + 3 * k, coefficients_.data(), coefficients_.data() + 3 * k,
          roots_.data(), scratch_.data(), scratch_.capacity(), status_.data(), root_failure_.data()),
          "complex kinetic channel roots");
    else
      checkRuntime(launchComplexKineticMatrixRoots(stream, layout_, channels_.data(), amplitude_.data(), inverse_.data(), head_.data(),
          outputs_.data(), outputs_.data() + 3 * k, coefficients_.data(), coefficients_.data() + 3 * k,
          roots_.data(), scratch_.data(), status_.data()), "complex kinetic matrix roots");
    admit();
  }

  /// Consume the same retained roots twice; only objective zero prepares the tile.
  template<class Ticket>
  void seed(const Ticket& ticket, std::size_t component, double* feature, double* cusp, double* response)
  {
    if (!component)
      prepareTile(ticket, cusp);
    const auto offset = [&](ParameterRole r, HeadTensorComponent c = HeadTensorComponent::REAL) {
      return image_->plan().parameter(r, NO_ATTENTION_BLOCK, c).begin;
    };
    const ComplexKineticHeadLayout head_layout{{layout_, response_.capacity(), offset(ParameterRole::BACKFLOW_UP_WEIGHT),
        offset(ParameterRole::BACKFLOW_DOWN_WEIGHT), offset(ParameterRole::ENVELOPE_PI_UP), offset(ParameterRole::ENVELOPE_PI_DOWN),
        offset(ParameterRole::ENVELOPE_ZETA_UP), offset(ParameterRole::ENVELOPE_ZETA_DOWN)},
        offset(ParameterRole::BACKFLOW_UP_WEIGHT, HeadTensorComponent::IMAG), offset(ParameterRole::BACKFLOW_DOWN_WEIGHT, HeadTensorComponent::IMAG)};
    if (cost_.parallel_head)
      checkRuntime(launchComplexKineticHeadReverseParallel(queue_.getNative(), head_layout, *image_->periodicCell(), ticket.positions,
          image_->nuclei(), image_->parameters(), ticket.final_features, envelope_.data(), projections_.data(),
          projections_.data() + cost_.head_elements, roots_.data(), component, feature, response, status_.data(),
          private_features_.data(), private_features_.capacity(), private_response_.data(), private_response_.capacity(),
          private_status_.data()), "complex kinetic parallel paired reverse");
    else
      checkRuntime(launchComplexKineticHeadReverse(queue_.getNative(), head_layout, *image_->periodicCell(), ticket.positions,
        image_->nuclei(), image_->parameters(), ticket.final_features, envelope_.data(), projections_.data(),
        projections_.data() + cost_.head_elements, roots_.data(), component, feature, response, status_.data()), "complex kinetic paired reverse");
    checkRuntime(launchComplexKineticCuspSeed(queue_.getNative(), ticket.features.laplacian_lanes, coefficients_.data(),
        coefficients_.data() + 3 * ticket.features.laplacian_lanes, component, cusp, status_.data()), "complex kinetic cusp roots");
    admit();
  }

  /// Admit oneP before adding its selected component to the sole complexP sum.
  void finishPass(std::size_t component)
  {
    trunk_->finishResponse();
    checkRuntime(launchComplexKineticPack(queue_.getNative(), response_.capacity(), trunk_->response(), component,
        response_.data(), status_.data()), "complex kinetic canonical sum");
    admit();
  }

  ComplexKineticCost cost_;
  std::shared_ptr<const PreparedDeviceModel> image_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  std::unique_ptr<PreparedDeviceKineticTrunk> trunk_;
  OpenOrbitalJetLayout layout_;
  DeviceBuffer<double> envelope_, projections_;
  DeviceBuffer<cuDoubleComplex> head_, roots_, inverse_, solve_, scratch_, outputs_, coefficients_, response_;
  DeviceBuffer<double> masses_, kappa_;
  DeviceBuffer<cuDoubleComplex> energy_;
  DeviceBuffer<std::size_t> permutation_;
  DeviceBuffer<ComplexValueChannel> channels_;
  DeviceBuffer<ComplexValueResult> amplitude_;
  DeviceBuffer<std::uint64_t> status_;
  DeviceBuffer<double> private_features_, private_response_;
  DeviceBuffer<std::uint64_t> private_status_, root_failure_;
  ComplexValueResult first_amplitude_;
  std::size_t generation_ = 0;
  bool ready_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
