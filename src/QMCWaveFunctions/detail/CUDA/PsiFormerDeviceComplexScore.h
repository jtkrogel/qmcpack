//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_H
#include "PsiFormerDeviceComplexValue.h"
#include "PsiFormerDeviceNetworkReverse.h"
#include "PsiFormerDeviceComplexScoreKernels.h"

namespace qmcplusplus::psiformer::device
{
/// Capture the score-only option once; no hot-loop environment or numeric owner.
inline bool configuredComplexScoreWeightedInversesCooperative()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_WEIGHTED_INVERSES_COOPERATIVE");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_WEIGHTED_INVERSES_COOPERATIVE must be 0 or 1");
  }();
  return enabled;
}

/// Capture the inverse implementation once; ordinary VALUE and raw-orbital VJPs keep their existing route.
inline bool configuredComplexScoreInverseColumns()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_INVERSE_COLUMNS");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_INVERSE_COLUMNS must be 0 or 1");
  }();
  return enabled;
}

/// Restrict initial retained qualification to the measured periodic production-size envelope.
inline bool complexScoreInverseColumnsEligible(const PsiFormerExecutionPlan& plan)
{
  const auto& shape = plan.modelShape();
  const auto& environment = plan.environment();
  return environment.boundary == BoundaryCondition::PERIODIC && environment.fixed_nuclei &&
      environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX &&
      environment.parameter_scalar_domain == ScalarDomain::REAL && environment.compute_scalar_domain == ScalarDomain::REAL &&
      environment.amplitude_scalar_domain == ScalarDomain::COMPLEX &&
      environment.periodic_axes[0] && environment.periodic_axes[1] && environment.periodic_axes[2] &&
      environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TORUS_V1 &&
      !environment.periodic_profile && shape.electrons() >= 64 && shape.electrons() <= 300 &&
      shape.determinants >= 1 && shape.determinants <= 16;
}

/// Capture only retained parameter-score VALUE geometry; ordinary and raw VJP routes are unchanged.
inline bool configuredComplexScorePeriodicGeometry()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_PERIODIC_GEOMETRY");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_PERIODIC_GEOMETRY must be 0 or 1");
  }();
  return enabled;
}

/// Retained VALUE geometry uses the same fixed paired FP64 envelope as the inverse qualification.
inline bool complexScorePeriodicGeometryEligible(const PsiFormerExecutionPlan& plan)
{
  return complexScoreInverseColumnsEligible(plan);
}

/// Capture the retained-only LU policy once; diagnostics and raw VJP do not consult this option.
inline bool configuredComplexScoreCooperativeLu()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_COOPERATIVE_LU");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_COOPERATIVE_LU must be 0 or 1");
  }();
  return enabled;
}

/// Capture the SCORE-only pair scan independently of unrelated ordinary VALUE options.
inline bool configuredComplexScorePairCheck()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_COOPERATIVE_PAIR_CHECK");
    if (!value || std::string_view(value) == "0")
      return false;
    if (std::string_view(value) == "1")
      return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_COOPERATIVE_PAIR_CHECK must be 0 or 1");
  }();
  return enabled;
}

/// Keep retained pair records inside the already qualified fixed paired FP64 envelope.
inline bool complexScorePairCheckEligible(const PsiFormerExecutionPlan& plan)
{
  return complexScoreInverseColumnsEligible(plan);
}

/// Capture only the retained SCORE grid; explicit primitive oracles do not consult it.
inline bool configuredComplexScoreWeightedGrid()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_SCORE_WEIGHTED_INVERSES_GRID");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_SCORE_WEIGHTED_INVERSES_GRID must be 0 or 1");
  }();
  return enabled;
}

/// Keep six-launch overhead outside small or unqualified physical SCORE profiles.
inline bool complexScoreWeightedGridEligible(const PsiFormerExecutionPlan& plan)
{
  return complexScoreInverseColumnsEligible(plan) && plan.modelShape().electrons() >= 100;
}

/// Standalone B1 score inventory; canonical bytes are exactly realP+complexP=24P.
struct ComplexScoreCost
{
  ComplexValueCost forward;
  std::size_t device_bytes, host_bytes, canonical_bytes, weighted_grid_bytes;
  std::size_t mixed_forward_bytes = 0, mixed_derivative_bytes = 0, mixed_pivot_bytes = 0;
};

/// Count the head-free retained VALUE profile, one network lane and one complexP.
inline ComplexScoreCost estimateComplexScoreCost(const PsiFormerExecutionPlan& plan,
    bool pair_check = configuredComplexScorePairCheck(), bool weighted_grid = configuredComplexScoreWeightedGrid())
{
  auto forward = estimateComplexValueCost(plan, 1, false);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex score extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex score bytes overflow"); };
  if (pair_check && complexScorePairCheckEligible(plan))
  {
    const auto pairs = mul(plan.modelShape().electrons(), plan.modelShape().electrons());
    if (pairs > std::numeric_limits<std::uint64_t>::max() / 2)
      throw std::overflow_error("Retained score pair event encoding exceeds uint64 admission");
    forward.pair_event_bytes = sizeof(std::uint64_t);
    forward.device_bytes = add(forward.device_bytes, forward.pair_event_bytes);
  }
  const auto& s = plan.modelShape();
  const auto n = s.electrons(), nf = mul(n, s.feature_dimension), probability = mul(s.attention_heads, mul(n, n));
  const auto p = plan.parameterCount(), q = mul(n, n - 1) / 2;
  if (!p || p - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8) ||
      (p - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("Complex score canonical launch/status extent exceeds admission");
  const auto tape = mul(8, mul(s.attention_blocks, add(mul(8, nf), probability)));
  forward.trunk_bytes = add(forward.trunk_bytes, tape);
  forward.device_bytes = add(forward.device_bytes, add(tape,
      add(mul(16, forward.matrices), mul(16 + sizeof(std::size_t), mul(s.determinants, n)))));
  const auto finite = mul(p / 4096 + (p % 4096 != 0), sizeof(DerivativeFiniteStatus));
  const auto reverse = add(mul(8, add(add(mul(10, nf), probability), p)), finite);
  const auto scratch = add(mul(8, add(add(forward.matrices, nf), add(mul(n, s.nuclei), add(q, 1)))), q);
  const auto grid_bytes = weighted_grid && complexScoreWeightedGridEligible(plan) ? sizeof(ComplexScoreWeightedGridState) : 0;
  return {forward, add(add(add(forward.device_bytes, reverse), add(mul(16, p), scratch)), grid_bytes),
      add(forward.host_bytes, finite), mul(24, p), grid_bytes};
}

/// Exact private mixed SCORE bytes; the immutable shared FP32/master images are excluded.
inline ComplexScoreCost estimateMixedComplexScoreCost(const PsiFormerExecutionPlan& plan)
{
  auto cost = estimateComplexScoreCost(plan);
  const auto rows = plan.modelShape().electrons();
  cost.mixed_forward_bytes = PreparedDeviceMixedValueDense::estimateDeviceWorkspaceBytes(plan, rows);
  cost.mixed_derivative_bytes = PreparedDeviceMixedAffineDerivatives::estimateDeviceWorkspaceBytes(plan, rows);
  cost.mixed_pivot_bytes = checkedDeviceBytes<double>(plan.modelShape().determinants);
  for (const auto bytes : {cost.mixed_forward_bytes, cost.mixed_derivative_bytes, cost.mixed_pivot_bytes})
    cost.device_bytes = spatial_detail::checkedAdd(cost.device_bytes, bytes, "Mixed complex score bytes overflow");
  cost.host_bytes = spatial_detail::checkedAdd(cost.host_bytes, 2 * sizeof(PsiFormerDeviceNumericalDiagnostics),
      "Mixed complex score host diagnostics overflow");
  return cost;
}

/// Raw scalar-objective VJP retains one tape/realP, with no inverse or complexP.
inline ComplexScoreCost estimateComplexOrbitalVJPCost(const PsiFormerExecutionPlan& plan)
{
  if (plan.environment().periodic_profile ||
      plan.environment().geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1)
    throw std::invalid_argument("Paired raw-orbital VJP requires the qualified periodic_torus_v1 profile");
  auto cost = estimateComplexScoreCost(plan, false, false);
  const auto multiply = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex orbital VJP bytes overflow"); };
  const auto& shape = plan.modelShape();
  const auto inverse_bytes = spatial_detail::checkedAdd(multiply(16, cost.forward.matrices),
      multiply(16 + sizeof(std::size_t), multiply(shape.determinants, shape.electrons())), "Complex orbital inverse bytes overflow");
  cost.forward.device_bytes -= inverse_bytes;
  cost.device_bytes -= inverse_bytes + multiply(16, plan.parameterCount());
  cost.device_bytes = spatial_detail::checkedAdd(cost.device_bytes, multiply(16, cost.forward.matrices),
      "Complex orbital cotangent bytes overflow");
  cost.canonical_bytes = multiply(8, plan.parameterCount());
  return cost;
}

/** Numerical CORE log-score for REAL canonical parameters, including imaginary heads.
 * Caller queue/BLAS/exclusive execution lease outlive this B1 owner. No public
 * GPUXML/optimizer/complex kinetic or FULL graph is admitted. Two real objectives
 * share ONE forward tape/P scratch; partial complexP is never publicly visible.
 */
class PreparedDeviceComplexScore
{
public:
  /// Borrow canonical score and CORE+cusp amplitude, without a common outer phase.
  struct ResultView
  {
    const PreparedDeviceComplexScore* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, forward_generation, count;
    const cuDoubleComplex* data;
  };

  /// Borrow raw E*(A+iB), without cusp/common phase or determinant factorization.
  struct OrbitalView
  {
    const PreparedDeviceComplexScore* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, forward_generation, count;
    const cuDoubleComplex* data;
  };

  /// Completed scalar-objective gradient with respect to real canonical parameters.
  struct OrbitalVJPView
  {
    const PreparedDeviceComplexScore* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, forward_generation, count;
    const double* data;
  };

  /// Exact combined admission precedes ALL trunk/reverse/head/canonical allocations.
  static std::unique_ptr<PreparedDeviceComplexScore> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget)
  {
    if (!image)
      throw std::invalid_argument("Complex score requires a paired numerical image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexScoreCost(image->plan());
    if (cost.device_bytes > budget)
      throw std::length_error("Complex score exceeds its standalone owner budget");
    return std::unique_ptr<PreparedDeviceComplexScore>(new PreparedDeviceComplexScore(
        std::move(image), queue, blas, cost, false, configuredComplexScoreInverseColumns(),
        configuredComplexScorePeriodicGeometry(), configuredComplexScoreCooperativeLu()));
  }

  /// Prepare one matched mixed primal/reverse, retaining the existing FP64 head and reductions.
  static std::unique_ptr<PreparedDeviceComplexScore> prepareMixed(
      std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget)
  {
    if (!parameters || parameters->trunkHeadPolicy() != MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD)
      throw std::invalid_argument("Mixed complex score requires a paired immutable affine cache");
    const auto& image = parameters->modelImage();
    image->validateCurrentDevice();
    const auto cost = estimateMixedComplexScoreCost(image->plan());
    if (cost.device_bytes > budget)
      throw std::length_error("Mixed complex score exceeds its standalone owner budget");
    auto primal = std::make_shared<PreparedDeviceMixedValueDense>(parameters,
        image->plan().modelShape().electrons(), queue, blas);
    auto derivative = std::make_shared<PreparedDeviceMixedAffineDerivatives>(parameters,
        image->plan().modelShape().electrons(), queue, blas);
    // The diagnostic factor route owns actual mixed dispatch, independently of ambient LU selectors.
    return std::unique_ptr<PreparedDeviceComplexScore>(new PreparedDeviceComplexScore(
        image, queue, blas, cost, false, false, false, false, std::move(primal), std::move(derivative)));
  }

  /// Report the frozen matched primal/reverse selection without consulting process flags.
  bool usesMixedAffine() const noexcept { return bool(mixed_forward_); }

  /// Prepare raw-orbital mode separately; ordinary score ownership is unchanged.
  static std::unique_ptr<PreparedDeviceComplexScore> prepareOrbitalVJP(std::shared_ptr<const PreparedDeviceModel> image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t budget)
  {
    if (!image)
      throw std::invalid_argument("Complex orbital VJP requires a paired numerical image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexOrbitalVJPCost(image->plan());
    if (cost.device_bytes > budget)
      throw std::length_error("Complex orbital VJP exceeds its standalone owner budget");
    return std::unique_ptr<PreparedDeviceComplexScore>(new PreparedDeviceComplexScore(std::move(image), queue, blas, cost, true));
  }

  /// Report the immutable cold selection without changing execution or ownership.
  bool usesCooperativeWeightedInverses() const noexcept { return cooperative_weighted_inverses_; }

  /// The preflight cost, not a second environment read, owns the grid selection.
  bool usesWeightedInverseGrid() const noexcept { return weighted_grid_.capacity() != 0; }

  /// Report the frozen retained inverse choice without consulting hot-loop environment state.
  bool usesInverseColumns() const noexcept { return inverse_columns_; }

  /// Report the immutable retained primal geometry policy without hot-loop environment access.
  bool usesPeriodicGeometry() const noexcept { return periodic_geometry_; }

  /// Report the frozen LU policy without changing the existing inverse-column and publication contracts.
  bool usesCooperativeLu() const noexcept { return cooperative_lu_; }

  /// Report the effective frozen pair-event owner, not the ambient ordinary VALUE flag.
  bool usesCooperativePairCheck() const noexcept { return cooperative_pair_check_; }

  /// Reuse one tape and oneP scratch for both real objectives before publishing.
  void evaluate(const std::vector<double>& positions)
  {
    if (orbital_only_)
      throw std::logic_error("Raw-orbital owner does not publish a log-score objective");
    prepareForward(positions);
    for (std::size_t component = 0; component < 2; ++component)
    {
      prepareObjective(component);
      finishObjective(component);
    }
    ready_ = true;
  }

  /// Observe the existing two reverse objectives on one admitted forward tape.
  void evaluateAffine(const std::vector<double>& positions, PreparedDeviceAffineStatistics& factors)
  {
    if (orbital_only_)
      throw std::logic_error("Raw-orbital owner cannot produce affine log-score statistics");
    factors.requireModel(forward_->trunk_->model_.get());
    prepareForward(positions);
    for (std::size_t objective = 0; objective < 2; ++objective)
    {
      AffineChannel channel(factors, objective);
      prepareObjective(objective, nullptr, &factors);
      finishObjective(objective, true, &channel);
    }
    ready_ = true;
  }

  /// Admit one retained raw head epoch, including configurations at wavefunction nodes.
  void evaluateOrbitals(const std::vector<double>& positions)
  {
    ready_ = orbital_forward_ready_ = false;
    if (!orbital_only_)
      throw std::logic_error("Raw-orbital forward requires its explicitly prepared owner");
    advanceGeneration();
    forward_->evaluateOrbitalHead(positions);
    auto& t = *forward_->trunk_;
    checkRuntime(launchScoreGeometry(t.queue_.getNative(), 1, t.ne_, t.shape_.nuclei, t.shape_.spin_up_electrons,
        t.positions_.data(), t.model_->nuclei(), ion_.data(), pair_.data(), same_.data(), t.model_->periodicCell()), "complex orbital VJP geometry");
    forward_->admitStatus();
    orbital_forward_ready_ = true;
  }

  OrbitalView orbitalView() const
  {
    if (!orbital_only_ || !orbital_forward_ready_ || !forward_->trunk_->value_tape_valid_)
      throw std::logic_error("Raw orbital head is unavailable");
    return {this, forward_->trunk_->model_.get(), forward_->generation_,
        forward_->trunk_->evaluationGeneration(), cost_.forward.matrices, forward_->matrix_.data()};
  }

  /// Check stamps before copying the complete independent orbital projection.
  void downloadOrbitals(const OrbitalView& view, std::vector<cuDoubleComplex>& destination) const
  {
    validateOrbitalView(view);
    if (destination.size() != view.count)
      throw std::invalid_argument("Raw orbital download has the wrong extent");
    forward_->matrix_.downloadOnStream(destination.data(), destination.size(), forward_->trunk_->queue_.getNative());
  }

  /// Contract analytic seeds C with deltaL=Re sum(C*deltaZ), not a log objective.
  void reverseOrbitals(const OrbitalView& view, const std::vector<cuDoubleComplex>& cotangents)
  {
    ready_ = false;
    advanceGeneration();
    validateOrbitalView(view);
    if (cotangents.size() != cost_.forward.matrices)
      throw std::invalid_argument("Raw orbital cotangent has the wrong extent");
    for (const auto value : cotangents)
      if (!device_math::isFiniteBinary64(cuCreal(value)) || !device_math::isFiniteBinary64(cuCimag(value)))
        throw std::invalid_argument("Raw orbital cotangent is nonfinite");
    orbital_cotangents_.uploadOnStream(cotangents.data(), cotangents.size(), forward_->trunk_->queue_.getNative());
    prepareObjective(0, orbital_cotangents_.data());
    finishObjective(0, false);
    ready_ = true;
  }

  OrbitalVJPView orbitalVJPView() const
  {
    if (!orbital_only_ || !ready_)
      throw std::logic_error("Raw orbital VJP is unavailable");
    return {this, forward_->trunk_->model_.get(), generation_, forward_->trunk_->evaluationGeneration(),
        network_->response_.capacity(), network_->response_.data()};
  }

  void downloadOrbitalVJP(const OrbitalVJPView& view, std::vector<double>& destination) const
  {
    forward_->trunk_->model_->validateCurrentDevice();
    if (!orbital_only_ || !ready_ || view.owner != this || view.image != forward_->trunk_->model_.get() ||
        view.generation != generation_ || view.forward_generation != forward_->trunk_->evaluationGeneration() ||
        view.count != network_->response_.capacity() || view.data != network_->response_.data() || destination.size() != view.count)
      throw std::invalid_argument("Raw orbital VJP is stale, foreign or wrong extent");
    network_->response_.downloadOnStream(destination.data(), destination.size(), forward_->trunk_->queue_.getNative());
  }

  /// Preflight the unchanged FP64 score leaves without revoking accepted results.
  void validateModelRebind(const std::shared_ptr<const PreparedDeviceModel>& image) const
  {
    if (mixed_forward_)
      throw std::invalid_argument("Mixed complex score requires its prepared-cache rebind contract");
    if (generation_ == std::numeric_limits<std::size_t>::max() ||
        forward_->generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex score generation exhausted");
    forward_->validateModelRebind(image);
    network_->validateModelRebindWithMixedParameters(image, {});
  }

  /// Adopt only after every composing FP64 leaf preflights under its owner lease.
  void adoptPreparedModel(const std::shared_ptr<const PreparedDeviceModel>& image) noexcept
  {
    forward_->adoptPreparedModel(image);
    network_->adoptPreparedModelWithMixedParameters(image, {});
    ready_ = orbital_forward_ready_ = false;
  }

  /// Preserve the ordinary rebind API through the same all-leaf transaction split.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> image)
  {
    validateModelRebind(image);
    adoptPreparedModel(image);
  }

  /// Validate all mixed score leaves while the accepted image and compute copy remain bound.
  void validateMixedParameterRebind(
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& incoming) const
  {
    if (!mixed_forward_ || !incoming)
      throw std::invalid_argument("Mixed complex score rebind requires a prepared successor cache");
    if (generation_ == std::numeric_limits<std::size_t>::max() ||
        forward_->generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Mixed complex score generation exhausted");
    const auto& image = incoming->modelImage();
    forward_->trunk_->validateModelRebindWithMixedParameters(image, incoming);
    network_->validateModelRebindWithMixedParameters(image, incoming);
  }

  /// No-throw adoption only after every native leaf and predecessor lifetime are secured.
  void adoptPreparedMixedParameters(
      const std::shared_ptr<const PreparedDeviceMixedDenseParameters>& incoming) noexcept
  {
    const auto& image = incoming->modelImage();
    mixed_forward_->adoptPreparedParameters(incoming);
    forward_->adoptPreparedModel(image);
    network_->adoptPreparedModelWithMixedParameters(image, incoming);
    ready_ = orbital_forward_ready_ = false;
    ++generation_;
  }

  /// Expose no matrix/inverse or incomplete objective products.
  ResultView resultView() const
  {
    if (orbital_only_ || !ready_)
      throw std::logic_error("Complex score response unavailable");
    return {this, forward_->trunk_->model_.get(), generation_, forward_->trunk_->evaluationGeneration(), score_.capacity(), score_.data()};
  }

  /// Check owner/image/tape generation and canonical extent before the finalP download.
  void download(const ResultView& view, std::vector<cuDoubleComplex>& response, ComplexValueResult& amplitude) const
  {
    forward_->trunk_->model_->validateCurrentDevice();
    if (orbital_only_ || !ready_ || view.owner != this || view.image != forward_->trunk_->model_.get() || view.generation != generation_ ||
        view.forward_generation != forward_->trunk_->evaluationGeneration() || view.count != score_.capacity() ||
        view.data != score_.data() || response.size() != view.count)
      throw std::invalid_argument("Complex score response is stale, foreign or wrong extent");
    score_.download(response.data(), response.size());
    forward_->results_.download(&amplitude, 1);
  }

  /// Report standalone numeric ownership, excluding shared image/vendor reserves.
  const ComplexScoreCost& cost() const noexcept { return cost_; }

private:
  friend struct ComplexScoreValidationAccess;
  friend struct ComplexScoreInverseValidationAccess;
  friend struct ComplexScoreGeometryValidationAccess;
  friend struct ComplexOrbitalVJPValidationAccess;

  bool hasNoOrbitalInverseStorage() const noexcept
  {
    return !forward_->inverse_.capacity() && !forward_->solve_.capacity() &&
        !forward_->permutation_.capacity() && !score_.capacity();
  }

  /// Allocate one paired tape and realP; only log-score mode adds complexP/inverses.
  PreparedDeviceComplexScore(std::shared_ptr<const PreparedDeviceModel> image, compute::Queue<PlatformKind::CUDA>& queue,
      AcceleratorBlasHandle& blas, ComplexScoreCost cost, bool orbital_only = false, bool inverse_columns = false,
      bool periodic_geometry = false, bool cooperative_lu = false,
      std::shared_ptr<PreparedDeviceMixedValueDense> mixed_forward = {},
      std::shared_ptr<PreparedDeviceMixedAffineDerivatives> mixed_derivative = {})
      : cost_(cost), orbital_only_(orbital_only),
        inverse_columns_(!orbital_only && inverse_columns &&
            complexScoreInverseColumnsEligible(image->plan())),
        periodic_geometry_(!orbital_only && periodic_geometry && complexScorePeriodicGeometryEligible(image->plan())),
        cooperative_lu_(!orbital_only && cooperative_lu && inverse_columns_ &&
            complexScoreInverseColumnsEligible(image->plan())),
        cooperative_pair_check_(!orbital_only && cost.forward.pair_event_bytes &&
            complexScorePairCheckEligible(image->plan())),
        cooperative_weighted_inverses_(!orbital_only && configuredComplexScoreWeightedInversesCooperative() &&
            image->plan().modelShape().electrons() >= 64 && image->plan().modelShape().electrons() <= 300 &&
            image->plan().modelShape().determinants >= 1 && image->plan().modelShape().determinants <= 16),
        mixed_forward_(std::move(mixed_forward)),
        mixed_pivots_(cost.mixed_pivot_bytes / sizeof(double), "ComplexScore::mixed_pivots", __FILE__, __LINE__),
        forward_(new PreparedDeviceComplexValue(image, 1, queue, blas, cost.forward, true, !orbital_only,
            inverse_columns_, periodic_geometry_, cooperative_lu_, cooperative_pair_check_)),
        network_(new PreparedDeviceNetworkReverse(std::move(image), 1, queue, blas,
            NetworkReverseOutput::PARAMETER_SCORE, std::move(mixed_derivative))), seed_(cost.forward.matrices, "ComplexScore::seed_", __FILE__, __LINE__),
        feature_(forward_->trunk_->ne_ * forward_->trunk_->shape_.feature_dimension, "ComplexScore::feature_", __FILE__, __LINE__),
        ion_(forward_->trunk_->ne_ * forward_->trunk_->shape_.nuclei, "ComplexScore::ion_", __FILE__, __LINE__),
        pair_(forward_->trunk_->ne_ * (forward_->trunk_->ne_ - 1) / 2, "ComplexScore::pair_", __FILE__, __LINE__),
        same_(pair_.capacity(), "ComplexScore::same_", __FILE__, __LINE__),
        unit_(1, "ComplexScore::unit_", __FILE__, __LINE__),
        score_(orbital_only ? 0 : forward_->trunk_->plan_.parameterCount(), "ComplexScore::score_", __FILE__, __LINE__),
        orbital_cotangents_(orbital_only ? cost.forward.matrices : 0, "ComplexScore::orbital_cotangents_", __FILE__, __LINE__),
        weighted_grid_(cost.weighted_grid_bytes ? 1 : 0, "ComplexScore::weighted_grid_", __FILE__, __LINE__)
  {
    if (mixed_forward_)
    {
      if (orbital_only_ || !network_->mixed_ ||
          network_->mixed_->parameters().get() != mixed_forward_->parameterCache().get())
        throw std::invalid_argument("Mixed complex score primal and derivative caches differ");
      forward_->trunk_->setMixedRetainedValueDense(mixed_forward_);
    }
    if ((cost.weighted_grid_bytes != 0 && cost.weighted_grid_bytes != sizeof(ComplexScoreWeightedGridState)) ||
        (cost.weighted_grid_bytes && (orbital_only || !complexScoreWeightedGridEligible(forward_->trunk_->plan_))))
      throw std::logic_error("Complex score grid differs from its frozen admission");
    const double one = 1;
    unit_.upload(&one, 1);
    const auto actual = forward_->cost().device_bytes + network_->deviceWorkspaceBytes() +
        (mixed_forward_ ? mixed_forward_->deviceWorkspaceBytes() : 0) + checkedDeviceBytes<double>(mixed_pivots_.capacity()) +
        8 * (seed_.capacity() + feature_.capacity() + ion_.capacity() + pair_.capacity() + unit_.capacity()) +
        same_.capacity() + 16 * (score_.capacity() + orbital_cotangents_.capacity()) +
        sizeof(ComplexScoreWeightedGridState) * weighted_grid_.capacity();
    if (actual != cost.device_bytes || forward_->cost().host_bytes + network_->hostWorkspaceBytes() +
        (mixed_forward_ ? mixed_forward_->hostWorkspaceBytes() : 0) != cost.host_bytes ||
        8 * network_->response_.capacity() + 16 * score_.capacity() != cost.canonical_bytes)
      throw std::logic_error("Complex score inventory differs from exact preflight/24P");
  }

  void advanceGeneration()
  {
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex score generation exhausted");
    ++generation_;
  }

  void validateOrbitalView(const OrbitalView& view) const
  {
    forward_->trunk_->model_->validateCurrentDevice();
    if (!orbital_only_ || !orbital_forward_ready_ || !forward_->trunk_->value_tape_valid_ || view.owner != this ||
        view.image != forward_->trunk_->model_.get() || view.generation != forward_->generation_ ||
        view.forward_generation != forward_->trunk_->evaluationGeneration() || view.count != cost_.forward.matrices ||
        view.data != forward_->matrix_.data())
      throw std::invalid_argument("Raw orbital view is stale, foreign or wrong extent");
  }

  /// Admit the sole forward/inverse epoch, then consume I into stable weighted I.
  void prepareForward(const std::vector<double>& positions)
  {
    ready_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex score generation exhausted");
    ++generation_;
    if (mixed_forward_) forward_->evaluateRetainedMixedCoordinates(positions, mixed_pivots_);
    else forward_->evaluate(positions);
    auto& t = *forward_->trunk_;
    const auto weighted_inverse = cooperative_weighted_inverses_ ? launchComplexScoreWeightedInversesCooperative :
        launchComplexScoreWeightedInverses;
    if (usesWeightedInverseGrid())
      checkRuntime(launchComplexScoreWeightedInversesGrid(t.queue_.getNative(), t.shape_.determinants, t.ne_,
          forward_->channels_.data(), forward_->inverse_.data(), t.cusp_jets_.data(), forward_->results_.data(),
          forward_->status_.data(), weighted_grid_.data(), weighted_grid_.capacity()), "complex score weighted grid epoch");
    else
      checkRuntime(weighted_inverse(t.queue_.getNative(), t.shape_.determinants, t.ne_,
          forward_->channels_.data(), forward_->inverse_.data(), t.cusp_jets_.data(), forward_->results_.data(),
          forward_->status_.data()), "complex score weighted inverse epoch");
    forward_->admitStatus();
    checkRuntime(launchScoreGeometry(t.queue_.getNative(), 1, t.ne_, t.shape_.nuclei, t.shape_.spin_up_electrons,
        t.positions_.data(), t.model_->nuclei(), ion_.data(), pair_.data(), same_.data(), t.model_->periodicCell()), "complex score geometry");
  }

  /// Map existing real-trunk callbacks to one channel of the paired transaction.
  class AffineChannel final : public DeviceAffineObservationSink
  {
  public:
    AffineChannel(PreparedDeviceAffineStatistics& factors, std::size_t objective)
        : factors_(factors), objective_(objective) {}

    void observe(ParameterRole role, std::size_t layer, const double* activations,
        const double* sensitivities, std::size_t rows, std::size_t input, std::size_t output) override
    {
      factors_.observeChannel(role, layer, HeadTensorComponent::REAL, objective_,
          activations, sensitivities, rows, input, output);
    }
  private:
    PreparedDeviceAffineStatistics& factors_;
    std::size_t objective_;
  };

  /// Seed canonical head/cusp intervals FIRST, accumulating both heads into realNF.
  void prepareObjective(std::size_t objective, const cuDoubleComplex* raw_cotangents = nullptr,
      PreparedDeviceAffineStatistics* factors = nullptr)
  {
    auto& t = *forward_->trunk_;
    const auto n = t.ne_, d = t.shape_.determinants, a = t.shape_.nuclei, up = t.shape_.spin_up_electrons;
    const auto stream = t.queue_.getNative();
    network_->result_ready_ = false;
    network_->zero(network_->response_, t.plan_.parameterCount());
    network_->zero(feature_, feature_.capacity());
    const DenseForwardLayout dense{n, t.shape_.feature_dimension, d * n, t.shape_.feature_dimension, d * n, d * n};
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t component = 0; component < 2; ++component)
      {
        if (raw_cotangents)
          checkRuntime(launchComplexOrbitalHeadSeed(stream, d, n, up, raw_cotangents, forward_->envelope_.data(),
              forward_->projections_.data(), forward_->projections_.data() + cost_.forward.matrices,
              component, spin, seed_.data(), forward_->status_.data()), "complex raw orbital affine cotangent");
        else
          checkRuntime(launchComplexScoreHeadSeed(stream, d, n, up, forward_->inverse_.data(), forward_->envelope_.data(),
            forward_->projections_.data(), forward_->projections_.data() + cost_.forward.matrices,
            objective, component, spin, seed_.data(), forward_->status_.data()), "complex score affine cotangent");
        const auto& interval = t.plan_.parameter(spin ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT,
            NO_ATTENTION_BLOCK, component ? HeadTensorComponent::IMAG : HeadTensorComponent::REAL);
        if (factors)
        {
          const auto begin = spin ? up : 0, rows = spin ? n - up : up;
          factors->observeChannel(spin ? ParameterRole::BACKFLOW_DOWN_WEIGHT : ParameterRole::BACKFLOW_UP_WEIGHT,
              NO_ATTENTION_BLOCK, component ? HeadTensorComponent::IMAG : HeadTensorComponent::REAL, objective,
              t.valueFinalFeatures() + begin * t.shape_.feature_dimension, seed_.data() + begin * d * n,
              rows, t.shape_.feature_dimension, d * n);
        }
        accumulateDenseWeightAdjoint(t.blas_, dense, t.valueFinalFeatures(), seed_.data(), network_->response_.data() + interval.begin);
        accumulateDenseInputAdjoint(t.blas_, dense, seed_.data(), t.model_->parameters() + interval.begin, feature_.data());
      }
    // H*Z uses the true affine projection at E0, never the already factored A/E.
    if (raw_cotangents)
      checkRuntime(launchComplexOrbitalHeadSeed(stream, d, n, up, raw_cotangents, forward_->envelope_.data(),
          forward_->projections_.data(), forward_->projections_.data() + cost_.forward.matrices,
          2, 0, seed_.data(), forward_->status_.data()), "complex raw orbital envelope cotangent");
    else
      checkRuntime(launchComplexScoreHeadSeed(stream, d, n, up, forward_->inverse_.data(), forward_->envelope_.data(),
        forward_->projections_.data(), forward_->projections_.data() + cost_.forward.matrices,
        objective, 2, 0, seed_.data(), forward_->status_.data()), "complex score envelope cotangent");
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto pi = t.plan_.parameter(spin ? ParameterRole::ENVELOPE_PI_DOWN : ParameterRole::ENVELOPE_PI_UP).begin;
      const auto zeta = t.plan_.parameter(spin ? ParameterRole::ENVELOPE_ZETA_DOWN : ParameterRole::ENVELOPE_ZETA_UP).begin;
      const EnvelopeParameterReverseLayout layout{1, d, n, a, spin ? up : 0, spin ? n : up,
          n, n * n, d * n * n, a, n * a, a, n * a, pi, zeta};
      checkRuntime(launchEnvelopeParameterReverse(stream, layout, ion_.data(), t.model_->parameters() + pi,
          t.model_->parameters() + zeta, seed_.data(), network_->response_.data()), "complex score envelope parameters");
    }
    if (!raw_cotangents && objective == 0 && pair_.capacity())
    {
      const auto* same = t.plan_.optionalParameter(ParameterRole::CUSP_SAME_ALPHA);
      const CuspAlphaReverseLayout cusp{1, pair_.capacity(), pair_.capacity(), same ? same->begin : 0,
          t.plan_.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin, same != nullptr};
      checkRuntime(launchCuspAlphaReverse(stream, cusp, pair_.data(), same_.data(), t.model_->sameAlpha(), t.model_->oppositeAlpha(),
          network_->response_.data(), unit_.data()), "complex score real cusp ONCE");
    }
    forward_->admitStatus();
  }

  /// Add the same taped trunk without clearing P, admit and pack this objective.
  void finishObjective(std::size_t objective, bool pack = true, DeviceAffineObservationSink* affine = nullptr)
  {
    auto& t = *forward_->trunk_;
    network_->reverseImpl(t, {}, feature_.data(), false, true, affine);
    if (pack)
      checkRuntime(launchComplexScorePack(t.queue_.getNative(), score_.capacity(), objective, network_->response_.data(),
        score_.data(), forward_->status_.data()), "complex score canonical objective pack");
    forward_->admitStatus();
  }

  ComplexScoreCost cost_;
  const bool orbital_only_;
  const bool inverse_columns_;
  const bool periodic_geometry_;
  const bool cooperative_lu_;
  const bool cooperative_pair_check_;
  const bool cooperative_weighted_inverses_;
  std::shared_ptr<PreparedDeviceMixedValueDense> mixed_forward_;
  DeviceBuffer<double> mixed_pivots_;
  std::unique_ptr<PreparedDeviceComplexValue> forward_;
  std::unique_ptr<PreparedDeviceNetworkReverse> network_;
  DeviceBuffer<double> seed_, feature_, ion_, pair_;
  DeviceBuffer<std::uint8_t> same_;
  DeviceBuffer<double> unit_;
  DeviceBuffer<cuDoubleComplex> score_, orbital_cotangents_;
  DeviceBuffer<ComplexScoreWeightedGridState> weighted_grid_;
  std::size_t generation_ = 0;
  bool ready_ = false;
  bool orbital_forward_ready_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
