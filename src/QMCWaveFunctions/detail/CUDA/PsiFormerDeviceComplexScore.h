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
/// Standalone B1 score inventory; canonical bytes are exactly realP+complexP=24P.
struct ComplexScoreCost { ComplexValueCost forward; std::size_t device_bytes, host_bytes, canonical_bytes; };

/// Count the head-free retained VALUE profile, one network lane and one complexP.
inline ComplexScoreCost estimateComplexScoreCost(const PsiFormerExecutionPlan& plan)
{
  auto forward = estimateComplexValueCost(plan, 1);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex score extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex score bytes overflow"); };
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
  return {forward, add(add(forward.device_bytes, reverse), add(mul(16, p), scratch)),
      add(forward.host_bytes, finite), mul(24, p)};
}

/// Raw scalar-objective VJP retains one tape/realP, with no inverse or complexP.
inline ComplexScoreCost estimateComplexOrbitalVJPCost(const PsiFormerExecutionPlan& plan)
{
  if (plan.environment().periodic_profile ||
      plan.environment().geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1)
    throw std::invalid_argument("Paired raw-orbital VJP requires the qualified periodic_torus_v1 profile");
  auto cost = estimateComplexScoreCost(plan);
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
    return std::unique_ptr<PreparedDeviceComplexScore>(new PreparedDeviceComplexScore(std::move(image), queue, blas, cost));
  }

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

  /// Validate one newer family image before replacing both internal references.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> image)
  {
    ready_ = orbital_forward_ready_ = false;
    forward_->trunk_->model_->validateSuccessor(image);
    forward_->rebindModel(image);
    network_->rebindModel(std::move(image));
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
  friend struct ComplexOrbitalVJPValidationAccess;

  bool hasNoOrbitalInverseStorage() const noexcept
  {
    return !forward_->inverse_.capacity() && !forward_->solve_.capacity() &&
        !forward_->permutation_.capacity() && !score_.capacity();
  }

  /// Allocate one paired tape and realP; only log-score mode adds complexP/inverses.
  PreparedDeviceComplexScore(std::shared_ptr<const PreparedDeviceModel> image, compute::Queue<PlatformKind::CUDA>& queue,
      AcceleratorBlasHandle& blas, ComplexScoreCost cost, bool orbital_only = false)
      : cost_(cost), orbital_only_(orbital_only), forward_(new PreparedDeviceComplexValue(image, 1, queue, blas, cost.forward, true, !orbital_only)),
        network_(new PreparedDeviceNetworkReverse(std::move(image), 1, queue, blas)), seed_(cost.forward.matrices, "ComplexScore::seed_", __FILE__, __LINE__),
        feature_(forward_->trunk_->ne_ * forward_->trunk_->shape_.feature_dimension, "ComplexScore::feature_", __FILE__, __LINE__),
        ion_(forward_->trunk_->ne_ * forward_->trunk_->shape_.nuclei, "ComplexScore::ion_", __FILE__, __LINE__),
        pair_(forward_->trunk_->ne_ * (forward_->trunk_->ne_ - 1) / 2, "ComplexScore::pair_", __FILE__, __LINE__),
        same_(pair_.capacity(), "ComplexScore::same_", __FILE__, __LINE__),
        unit_(1, "ComplexScore::unit_", __FILE__, __LINE__),
        score_(orbital_only ? 0 : forward_->trunk_->plan_.parameterCount(), "ComplexScore::score_", __FILE__, __LINE__),
        orbital_cotangents_(orbital_only ? cost.forward.matrices : 0, "ComplexScore::orbital_cotangents_", __FILE__, __LINE__)
  {
    const double one = 1;
    unit_.upload(&one, 1);
    const auto actual = forward_->cost().device_bytes + network_->deviceWorkspaceBytes() +
        8 * (seed_.capacity() + feature_.capacity() + ion_.capacity() + pair_.capacity() + unit_.capacity()) +
        same_.capacity() + 16 * (score_.capacity() + orbital_cotangents_.capacity());
    if (actual != cost.device_bytes || forward_->cost().host_bytes + network_->hostWorkspaceBytes() != cost.host_bytes ||
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
    forward_->evaluate(positions);
    auto& t = *forward_->trunk_;
    checkRuntime(launchComplexScoreWeightedInverses(t.queue_.getNative(), t.shape_.determinants, t.ne_,
        forward_->channels_.data(), forward_->inverse_.data(), t.cusp_jets_.data(), forward_->results_.data(),
        forward_->status_.data()), "complex score weighted inverse epoch");
    forward_->admitStatus();
    checkRuntime(launchScoreGeometry(t.queue_.getNative(), 1, t.ne_, t.shape_.nuclei, t.shape_.spin_up_electrons,
        t.positions_.data(), t.model_->nuclei(), ion_.data(), pair_.data(), same_.data(), t.model_->periodicCell()), "complex score geometry");
  }

  /// Seed canonical head/cusp intervals FIRST, accumulating both heads into realNF.
  void prepareObjective(std::size_t objective, const cuDoubleComplex* raw_cotangents = nullptr)
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
  void finishObjective(std::size_t objective, bool pack = true)
  {
    auto& t = *forward_->trunk_;
    network_->reverseImpl(t, {}, feature_.data(), false);
    if (pack)
      checkRuntime(launchComplexScorePack(t.queue_.getNative(), score_.capacity(), objective, network_->response_.data(),
        score_.data(), forward_->status_.data()), "complex score canonical objective pack");
    forward_->admitStatus();
  }

  ComplexScoreCost cost_;
  const bool orbital_only_;
  std::unique_ptr<PreparedDeviceComplexValue> forward_;
  std::unique_ptr<PreparedDeviceNetworkReverse> network_;
  DeviceBuffer<double> seed_, feature_, ion_, pair_;
  DeviceBuffer<std::uint8_t> same_;
  DeviceBuffer<double> unit_;
  DeviceBuffer<cuDoubleComplex> score_, orbital_cotangents_;
  std::size_t generation_ = 0;
  bool ready_ = false;
  bool orbital_forward_ready_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
