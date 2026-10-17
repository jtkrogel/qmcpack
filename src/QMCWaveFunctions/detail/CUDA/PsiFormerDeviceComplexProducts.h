//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_PRODUCTS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_PRODUCTS_H
#include "PsiFormerDeviceComplexScore.h"
#include "PsiFormerDeviceComplexKinetic.h"
#include <complex>

namespace qmcplusplus::psiformer::device
{
/// Direct numerical callers retain HOST results unless bounded output is explicit.
enum class ComplexProductsHostOutput { RETAINED, CHUNKED };

/// Complete numerical ownership, excluding shared immutable image/vendor/caller inputs.
struct ComplexProductsCost
{
  ComplexScoreCost score;
  ComplexKineticCost kinetic;
  std::size_t device_bytes, host_bytes, matrix_work, parameters, electrons;
  std::size_t channels, maximum_samples, chunk, statuses;
  std::size_t dot_partials, score_dot_scalar_work;
  ComplexProductsHostOutput host_output;
};

/// Sum both qualified owners and every prepared mixed-type product/status plane.
/// Work is maximum B times existing kinetic lifted-matrix sites, NOT score/neural FLOPs.
inline ComplexProductsCost estimateComplexProductsCost(const PsiFormerExecutionPlan& plan,
    std::size_t targets, std::size_t samples, std::size_t channels, std::size_t chunk,
    ComplexProductsHostOutput host_output = ComplexProductsHostOutput::RETAINED)
{
  const auto p = plan.parameterCount(), n = plan.modelShape().electrons();
  if (!samples || !channels || channels > 3 || !chunk || chunk > p)
    throw std::invalid_argument("Complex products require bounded B,1-3 channels and canonical chunk");
  if (host_output != ComplexProductsHostOutput::RETAINED && host_output != ComplexProductsHostOutput::CHUNKED)
    throw std::invalid_argument("Complex products HOST output policy is invalid");
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Complex products extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Complex products bytes overflow"); };
  const auto score = estimateComplexScoreCost(plan);
  const auto kinetic = estimateComplexKineticCost(plan, targets);
  const auto elements = mul(channels, p), doubles = mul(2, elements);
  const auto statuses = doubles / 4096 + (doubles % 4096 != 0);
  const auto status_bytes = mul(statuses, sizeof(DerivativeFiniteStatus));
  const auto partials = p / 4096 + (p % 4096 != 0);
  const auto dot_device = add(mul(8, add(p, add(partials, 2))), mul(add(partials, 1), sizeof(DerivativeFiniteStatus)));
  const auto dot_host = add(add(mul(8, p), mul(16, samples)), sizeof(DerivativeFiniteStatus));
  const auto device = add(add(add(score.device_bytes, kinetic.device_bytes), add(mul(16, elements), status_bytes)), dot_device);
  const auto host = add(add(score.host_bytes, kinetic.host_bytes),
      add(add(add(mul(sizeof(std::complex<double>), host_output == ComplexProductsHostOutput::RETAINED ? elements : chunk),
          mul(16, chunk)), add(mul(32, n), status_bytes)), dot_host));
  return {score, kinetic, device, host, mul(samples, kinetic.matrix_work), p, n, channels, samples, chunk, statuses,
      partials, mul(mul(2, p), samples), host_output};
}

/** TRANSPOSE/HERMITIAN complex SCORE/physical dK and full-score/device-dot REAL JVP.
 * Caller owns queue/BLAS/exclusive image execution lease for this owner's lifetime.
 * Immutable packed batch inputs/coefficient views are borrowed only during evaluate.
 * SOLE paired wavefunction only: optimizer policy/publication remain caller-owned.
 */
class PreparedDeviceComplexProducts
{
public:
  using Complex = std::complex<double>;
  /// Result receipts bind this address; owners cannot be copied or moved.
  PreparedDeviceComplexProducts(const PreparedDeviceComplexProducts&) = delete;
  PreparedDeviceComplexProducts& operator=(const PreparedDeviceComplexProducts&) = delete;
  PreparedDeviceComplexProducts(PreparedDeviceComplexProducts&&) = delete;
  PreparedDeviceComplexProducts& operator=(PreparedDeviceComplexProducts&&) = delete;

  /// Only the two qualified complete numerical responses may enter a contraction.
  enum class Role { SCORE, KINETIC };
  /// One caller-owned complex coefficient per packed sample, never conjugated.
  struct Channel { Role role; const Complex* values; std::size_t count; };
  /// Borrow the complete channel-major canonical host result until the next request.
  struct ResultView
  {
    const PreparedDeviceComplexProducts* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, channels, parameters;
    const Complex* data;
  };

  /// The execution lease owns this nonmovable owner; the receipt retains its exact old image.
  struct CompletedResult
  {
    const PreparedDeviceComplexProducts* owner = nullptr;
    std::shared_ptr<const PreparedDeviceModel> image;
    std::size_t generation = 0, channels = 0, parameters = 0;
  };

  /// Borrow completed per-sample complex JVPs, never a sample-by-parameter table.
  struct ScoreJvpView
  {
    const PreparedDeviceComplexProducts* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, samples;
    const Complex* data;
  };

  /// Borrow per-reference complete nonlocal JVPs, not plain score directional products.
  struct NonlocalJvpView
  {
    const PreparedDeviceComplexProducts* owner = nullptr;
    const PreparedDeviceModel* image = nullptr;
    std::size_t generation = 0, samples = 0;
    const Complex* data = nullptr;
  };

  enum class LocalTerms : unsigned { KINETIC = 1, NONLOCAL_ECP = 2, COMPLETE = 3 };

  /// A complete local-energy response cannot be consumed as a score or ECP-only receipt.
  struct LocalEnergyJvpView
  {
    const PreparedDeviceComplexProducts* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, samples;
    const Complex* data;
    LocalTerms terms;
  };

  /// Reject checked combined device/host/work caps before either numerical owner allocates.
  static std::unique_ptr<PreparedDeviceComplexProducts> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t targets, std::size_t samples, std::size_t channels, std::size_t chunk,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
      std::size_t device_budget, std::size_t host_budget, std::size_t work_budget,
      ComplexProductsHostOutput host_output = ComplexProductsHostOutput::RETAINED)
  {
    if (!image)
      throw std::invalid_argument("Complex products require an immutable paired image");
    image->validateCurrentDevice();
    const auto cost = estimateComplexProductsCost(image->plan(), targets, samples, channels, chunk, host_output);
    for (const auto cap : {device_budget, host_budget, work_budget})
      if (!cap || cap == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Complex products require positive finite numeric/work caps");
    if (cost.device_bytes > device_budget || cost.host_bytes > host_budget || cost.matrix_work > work_budget)
      throw std::length_error("Complex products exceed combined device/host/request-work admission");
    return std::unique_ptr<PreparedDeviceComplexProducts>(
        new PreparedDeviceComplexProducts(std::move(image), targets, queue, blas, cost));
  }

  /// Validate the WHOLE borrowed batch before any launch; publish only complete finite channels.
  void evaluate(const std::vector<double>& positions, const std::vector<double>& inverse_masses,
      const std::array<double, 3>& kappa, const std::vector<Channel>& channels, bool hermitian = false)
  {
    ready_ = jvp_ready_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex products generation exhausted");
    ++generation_;
    image_->validateCurrentDevice();
    const auto n = cost_.electrons, p = cost_.parameters, b = positions.size() / (3 * n);
    if (!b || b > cost_.maximum_samples || positions.size() != b * 3 * n ||
        inverse_masses.size() != b * n || channels.size() != cost_.channels)
      throw std::invalid_argument("Complex products packed batch/channel extent mismatch");
    const auto finite = [](double x) { return device_math::isFiniteBinary64(x); };
    for (double x : positions)
      if (!finite(x)) throw std::invalid_argument("Complex products nonfinite coordinate");
    for (double x : inverse_masses)
      if (!finite(x) || x <= 0) throw std::invalid_argument("Complex products require positive finite inverse masses");
    for (double x : kappa)
      if (!finite(x)) throw std::invalid_argument("Complex products nonfinite common kappa");
    bool score = false, kinetic = false;
    for (const auto& channel : channels)
    {
      if ((channel.role != Role::SCORE && channel.role != Role::KINETIC) || !channel.values || channel.count != b)
        throw std::invalid_argument("Complex products role/coefficient extent mismatch");
      score |= channel.role == Role::SCORE;
      kinetic |= channel.role == Role::KINETIC;
      for (std::size_t sample = 0; sample < b; ++sample)
        if (!finite(channel.values[sample].real()) || !finite(channel.values[sample].imag()))
          throw std::invalid_argument("Complex products nonfinite coefficient");
    }
    checkRuntime(launchDerivativeZero(queue_.getNative(), {2 * accumulated_.capacity(), 0, 0, 0, 1},
        reinterpret_cast<double*>(accumulated_.data())), "complex products zero");
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      std::copy_n(positions.data() + 3 * sample * n, 3 * n, positions_.data());
      std::copy_n(inverse_masses.data() + sample * n, n, masses_.data());
      const cuDoubleComplex *score_response = nullptr, *kinetic_response = nullptr;
      if (score)
      {
        score_->evaluate(positions_);
        const auto view = score_->resultView();
        if (view.image != image_.get() || view.count != p) throw std::logic_error("Complex products score binding changed");
        score_response = view.data;
      }
      if (kinetic)
      {
        kinetic_->evaluate(positions_, masses_, kappa);
        const auto view = kinetic_->resultView();
        if (view.image != image_.get() || view.count != p) throw std::logic_error("Complex products kinetic binding changed");
        kinetic_response = view.data;
      }
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        const auto* response = channels[channel].role == Role::SCORE ? score_response : kinetic_response;
        const auto coefficient = channels[channel].values[sample];
        accumulateResponse(response, coefficient, hermitian, channel);
      }
    }
    publishCompleted();
  }

  /** One regular-node complete nonlocal response, with caller-owned immutable grid/ratios.
   * C1 request only, using slot0 of any admitted channel capacity (other slots become zero).
   * B+Q counts all scalar score evaluations in the prepared request bound.
   * W contains the COMPLETE bare*total virtual/reference ratio, including fixed
   * components/outer phase. No public Hamiltonian or quadrature construction.
   */
  void evaluateNonlocalVJP(const std::vector<double>& base_positions,
      const std::vector<double>& virtual_positions, const std::vector<std::size_t>& references,
      const std::vector<Complex>& complete_weights, const std::vector<Complex>& coefficients,
      bool hermitian = false)
  {
    evaluateLocalEnergyVJP(base_positions, virtual_positions, references, complete_weights, coefficients,
        {}, {}, LocalTerms::NONLOCAL_ECP, hermitian);
  }

  /// Append kinetic and captured-grid responses before ONE finite publication in the same owner.
  void evaluateLocalEnergyVJP(const std::vector<double>& base_positions,
      const std::vector<double>& virtual_positions, const std::vector<std::size_t>& references,
      const std::vector<Complex>& complete_weights, const std::vector<Complex>& coefficients,
      const std::vector<double>& inverse_masses, const std::array<double, 3>& kappa,
      LocalTerms terms, bool hermitian = false)
  {
    ready_ = jvp_ready_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex products generation exhausted");
    ++generation_;
    image_->validateCurrentDevice();
    const auto stride = 3 * cost_.electrons;
    const auto b = base_positions.size() / stride, q = virtual_positions.size() / stride;
    const bool kinetic = hasKinetic(terms), nonlocal = hasNonlocal(terms);
    if (!cost_.channels || !b || b > cost_.maximum_samples || (nonlocal && q > cost_.maximum_samples - b) ||
        base_positions.size() % stride || virtual_positions.size() % stride ||
        coefficients.size() != b || references.size() != q || complete_weights.size() != q)
      throw std::invalid_argument("Complex nonlocal C1 packed grid/request extent mismatch");
    if (kinetic) validateKinetic(inverse_masses, kappa, b);
    const auto finite = [](Complex value) {
      return device_math::isFiniteBinary64(value.real()) && device_math::isFiniteBinary64(value.imag());
    };
    for (const auto* positions : {&base_positions, &virtual_positions})
      for (double value : *positions)
        if (!device_math::isFiniteBinary64(value))
          throw std::invalid_argument("Complex nonlocal nonfinite coordinate");
    for (auto value : coefficients)
      if (!finite(value)) throw std::invalid_argument("Complex nonlocal nonfinite sample coefficient");
    // Revoke JVP views before reusing their bounded O(B) storage for base weights.
    std::fill_n(sample_products_.begin(), b, Complex{});
    for (std::size_t point = 0; nonlocal && point < q; ++point)
    {
      const auto reference = references[point];
      if (reference >= b || !finite(complete_weights[point]))
        throw std::invalid_argument("Complex nonlocal invalid reference/complete weight");
      const auto weighted = coefficients[reference] *
          (hermitian ? std::conj(complete_weights[point]) : complete_weights[point]);
      if (!finite(weighted)) throw std::overflow_error("Complex nonlocal weighted coefficient overflow");
      sample_products_[reference] -= weighted;
      if (!finite(sample_products_[reference]))
        throw std::overflow_error("Complex nonlocal base coefficient overflow");
    }
    checkRuntime(launchDerivativeZero(queue_.getNative(), {2 * accumulated_.capacity(), 0, 0, 0, 1},
        reinterpret_cast<double*>(accumulated_.data())), "complex nonlocal zero");
    for (std::size_t point = 0; nonlocal && point < q; ++point)
    {
      const auto weighted = coefficients[references[point]] *
          (hermitian ? std::conj(complete_weights[point]) : complete_weights[point]);
      accumulateNonlocalScore(virtual_positions.data() + point * stride, weighted, hermitian);
    }
    // Evaluate even zero-weight references/virtuals: undefined node scores are not zero.
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      if (nonlocal)
        accumulateNonlocalScore(base_positions.data() + sample * stride, sample_products_[sample], hermitian);
      if (kinetic)
      {
        std::copy_n(base_positions.data() + sample * stride, stride, positions_.data());
        std::copy_n(inverse_masses.data() + sample * cost_.electrons, cost_.electrons, masses_.data());
        kinetic_->evaluate(positions_, masses_, kappa);
        const auto response = kinetic_->resultView();
        if (response.image != image_.get() || response.count != cost_.parameters)
          throw std::logic_error("Complex local-energy kinetic binding changed");
        accumulateResponse(response.data, coefficients[sample], hermitian);
      }
    }
    publishCompleted();
  }

  /** Full-score-then-device-dot JVP for a REAL canonical direction.
   * Upload oneP direction per request, reuse across B1 numerical samples, and
   * download only complex scalar/status results. No B*P Jacobian or CPU tape.
   */
  void evaluateScoreJVP(const std::vector<double>& positions, const Complex* direction, std::size_t count)
  {
    ready_ = jvp_ready_ = false;
    nonlocal_jvp_ = false;
    local_jvp_ = false;
    if (generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Complex products generation exhausted");
    ++generation_;
    image_->validateCurrentDevice();
    const auto n = cost_.electrons, p = cost_.parameters, b = positions.size() / (3 * n);
    if (!b || b > cost_.maximum_samples || positions.size() != b * 3 * n || !direction || count != p)
      throw std::invalid_argument("Complex score JVP coordinate/direction extent mismatch");
    for (double value : positions)
      if (!device_math::isFiniteBinary64(value))
        throw std::invalid_argument("Complex score JVP nonfinite coordinate");
    prepareDirection(direction, count);
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      std::copy_n(positions.data() + 3 * sample * n, 3 * n, positions_.data());
      score_->evaluate(positions_);
      const auto score = score_->resultView();
      if (score.image != image_.get() || score.count != p)
        throw std::logic_error("Complex score JVP numerical binding changed");
      sample_products_[sample] = dotResponse(score.data);
    }
    jvp_samples_ = b;
    jvp_ready_ = true;
  }

  /** Borrow combined [B base, Q virtual] packing and produce one complete response per base.
   * Existing full-score/device-dot work uploads one REAL direction; Wq includes
   * all fixed component and outer-phase ratios. No coordinate/scalar array is allocated.
   */
  void evaluateNonlocalJVP(const std::vector<double>& positions, std::size_t bases,
      const std::vector<std::size_t>& references, const std::vector<Complex>& complete_weights,
      const Complex* direction, std::size_t count)
  {
    ready_ = jvp_ready_ = false;
    const auto stride = 3 * cost_.electrons, total = positions.size() / stride;
    if (!cost_.channels || !bases || bases > total || total > cost_.maximum_samples ||
        positions.size() % stride || references.size() != total - bases || complete_weights.size() != references.size())
      throw std::invalid_argument("Complex nonlocal JVP C1 packed grid/request extent mismatch");
    const auto finite = [](Complex value) {
      return device_math::isFiniteBinary64(value.real()) && device_math::isFiniteBinary64(value.imag());
    };
    for (std::size_t point = 0; point < references.size(); ++point)
      if (references[point] >= bases || !finite(complete_weights[point]))
        throw std::invalid_argument("Complex nonlocal JVP invalid reference/complete weight");
    try
    {
      evaluateScoreJVP(positions, direction, count); // Checks every coordinate/direction and regular node.
      jvp_ready_ = false; // Internal score completion must never escape a later reduction failure.
      for (std::size_t point = 0; point < references.size(); ++point)
      {
        auto& term = sample_products_[bases + point];
        term = complete_weights[point] * (term - sample_products_[references[point]]);
        if (!finite(term)) throw std::overflow_error("Complex nonlocal JVP virtual product overflow");
      }
      // Every virtual difference is formed before any original base value is overwritten.
      std::fill_n(sample_products_.begin(), bases, Complex{});
      for (std::size_t point = 0; point < references.size(); ++point)
      {
        auto& sum = sample_products_[references[point]];
        sum += sample_products_[bases + point];
        if (!finite(sum)) throw std::overflow_error("Complex nonlocal JVP base accumulation overflow");
      }
      jvp_samples_ = bases;
      nonlocal_jvp_ = jvp_ready_ = true;
    }
    catch (...)
    {
      ready_ = jvp_ready_ = false;
      throw;
    }
  }

  /// Borrow [B base,Q virtual] packing; charge B extra dots when kinetic and ECP are combined.
  void evaluateLocalEnergyJVP(const std::vector<double>& positions, std::size_t bases,
      const std::vector<std::size_t>& references, const std::vector<Complex>& complete_weights,
      const std::vector<double>& inverse_masses, const std::array<double, 3>& kappa,
      LocalTerms terms, const Complex* direction, std::size_t count)
  {
    ready_ = jvp_ready_ = false;
    const bool kinetic = hasKinetic(terms), nonlocal = hasNonlocal(terms);
    const auto stride = 3 * cost_.electrons, total = positions.size() / stride;
    if (!bases || bases > total || bases > cost_.maximum_samples || positions.size() % stride ||
        (nonlocal && total > cost_.maximum_samples) ||
        (kinetic && nonlocal && bases > cost_.maximum_samples - total))
      throw std::invalid_argument("Complex local-energy JVP request work exceeds preparation");
    if (kinetic) validateKinetic(inverse_masses, kappa, bases);
    try
    {
      if (nonlocal)
        evaluateNonlocalJVP(positions, bases, references, complete_weights, direction, count);
      else
      {
        if (generation_ == std::numeric_limits<std::size_t>::max())
          throw std::overflow_error("Complex products generation exhausted");
        ++generation_;
        image_->validateCurrentDevice();
        for (double value : positions)
          if (!device_math::isFiniteBinary64(value))
            throw std::invalid_argument("Complex local-energy JVP nonfinite coordinate");
        prepareDirection(direction, count);
        std::fill_n(sample_products_.begin(), bases, Complex{});
      }
      jvp_ready_ = false;
      for (std::size_t sample = 0; kinetic && sample < bases; ++sample)
      {
        std::copy_n(positions.data() + sample * stride, stride, positions_.data());
        std::copy_n(inverse_masses.data() + sample * cost_.electrons, cost_.electrons, masses_.data());
        kinetic_->evaluate(positions_, masses_, kappa);
        const auto response = kinetic_->resultView();
        if (response.image != image_.get() || response.count != cost_.parameters)
          throw std::logic_error("Complex local-energy JVP kinetic binding changed");
        sample_products_[sample] += dotResponse(response.data);
        if (!finiteComplex(sample_products_[sample]))
          throw std::overflow_error("Complex local-energy JVP accumulation overflow");
      }
      jvp_samples_ = bases;
      local_terms_ = terms;
      local_jvp_ = jvp_ready_ = true;
    }
    catch (...)
    {
      ready_ = jvp_ready_ = false;
      throw;
    }
  }

  LocalEnergyJvpView localEnergyJvpView() const
  {
    image_->validateCurrentDevice();
    if (!jvp_ready_ || !local_jvp_)
      throw std::logic_error("Complex local-energy JVP response unavailable");
    return {this, image_.get(), generation_, jvp_samples_, sample_products_.data(), local_terms_};
  }

  void validate(const LocalEnergyJvpView& view) const
  {
    const auto current = localEnergyJvpView();
    if (view.owner != current.owner || view.image != current.image || view.generation != current.generation ||
        view.samples != current.samples || view.data != current.data || view.terms != current.terms)
      throw std::invalid_argument("Complex local-energy JVP stale/foreign result receipt");
  }

  /// Publish only the complete finite physical response with its distinct mathematical receipt.
  NonlocalJvpView nonlocalJvpView() const
  {
    image_->validateCurrentDevice();
    if (!jvp_ready_ || !nonlocal_jvp_ || local_jvp_) throw std::logic_error("Complex nonlocal JVP response unavailable");
    return {this, image_.get(), generation_, jvp_samples_, sample_products_.data()};
  }

  /// Reject a foreign/stale receipt, including replacement by a plain score-JVP request.
  void validate(const NonlocalJvpView& view) const
  {
    const auto current = nonlocalJvpView();
    if (view.owner != current.owner || view.image != current.image || view.generation != current.generation ||
        view.samples != current.samples || view.data != current.data)
      throw std::invalid_argument("Complex nonlocal JVP stale/foreign result receipt");
  }

  /// Publish only a complete finite JVP request at this image and generation.
  ScoreJvpView scoreJvpView() const
  {
    image_->validateCurrentDevice();
    if (!jvp_ready_ || nonlocal_jvp_ || local_jvp_)
      throw std::logic_error("Complex score JVP response unavailable");
    return {this, image_.get(), generation_, jvp_samples_, sample_products_.data()};
  }

  /// Reject a JVP receipt invalidated by any next or failed owner request.
  void validate(const ScoreJvpView& view) const
  {
    const auto current = scoreJvpView();
    if (view.owner != current.owner || view.image != current.image || view.generation != current.generation ||
        view.samples != current.samples || view.data != current.data)
      throw std::invalid_argument("Complex score JVP stale/foreign result receipt");
  }

  /// Publish no incomplete/failed request or unowned numerical storage.
  ResultView resultView() const
  {
    image_->validateCurrentDevice();
    if (!ready_ || cost_.host_output != ComplexProductsHostOutput::RETAINED)
      throw std::logic_error("Complex products retained HOST response unavailable");
    return {this, image_.get(), generation_, cost_.channels, cost_.parameters, completed_.data()};
  }

  /// Reject foreign/stale/image/address/count receipts before a consumer uses the result.
  void validate(const ResultView& view) const
  {
    const auto current = resultView();
    if (view.owner != current.owner || view.image != current.image || view.generation != current.generation ||
        view.channels != current.channels || view.parameters != current.parameters || view.data != current.data)
      throw std::invalid_argument("Complex products stale/foreign result receipt");
  }

  /// Bind only a fully finite DEVICE contraction, never a partial request.
  CompletedResult completedResult() const
  {
    image_->validateCurrentDevice();
    if (!ready_) throw std::logic_error("Complex products completion unavailable");
    return {this, image_, generation_, cost_.channels, cost_.parameters};
  }

  /// Download one bounded canonical range; the returned HOST view lasts until the next chunk/request.
  const Complex* downloadCompletedChunk(const CompletedResult& receipt, std::size_t channel,
      std::size_t offset, std::size_t count)
  {
    const auto current = completedResult();
    if (cost_.host_output != ComplexProductsHostOutput::CHUNKED || receipt.owner != this ||
        receipt.image != current.image || receipt.generation != current.generation ||
        receipt.channels != current.channels || receipt.parameters != current.parameters ||
        channel >= cost_.channels || !count || count > cost_.chunk ||
        offset > cost_.parameters || count > cost_.parameters - offset)
      throw std::invalid_argument("Complex products chunk receipt or canonical range is stale/foreign/invalid");
    accumulated_.downloadOnStream(staging_.data(), count, queue_.getNative(), channel * cost_.parameters + offset);
    for (std::size_t i = 0; i < count; ++i)
      completed_[i] = Complex(cuCreal(staging_[i]), cuCimag(staging_[i]));
    return completed_.data();
  }

  /// Exact inventory excludes shared image/vendor/caller-owned borrowed batch inputs.
  const ComplexProductsCost& cost() const noexcept { return cost_; }

  /// Track prepared owner/staging identities separately from request generations.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t result = 0;
    for (const void* pointer : std::array<const void*, 14>{score_.get(), kinetic_.get(), accumulated_.data(),
        status_.data(), host_status_.data(), completed_.data(), staging_.data(), positions_.data(), direction_.data(),
        direction_host_.data(), dot_workspace_.data(), dot_status_.data(), dot_host_status_.data(), sample_products_.data()})
      result = (result ^ reinterpret_cast<std::uintptr_t>(pointer)) * 1099511628211ULL;
    return result ^ reinterpret_cast<std::uintptr_t>(masses_.data()) ^ static_cast<std::size_t>(cost_.host_output);
  }

private:
  static bool finiteComplex(Complex value)
  {
    return device_math::isFiniteBinary64(value.real()) && device_math::isFiniteBinary64(value.imag());
  }

  static bool hasKinetic(LocalTerms terms)
  {
    if (terms != LocalTerms::KINETIC && terms != LocalTerms::NONLOCAL_ECP && terms != LocalTerms::COMPLETE)
      throw std::invalid_argument("Complex local-energy term mask unsupported");
    return terms != LocalTerms::NONLOCAL_ECP;
  }

  static bool hasNonlocal(LocalTerms terms) { return terms != LocalTerms::KINETIC; }

  void validateKinetic(const std::vector<double>& masses, const std::array<double, 3>& kappa, std::size_t samples) const
  {
    if (masses.size() != samples * cost_.electrons)
      throw std::invalid_argument("Complex local-energy inverse-mass extent mismatch");
    for (double value : masses)
      if (!device_math::isFiniteBinary64(value) || value <= 0)
        throw std::invalid_argument("Complex local-energy inverse masses must be positive finite");
    for (double value : kappa)
      if (!device_math::isFiniteBinary64(value)) throw std::invalid_argument("Complex local-energy kappa nonfinite");
  }

  void prepareDirection(const Complex* direction, std::size_t count)
  {
    if (!direction || count != cost_.parameters)
      throw std::invalid_argument("Complex directional response extent mismatch");
    for (std::size_t i = 0; i < count; ++i)
    {
      if (!device_math::isFiniteBinary64(direction[i].real()) || direction[i].imag() != 0)
        throw std::invalid_argument("Complex score JVP requires a finite REAL canonical direction");
      direction_host_[i] = direction[i].real();
    }
    direction_.upload(direction_host_.data(), count);
  }

  Complex dotResponse(const cuDoubleComplex* response)
  {
    for (std::size_t component = 0; component < 2; ++component)
    {
      const ResponseDotLayout layout{cost_.parameters, 4096, cost_.dot_partials, component, 2, 0, 1};
      checkRuntime(launchResponseDotDirection(queue_.getNative(), layout,
          reinterpret_cast<const double*>(response), direction_.data(), dot_workspace_.data(), dot_status_.data(),
          dot_workspace_.data() + cost_.dot_partials + component, dot_status_.data() + cost_.dot_partials),
          "complex response directional dot");
      queue_.sync();
      dot_status_.download(dot_host_status_.data(), 1, cost_.dot_partials);
      if (dot_host_status_[0].nonfinite_count) throw std::runtime_error("Complex directional dot is nonfinite");
    }
    std::array<double, 2> result;
    dot_workspace_.download(result.data(), 2, cost_.dot_partials);
    return {result[0], result[1]};
  }

  void accumulateResponse(const cuDoubleComplex* response, Complex coefficient, bool hermitian,
      std::size_t channel = 0)
  {
    static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double));
    for (std::size_t component = 0; component < 2; ++component)
    {
      const double sign = hermitian && component ? -1 : 1;
      checkRuntime(launchRealToComplexAxpy(queue_.getNative(),
          {cost_.parameters, component, 2, 2 * channel * cost_.parameters, 2},
          sign * (component ? -coefficient.imag() : coefficient.real()),
          sign * (component ? coefficient.real() : coefficient.imag()), reinterpret_cast<const double*>(response),
          reinterpret_cast<double*>(accumulated_.data())), "complex response AXPY");
    }
  }

  /// Accumulate one admitted B1 score with the same TRANSPOSE/HERMITIAN algebra as ordinary products.
  void accumulateNonlocalScore(const double* positions, Complex coefficient, bool hermitian)
  {
    std::copy_n(positions, positions_.size(), positions_.data());
    score_->evaluate(positions_);
    const auto score = score_->resultView();
    if (score.image != image_.get() || score.count != cost_.parameters)
      throw std::logic_error("Complex nonlocal score binding changed");
    accumulateResponse(score.data, coefficient, hermitian);
  }

  /// Publish only a complete finite contraction; both product routes share the same receipt protocol.
  void publishCompleted()
  {
    checkRuntime(launchDerivativeFiniteScan(queue_.getNative(), {2 * accumulated_.capacity(), 4096, cost_.statuses, 0, 1},
        reinterpret_cast<const double*>(accumulated_.data()), status_.data()), "complex products finite publication");
    queue_.sync(); // Nonblocking compute queues must complete before synchronous default-stream downloads.
    status_.download(host_status_.data(), host_status_.size());
    for (const auto& status : host_status_)
      if (status.nonfinite_count) throw std::runtime_error("Complex products complete contraction is nonfinite");
    for (std::size_t offset = 0; cost_.host_output == ComplexProductsHostOutput::RETAINED &&
        offset < completed_.size(); offset += cost_.chunk)
    {
      const auto count = std::min(cost_.chunk, completed_.size() - offset);
      accumulated_.download(staging_.data(), count, offset);
      for (std::size_t i = 0; i < count; ++i)
        completed_[offset + i] = Complex(cuCreal(staging_[i]), cuCimag(staging_[i]));
    }
    ready_ = true;
  }

  /// Materialize only checked capacities, including finite partials in BOTH memory domains.
  PreparedDeviceComplexProducts(std::shared_ptr<const PreparedDeviceModel> image, std::size_t targets,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, ComplexProductsCost cost)
      : image_(std::move(image)), queue_(queue), cost_(cost),
        score_(PreparedDeviceComplexScore::prepare(image_, queue, blas, cost.score.device_bytes)),
        kinetic_(PreparedDeviceComplexKinetic::prepare(image_, targets, queue, blas, cost.kinetic.device_bytes, cost.kinetic.matrix_work)),
        accumulated_(cost.channels * cost.parameters), status_(cost.statuses), host_status_(cost.statuses),
        completed_(cost.host_output == ComplexProductsHostOutput::RETAINED ? cost.channels * cost.parameters : cost.chunk),
        staging_(cost.chunk), positions_(3 * cost.electrons), masses_(cost.electrons),
        direction_(cost.parameters), dot_workspace_(cost.dot_partials + 2), dot_status_(cost.dot_partials + 1),
        direction_host_(cost.parameters), sample_products_(cost.maximum_samples), dot_host_status_(1)
  {
    const auto device = score_->cost().device_bytes + kinetic_->cost().device_bytes +
        16 * accumulated_.capacity() + sizeof(DerivativeFiniteStatus) * status_.capacity() +
        8 * (direction_.capacity() + dot_workspace_.capacity()) + sizeof(DerivativeFiniteStatus) * dot_status_.capacity();
    const auto host = score_->cost().host_bytes + kinetic_->cost().host_bytes +
        sizeof(Complex) * completed_.capacity() + 16 * staging_.capacity() +
        8 * (positions_.capacity() + masses_.capacity() + direction_host_.capacity()) +
        sizeof(DerivativeFiniteStatus) * (host_status_.capacity() + dot_host_status_.capacity()) +
        sizeof(Complex) * sample_products_.capacity();
    if (device != cost_.device_bytes || host != cost_.host_bytes)
      throw std::logic_error("Complex products actual inventory differs from checked admission");
  }

  const std::shared_ptr<const PreparedDeviceModel> image_;
  compute::Queue<PlatformKind::CUDA>& queue_;
  const ComplexProductsCost cost_;
  std::unique_ptr<PreparedDeviceComplexScore> score_;
  std::unique_ptr<PreparedDeviceComplexKinetic> kinetic_;
  DeviceBuffer<cuDoubleComplex> accumulated_;
  DeviceBuffer<DerivativeFiniteStatus> status_;
  std::vector<DerivativeFiniteStatus> host_status_;
  std::vector<Complex> completed_;
  std::vector<cuDoubleComplex> staging_;
  std::vector<double> positions_, masses_;
  DeviceBuffer<double> direction_, dot_workspace_;
  DeviceBuffer<DerivativeFiniteStatus> dot_status_;
  std::vector<double> direction_host_;
  std::vector<Complex> sample_products_;
  std::vector<DerivativeFiniteStatus> dot_host_status_;
  std::size_t jvp_samples_ = 0;
  std::size_t generation_ = 0;
  bool ready_ = false, jvp_ready_ = false, nonlocal_jvp_ = false;
  bool local_jvp_ = false;
  LocalTerms local_terms_ = LocalTerms::NONLOCAL_ECP;
};
} // namespace qmcplusplus::psiformer::device
#endif
