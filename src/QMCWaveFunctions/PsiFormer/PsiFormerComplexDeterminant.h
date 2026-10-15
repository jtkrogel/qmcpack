//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexDeterminant.h
 * @brief Bounded CPU complex full/spin-block determinant values and spatial algebra.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_DETERMINANT_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_DETERMINANT_H

#include "PsiFormerDeterminant.h"
#include <complex>
#include <array>

namespace qmcplusplus::psiformer::determinant
{
using Complex = std::complex<double>;
using ComplexLogAmplitude = LogAmplitude<Complex>;

/// Validate both Cartesian components using the release-safe binary64 predicate.
inline bool isFiniteComplex(Complex value) noexcept
{
  return isFiniteReal(value.real()) && isFiniteReal(value.imag());
}

/// Value, derivative availability and cancellation diagnostics for one reduction.
struct ComplexDeterminantResult
{
  ComplexLogAmplitude amplitude;
  std::size_t singular_channels = 0;
  bool all_required_inverses_available = false;
  /// |sum scaled channels| / sum |scaled channels|; zero denotes exact cancellation.
  double cancellation_ratio = 0.0;
};

/** Prepared CPU complex determinant algebra, independent of geometry and twist.
 *
 * Each channel stores consecutive row-major blocks of sizes first_size and
 * second_size. A full determinant uses (N,0); spin blocks use (Nup,Ndown).
 * An empty block has determinant one. The expansion is sum_d c_d det(U_d)det(D_d),
 * never the product of independent channel sums. Spatial arrays are lane-major
 * over the complete channel batch; both blocks may depend on every target lane.
 *
 * This scalar reference kernel owns O(D*(Nu^2+Nd^2)+lanes) storage, allocated at
 * construction. It does not imply public complex-model or GPU qualification.
 */
class ComplexDeterminantWorkspace
{
  using Wide = std::complex<long double>;
  /// Compensate real and imaginary components independently in extended precision.
  struct Sum
  {
    detail::CompensatedSum re, im;

    /// Accumulate one ordinary complex term.
    void add(Wide z)
    {
      re.add(z.real());
      im.add(z.imag());
    }

    /// Recover the corrected Cartesian sum.
    Wide value() const { return {re.value(), im.value()}; }
  };

public:
  /// Exact vector-element requirement without allocating factors or derivative buffers.
  static std::size_t storageRequirement(std::size_t channels, std::size_t first_size,
      std::size_t second_size = 0, std::size_t gradient_lanes = 0, std::size_t laplacian_lanes = 0)
  {
    if (!channels || (laplacian_lanes && gradient_lanes != product(3, laplacian_lanes)))
      throw std::invalid_argument("invalid complex determinant workspace dimensions");
    const auto first_square = product(first_size, first_size);
    const auto second_square = product(second_size, second_size);
    const auto matrices = product(channels, sum(first_square, second_square));
    std::size_t bytes = product(sizeof(Complex),
        sum(product(3, matrices), sum(gradient_lanes, product(2, laplacian_lanes))));
    bytes = sum(bytes, product(sizeof(std::size_t), product(channels, sum(first_size, second_size))));
    bytes = sum(bytes, product(sizeof(ComplexLogAmplitude), product(3, channels)));
    bytes = sum(bytes, product(sizeof(unsigned char), product(2, channels)));
    return sum(bytes, product(sizeof(Wide), sum(product(2, channels),
        sum(std::max(first_size, second_size), std::max(first_square, second_square)))));
  }

  /// Reserve factorization and requested derivative scratch before evaluation.
  ComplexDeterminantWorkspace(std::size_t channels, std::size_t first_size,
                              std::size_t second_size = 0, std::size_t gradient_lanes = 0,
                              std::size_t laplacian_lanes = 0)
      : channels_(channels), sizes_{first_size, second_size},
        offsets_{0, product(first_size, first_size)},
        elements_(sum(offsets_[1], product(second_size, second_size))),
        gradient_capacity_(gradient_lanes), laplacian_capacity_(laplacian_lanes),
        lu_(product(channels, elements_)), inverse_(lu_.size()),
        permutation_(product(channels, sum(first_size, second_size))),
        block_amplitude_(product(channels, 2)), available_(product(channels, 2)),
        terms_(channels), scaled_terms_(channels), weights_(channels),
        solve_(std::max(first_size, second_size)),
        matrix_product_(std::max(offsets_[1], product(second_size, second_size))),
        gradient_(gradient_lanes), lap_ratio_(laplacian_lanes), lap_log_(laplacian_lanes),
        adjoint_(lu_.size())
  {
    if (!channels || (laplacian_lanes && gradient_lanes != product(3, laplacian_lanes)))
      throw std::invalid_argument("invalid complex determinant workspace dimensions");
  }

  /// Mutable factorization scratch cannot be shared through copying.
  ComplexDeterminantWorkspace(const ComplexDeterminantWorkspace&) = delete;

  /// Preserve exclusive workspace ownership on assignment.
  ComplexDeterminantWorkspace& operator=(const ComplexDeterminantWorkspace&) = delete;

  /// Transfer the complete prepared allocation set.
  ComplexDeterminantWorkspace(ComplexDeterminantWorkspace&&) = default;

  /// Transfer ownership of mutable factors and staged results.
  ComplexDeterminantWorkspace& operator=(ComplexDeterminantWorkspace&&) = default;

  /// Number of matrix entries in a channel's consecutive blocks.
  std::size_t matrixElements() const noexcept { return elements_; }

  /// Number of paired terms in the determinant expansion.
  std::size_t channels() const noexcept { return channels_; }

  /// Factor and reduce, optionally omitting inverse solves for value-only requests.
  ComplexDeterminantResult evaluate(const Complex* matrices, const Complex* coefficients = nullptr,
                                    bool prepare_inverses = true)
  {
    evaluated_ = false;
    if (elements_ && !matrices)
      throw std::invalid_argument("null complex determinant matrices");
    double shift = -std::numeric_limits<double>::infinity();
    ComplexDeterminantResult result;
    result.all_required_inverses_available = true;
    for (std::size_t c = 0; c < channels_; ++c)
    {
      const Complex coefficient = coefficients ? coefficients[c] : Complex{1};
      if (!isFiniteComplex(coefficient))
        throw std::invalid_argument("nonfinite complex determinant coefficient");
      bool singular = false;
      ComplexLogAmplitude term{Complex{1}, 0};
      for (std::size_t b = 0; b < 2; ++b)
      {
        factor(matrices, c, b, prepare_inverses);
        const auto& block = block_amplitude_[2 * c + b];
        singular = singular || block.isZero();
        term.phase *= block.phase;
        term.log_abs += block.log_abs;
        if (coefficient != Complex{} && !available_[2 * c + b])
          result.all_required_inverses_available = false;
      }
      result.singular_channels += singular;
      if (coefficient == Complex{} || singular)
        term = {};
      else
      {
        const long double magnitude = std::abs(Wide(coefficient));
        term.phase *= narrow(Wide(coefficient) / magnitude);
        term.log_abs += static_cast<double>(std::log(magnitude));
        if (!isFiniteReal(term.log_abs))
          throw std::overflow_error("complex determinant channel logarithm overflow");
        shift = std::max(shift, term.log_abs);
      }
      terms_[c] = term;
    }
    Sum total;
    detail::CompensatedSum magnitudes;
    for (std::size_t c = 0; c < channels_; ++c)
    {
      scaled_terms_[c] = terms_[c].isZero() ? Wide{} : Wide(terms_[c].phase) *
          std::exp(static_cast<long double>(terms_[c].log_abs) - shift);
      total.add(scaled_terms_[c]);
      magnitudes.add(std::abs(scaled_terms_[c]));
    }
    const Wide value = total.value();
    const long double magnitude = std::abs(value);
    if (magnitude != 0)
    {
      result.amplitude = {narrow(value / magnitude), shift + static_cast<double>(std::log(magnitude))};
      result.cancellation_ratio = static_cast<double>(magnitude / magnitudes.value());
      for (std::size_t c = 0; c < channels_; ++c)
        weights_[c] = scaled_terms_[c] / value;
    }
    else
      std::fill(weights_.begin(), weights_.end(), Wide{});
    result_ = result;
    evaluated_ = true;
    return result;
  }

  /** Exact analytic spatial derivatives; every product is ordinary complex.
   * Outputs publish only after every lane succeeds. Singular nonzero-coefficient
   * channels reject derivatives even when their value contribution vanishes:
   * their determinant derivative need not vanish. Zero coefficients are exempt.
   */
  ComplexDeterminantResult evaluateSpatial(const Complex* matrices, const Complex* gradients,
      std::size_t gradient_lanes, const Complex* laplacians, std::size_t laplacian_lanes,
      Complex* output_gradient, Complex* output_lap_log, Complex* output_lap_ratio,
      const Complex* coefficients = nullptr)
  {
    evaluated_ = false;
    if (gradient_lanes > gradient_capacity_ || laplacian_lanes > laplacian_capacity_ ||
        (laplacian_lanes && gradient_lanes != product(3, laplacian_lanes)) ||
        (gradient_lanes && (!gradients || !output_gradient)) ||
        (laplacian_lanes && (!laplacians || !output_lap_log || !output_lap_ratio)))
      throw std::invalid_argument("invalid complex determinant spatial views");
    validate(gradients, product(gradient_lanes, lu_.size()));
    validate(laplacians, product(laplacian_lanes, lu_.size()));
    const auto result = evaluate(matrices, coefficients);
    requireDerivatives();
    evaluated_ = false;
    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    {
      Sum total;
      for (std::size_t c = 0; c < channels_; ++c)
        if (weights_[c] != Wide{})
          total.add(weights_[c] * channelTrace(c, gradients + lane * lu_.size()));
      gradient_[lane] = narrow(total.value());
    }
    for (std::size_t e = 0; e < laplacian_lanes; ++e)
    {
      Sum total;
      for (std::size_t c = 0; c < channels_; ++c)
      {
        if (weights_[c] == Wide{})
          continue;
        Wide ratio = channelTrace(c, laplacians + e * lu_.size());
        for (std::size_t dim = 0; dim < 3; ++dim)
        {
          const Complex* direction = gradients + (3 * e + dim) * lu_.size();
          const Wide trace = channelTrace(c, direction);
          // Squaring the SUM of block traces supplies both spin cross terms.
          ratio += trace * trace - channelSquareTrace(c, direction);
        }
        total.add(weights_[c] * ratio);
      }
      Wide squared_gradient{};
      for (std::size_t dim = 0; dim < 3; ++dim)
        squared_gradient += Wide(gradient_[3 * e + dim]) * Wide(gradient_[3 * e + dim]);
      lap_ratio_[e] = narrow(total.value());
      lap_log_[e] = narrow(total.value() - squared_gradient);
    }
    std::copy_n(gradient_.data(), gradient_lanes, output_gradient);
    std::copy_n(lap_log_.data(), laplacian_lanes, output_lap_log);
    std::copy_n(lap_ratio_.data(), laplacian_lanes, output_lap_ratio);
    evaluated_ = true;
    return result;
  }

  /** Analytic seed d log(S)/d A_ij = weight * inverse_ji, without conjugation.
   * For real parameters use this seed with complex dA/dtheta; any Hermitian
   * objective contraction and real projection belongs at its caller's boundary.
   */
  void fillLogMatrixAdjoints(Complex* output)
  {
    requireDerivatives();
    if (elements_ && !output)
      throw std::invalid_argument("null complex matrix adjoint");
    for (std::size_t c = 0; c < channels_; ++c)
      for (std::size_t b = 0; b < 2; ++b)
      {
        const std::size_t n = sizes_[b], offset = c * elements_ + offsets_[b];
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
            adjoint_[offset + i * n + j] = weights_[c] == Wide{} ? Complex{} :
                narrow(weights_[c] * Wide(inverse_[offset + j * n + i]));
      }
    std::copy(adjoint_.begin(), adjoint_.end(), output);
  }

  /// Expose only a completed value or compound spatial evaluation.
  const ComplexDeterminantResult& result() const
  {
    if (!evaluated_)
      throw std::logic_error("complex determinant result unavailable");
    return result_;
  }

  /// Count every explicitly owned vector; value-only calls still reserve inverse scratch.
  std::size_t storageBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&bytes](const auto& buffer) {
      using Element = typename std::decay_t<decltype(buffer)>::value_type;
      bytes = sum(bytes, product(buffer.capacity(), sizeof(Element)));
    };
    add(lu_);
    add(inverse_);
    add(permutation_);
    add(block_amplitude_);
    add(available_);
    add(terms_);
    add(scaled_terms_);
    add(weights_);
    add(solve_);
    add(matrix_product_);
    add(gradient_);
    add(lap_ratio_);
    add(lap_log_);
    add(adjoint_);
    return bytes;
  }

  /// Allocation identity permits warmed no-growth checks without a global allocator.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    detail::mixStorage(hash, lu_);
    detail::mixStorage(hash, inverse_);
    detail::mixStorage(hash, permutation_);
    detail::mixStorage(hash, block_amplitude_);
    detail::mixStorage(hash, available_);
    detail::mixStorage(hash, terms_);
    detail::mixStorage(hash, scaled_terms_);
    detail::mixStorage(hash, weights_);
    detail::mixStorage(hash, solve_);
    detail::mixStorage(hash, matrix_product_);
    detail::mixStorage(hash, gradient_);
    detail::mixStorage(hash, lap_ratio_);
    detail::mixStorage(hash, lap_log_);
    detail::mixStorage(hash, adjoint_);
    return hash;
  }

private:
  /// Reject element-count multiplication overflow before constructing vectors.
  static std::size_t product(std::size_t a, std::size_t b)
  {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
      throw std::overflow_error("complex determinant storage overflow");
    return a * b;
  }

  /// Reject element-count addition overflow before constructing vectors.
  static std::size_t sum(std::size_t a, std::size_t b)
  {
    if (a > std::numeric_limits<std::size_t>::max() - b)
      throw std::overflow_error("complex determinant storage overflow");
    return a + b;
  }

  /// Reject out-of-range Cartesian components before narrowing an analytic result.
  static Complex narrow(Wide value)
  {
    return {detail::narrowFinite(value.real(), "complex determinant real overflow"),
            detail::narrowFinite(value.imag(), "complex determinant imaginary overflow")};
  }

  /// Check each supplied matrix or spatial lane before publishing any result.
  static void validate(const Complex* data, std::size_t count)
  {
    for (std::size_t i = 0; i < count; ++i)
      if (!isFiniteComplex(data[i]))
        throw std::invalid_argument("nonfinite complex determinant input");
  }

  /// Reject nodes and every required unavailable inverse, invalidating failed state.
  void requireDerivatives()
  {
    if (!evaluated_)
      throw std::logic_error("complex determinant derivatives before evaluation");
    if (result_.amplitude.isZero() || !result_.all_required_inverses_available)
    {
      evaluated_ = false;
      throw std::domain_error("complex determinant derivatives unavailable at node or singular channel");
    }
  }

  /// Scale, partially pivot and solve each inverse column; empty blocks are identities.
  void factor(const Complex* matrices, std::size_t c, std::size_t b, bool invert)
  {
    const std::size_t n = sizes_[b], offset = c * elements_ + offsets_[b], slot = 2 * c + b;
    available_[slot] = false;
    block_amplitude_[slot] = {};
    if (!n)
    {
      block_amplitude_[slot] = {Complex{1}, 0};
      available_[slot] = true;
      return;
    }
    const Complex* input = matrices + offset;
    validate(input, n * n);
    double scale = 0;
    for (std::size_t i = 0; i < n * n; ++i)
      scale = std::max({scale, std::abs(input[i].real()), std::abs(input[i].imag())});
    if (scale == 0)
      return;
    Complex* lu = lu_.data() + offset;
    std::size_t* perm = permutation_.data() + c * (sizes_[0] + sizes_[1]) + (b ? sizes_[0] : 0);
    for (std::size_t i = 0; i < n * n; ++i)
      lu[i] = input[i] / scale;
    for (std::size_t i = 0; i < n; ++i)
      perm[i] = i;
    Wide phase{1};
    long double log_abs = static_cast<long double>(n) * std::log(static_cast<long double>(scale));
    for (std::size_t k = 0; k < n; ++k)
    {
      std::size_t pivot = k;
      for (std::size_t i = k + 1; i < n; ++i)
        if (std::abs(lu[i * n + k]) > std::abs(lu[pivot * n + k]))
          pivot = i;
      if (lu[pivot * n + k] == Complex{})
        return;
      if (pivot != k)
      {
        for (std::size_t j = 0; j < n; ++j)
          std::swap(lu[k * n + j], lu[pivot * n + j]);
        std::swap(perm[k], perm[pivot]);
        phase = -phase;
      }
      const Wide diagonal(lu[k * n + k]);
      const long double magnitude = std::abs(diagonal);
      phase *= diagonal / magnitude;
      log_abs += std::log(magnitude);
      for (std::size_t i = k + 1; i < n; ++i)
      {
        lu[i * n + k] /= lu[k * n + k];
        for (std::size_t j = k + 1; j < n; ++j)
          lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
      }
    }
    validate(lu, n * n);
    block_amplitude_[slot] = {narrow(phase / std::abs(phase)), static_cast<double>(log_abs)};
    if (!invert)
      return;
    // Inverse overflow leaves a valid value but explicitly unavailable derivatives.
    try
    {
      for (std::size_t col = 0; col < n; ++col)
      {
        for (std::size_t i = 0; i < n; ++i)
        {
          Wide value{perm[i] == col ? 1.0L : 0.0L};
          for (std::size_t j = 0; j < i; ++j)
            value -= Wide(lu[i * n + j]) * solve_[j];
          solve_[i] = value;
        }
        for (std::size_t i = n; i-- > 0;)
        {
          for (std::size_t j = i + 1; j < n; ++j)
            solve_[i] -= Wide(lu[i * n + j]) * solve_[j];
          solve_[i] /= Wide(lu[i * n + i]);
          inverse_[offset + i * n + col] = narrow(solve_[i] / static_cast<long double>(scale));
        }
      }
      available_[slot] = true;
    }
    catch (const std::overflow_error&)
    {
      // A finite logarithmic value remains usable without a representable inverse.
    }
  }

  /// Sum tr(A_block^-1 dA_block) over both blocks using ordinary products.
  Wide channelTrace(std::size_t c, const Complex* derivative) const
  {
    Sum trace;
    for (std::size_t b = 0; b < 2; ++b)
    {
      const std::size_t n = sizes_[b], offset = c * elements_ + offsets_[b];
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          trace.add(Wide(inverse_[offset + i * n + j]) * Wide(derivative[offset + j * n + i]));
    }
    return trace.value();
  }

  /// Sum tr((A_block^-1 dA_block)^2), explicitly avoiding Hermitian norms.
  Wide channelSquareTrace(std::size_t c, const Complex* derivative)
  {
    Sum trace;
    for (std::size_t b = 0; b < 2; ++b)
    {
      const std::size_t n = sizes_[b], offset = c * elements_ + offsets_[b];
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
          Sum entry;
          for (std::size_t k = 0; k < n; ++k)
            entry.add(Wide(inverse_[offset + i * n + k]) * Wide(derivative[offset + k * n + j]));
          matrix_product_[i * n + j] = entry.value();
        }
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          trace.add(matrix_product_[i * n + j] * matrix_product_[j * n + i]);
    }
    return trace.value();
  }

  std::size_t channels_;
  std::array<std::size_t, 2> sizes_, offsets_;
  std::size_t elements_, gradient_capacity_, laplacian_capacity_;
  std::vector<Complex> lu_, inverse_;
  std::vector<std::size_t> permutation_;
  std::vector<ComplexLogAmplitude> block_amplitude_;
  std::vector<unsigned char> available_;
  std::vector<ComplexLogAmplitude> terms_;
  std::vector<Wide> scaled_terms_, weights_, solve_, matrix_product_;
  std::vector<Complex> gradient_, lap_ratio_, lap_log_, adjoint_;
  ComplexDeterminantResult result_;
  bool evaluated_ = false;
};
} // namespace qmcplusplus::psiformer::determinant
#endif
