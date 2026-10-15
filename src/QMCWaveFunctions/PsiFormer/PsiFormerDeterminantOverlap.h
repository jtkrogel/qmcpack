//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeterminantOverlap.h
 * @brief Bounded-memory normalized-overlap statistics for PsiFormer channels.
 *
 * Samples are drawn from a frozen teacher T.  For every determinant channel k,
 * the accumulator forms a=Phi_k/T and b=Psi_k/T in signed-log arithmetic and
 * minimizes -w_k log(|<a b>|^2/(<a^2><b^2>)).  A first streaming pass retains
 * only O(M) moments.  A second pass returns M scalar channel-score seeds, which
 * can be consumed by DirectScoreExecutor's single shared-trunk reverse pass.
 * No walker-by-channel or channel-by-parameter matrix is retained.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DETERMINANT_OVERLAP_H
#define QMCPLUSPLUS_PSIFORMER_DETERMINANT_OVERLAP_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{

/** Represent one real amplitude by its sign and logarithmic magnitude. */
struct RealSignedLogAmplitude
{
  double phase   = 0.0;
  double log_abs = -std::numeric_limits<double>::infinity();

  /// Return whether this is the canonical representation of an exact zero.
  bool isZero() const noexcept { return phase == 0.0; }
};

/** Set numerical validity thresholds for normalized-overlap training. */
struct DeterminantOverlapPolicy
{
  double minimum_fidelity             = 1.0e-14;
  double minimum_effective_samples    = 1.0;
};

/** Report the normalized overlap and sampling quality of one channel. */
struct DeterminantOverlapChannelDiagnostics
{
  double log_fidelity             = -std::numeric_limits<double>::infinity();
  double fidelity                 = 0.0;
  double target_effective_samples = 0.0;
  double student_effective_samples = 0.0;
};

/** Serializable first-pass moments used by stable rank-wise reduction. */
struct DeterminantOverlapMomentSnapshot
{
  std::vector<RealSignedLogAmplitude> cross;
  std::vector<RealSignedLogAmplitude> target_square;
  std::vector<RealSignedLogAmplitude> student_square;
  std::vector<RealSignedLogAmplitude> target_absolute;
  std::vector<RealSignedLogAmplitude> student_absolute;
  std::size_t samples = 0;
};

namespace overlap_detail
{

/** Recognize the canonical zero log-magnitude under finite-math-only builds. */
inline bool isNegativeInfinity(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == UINT64_C(0xfff0000000000000);
}

/** Classify a binary64 value even in QMCPACK's finite-math-only release builds. */
inline bool isFinite(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/** Accumulate signed logarithmic terms with a dynamic exponent shift. */
class SignedLogSum
{
public:
  /// Restore the empty sum without changing any surrounding storage.
  void clear() noexcept
  {
    shift_      = -std::numeric_limits<double>::infinity();
    sum_        = 0.0L;
    correction_ = 0.0L;
  }

  /// Add one real signed-log term.
  void add(double phase, double log_abs)
  {
    if (phase == 0.0)
      return;
    if ((phase != -1.0 && phase != 1.0) || !isFinite(log_abs))
      throw std::invalid_argument("PsiFormer overlap received an invalid signed-log term");
    if (!isFinite(shift_))
    {
      shift_ = log_abs;
      addScaled(static_cast<long double>(phase));
      return;
    }
    if (log_abs > shift_)
    {
      const long double scale = std::exp(static_cast<long double>(shift_ - log_abs));
      sum_ *= scale;
      correction_ *= scale;
      shift_ = log_abs;
    }
    addScaled(static_cast<long double>(phase) *
              std::exp(static_cast<long double>(log_abs - shift_)));
  }

  /// Merge another independently accumulated stream, enabling MPI-style reduction.
  void merge(const SignedLogSum& other)
  {
    const RealSignedLogAmplitude value = other.value();
    add(value.phase, value.log_abs);
  }

  /// Return the accumulated sum without materializing its magnitude.
  RealSignedLogAmplitude value() const
  {
    const long double mantissa = sum_ + correction_;
    if (mantissa == 0.0L)
      return {};
    const double log_abs = shift_ + static_cast<double>(std::log(std::abs(mantissa)));
    if (!isFinite(log_abs))
      throw std::overflow_error("PsiFormer overlap moment became non-finite");
    return {mantissa < 0.0L ? -1.0 : 1.0, log_abs};
  }

private:
  /// Neumaier-add one scaled term while retaining its first-order residual.
  void addScaled(long double value) noexcept
  {
    const long double updated = sum_ + value;
    correction_ += std::abs(sum_) >= std::abs(value) ? (sum_ - updated) + value
                                                     : (value - updated) + sum_;
    sum_ = updated;
  }

  double shift_          = -std::numeric_limits<double>::infinity();
  long double sum_       = 0.0L;
  long double correction_ = 0.0L;
};

/** Validate a canonical real signed-log amplitude. */
inline void validateAmplitude(RealSignedLogAmplitude amplitude, const char* description)
{
  if (amplitude.phase == 0.0)
  {
    if (!isNegativeInfinity(amplitude.log_abs))
      throw std::invalid_argument(std::string("PsiFormer overlap ") + description +
                                  " has a noncanonical zero");
    return;
  }
  if ((amplitude.phase != -1.0 && amplitude.phase != 1.0) || !isFinite(amplitude.log_abs))
    throw std::invalid_argument(std::string("PsiFormer overlap ") + description +
                                " is not a finite real signed-log amplitude");
}

} // namespace overlap_detail

/** Accumulate and differentiate a real determinant-channel normalized-overlap loss. */
class DeterminantOverlapAccumulator
{
public:
  /** Allocate fixed O(M) moment and diagnostic storage.
   *
   * Objective weights need not sum to one, but must be finite and nonnegative.
   */
  explicit DeterminantOverlapAccumulator(std::vector<double> objective_weights,
                                         DeterminantOverlapPolicy policy = {})
      : weights_(std::move(objective_weights)),
        policy_(policy),
        cross_(weights_.size()),
        target_square_(weights_.size()),
        student_square_(weights_.size()),
        target_absolute_(weights_.size()),
        student_absolute_(weights_.size()),
        diagnostics_(weights_.size())
  {
    if (weights_.empty())
      throw std::invalid_argument("PsiFormer overlap requires determinant channels");
    for (double weight : weights_)
      if (!overlap_detail::isFinite(weight) || weight < 0.0)
        throw std::invalid_argument("PsiFormer overlap objective weight is invalid");
    if (!overlap_detail::isFinite(policy_.minimum_fidelity) ||
        policy_.minimum_fidelity < 0.0 || policy_.minimum_fidelity > 1.0 ||
        !overlap_detail::isFinite(policy_.minimum_effective_samples) ||
        policy_.minimum_effective_samples < 0.0)
      throw std::invalid_argument("PsiFormer overlap validity policy is invalid");
  }

  /// Reset all streaming moments while retaining every allocation.
  void reset() noexcept
  {
    for (std::size_t channel = 0; channel < weights_.size(); ++channel)
    {
      cross_[channel].clear();
      target_square_[channel].clear();
      student_square_[channel].clear();
      target_absolute_[channel].clear();
      student_absolute_[channel].clear();
      diagnostics_[channel] = {};
    }
    samples_   = 0;
    loss_      = 0.0;
    finalized_ = false;
  }

  /** Add one sample using full teacher, target-channel, and student-channel amplitudes. */
  void addSample(RealSignedLogAmplitude teacher,
                 const RealSignedLogAmplitude* targets,
                 const RealSignedLogAmplitude* students,
                 std::size_t channel_count)
  {
    if (finalized_)
      throw std::logic_error("PsiFormer overlap cannot add samples after finalization");
    if (channel_count != weights_.size() || !targets || !students)
      throw std::invalid_argument("PsiFormer overlap sample has the wrong channel count");
    overlap_detail::validateAmplitude(teacher, "teacher");
    if (teacher.isZero())
      throw std::domain_error("PsiFormer overlap sampling teacher is at an exact node");

    // Validate the complete sample before mutating any moment.
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      overlap_detail::validateAmplitude(targets[channel], "target channel");
      overlap_detail::validateAmplitude(students[channel], "student channel");
    }
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      const auto target  = targets[channel];
      const auto student = students[channel];
      if (!target.isZero())
      {
        const double log_a = target.log_abs - teacher.log_abs;
        target_absolute_[channel].add(1.0, log_a);
        target_square_[channel].add(1.0, 2.0 * log_a);
        if (!student.isZero())
        {
          const double log_b = student.log_abs - teacher.log_abs;
          cross_[channel].add(target.phase * student.phase, log_a + log_b);
        }
      }
      if (!student.isZero())
      {
        const double log_b = student.log_abs - teacher.log_abs;
        student_absolute_[channel].add(1.0, log_b);
        student_square_[channel].add(1.0, 2.0 * log_b);
      }
    }
    ++samples_;
  }

  /** Merge a compatible partial stream without retaining its samples. */
  void merge(const DeterminantOverlapAccumulator& other)
  {
    if (finalized_ || other.finalized_)
      throw std::logic_error("PsiFormer overlap can merge only collecting accumulators");
    if (weights_ != other.weights_ || policy_.minimum_fidelity != other.policy_.minimum_fidelity ||
        policy_.minimum_effective_samples != other.policy_.minimum_effective_samples)
      throw std::invalid_argument("PsiFormer overlap accumulator conventions do not match");
    for (std::size_t channel = 0; channel < weights_.size(); ++channel)
    {
      cross_[channel].merge(other.cross_[channel]);
      target_square_[channel].merge(other.target_square_[channel]);
      student_square_[channel].merge(other.student_square_[channel]);
      target_absolute_[channel].merge(other.target_absolute_[channel]);
      student_absolute_[channel].merge(other.student_absolute_[channel]);
    }
    samples_ += other.samples_;
  }

  /** Copy the O(M) collecting state for a stable external distributed reduction. */
  DeterminantOverlapMomentSnapshot snapshotMoments() const
  {
    if (finalized_)
      throw std::logic_error("PsiFormer overlap cannot snapshot finalized moments");
    DeterminantOverlapMomentSnapshot result;
    result.cross.resize(weights_.size());
    result.target_square.resize(weights_.size());
    result.student_square.resize(weights_.size());
    result.target_absolute.resize(weights_.size());
    result.student_absolute.resize(weights_.size());
    result.samples = samples_;
    for (std::size_t channel = 0; channel < weights_.size(); ++channel)
    {
      result.cross[channel]            = cross_[channel].value();
      result.target_square[channel]    = target_square_[channel].value();
      result.student_square[channel]   = student_square_[channel].value();
      result.target_absolute[channel]  = target_absolute_[channel].value();
      result.student_absolute[channel] = student_absolute_[channel].value();
    }
    return result;
  }

  /** Replace collecting state with a compatible globally reduced O(M) snapshot. */
  void replaceMoments(const DeterminantOverlapMomentSnapshot& snapshot)
  {
    if (finalized_)
      throw std::logic_error("PsiFormer overlap cannot replace finalized moments");
    const auto require_extent = [this](const auto& values) {
      return values.size() == weights_.size();
    };
    if (!require_extent(snapshot.cross) || !require_extent(snapshot.target_square) ||
        !require_extent(snapshot.student_square) ||
        !require_extent(snapshot.target_absolute) ||
        !require_extent(snapshot.student_absolute))
      throw std::invalid_argument("PsiFormer overlap moment snapshot has the wrong channel count");

    reset();
    for (std::size_t channel = 0; channel < weights_.size(); ++channel)
    {
      overlap_detail::validateAmplitude(snapshot.cross[channel], "cross moment");
      overlap_detail::validateAmplitude(snapshot.target_square[channel], "target-square moment");
      overlap_detail::validateAmplitude(snapshot.student_square[channel], "student-square moment");
      overlap_detail::validateAmplitude(snapshot.target_absolute[channel], "target-absolute moment");
      overlap_detail::validateAmplitude(snapshot.student_absolute[channel], "student-absolute moment");
      cross_[channel].add(snapshot.cross[channel].phase, snapshot.cross[channel].log_abs);
      target_square_[channel].add(snapshot.target_square[channel].phase,
                                  snapshot.target_square[channel].log_abs);
      student_square_[channel].add(snapshot.student_square[channel].phase,
                                   snapshot.student_square[channel].log_abs);
      target_absolute_[channel].add(snapshot.target_absolute[channel].phase,
                                    snapshot.target_absolute[channel].log_abs);
      student_absolute_[channel].add(snapshot.student_absolute[channel].phase,
                                     snapshot.student_absolute[channel].log_abs);
    }
    samples_ = snapshot.samples;
  }

  /** Finalize first-pass statistics and return the weighted negative log fidelity. */
  double finalize()
  {
    if (finalized_)
      return loss_;
    if (samples_ == 0)
      throw std::domain_error("PsiFormer overlap cannot finalize an empty sample stream");
    double candidate_loss = 0.0;
    for (std::size_t channel = 0; channel < weights_.size(); ++channel)
    {
      const auto cross = cross_[channel].value();
      const auto a2    = target_square_[channel].value();
      const auto b2    = student_square_[channel].value();
      const auto a1    = target_absolute_[channel].value();
      const auto b1    = student_absolute_[channel].value();
      if (cross.isZero() || a2.isZero() || b2.isZero() || a1.isZero() || b1.isZero())
        throw std::domain_error("PsiFormer overlap channel has a zero normalization or overlap");

      double log_fidelity = 2.0 * cross.log_abs - a2.log_abs - b2.log_abs;
      // Cauchy-Schwarz permits zero; tolerate a few ulps of positive roundoff.
      if (log_fidelity > 1.0e-12)
        throw std::domain_error("PsiFormer overlap fidelity exceeds one");
      log_fidelity = std::min(0.0, log_fidelity);
      const double fidelity = std::exp(log_fidelity);
      const double target_ess = std::exp(2.0 * a1.log_abs - a2.log_abs);
      const double student_ess = std::exp(2.0 * b1.log_abs - b2.log_abs);
      if (!overlap_detail::isFinite(fidelity) || !overlap_detail::isFinite(target_ess) ||
          !overlap_detail::isFinite(student_ess) || fidelity < policy_.minimum_fidelity ||
          target_ess < policy_.minimum_effective_samples ||
          student_ess < policy_.minimum_effective_samples)
        throw std::domain_error("PsiFormer overlap channel failed fidelity or effective-sample policy");
      diagnostics_[channel] = {log_fidelity, fidelity, target_ess, student_ess};
      candidate_loss -= weights_[channel] * log_fidelity;
    }
    if (!overlap_detail::isFinite(candidate_loss))
      throw std::overflow_error("PsiFormer overlap loss is non-finite");
    loss_      = candidate_loss;
    finalized_ = true;
    return loss_;
  }

  /** Produce one sample's M seeds for dL/dlog|Psi_k| during the second pass. */
  void channelScoreWeights(RealSignedLogAmplitude teacher,
                           const RealSignedLogAmplitude* targets,
                           const RealSignedLogAmplitude* students,
                           std::size_t channel_count,
                           double* output) const
  {
    if (!finalized_)
      throw std::logic_error("PsiFormer overlap score weights require finalized moments");
    if (channel_count != weights_.size() || !targets || !students || !output)
      throw std::invalid_argument("PsiFormer overlap score sample has the wrong channel count");
    overlap_detail::validateAmplitude(teacher, "teacher");
    if (teacher.isZero())
      throw std::domain_error("PsiFormer overlap sampling teacher is at an exact node");
    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      overlap_detail::validateAmplitude(targets[channel], "target channel");
      overlap_detail::validateAmplitude(students[channel], "student channel");
    }

    for (std::size_t channel = 0; channel < channel_count; ++channel)
    {
      const auto target  = targets[channel];
      const auto student = students[channel];
      if (student.isZero())
        throw std::domain_error("PsiFormer overlap score is undefined at a student channel node");
      const double log_b = student.log_abs - teacher.log_abs;
      const auto cross   = cross_[channel].value();
      const auto b2      = student_square_[channel].value();
      double normalized_cross = 0.0;
      if (!target.isZero())
      {
        const double log_a = target.log_abs - teacher.log_abs;
        normalized_cross = target.phase * student.phase * cross.phase *
            std::exp(log_a + log_b - cross.log_abs);
      }
      const double normalized_square = std::exp(2.0 * log_b - b2.log_abs);
      output[channel] = -2.0 * weights_[channel] *
          (normalized_cross - normalized_square);
      if (!overlap_detail::isFinite(output[channel]))
        throw std::overflow_error("PsiFormer overlap channel-score weight is non-finite");
    }
  }

  /// Return the number of determinant channels.
  std::size_t channelCount() const noexcept { return weights_.size(); }

  /// Return the number of first-pass samples represented by the moments.
  std::size_t sampleCount() const noexcept { return samples_; }

  /// Return whether first-pass moments have been finalized successfully.
  bool finalized() const noexcept { return finalized_; }

  /// Return fixed channel diagnostics after finalization.
  const std::vector<DeterminantOverlapChannelDiagnostics>& diagnostics() const noexcept
  {
    return diagnostics_;
  }

  /// Count explicit retained scalar storage, independent of sample count and P.
  std::size_t retainedScalarBytes() const noexcept
  {
    return weights_.capacity() * sizeof(double) +
        diagnostics_.capacity() * sizeof(DeterminantOverlapChannelDiagnostics) +
        (cross_.capacity() + target_square_.capacity() + student_square_.capacity() +
         target_absolute_.capacity() + student_absolute_.capacity()) *
            sizeof(overlap_detail::SignedLogSum);
  }

  /// Fingerprint retained buffer locations/capacities for warmed-call allocation tests.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    const auto fold = [&hash](const auto& storage) {
      hash ^= reinterpret_cast<std::uintptr_t>(storage.data());
      hash *= 1099511628211ULL;
      hash ^= storage.capacity();
      hash *= 1099511628211ULL;
    };
    fold(weights_);
    fold(cross_);
    fold(target_square_);
    fold(student_square_);
    fold(target_absolute_);
    fold(student_absolute_);
    fold(diagnostics_);
    return hash;
  }

private:
  std::vector<double> weights_;
  DeterminantOverlapPolicy policy_;
  std::vector<overlap_detail::SignedLogSum> cross_;
  std::vector<overlap_detail::SignedLogSum> target_square_;
  std::vector<overlap_detail::SignedLogSum> student_square_;
  std::vector<overlap_detail::SignedLogSum> target_absolute_;
  std::vector<overlap_detail::SignedLogSum> student_absolute_;
  std::vector<DeterminantOverlapChannelDiagnostics> diagnostics_;
  std::size_t samples_ = 0;
  double loss_         = 0.0;
  bool finalized_      = false;
};

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_DETERMINANT_OVERLAP_H
