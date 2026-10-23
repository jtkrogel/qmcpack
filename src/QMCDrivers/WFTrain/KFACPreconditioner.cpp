//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KFACPreconditioner.cpp
 * @brief Validation, accumulation, reduction, and split-damped KFAC solves.
 */

#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"

#include "Message/CommOperators.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Multiply dimensions with overflow checking before storage allocation.
std::size_t checkedProduct(std::size_t left, std::size_t right, const char* what)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(what) + " size overflow");
  return left * right;
}

/// Extend a compact FNV-1a registry identity with one byte interval.
void extendHash(std::uint64_t& hash, const void* bytes, std::size_t bytes_count) noexcept
{
  const auto* data = static_cast<const unsigned char*>(bytes);
  for (std::size_t index = 0; index < bytes_count; ++index)
  {
    hash ^= data[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend a compact identity with a string and an unambiguous terminator.
void extendHash(std::uint64_t& hash, std::string_view text) noexcept
{
  extendHash(hash, text.data(), text.size());
  const unsigned char terminator = 0;
  extendHash(hash, &terminator, 1);
}

/// Extend a compact identity with one platform-independent fixed-width integer.
void extendHash(std::uint64_t& hash, std::uint64_t value) noexcept
{
  std::array<unsigned char, 8> bytes{};
  for (std::size_t byte = 0; byte < bytes.size(); ++byte)
    bytes[byte] = static_cast<unsigned char>(value >> (8 * byte));
  extendHash(hash, bytes.data(), bytes.size());
}

/// Format a fixed-width registry identity without platform-dependent std::hash.
std::string formatFingerprint(std::uint64_t hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result(16, '0');
  for (std::size_t index = 0; index < result.size(); ++index)
  {
    result[result.size() - 1 - index] = digits[hash & 0xfU];
    hash >>= 4;
  }
  return result;
}

/// Return whether one real scalar is finite.
bool finite(DerivativeReal value) noexcept
{
  return isFiniteTrainingReal(value);
}

/// Validate factor dimensions and optionally allocate zeroed numeric storage.
KFACFactorStatistics makeStatistics(const KFACAffineBlockDescriptor& block, bool allocate = true)
{
  const std::size_t activation_dimension =
      block.input_width + (block.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1);
  const std::size_t activation_size =
      checkedProduct(activation_dimension, activation_dimension, "KFAC activation factor");
  const std::size_t sensitivity_size =
      checkedProduct(block.output_width, block.output_width, "KFAC sensitivity factor");
  return {activation_dimension, block.output_width, 0.0, 0.0, 0,
          std::vector<DerivativeReal>(allocate ? activation_size : 0),
          std::vector<DerivativeReal>(allocate ? sensitivity_size : 0)};
}

/// Compute an in-place lower Cholesky factor, rejecting a non-SPD matrix.
void cholesky(std::vector<DerivativeReal>& matrix, std::size_t dimension)
{
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column <= row; ++column)
    {
      DerivativeReal value = matrix[row * dimension + column];
      for (std::size_t inner = 0; inner < column; ++inner)
        value -= matrix[row * dimension + inner] * matrix[column * dimension + inner];
      if (row == column)
      {
        if (!finite(value) || value <= 0.0)
          throw MatrixFreeNumericalError("KFAC damped factor is not positive definite");
        matrix[row * dimension + column] = std::sqrt(value);
      }
      else
        matrix[row * dimension + column] = value / matrix[column * dimension + column];
    }
  for (std::size_t row = 0; row < dimension; ++row)
    std::fill(matrix.begin() + row * dimension + row + 1,
              matrix.begin() + (row + 1) * dimension, 0.0);
}

/// Solve L L^T x=b for one vector using caller-owned scratch.
void solveCholesky(const std::vector<DerivativeReal>& lower,
                   std::size_t dimension,
                   const DerivativeReal* rhs,
                   DerivativeReal* result,
                   DerivativeReal* scratch)
{
  for (std::size_t row = 0; row < dimension; ++row)
  {
    DerivativeReal value = rhs[row];
    for (std::size_t column = 0; column < row; ++column)
      value -= lower[row * dimension + column] * scratch[column];
    scratch[row] = value / lower[row * dimension + row];
  }
  for (std::size_t reverse = dimension; reverse > 0; --reverse)
  {
    const std::size_t row = reverse - 1;
    DerivativeReal value = scratch[row];
    for (std::size_t column = row + 1; column < dimension; ++column)
      value -= lower[column * dimension + row] * result[column];
    result[row] = value / lower[row * dimension + row];
  }
}

} // namespace

KFACBlockRegistry::KFACBlockRegistry(
    const StructuredParameterSchema& schema,
    std::vector<KFACAffineBlockDescriptor> affine_blocks,
    std::vector<std::size_t> fallback_blocks,
    std::vector<std::size_t> generic_blocks)
    : schema_(schema),
      affine_blocks_(std::move(affine_blocks)),
      fallback_blocks_(std::move(fallback_blocks)),
      generic_blocks_(std::move(generic_blocks))
{
  if (schema_.blocks().empty())
    throw std::invalid_argument("KFAC registry requires a nonempty parameter schema");
  std::vector<bool> covered(schema_.blocks().size(), false);
  for (std::size_t block = 0; block < schema_.blocks().size(); ++block)
  {
    if (schema_.blocks()[block].scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument("KFAC currently supports only real parameter schemas");
    // Frozen blocks are deliberately outside the curvature registry and remain zero.
    covered[block] = !schema_.blocks()[block].trainable;
  }
  std::vector<std::string> ids;
  ids.reserve(affine_blocks_.size());
  for (const KFACAffineBlockDescriptor& affine : affine_blocks_)
  {
    if (affine.id.empty() || affine.input_width == 0 || affine.output_width == 0 ||
        affine.weight_block >= schema_.blocks().size())
      throw std::invalid_argument("KFAC affine block has invalid identity or dimensions");
    if (std::find(ids.begin(), ids.end(), affine.id) != ids.end())
      throw std::invalid_argument("KFAC affine block identifiers must be unique");
    ids.push_back(affine.id);

    const ParameterBlockDescriptor& weight = schema_.blocks()[affine.weight_block];
    if (weight.scalar_domain != ParameterScalarDomain::REAL64 || !weight.trainable ||
        weight.count != checkedProduct(affine.input_width, affine.output_width,
                                       "KFAC weight tensor"))
      throw std::invalid_argument("KFAC affine weight block has an unsupported domain, state, or shape");
    if (covered[affine.weight_block])
      throw std::invalid_argument("KFAC schema block is covered more than once");
    covered[affine.weight_block] = true;

    if (affine.bias_block != NO_KFAC_BIAS_BLOCK)
    {
      if (affine.bias_block >= schema_.blocks().size())
        throw std::invalid_argument("KFAC affine bias block is out of range");
      const ParameterBlockDescriptor& bias = schema_.blocks()[affine.bias_block];
      if (bias.scalar_domain != ParameterScalarDomain::REAL64 || !bias.trainable ||
          bias.count != affine.output_width || covered[affine.bias_block])
        throw std::invalid_argument("KFAC affine bias block has an unsupported domain, state, or shape");
      covered[affine.bias_block] = true;
    }
  }

  for (std::size_t block : fallback_blocks_)
  {
    if (block >= schema_.blocks().size() || covered[block])
      throw std::invalid_argument("KFAC fallback block is out of range or covered more than once");
    const ParameterBlockDescriptor& descriptor = schema_.blocks()[block];
    if (descriptor.scalar_domain != ParameterScalarDomain::REAL64 || !descriptor.trainable)
      throw std::invalid_argument("KFAC fallback supports only trainable real blocks");
    covered[block] = true;
  }
  if (!std::all_of(covered.begin(), covered.end(), [](bool value) { return value; }))
    throw std::invalid_argument("KFAC registry leaves schema blocks unclassified");
  std::vector<bool> generic_seen(schema_.blocks().size(), false);
  for (std::size_t block : generic_blocks_)
  {
    if (std::find(fallback_blocks_.begin(), fallback_blocks_.end(), block) == fallback_blocks_.end() ||
        generic_seen[block])
      throw std::invalid_argument("KFAC generic blocks must be a unique subset of fallback blocks");
    generic_seen[block] = true;
  }

  std::uint64_t hash = UINT64_C(14695981039346656037);
  extendHash(hash, schema_.fingerprint());
  for (const KFACAffineBlockDescriptor& affine : affine_blocks_)
  {
    extendHash(hash, affine.id);
    extendHash(hash, static_cast<std::uint64_t>(affine.weight_block));
    extendHash(hash, affine.bias_block == NO_KFAC_BIAS_BLOCK
                         ? std::numeric_limits<std::uint64_t>::max()
                         : static_cast<std::uint64_t>(affine.bias_block));
    extendHash(hash, static_cast<std::uint64_t>(affine.input_width));
    extendHash(hash, static_cast<std::uint64_t>(affine.output_width));
  }
  for (std::size_t block : fallback_blocks_)
    extendHash(hash, static_cast<std::uint64_t>(block));
  if (!generic_blocks_.empty())
  {
    extendHash(hash, "registered_generic_blocks");
    for (std::size_t block : generic_blocks_)
      extendHash(hash, static_cast<std::uint64_t>(block));
  }
  fingerprint_ = formatFingerprint(hash);
}

KFACFactorAccumulator::KFACFactorAccumulator(KFACBlockRegistry registry,
                                             std::size_t parameter_version,
                                             KFACCurvatureConvention convention,
                                             KFACAffineStatisticsSource source)
    : registry_(std::move(registry)), convention_(convention), affine_source_(source),
      parameter_version_(parameter_version)
{
  if (source != KFACAffineStatisticsSource::OBSERVATION_ROWS &&
      source != KFACAffineStatisticsSource::PRECOMPUTED_SUMS)
    throw std::invalid_argument("KFAC affine statistics source is unsupported");
  if (convention_ != KFACCurvatureConvention::NATIVE_SCORE &&
      convention_ != KFACCurvatureConvention::REGISTERED_GAUSSIAN)
    throw std::invalid_argument("KFAC curvature convention is unsupported");
  factors_.reserve(registry_.affineBlocks().size());
  pending_.reserve(registry_.affineBlocks().size());
  for (const KFACAffineBlockDescriptor& block : registry_.affineBlocks())
  {
    // Imported factors replace this descriptor; do not allocate an unused bank.
    factors_.push_back(makeStatistics(block, affine_source_ == KFACAffineStatisticsSource::OBSERVATION_ROWS));
    if (affine_source_ == KFACAffineStatisticsSource::OBSERVATION_ROWS)
      pending_.push_back(makeStatistics(block));
  }
  observed_.assign(factors_.size(), false);
  std::size_t fallback_parameter_count = 0;
  for (std::size_t block : registry_.fallbackBlocks())
  {
    const std::size_t count = registry_.parameterSchema().blocks()[block].count;
    if (fallback_parameter_count > std::numeric_limits<std::size_t>::max() - count)
      throw std::overflow_error("KFAC fallback parameter extent overflow");
    fallback_parameter_count += count;
  }
  fallback_diagonal_sum_.assign(fallback_parameter_count, 0.0);
  pending_fallback_diagonal_.assign(fallback_parameter_count, 0.0);
  if (convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN && !registry_.genericBlocks().empty())
  {
    fallback_score_sum_.assign(fallback_parameter_count, 0.0);
    pending_fallback_score_.assign(fallback_parameter_count, 0.0);
    fallback_imaginary_score_sum_.assign(fallback_parameter_count, 0.0);
    pending_fallback_imaginary_score_.assign(fallback_parameter_count, 0.0);
    for (std::size_t block : registry_.fallbackBlocks())
      generic_entries_.insert(generic_entries_.end(), registry_.parameterSchema().blocks()[block].count,
          std::find(registry_.genericBlocks().begin(), registry_.genericBlocks().end(), block) !=
              registry_.genericBlocks().end());
  }
}

void KFACFactorAccumulator::beginSample(DerivativeReal sample_weight)
{
  if (sample_active_)
    throw std::logic_error("KFAC sample transaction is already active");
  if (globally_reduced_)
    throw std::logic_error("KFAC globally reduced statistics cannot accept new samples");
  if (imported_affine_)
    throw std::logic_error("KFAC imported affine statistics cannot accept new samples");
  if (affine_source_ == KFACAffineStatisticsSource::PRECOMPUTED_SUMS && sample_weight != 1.0)
    throw std::invalid_argument("KFAC precomputed affine statistics require unit sample weights");
  if (!finite(sample_weight) || sample_weight < 0.0)
    throw std::invalid_argument("KFAC sample weight must be finite and nonnegative");
  sample_weight_ = convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 1.0 : sample_weight;
  std::fill(observed_.begin(), observed_.end(), affine_source_ == KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
  for (KFACFactorStatistics& factor : pending_)
  {
    factor.sample_weight_sum = 0.0;
    factor.row_weight_sum = 0.0;
    factor.row_count = 0;
    std::fill(factor.activation_outer_sum.begin(), factor.activation_outer_sum.end(), 0.0);
    std::fill(factor.sensitivity_outer_sum.begin(), factor.sensitivity_outer_sum.end(), 0.0);
  }
  std::fill(pending_fallback_diagonal_.begin(), pending_fallback_diagonal_.end(), 0.0);
  std::fill(pending_fallback_score_.begin(), pending_fallback_score_.end(), 0.0);
  std::fill(pending_fallback_imaginary_score_.begin(), pending_fallback_imaginary_score_.end(), 0.0);
  fallback_observed_ = registry_.fallbackBlocks().empty();
  sample_active_ = true;
}

void KFACFactorAccumulator::addObservation(const KFACLayerObservation& observation)
{
  if (affine_source_ != KFACAffineStatisticsSource::OBSERVATION_ROWS)
    throw std::logic_error("KFAC precomputed factors cannot mix with observation rows");
  if (!sample_active_)
    throw std::logic_error("KFAC observation requires an active sample transaction");
  if (observation.affine_block >= pending_.size() || observed_[observation.affine_block])
    throw std::invalid_argument("KFAC observation has an invalid or duplicate block ordinal");
  const KFACAffineBlockDescriptor& block =
      registry_.affineBlocks()[observation.affine_block];
  if (observation.row_count == 0 ||
      observation.activations.size() !=
          checkedProduct(observation.row_count, block.input_width, "KFAC activation rows") ||
      observation.sensitivities.size() !=
          checkedProduct(observation.row_count, block.output_width, "KFAC sensitivity rows") ||
      !observation.activations.data() || !observation.sensitivities.data())
    throw std::invalid_argument("KFAC observation row extents do not match the registry");
  if (!observation.imaginary_sensitivities.empty() &&
      (observation.imaginary_sensitivities.size() != observation.sensitivities.size() ||
       !observation.imaginary_sensitivities.data()))
    throw std::invalid_argument("KFAC imaginary sensitivity extent does not match the real channel");
  for (DerivativeReal value : observation.activations)
    if (!finite(value))
      throw std::invalid_argument("KFAC activation observation contains a non-finite value");
  for (DerivativeReal value : observation.sensitivities)
    if (!finite(value))
      throw std::invalid_argument("KFAC sensitivity observation contains a non-finite value");
  for (DerivativeReal value : observation.imaginary_sensitivities)
    if (!finite(value))
      throw std::invalid_argument("KFAC imaginary sensitivity contains a non-finite value");

  KFACFactorStatistics& factor = pending_[observation.affine_block];
  for (std::size_t row = 0; row < observation.row_count; ++row)
  {
    for (std::size_t left = 0; left < factor.activation_dimension; ++left)
    {
      const DerivativeReal left_value = left == block.input_width
          ? 1.0
          : observation.activations[row * block.input_width + left];
      for (std::size_t right = 0; right < factor.activation_dimension; ++right)
      {
        const DerivativeReal right_value = right == block.input_width
            ? 1.0
            : observation.activations[row * block.input_width + right];
        factor.activation_outer_sum[left * factor.activation_dimension + right] +=
            sample_weight_ * left_value * right_value;
      }
    }
    for (std::size_t left = 0; left < block.output_width; ++left)
      for (std::size_t right = 0; right < block.output_width; ++right)
        factor.sensitivity_outer_sum[left * block.output_width + right] +=
            (convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 2.0 : 1.0) *
            sample_weight_ * observation.sensitivities[row * block.output_width + left] *
            observation.sensitivities[row * block.output_width + right];
    if (!observation.imaginary_sensitivities.empty())
      for (std::size_t left = 0; left < block.output_width; ++left)
        for (std::size_t right = 0; right < block.output_width; ++right)
          factor.sensitivity_outer_sum[left * block.output_width + right] +=
              (convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 2.0 : 1.0) *
              sample_weight_ * observation.imaginary_sensitivities[row * block.output_width + left] *
              observation.imaginary_sensitivities[row * block.output_width + right];
  }
  factor.sample_weight_sum = sample_weight_;
  factor.row_weight_sum = sample_weight_ * static_cast<DerivativeReal>(observation.row_count);
  factor.row_count = observation.row_count;
  observed_[observation.affine_block] = true;
}

void KFACFactorAccumulator::addFallbackScores(
    DerivativeArrayView<const DerivativeReal> score,
    DerivativeArrayView<const DerivativeReal> imaginary_score)
{
  if (!sample_active_)
    throw std::logic_error("KFAC fallback scores require an active sample transaction");
  if (fallback_observed_)
    throw std::invalid_argument("KFAC fallback scores were already supplied for this sample");
  if (score.size() != registry_.parameterSchema().parameterCount() ||
      (!score.empty() && !score.data()))
    throw std::invalid_argument("KFAC fallback score extent does not match the registry schema");
  if (!imaginary_score.empty() &&
      (imaginary_score.size() != score.size() || !imaginary_score.data()))
    throw std::invalid_argument("KFAC imaginary fallback score extent does not match the real channel");
  // Keep the load address valid even when an optimized empty-channel select is evaluated eagerly.
  const DerivativeReal zero_imaginary = 0.0;
  const DerivativeReal* imaginary_values = imaginary_score.empty() ? &zero_imaginary : imaginary_score.data();
  std::size_t packed = 0;
  for (std::size_t block_index : registry_.fallbackBlocks())
  {
    const ParameterBlockDescriptor& block =
        registry_.parameterSchema().blocks()[block_index];
    for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
         ++parameter)
    {
      if (!finite(score[parameter]))
        throw std::invalid_argument("KFAC fallback score contains a non-finite value");
      const std::size_t imaginary_index = imaginary_score.empty() ? 0 : parameter;
      const DerivativeReal imaginary = imaginary_values[imaginary_index];
      if (!finite(imaginary))
        throw std::invalid_argument("KFAC imaginary fallback score contains a non-finite value");
      if (!generic_entries_.empty() && generic_entries_[packed])
      {
        pending_fallback_score_[packed] = score[parameter];
        pending_fallback_imaginary_score_[packed] = imaginary;
      }
      else
      {
        pending_fallback_diagonal_[packed] =
            (convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 2.0 : 1.0) *
            sample_weight_ * score[parameter] * score[parameter];
        if (!imaginary_score.empty())
          pending_fallback_diagonal_[packed] +=
              (convention_ == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 2.0 : 1.0) *
              sample_weight_ * imaginary * imaginary;
      }
      ++packed;
    }
  }
  fallback_observed_ = true;
}

void KFACFactorAccumulator::endSample()
{
  if (!sample_active_)
    throw std::logic_error("KFAC sample completion requires an active transaction");
  if (!std::all_of(observed_.begin(), observed_.end(), [](bool value) { return value; }) ||
      !fallback_observed_)
  {
    abortSample();
    throw std::logic_error("KFAC sample ended before every affine block was observed");
  }

  // Validate the complete commit before touching published statistics.  This also
  // catches finite inputs whose products or accumulated sums overflowed.
  bool valid = sample_count_ != std::numeric_limits<std::size_t>::max() &&
      finite(fallback_weight_sum_ + sample_weight_);
  for (std::size_t block = 0; valid && block < pending_.size(); ++block)
  {
    const KFACFactorStatistics& destination = factors_[block];
    const KFACFactorStatistics& source = pending_[block];
    valid = source.row_count <= std::numeric_limits<std::size_t>::max() - destination.row_count &&
        finite(destination.sample_weight_sum + source.sample_weight_sum) &&
        finite(destination.row_weight_sum + source.row_weight_sum);
    for (std::size_t index = 0; valid && index < destination.activation_outer_sum.size(); ++index)
      valid = finite(destination.activation_outer_sum[index] + source.activation_outer_sum[index]);
    for (std::size_t index = 0; valid && index < destination.sensitivity_outer_sum.size(); ++index)
      valid = finite(destination.sensitivity_outer_sum[index] + source.sensitivity_outer_sum[index]);
  }
  for (std::size_t parameter = 0; valid && parameter < fallback_diagonal_sum_.size(); ++parameter)
    if (!generic_entries_.empty() && generic_entries_[parameter])
    {
      const DerivativeReal sum = fallback_score_sum_[parameter] + pending_fallback_score_[parameter];
      const DerivativeReal imaginary = fallback_imaginary_score_sum_[parameter] +
          pending_fallback_imaginary_score_[parameter];
      valid = finite(sum) && finite(imaginary) && finite(2.0 * sum * sum + 2.0 * imaginary * imaginary);
    }
    else
      valid = finite(fallback_diagonal_sum_[parameter] + pending_fallback_diagonal_[parameter]);
  if (!valid)
  {
    abortSample();
    throw std::overflow_error("KFAC sample statistics overflowed during accumulation");
  }

  for (std::size_t block = 0; block < pending_.size(); ++block)
  {
    KFACFactorStatistics& destination = factors_[block];
    const KFACFactorStatistics& source = pending_[block];
    destination.sample_weight_sum += source.sample_weight_sum;
    destination.row_weight_sum += source.row_weight_sum;
    destination.row_count += source.row_count;
    for (std::size_t index = 0; index < destination.activation_outer_sum.size(); ++index)
      destination.activation_outer_sum[index] += source.activation_outer_sum[index];
    for (std::size_t index = 0; index < destination.sensitivity_outer_sum.size(); ++index)
      destination.sensitivity_outer_sum[index] += source.sensitivity_outer_sum[index];
  }
  for (std::size_t parameter = 0; parameter < fallback_diagonal_sum_.size(); ++parameter)
    if (!generic_entries_.empty() && generic_entries_[parameter])
    {
      fallback_score_sum_[parameter] += pending_fallback_score_[parameter];
      fallback_diagonal_sum_[parameter] =
          2.0 * fallback_score_sum_[parameter] * fallback_score_sum_[parameter];
      fallback_imaginary_score_sum_[parameter] += pending_fallback_imaginary_score_[parameter];
      const DerivativeReal imaginary = fallback_imaginary_score_sum_[parameter];
      fallback_diagonal_sum_[parameter] += 2.0 * imaginary * imaginary;
    }
    else
      fallback_diagonal_sum_[parameter] += pending_fallback_diagonal_[parameter];
  fallback_weight_sum_ += sample_weight_;
  ++sample_count_;
  sample_active_ = false;
}

void KFACFactorAccumulator::abortSample() noexcept
{
  sample_active_ = false;
  std::fill(observed_.begin(), observed_.end(), false);
  fallback_observed_ = false;
}

void KFACFactorAccumulator::importAffineStatistics(const std::string& registry_fingerprint,
    std::size_t parameter_version, std::vector<KFACFactorStatistics> factors)
{
  if (affine_source_ != KFACAffineStatisticsSource::PRECOMPUTED_SUMS || imported_affine_ ||
      sample_active_ || globally_reduced_ || !sample_count_)
    throw std::logic_error("KFAC affine import requires completed local fallback collection");
  if (registry_fingerprint != registry_.fingerprint() || parameter_version != parameter_version_ ||
      factors.size() != factors_.size())
    throw std::invalid_argument("KFAC imported affine identity does not match the accumulator");
  for (std::size_t block = 0; block < factors.size(); ++block)
  {
    const auto& source = factors[block];
    const auto& destination = factors_[block];
    if (source.activation_dimension != destination.activation_dimension ||
        source.sensitivity_dimension != destination.sensitivity_dimension ||
        source.activation_outer_sum.size() != checkedProduct(destination.activation_dimension,
            destination.activation_dimension, "KFAC imported activation factor") ||
        source.sensitivity_outer_sum.size() != checkedProduct(destination.sensitivity_dimension,
            destination.sensitivity_dimension, "KFAC imported sensitivity factor") ||
        source.sample_weight_sum != static_cast<DerivativeReal>(sample_count_) ||
        source.sample_weight_sum != fallback_weight_sum_ || source.row_count < sample_count_ ||
        source.row_count % sample_count_ != 0 || source.row_weight_sum != static_cast<DerivativeReal>(source.row_count) ||
        !finite(source.sample_weight_sum) || !finite(source.row_weight_sum) ||
        source.sample_weight_sum <= 0 || source.row_weight_sum <= 0 ||
        !std::all_of(source.activation_outer_sum.begin(), source.activation_outer_sum.end(), finite) ||
        !std::all_of(source.sensitivity_outer_sum.begin(), source.sensitivity_outer_sum.end(), finite))
      throw std::invalid_argument("KFAC imported raw factor sums are incompatible or nonfinite");
  }
  factors_.swap(factors);
  imported_affine_ = true;
}

std::size_t KFACFactorAccumulator::retainedNumericBytes() const noexcept
{
  std::size_t elements = 0;
  for (const auto* collection : {&factors_, &pending_})
    for (const KFACFactorStatistics& factor : *collection)
      elements += factor.activation_outer_sum.capacity() +
          factor.sensitivity_outer_sum.capacity();
  elements += fallback_diagonal_sum_.capacity() +
      pending_fallback_diagonal_.capacity() + fallback_score_sum_.capacity() +
      pending_fallback_score_.capacity() + fallback_imaginary_score_sum_.capacity() +
      pending_fallback_imaginary_score_.capacity();
  return elements * sizeof(DerivativeReal);
}

void reduceKFACFactorStatistics(KFACFactorAccumulator& accumulator,
                                Communicate* communicator,
                                std::size_t maximum_chunk_size)
{
  const std::size_t participants = communicator ? communicator->size() : 1;
  std::uint64_t failure = 0;
  if (accumulator.sample_active_)
    failure = 1;
  else if (!accumulator.affineStatisticsComplete())
    failure = 5;
  else if (accumulator.globally_reduced_)
    failure = 2;
  else if (maximum_chunk_size == 0 ||
           maximum_chunk_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    failure = 3;
  else
  {
    for (const KFACFactorStatistics& factor : accumulator.factors_)
      if (!finite(factor.sample_weight_sum) || !finite(factor.row_weight_sum) ||
          !std::all_of(factor.activation_outer_sum.begin(),
                       factor.activation_outer_sum.end(), finite) ||
          !std::all_of(factor.sensitivity_outer_sum.begin(),
                       factor.sensitivity_outer_sum.end(), finite))
      {
        failure = 4;
        break;
      }
    if (failure == 0 &&
        (!finite(accumulator.fallback_weight_sum_) ||
         !std::all_of(accumulator.fallback_diagonal_sum_.begin(),
                      accumulator.fallback_diagonal_sum_.end(), finite)))
      failure = 4;
  }

  std::uint64_t identity = UINT64_C(14695981039346656037);
  extendHash(identity, accumulator.registry_.fingerprint());
  extendHash(identity, static_cast<std::uint64_t>(accumulator.convention_));
  std::size_t packed_scalar_count = accumulator.fallback_diagonal_sum_.size() +
      accumulator.fallback_score_sum_.size() + accumulator.fallback_imaginary_score_sum_.size();
  for (const KFACFactorStatistics& factor : accumulator.factors_)
    packed_scalar_count += factor.activation_outer_sum.size() +
        factor.sensitivity_outer_sum.size() + 2;
  if (communicator)
  {
    const std::array<std::uint64_t, 7> local{
        failure, identity, accumulator.parameter_version_, accumulator.factors_.size(),
        accumulator.registry_.parameterSchema().parameterCount(), packed_scalar_count,
        maximum_chunk_size};
    std::vector<std::uint64_t> records(participants * local.size());
    auto send = local;
    communicator->allgather(send.data(), records.data(), static_cast<int>(local.size()));
    for (std::size_t rank = 0; rank < participants; ++rank)
      if (records[rank * local.size()] != 0)
        throw std::runtime_error("Distributed KFAC factor production failed on rank " +
                                 std::to_string(rank));
    for (std::size_t rank = 1; rank < participants; ++rank)
      if (!std::equal(records.begin() + 1, records.begin() + local.size(),
                      records.begin() + rank * local.size() + 1))
        throw std::runtime_error("Distributed KFAC registry/version metadata mismatch");
  }
  else if (failure == 1)
    throw std::logic_error("KFAC reduction rejects an active sample transaction");
  else if (failure == 2)
    throw std::logic_error("KFAC statistics have already been globally reduced");
  else if (failure == 3)
    throw std::invalid_argument("KFAC reduction chunk size is invalid");
  else if (failure == 5)
    throw std::logic_error("KFAC reduction requires complete imported affine statistics");
  else if (failure != 0)
    throw std::invalid_argument("KFAC reduction received non-finite local statistics");

  unsigned long sample_count = static_cast<unsigned long>(accumulator.sample_count_);
  if (communicator)
    communicator->allreduce_in_place(&sample_count, 1);
  accumulator.sample_count_ = sample_count;
  auto reduce_real_array = [&](DerivativeReal* values, std::size_t size) {
    if (!communicator)
      return;
    for (std::size_t begin = 0; begin < size; begin += maximum_chunk_size)
    {
      const std::size_t count = std::min(maximum_chunk_size, size - begin);
      communicator->allreduce_in_place(values + begin, static_cast<int>(count));
    }
  };
  if (communicator)
  {
    communicator->allreduce_in_place(&accumulator.fallback_weight_sum_, 1);
  }
  reduce_real_array(accumulator.fallback_diagonal_sum_.data(),
                    accumulator.fallback_diagonal_sum_.size());
  reduce_real_array(accumulator.fallback_score_sum_.data(), accumulator.fallback_score_sum_.size());
  reduce_real_array(accumulator.fallback_imaginary_score_sum_.data(),
                    accumulator.fallback_imaginary_score_sum_.size());
  for (std::size_t parameter = 0; parameter < accumulator.generic_entries_.size(); ++parameter)
    if (accumulator.generic_entries_[parameter])
    {
      const DerivativeReal sum = accumulator.fallback_score_sum_[parameter];
      const DerivativeReal imaginary = accumulator.fallback_imaginary_score_sum_[parameter];
      const DerivativeReal diagonal = 2.0 * sum * sum + 2.0 * imaginary * imaginary;
      if (!finite(sum) || !finite(imaginary) || !finite(diagonal))
        throw MatrixFreeNumericalError("KFAC global generic curvature overflowed");
      accumulator.fallback_diagonal_sum_[parameter] = diagonal;
    }
  for (KFACFactorStatistics& factor : accumulator.factors_)
  {
    unsigned long row_count = static_cast<unsigned long>(factor.row_count);
    if (communicator)
    {
      communicator->allreduce_in_place(&row_count, 1);
      communicator->allreduce_in_place(&factor.sample_weight_sum, 1);
      communicator->allreduce_in_place(&factor.row_weight_sum, 1);
    }
    reduce_real_array(factor.activation_outer_sum.data(),
                      factor.activation_outer_sum.size());
    reduce_real_array(factor.sensitivity_outer_sum.data(),
                      factor.sensitivity_outer_sum.size());
    factor.row_count = row_count;
  }
  accumulator.globally_reduced_ = true;
}

namespace
{
std::size_t packedStateSize(const KFACBlockRegistry& registry)
{
  std::size_t size = 0;
  const auto add = [&](std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() - size)
      throw std::overflow_error("KFAC state packing overflow");
    size += count;
  };
  for (const auto& block : registry.affineBlocks())
  {
    const auto input = block.input_width + (block.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1);
    add(checkedProduct(input, input, "KFAC state activation"));
    add(checkedProduct(block.output_width, block.output_width, "KFAC state sensitivity"));
    add(1);
  }
  for (auto block : registry.fallbackBlocks())
    add(registry.parameterSchema().blocks()[block].count);
  return size;
}

bool validConvention(KFACCurvatureConvention convention)
{
  return convention == KFACCurvatureConvention::NATIVE_SCORE ||
      convention == KFACCurvatureConvention::REGISTERED_GAUSSIAN;
}
} // namespace

void validateKFACFactorState(const KFACBlockRegistry& registry, const KFACFactorState& state)
{
  if (state.registry_fingerprint != registry.fingerprint() || !validConvention(state.convention) ||
      state.update_count == 0 || !finite(state.normalizer) || state.normalizer < 1.0 ||
      state.normalizer > static_cast<DerivativeReal>(state.update_count) ||
      state.values.size() != packedStateSize(registry) ||
      !std::all_of(state.values.begin(), state.values.end(), finite))
    throw std::invalid_argument("KFAC factor state has incompatible identity, metadata or values");
  std::size_t offset = 0;
  for (const auto& block : registry.affineBlocks())
  {
    for (auto dimension : {block.input_width + (block.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1),
                           block.output_width})
    {
      for (std::size_t row = 0; row < dimension; ++row)
      {
        if (state.values[offset + row * dimension + row] < 0.0)
          throw std::invalid_argument("KFAC factor state has a negative diagonal");
        for (std::size_t column = 0; column < row; ++column)
          if (std::abs(state.values[offset + row * dimension + column] -
                       state.values[offset + column * dimension + row]) >
              64 * std::numeric_limits<DerivativeReal>::epsilon() *
                  std::max({DerivativeReal{1}, std::abs(state.values[offset + row * dimension + column]),
                             std::abs(state.values[offset + column * dimension + row])}))
            throw std::invalid_argument("KFAC factor state is not symmetric");
      }
      offset += dimension * dimension;
    }
    if (state.values[offset++] <= 0.0)
      throw std::invalid_argument("KFAC factor state has invalid repeat normalization");
  }
  for (; offset < state.values.size(); ++offset)
    if (state.values[offset] < 0.0)
      throw std::invalid_argument("KFAC factor state has a negative fallback diagonal");
}

KFACFactorState updateKFACFactorState(const KFACFactorState* previous,
                                     const KFACFactorAccumulator& batch, DerivativeReal decay)
{
  if (!finite(decay) || decay < 0.0 || decay > 1.0 || batch.sampleActive() ||
      batch.sampleCount() == 0 || !batch.globallyReduced())
    throw std::invalid_argument("KFAC EMA requires a valid decay and globally reduced complete batch");
  KFACFactorState next;
  next.registry_fingerprint = batch.registry().fingerprint();
  next.convention = batch.curvatureConvention();
  next.normalizer = 1.0;
  next.update_count = 1;
  next.values.reserve(packedStateSize(batch.registry()));
  for (const auto& factor : batch.factors())
  {
    if (!finite(factor.sample_weight_sum) || factor.sample_weight_sum <= 0.0)
      throw std::invalid_argument("KFAC EMA batch has invalid factor weight");
    for (const auto& matrix : {&factor.activation_outer_sum, &factor.sensitivity_outer_sum})
      for (auto value : *matrix)
        next.values.push_back(value / factor.sample_weight_sum);
    next.values.push_back(factor.row_weight_sum / factor.sample_weight_sum);
  }
  if (!batch.registry().fallbackBlocks().empty())
  {
    if (!finite(batch.fallbackWeightSum()) || batch.fallbackWeightSum() <= 0.0)
      throw std::invalid_argument("KFAC EMA batch has invalid fallback weight");
    for (auto value : batch.fallbackDiagonalSum())
      next.values.push_back(value / batch.fallbackWeightSum());
  }
  validateKFACFactorState(batch.registry(), next);
  if (previous)
  {
    validateKFACFactorState(batch.registry(), *previous);
    if (previous->convention != next.convention ||
        previous->update_count == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("KFAC EMA previous state convention or update counter is incompatible");
    next.normalizer = decay * previous->normalizer + 1.0;
    next.update_count = previous->update_count + 1;
    if (!finite(next.normalizer))
      throw std::overflow_error("KFAC EMA normalizer overflowed");
    const DerivativeReal eta = 1.0 / next.normalizer;
    for (std::size_t i = 0; i < next.values.size(); ++i)
      next.values[i] = (1.0 - eta) * previous->values[i] + eta * next.values[i];
    validateKFACFactorState(batch.registry(), next);
  }
  return next;
}

struct KFACPreconditioner::PreparedBlock
{
  KFACAffineBlockDescriptor descriptor;
  DerivativeReal curvature_inverse_scale = 1.0;
  std::vector<DerivativeReal> activation_cholesky;
  std::vector<DerivativeReal> sensitivity_cholesky;
};

KFACPreconditioner::~KFACPreconditioner() = default;

KFACPreconditioner::KFACPreconditioner(
    const KFACFactorAccumulator& statistics,
    KFACPreconditionerControl control)
    : MatrixFreePreconditioner(statistics.registry().parameterSchema(),
                               statistics.parameterVersion()),
      registry_(statistics.registry()),
      control_(control),
      convention_(statistics.curvatureConvention())
{
  if (statistics.sampleActive() || statistics.sampleCount() == 0 || !statistics.affineStatisticsComplete())
    throw std::invalid_argument("KFAC preconditioner requires complete nonempty statistics");
  if (!finite(control_.damping) || control_.damping <= 0.0 ||
      !finite(control_.fallback_damping) || control_.fallback_damping <= 0.0)
    throw std::invalid_argument("KFAC damping values must be finite and positive");
  if (control_.damping_policy != KFACDampingPolicy::EQUAL_SPLIT &&
      control_.damping_policy != KFACDampingPolicy::TRACE_BALANCED)
    throw std::invalid_argument("KFAC damping policy is unsupported");
  std::size_t maximum_matrix = 0;
  std::size_t maximum_dimension = 0;
  blocks_.reserve(statistics.factors().size());
  for (std::size_t ordinal = 0; ordinal < statistics.factors().size(); ++ordinal)
  {
    const KFACFactorStatistics& factor = statistics.factors()[ordinal];
    if (!finite(factor.sample_weight_sum) || factor.sample_weight_sum <= 0.0 ||
        !finite(factor.row_weight_sum) || factor.row_weight_sum <= 0.0 ||
        factor.row_count == 0)
      throw std::invalid_argument("KFAC factor has no positive finite row weight");
    PreparedBlock prepared;
    prepared.descriptor = registry_.affineBlocks()[ordinal];
    const DerivativeReal repeats = factor.row_weight_sum / factor.sample_weight_sum;
    if (!finite(repeats) || repeats <= 0.0)
      throw std::invalid_argument("KFAC factor has an invalid repeated-row count");
    prepared.curvature_inverse_scale = repeats;
    prepared.activation_cholesky = factor.activation_outer_sum;
    prepared.sensitivity_cholesky = factor.sensitivity_outer_sum;
    for (DerivativeReal& value : prepared.activation_cholesky)
      value /= factor.sample_weight_sum;
    for (DerivativeReal& value : prepared.sensitivity_cholesky)
      value /= factor.sample_weight_sum;
    const DerivativeReal effective_damping = control_.damping * repeats;
    if (!finite(effective_damping))
      throw MatrixFreeNumericalError("KFAC repeated-block damping overflowed");
    DerivativeReal activation_damping = std::sqrt(effective_damping);
    DerivativeReal sensitivity_damping = activation_damping;
    if (control_.damping_policy == KFACDampingPolicy::TRACE_BALANCED)
    {
      DerivativeReal a = 0.0, b = 0.0;
      for (std::size_t i = 0; i < factor.activation_dimension; ++i)
        a += prepared.activation_cholesky[i * factor.activation_dimension + i] /
            static_cast<DerivativeReal>(factor.activation_dimension);
      for (std::size_t i = 0; i < factor.sensitivity_dimension; ++i)
        b += prepared.sensitivity_cholesky[i * factor.sensitivity_dimension + i] /
            static_cast<DerivativeReal>(factor.sensitivity_dimension);
      if (!finite(a) || !finite(b) || a < 0.0 || b < 0.0)
        throw MatrixFreeNumericalError("KFAC trace-balanced factor trace is invalid");
      if (a == 0.0 || b == 0.0)
      {
        // The zero-factor branch represents damping*I.
        std::fill(prepared.activation_cholesky.begin(), prepared.activation_cholesky.end(), 0.0);
        std::fill(prepared.sensitivity_cholesky.begin(), prepared.sensitivity_cholesky.end(), 0.0);
      }
      else if (factor.activation_dimension == 1)
      {
        activation_damping = 0.0;
        sensitivity_damping = effective_damping / a;
      }
      else if (factor.sensitivity_dimension == 1)
      {
        activation_damping = effective_damping / b;
        sensitivity_damping = 0.0;
      }
      else
      {
        // sqrt ratios avoid unnecessary overflow of a/b or damping*a.
        activation_damping *= std::sqrt(a) / std::sqrt(b);
        sensitivity_damping *= std::sqrt(b) / std::sqrt(a);
      }
    }
    for (std::size_t diagonal = 0; diagonal < factor.activation_dimension; ++diagonal)
      prepared.activation_cholesky[diagonal * factor.activation_dimension + diagonal] +=
          activation_damping;
    for (std::size_t diagonal = 0; diagonal < factor.sensitivity_dimension; ++diagonal)
      prepared.sensitivity_cholesky[diagonal * factor.sensitivity_dimension + diagonal] +=
          sensitivity_damping;
    cholesky(prepared.activation_cholesky, factor.activation_dimension);
    cholesky(prepared.sensitivity_cholesky, factor.sensitivity_dimension);
    maximum_matrix = std::max(maximum_matrix,
                              factor.activation_dimension * factor.sensitivity_dimension);
    maximum_dimension = std::max({maximum_dimension, factor.activation_dimension,
                                  factor.sensitivity_dimension});
    blocks_.push_back(std::move(prepared));
  }
  matrix_scratch_.resize(maximum_matrix);
  solve_scratch_.resize(2 * maximum_dimension);

  fallback_inverse_diagonal_.assign(statistics.fallbackDiagonalSum().size(), 0.0);
  if (!registry_.fallbackBlocks().empty())
  {
    if (!finite(statistics.fallbackWeightSum()) || statistics.fallbackWeightSum() <= 0.0)
      throw std::invalid_argument("KFAC fallback diagonal has no positive finite sample weight");
    const auto diagonal = statistics.fallbackDiagonalSum();
    std::size_t packed = 0;
    for (std::size_t block_index : registry_.fallbackBlocks())
    {
      const ParameterBlockDescriptor& block =
          registry_.parameterSchema().blocks()[block_index];
      for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
           ++parameter)
      {
        const DerivativeReal curvature =
            diagonal[packed] / statistics.fallbackWeightSum();
        if (!finite(curvature) || curvature < 0.0)
          throw MatrixFreeNumericalError("KFAC fallback diagonal is invalid");
        fallback_inverse_diagonal_[packed++] =
            1.0 / (curvature + control_.fallback_damping);
      }
    }
  }
  if (psiformer::memory::enabled())
    trace_capacity_.snapshotAt(__FILE__, __LINE__, "KFACPreconditioner::factors_inverse_and_solve",
        storageDiagnostics().retained_numeric_bytes, reinterpret_cast<std::uintptr_t>(this), -1, parameterVersion());
}

KFACFactorAccumulator KFACPreconditioner::statisticsFromState(
    KFACBlockRegistry registry, std::size_t parameter_version, const KFACFactorState& state)
{
  validateKFACFactorState(registry, state);
  KFACFactorAccumulator statistics(std::move(registry), parameter_version, state.convention);
  std::size_t offset = 0;
  for (auto& factor : statistics.factors_)
  {
    for (auto* matrix : {&factor.activation_outer_sum, &factor.sensitivity_outer_sum})
      for (auto& value : *matrix)
        value = state.values[offset++];
    factor.sample_weight_sum = 1.0;
    factor.row_weight_sum = state.values[offset++];
    factor.row_count = 1; // Virtual normalized batch, not a physical sample receipt.
  }
  for (auto& value : statistics.fallback_diagonal_sum_)
    value = state.values[offset++];
  statistics.fallback_weight_sum_ = 1.0;
  statistics.sample_count_ = 1;
  statistics.globally_reduced_ = true;
  return statistics;
}

KFACPreconditioner::KFACPreconditioner(KFACBlockRegistry registry, std::size_t parameter_version,
    const KFACFactorState& state, KFACPreconditionerControl control)
    : KFACPreconditioner(statisticsFromState(std::move(registry), parameter_version, state), control)
{}

KFACInverseState KFACPreconditioner::exportInverseState() const
{
  KFACInverseState state{registry_.fingerprint(), convention_, control_.damping_policy,
                         control_.damping, control_.fallback_damping, {}};
  state.values.reserve(packedStateSize(registry_));
  for (const auto& block : blocks_)
  {
    state.values.insert(state.values.end(), block.activation_cholesky.begin(), block.activation_cholesky.end());
    state.values.insert(state.values.end(), block.sensitivity_cholesky.begin(), block.sensitivity_cholesky.end());
    state.values.push_back(block.curvature_inverse_scale);
  }
  state.values.insert(state.values.end(), fallback_inverse_diagonal_.begin(), fallback_inverse_diagonal_.end());
  return state;
}

void KFACPreconditioner::prepareScratch()
{
  std::size_t maximum_matrix = 0, maximum_dimension = 0;
  for (const auto& descriptor : registry_.affineBlocks())
  {
    const auto input = descriptor.input_width + (descriptor.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1);
    maximum_matrix = std::max(maximum_matrix, input * descriptor.output_width);
    maximum_dimension = std::max({maximum_dimension, input, descriptor.output_width});
  }
  matrix_scratch_.resize(maximum_matrix);
  solve_scratch_.resize(2 * maximum_dimension);
}

KFACPreconditioner::KFACPreconditioner(KFACBlockRegistry registry, std::size_t parameter_version,
    const KFACInverseState& state)
    : MatrixFreePreconditioner(registry.parameterSchema(), parameter_version), registry_(std::move(registry)),
      control_{state.damping, state.fallback_damping, state.damping_policy}, convention_(state.convention)
{
  if (state.registry_fingerprint != registry_.fingerprint() || !validConvention(state.convention) ||
      !finite(state.damping) || state.damping <= 0.0 ||
      !finite(state.fallback_damping) || state.fallback_damping <= 0.0 ||
      (state.damping_policy != KFACDampingPolicy::EQUAL_SPLIT &&
       state.damping_policy != KFACDampingPolicy::TRACE_BALANCED) ||
      state.values.size() != packedStateSize(registry_) ||
      !std::all_of(state.values.begin(), state.values.end(), finite))
    throw std::invalid_argument("KFAC inverse state identity, controls or packing are invalid");
  std::size_t offset = 0;
  for (const auto& descriptor : registry_.affineBlocks())
  {
    PreparedBlock block;
    block.descriptor = descriptor;
    const auto input = descriptor.input_width + (descriptor.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1);
    for (auto pair : {std::make_pair(&block.activation_cholesky, input),
                      std::make_pair(&block.sensitivity_cholesky, descriptor.output_width)})
    {
      pair.first->assign(state.values.begin() + offset, state.values.begin() + offset + pair.second * pair.second);
      offset += pair.second * pair.second;
      for (std::size_t row = 0; row < pair.second; ++row)
      {
        if ((*pair.first)[row * pair.second + row] <= 0.0)
          throw std::invalid_argument("KFAC inverse state has invalid Cholesky diagonal");
        for (std::size_t column = row + 1; column < pair.second; ++column)
          if ((*pair.first)[row * pair.second + column] != 0.0)
            throw std::invalid_argument("KFAC inverse state Cholesky storage is not lower triangular");
      }
    }
    block.curvature_inverse_scale = state.values[offset++];
    if (block.curvature_inverse_scale <= 0.0)
      throw std::invalid_argument("KFAC inverse state has invalid repeat scale");
    blocks_.push_back(std::move(block));
  }
  fallback_inverse_diagonal_.assign(state.values.begin() + offset, state.values.end());
  if (!std::all_of(fallback_inverse_diagonal_.begin(), fallback_inverse_diagonal_.end(),
                   [](auto value) { return value > 0.0; }))
    throw std::invalid_argument("KFAC inverse state has invalid fallback inverse");
  prepareScratch();
  if (psiformer::memory::enabled())
    trace_capacity_.snapshotAt(__FILE__, __LINE__, "KFACPreconditioner::factors_inverse_and_solve",
        storageDiagnostics().retained_numeric_bytes, reinterpret_cast<std::uintptr_t>(this), -1, parameterVersion());
}

void KFACPreconditioner::refreshFallbackDiagonal(const KFACFactorState& state, DerivativeReal damping)
{
  validateKFACFactorState(registry_, state);
  if (state.convention != convention_ || !finite(damping) || damping <= 0.0)
    throw std::invalid_argument("KFAC fallback refresh has incompatible convention or damping");
  std::vector<DerivativeReal> inverse(fallback_inverse_diagonal_.size());
  const auto offset = state.values.size() - inverse.size();
  for (std::size_t i = 0; i < inverse.size(); ++i)
  {
    const auto denominator = state.values[offset + i] + damping;
    inverse[i] = 1.0 / denominator;
    if (!finite(denominator) || !finite(inverse[i]) || inverse[i] <= 0.0)
      throw MatrixFreeNumericalError("KFAC refreshed fallback inverse is invalid");
  }
  fallback_inverse_diagonal_.swap(inverse);
  control_.fallback_damping = damping;
  if (psiformer::memory::enabled())
    trace_capacity_.snapshotAt(__FILE__, __LINE__, "KFACPreconditioner::factors_inverse_and_solve",
        storageDiagnostics().retained_numeric_bytes, reinterpret_cast<std::uintptr_t>(this), -1, parameterVersion());
}

KFACStorageDiagnostics KFACPreconditioner::storageDiagnostics() const noexcept
{
  std::size_t factor_elements = 0;
  for (const PreparedBlock& block : blocks_)
    factor_elements += block.activation_cholesky.capacity() +
        block.sensitivity_cholesky.capacity();
  const std::size_t scratch = matrix_scratch_.capacity() + solve_scratch_.capacity() +
      fallback_inverse_diagonal_.capacity();
  return {registry_.parameterSchema().parameterCount(), blocks_.size(), factor_elements,
          scratch, (factor_elements + scratch) * sizeof(DerivativeReal)};
}

void KFACPreconditioner::evaluate(
    const StructuredParameterVectorConstView& residual,
    DerivativeArrayView<DerivativeValue> result) const
{
  std::fill(result.begin(), result.end(), DerivativeValue{});
  const auto& schema_blocks = registry_.parameterSchema().blocks();
  for (const PreparedBlock& block : blocks_)
  {
    const auto& descriptor = block.descriptor;
    const std::size_t activation_dimension = descriptor.input_width +
        (descriptor.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1);
    const std::size_t output_width = descriptor.output_width;
    const ParameterBlockDescriptor& weight = schema_blocks[descriptor.weight_block];

    // Assemble the augmented input-by-output gradient matrix.
    for (std::size_t input = 0; input < descriptor.input_width; ++input)
      for (std::size_t output = 0; output < output_width; ++output)
        matrix_scratch_[input * output_width + output] =
            residual.values()[weight.offset + input * output_width + output].real();
    if (descriptor.bias_block != NO_KFAC_BIAS_BLOCK)
    {
      const ParameterBlockDescriptor& bias = schema_blocks[descriptor.bias_block];
      for (std::size_t output = 0; output < output_width; ++output)
        matrix_scratch_[descriptor.input_width * output_width + output] =
            residual.values()[bias.offset + output].real();
    }

    // Left solve by A.  Each output column is an independent right-hand side.
    for (std::size_t output = 0; output < output_width; ++output)
    {
      for (std::size_t input = 0; input < activation_dimension; ++input)
        solve_scratch_[input] = matrix_scratch_[input * output_width + output];
      solveCholesky(block.activation_cholesky, activation_dimension,
                    solve_scratch_.data(), solve_scratch_.data() + activation_dimension,
                    solve_scratch_.data());
      for (std::size_t input = 0; input < activation_dimension; ++input)
        matrix_scratch_[input * output_width + output] =
            solve_scratch_[activation_dimension + input];
    }

    // Right solve by G.  G is symmetric, so solve one input row directly.
    for (std::size_t input = 0; input < activation_dimension; ++input)
    {
      DerivativeReal* row = matrix_scratch_.data() + input * output_width;
      solveCholesky(block.sensitivity_cholesky, output_width, row,
                    solve_scratch_.data() + output_width, solve_scratch_.data());
      std::copy_n(solve_scratch_.data() + output_width, output_width, row);
    }

    for (std::size_t input = 0; input < descriptor.input_width; ++input)
      for (std::size_t output = 0; output < output_width; ++output)
        result[weight.offset + input * output_width + output] =
            block.curvature_inverse_scale *
            matrix_scratch_[input * output_width + output];
    if (descriptor.bias_block != NO_KFAC_BIAS_BLOCK)
    {
      const ParameterBlockDescriptor& bias = schema_blocks[descriptor.bias_block];
      for (std::size_t output = 0; output < output_width; ++output)
        result[bias.offset + output] =
            block.curvature_inverse_scale *
            matrix_scratch_[descriptor.input_width * output_width + output];
    }
  }

  std::size_t packed_fallback = 0;
  for (std::size_t block_index : registry_.fallbackBlocks())
  {
    const ParameterBlockDescriptor& block = schema_blocks[block_index];
    for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
         ++parameter)
      result[parameter] = residual.values()[parameter] *
          fallback_inverse_diagonal_[packed_fallback++];
  }
}

} // namespace qmcplusplus::wftrain
