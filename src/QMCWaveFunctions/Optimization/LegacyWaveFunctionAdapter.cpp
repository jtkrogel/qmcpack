//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file LegacyWaveFunctionAdapter.cpp
 * @brief Implementation of the bounded legacy-variable training bridge.
 */

#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr std::size_t MAXIMUM_VJP_CHANNELS = 3;

/// Test IEEE-754 finiteness without fast-math assumptions.
bool isFiniteDouble(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559,
                "Legacy adapter requires IEEE-754 binary64 parameters");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Multiply storage extents without allowing size_t wraparound.
std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs, const char* context)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::length_error(std::string(context) + " exceeds addressable storage");
  return lhs * rhs;
}

/// Add storage extents without allowing size_t wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* context)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::length_error(std::string(context) + " exceeds addressable storage");
  return lhs + rhs;
}

/// Return the conservative two-row plus three-channel derivative scratch bound.
std::size_t requiredScratchBytes(std::size_t parameter_count)
{
  const std::size_t legacy_rows = checkedMultiply(
      checkedMultiply(2, parameter_count, "Legacy derivative rows"), sizeof(QMCTraits::ValueType),
      "Legacy derivative row bytes");
  const std::size_t channels = checkedMultiply(
      checkedMultiply(MAXIMUM_VJP_CHANNELS, parameter_count, "Legacy derivative channels"),
      sizeof(DerivativeValue), "Legacy derivative channel bytes");
  return checkedAdd(legacy_rows, channels, "Legacy derivative scratch");
}

/// Map a legacy parameter category to a stable, human-readable update group.
std::string categoryName(int category)
{
  switch (category)
  {
  case optimize::OTHER_P:
    return "legacy_other";
  case optimize::LOGLINEAR_P:
    return "legacy_loglinear";
  case optimize::LOGLINEAR_K:
    return "legacy_kspace_loglinear";
  case optimize::LINEAR_P:
    return "legacy_linear";
  case optimize::SPO_P:
    return "legacy_spo";
  case optimize::BACKFLOW_P:
    return "legacy_backflow";
  default:
    return "legacy_category_" + std::to_string(category);
  }
}

/// Mix bytes into a deterministic FNV-1a digest independent of std::hash.
void mixDigest(std::uint64_t& digest, const void* data, std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index)
  {
    digest ^= bytes[index];
    digest *= UINT64_C(1099511628211);
  }
}

/// Form a stable block identifier from the ordered legacy names and category.
std::string blockId(const optimize::VariableSet& variables,
                    const std::vector<std::size_t>& active_locations,
                    std::size_t first,
                    std::size_t count,
                    int category)
{
  std::uint64_t digest = UINT64_C(14695981039346656037);
  const std::string category_identity = std::to_string(category);
  mixDigest(digest, category_identity.data(), category_identity.size());
  for (std::size_t index = first; index < first + count; ++index)
  {
    const std::string& name = variables.name(active_locations[index]);
    mixDigest(digest, name.data(), name.size());
    const unsigned char separator = 0;
    mixDigest(digest, &separator, sizeof(separator));
  }

  std::ostringstream id;
  id << categoryName(category) << '/' << std::hex << std::setfill('0') << std::setw(16) << digest;
  return id.str();
}

/// Build compact category runs and the canonical active-index lookup.
std::pair<std::vector<ParameterBlockDescriptor>, std::vector<std::size_t>> makeBlocks(
    const optimize::VariableSet& variables)
{
  const std::size_t active_count = static_cast<std::size_t>(variables.size_of_active());
  std::vector<std::size_t> active_locations(active_count, variables.size());
  for (std::size_t location = 0; location < variables.size(); ++location)
  {
    const int active_index = variables.where(static_cast<int>(location));
    if (active_index >= 0)
    {
      if (static_cast<std::size_t>(active_index) >= active_count ||
          active_locations[active_index] != variables.size())
        throw std::logic_error("Legacy variables do not have unique dense active indices");
      active_locations[active_index] = location;
    }
  }
  if (std::find(active_locations.begin(), active_locations.end(), variables.size()) != active_locations.end())
    throw std::logic_error("Legacy variables have an incomplete active-index ordering");

  std::vector<ParameterBlockDescriptor> blocks;
  for (std::size_t first = 0; first < active_count;)
  {
    const int category = variables.getType(active_locations[first]);
    std::size_t last   = first + 1;
    while (last < active_count && variables.getType(active_locations[last]) == category)
      ++last;
    const std::size_t count = last - first;
    blocks.push_back({blockId(variables, active_locations, first, count, category),
                      {count},
                      first,
                      count,
                      ParameterScalarDomain::REAL64,
                      true,
                      categoryName(category)});
    first = last;
  }
  return {std::move(blocks), std::move(active_locations)};
}

/// Verify that a disposable evaluator registers exactly the source variable layout.
optimize::VariableSet validateEvaluatorRegistration(TrialWaveFunction& evaluator,
                                                    const optimize::VariableSet& expected)
{
  if (!evaluator.extractStructuredParameterProviders().empty())
    throw std::invalid_argument("Legacy adapter evaluators must not contain structured parameter providers");

  optimize::VariableSet actual;
  evaluator.checkInVariables(actual);
  actual.resetIndex();
  if (actual.size() != expected.size() || actual.size_of_active() != expected.size_of_active())
    throw std::invalid_argument("Legacy adapter evaluator variable extent does not match registration source");
  for (std::size_t index = 0; index < actual.size(); ++index)
    if (actual.name(index) != expected.name(index) || actual.getType(index) != expected.getType(index) ||
        actual.where(static_cast<int>(index)) != expected.where(static_cast<int>(index)) ||
        (actual.where(static_cast<int>(index)) < 0 && actual[static_cast<int>(index)] !=
             expected[static_cast<int>(index)]))
      throw std::invalid_argument("Legacy adapter evaluator variable registration does not match source");
  return actual;
}

/// Install one detached snapshot in a disposable evaluator and recompute its state.
void prepareEvaluator(TrialWaveFunction& evaluator,
                      ParticleSet& particles,
                      const optimize::VariableSet& expected,
                      const std::vector<double>& values)
{
  optimize::VariableSet active = validateEvaluatorRegistration(evaluator, expected);
  for (std::size_t location = 0; location < active.size(); ++location)
  {
    const int active_index = active.where(static_cast<int>(location));
    if (active_index >= 0)
      active[static_cast<int>(location)] = values[active_index];
  }
  evaluator.checkOutVariables(active);
  evaluator.resetParameters(active);
  evaluator.evaluateLog(particles);
}

/// Convert either real- or complex-build wavefunction scalars without projection.
DerivativeValue asDerivativeValue(QMCTraits::ValueType value) noexcept
{
  return {static_cast<DerivativeReal>(std::real(value)), static_cast<DerivativeReal>(std::imag(value))};
}

/** Contract scalar legacy derivative rows immediately into bounded accumulators.
 *
 * Evaluators and particles are nonowning and exclusive for this operator's lifetime.
 * The only full-P storage is two legacy rows and three complex accumulators.
 */
class LegacyStreamingDerivativeOperator final : public StreamingDerivativeOperator
{
public:
  LegacyStreamingDerivativeOperator(std::shared_ptr<const StructuredParameterSchema> schema,
                                    std::shared_ptr<const optimize::VariableSet> active,
                                    StructuredParameterSnapshot snapshot,
                                    const RefVector<TrialWaveFunction>& evaluators,
                                    const RefVector<ParticleSet>& particle_sets,
                                    std::size_t batch_ordinal,
                                    std::size_t sample_offset,
                                    std::size_t maximum_chunk_size,
                                    std::size_t maximum_scratch_bytes)
      : schema_(std::move(schema)),
        active_(std::move(active)),
        parameter_version_(snapshot.version),
        evaluators_(evaluators),
        particle_sets_(particle_sets),
        batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset),
        chunk_plan_(*schema_, parameter_version_, maximum_chunk_size),
        score_row_(schema_->parameterCount()),
        kinetic_row_(schema_->parameterCount()),
        channel_accumulators_{std::vector<DerivativeValue>(schema_->parameterCount()),
                              std::vector<DerivativeValue>(schema_->parameterCount()),
                              std::vector<DerivativeValue>(schema_->parameterCount())}
  {
    // Measure allocator capacities before invoking any fallible legacy callback.
    initializeStorageDiagnostics();
    if (storage_diagnostics_.retained_numeric_bytes > maximum_scratch_bytes)
      throw std::length_error("Legacy adapter allocated derivative scratch beyond its mandatory byte ceiling");
    for (std::size_t sample = 0; sample < evaluators_.size(); ++sample)
      prepareEvaluator(evaluators_[sample], particle_sets_[sample], *active_, snapshot.values);
  }

  /// Advertise only the bounded host products implemented by the scalar bridge.
  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain       = ParameterScalarDomain::REAL64;
    result.result_scalar_domain          = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain              = ReductionDomain::CROWD_LOCAL;
    result.execution_domain              = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask        = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    result.maximum_vjp_channels          = MAXIMUM_VJP_CHANNELS;
    result.maximum_parameter_chunk_size  = chunk_plan_.maximumChunkSize();
    result.maximum_sample_tile_size      = 1;
    result.fixed_parameter_scratch_vectors = 2 + MAXIMUM_VJP_CHANNELS;
    result.block_streaming               = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return *schema_; }
  std::size_t parameterVersion() const noexcept override { return parameter_version_; }
  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }
  std::size_t sampleOffset() const noexcept override { return sample_offset_; }
  std::size_t sampleCount() const noexcept override { return evaluators_.size(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return chunk_plan_; }

  /// Report the exact numeric buffers retained by the compatibility operator.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    return storage_diagnostics_;
  }

protected:
  /// Evaluate one scalar row at a time and contract it into at most three channels.
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
      std::fill(channel_accumulators_[channel].begin(), channel_accumulators_[channel].end(), DerivativeValue{});

    for (std::size_t sample = 0; sample < evaluators_.size(); ++sample)
    {
      evaluateRows(sample);
      for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
          DerivativeValue derivative = channels[channel].product == DerivativeProduct::SCORE_VJP
              ? asDerivativeValue(score_row_[parameter])
              : asDerivativeValue(kinetic_row_[parameter]);
          if (adjoint == DerivativeAdjoint::HERMITIAN)
            derivative = std::conj(derivative);
          channel_accumulators_[channel][parameter] +=
              derivative * channels[channel].coefficients.values[sample];
        }
    }

    for (const ParameterChunkDescriptor& chunk : chunk_plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        sink.add(channel,
                 {chunk,
                  {channel_accumulators_[channel].data() + chunk.parameter_offset, chunk.count}});
  }

  /// Evaluate and emit one scalar score-direction product per sample.
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    for (std::size_t sample = 0; sample < evaluators_.size(); ++sample)
    {
      evaluateRows(sample);
      DerivativeValue product{};
      for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
        product += asDerivativeValue(score_row_[parameter]) * direction.values()[parameter];
      const SampleProductTileDescriptor descriptor{schema_->providerId(),
                                                   schema_->fingerprint(),
                                                   parameter_version_,
                                                   batch_ordinal_,
                                                   sample_offset_ + sample,
                                                   1,
                                                   sample};
      sink.add({descriptor, {&product, 1}});
    }
  }

private:
  /// Reuse the two legacy full-P rows for one evaluator configuration.
  void evaluateRows(std::size_t sample) const
  {
    score_row_   = QMCTraits::ValueType{};
    kinetic_row_ = QMCTraits::ValueType{};
    evaluators_[sample].get().evaluateDerivatives(
        particle_sets_[sample].get(), *active_, score_row_, kinetic_row_);
  }

  /// Capture the retained-buffer byte count after all construction allocations.
  void initializeStorageDiagnostics()
  {
    storage_diagnostics_.parameter_count = schema_->parameterCount();
    storage_diagnostics_.sample_count    = evaluators_.size();
#ifdef QMC_COMPLEX
    storage_diagnostics_.complex_parameter_vectors = 2 + MAXIMUM_VJP_CHANNELS;
#else
    storage_diagnostics_.real_parameter_vectors    = 2;
    storage_diagnostics_.complex_parameter_vectors = MAXIMUM_VJP_CHANNELS;
#endif
    const std::size_t legacy_row_entries = checkedAdd(score_row_.capacity(), kinetic_row_.capacity(),
                                                      "Legacy derivative row capacity");
    const std::size_t legacy_row_bytes = checkedMultiply(
        legacy_row_entries, sizeof(QMCTraits::ValueType), "Legacy derivative row capacity bytes");
    std::size_t channel_entries = 0;
    for (const auto& channel : channel_accumulators_)
      channel_entries = checkedAdd(channel_entries, channel.capacity(), "Legacy derivative channel capacity");
    const std::size_t channel_bytes = checkedMultiply(
        channel_entries, sizeof(DerivativeValue), "Legacy derivative channel capacity bytes");
    storage_diagnostics_.parameter_scratch_bytes = checkedAdd(
        legacy_row_bytes, channel_bytes, "Legacy retained derivative scratch");
    storage_diagnostics_.retained_numeric_bytes  = storage_diagnostics_.parameter_scratch_bytes;
    std::size_t fingerprint = reinterpret_cast<std::size_t>(score_row_.data());
    fingerprint ^= reinterpret_cast<std::size_t>(kinetic_row_.data()) + std::size_t{0x9e3779b9U};
    for (const auto& channel : channel_accumulators_)
      fingerprint ^= reinterpret_cast<std::size_t>(channel.data()) + (fingerprint << 6) + (fingerprint >> 2);
    storage_diagnostics_.storage_fingerprint = fingerprint == 0 ? 1 : fingerprint;
  }

  std::shared_ptr<const StructuredParameterSchema> schema_;
  std::shared_ptr<const optimize::VariableSet> active_;
  std::size_t parameter_version_ = 0;
  RefVector<TrialWaveFunction> evaluators_;
  RefVector<ParticleSet> particle_sets_;
  std::size_t batch_ordinal_ = 0;
  std::size_t sample_offset_ = 0;
  ParameterChunkPlan chunk_plan_;
  mutable Vector<QMCTraits::ValueType> score_row_;
  mutable Vector<QMCTraits::ValueType> kinetic_row_;
  mutable std::array<std::vector<DerivativeValue>, MAXIMUM_VJP_CHANNELS> channel_accumulators_;
  StreamingDerivativeStorageDiagnostics storage_diagnostics_;
};

} // namespace

LegacyWaveFunctionAdapter::LegacyWaveFunctionAdapter(std::string provider_id,
                                                     TrialWaveFunction& registration_source,
                                                     LegacyWaveFunctionAdapterOptions options)
    : options_(options)
{
  if (provider_id.empty())
    throw std::invalid_argument("Legacy adapter provider identity must not be empty");
  if (options_.maximum_parameter_count == 0 || options_.maximum_derivative_scratch_bytes == 0 ||
      options_.maximum_parameter_chunk_size == 0)
    throw std::invalid_argument("Legacy adapter requires positive parameter, scratch, and chunk limits");
  if (!registration_source.extractStructuredParameterProviders().empty())
    throw std::invalid_argument("Legacy adapter cannot aggregate existing structured parameter providers");

  optimize::VariableSet registration_template;
  registration_source.checkInVariables(registration_template);
  registration_template.resetIndex();
  if (registration_template.size_of_active() <= 0)
    throw std::invalid_argument("Legacy adapter requires at least one active parameter");
  for (std::size_t location = 0; location < registration_template.size(); ++location)
    if (!isFiniteDouble(registration_template[static_cast<int>(location)]))
      throw std::invalid_argument("Legacy adapter registration contains a non-finite parameter");

  const auto [blocks, active_locations] = makeBlocks(registration_template);
  const std::size_t parameter_count     = active_locations.size();
  if (parameter_count > options_.maximum_parameter_count)
    throw std::length_error("Legacy adapter parameter count exceeds its mandatory ceiling");
  if (requiredScratchBytes(parameter_count) > options_.maximum_derivative_scratch_bytes)
    throw std::length_error("Legacy adapter derivative scratch exceeds its mandatory byte ceiling");

  parameter_values_.resize(parameter_count);
  for (std::size_t index = 0; index < parameter_count; ++index)
    parameter_values_[index] = registration_template[static_cast<int>(active_locations[index])];
  registration_template_ =
      std::make_shared<const optimize::VariableSet>(std::move(registration_template));
  schema_ = std::make_shared<const StructuredParameterSchema>(std::move(provider_id), blocks);
}

StructuredParameterSnapshot LegacyWaveFunctionAdapter::snapshotParameters() const
{
  std::lock_guard lock(state_mutex_);
  return {schema_->fingerprint(), parameter_version_, parameter_values_};
}

std::size_t LegacyWaveFunctionAdapter::publishParameters(const StructuredParameterSnapshot& candidate,
                                                         std::size_t expected_version)
{
  if (candidate.schema_fingerprint != schema_->fingerprint())
    throw std::invalid_argument("Legacy adapter candidate has the wrong schema fingerprint");
  if (candidate.values.size() != schema_->parameterCount())
    throw std::invalid_argument("Legacy adapter candidate has the wrong parameter extent");
  if (std::any_of(candidate.values.begin(), candidate.values.end(),
                  [](double value) { return !isFiniteDouble(value); }))
    throw std::invalid_argument("Legacy adapter candidate contains a non-finite value");

  // Copy before locking so allocation failure cannot affect the authoritative state.
  std::vector<double> replacement(candidate.values);
  std::lock_guard lock(state_mutex_);
  if (expected_version != parameter_version_ || candidate.version != expected_version)
    throw std::runtime_error("Legacy adapter candidate parameter version is stale");
  if (parameter_version_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Legacy adapter parameter version overflow");
  parameter_values_.swap(replacement);
  return ++parameter_version_;
}

std::unique_ptr<StreamingDerivativeOperator> LegacyWaveFunctionAdapter::makeDerivativeOperator(
    const RefVector<TrialWaveFunction>& evaluators,
    const RefVector<ParticleSet>& particle_sets,
    std::size_t batch_ordinal,
    std::size_t sample_offset) const
{
  if (evaluators.size() != particle_sets.size())
    throw std::invalid_argument("Legacy adapter evaluator and particle batches have different sizes");
  if (sample_offset > std::numeric_limits<std::size_t>::max() - evaluators.size())
    throw std::overflow_error("Legacy adapter sample interval overflow");

  // Capability and resource failures occur before any legacy evaluator callback.
  if (schema_->parameterCount() > options_.maximum_parameter_count ||
      requiredScratchBytes(schema_->parameterCount()) > options_.maximum_derivative_scratch_bytes)
    throw std::length_error("Legacy adapter derivative request exceeds configured resource ceilings");
  for (const ParticleSet& particles : particle_sets)
  {
    const auto& masses = particles.get_mass_by_group();
    if (masses.size() == 0 || std::any_of(masses.begin(), masses.end(), [](auto mass) { return mass != 1.0; }))
      throw std::invalid_argument("Legacy adapter supports only initialized unit electron masses");
  }

  const StructuredParameterSnapshot snapshot = snapshotParameters();
  return std::make_unique<LegacyStreamingDerivativeOperator>(
      schema_, registration_template_, snapshot, evaluators, particle_sets, batch_ordinal, sample_offset,
      std::min(options_.maximum_parameter_chunk_size, schema_->parameterCount()),
      options_.maximum_derivative_scratch_bytes);
}

} // namespace qmcplusplus::wftrain
