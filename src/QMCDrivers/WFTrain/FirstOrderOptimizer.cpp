//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file FirstOrderOptimizer.cpp
 * @brief Transactional implementations of bounded first-order update rules.
 */

#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"

#include "io/hdf/hdf_archive.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceAdam.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

constexpr char CONTRIBUTOR_ID[] = "qmcpack/wftrain/first-order-optimizer";

/** Test IEEE-754 finiteness without relying on fast-math-sensitive classification.
 *
 * QMCPACK release builds enable -ffast-math, under which the compiler may fold
 * std::isfinite to true. Parameter validation must still reject NaN and infinity.
 */
bool isFiniteDouble(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Extend one FNV-1a fingerprint with a raw byte interval.
void extendFingerprint(std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
  const auto* input = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < bytes; ++index)
  {
    hash ^= input[index];
    hash *= UINT64_C(1099511628211);
  }
}

/// Extend a fingerprint with a length-delimited string.
void extendString(std::uint64_t& hash, const std::string& value) noexcept
{
  const std::uint64_t size = value.size();
  extendFingerprint(hash, &size, sizeof(size));
  extendFingerprint(hash, value.data(), value.size());
}

/// Render one fixed-width fingerprint for checkpoint identity metadata.
std::string formatFingerprint(std::uint64_t hash)
{
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

/// Return a stable name for diagnostics and configuration fingerprints.
const char* methodName(FirstOrderMethod method)
{
  switch (method)
  {
  case FirstOrderMethod::SGD:
    return "sgd";
  case FirstOrderMethod::MOMENTUM_SGD:
    return "momentum_sgd";
  case FirstOrderMethod::RMSPROP:
    return "rmsprop";
  case FirstOrderMethod::ADAM:
    return "adam";
  case FirstOrderMethod::AMSGRAD:
    return "amsgrad";
  }
  throw std::invalid_argument("Unknown first-order optimizer method");
}

/// Require a finite scalar in the half-open unit interval.
void validateDecay(double value, const char* name)
{
  if (!isFiniteDouble(value) || value < 0.0 || value >= 1.0)
    throw std::invalid_argument(std::string("First-order optimizer ") + name +
                                " must be finite and in [0, 1)");
}

/// Compute 1-beta^iteration without early-iteration cancellation.
double oneMinusPower(double beta, std::uint64_t iteration) noexcept
{
  if (beta == 0.0)
    return 1.0;
  return -std::expm1(static_cast<double>(iteration) * std::log(beta));
}

} // namespace

/// Owning restore transaction whose final state installation cannot throw.
class FirstOrderOptimizer::PreparedRestoreAction final
    : public OptimizerCheckpointContributor::PreparedRestore
{
public:
  PreparedRestoreAction(FirstOrderOptimizer& owner,
                        std::uint64_t update_count,
                        std::vector<double> first_moment,
                        std::vector<double> second_moment,
                        std::vector<double> maximum_second_moment)
      : owner_(owner),
        update_count_(update_count),
        first_moment_(std::move(first_moment)),
        second_moment_(std::move(second_moment)),
        maximum_second_moment_(std::move(maximum_second_moment))
  {}

  void commit() noexcept override
  {
    owner_.first_moment_.swap(first_moment_);
    owner_.second_moment_.swap(second_moment_);
    owner_.maximum_second_moment_.swap(maximum_second_moment_);
    owner_.accepted_update_count_ = update_count_;
    owner_.moment_mirrors_stale_ = false;
#ifdef ENABLE_CUDA
    if (owner_.device_adam_)
      owner_.device_adam_->invalidateSeed();
#endif
  }

private:
  FirstOrderOptimizer& owner_;
  std::uint64_t update_count_;
  std::vector<double> first_moment_;
  std::vector<double> second_moment_;
  std::vector<double> maximum_second_moment_;
};

FirstOrderOptimizer::FirstOrderOptimizer(const StructuredParameterSchema& schema,
                                         FirstOrderOptimizerOptions options)
    : options_(std::move(options)),
      provider_id_(schema.providerId()),
      schema_fingerprint_(schema.fingerprint()),
      parameter_count_(schema.parameterCount())
{
  // Validate the complete public configuration even when a field is inactive for
  // the selected method; this prevents latent invalid settings from surviving a
  // later method change in driver-side configuration code.
  validateDecay(options_.momentum_decay, "momentum decay");
  validateDecay(options_.rms_decay, "RMSProp decay");
  validateDecay(options_.adam_beta1, "Adam beta1");
  validateDecay(options_.adam_beta2, "Adam beta2");
  if (!isFiniteDouble(options_.epsilon) || options_.epsilon <= 0.0)
    throw std::invalid_argument("First-order optimizer epsilon must be positive and finite");

  // Exhaustively validate the enum before it controls storage or arithmetic.
  switch (options_.method)
  {
  case FirstOrderMethod::SGD:
  case FirstOrderMethod::MOMENTUM_SGD:
  case FirstOrderMethod::RMSPROP:
  case FirstOrderMethod::ADAM:
  case FirstOrderMethod::AMSGRAD:
    break;
  default:
    throw std::invalid_argument("Unknown first-order optimizer method");
  }

  // Resolve rates by named update group once; runtime updates retain only compact
  // block ranges and never allocate a per-parameter learning-rate vector.
  std::map<std::string, double> rates;
  for (const UpdateGroupLearningRate& entry : options_.learning_rates)
  {
    if (!isFiniteDouble(entry.learning_rate) || entry.learning_rate <= 0.0)
      throw std::invalid_argument(
          "First-order optimizer learning rates must be positive and finite");
    if (!rates.emplace(entry.update_group, entry.learning_rate).second)
      throw std::invalid_argument("Duplicate first-order optimizer learning-rate group: " +
                                  entry.update_group);
  }

  std::set<std::string> required_groups;
  blocks_.reserve(schema.blocks().size());
  for (const ParameterBlockDescriptor& block : schema.blocks())
  {
    if (block.scalar_domain != ParameterScalarDomain::REAL64)
      throw std::invalid_argument("First-order optimizer supports only real parameter blocks");

    double learning_rate = 0.0;
    if (block.trainable)
    {
      required_groups.insert(block.update_group);
      const auto rate = rates.find(block.update_group);
      if (rate == rates.end())
        throw std::invalid_argument("Missing first-order optimizer learning rate for group: " +
                                    block.update_group);
      learning_rate = rate->second;
    }
    blocks_.push_back({block.offset, block.count, block.trainable, learning_rate});
  }
  for (const auto& rate : rates)
    if (required_groups.count(rate.first) == 0)
      throw std::invalid_argument("Unknown first-order optimizer learning-rate group: " +
                                  rate.first);

  switch (options_.method)
  {
  case FirstOrderMethod::SGD:
    break;
  case FirstOrderMethod::MOMENTUM_SGD:
    first_moment_.assign(parameter_count_, 0.0);
    break;
  case FirstOrderMethod::RMSPROP:
    second_moment_.assign(parameter_count_, 0.0);
    break;
  case FirstOrderMethod::ADAM:
  case FirstOrderMethod::AMSGRAD:
    first_moment_.assign(parameter_count_, 0.0);
    second_moment_.assign(parameter_count_, 0.0);
    if (options_.method == FirstOrderMethod::AMSGRAD)
      maximum_second_moment_.assign(parameter_count_, 0.0);
    break;
  }

  // Configuration identity includes the schema, selected method, applicable
  // numerical constants, and the canonical name-sorted group-rate mapping.
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendString(hash, schema_fingerprint_);
  extendString(hash, methodName(options_.method));
  switch (options_.method)
  {
  case FirstOrderMethod::SGD:
    break;
  case FirstOrderMethod::MOMENTUM_SGD:
    extendFingerprint(hash, &options_.momentum_decay, sizeof(double));
    break;
  case FirstOrderMethod::RMSPROP:
    extendFingerprint(hash, &options_.rms_decay, sizeof(double));
    extendFingerprint(hash, &options_.epsilon, sizeof(double));
    break;
  case FirstOrderMethod::ADAM:
  case FirstOrderMethod::AMSGRAD:
    extendFingerprint(hash, &options_.adam_beta1, sizeof(double));
    extendFingerprint(hash, &options_.adam_beta2, sizeof(double));
    extendFingerprint(hash, &options_.epsilon, sizeof(double));
    break;
  }
  for (const auto& [group, rate] : rates)
  {
    extendString(hash, group);
    extendFingerprint(hash, &rate, sizeof(double));
  }
  configuration_fingerprint_ = formatFingerprint(hash);
}

/// Preserve old CPU destruction while allowing the optional owner to stay opaque.
FirstOrderOptimizer::~FirstOrderOptimizer() = default;

/// Prepare only the approved method; compact blocks retain existing resolved rates.
void FirstOrderOptimizer::prepareDeviceAdam(std::size_t device_budget_bytes)
{
  if (!device_budget_bytes)
    return;
#ifdef ENABLE_CUDA
  if (options_.method != FirstOrderMethod::ADAM || proposal_live_ || device_adam_)
    throw std::invalid_argument("Explicit device Adam needs a fresh real-Adam optimizer");
  std::vector<psiformer::device::AdamBlock> blocks;
  blocks.reserve(blocks_.size());
  for (const auto& block : blocks_)
    blocks.push_back({block.offset, block.count, block.learning_rate, block.trainable ? 1u : 0u});
  const auto bytes = psiformer::device::estimateAdamBytes(parameter_count_, blocks.size());
  if (bytes > std::numeric_limits<std::size_t>::max() - retainedBytes())
    throw std::overflow_error("Device Adam combined host/device retained bytes overflow");
  device_adam_ = psiformer::device::PreparedDeviceAdam::prepare(parameter_count_, blocks, device_budget_bytes);
#else
  throw std::invalid_argument("Explicit device Adam is not compiled");
#endif
}

/// Download only authoritative committed moments; failed copies leave mirrors stale.
void FirstOrderOptimizer::synchronizeMoments() const
{
#ifdef ENABLE_CUDA
  if (device_adam_ && moment_mirrors_stale_)
  {
    device_adam_->download(first_moment_, second_moment_);
    moment_mirrors_stale_ = false;
  }
#endif
}

/// Inspect committed first moments, with explicit checked synchronization if needed.
const std::vector<double>& FirstOrderOptimizer::firstMoment() const
{
  synchronizeMoments();
  return first_moment_;
}

/// Inspect committed second moments, with explicit checked synchronization if needed.
const std::vector<double>& FirstOrderOptimizer::secondMoment() const
{
  synchronizeMoments();
  return second_moment_;
}

void FirstOrderOptimizer::validateInputs(const StructuredParameterSchema& schema,
                                         const StructuredParameterSnapshot& parameters,
                                         ParameterGradientView objective) const
{
  if (proposal_live_)
    throw std::logic_error("First-order optimizer already has a live proposal");
  if (schema.providerId() != provider_id_ || schema.fingerprint() != schema_fingerprint_ ||
      schema.parameterCount() != parameter_count_)
    throw std::invalid_argument("First-order optimizer schema does not match its binding");
  if (parameters.schema_fingerprint != schema_fingerprint_ ||
      parameters.values.size() != parameter_count_)
    throw std::invalid_argument("First-order optimizer parameter snapshot is incompatible");
  if (objective.schema_fingerprint != schema_fingerprint_ ||
      objective.parameter_version != parameters.version ||
      objective.gradient.size() != parameter_count_)
    throw std::invalid_argument("First-order optimizer objective is incompatible");
  if (objective.reduction_domain != ReductionDomain::GLOBAL)
    throw std::invalid_argument("First-order optimizer requires a globally reduced objective");
  if (!std::all_of(parameters.values.begin(), parameters.values.end(),
                   [](double value) { return isFiniteDouble(value); }) ||
      !std::all_of(objective.gradient.begin(), objective.gradient.end(),
                   [](double value) { return isFiniteDouble(value); }))
    throw std::invalid_argument("First-order optimizer inputs must be finite");
  if (accepted_update_count_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("First-order optimizer update count overflow");
}

StructuredParameterSnapshot FirstOrderOptimizer::propose(
    const StructuredParameterSchema& schema,
    const StructuredParameterSnapshot& parameters,
    ParameterGradientView objective)
{
  validateInputs(schema, parameters, objective);
  StructuredParameterSnapshot candidate = parameters;
  const std::uint64_t next_iteration = accepted_update_count_ + 1;
  const double adam_first_correction =
      options_.method == FirstOrderMethod::ADAM || options_.method == FirstOrderMethod::AMSGRAD
      ? oneMinusPower(options_.adam_beta1, next_iteration)
      : 1.0;
  const double adam_second_correction =
      options_.method == FirstOrderMethod::ADAM || options_.method == FirstOrderMethod::AMSGRAD
      ? oneMinusPower(options_.adam_beta2, next_iteration)
      : 1.0;

#ifdef ENABLE_CUDA
  if (device_adam_)
  {
    if (!device_adam_->seeded())
      device_adam_->seed(first_moment_, second_moment_);
    device_adam_->propose(parameters.values, objective.gradient.data(), options_.adam_beta1,
        options_.adam_beta2, adam_first_correction, adam_second_correction, options_.epsilon, candidate.values);
    proposal_live_ = true;
    std::clog << "WFTrain device Adam recurrence completed: parameters=" << parameter_count_
              << ", accepted_updates_before=" << accepted_update_count_
              << ", owner_device_bytes=" << device_adam_->deviceBytes()
              << ", host_mirror_bytes=" << sizeof(double) * (first_moment_.capacity() + second_moment_.capacity())
              << ", global_gradient=host_upload, old_parameters=host_upload, raw_candidate=host_download"
              << ", selection_and_publication=host, fully_resident=no\n";
    return candidate;
  }
#endif

  for (const BlockUpdate& block : blocks_)
    if (block.trainable)
      for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
           ++parameter)
      {
        const double gradient = objective.gradient[parameter];
        double direction      = gradient;
        switch (options_.method)
        {
        case FirstOrderMethod::SGD:
          break;
        case FirstOrderMethod::MOMENTUM_SGD:
          direction = options_.momentum_decay * first_moment_[parameter] + gradient;
          break;
        case FirstOrderMethod::RMSPROP:
        {
          const double next_second = options_.rms_decay * second_moment_[parameter] +
              (1.0 - options_.rms_decay) * gradient * gradient;
          if (!isFiniteDouble(next_second))
            throw std::overflow_error("RMSProp second moment overflow");
          direction = gradient / (std::sqrt(next_second) + options_.epsilon);
          break;
        }
        case FirstOrderMethod::ADAM:
        case FirstOrderMethod::AMSGRAD:
        {
          const double next_first =
              options_.adam_beta1 * first_moment_[parameter] +
              (1.0 - options_.adam_beta1) * gradient;
          const double next_second =
              options_.adam_beta2 * second_moment_[parameter] +
              (1.0 - options_.adam_beta2) * gradient * gradient;
          if (!isFiniteDouble(next_first) || !isFiniteDouble(next_second))
            throw std::overflow_error("Adam moment overflow");
          const double corrected_first  = next_first / adam_first_correction;
          const double denominator_second = options_.method == FirstOrderMethod::AMSGRAD
              ? std::max(maximum_second_moment_[parameter], next_second)
              : next_second;
          const double corrected_second = denominator_second / adam_second_correction;
          direction = corrected_first / (std::sqrt(corrected_second) + options_.epsilon);
          break;
        }
        }
        if (!isFiniteDouble(direction))
          throw std::overflow_error("First-order optimizer update direction is non-finite");
        candidate.values[parameter] -= block.learning_rate * direction;
        if (!isFiniteDouble(candidate.values[parameter]))
          throw std::overflow_error("First-order optimizer candidate parameter is non-finite");
      }

  proposal_live_ = true;
  return candidate;
}

void FirstOrderOptimizer::commitRecurrence(ParameterGradientView objective) noexcept
{
  for (const BlockUpdate& block : blocks_)
    if (block.trainable)
      for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
           ++parameter)
      {
        const double gradient = objective.gradient[parameter];
        switch (options_.method)
        {
        case FirstOrderMethod::SGD:
          break;
        case FirstOrderMethod::MOMENTUM_SGD:
          first_moment_[parameter] =
              options_.momentum_decay * first_moment_[parameter] + gradient;
          break;
        case FirstOrderMethod::RMSPROP:
          second_moment_[parameter] = options_.rms_decay * second_moment_[parameter] +
              (1.0 - options_.rms_decay) * gradient * gradient;
          break;
        case FirstOrderMethod::ADAM:
        case FirstOrderMethod::AMSGRAD:
          first_moment_[parameter] = options_.adam_beta1 * first_moment_[parameter] +
              (1.0 - options_.adam_beta1) * gradient;
          second_moment_[parameter] = options_.adam_beta2 * second_moment_[parameter] +
              (1.0 - options_.adam_beta2) * gradient * gradient;
          if (options_.method == FirstOrderMethod::AMSGRAD)
            maximum_second_moment_[parameter] =
                std::max(maximum_second_moment_[parameter], second_moment_[parameter]);
          break;
        }
      }
}

void FirstOrderOptimizer::proposalAccepted(const StructuredParameterSchema& schema,
                                           const StructuredParameterSnapshot& parameters,
                                           ParameterGradientView objective) noexcept
{
  assert(proposal_live_);
  assert(schema.fingerprint() == schema_fingerprint_);
  assert(parameters.version == objective.parameter_version);
#ifdef ENABLE_CUDA
  if (device_adam_)
  {
    device_adam_->accept();
    moment_mirrors_stale_ = true;
  }
  else
#endif
    commitRecurrence(objective);
  ++accepted_update_count_;
  proposal_live_ = false;
}

void FirstOrderOptimizer::proposalRejected() noexcept
{
#ifdef ENABLE_CUDA
  if (device_adam_)
    device_adam_->reject();
#endif
  proposal_live_ = false;
}

std::size_t FirstOrderOptimizer::retainedBytes() const noexcept
{
  const auto host = (first_moment_.capacity() + second_moment_.capacity() +
                    maximum_second_moment_.capacity()) * sizeof(double);
#ifdef ENABLE_CUDA
  return host + (device_adam_ ? device_adam_->deviceBytes() : 0);
#else
  return host;
#endif
}

std::string FirstOrderOptimizer::stateFingerprint(
    std::uint64_t update_count,
    const std::vector<double>& first_moment,
    const std::vector<double>& second_moment,
    const std::vector<double>& maximum_second_moment) const
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  extendFingerprint(hash, &update_count, sizeof(update_count));
  const std::uint64_t first_size = first_moment.size();
  const std::uint64_t second_size = second_moment.size();
  extendFingerprint(hash, &first_size, sizeof(first_size));
  if (!first_moment.empty())
    extendFingerprint(hash, first_moment.data(), first_moment.size() * sizeof(double));
  extendFingerprint(hash, &second_size, sizeof(second_size));
  if (!second_moment.empty())
    extendFingerprint(hash, second_moment.data(), second_moment.size() * sizeof(double));
  // Preserve the original byte stream for all existing optimizer checkpoints.
  if (options_.method == FirstOrderMethod::AMSGRAD)
  {
    const std::uint64_t maximum_size = maximum_second_moment.size();
    extendFingerprint(hash, &maximum_size, sizeof(maximum_size));
    if (!maximum_second_moment.empty())
      extendFingerprint(hash, maximum_second_moment.data(), maximum_second_moment.size() * sizeof(double));
  }
  return formatFingerprint(hash);
}

CheckpointContributorMetadata FirstOrderOptimizer::checkpointMetadata() const
{
  if (proposal_live_)
    throw std::logic_error("Cannot checkpoint a first-order optimizer with a live proposal");
  synchronizeMoments();
  return {CONTRIBUTOR_ID, {1, 0, 0}, configuration_fingerprint_,
          stateFingerprint(accepted_update_count_, first_moment_, second_moment_, maximum_second_moment_)};
}

void FirstOrderOptimizer::writeCheckpointPayload(hdf_archive& archive) const
{
  if (proposal_live_)
    throw std::logic_error("Cannot checkpoint a first-order optimizer with a live proposal");
  synchronizeMoments();
  archive.write(accepted_update_count_, "accepted_update_count");
  if (!first_moment_.empty())
    archive.write(first_moment_, "first_moment");
  if (!second_moment_.empty())
    archive.write(second_moment_, "second_moment");
  if (!maximum_second_moment_.empty())
    archive.write(maximum_second_moment_, "maximum_second_moment");
}

void FirstOrderOptimizer::validateRestoredState(
    std::uint64_t update_count,
    const std::vector<double>& first_moment,
    const std::vector<double>& second_moment,
    const std::vector<double>& maximum_second_moment) const
{
  const std::size_t expected_first =
      options_.method == FirstOrderMethod::MOMENTUM_SGD ||
          options_.method == FirstOrderMethod::ADAM || options_.method == FirstOrderMethod::AMSGRAD
      ? parameter_count_
      : 0;
  const std::size_t expected_second =
      options_.method == FirstOrderMethod::RMSPROP || options_.method == FirstOrderMethod::ADAM ||
          options_.method == FirstOrderMethod::AMSGRAD
      ? parameter_count_
      : 0;
  const std::size_t expected_maximum =
      options_.method == FirstOrderMethod::AMSGRAD ? parameter_count_ : 0;
  if (first_moment.size() != expected_first || second_moment.size() != expected_second ||
      maximum_second_moment.size() != expected_maximum)
    throw std::runtime_error("First-order optimizer checkpoint moment shape mismatch");
  if (update_count == std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("First-order optimizer checkpoint update count is exhausted");
  if (!std::all_of(first_moment.begin(), first_moment.end(),
                   [](double value) { return isFiniteDouble(value); }) ||
      !std::all_of(second_moment.begin(), second_moment.end(),
                   [](double value) { return isFiniteDouble(value) && value >= 0.0; }) ||
      !std::all_of(maximum_second_moment.begin(), maximum_second_moment.end(),
                   [](double value) { return isFiniteDouble(value) && value >= 0.0; }))
    throw std::runtime_error("First-order optimizer checkpoint moments are invalid");
  for (std::size_t parameter = 0; parameter < maximum_second_moment.size(); ++parameter)
    if (maximum_second_moment[parameter] < second_moment[parameter])
      throw std::runtime_error("AMSGrad checkpoint maximum second moment is below its EMA");

  // Frozen entries are never part of a recurrence and must stay canonical zero.
  for (const BlockUpdate& block : blocks_)
    if (!block.trainable)
      for (std::size_t parameter = block.offset; parameter < block.offset + block.count;
           ++parameter)
      {
        if (!first_moment.empty() && first_moment[parameter] != 0.0)
          throw std::runtime_error(
              "First-order optimizer checkpoint has nonzero frozen first moment");
        if (!second_moment.empty() && second_moment[parameter] != 0.0)
          throw std::runtime_error(
              "First-order optimizer checkpoint has nonzero frozen second moment");
        if (!maximum_second_moment.empty() && maximum_second_moment[parameter] != 0.0)
          throw std::runtime_error(
              "AMSGrad checkpoint has nonzero frozen maximum second moment");
      }
}

std::unique_ptr<OptimizerCheckpointContributor::PreparedRestore>
FirstOrderOptimizer::prepareCheckpointRestore(
    hdf_archive& archive,
    const CheckpointContributorMetadata& saved_metadata)
{
  if (proposal_live_)
    throw std::logic_error("Cannot restore a first-order optimizer with a live proposal");
  if (saved_metadata.contributor_id != CONTRIBUTOR_ID ||
      saved_metadata.format_version[0] != 1 ||
      saved_metadata.configuration_fingerprint != configuration_fingerprint_)
    throw std::runtime_error("First-order optimizer checkpoint metadata mismatch");

  std::uint64_t update_count = 0;
  std::vector<double> first_moment;
  std::vector<double> second_moment;
  std::vector<double> maximum_second_moment;
  archive.read(update_count, "accepted_update_count");
  if (!first_moment_.empty())
    archive.read(first_moment, "first_moment");
  if (!second_moment_.empty())
    archive.read(second_moment, "second_moment");
  if (!maximum_second_moment_.empty())
    archive.read(maximum_second_moment, "maximum_second_moment");
  validateRestoredState(update_count, first_moment, second_moment, maximum_second_moment);
  if (saved_metadata.state_fingerprint !=
      stateFingerprint(update_count, first_moment, second_moment, maximum_second_moment))
    throw std::runtime_error("First-order optimizer checkpoint state fingerprint mismatch");

  return std::make_unique<PreparedRestoreAction>(
      *this, update_count, std::move(first_moment), std::move(second_moment),
      std::move(maximum_second_moment));
}

} // namespace qmcplusplus::wftrain
