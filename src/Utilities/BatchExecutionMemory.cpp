//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#include "BatchExecutionMemory.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace qmcplusplus
{
namespace
{

constexpr std::array<BatchExecutionMode, 4> TUNABLE_MODES{BatchExecutionMode::VALUE, BatchExecutionMode::FULL_VGL,
                                                          BatchExecutionMode::ACTIVE_GRADIENT,
                                                          BatchExecutionMode::ECP_OUTER};

[[noreturn]] void throwBatchMemoryOverflow(std::string_view context,
                                           std::string_view suffix = {})
{
  std::string message("Batch memory size overflow in ");
  if (!context.empty())
    message.append(context.data(), context.size());
  if (!suffix.empty())
    message.append(suffix.data(), suffix.size());
  throw std::overflow_error(message);
}

/** Return the capacity associated with one tunable operation family. */
std::size_t getCapacity(const BatchTileCapacities& capacities, BatchExecutionMode mode)
{
  switch (mode)
  {
  case BatchExecutionMode::VALUE:
    return capacities.value;
  case BatchExecutionMode::FULL_VGL:
    return capacities.full_vgl;
  case BatchExecutionMode::ACTIVE_GRADIENT:
    return capacities.active_gradient;
  case BatchExecutionMode::ECP_OUTER:
    return capacities.ecp_outer;
  default:
    throw std::logic_error("Operation family has no tunable batch capacity");
  }
}

/** Set the capacity associated with one tunable operation family. */
void setCapacity(BatchTileCapacities& capacities, BatchExecutionMode mode, std::size_t capacity)
{
  switch (mode)
  {
  case BatchExecutionMode::VALUE:
    capacities.value = capacity;
    return;
  case BatchExecutionMode::FULL_VGL:
    capacities.full_vgl = capacity;
    return;
  case BatchExecutionMode::ACTIVE_GRADIENT:
    capacities.active_gradient = capacity;
    return;
  case BatchExecutionMode::ECP_OUTER:
    capacities.ecp_outer = capacity;
    return;
  default:
    throw std::logic_error("Operation family has no tunable batch capacity");
  }
}

/** Return the request associated with one tunable operation family. */
const BatchTileRequest& getRequest(const BatchTileRequests& requests, BatchExecutionMode mode)
{
  switch (mode)
  {
  case BatchExecutionMode::VALUE:
    return requests.value;
  case BatchExecutionMode::FULL_VGL:
    return requests.full_vgl;
  case BatchExecutionMode::ACTIVE_GRADIENT:
    return requests.active_gradient;
  case BatchExecutionMode::ECP_OUTER:
    return requests.ecp_outer;
  default:
    throw std::logic_error("Operation family has no tunable batch request");
  }
}

/** Format a byte pair for setup-time policy errors. */
std::string formatBytes(BatchMemoryBytes bytes)
{
  std::ostringstream message;
  message << "host=" << bytes.host << "B, device=" << bytes.device << 'B';
  return message.str();
}

/** Check whether both independently optional hard budgets admit an estimate. */
bool fitsBudget(const BatchMemoryPolicy& policy, BatchMemoryBytes bytes)
{
  return (!policy.host_budget || bytes.host <= *policy.host_budget) &&
      (!policy.device_budget || bytes.device <= *policy.device_budget);
}

/** Compute an exact nonnegative difference and reject a nonmonotone estimator. */
BatchMemoryBytes checkedSaving(BatchMemoryBytes before, BatchMemoryBytes after)
{
  if (after.host > before.host || after.device > before.device)
    throw std::logic_error("Batch memory estimator increased when a tile capacity was reduced");
  return {before.host - after.host, before.device - after.device};
}

/** Exact two-word sum used to compare host-plus-device relief without overflow. */
struct ReliefScore
{
  bool carry      = false;
  std::size_t low = 0;

  bool isZero() const noexcept { return !carry && low == 0; }
};

/** Compare exact two-word relief scores. */
bool operator>(const ReliefScore& lhs, const ReliefScore& rhs) noexcept
{
  return lhs.carry != rhs.carry ? lhs.carry : lhs.low > rhs.low;
}

/** Score only savings that relieve a currently exceeded hard budget. */
ReliefScore savingScore(const BatchMemoryPolicy& policy, BatchMemoryBytes current, BatchMemoryBytes saving)
{
  const std::size_t host_relief = policy.host_budget && current.host > *policy.host_budget ? saving.host : 0;
  const std::size_t device_relief =
      policy.device_budget && current.device > *policy.device_budget ? saving.device : 0;
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (device_relief > maximum - host_relief)
    return {true, device_relief - (maximum - host_relief) - 1};
  return {false, host_relief + device_relief};
}

/** Mix one byte into a deterministic FNV-1a plan fingerprint. */
void mixByte(std::uint64_t& hash, std::uint8_t byte) noexcept
{
  hash ^= byte;
  hash *= 1099511628211ULL;
}

/** Mix one fixed-width integer without depending on host byte order. */
void mixInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned shift = 0; shift < 64; shift += 8)
    mixByte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
}

/** Mix a length-delimited string into a deterministic plan fingerprint. */
void mixString(std::uint64_t& hash, const std::string& value) noexcept
{
  mixInteger(hash, value.size());
  for (const unsigned char character : value)
    mixByte(hash, character);
}

/** Mix one optional size, distinguishing omission from an explicit zero. */
void mixOptional(std::uint64_t& hash, const std::optional<std::size_t>& value) noexcept
{
  mixByte(hash, value.has_value());
  if (value)
    mixInteger(hash, *value);
}

/** Mix all four capacities in their documented stable order. */
void mixCapacities(std::uint64_t& hash, const BatchTileCapacities& capacities) noexcept
{
  mixInteger(hash, capacities.value);
  mixInteger(hash, capacities.full_vgl);
  mixInteger(hash, capacities.active_gradient);
  mixInteger(hash, capacities.ecp_outer);
}

/** Mix one automatic/fixed request into the plan fingerprint. */
void mixRequest(std::uint64_t& hash, const BatchTileRequest& request) noexcept
{
  mixByte(hash, request.isAutomatic());
  if (!request.isAutomatic())
    mixInteger(hash, request.fixedCapacity());
}

/** Mix every category of one estimate in stable host/device order. */
void mixEstimate(std::uint64_t& hash, const BatchMemoryEstimate& estimate) noexcept
{
  for (std::size_t category = 0; category < static_cast<std::size_t>(BatchMemoryCategory::COUNT); ++category)
  {
    const BatchMemoryBytes bytes = estimate.at(static_cast<BatchMemoryCategory>(category));
    mixInteger(hash, bytes.host);
    mixInteger(hash, bytes.device);
  }
}

/** Produce a stable exact-content fingerprint for an immutable selected plan. */
std::uint64_t makeFingerprint(const BatchExecutionSelectionInput& input,
                              const BatchTileCapacities& minimum,
                              const BatchTileCapacities& selected,
                              const BatchMemoryEstimate& minimum_estimate,
                              const BatchMemoryEstimate& estimate,
                              const std::vector<BatchMemoryParticipantEvidence>& participant_evidence)
{
  std::uint64_t hash = 14695981039346656037ULL;
  mixString(hash, input.schema_id);
  mixString(hash, input.preference.id);
  mixInteger(hash, input.requirements.mask());
  mixOptional(hash, input.policy.host_budget);
  mixOptional(hash, input.policy.device_budget);
  mixRequest(hash, input.policy.tiles.value);
  mixRequest(hash, input.policy.tiles.full_vgl);
  mixRequest(hash, input.policy.tiles.active_gradient);
  mixRequest(hash, input.policy.tiles.ecp_outer);
  mixCapacities(hash, input.logical_maximum);
  mixCapacities(hash, input.preference.preferred);
  mixCapacities(hash, minimum);
  mixCapacities(hash, selected);
  mixInteger(hash, input.particle_count);
  mixInteger(hash, input.active_parameter_count);
  mixInteger(hash, input.parameter_derivative_width);
  mixByte(hash, static_cast<std::uint8_t>(input.target_coordinate));

  mixByte(hash, input.topology.serialized_walkers);
  mixString(hash, input.topology.run_kind);
  mixString(hash, input.topology.backend_id);
  mixOptional(hash, input.topology.device_id);
  mixInteger(hash, input.topology.initial_walkers_per_crowd.size());
  for (const std::size_t walkers : input.topology.initial_walkers_per_crowd)
    mixInteger(hash, walkers);
  mixInteger(hash, input.topology.reserve_walkers_per_crowd.size());
  for (const std::size_t walkers : input.topology.reserve_walkers_per_crowd)
    mixInteger(hash, walkers);

  mixInteger(hash, participant_evidence.size());
  for (const BatchMemoryParticipantEvidence& evidence : participant_evidence)
  {
    mixString(hash, evidence.participant_id);
    mixCapacities(hash, evidence.logical_maximum);
    mixInteger(hash, evidence.owner_multiplicity);
    mixByte(hash, evidence.fully_accounted);
    mixEstimate(hash, evidence.fixed_minimum_per_owner);
    mixEstimate(hash, evidence.selected_per_owner);
  }
  mixEstimate(hash, minimum_estimate);
  mixEstimate(hash, estimate);
  return hash;
}

/** Validate selection-wide fields before invoking any owner estimator. */
void validateSelectionInput(const BatchExecutionSelectionInput& input)
{
  if (input.schema_id.empty())
    throw std::invalid_argument("Batch execution memory schema ID must not be empty");
  if (input.preference.id.empty())
    throw std::invalid_argument("Batch execution memory preference ID must not be empty");
  switch (input.target_coordinate)
  {
  case BatchExecutionTargetCoordinate::UNKNOWN:
  case BatchExecutionTargetCoordinate::POS_ONLY:
  case BatchExecutionTargetCoordinate::POS_SPIN:
    break;
  default:
    throw std::invalid_argument("Batch execution target coordinate capability is invalid");
  }
  validateBatchExecutionTopology(input.topology);
}

/** Return whether every participant maximum fits within the driver envelope. */
bool capacitiesFitWithin(const BatchTileCapacities& capacities, const BatchTileCapacities& envelope) noexcept
{
  return capacities.value <= envelope.value && capacities.full_vgl <= envelope.full_vgl &&
      capacities.active_gradient <= envelope.active_gradient && capacities.ecp_outer <= envelope.ecp_outer;
}

/** Check invariant participant metadata against the first candidate evaluation. */
void validateContributionInvariants(
    const std::vector<BatchMemoryParticipantContribution>& contributions,
    const std::vector<BatchMemoryParticipantContribution>& reference)
{
  if (contributions.size() != reference.size())
    throw std::logic_error("Batch memory participant count changed across candidate estimates");

  for (std::size_t index = 0; index < contributions.size(); ++index)
  {
    const BatchMemoryParticipantContribution& contribution = contributions[index];
    const BatchMemoryParticipantContribution& expected     = reference[index];
    if (contribution.participant_id != expected.participant_id)
      throw std::logic_error("Batch memory participant identity or order changed across candidate estimates");
    if (!(contribution.contribution.logical_maximum == expected.contribution.logical_maximum))
      throw std::logic_error("Batch memory participant logical maximum changed across candidate estimates: " +
                             contribution.participant_id);
    if (contribution.contribution.owner_multiplicity != expected.contribution.owner_multiplicity)
      throw std::logic_error("Batch memory participant owner multiplicity changed across candidate estimates: " +
                             contribution.participant_id);
    if (contribution.contribution.fully_accounted != expected.contribution.fully_accounted)
      throw std::logic_error("Batch memory participant accounting status changed across candidate estimates: " +
                             contribution.participant_id);
  }
}

/** One provider evaluation paired with its checked rank-local aggregate. */
struct ContributionEvaluation
{
  std::vector<BatchMemoryParticipantContribution> contributions;
  BatchMemoryEstimate aggregate;
};

/** Resolve one request against whether the mode is reachable and its logical maximum. */
std::size_t initialCapacity(const BatchTileRequest& request,
                            std::size_t preference,
                            std::size_t logical_maximum,
                            bool required)
{
  if (!required)
    return 0;
  if (logical_maximum == 0)
    throw std::invalid_argument("A required batch execution mode has a zero logical maximum");
  const std::size_t requested = request.isAutomatic() ? preference : request.fixedCapacity();
  return request.isAutomatic() ? std::max(std::size_t{1}, std::min(requested, logical_maximum))
                               : std::min(requested, logical_maximum);
}

} // namespace

void BatchExecutionRequirements::require(BatchExecutionMode mode) noexcept
{
  mask_ |= static_cast<std::uint32_t>(mode);
}

bool BatchExecutionRequirements::requires(BatchExecutionMode mode) const noexcept
{
  return (mask_ & static_cast<std::uint32_t>(mode)) != 0;
}

bool batchExecutionModeIsRequired(const BatchExecutionRequirements& requirements,
                                  BatchExecutionMode mode) noexcept
{
  // Scalar compatibility shares the ordinary VALUE tile, while all flattened
  // nonlocal-ECP products share the outer replacement tile.
  if (mode == BatchExecutionMode::VALUE)
    return requirements.requires(BatchExecutionMode::VALUE) ||
        requirements.requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  if (mode == BatchExecutionMode::ECP_OUTER)
    return requirements.requires(BatchExecutionMode::ECP_OUTER) ||
        requirements.requires(BatchExecutionMode::ECP_WEIGHTED_SCORE) ||
        requirements.requires(BatchExecutionMode::ECP_TMOVE_CANDIDATES) ||
        requirements.requires(BatchExecutionMode::ECP_LISTENER_OUTPUT);
  return requirements.requires(mode);
}

BatchTileRequest BatchTileRequest::fixed(std::size_t capacity)
{
  if (capacity == 0)
    throw std::invalid_argument("A fixed batch tile capacity must be positive");
  return BatchTileRequest(false, capacity);
}

std::size_t BatchTileRequest::fixedCapacity() const
{
  if (automatic_)
    throw std::logic_error("An automatic batch tile request has no fixed capacity");
  return capacity_;
}

std::size_t BatchExecutionTopology::initialWalkerCount() const
{
  std::size_t total = 0;
  for (const std::size_t walkers : initial_walkers_per_crowd)
    total = checkedBatchMemoryAdd(total, walkers, "initial walker count");
  return total;
}

std::size_t BatchExecutionTopology::reserveWalkerCount() const
{
  const std::vector<std::size_t>& topology =
      reserve_walkers_per_crowd.empty() ? initial_walkers_per_crowd : reserve_walkers_per_crowd;
  std::size_t total = 0;
  for (const std::size_t walkers : topology)
    total = checkedBatchMemoryAdd(total, walkers, "reserve walker count");
  return total;
}

void validateBatchExecutionTopology(const BatchExecutionTopology& topology)
{
  if (!topology.reserve_walkers_per_crowd.empty() &&
      topology.reserve_walkers_per_crowd.size() != topology.initial_walkers_per_crowd.size())
    throw std::invalid_argument("Initial and reserve batch crowd topologies must have the same number of crowds");

  const std::size_t initial_walkers = topology.initialWalkerCount();
  const std::size_t reserve_walkers = topology.reserveWalkerCount();
  if (!topology.reserve_walkers_per_crowd.empty() && reserve_walkers < initial_walkers)
    throw std::invalid_argument("The rank reserve walker envelope is smaller than the initial population");
}

void BatchMemoryEstimate::add(BatchMemoryCategory category,
                              BatchMemoryBytes bytes,
                              std::string_view context)
{
  const std::size_t index = static_cast<std::size_t>(category);
  if (index >= categories_.size())
    throw std::out_of_range("Invalid batch memory category");
  const std::string_view checked_context =
      context.empty() ? std::string_view{"category bytes"} : context;
  categories_[index] =
      checkedBatchMemoryAdd(categories_[index], bytes, checked_context);
}

void BatchMemoryEstimate::add(const BatchMemoryEstimate& other,
                              std::string_view context)
{
  for (std::size_t category = 0; category < categories_.size(); ++category)
    add(static_cast<BatchMemoryCategory>(category), other.categories_[category], context);
}

const BatchMemoryBytes& BatchMemoryEstimate::at(BatchMemoryCategory category) const
{
  const std::size_t index = static_cast<std::size_t>(category);
  if (index >= categories_.size())
    throw std::out_of_range("Invalid batch memory category");
  return categories_[index];
}

BatchMemoryBytes BatchMemoryEstimate::total(std::string_view context) const
{
  BatchMemoryBytes total;
  const std::string_view checked_context =
      context.empty() ? std::string_view{"total bytes"} : context;
  for (const BatchMemoryBytes bytes : categories_)
    total = checkedBatchMemoryAdd(total, bytes, checked_context);
  return total;
}

std::size_t checkedBatchMemoryAdd(std::size_t lhs, std::size_t rhs,
                                  std::string_view context)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throwBatchMemoryOverflow(context);
  return lhs + rhs;
}

std::size_t checkedBatchMemoryMultiply(std::size_t lhs, std::size_t rhs,
                                       std::string_view context)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throwBatchMemoryOverflow(context);
  return lhs * rhs;
}

BatchMemoryBytes checkedBatchMemoryAdd(BatchMemoryBytes lhs,
                                       BatchMemoryBytes rhs,
                                       std::string_view context)
{
  if (rhs.host > std::numeric_limits<std::size_t>::max() - lhs.host)
    throwBatchMemoryOverflow(context, " host");
  if (rhs.device > std::numeric_limits<std::size_t>::max() - lhs.device)
    throwBatchMemoryOverflow(context, " device");
  return {lhs.host + rhs.host, lhs.device + rhs.device};
}

BatchMemoryBytes checkedBatchMemoryMultiply(BatchMemoryBytes bytes,
                                            std::size_t multiplicity,
                                            std::string_view context)
{
  if (bytes.host != 0 &&
      multiplicity > std::numeric_limits<std::size_t>::max() / bytes.host)
    throwBatchMemoryOverflow(context, " host");
  if (bytes.device != 0 &&
      multiplicity > std::numeric_limits<std::size_t>::max() / bytes.device)
    throwBatchMemoryOverflow(context, " device");
  return {bytes.host * multiplicity, bytes.device * multiplicity};
}

void includeBatchExecutionLogicalMaximum(BatchTileCapacities& aggregate,
                                         const BatchTileCapacities& participant) noexcept
{
  aggregate.value           = std::max(aggregate.value, participant.value);
  aggregate.full_vgl        = std::max(aggregate.full_vgl, participant.full_vgl);
  aggregate.active_gradient = std::max(aggregate.active_gradient, participant.active_gradient);
  aggregate.ecp_outer       = std::max(aggregate.ecp_outer, participant.ecp_outer);
}

std::string escapeBatchParticipantIdSegment(std::string_view segment)
{
  constexpr char HEX_DIGITS[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(segment.size());
  for (const unsigned char character : segment)
  {
    const bool unreserved = (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') ||
        character == '-' || character == '.' || character == '_' || character == '~';
    if (unreserved)
      escaped.push_back(static_cast<char>(character));
    else
    {
      escaped.push_back('%');
      escaped.push_back(HEX_DIGITS[character >> 4]);
      escaped.push_back(HEX_DIGITS[character & 0x0fU]);
    }
  }
  return escaped;
}

BatchMemoryEstimate aggregateBatchMemoryContributions(
    const std::vector<BatchMemoryParticipantContribution>& participants)
{
  BatchMemoryEstimate aggregate;
  std::unordered_set<std::string> participant_ids;
  for (const BatchMemoryParticipantContribution& participant : participants)
  {
    if (participant.participant_id.empty())
      throw std::invalid_argument("Batch memory participant ID must not be empty");
    if (!participant_ids.insert(participant.participant_id).second)
      throw std::invalid_argument("Duplicate batch memory participant ID: " + participant.participant_id);
    if (!participant.contribution.fully_accounted)
      throw std::invalid_argument("Batch memory participant is not fully accounted: " + participant.participant_id);
    if (participant.contribution.owner_multiplicity == 0)
      continue;

    for (std::size_t category = 0; category < static_cast<std::size_t>(BatchMemoryCategory::COUNT); ++category)
    {
      const BatchMemoryBytes bytes = checkedBatchMemoryMultiply(
          participant.contribution.per_owner.at(static_cast<BatchMemoryCategory>(category)),
          participant.contribution.owner_multiplicity, participant.participant_id);
      aggregate.add(static_cast<BatchMemoryCategory>(category), bytes, participant.participant_id);
    }
  }
  return aggregate;
}

const BatchExecutionPlan& BatchExecutionParticipantPlan::plan() const
{
  if (!plan_)
    throw std::logic_error("An empty batch execution participant view has no plan");
  return *plan_;
}

const BatchMemoryParticipantEvidence& BatchExecutionParticipantPlan::evidence() const
{
  if (!plan_)
    throw std::logic_error("An empty batch execution participant view has no evidence");
  const std::vector<BatchMemoryParticipantEvidence>& all_evidence = plan_->participantEvidence();
  if (participant_index_ >= all_evidence.size())
    throw std::logic_error("Batch execution participant view has an invalid evidence index");
  return all_evidence[participant_index_];
}

bool BatchExecutionParticipantPlan::sameBinding(const BatchExecutionParticipantPlan& other) const noexcept
{
  if (!plan_ || !other.plan_)
    return plan_.get() == other.plan_.get();
  return plan_.get() == other.plan_.get() && participant_index_ == other.participant_index_;
}

BatchExecutionParticipantPlan makeBatchExecutionParticipantPlan(
    std::shared_ptr<const BatchExecutionPlan> plan,
    std::string_view participant_id)
{
  // A null plan is the explicit no-policy binding. The participant identity is
  // intentionally ignored because no evidence exists to resolve it against.
  if (!plan)
    return {};
  if (participant_id.empty())
    throw std::invalid_argument("Batch memory participant ID must not be empty");

  const std::vector<BatchMemoryParticipantEvidence>& evidence = plan->participantEvidence();
  const auto match = std::find_if(evidence.begin(), evidence.end(), [participant_id](const auto& item) {
    return item.participant_id == participant_id;
  });
  if (match == evidence.end())
    throw std::invalid_argument("Batch execution plan has no evidence for participant: " +
                                std::string(participant_id));
  const std::size_t participant_index = static_cast<std::size_t>(std::distance(evidence.begin(), match));
  return BatchExecutionParticipantPlan(std::move(plan), participant_index);
}

BatchExecutionPlan selectBatchExecutionPlan(const BatchExecutionSelectionInput& input,
                                            const BatchMemoryContributionProvider& provider)
{
  validateSelectionInput(input);
  if (!provider)
    throw std::invalid_argument("Batch execution memory selection requires a contribution provider");

  BatchTileCapacities selected;
  for (const BatchExecutionMode mode : TUNABLE_MODES)
  {
    const bool required = batchExecutionModeIsRequired(input.requirements, mode);
    setCapacity(selected, mode,
                initialCapacity(getRequest(input.policy.tiles, mode), getCapacity(input.preference.preferred, mode),
                                getCapacity(input.logical_maximum, mode), required));
  }

  BatchTileCapacities minimum = selected;
  for (const BatchExecutionMode mode : TUNABLE_MODES)
    if (batchExecutionModeIsRequired(input.requirements, mode) &&
        getRequest(input.policy.tiles, mode).isAutomatic())
      setCapacity(minimum, mode, 1);

  std::vector<BatchMemoryParticipantContribution> reference_contributions;
  bool have_reference = false;
  auto evaluate = [&](const BatchTileCapacities& capacities) {
    BatchExecutionPlanningContext context{input.requirements,
                                          input.topology,
                                          input.logical_maximum,
                                          capacities,
                                          input.particle_count,
                                          input.active_parameter_count,
                                          input.parameter_derivative_width,
                                          input.target_coordinate};
    ContributionEvaluation evaluation;
    evaluation.contributions = provider(context);
    evaluation.aggregate     = aggregateBatchMemoryContributions(evaluation.contributions);

    for (const BatchMemoryParticipantContribution& participant : evaluation.contributions)
      if (!capacitiesFitWithin(participant.contribution.logical_maximum, input.logical_maximum))
        throw std::invalid_argument("Batch memory participant logical maximum exceeds the selection envelope: " +
                                    participant.participant_id);

    if (have_reference)
      validateContributionInvariants(evaluation.contributions, reference_contributions);
    else
    {
      reference_contributions = evaluation.contributions;
      have_reference          = true;
    }
    return evaluation;
  };

  ContributionEvaluation minimum_evaluation = evaluate(minimum);
  const BatchMemoryBytes minimum_bytes       = minimum_evaluation.aggregate.total("fixed minimum");
  if (!fitsBudget(input.policy, minimum_bytes))
  {
    std::ostringstream message;
    message << "Batch execution memory minimum does not fit the configured budget (" << formatBytes(minimum_bytes)
            << ")";
    throw std::runtime_error(message.str());
  }

  ContributionEvaluation selected_evaluation =
      selected == minimum ? minimum_evaluation : evaluate(selected);
  while (!fitsBudget(input.policy, selected_evaluation.aggregate.total("selected plan")))
  {
    const BatchMemoryBytes current_bytes = selected_evaluation.aggregate.total("selected plan");
    bool found_candidate                 = false;
    ReliefScore best_saving;
    BatchTileCapacities best_capacities;
    ContributionEvaluation best_evaluation;

    // Iteration order is the documented stable tie break: VALUE, FULL_VGL,
    // ACTIVE_GRADIENT, then ECP_OUTER.
    for (const BatchExecutionMode mode : TUNABLE_MODES)
    {
      if (!getRequest(input.policy.tiles, mode).isAutomatic() || getCapacity(selected, mode) <= 1)
        continue;

      BatchTileCapacities candidate = selected;
      setCapacity(candidate, mode, getCapacity(candidate, mode) - 1);
      ContributionEvaluation candidate_evaluation = evaluate(candidate);
      const BatchMemoryBytes saving =
          checkedSaving(current_bytes, candidate_evaluation.aggregate.total("candidate plan"));
      const ReliefScore score        = savingScore(input.policy, current_bytes, saving);
      if (!found_candidate || score > best_saving)
      {
        found_candidate = true;
        best_saving      = score;
        best_capacities  = candidate;
        best_evaluation  = std::move(candidate_evaluation);
      }
    }

    if (!found_candidate || best_saving.isZero())
    {
      std::ostringstream message;
      message << "Batch execution memory budget cannot be met without reducing a hard tile request ("
              << formatBytes(current_bytes) << ")";
      throw std::runtime_error(message.str());
    }
    selected            = best_capacities;
    selected_evaluation = std::move(best_evaluation);
  }

  std::vector<BatchMemoryParticipantEvidence> participant_evidence;
  participant_evidence.reserve(minimum_evaluation.contributions.size());
  for (std::size_t index = 0; index < minimum_evaluation.contributions.size(); ++index)
  {
    const BatchMemoryParticipantContribution& minimum_contribution  = minimum_evaluation.contributions[index];
    const BatchMemoryParticipantContribution& selected_contribution = selected_evaluation.contributions[index];
    participant_evidence.push_back(
        {minimum_contribution.participant_id, minimum_contribution.contribution.logical_maximum,
         minimum_contribution.contribution.owner_multiplicity, minimum_contribution.contribution.fully_accounted,
         minimum_contribution.contribution.per_owner, selected_contribution.contribution.per_owner});
  }

  BatchExecutionPlan plan;
  plan.schema_id_              = input.schema_id;
  plan.preference_id_          = input.preference.id;
  plan.requirements_           = input.requirements;
  plan.topology_               = input.topology;
  plan.policy_                 = input.policy;
  plan.logical_maximum_        = input.logical_maximum;
  plan.minimum_capacities_     = minimum;
  plan.selected_capacities_    = selected;
  plan.fixed_minimum_estimate_ = minimum_evaluation.aggregate;
  plan.selected_estimate_      = selected_evaluation.aggregate;
  plan.particle_count_             = input.particle_count;
  plan.active_parameter_count_     = input.active_parameter_count;
  plan.parameter_derivative_width_ = input.parameter_derivative_width;
  plan.target_coordinate_          = input.target_coordinate;
  plan.participant_evidence_        = std::move(participant_evidence);
  plan.fingerprint_ =
      makeFingerprint(input, minimum, selected, plan.fixed_minimum_estimate_,
                      plan.selected_estimate_, plan.participant_evidence_);
  return plan;
}

} // namespace qmcplusplus
