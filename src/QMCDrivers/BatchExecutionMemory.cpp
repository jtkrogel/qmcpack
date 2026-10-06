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

/** ECP weighted-score execution uses the same outer tile as ECP value execution. */
bool modeIsRequired(const BatchExecutionRequirements& requirements, BatchExecutionMode mode)
{
  if (mode == BatchExecutionMode::ECP_OUTER)
    return requirements.requires(BatchExecutionMode::ECP_OUTER) ||
        requirements.requires(BatchExecutionMode::ECP_WEIGHTED_SCORE);
  return requirements.requires(mode);
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

/** Score only savings that relieve a currently exceeded hard budget. */
std::size_t savingScore(const BatchMemoryPolicy& policy, BatchMemoryBytes current, BatchMemoryBytes saving)
{
  const std::size_t host_relief = policy.host_budget && current.host > *policy.host_budget ? saving.host : 0;
  const std::size_t device_relief =
      policy.device_budget && current.device > *policy.device_budget ? saving.device : 0;
  return checkedBatchMemoryAdd(host_relief, device_relief, "cross-space candidate relief");
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

/** Produce a stable exact-content fingerprint for an immutable selected plan. */
std::uint64_t makeFingerprint(const BatchExecutionSelectionInput& input,
                              const BatchTileCapacities& selected,
                              const BatchMemoryEstimate& estimate)
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
  mixCapacities(hash, selected);

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

  mixInteger(hash, input.participant_ids.size());
  for (const std::string& participant_id : input.participant_ids)
    mixString(hash, participant_id);
  for (std::size_t category = 0; category < static_cast<std::size_t>(BatchMemoryCategory::COUNT); ++category)
  {
    const BatchMemoryBytes bytes = estimate.at(static_cast<BatchMemoryCategory>(category));
    mixInteger(hash, bytes.host);
    mixInteger(hash, bytes.device);
  }
  return hash;
}

/** Validate topology and structural identifiers before invoking any owner estimator. */
void validateSelectionInput(const BatchExecutionSelectionInput& input)
{
  if (input.schema_id.empty())
    throw std::invalid_argument("Batch execution memory schema ID must not be empty");
  if (input.preference.id.empty())
    throw std::invalid_argument("Batch execution memory preference ID must not be empty");
  if (!input.topology.reserve_walkers_per_crowd.empty() &&
      input.topology.reserve_walkers_per_crowd.size() != input.topology.initial_walkers_per_crowd.size())
    throw std::invalid_argument("Initial and reserve batch crowd topologies must have the same number of crowds");

  // Checked totals validate even topologies whose sum is not otherwise needed by this selection boundary.
  input.topology.initialWalkerCount();
  input.topology.reserveWalkerCount();

  std::unordered_set<std::string> participant_ids;
  for (const std::string& participant_id : input.participant_ids)
  {
    if (participant_id.empty())
      throw std::invalid_argument("Batch memory participant ID must not be empty");
    if (!participant_ids.insert(participant_id).second)
      throw std::invalid_argument("Duplicate batch memory participant ID: " + participant_id);
  }
}

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

void BatchMemoryEstimate::add(BatchMemoryCategory category,
                              BatchMemoryBytes bytes,
                              const std::string& context)
{
  const std::size_t index = static_cast<std::size_t>(category);
  if (index >= categories_.size())
    throw std::out_of_range("Invalid batch memory category");
  categories_[index] = checkedBatchMemoryAdd(categories_[index], bytes, context.empty() ? "category bytes" : context);
}

void BatchMemoryEstimate::add(const BatchMemoryEstimate& other, const std::string& context)
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

BatchMemoryBytes BatchMemoryEstimate::total(const std::string& context) const
{
  BatchMemoryBytes total;
  for (const BatchMemoryBytes bytes : categories_)
    total = checkedBatchMemoryAdd(total, bytes, context.empty() ? "total bytes" : context);
  return total;
}

std::size_t checkedBatchMemoryAdd(std::size_t lhs, std::size_t rhs, const std::string& context)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error("Batch memory size overflow in " + context);
  return lhs + rhs;
}

std::size_t checkedBatchMemoryMultiply(std::size_t lhs, std::size_t rhs, const std::string& context)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error("Batch memory size overflow in " + context);
  return lhs * rhs;
}

BatchMemoryBytes checkedBatchMemoryAdd(BatchMemoryBytes lhs,
                                       BatchMemoryBytes rhs,
                                       const std::string& context)
{
  return {checkedBatchMemoryAdd(lhs.host, rhs.host, context + " host"),
          checkedBatchMemoryAdd(lhs.device, rhs.device, context + " device")};
}

BatchMemoryBytes checkedBatchMemoryMultiply(BatchMemoryBytes bytes,
                                            std::size_t multiplicity,
                                            const std::string& context)
{
  return {checkedBatchMemoryMultiply(bytes.host, multiplicity, context + " host"),
          checkedBatchMemoryMultiply(bytes.device, multiplicity, context + " device")};
}

BatchMemoryEstimate aggregateBatchMemoryEstimates(const std::vector<BatchMemoryParticipantEstimate>& participants)
{
  BatchMemoryEstimate aggregate;
  std::unordered_set<std::string> participant_ids;
  for (const BatchMemoryParticipantEstimate& participant : participants)
  {
    if (participant.participant_id.empty())
      throw std::invalid_argument("Batch memory participant ID must not be empty");
    if (!participant_ids.insert(participant.participant_id).second)
      throw std::invalid_argument("Duplicate batch memory participant ID: " + participant.participant_id);
    if (participant.owner_multiplicity == 0)
      continue;

    for (std::size_t category = 0; category < static_cast<std::size_t>(BatchMemoryCategory::COUNT); ++category)
    {
      const BatchMemoryBytes bytes = checkedBatchMemoryMultiply(
          participant.per_owner.at(static_cast<BatchMemoryCategory>(category)), participant.owner_multiplicity,
          participant.participant_id);
      aggregate.add(static_cast<BatchMemoryCategory>(category), bytes, participant.participant_id);
    }
  }
  return aggregate;
}

BatchExecutionPlan selectBatchExecutionPlan(const BatchExecutionSelectionInput& input,
                                            const BatchMemoryEstimator& estimator)
{
  validateSelectionInput(input);
  if (!estimator)
    throw std::invalid_argument("Batch execution memory selection requires an estimator");

  BatchTileCapacities selected;
  for (const BatchExecutionMode mode : TUNABLE_MODES)
  {
    const bool required = modeIsRequired(input.requirements, mode);
    setCapacity(selected, mode,
                initialCapacity(getRequest(input.policy.tiles, mode), getCapacity(input.preference.preferred, mode),
                                getCapacity(input.logical_maximum, mode), required));
  }

  BatchTileCapacities minimum = selected;
  for (const BatchExecutionMode mode : TUNABLE_MODES)
    if (modeIsRequired(input.requirements, mode) && getRequest(input.policy.tiles, mode).isAutomatic())
      setCapacity(minimum, mode, 1);

  const BatchMemoryEstimate minimum_estimate = estimator(minimum);
  const BatchMemoryBytes minimum_bytes        = minimum_estimate.total("fixed minimum");
  if (!fitsBudget(input.policy, minimum_bytes))
  {
    std::ostringstream message;
    message << "Batch execution memory minimum does not fit the configured budget (" << formatBytes(minimum_bytes)
            << ")";
    throw std::runtime_error(message.str());
  }

  BatchMemoryEstimate selected_estimate = estimator(selected);
  while (!fitsBudget(input.policy, selected_estimate.total("selected plan")))
  {
    const BatchMemoryBytes current_bytes = selected_estimate.total("selected plan");
    bool found_candidate                 = false;
    std::size_t best_saving              = 0;
    BatchTileCapacities best_capacities;
    BatchMemoryEstimate best_estimate;

    // Iteration order is the documented stable tie break: VALUE, FULL_VGL,
    // ACTIVE_GRADIENT, then ECP_OUTER.
    for (const BatchExecutionMode mode : TUNABLE_MODES)
    {
      if (!getRequest(input.policy.tiles, mode).isAutomatic() || getCapacity(selected, mode) <= 1)
        continue;

      BatchTileCapacities candidate = selected;
      setCapacity(candidate, mode, getCapacity(candidate, mode) - 1);
      BatchMemoryEstimate candidate_estimate = estimator(candidate);
      const BatchMemoryBytes saving = checkedSaving(current_bytes, candidate_estimate.total("candidate plan"));
      const std::size_t score        = savingScore(input.policy, current_bytes, saving);
      if (!found_candidate || score > best_saving)
      {
        found_candidate = true;
        best_saving     = score;
        best_capacities = candidate;
        best_estimate   = std::move(candidate_estimate);
      }
    }

    if (!found_candidate || best_saving == 0)
    {
      std::ostringstream message;
      message << "Batch execution memory budget cannot be met without reducing a hard tile request ("
              << formatBytes(current_bytes) << ")";
      throw std::runtime_error(message.str());
    }
    selected          = best_capacities;
    selected_estimate = std::move(best_estimate);
  }

  BatchExecutionPlan plan;
  plan.schema_id_              = input.schema_id;
  plan.preference_id_          = input.preference.id;
  plan.requirements_           = input.requirements;
  plan.topology_               = input.topology;
  plan.policy_                 = input.policy;
  plan.logical_maximum_        = input.logical_maximum;
  plan.selected_capacities_    = selected;
  plan.fixed_minimum_estimate_ = minimum_estimate;
  plan.selected_estimate_      = selected_estimate;
  plan.participant_ids_        = input.participant_ids;
  plan.fingerprint_            = makeFingerprint(input, selected, selected_estimate);
  return plan;
}

} // namespace qmcplusplus
