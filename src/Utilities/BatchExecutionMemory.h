//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_BATCH_EXECUTION_MEMORY_H
#define QMCPLUSPLUS_BATCH_EXECUTION_MEMORY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace qmcplusplus
{

/** Identifies the memory space charged by the batch execution planner. */
enum class BatchMemorySpace : std::uint8_t
{
  HOST,
  DEVICE
};

/** Holds independently checked host and device byte counts. */
struct BatchMemoryBytes
{
  std::size_t host   = 0;
  std::size_t device = 0;

  friend bool operator==(const BatchMemoryBytes& lhs, const BatchMemoryBytes& rhs)
  {
    return lhs.host == rhs.host && lhs.device == rhs.device;
  }
};

/** Categories make the setup-time memory report explain which storage dominates. */
enum class BatchMemoryCategory : std::size_t
{
  FIXED_CLONE_STATE,
  LOGICAL_INPUT_OUTPUT,
  INNER_TILE_SCRATCH,
  SCORE_TAPE,
  KINETIC_TAPE,
  ECP_METADATA,
  OUTER_TILE_SCRATCH,
  PUBLICATION_STAGING,
  RETAINED_HIGH_WATER,
  REALLOCATION_TRANSIENT,
  COUNT
};

/** Required operation families determine which tile dimensions may be nonzero. */
enum class BatchExecutionMode : std::uint32_t
{
  VALUE              = 1U << 0,
  FULL_VGL           = 1U << 1,
  ACTIVE_GRADIENT    = 1U << 2,
  SCORE              = 1U << 3,
  KINETIC            = 1U << 4,
  ECP_OUTER          = 1U << 5,
  ECP_WEIGHTED_SCORE = 1U << 6
};

/** Compact mask of operation families reachable from one driver section. */
class BatchExecutionRequirements
{
public:
  BatchExecutionRequirements() = default;
  explicit BatchExecutionRequirements(BatchExecutionMode mode) { require(mode); }

  void require(BatchExecutionMode mode) noexcept;
  bool requires(BatchExecutionMode mode) const noexcept;
  bool empty() const noexcept { return mask_ == 0; }
  std::uint32_t mask() const noexcept { return mask_; }

  friend bool operator==(const BatchExecutionRequirements& lhs, const BatchExecutionRequirements& rhs)
  {
    return lhs.mask_ == rhs.mask_;
  }

private:
  std::uint32_t mask_ = 0;
};

/** A tile request is either selected automatically or fixed by the input. */
class BatchTileRequest
{
public:
  static BatchTileRequest automatic() noexcept { return BatchTileRequest(true, 0); }
  static BatchTileRequest fixed(std::size_t capacity);

  bool isAutomatic() const noexcept { return automatic_; }
  std::size_t fixedCapacity() const;

  friend bool operator==(const BatchTileRequest& lhs, const BatchTileRequest& rhs)
  {
    return lhs.automatic_ == rhs.automatic_ && lhs.capacity_ == rhs.capacity_;
  }

private:
  BatchTileRequest(bool automatic, std::size_t capacity) : automatic_(automatic), capacity_(capacity) {}

  bool automatic_      = true;
  std::size_t capacity_ = 0;
};

/** Requested tile policy for all currently tunable batch dimensions. */
struct BatchTileRequests
{
  BatchTileRequest value           = BatchTileRequest::automatic();
  BatchTileRequest full_vgl        = BatchTileRequest::automatic();
  BatchTileRequest active_gradient = BatchTileRequest::automatic();
  BatchTileRequest ecp_outer       = BatchTileRequest::automatic();

  friend bool operator==(const BatchTileRequests& lhs, const BatchTileRequests& rhs)
  {
    return lhs.value == rhs.value && lhs.full_vgl == rhs.full_vgl && lhs.active_gradient == rhs.active_gradient &&
        lhs.ecp_outer == rhs.ecp_outer;
  }
};

/** Effective capacities selected for the four independent tile dimensions. */
struct BatchTileCapacities
{
  std::size_t value           = 0;
  std::size_t full_vgl        = 0;
  std::size_t active_gradient = 0;
  std::size_t ecp_outer       = 0;

  friend bool operator==(const BatchTileCapacities& lhs, const BatchTileCapacities& rhs)
  {
    return lhs.value == rhs.value && lhs.full_vgl == rhs.full_vgl &&
        lhs.active_gradient == rhs.active_gradient && lhs.ecp_outer == rhs.ecp_outer;
  }
};

/** Parsed section-local caps and hard/automatic tile requests. */
struct BatchMemoryPolicy
{
  std::optional<std::size_t> host_budget;
  std::optional<std::size_t> device_budget;
  BatchTileRequests tiles;

  bool hasHardBudget() const noexcept { return host_budget.has_value() || device_budget.has_value(); }

  friend bool operator==(const BatchMemoryPolicy& lhs, const BatchMemoryPolicy& rhs)
  {
    return lhs.host_budget == rhs.host_budget && lhs.device_budget == rhs.device_budget && lhs.tiles == rhs.tiles;
  }
};

/** Versioned backend preference; automatic selection never grows beyond it. */
struct BatchExecutionPreferenceProfile
{
  std::string id{"cpu-v1"};
  BatchTileCapacities preferred{4, 4, 4, 256};
};

/** Exact rank-local crowd topology used to make a plan reproducible. */
struct BatchExecutionTopology
{
  std::vector<std::size_t> initial_walkers_per_crowd;
  std::vector<std::size_t> reserve_walkers_per_crowd;
  bool serialized_walkers = false;
  std::string run_kind;
  std::string backend_id{"cpu"};
  std::optional<std::size_t> device_id;

  std::size_t initialWalkerCount() const;
  std::size_t reserveWalkerCount() const;
};

/** Categorized exact estimate returned by one or more registered owners. */
class BatchMemoryEstimate
{
public:
  void add(BatchMemoryCategory category, BatchMemoryBytes bytes, const std::string& context = {});
  void add(const BatchMemoryEstimate& other, const std::string& context = {});

  const BatchMemoryBytes& at(BatchMemoryCategory category) const;
  BatchMemoryBytes total(const std::string& context = {}) const;

  friend bool operator==(const BatchMemoryEstimate& lhs, const BatchMemoryEstimate& rhs)
  {
    return lhs.categories_ == rhs.categories_;
  }

private:
  std::array<BatchMemoryBytes, static_cast<std::size_t>(BatchMemoryCategory::COUNT)> categories_{};
};

/** Estimate from a stable participant, with the number of simultaneously resident owners. */
struct BatchMemoryParticipantEstimate
{
  std::string participant_id;
  std::size_t owner_multiplicity = 1;
  BatchMemoryEstimate per_owner;
};

/** Complete pure input to deterministic tile selection. */
struct BatchExecutionSelectionInput
{
  std::string schema_id{"batch-execution-memory-v1"};
  BatchMemoryPolicy policy;
  BatchExecutionRequirements requirements;
  BatchExecutionTopology topology;
  BatchTileCapacities logical_maximum;
  BatchExecutionPreferenceProfile preference;
  std::vector<std::string> participant_ids;
};

/** Logically immutable result bound to all resources created for one driver section. */
class BatchExecutionPlan
{
public:
  const std::string& schemaId() const noexcept { return schema_id_; }
  const std::string& preferenceId() const noexcept { return preference_id_; }
  std::uint64_t fingerprint() const noexcept { return fingerprint_; }
  const BatchExecutionRequirements& requirements() const noexcept { return requirements_; }
  const BatchExecutionTopology& topology() const noexcept { return topology_; }
  const BatchMemoryPolicy& policy() const noexcept { return policy_; }
  const BatchTileCapacities& logicalMaximum() const noexcept { return logical_maximum_; }
  const BatchTileCapacities& selectedCapacities() const noexcept { return selected_capacities_; }
  const BatchMemoryEstimate& fixedMinimumEstimate() const noexcept { return fixed_minimum_estimate_; }
  const BatchMemoryEstimate& selectedEstimate() const noexcept { return selected_estimate_; }
  const std::vector<std::string>& participantIds() const noexcept { return participant_ids_; }

private:
  friend BatchExecutionPlan selectBatchExecutionPlan(
      const BatchExecutionSelectionInput&,
      const std::function<BatchMemoryEstimate(const BatchTileCapacities&)>&);

  std::string schema_id_;
  std::string preference_id_;
  std::uint64_t fingerprint_ = 0;
  BatchExecutionRequirements requirements_;
  BatchExecutionTopology topology_;
  BatchMemoryPolicy policy_;
  BatchTileCapacities logical_maximum_;
  BatchTileCapacities selected_capacities_;
  BatchMemoryEstimate fixed_minimum_estimate_;
  BatchMemoryEstimate selected_estimate_;
  std::vector<std::string> participant_ids_;
};

using BatchMemoryEstimator = std::function<BatchMemoryEstimate(const BatchTileCapacities&)>;

/** Checked scalar addition used by estimators and storage owners. */
std::size_t checkedBatchMemoryAdd(std::size_t lhs, std::size_t rhs, const std::string& context);

/** Checked scalar multiplication used by estimators and storage owners. */
std::size_t checkedBatchMemoryMultiply(std::size_t lhs, std::size_t rhs, const std::string& context);

/** Checked addition of independently accounted host and device bytes. */
BatchMemoryBytes checkedBatchMemoryAdd(BatchMemoryBytes lhs,
                                       BatchMemoryBytes rhs,
                                       const std::string& context);

/** Checked multiplication of host and device bytes by an owner count. */
BatchMemoryBytes checkedBatchMemoryMultiply(BatchMemoryBytes bytes,
                                            std::size_t multiplicity,
                                            const std::string& context);

/** Aggregate stable participant estimates with checked owner multiplicities. */
BatchMemoryEstimate aggregateBatchMemoryEstimates(const std::vector<BatchMemoryParticipantEstimate>& participants);

/** Resolve hard requests and deterministic automatic tiles against exact estimates. */
BatchExecutionPlan selectBatchExecutionPlan(const BatchExecutionSelectionInput& input,
                                            const BatchMemoryEstimator& estimator);

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_BATCH_EXECUTION_MEMORY_H
