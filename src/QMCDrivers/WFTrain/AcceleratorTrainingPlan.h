//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorTrainingPlan.h
 * @brief Pure distributed accelerator planning for high-parameter training.
 *
 * The contract is wavefunction-independent and contains no MPI or vendor-runtime
 * objects.  It validates the fixed metadata that ranks must agree on before entering
 * collective work and produces the canonical bounded schedule later transports use.
 */

#ifndef QMCPLUSPLUS_ACCELERATOR_TRAINING_PLAN_H
#define QMCPLUSPLUS_ACCELERATOR_TRAINING_PLAN_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Select how one prepared accelerator buffer participates in an MPI reduction.
enum class AcceleratorCollectiveTransport : std::uint8_t
{
  HOST_STAGED_MPI,
  DEVICE_MPI
};

/// Express an automatic or strict transport request before rank consensus.
enum class AcceleratorTransportRequest : std::uint8_t
{
  AUTO,
  HOST_STAGED_MPI,
  DEVICE_MPI
};

/// Describe one rank's immutable placement, capability, and memory evidence.
struct AcceleratorRankPlanInput
{
  std::size_t rank              = 0;
  std::size_t participant_count = 0;
  std::uint64_t node_id         = 0;
  std::size_t device_id         = 0;
  std::size_t local_sample_count = 0;
  std::optional<std::size_t> device_budget_bytes;
  /// Device bytes required before adding this plan's collective buffers.
  std::size_t base_required_device_bytes = 0;
  std::size_t maximum_chunk_elements = 0;
  std::size_t parameter_version = 0;
  std::uint64_t model_fingerprint = 0;
  std::uint64_t precision_fingerprint = 0;
  bool device_mpi_available = false;
};

/// Identify one fixed canonical collective interval.
struct AcceleratorChunkDescriptor
{
  std::size_t ordinal = 0;
  std::size_t offset  = 0;
  std::size_t count   = 0;

  friend bool operator==(const AcceleratorChunkDescriptor& lhs,
                         const AcceleratorChunkDescriptor& rhs) noexcept
  {
    return lhs.ordinal == rhs.ordinal && lhs.offset == rhs.offset && lhs.count == rhs.count;
  }
};

/** Immutable unanimous plan used by every rank in one training stage.
 *
 * Device identifiers are rank-local and may repeat on different nodes.  Exactly one
 * rank may own a given (node_id, device_id) pair in the first supported topology.
 */
struct AcceleratorDistributedPlan
{
  AcceleratorCollectiveTransport transport = AcceleratorCollectiveTransport::HOST_STAGED_MPI;
  std::size_t participant_count = 0;
  std::size_t total_sample_count = 0;
  std::size_t parameter_count    = 0;
  std::size_t parameter_version  = 0;
  std::size_t chunk_elements     = 0;
  std::size_t collective_element_bytes = 0;
  std::size_t buffer_slots       = 1;
  std::size_t host_staging_bytes_per_rank = 0;
  std::size_t device_collective_bytes_per_rank = 0;
  /// Complete device requirement in rank order, including collective buffers.
  std::vector<std::size_t> required_device_bytes_per_rank;
  std::uint64_t model_fingerprint     = 0;
  std::uint64_t precision_fingerprint = 0;
  std::uint64_t fingerprint           = 0;
  std::vector<AcceleratorChunkDescriptor> chunks;
};

/// Classify one rank's candidate-update status without variable-length messages.
enum class AcceleratorPublicationStatus : std::uint8_t
{
  READY,
  LOCAL_FAILURE,
  NONFINITE_CANDIDATE
};

/// Fixed publication record contributed by one rank after device finite checks.
struct AcceleratorPublicationReport
{
  std::size_t rank              = 0;
  std::size_t participant_count = 0;
  std::size_t active_version    = 0;
  std::size_t candidate_version = 0;
  AcceleratorPublicationStatus status = AcceleratorPublicationStatus::READY;
};

/// Uniform publication result; rejection always retains the previous version.
struct AcceleratorPublicationDecision
{
  bool accepted = false;
  std::size_t active_version = 0;
  std::optional<std::size_t> failing_rank;
  AcceleratorPublicationStatus failure_status = AcceleratorPublicationStatus::READY;
};

/// Distinguish exact decomposition restart from a statistical walker repartition.
enum class AcceleratorRestartMode : std::uint8_t
{
  EXACT_DECOMPOSITION,
  STATISTICAL_REPARTITION
};

/// Minimal completed-manifest fields needed before rank-local shards are loaded.
struct AcceleratorCheckpointManifest
{
  bool complete = false;
  std::size_t participant_count = 0;
  std::size_t parameter_version = 0;
  std::uint64_t model_fingerprint = 0;
  std::uint64_t precision_fingerprint = 0;
  std::uint64_t plan_fingerprint = 0;
};

/// Build canonical chunks covering [0, element_count) exactly once.
std::vector<AcceleratorChunkDescriptor> makeAcceleratorChunkSchedule(
    std::size_t element_count,
    std::size_t maximum_chunk_elements);

/** Validate rank records and construct one deterministic collective plan.
 *
 * ``collective_element_bytes`` is explicit so the same planner supports real and
 * complex reductions without depending on a particular derivative scalar typedef.
 */
AcceleratorDistributedPlan makeAcceleratorDistributedPlan(
    const std::vector<AcceleratorRankPlanInput>& ranks,
    std::size_t parameter_count,
    std::size_t collective_element_bytes,
    AcceleratorTransportRequest transport_request,
    std::size_t buffer_slots = 1);

/// Decide failure-atomic publication from one fixed report per rank.
AcceleratorPublicationDecision decideAcceleratorPublication(
    const std::vector<AcceleratorPublicationReport>& reports);

/// Validate a complete manifest and select exact or statistical restart semantics.
AcceleratorRestartMode validateAcceleratorRestart(
    const AcceleratorCheckpointManifest& manifest,
    std::size_t requested_participant_count,
    std::uint64_t expected_model_fingerprint,
    std::uint64_t expected_precision_fingerprint,
    std::uint64_t expected_plan_fingerprint,
    bool allow_rank_count_change);

} // namespace qmcplusplus::wftrain

#endif // QMCPLUSPLUS_ACCELERATOR_TRAINING_PLAN_H
