//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file AcceleratorTrainingPlan.cpp
 * @brief Validation and deterministic schedules for distributed accelerator training.
 */

#include "QMCDrivers/WFTrain/AcceleratorTrainingPlan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::wftrain
{
namespace
{

/// Add two extents without permitting silent size_t wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* description)
{
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    throw std::overflow_error(std::string("Accelerator training overflow for ") + description);
  return lhs + rhs;
}

/// Multiply two extents without permitting silent size_t wraparound.
std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs, const char* description)
{
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    throw std::overflow_error(std::string("Accelerator training overflow for ") + description);
  return lhs * rhs;
}

/// Extend a stable FNV-1a plan identity with one fixed-width unsigned value.
void extendFingerprint(std::uint64_t& fingerprint, std::uint64_t value) noexcept
{
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
  {
    fingerprint ^= static_cast<unsigned char>(value & UINT64_C(0xff));
    fingerprint *= UINT64_C(1099511628211);
    value >>= 8;
  }
}

/// Validate that fixed rank records form one complete ordered participant set.
template<class Record>
std::vector<const Record*> orderRankRecords(const std::vector<Record>& records)
{
  if (records.empty())
    throw std::invalid_argument("Accelerator training requires at least one rank record");

  const std::size_t participant_count = records.size();
  std::vector<const Record*> ordered(participant_count, nullptr);
  for (const Record& record : records)
  {
    if (record.participant_count != participant_count)
      throw std::invalid_argument("Accelerator training participant-count metadata mismatch");
    if (record.rank >= participant_count || ordered[record.rank] != nullptr)
      throw std::invalid_argument("Accelerator training rank records must be unique and contiguous");
    ordered[record.rank] = &record;
  }
  return ordered;
}

} // namespace

std::vector<AcceleratorChunkDescriptor> makeAcceleratorChunkSchedule(
    std::size_t element_count,
    std::size_t maximum_chunk_elements)
{
  if (maximum_chunk_elements == 0)
    throw std::invalid_argument("Accelerator collective chunk size must be positive");

  std::vector<AcceleratorChunkDescriptor> chunks;
  std::size_t offset  = 0;
  std::size_t ordinal = 0;
  while (offset < element_count)
  {
    const std::size_t count = std::min(maximum_chunk_elements, element_count - offset);
    chunks.push_back({ordinal, offset, count});
    offset = checkedAdd(offset, count, "canonical chunk offset");
    ++ordinal;
  }
  return chunks;
}

AcceleratorDistributedPlan makeAcceleratorDistributedPlan(
    const std::vector<AcceleratorRankPlanInput>& ranks,
    std::size_t parameter_count,
    std::size_t collective_element_bytes,
    AcceleratorTransportRequest transport_request,
    std::size_t buffer_slots)
{
  const std::vector<const AcceleratorRankPlanInput*> ordered = orderRankRecords(ranks);
  if (parameter_count == 0)
    throw std::invalid_argument("Accelerator training requires a nonempty parameter vector");
  if (collective_element_bytes == 0)
    throw std::invalid_argument("Accelerator collective element width must be positive");
  if (buffer_slots == 0 || buffer_slots > 2)
    throw std::invalid_argument("Accelerator collective buffer slots must be one or two");

  const AcceleratorRankPlanInput& reference = *ordered.front();
  std::size_t total_samples = 0;
  std::size_t common_chunk  = std::numeric_limits<std::size_t>::max();
  bool all_device_mpi       = true;
  std::set<std::pair<std::uint64_t, std::size_t>> placements;
  for (const AcceleratorRankPlanInput* rank : ordered)
  {
    if (rank->parameter_version != reference.parameter_version ||
        rank->model_fingerprint != reference.model_fingerprint ||
        rank->precision_fingerprint != reference.precision_fingerprint)
      throw std::invalid_argument("Accelerator training rank semantic metadata mismatch");
    if (rank->maximum_chunk_elements == 0)
      throw std::invalid_argument("Accelerator rank reports a zero collective chunk capacity");
    if (!placements.emplace(rank->node_id, rank->device_id).second)
      throw std::invalid_argument("Accelerator topology assigns two local ranks to the same device");

    total_samples = checkedAdd(total_samples, rank->local_sample_count, "global sample count");
    common_chunk  = std::min(common_chunk, rank->maximum_chunk_elements);
    all_device_mpi = all_device_mpi && rank->device_mpi_available;
  }

  AcceleratorCollectiveTransport transport = AcceleratorCollectiveTransport::HOST_STAGED_MPI;
  switch (transport_request)
  {
  case AcceleratorTransportRequest::AUTO:
    transport = all_device_mpi ? AcceleratorCollectiveTransport::DEVICE_MPI
                               : AcceleratorCollectiveTransport::HOST_STAGED_MPI;
    break;
  case AcceleratorTransportRequest::HOST_STAGED_MPI:
    transport = AcceleratorCollectiveTransport::HOST_STAGED_MPI;
    break;
  case AcceleratorTransportRequest::DEVICE_MPI:
    if (!all_device_mpi)
      throw std::runtime_error("Direct device MPI was requested but is not available on every rank");
    transport = AcceleratorCollectiveTransport::DEVICE_MPI;
    break;
  }

  AcceleratorDistributedPlan plan;
  plan.transport             = transport;
  plan.participant_count     = ranks.size();
  plan.total_sample_count    = total_samples;
  plan.parameter_count       = parameter_count;
  plan.parameter_version     = reference.parameter_version;
  plan.chunk_elements        = std::min(common_chunk, parameter_count);
  plan.collective_element_bytes = collective_element_bytes;
  plan.buffer_slots          = buffer_slots;
  plan.model_fingerprint     = reference.model_fingerprint;
  plan.precision_fingerprint = reference.precision_fingerprint;
  plan.chunks = makeAcceleratorChunkSchedule(parameter_count, plan.chunk_elements);

  const std::size_t one_slot_bytes = checkedMultiply(
      plan.chunk_elements, collective_element_bytes, "one collective buffer");
  plan.device_collective_bytes_per_rank = checkedMultiply(
      one_slot_bytes, buffer_slots, "device collective buffers");
  if (transport == AcceleratorCollectiveTransport::HOST_STAGED_MPI)
    plan.host_staging_bytes_per_rank = checkedMultiply(
        one_slot_bytes, buffer_slots, "pinned host collective buffers");

  plan.required_device_bytes_per_rank.reserve(ordered.size());
  for (const AcceleratorRankPlanInput* rank : ordered)
  {
    const std::size_t required = checkedAdd(
        rank->base_required_device_bytes, plan.device_collective_bytes_per_rank,
        "rank device storage including collectives");
    if (rank->device_budget_bytes && required > *rank->device_budget_bytes)
      throw std::runtime_error("Accelerator rank device-memory plan exceeds its hard budget");
    plan.required_device_bytes_per_rank.push_back(required);
  }

  std::uint64_t fingerprint = UINT64_C(14695981039346656037);
  constexpr std::uint64_t protocol_version = 1;
  extendFingerprint(fingerprint, protocol_version);
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.transport));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.participant_count));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.parameter_count));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.parameter_version));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.chunk_elements));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.collective_element_bytes));
  extendFingerprint(fingerprint, static_cast<std::uint64_t>(plan.buffer_slots));
  extendFingerprint(fingerprint, plan.model_fingerprint);
  extendFingerprint(fingerprint, plan.precision_fingerprint);
  for (const AcceleratorRankPlanInput* rank : ordered)
  {
    extendFingerprint(fingerprint, static_cast<std::uint64_t>(rank->rank));
    extendFingerprint(fingerprint, rank->node_id);
    extendFingerprint(fingerprint, static_cast<std::uint64_t>(rank->device_id));
  }
  plan.fingerprint = fingerprint;
  return plan;
}

AcceleratorPublicationDecision decideAcceleratorPublication(
    const std::vector<AcceleratorPublicationReport>& reports)
{
  const std::vector<const AcceleratorPublicationReport*> ordered = orderRankRecords(reports);
  const AcceleratorPublicationReport& reference = *ordered.front();
  if (reference.candidate_version <= reference.active_version)
    throw std::invalid_argument("Accelerator candidate version must be newer than the active version");

  for (const AcceleratorPublicationReport* report : ordered)
    if (report->active_version != reference.active_version ||
        report->candidate_version != reference.candidate_version)
      throw std::invalid_argument("Accelerator publication version metadata mismatch");

  for (const AcceleratorPublicationReport* report : ordered)
    if (report->status != AcceleratorPublicationStatus::READY)
      return {/*accepted=*/false,
              /*active_version=*/reference.active_version,
              /*failing_rank=*/report->rank,
              /*failure_status=*/report->status};

  return {/*accepted=*/true,
          /*active_version=*/reference.candidate_version,
          /*failing_rank=*/std::nullopt,
          /*failure_status=*/AcceleratorPublicationStatus::READY};
}

AcceleratorRestartMode validateAcceleratorRestart(
    const AcceleratorCheckpointManifest& manifest,
    std::size_t requested_participant_count,
    std::uint64_t expected_model_fingerprint,
    std::uint64_t expected_precision_fingerprint,
    std::uint64_t expected_plan_fingerprint,
    bool allow_rank_count_change)
{
  if (!manifest.complete)
    throw std::runtime_error("Accelerator checkpoint manifest is incomplete");
  if (manifest.participant_count == 0 || requested_participant_count == 0)
    throw std::invalid_argument("Accelerator restart participant count must be positive");
  if (manifest.model_fingerprint != expected_model_fingerprint ||
      manifest.precision_fingerprint != expected_precision_fingerprint)
    throw std::invalid_argument("Accelerator checkpoint semantic fingerprint mismatch");
  if (manifest.participant_count == requested_participant_count)
  {
    if (manifest.plan_fingerprint != expected_plan_fingerprint)
      throw std::invalid_argument("Accelerator checkpoint execution-plan fingerprint mismatch");
    return AcceleratorRestartMode::EXACT_DECOMPOSITION;
  }
  if (!allow_rank_count_change)
    throw std::runtime_error("Accelerator checkpoint rank-count change is disabled");
  return AcceleratorRestartMode::STATISTICAL_REPARTITION;
}

} // namespace qmcplusplus::wftrain
