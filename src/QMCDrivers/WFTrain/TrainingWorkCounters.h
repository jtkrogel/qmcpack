//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_TRAINING_WORK_COUNTERS_H
#define QMCPLUSPLUS_TRAINING_WORK_COUNTERS_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>

namespace qmcplusplus::wftrain
{
/** One rank-local batch-work record. Product counts are API calls that requested
 * or computed a family, not configuration counts or mutually-exclusive modes.
 */
struct TrainingWorkCountSnapshot
{
  std::uint64_t proposal_rounds = 0;
  std::uint64_t ratio_api_batches = 0;
  std::uint64_t drift_gradient_api_batches = 0;
  std::uint64_t accepted_move_decisions = 0;
  std::uint64_t rejected_move_decisions = 0;
  std::uint64_t requested_value_product_batches = 0;
  std::uint64_t requested_gradient_product_batches = 0;
  std::uint64_t requested_laplacian_product_batches = 0;
  /// Accepted-state FULL VGL replay needed before production when products are invalid.
  std::uint64_t pre_move_full_spatial_handoff_batches = 0;
  std::uint64_t computed_value_product_batches = 0;
  std::uint64_t computed_gradient_product_batches = 0;
  std::uint64_t computed_laplacian_product_batches = 0;
  std::uint64_t computed_product_status_unknown_batches = 0;
  /// Successful physical local-energy evaluations; auxiliary component saves are excluded.
  std::uint64_t hamiltonian_walker_evaluations = 0;
  std::uint64_t ecp_bearing_hamiltonian_walker_evaluations = 0;
  std::uint64_t retained_configurations = 0;
};

/** Optional, bounded training counters updated once per batch/API call.
 *
 * Construct/pass this object only when PSIFORMER_WORK_COUNTER_TRACE is set to a
 * fresh path prefix. With no pointer, normal VMC paths pay no counter cost.
 * The per-rank JSONL writer is separate from legacy training_info.csv.
 */
class TrainingWorkCounters
{
public:
  void addProposalRounds(std::uint64_t count = 1) noexcept
  {
    proposal_rounds_.fetch_add(count, std::memory_order_relaxed);
  }
  void addRatioApiBatches(std::uint64_t count = 1) noexcept
  {
    ratio_api_batches_.fetch_add(count, std::memory_order_relaxed);
  }
  void addDriftGradientApiBatches(std::uint64_t count = 1) noexcept
  {
    drift_gradient_api_batches_.fetch_add(count, std::memory_order_relaxed);
  }
  void addMoveDecisions(std::uint64_t accepted, std::uint64_t rejected) noexcept
  {
    accepted_move_decisions_.fetch_add(accepted, std::memory_order_relaxed);
    rejected_move_decisions_.fetch_add(rejected, std::memory_order_relaxed);
  }
  void addRequestedProducts(bool value, bool gradient, bool laplacian) noexcept
  {
    if (value) requested_value_product_batches_.fetch_add(1, std::memory_order_relaxed);
    if (gradient) requested_gradient_product_batches_.fetch_add(1, std::memory_order_relaxed);
    if (laplacian) requested_laplacian_product_batches_.fetch_add(1, std::memory_order_relaxed);
  }
  void addComputedProducts(bool value, bool gradient, bool laplacian) noexcept
  {
    if (value) computed_value_product_batches_.fetch_add(1, std::memory_order_relaxed);
    if (gradient) computed_gradient_product_batches_.fetch_add(1, std::memory_order_relaxed);
    if (laplacian) computed_laplacian_product_batches_.fetch_add(1, std::memory_order_relaxed);
  }
  void addUnknownComputedProducts() noexcept
  {
    computed_product_status_unknown_batches_.fetch_add(1, std::memory_order_relaxed);
  }
  void addPreMoveFullSpatialHandoffBatch() noexcept
  {
    pre_move_full_spatial_handoff_batches_.fetch_add(1, std::memory_order_relaxed);
  }
  void addHamiltonianEvaluations(std::uint64_t walkers, std::uint64_t ecp_bearing_walkers = 0) noexcept
  {
    hamiltonian_walker_evaluations_.fetch_add(walkers, std::memory_order_relaxed);
    ecp_bearing_hamiltonian_walker_evaluations_.fetch_add(ecp_bearing_walkers, std::memory_order_relaxed);
  }
  void addRetainedConfigurations(std::uint64_t count) noexcept
  {
    retained_configurations_.fetch_add(count, std::memory_order_relaxed);
  }

  TrainingWorkCountSnapshot snapshotAndReset() noexcept
  {
    return {proposal_rounds_.exchange(0, std::memory_order_relaxed),
            ratio_api_batches_.exchange(0, std::memory_order_relaxed),
            drift_gradient_api_batches_.exchange(0, std::memory_order_relaxed),
            accepted_move_decisions_.exchange(0, std::memory_order_relaxed),
            rejected_move_decisions_.exchange(0, std::memory_order_relaxed),
            requested_value_product_batches_.exchange(0, std::memory_order_relaxed),
            requested_gradient_product_batches_.exchange(0, std::memory_order_relaxed),
            requested_laplacian_product_batches_.exchange(0, std::memory_order_relaxed),
            pre_move_full_spatial_handoff_batches_.exchange(0, std::memory_order_relaxed),
            computed_value_product_batches_.exchange(0, std::memory_order_relaxed),
            computed_gradient_product_batches_.exchange(0, std::memory_order_relaxed),
            computed_laplacian_product_batches_.exchange(0, std::memory_order_relaxed),
            computed_product_status_unknown_batches_.exchange(0, std::memory_order_relaxed),
            hamiltonian_walker_evaluations_.exchange(0, std::memory_order_relaxed),
            ecp_bearing_hamiltonian_walker_evaluations_.exchange(0, std::memory_order_relaxed),
            retained_configurations_.exchange(0, std::memory_order_relaxed)};
  }

  static bool traceEnabled() noexcept
  {
    static const bool enabled = [] {
      const char* setting = std::getenv("PSIFORMER_WORK_COUNTER_TRACE");
      return setting && *setting && std::string_view(setting) != "0";
    }();
    return enabled;
  }

private:
  std::atomic<std::uint64_t> proposal_rounds_{0};
  std::atomic<std::uint64_t> ratio_api_batches_{0};
  std::atomic<std::uint64_t> drift_gradient_api_batches_{0};
  std::atomic<std::uint64_t> accepted_move_decisions_{0};
  std::atomic<std::uint64_t> rejected_move_decisions_{0};
  std::atomic<std::uint64_t> requested_value_product_batches_{0};
  std::atomic<std::uint64_t> requested_gradient_product_batches_{0};
  std::atomic<std::uint64_t> requested_laplacian_product_batches_{0};
  std::atomic<std::uint64_t> pre_move_full_spatial_handoff_batches_{0};
  std::atomic<std::uint64_t> computed_value_product_batches_{0};
  std::atomic<std::uint64_t> computed_gradient_product_batches_{0};
  std::atomic<std::uint64_t> computed_laplacian_product_batches_{0};
  std::atomic<std::uint64_t> computed_product_status_unknown_batches_{0};
  std::atomic<std::uint64_t> hamiltonian_walker_evaluations_{0};
  std::atomic<std::uint64_t> ecp_bearing_hamiltonian_walker_evaluations_{0};
  std::atomic<std::uint64_t> retained_configurations_{0};
};

/** Append rank-local, iteration-scoped records only under explicit opt-in. */
class TrainingWorkCounterTrace
{
public:
  TrainingWorkCounterTrace() noexcept
  {
    if (!TrainingWorkCounters::traceEnabled())
      return;
    try
    {
      rank_ = "unknown";
      for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"})
        if (const char* value = std::getenv(name))
        {
          if (*value && std::string_view(value).find_first_not_of("0123456789") == std::string_view::npos)
            rank_ = value;
          break;
        }
      pid_ = getpid();
      const char* prefix = std::getenv("PSIFORMER_WORK_COUNTER_TRACE");
      const std::string path = std::string(prefix) + ".rank" + rank_ + ".pid" + std::to_string(pid_) + ".jsonl";
      file_.open(path, std::ios::out | std::ios::app);
      if (!file_)
        warn("cannot open sidecar path");
    }
    catch (...)
    {
      warn("sidecar initialization failed");
    }
  }

  void record(const char* phase, long long iteration, const TrainingWorkCountSnapshot& counts,
              std::uint64_t factor_generation_builds = 0,
              std::uint64_t inverse_generation_builds = 0,
              const char* source = "WFTrainBatched.cpp",
              const char* qmc_section = "unknown") noexcept
  {
    if (!TrainingWorkCounters::traceEnabled())
      return;
    try
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!file_)
        return;
      const auto mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      file_ << "{\"schema\":1,\"kind\":\"work_counters\",\"source\":";
      writeJsonString(file_, source ? source : "unknown");
      file_ << ",\"qmc_section\":";
      writeJsonString(file_, qmc_section ? qmc_section : "unknown");
      file_ << ",\"scope\":\"rank_local_batch_counts\",\"phase\":";
      writeJsonString(file_, phase ? phase : "unknown");
      file_ << ",\"iteration\":" << iteration << ",\"rank\":\"" << rank_
            << "\",\"pid\":" << pid_ << ",\"mono_ns\":" << mono_ns
            << ",\"proposal_rounds\":" << counts.proposal_rounds
            << ",\"ratio_api_batches\":" << counts.ratio_api_batches
            << ",\"drift_gradient_api_batches\":" << counts.drift_gradient_api_batches
            << ",\"accepted_move_decisions\":" << counts.accepted_move_decisions
            << ",\"rejected_move_decisions\":" << counts.rejected_move_decisions
            << ",\"requested_value_product_batches\":" << counts.requested_value_product_batches
            << ",\"requested_gradient_product_batches\":" << counts.requested_gradient_product_batches
            << ",\"requested_laplacian_product_batches\":" << counts.requested_laplacian_product_batches
            << ",\"pre_move_full_spatial_handoff_batches\":" << counts.pre_move_full_spatial_handoff_batches
            << ",\"computed_value_product_batches\":" << counts.computed_value_product_batches
            << ",\"computed_gradient_product_batches\":" << counts.computed_gradient_product_batches
            << ",\"computed_laplacian_product_batches\":" << counts.computed_laplacian_product_batches
            << ",\"computed_product_status_unknown_batches\":" << counts.computed_product_status_unknown_batches
            << ",\"hamiltonian_walker_evaluations\":" << counts.hamiltonian_walker_evaluations
            << ",\"ecp_bearing_hamiltonian_walker_evaluations\":"
            << counts.ecp_bearing_hamiltonian_walker_evaluations
            << ",\"retained_configurations\":" << counts.retained_configurations
            << ",\"kfac_factor_generation_builds\":" << factor_generation_builds
            << ",\"kfac_inverse_generation_builds\":" << inverse_generation_builds << "}\n";
      file_.flush();
      if (!file_)
        warn("sidecar write failed");
    }
    catch (...)
    {
      warn("sidecar write failed");
    }
  }

  static TrainingWorkCounterTrace& instance() noexcept
  {
    static auto* trace = new TrainingWorkCounterTrace();
    return *trace;
  }

private:
  static void writeJsonString(std::ofstream& output, const char* value) noexcept
  {
    static constexpr char hex[] = "0123456789abcdef";
    output.put('"');
    for (const auto* cursor = reinterpret_cast<const unsigned char*>(value ? value : "unknown"); *cursor;
         ++cursor)
    {
      switch (*cursor)
      {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (*cursor < 0x20)
        {
          output << "\\u00" << hex[*cursor >> 4] << hex[*cursor & 0x0f];
          break;
        }
        output.put(static_cast<char>(*cursor));
      }
    }
    output.put('"');
  }

  void warn(const char* message) noexcept
  {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
      std::fprintf(stderr, "WFTrain work counter trace disabled: %s; scientific execution continues\n", message);
  }

  std::mutex mutex_;
  std::ofstream file_;
  std::string rank_ = "unknown";
  long long pid_ = -1;
};
}

#endif
