//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_CPU_COMPONENT_TRACE_H
#define QMCPLUSPLUS_PSIFORMER_CPU_COMPONENT_TRACE_H

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pf
{
namespace cpu_component_trace
{
/** Optional host CPU wall-time attribution, independent of QMCPACK global timers.
 * Set PSIFORMER_CPU_COMPONENT_TRACE to an output filename prefix before first use.
 * Each participating worker accumulates fixed counters locally; there is no hot
 * global lock or per-evaluation I/O. Registration locks once per worker. Disabled
 * scopes only test a cached pointer and do not read the clock or allocate.
 *
 * Call flushAllThreads() ONLY after all instrumented evaluations have stopped,
 * e.g. after an OpenMP parallel region / worker join. It snapshots every worker,
 * including persistent OpenMP workers. No shutdown destructor is relied upon:
 * missing explicit flush, abnormal exit, and killed jobs produce no output.
 * Repeated flushes overwrite cumulative snapshots. Failed/partial operations are
 * counted too. I/O/allocation errors are swallowed and cannot fail science.
 *
 * Stage intervals within one scope do not overlap. Labels ending in inclusive
 * contain nested real VALUE stages: never sum inclusive and exclusive rows.
 * Real spatial trunk includes geometry, neural derivatives, and cusp derivatives;
 * spatial cusp_finalize is only the final combination/validation/publication.
 * VALUE cusp_finalize similarly excludes cusp evaluation inside the real trunk.
 * FULL, TARGET and ACTIVE spatial modes share stage labels. Counts are stage
 * entries (target VGL tiles each count separately), not walkers. Snapshots never
 * reset counters; do not add repeated snapshots of the same process together.
 * Opt-in adds clock reads and counter writes at stage boundaries; timings include
 * host scheduling stalls and must not be interpreted as accelerator timings.
 */
enum class Stage : std::size_t
{
  value_real_trunk_inclusive,
  value_envelope,
  value_head,
  value_determinant,
  value_cusp_finalize,
  real_value_geometry,
  real_value_neural_trunk,
  real_value_cusp,
  spatial_real_trunk_inclusive,
  spatial_envelope,
  spatial_head,
  spatial_determinant,
  spatial_cusp_finalize,
  count
};

namespace detail
{
constexpr std::size_t stage_count = static_cast<std::size_t>(Stage::count);
inline const char* label(std::size_t i) noexcept
{
  static const char* const labels[] = {
    "value.real_trunk.inclusive", "value.envelope.exclusive", "value.head.exclusive",
    "value.determinant.exclusive", "value.cusp_finalize.exclusive",
    "real_value.geometry.exclusive", "real_value.neural_trunk.exclusive", "real_value.cusp.exclusive",
    "spatial.real_trunk.inclusive", "spatial.envelope.exclusive", "spatial.head.exclusive",
    "spatial.determinant.exclusive", "spatial.cusp_finalize.exclusive"};
  return labels[i];
}
struct Record
{
  std::array<std::uint64_t, stage_count> calls{};
  std::array<double, stage_count> seconds{};
};
struct Registry
{
  std::mutex mutex;
  std::vector<Record*> records;
  std::string prefix;
  std::string rank = "unknown";
  std::string run;
};
// Deliberately retained until process exit: records remain valid after worker exit.
inline Registry* registry() noexcept
{
  static Registry* const value = []() noexcept -> Registry* {
    try
    {
      const char* prefix = std::getenv("PSIFORMER_CPU_COMPONENT_TRACE");
      if (!prefix || !*prefix) return nullptr;
      std::unique_ptr<Registry> r(new Registry);
      r->prefix = prefix;
      for (const char* key : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"})
        if (const char* rank = std::getenv(key))
        {
          // Only numeric rank text may enter filenames / TSV fields.
          if (*rank && std::string(rank).find_first_not_of("0123456789") == std::string::npos)
            r->rank = rank;
          break;
        }
      r->run = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
      return r.release();
    }
    catch (...) { return nullptr; }
  }();
  return value;
}
inline Record* localRecord() noexcept
{
  static thread_local Record* const record = []() noexcept -> Record* {
    auto* r = registry();
    if (!r) return nullptr;
    try
    {
      std::lock_guard<std::mutex> lock(r->mutex);
      auto* entry = new Record;
      try { r->records.push_back(entry); }
      catch (...) { delete entry; return nullptr; }
      return entry;
    }
    catch (...) { return nullptr; }
  }();
  return record;
}
} // namespace detail

class Scope
{
  using Clock = std::chrono::steady_clock;
public:
  explicit Scope(Stage stage) noexcept : record_(detail::localRecord()), stage_(stage)
  {
    if (record_) start_ = Clock::now();
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept { stop(); }
  void next(Stage stage) noexcept
  {
    if (!record_) return;
    const auto now = Clock::now();
    accumulate(now);
    start_ = now;
    stage_ = stage;
  }
  void stop() noexcept
  {
    if (!record_) return;
    accumulate(Clock::now());
    record_ = nullptr;
  }
private:
  void accumulate(Clock::time_point now) noexcept
  {
    const auto index = static_cast<std::size_t>(stage_);
    ++record_->calls[index];
    record_->seconds[index] += std::chrono::duration<double>(now - start_).count();
  }
  detail::Record* record_;
  Stage stage_;
  Clock::time_point start_;
};

/** Quiescent-boundary snapshot. Returns files written, zero if off or on error.
 * Filename: PREFIX.run<TIMESTAMP>.rank<RANK>.thread<REGISTRATION_INDEX>.tsv.
 * RANK comes from common MPI launcher environment, or 'unknown' without it.
 * Use a job/process-unique PREFIX when ranks are unavailable or across jobs;
 * timestamp reduces collisions but is not a cross-process uniqueness guarantee.
 * Registration indices identify workers within this process, not OpenMP IDs.
 * No MPI, POSIX, or device runtime calls are required.
 */
inline std::size_t flushAllThreads() noexcept
{
  auto* r = detail::registry();
  if (!r) return 0;
  std::size_t written = 0;
  try
  {
    std::lock_guard<std::mutex> lock(r->mutex);
    for (std::size_t thread = 0; thread < r->records.size(); ++thread)
    {
      std::ofstream out(r->prefix + ".run" + r->run + ".rank" + r->rank +
                        ".thread" + std::to_string(thread) + ".tsv");
      out << "schema\trank\tthread\tstage\tcalls\tseconds\n" << std::setprecision(17);
      const auto& record = *r->records[thread];
      for (std::size_t stage = 0; stage < detail::stage_count; ++stage)
        if (record.calls[stage])
          out << "1\t" << r->rank << '\t' << thread << '\t' << detail::label(stage) << '\t'
              << record.calls[stage] << '\t' << record.seconds[stage] << '\n';
      out.close();
      if (out) ++written;
    }
  }
  catch (...) {}
  return written;
}
} // namespace cpu_component_trace
} // namespace pf
#endif
