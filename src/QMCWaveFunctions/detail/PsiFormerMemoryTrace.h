//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_MEMORY_TRACE_H
#define QMCPLUSPLUS_PSIFORMER_MEMORY_TRACE_H

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>

namespace qmcplusplus::psiformer::memory
{
/// Frozen opt-in; the disabled path does not construct the sink or metadata.
inline bool enabled() noexcept
{
  static const bool value = [] {
    const char* prefix = std::getenv("PSIFORMER_MEMORY_TRACE");
    return prefix && *prefix;
  }();
  return value;
}

inline std::uint64_t nextIdentity() noexcept
{
  static std::atomic<std::uint64_t> counter{1};
  return enabled() ? counter.fetch_add(1, std::memory_order_relaxed) : 0;
}

/// All strings are borrowed for this call only. No allocation registry is retained.
struct Event
{
  const char* kind;
  std::uint64_t id = 0;
  const char* owner = "unknown";
  const char* category = "device_capacity";
  const char* backend = "unknown";
  int device = -1;
  std::size_t bytes = 0;
  std::size_t logical = 0;
  const char* file = "unknown";
  int line = 0;
  std::uint64_t parent = 0;
  std::size_t offset = 0;
  const char* dtype = "bytes";
  const char* shape = "unknown";
  std::uint64_t generation = 0;
  std::uintptr_t resource = 0;
  long long crowd = -1;
  bool native_snapshot = false;
  std::size_t native_live = 0;
  std::size_t native_peak = 0;
};

class Sink
{
public:
  explicit Sink(const char* prefix) noexcept
  {
    try
    {
      const char* rank = nullptr;
      for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"})
        if ((rank = std::getenv(name)))
          break;
      rank_ = rank ? rank : "unknown";
      if (rank && (rank_.empty() || rank_.find_first_not_of("0123456789") != std::string::npos))
        rank_ = "unknown";
      const std::string path = std::string(prefix ? prefix : "") + ".rank" + rank_ + ".pid" +
          std::to_string(getpid()) + ".jsonl";
      file_ = std::fopen(path.c_str(), "w");
      if (!file_)
      {
        warn();
        return;
      }
      const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      std::fprintf(file_, "{\"schema\":1,\"kind\":\"begin\",\"wall_ns\":%lld,\"mono_ns\":%lld,\"pid\":%ld,\"rank\":\"%s\"}\n",
                   static_cast<long long>(wall), now(), static_cast<long>(getpid()), rank_.c_str());
    }
    catch (...)
    {
      warn();
    }
  }

  Sink(const Sink&) = delete;

  ~Sink() noexcept
  {
    if (file_)
    {
      finish();
      if (std::fclose(file_) != 0)
        warn();
    }
  }

  void finish() noexcept
  {
    try
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (file_ && (std::fprintf(file_, "{\"schema\":1,\"kind\":\"end\",\"events\":%llu,\"dropped_events\":%llu}\n",
                    count_, static_cast<unsigned long long>(dropped_.load())) < 0 ||
                    std::fflush(file_) != 0))
        warn();
    }
    catch (...)
    {
      warn();
    }
  }

  void emit(const Event& e) noexcept
  {
    try
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!file_ || broken_.load())
      {
        ++dropped_;
        return;
      }
      // Streaming bounds history memory independently of the number of events.
      const auto owner = quote(e.owner), category = quote(e.category), backend = quote(e.backend);
      const auto source = quote(e.file), dtype = quote(e.dtype), shape = quote(e.shape);
      const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
      const int result = std::fprintf(file_,
          "{\"schema\":1,\"kind\":\"%s\",\"mono_ns\":%lld,\"thread\":%llu,\"id\":%llu,"
          "\"owner\":%s,\"category\":%s,\"backend\":%s,\"device\":%d,\"bytes\":%zu,\"logical_bytes\":%zu,"
          "\"source\":%s,\"line\":%d,\"parent\":%llu,\"offset_bytes\":%zu,\"dtype\":%s,\"shape\":%s,"
          "\"generation\":%llu,\"resource\":%llu,\"crowd\":%lld,"
          "\"native_snapshot\":%s,\"native_live_bytes\":%zu,\"native_peak_bytes\":%zu}\n",
          e.kind, now(), static_cast<unsigned long long>(thread), static_cast<unsigned long long>(e.id), owner.c_str(),
          category.c_str(), backend.c_str(), e.device, e.bytes, e.logical, source.c_str(), e.line,
          static_cast<unsigned long long>(e.parent), e.offset, dtype.c_str(), shape.c_str(),
          static_cast<unsigned long long>(e.generation), static_cast<unsigned long long>(e.resource), e.crowd,
          e.native_snapshot ? "true" : "false", e.native_live, e.native_peak);
      ++count_;
      if (result < 0 || std::fflush(file_) != 0)
      {
        broken_.store(true);
        ++dropped_;
        warn();
      }
    }
    catch (...)
    {
      broken_.store(true);
      ++dropped_;
      warn();
    }
  }
private:
  static long long now() noexcept
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  static std::string quote(const char* s)
  {
    std::string out = "\"";
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s ? s : "unknown"); *p; ++p)
    {
      if (*p == '"' || *p == '\\')
      {
        out += '\\';
        out += *p;
      }
      else if (*p < 32)
      {
        const char hex[] = "0123456789abcdef";
        out += "\\u00";
        out += hex[*p >> 4];
        out += hex[*p & 15];
      }
      else
        out += *p;
    }
    return out + '"';
  }

  void warn() noexcept
  {
    if (!warned_.exchange(true))
      std::fputs("PsiFormer memory trace I/O failure; scientific execution continues\n", stderr);
  }
  FILE* file_ = nullptr;
  std::string rank_;
  std::mutex mutex_;
  std::atomic<bool> warned_{false};
  std::atomic<bool> broken_{false};
  std::atomic<std::size_t> dropped_{0};
  unsigned long long count_ = 0;
};

inline Sink* processSink() noexcept
{
  // Process lifetime: late static buffer destructors must not use a destroyed sink.
  static Sink* sink = []() noexcept -> Sink* {
    try
    {
      auto* result = new Sink(std::getenv("PSIFORMER_MEMORY_TRACE"));
      std::atexit([] {
        if (auto* target = processSink())
          target->finish();
      });
      return result;
    }
    catch (...)
    {
      std::fputs("PsiFormer memory trace initialization failed; scientific execution continues\n", stderr);
      return nullptr;
    }
  }();
  return sink;
}

inline void emit(const Event& event) noexcept
{
  if (enabled())
    if (auto* sink = processSink())
      sink->emit(event);
}

/// Host scopes mark submission/lifetime boundaries, never device completion.
class Phase
{
public:
  explicit Phase(const char* name) noexcept : name_(name), id_(nextIdentity())
  {
    if (id_)
      emit({"phase_begin", id_, name_, "phase", "host"});
  }

  ~Phase() noexcept
  {
    if (id_)
      emit({"phase_end", id_, name_, "phase", "host"});
  }

  Phase(const Phase&) = delete;

  void transition(const char* name) noexcept
  {
    if (id_)
      emit({"phase_end", id_, name_, "phase", "host"});
    name_ = name;
    id_ = nextIdentity();
    if (id_)
      emit({"phase_begin", id_, name_, "phase", "host"});
  }
private:
  const char* name_;
  std::uint64_t id_;
};

/// Explicit physical capacity owner; snapshots replace capacity, views never add it.
class HostCapacity
{
public:
  HostCapacity() noexcept = default;

  ~HostCapacity() noexcept
  {
    if (id_)
      emit({"release", id_, name_, "host_capacity", "host"});
  }

  HostCapacity(const HostCapacity&) = delete;

  HostCapacity(HostCapacity&& other) noexcept : id_(other.id_), name_(other.name_), bytes_(other.bytes_)
  {
    other.id_ = 0;
    other.bytes_ = 0;
  }

  HostCapacity& operator=(HostCapacity&& other) noexcept
  {
    if (this != &other)
    {
      if (id_)
        emit({"release", id_, name_, "host_capacity", "host"});
      id_ = other.id_;
      name_ = other.name_;
      bytes_ = other.bytes_;
      other.id_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }

  void snapshot(const char* name, std::size_t bytes, std::uintptr_t resource = 0,
                long long crowd = -1, std::uint64_t generation = 0) noexcept
  {
    snapshotAt("unknown", 0, name, bytes, resource, crowd, generation);
  }

  void snapshotAt(const char* file, int line, const char* name, std::size_t bytes, std::uintptr_t resource = 0,
                  long long crowd = -1, std::uint64_t generation = 0) noexcept
  {
    if (!enabled() || (!id_ && !bytes))
      return;
    if (!id_)
      id_ = nextIdentity();
    name_ = name;
    if (bytes != bytes_)
    {
      Event event{"capacity", id_, name_, "host_capacity", "host", -1, bytes};
      event.resource = resource;
      event.crowd = crowd;
      event.generation = generation;
      event.file = file;
      event.line = line;
      emit(event);
      bytes_ = bytes;
    }
  }
private:
  std::uint64_t id_ = 0;
  const char* name_ = "unknown";
  std::size_t bytes_ = 0;
};

/** Trace one selected owning HOST allocation whose vector lifetime is explicit.
 *
 * This is a narrow ownership annotation, not a HOST allocator interceptor.
 * Call requestAt immediately before the backing container allocates, then
 * allocationSucceeded after success (or allocationFailed on exception).
 * Declare this object before the backing container so C++ destroys the storage
 * before the tracker emits its release event. Bytes are the container's owned
 * capacity payload, not allocator overhead, RSS, pinned memory, or physical
 * residency.
 */
class HostAllocation
{
public:
  HostAllocation() noexcept = default;
  HostAllocation(const HostAllocation&) = delete;
  HostAllocation& operator=(const HostAllocation&) = delete;

  HostAllocation(HostAllocation&& other) noexcept { take(other); }
  HostAllocation& operator=(HostAllocation&&) = delete;

  ~HostAllocation() noexcept { release(); }

  /// Record intent immediately before a selected container allocates storage.
  void requestAt(const char* file, int line, const char* owner, std::size_t requested_bytes,
                 const char* dtype, const char* shape, std::uintptr_t resource = 0,
                 long long crowd = -1, std::uint64_t generation = 0) noexcept
  {
    if (!enabled() || !requested_bytes)
      return;
    release();
    id_ = nextIdentity();
    if (!id_)
      return;
    owner_ = owner ? owner : "unknown";
    category_ = "host_tracked_capacity";
    backend_ = "host";
    file_ = file ? file : "unknown";
    line_ = line;
    requested_bytes_ = requested_bytes;
    bytes_ = 0;
    dtype_ = dtype ? dtype : "unknown";
    resource_ = resource;
    crowd_ = crowd;
    generation_ = generation;
    std::snprintf(shape_.data(), shape_.size(), "%s", shape ? shape : "unknown");
    emitRecord("request", requested_bytes_);
  }

  /// Mark successful storage creation using its observed container capacity.
  void allocationSucceeded(std::size_t capacity_bytes) noexcept
  {
    if (!id_)
      return;
    bytes_ = capacity_bytes;
    allocated_ = true;
    emitRecord("allocated", bytes_);
  }

  /// Preserve failed allocation intent without claiming bytes became live.
  void allocationFailed() noexcept
  {
    if (!id_)
      return;
    emitRecord("allocation_failed", requested_bytes_);
    id_ = 0;
    requested_bytes_ = 0;
    bytes_ = 0;
    allocated_ = false;
  }

  /// End tracking after the backing container explicitly relinquishes capacity.
  void storageReleased() noexcept
  {
    if (!id_ || !allocated_)
      return;
    emitRecord("release", bytes_);
    id_ = 0;
    requested_bytes_ = 0;
    bytes_ = 0;
    allocated_ = false;
  }

private:
  void emitRecord(const char* kind, std::size_t bytes) noexcept
  {
    Event event{kind, id_, owner_, category_, backend_, -1, bytes};
    event.logical = bytes;
    event.file = file_;
    event.line = line_;
    event.dtype = dtype_;
    event.shape = shape_.data();
    event.resource = resource_;
    event.crowd = crowd_;
    event.generation = generation_;
    emit(event);
  }

  void take(HostAllocation& other) noexcept
  {
    id_ = std::exchange(other.id_, 0);
    owner_ = other.owner_;
    category_ = other.category_;
    backend_ = other.backend_;
    file_ = other.file_;
    line_ = other.line_;
    bytes_ = std::exchange(other.bytes_, 0);
    requested_bytes_ = std::exchange(other.requested_bytes_, 0);
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    resource_ = other.resource_;
    crowd_ = other.crowd_;
    generation_ = other.generation_;
    allocated_ = std::exchange(other.allocated_, false);
  }

  void release() noexcept
  {
    if (!id_)
      return;
    emitRecord(allocated_ ? "release" : "allocation_failed", allocated_ ? bytes_ : requested_bytes_);
    id_ = 0;
    requested_bytes_ = 0;
    bytes_ = 0;
    allocated_ = false;
  }

  std::uint64_t id_ = 0;
  const char* owner_ = "unknown";
  const char* category_ = "host_tracked_capacity";
  const char* backend_ = "host";
  const char* file_ = "unknown";
  int line_ = 0;
  std::size_t requested_bytes_ = 0;
  std::size_t bytes_ = 0;
  const char* dtype_ = "unknown";
  std::array<char, 128> shape_{{'u', 'n', 'k', 'n', 'o', 'w', 'n', '\0'}};
  std::uintptr_t resource_ = 0;
  long long crowd_ = -1;
  std::uint64_t generation_ = 0;
  bool allocated_ = false;
};
}
#endif
