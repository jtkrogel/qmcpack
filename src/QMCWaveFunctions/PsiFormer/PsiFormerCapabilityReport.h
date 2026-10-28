//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_CAPABILITY_REPORT_H
#define QMCPLUSPLUS_PSIFORMER_CAPABILITY_REPORT_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <new>
#include <mutex>
#include <sstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmcplusplus::psiformer
{
/// Admission/qualification restrictions describe this implementation, not mathematical limits.
enum class CapabilityFailure
{
  UNSUPPORTED_OPERATION,
  INVALID_INPUT,
  INSUFFICIENT_RESOURCES,
  UNQUALIFIED_HARDWARE,
  UNCLASSIFIED
};

inline const char* capabilityFailureName(CapabilityFailure failure) noexcept
{
  switch (failure)
  {
  case CapabilityFailure::UNSUPPORTED_OPERATION: return "unsupported_operation";
  case CapabilityFailure::INVALID_INPUT: return "invalid_input";
  case CapabilityFailure::INSUFFICIENT_RESOURCES: return "insufficient_resources";
  case CapabilityFailure::UNQUALIFIED_HARDWARE: return "unqualified_hardware";
  default: return "unclassified";
  }
}

/// Preserve runtime_error and its exact diagnostic while attaching machine-readable evidence.
class CapabilityError : public std::runtime_error
{
public:
  CapabilityError(CapabilityFailure failure, const std::string& message)
      : std::runtime_error(message), failure(failure) {}
  const CapabilityFailure failure;
};

inline CapabilityFailure capabilityFailure(const std::exception& error) noexcept
{
  if (const auto* typed = dynamic_cast<const CapabilityError*>(&error)) return typed->failure;
  if (dynamic_cast<const std::invalid_argument*>(&error)) return CapabilityFailure::INVALID_INPUT;
  if (dynamic_cast<const std::length_error*>(&error) || dynamic_cast<const std::bad_alloc*>(&error))
    return CapabilityFailure::INSUFFICIENT_RESOURCES;
  return CapabilityFailure::UNCLASSIFIED;
}

inline void writeCapabilityString(std::ostream& out, const std::string& value)
{
  constexpr char hex[] = "0123456789abcdef";
  out << '"';
  for (unsigned char c : value)
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
    else out << c;
  out << '"';
}

/** One JSON object per lifecycle boundary. All field values are strings;
 * unknown/deferred fields are explicit and must never be interpreted as zero.
 * Callers assemble reports outside numerical kernels. No disk/file ownership here.
 */
inline void writeCapabilityReceipt(std::ostream& out, const char* phase,
    std::initializer_list<std::pair<std::string, std::string>> fields)
{
  std::ostringstream receipt;
  receipt << "PsiFormer capability {\"schema\":\"cp0_v1\",\"phase\":";
  writeCapabilityString(receipt, phase);
  for (const auto& field : fields)
  {
    receipt << ',';
    writeCapabilityString(receipt, field.first);
    receipt << ':';
    writeCapabilityString(receipt, field.second);
  }
  receipt << "}\n";
  static std::mutex report_mutex;
  std::lock_guard lock(report_mutex);
  out << receipt.str();
}

/** Completed native API dispatches, shared by a clone family. These are API
 * counts, not GPU launch counts or overlap evidence. Nested scalar fallback
 * dispatches are counted separately from their enclosing crowd call.
 */
struct CapabilityDispatchCounters
{
  std::atomic<std::size_t> scalar_full{0}, crowd_full{0}, maximum_crowd_walkers{0};

  void report(std::ostream& out, std::uint64_t identity) const
  {
    writeCapabilityReceipt(out, "dispatch_summary", {
      {"model_identity", std::to_string(identity)}, {"scope", "clone_family_completed_full_vgl_api_calls"},
      {"scalar_full", std::to_string(scalar_full.load(std::memory_order_relaxed))},
      {"crowd_full", std::to_string(crowd_full.load(std::memory_order_relaxed))},
      {"maximum_crowd_walkers", std::to_string(maximum_crowd_walkers.load(std::memory_order_relaxed))},
      {"gpu_overlap", "not_measured"}});
  }
};

/// Count only normally completed calls; a rejected dispatch contributes nothing.
class CompletedCapabilityDispatch
{
public:
  CompletedCapabilityDispatch(CapabilityDispatchCounters& counters, std::size_t walkers, bool crowd)
      : counters_(counters), walkers_(walkers), crowd_(crowd), exceptions_(std::uncaught_exceptions()) {}
  CompletedCapabilityDispatch(const CompletedCapabilityDispatch&) = delete;
  CompletedCapabilityDispatch& operator=(const CompletedCapabilityDispatch&) = delete;
  ~CompletedCapabilityDispatch()
  {
    if (std::uncaught_exceptions() != exceptions_) return;
    (crowd_ ? counters_.crowd_full : counters_.scalar_full).fetch_add(1, std::memory_order_relaxed);
    if (crowd_)
    {
      auto previous = counters_.maximum_crowd_walkers.load(std::memory_order_relaxed);
      while (previous < walkers_ && !counters_.maximum_crowd_walkers.compare_exchange_weak(
          previous, walkers_, std::memory_order_relaxed)) {}
    }
  }
private:
  CapabilityDispatchCounters& counters_;
  const std::size_t walkers_;
  const bool crowd_;
  const int exceptions_;
};
} // namespace qmcplusplus::psiformer
#endif
