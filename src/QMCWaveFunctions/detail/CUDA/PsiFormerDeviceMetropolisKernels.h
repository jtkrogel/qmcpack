//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_METROPOLIS_KERNELS_H

#include "PsiFormerDeviceRuntime.h"
#include <cstddef>
#include <cstdint>

namespace qmcplusplus::psiformer::device
{
/// Per-walker native draw eligibility and completed acceptance, not fatal errors.
enum MetropolisMask : std::uint32_t { METROPOLIS_ELIGIBLE = 1, METROPOLIS_ACCEPTED = 2 };

/// Keep the native strict comparison independently testable at an exact boundary.
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline bool metropolisLogComparison(double log_uniform, double log_acceptance) noexcept
{ return log_uniform < log_acceptance; }

/// Generate all coordinates without touching the accepted source.
Error launchMetropolisProposal(Stream stream, std::size_t population, std::size_t electrons,
    const double* accepted, const double* gaussian, const double* inverse_mass, double tau,
    double* proposed, std::uint64_t* status, bool physical_displacement = false);

/// Consume a VALUE microtile; optional loggb/loggf arrays require offset=0, seed=false and extent=2*count.
Error launchMetropolisStageValues(Stream stream, std::size_t count, std::size_t offset, bool seed,
    const double* phase, const double* logabs, const double* accepted_log,
    double* staged_phase, double* staged_log, double* log_acceptance, std::uint32_t* mask, std::uint64_t* status,
    const double* green_correction = nullptr);

/// Merge rejections into staging; accepted coordinates/cache remain read-only.
Error launchMetropolisResolve(Stream stream, std::size_t population, std::size_t electrons,
    const double* accepted, double* proposed, const double* accepted_phase, const double* accepted_log,
    double* staged_phase, double* staged_log, const double* uniforms, const double* log_acceptance,
    std::uint32_t* mask, std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
