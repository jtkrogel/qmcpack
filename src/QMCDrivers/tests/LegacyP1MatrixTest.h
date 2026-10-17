//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

// Test-only old/new pure-CI matrix seam; no production optimizer interface.
#ifndef QMCPLUSPLUS_LEGACY_P1_MATRIX_TEST_H
#define QMCPLUSPLUS_LEGACY_P1_MATRIX_TEST_H

#include <array>

namespace qmcplusplus::testing
{
/// One physically evaluated weighted sample in the single-active-CI coordinate.
struct LegacyP1Sample
{
  double weight, energy, score, energy_derivative; // raw O and dEL, not d(Hpsi)/psi
};

/// Actual legacy raw centered matrices and its pure-linear rescaling policy.
struct LegacyP1Matrices
{
  std::array<double, 4> hamiltonian, overlap; // row-major, raw centered H (not symmetrized)
  double linear_rescale;
};

/// Feed four physical records into the actual old beta-zero matrix builder.
LegacyP1Matrices legacyP1Matrices(const std::array<LegacyP1Sample, 4>& records);
}
#endif
