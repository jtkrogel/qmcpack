// Test-only old/new pure-CI matrix seam; no production optimizer interface.
#ifndef QMCPLUSPLUS_LEGACY_P1_MATRIX_TEST_H
#define QMCPLUSPLUS_LEGACY_P1_MATRIX_TEST_H

#include <array>

namespace qmcplusplus::testing
{
struct LegacyP1Sample
{
  double weight, energy, score, energy_derivative; // raw O and dEL, not d(Hpsi)/psi
};

struct LegacyP1Matrices
{
  std::array<double, 4> hamiltonian, overlap; // row-major, raw centered H (not symmetrized)
  double linear_rescale;
};

/// Feed four physical records into the actual old beta-zero matrix builder.
LegacyP1Matrices legacyP1Matrices(const std::array<LegacyP1Sample, 4>& records);
}
#endif
