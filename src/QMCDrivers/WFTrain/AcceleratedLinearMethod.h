//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_ACCELERATED_LINEAR_METHOD_H
#define QMCPLUSPLUS_ACCELERATED_LINEAR_METHOD_H

#include "QMCDrivers/WFTrain/MatrixFreeLinearMethod.h"

namespace qmcplusplus::wftrain
{
/// Bound outer Ritz work, inner GMRES work, and the trusted-vector restart.
struct AcceleratedLinearMethodControl
{
  MatrixFreeLinearMethodControl outer;
  std::size_t inner_iterations = 12;
  std::size_t inner_restart_length = 6;
  DerivativeReal inner_relative_tolerance = 0.2;
  std::size_t retained_ritz_vectors = 2;
};

/** Bounded real nonsymmetric generalized Jacobi--Davidson solve.
 * H and S must have matching GLOBAL REAL64 identities; only S must be Hermitian.
 * The canonical reference initial vector defines E0=H00/S00. DGGEV selects the
 * real, finite, reference-connected Ritz root nearest E0, rejecting complex roots,
 * nonpositive overlap norms and small c0. Inner right-preconditioned restarted
 * GMRES solves (I-Su*u^T)(H-theta*S)(I-u*u^T*S)t=-r for S-normalized u.
 * The preconditioner acts on tangent coordinates, with an identity reference block.
 * Both H and S applications, including inner solves, share the outer action/time
 * budgets. Storage is O((m+k)P + m^2+k^2), for fixed outer/inner restart bounds.
 * Unconverged results are diagnostics and must not be published as updates.
 */
MatrixFreeLinearMethodResult solveAcceleratedMatrixFreeLinearMethod(
    const AugmentedLinearMethodOperator& hamiltonian,
    const AugmentedLinearMethodOperator& overlap,
    const MatrixFreePreconditioner& preconditioner,
    DerivativeArrayView<const DerivativeValue> initial_vector,
    const AcceleratedLinearMethodControl& control = {});
} // namespace qmcplusplus::wftrain
#endif
