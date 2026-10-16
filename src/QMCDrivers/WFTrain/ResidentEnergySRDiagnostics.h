//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_RESIDENT_ENERGY_SR_DIAGNOSTICS_H
#define QMCPLUSPLUS_RESIDENT_ENERGY_SR_DIAGNOSTICS_H
#include "QMCDrivers/WFTrain/KrylovSolver.h"

namespace qmcplusplus::wftrain
{
/// Scalar SR completion, without pretending that resident vectors are host results.
struct ResidentEnergySRDiagnostics
{
  KrylovSolveDiagnostics solve;
  double gradient_norm = 0, direction_norm = 0, metric_norm = 0;
};
}
#endif
