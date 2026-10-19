//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_ORBITAL_LOSS_NORMALIZATION_H
#define QMCPLUSPLUS_ORBITAL_LOSS_NORMALIZATION_H

namespace qmcplusplus::wftrain
{
/// Preserve the legacy sum of spin-sector means or average every full-matrix entry.
enum class OrbitalLossNormalization { SPIN_SECTOR_SUM, FULL_MATRIX_MEAN };
}
#endif
