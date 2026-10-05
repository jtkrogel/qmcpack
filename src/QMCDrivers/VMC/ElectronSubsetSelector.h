//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_ELECTRONSUBSETSELECTOR_H
#define QMCPLUSPLUS_ELECTRONSUBSETSELECTOR_H

#include <vector>

#include "Configuration.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "Utilities/RandomBase.h"

namespace qmcplusplus
{
/** Select a canonical subset of electron indices for an n-electron VMC move.
 *
 * Random selection is uniform without replacement. Cyclic selection consumes a
 * contiguous stream of indices, advancing by electrons_per_move on each call.
 * Wrapped cyclic subsets are sorted before being returned, so every result is
 * strictly increasing and contains no duplicates.
 */
class ElectronSubsetSelector
{
public:
  using IndexType             = QMCTraits::IndexType;
  using FullPrecisionRealType = QMCTraits::FullPrecRealType;
  using Selection             = VMCDriverInput::ElectronSelection;

  ElectronSubsetSelector(IndexType electron_count, IndexType electrons_per_move, Selection selection);

  /** Return the next sorted, unique subset.
   *
   * The reference remains valid until the next call to select on this object.
   * The random number generator is not consumed for cyclic selection.
   */
  const std::vector<IndexType>& select(RandomBase<FullPrecisionRealType>& random_gen);

  /// Restart cyclic selection at electron zero. Random selection has no cursor.
  void reset() noexcept { cyclic_cursor_ = 0; }

  IndexType getElectronCount() const noexcept { return electron_count_; }
  IndexType getElectronsPerMove() const noexcept { return electrons_per_move_; }
  Selection getSelection() const noexcept { return selection_; }

private:
  IndexType electron_count_;
  IndexType electrons_per_move_;
  Selection selection_;
  IndexType cyclic_cursor_ = 0;
  std::vector<IndexType> scratch_indices_;
  std::vector<IndexType> selected_indices_;
};

} // namespace qmcplusplus

#endif
