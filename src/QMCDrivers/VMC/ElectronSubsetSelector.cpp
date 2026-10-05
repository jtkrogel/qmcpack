//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "ElectronSubsetSelector.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace qmcplusplus
{
ElectronSubsetSelector::ElectronSubsetSelector(IndexType electron_count,
                                               IndexType electrons_per_move,
                                               Selection selection)
    : electron_count_(electron_count), electrons_per_move_(electrons_per_move), selection_(selection)
{
  if (electron_count_ <= 0)
    throw std::invalid_argument("ElectronSubsetSelector requires a positive electron count.");
  if (electrons_per_move_ <= 0 || electrons_per_move_ > electron_count_)
    throw std::invalid_argument(
        "ElectronSubsetSelector electrons_per_move must be in the range [1, electron_count].");
  if (selection_ != Selection::RANDOM && selection_ != Selection::CYCLIC)
    throw std::invalid_argument("ElectronSubsetSelector received an invalid selection policy.");

  scratch_indices_.resize(electron_count_);
  selected_indices_.resize(electrons_per_move_);
}

const std::vector<ElectronSubsetSelector::IndexType>& ElectronSubsetSelector::select(
    RandomBase<FullPrecisionRealType>& random_gen)
{
  switch (selection_)
  {
  case Selection::RANDOM:
    std::iota(scratch_indices_.begin(), scratch_indices_.end(), IndexType{0});
    for (IndexType selected = 0; selected < electrons_per_move_; ++selected)
    {
      const FullPrecisionRealType random_value = random_gen();
      if (!(random_value >= FullPrecisionRealType{0} && random_value < FullPrecisionRealType{1}))
        throw std::domain_error("ElectronSubsetSelector requires random values in the half-open interval [0, 1).");

      const IndexType remaining = electron_count_ - selected;
      const IndexType offset    = std::min(static_cast<IndexType>(random_value * remaining), remaining - 1);
      const IndexType chosen    = selected + offset;
      std::swap(scratch_indices_[selected], scratch_indices_[chosen]);
      selected_indices_[selected] = scratch_indices_[selected];
    }
    break;
  case Selection::CYCLIC:
  {
    IndexType next_index = cyclic_cursor_;
    for (IndexType selected = 0; selected < electrons_per_move_; ++selected)
    {
      selected_indices_[selected] = next_index;
      next_index                  = next_index + 1 == electron_count_ ? 0 : next_index + 1;
    }
    cyclic_cursor_ = next_index;
    break;
  }
  default:
    throw std::logic_error("ElectronSubsetSelector received an invalid selection policy.");
  }

  std::sort(selected_indices_.begin(), selected_indices_.end());
  return selected_indices_;
}

} // namespace qmcplusplus
