//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2025 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_NEIGHBORLIST_H
#define QMCPLUSPLUS_NEIGHBORLIST_H

#include <vector>
#include "NonLocalECPComponent.h"
#include <DistanceTable.h>

namespace qmcplusplus
{
class NeighborLists
{
  /// neighboring particle IDs of all the reference particles
  std::vector<std::vector<int>> neighborIDs_;

public:
  /** constructor
   * @param num_ref number of reference particles
   */
  NeighborLists(size_t num_ref) : neighborIDs_(num_ref) {}

  /// get the number of neignbor lists / reference particles.
  size_t size() const { return neighborIDs_.size(); }
  /// get the neighbor list of the source particle
  std::vector<int>& getNeighborList(int source) { return neighborIDs_[source]; }
  const std::vector<int>& getNeighborList(int source) const { return neighborIDs_[source]; }

  /// Exchange identically shaped owned list storage without allocating.
  void swap(NeighborLists& other) noexcept { neighborIDs_.swap(other.neighborIDs_); }
};

class NeighborListsForPseudo
{
public:
  /** Independently owned neighbor-list state suitable for transactional staging. */
  class OwnedLists
  {
    friend class NeighborListsForPseudo;

    OwnedLists(size_t num_elecs, size_t num_ions)
        : elec_neighbor_ions_(num_elecs), ion_neighbor_elecs_(num_ions)
    {}

    NeighborLists elec_neighbor_ions_;
    NeighborLists ion_neighbor_elecs_;

  public:
    OwnedLists(OwnedLists&&) noexcept            = default;
    OwnedLists& operator=(OwnedLists&&) noexcept = default;
    OwnedLists(const OwnedLists&)                = delete;
    OwnedLists& operator=(const OwnedLists&)     = delete;

    /// Remove all logical entries while retaining owned capacities.
    void clear();

    /// Add one electron--ion pair to both directional lists.
    void addElecIonPair(int jel, int iat);

    /// Read the staged neighboring ions for one electron.
    const std::vector<int>& getNeighboringIons(int jel) const
    {
      return elec_neighbor_ions_.getNeighborList(jel);
    }

    /// Read the staged neighboring electrons for one ion.
    const std::vector<int>& getNeighboringElectrons(int iat) const
    {
      return ion_neighbor_elecs_.getNeighborList(iat);
    }
  };

private:
  ///neighborlist of electrons
  NeighborLists elec_neighbor_ions_;
  ///neighborlist of ions
  NeighborLists ion_neighbor_elecs_;
  ///the set of local-potentials (one for each ion)
  const std::vector<NonLocalECPComponent*>& PP;

public:
  NeighborListsForPseudo(size_t num_elecs, size_t num_ions, const std::vector<NonLocalECPComponent*>& pp);

  NeighborListsForPseudo(const NeighborListsForPseudo&)            = delete;
  NeighborListsForPseudo& operator=(const NeighborListsForPseudo&) = delete;

  /// Create empty independently owned state with the same electron/ion shape.
  OwnedLists makeOwnedLists() const;

  /// Validate staged list shape before entering a no-throw publication phase.
  void validateOwnedLists(const OwnedLists& lists) const;

  /** Publish owned list state without allocation.
   * @return true on success; false leaves both objects unchanged when shapes differ.
   */
  bool swapOwnedLists(OwnedLists& lists) noexcept;

  /// Test whether this object refers to the expected pseudopotential vector.
  bool isBoundTo(const std::vector<NonLocalECPComponent*>& pp) const noexcept { return &PP == &pp; }

  /// get the neighboring ion list of a given electron
  const std::vector<int>& getNeighboringIons(int jel) const { return elec_neighbor_ions_.getNeighborList(jel); }

  /// get the neighboring electron list of a given ion
  const std::vector<int>& getNeighboringElectrons(int iat) const { return ion_neighbor_elecs_.getNeighborList(iat); }

  /** mark all the electrons affected by T-moves and update elec_neighbor_ions_ and ion_neighbor_elecs_
   * @param myTable electron ion distance table
   * @param iel reference electron
   * Note this function should be called before acceptMove for a Tmove
   */
  void markAffectedElecs(const DistanceTableAB& myTable, int iel, std::vector<bool>& elecTMAffected);

  /// clear all the electron and ion neighbor lists
  void clear();

  /** add electron ion pair to the neighbor lists
   * @param jel electron index
   * @param iat ion index
   */
  void addElecIonPair(int jel, int iat);
};

} // namespace qmcplusplus
#endif
