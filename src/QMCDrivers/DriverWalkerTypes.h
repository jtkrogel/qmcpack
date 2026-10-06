//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/**
 * @file
 * Driver level walker (DriverWalker) related data structures.
 * Unlike MCWalkerConfiguration which only holds electron positions and weights.
 * Driver level walker includes all the per-walker data structures which depends on the type of driver
 */

#ifndef QMCPLUSPLUS_DRIVERWALKERTYPES_H
#define QMCPLUSPLUS_DRIVERWALKERTYPES_H

#include <ResourceCollection.h>
#include "Utilities/BatchResourcePreparation.h"

namespace qmcplusplus
{
/** DriverWalker multi walker resource collections
 * It currently supports VMC and DMC only
 */
struct DriverWalkerResourceCollection
{
  ResourceCollection pset_res;
  ResourceCollection twf_res;
  ResourceCollection ham_res;

  DriverWalkerResourceCollection() : pset_res("ParticleSet"), twf_res("TrialWaveFunction"), ham_res("Hamiltonian") {}

  /** Prepare all three collection families as one all-or-nothing publication. */
  void prepareBatchResources(const BatchResourcePreparationContext& context)
  {
    // Construct every candidate before publishing any of them.  This preserves
    // all original resources when cloning or any preparation hook fails.
    ResourceCollection prepared_pset = pset_res.makePreparedBatchResources(context);
    ResourceCollection prepared_twf  = twf_res.makePreparedBatchResources(context);
    ResourceCollection prepared_ham  = ham_res.makePreparedBatchResources(context);

    pset_res.swapResourceStorage(prepared_pset);
    twf_res.swapResourceStorage(prepared_twf);
    ham_res.swapResourceStorage(prepared_ham);
  }
};
} // namespace qmcplusplus
#endif
