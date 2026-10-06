//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Peter W. Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_NONLOCAL_ECPOTENTIAL_H
#define QMCPLUSPLUS_NONLOCAL_ECPOTENTIAL_H

#include <memory>
#include <utility>

#include "Configuration.h"
#include "ForceBase.h"
#include "OperatorBase.h"
#include "NeighborListsForPseudo.h"
#include "type_traits/OptionalRef.hpp"

namespace qmcplusplus
{
class NonLocalECPComponent;
class NLPPVirtualBatchStorage;
template<typename T>
struct NLPPJob;

namespace testing
{
class TestNonLocalECPotential;
}

/** @ingroup hamiltonian
 * \brief Evaluate the semi local potentials
 */
class NonLocalECPotential : public OperatorBase, public ForceBase
{
  using Real = QMCTraits::RealType;

  struct MultiWalkerResourceIdentity
  {};

  struct MultiWalkerResourceSchema
  {
    bool uses_virtual_particles;
    std::size_t virtual_particle_distance_tables;
    std::size_t electron_count;
    std::size_t electron_group_count;
    std::size_t ion_count;
    std::size_t outer_tile_capacity;

    bool operator==(const MultiWalkerResourceSchema& other) const noexcept
    {
      return uses_virtual_particles == other.uses_virtual_particles &&
          virtual_particle_distance_tables == other.virtual_particle_distance_tables &&
          electron_count == other.electron_count && electron_group_count == other.electron_group_count &&
          ion_count == other.ion_count && outer_tile_capacity == other.outer_tile_capacity;
    }

    bool operator!=(const MultiWalkerResourceSchema& other) const noexcept { return !(*this == other); }
  };

  /** Focused accounting for the private flattened derivative transaction. */
  struct MultiWalkerDerivativeStatistics
  {
    std::size_t tiles_packed;
    std::size_t split_job_continuations;
    std::size_t max_tile_occupancy;
    std::size_t derivative_staging_size;
    std::size_t derivative_staging_capacity;
    std::size_t bounded_weight_size;
    std::size_t bounded_weight_capacity;
  };

  struct NonLocalECPotentialMultiWalkerResource;

public:
  NonLocalECPotential(ParticleSet& ions, ParticleSet& els, bool enable_DLA, bool use_VP);
  NonLocalECPotential(const NonLocalECPotential& nlpp, ParticleSet& els);
  ~NonLocalECPotential() override;

  std::string getClassName() const override { return "NonLocalECPotential"; }

#if !defined(REMOVE_TRACEMANAGER)
  void contributeParticleQuantities() override;
  void checkoutParticleQuantities(TraceManager& tm) override;
  void deleteParticleQuantities() override;
#endif

  Return_t evaluate(TrialWaveFunction& psi, ParticleSet& P) override;
  Return_t evaluateDeterministic(TrialWaveFunction& psi, ParticleSet& P) override;
  void mw_evaluate(const RefVectorWithLeader<OperatorBase>& o_list,
                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                   const RefVectorWithLeader<ParticleSet>& p_list) const override;

  Return_t evaluateWithToperator(TrialWaveFunction& psi, ParticleSet& P) override;

  void mw_evaluateWithToperator(const RefVectorWithLeader<OperatorBase>& o_list,
                                const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                const RefVectorWithLeader<ParticleSet>& p_list) const override;

  void mw_evaluatePerParticle(const RefVectorWithLeader<OperatorBase>& o_list,
                              const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                              const RefVectorWithLeader<ParticleSet>& p_list,
                              const std::vector<ListenerVector<Real>>& listeners,
                              const std::vector<ListenerVector<Real>>& listeners_ions) const override;

  void mw_evaluatePerParticleWithToperator(const RefVectorWithLeader<OperatorBase>& o_list,
                                           const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                           const std::vector<ListenerVector<Real>>& listeners,
                                           const std::vector<ListenerVector<Real>>& listeners_ions) const override;

  void evaluateIonDerivs(ParticleSet& P,
                         ParticleSet& ions,
                         TrialWaveFunction& psi,
                         ParticleSet::ParticlePos& hf_terms,
                         ParticleSet::ParticlePos& pulay_terms) override;

  void evaluateOneBodyOpMatrix(ParticleSet& P, const TWFFastDerivWrapper& psi, std::vector<ValueMatrix>& B) override;

  void evaluateOneBodyOpMatrixForceDeriv(ParticleSet& P,
                                         ParticleSet& source,
                                         const TWFFastDerivWrapper& psi,
                                         const int iat,
                                         std::vector<std::vector<ValueMatrix>>& Bforce) override;


  /** make non local moves with particle-by-particle moves
   * @param psi trial wavefunction
   * @param P particle set
   * @return the number of accepted moves
   */
  int makeNonLocalMovesPbyP(TrialWaveFunction& psi, ParticleSet& P, NonLocalTOperator& move_op) override;

  /** make non local moves for a batch of walkers
   *
   * For TmoveKind::V1 the candidate-ratio evaluations of each electron are
   * batched across walkers.  The VP path flattens every current-electron job
   * into bounded outer tiles; the non-VP path retains the legacy job
   * wavefront.  Move selection and the (rare) accepts stay per walker, so each
   * walker reproduces the sequential single-walker v1 sweep, including its
   * RNG draw order. Other T-move kinds fall back to the per-walker
   * implementation.
   *
   * @return the number of accepted moves per walker
   */
  static std::vector<int> mw_makeNonLocalMovesPbyP(const RefVectorWithLeader<OperatorBase>& o_list,
                                                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                   const RefVectorWithLeader<ParticleSet>& p_list,
                                                   NonLocalTOperator& move_op);

  Return_t evaluateValueAndDerivatives(TrialWaveFunction& psi,
                                       ParticleSet& P,
                                       const OptVariables& optvars,
                                       const Vector<ValueType>& dlogpsi,
                                       Vector<ValueType>& dhpsioverpsi) override;

  /** Evaluate nonlocal energies and weighted parameter derivatives for a walker batch.
   *
   * Virtual-particle jobs are compacted across walkers and avoid the legacy
   * knot-by-parameter matrix when a component supplies a direct weighted sink.
   */
  void mw_evaluateWithParameterDerivatives(const RefVectorWithLeader<OperatorBase>& o_list,
                                           const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                           const OptVariables& optvars,
                                           const RecordArray<ValueType>& dlogpsi,
                                           RecordArray<ValueType>& dhpsioverpsi) const override;

  /** Do nothing */
  bool put(xmlNodePtr cur) override { return true; }

  bool get(std::ostream& os) const override
  {
    os << "NonLocalECPotential: " << IonConfig.getName();
    return true;
  }

  /** initialize a shared resource and hand it to a collection
   */
  void createResource(ResourceCollection& collection) const override;

  /** acquire a shared resource from a collection
   */
  void acquireResource(ResourceCollection& collection, const RefVectorWithLeader<OperatorBase>& o_list) const override;

  /** return a shared resource to a collection
   */
  void releaseResource(ResourceCollection& collection, const RefVectorWithLeader<OperatorBase>& o_list) const override;

  std::unique_ptr<OperatorBase> makeClone(ParticleSet& qp, TrialWaveFunction& psi) const override;

  void addComponent(int groupID, std::unique_ptr<NonLocalECPComponent>&& pp);

  /** set the internal RNG pointer as the given pointer
   * @param rng input RNG pointer
   */
  void setRandomGenerator(RandomBase<FullPrecRealType>* rng) override { myRNG = rng; }

protected:
  /** the actual implementation for batched walkers, used by mw_evaluate, mw_evaluateWithToperator
   *  mw_evaluatePerPaticleWithToperator
   * @param o_list     the list of NonLocalECPotential in a walker batch
   * @param wf_list    the list of TrialWaveFunction in a walker batch
   * @param p_list     the list of ParticleSet in a walker batch
   * @param compute_txy_all whether to compute Txy for all the electrons affected by NLPP
   * @param listeners  optional listeners which allow per particle and reduced to share impl
   */
  static void mw_evaluateImpl(const RefVectorWithLeader<OperatorBase>& o_list,
                              const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                              const RefVectorWithLeader<ParticleSet>& p_list,
                              bool compute_txy_all,
                              std::optional<ListenerOption<Real>> listeners,
                              bool keepGrid = false);

  ///random number generator
  RandomBase<FullPrecRealType>* myRNG;
  ///the set of local-potentials (one for each ion)
  std::vector<NonLocalECPComponent*> PP;
  ///unique NonLocalECPComponent to remove
  std::vector<std::unique_ptr<NonLocalECPComponent>> PPset;
  ///reference to the center ion
  ParticleSet& IonConfig;
  ///true, determinant localization approximation(DLA) is enabled
  bool use_DLA;

private:
  /// Conservative explicit Task 08 outer capacity; Task 09 owns runtime policy.
  static constexpr std::size_t default_outer_tile_capacity_ = 256;

  /// Immutable identity shared only by clones belonging to this operator family.
  const std::shared_ptr<const MultiWalkerResourceIdentity> mw_resource_identity_;
  /// Maximum number of virtual knots held by the crowd-local outer tile.
  std::size_t outer_tile_capacity_ = default_outer_tile_capacity_;
  ///virtual particle set
  const std::unique_ptr<VirtualParticleSet> vp_;
  ///index of distance table for the ion-el pair
  int myTableIndex;
  ///reference to the electrons
  ParticleSet& Peln;
  ///neighbor lists for marking electrons touched by T-moves
  NeighborListsForPseudo neighbor_lists;
  ///ture if an electron is affected by other electrons moved by T-moves
  std::vector<bool> elecTMAffected;
  ///Pulay force vector
  ParticleSet::ParticlePos PulayTerm;
  /// Tmove data collected for all the electrons
  std::vector<NonLocalData> tmove_xy_all_;
#if !defined(REMOVE_TRACEMANAGER)
  ///single particle trace samples

  Array<TraceReal, 1>* Ve_sample;
  Array<TraceReal, 1>* Vi_sample;
#endif
  ///NLPP job list of ion-electron pairs by spin group
  std::vector<std::vector<NLPPJob<Real>>> nlpp_jobs;
  /// mult walker shared resource
  ResourceHandle<NonLocalECPotentialMultiWalkerResource> mw_res_handle_;

  /// Describe every shape property needed by the current multi-walker resource schema.
  MultiWalkerResourceSchema multiWalkerResourceSchema() const noexcept;

  /// Focused resource-clone diagnostics used by the ownership regression.
  void resizeMultiWalkerListenerScratchForTesting(std::size_t walkers,
                                                  std::size_t electrons,
                                                  std::size_t ions);
  std::pair<std::size_t, std::size_t> multiWalkerListenerScratchSizesForTesting() const;
  MultiWalkerDerivativeStatistics multiWalkerDerivativeStatisticsForTesting() const;

  /// Select a small deterministic capacity before resource creation in focused tests.
  void setOuterTileCapacityForTesting(std::size_t capacity);

  /** Evaluate supported VP modes through one staged, flattened outer-tile transaction. */
  static void mw_evaluateImplFlattenedVP(const RefVectorWithLeader<OperatorBase>& o_list,
                                         const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         bool compute_txy_all,
                                         const std::optional<ListenerOption<Real>>& listeners,
                                         bool keep_grid);

  /** Evaluate VP nonlocal energies and parameter derivatives as one tiled transaction. */
  static void mw_evaluateWithParameterDerivativesFlattenedVP(
      const RefVectorWithLeader<OperatorBase>& o_list,
      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const OptVariables& optvars,
      RecordArray<ValueType>& dhpsioverpsi);

  /** Consume all packed outer tiles belonging to one already prepared electron group. */
  static void mw_consumeFlattenedVPPreparedGroup(
      NonLocalECPotentialMultiWalkerResource& resource,
      const RefVectorWithLeader<OperatorBase>& o_list,
      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      int group,
      bool compute_txy_all,
      bool accumulate_pair_results,
      const std::optional<ListenerOption<Real>>& listeners,
      bool& have_reference_stamps,
      bool& have_nonfermionic_reference_stamps,
      bool& tile_available);

  /** Consume derivative tiles for one prepared group into private request staging. */
  static void mw_consumeFlattenedVPDerivativePreparedGroup(
      NonLocalECPotentialMultiWalkerResource& resource,
      const RefVectorWithLeader<OperatorBase>& o_list,
      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const OptVariables& optvars,
      int group,
      bool& have_reference_stamps,
      bool& tile_available);

  /** Build one V1 electron's VP candidates without crossing its selection boundary. */
  static NLPPVirtualBatchStorage& mw_evaluateV1ElectronCandidates(
      const RefVectorWithLeader<OperatorBase>& o_list,
      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      int group,
      int electron);

  /** Size reusable VP V1 metadata and component scratch before an electron sweep. */
  static void mw_prepareV1FlattenedVPResource(const RefVectorWithLeader<OperatorBase>& o_list,
                                               const RefVectorWithLeader<ParticleSet>& p_list);

  /** the actual implementation, used by evaluate and evaluateWithToperator
   * @param P particle set
   * @param compute_txy_all whether to compute Txy for all the electrons affected by NLPP
   * @param keepGrid.  If true, does not randomize the quadrature grid before evaluation.  
   */
  void evaluateImpl(TrialWaveFunction& psi, ParticleSet& P, bool compute_txy_all, bool keepGrid = false);

  /** compute the T move transition probability for a given electron
   * member variable nonLocalOps.Txy is updated
   * @param psi trial wavefunction
   * @param P particle set
   * @param ref_elec reference electron id
   * @param tmove_xy off-diagonal terms for one electron.
   */
  void computeOneElectronTxy(TrialWaveFunction& psi,
                             ParticleSet& P,
                             const int ref_elec,
                             std::vector<NonLocalData>& tmove_xy);

  friend class testing::TestNonLocalECPotential;
};
} // namespace qmcplusplus
#endif
