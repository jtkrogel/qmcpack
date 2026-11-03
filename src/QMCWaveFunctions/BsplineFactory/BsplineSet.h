//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/** @file BsplineSet.h
 *
 * BsplineSet is a SPOSet derived class and serves as a base class for B-spline SPO C2C/C2R/R2R implementation
 */
#ifndef QMCPLUSPLUS_BSPLINESET_H
#define QMCPLUSPLUS_BSPLINESET_H

#include "QMCWaveFunctions/SPOSet.h"
#include "QMCWaveFunctions/BandInfo.h"
#include <complex>
#include <string>

namespace qmcplusplus
{

template<class ST>
class SplineSetReader;
template<class ST>
class HybridRepSetReader;

/** BsplineSet is the base class for SplineC2C, SplineC2R, SplineR2R.
 * Its derived template classes manage the storage and evaluation at given precision.
 * BsplineSet also implements a few fallback routines in case optimized implementation is not necessary in the derived class.
 */
class BsplineSet : public SPOSet
{
public:
  enum class MFRCoreActionPart
  {
    COMPLEX,
    REAL,
    IMAGINARY
  };

  struct MFRCoreActionOrbitalSource
  {
    int spin = 0;
    int twist = 0;
    int band = -1;
    int distinct_orbital = -1;
    MFRCoreActionPart part = MFRCoreActionPart::REAL;
    PosType qmc_k_cart{0.0};
    std::complex<double> rotation_phase{1.0, 0.0};
  };

protected:
  static const int D = DIM;
  ///primitive cell
  const Lattice prim_lattice_;
  ///sign bits at the G/2 boundaries
  TinyVector<int, D> HalfG;
  ///flags to unpack sin/cos
  std::vector<bool> MakeTwoCopies;
  /** kpoints for each unique orbitals.
   * Note: for historic reason, this sign is opposite to what was used in DFT when orbitals were generated.
   * Changing the sign requires updating all the evaluation code.
   */
  std::vector<SPOSet::PosType> kPoints;
  ///remap splines to orbitals
  aligned_vector<int> BandIndexMap;
  std::vector<MFRCoreActionOrbitalSource> mfr_core_action_sources_;
  TinyVector<int, 3> mfr_core_action_mesh_{0, 0, 0};
  TinyVector<int, 3> mfr_core_action_half_g_{0, 0, 0};
  bool mfr_core_action_use_duplex_ = false;
  bool mfr_core_action_rotate_ = true;
  std::string mfr_core_action_orbital_file_;
  const ParticleSet* mfr_core_action_ions_ = nullptr;

public:
  BsplineSet(const std::string& my_name, size_t size, const Lattice& prim_lattice)
      : SPOSet(my_name, size), prim_lattice_(prim_lattice)
  {}

  virtual std::string getKeyword() const = 0;

  auto& getHalfG() const { return HalfG; }

  /** Record the source-state ordering used by the selected B-spline SPOs.
   *  Real builds can split one complex state into its real and imaginary
   *  time-reversal pair; this mapping preserves that exact order.
   */
  void setMFRCoreActionSources(const BandInfoGroup& bandgroup,
                               int spin,
                               bool use_duplex_splines,
                               bool rotate,
                               const TinyVector<int, 3>& mesh_size,
                               const TinyVector<int, 3>& half_g,
                               std::string orbital_file,
                               const ParticleSet* ions = nullptr)
  {
    mfr_core_action_ions_ = ions;
    mfr_core_action_sources_.clear();
    mfr_core_action_mesh_ = mesh_size;
    mfr_core_action_half_g_ = half_g;
    mfr_core_action_use_duplex_ = use_duplex_splines;
    mfr_core_action_rotate_ = rotate;
    mfr_core_action_orbital_file_ = std::move(orbital_file);

    const int num_distinct = bandgroup.getNumDistinctOrbitals();
    for (int distinct = 0; distinct < num_distinct; ++distinct)
    {
      const BandInfo& source_band = bandgroup.myBands[BandIndexMap[distinct]];
      MFRCoreActionOrbitalSource source;
      source.spin = spin;
      source.twist = source_band.TwistIndex;
      source.band = source_band.BandIndex;
      source.distinct_orbital = distinct;
#if defined(QMC_COMPLEX)
      source.part = MFRCoreActionPart::COMPLEX;
      source.qmc_k_cart = kPoints[distinct];
      mfr_core_action_sources_.push_back(source);
#else
      if (use_duplex_splines && MakeTwoCopies[distinct])
      {
        source.part = MFRCoreActionPart::REAL;
        source.qmc_k_cart = kPoints[distinct];
        mfr_core_action_sources_.push_back(source);
        source.part = MFRCoreActionPart::IMAGINARY;
        mfr_core_action_sources_.push_back(source);
      }
      else
      {
        source.part = MFRCoreActionPart::REAL;
        source.qmc_k_cart = use_duplex_splines ? kPoints[distinct] : PosType(0.0);
        mfr_core_action_sources_.push_back(source);
      }
#endif
    }
    if (mfr_core_action_sources_.size() != OrbitalSetSize)
      throw std::runtime_error("B-spline MFR action source map does not match the SPO count");
  }

  const std::vector<MFRCoreActionOrbitalSource>& getMFRCoreActionSources() const
  { return mfr_core_action_sources_; }
  std::vector<MFRCoreActionOrbitalSource>& getMFRCoreActionSources() { return mfr_core_action_sources_; }
  const ParticleSet* getMFRCoreActionIons() const { return mfr_core_action_ions_; }
  const std::string& getMFRCoreActionOrbitalFile() const { return mfr_core_action_orbital_file_; }
  void setMFRCoreActionRotationPhase(int orbital, const std::complex<double>& phase)
  { mfr_core_action_sources_.at(orbital).rotation_phase = phase; }
  const TinyVector<int, 3>& getMFRCoreActionMesh() const { return mfr_core_action_mesh_; }
  const TinyVector<int, 3>& getMFRCoreActionHalfG() const { return mfr_core_action_half_g_; }
  bool usesDuplexSplinesForMFRCoreAction() const { return mfr_core_action_use_duplex_; }
  bool rotatesOrbitalsForMFRCoreAction() const { return mfr_core_action_rotate_; }

  inline void init_base(int n)
  {
    kPoints.resize(n);
    MakeTwoCopies.resize(n);
    BandIndexMap.resize(n);
    for (int i = 0; i < n; i++)
      BandIndexMap[i] = i;
  }

  /// resize vectors related to spline evaluaton results.
  virtual void resizeStorage(size_t n) = 0;

  /** remap kPoints to pack the double copy */
  virtual void resize_kpoints() {}

  ///remap kpoints to group general kpoints & special kpoints
  int remap_kpoints()
  {
    std::vector<SPOSet::PosType> k_copy(kPoints);
    const int nk = kPoints.size();
    int nCB      = 0;
    //two pass
    for (int i = 0; i < nk; ++i)
    {
      if (MakeTwoCopies[i])
      {
        kPoints[nCB]        = k_copy[i];
        BandIndexMap[nCB++] = i;
      }
    }
    int nRealBands = nCB;
    for (int i = 0; i < nk; ++i)
    {
      if (!MakeTwoCopies[i])
      {
        kPoints[nRealBands]        = k_copy[i];
        BandIndexMap[nRealBands++] = i;
      }
    }
    return nCB; //return the number of complex bands
  }

  // propagate SPOSet virtual functions
  using SPOSet::evaluateDetRatios;
  using SPOSet::evaluateValue;
  using SPOSet::evaluateVGH;
  using SPOSet::evaluateVGHGH;
  using SPOSet::evaluateVGL;
  using SPOSet::finalizeConstruction;
  using SPOSet::mw_evaluateDetRatios;
  using SPOSet::mw_evaluateVGL;
  using SPOSet::mw_evaluateVGLandDetRatioGrads;

  using SPOSet::acquireResource;
  using SPOSet::createResource;
  using SPOSet::releaseResource;

  std::unique_ptr<SPOSet> makeClone() const override = 0;

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix& logdet,
                            GradMatrix& dlogdet,
                            ValueMatrix& d2logdet) override
  {
    using value_type = ValueMatrix::value_type;
    using grad_type  = GradMatrix::value_type;
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      ValueVector v(logdet[i], logdet.cols());
      GradVector g(dlogdet[i], dlogdet.cols());
      ValueVector l(d2logdet[i], d2logdet.cols());
      evaluateVGL(P, iat, v, g, l);
    }
  }

  void mw_evaluate_notranspose(const RefVectorWithLeader<SPOSet>& spo_list,
                               const RefVectorWithLeader<ParticleSet>& P_list,
                               int first,
                               int last,
                               const RefVector<ValueMatrix>& logdet_list,
                               const RefVector<GradMatrix>& dlogdet_list,
                               const RefVector<ValueMatrix>& d2logdet_list) const override
  {
    assert(this == &spo_list.getLeader());
    using value_type = ValueMatrix::value_type;
    using grad_type  = GradMatrix::value_type;

    const size_t nw = spo_list.size();
    std::vector<ValueVector> mw_psi_v;
    std::vector<GradVector> mw_dpsi_v;
    std::vector<ValueVector> mw_d2psi_v;
    RefVector<ValueVector> psi_v_list;
    RefVector<GradVector> dpsi_v_list;
    RefVector<ValueVector> d2psi_v_list;
    mw_psi_v.reserve(nw);
    mw_dpsi_v.reserve(nw);
    mw_d2psi_v.reserve(nw);
    psi_v_list.reserve(nw);
    dpsi_v_list.reserve(nw);
    d2psi_v_list.reserve(nw);

    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      mw_psi_v.clear();
      mw_dpsi_v.clear();
      mw_d2psi_v.clear();
      psi_v_list.clear();
      dpsi_v_list.clear();
      d2psi_v_list.clear();

      for (int iw = 0; iw < nw; iw++)
      {
        mw_psi_v.emplace_back(logdet_list[iw].get()[i], logdet_list[iw].get().cols());
        mw_dpsi_v.emplace_back(dlogdet_list[iw].get()[i], dlogdet_list[iw].get().cols());
        mw_d2psi_v.emplace_back(d2logdet_list[iw].get()[i], d2logdet_list[iw].get().cols());
        psi_v_list.push_back(mw_psi_v.back());
        dpsi_v_list.push_back(mw_dpsi_v.back());
        d2psi_v_list.push_back(mw_d2psi_v.back());
      }

      mw_evaluateVGL(spo_list, P_list, iat, psi_v_list, dpsi_v_list, d2psi_v_list);
    }
  }

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix& logdet,
                            GradMatrix& dlogdet,
                            HessMatrix& grad_grad_logdet) override
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      ValueVector v(logdet[i], logdet.cols());
      GradVector g(dlogdet[i], dlogdet.cols());
      HessVector h(grad_grad_logdet[i], grad_grad_logdet.cols());
      evaluateVGH(P, iat, v, g, h);
    }
  }

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix& logdet,
                            GradMatrix& dlogdet,
                            HessMatrix& grad_grad_logdet,
                            GGGMatrix& grad_grad_grad_logdet) override
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      ValueVector v(logdet[i], logdet.cols());
      GradVector g(dlogdet[i], dlogdet.cols());
      HessVector h(grad_grad_logdet[i], grad_grad_logdet.cols());
      GGGVector gh(grad_grad_grad_logdet[i], grad_grad_grad_logdet.cols());
      evaluateVGHGH(P, iat, v, g, h, gh);
    }
  }

  void evaluateGradSource(const ParticleSet& P,
                          int first,
                          int last,
                          const ParticleSet& source,
                          int iat_src,
                          GradMatrix& gradphi) override
  {
    //Do nothing, since Einsplines don't explicitly depend on ion positions.
  }

  void evaluateGradSource(const ParticleSet& P,
                          int first,
                          int last,
                          const ParticleSet& source,
                          int iat_src,
                          GradMatrix& grad_phi,
                          HessMatrix& grad_grad_phi,
                          GradMatrix& grad_lapl_phi) override
  {
    //Do nothing, since Einsplines don't explicitly depend on ion positions.
  }

  template<class ST>
  friend class SplineSetReader;
  template<class ST>
  friend class HybridRepSetReader;
  friend class BsplineReader;
};

} // namespace qmcplusplus
#endif
