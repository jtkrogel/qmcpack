//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_MFR_POTENTIAL_H
#define QMCPLUSPLUS_MFR_POTENTIAL_H

#include <memory>
#include <string>
#include <vector>

#include "QMCHamiltonians/OperatorBase.h"
#include "einspline/bspline.h"

namespace qmcplusplus
{
/** Positive mean-field electron-electron potential, V_H + V_xc, read from QE. */
class MFRPotential : public OperatorDependsOnlyOnParticleSet
{
public:
  explicit MFRPotential(ParticleSet& pset);

  std::string getClassName() const override { return "MFRPotential"; }
  bool put(xmlNodePtr cur) override;
  bool get(std::ostream& os) const override;
  Return_t evaluate(ParticleSet& pset) override;
  void addObservables(PropertySetType& plist, BufferType& collectables) override;
  void registerObservables(std::vector<ObservableHelper>& h5desc, hdf_archive& file) const override;
  void setObservables(PropertySetType& plist) override;
  void setParticlePropertyList(PropertySetType& plist, int offset) override;
  std::unique_ptr<OperatorBase> makeClone(ParticleSet& pset) const final;

#if !defined(REMOVE_TRACEMANAGER)
  void contributeParticleQuantities() override;
  void checkoutParticleQuantities(TraceManager& tm) override;
  void deleteParticleQuantities() override;
#endif

private:
  MFRPotential(ParticleSet& pset,
               const Lattice& field_lattice,
               std::vector<std::shared_ptr<UBspline_3d_d>> splines,
               int spin_channels,
               std::string file_name,
               RealType total_energy_mf);

  RealType evaluateOne(const ParticleSet& pset, int particle_index) const;

  const ParticleSet& pset_;
  Lattice field_lattice_;
  std::vector<std::shared_ptr<UBspline_3d_d>> splines_;
  int spin_channels_ = 0;
  std::string file_name_;
  RealType total_energy_mf_ = 0.0;

#if !defined(REMOVE_TRACEMANAGER)
  Array<TraceReal, 1>* v_sample_ = nullptr;
#endif
};
} // namespace qmcplusplus

#endif
