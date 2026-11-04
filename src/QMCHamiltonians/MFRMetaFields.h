//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_MFR_META_FIELDS_H
#define QMCPLUSPLUS_MFR_META_FIELDS_H

#include "MFRCoreAction.h"

namespace qmcplusplus
{
/** Two-field meta-GGA reference: u and Q=-div(q grad)/2, acting on D only. */
class MFRMetaFields : public OperatorDependsOnlyOnParticleSet
{
public:
  struct FieldValues
  {
    double u = 0.0;
    double q = 0.0;
    TinyVector<double, 3> grad_q;
  };

  explicit MFRMetaFields(ParticleSet& pset, TrialWaveFunction* psi = nullptr);
  std::string getClassName() const override { return "MFRMetaFields"; }
  bool put(xmlNodePtr cur) override;
  bool get(std::ostream& os) const override;
  Return_t evaluate(ParticleSet& pset) override { return value_; }
  // The energy caller supplies current accepted total log derivatives in P.G/P.L.
  FullPrecValueType evaluateTrace(TrialWaveFunction& psi, ParticleSet& pset);
  FieldValues evaluateFields(int spin, const PosType& position) const;
  bool matchesCoreReference(const MFRCoreAction& core) const { return reference_->matchesReference(core); }
  FullPrecRealType getDifferential() const { return differential_; }
  FullPrecRealType getImaginaryDifferential() const { return imaginary_differential_; }
  FullPrecRealType getTotalEnergyMF() const { return reference_->getTotalEnergyMF(); }
  FullPrecRealType getIonIonEnergyMF() const { return reference_->getIonIonEnergyMF(); }
  void addObservables(PropertySetType& plist, BufferType& collectables) override;
  void registerObservables(std::vector<ObservableHelper>& h5desc, hdf_archive& file) const override;
  void setObservables(PropertySetType& plist) override;
  void setParticlePropertyList(PropertySetType& plist, int offset) override;
  std::unique_ptr<OperatorBase> makeClone(ParticleSet& pset) const final;

private:
  struct Fields
  {
    std::vector<std::shared_ptr<UBspline_3d_d>> u, q;
  };
  void writeProperties(PropertySetType& plist, int offset) const;
  std::unique_ptr<MFRCoreAction> reference_;
  std::shared_ptr<const Fields> fields_;
  std::string filename_;
  ParticleSet::ParticleGradient nonfermion_g_;
  ParticleSet::ParticleLaplacian nonfermion_l_;
  FullPrecRealType differential_ = 0.0;
  FullPrecRealType imaginary_differential_ = 0.0;
  std::array<int,5> indices_{{-1,-1,-1,-1,-1}};
};
} // namespace qmcplusplus
#endif
