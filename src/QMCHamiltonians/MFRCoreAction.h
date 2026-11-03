//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_MFR_CORE_ACTION_H
#define QMCPLUSPLUS_MFR_CORE_ACTION_H

#include <array>
#include <hdf5.h>
#include <complex>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "QMCHamiltonians/OperatorBase.h"
#include "einspline/bspline.h"

namespace qmcplusplus
{
class TrialWaveFunction;
class BsplineSet;

/** Direct bare-core action (T + local ionic + bare NC nonlocal) on selected KS orbitals. */
class MFRCoreAction : public OperatorDependsOnlyOnParticleSet
{
public:
  struct StateKey
  {
    int spin;
    int twist;
    int band;
    bool operator<(const StateKey& rhs) const
    {
      return std::tie(spin, twist, band) < std::tie(rhs.spin, rhs.twist, rhs.band);
    }
  };

  explicit MFRCoreAction(ParticleSet& pset, TrialWaveFunction* psi = nullptr);
  std::string getClassName() const override { return "MFRCoreAction"; }
  bool put(xmlNodePtr cur) override;
  bool get(std::ostream& os) const override;
  Return_t evaluate(ParticleSet& pset) override;
  void addObservables(PropertySetType& plist, BufferType& collectables) override;
  void registerObservables(std::vector<ObservableHelper>& h5desc, hdf_archive& file) const override;
  void setObservables(PropertySetType& plist) override;
  void setParticlePropertyList(PropertySetType& plist, int offset) override;
  std::unique_ptr<OperatorBase> makeClone(ParticleSet& pset) const final;

  /** Compute and store the determinant-contracted core action at an energy boundary. */
  FullPrecValueType evaluateTrace(TrialWaveFunction& psi, ParticleSet& pset);

  /** Load one explicitly named state; exposed for deterministic grid-reader tests. */
  void loadActionState(int spin, int twist, int band);

  /** Evaluate the already loaded action field before orbital gauge/Bloch transforms. */
  std::complex<double> evaluatePeriodicAction(int spin, int twist, int band, const PosType& position) const;

  /** Evaluate a loaded action in the SPO's gauge, including its full Bloch phase. */
  std::complex<double> evaluateOrbitalAction(const BsplineSet& orbital_set,
                                             int orbital_index,
                                             const PosType& position) const;

  FullPrecRealType getIonIonEnergyMF() const { return ion_ion_energy_mf_; }
  FullPrecRealType getTotalEnergyMF() const { return total_energy_mf_; }
  FullPrecRealType getImaginaryTrace() const { return imaginary_trace_; }
  /// Cumulative number of refreshed electron rows (also useful for diagnostics).
  size_t getActionRowEvaluationCount() const { return action_row_evaluations_; }

private:
  struct ActionState
  {
    std::shared_ptr<UBspline_3d_d> real;
    std::shared_ptr<UBspline_3d_d> imag;
    std::array<double, 3> reduced_k{};
    FullPrecRealType eigenvalue = 0.0;
  };
  struct SharedData
  {
    Lattice field_lattice;
    std::map<StateKey, ActionState> states;
  };

  // Instance-local, never shared between walker/operator clones. At an energy
  // boundary the accepted coordinates identify changed rows, including buffer
  // restores and walker reassignment, without adding trial-move callbacks.
  struct BlockCache
  {
    const BsplineSet* orbital_set = nullptr;
    Matrix<FullPrecValueType> action;
    std::vector<PosType> positions;
    std::vector<bool> valid;
  };
  std::vector<BlockCache> row_cache_;
  size_t action_row_evaluations_ = 0;

  void validateIons(hid_t file, const BsplineSet& orbital_set) const;
  void readHeader();
  void prepareWaveFunction(TrialWaveFunction& psi);
  ActionState readState(const StateKey& key, BsplineSet* orbital_set, int orbital_index);
  void validateCell(const ParticleSet& pset) const;

  const ParticleSet& pset_;
  TrialWaveFunction* psi_;
  std::string file_name_;
  std::array<int, 3> grid_dimensions_{};
  int spin_channels_                  = 0;
  int number_of_kpoints_              = 0;
  int number_of_states_               = 0;
  FullPrecRealType ion_ion_energy_mf_ = 0.0;
  FullPrecRealType total_energy_mf_   = 0.0;
  FullPrecRealType imaginary_trace_   = 0.0;
  std::shared_ptr<SharedData> data_   = std::make_shared<SharedData>();
  int my_index_                       = -1;
};
} // namespace qmcplusplus

#endif
