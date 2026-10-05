//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_VMCDRIVERINPUT_H
#define QMCPLUSPLUS_VMCDRIVERINPUT_H

#include "Configuration.h"
#include "OhmmsData/ParameterSet.h"

namespace qmcplusplus
{
/** Input representation for VMC driver class runtime parameters
 */
class VMCDriverInput
{
public:
  using IndexType             = QMCTraits::IndexType;
  using RealType              = QMCTraits::RealType;
  using FullPrecisionRealType = QMCTraits::FullPrecRealType;

  enum class MoveKind
  {
    PBYP,
    ALL_ELECTRON,
    N_ELECTRON
  };

  enum class ElectronSelection
  {
    RANDOM,
    CYCLIC
  };

  VMCDriverInput(){};
  VMCDriverInput(bool use_drift);
  void readXML(xmlNodePtr xml_input);

protected:
  /** @ingroup Parameters for VMC Driver
   *  @{
   *  
   *  Do not write out blocks of gets for variables like this
   *  there is are code_generation tools in QMCPACK_ROOT/utils/code_tools
   */
  bool use_drift_    = true;
  IndexType samples_ = -1;
  MoveKind move_kind_                   = MoveKind::PBYP;
  IndexType electrons_per_move_         = 0;
  ElectronSelection electron_selection_ = ElectronSelection::RANDOM;
  bool electrons_per_move_was_provided_ = false;
  bool electron_selection_was_provided_ = false;
  /** @} */

public:
  bool get_use_drift() const { return use_drift_; }
  IndexType get_samples() const { return samples_; }
  MoveKind get_move_kind() const { return move_kind_; }
  IndexType get_electrons_per_move() const { return electrons_per_move_; }
  ElectronSelection get_electron_selection() const { return electron_selection_; }
  bool was_electrons_per_move_provided() const { return electrons_per_move_was_provided_; }
  bool was_electron_selection_provided() const { return electron_selection_was_provided_; }

  friend std::ostream& operator<<(std::ostream& o_stream, const VMCDriverInput& vmci);
};

extern std::ostream& operator<<(std::ostream& o_stream, const VMCDriverInput& vmci);

} // namespace qmcplusplus
#endif
