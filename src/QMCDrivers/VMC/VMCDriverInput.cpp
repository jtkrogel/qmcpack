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

#include "VMCDriverInput.h"

#include <stdexcept>

#include "ModernStringUtils.hpp"
#include "OhmmsData/AttributeSet.h"
#include "OhmmsData/XMLParsingString.h"

namespace qmcplusplus
{
namespace
{
bool hasInputParameter(xmlNodePtr node, const std::string& parameter_name)
{
  for (xmlNodePtr child = node == nullptr ? nullptr : node->children; child != nullptr; child = child->next)
  {
    const std::string child_name = lowerCase(castXMLCharToChar(child->name));
    if (child_name == parameter_name)
      return true;
    if (child_name == "parameter" && lowerCase(getXMLAttributeValue(child, "name")) == parameter_name)
      return true;
  }
  return false;
}

VMCDriverInput::MoveKind parseMoveKind(const std::string& move)
{
  if (move == "pbyp")
    return VMCDriverInput::MoveKind::PBYP;
  if (move == "alle")
    return VMCDriverInput::MoveKind::ALL_ELECTRON;
  if (move == "n_electron")
    return VMCDriverInput::MoveKind::N_ELECTRON;

  throw std::runtime_error("VMCDriverInput: invalid move=\"" + move +
                           "\"; expected exactly one of pbyp, alle, or n_electron.");
}

VMCDriverInput::ElectronSelection parseElectronSelection(const std::string& selection)
{
  if (selection == "random")
    return VMCDriverInput::ElectronSelection::RANDOM;
  if (selection == "cyclic")
    return VMCDriverInput::ElectronSelection::CYCLIC;

  throw std::runtime_error("VMCDriverInput: invalid electron_selection=\"" + selection +
                           "\"; expected exactly random or cyclic.");
}
} // namespace

VMCDriverInput::VMCDriverInput(bool use_drift) : use_drift_(use_drift) {}

void VMCDriverInput::readXML(xmlNodePtr node)
{
  std::string move{"pbyp"};
  OhmmsAttributeSet attributes;
  attributes.add(move, "move");
  attributes.put(node);

  electrons_per_move_               = 0;
  electron_selection_               = ElectronSelection::RANDOM;
  electrons_per_move_was_provided_  = hasInputParameter(node, "electrons_per_move");
  electron_selection_was_provided_  = hasInputParameter(node, "electron_selection");

  ParameterSet parameter_set_;
  std::string use_drift;
  std::string electron_selection{"random"};
  parameter_set_.add(use_drift, "usedrift", {"yes", "no"});
  parameter_set_.add(use_drift, "use_drift", {"yes", "no"});
  parameter_set_.add(samples_, "samples");
  parameter_set_.add(electrons_per_move_, "electrons_per_move");
  parameter_set_.add(electron_selection, "electron_selection");
  parameter_set_.put(node);

  move_kind_          = parseMoveKind(move);
  electron_selection_ = parseElectronSelection(electron_selection);

  if (move_kind_ == MoveKind::N_ELECTRON)
  {
    if (!electrons_per_move_was_provided_)
      throw std::runtime_error(
          "VMCDriverInput: move=\"n_electron\" requires an explicit electrons_per_move parameter.");
    if (electrons_per_move_ <= 0)
      throw std::runtime_error("VMCDriverInput: electrons_per_move must be greater than zero for move=\"n_electron\".");
  }
  else if (electrons_per_move_was_provided_ || electron_selection_was_provided_)
    throw std::runtime_error(
        "VMCDriverInput: electrons_per_move and electron_selection are only valid for move=\"n_electron\".");

  use_drift_ = use_drift == "yes";
  if (use_drift_)
    app_log() << "  Random walking with drift" << std::endl;
  else
    app_log() << "  Random walking without drift" << std::endl;
}

std::ostream& operator<<(std::ostream& o_stream, const VMCDriverInput& vmci) { return o_stream; }

} // namespace qmcplusplus
