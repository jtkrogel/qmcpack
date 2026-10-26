//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file WaveFunctionPool.cpp
 * @brief Implements WaveFunctionPool operators.
 */
#include <sstream>
#include "WaveFunctionPool.h"
#include "Message/UniformCommunicateError.h"
#include "Particle/ParticleSetPool.h"
#include "OhmmsData/AttributeSet.h"
#include "StlPrettyPrint.hpp"

namespace qmcplusplus
{
template class ObjectPool<TrialWaveFunction>;

WaveFunctionPool::WaveFunctionPool(const RuntimeOptions& runtime_options, ParticleSetPool& pset_pool, Communicate* c)
    : MPIObjectBase(c), runtime_options_(runtime_options), ptcl_pool_(pset_pool)
{}

WaveFunctionPool::~WaveFunctionPool() = default;

bool WaveFunctionPool::put(xmlNodePtr cur)
{
  std::string target("e"), psi_name("psi0"), parameter_file;
  OhmmsAttributeSet pAttrib;
  pAttrib.add(target, "target");
  pAttrib.add(target, "ref");
  pAttrib.add(psi_name, "id");
  pAttrib.add(psi_name, "name");
  pAttrib.add(parameter_file, "parameter_file");
  pAttrib.put(cur);

  ParticleSet* qp = ptcl_pool_.getParticleSet(target);

  if (qp == nullptr)
    myComm->barrier_and_abort("target particle set named '" + target + "' not found");

  if (psi_name.empty())
    throw UniformCommunicateError(
        "A wavefunction node cannot be named with an empty string. Please check the name attribute.");

  if (contains(psi_name))
    throw UniformCommunicateError("\"" + psi_name +
                                  "\" already exists in the wavefunction pool. Please rename this wavefunction node.");

  // Resolve only completed prior definitions; pending conventional imports are not frozen values.
  WaveFunctionFactory psiFactory(*qp, ptcl_pool_.getPool(), myComm,
      [this, target](const std::string& name) -> const TrialWaveFunction& {
        if (const auto* request = getParameterFileRequest(name))
          if (!request->imported_hamiltonian)
            throw UniformCommunicateError("Frozen reference sum cannot clone a pending conventional parameter_file import: " + name);
        auto wavefunction = getWaveFunction(name);
        if (!wavefunction)
          throw UniformCommunicateError("Frozen reference sum requires a previously defined wavefunction: " + name);
        auto* definition = wavefunction->get().getNode();
        std::string branch_target("e");
        OhmmsAttributeSet attributes;
        attributes.add(branch_target, "target");
        attributes.add(branch_target, "ref");
        attributes.put(definition);
        if (branch_target != target)
          throw UniformCommunicateError("Frozen reference sum branches must target the same particle set");
        return *wavefunction;
      });
  std::unique_ptr<TrialWaveFunction> psi;
  if (xmlHasProp(cur, BAD_CAST "parameter_file"))
  {
    if (parameter_file.empty())
      throw UniformCommunicateError("wavefunction parameter_file must name a conventional model export");
    // Artifact location is routing metadata, not part of the inline physical
    // definition stored in the wavefunction and compared with exported identity.
    std::unique_ptr<xmlNode, decltype(&xmlFreeNode)> definition(xmlCopyNode(cur, 1), xmlFreeNode);
    if (!definition)
      throw std::bad_alloc();
    xmlUnsetProp(definition.get(), BAD_CAST "parameter_file");
    psi = psiFactory.buildTWF(definition.get(), runtime_options_, psi_name);
    parameter_file_requests_.emplace(psi_name, ParameterFileRequest{parameter_file, target, nullptr});
  }
  else
    psi = psiFactory.buildTWF(cur, runtime_options_, psi_name);
  add(psi->getName(), std::move(psi));
  app_log() << "  Added \"" << psi_name << "\" to the wavefunction pool" << std::endl;
  return true;
}

const WaveFunctionPool::ParameterFileRequest* WaveFunctionPool::getParameterFileRequest(const std::string& name) const
{
  const auto request = parameter_file_requests_.find(name);
  return request == parameter_file_requests_.end() ? nullptr : &request->second;
}

void WaveFunctionPool::markParameterFileImported(const std::string& name, const QMCHamiltonian& hamiltonian)
{
  const auto request = parameter_file_requests_.find(name);
  if (request == parameter_file_requests_.end() || request->second.imported_hamiltonian)
    throw std::logic_error("Conventional parameter_file import has no pending request");
  request->second.imported_hamiltonian = &hamiltonian;
}

OptionalRef<TrialWaveFunction> WaveFunctionPool::getWaveFunction(const std::string& pname)
{
  if (empty())
    throw UniformCommunicateError(
        "The wavefunction pool is empty. Need at least one wavefunction node in the xml input!");

  if (pname.empty())
    return getTheOneAndOnly();
  else
    return getObjectByName(pname);
}

xmlNodePtr WaveFunctionPool::getWaveFunctionNode(const std::string& id)
{
  auto wf_optional = getWaveFunction(id);
  if (!wf_optional)
  {
    std::ostringstream msg;
    msg << "Wavefunction pool contains " << myPool << "." << std::endl
        << "Cannot find wavefunction named \"" + id + "\"!";
    throw UniformCommunicateError(msg.str());
  }

  TrialWaveFunction& wf = *wf_optional;
  return wf.getNode();
}
} // namespace qmcplusplus
