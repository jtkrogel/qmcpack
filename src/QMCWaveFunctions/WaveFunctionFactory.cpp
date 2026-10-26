//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file WaveFunctionFactory.cpp
 *@brief Definition of a WaveFunctionFactory
 */
#include "WaveFunctionFactory.h"
#include "FrozenReferenceResidualWF.h"
#include "Message/UniformCommunicateError.h"
#include <cmath>
#include "QMCWaveFunctions/Jastrow/JastrowBuilder.h"
#include "QMCWaveFunctions/Fermion/SlaterDetBuilder.h"
#include "QMCWaveFunctions/LatticeGaussianProductBuilder.h"
#include "QMCWaveFunctions/ExampleHeBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#ifdef ENABLE_DEEPQMC_INFERENCE
#include "QMCWaveFunctions/DeepQMC/DeepQMCWaveFunctionBuilder.h"
#endif

#include "Utilities/ProgressReportEngine.h"
#include "Utilities/IteratorUtility.h"
#include "OhmmsData/AttributeSet.h"
namespace qmcplusplus
{
WaveFunctionFactory::WaveFunctionFactory(ParticleSet& qp, const PSetMap& pset, Communicate* c,
    NamedWaveFunctionResolver resolver)
    : MPIObjectBase(c), targetPtcl(qp), ptclPool(pset), class_name_("WaveFunctionFactory"),
      named_wavefunction_resolver_(std::move(resolver))
{}

WaveFunctionFactory::~WaveFunctionFactory() = default;

std::unique_ptr<TrialWaveFunction> WaveFunctionFactory::buildTWF(xmlNodePtr cur,
                                                                 const RuntimeOptions& runtime_options,
                                                                 const std::string psi_name)
{
  // YL: how can this happen?
  if (cur == NULL)
    return nullptr;

  ReportEngine PRE(class_name_, "build");

  std::string tasking;
  std::string print_parameters = "yes";
  OhmmsAttributeSet pAttrib;
  pAttrib.add(tasking, "tasking", {"no", "yes"});
  pAttrib.add(print_parameters, "print_parameters", {"no", "yes"});
  pAttrib.put(cur);

  app_summary() << std::endl;
  app_summary() << " Many-body wavefunction" << std::endl;
  app_summary() << " -------------------" << std::endl;
  app_summary() << "  Name: " << psi_name << "   Tasking: " << (tasking == "yes" ? "yes" : "no") << std::endl;
  app_summary() << std::endl;

  auto targetPsi = std::make_unique<TrialWaveFunction>(runtime_options, psi_name, tasking == "yes");
  targetPsi->storeXMLNode(cur);

  SPOSetBuilderFactory sposet_builder_factory(myComm, targetPtcl, ptclPool);

  std::string vp_file_to_load;
  bool contains_frozen_sum = false;
  cur = cur->children;
  while (cur != NULL)
  {
    std::string cname((const char*)(cur->name));
    if (cname == "sposet_builder" || cname == "sposet_collection")
      sposet_builder_factory.buildSPOSetCollection(cur);
    else if (cname == WaveFunctionComponentBuilder::detset_tag)
    {
      addFermionTerm(*targetPsi, sposet_builder_factory, cur);
      //default twist is [0 0 0]
      targetPsi->setTwist(std::vector<ParticleSet::RealType>(3, 0));
    }
    else if (cname == WaveFunctionComponentBuilder::jastrow_tag)
    {
      auto jbuilder = std::make_unique<JastrowBuilder>(myComm, targetPtcl, ptclPool);
      targetPsi->addComponent(jbuilder->buildComponent(cur));
    }
    else if (cname == "fdlrwfn")
    {
      APP_ABORT("FDLR wave functions are not currently supported.");
    }
    else if (cname == WaveFunctionComponentBuilder::ionorb_tag)
    {
      auto builder = std::make_unique<LatticeGaussianProductBuilder>(myComm, targetPtcl, ptclPool);
      targetPsi->addComponent(builder->buildComponent(cur));
    }
    else if ((cname == "Molecular") || (cname == "molecular"))
    {
      APP_ABORT("  Removed Helium Molecular terms from qmcpack ");
    }
    else if (cname == "example_he")
    {
      auto exampleHe_builder = std::make_unique<ExampleHeBuilder>(myComm, targetPtcl, ptclPool);
      targetPsi->addComponent(exampleHe_builder->buildComponent(cur));
    }
    else if (cname == "deepqmc")
    {
#ifdef ENABLE_DEEPQMC_INFERENCE
      auto deepqmc_builder = std::make_unique<DeepQMCWaveFunctionBuilder>(myComm, targetPtcl, ptclPool);
      targetPsi->addComponent(deepqmc_builder->buildComponent(cur));
#else
      APP_ABORT("DeepQMC wavefunction input requires ENABLE_DEEPQMC_INFERENCE=ON");
#endif
    }
    else if (cname == "psiformer")
    {
      auto builder = std::make_unique<PsiFormerWaveFunctionBuilder>(myComm, targetPtcl, ptclPool);
      targetPsi->addComponent(builder->buildComponent(cur));
    }
    else if (cname == "frozen_reference_sum")
    {
      if (!named_wavefunction_resolver_)
        throw UniformCommunicateError("frozen_reference_sum requires named definitions in the wavefunction pool");
      std::string reference_name, neural_name, coefficient_text, optimize = "no";
      OhmmsAttributeSet attributes;
      attributes.add(reference_name, "reference");
      attributes.add(neural_name, "neural");
      attributes.add(coefficient_text, "coefficient");
      attributes.add(optimize, "optimize");
      attributes.put(cur);
      if (reference_name.empty() || neural_name.empty() || coefficient_text.empty() ||
          reference_name == neural_name || reference_name == psi_name || neural_name == psi_name)
        throw UniformCommunicateError("frozen_reference_sum requires distinct prior reference/neural names and explicit coefficient");
      for (xmlAttrPtr attribute = cur->properties; attribute; attribute = attribute->next)
      {
        const std::string key(reinterpret_cast<const char*>(attribute->name));
        if (key != "reference" && key != "neural" && key != "coefficient" && key != "optimize")
          throw UniformCommunicateError("Unknown frozen_reference_sum attribute: " + key);
      }
      for (xmlNodePtr child = cur->children; child; child = child->next)
        if (child->type == XML_ELEMENT_NODE)
          throw UniformCommunicateError("frozen_reference_sum does not accept nested component or training controls");
      if (optimize != "yes" && optimize != "no")
        throw UniformCommunicateError("frozen_reference_sum optimize must be yes or no");
      double coefficient;
      try
      {
        std::size_t consumed = 0;
        coefficient = std::stod(coefficient_text, &consumed);
        if (consumed != coefficient_text.size() || !residual::finiteReal(coefficient))
          throw std::invalid_argument("not a finite complete scalar");
      }
      catch (const std::exception&)
      {
        throw UniformCommunicateError("frozen_reference_sum coefficient must be a finite real scalar");
      }
      const auto& reference = named_wavefunction_resolver_(reference_name);
      const auto& neural = named_wavefunction_resolver_(neural_name);
      auto sum = std::make_unique<FrozenReferenceResidualWF>(targetPtcl, reference, neural, coefficient);
      if (optimize == "yes")
        sum->enableNeuralTraining();
      targetPsi->addComponent(std::move(sum));
      targetPsi->setTwist(reference.twist());
      contains_frozen_sum = true;
      app_log() << "  Frozen reference sum: reference=" << reference_name << ", neural=" << neural_name
                << ", fixed coefficient=" << coefficient << ", CPU FP64 Gamma VMC" << std::endl;
    }
    else if (cname == "override_variational_parameters")
    {
      OhmmsAttributeSet attribs;
      attribs.add(vp_file_to_load, "href");
      attribs.put(cur);
    }

    cur = cur->next;
  }
  if (contains_frozen_sum && (targetPsi->getOrbitals().size() != 1 || !vp_file_to_load.empty()))
    throw UniformCommunicateError("frozen_reference_sum must be the sole parent component with no parameter override");
  //{
  //  ReportEngine PREA("TrialWaveFunction","print");
  //  targetPsi->VarList.print(app_log());
  //}
  // synch all parameters. You don't want some being different if same name.
  OptVariables dummy;
  targetPsi->checkInVariables(dummy);
  dummy.resetIndex();
  targetPsi->checkOutVariables(dummy);

  if (!vp_file_to_load.empty())
  {
    app_log() << "  Reading variational parameters from " << vp_file_to_load << std::endl;
    hdf_archive hin;
    dummy.readFromHDF(vp_file_to_load, hin);

    UniqueOptObjRefs opt_obj_refs = targetPsi->extractOptimizableObjectRefs();
    for (auto opt_obj : opt_obj_refs)
      opt_obj.get().readVariationalParameters(hin);
  }

  targetPsi->resetParameters(dummy);
  if (print_parameters == "yes")
  {
    app_log() << "  Wave function " << targetPsi->getName() << " parameters:" << std::endl;
    dummy.print(app_log(), 4 /* left padding spaces */, true);
  }
  else
    app_log() << "  Wave function " << targetPsi->getName() << ": " << dummy.size()
              << " parameters (listing suppressed)" << std::endl;
  app_log() << std::endl;
  targetPsi->storeSPOMap(sposet_builder_factory.exportSPOSets());
  return targetPsi;
}

bool WaveFunctionFactory::addFermionTerm(TrialWaveFunction& psi, SPOSetBuilderFactory& spo_factory, xmlNodePtr cur)
{
  ReportEngine PRE(class_name_, "addFermionTerm");
  std::string orbtype("MolecularOrbital");
  std::string nuclei("i");
  OhmmsAttributeSet oAttrib;
  oAttrib.add(orbtype, "type");
  oAttrib.add(nuclei, "source");
  oAttrib.put(cur);
  std::unique_ptr<WaveFunctionComponentBuilder> detbuilder;
  if (orbtype == "electron-gas")
  {
    std::ostringstream msg;
    msg << "electron-gas in determinantset is deprecated";
    msg << " please use \"free\" orbitals in sposet_builder" << std::endl;
    throw std::runtime_error(msg.str());
  }
  else
    detbuilder = std::make_unique<SlaterDetBuilder>(myComm, spo_factory, targetPtcl, psi, ptclPool);
  psi.addComponent(detbuilder->buildComponent(cur));
  return true;
}

} // namespace qmcplusplus
