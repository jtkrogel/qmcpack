//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Bryan Clark, bclark@Princeton.edu, Princeton University
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file QMCDriverFactory.cpp
 * @brief Implments QMCMain operators.
 */

#include "QMCDriverFactory.h"
#include <exception>
#include <queue>
#include "QMCDrivers/MCPopulation.h"
#include "Utilities/qmc_common.h"
#include "QMCWaveFunctions/WaveFunctionPool.h"
#include "QMCHamiltonians/HamiltonianPool.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "PsiHamNamePairReader.h"
#include "QMCDrivers/VMC/VMC.h"
#include "QMCDrivers/CorrelatedSampling/CSVMC.h"
#include "QMCDrivers/DMC/DMCFactory.h"
#include "QMCDrivers/RMC/RMCFactory.h"
#include "VMC/VMCDriverInput.h"
#include "VMC/VMCBatched.h"
#include "WFTrain/WFTrainBatched.h"
#include "WFTrain/WFTrainDriverInput.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"
#include "DMC/DMCDriverInput.h"
#include "DMC/DMCBatched.h"
#include "QMCDrivers/WFOpt/QMCFixedSampleLinearOptimize.h"
#include "QMCDrivers/WFOpt/QMCFixedSampleLinearOptimizeBatched.h"
#include "QMCDrivers/WaveFunctionTester.h"
#include "OhmmsData/AttributeSet.h"
#include "OhmmsData/ParameterSet.h"
#include "Estimators/EstimatorInputDelegates.h"
#include "Estimators/EstimatorManagerNew.h"
#include "Message/UniformCommunicateError.h"
#include "Message/CommOperators.h"
#include "RandomNumberControl.h"
#ifdef QMC_COMPLEX
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPBCAA.h"
#include "QMCHamiltonians/CoulombPBCAB.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include <array>
#include <typeinfo>
#endif

namespace qmcplusplus
{
#ifdef QMC_COMPLEX
namespace
{
/// Find the explicit profile in the actual selected wavefunction, never an inferred pool default.
const PsiFormerComplexWF* selectedComplexProfile(const TrialWaveFunction& psi)
{
  for (const auto& component : psi.getOrbitals())
    if (auto complex = dynamic_cast<const PsiFormerComplexWF*>(component.get()))
      return complex;
  return nullptr;
}

/// Validate concrete selected objects before estimators/drivers/clones; store no admission pointer/token.
void requireSelectedComplexCoulombSystem(const TrialWaveFunction& psi, const QMCHamiltonian& ham,
                                        const ParticleSet& target, QMCRunType run_type)
{
  const auto* complex = selectedComplexProfile(psi);
  if (!complex)
    return;
  auto reject = []() { throw UniformCommunicateError(
      "Complex PsiFormer selected-system admission requires a sole component, CPU bare kinetic and "
      "three physical periodic Coulomb terms bound to its actual target/source"); };
  // Ordinary scalar PP inference uses the same checked batched VMC route on CPU/GPU.
  const bool ecp_run = run_type == QMCRunType::WF_TRAIN_BATCH ||
      run_type == QMCRunType::VMC_BATCH;
  const bool ecp = ecp_run && ham.hasPhysicalNLPP();
  const auto& provenance = ham.ecpModelProvenance();
  if (complex->isPseudopotentialSystem() != ecp || ham.hasPhysicalNLPP() != ecp ||
      (ecp && (!provenance || !provenance->ordinary_locality_vp || provenance->scientific_definition.empty())) ||
      psi.getOrbitals().size() != 1 || &complex->targetParticles() != &target ||
      ham.size() != (ecp ? 5 : 4) || ham.total_size() != ham.size())
    reject();
  const auto* kinetic = dynamic_cast<const BareKineticEnergy*>(ham.getComponent(0));
  if (!kinetic || typeid(*kinetic) != typeid(BareKineticEnergy) || &kinetic->getTargetPSet() != &target)
    reject();
  std::array<int, 3> counts{};
  int nonlocal_count = 0;
  for (int i = 1; i < ham.size(); ++i)
  {
    const auto* op = ham.getComponent(i);
    if (const auto* aa = dynamic_cast<const CoulombPBCAA*>(op))
    {
      if (typeid(*aa) != typeid(CoulombPBCAA) || aa->ComputeForces || aa->quasi2d || aa->usesOffload())
        reject();
      if (ecp && aa->ewaldModelDefinition().empty())
        reject();
      if (&aa->Ps == &target && aa->is_active)
        ++counts[0];
      else if (&aa->Ps == &complex->sourceParticles() && !aa->is_active)
        ++counts[1];
      else
        reject();
    }
    else if (const auto* ab = dynamic_cast<const CoulombPBCAB*>(op))
    {
      if (typeid(*ab) != typeid(CoulombPBCAB) || ab->ComputeForces || &ab->Peln != &target ||
          &ab->getSourcePSet() != &complex->sourceParticles())
        reject();
      // This is the built Ewald/local-PP definition; parsed PP bytes are separately bound above.
      if (ecp && ab->ewaldModelDefinition().empty())
        reject();
      ++counts[2];
    }
    else if (const auto* nl = dynamic_cast<const NonLocalECPotential*>(op))
    {
      if (!ecp || typeid(*nl) != typeid(NonLocalECPotential) || nl->usesDLA() ||
          !nl->usesVirtualParticles() || &nl->sourceParticles() != &complex->sourceParticles() ||
          nl->virtualTargetParticles() != &target)
        reject();
      ++nonlocal_count;
    }
    else
      reject();
  }
  if (counts != std::array<int, 3>{1, 1, 1} || nonlocal_count != (ecp ? 1 : 0))
    reject();
}
} // namespace
#endif

QMCDriverFactory::QMCDriverFactory(const ProjectData& project_data) : project_data_(project_data) {}

/** Read the xml defining the driver for this QMC section
 *
 *  Copy elision should result in just a move of the
 *  DriverAssemblyState
 *
 *  Most (all) of this should be done by calling QMCDriverInput::readXML
 *  At some point in driver refactoring this should go there and
 *  QMCDriverInput created before the giant switch
 */
QMCDriverFactory::DriverAssemblyState QMCDriverFactory::readSection(xmlNodePtr cur) const
{
  DriverAssemblyState das;
  std::string curName(castXMLCharToChar(cur->name));
  std::string update_mode("pbyp");
  std::string qmc_mode;
  std::string append_tag("no");
  std::string profiling_tag("no");
  OhmmsAttributeSet aAttrib;
  aAttrib.add(qmc_mode, "method", {"", "vmc", "vmc_batch", "dmc", "dmc_batch", "csvmc", "rmc", "linear",
                                    "linear_batch", "wf_train", "wftest"});
  aAttrib.add(update_mode, "move");
  aAttrib.add(append_tag, "append");
  aAttrib.add(profiling_tag, "profiling");
  aAttrib.add(das.traces_tag, "trace");
  aAttrib.add(das.walkerlogs_tag, "walkerlog");
  aAttrib.put(cur);
  das.append_run                 = (append_tag == "yes");
  das.enable_profiling           = (profiling_tag == "yes");
  das.what_to_do[UPDATE_MODE]    = (update_mode == "pbyp");
  infoSummary.flush();
  infoLog.flush();

  // Really by position if you don't write <qmc name="xxx",,,>
  // you can write <vmc ...> so hacky
  if (curName != "qmc")
    qmc_mode = curName;

  const int nchars = qmc_mode.size();

  using DV = ProjectData::DriverVersion;
  switch (project_data_.getDriverVersion())
  {
  case DV::BATCH:
    if (qmc_mode == "wf_train")
      das.new_run_type = QMCRunType::WF_TRAIN_BATCH;
    else if (qmc_mode.find("vmc") < nchars) // order matters here
      das.new_run_type = QMCRunType::VMC_BATCH;
    else if (qmc_mode.find("dmc") < nchars) // order matters here
      das.new_run_type = QMCRunType::DMC_BATCH;
    else if (qmc_mode.find("linear") < nchars)
      das.new_run_type = QMCRunType::LINEAR_OPTIMIZE_BATCH;
    else
      throw UniformCommunicateError(
          "QMC mode unknown. Valid modes for batched drivers are : vmc, dmc, linear, wf_train.");
    break;
  // Begin to separate driver version = batch input reading from the legacy input parsing
  case DV::LEGACY:
    if (qmc_mode == "wf_train")
      das.new_run_type = QMCRunType::WF_TRAIN_BATCH;
    else if (qmc_mode.find("linear_batch") < nchars) // order matters here
      das.new_run_type = QMCRunType::LINEAR_OPTIMIZE_BATCH;
    else if (qmc_mode.find("linear") < nchars)
      das.new_run_type = QMCRunType::LINEAR_OPTIMIZE;
    else if (qmc_mode.find("rmc") < nchars)
      das.new_run_type = QMCRunType::RMC;
    else if (qmc_mode.find("csvmc") < nchars)
      das.new_run_type = QMCRunType::CSVMC;
    else if (qmc_mode.find("vmc_batch") < nchars) // order matters here
      das.new_run_type = QMCRunType::VMC_BATCH;
    else if (qmc_mode.find("vmc") < nchars)
      das.new_run_type = QMCRunType::VMC;
    else if (qmc_mode.find("dmc_batch") < nchars) // order matters here
      das.new_run_type = QMCRunType::DMC_BATCH;
    else if (qmc_mode.find("dmc") < nchars)
      das.new_run_type = QMCRunType::DMC;
    else if (qmc_mode == "wftest")
      das.new_run_type = QMCRunType::WF_TEST;
    else
      throw std::runtime_error("qmc method cannot be empty!");
  }
  return das;
}

std::unique_ptr<QMCDriverInterface> QMCDriverFactory::createQMCDriver(xmlNodePtr cur,
                                                                      DriverAssemblyState& das,
                                                                      const std::optional<EstimatorManagerInput>& emi,
                                                                      MCWalkerConfiguration& qmc_system,
                                                                      ParticleSetPool& particle_pool,
                                                                      WaveFunctionPool& wavefunction_pool,
                                                                      HamiltonianPool& hamiltonian_pool,
                                                                      Communicate* comm) const
{
  std::unique_ptr<QMCDriverInterface> new_driver;

  auto getPsi = [&wavefunction_pool](const std::string& name) -> TrialWaveFunction& {
    if (auto psi_optional = wavefunction_pool.getWaveFunction(name); psi_optional)
      return *psi_optional;
    else if (name.empty())
      throw UniformCommunicateError("Failed to find a wavefunction! Please specify the name of wavefunction using a "
                                    "qmcsystem node in the driver input.");
    else
      throw UniformCommunicateError("Failed to find the wavefunction named \"" + name + "\"!");
  };

  auto getHam = [&hamiltonian_pool](const std::string& name) -> QMCHamiltonian& {
    if (auto ham_optional = hamiltonian_pool.getHamiltonian(name); ham_optional)
      return *ham_optional;
    else if (name.empty())
      throw UniformCommunicateError("Failed to find a hamiltonian! Please specify the name of hamiltonian using a "
                                    "qmcsystem node in the driver input.");
    else
      throw UniformCommunicateError("Failed to find the hamiltonian named \"" + name + "\"!");
  };

  if (das.new_run_type == QMCRunType::CSVMC)
  { // CSVMC requires multiple pairs of Psi and Ham
    std::vector<TrialWaveFunction*> multi_psi;
    std::vector<QMCHamiltonian*> multi_ham;
    for (const auto& name_pair : PsiHamNamePairReader::readMultiplePairs(cur))
    {
      multi_psi.emplace_back(&getPsi(name_pair.first));
      if (wavefunction_pool.getParameterFileRequest(multi_psi.back()->getName()))
        throw UniformCommunicateError("wavefunction parameter_file does not support CSVMC or multiple selected systems");
#ifdef QMC_COMPLEX
      if (selectedComplexProfile(*multi_psi.back()))
        throw UniformCommunicateError("Complex PsiFormer selected-system admission does not support CSVMC");
#endif
      multi_ham.emplace_back(&getHam(name_pair.second));
    }
    new_driver = std::make_unique<CSVMC>(project_data_, qmc_system, std::move(multi_psi), std::move(multi_ham), comm);
    new_driver->setUpdateMode(das.what_to_do[UPDATE_MODE]);
  }
  else
  { // other drivers require at most one pair of Psi and Ham
    auto one_pair = PsiHamNamePairReader::readOnePair(cur);
    // get primaryPsi
    auto& primaryPsi = getPsi(one_pair ? one_pair->first : "");
    // get primaryH
    auto& primaryH = getHam(one_pair ? one_pair->second : "");
    bool imported_md_orbital_ecp = false;
    if (const auto* request = wavefunction_pool.getParameterFileRequest(primaryPsi.getName()))
    {
      // Defer loading until physical/PP identity exists, but finish before any
      // estimator/population construction can clone or acquire model resources.
      if (das.new_run_type != QMCRunType::VMC_BATCH && das.new_run_type != QMCRunType::WF_TRAIN_BATCH)
        throw UniformCommunicateError("wavefunction parameter_file supports only CPU batched VMC and WFTrain");
      if (das.new_run_type == QMCRunType::WF_TRAIN_BATCH)
      {
        WFTrainDriverInput input;
        input.readXML(cur);
        if (input.getObjective() != WFTrainDriverInput::Objective::ENERGY)
          throw UniformCommunicateError("wavefunction parameter_file supports conventional ENERGY training only");
      }
      if (!primaryPsi.extractStructuredParameterProviders().empty() || qmc_system.isSpinor() ||
          qmc_system.getCoordinates().getKind() != DynamicCoordinateKind::DC_POS)
        throw UniformCommunicateError("wavefunction parameter_file requires a qualified conventional CPU model");
      // Binding/setHamiltonian below admits the precise scalar/model profile
      // before import; a complex build alone does not qualify an arbitrary model.
      if (particle_pool.getParticleSet(request->target) != &qmc_system)
        throw UniformCommunicateError("wavefunction parameter_file selected a different electron target");
      if (request->imported_hamiltonian && request->imported_hamiltonian != &primaryH)
        throw UniformCommunicateError("wavefunction parameter_file cannot reuse an imported model with another Hamiltonian");
      if (!request->imported_hamiltonian)
      {
        // Fixed inference-loader ceilings avoid adding optimizer controls to VMC.
        const wftrain::LegacyWaveFunctionAdapterOptions limits{65536, 4 * 1024 * 1024, 256};
        std::unique_ptr<wftrain::TrainingWaveFunctionBinding> binding;
        std::exception_ptr preparation_failure;
        try
        {
          optimize::VariableSet registration;
          primaryPsi.checkInVariables(registration);
          registration.resetIndex();
          if (registration.size_of_active() > limits.maximum_parameter_count)
            throw UniformCommunicateError("Conventional parameter_file loader requires " +
                std::to_string(registration.size_of_active()) + " active parameters; loader limit is " +
                std::to_string(limits.maximum_parameter_count));
          binding = std::make_unique<wftrain::TrainingWaveFunctionBinding>(primaryPsi, qmc_system, limits, *comm);
          binding->setHamiltonian(primaryH);
        }
        catch (...)
        {
          preparation_failure = std::current_exception();
        }
        int failed = preparation_failure ? 1 : 0;
        comm->allreduce(failed);
        if (failed)
        {
          if (preparation_failure)
            std::rethrow_exception(preparation_failure);
          throw UniformCommunicateError("Conventional parameter_file preparation failed on another rank");
        }
        binding->importModel(request->path);
        wavefunction_pool.markParameterFileImported(primaryPsi.getName(), primaryH);
      }
      const auto determinants = primaryPsi.findMSD();
      if (primaryH.hasPhysicalNLPP() && !determinants.empty() && determinants.front().get().getDet(0).isOptimizable())
      {
        // Revalidate this exact imported subset on later sections too; never flag the pool-owned model.
        std::exception_ptr failure;
        try { determinants.front().get().validateAcceptedReferenceOrbitalProfile(); }
        catch (...) { failure = std::current_exception(); }
        int failed = failure ? 1 : 0;
        comm->allreduce(failed);
        if (failed)
        {
          if (failure) std::rethrow_exception(failure);
          throw UniformCommunicateError("Imported orbital ECP structural admission failed on another rank");
        }
        imported_md_orbital_ecp = true;
      }
    }
#ifdef QMC_COMPLEX
    requireSelectedComplexCoulombSystem(primaryPsi, primaryH, qmc_system, das.new_run_type);
    if (const auto* complex = selectedComplexProfile(primaryPsi); complex && complex->gpuInferenceEnabled() &&
        das.new_run_type != QMCRunType::VMC_BATCH && das.new_run_type != QMCRunType::WF_TRAIN_BATCH)
      throw UniformCommunicateError("Complex GPU requires PBYP VMC_BATCH or explicitly admitted WF_TRAIN_BATCH");
#endif

    auto makeEstimatorManager =
        [&](const std::optional<EstimatorManagerInput>& global_emi,
            const std::optional<EstimatorManagerInput>& driver_emi,
            TrialWaveFunction& estimator_wavefunction) -> UPtr<EstimatorManagerNew> {
      // This is done so that the application level input structures reflect the actual input to the code.
      // While the actual simulation objects still take singular input structures at construction.
      auto makeEstimatorManagerInput = [](auto& global_emi, auto& local_emi) -> EstimatorManagerInput {
        if (global_emi.has_value() && local_emi.has_value())
          return {global_emi.value(), local_emi.value()};
        else if (global_emi.has_value())
          return {global_emi.value()};
        else if (local_emi.has_value())
          return {local_emi.value()};
        else
          return {};
      };

      auto estimator_manager = std::make_unique<EstimatorManagerNew>(primaryH, comm);
      estimator_manager->constructEstimators(makeEstimatorManagerInput(global_emi, driver_emi), qmc_system,
                                             estimator_wavefunction, primaryH, particle_pool.getPool());
      return estimator_manager;
    };

    if (das.new_run_type == QMCRunType::VMC)
    {
      new_driver = std::make_unique<VMC>(project_data_, qmc_system, primaryPsi, primaryH,
                                         RandomNumberControl::getChildren(), comm, das.enable_profiling);
      new_driver->setUpdateMode(das.what_to_do[UPDATE_MODE]);
    }
    else if (das.new_run_type == QMCRunType::VMC_BATCH)
    {
      app_summary() << "\n========================================"
                       "\n  Reading VMC driver XML input section"
                       "\n========================================"
                    << std::endl;

      QMCDriverInput qmcdriver_input;
      VMCDriverInput vmcdriver_input;
      try
      {
        qmcdriver_input.readXML(cur);
        vmcdriver_input.readXML(cur);
      }
      catch (const std::exception& e)
      {
        throw UniformCommunicateError(e.what());
      }

      const bool is_pbyp_move = vmcdriver_input.get_move_kind() == VMCDriverInput::MoveKind::PBYP;
#ifdef QMC_COMPLEX
      if (const auto* complex = selectedComplexProfile(primaryPsi); complex && complex->gpuInferenceEnabled() &&
          !is_pbyp_move && !complex->supportsMultiParticleMoves())
        throw UniformCommunicateError("Complex GPU inference requires parsed PBYP VMC_BATCH moves");
      if (const auto* complex = selectedComplexProfile(primaryPsi); complex &&
          complex->isPseudopotentialSystem() && !is_pbyp_move &&
          (vmcdriver_input.get_move_kind() != VMCDriverInput::MoveKind::ALL_ELECTRON ||
           !complex->supportsMultiParticleMoves()))
        throw UniformCommunicateError("Scalar PP inference requires PBYP or supported all-electron VMC_BATCH moves");
#endif

      // I don't like that QMCDriverFactory is unpacking the driver input here, ideally only the driver should need to
      // depend on the content and implementation of the input.  This seems to be to be a bigger deal that passing the
      // known at this level PSPool down.
      new_driver =
          std::make_unique<VMCBatched>(project_data_, std::move(qmcdriver_input),
                                       makeEstimatorManager(emi, qmcdriver_input.get_estimator_manager_input(),
                                                            primaryPsi),
                                       std::move(vmcdriver_input), qmc_system,
                                       MCPopulation(comm->size(), comm->rank(), qmc_system, primaryPsi, primaryH),
                                       RandomNumberControl::getChildrenRefs(), qmc_system.getSampleStack(), comm,
                                       imported_md_orbital_ecp);

      new_driver->setUpdateMode(is_pbyp_move);
    }
    else if (das.new_run_type == QMCRunType::WF_TRAIN_BATCH)
    {
      app_summary() << "\n========================================"
                       "\n  Reading wavefunction-training XML input section"
                       "\n========================================"
                    << std::endl;

      QMCDriverInput qmcdriver_input;
      VMCDriverInput vmcdriver_input;
      WFTrainDriverInput training_input;
      try
      {
        qmcdriver_input.readXML(cur);
        vmcdriver_input.readXML(cur);
        training_input.readXML(cur);
      }
      catch (const std::exception& error)
      {
        throw UniformCommunicateError(error.what());
      }

      const bool is_pbyp_move = vmcdriver_input.get_move_kind() == VMCDriverInput::MoveKind::PBYP;
      // Keep immutable targets separate from the selected Markov-chain wavefunction.
      const TrialWaveFunction* frozen_teacher = training_input.getObjective() != WFTrainDriverInput::Objective::ENERGY
          ? &getPsi(training_input.getTeacherWaveFunction()) : nullptr;
      if (frozen_teacher == &primaryPsi)
        throw UniformCommunicateError("WFTrain teacher objectives require a wavefunction distinct from the student");
      TrialWaveFunction& sampling_wavefunction =
          frozen_teacher && !training_input.samplesOrbitalStudent()
          ? getPsi(training_input.getTeacherWaveFunction())
          : primaryPsi;
      // Training owns scalar energy/variance diagnostics. Ignore supplemental
      // requests before constructing estimator resources, including teacher runs.
      if ((emi || qmcdriver_input.get_estimator_manager_input()) && comm->rank() == 0)
        app_warning() << "WFTrain ignores supplemental global/local estimator requests; "
                         "sampling retains energy, variance and basic diagnostics." << std::endl;
      UPtr<EstimatorManagerNew> estimator_manager = makeEstimatorManager(
          std::nullopt, std::nullopt, sampling_wavefunction);
      new_driver = std::make_unique<WFTrainBatched>(
          project_data_, std::move(qmcdriver_input), std::move(estimator_manager), std::move(vmcdriver_input),
          std::move(training_input), qmc_system, primaryPsi,
          MCPopulation(comm->size(), comm->rank(), qmc_system, sampling_wavefunction, primaryH),
          RandomNumberControl::getChildrenRefs(), comm, frozen_teacher);
      new_driver->setUpdateMode(is_pbyp_move);
    }
    else if (das.new_run_type == QMCRunType::DMC)
    {
      DMCFactory fac(das.what_to_do[UPDATE_MODE], das.what_to_do[GPU_MODE], cur);
      new_driver = fac.create(project_data_, qmc_system, primaryPsi, primaryH, comm, das.enable_profiling);
    }
    else if (das.new_run_type == QMCRunType::DMC_BATCH)
    {
      app_summary() << "\n========================================"
                       "\n  Reading DMC driver XML input section"
                       "\n========================================"
                    << std::endl;

      QMCDriverInput qmcdriver_input;
      DMCDriverInput dmcdriver_input;
      try
      {
        qmcdriver_input.readXML(cur);
        dmcdriver_input.readXML(cur);
      }
      catch (const std::exception& e)
      {
        throw UniformCommunicateError(e.what());
      }

      if (dmcdriver_input.get_l2_diffusion())
      {
        if (qmc_system.isSpinor())
          throw UniformCommunicateError("L2 diffusion is not supported for spinor particle sets.");
        if (!primaryH.has_L2())
          throw UniformCommunicateError("L2 diffusion was requested, but the Hamiltonian has no L2 potential.");
      }

      new_driver =
          std::make_unique<DMCBatched>(project_data_, std::move(qmcdriver_input),
                                       makeEstimatorManager(emi, qmcdriver_input.get_estimator_manager_input(),
                                                            primaryPsi),
                                       std::move(dmcdriver_input), qmc_system,
                                       MCPopulation(comm->size(), comm->rank(), qmc_system, primaryPsi, primaryH),
                                       RandomNumberControl::getChildrenRefs(), comm);
    }
    else if (das.new_run_type == QMCRunType::RMC)
    {
      RMCFactory fac(das.what_to_do[UPDATE_MODE], cur);
      new_driver = fac.create(project_data_, qmc_system, primaryPsi, primaryH, comm);
    }
    else if (das.new_run_type == QMCRunType::LINEAR_OPTIMIZE)
    {
#ifdef MIXED_PRECISION
      APP_ABORT(
          "QMCDriverFactory::createQMCDriver : method=\"linear\" is not safe with CPU mixed precision. Please use "
          "full precision build instead.");
#endif
      QMCFixedSampleLinearOptimize* opt =
          new QMCFixedSampleLinearOptimize(project_data_, qmc_system, primaryPsi, primaryH, comm);
      //ZeroVarianceOptimize *opt = new ZeroVarianceOptimize(qmc_system,primaryPsi,primaryH );
      opt->setWaveFunctionNode(wavefunction_pool.getWaveFunctionNode("psi0"));
      new_driver.reset(opt);
    }
    else if (das.new_run_type == QMCRunType::LINEAR_OPTIMIZE_BATCH)
    {
#ifdef MIXED_PRECISION
      APP_ABORT("QMCDriverFactory::createQMCDriver : method=\"linear_batch\" is not safe with CPU mixed precision. "
                "Please use full precision build instead.");
#endif
      app_summary() << "\n========================================"
                       "\n  Reading WFOpt driver XML input section"
                       "\n========================================"
                    << std::endl;

      QMCDriverInput qmcdriver_input;
      VMCDriverInput vmcdriver_input;
      try
      {
        qmcdriver_input.readXML(cur);
        vmcdriver_input.readXML(cur);
      }
      catch (const std::exception& e)
      {
        throw UniformCommunicateError(e.what());
      }

      if (qmcdriver_input.get_estimator_manager_input())
        app_warning() << "The batched wavefunction optimization driver ignores local estimator input.";
      if (emi)
        app_warning() << "The batched wavefunction optimization driver ignores global estimator input.";

      auto opt = std::make_unique<QMCFixedSampleLinearOptimizeBatched>(project_data_, std::move(qmcdriver_input),
                                                                       std::move(vmcdriver_input), qmc_system,
                                                                       MCPopulation(comm->size(), comm->rank(),
                                                                                    qmc_system, primaryPsi, primaryH),
                                                                       RandomNumberControl::getChildrenRefs(),
                                                                       qmc_system.getSampleStack(), comm);
      opt->setWaveFunctionNode(wavefunction_pool.getWaveFunctionNode("psi0"));
      new_driver = std::move(opt);
    }
    else if (das.new_run_type == QMCRunType::WF_TEST)
    {
      app_log() << "Testing wavefunctions." << std::endl;
      QMCDriverInterface* temp_ptr =
          new WaveFunctionTester(project_data_, qmc_system, primaryPsi, primaryH, particle_pool, comm);
      new_driver.reset(temp_ptr);
    }
    else
    {
      APP_ABORT("Unhandled run type: " << static_cast<int>(das.new_run_type));
    }
  }

  infoSummary.flush();
  infoLog.flush();
  //add trace information
  bool allow_traces = das.traces_tag == "yes" ||
      (das.traces_tag == "none" && (das.new_run_type == QMCRunType::VMC || das.new_run_type == QMCRunType::DMC));
  new_driver->requestTraces(allow_traces);

  //add trace information
  bool allow_walker_logs = das.walkerlogs_tag == "yes" ||
      (das.walkerlogs_tag == "none" &&
       (das.new_run_type == QMCRunType::VMC || das.new_run_type == QMCRunType::DMC ||
        das.new_run_type == QMCRunType::VMC_BATCH || das.new_run_type == QMCRunType::DMC_BATCH));
  new_driver->requestWalkerLogs(allow_walker_logs);

  return new_driver;
}

} // namespace qmcplusplus
