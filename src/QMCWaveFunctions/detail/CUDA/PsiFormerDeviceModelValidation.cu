//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceModelValidation.cu
 * @brief Opt-in full-model GPU forward/VGL validation against native CPU execution.
 *
 * CPU-only mode produces references without consulting a GPU. Device mode runs
 * the prepared graph and checks every stored configuration; no CPU fallback is
 * accepted. Ordinary weighted parameter reverse is checked for regular channels;
 * Numerical kinetic and synthetic captured-grid responses are also checked;
 * production training/physical Hamiltonian binding are not claimed here.
 */

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceNetworkReverse.h"
#include "PsiFormerDeviceRawFeatureReverseValidation.h"
#include "PsiFormerDeviceCoordinateGradientValidation.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "PsiFormerDeviceKineticTrunkValidation.h"
#include "PsiFormerDeviceTargetKineticValidation.h"
#include "PsiFormerDeviceTargetKineticSpotValidation.h"
#include "PsiFormerDeviceNonlocalResponse.h"
#include "PsiFormerDeviceCrowdHooks.h"
#include "PsiFormerDeviceBatchBackend.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "PsiFormerDeviceTargetSelectionValidation.h"
#include "PsiFormerDeviceKineticContractionValidation.h"
#include "PsiFormerDeviceCoordinatesValidation.h"
#include "PsiFormerDeviceMetropolisValidation.h"
#include "PsiFormerDeviceResidentAdamValidation.h"
#include "PsiFormerDeviceSrMetricValidation.h"
#include "PsiFormerDevicePcgValidation.h"
#include "PsiFormerDeviceEnergySrValidation.h"
#include "PsiFormerDeviceRetainedEnergySrValidation.h"

#include <fstream>
#include <iostream>

namespace
{
using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;

/// Reject nonfinite or inaccurate GPU observables even under host fast-math.
void near(double actual, double expected, const char* field)
{
  if (!device_math::isFiniteBinary64(actual) || !device_math::isFiniteBinary64(expected) ||
      std::abs(actual - expected) > 2e-7 * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(field) + " differs between device and CPU");
}

/// Retain independent CPU observables in a portable row per configuration.
struct Reference
{
  double sign = 0, logabs = 0, kinetic = 0;
  std::vector<double> gradient, lap_log, lap_ratio;
};

/// Evaluate all configurations with the current direct CPU path and check one native oracle.
std::vector<Reference> cpuReferences(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
                                   std::size_t limit = 0)
{
  pf::DirectValueExecutor value(model, plan);
  pf::DirectSpatialExecutor spatial(model, value, plan);
  auto workspace = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  std::vector<Reference> result;
  const auto count = limit == 0 ? model.cfg.nconfig : std::min(limit, model.cfg.nconfig);
  for (std::size_t index = 0; index < count; ++index)
  {
    auto positions = model.cfg.configuration(index);
    workspace->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
    const auto evaluated = spatial.evaluateFull(*workspace);
    Reference row;
    row.sign = evaluated.sign;
    row.logabs = evaluated.logabs;
    row.gradient.assign(evaluated.gradient.data(), evaluated.gradient.data() + evaluated.gradient.size());
    row.lap_log.assign(evaluated.lap_log.data(), evaluated.lap_log.data() + evaluated.lap_log.size());
    row.lap_ratio.assign(evaluated.lap_ratio.data(), evaluated.lap_ratio.data() + evaluated.lap_ratio.size());
    for (double laplacian : row.lap_ratio) row.kinetic -= 0.5 * laplacian;
    if (index == 0)
    {
      pf::EvaluationRequest request;
      request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
      const auto oracle = model.evaluate(positions, request);
      near(row.sign, oracle.sign, "CPU sign/oracle");
      near(row.logabs, oracle.logabs, "CPU logabs/oracle");
      for (std::size_t lane = 0; lane < row.gradient.size(); ++lane)
        near(row.gradient[lane], oracle.gradient[lane], "CPU gradient/oracle");
      for (std::size_t electron = 0; electron < model.ne; ++electron)
        near(row.lap_ratio[electron], oracle.lap_ratio[electron], "CPU Laplacian/oracle");
    }
    result.push_back(std::move(row));
  }
  return result;
}

/// Save complete VGL references, not statistical Monte Carlo error bars.
void writeReferences(const std::string& filename, const std::vector<Reference>& references)
{
  std::ofstream output(filename);
  if (!output) throw std::runtime_error("Cannot open reference output");
  output << std::setprecision(17) << "# index sign logabs kinetic gradient... lap_log... lap_ratio...\n";
  for (std::size_t index = 0; index < references.size(); ++index)
  {
    const auto& row = references[index];
    output << index << ' ' << row.sign << ' ' << row.logabs << ' ' << row.kinetic;
    for (const auto* values : {&row.gradient, &row.lap_log, &row.lap_ratio})
      for (double value : *values) output << ' ' << value;
    output << '\n';
  }
  if (!output) throw std::runtime_error("Reference output write failed");
}
} // namespace

/// Load portable fixtures; require actual-device work unless explicitly CPU-only.
int main(int argc, char** argv)
{
  try
  {
    // Keep metadata qualification before fixtures and every device/runtime query.
    if (argc == 2 && std::string(argv[1]) == "--target-cost-check")
    {
      validateTargetCosts();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--target-selection-check")
    {
      validateTargetSelectionCosts();
      return 0;
    }
    if (argc < 4)
      throw std::invalid_argument("Usage: psiformer_device_model_validation --target-cost-check OR PARAMETERS CONFIGURATIONS OUTPUT "
                                  "[--cpu-only] [--coordinate-only] [--device-coordinates-only] [--target-kinetic-spot] [--kinetic-trunk-only] [--target-selection-spot] [--metropolis-only] [--resident-adam-only] [--sr-metric-only] [--resident-pcg-only] [--energy-sr-only] [--retained-energy-sr-only] "
                                  "[--shape DETERMINANTS WIDTH HEADS BLOCKS] "
                                  "[--score-output CANONICAL_F64] [--jvp-output CONFIGURATION_F64] "
                                  "[--kinetic-output CANONICAL_F64] [--nonlocal-output CANONICAL_F64]");

    // Explicit dimensions permit compact generated fixtures without changing
    // the established DeepQMC profile used by existing invocation scripts.
    bool cpu_only = false;
    bool coordinate_only = false;
    bool device_coordinates_only = false;
    bool target_kinetic_spot = false;
    bool kinetic_trunk_only = false;
    bool target_selection_spot = false;
    bool metropolis_only = false;
    bool resident_adam_only = false;
    bool sr_metric_only = false;
    bool resident_pcg_only = false;
    bool energy_sr_only = false;
    bool retained_energy_sr_only = false;
    std::size_t determinants = 16, width = 256, heads = 4, blocks = 4;
    bool shape_supplied = false;
    std::string score_output;
    std::string jvp_output;
    std::string kinetic_output;
    std::string nonlocal_output;
    for (int argument = 4; argument < argc; ++argument)
    {
      const std::string option(argv[argument]);
      if (option == "--cpu-only" && !cpu_only)
        cpu_only = true;
      else if (option == "--coordinate-only" && !coordinate_only)
        coordinate_only = true;
      else if (option == "--device-coordinates-only" && !device_coordinates_only)
        device_coordinates_only = true;
      else if (option == "--kinetic-trunk-only" && !kinetic_trunk_only)
        kinetic_trunk_only = true;
      else if (option == "--target-kinetic-spot" && !target_kinetic_spot)
        target_kinetic_spot = true;
      else if (option == "--target-selection-spot" && !target_selection_spot)
        target_selection_spot = true;
      else if (option == "--metropolis-only" && !metropolis_only)
        metropolis_only = true;
      else if (option == "--resident-adam-only" && !resident_adam_only)
        resident_adam_only = true;
      else if (option == "--sr-metric-only" && !sr_metric_only)
        sr_metric_only = true;
      else if (option == "--resident-pcg-only" && !resident_pcg_only)
        resident_pcg_only = true;
      else if (option == "--energy-sr-only" && !energy_sr_only)
        energy_sr_only = true;
      else if (option == "--retained-energy-sr-only" && !retained_energy_sr_only)
        retained_energy_sr_only = true;
      else if (option == "--score-output" && score_output.empty() && argument + 1 < argc)
        score_output = argv[++argument];
      else if (option == "--jvp-output" && jvp_output.empty() && argument + 1 < argc)
        jvp_output = argv[++argument];
      else if (option == "--kinetic-output" && kinetic_output.empty() && argument + 1 < argc)
        kinetic_output = argv[++argument];
      else if (option == "--nonlocal-output" && nonlocal_output.empty() && argument + 1 < argc)
        nonlocal_output = argv[++argument];
      else if (option == "--shape" && !shape_supplied && argument + 4 < argc)
      {
        std::size_t* dimensions[] = {&determinants, &width, &heads, &blocks};
        for (auto* dimension : dimensions)
        {
          const std::string token(argv[++argument]);
          if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Shape dimensions must be positive decimal integers");
          std::size_t consumed = 0;
          *dimension = std::stoull(token, &consumed);
          if (*dimension == 0 || consumed != token.size())
            throw std::invalid_argument("Shape dimensions must be positive decimal integers");
        }
        shape_supplied = true;
      }
      else
        throw std::invalid_argument("Unknown, duplicate or incomplete validation option: " + option);
    }
    if (int(sr_metric_only) + int(resident_adam_only) + int(resident_pcg_only) + int(energy_sr_only) + int(retained_energy_sr_only) > 1)
      throw std::invalid_argument("Resident Adam/SR metric/PCG/ENERGY SR qualifications are mutually exclusive");
    if ((resident_adam_only || sr_metric_only || resident_pcg_only || energy_sr_only || retained_energy_sr_only) && (cpu_only || coordinate_only || device_coordinates_only || metropolis_only ||
        target_kinetic_spot || kinetic_trunk_only || target_selection_spot || !score_output.empty() || !jvp_output.empty() ||
        !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("Resident Adam/SR metric/PCG qualification cannot request other modes or exports");
    if (cpu_only && (!score_output.empty() || !jvp_output.empty() || !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("Canonical GPU score export is incompatible with --cpu-only");
    if (coordinate_only && (cpu_only || !score_output.empty() || !jvp_output.empty() ||
        !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("Coordinate-only validation cannot request CPU-only or parameter exports");
    if (target_selection_spot && target_kinetic_spot)
      throw std::invalid_argument("Target selection spot cannot request another target spot mode");
    if ((target_kinetic_spot || target_selection_spot) && (cpu_only || coordinate_only || device_coordinates_only || metropolis_only ||
        !score_output.empty() || !jvp_output.empty() || !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("TARGET kinetic spot cannot request other modes or exports");
    if (metropolis_only && (cpu_only || coordinate_only || device_coordinates_only || target_kinetic_spot || target_selection_spot ||
        !score_output.empty() || !jvp_output.empty() || !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("Metropolis-only validation cannot request other modes or exports");
    if ((resident_pcg_only || energy_sr_only || retained_energy_sr_only) && (!shape_supplied || determinants != 4 || width != 16 || heads != 4 || blocks != 2))
      throw std::invalid_argument("Resident PCG/ENERGY SR gate requires explicit tiny --shape 4 16 4 2");
    pf::PsiFormer model(pf::Parameters(argv[1]), pf::ConfigData(argv[2]),
                        determinants, width, heads, blocks);
    const ModelShape shape{model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0],
                            model.ndet, model.dim, model.heads, model.blocks};
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape);
    if ((resident_pcg_only || energy_sr_only || retained_energy_sr_only) && (plan.parameterCount() != 3922 || model.ne != 4 || model.cfg.nup != 2))
      throw std::invalid_argument("Resident PCG/ENERGY SR gate requires P3922/Ne4/Nup2, before GPU image preparation");
    if (target_kinetic_spot || target_selection_spot || kinetic_trunk_only)
    {
      // Branch before CPU references, broad validation owners or parameter traces.
      int device_count = 0;
#ifdef QMC_CUDA2HIP
      const auto discovered = hipGetDeviceCount(&device_count);
#else
      const auto discovered = cudaGetDeviceCount(&device_count);
#endif
      if (discovered != success || device_count == 0)
      {
        std::cerr << "NOT_RUN: no usable GPU for TARGET kinetic spot; exit 77\n";
        return 77;
      }
      if (model.cfg.electrons.x.size() < 3 * model.ne)
        throw std::invalid_argument("TARGET kinetic spot requires one complete configuration");
      qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
      AcceleratorBlasHandle handle(queue);
      const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
      const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      std::ofstream output(argv[3]);
      if (!output)
        throw std::runtime_error("Cannot open TARGET kinetic spot output");
      if (kinetic_trunk_only)
        KineticTrunkValidationAccess::run(model, image, queue, handle);
      else if (target_selection_spot)
        validateTargetSelectionNumerics(image, positions, queue, handle, output);
      else
        validateTargetKineticSpot(image, positions, queue, handle, output);
      if (!output)
        throw std::runtime_error("Cannot write TARGET kinetic spot output");
      return 0;
    }
    const auto references = (metropolis_only || resident_adam_only || sr_metric_only || resident_pcg_only || energy_sr_only || retained_energy_sr_only) ? std::vector<Reference>{} : cpuReferences(model, plan, coordinate_only ? 3 : 0);
    if (!metropolis_only && !resident_adam_only && !sr_metric_only && !resident_pcg_only && !energy_sr_only && !retained_energy_sr_only)
      writeReferences(argv[3], references);
    if (cpu_only)
    {
      std::cout << "PASS CPU direct VGL/native oracle; GPU model validation NOT_RUN\n";
      return 0;
    }
    int device_count = 0;
#ifdef QMC_CUDA2HIP
    const auto discovered = hipGetDeviceCount(&device_count);
#else
    const auto discovered = cudaGetDeviceCount(&device_count);
#endif
    if (discovered != success || device_count == 0)
    {
      std::cerr << "NOT_RUN: no usable GPU; CPU references saved; exit 77\n";
      return 77;
    }
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle handle(queue);
    if (resident_adam_only || sr_metric_only || resident_pcg_only || energy_sr_only || retained_energy_sr_only)
    {
      const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
      if (retained_energy_sr_only)
        validateRetainedEnergySr(model, plan, image);
      else if (energy_sr_only)
        validateEnergySr(model, plan, image);
      else if (resident_pcg_only)
        validateResidentPcg(model, plan, image);
      else if (sr_metric_only)
        validateSrMetric(model, plan, image);
      else
        validateResidentAdam(model, plan, image);
      return 0;
    }
    if (metropolis_only)
    {
      // No broad CPU/GPU reference matrix or production sampler is constructed here.
      const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
      validateDeviceMetropolis(model, image, queue, handle);
      std::ofstream output(argv[3]);
      output << "PASS injected numerical Metropolis; costs/transfers in stdout; XML/RNG/Hamiltonian not integrated\n";
      if (!output)
        throw std::runtime_error("Cannot save injected Metropolis qualification result");
      return 0;
    }
    if (device_coordinates_only)
    {
      // Qualify only the typed source contract, without the broad model matrix.
      const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
      validateDeviceCoordinates(model, image, queue, handle);
      validateRetainedCoordinateReplays(model, image);
      return 0;
    }
    if (coordinate_only)
    {
      // Exit before constructing any score/kinetic owners or parameter traces.
      const auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
      const std::vector<double> prefix(model.cfg.electrons.x.begin(),
          model.cfg.electrons.x.begin() + references.size() * model.ne * 3);
      // Include no-P admission and lifecycle contracts, not only the warmed spot.
      validateCoordinateGradient(image, model, prefix, queue, handle, false);
      std::cout << "PASS coordinate-only: independent CPU parity and no-P owner contracts\n";
      return 0;
    }
    // Successor images share fixed geometry/plan metadata, not old parameters.
    // Rebinding keeps scratch stable and rejects stale results or foreign families.
    {
      auto original = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 30);
      std::weak_ptr<const PreparedDeviceModel> old_image = original;
      PreparedDeviceInference rebound(original, 1, queue, handle, SpatialJetMode::VALUE_ONLY, true);
      const auto prepared_bytes = rebound.deviceWorkspaceBytes();
      const auto* stable_plan = &rebound.modelImage()->plan();
      std::vector<double> one_position(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      rebound.evaluate(one_position);
      PreparedDeviceScoreReverse rebound_score(original, 1, queue, handle);
      PreparedDeviceScoreJvp rebound_jvp(original, 1, queue, handle);
      PreparedDeviceKineticReverse rebound_kinetic(original, 1, queue, handle);
      PreparedDeviceNonlocalResponse rebound_nonlocal(original, 1, 1, queue, handle);
      const auto product_bytes = rebound_score.deviceWorkspaceBytes() + rebound_jvp.deviceWorkspaceBytes() +
          rebound_kinetic.deviceWorkspaceBytes() + rebound_nonlocal.deviceWorkspaceBytes();
      rebound_score.reverse(rebound, {1.0});
      std::vector<double> rebound_direction(plan.parameterCount(), 0.0);
      rebound_direction[0] = 1.0;
      rebound_jvp.evaluate(rebound, rebound_direction);
      auto changed = model.p.flat_values();
      changed[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += 0.003;
      changed[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] += 0.01;
      auto successor = original->makeUpdatedImage(changed, 31);
      if (successor->metadataFamily().get() != original->metadataFamily().get() ||
          successor->nuclei() != original->nuclei())
        throw std::runtime_error("Successor duplicated fixed metadata/geometry");
      rebound.rebindModel(successor);
      rebound_score.rebindModel(successor);
      rebound_jvp.rebindModel(successor);
      rebound_kinetic.rebindModel(successor);
      rebound_nonlocal.rebindModel(successor);
      original.reset();
      if (!old_image.expired() || &rebound.modelImage()->plan() != stable_plan ||
          rebound.deviceWorkspaceBytes() != prepared_bytes)
        throw std::runtime_error("Graph rebind retained retired parameters or changed prepared storage");
      std::vector<double> phase(1), logabs(1), no_laplacians;
      bool stale_image_result_rejected = false;
      try { rebound.downloadResults(phase, logabs, no_laplacians); }
      catch (const std::logic_error&) { stale_image_result_rejected = true; }
      if (!stale_image_result_rejected)
        throw std::runtime_error("Graph rebind exposed old-image results");
      std::vector<double> rebound_response(plan.parameterCount()), rebound_products(1);
      bool stale_score_after_rebind = false, stale_jvp_after_rebind = false;
      try { rebound_score.downloadResponse(rebound_response); }
      catch (const std::logic_error&) { stale_score_after_rebind = true; }
      try { rebound_jvp.downloadProducts(rebound_products); }
      catch (const std::logic_error&) { stale_jvp_after_rebind = true; }
      if (!stale_score_after_rebind || !stale_jvp_after_rebind)
        throw std::runtime_error("Product rebind exposed old-image results");
      rebound.evaluate(one_position);
      bool rebound_direction_rejected = false;
      try { (void)rebound_jvp.evaluatePrepared(rebound); }
      catch (const std::logic_error&) { rebound_direction_rejected = true; }
      if (!rebound_direction_rejected)
        throw std::runtime_error("Product rebind retained a direction from the previous image");
      rebound.downloadResults(phase, logabs, no_laplacians);
      pf::PsiFormer changed_model = model;
      changed_model.p.set_flat_values(changed);
      pf::DirectValueExecutor changed_value(changed_model, plan);
      auto changed_workspace = changed_value.makeWorkspace();
      changed_workspace->setPositions(pf::GeometryPositionView::interleaved(one_position.data(), model.ne));
      const auto expected_changed = changed_value.evaluate(*changed_workspace);
      near(phase[0], expected_changed.sign, "rebound image sign");
      near(logabs[0], expected_changed.logabs, "rebound image value");
      rebound_score.reverse(rebound, {1.0});
      rebound_score.downloadResponse(rebound_response);
      rebound_jvp.evaluate(rebound, rebound_direction);
      rebound_jvp.downloadProducts(rebound_products);
      pf::DirectScoreExecutor changed_score(changed_model, plan);
      auto changed_score_workspace = changed_score.makeWorkspace();
      changed_score_workspace->setPositions(pf::GeometryPositionView::interleaved(one_position.data(), model.ne));
      const auto expected_score = changed_score.evaluate(*changed_score_workspace);
      near(rebound_products[0], expected_score.parameter_score[0], "rebound JVP CPU parity");
      for (std::size_t parameter = 0; parameter < rebound_response.size(); ++parameter)
        near(rebound_response[parameter], expected_score.parameter_score[parameter], "rebound score CPU parity");
      if (product_bytes != rebound_score.deviceWorkspaceBytes() + rebound_jvp.deviceWorkspaceBytes() +
          rebound_kinetic.deviceWorkspaceBytes() + rebound_nonlocal.deviceWorkspaceBytes())
        throw std::runtime_error("Product image rebind reallocated prepared scratch");
      auto rollback = successor->makeUpdatedImage(model.p.flat_values(), 32);
      rebound.rebindModel(rollback);
      rebound.evaluate(one_position);
      rebound.downloadResults(phase, logabs, no_laplacians);
      near(logabs[0], references[0].logabs, "rebound rollback value");
      auto foreign_family = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 33);
      bool foreign_rebind_rejected = false, older_rebind_rejected = false;
      try { rebound.rebindModel(foreign_family); }
      catch (const std::invalid_argument&) { foreign_rebind_rejected = true; }
      try { rebound.rebindModel(successor); }
      catch (const std::invalid_argument&) { older_rebind_rejected = true; }
      if (!foreign_rebind_rejected || !older_rebind_rejected || rebound.modelImage().get() != rollback.get())
        throw std::runtime_error("Graph rebind accepted an incompatible/non-increasing image");
      std::cout << "PASS immutable image successors: shared geometry, retired-image release, stable scratch, CPU parity, rollback\n";
    }
    // Distinct numerical workspaces share one immutable uploaded parameter image.
    auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 17);
    PreparedDeviceInference inference(image, model.cfg.nconfig, queue, handle);
    inference.evaluate(model.cfg.electrons.x);
    const auto& layout = inference.resultLayout();
    std::vector<double> phases(model.cfg.nconfig), jets(spatialJetSpanElements(layout)),
        ratios(model.cfg.nconfig * model.ne);
    inference.downloadResults(phases, jets, ratios);
    auto verify = [&]()
    {
      for (std::size_t index = 0; index < references.size(); ++index)
      {
        const auto& reference = references[index];
        near(phases[index], reference.sign, "model sign");
        near(jets[layout.uncheckedValueOffset(index, 0)], reference.logabs, "model logabs");
        double kinetic = 0;
        for (std::size_t lane = 0; lane < reference.gradient.size(); ++lane)
          near(jets[layout.uncheckedGradientOffset(index, lane, 0)], reference.gradient[lane], "model gradient");
        for (std::size_t electron = 0; electron < model.ne; ++electron)
        {
          near(jets[layout.uncheckedLaplacianOffset(index, electron, 0)], reference.lap_log[electron], "model log Laplacian");
          near(ratios[index * model.ne + electron], reference.lap_ratio[electron], "model Laplacian ratio");
          kinetic -= 0.5 * ratios[index * model.ne + electron];
        }
        near(kinetic, reference.kinetic, "model kinetic energy");
      }
    };
    verify();
    // Repeat the same prepared graph to constrain stale scratch and buffer reuse.
    inference.evaluate(model.cfg.electrons.x);
    inference.downloadResults(phases, jets, ratios);
    verify();

    // Explicit non-retained target tiling must preserve one physical result and
    // one geometry upload/generation while keeping max-K scratch stable.
    {
      constexpr std::size_t capacity = 3;
      std::vector<double> target_positions;
      for (std::size_t sample = 0; sample < capacity; ++sample)
      {
        const auto first = model.cfg.electrons.x.begin() + (sample % references.size()) * 3 * model.ne;
        target_positions.insert(target_positions.end(), first, first + 3 * model.ne);
      }
      int ordinal = 0;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&ordinal), "target VGL metrics device");
#else
      checkRuntime(cudaGetDevice(&ordinal), "target VGL metrics device");
#endif
      auto& metrics = deviceMemoryMetrics(ordinal);
      const auto policy = InferenceStatusPolicy::DEFERRED_NEURAL;
      const auto admitted = [&](std::size_t count) {
        const auto cost = estimateTargetVglCost(plan, capacity, count);
        return TargetVglPreparation{count, cost.device_workspace_bytes, cost.charged_scalar_work_units};
      };
      const auto expect_invalid = [&](TargetVglPreparation preparation, SpatialJetMode mode,
                                      bool score_tape, bool spatial_tape) {
        const auto before = metrics.snapshot();
        bool rejected = false;
        try { PreparedDeviceInference graph(image, capacity, queue, handle, mode,
                                             score_tape, spatial_tape, policy, preparation); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (const std::length_error&) { rejected = true; }
        const auto after = metrics.snapshot();
        if (!rejected || before.allocation_calls != after.allocation_calls || before.live_bytes != after.live_bytes)
          throw std::runtime_error("Target VGL invalid preparation allocated device scratch or was admitted");
      };
      auto options = admitted(1);
      auto insufficient = options;
      --insufficient.device_workspace_budget_bytes;
      expect_invalid(insufficient, SpatialJetMode::FULL_VGL, false, false);
      insufficient = options;
      --insufficient.scalar_work_budget;
      expect_invalid(insufficient, SpatialJetMode::FULL_VGL, false, false);
      expect_invalid({0, 1, 1}, SpatialJetMode::FULL_VGL, false, false);
      expect_invalid({1, 0, options.scalar_work_budget}, SpatialJetMode::FULL_VGL, false, false);
      expect_invalid({1, options.device_workspace_budget_bytes, 0}, SpatialJetMode::FULL_VGL, false, false);
      expect_invalid({model.ne + 1, options.device_workspace_budget_bytes, options.scalar_work_budget},
                     SpatialJetMode::FULL_VGL, false, false);
      expect_invalid(options, SpatialJetMode::VALUE_ONLY, false, false);
      expect_invalid(options, SpatialJetMode::FULL_VGL, false, true);
      expect_invalid(options, SpatialJetMode::VALUE_ONLY, true, false);
      bool overflow_rejected = false;
      try { (void)estimateTargetVglCost(plan, std::numeric_limits<std::size_t>::max(), 1); }
      catch (const std::length_error&) { overflow_rejected = true; }
      if (!overflow_rejected)
        throw std::runtime_error("Target VGL preparation cost overflow was admitted");

      device_determinant::CombinationMetadata combination;
      combination.status = device_determinant::CombinationStatus::REGULAR;
      admitTargetVglBase(1, -10, 1, -10, combination);
      const auto expect_bad_base = [&](double phase, double logabs,
                                        device_determinant::CombinationMetadata metadata) {
        bool rejected = false;
        try { admitTargetVglBase(phase, logabs, 1, -10, metadata); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected)
          throw std::runtime_error("Target VGL inconsistent/cancelling base was admitted");
      };
      expect_bad_base(-1, -10, combination);
      expect_bad_base(1, -9, combination);
      expect_bad_base(1, std::numeric_limits<double>::infinity(), combination);
      combination.severe_cancellation = true;
      expect_bad_base(1, -10, combination);

      PreparedDeviceInference full_target_reference(image, capacity, queue, handle);
      std::size_t smallest_target_bytes = 0;
      for (const auto target_count : {std::size_t{1}, std::min<std::size_t>(2, model.ne),
                                     std::min<std::size_t>(3, model.ne), model.ne})
      {
        PreparedDeviceInference target_graph(image, capacity, queue, handle, SpatialJetMode::FULL_VGL,
                                              false, false, policy, admitted(target_count));
        const auto prepared_bytes = target_graph.deviceWorkspaceBytes();
        const auto cost = estimateTargetVglCost(plan, capacity, target_count);
        if (prepared_bytes != cost.device_workspace_bytes ||
            target_graph.hostWorkspaceBytes() != cost.host_workspace_bytes ||
            target_graph.targetVglCost().tile_count != cost.tile_count)
          throw std::runtime_error("Target VGL exact pre-allocation cost disagrees with prepared storage");
        if (target_count == 1)
          smallest_target_bytes = prepared_bytes;
        for (const std::size_t count : {capacity, std::size_t{1}, std::size_t{2}, capacity})
        {
          const std::vector<double> prefix(target_positions.begin(), target_positions.begin() + count * 3 * model.ne);
          const auto before = metrics.snapshot();
          const auto generation = target_graph.evaluationGeneration();
          target_graph.evaluate(prefix);
          const auto after = metrics.snapshot();
          if (after.upload_calls != before.upload_calls + 1 ||
              after.uploaded_bytes != before.uploaded_bytes + checkedDeviceBytes<double>(prefix.size()) ||
              after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
              after.live_bytes != before.live_bytes || target_graph.evaluationGeneration() != generation + 1 ||
              target_graph.admittedTargetTiles() != cost.tile_count ||
              target_graph.deviceWorkspaceBytes() != prepared_bytes)
            throw std::runtime_error("Target VGL request lost one-upload/generation or stable preparation");
          const auto& target_layout = target_graph.resultLayout();
          if (!hasCompleteSpatialTargets(target_layout) || target_layout.configuration_count != count)
            throw std::runtime_error("Target VGL exposed a local rather than physical result layout");
          std::vector<double> target_phase(count), target_jets(spatialJetSpanElements(target_layout)),
              target_ratios(count * model.ne);
          target_graph.downloadResults(target_phase, target_jets, target_ratios);
          for (std::size_t sample = 0; sample < count; ++sample)
          {
            const auto& reference = references[sample % references.size()];
            near(target_phase[sample], reference.sign, "target VGL physical phase");
            near(target_jets[target_layout.uncheckedValueOffset(sample, 0)], reference.logabs,
                "target VGL physical log amplitude");
            for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
              near(target_jets[target_layout.uncheckedGradientOffset(sample, lane, 0)], reference.gradient[lane],
                  "target VGL physical gradient");
            for (std::size_t electron = 0; electron < model.ne; ++electron)
            {
              near(target_jets[target_layout.uncheckedLaplacianOffset(sample, electron, 0)],
                  reference.lap_log[electron], "target VGL physical log Laplacian");
              near(target_ratios[sample * model.ne + electron], reference.lap_ratio[electron],
                  "target VGL physical Laplacian ratio");
            }
          }
        }
        bool reverse_rejected = false;
        try { (void)target_graph.spatialHeadTape(); }
        catch (const std::logic_error&) { reverse_rejected = true; }
        if (!reverse_rejected)
          throw std::runtime_error("Target VGL exposed an unsupported retained spatial tape");
        auto compute_parameters = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue);
        auto adapter = std::make_shared<PreparedDeviceMixedValueDense>(compute_parameters,
            capacity * model.ne * (1 + 4 * target_count), queue, handle);
        bool mixed_rejected = false;
        try { target_graph.setMixedDense(adapter); }
        catch (const std::invalid_argument&) { mixed_rejected = true; }
        if (!mixed_rejected)
          throw std::runtime_error("Target VGL admitted unsupported mixed arithmetic");
        const auto successor = image->makeUpdatedImage(model.p.flat_values(), 18);
        target_graph.rebindModel(successor);
        bool stale_rejected = false;
        std::vector<double> stale_phase(capacity), stale_jets(spatialJetSpanElements(target_graph.resultLayout())),
            stale_ratios(capacity * model.ne);
        try { target_graph.downloadResults(stale_phase, stale_jets, stale_ratios); }
        catch (const std::logic_error&) { stale_rejected = true; }
        if (!stale_rejected)
          throw std::runtime_error("Target VGL publication rebind exposed an earlier result");
        target_graph.evaluate(target_positions);
        target_graph.downloadResults(stale_phase, stale_jets, stale_ratios);
      }
      if (model.ne > 1 && smallest_target_bytes >= full_target_reference.deviceWorkspaceBytes())
        throw std::runtime_error("Target VGL K1 did not reduce non-retained graph workspace");
      std::cout << "PASS explicit target VGL: K1/2/3/Ne, B3/1/2/3, complete CPU VGL parity, "
                   "budgets before allocation, one upload/generation, fixed storage, rebind/reverse/mixed guards; "
                << "full_device_bytes=" << full_target_reference.deviceWorkspaceBytes()
                << "; K1_device_bytes=" << smallest_target_bytes << '\n';
    }

    {
      // Reuse this compact architecture with a periodic embedding. Rebuild each
      // canonical tensor interval independently: resizing embedding shifts every
      // following parameter offset, so a raw whole-vector copy would be invalid.
      ExecutionEnvironment environment = plan.environment();
      environment.boundary = BoundaryCondition::PERIODIC;
      environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
      environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
      environment.periodic_axes = {true, true, true};
      const auto original_parameters = model.p.flat_values();
      std::vector<double> periodic_parameters;
      std::vector<pf::Layout> periodic_layouts;
      for (const auto& tensor : plan.parameterTensors())
      {
        auto tensor_shape = tensor.shape;
        if (tensor.role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT)
          tensor_shape[0] = 7 * shape.nuclei + 1;
        std::size_t count = 1;
        for (const auto extent : tensor_shape)
          count = spatial_detail::checkedProduct(count, extent, "Periodic target fixture extent overflow");
        const auto begin = periodic_parameters.size();
        for (std::size_t index = 0; index < count; ++index)
          periodic_parameters.push_back(original_parameters[tensor.begin + index % (tensor.end - tensor.begin)]);
        periodic_layouts.push_back({tensor.module, tensor.name, tensor_shape, begin, periodic_parameters.size()});
      }
      pf::PsiFormer periodic_model(pf::Parameters(std::move(periodic_parameters), std::move(periodic_layouts)),
                                  model.cfg, model.ndet, model.dim, model.heads, model.blocks);
      const auto periodic_plan = PsiFormerExecutionPlan::fromParameters(periodic_model.p, shape, environment);
      pf::DirectValueExecutor periodic_value(periodic_model, periodic_plan);
      pf::DirectSpatialExecutor periodic_spatial(periodic_model, periodic_value, periodic_plan);
      auto cpu_workspace = periodic_spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
      const auto periodic_image = std::make_shared<PreparedDeviceModel>(periodic_plan,
          periodic_model.p.flat_values(), model.cfg.nuclei.x, 0);
      std::vector<double> positions;
      for (std::size_t sample = 0; sample < 3; ++sample)
        positions.insert(positions.end(), model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      positions[3 * model.ne] += 0.031;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        positions[6 * model.ne + dimension] += environment.lattice_vectors[0][dimension] -
            2 * environment.lattice_vectors[2][dimension];
      for (const auto target_count : {std::size_t{1}, std::min<std::size_t>(3, model.ne)})
      {
        const auto cost = estimateTargetVglCost(periodic_plan, 3, target_count);
        PreparedDeviceInference target(periodic_image, 3, queue, handle, SpatialJetMode::FULL_VGL, false, false,
            InferenceStatusPolicy::DEFERRED_NEURAL,
            {target_count, cost.device_workspace_bytes, cost.charged_scalar_work_units});
        const auto bytes = target.deviceWorkspaceBytes();
        for (const std::size_t count : {std::size_t{3}, std::size_t{1}, std::size_t{2}, std::size_t{3}})
        {
          const std::vector<double> prefix(positions.begin(), positions.begin() + count * 3 * model.ne);
          target.evaluate(prefix);
          const auto& target_layout = target.resultLayout();
          std::vector<double> target_phase(count), target_jets(spatialJetSpanElements(target_layout)),
              target_ratios(count * model.ne);
          target.downloadResults(target_phase, target_jets, target_ratios);
          for (std::size_t sample = 0; sample < count; ++sample)
          {
            cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(prefix.data() + sample * 3 * model.ne,
                                                                              model.ne));
            const auto reference = periodic_spatial.evaluateFull(*cpu_workspace);
            near(target_phase[sample], reference.sign, "periodic target phase");
            near(target_jets[target_layout.uncheckedValueOffset(sample, 0)], reference.logabs,
                 "periodic target log amplitude");
            for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
              near(target_jets[target_layout.uncheckedGradientOffset(sample, lane, 0)], reference.gradient[lane],
                   "periodic target gradient");
            for (std::size_t electron = 0; electron < model.ne; ++electron)
            {
              near(target_jets[target_layout.uncheckedLaplacianOffset(sample, electron, 0)], reference.lap_log[electron],
                   "periodic target log Laplacian");
              near(target_ratios[sample * model.ne + electron], reference.lap_ratio[electron],
                   "periodic target Laplacian ratio");
            }
          }
          if (count == 3)
            for (std::size_t plane = 0; plane < target_layout.plane_count; ++plane)
              near(target_jets[target_layout.uncheckedPlaneOffset(2, plane, 0)],
                   target_jets[target_layout.uncheckedPlaneOffset(0, plane, 0)], "periodic target image invariance");
          if (target.deviceWorkspaceBytes() != bytes || target.admittedTargetTiles() != cost.tile_count)
            throw std::runtime_error("Periodic target VGL reuse changed prepared storage or target coverage");
        }
      }
      validateCoordinateGradient(periodic_image, periodic_model, positions, queue, handle);
      std::cout << "PASS complete skew-periodic coordinate gradient: CPU full VGL, FD, lifecycle, exact storage\n";
      std::cout << "PASS skew-periodic target VGL: K1/3, compact tails, B3/1/2/3 reuse, "
                   "complete independent CPU parity and integer-image invariance\n";
    }

    if (model.ne > 1)
    {
      // Make only the final physical electron's radial Laplacian overflow in
      // embedding. Its ordinary feature remains finite and is masked from the
      // orbital head; earlier targets therefore complete real scatter work.
      auto late_parameters = model.p.flat_values();
      const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
      for (std::size_t input = 0; input < embedding.shape[0]; ++input)
        late_parameters[embedding.begin + input * model.dim] = input == 0 ? 1e160 : 0;
      for (std::size_t layer = 0; layer < model.blocks; ++layer)
        for (const auto role : {ParameterRole::ATTENTION_QUERY_WEIGHT, ParameterRole::ATTENTION_KEY_WEIGHT,
            ParameterRole::ATTENTION_VALUE_WEIGHT, ParameterRole::ATTENTION_OUTPUT_WEIGHT,
            ParameterRole::UPDATE_HIDDEN_WEIGHT, ParameterRole::UPDATE_HIDDEN_BIAS,
            ParameterRole::UPDATE_OUTPUT_WEIGHT, ParameterRole::UPDATE_OUTPUT_BIAS})
        {
          const auto& tensor = plan.parameter(role, layer);
          std::fill(late_parameters.begin() + tensor.begin, late_parameters.begin() + tensor.end, 0);
        }
      for (const auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
      {
        const auto& tensor = plan.parameter(role);
        std::fill(late_parameters.begin() + tensor.begin,
                  late_parameters.begin() + tensor.begin + tensor.shape[1], 0);
      }
      auto nuclei = model.cfg.nuclei.x;
      std::fill(nuclei.begin(), nuclei.begin() + 3, 0);
      const auto late_image = std::make_shared<PreparedDeviceModel>(plan, late_parameters, nuclei, 20);
      const auto cost = estimateTargetVglCost(plan, 1, 1);
      const TargetVglPreparation preparation{1, cost.device_workspace_bytes, cost.charged_scalar_work_units};
      std::vector<double> late_positions(3 * model.ne);
      for (std::size_t electron = 0; electron < model.ne; ++electron)
      {
        late_positions[3 * electron] = 0.4 + 0.3 * electron;
        late_positions[3 * electron + 1] = -0.25 + 0.13 * electron;
        late_positions[3 * electron + 2] = 0.2 + 0.07 * electron;
      }
      late_positions[3 * (model.ne - 1)] = 1e-150;
      late_positions[3 * (model.ne - 1) + 1] = late_positions[3 * (model.ne - 1) + 2] = 0;
      for (const auto policy : {InferenceStatusPolicy::SYNCHRONOUS_DEBUG, InferenceStatusPolicy::DEFERRED_NEURAL})
      {
        PreparedDeviceInference late_graph(late_image, 1, queue, handle, SpatialJetMode::FULL_VGL,
                                           false, false, policy, preparation);
        const auto generation = late_graph.evaluationGeneration() + 1;
        bool late_rejected = false;
        try { late_graph.evaluate(late_positions); }
        catch (const InferenceStatusError& error)
        {
          late_rejected = error.generation() == generation;
        }
        if (!late_rejected || late_graph.admittedTargetTiles() != model.ne - 1 ||
            late_graph.evaluationGeneration() != generation)
          throw std::runtime_error("Target VGL did not reject the last target after admitted earlier tiles");
        std::vector<double> late_phase(1), late_jets(spatialJetSpanElements(late_graph.resultLayout())),
            late_ratios(model.ne);
        bool partial_rejected = false;
        try { late_graph.downloadResults(late_phase, late_jets, late_ratios); }
        catch (const std::logic_error&) { partial_rejected = true; }
        if (!partial_rejected)
          throw std::runtime_error("Target VGL exposed partial output after a late tile failure");
        auto recovered_positions = late_positions;
        recovered_positions[3 * (model.ne - 1)] = 0.2;
        const auto bytes = late_graph.deviceWorkspaceBytes();
        late_graph.evaluate(recovered_positions);
        late_graph.downloadResults(late_phase, late_jets, late_ratios);
        if (late_graph.admittedTargetTiles() != model.ne || late_graph.deviceWorkspaceBytes() != bytes)
          throw std::runtime_error("Target VGL late failure recovery changed prepared storage");
      }
      std::cout << "PASS target VGL late-tile failure: earlier real scatters, no partial publication, "
                   "one generation, both status policies, recovery\n";
    }

    // Qualify both diagnostic policies independently of the environment default.
    // Deferred neural failures may finish queued arithmetic, but can never expose
    // a partial result/tape or erase the first failing stage's typed identity.
    {
      const auto original_policy = inference.statusPolicy();
      const auto prepared_bytes = inference.deviceWorkspaceBytes();
      auto huge_parameters = model.p.flat_values();
      for (const auto role : {ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
                              ParameterRole::ATTENTION_QUERY_WEIGHT, ParameterRole::ATTENTION_KEY_WEIGHT})
      {
        const auto& parameter = plan.parameter(role, role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT
            ? NO_ATTENTION_BLOCK : 0);
        std::fill(huge_parameters.begin() + parameter.begin,
                  huge_parameters.begin() + parameter.end, std::numeric_limits<double>::max());
      }
      const auto huge_image = image->makeUpdatedImage(huge_parameters, 18);
      const auto recovered_image = huge_image->makeUpdatedImage(model.p.flat_values(), 19);
      for (const auto policy : {InferenceStatusPolicy::SYNCHRONOUS_DEBUG, InferenceStatusPolicy::DEFERRED_NEURAL})
      {
        inference.setStatusPolicy(policy);
        inference.evaluate(model.cfg.electrons.x);
        inference.downloadResults(phases, jets, ratios);
        verify();
        if (inference.deviceWorkspaceBytes() != prepared_bytes)
          throw std::runtime_error("Status policy selection changed prepared VGL storage");

        PreparedDeviceInference retained(image, model.cfg.nconfig, queue, handle,
            SpatialJetMode::VALUE_ONLY, true, false, policy);
        retained.evaluate(model.cfg.electrons.x);
        const auto retained_bytes = retained.deviceWorkspaceBytes();
        std::vector<double> policy_phases(model.cfg.nconfig), policy_values(model.cfg.nconfig), no_spatial;
        retained.downloadResults(policy_phases, policy_values, no_spatial);
        for (std::size_t sample = 0; sample < model.cfg.nconfig; ++sample)
          near(policy_values[sample], references[sample].logabs, "status-policy value parity");

        auto invalid_geometry = model.cfg.electrons.x;
        invalid_geometry.front() = std::numeric_limits<double>::quiet_NaN();
        const auto invalid_generation = retained.evaluationGeneration() + 1;
        bool geometry_rejected = false, stale_tape_rejected = false, stale_result_rejected = false;
        try { retained.evaluate(invalid_geometry); }
        catch (const InferenceStatusError& error)
        {
          geometry_rejected = error.domain() == InferenceStatusDomain::SPATIAL && error.stage() == 0 &&
              error.spatialStatus() != OpenSpatialStatus::REGULAR && error.generation() == invalid_generation;
        }
        try { (void)retained.valueHeadTape(); }
        catch (const std::logic_error&) { stale_tape_rejected = true; }
        try { retained.downloadResults(policy_phases, policy_values, no_spatial); }
        catch (const std::logic_error&) { stale_result_rejected = true; }
        if (!geometry_rejected || !stale_tape_rejected || !stale_result_rejected)
          throw std::runtime_error("Status policy lost immediate geometry admission or exposed a failed snapshot");
        retained.evaluate(model.cfg.electrons.x);
        retained.downloadResults(policy_phases, policy_values, no_spatial);
        for (std::size_t sample = 0; sample < model.cfg.nconfig; ++sample)
          near(policy_values[sample], references[sample].logabs, "status-policy geometry recovery");

        // Finite parameters can still overflow the neural GEMMs. Both policies
        // must report the first softmax failure, not a later head/node error.
        retained.rebindModel(huge_image);
        const auto neural_generation = retained.evaluationGeneration() + 1;
        bool neural_rejected = false;
        try { retained.evaluate(model.cfg.electrons.x); }
        catch (const InferenceStatusError& error)
        {
          neural_rejected = error.domain() == InferenceStatusDomain::SOFTMAX && error.stage() == 1 &&
              error.jetStatus() == device_math::JetMathStatus::NONFINITE_INPUT &&
              error.generation() == neural_generation;
        }
        if (!neural_rejected)
          throw std::runtime_error("Status policy did not preserve the first typed neural failure");
        stale_tape_rejected = false;
        stale_result_rejected = false;
        try { (void)retained.valueHeadTape(); }
        catch (const std::logic_error&) { stale_tape_rejected = true; }
        try { retained.downloadResults(policy_phases, policy_values, no_spatial); }
        catch (const std::logic_error&) { stale_result_rejected = true; }
        if (!stale_tape_rejected || !stale_result_rejected)
          throw std::runtime_error("Neural status failure exposed an earlier result or reverse tape");
        retained.rebindModel(recovered_image);
        retained.evaluate(model.cfg.electrons.x);
        retained.downloadResults(policy_phases, policy_values, no_spatial);
        for (std::size_t sample = 0; sample < model.cfg.nconfig; ++sample)
          near(policy_values[sample], references[sample].logabs, "status-policy neural recovery");
        if (retained.deviceWorkspaceBytes() != retained_bytes)
          throw std::runtime_error("Status failure recovery changed prepared storage");
      }
      inference.setStatusPolicy(original_policy);
      bool invalid_policy_rejected = false;
      try { inference.setStatusPolicy(static_cast<InferenceStatusPolicy>(999)); }
      catch (const std::invalid_argument&) { invalid_policy_rejected = true; }
      if (!invalid_policy_rejected || inference.statusPolicy() != original_policy)
        throw std::runtime_error("Invalid status policy changed graph configuration");
      std::cout << "PASS status policies: value/VGL parity, typed first failure, stale snapshots, recovery, stable storage\n";
    }

    // Stage-6 preparation retains full spatial planes without exposing them as
    // ordinary score values. Check B*E value/gradient/Laplacian product identities.
    PreparedDeviceInference spatial_taped(image, model.cfg.nconfig, queue, handle,
                                          SpatialJetMode::FULL_VGL, false, true);
    const auto spatial_tape_bytes = spatial_taped.deviceWorkspaceBytes();
    spatial_taped.evaluate(model.cfg.electrons.x);
    spatial_taped.downloadResults(phases, jets, ratios);
    verify();
    const auto spatial_head = spatial_taped.spatialHeadTape();
    // Kinetic root seeds use signed sample contractions, optional physical masses
    // and complete product-wavefunction drift, not an implicit PsiFormer-only drift.
    std::vector<double> kinetic_weights(model.cfg.nconfig), inverse_masses(model.ne);
    for (std::size_t configuration = 0; configuration < kinetic_weights.size(); ++configuration)
      kinetic_weights[configuration] = configuration % 2 == 0 ? 0.7 : -0.25;
    for (std::size_t electron = 0; electron < model.ne; ++electron)
      inverse_masses[electron] = 1.0 + 0.1 * electron;
    DeviceBuffer<double> kinetic_weight_buffer(kinetic_weights.size()), mass_buffer(inverse_masses.size());
    DeviceBuffer<double> root_adjoint(spatialJetSpanElements(spatial_head.root_layout));
    DeviceBuffer<OpenSpatialStatus> root_status(model.cfg.nconfig);
    kinetic_weight_buffer.upload(kinetic_weights.data(), kinetic_weights.size());
    mass_buffer.upload(inverse_masses.data(), inverse_masses.size());
    std::vector<double> root_seeds(root_adjoint.capacity());
    std::vector<OpenSpatialStatus> root_flags(model.cfg.nconfig);
    checkRuntime(launchKineticRootAdjoints(queue.getNative(), spatial_head.root_layout, spatial_head.root,
        kinetic_weight_buffer.data(), nullptr, nullptr, 0, root_adjoint.data(), root_status.data()), "unit-mass kinetic root seed");
    queue.sync();
    root_adjoint.download(root_seeds.data(), root_seeds.size());
    root_status.download(root_flags.data(), root_flags.size());
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      if (root_flags[configuration] != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Regular kinetic root seed failed");
      near(root_seeds[spatial_head.root_layout.uncheckedValueOffset(configuration, 0)], 0, "kinetic value seed");
      for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
      {
        const auto offset = spatial_head.root_layout.uncheckedGradientOffset(configuration, lane, 0);
        near(root_seeds[offset], -kinetic_weights[configuration] * jets[offset], "unit-mass total drift seed");
      }
      for (std::size_t electron = 0; electron < model.ne; ++electron)
        near(root_seeds[spatial_head.root_layout.uncheckedLaplacianOffset(configuration, electron, 0)],
             -0.5 * kinetic_weights[configuration], "unit-mass Laplacian seed");
    }
    const auto external_stride = 3 * model.ne + 2;
    std::vector<double> external_drift(model.cfg.nconfig * external_stride, 999.0);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
      for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
        external_drift[configuration * external_stride + lane] =
            jets[spatial_head.root_layout.uncheckedGradientOffset(configuration, lane, 0)] + 0.02 * (lane + 1);
    DeviceBuffer<double> external_drift_buffer(external_drift.size());
    external_drift_buffer.upload(external_drift.data(), external_drift.size());
    checkRuntime(launchKineticRootAdjoints(queue.getNative(), spatial_head.root_layout, spatial_head.root,
        kinetic_weight_buffer.data(), mass_buffer.data(), external_drift_buffer.data(), external_stride,
        root_adjoint.data(), root_status.data()), "external-drift mass-weighted kinetic root seed");
    queue.sync();
    root_adjoint.download(root_seeds.data(), root_seeds.size());
    root_status.download(root_flags.data(), root_flags.size());
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      if (root_flags[configuration] != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Mass-weighted external kinetic drift seed failed");
      for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
        near(root_seeds[spatial_head.root_layout.uncheckedGradientOffset(configuration, lane, 0)],
             -kinetic_weights[configuration] * inverse_masses[lane / 3] *
              external_drift[configuration * external_stride + lane], "external total drift seed");
      for (std::size_t electron = 0; electron < model.ne; ++electron)
        near(root_seeds[spatial_head.root_layout.uncheckedLaplacianOffset(configuration, electron, 0)],
             -0.5 * kinetic_weights[configuration] * inverse_masses[electron], "inverse-mass Laplacian seed");
    }
    // Compact roots retain physical mass/drift indexing, including validation
    // outside this tile. The unused full-capacity tail must remain untouched.
    if (model.ne > 1)
    {
      const auto local_root = withSpatialTargetRange(spatial_head.root_layout, 1, 1);
      const auto live_span = spatialJetSpanElements(local_root);
      std::vector<double> compact_root(root_seeds.size(), std::numeric_limits<double>::quiet_NaN());
      std::vector<double> compact_seeds(root_seeds.size(), -817.25);
      for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
      {
        compact_root[local_root.uncheckedValueOffset(configuration, 0)] = 0.0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          compact_root[local_root.uncheckedGradientOffset(configuration, dimension, 0)] =
              jets[spatial_head.root_layout.uncheckedGradientOffset(configuration, 3 + dimension, 0)];
        compact_root[local_root.uncheckedLaplacianOffset(configuration, 0, 0)] =
            jets[spatial_head.root_layout.uncheckedLaplacianOffset(configuration, 1, 0)];
      }
      DeviceBuffer<double> compact_input(compact_root.size());
      compact_input.upload(compact_root.data(), compact_root.size());
      for (bool gradient_only : {false, true})
      {
        root_adjoint.upload(compact_seeds.data(), compact_seeds.size());
        checkRuntime(launchKineticRootAdjoints(queue.getNative(), local_root, compact_input.data(),
            kinetic_weight_buffer.data(), mass_buffer.data(), external_drift_buffer.data(), external_stride,
            root_adjoint.data(), root_status.data(), gradient_only), "mapped compact kinetic root seed");
        queue.sync();
        root_adjoint.download(root_seeds.data(), root_seeds.size());
        root_status.download(root_flags.data(), root_flags.size());
        for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
        {
          if (root_flags[configuration] != OpenSpatialStatus::REGULAR)
            throw std::runtime_error("Compact kinetic root seed rejected regular physical inputs");
          near(root_seeds[local_root.uncheckedValueOffset(configuration, 0)], 0, "compact value seed");
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            near(root_seeds[local_root.uncheckedGradientOffset(configuration, dimension, 0)],
                -kinetic_weights[configuration] * inverse_masses[1] *
                    external_drift[configuration * external_stride + 3 + dimension], "mapped compact drift seed");
          near(root_seeds[local_root.uncheckedLaplacianOffset(configuration, 0, 0)],
              gradient_only ? 0.0 : -0.5 * kinetic_weights[configuration] * inverse_masses[1],
              "mapped compact Laplacian seed");
        }
        for (std::size_t index = live_span; index < root_seeds.size(); ++index)
          near(root_seeds[index], -817.25, "compact root unused tail");
      }
      // Electron zero lies outside [1,2); either invalid physical coefficient
      // must invalidate the tile rather than escape validation until a later tile.
      for (bool bad_mass : {true, false})
      {
        inverse_masses[0] = bad_mass ? 0.0 : 1.0;
        mass_buffer.upload(inverse_masses.data(), inverse_masses.size());
        if (!bad_mass)
        {
          auto invalid_drift = external_drift;
          for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
            invalid_drift[configuration * external_stride] = std::numeric_limits<double>::quiet_NaN();
          external_drift_buffer.upload(invalid_drift.data(), invalid_drift.size());
        }
        checkRuntime(launchKineticRootAdjoints(queue.getNative(), local_root, compact_input.data(),
            kinetic_weight_buffer.data(), mass_buffer.data(), external_drift_buffer.data(), external_stride,
            root_adjoint.data(), root_status.data()), "compact root outside-tile coefficient guard");
        queue.sync();
        root_status.download(root_flags.data(), root_flags.size());
        for (auto flag : root_flags)
          if (flag == OpenSpatialStatus::REGULAR)
            throw std::runtime_error("Compact kinetic root accepted invalid coefficients outside its tile");
      }
      external_drift_buffer.upload(external_drift.data(), external_drift.size());
      std::cout << "PASS compact kinetic root: physical mass/drift, gradient-only, poisoned tail, outside-tile guards\n";
    }
    inverse_masses[0] = 0;
    mass_buffer.upload(inverse_masses.data(), inverse_masses.size());
    checkRuntime(launchKineticRootAdjoints(queue.getNative(), spatial_head.root_layout, spatial_head.root,
        kinetic_weight_buffer.data(), mass_buffer.data(), nullptr, 0, root_adjoint.data(), root_status.data()), "invalid kinetic mass guard");
    queue.sync();
    root_status.download(root_flags.data(), root_flags.size());
    for (auto flag : root_flags)
      if (flag == OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Kinetic seed accepted a nonpositive inverse mass");
    std::cout << "PASS kinetic root seeds: weights, masses, external total drift, padded stride, invalid-mass guard\n";
    PreparedDeviceKineticReverse kinetic_reverse(image, model.cfg.nconfig, queue, handle);
    kinetic_reverse.evaluate(spatial_taped, kinetic_weights);
    std::vector<double> gpu_kinetic(plan.parameterCount()), cpu_kinetic(plan.parameterCount(), 0.0);
    kinetic_reverse.downloadResponse(gpu_kinetic);
    const auto kinetic_bytes = kinetic_reverse.deviceWorkspaceBytes();
    bool invalid_kinetic_weights = false, stale_kinetic_rejected = false;
    try { (void)kinetic_reverse.evaluate(spatial_taped, {}); }
    catch (const std::invalid_argument&) { invalid_kinetic_weights = true; }
    try { kinetic_reverse.downloadResponse(gpu_kinetic); }
    catch (const std::logic_error&) { stale_kinetic_rejected = true; }
    if (!invalid_kinetic_weights || !stale_kinetic_rejected)
      throw std::runtime_error("Failed kinetic request exposed a stale response");
    kinetic_reverse.evaluate(spatial_taped, kinetic_weights);
    kinetic_reverse.downloadResponse(gpu_kinetic);
    if (kinetic_reverse.deviceWorkspaceBytes() != kinetic_bytes)
      throw std::runtime_error("Kinetic reverse allocated during reuse");
    pf::DirectKineticExecutor kinetic_cpu(model, plan);
    auto kinetic_workspace = kinetic_cpu.makeWorkspace();
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * model.ne * 3, model.ne));
      const auto response = kinetic_cpu.evaluate(*kinetic_workspace);
      for (std::size_t parameter = 0; parameter < cpu_kinetic.size(); ++parameter)
        cpu_kinetic[parameter] += kinetic_weights[configuration] * response.kinetic_parameter_response[parameter];
    }
    double kinetic_max_difference = 0;
    for (std::size_t parameter = 0; parameter < cpu_kinetic.size(); ++parameter)
    {
      near(gpu_kinetic[parameter], cpu_kinetic[parameter], "full weighted kinetic parameter response");
      kinetic_max_difference = std::max(kinetic_max_difference, std::abs(gpu_kinetic[parameter] - cpu_kinetic[parameter]));
    }
    std::cout << "PASS complete regular-channel kinetic reverse: CPU parity; max_abs_difference="
              << kinetic_max_difference << '\n';
    if (!kinetic_output.empty())
    {
      std::ofstream output(kinetic_output, std::ios::binary);
      output.write(reinterpret_cast<const char*>(gpu_kinetic.data()), gpu_kinetic.size() * sizeof(double));
      if (!output)
        throw std::runtime_error("Kinetic canonical response export failed");
    }
    std::vector<double> external_contiguous(model.cfg.nconfig * 3 * model.ne);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
      std::copy_n(external_drift.data() + configuration * external_stride, 3 * model.ne,
                   external_contiguous.data() + configuration * 3 * model.ne);
    inverse_masses[0] = 1.0;
    kinetic_reverse.evaluate(spatial_taped, kinetic_weights, inverse_masses, external_contiguous);
    kinetic_reverse.downloadResponse(gpu_kinetic);
    std::fill(cpu_kinetic.begin(), cpu_kinetic.end(), 0.0);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * model.ne * 3, model.ne));
      const auto response = kinetic_cpu.evaluate(*kinetic_workspace,
          external_contiguous.data() + configuration * 3 * model.ne, 3 * model.ne,
          inverse_masses.data(), inverse_masses.size());
      for (std::size_t parameter = 0; parameter < cpu_kinetic.size(); ++parameter)
        cpu_kinetic[parameter] += kinetic_weights[configuration] * response.kinetic_parameter_response[parameter];
    }
    for (std::size_t parameter = 0; parameter < cpu_kinetic.size(); ++parameter)
      near(gpu_kinetic[parameter], cpu_kinetic[parameter], "mass/external-drift kinetic response CPU parity");
    std::cout << "PASS full kinetic reverse: nonunit masses and external total-wavefunction drift\n";
    validateTargetKinetic(model, plan, image, queue, handle);
    validateKineticContractions(model, plan, image);
    validateDeviceCoordinates(model, image, queue, handle);
    validateRetainedCoordinateReplays(model, image);
    {
      const auto targets = std::min(std::size_t(3), model.ne);
      const auto vgl_cost = estimateTargetVglCost(plan, 2, targets);
      const auto kinetic_cost = estimateTargetKineticCost(plan, targets);
      const DeviceExecutionPreparation preparation{
          {targets, vgl_cost.device_workspace_bytes, vgl_cost.charged_scalar_work_units},
          {targets, kinetic_cost.device_workspace_bytes, kinetic_cost.charged_scalar_work_units}};
      PreparedDeviceBatchBackend targeted(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
      auto target_session = targeted.prepareProductSession();
      auto score_session = targeted.prepareProductSession(ProductSessionProfile::SCORE_ONLY);
      const auto bytes = target_session->deviceWorkspaceBytes();
      if (target_session->kineticTargetCapacity() != targets || score_session->kineticTargetCapacity() != 0 ||
          bytes != score_session->deviceWorkspaceBytes() + kinetic_cost.device_workspace_bytes)
        throw std::runtime_error("Target product session retained an extra full kinetic graph");
      kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), model.ne));
      const auto expected = kinetic_cpu.evaluate(*kinetic_workspace, external_contiguous.data(), 3 * model.ne,
          inverse_masses.data(), inverse_masses.size());
      const auto& actual = target_session->kinetic(model.cfg.electrons.x.data(), external_contiguous.data(),
          inverse_masses.data(), image->version());
      for (std::size_t parameter = 0; parameter < actual.size(); ++parameter)
        near(actual[parameter], expected.kinetic_parameter_response[parameter], "target product real kinetic CPU parity");
      (void)target_session->score(model.cfg.electrons.x.data(), image->version());
      if (target_session->numericalProducts() != 2)
        throw std::runtime_error("Target kinetic repeated the separate ordinary score operation");
      const auto expected_imaginary = kinetic_cpu.evaluateGradientParameterResponse(*kinetic_workspace,
          external_contiguous.data(), 3 * model.ne, inverse_masses.data(), inverse_masses.size());
      const auto& actual_imaginary = target_session->gradientResponse(model.cfg.electrons.x.data(),
          external_contiguous.data(), inverse_masses.data(), image->version());
      for (std::size_t parameter = 0; parameter < actual_imaginary.size(); ++parameter)
        near(actual_imaginary[parameter], expected_imaginary[parameter], "target product imaginary kinetic CPU parity");
      auto crowd = targeted.prepare(2);
      const auto result = targeted.evaluate(*crowd, pf::DirectBatchMode::FULL_VGL,
          model.cfg.electrons.x.data(), 2, image->version());
      for (std::size_t sample = 0; sample < 2; ++sample)
      {
        near(result.logabs[sample * result.value_stride], references[sample].logabs, "target production crowd VGL value");
        for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
          near(result.gradient[sample * result.gradient_configuration_stride + lane * result.gradient_lane_stride],
              references[sample].gradient[lane], "target production crowd VGL gradient");
      }
      targeted.publishParameters(model.p.flat_values(), image->version() + 1, []() {});
      (void)target_session->kinetic(model.cfg.electrons.x.data(), external_contiguous.data(),
          inverse_masses.data(), image->version() + 1);
      (void)targeted.evaluate(*crowd, pf::DirectBatchMode::FULL_VGL,
          model.cfg.electrons.x.data(), 1, image->version() + 1);
      if (target_session->deviceWorkspaceBytes() != bytes)
        throw std::runtime_error("Target product publication changed prepared capacity");
      bool mixed_rejected = false;
      try { PreparedDeviceBatchBackend mixed_target(image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS, preparation); }
      catch (const std::invalid_argument&) { mixed_rejected = true; }
      if (!mixed_rejected)
        throw std::runtime_error("Mixed production backend admitted explicit targets");
      std::cout << "PASS target production owners: retained kinetic exclusively replaces full graph, "
                   "score once separately, real/imaginary CPU parity, non-retained crowd VGL, "
                   "publication rebind and mixed-target rejection; K=" << targets << '\n';
    }
    const auto& orbital_layout = spatial_head.orbital_layout.orbitals;
    const auto orbital_span = spatialJetSpanElements(orbital_layout);
    std::vector<double> spatial_backflow(orbital_span), spatial_envelope(orbital_span), spatial_orbitals(orbital_span);
    spatial_taped.downloadSpatialOrbitalTape(spatial_backflow, spatial_envelope, spatial_orbitals);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
      for (std::size_t element = 0; element < orbital_layout.element_count; ++element)
      {
        const auto value = orbital_layout.uncheckedValueOffset(configuration, element);
        near(spatial_orbitals[value], spatial_backflow[value] * spatial_envelope[value], "retained spatial orbital product value");
        for (std::size_t lane = 0; lane < orbital_layout.gradient_lanes; ++lane)
        {
          const auto gradient = orbital_layout.uncheckedGradientOffset(configuration, lane, element);
          near(spatial_orbitals[gradient], spatial_backflow[gradient] * spatial_envelope[value] +
               spatial_backflow[value] * spatial_envelope[gradient], "retained spatial orbital product gradient");
        }
        for (std::size_t electron = 0; electron < orbital_layout.laplacian_lanes; ++electron)
        {
          const auto laplacian = orbital_layout.uncheckedLaplacianOffset(configuration, electron, element);
          double product = spatial_backflow[laplacian] * spatial_envelope[value] +
              spatial_backflow[value] * spatial_envelope[laplacian];
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const auto gradient = orbital_layout.uncheckedGradientOffset(configuration, 3 * electron + dimension, element);
            product += 2 * spatial_backflow[gradient] * spatial_envelope[gradient];
          }
          near(spatial_orbitals[laplacian], product, "retained spatial orbital product Laplacian");
        }
      }
    bool spatial_as_value_rejected = false;
    const auto feature_layout = makeSpatialDenseJetLayout(model.cfg.nconfig, model.ne, model.dim,
        model.dim, model.ne, SpatialJetMode::FULL_VGL).source;
    for (std::size_t layer = 0; layer < model.blocks; ++layer)
    {
      const auto& tape = spatial_taped.spatialBlockTape(layer);
      for (bool output_layer : {false, true})
      {
        const auto& pre_buffer = output_layer ? tape.projected_pre : tape.hidden_pre;
        const auto& post_buffer = output_layer ? tape.projected : tape.hidden;
        std::vector<double> pre(pre_buffer.capacity()), post(post_buffer.capacity());
        pre_buffer.download(pre.data(), pre.size());
        post_buffer.download(post.data(), post.size());
        // Independent scalar tanh chain rules check biased preactivations and
        // every retained first/contracted-second spatial derivative plane.
        for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
          for (std::size_t element = 0; element < feature_layout.element_count; ++element)
          {
            const auto value = feature_layout.uncheckedValueOffset(configuration, element);
            const double y = std::tanh(pre[value]), first = 1 - y * y, second = -2 * y * first;
            near(post[value], y, "retained biased preactivation value");
            for (std::size_t lane = 0; lane < feature_layout.gradient_lanes; ++lane)
            {
              const auto gradient = feature_layout.uncheckedGradientOffset(configuration, lane, element);
              if (!device_math::isFiniteBinary64(post[gradient]) || !device_math::isFiniteBinary64(pre[gradient]) ||
                  std::abs(post[gradient] - first * pre[gradient]) > 2e-7 * (1 + std::abs(first * pre[gradient])))
              {
                std::cerr.precision(17);
                std::cerr << "Preactivation diagnostic layer=" << layer << " output=" << output_layer
                          << " configuration=" << configuration << " element=" << element << " lane=" << lane
                          << " pre_value=" << pre[value] << " post_value=" << post[value]
                          << " pre_gradient=" << pre[gradient] << " post_gradient=" << post[gradient]
                          << " expected=" << first * pre[gradient] << '\n';
              }
              near(post[gradient], first * pre[gradient], "retained preactivation gradient");
            }
            for (std::size_t electron = 0; electron < feature_layout.laplacian_lanes; ++electron)
            {
              const auto laplacian = feature_layout.uncheckedLaplacianOffset(configuration, electron, element);
              double squared = 0;
              for (std::size_t dimension = 0; dimension < 3; ++dimension)
              {
                const auto gradient = feature_layout.uncheckedGradientOffset(configuration, 3 * electron + dimension, element);
                squared += pre[gradient] * pre[gradient];
              }
              near(post[laplacian], first * pre[laplacian] + second * squared, "retained preactivation Laplacian");
            }
          }
      }
      const auto attention_layout = makeSpatialAttentionJetLayout(model.cfg.nconfig, model.ne,
          model.heads, model.dim / model.heads, model.ne, SpatialJetMode::FULL_VGL);
      std::vector<double> logits(tape.logits.capacity()), probabilities(tape.probabilities.capacity());
      tape.logits.download(logits.data(), logits.size());
      tape.probabilities.download(probabilities.data(), probabilities.size());
      for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
        for (std::size_t head = 0; head < model.heads; ++head)
          for (std::size_t row = 0; row < model.ne; ++row)
          {
            const auto origin = attention_layout.attention.uncheckedValueOffset(configuration,
                head * model.ne * model.ne + row * model.ne);
            double maximum = logits[origin], sum = 0;
            for (std::size_t column = 0; column < model.ne; ++column)
              maximum = std::max(maximum, logits[origin + column]);
            for (std::size_t column = 0; column < model.ne; ++column)
              sum += std::exp(logits[origin + column] - maximum);
            for (std::size_t column = 0; column < model.ne; ++column)
              near(probabilities[origin + column], std::exp(logits[origin + column] - maximum) / sum,
                   "retained raw attention logits");
          }
    }
    try { (void)spatial_taped.valueBlockTape(0); }
    catch (const std::logic_error&) { spatial_as_value_rejected = true; }
    if (!spatial_as_value_rejected)
      throw std::runtime_error("Full-spatial tape was accepted as ordinary values");
    spatial_taped.evaluate(model.cfg.electrons.x);
    if (spatial_taped.deviceWorkspaceBytes() != spatial_tape_bytes)
      throw std::runtime_error("Spatial tape allocated storage during reuse");
    std::cout << "PASS retained full-VGL tape: CPU parity, orbital product rules, mode isolation, reuse\n";

    // Values must take the one-plane route, not secretly compute and discard VGL.
    PreparedDeviceInference values(image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY);
    if (values.modelImage().get() != inference.modelImage().get() || values.modelImage()->version() != 17 ||
        values.resultLayout().plane_count != 1 || values.resultLayout().gradient_lanes != 0 ||
        values.resultLayout().laplacian_lanes != 0)
      throw std::runtime_error("Value graph lost shared model identity or one-plane layout");
    std::vector<double> value_phases(model.cfg.nconfig), value_logs(model.cfg.nconfig), no_ratios;
    const auto value_bytes = values.deviceWorkspaceBytes();
    if (value_bytes >= inference.deviceWorkspaceBytes())
      throw std::runtime_error("Value-only workspace did not eliminate derivative storage");
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      values.evaluate(model.cfg.electrons.x);
      values.downloadResults(value_phases, value_logs, no_ratios);
      for (std::size_t index = 0; index < references.size(); ++index)
      {
        near(value_phases[index], references[index].sign, "value-only sign");
        near(value_logs[index], references[index].logabs, "value-only logabs");
      }
    }

    // This opt-in experiment changes only affine value products. It is never
    // combined with unmatched VGL/reverse tapes or enabled by production XML.
    {
      PreparedDeviceMixedValueInference mixed_values(image, model.cfg.nconfig, queue, handle);
      const auto prepared_bytes = mixed_values.deviceWorkspaceBytes();
      double maximum_mixed_error = 0;
      std::size_t reference_retries = 0;
      for (int repeat = 0; repeat < 2; ++repeat)
      {
        mixed_values.evaluate(model.cfg.electrons.x);
        mixed_values.downloadResults(value_phases, value_logs, no_ratios);
        reference_retries += mixed_values.usedFp64Retry();
        for (std::size_t sample = 0; sample < references.size(); ++sample)
        {
          near(value_phases[sample], references[sample].sign, "mixed value sign");
          const auto difference = std::abs(value_logs[sample] - references[sample].logabs);
          maximum_mixed_error = std::max(maximum_mixed_error, difference);
          if (!device_math::isFiniteBinary64(value_logs[sample]) ||
              difference > 5e-5 * (1 + std::abs(references[sample].logabs)))
            throw std::runtime_error("Explicit mixed affine value exceeds its numerical test tolerance");
          if (mixed_values.usedFp64Retry())
            near(value_logs[sample], references[sample].logabs, "whole-graph FP64 retry value");
        }
        if (mixed_values.deviceWorkspaceBytes() != prepared_bytes ||
            mixed_values.statistics().projections != 1 + 6 * model.blocks ||
            mixed_values.statistics().roundTripBytes() == 0)
          throw std::runtime_error("Mixed value graph lost stable storage or explicit affine cast accounting");
      }
      if (model.cfg.nconfig > 1)
      {
        const std::vector<double> prefix(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
        mixed_values.evaluate(prefix);
        std::vector<double> one_phase(1), one_value(1);
        mixed_values.downloadResults(one_phase, one_value, no_ratios);
        if (std::abs(one_value[0] - references[0].logabs) > 5e-5 * (1 + std::abs(references[0].logabs)))
          throw std::runtime_error("Mixed affine tail differs from the FP64 reference");
      }

      auto compute_parameters = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue);
      auto adapter = std::make_shared<PreparedDeviceMixedValueDense>(compute_parameters,
          model.cfg.nconfig * model.ne, queue, handle);
      bool spatial_rejected = false, tape_rejected = false;
      try { inference.setMixedValueDense(adapter); }
      catch (const std::invalid_argument&) { spatial_rejected = true; }
      PreparedDeviceInference tape_guard(image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY, true);
      try { tape_guard.setMixedValueDense(adapter); }
      catch (const std::invalid_argument&) { tape_rejected = true; }
      if (!spatial_rejected || !tape_rejected)
        throw std::runtime_error("Mixed affine adapter admitted unmatched VGL or reverse tapes");

      // A finite but FP32-unrepresentable activation must request a whole graph
      // retry, not let its deterministic zero placeholder become an accepted value.
      const auto& embedding_tensor = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
      const DenseForwardLayout cast_layout{1, embedding_tensor.shape[0], embedding_tensor.shape[1],
          embedding_tensor.shape[0], embedding_tensor.shape[1], embedding_tensor.shape[1]};
      std::vector<double> huge_input(cast_layout.sourceElements(), 1e100);
      DeviceBuffer<double> input_buffer(huge_input.size()), output_buffer(cast_layout.targetElements());
      input_buffer.upload(huge_input.data(), huge_input.size());
      adapter->beginEvaluation(image);
      adapter->project(cast_layout, input_buffer.data(), image->parameters() + embedding_tensor.begin, output_buffer.data());
      bool cast_rejected = false;
      try { adapter->finishEvaluation(); }
      catch (const MixedPrecisionRetryRequired&) { cast_rejected = true; }
      if (!cast_rejected || adapter->diagnostics().execution.nonfinite_count == 0)
        throw std::runtime_error("Mixed affine casts hid a finite overflow hazard");

      // Rebinding selects a separately admitted immutable cache. Versions and
      // exact image pointers remain independent of shape-compatible tensors.
      const auto successor_image = image->makeUpdatedImage(model.p.flat_values(), 18);
      const auto successor_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(successor_image, queue);
      const auto adapter_bytes = adapter->deviceWorkspaceBytes();
      adapter->rebindParameters(successor_cache);
      bool stale_image_rejected = false, older_cache_rejected = false;
      try { adapter->beginEvaluation(image); }
      catch (const std::invalid_argument&) { stale_image_rejected = true; }
      try { adapter->rebindParameters(compute_parameters); }
      catch (const std::invalid_argument&) { older_cache_rejected = true; }
      adapter->beginEvaluation(successor_image);
      adapter->cancelEvaluation();
      if (!stale_image_rejected || !older_cache_rejected || adapter->deviceWorkspaceBytes() != adapter_bytes)
        throw std::runtime_error("Mixed affine cache rebind lost identity guards or stable scratch");

      // Parameter overflow is a preparation hazard. The complete FP64 graph is
      // still well defined for this large finite first-layer query coefficient.
      auto overflow_parameters = model.p.flat_values();
      overflow_parameters[plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0).begin] = 1e40;
      const auto overflow_image = image->makeUpdatedImage(overflow_parameters, 18);
      PreparedDeviceMixedValueInference fallback(overflow_image, model.cfg.nconfig, queue, handle);
      PreparedDeviceInference full_precision(overflow_image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY);
      fallback.evaluate(model.cfg.electrons.x);
      full_precision.evaluate(model.cfg.electrons.x);
      fallback.downloadResults(value_phases, value_logs, no_ratios);
      std::vector<double> reference_phases(model.cfg.nconfig), reference_values(model.cfg.nconfig);
      full_precision.downloadResults(reference_phases, reference_values, no_ratios);
      if (!fallback.usedFp64Retry() || fallback.retryReason().empty())
        throw std::runtime_error("Unsafe mixed parameter conversion did not explicitly select FP64");
      for (std::size_t sample = 0; sample < references.size(); ++sample)
      {
        near(value_phases[sample], reference_phases[sample], "parameter-hazard whole FP64 sign");
        near(value_logs[sample], reference_values[sample], "parameter-hazard whole FP64 value");
      }

      // Every weight converts finitely, but the embedding SGEMM overflows.
      // Exercise production's same-graph replay, including errors detected by
      // intermediate neural admission before finishEvaluation can run.
      auto activation_parameters = model.p.flat_values();
      std::fill(activation_parameters.begin() + embedding_tensor.begin,
                activation_parameters.begin() + embedding_tensor.end, 3e38);
      const auto activation_image = image->makeUpdatedImage(activation_parameters, 19);
      const auto activation_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(activation_image, queue);
      auto activation_adapter = std::make_shared<PreparedDeviceMixedValueDense>(activation_cache,
          model.cfg.nconfig * model.ne, queue, handle);
      PreparedDeviceInference activation_graph(activation_image, model.cfg.nconfig, queue, handle,
                                               SpatialJetMode::VALUE_ONLY);
      PreparedDeviceInference activation_reference(activation_image, model.cfg.nconfig, queue, handle,
                                                   SpatialJetMode::VALUE_ONLY);
      activation_graph.setMixedDense(activation_adapter);
      std::string activation_reason;
      if (!activation_graph.evaluateWithFp64Retry(model.cfg.electrons.x, activation_reason) || activation_reason.empty())
        throw std::runtime_error("Finite compute weights did not trigger complete activation-overflow FP64 retry");
      activation_reference.evaluate(model.cfg.electrons.x);
      activation_graph.downloadResults(value_phases, value_logs, no_ratios);
      activation_reference.downloadResults(reference_phases, reference_values, no_ratios);
      for (std::size_t sample = 0; sample < references.size(); ++sample)
      {
        near(value_phases[sample], reference_phases[sample], "activation-overflow whole FP64 sign");
        near(value_logs[sample], reference_values[sample], "activation-overflow whole FP64 value");
      }

      // Failed geometry invalidates the wrapper's previous result, and RAII must
      // cancel the approximate adapter so the same prepared resource can recover.
      auto invalid = model.cfg.electrons.x;
      invalid.front() = std::numeric_limits<double>::quiet_NaN();
      bool invalid_rejected = false, stale_rejected = false;
      try { mixed_values.evaluate(invalid); }
      catch (const InferenceStatusError&) { invalid_rejected = true; }
      try { mixed_values.downloadResults(value_phases, value_logs, no_ratios); }
      catch (const std::logic_error&) { stale_rejected = true; }
      mixed_values.evaluate(model.cfg.electrons.x);
      if (!invalid_rejected || !stale_rejected || mixed_values.deviceWorkspaceBytes() != prepared_bytes)
        throw std::runtime_error("Mixed value failure/recovery exposed stale results or changed storage");
      std::cout << "PASS explicit mixed affine values: max_abs_log_error=" << maximum_mixed_error
                << " reference_retries=" << reference_retries << "/2 cast_bytes="
                << mixed_values.statistics().roundTripBytes() << " workspace_bytes=" << prepared_bytes
                << "; overflow fallback, VGL/tape guards, tails and recovery\n";
    }

    // Matched mixed spatial jets use FP32 affine products for every plane, not
    // FP64 derivatives of a separately rounded value-only forward pass.
    {
      PreparedDeviceMixedInference mixed_spatial(image, model.cfg.nconfig, queue, handle, SpatialJetMode::FULL_VGL);
      PreparedDeviceMixedInference mixed_value(image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY);
      const auto prepared_bytes = mixed_spatial.deviceWorkspaceBytes();
      std::vector<double> mixed_phases(model.cfg.nconfig), mixed_jets(spatialJetSpanElements(mixed_spatial.resultLayout())),
          mixed_ratios(model.cfg.nconfig * model.ne), mixed_logs(model.cfg.nconfig), unused;
      double maximum_value_difference = 0, maximum_gradient_scaled = 0, maximum_laplacian_scaled = 0,
          maximum_kinetic_scaled = 0;
      std::size_t fp64_retries = 0;
      const auto compare = [](double actual, double expected, double budget, const char* description) {
        if (!device_math::isFiniteBinary64(actual) ||
            std::abs(actual - expected) > budget * (1 + std::abs(expected)))
          throw std::runtime_error(std::string("Mixed full-VGL exceeds ") + description + " error budget");
        return std::abs(actual - expected) / (1 + std::abs(expected));
      };
      for (int repeat = 0; repeat < 2; ++repeat)
      {
        mixed_spatial.evaluate(model.cfg.electrons.x);
        mixed_spatial.downloadResults(mixed_phases, mixed_jets, mixed_ratios);
        mixed_value.evaluate(model.cfg.electrons.x);
        mixed_value.downloadResults(value_phases, mixed_logs, unused);
        fp64_retries += mixed_spatial.usedFp64Retry();
        const auto& mixed_layout = mixed_spatial.resultLayout();
        for (std::size_t sample = 0; sample < references.size(); ++sample)
        {
          near(mixed_phases[sample], references[sample].sign, "matched mixed VGL sign");
          const auto log_value = mixed_jets[mixed_layout.uncheckedValueOffset(sample, 0)];
          compare(log_value, references[sample].logabs, 5e-5, "value");
          compare(log_value, mixed_logs[sample], 5e-5, "value/VGL consistency");
          maximum_value_difference = std::max(maximum_value_difference, std::abs(log_value - mixed_logs[sample]));
          for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
            maximum_gradient_scaled = std::max(maximum_gradient_scaled,
                compare(mixed_jets[mixed_layout.uncheckedGradientOffset(sample, lane, 0)],
                        references[sample].gradient[lane], 5e-4, "gradient"));
          double kinetic = 0;
          for (std::size_t electron = 0; electron < model.ne; ++electron)
          {
            maximum_laplacian_scaled = std::max(maximum_laplacian_scaled,
                compare(mixed_jets[mixed_layout.uncheckedLaplacianOffset(sample, electron, 0)],
                        references[sample].lap_log[electron], 1e-3, "log Laplacian"));
            maximum_laplacian_scaled = std::max(maximum_laplacian_scaled,
                compare(mixed_ratios[sample * model.ne + electron], references[sample].lap_ratio[electron],
                        1e-3, "Laplacian ratio"));
            kinetic -= 0.5 * mixed_ratios[sample * model.ne + electron];
          }
          maximum_kinetic_scaled = std::max(maximum_kinetic_scaled,
              compare(kinetic, references[sample].kinetic, 1e-3, "kinetic"));
        }
        if (mixed_spatial.deviceWorkspaceBytes() != prepared_bytes)
          throw std::runtime_error("Mixed full-VGL reuse changed its prepared storage");
      }
      // Mixed finite differences need a same-step FP64 control: excessive h
      // truncates curved trained states, while tiny h amplifies FP32 rounding.
      const auto full_cast_bytes = mixed_spatial.statistics().roundTripBytes();
      const double base_log = mixed_logs[0];
      const auto& mixed_layout = mixed_spatial.resultLayout();
      const double jet_gradient = mixed_jets[mixed_layout.uncheckedGradientOffset(0, 0, 0)];
      const double jet_laplacian = mixed_jets[mixed_layout.uncheckedLaplacianOffset(0, 0, 0)];
      PreparedDeviceInference fd_reference(image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY);
      std::vector<double> fd_phases(model.cfg.nconfig), fd_logs(model.cfg.nconfig);
      bool fd_control_admitted = false, mixed_fd_admitted = false;
      for (const double h : {1e-2, 3e-3, 1e-3})
      {
        double mixed_laplacian_fd = 0, mixed_gradient_fd = 0, fp64_laplacian_fd = 0, fp64_gradient_fd = 0;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          auto displaced = model.cfg.electrons.x;
          displaced[dimension] += h;
          mixed_value.evaluate(displaced);
          mixed_value.downloadResults(value_phases, value_logs, unused);
          fd_reference.evaluate(displaced);
          fd_reference.downloadResults(fd_phases, fd_logs, unused);
          const double mixed_plus = value_logs[0], fp64_plus = fd_logs[0];
          displaced[dimension] -= 2 * h;
          mixed_value.evaluate(displaced);
          mixed_value.downloadResults(value_phases, value_logs, unused);
          fd_reference.evaluate(displaced);
          fd_reference.downloadResults(fd_phases, fd_logs, unused);
          const double mixed_minus = value_logs[0], fp64_minus = fd_logs[0];
          mixed_laplacian_fd += (mixed_plus + mixed_minus - 2 * base_log) / (h * h);
          fp64_laplacian_fd += (fp64_plus + fp64_minus - 2 * references[0].logabs) / (h * h);
          if (dimension == 0)
          {
            mixed_gradient_fd = (mixed_plus - mixed_minus) / (2 * h);
            fp64_gradient_fd = (fp64_plus - fp64_minus) / (2 * h);
          }
        }
        const auto scaled = [](double x, double y) { return std::abs(x - y) / (1 + std::abs(y)); };
        const bool controlled = scaled(fp64_gradient_fd, references[0].gradient[0]) <= 3e-2 &&
                                scaled(fp64_laplacian_fd, references[0].lap_log[0]) <= 1e-1;
        const bool mixed_admitted = scaled(mixed_gradient_fd, jet_gradient) <= 3e-2 &&
                                    scaled(mixed_laplacian_fd, jet_laplacian) <= 1e-1;
        std::cout << "Mixed FD control h=" << h << " gradient(mixedAD,mixedFD,fp64AD,fp64FD)="
                  << jet_gradient << ',' << mixed_gradient_fd << ',' << references[0].gradient[0] << ','
                  << fp64_gradient_fd << " laplacian=" << jet_laplacian << ',' << mixed_laplacian_fd << ','
                  << references[0].lap_log[0] << ',' << fp64_laplacian_fd << " controlled=" << controlled
                  << " mixed_admitted=" << mixed_admitted << '\n';
        fd_control_admitted = fd_control_admitted || controlled;
        mixed_fd_admitted = mixed_fd_admitted || (controlled && mixed_admitted);
      }
      if (!fd_control_admitted || !mixed_fd_admitted)
        throw std::runtime_error("Mixed full-VGL finite-difference scan lacks a controlled admissible step");
      if (model.cfg.nconfig > 1)
      {
        const std::vector<double> prefix(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
        mixed_spatial.evaluate(prefix);
        std::vector<double> tail_phase(1), tail_jets(spatialJetSpanElements(mixed_spatial.resultLayout())), tail_ratios(model.ne);
        mixed_spatial.downloadResults(tail_phase, tail_jets, tail_ratios);
        compare(tail_jets[mixed_spatial.resultLayout().uncheckedGradientOffset(0, 0, 0)],
                references[0].gradient[0], 5e-4, "tail gradient");
      }
      std::cout << "PASS matched mixed affine VGL: value_mode_difference=" << maximum_value_difference
                << " gradient_scaled_error=" << maximum_gradient_scaled
                << " laplacian_scaled_error=" << maximum_laplacian_scaled
                << " kinetic_scaled_error=" << maximum_kinetic_scaled << " reference_retries=" << fp64_retries
                << "/2 cast_bytes=" << full_cast_bytes
                << " workspace_bytes=" << prepared_bytes << "; rounding-aware finite differences and tails\n";
    }

    // Score preparation retains only ordinary block values, not per-walker scores.
    PreparedDeviceInference taped(image, model.cfg.nconfig, queue, handle,
                                   SpatialJetMode::VALUE_ONLY, true);
    const auto tape_bytes = taped.deviceWorkspaceBytes();
    taped.evaluate(model.cfg.electrons.x);
    taped.downloadResults(value_phases, value_logs, no_ratios);
    const auto canonical_parameters = model.p.flat_values();
    const auto matrix_count = model.cfg.nconfig * model.ndet * model.ne * model.ne;
    std::vector<double> retained_backflow(matrix_count), retained_envelope(matrix_count);
    std::vector<double> final_features(model.cfg.nconfig * model.ne * model.dim);
    taped.downloadValueOrbitalFactors(retained_backflow, retained_envelope);
    taped.downloadValueFinalFeatures(final_features);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
      for (std::size_t determinant = 0; determinant < model.ndet; ++determinant)
        for (std::size_t electron = 0; electron < model.ne; ++electron)
          for (std::size_t orbital = 0; orbital < model.ne; ++orbital)
          {
            const bool spin_up = electron < model.cfg.nup;
            const auto backflow_offset = plan.parameter(spin_up ? ParameterRole::BACKFLOW_UP_WEIGHT :
                                                                 ParameterRole::BACKFLOW_DOWN_WEIGHT).begin;
            const auto pi_offset = plan.parameter(spin_up ? ParameterRole::ENVELOPE_PI_UP :
                                                           ParameterRole::ENVELOPE_PI_DOWN).begin;
            const auto zeta_offset = plan.parameter(spin_up ? ParameterRole::ENVELOPE_ZETA_UP :
                                                             ParameterRole::ENVELOPE_ZETA_DOWN).begin;
            double backflow = 0, envelope = 0;
            for (std::size_t feature = 0; feature < model.dim; ++feature)
              backflow += final_features[(configuration * model.ne + electron) * model.dim + feature] *
                  canonical_parameters[backflow_offset + (feature * model.ndet + determinant) * model.ne + orbital];
            for (std::size_t nucleus = 0; nucleus < shape.nuclei; ++nucleus)
            {
              double radius_squared = 0;
              for (std::size_t dimension = 0; dimension < 3; ++dimension)
              {
                const double displacement = model.cfg.electrons.x[(configuration * model.ne + electron) * 3 + dimension] -
                    model.cfg.nuclei.x[nucleus * 3 + dimension];
                radius_squared += displacement * displacement;
              }
              const auto parameter = (determinant * model.ne + orbital) * shape.nuclei + nucleus;
              envelope += canonical_parameters[pi_offset + parameter] *
                  std::exp(-std::abs(canonical_parameters[zeta_offset + parameter]) * std::sqrt(radius_squared));
            }
            const auto index = ((configuration * model.ndet + determinant) * model.ne + electron) * model.ne + orbital;
            near(retained_backflow[index], backflow, "retained spin-conditioned backflow");
            near(retained_envelope[index], envelope, "retained orbital envelope");
          }
    if (!taped.valueHeadTape().inverses)
      throw std::runtime_error("Score tape did not prepare determinant inverse storage");
    std::cout << "PASS retained orbital factors: independent spin/backflow/envelope formulas\n";
    for (std::size_t layer = 0; layer < model.blocks; ++layer)
    {
      const auto& tape = taped.valueBlockTape(layer);
      std::vector<double> input(tape.input.capacity()), query(tape.query.capacity());
      std::vector<double> residual(tape.residual.capacity()), hidden(tape.hidden.capacity());
      tape.input.download(input.data(), input.size());
      tape.query.download(query.data(), query.size());
      tape.residual.download(residual.data(), residual.size());
      tape.hidden.download(hidden.data(), hidden.size());
      const auto query_offset = plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, layer).begin;
      const auto hidden_offset = plan.parameter(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer).begin;
      const auto bias_offset = plan.parameter(ParameterRole::UPDATE_HIDDEN_BIAS, layer).begin;
      // Independent first-electron dense/tanh formulas detect stale layer captures.
      for (std::size_t column = 0; column < model.dim; ++column)
      {
        double expected_query = 0, expected_hidden = canonical_parameters[bias_offset + column];
        for (std::size_t row = 0; row < model.dim; ++row)
        {
          expected_query += input[row] * canonical_parameters[query_offset + row * model.dim + column];
          expected_hidden += residual[row] * canonical_parameters[hidden_offset + row * model.dim + column];
        }
        near(query[column], expected_query, "retained query tape");
        near(hidden[column], std::tanh(expected_hidden), "retained hidden tape");
      }
    }
    for (std::size_t index = 0; index < references.size(); ++index)
      near(value_logs[index], references[index].logabs, "tape preserves values");
    if (taped.valueTapeBatchCount() != model.cfg.nconfig || tape_bytes <= value_bytes)
      throw std::runtime_error("Tape extent/storage accounting mismatch");
    bool invalid_input_rejected = false, stale_tape_rejected = false;
    try { taped.evaluate({}); }
    catch (const std::invalid_argument&) { invalid_input_rejected = true; }
    try { (void)taped.valueBlockTape(0); }
    catch (const std::logic_error&) { stale_tape_rejected = true; }
    if (!invalid_input_rejected || !stale_tape_rejected)
      throw std::runtime_error("Failed evaluation exposed a stale value tape");
    taped.evaluate(model.cfg.electrons.x);
    if (taped.deviceWorkspaceBytes() != tape_bytes)
      throw std::runtime_error("Tape allocated storage during reuse");
    std::cout << "PASS retained value tape: layer captures, values, invalidation, stable storage\n";

    // Contract the whole feature/attention/MLP reverse against an independent
    // central difference of a seeded final-feature scalar objective.
    std::vector<double> seed(model.cfg.nconfig * model.ne * model.dim);
    for (std::size_t index = 0; index < seed.size(); ++index)
      seed[index] = (static_cast<int>(index % 7) - 3) * 0.01;
    PreparedDeviceNetworkReverse network_reverse(image, model.cfg.nconfig, queue, handle);
    network_reverse.reverse(taped, seed);
    std::vector<double> neural_response(plan.parameterCount()), direction(plan.parameterCount(), 0.0);
    network_reverse.downloadResponse(neural_response);
    const auto reverse_bytes = network_reverse.deviceWorkspaceBytes();
    bool bad_seed_rejected = false, stale_response_rejected = false;
    try { (void)network_reverse.reverse(taped, {}); }
    catch (const std::invalid_argument&) { bad_seed_rejected = true; }
    try { network_reverse.downloadResponse(neural_response); }
    catch (const std::logic_error&) { stale_response_rejected = true; }
    if (!bad_seed_rejected || !stale_response_rejected)
      throw std::runtime_error("Failed network reverse exposed a stale response");
    network_reverse.reverse(taped, seed);
    network_reverse.downloadResponse(neural_response);
    if (network_reverse.deviceWorkspaceBytes() != reverse_bytes)
      throw std::runtime_error("Network reverse allocated storage during reuse");
    const auto add_direction = [&](ParameterRole role, std::size_t layer) {
      const auto& interval = plan.parameter(role, layer);
      direction[interval.begin] += 0.3;
      direction[interval.end - 1] -= 0.2;
    };
    add_direction(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK);
    for (std::size_t layer = 0; layer < model.blocks; ++layer)
      for (auto role : {ParameterRole::ATTENTION_QUERY_WEIGHT, ParameterRole::ATTENTION_KEY_WEIGHT,
                       ParameterRole::ATTENTION_VALUE_WEIGHT, ParameterRole::ATTENTION_OUTPUT_WEIGHT,
                       ParameterRole::UPDATE_HIDDEN_WEIGHT, ParameterRole::UPDATE_HIDDEN_BIAS,
                       ParameterRole::UPDATE_OUTPUT_WEIGHT, ParameterRole::UPDATE_OUTPUT_BIAS})
        add_direction(role, layer);
    double reverse_product = 0;
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    {
      if (!device_math::isFiniteBinary64(neural_response[parameter]))
        throw std::runtime_error("Network reverse produced a nonfinite response");
      reverse_product += direction[parameter] * neural_response[parameter];
    }
    const double delta = 1e-5;
    double objectives[2]{};
    for (std::size_t side = 0; side < 2; ++side)
    {
      auto shifted = canonical_parameters;
      for (std::size_t parameter = 0; parameter < shifted.size(); ++parameter)
        shifted[parameter] += (side == 0 ? -delta : delta) * direction[parameter];
      auto shifted_image = std::make_shared<PreparedDeviceModel>(plan, shifted, model.cfg.nuclei.x, 17);
      PreparedDeviceInference shifted_graph(shifted_image, model.cfg.nconfig, queue, handle,
                                            SpatialJetMode::VALUE_ONLY, true);
      shifted_graph.evaluate(model.cfg.electrons.x);
      std::vector<double> features(seed.size());
      shifted_graph.downloadValueFinalFeatures(features);
      for (std::size_t index = 0; index < seed.size(); ++index)
        objectives[side] += seed[index] * features[index];
    }
    near(reverse_product, (objectives[1] - objectives[0]) / (2 * delta),
         "embedding/attention/MLP weighted reverse finite difference");
    std::cout << "PASS neural trunk weighted VJP: canonical finite difference, no B-by-P matrix\n";
    validateRawFeatureNetworkReverse(image, taped, queue, handle,
        model.cfg.electrons.x, model.cfg.nuclei.x, seed);
    std::cout << "PASS raw-feature-only trunk VJP: bounded input adjoints, no P response, coordinate-direction difference\n";
    validateCoordinateGradient(image, model, model.cfg.electrons.x, queue, handle);
    std::cout << "PASS complete open coordinate gradient: CPU full VGL, FD, lifecycle, exact storage\n";

    // Complete canonical score contraction against the independent CPU evaluator.
    std::vector<double> score_weights(model.cfg.nconfig);
    for (std::size_t configuration = 0; configuration < score_weights.size(); ++configuration)
      score_weights[configuration] = configuration % 2 == 0 ? 0.7 : -0.25;
    PreparedDeviceScoreReverse score_reverse(image, model.cfg.nconfig, queue, handle);
    score_reverse.reverse(taped, score_weights);
    std::vector<double> gpu_score(plan.parameterCount()), cpu_score(plan.parameterCount(), 0.0);
    score_reverse.downloadResponse(gpu_score);
    const auto score_bytes = score_reverse.deviceWorkspaceBytes();
    bool bad_weights_rejected = false, stale_score_rejected = false;
    try { (void)score_reverse.reverse(taped, {}); }
    catch (const std::invalid_argument&) { bad_weights_rejected = true; }
    try { score_reverse.downloadResponse(gpu_score); }
    catch (const std::logic_error&) { stale_score_rejected = true; }
    if (!bad_weights_rejected || !stale_score_rejected)
      throw std::runtime_error("Failed score contraction exposed stale output");
    score_reverse.reverse(taped, score_weights);
    score_reverse.downloadResponse(gpu_score);
    if (score_reverse.deviceWorkspaceBytes() != score_bytes)
      throw std::runtime_error("Score reverse allocated storage during reuse");
    pf::DirectScoreExecutor cpu_score_executor(model, plan);
    auto cpu_score_workspace = cpu_score_executor.makeWorkspace();
    std::vector<double> product_direction(plan.parameterCount());
    for (std::size_t parameter = 0; parameter < product_direction.size(); ++parameter)
      product_direction[parameter] = (static_cast<int>(parameter % 11) - 5) * 0.01 / std::sqrt(product_direction.size());
    PreparedDeviceScoreJvp score_jvp(image, model.cfg.nconfig, queue, handle);
    score_jvp.evaluate(taped, product_direction);
    std::vector<double> products(model.cfg.nconfig), cpu_products(model.cfg.nconfig, 0.0);
    score_jvp.downloadProducts(products);
    const auto jvp_bytes = score_jvp.deviceWorkspaceBytes();
    bool invalid_direction_rejected = false, stale_jvp_rejected = false, invalid_prepared_rejected = false;
    try { (void)score_jvp.evaluate(taped, {}); }
    catch (const std::invalid_argument&) { invalid_direction_rejected = true; }
    try { score_jvp.downloadProducts(products); }
    catch (const std::logic_error&) { stale_jvp_rejected = true; }
    try { (void)score_jvp.evaluatePrepared(taped); }
    catch (const std::logic_error&) { invalid_prepared_rejected = true; }
    if (!invalid_direction_rejected || !stale_jvp_rejected || !invalid_prepared_rejected)
      throw std::runtime_error("Failed JVP request exposed stale products");

    // A failed finite-value replacement must revoke the previous direction too.
    score_jvp.prepareDirection(product_direction);
    auto nonfinite_direction = product_direction;
    nonfinite_direction[0] = std::numeric_limits<double>::quiet_NaN();
    bool nonfinite_direction_rejected = false;
    invalid_prepared_rejected = false;
    try { score_jvp.prepareDirection(nonfinite_direction); }
    catch (const std::invalid_argument&) { nonfinite_direction_rejected = true; }
    try { (void)score_jvp.evaluatePrepared(taped); }
    catch (const std::logic_error&) { invalid_prepared_rejected = true; }
    if (!nonfinite_direction_rejected || !invalid_prepared_rejected)
      throw std::runtime_error("Nonfinite replacement retained a prepared JVP direction");
    score_jvp.evaluate(taped, product_direction);
    score_jvp.downloadProducts(products);
    if (score_jvp.deviceWorkspaceBytes() != jvp_bytes)
      throw std::runtime_error("Score JVP allocated storage during reuse");
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * model.ne * 3, model.ne));
      const auto score = cpu_score_executor.evaluate(*cpu_score_workspace);
      for (std::size_t parameter = 0; parameter < cpu_score.size(); ++parameter)
      {
        cpu_score[parameter] += score_weights[configuration] * score.parameter_score[parameter];
        cpu_products[configuration] += product_direction[parameter] * score.parameter_score[parameter];
      }
    }
    double score_max_difference = 0;
    for (std::size_t parameter = 0; parameter < cpu_score.size(); ++parameter)
    {
      near(gpu_score[parameter], cpu_score[parameter], "full canonical weighted parameter score");
      score_max_difference = std::max(score_max_difference, std::abs(gpu_score[parameter] - cpu_score[parameter]));
    }
    std::cout << "PASS full weighted canonical score VJP: CPU parity; parameters=" << cpu_score.size()
              << " max_abs_difference=" << score_max_difference << '\n';
    std::vector<double> chunk(std::min<std::size_t>(65536, gpu_score.size()));
    for (std::size_t begin = 0; begin < gpu_score.size(); begin += chunk.size())
    {
      const auto count = std::min(chunk.size(), gpu_score.size() - begin);
      score_reverse.downloadResponseChunk(begin, chunk.data(), count);
      for (std::size_t index = 0; index < count; ++index)
        near(chunk[index], gpu_score[begin + index], "streamed canonical score chunk");
    }
    score_reverse.downloadResponseChunk(gpu_score.size(), nullptr, 0);
    bool invalid_chunk_rejected = false;
    try { score_reverse.downloadResponseChunk(gpu_score.size(), chunk.data(), 1); }
    catch (const std::out_of_range&) { invalid_chunk_rejected = true; }
    if (!invalid_chunk_rejected || score_reverse.parameterVersion() != image->version())
      throw std::runtime_error("GPU score chunk bounds/version contract failed");
    double weighted_jvp = 0, vjp_dot_direction = 0, jvp_max_difference = 0;
    for (std::size_t configuration = 0; configuration < products.size(); ++configuration)
    {
      near(products[configuration], cpu_products[configuration], "explicit parameter score JVP CPU parity");
      if (std::abs(products[configuration] - cpu_products[configuration]) > 5e-10 * (1 + std::abs(cpu_products[configuration])))
        throw std::runtime_error("Score JVP exceeds tight directional-product tolerance");
      weighted_jvp += score_weights[configuration] * products[configuration];
      jvp_max_difference = std::max(jvp_max_difference, std::abs(products[configuration] - cpu_products[configuration]));
    }
    for (std::size_t parameter = 0; parameter < gpu_score.size(); ++parameter)
      vjp_dot_direction += gpu_score[parameter] * product_direction[parameter];
    near(weighted_jvp, vjp_dot_direction, "JVP/VJP adjoint identity");
    if (std::abs(weighted_jvp - vjp_dot_direction) > 5e-10 * (1 + std::abs(vjp_dot_direction)))
      throw std::runtime_error("Score JVP/VJP adjoint identity exceeds tight tolerance");
    std::cout << "PASS explicit parameter score JVP and adjoint identity: max_abs_difference="
              << jvp_max_difference << '\n';

    // Reuse one uploaded direction at a second coordinate population. Snapshot
    // only the tangent call: forward-coordinate uploads are intentionally outside
    // this interval, so any H2D transfer here would be a direction regression.
    {
      int metrics_device = -1;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&metrics_device), "resident JVP metrics device");
#else
      checkRuntime(cudaGetDevice(&metrics_device), "resident JVP metrics device");
#endif
      auto& metrics = deviceMemoryMetrics(metrics_device);
      score_jvp.prepareDirection(product_direction);
      const auto before = metrics.snapshot();
      score_jvp.evaluatePrepared(taped);
      score_jvp.downloadProducts(products);
      const auto after = metrics.snapshot();
      if (before.upload_calls != after.upload_calls || before.uploaded_bytes != after.uploaded_bytes ||
          before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
          before.live_bytes != after.live_bytes)
        throw std::runtime_error("Prepared JVP uploaded a direction or changed device ownership");
      for (std::size_t configuration = 0; configuration < products.size(); ++configuration)
        near(products[configuration], cpu_products[configuration], "resident JVP original-coordinate parity");

      auto shifted_positions = model.cfg.electrons.x;
      for (std::size_t index = 0; index < shifted_positions.size(); ++index)
        shifted_positions[index] += (index % 2 == 0 ? 0.0003 : -0.0002);
      taped.evaluate(shifted_positions);
      score_jvp.evaluatePrepared(taped);
      std::vector<double> shifted_products(products.size());
      score_jvp.downloadProducts(shifted_products);
      for (std::size_t configuration = 0; configuration < products.size(); ++configuration)
      {
        cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
            shifted_positions.data() + configuration * model.ne * 3, model.ne));
        const auto shifted_score = cpu_score_executor.evaluate(*cpu_score_workspace);
        double expected_product = 0;
        for (std::size_t parameter = 0; parameter < product_direction.size(); ++parameter)
          expected_product += product_direction[parameter] * shifted_score.parameter_score[parameter];
        near(shifted_products[configuration], expected_product, "resident JVP changed-coordinate CPU parity");
      }
      score_jvp.evaluate(taped, product_direction);
      score_jvp.downloadProducts(products);
      for (std::size_t configuration = 0; configuration < products.size(); ++configuration)
        near(products[configuration], shifted_products[configuration], "resident and compatibility JVP parity");

      // Preserve the original fixture/tape and exported products for later tests.
      taped.evaluate(model.cfg.electrons.x);
      score_jvp.evaluatePrepared(taped);
      score_jvp.downloadProducts(products);
      if (score_jvp.deviceWorkspaceBytes() != jvp_bytes)
        throw std::runtime_error("Resident JVP coordinate reuse changed prepared storage");
      std::cout << "PASS resident JVP: two coordinate populations, CPU/compatibility parity, no tangent uploads/allocations\n";
    }
    if (!jvp_output.empty())
    {
      std::ofstream output(jvp_output, std::ios::binary);
      output.write(reinterpret_cast<const char*>(products.data()), products.size() * sizeof(double));
      if (!output)
        throw std::runtime_error("Score JVP output write failed");
    }
    if (!score_output.empty())
    {
      // Developer interchange on the qualified little-endian binary64 machines:
      // canonical vector only, never a walker-by-parameter table.
      std::ofstream output(score_output, std::ios::binary);
      output.write(reinterpret_cast<const char*>(gpu_score.data()), gpu_score.size() * sizeof(double));
      if (!output)
        throw std::runtime_error("Canonical score output write failed");
    }
    // A captured-grid nonlocal response uses COMPLETE bare*total-ratio weights.
    // Split seven points into 4+3 to exercise streaming, tails and repeated bases.
    constexpr std::size_t virtual_count = 7, virtual_capacity = 4;
    std::vector<double> virtual_positions(virtual_count * model.ne * 3), complete_weights(virtual_count);
    std::vector<std::size_t> references_for_virtual(virtual_count);
    std::vector<double> bare_weights(virtual_count);
    for (std::size_t point = 0; point < virtual_count; ++point)
    {
      const auto reference = (2 * point + 1) % model.cfg.nconfig;
      references_for_virtual[point] = reference;
      std::copy_n(model.cfg.electrons.x.data() + reference * model.ne * 3, model.ne * 3,
                   virtual_positions.data() + point * model.ne * 3);
      const auto electron = point % model.ne;
      virtual_positions[(point * model.ne + electron) * 3] += 0.01 * (point + 1);
      virtual_positions[(point * model.ne + electron) * 3 + 1] -= 0.02;
      virtual_positions[(point * model.ne + electron) * 3 + 2] += 0.005;
      bare_weights[point] = 0.03 * (static_cast<int>(point % 4) - 1.5);
    }
    PreparedDeviceInference virtual_forward(image, virtual_capacity, queue, handle, SpatialJetMode::VALUE_ONLY, true);
    PreparedDeviceNonlocalResponse nonlocal_response(image, model.cfg.nconfig, virtual_capacity, queue, handle);
    taped.evaluate(model.cfg.electrons.x);
    taped.downloadResults(value_phases, value_logs, no_ratios);
    nonlocal_response.begin(taped, score_weights);
    std::vector<double> gpu_nonlocal(plan.parameterCount()), cpu_nonlocal(plan.parameterCount(), 0.0);
    bool premature_nonlocal_rejected = false;
    try { nonlocal_response.downloadResponse(gpu_nonlocal); }
    catch (const std::logic_error&) { premature_nonlocal_rejected = true; }
    if (!premature_nonlocal_rejected)
      throw std::runtime_error("Nonlocal response exposed an unfinished transaction");
    for (std::size_t first = 0; first < virtual_count; first += virtual_capacity)
    {
      const auto count = std::min(virtual_capacity, virtual_count - first);
      std::vector<double> positions(virtual_positions.begin() + first * model.ne * 3,
          virtual_positions.begin() + (first + count) * model.ne * 3);
      virtual_forward.evaluate(positions);
      std::vector<double> phases(count), logs(count), weights(count);
      std::vector<std::size_t> reference_indices(references_for_virtual.begin() + first,
                                               references_for_virtual.begin() + first + count);
      virtual_forward.downloadResults(phases, logs, no_ratios);
      for (std::size_t point = 0; point < count; ++point)
      {
        const auto reference = reference_indices[point];
        weights[point] = bare_weights[first + point] * phases[point] / value_phases[reference] *
            std::exp(logs[point] - value_logs[reference]);
        complete_weights[first + point] = weights[point];
      }
      nonlocal_response.accumulateVirtualTile(virtual_forward, reference_indices, weights);
    }
    nonlocal_response.finish();
    nonlocal_response.downloadResponse(gpu_nonlocal);
    if (!nonlocal_output.empty())
    {
      std::ofstream output(nonlocal_output, std::ios::binary);
      output.write(reinterpret_cast<const char*>(gpu_nonlocal.data()), gpu_nonlocal.size() * sizeof(double));
      if (!output)
        throw std::runtime_error("Nonlocal canonical response export failed");
    }
    for (std::size_t point = 0; point < virtual_count; ++point)
    {
      const auto reference = references_for_virtual[point];
      const double weight = score_weights[reference] * complete_weights[point];
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
          virtual_positions.data() + point * model.ne * 3, model.ne));
      const auto virtual_score = cpu_score_executor.evaluate(*cpu_score_workspace);
      for (std::size_t parameter = 0; parameter < cpu_nonlocal.size(); ++parameter)
        cpu_nonlocal[parameter] += weight * virtual_score.parameter_score[parameter];
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + reference * model.ne * 3, model.ne));
      const auto base_score = cpu_score_executor.evaluate(*cpu_score_workspace);
      for (std::size_t parameter = 0; parameter < cpu_nonlocal.size(); ++parameter)
        cpu_nonlocal[parameter] -= weight * base_score.parameter_score[parameter];
    }
    double nonlocal_max_difference = 0;
    for (std::size_t parameter = 0; parameter < cpu_nonlocal.size(); ++parameter)
    {
      near(gpu_nonlocal[parameter], cpu_nonlocal[parameter], "captured nonlocal full-ratio parameter response");
      nonlocal_max_difference = std::max(nonlocal_max_difference, std::abs(gpu_nonlocal[parameter] - cpu_nonlocal[parameter]));
    }
    nonlocal_response.begin(taped, score_weights);
    taped.evaluate(model.cfg.electrons.x);
    bool stale_nonlocal_base_rejected = false;
    try { (void)nonlocal_response.finish(); }
    catch (const std::logic_error&) { stale_nonlocal_base_rejected = true; }
    bool stale_nonlocal_output_rejected = false;
    try { nonlocal_response.downloadResponse(gpu_nonlocal); }
    catch (const std::logic_error&) { stale_nonlocal_output_rejected = true; }
    if (!stale_nonlocal_base_rejected || !stale_nonlocal_output_rejected)
      throw std::runtime_error("Nonlocal response accepted a replaced base snapshot");
    std::cout << "PASS captured-grid nonlocal response: CPU parity, 4+3 tiles, complete ratios, base generation guard; max_abs_difference="
              << nonlocal_max_difference << '\n';

    if (model.cfg.nconfig > 1)
    {
      std::vector<double> prefix(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      taped.evaluate(prefix);
      const std::vector<double> prefix_seed(seed.begin(), seed.begin() + model.ne * model.dim);
      network_reverse.reverse(taped, prefix_seed);
      score_reverse.reverse(taped, {score_weights[0]});
      score_jvp.evaluate(taped, product_direction);
      std::vector<double> prefix_products(1);
      score_jvp.downloadProducts(prefix_products);
      near(prefix_products[0], cpu_products[0], "score JVP tail-prefix CPU parity");
      score_reverse.downloadResponse(gpu_score);
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), model.ne));
      const auto prefix_score = cpu_score_executor.evaluate(*cpu_score_workspace);
      for (std::size_t parameter = 0; parameter < gpu_score.size(); ++parameter)
        near(gpu_score[parameter], score_weights[0] * prefix_score.parameter_score[parameter], "score tail-prefix CPU parity");
      if (taped.valueTapeBatchCount() != 1 || taped.deviceWorkspaceBytes() != tape_bytes ||
          network_reverse.deviceWorkspaceBytes() != reverse_bytes || score_reverse.deviceWorkspaceBytes() != score_bytes ||
          score_jvp.deviceWorkspaceBytes() != jvp_bytes)
        throw std::runtime_error("Short score tape/reverse prefix changed prepared storage");
    }

    // A short live prefix must neither touch stale tail status nor resize scratch.
    if (model.cfg.nconfig > 1)
    {
      std::vector<double> prefix(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      std::vector<double> short_phases(1), short_logs(1);
      values.evaluate(prefix);
      values.downloadResults(short_phases, short_logs, no_ratios);
      near(short_phases[0], references[0].sign, "tail value sign");
      near(short_logs[0], references[0].logabs, "tail value logabs");
      if (values.batchCapacity() != model.cfg.nconfig || values.deviceWorkspaceBytes() != value_bytes)
        throw std::runtime_error("Short live batch changed prepared device storage");
      values.evaluate(model.cfg.electrons.x);
      values.downloadResults(value_phases, value_logs, no_ratios);
      for (std::size_t index = 0; index < references.size(); ++index)
        near(value_logs[index], references[index].logabs, "restored full batch logabs");

      // VGL uses the same live-prefix protocol with derivative output strides.
      inference.evaluate(prefix);
      std::vector<double> short_jets(spatialJetSpanElements(inference.resultLayout())), short_ratios(model.ne);
      inference.downloadResults(short_phases, short_jets, short_ratios);
      const auto& short_layout = inference.resultLayout();
      for (std::size_t lane = 0; lane < references[0].gradient.size(); ++lane)
        near(short_jets[short_layout.uncheckedGradientOffset(0, lane, 0)], references[0].gradient[lane],
             "tail VGL gradient");
      for (std::size_t electron = 0; electron < model.ne; ++electron)
        near(short_ratios[electron], references[0].lap_ratio[electron], "tail VGL Laplacian");
      inference.evaluate(model.cfg.electrons.x);
      inference.downloadResults(phases, jets, ratios);
      verify();
    }

    // Zero orbital projections produce genuine determinant nodes. Their values
    // are defined, even though an inverse-based log-VGL request would not be.
    // Nuclear and electron coalescences are valid value requests even though
    // the corresponding log spatial derivatives are not generally defined.
    pf::DirectValueExecutor coalescence_reference(model, plan);
    auto coalescence_workspace = coalescence_reference.makeWorkspace();
    for (bool nuclear : {true, false})
    {
      auto coalescence_positions = model.cfg.electrons.x;
      if (nuclear)
        std::copy_n(model.cfg.nuclei.x.begin(), 3, coalescence_positions.begin());
      else
        std::copy_n(coalescence_positions.begin(), 3, coalescence_positions.begin() + 3);
      values.evaluate(coalescence_positions);
      values.downloadResults(value_phases, value_logs, no_ratios);
      coalescence_workspace->setPositions(pf::GeometryPositionView::interleaved(coalescence_positions.data(), model.ne));
      const auto expected = coalescence_reference.evaluate(*coalescence_workspace);
      if (expected.sign == 0)
      {
        if (value_phases[0] != 0 || value_logs[0] != -HUGE_VAL)
          throw std::runtime_error("Value-only graph lost the same-spin coalescence node");
      }
      else
      {
        near(value_phases[0], expected.sign, "value coalescence sign");
        near(value_logs[0], expected.logabs, "value coalescence logabs");
      }
    }
    auto node_parameters = model.p.flat_values();
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& interval = plan.parameter(role);
      std::fill(node_parameters.begin() + interval.begin,
                node_parameters.begin() + interval.end, 0.0);
    }
    auto node_image = std::make_shared<PreparedDeviceModel>(plan, node_parameters, model.cfg.nuclei.x, 18);
    PreparedDeviceInference node_tape(node_image, model.cfg.nconfig, queue, handle,
                                      SpatialJetMode::VALUE_ONLY, true);
    node_tape.evaluate(model.cfg.electrons.x);
    PreparedDeviceScoreReverse node_score(node_image, model.cfg.nconfig, queue, handle);
    bool undefined_node_score_rejected = false;
    try { (void)node_score.reverse(node_tape, score_weights); }
    catch (const std::runtime_error&) { undefined_node_score_rejected = true; }
    if (!undefined_node_score_rejected)
      throw std::runtime_error("GPU log score accepted an exact total node");
    PreparedDeviceScoreJvp node_jvp(node_image, model.cfg.nconfig, queue, handle);
    bool undefined_node_jvp_rejected = false;
    try { (void)node_jvp.evaluate(node_tape, product_direction); }
    catch (const std::runtime_error&) { undefined_node_jvp_rejected = true; }
    if (!undefined_node_jvp_rejected)
      throw std::runtime_error("GPU score JVP accepted an exact total node");
    // Raw orbital losses remain differentiable at singular determinant heads.
    // Their cotangents bypass both determinant inverses and the common cusp.
    std::vector<double> node_raw_seeds(model.cfg.nconfig * model.ndet * model.ne * model.ne);
    for (std::size_t index = 0; index < node_raw_seeds.size(); ++index)
    {
      const auto electron = (index / model.ne) % model.ne;
      const auto spin_size = electron < model.cfg.nup ? model.cfg.nup : model.cfg.ndown;
      node_raw_seeds[index] = -2.0 / static_cast<double>(model.ndet * spin_size * model.ne);
    }
    node_score.reverseOrbitals(node_tape, node_raw_seeds);
    std::vector<double> node_raw_response(plan.parameterCount()), node_raw_reference(plan.parameterCount(), 0.0);
    node_score.downloadResponse(node_raw_response);
    pf::PsiFormer node_native(pf::Parameters(node_parameters, model.p.layouts), model.cfg,
                              model.ndet, model.dim, model.heads, model.blocks);
    pf::DirectScoreExecutor node_cpu(node_native, plan);
    auto node_cpu_workspace = node_cpu.makeWorkspace();
    std::vector<double> node_target(model.ndet * model.ne * model.ne, 1.0);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      node_cpu_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * 3 * model.ne, model.ne));
      const auto result = node_cpu.evaluateOrbitalMSE(*node_cpu_workspace, {node_target.data(), node_target.size()});
      for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        node_raw_reference[parameter] += result.parameter_gradient[parameter];
    }
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(node_raw_response[parameter], node_raw_reference[parameter], "singular-head inverse-free orbital-MSE reverse");
    std::vector<double> zero_channel_weights(model.cfg.nconfig * model.ndet, 0.0);
    node_score.reverseChannels(node_tape, zero_channel_weights);
    node_score.downloadResponse(node_raw_response);
    if (std::any_of(node_raw_response.begin(), node_raw_response.end(), [](double value) { return value != 0.0; }))
      throw std::runtime_error("Zero independent-channel objective has a nonzero derivative");
    auto mixed_parameters = canonical_parameters;
    // Zero only one determinant column: its cofactor derivative can be nonzero
    // even though its normalized determinant weight is zero. Inverse-only score
    // code must reject this rather than silently omit the channel contribution.
    for (auto role : {ParameterRole::ENVELOPE_PI_UP, ParameterRole::ENVELOPE_PI_DOWN})
    {
      const auto begin = plan.parameter(role).begin;
      std::fill_n(mixed_parameters.begin() + begin, shape.nuclei, 0.0);
    }
    auto mixed_image = std::make_shared<PreparedDeviceModel>(plan, mixed_parameters, model.cfg.nuclei.x, 19);
    PreparedDeviceInference mixed_tape(mixed_image, model.cfg.nconfig, queue, handle,
                                       SpatialJetMode::VALUE_ONLY, true);
    mixed_tape.evaluate(model.cfg.electrons.x);
    mixed_tape.downloadResults(value_phases, value_logs, no_ratios);
    if (value_phases[0] == 0)
      throw std::runtime_error("Mixed singular-channel fixture unexpectedly has a total node");
    PreparedDeviceScoreReverse mixed_score(mixed_image, model.cfg.nconfig, queue, handle);
    bool mixed_channel_rejected = false;
    try { (void)mixed_score.reverse(mixed_tape, score_weights); }
    catch (const std::runtime_error&) { mixed_channel_rejected = true; }
    if (!mixed_channel_rejected)
      throw std::runtime_error("Inverse-only GPU score silently accepted a singular determinant channel");
    PreparedDeviceScoreJvp mixed_jvp(mixed_image, model.cfg.nconfig, queue, handle);
    bool mixed_jvp_rejected = false;
    try { (void)mixed_jvp.evaluate(mixed_tape, product_direction); }
    catch (const std::runtime_error&) { mixed_jvp_rejected = true; }
    if (!mixed_jvp_rejected)
      throw std::runtime_error("Inverse-only GPU JVP accepted a singular determinant channel");
    std::cout << "PASS score failure guards: total node and singular channel with nonzero total amplitude\n";
    PreparedDeviceInference nodes(node_image, model.cfg.nconfig, queue, handle, SpatialJetMode::VALUE_ONLY);
    nodes.evaluate(model.cfg.electrons.x);
    nodes.downloadResults(value_phases, value_logs, no_ratios);
    for (std::size_t index = 0; index < value_logs.size(); ++index)
      if (value_phases[index] != 0 || value_logs[index] != -HUGE_VAL)
        throw std::runtime_error("Value-only graph did not preserve signed-log node semantics");
    PreparedDeviceMixedValueInference mixed_nodes(node_image, model.cfg.nconfig, queue, handle);
    mixed_nodes.evaluate(model.cfg.electrons.x);
    mixed_nodes.downloadResults(value_phases, value_logs, no_ratios);
    if (!mixed_nodes.usedFp64Retry())
      throw std::runtime_error("Mixed value conditioning did not retry an exact determinant node");
    for (std::size_t index = 0; index < value_logs.size(); ++index)
      if (value_phases[index] != 0 || value_logs[index] != -HUGE_VAL)
        throw std::runtime_error("Whole-graph mixed retry did not preserve exact signed-log nodes");
    std::cout << "Prepared device bytes: shared_model=" << image->deviceStorageBytes()
              << " value_workspace=" << value_bytes << " VGL_workspace=" << inference.deviceWorkspaceBytes() << '\n';

    // Exercise real numerical hooks through production-shaped ownership/events.
    // The standalone graph alone cannot constrain this provenance boundary.
    const auto backend = PreparedDeviceCrowdHooks::backend();
    const std::uint64_t layout_fingerprint = UINT64_C(0x5053494649585455);
    PsiFormerAcceleratorSharedModel owner(backend, layout_fingerprint, image->version());
    PsiFormerAcceleratorExecutor executor(backend, PreparedDeviceCrowdHooks::coverage());
    PsiFormerAcceleratorCrowdState crowd(1);
    PreparedDeviceCrowdHooks hooks(image, owner, 1);
    executor.prepare(crowd, layout_fingerprint, model.cfg.nconfig, hooks);
    executor.acquire(owner, crowd, hooks);
    const auto value_mask = acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES);
    hooks.bindRequest(model.cfg.electrons.x, value_mask, 1);
    const auto pending_value = executor.evaluate(owner, crowd, value_mask, model.cfg.nconfig, 1, hooks);
    bool early_result_rejected = false;
    try { (void)hooks.jets(); }
    catch (const std::logic_error&) { early_result_rejected = true; }
    if (!early_result_rejected)
      throw std::runtime_error("Numerical hooks exposed a result before its wait");
    executor.wait(crowd, pending_value.result_generation, hooks);
    for (std::size_t index = 0; index < references.size(); ++index)
    {
      near(hooks.phases()[index], references[index].sign, "hook value sign");
      near(hooks.jets()[index], references[index].logabs, "hook value logabs");
    }
    const auto vgl_mask = value_mask | acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL);
    hooks.bindRequest(model.cfg.electrons.x, vgl_mask, 2);
    const auto pending_vgl = executor.evaluate(owner, crowd, vgl_mask, model.cfg.nconfig, 2, hooks);
    executor.wait(crowd, pending_vgl.result_generation, hooks);
    const auto& hook_layout = hooks.resultLayout();
    for (std::size_t index = 0; index < references.size(); ++index)
    {
      near(hooks.jets()[hook_layout.uncheckedValueOffset(index, 0)], references[index].logabs, "hook VGL logabs");
      for (std::size_t lane = 0; lane < references[index].gradient.size(); ++lane)
        near(hooks.jets()[hook_layout.uncheckedGradientOffset(index, lane, 0)],
             references[index].gradient[lane], "hook VGL gradient");
      for (std::size_t electron = 0; electron < model.ne; ++electron)
        near(hooks.laplacianRatios()[index * model.ne + electron],
             references[index].lap_ratio[electron], "hook VGL Laplacian");
    }

    // A numerical failure after launch must poison the control-plane crowd,
    // synchronize recovery, and leave the same model image usable afterward.
    auto invalid_positions = model.cfg.electrons.x;
    invalid_positions[0] = std::numeric_limits<double>::quiet_NaN();
    hooks.bindRequest(invalid_positions, value_mask, 3);
    bool numerical_failure_rejected = false;
    try { (void)executor.evaluate(owner, crowd, value_mask, model.cfg.nconfig, 3, hooks); }
    catch (const std::runtime_error&) { numerical_failure_rejected = true; }
    if (!numerical_failure_rejected || !crowd.poisoned())
      throw std::runtime_error("Numerical crowd failure did not enter required poison state");
    executor.recover(owner, crowd, hooks);
    executor.prepare(crowd, layout_fingerprint, model.cfg.nconfig, hooks);
    executor.acquire(owner, crowd, hooks);
    hooks.bindRequest(model.cfg.electrons.x, value_mask, 4);
    const auto recovered = executor.evaluate(owner, crowd, value_mask, model.cfg.nconfig, 4, hooks);
    executor.wait(crowd, recovered.result_generation, hooks);
    for (std::size_t index = 0; index < references.size(); ++index)
      near(hooks.jets()[index], references[index].logabs, "recovered hook value");
    if (hooks.numericalBatches() != 3)
      throw std::runtime_error("Numerical hooks did not run exactly the three successful batches");
    executor.release(owner, crowd, hooks);

    // Two queues/crowds retain the same image but must not alias typed results.
    PsiFormerAcceleratorCrowdState second_crowd(2);
    PreparedDeviceCrowdHooks second_hooks(image, owner, 2);
    executor.prepare(second_crowd, layout_fingerprint, model.cfg.nconfig, second_hooks);
    executor.acquire(owner, crowd, hooks);
    executor.acquire(owner, second_crowd, second_hooks);
    auto reversed_positions = model.cfg.electrons.x;
    for (std::size_t index = 0; index < model.cfg.nconfig; ++index)
      std::copy_n(model.cfg.electrons.x.begin() + (model.cfg.nconfig - 1 - index) * 3 * model.ne,
                  3 * model.ne, reversed_positions.begin() + index * 3 * model.ne);
    hooks.bindRequest(model.cfg.electrons.x, value_mask, 5);
    second_hooks.bindRequest(reversed_positions, value_mask, 6);
    const auto first_result = executor.evaluate(owner, crowd, value_mask, model.cfg.nconfig, 5, hooks);
    const auto second_result = executor.evaluate(owner, second_crowd, value_mask, model.cfg.nconfig, 6, second_hooks);
    executor.wait(crowd, first_result.result_generation, hooks);
    executor.wait(second_crowd, second_result.result_generation, second_hooks);
    for (std::size_t index = 0; index < references.size(); ++index)
    {
      near(hooks.jets()[index], references[index].logabs, "first crowd isolated value");
      near(second_hooks.jets()[index], references[references.size() - 1 - index].logabs,
           "second crowd isolated value");
    }
    executor.release(owner, crowd, hooks);
    executor.release(owner, second_crowd, second_hooks);
    std::cout << "PASS numerical crowd hooks: typed value/VGL, provenance, wait visibility, failure/recovery, two crowds\n";

    // Production mixed crowds share one compute image but retain exclusive
    // scratch. Publication may skip versions for a dormant released crowd.
    {
      const auto accepted = model.p.flat_values();
      const auto version = model.p.version();
      auto mixed_image = std::make_shared<PreparedDeviceModel>(plan, accepted, model.cfg.nuclei.x, version);
      PreparedDeviceBatchBackend mixed_backend(mixed_image, {}, PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS);
      auto warm = mixed_backend.prepare(1), dormant = mixed_backend.prepare(1);
      const auto evaluate_mixed = [&](pf::DirectBatchTileBackendWorkspace& crowd, std::size_t current_version) {
        return mixed_backend.evaluate(crowd, pf::DirectBatchMode::FULL_VGL,
                                      model.cfg.electrons.x.data(), 1, current_version);
      };
      const auto initial = evaluate_mixed(*warm, version).logabs[0];
      bool rejected_product = false;
      try { (void)mixed_backend.prepareProductSession(); }
      catch (const std::invalid_argument&) { rejected_product = true; }
      if (!rejected_product)
        throw std::runtime_error("Mixed production backend admitted reverse products");
      auto candidate = accepted;
      candidate[0] += .003;
      bool rejected_commit = false, committed = false;
      try
      {
        mixed_backend.publishParameters(candidate, version + 1,
            [] { throw std::runtime_error("injected mixed native commit rejection"); });
      }
      catch (const std::runtime_error&) { rejected_commit = true; }
      near(evaluate_mixed(*warm, version).logabs[0], initial, "mixed rejected commit retains image/cache");
      auto invalid = candidate;
      invalid[0] = std::numeric_limits<double>::quiet_NaN();
      bool rejected_candidate = false;
      try { mixed_backend.publishParameters(invalid, version + 1, [&] { committed = true; }); }
      catch (const std::invalid_argument&) { rejected_candidate = true; }
      if (!rejected_commit || !rejected_candidate || committed)
        throw std::runtime_error("Mixed rejected publication reached native commit or changed accepted pair");
      mixed_backend.publishParameters(candidate, version + 1, [&] { committed = true; });
      (void)evaluate_mixed(*warm, version + 1);
      mixed_backend.publishParameters(accepted, version + 2, [] {});
      near(evaluate_mixed(*warm, version + 2).logabs[0], initial, "mixed accepted rollback cache rebind");
      near(evaluate_mixed(*dormant, version + 2).logabs[0], initial, "mixed dormant crowd skips versions");
      int metrics_device = 0;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&metrics_device), "mixed residency device");
#else
      checkRuntime(cudaGetDevice(&metrics_device), "mixed residency device");
#endif
      auto& metrics = deviceMemoryMetrics(metrics_device);
      const auto before = metrics.snapshot();
      (void)evaluate_mixed(*warm, version + 2);
      (void)evaluate_mixed(*dormant, version + 2);
      const auto after = metrics.snapshot();
      if (!committed || before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
          before.live_bytes != after.live_bytes)
        throw std::runtime_error("Warmed mixed crowds reallocated compute images or private scratch");
      std::cout << "PASS production mixed cache publication: two crowds, failed/invalid commit, skipped versions, zero warm allocations\n";
    }

    // Run the exact native batch-publication boundary used by PsiFormerWF.
    // This catches stride, active-electron and sparse-input integration mistakes
    // that raw graph/hook comparisons cannot detect.
    pf::DirectValueExecutor batch_value(model, plan);
    pf::DirectSpatialExecutor batch_spatial(model, batch_value, plan);
    auto batch_image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, model.p.version());
    auto backend_tiles = std::make_shared<PreparedDeviceBatchBackend>(batch_image);
    pf::DirectBatchExecutor batch_executor(batch_value, batch_spatial, backend_tiles);
    auto batch_workspace = batch_executor.makeWorkspace();
    batch_workspace->prepareTileCapacity(2);
    for (auto mode : {pf::DirectBatchMode::VALUE_ONLY, pf::DirectBatchMode::FULL_VGL,
                      pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT})
    {
      batch_workspace->resize(mode, model.cfg.nconfig);
      for (std::size_t index = 0; index < model.cfg.nconfig; ++index)
        batch_workspace->setPositions(index, pf::GeometryPositionView::interleaved(
            model.cfg.electrons.x.data() + index * 3 * model.ne, model.ne));
      if (mode == pf::DirectBatchMode::VALUE_ONLY)
      {
        const auto batch_result = batch_executor.evaluateValues(*batch_workspace);
        for (std::size_t index = 0; index < references.size(); ++index)
          near(batch_result.logabs[index], references[index].logabs, "native tile value publication");
      }
      else
      {
        std::vector<std::size_t> active(model.cfg.nconfig);
        for (std::size_t index = 0; index < active.size(); ++index) active[index] = index % model.ne;
        const auto batch_result = mode == pf::DirectBatchMode::FULL_VGL
            ? batch_executor.evaluateFull(*batch_workspace) : batch_executor.evaluateActive(*batch_workspace, active.data());
        for (std::size_t index = 0; index < references.size(); ++index)
        {
          near(batch_result.logabs[index], references[index].logabs, "native tile spatial publication");
          for (std::size_t lane = 0; lane < batch_result.gradient_stride; ++lane)
          {
            const auto reference_lane = mode == pf::DirectBatchMode::FULL_VGL ? lane : 3 * active[index] + lane;
            near(batch_result.gradient[index * batch_result.gradient_stride + lane],
                 references[index].gradient[reference_lane], "native tile gradient publication");
          }
          if (mode == pf::DirectBatchMode::FULL_VGL)
            for (std::size_t electron = 0; electron < model.ne; ++electron)
              near(batch_result.lap_ratio[index * model.ne + electron], references[index].lap_ratio[electron],
                   "native tile Laplacian publication");
        }
      }
    }
    batch_workspace->resizeSparseValues(1, 1);
    const auto first_position = pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), model.ne);
    batch_workspace->setReferenceConfiguration(0, first_position);
    batch_workspace->setVirtualReplacement(0, 0, 0, {first_position(0, 0), first_position(0, 1), first_position(0, 2)});
    const auto sparse_result = batch_executor.evaluateValues(*batch_workspace);
    near(sparse_result.logabs[0], references[0].logabs, "native sparse reference publication");
    near(sparse_result.logabs[1], references[0].logabs, "native sparse replacement publication");
    std::cout << "PASS native numerical tiles: values/VGL/active gradients/sparse replacements/tails\n";

    // Exercise the same publication barrier used by structured updates, legacy
    // resets and VP restore. Reuse existing crowds across failures and versions.
    // Independent channel and raw-orbital objectives share the neural reverse,
    // but must not inherit the normalized determinant-sum or cusp seed rules.
    taped.evaluate(model.cfg.electrons.x);
    std::vector<double> channel_coefficients(model.cfg.nconfig * model.ndet);
    std::vector<double> channel_reference(plan.parameterCount(), 0.0), head_response(plan.parameterCount());
    for (std::size_t index = 0; index < channel_coefficients.size(); ++index)
      channel_coefficients[index] = .13 * (static_cast<int>(index % 5) - 2);
    score_reverse.reverseChannels(taped, channel_coefficients);
    score_reverse.downloadResponse(head_response);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * 3 * model.ne, model.ne));
      const auto result = cpu_score_executor.evaluateWeightedChannelScore(*cpu_score_workspace,
          channel_coefficients.data() + configuration * model.ndet, model.ndet);
      for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        channel_reference[parameter] += result.parameter_score[parameter];
    }
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(head_response[parameter], channel_reference[parameter], "independent weighted channel score CPU parity");

    const auto orbital_count = model.ndet * model.ne * model.ne;
    std::vector<double> backflow(model.cfg.nconfig * orbital_count), envelope(backflow.size());
    taped.downloadValueOrbitalFactors(backflow, envelope);
    std::vector<double> raw_adjoints(backflow.size()), orbital_target(orbital_count);
    std::fill(channel_reference.begin(), channel_reference.end(), 0.0);
    for (std::size_t index = 0; index < orbital_count; ++index)
      orbital_target[index] = .07 * (static_cast<int>(index % 7) - 3);
    for (std::size_t configuration = 0; configuration < model.cfg.nconfig; ++configuration)
    {
      for (std::size_t index = 0; index < orbital_count; ++index)
      {
        const auto electron = (index / model.ne) % model.ne;
        const auto spin_size = electron < model.cfg.nup ? model.cfg.nup : model.cfg.ndown;
        const auto offset = configuration * orbital_count + index;
        raw_adjoints[offset] = 2.0 * (backflow[offset] * envelope[offset] - orbital_target[index]) /
            static_cast<double>(model.ndet * spin_size * model.ne);
      }
      cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
          model.cfg.electrons.x.data() + configuration * 3 * model.ne, model.ne));
      const auto result = cpu_score_executor.evaluateOrbitalMSE(*cpu_score_workspace,
          {orbital_target.data(), orbital_target.size()});
      for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        channel_reference[parameter] += result.parameter_gradient[parameter];
    }
    score_reverse.reverseOrbitals(taped, raw_adjoints);
    score_reverse.downloadResponse(head_response);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(head_response[parameter], channel_reference[parameter], "raw orbital-MSE gradient CPU parity");
    if (head_response[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] != 0.0)
      throw std::runtime_error("Raw orbital objective incorrectly differentiated the common cusp");
    std::cout << "PASS independent determinant-channel and raw orbital-MSE head seeds: all-parameter CPU parity\n";
    if (model.ndet % 2 == 0)
    {
      // Pair identical regular channels with an exact column sign reversal.
      // Their sum is nodal, but each independent channel log score is defined.
      auto cancelling_parameters = model.p.flat_values();
      for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
      {
        const auto& interval = plan.parameter(role);
        for (std::size_t feature = 0; feature < model.dim; ++feature)
          for (std::size_t determinant = 0; determinant < model.ndet; ++determinant)
            for (std::size_t orbital = 0; orbital < model.ne; ++orbital)
            {
              const auto source = interval.begin + feature * model.ndet * model.ne + orbital;
              const auto destination = source + determinant * model.ne;
              cancelling_parameters[destination] = canonical_parameters[source] *
                  (determinant % 2 != 0 && orbital == 0 ? -1.0 : 1.0);
            }
      }
      for (auto role : {ParameterRole::ENVELOPE_PI_UP, ParameterRole::ENVELOPE_PI_DOWN,
                        ParameterRole::ENVELOPE_ZETA_UP, ParameterRole::ENVELOPE_ZETA_DOWN})
      {
        const auto& interval = plan.parameter(role);
        const auto per_channel = (interval.end - interval.begin) / model.ndet;
        for (std::size_t determinant = 0; determinant < model.ndet; ++determinant)
          std::copy_n(canonical_parameters.data() + interval.begin, per_channel,
                       cancelling_parameters.data() + interval.begin + determinant * per_channel);
      }
      auto cancelling_image = std::make_shared<PreparedDeviceModel>(plan, cancelling_parameters, model.cfg.nuclei.x, 28);
      PreparedDeviceInference cancelling_tape(cancelling_image, 1, queue, handle, SpatialJetMode::VALUE_ONLY, true);
      std::vector<double> first_coordinates(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
      cancelling_tape.evaluate(first_coordinates);
      std::vector<double> cancelling_phase(1), cancelling_log(1), no_spatial_output;
      cancelling_tape.downloadResults(cancelling_phase, cancelling_log, no_spatial_output);
      if (cancelling_phase[0] != 0.0)
        throw std::runtime_error("Independent-channel cancellation fixture is not an exact total node");
      PreparedDeviceScoreReverse cancelling_reverse(cancelling_image, 1, queue, handle);
      std::vector<double> select_channel(model.ndet, 0.0);
      select_channel[0] = 1.0;
      cancelling_reverse.reverseChannels(cancelling_tape, select_channel);
      cancelling_reverse.downloadResponse(head_response);
      pf::PsiFormer cancelling_native(pf::Parameters(cancelling_parameters, model.p.layouts), model.cfg,
                                       model.ndet, model.dim, model.heads, model.blocks);
      pf::DirectScoreExecutor cancelling_cpu(cancelling_native, plan);
      auto cancelling_workspace = cancelling_cpu.makeWorkspace();
      cancelling_workspace->setPositions(first_position);
      const auto cancelling_reference = cancelling_cpu.evaluateWeightedChannelScore(*cancelling_workspace,
          select_channel.data(), select_channel.size());
      for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        near(head_response[parameter], cancelling_reference.parameter_score[parameter],
             "independent channel score at a total cancellation node");
      std::cout << "PASS independent channel at exact determinant-sum node; raw orbital reverse at singular heads\n";
      PreparedDeviceBatchBackend cancelling_backend(cancelling_image);
      auto zero_node_lane = cancelling_backend.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4);
      zero_node_lane->prepareScoreContractions(1, cancelling_image->version());
      zero_node_lane->beginScoreContraction(1, cancelling_image->version());
      const double zero_nodal_weight = 0;
      bool zero_node_rejected = false;
      try { zero_node_lane->accumulateWeightedScoreTile(first_coordinates.data(), &zero_nodal_weight,
          1, cancelling_image->version()); }
      catch (const std::runtime_error&) { zero_node_rejected = true; }
      if (!zero_node_rejected)
        throw std::runtime_error("Zero-weight score tile bypassed total-node admission");
      std::cout << "PASS weighted score tile rejects an exact total node even at zero coefficient\n";
    }

    auto dormant_tile = backend_tiles->prepare(1);
    auto product_session = backend_tiles->prepareProductSession();
    const auto session_bytes = product_session->deviceWorkspaceBytes();
    auto score_only_session = backend_tiles->prepareProductSession(ProductSessionProfile::SCORE_ONLY);
    const auto score_only_bytes = score_only_session->deviceWorkspaceBytes();
    const auto score_only_host_bytes = score_only_session->hostWorkspaceBytes();
    if (product_session->profile() != ProductSessionProfile::SCORE_AND_KINETIC ||
        score_only_session->profile() != ProductSessionProfile::SCORE_ONLY ||
        score_only_bytes >= session_bytes || score_only_host_bytes >= product_session->hostWorkspaceBytes())
      throw std::runtime_error("Product session preparation profile/memory contract failed");
    if (session_bytes - score_only_bytes < checkedDeviceBytes<double>(plan.parameterCount()) ||
        product_session->hostWorkspaceBytes() - score_only_host_bytes < checkedDeviceBytes<double>(plan.parameterCount()))
      throw std::runtime_error("Score-only profile retained canonical kinetic response storage");
    std::cout << "PASS product session profiles: full_device_bytes=" << session_bytes
              << " score_only_device_bytes=" << score_only_bytes
              << " saved_device_bytes=" << session_bytes - score_only_bytes
              << " full_host_bytes=" << product_session->hostWorkspaceBytes()
              << " score_only_host_bytes=" << score_only_host_bytes << '\n';
    const auto& session_score = product_session->score(model.cfg.electrons.x.data(), model.p.version());
    cpu_score_workspace->setPositions(first_position);
    const auto session_reference = cpu_score_executor.evaluate(*cpu_score_workspace);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(session_score[parameter], session_reference.parameter_score[parameter], "leased score session CPU parity");
    near(product_session->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
         cpu_products[0], "leased explicit JVP CPU parity");
    const auto& score_only_score = score_only_session->score(model.cfg.electrons.x.data(), model.p.version());
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(score_only_score[parameter], session_reference.parameter_score[parameter],
           "score-only session canonical score CPU parity");
    near(score_only_session->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
         cpu_products[0], "score-only session explicit JVP CPU parity");
    score_only_session->prepareScoreDirection(product_direction, model.p.version());
    const auto before_unsupported_kinetic = score_only_session->numericalProducts();
    bool unsupported_kinetic_rejected = false, unsupported_gradient_rejected = false;
    try { (void)score_only_session->kinetic(model.cfg.electrons.x.data(),
        external_contiguous.data(), inverse_masses.data(), model.p.version()); }
    catch (const std::logic_error&) { unsupported_kinetic_rejected = true; }
    try { (void)score_only_session->gradientResponse(model.cfg.electrons.x.data(),
        external_contiguous.data(), inverse_masses.data(), model.p.version()); }
    catch (const std::logic_error&) { unsupported_gradient_rejected = true; }
    if (!unsupported_kinetic_rejected || !unsupported_gradient_rejected ||
        score_only_session->numericalProducts() != before_unsupported_kinetic ||
        score_only_session->deviceWorkspaceBytes() != score_only_bytes ||
        score_only_session->hostWorkspaceBytes() != score_only_host_bytes)
      throw std::runtime_error("Score-only session kinetic rejection changed products or prepared storage");
    near(score_only_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()),
         cpu_products[0], "score-only unsupported kinetic preserves prepared direction");

    bool stale_product_rejected = false;
    try { product_session->score(model.cfg.electrons.x.data(), model.p.version() + 1); }
    catch (const std::invalid_argument&) { stale_product_rejected = true; }
    if (!stale_product_rejected || product_session->numericalProducts() != 2)
      throw std::runtime_error("Product session stale request/counter contract failed");
    product_session->prepareScoreDirection(product_direction, model.p.version());
    if (product_session->numericalProducts() != 2)
      throw std::runtime_error("Direction preparation fabricated a numerical product");
    near(product_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()),
         cpu_products[0], "leased resident JVP CPU parity");
    bool stale_direction_preparation_rejected = false, cleared_session_direction_rejected = false;
    try { product_session->prepareScoreDirection(product_direction, model.p.version() + 1); }
    catch (const std::invalid_argument&) { stale_direction_preparation_rejected = true; }
    try { (void)product_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()); }
    catch (const std::logic_error&) { cleared_session_direction_rejected = true; }
    if (!stale_direction_preparation_rejected || !cleared_session_direction_rejected)
      throw std::runtime_error("Failed session preparation retained an old direction");
    product_session->prepareScoreDirection(product_direction, model.p.version());
    near(product_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()),
         cpu_products[0], "leased resident JVP recovery");
    const auto& session_kinetic = product_session->kinetic(model.cfg.electrons.x.data(),
        external_contiguous.data(), inverse_masses.data(), model.p.version());
    kinetic_workspace->setPositions(first_position);
    const auto session_kinetic_reference = kinetic_cpu.evaluate(*kinetic_workspace,
        external_contiguous.data(), 3 * model.ne, inverse_masses.data(), model.ne);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(session_kinetic[parameter], session_kinetic_reference.kinetic_parameter_response[parameter],
           "leased kinetic session total-drift/mass CPU parity");
    std::vector<double> invalid_product_positions(model.cfg.electrons.x.begin(),
        model.cfg.electrons.x.begin() + 3 * model.ne);
    invalid_product_positions[0] = std::numeric_limits<double>::quiet_NaN();
    bool failed_product_rejected = false;
    try { product_session->score(invalid_product_positions.data(), model.p.version()); }
    catch (const std::exception&) { failed_product_rejected = true; }
    if (!failed_product_rejected)
      throw std::runtime_error("Product session accepted nonfinite numerical input");
    const auto& recovered_score = product_session->score(model.cfg.electrons.x.data(), model.p.version());
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(recovered_score[parameter], session_reference.parameter_score[parameter],
           "leased product failure recovery preserves model and prepared scratch");
    std::vector<double> device_channel_phases(model.ndet), device_channel_logs(model.ndet);
    std::vector<double> cpu_channel_phases(model.ndet), cpu_channel_logs(model.ndet);
    product_session->channelAmplitudes(model.cfg.electrons.x.data(), device_channel_phases.data(),
                                       device_channel_logs.data(), model.ndet, model.p.version());
    cpu_score_executor.evaluateChannelAmplitudes(*cpu_score_workspace, cpu_channel_phases.data(),
                                                 cpu_channel_logs.data(), model.ndet);
    for (std::size_t channel = 0; channel < model.ndet; ++channel)
    {
      near(device_channel_phases[channel], cpu_channel_phases[channel], "leased independent channel phase");
      near(device_channel_logs[channel], cpu_channel_logs[channel], "leased independent channel log amplitude");
    }
    const auto& leased_channel_response = product_session->channelScore(model.cfg.electrons.x.data(),
        channel_coefficients.data(), model.ndet, model.p.version());
    const auto leased_channel_reference = cpu_score_executor.evaluateWeightedChannelScore(*cpu_score_workspace,
        channel_coefficients.data(), model.ndet);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(leased_channel_response[parameter], leased_channel_reference.parameter_score[parameter],
           "leased independent channel response");
    score_only_session->channelAmplitudes(model.cfg.electrons.x.data(), device_channel_phases.data(),
                                          device_channel_logs.data(), model.ndet, model.p.version());
    for (std::size_t channel = 0; channel < model.ndet; ++channel)
    {
      near(device_channel_phases[channel], cpu_channel_phases[channel], "score-only independent channel phase");
      near(device_channel_logs[channel], cpu_channel_logs[channel], "score-only independent channel log amplitude");
    }
    const auto& score_only_channel_response = score_only_session->channelScore(model.cfg.electrons.x.data(),
        channel_coefficients.data(), model.ndet, model.p.version());
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(score_only_channel_response[parameter], leased_channel_reference.parameter_score[parameter],
           "score-only independent channel response");

    const auto leased_orbital_loss = product_session->orbitalMse(model.cfg.electrons.x.data(),
        orbital_target.data(), orbital_target.size(), head_response.data(), head_response.size(), model.p.version());
    const auto leased_orbital_reference = cpu_score_executor.evaluateOrbitalMSE(*cpu_score_workspace,
        {orbital_target.data(), orbital_target.size()});
    near(leased_orbital_loss, leased_orbital_reference.loss, "leased orbital-MSE loss");
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(head_response[parameter], leased_orbital_reference.parameter_gradient[parameter],
           "leased orbital-MSE gradient");
    const auto accepted_values = model.p.flat_values();
    auto proposed_values = accepted_values;
    proposed_values[0] += .003;
    const auto accepted_version = model.p.version();
    bool native_called = false;
    bool publication_rejected = false;
    try
    {
      backend_tiles->publishParameters(proposed_values, accepted_version + 1, [&]() {
        native_called = true;
        throw std::runtime_error("injected native validation failure");
      });
    }
    catch (const std::runtime_error&) { publication_rejected = true; }
    if (!native_called || !publication_rejected || model.p.version() != accepted_version)
      throw std::runtime_error("Device publication failure changed the accepted native model");
    const auto preserved_result = batch_executor.evaluateValues(*batch_workspace);
    near(preserved_result.logabs[0], references[0].logabs, "failed publication preserves accepted device image");

    auto invalid_values = proposed_values;
    invalid_values[0] = std::numeric_limits<double>::quiet_NaN();
    native_called = false;
    publication_rejected = false;
    try
    {
      backend_tiles->publishParameters(invalid_values, accepted_version + 1, [&]() { native_called = true; });
    }
    catch (const std::invalid_argument&) { publication_rejected = true; }
    if (native_called || !publication_rejected)
      throw std::runtime_error("Invalid candidate reached the native publication callback");

    backend_tiles->publishParameters(proposed_values, accepted_version + 1,
                                     [&]() { model.p.set_flat_values(proposed_values); });
    bool published_direction_rejected = false;
    try { (void)product_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()); }
    catch (const std::logic_error&) { published_direction_rejected = true; }
    if (!published_direction_rejected)
      throw std::runtime_error("Session publication retained an old-image JVP direction");
    bool score_only_published_direction_rejected = false;
    try { (void)score_only_session->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()); }
    catch (const std::logic_error&) { score_only_published_direction_rejected = true; }
    if (!score_only_published_direction_rejected)
      throw std::runtime_error("Score-only publication retained an old-image direction");

    auto changed_workspace = batch_value.makeWorkspace();
    changed_workspace->setPositions(first_position);
    const auto changed_value = batch_value.evaluate(*changed_workspace);
    const auto changed_result = batch_executor.evaluateValues(*batch_workspace);
    near(changed_result.logabs[0], changed_value.logabs, "published native/device parameter agreement");
    const auto& changed_score = product_session->score(model.cfg.electrons.x.data(), model.p.version());
    const auto changed_score_reference = cpu_score_executor.evaluate(*cpu_score_workspace);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(changed_score[parameter], changed_score_reference.parameter_score[parameter],
           "leased product rebind CPU parity");
    const auto& changed_score_only = score_only_session->score(model.cfg.electrons.x.data(), model.p.version());
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(changed_score_only[parameter], changed_score_reference.parameter_score[parameter],
           "score-only session rebind CPU parity");
    if (score_only_session->profile() != ProductSessionProfile::SCORE_ONLY ||
        score_only_session->deviceWorkspaceBytes() != score_only_bytes ||
        score_only_session->hostWorkspaceBytes() != score_only_host_bytes)
      throw std::runtime_error("Score-only rebind changed immutable profile or prepared scratch");

    backend_tiles->publishParameters(accepted_values, accepted_version + 2,
                                     [&]() { model.p.set_flat_values(accepted_values); });
    const auto restored_result = batch_executor.evaluateValues(*batch_workspace);
    near(restored_result.logabs[0], references[0].logabs, "rollback publishes a newer image of accepted values");
    const auto dormant_result = backend_tiles->evaluate(*dormant_tile, pf::DirectBatchMode::FULL_VGL,
                                                       model.cfg.electrons.x.data(), 1, model.p.version());
    near(dormant_result.logabs[0], references[0].logabs, "dormant context skips retired parameter versions");
    for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
      near(dormant_result.gradient[lane * dormant_result.gradient_lane_stride], references[0].gradient[lane],
           "rebound spatial scratch after rollback");
    native_called = false;
    publication_rejected = false;
    try
    {
      backend_tiles->publishParameters(proposed_values, accepted_version + 1, [&]() { native_called = true; });
    }
    catch (const std::invalid_argument&) { publication_rejected = true; }
    if (native_called || !publication_rejected)
      throw std::runtime_error("Stale publication reached the native commit");
    std::cout << "PASS production image publication: failed commit, invalid/stale candidate, reused/dormant crowds, rollback\n";
    if (session_bytes != product_session->deviceWorkspaceBytes())
      throw std::runtime_error("Product session reallocated prepared scratch");
    std::cout << "PASS leased numerical product session: score/JVP, stale version, publication rebind, fixed scratch\n";
    {
      // Production uses one real reverse per configuration, retaining only C
      // complex sums. Exercise the same leases with genuinely complex weights.
      using Complex = std::complex<double>;
      std::array<std::vector<Complex>, 2> contracted, expected;
      for (auto* collection : {&contracted, &expected})
        for (auto& channel : *collection)
          channel.resize(plan.parameterCount());
      product_session->prepareScoreContractions(2, model.p.version());
      bool empty_begin_rejected = false;
      try { product_session->beginScoreContraction(0, model.p.version()); }
      catch (const std::invalid_argument&) { empty_begin_rejected = true; }
      if (!empty_begin_rejected)
        throw std::runtime_error("Score contraction accepted an empty channel count");
      const auto prepared_bytes = product_session->deviceWorkspaceBytes();
      const auto contraction_version = model.p.version();
      auto run_contraction = [&]() {
        product_session->beginScoreContraction(2, contraction_version);
        for (std::size_t sample = 0; sample < model.cfg.nconfig; ++sample)
        {
          const std::array<Complex, 2> coefficients{{Complex(.17 * (sample + 1), -.11),
              sample == 0 ? Complex{} : Complex(-.23, .09 * sample)}};
          const double* coordinates = model.cfg.electrons.x.data() + sample * 3 * model.ne;
          product_session->accumulateScoreContraction(coordinates, coefficients.data(), 2, contraction_version);
        }
        product_session->finishScoreContraction(contraction_version);
      };
      for (std::size_t sample = 0; sample < model.cfg.nconfig; ++sample)
      {
        cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(
            model.cfg.electrons.x.data() + sample * 3 * model.ne, model.ne));
        const auto reference = cpu_score_executor.evaluate(*cpu_score_workspace);
        const std::array<Complex, 2> coefficients{{Complex(.17 * (sample + 1), -.11),
            sample == 0 ? Complex{} : Complex(-.23, .09 * sample)}};
        for (std::size_t channel = 0; channel < 2; ++channel)
          for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
            expected[channel][parameter] += coefficients[channel] * reference.parameter_score[parameter];
      }
      run_contraction();
      int contraction_device = -1;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&contraction_device), "score contraction metrics device");
#else
      checkRuntime(cudaGetDevice(&contraction_device), "score contraction metrics device");
#endif
      auto& contraction_metrics = deviceMemoryMetrics(contraction_device);
      const auto before_reuse = contraction_metrics.snapshot();
      run_contraction();
      const auto before_download = contraction_metrics.snapshot();
      for (std::size_t channel = 0; channel < 2; ++channel)
      {
        product_session->downloadScoreContraction(channel, contracted[channel].data(),
            contracted[channel].size(), contraction_version);
        for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
        {
          near(contracted[channel][parameter].real(), expected[channel][parameter].real(),
               "resident complex score contraction real CPU parity");
          near(contracted[channel][parameter].imag(), expected[channel][parameter].imag(),
               "resident complex score contraction imaginary CPU parity");
        }
      }
      const auto after_download = contraction_metrics.snapshot();
      if (after_download.allocation_calls != before_reuse.allocation_calls ||
          after_download.free_calls != before_reuse.free_calls ||
          after_download.download_calls - before_download.download_calls != 2 ||
          after_download.downloaded_bytes - before_download.downloaded_bytes != 4 * plan.parameterCount() * sizeof(double) ||
          product_session->deviceWorkspaceBytes() != prepared_bytes)
        throw std::runtime_error("Resident score contraction warmed allocation/download contract failed");

      // Bounded canonical ranges preserve the completed whole-vector result.
      const auto chunk_capacity = plan.parameterCount() / 3 + 1;
      std::vector<Complex> chunk_output(chunk_capacity);
      for (std::size_t offset = 0; offset < plan.parameterCount(); offset += chunk_capacity)
      {
        const auto count = std::min(chunk_capacity, plan.parameterCount() - offset);
        product_session->downloadScoreContractionChunk(1, offset, chunk_output.data(), count, contraction_version);
        for (std::size_t i = 0; i < count; ++i)
        {
          near(chunk_output[i].real(), contracted[1][offset + i].real(), "score chunk real parity");
          near(chunk_output[i].imag(), contracted[1][offset + i].imag(), "score chunk imaginary parity");
        }
      }
      bool invalid_chunk_rejected = false;
      try { product_session->downloadScoreContractionChunk(0, plan.parameterCount(), chunk_output.data(), 1,
                                                           contraction_version); }
      catch (const std::invalid_argument&) { invalid_chunk_rejected = true; }
      if (!invalid_chunk_rejected)
        throw std::runtime_error("Score contraction accepted an out-of-range chunk");
      run_contraction();
      bool stale_chunk_rejected = false;
      try { product_session->downloadScoreContractionChunk(0, 0, chunk_output.data(), 1, contraction_version + 1); }
      catch (const std::invalid_argument&) { stale_chunk_rejected = true; }
      if (!stale_chunk_rejected)
        throw std::runtime_error("Score contraction accepted a stale chunk");

      auto expect_no_contraction = [&]() {
        bool rejected = false;
        try { product_session->downloadScoreContraction(0, contracted[0].data(), contracted[0].size(), model.p.version()); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected)
          throw std::runtime_error("Failed or unfinished score contraction exposed a result");
      };
      product_session->beginScoreContraction(2, contraction_version);
      expect_no_contraction();
      bool unfinished_chunk_rejected = false;
      try { product_session->downloadScoreContractionChunk(0, 0, chunk_output.data(), 1, contraction_version); }
      catch (const std::logic_error&) { unfinished_chunk_rejected = true; }
      if (!unfinished_chunk_rejected)
        throw std::runtime_error("Score contraction exposed an unfinished chunk");
      run_contraction();
      bool stale_finish_rejected = false;
      try { product_session->finishScoreContraction(contraction_version + 1); }
      catch (const std::exception&) { stale_finish_rejected = true; }
      if (!stale_finish_rejected)
        throw std::runtime_error("Score contraction accepted stale finalization");
      expect_no_contraction();
      product_session->beginScoreContraction(2, contraction_version);
      // Even zero coefficients must admit every geometry and score channel.
      const std::array<Complex, 2> unit_coefficients{};
      bool invalid_contraction_rejected = false;
      try { product_session->accumulateScoreContraction(invalid_product_positions.data(),
          unit_coefficients.data(), 2, contraction_version); }
      catch (const std::exception&) { invalid_contraction_rejected = true; }
      if (!invalid_contraction_rejected)
        throw std::runtime_error("Score contraction accepted nonfinite geometry");
      expect_no_contraction();
      product_session->beginScoreContraction(2, contraction_version);
      const std::array<Complex, 2> invalid_coefficients{{Complex(1, 0),
          Complex(0, std::numeric_limits<double>::quiet_NaN())}};
      bool invalid_coefficient_rejected = false;
      try { product_session->accumulateScoreContraction(model.cfg.electrons.x.data(),
          invalid_coefficients.data(), 2, contraction_version); }
      catch (const std::invalid_argument&) { invalid_coefficient_rejected = true; }
      if (!invalid_coefficient_rejected)
        throw std::runtime_error("Score contraction accepted a nonfinite algebraic coefficient");
      expect_no_contraction();
      run_contraction();
      backend_tiles->publishParameters(accepted_values, contraction_version + 1,
                                       [&]() { model.p.set_flat_values(accepted_values); });
      expect_no_contraction();

      // Finite responses and coefficients can overflow only after contraction.
      PreparedDeviceScoreContraction overflow_contraction(1, 1, queue);
      DeviceBuffer<double> overflow_response(1);
      const double maximum = std::numeric_limits<double>::max();
      overflow_response.upload(&maximum, 1);
      const Complex double_coefficient(2, 0);
      overflow_contraction.begin(1);
      overflow_contraction.accumulate(overflow_response.data(), &double_coefficient, 1);
      bool overflow_rejected = false;
      try { overflow_contraction.finish(); }
      catch (const std::runtime_error&) { overflow_rejected = true; }
      if (!overflow_rejected)
        throw std::runtime_error("Score contraction admitted a nonfinite complex sum");
      std::cout << "PASS resident serial score contractions: complex channels/CPU parity, exact downloads, "
                   "warmed reuse, failure/stale/publication invalidation, finite-sum admission\n";
    }
    if (PreparedDeviceProductSession::admittedScoreCapacity(shape, 4) == 4)
    {
      using Complex = std::complex<double>;
      const auto before_lanes = product_session->transferSnapshot();
      auto single_lane = backend_tiles->prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 1);
      auto tiled_lane = backend_tiles->prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 4);
      const auto after_lanes = tiled_lane->transferSnapshot();
      if (after_lanes.live_bytes - before_lanes.live_bytes !=
          single_lane->deviceWorkspaceBytes() + tiled_lane->deviceWorkspaceBytes())
        throw std::runtime_error("Score/JVP session accounting differs from actual owned buffers");
      if (single_lane->scoreCapacity() != 1 || tiled_lane->scoreCapacity() != 4)
        throw std::runtime_error("Explicit score tile preparation lost its immutable capacity");
      for (const auto electrons : {std::size_t{100}, std::size_t{200}, std::size_t{300}})
      {
        auto large_shape = shape;
        large_shape.spin_up_electrons = electrons / 2;
        large_shape.spin_down_electrons = electrons - electrons / 2;
        if (PreparedDeviceProductSession::admittedScoreCapacity(large_shape, 4) != 1)
          throw std::runtime_error("Large-electron model bypassed score tile qualification admission");
      }
      auto large_width_shape = shape;
      large_width_shape.feature_dimension = 1024;
      auto large_head_shape = shape;
      large_head_shape.determinants = 64;
      if (PreparedDeviceProductSession::admittedScoreCapacity(large_width_shape, 4) != 1 ||
          PreparedDeviceProductSession::admittedScoreCapacity(large_head_shape, 4) != 1)
        throw std::runtime_error("Unqualified large model bypassed bounded score capacity admission");
      std::cout << "Score tile preparation: capacity1_device_bytes=" << single_lane->deviceWorkspaceBytes()
                << " capacity4_device_bytes=" << tiled_lane->deviceWorkspaceBytes()
                << " additional_device_bytes=" << tiled_lane->deviceWorkspaceBytes() - single_lane->deviceWorkspaceBytes()
                << " additional_host_bytes=" << tiled_lane->hostWorkspaceBytes() - single_lane->hostWorkspaceBytes() << '\n';
      tiled_lane->prepareScoreContractions(1, model.p.version());
      const auto fixed_bytes = tiled_lane->deviceWorkspaceBytes();
      std::vector<Complex> output(plan.parameterCount());
      std::vector<double> expected(plan.parameterCount());
      // These populations cover full/tail tiles and repeated 4->1->3->4 counts.
      for (const auto population : {std::size_t{1}, std::size_t{4}, std::size_t{5}, std::size_t{7},
                                    std::size_t{4}, std::size_t{1}, std::size_t{3}, std::size_t{4}})
      {
        std::vector<double> coordinates(population * 3 * model.ne), weights(population);
        std::fill(expected.begin(), expected.end(), 0.0);
        for (std::size_t sample = 0; sample < population; ++sample)
        {
          const double* position = model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * 3 * model.ne;
          std::copy_n(position, 3 * model.ne, coordinates.data() + sample * 3 * model.ne);
          weights[sample] = .17 * (static_cast<int>(sample % 3) - 1);
          cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(position, model.ne));
          const auto reference = cpu_score_executor.evaluate(*cpu_score_workspace);
          for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
            expected[parameter] += weights[sample] * reference.parameter_score[parameter];
        }
        auto run_tiles = [&]() {
          tiled_lane->beginScoreContraction(1, model.p.version());
          for (std::size_t offset = 0; offset < population; offset += 4)
            tiled_lane->accumulateWeightedScoreTile(coordinates.data() + offset * 3 * model.ne,
                weights.data() + offset, std::min(std::size_t{4}, population - offset), model.p.version());
          tiled_lane->finishScoreContraction(model.p.version());
          tiled_lane->downloadScoreContraction(0, output.data(), output.size(), model.p.version());
        };
        run_tiles();
        int ordinal = -1;
#ifdef QMC_CUDA2HIP
        checkRuntime(hipGetDevice(&ordinal), "weighted score tile metrics device");
#else
        checkRuntime(cudaGetDevice(&ordinal), "weighted score tile metrics device");
#endif
        auto& metrics = deviceMemoryMetrics(ordinal);
        const auto before = metrics.snapshot();
        const auto before_tiles = tiled_lane->weightedScoreTiles();
        const auto before_configurations = tiled_lane->weightedScoreConfigurations();
        run_tiles();
        const auto after = metrics.snapshot();
        if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
            tiled_lane->deviceWorkspaceBytes() != fixed_bytes ||
            tiled_lane->weightedScoreTiles() - before_tiles != (population + 3) / 4 ||
            tiled_lane->weightedScoreConfigurations() - before_configurations != population)
          throw std::runtime_error("Weighted score tiles changed prepared capacity or execution counts");
        for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
        {
          near(output[parameter].real(), expected[parameter], "weighted score full/tail CPU parity");
          near(output[parameter].imag(), 0, "weighted real score tile imaginary response");
        }
      }
      // A single resident direction serves every full/tail tile and algebraic component.
      const auto fixed_host_bytes = tiled_lane->hostWorkspaceBytes();
      const auto imaginary_direction = [&]() {
        auto result = product_direction;
        for (std::size_t parameter = 0; parameter < result.size(); ++parameter)
          result[parameter] *= parameter % 2 == 0 ? -.3 : .2;
        return result;
      }();
      double maximum_jvp_error = 0;
      std::size_t measured_direction_uploads = 0;
      for (const auto population : {std::size_t{1}, std::size_t{4}, std::size_t{5}, std::size_t{7},
                                    std::size_t{4}, std::size_t{1}, std::size_t{3}, std::size_t{4}})
      {
        std::vector<double> coordinates(population * 3 * model.ne);
        std::vector<Complex> reference(population), products(population);
        for (std::size_t sample = 0; sample < population; ++sample)
        {
          const auto* position = model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * 3 * model.ne;
          std::copy_n(position, 3 * model.ne, coordinates.data() + sample * 3 * model.ne);
          cpu_score_workspace->setPositions(pf::GeometryPositionView::interleaved(position, model.ne));
          const auto response = cpu_score_executor.evaluate(*cpu_score_workspace);
          for (std::size_t parameter = 0; parameter < product_direction.size(); ++parameter)
            reference[sample] += response.parameter_score[parameter] *
                Complex(product_direction[parameter], imaginary_direction[parameter]);
        }
        const auto before = tiled_lane->transferSnapshot();
        const auto before_products = tiled_lane->numericalProducts();
        for (unsigned component = 0; component < 2; ++component)
        {
          const auto before_direction = tiled_lane->transferSnapshot();
          tiled_lane->prepareScoreDirection(component == 0 ? product_direction : imaginary_direction, model.p.version());
          const auto after_direction = tiled_lane->transferSnapshot();
          if (after_direction.upload_calls - before_direction.upload_calls != 1 ||
              after_direction.uploaded_bytes - before_direction.uploaded_bytes != product_direction.size() * sizeof(double))
            throw std::runtime_error("Prepared JVP did not upload exactly one canonical direction");
          ++measured_direction_uploads;
          for (std::size_t offset = 0; offset < population; offset += 4)
          {
            const auto count = std::min(std::size_t{4}, population - offset);
            const auto& tile = tiled_lane->scoreJvpPreparedTile(
                coordinates.data() + offset * 3 * model.ne, count, model.p.version());
            if (tile.size() != count)
              throw std::runtime_error("Prepared JVP exposed capacity instead of live tail count");
            for (std::size_t sample = 0; sample < count; ++sample)
              products[offset + sample] += (component == 0 ? Complex(1, 0) : Complex(0, 1)) * tile[sample];
          }
        }
        const auto after = tiled_lane->transferSnapshot();
        if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
            tiled_lane->deviceWorkspaceBytes() != fixed_bytes || tiled_lane->hostWorkspaceBytes() != fixed_host_bytes ||
            tiled_lane->numericalProducts() - before_products != 2 * ((population + 3) / 4))
          throw std::runtime_error("Prepared JVP tiles changed allocation, accounting or execution counts");
        for (std::size_t sample = 0; sample < population; ++sample)
        {
          maximum_jvp_error = std::max(maximum_jvp_error, std::abs(products[sample] - reference[sample]));
          near(products[sample].real(), reference[sample].real(), "tiled JVP real CPU dot parity");
          near(products[sample].imag(), reference[sample].imag(), "tiled JVP imaginary CPU dot parity");
        }
      }

      // Failure and stale calls publish no scalar tail; recovery reuses the direction.
      tiled_lane->prepareScoreDirection(product_direction, model.p.version());
      const auto& ephemeral_products = tiled_lane->scoreJvpPreparedTile(model.cfg.electrons.x.data(), 1, model.p.version());
      auto expect_failed_jvp = [&](const double* positions, std::size_t count, std::size_t version) {
        bool rejected = false;
        try { tiled_lane->scoreJvpPreparedTile(positions, count, version); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected || !ephemeral_products.empty())
          throw std::runtime_error("Failed JVP tile retained published products");
      };
      expect_failed_jvp(model.cfg.electrons.x.data(), 0, model.p.version());
      expect_failed_jvp(model.cfg.electrons.x.data(), 5, model.p.version());
      expect_failed_jvp(model.cfg.electrons.x.data(), 1, model.p.version() + 1);
      std::vector<double> invalid_jvp_tile(2 * 3 * model.ne);
      std::copy_n(model.cfg.electrons.x.data(), 3 * model.ne, invalid_jvp_tile.data());
      std::copy_n(invalid_product_positions.data(), 3 * model.ne, invalid_jvp_tile.data() + 3 * model.ne);
      expect_failed_jvp(invalid_jvp_tile.data(), 2, model.p.version());
      tiled_lane->prepareScoreDirection(product_direction, model.p.version());
      near(tiled_lane->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()),
           single_lane->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
           "prepared singleton JVP recovery");
      std::cout << "PASS prepared score JVP tiles: populations1/4/5/7, tails4/1/3/4, signed complex CPU dot parity, "
                   "exact owned bytes, zero warmed allocations, atomic failure/stale recovery; max_abs_error="
                << maximum_jvp_error << " measured_direction_uploads=" << measured_direction_uploads
                << " transfer_counter_scope=device_wide\n";

      // Ordinary roots must reset the larger value tape to one live configuration.
      cpu_score_workspace->setPositions(first_position);
      const auto single_reference = cpu_score_executor.evaluate(*cpu_score_workspace);
      const auto& single_score = tiled_lane->score(model.cfg.electrons.x.data(), model.p.version());
      for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
        near(single_score[parameter], single_reference.parameter_score[parameter], "single score after weighted tiles");
      double jvp_reference = 0;
      for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
        jvp_reference += single_reference.parameter_score[parameter] * product_direction[parameter];
      near(tiled_lane->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
           jvp_reference, "single JVP after weighted tiles");
      const auto channel_reference = cpu_score_executor.evaluateWeightedChannelScore(*cpu_score_workspace,
          channel_coefficients.data(), model.ndet);
      const auto& single_channel = tiled_lane->channelScore(model.cfg.electrons.x.data(),
          channel_coefficients.data(), model.ndet, model.p.version());
      for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
        near(single_channel[parameter], channel_reference.parameter_score[parameter], "single channel after weighted tiles");
      kinetic_workspace->setPositions(first_position);
      const auto kinetic_reference = kinetic_cpu.evaluate(*kinetic_workspace, external_contiguous.data(),
          3 * model.ne, inverse_masses.data(), model.ne);
      const auto& single_kinetic = tiled_lane->kinetic(model.cfg.electrons.x.data(), external_contiguous.data(),
          inverse_masses.data(), model.p.version());
      for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
        near(single_kinetic[parameter], kinetic_reference.kinetic_parameter_response[parameter],
             "single kinetic drift extent after weighted tiles");

      auto expect_missing = [&]() {
        bool rejected = false;
        try { tiled_lane->downloadScoreContraction(0, output.data(), output.size(), model.p.version()); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected)
          throw std::runtime_error("Failed weighted tile retained a completed contraction");
      };
      const double zero_weight = 0;
      tiled_lane->beginScoreContraction(1, model.p.version());
      bool zero_invalid_rejected = false;
      try { tiled_lane->accumulateWeightedScoreTile(invalid_product_positions.data(), &zero_weight, 1, model.p.version()); }
      catch (const std::exception&) { zero_invalid_rejected = true; }
      if (!zero_invalid_rejected)
        throw std::runtime_error("Zero-weight tile bypassed geometry admission");
      expect_missing();
      const double nonfinite_weight = std::numeric_limits<double>::quiet_NaN();
      tiled_lane->beginScoreContraction(1, model.p.version());
      bool nonfinite_rejected = false;
      try { tiled_lane->accumulateWeightedScoreTile(model.cfg.electrons.x.data(), &nonfinite_weight, 1, model.p.version()); }
      catch (const std::exception&) { nonfinite_rejected = true; }
      if (!nonfinite_rejected)
        throw std::runtime_error("Weighted tile accepted a nonfinite coefficient");
      expect_missing();
      tiled_lane->beginScoreContraction(1, model.p.version());
      tiled_lane->accumulateWeightedScoreTile(model.cfg.electrons.x.data(), &zero_weight, 1, model.p.version());
      tiled_lane->finishScoreContraction(model.p.version());
      backend_tiles->publishParameters(accepted_values, model.p.version() + 1,
                                       [&]() { model.p.set_flat_values(accepted_values); });
      expect_missing();
      expect_failed_jvp(model.cfg.electrons.x.data(), 1, model.p.version());
      tiled_lane->prepareScoreDirection(product_direction, model.p.version());
      near(tiled_lane->scoreJvpPrepared(model.cfg.electrons.x.data(), model.p.version()),
           single_lane->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
           "prepared tiled JVP publication rebind");
      tiled_lane->beginScoreContraction(1, model.p.version());
      tiled_lane->accumulateWeightedScoreTile(model.cfg.electrons.x.data(), &zero_weight, 1, model.p.version());
      tiled_lane->finishScoreContraction(model.p.version());
      tiled_lane->downloadScoreContraction(0, output.data(), output.size(), model.p.version());
      for (const auto value : output)
        near(std::abs(value), 0, "weighted tile zero-coefficient recovery after publication");
      std::cout << "PASS true weighted score tiles: signed/zero CPU parity B1/4/5/7, tails4/1/3/4, "
                   "exact tile counts, zero warm allocations, single score/JVP/channel/kinetic reset, "
                   "failure/publication recovery\n";
    }
    std::cout << "PASS actual-device full-model FP64 value/gradient/Laplacian/kinetic/reuse; configurations="
              << references.size() << "; shared-image/value-only/reuse/nodes/tails PASS; "
              << "regular-channel score/kinetic/captured-grid response PASS; production training/Hamiltonian binding NOT_VALIDATED\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
