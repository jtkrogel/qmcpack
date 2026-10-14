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
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "PsiFormerDeviceNonlocalResponse.h"
#include "PsiFormerDeviceCrowdHooks.h"
#include "PsiFormerDeviceBatchBackend.h"

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
std::vector<Reference> cpuReferences(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan)
{
  pf::DirectValueExecutor value(model, plan);
  pf::DirectSpatialExecutor spatial(model, value, plan);
  auto workspace = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  std::vector<Reference> result;
  for (std::size_t index = 0; index < model.cfg.nconfig; ++index)
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
    if (argc < 4)
      throw std::invalid_argument("Usage: psiformer_device_model_validation PARAMETERS CONFIGURATIONS OUTPUT "
                                  "[--cpu-only] [--shape DETERMINANTS WIDTH HEADS BLOCKS] "
                                  "[--score-output CANONICAL_F64] [--jvp-output CONFIGURATION_F64] "
                                  "[--kinetic-output CANONICAL_F64] [--nonlocal-output CANONICAL_F64]");

    // Explicit dimensions permit compact generated fixtures without changing
    // the established DeepQMC profile used by existing invocation scripts.
    bool cpu_only = false;
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
    if (cpu_only && (!score_output.empty() || !jvp_output.empty() || !kinetic_output.empty() || !nonlocal_output.empty()))
      throw std::invalid_argument("Canonical GPU score export is incompatible with --cpu-only");
    pf::PsiFormer model(pf::Parameters(argv[1]), pf::ConfigData(argv[2]),
                        determinants, width, heads, blocks);
    const ModelShape shape{model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0],
                            model.ndet, model.dim, model.heads, model.blocks};
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape);
    const auto references = cpuReferences(model, plan);
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
    bool invalid_direction_rejected = false, stale_jvp_rejected = false;
    try { (void)score_jvp.evaluate(taped, {}); }
    catch (const std::invalid_argument&) { invalid_direction_rejected = true; }
    try { score_jvp.downloadProducts(products); }
    catch (const std::logic_error&) { stale_jvp_rejected = true; }
    if (!invalid_direction_rejected || !stale_jvp_rejected)
      throw std::runtime_error("Failed JVP request exposed stale products");
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
    auto dormant_tile = backend_tiles->prepare(1);
    auto product_session = backend_tiles->prepareProductSession();
    const auto session_bytes = product_session->deviceWorkspaceBytes();
    const auto& session_score = product_session->score(model.cfg.electrons.x.data(), model.p.version());
    cpu_score_workspace->setPositions(first_position);
    const auto session_reference = cpu_score_executor.evaluate(*cpu_score_workspace);
    for (std::size_t parameter = 0; parameter < plan.parameterCount(); ++parameter)
      near(session_score[parameter], session_reference.parameter_score[parameter], "leased score session CPU parity");
    near(product_session->scoreJvp(model.cfg.electrons.x.data(), product_direction, model.p.version()),
         cpu_products[0], "leased explicit JVP CPU parity");
    bool stale_product_rejected = false;
    try { product_session->score(model.cfg.electrons.x.data(), model.p.version() + 1); }
    catch (const std::invalid_argument&) { stale_product_rejected = true; }
    if (!stale_product_rejected || product_session->numericalProducts() != 2)
      throw std::runtime_error("Product session stale request/counter contract failed");
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
