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
 * accepted. ECP potential/quadrature and parameter reverse are not claimed here.
 */

#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "PsiFormerDeviceInference.h"
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
                                  "[--cpu-only] [--shape DETERMINANTS WIDTH HEADS BLOCKS]");

    // Explicit dimensions permit compact generated fixtures without changing
    // the established DeepQMC profile used by existing invocation scripts.
    bool cpu_only = false;
    std::size_t determinants = 16, width = 256, heads = 4, blocks = 4;
    bool shape_supplied = false;
    for (int argument = 4; argument < argc; ++argument)
    {
      const std::string option(argv[argument]);
      if (option == "--cpu-only" && !cpu_only)
        cpu_only = true;
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
    std::cout << "PASS actual-device full-model FP64 value/gradient/Laplacian/kinetic/reuse; configurations="
              << references.size() << "; shared-image/value-only/reuse/nodes/tails PASS; "
              << "production dispatch and parameter reverse NOT_VALIDATED\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
