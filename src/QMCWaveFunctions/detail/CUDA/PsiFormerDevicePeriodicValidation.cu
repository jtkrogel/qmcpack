//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDevicePeriodicValidation.cu
 * @brief External-data-free Gamma-periodic CUDA/HIP value and derivative checks.
 *
 * A small deterministic model uses canonical metadata from the production
 * initializer. Independent native periodic executors supply the references;
 * the original open-only native graph is deliberately not used as an oracle.
 */
#define PSIFORMER_LIBRARY
// Guard the scalar cores' SDK-free include contract before native/runtime headers.
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDerivativeMath.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicFeatures.h"
#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceScoreJvp.h"
#include "PsiFormerDeviceKineticReverse.h"
#include "PsiFormerDeviceTargetKineticValidation.h"

#include <iostream>
#include <numeric>

namespace
{
using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;

/// Reject nonfinite results as well as ordinary CPU/device tolerance failures.
void near(double actual, double expected, const char* quantity, double tolerance = 3e-8)
{
  if (!device_math::isFiniteBinary64(actual) || !device_math::isFiniteBinary64(expected) ||
      std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(quantity) + " periodic CPU/device mismatch: " +
                             std::to_string(actual) + " vs " + std::to_string(expected));
}

/// Reduce production canonical metadata to a small test-only architecture.
/// This does not change or weaken the versioned production initializer profile.
pf::Parameters smallParameters(const ExecutionEnvironment& environment)
{
  const ModelShape production{2, 2, 2, 16, 256, 4, 4};
  const auto profile = environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1
      ? PSIFORMER_PERIODIC_PAIR_V1 : environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX
      ? PSIFORMER_COMPLEX_PERIODIC_V1 : DEEPQMC_PSIFORMER_V1;
  const auto initialized = initializePsiFormerParameters(production, 1729, profile, environment);
  const PsiFormerExecutionPlan production_plan(production, initialized.layouts, environment);
  std::vector<pf::Layout> layouts;
  std::vector<double> values;
  for (const auto& tensor : production_plan.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= 2)
      continue;
    auto shape = tensor.shape;
    if (tensor.role == ParameterRole::PAIR_ATTENTION_WEIGHT || tensor.role == ParameterRole::PAIR_ATTENTION_BIAS)
      shape.back() = 2;
    std::size_t count = 1;
    for (auto& extent : shape)
    {
      if (extent == 256 || extent == 64)
        extent = 16; // width 16, and four determinants times four electrons.
      count *= extent;
    }
    const auto begin = values.size();
    const bool rescale = tensor.role != ParameterRole::ELECTRON_EMBEDDING_WEIGHT &&
        (tensor.shape.size() == 1 || (tensor.shape.size() == 2 && tensor.shape[0] == 256));
    for (std::size_t index = 0; index < count; ++index)
      values.push_back(tensor.role == ParameterRole::PAIR_ATTENTION_WEIGHT || tensor.role == ParameterRole::PAIR_ATTENTION_BIAS
          ? .03 * (static_cast<int>((index + 3 * tensor.attention_block) % 11) - 5)
          : initialized.values[tensor.begin + index] * (rescale ? 4.0 : 1.0));
    layouts.push_back({tensor.module, tensor.name, shape, begin, values.size()});
  }
  return pf::Parameters(std::move(values), std::move(layouts));
}

/// Native, file-free configurations include an independent perturbation and image.
pf::ConfigData configurations(const ExecutionEnvironment& environment)
{
  std::vector<double> positions{0.4, 0.7, 0.9, -0.2, 1.3, -0.7,
                                1.1, -0.6, 0.3, -0.8, -0.4, 1.2};
  const auto base = positions;
  positions.insert(positions.end(), base.begin(), base.end());
  positions[12] += 0.031;
  positions[17] -= 0.017;
  positions.insert(positions.end(), base.begin(), base.end());
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    positions[24 + dimension] += environment.lattice_vectors[0][dimension] -
        2 * environment.lattice_vectors[2][dimension];
  return pf::ConfigData(pf::Tensor({3, 4, 3}, std::move(positions)),
      pf::Tensor({2, 3}, {0.1, 0.2, -0.1, 0.8, -0.3, 0.7}),
      pf::Tensor({2}, {3.0, 1.0}), 2, 2, 4);
}

/// Validate one skew-cell model using real numerical GPU roots, never fallback.
void validate(bool selected_two_cell = false)
{
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  if (selected_two_cell)
  {
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    auto metadata = std::make_shared<PeriodicProfileMetadata>();
    metadata->cells.primitive = metadata->cells.simulation =
        {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
    for (double& entry : metadata->cells.primitive.lattice_vectors[0]) entry *= .5;
    metadata->cells.tiling[0][0] = 2;
    metadata->primitive_motif = {{.1, .2, -.1}, {.8, -.3, .7}};
    metadata->motif_species = {3, 1};
    metadata->occupied_columns = {{{}, 0, 0, 1}, {{}, 1, 0, 1}, {{}, 0, 1, 1}, {{}, 1, 1, 1}};
    environment.periodic_profile = std::move(metadata);
  }
  const ModelShape shape{2, 2, 2, 4, 16, 2, 2};
  pf::PsiFormer model(smallParameters(environment), configurations(environment), 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
  pf::DirectValueExecutor cpu_value(model, plan);
  pf::DirectSpatialExecutor cpu_spatial(model, cpu_value, plan);
  pf::DirectScoreExecutor cpu_score(model, plan);
  pf::DirectKineticExecutor cpu_kinetic(model, plan);
  auto spatial_workspace = cpu_spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto score_workspace = cpu_score.makeWorkspace();
  auto kinetic_workspace = cpu_kinetic.makeWorkspace();

  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  std::shared_ptr<const PreparedDeviceModel> image = selected_two_cell
      ? PreparedDeviceModel::prepareRealTwoCellProfile(plan, model.p.flat_values(), model.cfg.nuclei.x)
      : std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
  PreparedDeviceInference value(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  PreparedDeviceInference spatial(image, 3, queue, blas, SpatialJetMode::FULL_VGL, false, true);
  value.evaluate(model.cfg.electrons.x);
  spatial.evaluate(model.cfg.electrons.x);
  std::vector<double> phases(3), logs(3), no_spatial, spatial_phases(3);
  const auto layout = spatial.resultLayout();
  std::vector<double> jets(spatialJetSpanElements(layout)), lap_ratio(12);
  value.downloadResults(phases, logs, no_spatial);
  spatial.downloadResults(spatial_phases, jets, lap_ratio);

  // Reuse the unchanged mixed error contracts on an independent periodic CPU oracle.
  // Both owners must actually execute mixed arithmetic, not merely pass via FP64 replay.
  if (!selected_two_cell)
  {
    PreparedDeviceMixedInference mixed_value(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY);
    PreparedDeviceMixedInference mixed_spatial(image, 3, queue, blas, SpatialJetMode::FULL_VGL);
    const auto prepared_bytes = mixed_spatial.deviceWorkspaceBytes();
    std::vector<double> mixed_phases(3), mixed_logs(3), unused;
    std::vector<double> mixed_spatial_phases(3), mixed_jets(jets.size()), mixed_ratios(lap_ratio.size());
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      mixed_value.evaluate(model.cfg.electrons.x);
      mixed_value.downloadResults(mixed_phases, mixed_logs, unused);
      mixed_spatial.evaluate(model.cfg.electrons.x);
      mixed_spatial.downloadResults(mixed_spatial_phases, mixed_jets, mixed_ratios);
      if (mixed_value.usedFp64Retry() || mixed_spatial.usedFp64Retry())
        throw std::runtime_error("Periodic mixed oracle did not exercise accepted mixed arithmetic");
      for (std::size_t sample = 0; sample < 3; ++sample)
      {
        spatial_workspace->setPositions(
            pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data() + 12 * sample, 4));
        const auto reference = cpu_spatial.evaluateFull(*spatial_workspace);
        near(mixed_phases[sample], reference.sign, "mixed periodic value phase");
        near(mixed_spatial_phases[sample], reference.sign, "mixed periodic VGL phase");
        near(mixed_logs[sample], reference.logabs, "mixed periodic value", 5e-5);
        near(mixed_jets[layout.uncheckedValueOffset(sample, 0)], reference.logabs,
             "mixed periodic VGL value", 5e-5);
        for (std::size_t lane = 0; lane < 12; ++lane)
          near(mixed_jets[layout.uncheckedGradientOffset(sample, lane, 0)], reference.gradient[lane],
               "mixed periodic gradient", 5e-4);
        double kinetic = 0;
        double reference_kinetic = 0;
        for (std::size_t electron = 0; electron < 4; ++electron)
        {
          near(mixed_jets[layout.uncheckedLaplacianOffset(sample, electron, 0)], reference.lap_log[electron],
               "mixed periodic log Laplacian", 1e-3);
          near(mixed_ratios[4 * sample + electron], reference.lap_ratio[electron],
               "mixed periodic Laplacian ratio", 1e-3);
          kinetic -= .5 * mixed_ratios[4 * sample + electron];
          reference_kinetic -= .5 * reference.lap_ratio[electron];
        }
        near(kinetic, reference_kinetic, "mixed periodic kinetic", 1e-3);
      }
      if (mixed_spatial.deviceWorkspaceBytes() != prepared_bytes)
        throw std::runtime_error("Periodic mixed warm execution changed prepared owner bytes");
    }
    auto unsupported = environment;
    unsupported.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    if (supportsMixedRealInferenceEnvironment(unsupported))
      throw std::runtime_error("Real mixed eligibility admitted a paired head");
    unsupported = environment;
    unsupported.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    if (supportsMixedRealInferenceEnvironment(unsupported))
      throw std::runtime_error("Real mixed eligibility admitted a versioned periodic feature contract");
    unsupported = environment;
    unsupported.periodic_axes[2] = false;
    if (supportsMixedRealInferenceEnvironment(unsupported))
      throw std::runtime_error("Real mixed eligibility admitted a partial-periodic geometry");
    std::cout << "PASS legacy Gamma mixed VALUE/FULL: independent CPU oracle, actual mixed batches, warm reuse\n";
  }

  const std::vector<double> weights{0.4, -0.3, 0.7}, masses{1.0, 0.8, 1.2, 0.9};
  std::vector<double> drift(36), imaginary_drift(36), direction(plan.parameterCount()), cpu_vjp(direction.size(), 0.0),
      cpu_kinetic_response(direction.size(), 0.0), cpu_gradient_response(direction.size(), 0.0), cpu_products(3, 0.0);
  const double cartesian_twist[3]{0.13, -0.07, 0.04};
  for (std::size_t lane = 0; lane < imaginary_drift.size(); ++lane)
    imaginary_drift[lane] = cartesian_twist[lane % 3];
  for (std::size_t index = 0; index < direction.size(); ++index)
    direction[index] = (static_cast<int>(index % 13) - 6) * 0.01 / std::sqrt(direction.size());
  for (std::size_t sample = 0; sample < 3; ++sample)
  {
    const auto position = pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data() + 12 * sample, 4);
    spatial_workspace->setPositions(position);
    const auto reference = cpu_spatial.evaluateFull(*spatial_workspace);
    near(phases[sample], reference.sign, "value phase");
    near(spatial_phases[sample], reference.sign, "spatial phase");
    near(logs[sample], reference.logabs, "value logabs");
    near(jets[layout.uncheckedValueOffset(sample, 0)], reference.logabs, "spatial logabs");
    for (std::size_t lane = 0; lane < 12; ++lane)
    {
      near(jets[layout.uncheckedGradientOffset(sample, lane, 0)], reference.gradient[lane], "gradient");
      drift[12 * sample + lane] = reference.gradient[lane] + 0.02 * (static_cast<int>(lane % 5) - 2);
    }
    for (std::size_t electron = 0; electron < 4; ++electron)
    {
      near(jets[layout.uncheckedLaplacianOffset(sample, electron, 0)], reference.lap_log[electron], "log Laplacian");
      near(lap_ratio[4 * sample + electron], reference.lap_ratio[electron], "Laplacian ratio");
    }
    score_workspace->setPositions(position);
    const auto score = cpu_score.evaluate(*score_workspace);
    kinetic_workspace->setPositions(position);
    const auto kinetic = cpu_kinetic.evaluate(*kinetic_workspace, drift.data() + 12 * sample, 12,
                                              masses.data(), masses.size());
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    {
      cpu_vjp[parameter] += weights[sample] * score.parameter_score[parameter];
      cpu_products[sample] += direction[parameter] * score.parameter_score[parameter];
      cpu_kinetic_response[parameter] += weights[sample] * kinetic.kinetic_parameter_response[parameter];
    }
    // The fixed outer phase is parameter independent. Only its imaginary drift
    // multiplies the parameter response of the real core's spatial gradient.
    const auto gradient_response = cpu_kinetic.evaluateGradientParameterResponse(*kinetic_workspace,
        imaginary_drift.data() + 12 * sample, 12, masses.data(), masses.size());
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      cpu_gradient_response[parameter] += weights[sample] * gradient_response[parameter];
  }
  near(logs[2], logs[0], "integer-image value");
  for (std::size_t plane = 0; plane < layout.plane_count; ++plane)
    near(jets[layout.uncheckedPlaneOffset(2, plane, 0)], jets[layout.uncheckedPlaneOffset(0, plane, 0)], "integer-image VGL");

  // The bounded trace owner consumes ordinary orbital jets in either geometry.
  // Exercise target tails, changing requests, parameter successors and rollback.
  if (!selected_two_cell)
    for (const std::size_t targets : {std::size_t(1), std::size_t(3)})
    {
      TargetVglPreparation preparation{targets, 0, 0, 1024 * 1024};
      const auto cost = estimatePreparedTargetVglCost(plan, 3, preparation);
      preparation.device_workspace_budget_bytes = cost.device_workspace_bytes;
      preparation.scalar_work_budget = cost.charged_scalar_work_units;
      PreparedDeviceInference parallel(image, 3, queue, blas, SpatialJetMode::FULL_VGL, false, false,
          InferenceStatusPolicy::SYNCHRONOUS_DEBUG, preparation);
      PreparedDeviceInference reference(image, 3, queue, blas);
      if (parallel.deviceWorkspaceBytes() != cost.device_workspace_bytes)
        throw std::runtime_error("Periodic parallel traces changed charged owner bytes");
      const auto compare = [&](const std::vector<double>& coordinates) {
        parallel.evaluate(coordinates);
        reference.evaluate(coordinates);
        std::vector<double> pp(3), rp(3), pj(jets.size()), rj(jets.size()), pl(lap_ratio.size()), rl(lap_ratio.size());
        parallel.downloadResults(pp, pj, pl);
        reference.downloadResults(rp, rj, rl);
        for (std::size_t i = 0; i < pp.size(); ++i) near(pp[i], rp[i], "periodic parallel phase");
        for (std::size_t i = 0; i < pj.size(); ++i) near(pj[i], rj[i], "periodic parallel VGL");
        for (std::size_t i = 0; i < pl.size(); ++i) near(pl[i], rl[i], "periodic parallel ratio");
      };
      compare(model.cfg.electrons.x);
      auto shifted = model.cfg.electrons.x;
      shifted[0] += .013;
      compare(shifted);
      compare(model.cfg.electrons.x);
      auto invalid = model.cfg.electrons.x;
      invalid.back() = std::numeric_limits<double>::quiet_NaN();
      bool rejected = false, revoked = false;
      try { parallel.evaluate(invalid); } catch (const std::runtime_error&) { rejected = true; }
      try { parallel.downloadResults(spatial_phases, jets, lap_ratio); }
      catch (const std::logic_error&) { revoked = true; }
      if (!rejected || !revoked) throw std::runtime_error("Periodic parallel failure exposed stale VGL");
      compare(model.cfg.electrons.x);
      auto changed = model.p.flat_values();
      changed.front() += .01;
      const auto updated = image->makeUpdatedImage(changed, image->version() + 1);
      parallel.rebindModel(updated);
      reference.rebindModel(updated);
      compare(model.cfg.electrons.x);
      std::cout << "PASS periodic real parallel K" << targets
                << " full-VGL parity, exact owner bytes, positions/reject/retry/parameter successor\n";
    }

  PreparedDeviceScoreReverse reverse(image, 3, queue, blas);
  if (!selected_two_cell)
    validateTargetKinetic(model, plan, image, queue, blas);
  PreparedDeviceScoreJvp jvp(image, 3, queue, blas);
  PreparedDeviceKineticReverse kinetic(image, 3, queue, blas);
  std::vector<double> response(direction.size()), products(3);
  reverse.reverse(value, weights);
  reverse.downloadResponse(response);
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    near(response[parameter], cpu_vjp[parameter], "score VJP");
  jvp.prepareDirection(direction);
  jvp.evaluatePrepared(value);
  jvp.downloadProducts(products);
  for (std::size_t sample = 0; sample < 3; ++sample)
    near(products[sample], cpu_products[sample], "resident score JVP");
  near(std::inner_product(weights.begin(), weights.end(), products.begin(), 0.0),
       std::inner_product(direction.begin(), direction.end(), response.begin(), 0.0), "JVP/VJP adjoint identity");
  kinetic.evaluate(spatial, weights, masses, drift);
  kinetic.downloadResponse(response);
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    near(response[parameter], cpu_kinetic_response[parameter], "complete-drift/mass kinetic response", 2e-7);

  if (selected_two_cell)
  {
    const auto cost = estimateTargetVglCost(plan, 3, 2);
    PreparedDeviceInference tiled(image, 3, queue, blas, SpatialJetMode::FULL_VGL, false, false,
        InferenceStatusPolicy::SYNCHRONOUS_DEBUG,
        {2, cost.device_workspace_bytes, cost.charged_scalar_work_units});
    tiled.evaluate(model.cfg.electrons.x);
    std::vector<double> tiled_phases(3), tiled_jets(jets.size()), tiled_ratios(lap_ratio.size());
    tiled.downloadResults(tiled_phases, tiled_jets, tiled_ratios);
    for (std::size_t index = 0; index < jets.size(); ++index) near(tiled_jets[index], jets[index], "explicit K2/full VGL");
    for (std::size_t index = 0; index < lap_ratio.size(); ++index) near(tiled_ratios[index], lap_ratio[index], "explicit K2/full ratio");
    for (std::size_t index = 0; index < phases.size(); ++index) near(tiled_phases[index], phases[index], "explicit K2/full phase");
    const auto successor = image->makeUpdatedImage(model.p.flat_values(), image->version() + 1);
    tiled.rebindModel(successor);
    bool unavailable = false;
    try { tiled.downloadResults(tiled_phases, tiled_jets, tiled_ratios); }
    catch (const std::logic_error&) { unavailable = true; }
    if (!unavailable) throw std::runtime_error("Selected two-cell rebind exposed old VGL");
    tiled.evaluate(model.cfg.electrons.x);
    tiled.downloadResults(tiled_phases, tiled_jets, tiled_ratios);
    for (std::size_t index = 0; index < jets.size(); ++index) near(tiled_jets[index], jets[index], "successor K2/full VGL");
  }

  const auto kinetic_bytes = kinetic.deviceWorkspaceBytes();
  kinetic.evaluateGradientResponse(spatial, weights, masses, imaginary_drift);
  kinetic.downloadResponse(response);
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    near(response[parameter], cpu_gradient_response[parameter], "fixed-twist gradient-only response", 2e-7);
  std::fill(imaginary_drift.begin(), imaginary_drift.end(), 0.0);
  kinetic.evaluateGradientResponse(spatial, weights, masses, imaginary_drift);
  kinetic.downloadResponse(response);
  for (double derivative : response)
    near(derivative, 0.0, "zero-drift response has no Laplacian seed", 1e-14);
  if (kinetic.deviceWorkspaceBytes() != kinetic_bytes)
    throw std::runtime_error("Gradient-only response changed prepared device storage");

  // Coalescence is value-defined but does not have an admissible spatial jet.
  // A failed refresh must hide the old snapshot and allow subsequent recovery.
  auto coincident = model.cfg.electrons.x;
  std::copy_n(model.cfg.nuclei.x.data(), 3, coincident.data());
  bool coalescence_rejected = false, stale_result_rejected = false;
  try { spatial.evaluate(coincident); }
  catch (const std::runtime_error&) { coalescence_rejected = true; }
  try { spatial.downloadResults(spatial_phases, jets, lap_ratio); }
  catch (const std::logic_error&) { stale_result_rejected = true; }
  if (!coalescence_rejected || !stale_result_rejected)
    throw std::runtime_error("Periodic coalescence exposed an invalid spatial snapshot");
  spatial.evaluate(model.cfg.electrons.x);
  spatial.downloadResults(spatial_phases, jets, lap_ratio);
  near(jets[layout.uncheckedValueOffset(0, 0)], logs[0], "coalescence recovery");

  // Independently differentiate a scalar value in the same canonical direction.
  const auto accepted = model.p.flat_values();
  auto plus = accepted, minus = accepted;
  constexpr double h = 1e-4;
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
  {
    plus[parameter] += h * direction[parameter];
    minus[parameter] -= h * direction[parameter];
  }
  auto value_workspace = cpu_value.makeWorkspace();
  value_workspace->setPositions(pf::GeometryPositionView::interleaved(model.cfg.electrons.x.data(), 4));
  model.p.set_flat_values(plus);
  const auto plus_log = cpu_value.evaluate(*value_workspace).logabs;
  model.p.set_flat_values(minus);
  const auto minus_log = cpu_value.evaluate(*value_workspace).logabs;
  near(products[0], (plus_log - minus_log) / (2 * h), "directional log-value finite difference", 2e-7);
  std::cout << (selected_two_cell ? "PASS selected two-cell real/log/atom/cusp GPU: " : "PASS Gamma-periodic skew-cell GPU: ")
            << "native initialized small model; value/VGL, image invariance, "
               "canonical VJP, resident JVP/adjoint/finite-difference, complete-drift/mass kinetic and fixed-twist gradient-only responses; P="
            << direction.size() << '\n';
}
} // namespace

/// Select the numerical paired atom-centered profile, retaining legacy fixtures unchanged.
void selectPairedTwoCellValidation(ExecutionEnvironment& environment, bool selected)
{
  if (!selected)
    return;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  auto metadata = std::make_shared<PeriodicProfileMetadata>();
  metadata->cells.primitive = metadata->cells.simulation =
      {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  for (double& entry : metadata->cells.primitive.lattice_vectors[0]) entry *= .5;
  metadata->cells.tiling[0][0] = 2;
  metadata->primitive_motif = {{.1, .2, -.1}, {.8, -.3, .7}};
  metadata->motif_species = {3, 1};
  metadata->occupied_columns = {{{}, 0, 0, 1}, {{}, 1, 0, 1}, {{}, 0, 1, 1}, {{}, 1, 1, 1}};
  environment.periodic_profile = std::move(metadata);
}

#include "PsiFormerDeviceComplexValueValidation.h"
#include "PsiFormerDeviceComplexSpatialValidation.h"
#include "PsiFormerDeviceComplexScoreValidation.h"
#include "PsiFormerDeviceComplexOrbitalVJPValidation.h"
#include "PsiFormerDeviceComplexLUValidation.h"
#include "PsiFormerDeviceComplexKineticValidation.h"
#include "PsiFormerDeviceComplexKineticOwnerValidation.h"
#include "PsiFormerDeviceComplexProductsValidation.h"
#include "PsiFormerDeviceComplexSpatialTargetValidation.h"
#include "PsiFormerDeviceKineticTrunkValidation.h"

/// Exercise physical K ranges on the existing skew-periodic real and paired fixtures.
void validateKineticTrunkPeriodic()
{
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  for (bool paired : {false, true})
  {
    environment.head_representation = paired ? HeadRepresentation::PAIRED_REAL_COMPLEX : HeadRepresentation::REAL;
    environment.amplitude_scalar_domain = paired ? ScalarDomain::COMPLEX : ScalarDomain::REAL;
    pf::PsiFormer model(smallParameters(environment), configurations(environment), 4, 16, 2, 2);
    const ModelShape shape{2, 2, 2, 4, 16, 2, 2};
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
    auto image = paired ? PreparedDeviceModel::preparePairedComplexValue(plan, model.p.flat_values(), model.cfg.nuclei.x) :
        std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle blas(queue);
    KineticTrunkValidationAccess::run(model, image, queue, blas);
  }
}

/// Compare pair logits and their parameter VJP with independent native feature jets.
void validatePeriodicPairAttention()
{
  namespace pg = qmcplusplus::psiformer::periodic_geometry;
  namespace feature = qmcplusplus::psiformer::periodic_feature;
  constexpr std::size_t batches = 2, ne = 3, heads = 2;
  const pf::GeometryBoundary boundary{pf::GeometryBoundaryKind::PERIODIC,
      {{{6.4, .8, -.4}, {.3, 2.8, .5}, {-.1, .6, 3.6}}}, {{true, true, true}}};
  auto primitive = boundary.lattice_vectors;
  for (double& value : primitive[0]) value *= .5;
  const auto cell = pg::makeTwoCell(boundary.lattice_vectors, primitive, boundary.periodic_axes);
  const std::vector<double> positions{.71, .52, -.41, -.27, 1.18, .84, 1.4, -.3, .6,
                                      .74, .52, -.41, -.27, 1.16, .84, 1.4, -.3, .61};
  std::vector<double> weights(8 * heads), bias{.31, -.17};
  for (std::size_t index = 0; index < weights.size(); ++index)
    weights[index] = .07 * (static_cast<int>(index) - 6);
  std::vector<feature::Jet> references(batches * ne * ne);
  for (std::size_t configuration = 0; configuration < batches; ++configuration)
  {
    const std::array<double, 3> nucleus{.12, -.23, .31};
    pf::PsiFormerGeometryCache geometry(ne,
        pf::GeometryPositionView::interleaved(nucleus.data(), 1), boundary);
    geometry.update(pf::GeometryPositionView::interleaved(positions.data() + 3 * ne * configuration, ne));
    for (std::size_t query = 0; query < ne; ++query)
      for (std::size_t key = 0; key < ne; ++key)
        if (query != key)
        {
          const auto first = std::min(query, key), second = std::max(query, key);
          std::size_t pair = second - first - 1;
          for (std::size_t row = 0; row < first; ++row) pair += ne - row - 1;
          references[(configuration * ne + query) * ne + key] = feature::evaluate(
              geometry.electronElectronPairs(), pair, boundary, query < key ? 1 : -1,
              PeriodicFeatureTransform::LOG_COMPLEMENT_V1, (query < 2) == (key < 2) ? 1 : -1);
        }
  }
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  DeviceBuffer<double> device_positions(positions.size()), device_weights(weights.size()), device_bias(bias.size());
  device_positions.upload(positions.data(), positions.size());
  device_weights.upload(weights.data(), weights.size()); device_bias.upload(bias.data(), bias.size());
  const auto forward = makeBatchedAttentionForwardLayout(batches, makeAttentionForwardLayout(ne, heads, 2, 5, 4, 15));
  std::vector<double> probabilities(forward.attentionElements(), 0), adjoints(probabilities.size(), 0),
      observed_probabilities(probabilities.size(), 0);
  std::vector<double> expected_weights(weights.size(), 0), expected_bias(heads, 0);
  for (std::size_t configuration = 0; configuration < batches; ++configuration)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t query = 0; query < ne; ++query)
      {
        double normalizer = 0;
        for (std::size_t key = 0; key < ne; ++key)
        {
          double logit = query == key ? 0 : bias[head];
          if (query != key)
            for (std::size_t f = 0; f < 8; ++f)
              logit += weights[f * heads + head] * references[(configuration * ne + query) * ne + key].value[f];
          const auto offset = forward.attentionOffset(configuration, head, query, key);
          probabilities[offset] = std::exp(logit); normalizer += probabilities[offset];
          adjoints[offset] = .02 * (1 + static_cast<int>(configuration + head + 3 * query) - 2 * static_cast<int>(key));
        }
        double mean_adjoint = 0;
        for (std::size_t key = 0; key < ne; ++key)
        {
          const auto offset = forward.attentionOffset(configuration, head, query, key);
          probabilities[offset] /= normalizer;
          mean_adjoint += probabilities[offset] * adjoints[offset];
        }
        for (std::size_t key = 0; key < ne; ++key)
          if (query != key)
          {
            const auto offset = forward.attentionOffset(configuration, head, query, key);
            const double seed = probabilities[offset] * (adjoints[offset] - mean_adjoint);
            expected_bias[head] += seed;
            for (std::size_t f = 0; f < 8; ++f)
              expected_weights[f * heads + head] += seed * references[(configuration * ne + query) * ne + key].value[f];
          }
      }
  for (auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::ACTIVE, SpatialJetMode::FULL_VGL})
    for (std::size_t begin : {std::size_t(0), std::size_t(2)})
    {
      auto layout = makeSpatialAttentionJetLayout(batches, ne, heads, 2, ne, mode, 5, 4, 15);
      if (mode == SpatialJetMode::FULL_VGL)
      {
        layout.features = withSpatialTargetRange(layout.features, begin, ne - begin);
        layout.attention = withSpatialTargetRange(layout.attention, begin, ne - begin);
      }
      const std::vector<std::size_t> active{begin, 1};
      DeviceBuffer<std::size_t> device_active(active.size()); device_active.upload(active.data(), active.size());
      std::vector<double> zeros(spatialJetSpanElements(layout.features), 0);
      std::vector<double> attention(spatialJetSpanElements(layout.attention), 9.125), logits(attention.size());
      DeviceBuffer<double> query(zeros.size()), key(zeros.size()), output(attention.size()), retained(logits.size());
      DeviceBuffer<device_math::JetMathStatus> status(batches * heads * ne);
      query.upload(zeros.data(), zeros.size()); key.upload(zeros.data(), zeros.size());
      output.upload(attention.data(), attention.size());
      const PeriodicPairAttentionInputs inputs{&cell, device_positions.data(), device_weights.data(),
          device_bias.data(), device_active.data(), 2};
      checkRuntime(launchAttentionJetWeights(queue.getNative(), layout, query.data(), key.data(), output.data(),
          status.data(), retained.data(), &inputs), "pair attention forward");
      queue.sync(); output.download(attention.data(), attention.size()); retained.download(logits.data(), logits.size());
      std::vector<device_math::JetMathStatus> statuses(status.capacity()); status.download(statuses.data(), statuses.size());
      for (const auto value : statuses)
        if (value != device_math::JetMathStatus::REGULAR) throw std::runtime_error("Pair attention returned nonregular jets");
      for (std::size_t configuration = 0; configuration < batches; ++configuration)
        for (std::size_t head = 0; head < heads; ++head)
          for (std::size_t q = 0; q < ne; ++q)
            for (std::size_t k = 0; k < ne; ++k)
            {
              const auto element = head * layout.attention_head_stride + q * layout.attention_row_stride + k;
              const auto& reference = references[(configuration * ne + q) * ne + k];
              double expected = q == k ? 0 : bias[head];
              if (q != k) for (std::size_t f = 0; f < 8; ++f) expected += weights[f * heads + head] * reference.value[f];
              near(logits[layout.attention.uncheckedValueOffset(configuration, element)], expected, "pair logit", 3e-11);
              near(attention[layout.attention.uncheckedValueOffset(configuration, element)],
                   probabilities[forward.attentionOffset(configuration, head, q, k)], "pair softmax", 3e-11);
              observed_probabilities[forward.attentionOffset(configuration, head, q, k)] =
                  attention[layout.attention.uncheckedValueOffset(configuration, element)];
              for (std::size_t lane = 0; lane < layout.attention.gradient_lanes; ++lane)
              {
                const auto target = mode == SpatialJetMode::ACTIVE ? active[configuration] : begin + lane / 3;
                const double sign = q == k ? 0 : target == q ? 1 : target == k ? -1 : 0;
                double gradient = 0;
                for (std::size_t f = 0; f < 8; ++f) gradient += sign * weights[f * heads + head] * reference.gradient[f][lane % 3];
                near(logits[layout.attention.uncheckedGradientOffset(configuration, lane, element)], gradient, "pair logit gradient", 3e-10);
              }
              for (std::size_t target = 0; target < layout.attention.laplacian_lanes; ++target)
              {
                const auto physical = mode == SpatialJetMode::ACTIVE ? active[configuration] : begin + target;
                double laplacian = 0;
                if (q != k && (physical == q || physical == k))
                  for (std::size_t f = 0; f < 8; ++f) laplacian += weights[f * heads + head] * reference.laplacian[f];
                near(logits[layout.attention.uncheckedLaplacianOffset(configuration, target, element)], laplacian,
                     "pair logit Laplacian", 3e-10);
              }
            }
      for (std::size_t configuration = 0; configuration < batches; ++configuration)
        for (std::size_t plane = 0; plane < layout.attention.plane_count; ++plane)
          near(logits[layout.attention.uncheckedPlaneOffset(configuration, plane, 3)], 9.125, "pair padding", 0);
    }
  DeviceBuffer<double> device_probabilities(probabilities.size()), device_adjoints(adjoints.size());
  DeviceBuffer<double> weight_response(weights.size()), bias_response(heads);
  std::vector<double> weight_result(weights.size(), .125), bias_result(heads, -.25);
  device_probabilities.upload(observed_probabilities.data(), observed_probabilities.size());
  device_adjoints.upload(adjoints.data(), adjoints.size());
  weight_response.upload(weight_result.data(), weight_result.size()); bias_response.upload(bias_result.data(), bias_result.size());
  for (int repeat = 0; repeat < 2; ++repeat)
    checkRuntime(launchPeriodicPairAttentionParameterReverse(queue.getNative(), forward, cell, 2,
        device_positions.data(), device_probabilities.data(), device_adjoints.data(), weight_response.data(), bias_response.data()),
        "pair parameter reverse");
  queue.sync(); weight_response.download(weight_result.data(), weight_result.size()); bias_response.download(bias_result.data(), bias_result.size());
  for (std::size_t index = 0; index < weights.size(); ++index)
    near(weight_result[index], .125 + 2 * expected_weights[index], "pair weight VJP", 3e-11);
  for (std::size_t head = 0; head < heads; ++head)
    near(bias_result[head], -.25 + 2 * expected_bias[head], "pair bias VJP", 3e-11);
  std::cout << "PASS selected pair attention logits/jets/softmax and accumulated parameter VJP\n";
}

/// Return a standard unavailable-runtime status without disguising CPU-only work.
int main(int argc, char** argv)
{
  try
  {
    if (argc == 2 && std::string_view(argv[1]) == "--complex-spatial-target-cost-check")
    {
      validateComplexSpatialTargetCosts();
      return 0; // Metadata-only: deliberately precedes runtime discovery/fixtures.
    }
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--complex-value-only" &&
        std::string_view(argv[1]) != "--complex-spatial-only" && std::string_view(argv[1]) != "--complex-score-only" && std::string_view(argv[1]) != "--kinetic-trunk-only" && std::string_view(argv[1]) != "--complex-kinetic-leaves-only" && std::string_view(argv[1]) != "--complex-kinetic-only" &&
        std::string_view(argv[1]) != "--complex-spatial-target-only" && std::string_view(argv[1]) != "--complex-products-only" &&
        std::string_view(argv[1]) != "--periodic-pair-only" && std::string_view(argv[1]) != "--two-cell-model-only" &&
        std::string_view(argv[1]) != "--paired-two-cell-only" && std::string_view(argv[1]) != "--complex-orbital-vjp-only" && std::string_view(argv[1]) != "--complex-lu-only" && std::string_view(argv[1]) != "--complex-lu-growth-only"))
      throw std::invalid_argument("Usage: periodic_validation [--complex-value-only|--complex-spatial-only|--complex-score-only|--complex-orbital-vjp-only|--complex-lu-only|--complex-lu-growth-only|--kinetic-trunk-only|--complex-kinetic-leaves-only|--complex-kinetic-only|--complex-spatial-target-only|--complex-spatial-target-cost-check|--complex-products-only|--periodic-pair-only|--two-cell-model-only|--paired-two-cell-only]");
    int count = 0;
#ifdef QMC_CUDA2HIP
    const auto discovery = hipGetDeviceCount(&count);
#else
    const auto discovery = cudaGetDeviceCount(&count);
#endif
    if (discovery != success || count == 0)
    {
      std::cout << "NOT_RUN: periodic validator requires a usable GPU\n";
      return 77;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--complex-lu-growth-only")
      validateComplexLU(true);
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-lu-only")
      validateComplexLU();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-orbital-vjp-only")
      validateComplexOrbitalVJP();
    else if (argc == 2 && std::string_view(argv[1]) == "--paired-two-cell-only")
    {
      validateComplexSpatial(true);
      validateComplexScore(true);
      validateComplexKineticOwner(true);
      validateComplexProducts(true);
      std::cout << "PASS selected paired two-cell numerical products: atom/log/pair8/cusp; "
                   "value/phase/G/L/score/JVP/kinetic/publication; native admission unchanged\n";
    }
    else if (argc == 2 && std::string_view(argv[1]) == "--two-cell-model-only")
      validate(true);
    else if (argc == 2 && std::string_view(argv[1]) == "--periodic-pair-only")
      validatePeriodicPairAttention();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-spatial-target-only")
      validateComplexSpatialTargets();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-products-only")
      validateComplexProducts();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-kinetic-only")
      validateComplexKineticOwner();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-kinetic-leaves-only")
      validateComplexKineticLeaves();
    else if (argc == 2 && std::string_view(argv[1]) == "--kinetic-trunk-only")
      validateKineticTrunkPeriodic();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-score-only")
      validateComplexScore();
    else if (argc == 2 && std::string_view(argv[1]) == "--complex-spatial-only")
      validateComplexSpatial();
    else if (argc == 2)
      validateComplexValue();
    else
      validate();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL periodic validation: " << error.what() << '\n';
    return 1;
  }
}
