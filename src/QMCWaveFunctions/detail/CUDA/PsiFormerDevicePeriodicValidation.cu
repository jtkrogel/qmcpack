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
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
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
  const auto initialized = initializePsiFormerParameters(production, 1729, DEEPQMC_PSIFORMER_V1, environment);
  const PsiFormerExecutionPlan production_plan(production, initialized.layouts, environment);
  std::vector<pf::Layout> layouts;
  std::vector<double> values;
  for (const auto& tensor : production_plan.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= 2)
      continue;
    auto shape = tensor.shape;
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
      values.push_back(initialized.values[tensor.begin + index] * (rescale ? 4.0 : 1.0));
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
void validate()
{
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
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
  auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, 0);
  PreparedDeviceInference value(image, 3, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  PreparedDeviceInference spatial(image, 3, queue, blas, SpatialJetMode::FULL_VGL, false, true);
  value.evaluate(model.cfg.electrons.x);
  spatial.evaluate(model.cfg.electrons.x);
  std::vector<double> phases(3), logs(3), no_spatial, spatial_phases(3);
  const auto layout = spatial.resultLayout();
  std::vector<double> jets(spatialJetSpanElements(layout)), lap_ratio(12);
  value.downloadResults(phases, logs, no_spatial);
  spatial.downloadResults(spatial_phases, jets, lap_ratio);

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

  PreparedDeviceScoreReverse reverse(image, 3, queue, blas);
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
  std::cout << "PASS Gamma-periodic skew-cell GPU: native initialized small model; value/VGL, image invariance, "
               "canonical VJP, resident JVP/adjoint/finite-difference, complete-drift/mass kinetic and fixed-twist gradient-only responses; P="
            << direction.size() << '\n';
}
} // namespace

/// Return a standard unavailable-runtime status without disguising CPU-only work.
int main()
{
  try
  {
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
    validate();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL periodic validation: " << error.what() << '\n';
    return 1;
  }
}
