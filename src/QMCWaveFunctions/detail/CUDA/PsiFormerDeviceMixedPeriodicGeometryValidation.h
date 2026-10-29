//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_PERIODIC_GEOMETRY_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MIXED_PERIODIC_GEOMETRY_VALIDATION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "PsiFormerDeviceInference.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <cstring>
#include <iostream>
#include <optional>

namespace qmcplusplus::psiformer::device
{
namespace mixed_periodic_validation
{
/// Cold process settings are restored; prepared owners must not reread them.
struct Settings
{
  std::vector<std::pair<std::string, std::optional<std::string>>> previous;
  Settings()
  {
    for (const char* key : {"PSIFORMER_GPU_PERIODIC_VALUE_GEOMETRY_COOPERATIVE",
         "PSIFORMER_GPU_PERIODIC_FEATURE_JETS_COOPERATIVE", "PSIFORMER_GPU_VALUE_COOPERATIVE_LU",
         "PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE"})
    {
      const char* value = std::getenv(key);
      previous.emplace_back(key, value ? std::optional<std::string>(value) : std::nullopt);
    }
    setenv("PSIFORMER_GPU_VALUE_COOPERATIVE_LU", "1", 1);
    setenv("PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE", "1", 1);
  }
  ~Settings()
  {
    for (const auto& entry : previous)
      if (entry.second)
        setenv(entry.first.c_str(), entry.second->c_str(), 1);
      else
        unsetenv(entry.first.c_str());
  }
  static void geometry(bool enabled)
  {
    setenv("PSIFORMER_GPU_PERIODIC_VALUE_GEOMETRY_COOPERATIVE", enabled ? "1" : "0", 1);
    setenv("PSIFORMER_GPU_PERIODIC_FEATURE_JETS_COOPERATIVE", enabled ? "1" : "0", 1);
  }
};

/// Canonical names/roles, reduced test-only width/depth and a well-conditioned localized head.
inline pf::Parameters parameters(const ExecutionEnvironment& environment, const ModelShape& shape)
{
  const ModelShape production{shape.spin_up_electrons, shape.spin_down_electrons,
                              shape.nuclei, 16, 256, 4, 4};
  auto profile = initializePsiFormerParameters(production, 17).profile;
  if (environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX)
    profile = PSIFORMER_COMPLEX_PERIODIC_V1;
  const auto canonical = initializePsiFormerParameters(production, 17, profile, environment);
  const PsiFormerExecutionPlan original(production, canonical.layouts, environment);
  std::vector<pf::Layout> layouts;
  std::vector<double> values;
  const auto channels = shape.determinants * shape.electrons();
  for (const auto& tensor : original.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= shape.attention_blocks)
      continue;
    auto extents = tensor.shape;
    switch (tensor.role)
    {
    case ParameterRole::ELECTRON_EMBEDDING_WEIGHT:
      extents = {(environment.boundary == BoundaryCondition::PERIODIC ? 7 : 4) * shape.nuclei + 1,
                 shape.feature_dimension};
      break;
    case ParameterRole::BACKFLOW_UP_WEIGHT:
    case ParameterRole::BACKFLOW_DOWN_WEIGHT:
      extents = {shape.feature_dimension, channels};
      break;
    case ParameterRole::ENVELOPE_PI_UP:
    case ParameterRole::ENVELOPE_PI_DOWN:
    case ParameterRole::ENVELOPE_ZETA_UP:
    case ParameterRole::ENVELOPE_ZETA_DOWN:
      extents = {channels, shape.nuclei};
      break;
    case ParameterRole::CUSP_SAME_ALPHA:
    case ParameterRole::CUSP_OPPOSITE_ALPHA:
      extents = {};
      break;
    case ParameterRole::UPDATE_HIDDEN_BIAS:
    case ParameterRole::UPDATE_OUTPUT_BIAS:
      extents = {shape.feature_dimension};
      break;
    default:
      extents = {shape.feature_dimension, shape.feature_dimension};
      break;
    }
    std::size_t count = 1;
    for (const auto extent : extents)
      count *= extent;
    const auto begin = values.size();
    values.resize(begin + count, 0.);
    if (tensor.role == ParameterRole::CUSP_SAME_ALPHA || tensor.role == ParameterRole::CUSP_OPPOSITE_ALPHA)
      values[begin] = 1.;
    else if (tensor.role == ParameterRole::ELECTRON_EMBEDDING_WEIGHT)
      values[begin] = 1.;
    else if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      std::fill(values.begin() + begin, values.begin() + begin + channels, 1.);
    else if (tensor.role == ParameterRole::ENVELOPE_ZETA_UP || tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN)
      std::fill(values.begin() + begin, values.end(), 4.);
    else if (tensor.role == ParameterRole::ENVELOPE_PI_UP || tensor.role == ParameterRole::ENVELOPE_PI_DOWN)
      for (std::size_t column = 0; column < channels; ++column)
        values[begin + column * shape.nuclei + column % shape.nuclei] = 1.;
    layouts.push_back({tensor.module, tensor.name, extents, begin, values.size()});
  }
  return pf::Parameters(std::move(values), std::move(layouts));
}

struct Output
{
  std::vector<double> phases, jets, ratios;
};

template<class Graph>
Output download(Graph& graph, SpatialJetMode mode, std::size_t ne)
{
  Output result;
  result.phases.resize(1);
  result.jets.resize(mode == SpatialJetMode::VALUE_ONLY ? 1 : spatialJetSpanElements(graph.resultLayout()));
  if (mode == SpatialJetMode::FULL_VGL)
    result.ratios.resize(ne);
  graph.downloadResults(result.phases, result.jets, result.ratios);
  return result;
}

/// Same precision/shape must retain every output bit; mixed accuracy is a separate check.
inline void identical(const Output& first, const Output& second)
{
  const auto check = [](const auto& a, const auto& b) {
    if (a.size() != b.size() ||
        (!a.empty() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double))))
      throw std::runtime_error("Mixed periodic geometry changed same-precision output bits");
    for (const double value : b)
      if (!device_math::isFiniteBinary64(value))
        throw std::runtime_error("Mixed periodic geometry accepted a nonfinite output");
  };
  check(first.phases, second.phases);
  check(first.jets, second.jets);
  check(first.ratios, second.ratios);
}

inline void accuracy(const Output& actual, const Output& reference, SpatialJetMode mode,
                     const SpatialJetLayout& layout, std::size_t ne)
{
  const auto near = [](double a, double b, double tolerance) {
    if (!device_math::isFiniteBinary64(a) || !device_math::isFiniteBinary64(b) ||
        std::abs(a - b) > tolerance * (1. + std::abs(b)))
      throw std::runtime_error("Mixed periodic geometry exceeded the existing mixed core error contract");
  };
  near(actual.phases[0], reference.phases[0], 0.);
  if (mode == SpatialJetMode::VALUE_ONLY)
  {
    near(actual.jets[0], reference.jets[0], 5e-5);
    return;
  }
  near(actual.jets[layout.uncheckedValueOffset(0, 0)], reference.jets[layout.uncheckedValueOffset(0, 0)], 5e-5);
  double kinetic = 0., reference_kinetic = 0.;
  for (std::size_t electron = 0; electron < ne; ++electron)
  {
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      near(actual.jets[layout.uncheckedGradientOffset(0, 3 * electron + dimension, 0)],
           reference.jets[layout.uncheckedGradientOffset(0, 3 * electron + dimension, 0)], 5e-4);
    near(actual.jets[layout.uncheckedLaplacianOffset(0, electron, 0)],
         reference.jets[layout.uncheckedLaplacianOffset(0, electron, 0)], 1e-3);
    near(actual.ratios[electron], reference.ratios[electron], 1e-3);
    kinetic -= .5 * actual.ratios[electron];
    reference_kinetic -= .5 * reference.ratios[electron];
  }
  near(kinetic, reference_kinetic, 1e-3);
}
} // namespace mixed_periodic_validation

/// Inspect each prepared graph, not an OR of public preselection probes.
struct MixedPeriodicGeometryValidationAccess
{
  static void run(compute::Queue<PlatformKind::CUDA>& queue)
  {
    using namespace mixed_periodic_validation;
    Settings settings;
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const ModelShape shape{32, 32, 64, 1, 64, 4, 1};
    auto p = parameters(environment, shape);
    const auto plan = PsiFormerExecutionPlan::fromParameters(p, shape, environment);
    std::vector<double> nuclei(3 * shape.nuclei), positions(3 * shape.electrons());
    for (std::size_t e = 0; e < shape.electrons(); ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const std::size_t grid = d == 0 ? e % 4 : d == 1 ? (e / 4) % 4 : e / 16;
        nuclei[3 * e + d] = 1. + 6. * grid;
        positions[3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d;
      }
    auto image = std::make_shared<PreparedDeviceModel>(plan, p.flat_values(), nuclei, 0);
    AcceleratorBlasHandle blas(queue);
    std::size_t completed = 0;
    for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::FULL_VGL})
    {
      Settings::geometry(false);
      PreparedDeviceMixedInference scalar(image, 1, queue, blas, mode);
      Settings::geometry(true);
      PreparedDeviceMixedInference cooperative(image, 1, queue, blas, mode);
      const auto selected = [&](const PreparedDeviceInference& graph, bool expected) {
        const bool value = graph.usesCooperativePeriodicValueGeometry();
        const bool features = graph.usesCooperativePeriodicFeatureJets();
        if (value != (expected && mode == SpatialJetMode::VALUE_ONLY) ||
            features != (expected && mode == SpatialJetMode::FULL_VGL) ||
            graph.usesCooperativeValueLu() || graph.usesCooperativeOpenFeatures())
          throw std::runtime_error("Mixed periodic prepared selection or unchanged LU/OPEN exclusion differs");
      };
      if (!scalar.approximate_ || !cooperative.approximate_)
        throw std::runtime_error("Well-conditioned mixed fixture did not prepare an approximate graph");
      selected(scalar.reference_, false);
      selected(*scalar.approximate_, false);
      selected(cooperative.reference_, true);
      selected(*cooperative.approximate_, true);
      if (scalar.deviceWorkspaceBytes() != cooperative.deviceWorkspaceBytes())
        throw std::runtime_error("Mixed periodic geometry added a numeric owner");
      Settings::geometry(false); // Existing ON owners must keep their prepared map.
      scalar.evaluate(positions);
      cooperative.evaluate(positions);
      if (scalar.usedFp64Retry() || cooperative.usedFp64Retry() ||
          cooperative.selected_ != cooperative.approximate_.get() ||
          cooperative.statistics().roundTripBytes() == 0)
        throw std::runtime_error("Mixed periodic ordinary fixture did not execute accepted mixed arithmetic");
      identical(download(scalar, mode, shape.electrons()), download(cooperative, mode, shape.electrons()));
      selected(*cooperative.selected_, true);
      const auto before = deviceMemoryMetrics(0).snapshot();
      cooperative.evaluate(positions);
      const auto after = deviceMemoryMetrics(0).snapshot();
      if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
          before.live_bytes != after.live_bytes)
        throw std::runtime_error("Mixed periodic warm execution changed tracked owners");

      // Independent complete FP64 core is an accuracy control, not the bitwise mixed comparison.
      Settings::geometry(true);
      PreparedDeviceInference reference(image, 1, queue, blas, mode);
      reference.excludeCooperativeValueLuForMixedRequest();
      reference.evaluate(positions);
      accuracy(download(cooperative, mode, shape.electrons()), download(reference, mode, shape.electrons()),
               mode, reference.resultLayout(), shape.electrons());

      // Both APIs must reject geometry before retry and invalidate previous publication.
      auto invalid = positions;
      invalid[0] = std::numeric_limits<double>::quiet_NaN();
      bool rejected = false, stale = false;
      try { cooperative.evaluate(invalid); }
      catch (const InferenceStatusError& error) { rejected = error.stage() == 0; }
      try { download(cooperative, mode, shape.electrons()); }
      catch (const std::logic_error&) { stale = true; }
      if (!rejected || !stale || cooperative.selected_)
        throw std::runtime_error("Mixed periodic geometry failure hid a prior published result");
      cooperative.evaluate(positions);
      selected(*cooperative.selected_, true);

      // Conversion can fail before approximate ownership exists; shadow selection is already bound.
      auto conversion = p.flat_values();
      conversion[plan.parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, 0).begin] = 1e40;
      auto conversion_image = image->makeUpdatedImage(conversion, 1);
      PreparedDeviceMixedInference recovery(conversion_image, 1, queue, blas, mode);
      if (recovery.approximate_)
        throw std::runtime_error("Forced parameter conversion hazard prepared a mixed graph");
      selected(recovery.reference_, true);
      recovery.evaluate(positions);
      if (!recovery.usedFp64Retry() || recovery.retryReason().empty() ||
          recovery.selected_ != &recovery.reference_)
        throw std::runtime_error("Conversion recovery did not select a complete FP64 shadow");
      selected(*recovery.selected_, true);
      PreparedDeviceInference conversion_reference(conversion_image, 1, queue, blas, mode);
      conversion_reference.excludeCooperativeValueLuForMixedRequest();
      conversion_reference.evaluate(positions);
      identical(download(recovery, mode, shape.electrons()),
                download(conversion_reference, mode, shape.electrons()));

      // Finite FP32 weights overflow their activation projection; direct replay detaches the adapter.
      auto activation = p.flat_values();
      const auto& embedding = plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
      // Keep one nonzero feature after FP64 normalization, rather than making
      // every feature identical and accidentally constructing a nodal reference.
      for (std::size_t input = 0; input < embedding.shape[0]; ++input)
        activation[embedding.begin + input * embedding.shape[1]] = 3e38;
      auto activation_image = image->makeUpdatedImage(activation, 2);
      auto cache = std::make_shared<PreparedDeviceMixedDenseParameters>(activation_image, queue);
      auto adapter = std::make_shared<PreparedDeviceMixedValueDense>(cache,
          shape.electrons() * reference.resultLayout().plane_count, queue, blas);
      PreparedDeviceInference direct(activation_image, 1, queue, blas, mode);
      direct.setMixedDense(adapter);
      selected(direct, true);
      std::string reason;
      if (!direct.evaluateWithFp64Retry(positions, reason) || reason.empty())
        throw std::runtime_error("Direct periodic mixed graph did not replay activation overflow");
      selected(direct, true);
      PreparedDeviceInference activation_reference(activation_image, 1, queue, blas, mode);
      activation_reference.excludeCooperativeValueLuForMixedRequest();
      activation_reference.evaluate(positions);
      identical(download(direct, mode, shape.electrons()), download(activation_reference, mode, shape.electrons()));
      rejected = false;
      stale = false;
      try { direct.evaluateWithFp64Retry(invalid, reason); }
      catch (const InferenceStatusError& error) { rejected = error.stage() == 0; }
      try { download(direct, mode, shape.electrons()); }
      catch (const std::logic_error&) { stale = true; }
      if (!rejected || !stale)
        throw std::runtime_error("Direct mixed geometry failure was hidden or published stale output");
      if (!direct.evaluateWithFp64Retry(positions, reason))
        throw std::runtime_error("Direct mixed adapter was not restored after geometry rejection");
      selected(direct, true);
      ++completed;
    }
    // The periodic exception must not weaken the independently sticky OPEN/LU policy.
    ExecutionEnvironment open_environment;
    open_environment.boundary = BoundaryCondition::OPEN;
    open_environment.geometry_feature_policy = GeometryFeaturePolicy::OPEN_EUCLIDEAN_V1;
    auto open_parameters = parameters(open_environment, shape);
    const auto open_plan = PsiFormerExecutionPlan::fromParameters(open_parameters, shape, open_environment);
    auto open_image = std::make_shared<PreparedDeviceModel>(open_plan, open_parameters.flat_values(), nuclei, 0);
    PreparedDeviceInference open_graph(open_image, 1, queue, blas, SpatialJetMode::VALUE_ONLY);
    if (!open_graph.usesCooperativeOpenFeatures() || !open_graph.usesCooperativeValueLu())
      throw std::runtime_error("OPEN/LU negative control did not begin on the requested ordinary routes");
    auto open_cache = std::make_shared<PreparedDeviceMixedDenseParameters>(open_image, queue);
    auto open_adapter = std::make_shared<PreparedDeviceMixedValueDense>(open_cache, shape.electrons(), queue, blas);
    open_graph.setMixedDense(open_adapter);
    open_graph.setMixedDense({});
    if (open_graph.usesCooperativeOpenFeatures() || open_graph.usesCooperativeValueLu())
      throw std::runtime_error("Periodic mixed exception weakened sticky OPEN/LU replay exclusion");
    std::cout << "PASS mixed_periodic_geometry Ne64 modes=" << completed
              << " same_precision_bits both_prepared_routes conversion_and_detach_replay"
                 " geometry_stage0 publication warm_owner_bytes\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
