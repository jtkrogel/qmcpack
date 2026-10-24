//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_FEATURE_JETS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_FEATURE_JETS_VALIDATION_H

#include "PsiFormerDeviceInference.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>

namespace qmcplusplus::psiformer::device
{
/// One bounded fixture list, not a Cartesian matrix of model/target/batch shapes.
inline void validatePeriodicFeatureJets(compute::Queue<PlatformKind::CUDA>& queue)
{
  const auto cell = periodic_geometry::makeCell({{{9., .3, .1}, {0., 10., .2}, {.1, 0., 11.}}},
                                               {{true, true, true}});
  struct Fixture { std::size_t ne, batch, begin, targets; };
  const Fixture fixtures[] = {{63, 1, 0, 1}, {64, 1, 0, 64}, {100, 1, 0, 1},
                             {100, 3, 97, 3}, {100, 1, 0, 8}, {200, 1, 192, 8}};
  std::size_t cases = 0;
  double maximum_error = 0.;
  for (const auto fixture : fixtures)
  {
    const auto ne = fixture.ne;
    // Three ions suffice to exercise all canonical pair columns and mixed spins.
    const std::size_t ni = 3;
    auto layout = makePeriodicFeatureJetLayout(fixture.batch, ne, ni, ne / 3,
        SpatialJetMode::FULL_VGL, 5, 5 * ne + 11, 7);
    layout.output = makeSpatialTargetJetLayout(fixture.batch, layout.output.element_count,
        ne, fixture.begin, fixture.targets, layout.output.element_count + 13);
    layout.output.configuration_stride += 19;
    validateOpenFeatureJetLayout(layout, true);
    std::vector<double> base(fixture.batch * layout.position_configuration_stride, 17.);
    std::vector<double> nuclei(ni * 7, 23.);
    for (std::size_t b = 0; b < fixture.batch; ++b)
      for (std::size_t e = 0; e < ne; ++e)
        for (std::size_t d = 0; d < 3; ++d)
          base[b * layout.position_configuration_stride + 5 * e + d] =
              .13 + .037 * e + .17 * d + .03 * b;
    for (std::size_t n = 0; n < ni; ++n)
      for (std::size_t d = 0; d < 3; ++d)
        nuclei[7 * n + d] = .09 + .051 * n + .23 * d;
    const auto span = spatialJetSpanElements(layout.output) + 7;
    std::vector<double> initial(span, 97.), old(span), actual(span), host(span);
    std::vector<OpenSpatialStatus> old_codes(fixture.batch), actual_codes(fixture.batch), host_codes(fixture.batch);
    DeviceBuffer<double> positions(base.size()), ions(nuclei.size()), old_output(span), output(span);
    DeviceBuffer<std::size_t> active_device(fixture.batch);
    DeviceBuffer<OpenSpatialStatus> old_status(fixture.batch), status(fixture.batch);
    ions.upload(nuclei.data(), nuclei.size());

    // Bad-index, nonfinite, coalescence, competing failure order and recovery.
    for (int variant = 0; variant < 8; ++variant)
    {
      auto coordinates = base;
      auto ion_coordinates = nuclei;
      std::vector<std::size_t> active(fixture.batch, ne - 1);
      if (variant == 1)
        active.back() = ne;
      if (variant == 2)
        coordinates[5 * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
      if (variant == 3 || variant == 4)
        for (std::size_t d = 0; d < 3; ++d)
          coordinates[d] = nuclei[d];
      if (variant == 4)
        coordinates[5 * (ne - 1)] = std::numeric_limits<double>::infinity();
      if (variant == 5)
      {
        coordinates[5 * (ne - 1)] = std::numeric_limits<double>::max();
        ion_coordinates[0] = -std::numeric_limits<double>::max(); // Finite subtraction overflow.
      }
      if (variant == 6)
        for (std::size_t d = 0; d < 3; ++d)
          coordinates[d] += cell.lattice[0][d];
      positions.upload(coordinates.data(), coordinates.size());
      ions.upload(ion_coordinates.data(), ion_coordinates.size());
      active_device.upload(active.data(), active.size());
      old_output.upload(initial.data(), span);
      output.upload(initial.data(), span);
      host = initial;
      for (std::size_t b = 0; b < fixture.batch; ++b)
        host_codes[b] = open_spatial::buildOpenFeatureJetsForConfiguration(
            layout, coordinates.data(), ion_coordinates.data(), b, active[b], host.data(), &cell);
      const auto before = deviceMemoryMetrics(0).snapshot();
      checkRuntime(launchPeriodicFeatureJets(queue.getNative(), layout, cell, positions.data(),
          ions.data(), active_device.data(), old_output.data(), old_status.data()), "scalar periodic feature jets");
      checkRuntime(launchPeriodicFeatureJetsCooperative(queue.getNative(), layout, cell, positions.data(),
          ions.data(), active_device.data(), output.data(), status.data()), "cooperative periodic feature jets");
      queue.sync();
      const auto after = deviceMemoryMetrics(0).snapshot();
      if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
          before.live_bytes != after.live_bytes)
        throw std::runtime_error("Periodic feature jets changed tracked owners");
      old_output.download(old.data(), span);
      output.download(actual.data(), span);
      old_status.download(old_codes.data(), fixture.batch);
      status.download(actual_codes.data(), fixture.batch);
      if (old_codes != actual_codes || old_codes != host_codes)
        throw std::runtime_error("Periodic feature jet status oracle differs");
      if (std::memcmp(old.data(), actual.data(), span * sizeof(double)))
        throw std::runtime_error("Periodic feature jets GPU bits/padding/failed partial output differ");
      for (std::size_t b = 0; b < fixture.batch; ++b)
        if (actual_codes[b] == OpenSpatialStatus::REGULAR)
          for (std::size_t plane = 0; plane < layout.output.plane_count; ++plane)
            for (std::size_t e = 0; e < layout.output.element_count; ++e)
            {
              const auto i = layout.output.uncheckedPlaneOffset(b, plane, e);
              const double error = std::abs(actual[i] - host[i]);
              maximum_error = std::max(maximum_error, error);
              if (!device_math::isFiniteBinary64(actual[i]) || error > 3e-12 * (1. + std::abs(host[i])))
                throw std::runtime_error("Periodic feature jets CPU analytic oracle differs");
            }
      ++cases;
    }
  }
  std::cout << "PASS periodic_FULL_feature_jets cases=" << cases << " CPU_max_abs=" << maximum_error
            << " scalar_GPU_full_buffer_bits_exact padding failure_order recovery warm_alloc0\n";
}

/// Access only the existing prefix/raw storage; no production diagnostic API.
struct PeriodicFeatureJetsValidationAccess
{
  static void run(compute::Queue<PlatformKind::CUDA>& queue)
  {
    const char* key = "PSIFORMER_GPU_PERIODIC_FEATURE_JETS_COOPERATIVE";
    std::optional<std::string> previous;
    if (const char* value = std::getenv(key))
      previous = value;
    struct Restore
    {
      const char* key;
      const std::optional<std::string>& value;
      ~Restore() { if (value) setenv(key, value->c_str(), 1); else unsetenv(key); }
    } restore{key, previous};
    unsetenv(key);
    if (configuredPeriodicFeatureJetsCooperative())
      throw std::runtime_error("Periodic feature jets default is not OFF");
    setenv(key, "invalid", 1);
    bool rejected = false;
    try { configuredPeriodicFeatureJetsCooperative(); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Periodic feature jets accepted an invalid selector");

    AcceleratorBlasHandle blas(queue);
    for (const bool paired : {false, true})
    {
      ExecutionEnvironment environment;
      environment.boundary = BoundaryCondition::PERIODIC;
      environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
      environment.lattice_vectors = {{{9., .3, .1}, {0., 10., .2}, {.1, 0., 11.}}};
      environment.periodic_axes = {true, true, true};
      environment.head_representation = paired ? HeadRepresentation::PAIRED_REAL_COMPLEX : HeadRepresentation::REAL;
      environment.amplitude_scalar_domain = paired ? ScalarDomain::COMPLEX : ScalarDomain::REAL;
      const ModelShape shape{32, 32, 2, 16, 256, 4, 4};
      const auto profile = paired ? std::string(PSIFORMER_COMPLEX_PERIODIC_V1) :
          initializePsiFormerParameters(shape, 17).profile;
      const auto initialized = initializePsiFormerParameters(shape, 17, profile, environment);
      const PsiFormerExecutionPlan plan(shape, initialized.layouts, environment);
      const std::vector<double> nuclei{.09, .32, .55, .141, .371, .601};
      auto image = paired ? PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei) :
          std::make_shared<PreparedDeviceModel>(plan, initialized.values, nuclei, 0);
      const TargetVglPreparation preparation{1, std::numeric_limits<std::size_t>::max(),
                                               std::numeric_limits<std::size_t>::max(), 0, false};
      const auto make = [&] {
        return std::unique_ptr<PreparedDeviceInference>(new PreparedDeviceInference(image, 1, queue, blas,
            SpatialJetMode::FULL_VGL, false, false, configuredInferenceStatusPolicy(), preparation, false, paired));
      };
      setenv(key, "0", 1);
      auto scalar = make();
      setenv(key, "1", 1);
      auto candidate = make();
      if (scalar->usesCooperativePeriodicFeatureJets() || !candidate->usesCooperativePeriodicFeatureJets() ||
          scalar->deviceWorkspaceBytes() != candidate->deviceWorkspaceBytes() ||
          scalar->hostWorkspaceBytes() != candidate->hostWorkspaceBytes())
        throw std::runtime_error("Periodic feature jets admission or exact owner cost differs");
      candidate->mixed_open_features_excluded_ = true;
      if (candidate->usesCooperativePeriodicFeatureJets())
        throw std::runtime_error("Periodic feature jets bypassed sticky mixed exclusion");
      candidate->mixed_open_features_excluded_ = false;

      std::vector<double> positions(3 * shape.electrons());
      for (std::size_t e = 0; e < shape.electrons(); ++e)
        for (std::size_t d = 0; d < 3; ++d)
          positions[3 * e + d] = .13 + .037 * e + .17 * d;
      const std::size_t active = 0;
      const auto evaluate = [&](PreparedDeviceInference& graph) {
        graph.positions_.upload(positions.data(), positions.size());
        graph.active_.upload(&active, 1);
        graph.resetStatusAdmission();
        graph.evaluatePreparedTrunk();
        queue.sync();
        std::vector<double> result(spatialJetSpanElements(graph.features_.output));
        graph.raw_.download(result.data(), result.size());
        return result;
      };
      const auto expected = evaluate(*scalar);
      const auto before = deviceMemoryMetrics(0).snapshot();
      const auto actual = evaluate(*candidate);
      const auto after = deviceMemoryMetrics(0).snapshot();
      if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)) ||
          before.allocation_calls != after.allocation_calls || before.live_bytes != after.live_bytes)
        throw std::runtime_error("Periodic feature jet owner prefix bits/allocation differ");
    }
    std::cout << "PASS periodic_feature_jet_owners real paired exact_bytes raw_bits cold_flag sticky_mixed\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
