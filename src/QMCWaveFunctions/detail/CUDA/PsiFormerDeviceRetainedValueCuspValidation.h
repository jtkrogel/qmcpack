//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_VALUE_CUSP_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_VALUE_CUSP_VALIDATION_H

#include "PsiFormerDeviceRetainedOpenFeatureValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"

namespace qmcplusplus::psiformer::device
{
/// Compare retained consumers while leaving both canonical cusp oracles unchanged.
struct RetainedValueCuspValidationAccess
{
  using Fixture = RetainedOpenFeatureValidationAccess;

  /// Retain exact IEEE bits, including signed zero and defined failed-output lanes.
  static void exactBits(const std::vector<double>& actual, const std::vector<double>& expected)
  {
    if (actual.size() != expected.size() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)) != 0)
      throw std::runtime_error("Retained VALUE cusp changed defined GPU output/response bits");
  }

  /// Host readback is native-test preparation only, not production numerical staging.
  static std::vector<double> download(const double* source, std::size_t count)
  {
    std::vector<double> result(count);
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy(result.data(), source, checkedDeviceBytes<double>(count), hipMemcpyDeviceToHost),
        "retained cusp native fixture readback");
#else
    checkRuntime(cudaMemcpy(result.data(), source, checkedDeviceBytes<double>(count), cudaMemcpyDeviceToHost),
        "retained cusp native fixture readback");
#endif
    return result;
  }

  /// Inspect the existing logical VALUE cusp buffer without introducing a new tape.
  static std::vector<double> cusp(const PreparedDeviceInference& graph)
  {
    std::vector<double> result(spatialJetSpanElements(graph.cusp_.output));
    graph.cusp_jets_.download(result.data(), result.size());
    return result;
  }

  /// Independent native CPU all-P score, using the exact same binary64 parameters.
  static std::vector<double> cpuScore(const std::shared_ptr<const PreparedDeviceModel>& image,
      const std::vector<double>& parameters, const std::vector<double>& positions)
  {
    const auto& shape = image->plan().modelShape();
    const auto n = shape.electrons();
    std::vector<pf::Layout> layouts;
    for (const auto& tensor : image->plan().parameterTensors())
      layouts.push_back({tensor.module, tensor.name, tensor.shape, tensor.begin, tensor.end});
    pf::ConfigData configuration(pf::Tensor({1, n, 3}, positions),
        pf::Tensor({shape.nuclei, 3}, download(image->nuclei(), 3 * shape.nuclei)),
        pf::Tensor({shape.nuclei}, std::vector<double>(shape.nuclei, 1.0)),
        shape.spin_up_electrons, shape.spin_down_electrons, shape.determinants);
    pf::PsiFormer model(pf::Parameters(parameters, std::move(layouts)), configuration,
        shape.determinants, shape.feature_dimension, shape.attention_heads, shape.attention_blocks);
    pf::DirectScoreExecutor cpu(model, image->plan());
    auto workspace = cpu.makeWorkspace();
    workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), n));
    const auto result = cpu.evaluate(*workspace);
    return {result.parameter_score.begin(), result.parameter_score.end()};
  }

  /// Ne63 proves fallback; Ne64/65 exercise the existing ordered leaf and spin tails.
  static void run(compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
  {
    Fixture::RestoreSelector restore("PSIFORMER_GPU_RETAINED_VALUE_CUSP");
    Fixture::RestoreSelector ordinary("PSIFORMER_GPU_CUSP_VALUE_COOPERATIVE");
    int device = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device), "retained cusp native metrics device");
#else
    checkRuntime(cudaGetDevice(&device), "retained cusp native metrics device");
#endif
    auto& metrics = deviceMemoryMetrics(device);
    for (std::size_t n : {std::size_t(63), std::size_t(64), std::size_t(65)})
    {
      std::vector<double> positions;
      const auto image = Fixture::fixture(n, positions);
      const auto p = image->plan().parameterCount();
      const auto parameters = download(image->parameters(), p);
      const auto cpu = cpuScore(image, parameters, positions);
      const auto value = [&](const char* selector) { setenv("PSIFORMER_GPU_RETAINED_VALUE_CUSP", selector, 1); };

      value("invalid");
      const auto before_invalid = metrics.snapshot();
      bool rejected = false;
      try { PreparedDeviceInference invalid(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected || metrics.snapshot().allocation_calls != before_invalid.allocation_calls)
        throw std::runtime_error("Malformed retained cusp selector allocated a graph");

      // Existing ordinary opt-in cannot activate retained score graphs, or vice versa.
      setenv("PSIFORMER_GPU_CUSP_VALUE_COOPERATIVE", "1", 1);
      value("0");
      PreparedDeviceInference reference(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      setenv("PSIFORMER_GPU_CUSP_VALUE_COOPERATIVE", "0", 1);
      value("1");
      PreparedDeviceInference selected(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      PreparedDeviceInference ordinary_only(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY);
      value("0"); // Captured selection must survive caller changes.
      if (reference.usesCooperativeValueCusp() || ordinary_only.usesCooperativeValueCusp() ||
          selected.usesCooperativeValueCusp() != (n >= 64) ||
          selected.deviceWorkspaceBytes() != reference.deviceWorkspaceBytes() ||
          selected.hostWorkspaceBytes() != reference.hostWorkspaceBytes())
        throw std::runtime_error("Retained cusp selection changed cutoff, independence or owners");
      reference.evaluate(positions);
      selected.evaluate(positions);
      exactBits(cusp(selected), cusp(reference));
      Fixture::exact(Fixture::tape(selected), Fixture::tape(reference));
      Fixture::qualifyMatrix(reference);

      PreparedDeviceScoreReverse reverse(image, 1, queue, blas);
      std::vector<double> expected(p), actual(p);
      for (double weight : {0.7, -0.25, 0.0})
      {
        reverse.reverse(reference, {weight});
        reverse.downloadResponse(expected);
        reverse.reverse(selected, {weight});
        reverse.downloadResponse(actual);
        exactBits(actual, expected);
        for (std::size_t i = 0; i < p; ++i)
          if (!device_math::isFiniteBinary64(actual[i]) || !device_math::isFiniteBinary64(cpu[i]) ||
              std::abs(actual[i] - weight * cpu[i]) > 2e-7 * (1 + std::abs(weight * cpu[i])))
            throw std::runtime_error("Retained VALUE cusp exceeds existing Model analytical CPU score bound");
      }
      const auto before_warm = metrics.snapshot();
      selected.evaluate(positions);
      reverse.reverse(selected, {-0.25});
      reverse.downloadResponse(actual);
      if (metrics.snapshot().allocation_calls != before_warm.allocation_calls ||
          metrics.snapshot().free_calls != before_warm.free_calls || metrics.snapshot().live_bytes != before_warm.live_bytes)
        throw std::runtime_error("Retained VALUE cusp warmed consumer changed device owners");

      // New parameter images must update alpha without preserving a previous score.
      auto changed_parameters = parameters;
      changed_parameters[image->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] += .125;
      const auto successor = image->makeUpdatedImage(changed_parameters, image->version() + 1);
      reference.rebindModel(successor);
      selected.rebindModel(successor);
      reverse.rebindModel(successor);
      bool unavailable = false;
      try { reverse.downloadResponse(actual); }
      catch (const std::logic_error&) { unavailable = true; }
      if (!unavailable || selected.value_tape_valid_ || selected.result_valid_)
        throw std::runtime_error("Retained cusp rebind exposed an old root/response");
      reference.evaluate(positions);
      selected.evaluate(positions);
      exactBits(cusp(selected), cusp(reference));
      Fixture::exact(Fixture::tape(selected), Fixture::tape(reference));
      reverse.reverse(reference, {0.7}); reverse.downloadResponse(expected);
      reverse.reverse(selected, {0.7}); reverse.downloadResponse(actual);
      exactBits(actual, expected);

      // Hazardous finite alpha must preserve the canonical GPU status/partial bits.
      changed_parameters[image->plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] = 1e308;
      const auto hazard = successor->makeUpdatedImage(changed_parameters, successor->version() + 1);
      reference.rebindModel(hazard); selected.rebindModel(hazard); reverse.rebindModel(hazard);
      bool reference_failed = false, selected_failed = false;
      try { reference.evaluate(positions); } catch (const std::runtime_error&) { reference_failed = true; }
      try { selected.evaluate(positions); } catch (const std::runtime_error&) { selected_failed = true; }
      OpenSpatialStatus reference_status, selected_status;
      reference.open_status_.download(&reference_status, 1);
      selected.open_status_.download(&selected_status, 1);
      exactBits(cusp(selected), cusp(reference));
      if (reference_failed != selected_failed || reference_status != selected_status)
        throw std::runtime_error("Retained cusp hazard changed canonical status admission");

      const auto recovered = hazard->makeUpdatedImage(parameters, hazard->version() + 1);
      reference.rebindModel(recovered); selected.rebindModel(recovered); reverse.rebindModel(recovered);
      auto bad_positions = positions;
      bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
      bool failed = false;
      try { selected.evaluate(bad_positions); } catch (const std::runtime_error&) { failed = true; }
      unavailable = false;
      try { reverse.reverse(selected, {0.7}); } catch (const std::exception&) { unavailable = true; }
      if (!failed || !unavailable)
        throw std::runtime_error("Retained cusp failure exposed a stale tape");
      reference.evaluate(positions); selected.evaluate(positions);
      exactBits(cusp(selected), cusp(reference));
      Fixture::exact(Fixture::tape(selected), Fixture::tape(reference));
      reverse.reverse(reference, {0.7}); reverse.downloadResponse(expected);
      reverse.reverse(selected, {0.7}); reverse.downloadResponse(actual);
      exactBits(actual, expected);

      value("1");
      PreparedDeviceInference excluded(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
      excluded.excludeCooperativeOpenFeaturesForMixedRequest();
      if (excluded.usesCooperativeValueCusp())
        throw std::runtime_error("Retained cusp ignored the permanent mixed-request exclusion");
      std::cout << "PASS retained VALUE cusp Ne=" << n << " selected=" << (n >= 64)
                << " exact_GPU_cusp_tape_status_allP_response analytical_CPU_bound=2e-7 "
                   "rebind_hazard_recovery owners_unchanged warm_zero_alloc\n";
    }
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
