//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_spatial.cpp
 * @brief Complete complex spatial composition against independent VALUE differences.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using namespace qmcplusplus::testing::psiformer::value;
using C = std::complex<double>;

/// Compare complex components with an explicit absolute tolerance for tiny fixtures.
void close(C actual, C expected, double tolerance = 2e-10)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).margin(tolerance));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).margin(tolerance));
}

/// Materialize only the tiny generated amplitude for independent finite differences.
C amplitude(const pf::DirectComplexValueResult& value)
{
  return value.amplitude.phase * std::exp(value.amplitude.log_abs);
}
} // namespace

TEST_CASE("PsiFormer complete complex spatial jets match VALUE finite differences and ACTIVE lanes",
          "[wavefunction][psiformer][complex_spatial]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexSpatialExecutor spatial(model, execution_plan);
  pf::DirectComplexValueExecutor value(model, execution_plan);
  auto full = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto active = spatial.makeWorkspace(pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
  auto reference = value.makeWorkspace();
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  full->setPositions(view);
  active->setPositions(view);
  reference->setPositions(view);
  const auto result = spatial.evaluateFull(*full);
  const C center = amplitude(value.evaluate(*reference));
  CHECK(result.amplitude.log_abs == Catch::Approx(std::log(std::abs(center))).margin(2e-11));
  close(result.amplitude.phase, center / std::abs(center));
  const auto fingerprint = full->storageFingerprint();
  const auto active_fingerprint = active->storageFingerprint();
  const auto bytes = full->vectorStorageBytes();
  const double step = 2e-4;
  for (std::size_t electron = 0; electron < model.ne; ++electron)
  {
    C lap{};
    const auto selected = spatial.evaluateActive(*active, electron);
    CHECK(selected.laplacian_count == 0);
    for (std::size_t dim = 0; dim < 3; ++dim)
    {
      const std::size_t lane = 3 * electron + dim;
      close(selected.gradient[dim], result.gradient[lane]);
      const double coordinate = positions.x[lane];
      reference->setPosition(electron, dim, coordinate + step);
      const C plus = amplitude(value.evaluate(*reference));
      reference->setPosition(electron, dim, coordinate - step);
      const C minus = amplitude(value.evaluate(*reference));
      reference->setPosition(electron, dim, coordinate);
      close(result.gradient[lane], (plus - minus) / (2 * step * center), 2e-6);
      lap += (plus + minus - 2.0 * center) / (step * step * center);
    }
    close(result.laplacian_ratio[electron], lap, 3e-5);
    C reconstructed = result.laplacian_log[electron];
    for (std::size_t dim = 0; dim < 3; ++dim)
      reconstructed += result.gradient[3 * electron + dim] * result.gradient[3 * electron + dim];
    close(result.laplacian_ratio[electron], reconstructed);
  }
  spatial.evaluateFull(*full);
  CHECK(full->storageFingerprint() == fingerprint);
  CHECK(active->storageFingerprint() == active_fingerprint);
  CHECK(full->vectorStorageBytes() == bytes);
  CHECK(full->vectorStorageBytes() > active->vectorStorageBytes());
  CHECK_THROWS_AS(spatial.evaluateFull(*active), std::invalid_argument);
  CHECK_THROWS(active->result());
  CHECK_THROWS_AS(spatial.evaluateActive(*active, model.ne), std::invalid_argument);
  auto parameters = model.p.flat_values();
  const auto& imaginary = execution_plan.parameter(ps::ParameterRole::BACKFLOW_UP_WEIGHT,
      ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
  parameters[imaginary.begin] += .02;
  model.p.set_flat_values(parameters);
  const auto updated = spatial.evaluateFull(*full);
  CHECK(updated.parameter_version == model.p.version());
  CHECK(updated.parameter_version != result.parameter_version);
  CHECK(updated.evaluation_generation != result.evaluation_generation);
  CHECK(std::abs(amplitude(value.evaluate(*reference)) / center - C{1}) > 1e-8);
  full->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(spatial.evaluateFull(*full));
  CHECK_THROWS(full->result());
  CHECK(full->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
}

TEST_CASE("PsiFormer zero-imaginary spatial owner specializes to real periodic VGL",
          "[wavefunction][psiformer][complex_spatial]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  auto parameters = model.p.flat_values();
  for (auto role : {ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& imaginary = execution_plan.parameter(role, ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG);
    std::fill(parameters.begin() + imaginary.begin, parameters.begin() + imaginary.end, 0);
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexSpatialExecutor complex(model, execution_plan);
  auto workspace = complex.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto real_model = tinyModel(true, false);
  const auto real_plan = plan(real_model, true);
  pf::DirectValueExecutor real_value(real_model, real_plan);
  pf::DirectSpatialExecutor real(real_model, real_value, real_plan);
  auto real_workspace = real.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  workspace->setPositions(view);
  real_workspace->setPositions(view);
  const auto result = complex.evaluateFull(*workspace);
  const auto expected = real.evaluateFull(*real_workspace);
  CHECK(result.amplitude.log_abs == Catch::Approx(expected.logabs).margin(2e-11));
  close(result.amplitude.phase, C{expected.sign});
  for (std::size_t lane = 0; lane < 3 * model.ne; ++lane)
    close(result.gradient[lane], C{expected.gradient[lane]});
  for (std::size_t electron = 0; electron < model.ne; ++electron)
  {
    close(result.laplacian_log[electron], C{expected.lap_log[electron]});
    close(result.laplacian_ratio[electron], C{expected.lap_ratio[electron]});
  }
  pf::DirectComplexSpatialExecutor foreign(model, execution_plan);
  CHECK_THROWS_AS(foreign.evaluateFull(*workspace), std::invalid_argument);
  CHECK_THROWS(workspace->result());
  workspace->setPositions(view);
  for (std::size_t dim = 0; dim < 3; ++dim)
    workspace->setPosition(1, dim, positions.x[dim]);
  CHECK_THROWS(complex.evaluateFull(*workspace));
  CHECK_THROWS(workspace->result());
  CHECK_THROWS_AS(pf::DirectComplexSpatialExecutor(real_model, real_plan), std::invalid_argument);
}

TEST_CASE("PsiFormer split real feature views preserve padded packed and partial-target head algebra",
          "[wavefunction][psiformer][complex_spatial]")
{
  const ps::ComplexOrbitalHeadDescriptor descriptor{2, 1, 1, 2,
      ps::ComplexOrbitalStructure::FULL, ps::ComplexOrbitalPhase::NONE, {}};
  const auto features = ps::makeSpatialTargetJetLayout(2, 4, 2, 1, 1, 7, 40);
  const auto envelopes = ps::makeSpatialTargetJetLayout(2, 4, 2, 1, 1);
  ps::ComplexOrbitalHeadWorkspace packed(descriptor, features, envelopes);
  ps::ComplexOrbitalHeadWorkspace split(descriptor, features, envelopes);
  std::vector<double> input(ps::spatialJetSpanElements(features), 0);
  std::vector<double> values(8), gradients(24), laplacians(8);
  std::vector<double> envelope(ps::spatialJetSpanElements(envelopes), .3), positions(12, .2);
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
    for (std::size_t plane = 0; plane < 5; ++plane)
      for (std::size_t element = 0; element < 4; ++element)
      {
        const double value = .1 + .03 * (configuration * 20 + plane * 4 + element);
        input[features.uncheckedPlaneOffset(configuration, plane, element)] = value;
        if (plane == 0)
          values[configuration * 4 + element] = value;
        else if (plane < 4)
          gradients[(configuration * 3 + plane - 1) * 4 + element] = value;
        else
          laplacians[configuration * 4 + element] = value;
      }
  const double real[] = {.2, -.1, .4, .6}, imaginary[] = {-.3, .1, .2, .5};
  const ps::ComplexOrbitalSpinParameters spin{real, imaginary, nullptr, nullptr, 4, 0};
  const ps::ComplexOrbitalHeadParameters parameters{spin, spin};
  ps::ComplexOrbitalFeatureJetView view{values.data(), gradients.data(), laplacians.data(), 8, 24, 8};
  packed.evaluate(input.data(), input.size(), envelope.data(), envelope.size(), positions.data(), 12, parameters);
  split.evaluate(view, envelope.data(), envelope.size(), positions.data(), 12, parameters);
  for (std::size_t i = 0; i < ps::spatialJetSpanElements(split.layout()); ++i)
    close(split.data()[i], packed.data()[i], 0);
  view.gradient_count -= 1;
  CHECK_THROWS_AS(split.evaluate(view, envelope.data(), envelope.size(), positions.data(), 12, parameters),
                  std::invalid_argument);
  CHECK_THROWS(split.data());
}

TEST_CASE("Complex target streaming preserves complete products with bounded physical storage",
          "[wavefunction][psiformer][complex_spatial][complex_cpu_streamed]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexSpatialExecutor spatial(model, execution_plan);
  auto full = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  const auto positions = model.cfg.configuration(0);
  const auto view = pf::GeometryPositionView::interleaved(positions.x.data(), model.ne);
  const pf::PsiFormerStorageShape shape{model.ne, model.cfg.nuclei.shape[0], model.ndet, model.dim,
      model.heads, execution_plan.parameter(ps::ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0],
      model.blocks, model.p.size()};
  CHECK(full->vectorStorageBytes() ==
        pf::DirectComplexSpatialWorkspace::storageRequirement(shape, pf::DirectSpatialMode::FULL_VGL));

  for (const std::size_t targets : {std::size_t{1}, std::size_t{3}, model.ne})
  {
    const auto bytes = pf::DirectComplexStreamedSpatialWorkspace::storageRequirement(shape, targets);
    CHECK_THROWS_AS(spatial.makeStreamedWorkspace(targets, bytes-1), std::length_error);
    auto stream = spatial.makeStreamedWorkspace(targets, bytes);
    CHECK(stream->vectorStorageBytes() == bytes);
    CHECK(stream->targetCapacity() == targets);
    if (targets < model.ne) CHECK(bytes < full->vectorStorageBytes());
    const auto fingerprint = stream->storageFingerprint();
    const auto compare = [&] {
      const auto reference = spatial.evaluateFull(*full);
      const auto result = spatial.evaluateFull(*stream);
      CHECK(result.gradient_count == 3*model.ne);
      CHECK(result.laplacian_count == model.ne);
      CHECK(result.parameter_version == model.p.version());
      CHECK(result.amplitude.log_abs == Catch::Approx(reference.amplitude.log_abs).margin(2e-11));
      close(result.amplitude.phase, reference.amplitude.phase);
      C kinetic{}, reference_kinetic{};
      for (std::size_t lane = 0; lane < 3*model.ne; ++lane)
        close(result.gradient[lane], reference.gradient[lane]);
      for (std::size_t electron = 0; electron < model.ne; ++electron)
      {
        close(result.laplacian_log[electron], reference.laplacian_log[electron]);
        close(result.laplacian_ratio[electron], reference.laplacian_ratio[electron]);
        kinetic -= .5*result.laplacian_ratio[electron];
        reference_kinetic -= .5*reference.laplacian_ratio[electron];
      }
      close(kinetic, reference_kinetic);
      CHECK(stream->storageFingerprint() == fingerprint);
      CHECK(stream->vectorStorageBytes() == bytes);
    };
    full->setPositions(view);
    stream->setPositions(view);
    compare();
    const auto generation = stream->evaluationGeneration();
    stream->setPosition(1, 2, positions.x[5]+.013);
    full->setPosition(1, 2, positions.x[5]+.013);
    CHECK_THROWS(stream->result());
    compare();
    CHECK(stream->evaluationGeneration() > generation);
    const auto old_version = stream->observedParameterVersion();
    auto values = model.p.flat_values();
    values[execution_plan.parameter(ps::ParameterRole::BACKFLOW_UP_WEIGHT,
        ps::NO_ATTENTION_BLOCK, ps::HeadTensorComponent::IMAG).begin] += .003;
    model.p.set_flat_values(values);
    CHECK(old_version != model.p.version());
    compare();
    pf::DirectComplexSpatialExecutor foreign(model, execution_plan);
    CHECK_THROWS_AS(foreign.evaluateFull(*stream), std::invalid_argument);
    CHECK_THROWS(stream->result());
    stream->setPosition(0, 0, std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS(spatial.evaluateFull(*stream));
    CHECK_THROWS(stream->result());
    CHECK(stream->observedParameterVersion() == std::numeric_limits<std::size_t>::max());
  }
  CHECK_THROWS_AS(spatial.makeStreamedWorkspace(0, 1000000), std::invalid_argument);
  CHECK_THROWS_AS(spatial.makeStreamedWorkspace(model.ne+1, 1000000), std::invalid_argument);
  CHECK_THROWS_AS(spatial.makeStreamedWorkspace(1, 0), std::invalid_argument);
  CHECK_THROWS_AS(spatial.makeStreamedWorkspace(1, std::numeric_limits<std::size_t>::max()), std::invalid_argument);
}

TEST_CASE("Complex target lane rejects invalid ranges without retaining a published result",
          "[wavefunction][psiformer][complex_spatial][complex_cpu_streamed]")
{
  auto model = tinyModel(true, true);
  const auto execution_plan = plan(model, true, true);
  pf::DirectComplexSpatialExecutor spatial(model, execution_plan);
  auto target = spatial.makeTargetWorkspace(3, 100000000);
  const auto positions = model.cfg.configuration(0);
  target->setPositions(pf::GeometryPositionView::interleaved(positions.x.data(), model.ne));
  const auto bytes = target->vectorStorageBytes(), fingerprint = target->storageFingerprint();
  for (const auto range : {std::array<std::size_t,2>{0,3}, {3,1}, {1,2}, {0,3}})
  {
    const auto result = spatial.evaluateTarget(*target, range[0], range[1]);
    CHECK(result.gradient_count == 3*range[1]);
    CHECK(result.laplacian_count == range[1]);
    CHECK(target->vectorStorageBytes() == bytes);
    CHECK(target->storageFingerprint() == fingerprint);
  }
  for (const auto range : {std::array<std::size_t,2>{0,0}, {0,4}, {4,1}, {3,2}})
  {
    CHECK_THROWS(spatial.evaluateTarget(*target, range[0], range[1]));
    CHECK_THROWS(target->result());
  }
}
