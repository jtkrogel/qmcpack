//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexKineticExecutor.h"
#include "psiformer_value_test_utils.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using C = std::complex<double>;

pf::PsiFormer modelWithPairBias(bool paired = true)
{
  auto original = qmcplusplus::testing::psiformer::value::tinyModel(true, paired);
  auto values = original.p.flat_values();
  auto layouts = original.p.layouts;
  for (std::size_t block = 0; block < original.blocks; ++block)
  {
    const auto layer = block == 0 ? std::string("electron_gnn_layer") : "electron_gnn_layer_" + std::to_string(block);
    for (bool bias : {false, true})
    {
      const auto begin = values.size();
      const auto count = (bias ? 1 : 8) * original.heads;
      values.resize(begin + count, 0);
      layouts.push_back({"periodic/" + layer + "/pair_attention", bias ? "b" : "w",
          bias ? std::vector<std::size_t>{original.heads} : std::vector<std::size_t>{8, original.heads},
          begin, begin + count});
    }
  }
  return {pf::Parameters(std::move(values), std::move(layouts)), std::move(original.cfg),
          original.ndet, original.dim, original.heads, original.blocks};
}

ps::PsiFormerExecutionPlan profilePlan(const pf::PsiFormer& model, ps::PeriodicFeatureTransform transform,
                                     bool two_cells, bool cusp = true, bool paired = true)
{
  auto environment = qmcplusplus::testing::psiformer::value::environment(true, paired);
  environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  auto metadata = std::make_shared<ps::PeriodicProfileMetadata>();
  metadata->cells.primitive = metadata->cells.simulation =
      {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  if (two_cells)
  {
    metadata->cells.primitive.lattice_vectors[0][0] *= 0.5;
    metadata->cells.tiling[0][0] = 2;
  }
  metadata->feature_transform = transform;
  metadata->electron_electron_cusp = cusp;
  for (std::size_t ion = 0; ion < model.cfg.nuclei.shape[0]; ++ion)
  {
    metadata->primitive_motif.push_back({model.cfg.nuclei.x[3 * ion], model.cfg.nuclei.x[3 * ion + 1],
                                        model.cfg.nuclei.x[3 * ion + 2]});
    metadata->motif_species.push_back(static_cast<int>(ion));
  }
  for (std::size_t column = 0; column < model.ne; ++column)
    metadata->occupied_columns.push_back({{}, column < model.cfg.nup ? column : column - model.cfg.nup,
                                          column < model.cfg.nup ? 0 : 1, 1});
  environment.periodic_profile = std::move(metadata);
  return ps::PsiFormerExecutionPlan::fromParameters(model.p,
      {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet, model.dim, model.heads, model.blocks},
      environment);
}

C amplitude(const pf::DirectComplexValueResult& result)
{
  return result.amplitude.phase * std::exp(result.amplitude.log_abs);
}
}

TEST_CASE("Periodic pair bias transmits simulation information beyond primitive tokens",
          "[wavefunction][psiformer][periodic_profile]")
{
  auto model = modelWithPairBias();
  const auto plan = profilePlan(model, ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1, true);
  pf::DirectValueExecutor trunk(model, plan, pf::DirectValuePreparation::REAL_TRUNK_ONLY);
  auto work = trunk.makeTrunkWorkspace();
  auto positions = model.cfg.configuration(0).x;
  auto shifted = positions;
  shifted[0] += 4;
  const auto features = [&](const auto& coordinates) {
    work->setPositions(pf::GeometryPositionView::interleaved(coordinates.data(), model.ne));
    const auto result = trunk.evaluateTrunk(*work);
    return std::vector<double>(result.features, result.features + model.ne * model.dim);
  };
  const auto base = features(positions), image = features(shifted);
  for (std::size_t element = 0; element < base.size(); ++element)
    CHECK(base[element] == Catch::Approx(image[element]).margin(1e-11));
  auto parameters = model.p.flat_values();
  for (std::size_t block = 0; block < model.blocks; ++block)
    parameters[plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, block).begin] = 0.7;
  model.p.set_flat_values(parameters);
  const auto informed = features(positions), informed_image = features(shifted);
  double difference = 0;
  for (std::size_t element = 0; element < informed.size(); ++element)
    difference = std::max(difference, std::abs(informed[element] - informed_image[element]));
  CHECK(difference > 1e-6);
}

TEST_CASE("Periodic pair value and spatial products share the same feature contract",
          "[wavefunction][psiformer][periodic_profile]")
{
  for (auto transform : {ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1,
                         ps::PeriodicFeatureTransform::LOG_COMPLEMENT_V1})
  {
    auto model = modelWithPairBias();
    const auto plan = profilePlan(model, transform, true);
    auto parameters = model.p.flat_values();
    for (std::size_t block = 0; block < model.blocks; ++block)
    {
      const auto& weight = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, block);
      const auto& bias = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_BIAS, block);
      for (std::size_t p = 0; p < weight.size(); ++p)
        parameters[weight.begin + p] = 0.03 * std::sin(static_cast<double>(p + 1));
      for (std::size_t h = 0; h < model.heads; ++h)
        parameters[bias.begin + h] = 0.02 * (h + 1);
    }
    model.p.set_flat_values(parameters);
    pf::DirectComplexValueExecutor value(model, plan);
    pf::DirectComplexSpatialExecutor spatial(model, plan);
    auto v = value.makeWorkspace();
    auto s = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    const auto positions = model.cfg.configuration(0).x;
    const auto view = pf::GeometryPositionView::interleaved(positions.data(), model.ne);
    v->setPositions(view);
    s->setPositions(view);
    const auto result = spatial.evaluateFull(*s);
    const C center = amplitude(value.evaluate(*v));
    CHECK(result.amplitude.log_abs == Catch::Approx(std::log(std::abs(center))).margin(1e-11));
    const auto fingerprint = s->storageFingerprint();
    constexpr double step = 1e-4;
    for (std::size_t electron = 0; electron < model.ne; ++electron)
    {
      C laplacian{};
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto lane = 3 * electron + d;
        v->setPosition(electron, d, positions[lane] + step);
        const C plus = amplitude(value.evaluate(*v));
        v->setPosition(electron, d, positions[lane] - step);
        const C minus = amplitude(value.evaluate(*v));
        v->setPosition(electron, d, positions[lane]);
        CHECK(std::abs((plus - minus) / (2 * step * center) - result.gradient[lane]) < 2e-6);
        laplacian += (plus - 2.0 * center + minus) / (step * step * center);
      }
      CHECK(std::abs(laplacian - result.laplacian_ratio[electron]) < 2e-4);
    }
    CHECK(s->storageFingerprint() == fingerprint);
  }
}

TEST_CASE("Periodic pair parameter scores and affine rows match full complex differences",
          "[wavefunction][psiformer][periodic_profile]")
{
  auto model = modelWithPairBias();
  const auto plan = profilePlan(model, ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1, true);
  auto parameters = model.p.flat_values();
  for (std::size_t block = 0; block < model.blocks; ++block)
  {
    const auto& weight = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, block);
    for (std::size_t p = 0; p < weight.size(); ++p)
      parameters[weight.begin + p] = 0.03 * std::sin(static_cast<double>(p + 1));
  }
  model.p.set_flat_values(parameters);
  pf::DirectComplexScoreExecutor score(model, plan);
  pf::DirectComplexValueExecutor value(model, plan);
  auto s = score.makeWorkspace();
  auto v = value.makeWorkspace();
  const auto positions = model.cfg.configuration(0).x;
  const auto view = pf::GeometryPositionView::interleaved(positions.data(), model.ne);
  s->setPositions(view);
  v->setPositions(view);
  struct Observer : pf::DirectAffineObservationSink
  {
    const ps::PsiFormerExecutionPlan& plan;
    std::vector<double> reconstructed;
    std::size_t calls = 0;
    const double* activations = nullptr;
    const double* sensitivities = nullptr;
    explicit Observer(const ps::PsiFormerExecutionPlan& p) : plan(p), reconstructed(p.parameterCount()) {}
    void observe(const pf::DirectAffineObservation& observation) override
    {
      if (observation.role != ps::ParameterRole::PAIR_ATTENTION_WEIGHT)
        return;
      ++calls;
      const auto& shape = plan.modelShape();
      CHECK(observation.row_count == shape.electrons() * (shape.electrons() - 1));
      CHECK(observation.input_width == 8);
      CHECK(observation.output_width == shape.attention_heads);
      if (activations)
      {
        CHECK(activations == observation.activations);
        CHECK(sensitivities == observation.sensitivities);
      }
      activations = observation.activations;
      sensitivities = observation.sensitivities;
      const auto& weight = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, observation.attention_block);
      const auto& bias = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_BIAS, observation.attention_block);
      for (std::size_t row = 0; row < observation.row_count; ++row)
        for (std::size_t head = 0; head < shape.attention_heads; ++head)
        {
          const auto delta = observation.sensitivities[row * shape.attention_heads + head];
          reconstructed[bias.begin + head] += delta;
          for (std::size_t f = 0; f < 8; ++f)
            reconstructed[weight.begin + f * shape.attention_heads + head] +=
                observation.activations[8 * row + f] * delta;
        }
    }
  } real(plan), imaginary(plan);
  const auto evaluated = score.evaluate(*s, &real, &imaginary);
  const std::vector<C> analytic(evaluated.parameter_score, evaluated.parameter_score + plan.parameterCount());
  CHECK(real.calls == model.blocks);
  CHECK(imaginary.calls == model.blocks);
  for (std::size_t block = 0; block < model.blocks; ++block)
    for (auto role : {ps::ParameterRole::PAIR_ATTENTION_WEIGHT, ps::ParameterRole::PAIR_ATTENTION_BIAS})
    {
      const auto& tensor = plan.parameter(role, block);
      for (std::size_t p = tensor.begin; p < tensor.end; ++p)
        CHECK(std::abs(analytic[p] - C{real.reconstructed[p], imaginary.reconstructed[p]}) < 1e-12);
    }
  const C center = amplitude(value.evaluate(*v));
  const auto fingerprint = s->storageFingerprint();
  constexpr double step = 2e-6;
  for (std::size_t p = 0; p < parameters.size(); ++p)
  {
    model.p.set_flat_value(p, parameters[p] + step);
    const C plus = amplitude(value.evaluate(*v));
    model.p.set_flat_value(p, parameters[p] - step);
    const C minus = amplitude(value.evaluate(*v));
    model.p.set_flat_value(p, parameters[p]);
    CHECK(std::abs((plus - minus) / (2 * step * center) - analytic[p]) < 3e-6);
  }
  score.evaluate(*s);
  CHECK(s->storageFingerprint() == fingerprint);
}

TEST_CASE("Periodic pair kinetic parameter responses include full geometric jets",
          "[wavefunction][psiformer][periodic_profile]")
{
  const std::array<double, 3> outer{0.11, -0.07, 0.03};
  const std::vector<double> masses{1, 0.7, 1.2, 0.9};
  for (auto transform : {ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1,
                         ps::PeriodicFeatureTransform::LOG_COMPLEMENT_V1})
  {
    auto model = modelWithPairBias();
    const auto plan = profilePlan(model, transform, true, false);
    auto parameters = model.p.flat_values();
    for (std::size_t block = 0; block < model.blocks; ++block)
    {
      const auto& tensor = plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, block);
      for (std::size_t p = 0; p < tensor.size(); ++p)
        parameters[tensor.begin + p] = 0.04 * std::cos(static_cast<double>(p + 1));
    }
    model.p.set_flat_values(parameters);
    pf::DirectComplexKineticExecutor kinetic(model, plan, outer);
    pf::DirectComplexSpatialExecutor spatial(model, plan);
    const auto bytes = kinetic.requiredStorageBytes();
    CHECK_THROWS(kinetic.makeWorkspace(bytes - 1));
    auto k = kinetic.makeWorkspace(bytes);
    auto s = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
    const auto positions = model.cfg.configuration(0).x;
    const auto view = pf::GeometryPositionView::interleaved(positions.data(), model.ne);
    k->setPositions(view);
    s->setPositions(view);
    const auto spatial_kinetic = [&] {
      const auto value = spatial.evaluateFull(*s);
      C total{};
      for (std::size_t electron = 0; electron < model.ne; ++electron)
      {
        C ratio = value.laplacian_log[electron];
        for (std::size_t d = 0; d < 3; ++d)
        {
          const C gradient = value.gradient[3 * electron + d] + C{0, outer[d]};
          ratio += gradient * gradient;
        }
        total -= 0.5 * masses[electron] * ratio;
      }
      return total;
    };
    const auto result = kinetic.evaluate(*k, masses.data(), masses.size());
    CHECK(std::abs(result.kinetic - spatial_kinetic()) < 2e-10);
    const std::vector<C> response(result.parameter_response, result.parameter_response + result.parameter_count);
    const auto fingerprint = k->storageFingerprint();
    constexpr double step = 1e-5;
    for (std::size_t p = 0; p < parameters.size(); ++p)
    {
      model.p.set_flat_value(p, parameters[p] + step);
      const C plus = spatial_kinetic();
      model.p.set_flat_value(p, parameters[p] - step);
      const C minus = spatial_kinetic();
      model.p.set_flat_value(p, parameters[p]);
      const C difference = (plus - minus) / (2 * step);
      CHECK(std::abs(difference - response[p]) < 3e-5 * (1 + std::abs(difference)));
    }
    kinetic.evaluate(*k, masses.data(), masses.size());
    CHECK(k->storageFingerprint() == fingerprint);
  }
}

TEST_CASE("Real Gamma periodic pair products agree with the explicit complex real subspace",
          "[wavefunction][psiformer][periodic_profile]")
{
  auto real = modelWithPairBias(false);
  auto complex = modelWithPairBias(true);
  const auto real_plan = profilePlan(real, ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1, true, true, false);
  const auto complex_plan = profilePlan(complex, ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1, true);
  auto real_parameters = real.p.flat_values();
  for (std::size_t block = 0; block < real.blocks; ++block)
    real_parameters[real_plan.parameter(ps::ParameterRole::PAIR_ATTENTION_WEIGHT, block).begin] = 0.12;
  real.p.set_flat_values(real_parameters);
  auto complex_parameters = complex.p.flat_values();
  for (const auto& tensor : complex_plan.parameterTensors())
    if (tensor.head_component == ps::HeadTensorComponent::IMAG)
      std::fill(complex_parameters.begin() + tensor.begin, complex_parameters.begin() + tensor.end, 0);
  for (const auto& tensor : real_plan.parameterTensors())
  {
    const auto& target = complex_plan.parameter(tensor.role, tensor.attention_block);
    std::copy_n(real_parameters.data() + tensor.begin, tensor.size(), complex_parameters.data() + target.begin);
  }
  complex.p.set_flat_values(complex_parameters);
  pf::DirectValueExecutor value(real, real_plan);
  pf::DirectSpatialExecutor spatial(real, value, real_plan);
  pf::DirectScoreExecutor score(real, real_plan);
  pf::DirectKineticExecutor kinetic(real, real_plan);
  pf::DirectComplexKineticExecutor complex_kinetic(complex, complex_plan);
  pf::DirectComplexScoreExecutor complex_score(complex, complex_plan);
  auto v = value.makeWorkspace();
  auto s = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  auto o = score.makeWorkspace();
  auto k = kinetic.makeWorkspace();
  auto ck = complex_kinetic.makeWorkspace(complex_kinetic.requiredStorageBytes());
  auto co = complex_score.makeWorkspace();
  const auto positions = real.cfg.configuration(0).x;
  const auto view = pf::GeometryPositionView::interleaved(positions.data(), real.ne);
  v->setPositions(view); s->setPositions(view); o->setPositions(view); k->setPositions(view);
  ck->setPositions(view); co->setPositions(view);
  const auto value_result = value.evaluate(*v);
  const auto spatial_result = spatial.evaluateFull(*s);
  const auto score_result = score.evaluate(*o);
  const auto kinetic_result = kinetic.evaluate(*k);
  const std::vector<double> masses(real.ne, 1);
  const auto ck_result = complex_kinetic.evaluate(*ck, masses.data(), masses.size());
  const auto co_result = complex_score.evaluate(*co);
  CHECK(value_result.logabs == Catch::Approx(spatial_result.logabs).margin(1e-11));
  CHECK(value_result.logabs == Catch::Approx(score_result.logabs).margin(1e-11));
  CHECK(value_result.logabs == Catch::Approx(kinetic_result.logabs).margin(1e-11));
  CHECK(value_result.logabs == Catch::Approx(ck_result.amplitude.log_abs).margin(1e-11));
  for (std::size_t lane = 0; lane < 3 * real.ne; ++lane)
    CHECK(spatial_result.gradient[lane] == Catch::Approx(kinetic_result.gradient[lane]).margin(1e-10));
  for (const auto& tensor : real_plan.parameterTensors())
  {
    const auto& paired = complex_plan.parameter(tensor.role, tensor.attention_block);
    for (std::size_t p = 0; p < tensor.size(); ++p)
    {
      CHECK(score_result.parameter_score[tensor.begin + p] ==
            Catch::Approx(kinetic_result.parameter_score[tensor.begin + p]).margin(1e-10));
      CHECK(std::abs(co_result.parameter_score[paired.begin + p] -
                     C{score_result.parameter_score[tensor.begin + p]}) < 2e-9);
      CHECK(std::abs(ck_result.parameter_response[paired.begin + p] -
                     C{kinetic_result.kinetic_parameter_response[tensor.begin + p]}) < 2e-8);
    }
  }
}
