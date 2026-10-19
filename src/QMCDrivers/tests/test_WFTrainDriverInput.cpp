//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"
#include "QMCDrivers/WFTrain/WFTrainOptimizerInput.h"
#include "QMCDrivers/WFTrain/WFTrainInferenceInput.h"

#include <array>
#include <limits>
#include <string>

namespace qmcplusplus
{
namespace
{

/** Parse a complete driver section while its owning XML document is alive. */
WFTrainDriverInput parseTrainingInput(const std::string& xml)
{
  Libxml2Document document;
  REQUIRE(document.parseFromString(xml));
  WFTrainDriverInput input;
  input.readXML(document.getRoot());
  return input;
}

} // namespace

TEST_CASE("KFAC parity controls are explicit and optimizer specific", "[drivers][wf_train][kfac]")
{
  using I = WFTrainDriverInput;
  const auto defaults = parseTrainingInput("<qmc><training optimizer='kfac'/></qmc>");
  CHECK(defaults.getKFACObjective() == I::KFACObjective::RAW_COVARIANCE);
  CHECK(defaults.getKFACCurvature() == I::KFACCurvature::NATIVE_SCORE);
  CHECK(defaults.getKFACDampingPolicy() == I::KFACDamping::EQUAL_SPLIT);
  CHECK(defaults.getKFACConstraint() == I::KFACConstraint::NONE);
  const auto parse = [](const std::string& controls) {
    return parseTrainingInput("<qmc><training optimizer='kfac'>" + controls + "</training></qmc>");
  };
  const auto selected = parse(R"(
    <parameter name="kfac_objective">clipped_covariance</parameter>
    <parameter name="kfac_clip_mode">soft_median</parameter>
    <parameter name="kfac_clip_width">1</parameter>
    <parameter name="kfac_clip_quantile">0.95</parameter>
    <parameter name="kfac_exclude_width">none</parameter>
    <parameter name="kfac_curvature">registered_gaussian</parameter>
    <parameter name="kfac_damping_policy">trace_balanced</parameter>
    <parameter name="kfac_constraint">damped_kfac</parameter>
    <parameter name="kfac_norm_constraint">0.001</parameter>)");
  CHECK(selected.getKFACObjective() == I::KFACObjective::CLIPPED_COVARIANCE);
  CHECK(selected.getKFACClipQuantile() == Catch::Approx(0.95));
  CHECK(selected.getKFACNormConstraint() == Catch::Approx(0.001));
  CHECK_THROWS(parse("<parameter name='kfac_objective'>invalid</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_clip_width'>1</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_constraint'>damped_kfac</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_norm_constraint'>0.001</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_objective'>clipped_covariance</parameter>"
                     "<parameter name='kfac_clip_quantile'>1.1</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_objective'>clipped_covariance</parameter>"
                     "<parameter name='kfac_clip_mode'>hard_median</parameter>"
                     "<parameter name='kfac_clip_quantile'>0.95</parameter>"));
  CHECK_THROWS(parseTrainingInput("<qmc><training optimizer='adam'>"
      "<parameter name='kfac_curvature'>registered_gaussian</parameter></training></qmc>"));
}

TEST_CASE("Complex median covariance is an explicit independent KFAC policy", "[drivers][wf_train][kfac]")
{
  const auto parse = [](const std::string& controls) {
    return parseTrainingInput("<qmc><training optimizer='kfac'>"
        "<parameter name='kfac_objective'>componentwise_mad_covariance</parameter>" + controls + "</training></qmc>");
  };
  const auto defaults = parse("");
  CHECK(defaults.getKFACObjective() == WFTrainDriverInput::KFACObjective::COMPONENTWISE_MAD_COVARIANCE);
  CHECK(defaults.getKFACComplexClipWidth() == 5);
  CHECK(defaults.getKFACClipWidth() == 1);
  CHECK(parse("<parameter name='kfac_complex_clip_width'>2</parameter>").getKFACComplexClipWidth() == 2);
  CHECK_THROWS(parse("<parameter name='kfac_complex_clip_width'>0</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_clip_width'>5</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_clip_mode'>hard_median</parameter>"));
  CHECK_THROWS(parseTrainingInput("<qmc><training optimizer='kfac'>"
      "<parameter name='kfac_complex_clip_width'>5</parameter></training></qmc>"));
  CHECK_THROWS(parseTrainingInput("<qmc><training optimizer='adam'>"
      "<parameter name='kfac_complex_clip_width'>5</parameter></training></qmc>"));
}

TEST_CASE("Typed curvature bundles expand before explicit controls without changing legacy defaults",
          "[drivers][wf_train][kfac][kfac_presets]")
{
  using I = WFTrainDriverInput;
  const auto parse = [](const std::string& controls) {
    return parseTrainingInput("<qmc><training objective='energy' optimizer='kfac'>" + controls + "</training></qmc>");
  };
  const auto legacy = parse("");
  CHECK(legacy.getKFACDefaults() == I::KFACDefaults::LEGACY);
  CHECK(legacy.getKFACGradient() == I::KFACGradient::LEGACY);
  CHECK(legacy.getLearningRate() == .001);
  CHECK_FALSE(legacy.getKFACPersistState());
  const std::array<std::string, 3> labels{"deepqmc", "deepsolid", "jaqmc"};
  const std::array<I::KFACObjective, 3> objectives{I::KFACObjective::CLIPPED_COVARIANCE,
      I::KFACObjective::COMPONENTWISE_MEAN_RESIDUAL, I::KFACObjective::COMPONENTWISE_MAD_COVARIANCE};
  const std::array<I::KFACGradient, 3> gradients{I::KFACGradient::HALF_COVARIANCE,
      I::KFACGradient::REAL_CENTERED_RESIDUAL, I::KFACGradient::PHYSICAL_COVARIANCE};
  for (std::size_t i = 0; i < labels.size(); ++i)
  {
    const auto selector = "<parameter name='defaults'>" + labels[i] + "</parameter>";
    const auto input = parse(selector);
    CHECK(input.getLearningRate() == .05);
    CHECK(input.getDamping() == .001);
    CHECK(input.getKFACFallbackDamping() == .001);
    CHECK(input.getKFACObjective() == objectives[i]);
    CHECK(input.getKFACGradient() == gradients[i]);
    CHECK(input.getKFACCurvature() == I::KFACCurvature::REGISTERED_GAUSSIAN);
    CHECK(input.getKFACDampingPolicy() == I::KFACDamping::TRACE_BALANCED);
    CHECK(input.getKFACPersistState());
    CHECK(input.getKFACEMADecay() == .95);
    CHECK(input.getKFACInversePeriod() == 1);
    CHECK(input.getKFACNormConstraint() == .001);
    CHECK(input.getKFACConstraint() == I::KFACConstraint::DAMPED_KFAC);
    CHECK(input.getKFACLearningRateDecay() == (i == 2 ? 2000 : 10000));
    CHECK(input.getKFACScheduleFloat32() == (i == 2));
    CHECK(input.getKFACScheduleArithmetic() == (i == 2 ? I::KFACScheduleArithmetic::MULTIPLY_RECIPROCAL : I::KFACScheduleArithmetic::DIVIDE_RATE));
    CHECK(input.getKFACUpdateOrder() == (i == 2 ? I::KFACUpdateOrder::PRE_INVERSE_RATE : I::KFACUpdateOrder::POST_INVERSE_RATE));
    CHECK(input.getKFACClipWidth() == 1);
    CHECK(input.getKFACClipQuantile() == .95);
    CHECK(input.getKFACComplexClipWidth() == 5);
    const auto overridden = parse("<parameter name='learning_rate'>.007</parameter>" + selector +
        "<parameter name='kfac_ema_decay'>.7</parameter><parameter name='kfac_objective'>raw_covariance</parameter>"
        "<parameter name='kfac_gradient_convention'>physical_covariance</parameter>"
        "<parameter name='kfac_update_order'>post_inverse_rate</parameter>"
        "<parameter name='kfac_schedule_arithmetic'>divide_rate</parameter>");
    CHECK(overridden.getLearningRate() == .007);
    CHECK(overridden.getKFACEMADecay() == .7);
    CHECK(overridden.getKFACObjective() == I::KFACObjective::RAW_COVARIANCE);
    CHECK(overridden.getKFACGradient() == I::KFACGradient::PHYSICAL_COVARIANCE);
    CHECK(overridden.getKFACUpdateOrder() == I::KFACUpdateOrder::POST_INVERSE_RATE);
    CHECK(overridden.getKFACScheduleArithmetic() == I::KFACScheduleArithmetic::DIVIDE_RATE);
  }
  CHECK_THROWS(parse("<parameter name='defaults'>unknown</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_gradient_convention'>unknown</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_update_order'>unknown</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_schedule_arithmetic'>unknown</parameter>"));
  CHECK_THROWS(parse("<parameter name='kfac_update_order'>pre_inverse_rate</parameter>"));
  CHECK_THROWS(parseTrainingInput("<qmc><training optimizer='adam'><parameter name='defaults'>jaqmc</parameter></training></qmc>"));
  CHECK_THROWS(parseTrainingInput("<qmc><training optimizer='kfac' defaults='jaqmc'/></qmc>"));
}

TEST_CASE("Persistent KFAC inputs require explicit state ownership", "[drivers][wf_train][kfac]")
{
  const auto parse = [](const std::string& controls) {
    return parseTrainingInput("<qmc><training optimizer='kfac'>" + controls + "</training></qmc>");
  };
  CHECK_FALSE(parse("").getKFACPersistState());
  CHECK_THROWS(parse("<parameter name='kfac_ema_decay'>0.95</parameter>"));
  const auto input = parse("<parameter name='kfac_persist_state'>yes</parameter>"
      "<parameter name='kfac_ema_decay'>0.95</parameter>"
      "<parameter name='kfac_inverse_period'>3</parameter>"
      "<parameter name='kfac_learning_rate_decay'>10000</parameter>"
      "<parameter name='kfac_schedule_precision'>float32</parameter>"
      "<parameter name='kfac_momentum'>0</parameter><parameter name='kfac_burn_in'>0</parameter>"
      "<parameter name='kfac_curvature_period'>1</parameter>");
  CHECK(input.getKFACPersistState());
  CHECK(input.getKFACEMADecay() == Catch::Approx(0.95));
  CHECK(input.getKFACInversePeriod() == 3);
  CHECK(input.getKFACScheduleFloat32());
  for (const auto& invalid : {"<parameter name='kfac_ema_decay'>1</parameter>",
         "<parameter name='kfac_inverse_period'>0</parameter>",
         "<parameter name='kfac_momentum'>0.1</parameter>",
         "<parameter name='kfac_burn_in'>1</parameter>",
         "<parameter name='kfac_curvature_period'>2</parameter>"})
    CHECK_THROWS(parse("<parameter name='kfac_persist_state'>yes</parameter>" + std::string(invalid)));
}

TEST_CASE("First-order input admits the shared Euclidean update bound only",
          "[drivers][wf_train][host_controls]")
{
  for (const std::string optimizer : {"sgd", "momentum_sgd", "rmsprop", "adam", "amsgrad"})
  {
    const auto parse = [&](const std::string& control) {
      return parseTrainingInput("<qmc><training optimizer=\"" + optimizer + "\">" + control +
                                "</training></qmc>");
    };
    CHECK(parse(R"(<parameter name="maximum_update_norm">0.00001</parameter>)").getMaximumUpdateNorm() ==
          Catch::Approx(0.00001));
    CHECK_THROWS(parse(R"(<parameter name="maximum_metric_norm">1</parameter>)"));
    CHECK_THROWS(parse(R"(<parameter name="maximum_update_norm">-1</parameter>)"));
  }
}

TEST_CASE("WFTrain inference uses native VMC inputs and completed-attempt cadence", "[drivers][wf_train]")
{
  const auto parse = [](const std::string& xml) {
    Libxml2Document document;
    REQUIRE(document.parseFromString("<qmc>" + xml + "</qmc>"));
    WFTrainInferenceInput input;
    input.readXML(document.getRoot());
    return input;
  };
  CHECK_FALSE(parse("").enabled);
  CHECK_FALSE(parse("").due(1000, true));
  const auto defaults = parse("<inference/>");
  CHECK(defaults.interval == 1000);
  CHECK_FALSE(defaults.last);
  CHECK_FALSE(defaults.due(0, true));
  CHECK_FALSE(defaults.due(999, false));
  CHECK(defaults.due(1000, false));
  const auto interval = parse("<inference interval=\"2\"/>");
  CHECK_FALSE(interval.due(1, false));
  CHECK(interval.due(2, false));
  CHECK_FALSE(interval.due(3, true));
  const auto last = parse("<inference interval=\"2\" last=\"yes\"/>");
  CHECK(last.due(2, true));
  CHECK(last.due(3, true));
  const auto sampling = parse(R"(<inference interval="2" last="no" move="pbyp">
    <parameter name="walkers_per_rank">3</parameter><parameter name="crowds">2</parameter>
    <parameter name="blocks">4</parameter><parameter name="steps">5</parameter>
    <parameter name="warmupsteps">6</parameter><parameter name="substeps">7</parameter>
    <parameter name="timestep">0.125</parameter><parameter name="usedrift">yes</parameter>
    </inference>)");
  CHECK(sampling.sampling.get_walkers_per_rank() == 3);
  CHECK(sampling.sampling.get_num_crowds() == 2);
  CHECK(sampling.sampling.get_max_blocks() == 4);
  CHECK(sampling.sampling.get_requested_steps() == 5);
  CHECK(sampling.sampling.get_warmup_steps() == 6);
  CHECK(sampling.sampling.get_sub_steps() == 7);
  CHECK(sampling.sampling.get_tau() == 0.125);
  CHECK(sampling.vmc.get_use_drift());
  CHECK(parse("<inference/>").evaluate_energy);
  CHECK_FALSE(parse("<inference evaluate_energy=\"no\"/>").evaluate_energy);
  CHECK(parse("<inference evaluate_energy=\"yes\"/>").evaluate_energy);
  CHECK_THROWS(parse("<inference evaluate_energy=\"maybe\"/>"));
  for (const std::string invalid : {
      "<inference/><inference/>", "<inference interval=\"0\"/>", "<inference interval=\"-1\"/>",
      "<inference interval=\"184467440737095516160\"/>", "<inference last=\"true\"/>",
      "<inference mystery=\"1\"/>", "<inference><parameter name=\"mystery\">1</parameter></inference>",
      "<inference><parameter name=\"steps\">bad</parameter></inference>",
      "<inference><parameter name=\"estimator_period\">0</parameter></inference>",
      "<inference><parameter name=\"usedrift\">maybe</parameter></inference>",
      "<inference><parameter name=\"steps\">1</parameter><parameter name=\"steps\">2</parameter></inference>",
      "<inference><parameter name=\"tau\">1</parameter><parameter name=\"timestep\">2</parameter></inference>"})
    CHECK_THROWS(parse(invalid));
  CHECK_NOTHROW(parse("<inference><estimators><unknown/></estimators></inference>"));
}

TEST_CASE("WFTrainDriverInput accepts conventional orbital pretraining", "[drivers][wf_train]")
{
  const std::string controls = R"(
    <parameter name="teacher">slater_teacher</parameter>
    <parameter name="target_provenance">inline-gaussian-v1</parameter>)";
  const auto parse = [&](const std::string& extra, const std::string& optimizer = "adam") {
    return parseTrainingInput("<qmc method=\"wf_train\"><training objective=\"orbital_mse\" optimizer=\"" +
        optimizer + "\">" + controls + extra + "</training></qmc>");
  };
  CHECK(parse("").getObjective() == WFTrainDriverInput::Objective::ORBITAL_MSE);
  CHECK(parse("").getOrbitalTarget() == WFTrainDriverInput::OrbitalTarget::SINGLE_REFERENCE);
  CHECK(parse("").getOrbitalSampling() == WFTrainDriverInput::OrbitalSampling::TEACHER);
  CHECK(parse("").getOrbitalNormalization() == wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM);
  CHECK(parse("<parameter name=\"orbital_normalization\">spin_sector_sum</parameter>").getOrbitalNormalization() ==
        wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM);
  CHECK(parse("<parameter name=\"orbital_normalization\">full_matrix_mean</parameter>").getOrbitalNormalization() ==
        wftrain::OrbitalLossNormalization::FULL_MATRIX_MEAN);
  CHECK_THROWS(parse("<parameter name=\"orbital_normalization\">unknown</parameter>"));
  CHECK_THROWS(parseTrainingInput(R"(<qmc><training objective="energy">
    <parameter name="orbital_normalization">full_matrix_mean</parameter></training></qmc>)"));
  CHECK_FALSE(parse("").samplesOrbitalStudent());
  CHECK(parse("<parameter name=\"orbital_sampling\">student</parameter>").samplesOrbitalStudent());
  CHECK_FALSE(parse("<parameter name=\"orbital_sampling\">teacher</parameter>").samplesOrbitalStudent());
  CHECK_THROWS(parse("<parameter name=\"orbital_sampling\">unknown</parameter>"));
  CHECK_THROWS(parseTrainingInput(R"(<qmc><training objective="energy">
    <parameter name="orbital_sampling">teacher</parameter></training></qmc>)"));
  CHECK(parse("<parameter name=\"orbital_target\">top_coefficients</parameter>").getOrbitalTarget() ==
        WFTrainDriverInput::OrbitalTarget::TOP_COEFFICIENT_MULTIDETERMINANT);
  for (const std::string optimizer : {"sgd", "momentum_sgd", "rmsprop", "adam", "amsgrad"})
    CHECK_NOTHROW(parse("", optimizer));
  CHECK_THROWS(parse("", "sr"));
  CHECK_THROWS(parse("<parameter name=\"line_search\">reference</parameter>"));
  CHECK_THROWS(parse("<parameter name=\"maximum_metric_norm\">1</parameter>"));
  CHECK_THROWS(parse("<parameter name=\"overlap_target\">general</parameter>"));
  CHECK_THROWS(parse("<parameter name=\"orbital_target\">unsupported</parameter>"));
  CHECK_THROWS(parseTrainingInput(R"(<qmc><training objective="orbital_mse"/></qmc>)"));
  CHECK_THROWS(parseTrainingInput(R"(<qmc><training objective="energy">
    <parameter name="orbital_target">single_reference</parameter></training></qmc>)"));
}

TEST_CASE("WFTrainDriverInput validates native checkpoint cadence", "[drivers][wf_train]")
{
  const auto parse = [](const std::string& value) {
    return parseTrainingInput("<qmc><training><parameter name=\"checkpoint_interval\">" + value +
                              "</parameter></training></qmc>");
  };
  CHECK(parse("1").getCheckpointInterval() == 1);
  CHECK(parse("1000").getCheckpointInterval() == 1000);
  for (const std::string value : {"0", "-1", "1.5", "bad", "184467440737095516160"})
    CHECK_THROWS(parse(value));
}

TEST_CASE("WFTrainDriverInput accepts strict energy/adam input", "[drivers][wf_train]")
{
  SECTION("defaults")
  {
    const auto input = parseTrainingInput(R"(<qmc method="wf_train"><training></training></qmc>)");
    CHECK(input.getIterations() == 1);
    CHECK(input.getCheckpointInterval() == 1000);
    CHECK(input.getWarmupSteps() == 0);
    CHECK(input.getObjective() == WFTrainDriverInput::Objective::ENERGY);
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM);
    CHECK(input.getParameterChunkSize() == 65536);
    CHECK(input.getLearningRate() == 1.0e-3);
    CHECK(input.getAdamBeta1() == 0.9);
    CHECK(input.getAdamBeta2() == 0.999);
    CHECK(input.getEpsilon() == 1.0e-8);
    CHECK(input.getDamping() == 1.0e-3);
    CHECK(input.getDampingMultiplier() == 10.0);
    CHECK(input.getMinimumRetryDamping() == 1.0e-8);
    CHECK(input.getMaximumDampingAttempts() == 3);
    CHECK(input.getMaximumUpdateNorm() == 0.0);
    CHECK(input.getMaximumMetricNorm() == 0.0);
    CHECK(input.getCenterShift() == 1.0);
    CHECK(input.getObjectiveTolerance() == 1.0e-9);
    CHECK(input.getSpringHistoryDecay() == 0.99);
    CHECK(input.getSpringMaximumUpdateNormSquared() == 0.0);
    CHECK(input.getKrylovRelativeTolerance() == 1.0e-8);
    CHECK(input.getKrylovAbsoluteTolerance() == 0.0);
    CHECK(input.getKrylovMaximumIterations() == 100);
    CHECK(input.getKrylovMaximumOperatorApplications() == 101);
    CHECK(input.getKrylovMaximumSeconds() == 0.0);
    CHECK(input.getKrylovStagnationWindow() == 0);
    CHECK(input.getKrylovMinimumRelativeImprovement() == 1.0e-4);
    CHECK(input.getKrylovCurvatureTolerance() == 0.0);
    CHECK(input.getKrylovImaginaryTolerance() == 64.0);
    CHECK(input.getMaximumRetainedBytes() == 0);
    CHECK(input.getCheckpointPath().empty());
    CHECK(input.getRestartPath().empty());
    CHECK(input.getProviderID().empty());
  }

  SECTION("scalar parameter children")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training objective="energy" optimizer="adam">
          <parameter name="parameter_chunk_size">2048</parameter>
          <parameter name="learning_rate">2e-4</parameter>
          <parameter name="adam_beta1">0.8</parameter>
          <parameter name="adam_beta2">0.95</parameter>
          <parameter name="epsilon">1e-7</parameter>
          <parameter name="checkpoint_path">continue.h5</parameter>
          <parameter name="restart_path">continue.h5</parameter>
          <parameter name="provider">neural</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getParameterChunkSize() == 2048);
    CHECK(input.getLearningRate() == 2.0e-4);
    CHECK(input.getAdamBeta1() == 0.8);
    CHECK(input.getAdamBeta2() == 0.95);
    CHECK(input.getEpsilon() == 1.0e-7);
    CHECK(input.getCheckpointPath() == "continue.h5");
    CHECK(input.getRestartPath() == "continue.h5");
    CHECK(input.getProviderID() == "neural");
  }
}

TEST_CASE("WFTrainDriverInput accepts strict determinant-overlap input",
          "[drivers][wf_train][overlap]")
{
  const auto input = parseTrainingInput(
      R"(
      <qmc method="wf_train">
        <training objective="determinant_overlap" optimizer="adam">
          <parameter name="teacher">teacher_psi</parameter>
          <parameter name="overlap_target">top_coefficients</parameter>
          <parameter name="overlap_weighting">coefficient_squared</parameter>
          <parameter name="target_provenance">ci-hash-42</parameter>
          <parameter name="minimum_fidelity">1e-10</parameter>
          <parameter name="minimum_effective_samples">3</parameter>
        </training>
      </qmc>
      )");
  CHECK(input.getObjective() == WFTrainDriverInput::Objective::DETERMINANT_OVERLAP);
  CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM);
  CHECK(input.getTeacherWaveFunction() == "teacher_psi");
  CHECK(input.getOverlapTarget() ==
        WFTrainDriverInput::OverlapTarget::TOP_COEFFICIENT_MULTIDETERMINANT);
  CHECK(input.getOverlapWeighting() ==
        WFTrainDriverInput::OverlapWeighting::COEFFICIENT_SQUARED);
  CHECK(input.getTargetProvenance() == "ci-hash-42");
  CHECK(input.getMinimumFidelity() == Catch::Approx(1.0e-10));
  CHECK(input.getMinimumEffectiveSamples() == Catch::Approx(3.0));
}

TEST_CASE("WFTrainDriverInput accepts typed matrix-free optimizer controls",
          "[drivers][wf_train]")
{
  SECTION("parameter-space SR")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="learning_rate">0.02</parameter>
          <parameter name="damping">0</parameter>
          <parameter name="damping_multiplier">4</parameter>
          <parameter name="minimum_retry_damping">2e-7</parameter>
          <parameter name="maximum_damping_attempts">5</parameter>
          <parameter name="maximum_update_norm">2</parameter>
          <parameter name="maximum_metric_norm">3</parameter>
          <parameter name="krylov_relative_tolerance">2e-6</parameter>
          <parameter name="krylov_absolute_tolerance">3e-9</parameter>
          <parameter name="krylov_maximum_iterations">40</parameter>
          <parameter name="krylov_maximum_operator_applications">44</parameter>
          <parameter name="krylov_maximum_seconds">3</parameter>
          <parameter name="krylov_stagnation_window">7</parameter>
          <parameter name="krylov_minimum_relative_improvement">5e-5</parameter>
          <parameter name="krylov_curvature_tolerance">6e-12</parameter>
          <parameter name="krylov_imaginary_tolerance">32</parameter>
          <parameter name="maximum_retained_bytes">123456</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() ==
          WFTrainDriverInput::Optimizer::STOCHASTIC_RECONFIGURATION);
    CHECK(input.getLearningRate() == 0.02);
    CHECK(input.getDamping() == 0.0);
    CHECK(input.getDampingMultiplier() == 4.0);
    CHECK(input.getMinimumRetryDamping() == 2.0e-7);
    CHECK(input.getMaximumDampingAttempts() == 5);
    CHECK(input.getMaximumUpdateNorm() == 2.0);
    CHECK(input.getMaximumMetricNorm() == 3.0);
    CHECK(input.getKrylovRelativeTolerance() == 2.0e-6);
    CHECK(input.getKrylovAbsoluteTolerance() == 3.0e-9);
    CHECK(input.getKrylovMaximumIterations() == 40);
    CHECK(input.getKrylovMaximumOperatorApplications() == 44);
    CHECK(input.getKrylovMaximumSeconds() == 3.0);
    CHECK(input.getKrylovStagnationWindow() == 7);
    CHECK(input.getKrylovMinimumRelativeImprovement() == 5.0e-5);
    CHECK(input.getKrylovCurvatureTolerance() == 6.0e-12);
    CHECK(input.getKrylovImaginaryTolerance() == 32.0);
    CHECK(input.getMaximumRetainedBytes() == 123456);
  }

  SECTION("minSR")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="minsr">
          <parameter name="damping">0.02</parameter>
          <parameter name="center_shift">0.5</parameter>
          <parameter name="objective_tolerance">2e-8</parameter>
          <parameter name="maximum_update_norm">1.5</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR);
    CHECK(input.getDamping() == 0.02);
    CHECK(input.getCenterShift() == 0.5);
    CHECK(input.getObjectiveTolerance() == 2.0e-8);
    CHECK(input.getMaximumUpdateNorm() == 1.5);
  }

  SECTION("SPRING")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="spring">
          <parameter name="damping">0.03</parameter>
          <parameter name="center_shift">0.25</parameter>
          <parameter name="spring_history_decay">0.8</parameter>
          <parameter name="spring_maximum_update_norm_squared">4</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::SPRING);
    CHECK(input.getDamping() == 0.03);
    CHECK(input.getCenterShift() == 0.25);
    CHECK(input.getSpringHistoryDecay() == 0.8);
    CHECK(input.getSpringMaximumUpdateNormSquared() == 4.0);
  }
}

TEST_CASE("WFTrainDriverInput maps strict PII and MinPII controls",
          "[drivers][wf_train][pii]")
{
  SECTION("explicit-shift PII")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="pii">
          <parameter name="learning_rate">0.4</parameter>
          <parameter name="pii_shift_mode">explicit</parameter>
          <parameter name="pii_explicit_tau">-1.25</parameter>
          <parameter name="damping">0.02</parameter>
          <parameter name="damping_multiplier">3</parameter>
          <parameter name="maximum_damping_retries">4</parameter>
          <parameter name="maximum_update_norm">2</parameter>
          <parameter name="objective_tolerance">3e-10</parameter>
          <parameter name="gmres_restart_length">17</parameter>
          <parameter name="krylov_relative_tolerance">2e-7</parameter>
          <parameter name="krylov_absolute_tolerance">4e-12</parameter>
          <parameter name="krylov_maximum_iterations">31</parameter>
          <parameter name="krylov_maximum_operator_applications">37</parameter>
          <parameter name="krylov_maximum_seconds">5</parameter>
          <parameter name="krylov_stagnation_window">6</parameter>
          <parameter name="krylov_minimum_relative_improvement">8e-5</parameter>
          <parameter name="krylov_curvature_tolerance">9e-13</parameter>
          <parameter name="krylov_imaginary_tolerance">16</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() ==
          WFTrainDriverInput::Optimizer::PROJECTED_INVERSE_ITERATION);
    CHECK(input.getPIIShiftMode() == WFTrainDriverInput::PIIShiftMode::EXPLICIT);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "pii");
    CHECK(wftrain::optimizerRequiresRealParameters(input.getOptimizer()));

    const auto control = wftrain::makePIIUpdateControl(input);
    CHECK(control.learning_rate == 0.4);
    CHECK(control.shift_mode == wftrain::PIIShiftMode::EXPLICIT);
    CHECK(control.explicit_shift == -1.25);
    CHECK(control.energy_undershoot == 0.1);
    CHECK(control.damping == 0.02);
    CHECK(control.damping_multiplier == 3.0);
    CHECK(control.maximum_damping_retries == 4);
    CHECK(control.maximum_update_norm == 2.0);
    CHECK(control.objective_tolerance == 3.0e-10);
    CHECK(control.gmres.restart_length == 17);
    CHECK(control.gmres.krylov.relative_tolerance == 2.0e-7);
    CHECK(control.gmres.krylov.absolute_tolerance == 4.0e-12);
    CHECK(control.gmres.krylov.maximum_iterations == 31);
    CHECK(control.gmres.krylov.maximum_operator_applications == 37);
    CHECK(control.gmres.krylov.maximum_seconds == 5.0);
    CHECK(control.gmres.krylov.stagnation_window == 6);
    CHECK(control.gmres.krylov.minimum_relative_improvement == 8.0e-5);
    CHECK(control.gmres.krylov.curvature_tolerance == 9.0e-13);
    CHECK(control.gmres.krylov.imaginary_tolerance == 16.0);
  }

  SECTION("mean-energy-shift MinPII defaults")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="minpii">
          <parameter name="pii_energy_undershoot">0.35</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() ==
          WFTrainDriverInput::Optimizer::MINIMUM_PROJECTED_INVERSE_ITERATION);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "minpii");
    const auto control = wftrain::makePIIUpdateControl(input);
    CHECK(control.learning_rate == 1.0);
    CHECK(control.shift_mode == wftrain::PIIShiftMode::MEAN_ENERGY_MINUS_UNDERSHOOT);
    CHECK(control.energy_undershoot == 0.35);
    CHECK(control.maximum_damping_retries == 2);
    CHECK(control.gmres.restart_length == 20);
  }
}

TEST_CASE("WFTrainDriverInput accepts strict SGD, RMSProp, AMSGrad, LM, ALM, and KFAC input",
          "[drivers][wf_train][optimizers]")
{
  SECTION("SGD defaults and selection")
  {
    const auto input = parseTrainingInput(
        R"(<qmc method="wf_train"><training optimizer="sgd"></training></qmc>)");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::SGD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "sgd");
    CHECK(input.getLearningRate() == 1.0e-3);
    CHECK(input.getMomentumDecay() == 0.9);
    CHECK(input.getRMSDecay() == 0.9);
  }

  SECTION("momentum SGD maps momentum_decay")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="momentum_sgd">
          <parameter name="learning_rate">0.03</parameter>
          <parameter name="momentum_decay">0.75</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::MOMENTUM_SGD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "momentum_sgd");
    CHECK(input.getLearningRate() == 0.03);
    CHECK(input.getMomentumDecay() == 0.75);
  }

  SECTION("RMSProp maps rms_decay")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="rmsprop">
          <parameter name="learning_rate">0.02</parameter>
          <parameter name="rms_decay">0.8</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::RMSPROP);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "rmsprop");
    CHECK(input.getLearningRate() == 0.02);
    CHECK(input.getRMSDecay() == 0.8);
  }

  SECTION("AMSGrad accepts its Adam moment controls")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="amsgrad">
          <parameter name="learning_rate">0.01</parameter>
          <parameter name="adam_beta1">0.8</parameter>
          <parameter name="adam_beta2">0.95</parameter>
          <parameter name="epsilon">1e-7</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::AMSGRAD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "amsgrad");
    CHECK(input.getLearningRate() == 0.01);
    CHECK(input.getAdamBeta1() == 0.8);
    CHECK(input.getAdamBeta2() == 0.95);
    CHECK(input.getEpsilon() == 1.0e-7);
  }

  SECTION("LM maps LM controls")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="lm">
          <parameter name="damping">0.02</parameter>
          <parameter name="krylov_relative_tolerance">2e-6</parameter>
          <parameter name="lm_overlap_shift">0.03</parameter>
          <parameter name="lm_normalization_xi">0.4</parameter>
          <parameter name="lm_maximum_subspace_dimension">12</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::LINEAR_METHOD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "lm");
    CHECK(input.getLearningRate() == 1.0);
    CHECK(input.getDamping() == 0.02);
    CHECK(input.getKrylovRelativeTolerance() == 2.0e-6);
    CHECK(input.getLMOverlapShift() == 0.03);
    CHECK(input.getLMNormalizationXi() == 0.4);
    CHECK(input.getLMMaximumSubspaceDimension() == 12);
  }

  SECTION("ALM maps iterative controls")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="alm">
          <parameter name="damping">0.03</parameter>
          <parameter name="krylov_relative_tolerance">3e-6</parameter>
          <parameter name="lm_inner_iterations">9</parameter>
          <parameter name="lm_inner_restart_length">5</parameter>
          <parameter name="lm_inner_relative_tolerance">0.02</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::ACCELERATED_LINEAR_METHOD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "alm");
    CHECK(input.getDamping() == 0.03);
    CHECK(input.getKrylovRelativeTolerance() == 3.0e-6);
    CHECK(input.getLMInnerIterations() == 9);
    CHECK(input.getLMInnerRestartLength() == 5);
    CHECK(input.getLMInnerRelativeTolerance() == 0.02);
  }

  SECTION("KFAC maps fallback damping")
  {
    const auto input = parseTrainingInput(
        R"(
      <qmc method="wf_train">
        <training optimizer="kfac">
          <parameter name="damping">0.04</parameter>
          <parameter name="kfac_fallback_damping">0.006</parameter>
        </training>
      </qmc>
      )");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::KFAC);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "kfac");
    CHECK(input.getDamping() == 0.04);
    CHECK(input.getKFACFallbackDamping() == 0.006);
  }
}

TEST_CASE("WFTrainDriverInput rejects ambiguous or unsupported input", "[drivers][wf_train]")
{
  WFTrainDriverInput input;

  auto rejects = [&input](const std::string& xml) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    CHECK_THROWS(input.readXML(document.getRoot()));
  };

  rejects(R"(<qmc method="wf_train"/>)");
  rejects(R"(<qmc method="wf_train"><training></training><training></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="mystery">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><optimizer/></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training objective="overlap"></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="teacher">teacher</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training objective="determinant_overlap"></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training objective="determinant_overlap" optimizer="sr">
          <parameter name="teacher">teacher</parameter>
          <parameter name="target_provenance">x</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training objective="determinant_overlap">
          <parameter name="teacher">teacher</parameter>
          <parameter name="target_provenance">x</parameter>
          <parameter name="overlap_weighting">coefficient_squared</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="unknown"></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="parameter_chunk_size">0</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="parameter_chunk_size">1k</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="learning_rate">0</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="learning_rate">nan</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="learning_rate">inf</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="adam_beta1">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="adam_beta2">-0.1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="epsilon">0</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="damping">1e-3</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="adam_beta1">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="center_shift">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="minsr"><parameter name="damping">0</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="minsr"><parameter name="damping">nan</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="minsr"><parameter name="damping_multiplier">2</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="spring"><parameter name="maximum_metric_norm">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="spring"><parameter name="damping">-1</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="spring">
          <parameter name="spring_history_decay">1</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="damping_multiplier">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="minimum_retry_damping">0</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="maximum_damping_attempts">0</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="maximum_update_norm">-1</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="krylov_relative_tolerance">0</parameter>
          <parameter name="krylov_absolute_tolerance">0</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="krylov_maximum_iterations">0</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="krylov_maximum_operator_applications">0</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="krylov_maximum_seconds">-1</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="sr">
          <parameter name="krylov_maximum_seconds">inf</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training><parameter name="maximum_retained_bytes">-1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training><parameter name="provider"></parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="pii"><parameter name="pii_shift_mode">fixed</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="pii"><parameter name="pii_shift_mode">explicit</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="pii"><parameter name="pii_explicit_tau">-1</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="pii">
          <parameter name="pii_shift_mode">explicit</parameter>
          <parameter name="pii_explicit_tau">-1</parameter>
          <parameter name="pii_energy_undershoot">0.1</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="pii"><parameter name="damping">0</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="pii">
          <parameter name="maximum_damping_attempts">2</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="pii">
          <parameter name="maximum_damping_retries">65</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(<qmc method="wf_train"><training optimizer="pii"><parameter name="gmres_restart_length">0</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="minpii"><parameter name="center_shift">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="minpii"><parameter name="adam_beta1">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sr"><parameter name="maximum_damping_retries">2</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sgd"><parameter name="adam_beta1">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="sgd"><parameter name="momentum_decay">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="momentum_sgd"><parameter name="rms_decay">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="rmsprop"><parameter name="adam_beta1">0.8</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="rmsprop"><parameter name="rms_decay">1</parameter></training></qmc>)");
  rejects(R"(<qmc method="wf_train"><training optimizer="amsgrad"><parameter name="rms_decay">0.8</parameter></training></qmc>)");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="momentum_sgd">
          <parameter name="momentum_decay">1</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="lm">
          <parameter name="damping">0.02</parameter>
          <parameter name="lm_overlap_shift">0</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="lm">
          <parameter name="damping">0.02</parameter>
          <parameter name="krylov_relative_tolerance">1e-6</parameter>
          <parameter name="lm_inner_iterations">3</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="lm">
          <parameter name="damping">0.02</parameter>
          <parameter name="krylov_relative_tolerance">1e-6</parameter>
          <parameter name="maximum_damping_attempts">2</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="alm">
          <parameter name="damping">0.02</parameter>
          <parameter name="krylov_relative_tolerance">1e-6</parameter>
          <parameter name="spring_history_decay">0.8</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="kfac">
          <parameter name="damping">0.02</parameter>
          <parameter name="krylov_relative_tolerance">1e-6</parameter>
        </training>
      </qmc>
      )");
  rejects(R"(
      <qmc method="wf_train">
        <training optimizer="kfac">
          <parameter name="damping">0.02</parameter>
          <parameter name="kfac_fallback_damping">0</parameter>
        </training>
      </qmc>
      )");
}

TEST_CASE("WFTrainDriverInput separates training iterations and warmup from sample blocks",
          "[drivers][wf_train]")
{
  const auto input = parseTrainingInput(R"(
    <qmc method="wf_train">
      <parameter name="blocks">4</parameter>
      <training objective="energy" optimizer="adam">
        <!-- Indentation and scalar CDATA are accepted. -->
        <parameter name="iterations">
          100
        </parameter>
        <parameter name="warmupsteps"><![CDATA[ 2 ]]></parameter>
        <parameter name="learning_rate"> 1e-5 </parameter>
      </training>
    </qmc>
  )");
  CHECK(input.getIterations() == 100);
  CHECK(input.getWarmupSteps() == 2);
  CHECK(input.getLearningRate() == 1e-5);

  const auto defaults = parseTrainingInput(R"(<qmc><training/></qmc>)");
  CHECK(defaults.getIterations() == 1);
  CHECK(defaults.getWarmupSteps() == 0);
}

TEST_CASE("WFTrainDriverInput rejects obsolete attributes and malformed parameter children",
          "[drivers][wf_train]")
{
  const std::string bad_inputs[] = {
      R"(<qmc><training learning_rate="1e-3"/></qmc>)",
      R"(<qmc><training iterations="2"/></qmc>)",
      R"(<qmc><training><parameter name="objective">energy</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="optimizer">adam</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations">0</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations">-1</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations">184467440737095516160</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="warmupsteps">-1</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="warmupsteps">1.5</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="warmupsteps">184467440737095516160</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="mystery">1</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations">2</parameter><parameter name="iterations">3</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="learning_rate">  </parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations" value="2">2</parameter></training></qmc>)",
      R"(<qmc><training><parameter>2</parameter></training></qmc>)",
      R"(<qmc><training><parameter name="iterations"><value>2</value></parameter></training></qmc>)",
      R"(<qmc><training>unexpected text</training></qmc>)"};
  for (const auto& xml : bad_inputs)
  {
    INFO(xml);
    CHECK_THROWS(parseTrainingInput(xml));
  }
}

TEST_CASE("WFTrainDriverInput distinguishes RSS and numeric storage guards", "[drivers][wf_train]")
{
  const auto defaults = parseTrainingInput("<qmc><training/></qmc>");
  CHECK(defaults.getMaximumResidentBytes() == 0);
  const auto guarded = parseTrainingInput(
      "<qmc><training><parameter name='maximum_resident_bytes'>1073741824</parameter>"
      "<parameter name='maximum_retained_bytes'>524288</parameter></training></qmc>");
  CHECK(guarded.getMaximumResidentBytes() == 1073741824);
  CHECK(guarded.getMaximumRetainedBytes() == 524288);
  CHECK_THROWS(parseTrainingInput(
      "<qmc><training><parameter name='maximum_resident_bytes'>-1</parameter></training></qmc>"));
}

TEST_CASE("WFTrainDriverInput validates complete bounded complex GPU opt-in",
          "[drivers][wf_train][complex_gpu_training_input]")
{
  const std::array<std::string, 4> names{"complex_gpu_device_bytes", "complex_gpu_host_bytes",
      "complex_gpu_kinetic_targets", "complex_gpu_maximum_matrix_work"};
  const std::array<std::string, 4> values{"8192", "4096", "3", "777"};
  const auto controls = [&](unsigned mask, int invalid = -1, const std::string& spelling = "") {
    std::string result;
    for (std::size_t i = 0; i < names.size(); ++i)
      if (mask & (1U << i))
        result += "<parameter name='" + names[i] + "'>" +
            (static_cast<int>(i) == invalid ? spelling : values[i]) + "</parameter>";
    return result;
  };
  const auto parse = [&](const std::string& extra) {
    return parseTrainingInput("<qmc><training objective='energy' optimizer='adam'>" + extra + "</training></qmc>");
  };
  const auto check_disabled = [](const WFTrainDriverInput& input) {
    const auto& gpu = input.getComplexGpuTraining();
    CHECK_FALSE(gpu.enabled());
    CHECK(gpu.device_bytes == 0);
    CHECK(gpu.host_bytes == 0);
    CHECK(gpu.kinetic_targets == 0);
    CHECK(gpu.maximum_matrix_work == 0);
  };

  const auto defaults = parse("");
  check_disabled(defaults);
  CHECK(defaults.getObjective() == WFTrainDriverInput::Objective::ENERGY);
  CHECK(defaults.getOptimizer() == WFTrainDriverInput::Optimizer::ADAM);
  CHECK(defaults.getLearningRate() == 1e-3);
  CHECK(defaults.getMaximumRetainedBytes() == 0);
  const auto input = parse(controls(15));
  const auto& gpu = input.getComplexGpuTraining();
  CHECK(gpu.enabled());
  CHECK(gpu.device_bytes == 8192);
  CHECK(gpu.host_bytes == 4096);
  CHECK(gpu.kinetic_targets == 3);
  CHECK(gpu.maximum_matrix_work == 777);
  CHECK(input.getLearningRate() == defaults.getLearningRate());
  CHECK(input.getDamping() == defaults.getDamping());
  CHECK(input.getMaximumRetainedBytes() == defaults.getMaximumRetainedBytes());

  // Every nonempty proper subset fails, and each bad field is isolated among three valid controls.
  for (unsigned mask = 1; mask < 15; ++mask)
  {
    INFO("control mask=" << mask);
    CHECK_THROWS(parse(controls(mask)));
  }
  const auto largest = std::to_string(std::numeric_limits<std::size_t>::max());
  for (int field = 0; field < 4; ++field)
    for (const auto& invalid : {std::string("0"), std::string("-1"), std::string("1x"), std::string("1.5"),
             std::string(""), std::string("nan"), largest, largest + "0"})
    {
      INFO(names[field] << "='" << invalid << "'");
      CHECK_THROWS(parse(controls(15, field, invalid)));
    }
  CHECK_THROWS(parse(controls(15) + "<parameter name='complex_gpu_host_bytes'>4096</parameter>"));

  auto reused = parse(controls(15) + "<parameter name='learning_rate'>0.0007</parameter>"
      "<parameter name='maximum_retained_bytes'>524288</parameter>");
  CHECK(reused.getLearningRate() == 0.0007);
  CHECK(reused.getMaximumRetainedBytes() == 524288);
  Libxml2Document second;
  REQUIRE(second.parseFromString("<qmc><training/></qmc>"));
  reused.readXML(second.getRoot());
  check_disabled(reused);
  CHECK(reused.getLearningRate() == defaults.getLearningRate());
  CHECK(reused.getMaximumRetainedBytes() == defaults.getMaximumRetainedBytes());
  CHECK(reused.getParameterChunkSize() == defaults.getParameterChunkSize());
  CHECK(reused.getOptimizer() == defaults.getOptimizer());
}

} // namespace qmcplusplus
