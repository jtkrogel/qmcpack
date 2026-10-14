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

TEST_CASE("WFTrainDriverInput accepts strict energy/adam input", "[drivers][wf_train]")
{
  SECTION("defaults")
  {
    const auto input = parseTrainingInput(R"(<qmc method="wf_train"><training></training></qmc>)");
    CHECK(input.getIterations() == 1);
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

} // namespace qmcplusplus
