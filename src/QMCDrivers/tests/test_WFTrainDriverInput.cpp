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
    const auto input = parseTrainingInput("<qmc method=\"wf_train\"><training/></qmc>");
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

  SECTION("all attributes")
  {
    const auto input = parseTrainingInput(
        "<qmc method=\"wf_train\"><training objective=\"energy\" optimizer=\"adam\" "
        "parameter_chunk_size=\"2048\" learning_rate=\"2e-4\" adam_beta1=\"0.8\" "
        "adam_beta2=\"0.95\" epsilon=\"1e-7\" checkpoint_path=\"continue.h5\" restart_path=\"continue.h5\" "
        "provider=\"neural\"/></qmc>");
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
      "<qmc method=\"wf_train\"><training objective=\"determinant_overlap\" "
      "optimizer=\"adam\" teacher=\"teacher_psi\" overlap_target=\"top_coefficients\" "
      "overlap_weighting=\"coefficient_squared\" target_provenance=\"ci-hash-42\" "
      "minimum_fidelity=\"1e-10\" minimum_effective_samples=\"3\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"sr\" learning_rate=\"0.02\" "
        "damping=\"0\" damping_multiplier=\"4\" minimum_retry_damping=\"2e-7\" "
        "maximum_damping_attempts=\"5\" maximum_update_norm=\"2\" "
        "maximum_metric_norm=\"3\" krylov_relative_tolerance=\"2e-6\" "
        "krylov_absolute_tolerance=\"3e-9\" krylov_maximum_iterations=\"40\" "
        "krylov_maximum_operator_applications=\"44\" krylov_maximum_seconds=\"3\" "
        "krylov_stagnation_window=\"7\" krylov_minimum_relative_improvement=\"5e-5\" "
        "krylov_curvature_tolerance=\"6e-12\" krylov_imaginary_tolerance=\"32\" "
        "maximum_retained_bytes=\"123456\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"minsr\" damping=\"0.02\" "
        "center_shift=\"0.5\" objective_tolerance=\"2e-8\" "
        "maximum_update_norm=\"1.5\"/></qmc>");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::MINIMUM_STEP_SR);
    CHECK(input.getDamping() == 0.02);
    CHECK(input.getCenterShift() == 0.5);
    CHECK(input.getObjectiveTolerance() == 2.0e-8);
    CHECK(input.getMaximumUpdateNorm() == 1.5);
  }

  SECTION("SPRING")
  {
    const auto input = parseTrainingInput(
        "<qmc method=\"wf_train\"><training optimizer=\"spring\" damping=\"0.03\" "
        "center_shift=\"0.25\" spring_history_decay=\"0.8\" "
        "spring_maximum_update_norm_squared=\"4\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"pii\" learning_rate=\"0.4\" "
        "pii_shift_mode=\"explicit\" pii_explicit_tau=\"-1.25\" damping=\"0.02\" "
        "damping_multiplier=\"3\" maximum_damping_retries=\"4\" maximum_update_norm=\"2\" "
        "objective_tolerance=\"3e-10\" gmres_restart_length=\"17\" "
        "krylov_relative_tolerance=\"2e-7\" krylov_absolute_tolerance=\"4e-12\" "
        "krylov_maximum_iterations=\"31\" krylov_maximum_operator_applications=\"37\" "
        "krylov_maximum_seconds=\"5\" krylov_stagnation_window=\"6\" "
        "krylov_minimum_relative_improvement=\"8e-5\" krylov_curvature_tolerance=\"9e-13\" "
        "krylov_imaginary_tolerance=\"16\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"minpii\" "
        "pii_energy_undershoot=\"0.35\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"sgd\"/></qmc>");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::SGD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "sgd");
    CHECK(input.getLearningRate() == 1.0e-3);
    CHECK(input.getMomentumDecay() == 0.9);
    CHECK(input.getRMSDecay() == 0.9);
  }

  SECTION("momentum SGD maps momentum_decay")
  {
    const auto input = parseTrainingInput(
        "<qmc method=\"wf_train\"><training optimizer=\"momentum_sgd\" "
        "learning_rate=\"0.03\" momentum_decay=\"0.75\"/></qmc>");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::MOMENTUM_SGD);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "momentum_sgd");
    CHECK(input.getLearningRate() == 0.03);
    CHECK(input.getMomentumDecay() == 0.75);
  }

  SECTION("RMSProp maps rms_decay")
  {
    const auto input = parseTrainingInput(
        "<qmc method=\"wf_train\"><training optimizer=\"rmsprop\" "
        "learning_rate=\"0.02\" rms_decay=\"0.8\"/></qmc>");
    CHECK(input.getOptimizer() == WFTrainDriverInput::Optimizer::RMSPROP);
    CHECK(wftrain::optimizerName(input.getOptimizer()) == "rmsprop");
    CHECK(input.getLearningRate() == 0.02);
    CHECK(input.getRMSDecay() == 0.8);
  }

  SECTION("AMSGrad accepts its Adam moment controls")
  {
    const auto input = parseTrainingInput(
        "<qmc method=\"wf_train\"><training optimizer=\"amsgrad\" "
        "learning_rate=\"0.01\" adam_beta1=\"0.8\" adam_beta2=\"0.95\" "
        "epsilon=\"1e-7\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"lm\" damping=\"0.02\" "
        "krylov_relative_tolerance=\"2e-6\" lm_overlap_shift=\"0.03\" "
        "lm_normalization_xi=\"0.4\" lm_maximum_subspace_dimension=\"12\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"alm\" damping=\"0.03\" "
        "krylov_relative_tolerance=\"3e-6\" lm_inner_iterations=\"9\" "
        "lm_inner_restart_length=\"5\" lm_inner_relative_tolerance=\"0.02\"/></qmc>");
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
        "<qmc method=\"wf_train\"><training optimizer=\"kfac\" damping=\"0.04\" "
        "kfac_fallback_damping=\"0.006\"/></qmc>");
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

  rejects("<qmc method=\"wf_train\"/>");
  rejects("<qmc method=\"wf_train\"><training/><training/></qmc>");
  rejects("<qmc method=\"wf_train\"><training mystery=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training><optimizer/></training></qmc>");
  rejects("<qmc method=\"wf_train\"><training objective=\"overlap\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training teacher=\"teacher\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training objective=\"determinant_overlap\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training objective=\"determinant_overlap\" "
          "teacher=\"teacher\" target_provenance=\"x\" optimizer=\"sr\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training objective=\"determinant_overlap\" "
          "teacher=\"teacher\" target_provenance=\"x\" "
          "overlap_weighting=\"coefficient_squared\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"unknown\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training parameter_chunk_size=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training parameter_chunk_size=\"1k\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training learning_rate=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training learning_rate=\"nan\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training learning_rate=\"inf\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training adam_beta1=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training adam_beta2=\"-0.1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training epsilon=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training damping=\"1e-3\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" adam_beta1=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" center_shift=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"minsr\" damping=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"minsr\" damping=\"nan\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"minsr\" damping_multiplier=\"2\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"spring\" maximum_metric_norm=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"spring\" damping=\"-1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"spring\" spring_history_decay=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" damping_multiplier=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" minimum_retry_damping=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" maximum_damping_attempts=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" maximum_update_norm=\"-1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" krylov_relative_tolerance=\"0\" "
          "krylov_absolute_tolerance=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" krylov_maximum_iterations=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" "
          "krylov_maximum_operator_applications=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" krylov_maximum_seconds=\"-1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" krylov_maximum_seconds=\"inf\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training maximum_retained_bytes=\"-1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training provider=\"\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" pii_shift_mode=\"fixed\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" pii_shift_mode=\"explicit\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" pii_explicit_tau=\"-1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" pii_shift_mode=\"explicit\" "
          "pii_explicit_tau=\"-1\" pii_energy_undershoot=\"0.1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" damping=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" maximum_damping_attempts=\"2\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" maximum_damping_retries=\"65\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"pii\" gmres_restart_length=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"minpii\" center_shift=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"minpii\" adam_beta1=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sr\" maximum_damping_retries=\"2\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sgd\" adam_beta1=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"sgd\" momentum_decay=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"momentum_sgd\" rms_decay=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"rmsprop\" adam_beta1=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"rmsprop\" rms_decay=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"amsgrad\" rms_decay=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"momentum_sgd\" momentum_decay=\"1\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"lm\" damping=\"0.02\" "
          "lm_overlap_shift=\"0\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"lm\" damping=\"0.02\" krylov_relative_tolerance=\"1e-6\" "
          "lm_inner_iterations=\"3\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"lm\" damping=\"0.02\" krylov_relative_tolerance=\"1e-6\" "
          "maximum_damping_attempts=\"2\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"alm\" damping=\"0.02\" krylov_relative_tolerance=\"1e-6\" "
          "spring_history_decay=\"0.8\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"kfac\" damping=\"0.02\" "
          "krylov_relative_tolerance=\"1e-6\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training optimizer=\"kfac\" damping=\"0.02\" "
          "kfac_fallback_damping=\"0\"/></qmc>");
}

} // namespace qmcplusplus
