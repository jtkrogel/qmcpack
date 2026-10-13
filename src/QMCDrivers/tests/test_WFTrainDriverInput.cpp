//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"

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
        "adam_beta2=\"0.95\" epsilon=\"1e-7\" checkpoint_path=\"next.h5\" restart_path=\"previous.h5\" "
        "provider=\"neural\"/></qmc>");
    CHECK(input.getParameterChunkSize() == 2048);
    CHECK(input.getLearningRate() == 2.0e-4);
    CHECK(input.getAdamBeta1() == 0.8);
    CHECK(input.getAdamBeta2() == 0.95);
    CHECK(input.getEpsilon() == 1.0e-7);
    CHECK(input.getCheckpointPath() == "next.h5");
    CHECK(input.getRestartPath() == "previous.h5");
    CHECK(input.getProviderID() == "neural");
  }
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
  rejects("<qmc method=\"wf_train\"><training optimizer=\"rmsprop\"/></qmc>");
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
  rejects("<qmc method=\"wf_train\"><training checkpoint_path=\"same.h5\" restart_path=\"same.h5\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training provider=\"\"/></qmc>");
}

} // namespace qmcplusplus
