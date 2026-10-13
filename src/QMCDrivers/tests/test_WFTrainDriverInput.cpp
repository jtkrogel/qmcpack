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
  rejects("<qmc method=\"wf_train\"><training checkpoint_path=\"same.h5\" restart_path=\"same.h5\"/></qmc>");
  rejects("<qmc method=\"wf_train\"><training provider=\"\"/></qmc>");
}

} // namespace qmcplusplus
