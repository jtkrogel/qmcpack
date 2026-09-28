//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCDrivers/QMCDriverInput.h"
#include "EstimatorInputDelegates.h"
#include "QMCDrivers/tests/ValidQMCInputSections.h"
#include "OhmmsData/Libxml2Doc.h"

namespace qmcplusplus
{
TEST_CASE("QMCDriverInput Instantiation", "[drivers]") { QMCDriverInput driver_input; }

TEST_CASE("QMCDriverInput readXML", "[drivers]")
{
  auto xml_test = [](const char* driver_xml) {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(driver_xml));
    xmlNodePtr node = doc.getRoot();
    QMCDriverInput qmcdriver_input;
    qmcdriver_input.readXML(node);
    REQUIRE(qmcdriver_input.get_qmc_method().size() > 0);
  };

  std::for_each(testing::valid_vmc_input_sections.begin() + testing::valid_vmc_input_vmc_batch_index,
                testing::valid_vmc_input_sections.end(), xml_test);

  std::for_each(testing::valid_dmc_input_sections.begin() + testing::valid_dmc_input_dmc_batch_index,
                testing::valid_dmc_input_sections.end(), xml_test);
}

TEST_CASE("QMCDriverInput multi-timestep readXML", "[drivers]")
{
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(R"(
    <qmc method="dmc" move="pbyp">
      <parameter name="blocks">2</parameter>
      <parameter name="mts_cycles">3</parameter>
      <parameter name="timestep">0.04 0.02 0.02 0.01</parameter>
    </qmc>)"));

  QMCDriverInput input;
  input.readXML(doc.getRoot());

  const auto& time_steps = input.get_time_steps();
  REQUIRE(time_steps.size() == 4);
  CHECK(time_steps[0] == Approx(0.04));
  CHECK(time_steps[1] == Approx(0.02));
  CHECK(time_steps[2] == Approx(0.02));
  CHECK(time_steps[3] == Approx(0.01));
  CHECK(input.get_tau() == Approx(0.04));
  CHECK(input.get_mts_cycles() == 3);
  CHECK_FALSE(input.has_steps_input());

  Libxml2Document steps_doc;
  REQUIRE(steps_doc.parseFromString(R"(
    <qmc method="dmc" move="pbyp">
      <parameter name="steps">4</parameter>
      <parameter name="timestep">0.04 0.02</parameter>
    </qmc>)"));
  QMCDriverInput steps_input;
  steps_input.readXML(steps_doc.getRoot());
  CHECK(steps_input.has_steps_input());

  Libxml2Document vmc_doc;
  REQUIRE(vmc_doc.parseFromString(R"(
    <qmc method="vmc" move="pbyp">
      <parameter name="timestep">0.04 0.02</parameter>
    </qmc>)"));
  QMCDriverInput vmc_input;
  CHECK_THROWS(vmc_input.readXML(vmc_doc.getRoot()));
}

} // namespace qmcplusplus
