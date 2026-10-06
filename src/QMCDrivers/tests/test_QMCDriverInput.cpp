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

#include <string>
#include <utility>

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

TEST_CASE("QMCDriverInput batch memory policy", "[drivers][batch_memory]")
{
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(R"(<qmc method="vmc">
    <batch_memory host_budget="2GiB" device_budget="0B" value_tile="8"
                  full_vgl_tile="auto" active_gradient_tile="3" ecp_outer_tile="64"/>
  </qmc>)"));

  QMCDriverInput input;
  input.readXML(doc.getRoot());
  REQUIRE(input.get_batch_memory_input());
  const BatchMemoryPolicy& policy = input.get_batch_memory_input()->getPolicy();
  CHECK(policy.host_budget == (std::size_t{2} << 30));
  CHECK(policy.device_budget == 0);
  CHECK(policy.tiles.value.fixedCapacity() == 8);
  CHECK(policy.tiles.full_vgl.isAutomatic());
  CHECK(policy.tiles.active_gradient.fixedCapacity() == 3);
  CHECK(policy.tiles.ecp_outer.fixedCapacity() == 64);

  SECTION("copy and move preserve the parsed structured input")
  {
    QMCDriverInput copied(input);
    REQUIRE(copied.get_batch_memory_input());
    CHECK(copied.get_batch_memory_input()->getPolicy() == policy);

    QMCDriverInput moved(std::move(copied));
    REQUIRE(moved.get_batch_memory_input());
    CHECK(moved.get_batch_memory_input()->getPolicy() == policy);

    QMCDriverInput copy_assigned;
    copy_assigned = input;
    REQUIRE(copy_assigned.get_batch_memory_input());
    CHECK(copy_assigned.get_batch_memory_input()->getPolicy() == policy);

    QMCDriverInput move_assigned;
    move_assigned = std::move(copy_assigned);
    REQUIRE(move_assigned.get_batch_memory_input());
    CHECK(move_assigned.get_batch_memory_input()->getPolicy() == policy);
  }

  SECTION("a later read resets an absent section-local policy")
  {
    Libxml2Document second_doc;
    REQUIRE(second_doc.parseFromString(R"(<qmc method="vmc"><parameter name="steps">1</parameter></qmc>)"));
    input.readXML(second_doc.getRoot());
    CHECK_FALSE(input.get_batch_memory_input());
  }
}

TEST_CASE("QMCDriverInput batch memory policy rejects malformed input", "[drivers][batch_memory]")
{
  auto check_rejected = [](const std::string& batch_memory) {
    Libxml2Document doc;
    const std::string xml = "<qmc method=\"vmc\">" + batch_memory + "</qmc>";
    REQUIRE(doc.parseFromString(xml));
    QMCDriverInput input;
    CHECK_THROWS(input.readXML(doc.getRoot()));
  };

  check_rejected(R"(<batch_memory host_budget="2GB"/>)");
  check_rejected(R"(<batch_memory host_budget="-1B"/>)");
  check_rejected(R"(<batch_memory host_budget="1.5GiB"/>)");
  check_rejected(R"(<batch_memory host_budget="1 GiB"/>)");
  check_rejected(R"(<batch_memory host_budget="1GiB trailing"/>)");
  check_rejected(R"(<batch_memory host_budget="999999999999999999999999999999999B"/>)");
  check_rejected(R"(<batch_memory value_tile="0"/>)");
  check_rejected(R"(<batch_memory value_tile="Auto"/>)");
  check_rejected(R"(<batch_memory value_tile="1.0"/>)");
  check_rejected(R"(<batch_memory value_tile="999999999999999999999999999999999"/>)");
  check_rejected(R"(<batch_memory mystery="1B"/>)");
  check_rejected(R"(<batch_memory type="unexpected"/>)");
  check_rejected(R"(<batch_memory><unexpected/></batch_memory>)");
  check_rejected(R"(<batch_memory>unexpected</batch_memory>)");
  check_rejected(R"(<batch_memory/><batch_memory/>)");
}

TEST_CASE("QMCDriverInput batch memory zero and omitted values", "[drivers][batch_memory]")
{
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(R"(<qmc method="vmc"><batch_memory host_budget="0B"/></qmc>)"));

  QMCDriverInput input;
  input.readXML(doc.getRoot());
  REQUIRE(input.get_batch_memory_input());
  const BatchMemoryPolicy& policy = input.get_batch_memory_input()->getPolicy();
  CHECK(policy.host_budget == 0);
  CHECK_FALSE(policy.device_budget);
  CHECK(policy.tiles.value.isAutomatic());
  CHECK(policy.tiles.full_vgl.isAutomatic());
  CHECK(policy.tiles.active_gradient.isAutomatic());
  CHECK(policy.tiles.ecp_outer.isAutomatic());
}
} // namespace qmcplusplus
