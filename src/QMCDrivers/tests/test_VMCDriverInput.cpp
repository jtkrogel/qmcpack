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

#include <algorithm>
#include <vector>

#include "QMCDrivers/VMC/ElectronSubsetSelector.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/tests/ValidQMCInputSections.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Utilities/FakeRandom.h"

namespace qmcplusplus
{
TEST_CASE("VMCDriverInput readXML", "[drivers]")
{
  auto xml_test = [](const char* driver_xml) {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(driver_xml));
    xmlNodePtr node = doc.getRoot();
    VMCDriverInput vmcdriver_input;
    vmcdriver_input.readXML(node);
    REQUIRE(vmcdriver_input.get_use_drift() == false);
  };

  std::for_each(testing::valid_vmc_input_sections.begin() + testing::valid_vmc_input_vmc_batch_index,
                testing::valid_vmc_input_sections.end(), xml_test);
}

TEST_CASE("VMCDriverInput move configuration", "[drivers]")
{
  auto parse_input = [](const char* driver_xml) {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(driver_xml));
    VMCDriverInput input;
    input.readXML(doc.getRoot());
    return input;
  };

  SECTION("defaults")
  {
    const VMCDriverInput input = parse_input(R"(<qmc method="vmc"/>)");
    CHECK(input.get_move_kind() == VMCDriverInput::MoveKind::PBYP);
    CHECK(input.get_electrons_per_move() == 0);
    CHECK(input.get_electron_selection() == VMCDriverInput::ElectronSelection::RANDOM);
    CHECK_FALSE(input.was_electrons_per_move_provided());
    CHECK_FALSE(input.was_electron_selection_provided());
  }

  SECTION("all-electron")
  {
    const VMCDriverInput input = parse_input(R"(<qmc method="vmc" move="alle"/>)");
    CHECK(input.get_move_kind() == VMCDriverInput::MoveKind::ALL_ELECTRON);
  }

  SECTION("random n-electron")
  {
    const VMCDriverInput input = parse_input(R"(
      <qmc method="vmc" move="n_electron">
        <parameter name="electrons_per_move">3</parameter>
        <parameter name="electron_selection">random</parameter>
      </qmc>)");
    CHECK(input.get_move_kind() == VMCDriverInput::MoveKind::N_ELECTRON);
    CHECK(input.get_electrons_per_move() == 3);
    CHECK(input.get_electron_selection() == VMCDriverInput::ElectronSelection::RANDOM);
    CHECK(input.was_electrons_per_move_provided());
    CHECK(input.was_electron_selection_provided());
  }

  SECTION("cyclic n-electron and direct parameter element")
  {
    const VMCDriverInput input = parse_input(R"(
      <qmc method="vmc" move="n_electron">
        <electrons_per_move>2</electrons_per_move>
        <electron_selection>cyclic</electron_selection>
      </qmc>)");
    CHECK(input.get_electrons_per_move() == 2);
    CHECK(input.get_electron_selection() == VMCDriverInput::ElectronSelection::CYCLIC);
  }

  SECTION("n-electron count is required and positive")
  {
    REQUIRE_THROWS(parse_input(R"(<qmc method="vmc" move="n_electron"/>)"));
    REQUIRE_THROWS(parse_input(R"(
      <qmc method="vmc" move="n_electron">
        <parameter name="electrons_per_move">0</parameter>
      </qmc>)"));
    REQUIRE_THROWS(parse_input(R"(
      <qmc method="vmc" move="n_electron">
        <parameter name="electrons_per_move">-2</parameter>
      </qmc>)"));
  }

  SECTION("subset controls require n-electron moves")
  {
    REQUIRE_THROWS(parse_input(R"(
      <qmc method="vmc" move="pbyp">
        <parameter name="electrons_per_move">1</parameter>
      </qmc>)"));
    REQUIRE_THROWS(parse_input(R"(
      <qmc method="vmc" move="alle">
        <parameter name="electron_selection">random</parameter>
      </qmc>)"));
  }

  SECTION("unknown spellings are rejected")
  {
    REQUIRE_THROWS(parse_input(R"(<qmc method="vmc" move="all"/>)"));
    REQUIRE_THROWS(parse_input(R"(<qmc method="vmc" move="PBYP"/>)"));
    REQUIRE_THROWS(parse_input(R"(
      <qmc method="vmc" move="n_electron">
        <parameter name="electrons_per_move">2</parameter>
        <parameter name="electron_selection">round_robin</parameter>
      </qmc>)"));
  }
}

TEST_CASE("ElectronSubsetSelector", "[drivers]")
{
  using Selection = VMCDriverInput::ElectronSelection;
  using IndexType = ElectronSubsetSelector::IndexType;
  FakeRandom<ElectronSubsetSelector::FullPrecisionRealType> random_gen;

  SECTION("cyclic selection advances by the subset size")
  {
    ElectronSubsetSelector selector(5, 2, Selection::CYCLIC);
    CHECK((selector.select(random_gen) == std::vector<IndexType>{0, 1}));
    CHECK((selector.select(random_gen) == std::vector<IndexType>{2, 3}));
    CHECK((selector.select(random_gen) == std::vector<IndexType>{0, 4}));
    CHECK((selector.select(random_gen) == std::vector<IndexType>{1, 2}));
    selector.reset();
    CHECK((selector.select(random_gen) == std::vector<IndexType>{0, 1}));
  }

  SECTION("random selection is sorted and without replacement")
  {
    random_gen.set_value(0.5);
    ElectronSubsetSelector selector(6, 3, Selection::RANDOM);
    const std::vector<IndexType>& selected = selector.select(random_gen);
    CHECK((selected == std::vector<IndexType>{0, 3, 4}));
    CHECK(std::is_sorted(selected.begin(), selected.end()));
    CHECK(std::adjacent_find(selected.begin(), selected.end()) == selected.end());
  }

  SECTION("full selection")
  {
    ElectronSubsetSelector selector(4, 4, Selection::CYCLIC);
    CHECK((selector.select(random_gen) == std::vector<IndexType>{0, 1, 2, 3}));
    CHECK((selector.select(random_gen) == std::vector<IndexType>{0, 1, 2, 3}));
  }

  SECTION("invalid sizes and random samples")
  {
    REQUIRE_THROWS_AS(ElectronSubsetSelector(0, 0, Selection::CYCLIC), std::invalid_argument);
    REQUIRE_THROWS_AS(ElectronSubsetSelector(4, 0, Selection::CYCLIC), std::invalid_argument);
    REQUIRE_THROWS_AS(ElectronSubsetSelector(4, 5, Selection::RANDOM), std::invalid_argument);
    REQUIRE_THROWS_AS(ElectronSubsetSelector(4, 2, static_cast<Selection>(-1)), std::invalid_argument);

    ElectronSubsetSelector selector(4, 2, Selection::RANDOM);
    random_gen.set_value(1.0);
    REQUIRE_THROWS_AS(selector.select(random_gen), std::domain_error);
  }
}


} // namespace qmcplusplus
