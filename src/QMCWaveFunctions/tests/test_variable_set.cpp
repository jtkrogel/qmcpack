//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "VariableSet.h"
#include "io/hdf/hdf_archive.h"

#include <stdio.h>
#include <random>
#include <string>

using std::string;

namespace optimize
{
TEST_CASE("VariableSet empty", "[optimize]")
{
  VariableSet vs;

  REQUIRE(vs.is_optimizable() == false);
  REQUIRE(vs.size_of_active() == 0);
  REQUIRE(vs.find("something") == vs.end());
  REQUIRE(vs.getIndex("something") == -1);
}

TEST_CASE("VariableSet one", "[optimize]")
{
  VariableSet vs;
  VariableSet::real_type first_val(1.123456789);
  vs.insert("first", first_val);
  vs.resetIndex();

  REQUIRE(vs.is_optimizable() == true);
  REQUIRE(vs.size_of_active() == 1);
  CHECK(vs.getIndex("first") == 0);
  CHECK(vs.name(0) == "first");
  double first_val_real = 1.123456789;
  CHECK(vs[0] == Approx(first_val_real));

  std::ostringstream o;
  vs.print(o, 0, false);
  //app_log() << o.str() << std::endl;
  REQUIRE(o.str() == "first                 1.123457e+00 0  ON 0\n");

  std::ostringstream o2;
  vs.print(o2, 1, true);
  //app_log() << o2.str() << std::endl;

  char formatted_output[] = "  Name                        Value Type Use Index\n"
                            " ----- ---------------------------- ---- --- -----\n"
                            " first                 1.123457e+00    0  ON     0\n";


  REQUIRE(o2.str() == formatted_output);

  VariableSet vs2;
  VariableSet::real_type second_val(2.23);
  vs2.insert("second", second_val);
  vs.insertFrom(vs2);
  vs.resetIndex();
  REQUIRE(vs.size_of_active() == 2);
  CHECK(vs.name(1) == "second");
  CHECK(vs2.findIndexOfFirstParam(vs) == 1);

  VariableSet vs3;
  CHECK(vs3.findIndexOfFirstParam(vs) == -1);
}

TEST_CASE("VariableSet output", "[optimize]")
{
  VariableSet vs;
  VariableSet::real_type first_val(11234.56789);
  VariableSet::real_type second_val(0.000256789);
  VariableSet::real_type third_val(-1.2);
  vs.insert("s", first_val);
  vs.insert("second", second_val);
  vs.insert("really_long_name", third_val);
  vs.resetIndex();

  std::ostringstream o;
  vs.print(o, 0, true);
  //app_log() << o.str() << std::endl;

  char formatted_output[] = "            Name                        Value Type Use Index\n"
                            "---------------- ---------------------------- ---- --- -----\n"
                            "               s                 1.123457e+04    0  ON     0\n"
                            "          second                 2.567890e-04    0  ON     1\n"
                            "really_long_name                -1.200000e+00    0  ON     2\n";

  REQUIRE(o.str() == formatted_output);
}

TEST_CASE("VariableSet HDF output and input", "[optimize]")
{
  VariableSet vs;
  VariableSet::real_type first_val(11234.56789);
  VariableSet::real_type second_val(0.000256789);
  VariableSet::real_type third_val(-1.2);
  vs.insert("s", first_val);
  vs.insert("second", second_val);
  vs.insert("really_really_really_long_name", third_val);
  qmcplusplus::hdf_archive hout;
  vs.writeToHDF("vp.h5", hout);

  VariableSet vs2;
  vs2.insert("s", 0.0);
  vs2.insert("second", 0.0);
  qmcplusplus::hdf_archive hin;
  vs2.readFromHDF("vp.h5", hin);
  CHECK(vs2.find("s")->second == Approx(first_val));
  CHECK(vs2.find("second")->second == Approx(second_val));
  // This value as in the file, but not in the VariableSet that loaded the file,
  // so the value does not get added.
  CHECK(vs2.find("really_really_really_long_name") == vs2.end());
}

TEST_CASE("VariableSet duplicate and disabled contracts", "[optimize]")
{
  VariableSet variables;
  variables.insert("alpha", 1.0, true, LINEAR_P);

  // A later insertion with the same name does not replace either the value
  // or parameter type established by the first insertion.
  variables.insert("alpha", 2.0, true, OTHER_P);
  REQUIRE(variables.size() == 1);
  CHECK(variables[0] == Approx(1.0));
  CHECK(variables.getType(0) == LINEAR_P);

  // Disabling an existing variable is sticky: a subsequent enabled insertion
  // does not silently opt the same name back in.
  variables.insert("alpha", 3.0, false, BACKFLOW_P);
  variables.insert("alpha", 4.0, true, OTHER_P);
  variables.resetIndex();
  CHECK(variables.where(0) == -1);
  CHECK(variables.size_of_active() == 0);

  // map-like insertion creates a stored, but disabled, variable.
  variables["implicit"] = 5.0;
  REQUIRE(variables.size() == 2);
  CHECK(variables.getLoc("implicit") == 1);
  CHECK(variables.where(1) == -1);
  CHECK(variables[1] == Approx(5.0));
}

TEST_CASE("VariableSet insertFrom preserves ordering and metadata", "[optimize]")
{
  VariableSet destination;
  destination.insert("first", 1.0, true, OTHER_P);
  destination.insert("overlap", 2.0, true, LINEAR_P);
  destination.resetIndex();

  VariableSet source;
  source.insert("overlap", 20.0, true, BACKFLOW_P);
  source.insert("disabled_tail", 30.0, false, SPO_P);
  source.resetIndex();

  destination.insertFrom(source);
  REQUIRE(destination.size() == 3);
  CHECK(destination.name(0) == "first");
  CHECK(destination.name(1) == "overlap");
  CHECK(destination.name(2) == "disabled_tail");

  // Existing variables receive the incoming value while retaining their
  // established type and enabled state. New variables retain source metadata.
  CHECK(destination[1] == Approx(20.0));
  CHECK(destination.getType(1) == LINEAR_P);
  CHECK(destination[2] == Approx(30.0));
  CHECK(destination.getType(2) == SPO_P);

  destination.resetIndex();
  CHECK(destination.where(0) == 0);
  CHECK(destination.where(1) == 1);
  CHECK(destination.where(2) == -1);
  CHECK(destination.size_of_active() == 2);
}

TEST_CASE("VariableSet global mapping copy move and clear", "[optimize]")
{
  VariableSet selected;
  selected.insert("leading", -1.0);
  selected.insert("alpha", 1.0);
  selected.insert("beta", 2.0);
  selected.resetIndex();

  VariableSet local;
  local.insert("beta", 20.0, true, LINEAR_P);
  local.insert("missing", 30.0, true, SPO_P);
  local.insert("alpha", 10.0, true, OTHER_P);
  local.getIndex(selected);

  CHECK(local.where(0) == 2);
  CHECK(local.where(1) == -1);
  CHECK(local.where(2) == 1);
  CHECK(local.size_of_active() == 2);

  VariableSet copied(local);
  CHECK(copied.getLoc("beta") == 0);
  CHECK(copied.getIndex("beta") == 2);
  CHECK(copied.getType(0) == LINEAR_P);

  VariableSet moved(std::move(copied));
  CHECK(moved.getLoc("alpha") == 2);
  CHECK(moved.getIndex("alpha") == 1);
  CHECK(moved.getType(2) == OTHER_P);

  CHECK(copied.size() == 0);
  CHECK(copied.size_of_active() == 0);
  CHECK(copied.find("alpha") == copied.end());
  moved.clear();
  CHECK(moved.size() == 0);
  CHECK(moved.size_of_active() == 0);
  CHECK(moved.find("alpha") == moved.end());

  moved.insert("after_clear", 7.0);
  moved.resetIndex();
  CHECK(moved.getLoc("after_clear") == 0);
  CHECK(moved.getIndex("after_clear") == 0);

  VariableSet move_assigned;
  move_assigned.insert("discarded", -1.0);
  move_assigned = std::move(moved);
  CHECK(move_assigned.getLoc("after_clear") == 0);
  CHECK(move_assigned.getIndex("after_clear") == 0);
  CHECK(moved.size() == 0);
  CHECK(moved.size_of_active() == 0);

  moved.insert("reused_source", 8.0);
  moved.resetIndex();
  CHECK(moved.getIndex("reused_source") == 0);
}

TEST_CASE("VariableSet bulk insertion matches scalar insertion", "[optimize]")
{
  constexpr std::size_t variable_count = 257;
  std::mt19937 generator(19);
  std::uniform_real_distribution<double> value_distribution(-2.0, 2.0);

  std::vector<VariableSet::pair_type> variables;
  std::vector<bool> enabled;
  std::vector<int> types;
  variables.reserve(variable_count + 1);
  enabled.reserve(variable_count + 1);
  types.reserve(variable_count + 1);
  for (std::size_t variable_index = 0; variable_index < variable_count; ++variable_index)
  {
    variables.emplace_back("random_" + std::to_string(variable_index), value_distribution(generator));
    enabled.push_back(generator() % 5 != 0);
    types.push_back(static_cast<int>(generator() % (BACKFLOW_P + 1)));
  }

  // Include a deterministic duplicate to exercise the same first-value-wins
  // and sticky-disable semantics through the batched path.
  variables.emplace_back("random_17", 123.0);
  enabled.push_back(false);
  types.push_back(BACKFLOW_P);

  VariableSet scalar;
  scalar.reserve(variables.size());
  for (std::size_t variable_index = 0; variable_index < variables.size(); ++variable_index)
    scalar.insert(variables[variable_index].first, variables[variable_index].second, enabled[variable_index],
                  types[variable_index]);

  VariableSet bulk;
  bulk.reserve(4 * variables.size());
  CHECK(bulk.size() == 0);
  bulk.insertBulk(variables, enabled, types);

  scalar.resetIndex();
  bulk.resetIndex();
  REQUIRE(bulk.size() == scalar.size());
  REQUIRE(bulk.size_of_active() == scalar.size_of_active());
  for (std::size_t variable_index = 0; variable_index < scalar.size(); ++variable_index)
  {
    CHECK(bulk.name(variable_index) == scalar.name(variable_index));
    CHECK(bulk[variable_index] == Approx(scalar[variable_index]));
    CHECK(bulk.getType(variable_index) == scalar.getType(variable_index));
    CHECK(bulk.where(variable_index) == scalar.where(variable_index));
    CHECK(bulk.getLoc(scalar.name(variable_index)) == static_cast<int>(variable_index));
    CHECK(bulk.getIndex(scalar.name(variable_index)) == scalar.getIndex(scalar.name(variable_index)));
  }

  std::vector<VariableSet::pair_type> malformed{{"one", 1.0}, {"two", 2.0}};
  CHECK_THROWS_AS(bulk.insertBulk(std::move(malformed), std::vector<bool>{true}, std::vector<int>{OTHER_P}),
                  std::invalid_argument);
}

TEST_CASE("VariableSet common-metadata bulk insertion", "[optimize]")
{
  VariableSet variables;
  variables.insertBulk({{"alpha", 1.0}, {"beta", 2.0}}, false, LINEAR_P);
  REQUIRE(variables.size() == 2);
  CHECK(variables.getType(0) == LINEAR_P);
  CHECK(variables.getType(1) == LINEAR_P);
  CHECK(variables.where(0) == -1);
  CHECK(variables.where(1) == -1);
}

TEST_CASE("VariableSet bounded summary output", "[optimize]")
{
  VariableSet variables;
  for (int variable_index = 0; variable_index < 10; ++variable_index)
    variables.insert("parameter_" + std::to_string(variable_index), variable_index);
  variables.resetIndex();

  std::ostringstream output;
  variables.printSummary(output, 2, 2);
  const std::string summary = output.str();
  CHECK(summary.find("  10 stored parameters, 10 active") != std::string::npos);
  CHECK(summary.find("parameter_0") != std::string::npos);
  CHECK(summary.find("parameter_1") != std::string::npos);
  CHECK(summary.find("6 parameters omitted") != std::string::npos);
  CHECK(summary.find("parameter_8") != std::string::npos);
  CHECK(summary.find("parameter_9") != std::string::npos);
  CHECK(summary.find("parameter_2") == std::string::npos);
  CHECK(summary.find("parameter_7") == std::string::npos);
}


} // namespace optimize
