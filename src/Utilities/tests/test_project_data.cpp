//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2017 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>


#include "OhmmsData/Libxml2Doc.h"
#include "ProjectData.h"


#include <stdio.h>
#include <string>
#include <sstream>


using std::string;

namespace qmcplusplus
{
TEST_CASE("ProjectData", "[ohmmsapp]")
{
  Communicate* c;
  c = OHMMS::Controller;


  ProjectData proj1;
  // If no name given, it gets set to time and date
  //   and the title is set equal to the name
  REQUIRE(proj1.getTitle().size() > 0);

  ProjectData proj2("asample");
  REQUIRE(proj2.getSeriesIndex() == 0);
  proj2.advance();
  REQUIRE(proj2.getSeriesIndex() == 1);
  REQUIRE(proj2.getTitle() == std::string("asample"));

  proj2.setCommunicator(c);
  std::stringstream o2;
  proj2.get(o2);
}

TEST_CASE("ProjectData::put no series", "[ohmmsapp]")
{
  ProjectData proj;

  const char* xml_input = R"(<project id="test1"></project>)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(xml_input));

  xmlNodePtr root = doc.getRoot();

  proj.put(root);
  REQUIRE(proj.getSeriesIndex() == 0);
  REQUIRE(proj.getTitle() == std::string("test1"));
}

TEST_CASE("ProjectData::put with series", "[ohmmsapp]")
{
  ProjectData proj;

  const char* xml_input = R"(<project id="test1" series="1"></project>)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(xml_input));

  xmlNodePtr root = doc.getRoot();

  proj.put(root);
  REQUIRE(proj.getSeriesIndex() == 1);
  REQUIRE(proj.getTitle() == std::string("test1"));

  // host and date nodes get added for output to the .cont.xml file
}

TEST_CASE("ProjectData::TestDriverVersion", "[ohmmsapp]")
{
  using DV = ProjectData::DriverVersion;
  SECTION("driver version batch")
  {
    ProjectData proj;
    REQUIRE(proj.getDriverVersion() == DV::BATCH);

    const char* xml_input = R"(
      <project id="test1" series="1">
        <parameter name='driver_version'>
          batch
        </parameter>
      </project>
      )";
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(xml_input));

    xmlNodePtr root = doc.getRoot();

    proj.put(root);
    REQUIRE(proj.getSeriesIndex() == 1);
    REQUIRE(proj.getDriverVersion() == DV::BATCH);
  }
  SECTION("driver version legacy")
  {
    ProjectData proj;

    const char* xml_input = R"(
      <project id="test1" series="1">
        <parameter name='driver_version'>
          legacy
        </parameter>
      </project>
      )";
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(xml_input));

    xmlNodePtr root = doc.getRoot();

    proj.put(root);
    REQUIRE(proj.getSeriesIndex() == 1);
    REQUIRE(proj.getDriverVersion() == DV::LEGACY);
  }
  SECTION("driver version bad value")
  {
    ProjectData proj;

    const char* xml_input = R"(
      <project id=" test1 " series=" 1 ">
        <parameter name = 'driver_version' >
          linear
        </parameter>
      </project>)";
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(xml_input));

    xmlNodePtr root = doc.getRoot();

    REQUIRE_THROWS(proj.put(root));
  }

  // host and date nodes get added for output to the .cont.xml file
}

TEST_CASE("ProjectData::TestIsComplex", "[ohmmsapp]")
{
  ProjectData proj;
#ifdef QMC_COMPLEX
  REQUIRE(proj.isComplex());
#else
  REQUIRE(!proj.isComplex());
#endif
}

TEST_CASE("ProjectData::timer_level", "[ohmmsapp]")
{
  ProjectData project;
  REQUIRE_FALSE(project.getTimerLevel());
  REQUIRE_FALSE(ProjectData::readTimerLevel(nullptr));

  for (const std::string level : {"none", "coarse", "medium", "fine"})
  {
    CAPTURE(level);
    Libxml2Document doc;
    REQUIRE(doc.parseFromString("<project id='timers' series='0'><parameter name='driver_version'>batch</parameter>"
                               "<parameter name='timer_level'>\n  " + level + " \t</parameter></project>"));
    const auto root = doc.getRoot();
    const auto parameter = root->children->next;
    const std::string original_content = XMLNodeString(parameter);
    REQUIRE(ProjectData::readTimerLevel(root) == level);
    REQUIRE(getXMLAttributeValue(root, "series") == "0");
    REQUIRE(std::string(XMLNodeString(parameter)) == original_content);
    project.put(root);
    REQUIRE(project.getTimerLevel() == level);
    project.advance();
    REQUIRE(ProjectData::readTimerLevel(root) == level);
    REQUIRE(std::string(XMLNodeString(parameter)) == original_content);
  }

  Libxml2Document absent;
  REQUIRE(absent.parseFromString("<project id='absent'><parameter name='driver_version'>legacy</parameter></project>"));
  REQUIRE_FALSE(ProjectData::readTimerLevel(absent.getRoot()));
  project.put(absent.getRoot());
  REQUIRE_FALSE(project.getTimerLevel());
}

TEST_CASE("ProjectData::invalid timer_level", "[ohmmsapp]")
{
  for (const std::string value : {"", " ", "invalid", "Fine", "medium fine"})
  {
    CAPTURE(value);
    Libxml2Document doc;
    REQUIRE(doc.parseFromString("<project id='invalid'><parameter name='timer_level'>" + value +
                               "</parameter></project>"));
    ProjectData project;
    REQUIRE_THROWS_WITH(ProjectData::readTimerLevel(doc.getRoot()),
                        "Invalid project timer_level '" + (value == " " ? "" : value) +
                            "': expected none, coarse, medium, or fine");
    REQUIRE_THROWS(project.put(doc.getRoot()));
  }

  for (const std::string second : {"fine", "coarse"})
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString("<project id='duplicate'><parameter name='timer_level'>fine</parameter>"
                               "<parameter name='timer_level'>" + second + "</parameter></project>"));
    ProjectData project;
    REQUIRE_THROWS_WITH(ProjectData::readTimerLevel(doc.getRoot()), "Duplicate project parameter: timer_level");
    REQUIRE_THROWS(project.put(doc.getRoot()));
  }
}

} // namespace qmcplusplus
