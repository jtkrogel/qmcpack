//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Leon Otis, leon_otis@berkeley.edu University, University of California Berkeley
//		      Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Leon Otis, leon_otis@berkeley.edu University, University of California Berkeley
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/Optimizers/DescentEngine.h"
#include "VariableSet.h"
#include "Message/Communicate.h"

#include <filesystem>
#include <unistd.h>

namespace qmcplusplus
{

using FullPrecValueType = qmcplusplus::QMCTraits::FullPrecValueType;
using ValueType         = qmcplusplus::QMCTraits::ValueType;

#if !defined(MIXED_PRECISION)
///This provides a basic test of the descent engine's parameter update algorithm
TEST_CASE("DescentEngine RMSprop update", "[drivers][descent]")
{
  Communicate* c = OHMMS::Controller;


  const std::string engine_input("<tmp> </tmp>");

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(engine_input));

  xmlNodePtr fakeXML = doc.getRoot();

  std::unique_ptr<DescentEngine> descentEngineObj = std::make_unique<DescentEngine>(c, fakeXML);

  optimize::VariableSet myVars;

  //Two fake parameters are specified
  optimize::VariableSet::real_type first_param(1.0);
  optimize::VariableSet::real_type second_param(-2.0);

  myVars.insert("first", first_param);
  myVars.insert("second", second_param);

  std::vector<ValueType> LDerivs;

  //Corresponding fake derivatives are specified and given to the engine
  ValueType first_deriv  = 5;
  ValueType second_deriv = 1;

  LDerivs.push_back(first_deriv);
  LDerivs.push_back(second_deriv);

  descentEngineObj->setDerivs(LDerivs);

  descentEngineObj->setupUpdate(myVars);

  descentEngineObj->storeDerivRecord();
  descentEngineObj->updateParameters();

  std::vector<ValueType> results = descentEngineObj->retrieveNewParams();

  app_log() << "Descent engine test of parameter update" << std::endl;
  app_log() << "First parameter: " << results[0] << std::endl;
  app_log() << "Second parameter: " << results[1] << std::endl;

  //The engine should update the parameters using the generic default step size of .001 and obtain these values.
  CHECK(std::real(results[0]) == Approx(.995));
  CHECK(std::real(results[1]) == Approx(-2.001));

  //Provide fake data to test mpi_unbiased_ratio_of_means
  int n              = 2;
  ValueType mean     = 0;
  ValueType variance = 0;
  ValueType stdErr   = 0;

  std::vector<ValueType> weights;
  weights.push_back(1.0);
  weights.push_back(1.0);
  std::vector<ValueType> numerSamples;
  numerSamples.push_back(-2.0);
  numerSamples.push_back(-2.0);
  std::vector<ValueType> denomSamples;
  denomSamples.push_back(1.0);
  denomSamples.push_back(1.0);

  descentEngineObj->mpi_unbiased_ratio_of_means(n, weights, numerSamples, denomSamples, mean, variance, stdErr);
  app_log() << "Descent engine test of mpi_unbiased_ratio_of_means" << std::endl;
  app_log() << "Mean: " << mean << std::endl;
  app_log() << "Variance: " << variance << std::endl;
  app_log() << "Standard Error: " << stdErr << std::endl;

  //mpi_unbiased_ratio_of_means should calculate the mean, variance, and standard error and obtain the values below
  CHECK(std::real(mean) == Approx(-2.0));
  CHECK(std::real(variance) == Approx(0.0));
  CHECK(std::real(stdErr) == Approx(0.0));
}

/// Verify bounded history and equivalent continuation from an optimizer checkpoint.
TEST_CASE("DescentEngine ADAM restart", "[drivers][descent]")
{
  Communicate* communicator = OHMMS::Controller;
  const std::filesystem::path state_path = std::filesystem::temp_directory_path() /
      ("qmcpack_descent_state_" + std::to_string(static_cast<long long>(getpid())) + ".h5");
  std::error_code error;
  std::filesystem::remove(state_path, error);

  Libxml2Document document;
  const std::string input = "<tmp><parameter name=\"flavor\">ADAM</parameter>"
      "<parameter name=\"descent_state_file\">" +
      state_path.string() + "</parameter></tmp>";
  REQUIRE(document.parseFromString(input));

  optimize::VariableSet variables;
  variables.insert("pf_pf_0000000", 1.0);
  variables.insert("pf_pf_0000001", -2.0);

  DescentEngine uninterrupted(communicator, document.getRoot());
  uninterrupted.setupUpdate(variables);
  auto take_step = [](DescentEngine& engine, std::vector<ValueType> derivatives) {
    engine.setDerivs(derivatives);
    engine.storeDerivRecord();
    engine.updateParameters();
  };
  take_step(uninterrupted, {5.0, 1.0});
  take_step(uninterrupted, {-2.0, 3.0});
  CHECK(uninterrupted.getDerivativeHistorySize() == 2);

  uninterrupted.writeConfiguredState();
  take_step(uninterrupted, {0.25, -0.75});
  const std::vector<ValueType> expected = uninterrupted.retrieveNewParams();

  DescentEngine restarted(communicator, document.getRoot());
  optimize::VariableSet restart_variables;
  const std::vector<ValueType> checkpoint_parameters = restarted.retrieveNewParams();
  restart_variables.insert("pf_pf_0000000", std::real(checkpoint_parameters[0]));
  restart_variables.insert("pf_pf_0000001", std::real(checkpoint_parameters[1]));
  restarted.setupUpdate(restart_variables);
  take_step(restarted, {0.25, -0.75});

  const std::vector<ValueType> actual = restarted.retrieveNewParams();
  REQUIRE(actual.size() == expected.size());
  for (std::size_t parameter = 0; parameter < actual.size(); ++parameter)
  {
    CHECK(std::real(actual[parameter]) == Approx(std::real(expected[parameter])).epsilon(1e-14));
    CHECK(std::imag(actual[parameter]) == Approx(std::imag(expected[parameter])).epsilon(1e-14));
  }
  CHECK(restarted.getDescentNum() == uninterrupted.getDescentNum());
  CHECK(restarted.getDerivativeHistorySize() == 2);

  std::filesystem::remove(state_path, error);
}
#endif
} // namespace qmcplusplus
