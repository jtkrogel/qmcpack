//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Leon Otis, leon_otis@berkeley.edu, University of California Berkeley
//
// File created by: Leon Otis, leon_otis@berkeley.edu, University of California Berkeley
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <array>
#include <complex>

#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/Optimizers/DescentEngine.h"
#include "QMCDrivers/WFOpt/EngineHandle.h"
#include "Message/Communicate.h"

namespace qmcplusplus
{

using FullPrecValueType = qmcplusplus::QMCTraits::FullPrecValueType;
using ValueType         = qmcplusplus::QMCTraits::ValueType;

///This provides a basic test of constructing an EngineHandle object and checking information in it
TEST_CASE("EngineHandle construction", "[drivers]")
{
  Communicate* c = OHMMS::Controller;


  const std::string engine_input("<tmp> </tmp>");

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(engine_input));

  xmlNodePtr fakeXML = doc.getRoot();

  DescentEngine descentEngineObj = DescentEngine(c, fakeXML);

  descentEngineObj.processXML(fakeXML);

  app_log() << "Test of DescentEngineHandle construction" << std::endl;
  std::unique_ptr<DescentEngineHandle> handle = std::make_unique<DescentEngineHandle>(descentEngineObj);


  const int fake_num_params = 5;
  const int fake_sample_num = 100;
  handle->prepareSampling(fake_num_params, fake_sample_num, 2);
  auto& test_der_rat_samp = handle->getVector();

  REQUIRE(test_der_rat_samp.size() == 6);
  REQUIRE(test_der_rat_samp[0] == 0.0);

  const EngineHandle::SamplingRequirements requirements = handle->getSamplingRequirements();
  CHECK(requirements.needs_log_derivatives);
  CHECK(requirements.needs_energy_derivatives);
  CHECK_FALSE(requirements.persists_log_derivatives);
  CHECK_FALSE(requirements.persists_energy_derivatives);
  CHECK(requirements.consumes_batches_online);
}

/// Check that crowd-streamed batches, including a short final batch, reproduce direct accumulation.
TEST_CASE("DescentEngineHandle streams crowd batches", "[drivers][descent]")
{
  Communicate* communicator = OHMMS::Controller;
  Libxml2Document document;
  REQUIRE(document.parseFromString("<tmp/>"));

  DescentEngine streamed_engine(communicator, document.getRoot());
  DescentEngine reference_engine(communicator, document.getRoot());
  DescentEngineHandle handle(streamed_engine);
  handle.prepareSampling(2, 3, 2);
  reference_engine.prepareStorage(2, 2);

  auto submit_batch = [&](int crowd,
                          int base_sample,
                          const std::vector<double>& energies,
                          const std::vector<std::array<double, 2>>& dlog,
                          const std::vector<std::array<double, 2>>& denergy) {
    RecordArray<ValueType> dlog_array(energies.size(), 2);
    RecordArray<ValueType> denergy_array(energies.size(), 2);
    for (std::size_t sample = 0; sample < energies.size(); ++sample)
      for (int parameter = 0; parameter < 2; ++parameter)
      {
        dlog_array[sample][parameter]    = dlog[sample][parameter];
        denergy_array[sample][parameter] = denergy[sample][parameter];
      }
    handle.takeSample(energies, dlog_array, denergy_array, base_sample, crowd);

    for (std::size_t sample = 0; sample < energies.size(); ++sample)
    {
      std::vector<FullPrecValueType> derivative_ratio{1.0, dlog[sample][0], dlog[sample][1]};
      std::vector<FullPrecValueType> energy_derivative{
          energies[sample], denergy[sample][0] + energies[sample] * dlog[sample][0],
          denergy[sample][1] + energies[sample] * dlog[sample][1]};
      reference_engine.takeSample(crowd, derivative_ratio, energy_derivative, energy_derivative, 1.0, 1.0);
    }
  };

  submit_batch(0, 0, {-1.0, -1.2}, {{0.2, -0.1}, {0.3, 0.4}}, {{0.05, -0.02}, {-0.03, 0.08}});
  submit_batch(1, 2, {-0.8}, {{-0.2, 0.15}}, {{0.01, -0.04}});
  handle.finishSampling();
  reference_engine.sample_finish();

  const auto& streamed  = streamed_engine.getAveragedDerivatives();
  const auto& reference = reference_engine.getAveragedDerivatives();
  REQUIRE(streamed.size() == reference.size());
  for (std::size_t parameter = 0; parameter < streamed.size(); ++parameter)
  {
    CHECK(std::real(streamed[parameter]) == Approx(std::real(reference[parameter])));
    CHECK(std::imag(streamed[parameter]) == Approx(std::imag(reference[parameter])));
  }
}
} // namespace qmcplusplus
