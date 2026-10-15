//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceKineticContractionValidation.h
 * @brief Same-sample resident mixed score/kinetic contraction qualification.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_CONTRACTION_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_CONTRACTION_VALIDATION_H

#include "PsiFormerDeviceBatchBackend.h"

#include <array>
#include <complex>
#include <iostream>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/// Compare both prepared kinetic routes against their existing host wrappers.
inline void validateKineticContractions(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  constexpr std::size_t channels = 3, samples = 8;
  const auto ne = plan.modelShape().electrons(), parameters = plan.parameterCount();
  std::vector<double> masses(ne), drift(3 * ne), imaginary_drift(3 * ne);
  for (std::size_t electron = 0; electron < ne; ++electron)
    masses[electron] = 0.8 + 0.17 * electron;
  for (std::size_t lane = 0; lane < 3 * ne; ++lane)
  {
    drift[lane] = 0.03 * (static_cast<int>(lane % 7) - 3);
    imaginary_drift[lane] = -0.07 + 0.011 * lane;
  }
  std::vector<std::complex<double>> expected(channels * parameters), actual(expected.size());
  std::array<std::complex<double>, channels> score_coefficients{}, kinetic_coefficients{}, imaginary_coefficients{};
  const auto coefficients = [&](std::size_t sample, double imaginary_sign) {
    score_coefficients = {};
    kinetic_coefficients = {};
    imaginary_coefficients = {};
    score_coefficients[0] = {0.1 * (sample + 1), -0.03 * (sample + 2)};
    kinetic_coefficients[1] = sample % 3 == 0 ? std::complex<double>{}
        : std::complex<double>{-0.2 * (sample + 1), 0.04 * sample};
    kinetic_coefficients[2] = {0.07 * (sample + 1), -0.09 * (sample + 2)};
    for (std::size_t channel = 0; channel < channels; ++channel)
      imaginary_coefficients[channel] = kinetic_coefficients[channel] * std::complex<double>{0.0, imaginary_sign};
  };
  for (bool targets : {false, true})
  {
    DeviceExecutionPreparation preparation;
    if (targets)
    {
      const auto count = std::min(std::size_t(3), ne);
      const auto cost = estimateTargetKineticCost(plan, count);
      preparation.kinetic = {count, cost.device_workspace_bytes, cost.charged_scalar_work_units};
    }
    PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
    auto session = backend.prepareProductSession();
    session->prepareScoreContractions(channels, image->version());
    for (double imaginary_sign : {-1.0, 1.0})
    {
      std::fill(expected.begin(), expected.end(), std::complex<double>{});
      const auto compatibility_before = session->transferSnapshot();
      for (std::size_t sample = 0; sample < samples; ++sample)
      {
        coefficients(sample, imaginary_sign);
        const auto* positions = model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * 3 * ne;
        const auto add = [&](const std::vector<double>& response, const auto& weights) {
          for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::size_t parameter = 0; parameter < parameters; ++parameter)
              expected[channel * parameters + parameter] += weights[channel] * response[parameter];
        };
        add(session->kinetic(positions, drift.data(), masses.data(), image->version()), kinetic_coefficients);
        add(session->score(positions, image->version()), score_coefficients);
        add(session->gradientResponse(positions, imaginary_drift.data(), masses.data(), image->version()), imaginary_coefficients);
      }
      const auto compatibility_after = session->transferSnapshot();
      const auto resident_before = session->transferSnapshot();
      session->beginScoreContraction(channels, image->version());
      for (std::size_t sample = 0; sample < samples; ++sample)
      {
        coefficients(sample, imaginary_sign);
        session->accumulateKineticContraction(
            model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * 3 * ne,
            drift.data(), imaginary_drift.data(), masses.data(), score_coefficients.data(),
            kinetic_coefficients.data(), imaginary_coefficients.data(), channels, image->version());
      }
      session->finishScoreContraction(image->version());
      for (std::size_t channel = 0; channel < channels; ++channel)
        session->downloadScoreContraction(channel, actual.data() + channel * parameters, parameters, image->version());
      const auto resident_after = session->transferSnapshot();
      double maximum = 0.0;
      for (std::size_t index = 0; index < actual.size(); ++index)
      {
        const auto error = std::abs(actual[index] - expected[index]);
        maximum = std::max(maximum, error);
        if (!device_math::isFiniteBinary64(actual[index].real()) ||
            !device_math::isFiniteBinary64(actual[index].imag()) || error > 2e-7 * (1 + std::abs(expected[index])))
          throw std::runtime_error("Resident mixed score/kinetic contraction disagrees with host-wrapper reference");
      }
      const auto compatibility_bytes = compatibility_after.downloaded_bytes - compatibility_before.downloaded_bytes;
      const auto resident_bytes = resident_after.downloaded_bytes - resident_before.downloaded_bytes;
      if (resident_after.allocation_calls != resident_before.allocation_calls ||
          resident_after.free_calls != resident_before.free_calls || resident_after.live_bytes != resident_before.live_bytes ||
          resident_bytes >= compatibility_bytes)
        throw std::runtime_error("Resident kinetic contraction allocated warm storage or failed to reduce measured downloads");
      std::cout << "Resident kinetic transfers: target=" << targets << " imaginary_sign=" << imaginary_sign
                << " compatibility_downloaded_bytes=" << compatibility_bytes
                << " resident_downloaded_bytes=" << resident_bytes
                << " saved_downloaded_bytes=" << compatibility_bytes - resident_bytes
                << " compatibility_download_calls=" << compatibility_after.download_calls - compatibility_before.download_calls
                << " resident_download_calls=" << resident_after.download_calls - resident_before.download_calls
                << " max_abs_difference=" << maximum << " transfer_counter_scope=device_wide\n";
    }

    // Failure after an earlier accepted contribution invalidates all channels.
    coefficients(0, 1.0);
    const auto accumulate = [&](const double* inverse_masses, std::size_t version) {
      session->accumulateKineticContraction(model.cfg.electrons.x.data(), drift.data(), imaginary_drift.data(),
          inverse_masses, score_coefficients.data(), kinetic_coefficients.data(), imaginary_coefficients.data(),
          channels, version);
    };
    session->beginScoreContraction(channels, image->version());
    accumulate(masses.data(), image->version());
    auto invalid_masses = masses;
    invalid_masses[0] = 0.0;
    bool failed = false, hidden = false;
    try { accumulate(invalid_masses.data(), image->version()); }
    catch (const std::invalid_argument&) { failed = true; }
    try { session->finishScoreContraction(image->version()); }
    catch (const std::logic_error&) { hidden = true; }
    if (!failed || !hidden)
      throw std::runtime_error("Resident kinetic failure exposed a partial contraction");
    session->beginScoreContraction(channels, image->version());
    accumulate(masses.data(), image->version());
    session->finishScoreContraction(image->version());
    backend.publishParameters(model.p.flat_values(), image->version() + 1, []() {});
    bool stale = false;
    try { session->downloadScoreContraction(0, actual.data(), parameters, image->version()); }
    catch (const std::invalid_argument&) { stale = true; }
    if (!stale)
      throw std::runtime_error("Resident kinetic contraction accepted stale publication provenance");
    session->beginScoreContraction(channels, image->version() + 1);
    accumulate(masses.data(), image->version() + 1);
    session->finishScoreContraction(image->version() + 1);
  }
  std::cout << "PASS resident mixed score/kinetic contraction: full+target owners, three signed complex channels, "
               "both imaginary adjoints, measured transfers, zero warm device allocations, failure/stale/rebind recovery\n";
}

} // namespace qmcplusplus::psiformer::device
#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_CONTRACTION_VALIDATION_H
