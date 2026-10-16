//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RESIDENT_ADAM_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RESIDENT_ADAM_VALIDATION_H
#include "PsiFormerDeviceBatchBackend.h"
#include <vector>
#include <iostream>

namespace qmcplusplus::psiformer::device
{
/// Qualify only the numerical ENERGY C3/Adam bridge, not a training driver binding.
inline void validateResidentAdam(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  const auto count = plan.parameterCount(), ne = model.ne, version = image->version();
  const auto require = [](bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
  };
  std::size_t rejection_case = 0;
  const auto fails = [&](const auto& action) {
    ++rejection_case;
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Resident Adam accepted invalid input/provenance, case=" + std::to_string(rejection_case));
  };
  const auto near = [&](double actual, double expected) {
    require(device_math::isFiniteBinary64(actual) &&
        std::abs(actual - expected) <= 3e-11 * (1 + std::abs(expected)),
        "Resident Adam differs from host ENERGY/Adam arithmetic");
  };
  const std::vector<AdamBlock> blocks{{0, count - 1, 0.0017, 1}, {count - 1, 1, 0, 0}};
  const auto base_bytes = estimateAdamBytes(count, blocks.size());
  auto adam = PreparedDeviceAdam::prepare(count, blocks, base_bytes);
  auto compatibility = PreparedDeviceAdam::prepare(count, blocks, base_bytes);
  std::vector<double> zero(count), first(count), second(count), reference_first(count), reference_second(count);
  adam->seed(zero, zero);
  compatibility->seed(zero, zero);
  require(adam->energyGradientDeviceBytes() == 0 && adam->deviceBytes() == base_bytes,
      "Ordinary Adam unexpectedly allocated ENERGY scratch");
  PreparedDeviceBatchBackend backend(image);
  auto session = backend.prepareProductSession();
  const auto session_base_bytes = session->deviceWorkspaceBytes();
  const auto c3_bytes = PreparedDeviceScoreContraction::requiredBytes(count, 3);
  const auto norm_bytes = PreparedDeviceAdam::energyGradientRequiredBytes(count);
  auto before = session->transferSnapshot();
  fails([&] { session->prepareScoreContractions(3, version, c3_bytes - 1); });
  fails([&] { adam->prepareEnergyGradient(norm_bytes - 1); });
  fails([&] { PreparedDeviceScoreContraction::requiredBytes(std::numeric_limits<std::size_t>::max(), 3); });
  fails([&] { PreparedDeviceAdam::energyGradientRequiredBytes(std::numeric_limits<std::size_t>::max()); });
  require(session->transferSnapshot().allocation_calls == before.allocation_calls,
      "Resident Adam cap-minus-one allocated device storage");
  session->prepareScoreContractions(3, version, c3_bytes);
  adam->prepareEnergyGradient(norm_bytes);
  require(adam->energyGradientDeviceBytes() == norm_bytes && adam->energyGradientHostBytes() == norm_bytes,
      "Resident Adam exact norm capacity mismatch");
  const auto session_bytes = session->deviceWorkspaceBytes();
  require(session_bytes - session_base_bytes == c3_bytes, "Resident Adam exact C3 capacity mismatch");
  fails([&] { session->beginScoreContraction(2, version, ScoreContractionRole::ENERGY_T_EO_O); });
  std::vector<double> parameters = model.p.flat_values(), candidate(count), expected(count), gradient(count);
  std::vector<double> masses(ne), drift(3 * ne), imaginary_drift(3 * ne);
  for (std::size_t electron = 0; electron < ne; ++electron) masses[electron] = 0.7 + 0.13 * electron;
  for (std::size_t lane = 0; lane < drift.size(); ++lane) drift[lane] = 0.01 * (int(lane % 5) - 2);
  std::vector<std::complex<double>> sums(3 * count);
  double weight = 0, energy_sum = 0;
  for (std::size_t sample = 0; sample < 8; ++sample)
  { weight += 0.8 + 0.07 * sample; energy_sum += (0.8 + 0.07 * sample) * (-2.3 + 0.09 * sample); }
  const double energy = energy_sum / weight, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  using Channels = std::vector<std::complex<double>>;
  const auto coefficients = [](std::size_t sample) {
    const double w = 0.8 + 0.07 * sample, e = -2.3 + 0.09 * sample;
    Channels result(3);
    result[1] = w * e;
    result[2] = w;
    return result;
  };
  const auto positions = [&](std::size_t sample) {
    return model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * 3 * ne;
  };
  const auto build = [&](std::size_t current_version, ScoreContractionRole role = ScoreContractionRole::ENERGY_T_EO_O) {
    const auto generation = session->beginScoreContraction(3, current_version, role);
    for (std::size_t sample = 0; sample < 8; ++sample)
    {
      const auto score = coefficients(sample);
      Channels kinetic(3), imaginary(3);
      kinetic[0] = 0.8 + 0.07 * sample;
      session->accumulateKineticContraction(positions(sample), drift.data(), imaginary_drift.data(), masses.data(),
          score.data(), kinetic.data(), imaginary.data(), 3, current_version);
    }
    session->finishScoreContraction(current_version);
    return generation;
  };
  const auto consume = [&](std::size_t current_version, std::size_t generation, double w = 0) {
    return session->consumeEnergyAdam(*adam, current_version, generation, parameters, w == 0 ? weight : w,
        energy, beta1, beta2, 1 - std::pow(beta1, current_version + 1),
        1 - std::pow(beta2, current_version + 1), epsilon, candidate);
  };
  // Untagged, wrong-generation, stale-version and mismatching old-P inputs never publish candidates.
  fails([&] { consume(version, build(version, ScoreContractionRole::GENERIC)); });
  auto generation = build(version);
  fails([&] { consume(version, generation + 1); });
  generation = build(version);
  fails([&] { consume(version + 1, generation); });
  generation = build(version);
  parameters[0] += 1;
  fails([&] { consume(version, generation); });
  parameters = model.p.flat_values();
  generation = build(version);
  const double tiny_weight = std::ldexp(1.0, -600), huge_energy = std::ldexp(1.0, 600);
  for (std::size_t channel = 0; channel < 3; ++channel)
    session->downloadScoreContraction(channel, sums.data() + channel * count, count, version);
  bool gradient_overflow = false;
  for (std::size_t parameter = 0; parameter < count; ++parameter)
  {
    const long double g = 2.0L * (static_cast<long double>(sums[parameter].real()) +
        sums[count + parameter].real() - static_cast<long double>(huge_energy) *
        sums[2 * count + parameter].real()) / tiny_weight;
    gradient_overflow |= sums[2 * count + parameter].real() != 0 &&
        std::fabs(g) > std::numeric_limits<double>::max();
  }
  require(gradient_overflow, "Resident Adam overflow negative lacks a nonzero actual O / long-double witness");
  fails([&] { session->consumeEnergyAdam(*adam, version, generation, parameters, tiny_weight, huge_energy,
      beta1, beta2, 0.1, 0.001, epsilon, candidate); });
  const std::vector<AdamBlock> frozen_blocks{{0, count, 0, 0}};
  auto frozen = PreparedDeviceAdam::prepare(count, frozen_blocks, estimateAdamBytes(count, 1));
  frozen->seed(zero, zero);
  frozen->prepareEnergyGradient(norm_bytes);
  generation = build(version);
  fails([&] { session->consumeEnergyAdam(*frozen, version, generation, parameters,
      tiny_weight, huge_energy, beta1, beta2, 0.1, 0.001, epsilon, candidate); });
  for (std::size_t step = 0; step < 2; ++step)
  {
    const auto current_version = version + step;
    std::fill(sums.begin(), sums.end(), std::complex<double>{});
    // Existing separately qualified host wrappers supply independent contraction arithmetic.
    for (std::size_t sample = 0; sample < 8; ++sample)
    {
      const auto score_weights = coefficients(sample);
      const auto score = session->score(positions(sample), current_version);
      const auto kinetic = session->kinetic(positions(sample), drift.data(), masses.data(), current_version);
      for (std::size_t parameter = 0; parameter < count; ++parameter)
      {
        sums[parameter] += (0.8 + 0.07 * sample) * kinetic[parameter];
        sums[count + parameter] += score_weights[1] * score[parameter];
        sums[2 * count + parameter] += score_weights[2] * score[parameter];
      }
    }
    double norm = 0;
    for (std::size_t parameter = 0; parameter < count; ++parameter)
    {
      gradient[parameter] = 2 * (sums[parameter] + sums[count + parameter] -
          energy * sums[2 * count + parameter]).real() / weight;
      norm = std::hypot(norm, gradient[parameter]);
    }
    const double correction1 = 1 - std::pow(beta1, step + 1), correction2 = 1 - std::pow(beta2, step + 1);
    compatibility->propose(parameters, gradient.data(), beta1, beta2, correction1, correction2, epsilon, expected);
    if (step == 0)
    {
      generation = build(current_version);
      const double huge_norm = session->consumeEnergyAdam(*frozen, current_version, generation, parameters,
          weight * 1e-200, energy, beta1, beta2, correction1, correction2, epsilon, candidate);
      near(huge_norm * 1e-200, norm); // Stable norm without squaring 1e200-sized frozen gradients.
      frozen->reject();
    }
    generation = build(current_version);
    before = session->transferSnapshot();
    near(consume(current_version, generation), norm);
    const auto after = session->transferSnapshot();
    require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls &&
        after.uploaded_bytes - before.uploaded_bytes == sizeof(double) * count &&
        after.downloaded_bytes - before.downloaded_bytes == sizeof(double) * count + norm_bytes + 16,
        "Resident Adam warm window allocated or transferred C3/host gradient");
    for (std::size_t parameter = 0; parameter < count; ++parameter)
    {
      near(candidate[parameter], expected[parameter]);
      if (parameter + 1 < count)
      {
        reference_first[parameter] = beta1 * reference_first[parameter] + (1 - beta1) * gradient[parameter];
        reference_second[parameter] = beta2 * reference_second[parameter] + (1 - beta2) * gradient[parameter] * gradient[parameter];
      }
    }
    if (step == 0)
    {
      adam->reject();
      adam->download(first, second);
      require(first == zero && second == zero, "Rejected resident Adam modified committed moments");
      generation = build(current_version);
      near(consume(current_version, generation), norm);
    }
    adam->accept();
    compatibility->accept();
    adam->download(first, second); // Explicit inspection/checkpoint boundary, outside warm window.
    for (std::size_t parameter = 0; parameter < count; ++parameter)
    { near(first[parameter], reference_first[parameter]); near(second[parameter], reference_second[parameter]); }
    require(session->deviceWorkspaceBytes() == session_bytes && adam->deviceBytes() == base_bytes,
        "Resident Adam changed fixed prepared capacity");
    parameters = candidate;
    backend.publishParameters(parameters, current_version + 1, []() {});
    std::cout << "Resident ENERGY Adam: B=8 step=" << step + 1 << " P=" << count
              << " C3_device_bytes=" << c3_bytes << " norm_device_bytes=" << norm_bytes
              << " norm_host_bytes=" << adam->energyGradientHostBytes()
              << " uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
              << " downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes << '\n';
  }
  compatibility->seed(first, second); // Existing checkpoint cold-seed path remains usable.
  compatibility->download(reference_first, reference_second);
  require(first == reference_first && second == reference_second, "Resident Adam checkpoint seed changed moments");
  std::cout << "PASS numerical resident ENERGY Adam: two B8 updates, moments, accept/reject, exact caps, "
               "provenance/failure recovery and no warm C3 download/gradient upload; production NOT_BOUND\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
