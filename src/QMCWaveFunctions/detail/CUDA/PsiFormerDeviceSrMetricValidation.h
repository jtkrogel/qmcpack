//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SR_METRIC_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SR_METRIC_VALIDATION_H
#include "PsiFormerDeviceBatchBackend.h"
#include "PsiFormerDeviceSrMetric.h"
#include <iostream>

namespace qmcplusplus::psiformer::device
{
/// Independent CPU score covariance oracle and lifecycle/transfer checks, not PCG/driver qualification.
inline void validateSrMetric(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  using Metric = PreparedDeviceSrMetric;
  const auto p = plan.parameterCount(), ne = model.ne, version = image->version();
  const auto require = [](bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
  };
  const auto fails = [&](const auto& action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "SR metric accepted invalid input/provenance");
  };
  const auto near = [&](double actual, double expected) {
    require(device_math::isFiniteBinary64(actual) && device_math::isFiniteBinary64(expected) &&
        std::abs(actual - expected) <= 3e-11 * (1 + std::abs(expected)),
        "Resident SR metric differs from independent CPU covariance");
  };
  require(p > 3 && model.cfg.nconfig > 0, "SR metric tiny fixture is incomplete");
  PreparedDeviceBatchBackend backend(image);
  auto session = backend.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4);
  const std::vector<Metric::FrozenInterval> frozen{{0, 2}, {p - 1, 1}};
  const auto trainable = [&](std::size_t index) { return index >= 2 && index + 1 < p; };
  std::vector<double> x(p), y(p), actual(p), second(p);
  for (std::size_t i = 0; i < p; ++i)
  {
    x[i] = (int(i % 11) - 5) * 0.01 / std::sqrt(p);
    y[i] = (int(i % 7) - 3) * 0.02 / std::sqrt(p);
  }
  pf::DirectScoreExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  // B5 with capacity4 exercises a non-divisor tail; B1 must give exactly zero.
  for (const auto b : {std::size_t{5}, std::size_t{1}})
  {
    std::vector<double> positions(3 * b * ne), weights(b), scores(b * p);
    double weight = 0;
    for (std::size_t s = 0; s < b; ++s)
    {
      std::copy_n(model.cfg.electrons.x.data() + (s % model.cfg.nconfig) * 3 * ne,
          3 * ne, positions.data() + s * 3 * ne);
      weights[s] = b == 1 ? 1.0 : (s == 1 ? 0.0 : 0.8 + 0.07 * s);
      weight += weights[s];
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + s * 3 * ne, ne));
      const auto score = cpu.evaluate(*workspace);
      std::copy_n(score.parameter_score.data, p, scores.data() + s * p);
    }
    const auto oracle_weights = weights; // Independent oracle survives caller-lifetime poisoning below.
    std::vector<double> oracle_products(b);
    double oracle_mean = 0;
    const auto oracle = [&](const std::vector<double>& direction) {
      std::vector<double> products(b), result(p);
      double mean = 0;
      for (std::size_t s = 0; s < b; ++s)
      {
        for (std::size_t i = 0; i < p; ++i)
          if (trainable(i)) products[s] += scores[s * p + i] * direction[i];
        mean += oracle_weights[s] * products[s];
      }
      mean /= weight;
      oracle_products = products;
      oracle_mean = mean;
      for (std::size_t s = 0; s < b; ++s)
        for (std::size_t i = 0; i < p; ++i)
          if (trainable(i)) result[i] += oracle_weights[s] / weight * (products[s] - mean) * scores[s * p + i];
      return result;
    };
    const auto cost = Metric::requiredBytes(p, b, ne, session->scoreContractionCapacity() ? session->scoreContractionCapacity() : 1, frozen.size());
    auto before = session->transferSnapshot();
    fails([&] { Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes - 1, cost.host_bytes); });
    fails([&] { Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes, cost.host_bytes - 1); });
    fails([&] { Metric::prepare(*session, positions, weights, frozen, version, 0, cost.host_bytes); });
    fails([&] { Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes,
        std::numeric_limits<std::size_t>::max()); });
    auto invalid_weights = weights;
    invalid_weights[0] = -1;
    fails([&] { Metric::prepare(*session, positions, invalid_weights, frozen, version, cost.device_bytes, cost.host_bytes); });
    fails([&] { Metric::requiredBytes(std::numeric_limits<std::size_t>::max(), b, ne, 3); });
    require(session->transferSnapshot().allocation_calls == before.allocation_calls,
        "SR metric cap/overflow rejection allocated numeric device storage");
    auto metric = Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes, cost.host_bytes);
    const auto allocated = metric->allocatedBytes();
    require(allocated.device_bytes == cost.device_bytes && allocated.host_bytes == cost.host_bytes,
        "SR metric requirement differs from actual fixed numeric capacities");
    // Destroy caller contents after preparation: the metric owns coordinates/weights.
    std::fill(positions.begin(), positions.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill(weights.begin(), weights.end(), -1.0);
    auto input = metric->uploadDirection(x);
    const auto stale = input;
    input = metric->uploadDirection(x);
    fails([&] { metric->apply(stale); });
    input = metric->uploadDirection(x);
    auto output = metric->apply(input); // Warm graph/BLAS before the measured window.
    before = session->transferSnapshot();
    output = metric->apply(input);
    const auto after = session->transferSnapshot();
    require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls &&
        after.uploaded_bytes - before.uploaded_bytes == (48 * b * ne + 8 * b) &&
        after.downloaded_bytes - before.downloaded_bytes < 8 * p,
        "Warm SR metric allocated or transferred a P-sized vector");
    metric->downloadAction(output, actual); // Explicit P inspection outside the window.
    const auto expected = oracle(x);
    // A failed oracle prints indexed evidence and separately exercises the old
    // host-direction JVP/C1 route, strictly outside the measured resident window.
    std::size_t worst = 0;
    double max_error = 0, scale = 0;
    for (std::size_t i = 0; i < p; ++i)
    {
      const double error = std::abs(actual[i] - expected[i]);
      if (error > max_error) { max_error = error; worst = i; }
      scale = std::max(scale, std::abs(expected[i]));
    }
    if (max_error > 3e-11 * (1 + std::abs(expected[worst])))
    {
      std::cerr.precision(17);
      std::cerr << "DIAGNOSTIC phase=x covariance B=" << b << " index=" << worst
          << " actual=" << actual[worst] << " expected=" << expected[worst]
          << " max_error=" << max_error << " covariance_scale=" << scale
          << " cpu_mean=" << oracle_mean << '\n';
      auto projected = x;
      for (std::size_t i = 0; i < p; ++i) if (!trainable(i)) projected[i] = 0;
      session->prepareScoreDirection(projected, version);
      std::vector<double> gpu_products(b);
      double gpu_mean = 0;
      for (std::size_t s = 0; s < b; ++s)
      {
        gpu_products[s] = session->scoreJvpPrepared(model.cfg.electrons.x.data() +
            (s % model.cfg.nconfig) * 3 * ne, version);
        gpu_mean += (b == 1 ? 1.0 : (s == 1 ? 0.0 : 0.8 + 0.07 * s)) * gpu_products[s];
        std::cerr << "DIAGNOSTIC sample=" << s << " cpu_jvp=" << oracle_products[s]
            << " old_host_ingress_gpu_jvp=" << gpu_products[s] << '\n';
      }
      gpu_mean /= weight;
      for (bool cpu_coefficients : {false, true})
      {
        session->beginScoreContraction(1, version);
        for (std::size_t s = 0; s < b; ++s)
        {
          const double w = b == 1 ? 1.0 : (s == 1 ? 0.0 : 0.8 + 0.07 * s);
          const std::complex<double> coefficient(w / weight *
              ((cpu_coefficients ? oracle_products[s] : gpu_products[s]) -
                  (cpu_coefficients ? oracle_mean : gpu_mean)), 0);
          session->accumulateScoreContraction(model.cfg.electrons.x.data() +
              (s % model.cfg.nconfig) * 3 * ne, &coefficient, 1, version);
        }
        session->finishScoreContraction(version);
        std::vector<std::complex<double>> reference(p);
        session->downloadScoreContraction(0, reference.data(), p, version);
        double resident_error = 0, cpu_error = 0;
        for (std::size_t i = 0; i < p; ++i)
        {
          const double value = trainable(i) ? reference[i].real() : 0;
          resident_error = std::max(resident_error, std::abs(value - actual[i]));
          cpu_error = std::max(cpu_error, std::abs(value - expected[i]));
        }
        std::cerr << "DIAGNOSTIC old_C1 cpu_coefficients=" << cpu_coefficients
            << " resident_action_max_error=" << resident_error << " cpu_covariance_max_error=" << cpu_error << '\n';
      }
    }
    for (std::size_t i = 0; i < p; ++i) near(actual[i], expected[i]);
    const auto old_output = output;
    input = metric->uploadDirection(y);
    fails([&] { metric->downloadAction(old_output, second); });
    output = metric->apply(input);
    metric->downloadAction(output, second);
    const auto expected_y = oracle(y);
    double xy = 0, yx = 0, xx = 0;
    for (std::size_t i = 0; i < p; ++i)
    {
      near(second[i], expected_y[i]);
      xy += x[i] * second[i]; yx += y[i] * actual[i]; xx += x[i] * actual[i];
      if (!trainable(i)) require(actual[i] == 0 && second[i] == 0, "SR frozen output is nonzero");
    }
    near(xy, yx);
    require(xx >= -1e-13, "SR metric violates positive semidefiniteness");
    if (b == 1)
      for (double value : actual) require(value == 0.0, "B1 SR metric is not exactly zero");
    auto bad = x;
    bad[0] = std::numeric_limits<double>::quiet_NaN(); // Frozen, but still admitted before projection.
    const auto bad_input = metric->uploadDirection(bad);
    fails([&] { metric->apply(bad_input); });
    fails([&] { metric->downloadAction(output, actual); });
    std::cout << "PASS resident SR numerical metric B=" << b << " P=" << p
        << " device_bytes=" << cost.device_bytes << " host_bytes=" << cost.host_bytes
        << " warm_uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
        << " warm_downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes << '\n';
  }
  std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * ne), weights{1};
  const auto c1_cost = Metric::requiredBytes(p, 1, ne, 1, frozen.size());
  auto old_capacity = Metric::prepare(*session, positions, weights, frozen, version,
      c1_cost.device_bytes, c1_cost.host_bytes);
  // Fixed C3 is retained and charged even though this metric consumes only live C1.
  session->prepareScoreContractions(3, version);
  fails([&] { old_capacity->uploadDirection(x); });
  const auto cost = Metric::requiredBytes(p, 1, ne, 3, frozen.size());
  auto metric = Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes, cost.host_bytes);
  const auto allocated = metric->allocatedBytes();
  require(allocated.device_bytes == cost.device_bytes && allocated.host_bytes == cost.host_bytes,
      "Live C1 hid retained C3 capacity");
  auto other = Metric::prepare(*session, positions, weights, frozen, version, cost.device_bytes, cost.host_bytes);
  auto input = metric->uploadDirection(x);
  fails([&] { other->apply(input); });
  const auto output = metric->apply(input);
  metric->downloadAction(output, actual);
  backend.publishParameters(model.p.flat_values(), version + 1, []() {});
  fails([&] { metric->apply(input); });
  fails([&] { metric->downloadAction(output, actual); });
}
} // namespace qmcplusplus::psiformer::device
#endif
