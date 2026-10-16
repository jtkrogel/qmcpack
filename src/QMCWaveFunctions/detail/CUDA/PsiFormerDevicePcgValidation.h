//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PCG_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PCG_VALIDATION_H
#include "PsiFormerDeviceBatchBackend.h"
#include "PsiFormerDevicePcg.h"
#include "QMCDrivers/WFTrain/KrylovPcgControl.h"
#include <array>
#include <iostream>
#include <type_traits>

namespace qmcplusplus::psiformer::device
{
// Receipts bind owner addresses; unique_ptr may move, its pointee must not.
static_assert(!std::is_move_constructible_v<PreparedDevicePcg> && !std::is_copy_constructible_v<PreparedDevicePcg>);
static_assert(!std::is_move_constructible_v<PreparedDeviceSrMetric> && !std::is_copy_constructible_v<PreparedDeviceSrMetric>);

/// Independent HOST score covariance storage for the already-qualified common loop.
/// No WFTrain .cpp functions/libraries or generic fault operator are linked.
struct PcgCpuReference
{
  using Slot = wftrain::krylov_detail::PcgSlot;
  using Value = wftrain::DerivativeValue;
  const std::vector<double>& scores;
  const std::vector<double>& weights;
  const std::size_t p;
  const double damping;
  std::array<std::vector<double>, 7> slots;
  std::vector<double> products;
  std::size_t scans = 0, dots = 0, applications = 0;

  /// Cold reference storage is outside every measured DEVICE solve window.
  PcgCpuReference(const std::vector<double>& score, const std::vector<double>& weight,
      const std::vector<double>& rhs, double shift)
      : scores(score), weights(weight), p(rhs.size()), damping(shift), products(weight.size())
  {
    for (auto& values : slots) values.assign(p, 0);
    get(Slot::RHS) = rhs;
  }

  /// Name only concrete test-owned host vectors, never a device pointer/view.
  std::vector<double>& get(Slot slot) { return slots[static_cast<std::size_t>(slot)]; }

  /// This qualified numerical route has no warm-start ingress.
  bool hasWarmStart() const { return false; }

  /// Match the concrete fixture's two frozen intervals, not a second schema engine.
  bool trainable(std::size_t index) const { return index >= 2 && index + 1 < p; }

  /// Independent all-lane reference admission throws the common numerical category.
  void finite(const std::vector<double>& values) const
  {
    for (double value : values)
      if (!device_math::isFiniteBinary64(value))
        throw wftrain::MatrixFreeNumericalError("CPU PCG reference nonfinite vector");
  }

  /// Concrete owner copy/preconditioner scans source once, then device-to-device copies.
  void copy(Slot source, Slot target)
  {
    ++scans;
    finite(get(source));
    get(target) = get(source);
  }

  /// Count the owner's two scans; do not fuse this operation with following AXPY.
  void scale(Value alpha, Slot target)
  {
    scans += 2;
    finite(get(target));
    for (auto& value : get(target)) value *= alpha.real();
    finite(get(target));
  }

  /// Independent separate-product/add reference and three whole-P scan boundaries.
  void axpy(Value alpha, Slot source, Slot target)
  {
    scans += 3;
    finite(get(source)); finite(get(target));
    for (std::size_t i = 0; i < p; ++i)
    {
      volatile double product = alpha.real() * get(source)[i];
      get(target)[i] += product;
    }
    finite(get(target));
  }

  /// Existing device dot performs its own input/product admission, no separate P scans.
  Value dot(Slot left, Slot right)
  {
    ++dots;
    finite(get(left)); finite(get(right));
    double sum = 0;
    for (std::size_t i = 0; i < p; ++i) sum += get(left)[i] * get(right)[i];
    if (!device_math::isFiniteBinary64(sum))
      throw wftrain::MatrixFreeNumericalError("CPU PCG reference nonfinite dot");
    return {sum, 0};
  }

  /// Preserve raw-square dot/sqrt norm, not a new stable-L2 algorithm.
  double norm(Slot slot) { return std::sqrt(std::max(0.0, dot(slot, slot).real())); }

  /// Identity preconditioning still copies into distinct Z.
  void applyPreconditioner(Slot source, Slot target) { copy(source, target); }

  /// Independent score covariance projects only its input/output, NEVER lambda*v.
  void applyOperator(Slot source, Slot target)
  {
    ++applications;
    std::fill(products.begin(), products.end(), 0);
    double weight = 0, mean = 0;
    for (std::size_t s = 0; s < weights.size(); ++s)
    {
      for (std::size_t i = 0; i < p; ++i)
        if (trainable(i)) products[s] += scores[s * p + i] * get(source)[i];
      weight += weights[s];
      mean += weights[s] * products[s];
    }
    mean /= weight;
    std::fill(get(target).begin(), get(target).end(), 0);
    for (std::size_t s = 0; s < weights.size(); ++s)
      for (std::size_t i = 0; i < p; ++i)
        if (trainable(i)) get(target)[i] += weights[s] / weight * (products[s] - mean) * scores[s * p + i];
    // The concrete owner uses this SAME checked AXPY after its measured Metric action.
    axpy({damping, 0}, source, target);
  }
};

/// Tiny real/open concrete-device PCG qualification, not a public optimizer/driver solve.
inline void validateResidentPcg(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  using Owner = PreparedDevicePcg;
  using Metric = PreparedDeviceSrMetric;
  using Stop = wftrain::KrylovStopReason;
  using Slot = wftrain::krylov_detail::PcgSlot;
  const auto require = [](bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
  };
  const auto fails = [&](auto operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Resident PCG accepted invalid input/receipt");
  };
  const auto near = [&](double actual, double expected, double tolerance = 3e-10) {
    require(device_math::isFiniteBinary64(actual) && device_math::isFiniteBinary64(expected) &&
        std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)), "Resident PCG CPU reference mismatch");
  };
  const auto p = plan.parameterCount(), ne = model.ne, version = image->version();
  require(p == 3922 && ne == 4 && model.cfg.nup == 2 && model.cfg.nconfig >= 3,
      "Resident PCG gate requires tiny P3922/Ne4/Nup2 fixture");
  constexpr double damping = 0.5;
  PreparedDeviceBatchBackend backend(image);
  auto session = backend.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4);
  require(session->scoreCapacity() == 4, "Resident PCG test lost score microtile capacity4");
  const std::vector<Owner::FrozenInterval> frozen{{0, 2}, {p - 1, 1}};
  pf::DirectScoreExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  wftrain::KrylovSolverControl control;
  control.relative_tolerance = 1e-8;
  control.absolute_tolerance = 1e-13;
  control.maximum_iterations = 20;
  control.maximum_operator_applications = 24;
  const auto q = p / 4096 + (p % 4096 != 0);
  double max_solution = 0, max_residual = 0;
  std::size_t solves = 0;
  for (std::size_t b : {std::size_t(5), std::size_t(1)})
  {
    std::vector<double> positions(3 * b * ne), weights(b), scores(b * p), rhs(p), actual(p);
    for (std::size_t s = 0; s < b; ++s)
    {
      std::copy_n(model.cfg.electrons.x.data() + (s % model.cfg.nconfig) * 3 * ne, 3 * ne, positions.data() + s * 3 * ne);
      weights[s] = b == 1 ? 1 : (s == 1 ? 0 : 0.8 + 0.07 * s);
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + s * 3 * ne, ne));
      const auto score = cpu.evaluate(*workspace);
      std::copy_n(score.parameter_score.data, p, scores.data() + s * p);
    }
    for (std::size_t i = 0; i < p; ++i) rhs[i] = (int(i % 17) - 8) / 1024.0;
    const auto oracle_weights = weights, oracle_rhs = rhs;
    const auto cost = Owner::requiredBytes(p, b, ne, 1, frozen.size());
    const auto metric_cost = Metric::requiredBytes(p, b, ne, 1, frozen.size());
    require(cost.device_bytes == metric_cost.device_bytes + 40 * p + 40 * q + 24 &&
        cost.host_bytes == metric_cost.host_bytes + 16 * q + 24, "Resident PCG cost failed independent composition");
    auto before = session->transferSnapshot();
    fails([&] { Owner::prepare(*session, positions, weights, frozen, version, damping, cost.device_bytes - 1, cost.host_bytes); });
    fails([&] { Owner::prepare(*session, positions, weights, frozen, version, damping, cost.device_bytes, cost.host_bytes - 1); });
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
      fails([&] { Owner::prepare(*session, positions, weights, frozen, version, invalid, cost.device_bytes, cost.host_bytes); });
    fails([&] { Owner::prepare(*session, positions, weights, frozen, version, damping, 0, cost.host_bytes); });
    fails([&] { Owner::prepare(*session, positions, weights, frozen, version, damping, cost.device_bytes,
        std::numeric_limits<std::size_t>::max()); });
    fails([&] { Owner::requiredBytes(std::numeric_limits<std::size_t>::max(), b, ne, 3); });
    require(session->transferSnapshot().allocation_calls == before.allocation_calls, "Resident PCG preflight allocated");

    // One existing Metric baseline outside solves accounts its honest host boundaries.
    std::size_t metric_download = 0;
    {
      auto metric = Metric::prepare(*session, positions, weights, frozen, version, metric_cost.device_bytes, metric_cost.host_bytes);
      const auto input = metric->uploadDirection(rhs);
      (void)metric->apply(input);
      before = session->transferSnapshot();
      (void)metric->apply(input);
      const auto after = session->transferSnapshot();
      metric_download = after.downloaded_bytes - before.downloaded_bytes;
      require(after.uploaded_bytes - before.uploaded_bytes == 48 * b * ne + 8 * b, "PCG Metric baseline ingress changed");
    }
    auto gpu = Owner::prepare(*session, positions, weights, frozen, version, damping, cost.device_bytes, cost.host_bytes);
    const auto allocated = gpu->allocatedBytes();
    require(allocated.device_bytes == cost.device_bytes && allocated.host_bytes == cost.host_bytes,
        "Resident PCG exact prepared capacities differ from preflight");
    const auto ingress = session->transferSnapshot();
    auto input = gpu->uploadRhs(rhs, version);
    require(session->transferSnapshot().uploaded_bytes - ingress.uploaded_bytes == 8 * p, "PCG RHS cold ingress not exactly8P");
    std::fill(positions.begin(), positions.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill(weights.begin(), weights.end(), -1);
    std::fill(rhs.begin(), rhs.end(), std::numeric_limits<double>::quiet_NaN());

    const auto compare = [&](const wftrain::KrylovSolverControl& policy, Stop required_stop) {
      PcgCpuReference reference(scores, oracle_weights, oracle_rhs, damping);
      wftrain::KrylovSolveDiagnostics expected;
      wftrain::krylov_detail::validateControl(policy);
      wftrain::krylov_detail::runPreconditionedConjugateGradient(reference, policy, p, expected);
      const auto dot_calls = reference.dots, scan_calls = reference.scans, applications = reference.applications;
      require(expected.stop_reason == required_stop, "CPU PCG fixture did not exercise its requested bounded stop");
      const auto warmup = gpu->solve(input, policy);
      before = session->transferSnapshot();
      const auto output = gpu->solve(input, policy); // Same retained RHS, explicit zero start.
      const auto after = session->transferSnapshot();
      const auto& d = output.diagnostics;
      require(d.stop_reason == expected.stop_reason && d.converged == expected.converged &&
          d.iterations == expected.iterations && d.operator_applications == expected.operator_applications,
          "Resident PCG common diagnostic/control counts differ");
      near(d.initial_residual_norm, expected.initial_residual_norm);
      near(d.final_residual_norm, expected.final_residual_norm);
      near(d.relative_residual_norm, expected.relative_residual_norm);
      require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls,
          "Resident PCG warmed DEVICE allocation/free");
      const auto uploads = after.uploaded_bytes - before.uploaded_bytes, downloads = after.downloaded_bytes - before.downloaded_bytes;
      require(applications == d.operator_applications && uploads == d.operator_applications * (48 * b * ne + 8 * b) &&
          downloads == d.operator_applications * metric_download + 24 * dot_calls + 16 * q * scan_calls,
          "Resident PCG warm transfer trace differs from Metric+scalar/status boundary");
      const auto inspect = session->transferSnapshot();
      gpu->downloadIterate(output.iterate, actual);
      require(session->transferSnapshot().downloaded_bytes - inspect.downloaded_bytes == 8 * p, "PCG inspection not exactly8P");
      fails([&] { gpu->downloadIterate(warmup.iterate, actual); });
      for (std::size_t i = 0; i < p; ++i)
      {
        max_solution = std::max(max_solution, std::abs(actual[i] - reference.get(Slot::X)[i]));
        near(actual[i], reference.get(Slot::X)[i]);
      }
      if (d.converged)
      {
        for (auto index : {std::size_t(0), std::size_t(1), p - 1}) near(damping * actual[index], oracle_rhs[index]);
        reference.get(Slot::X) = actual;
        reference.applyOperator(Slot::X, Slot::AP);
        double residual = 0, norm = 0;
        for (std::size_t i = 0; i < p; ++i)
        {
          const auto delta = reference.get(Slot::AP)[i] - oracle_rhs[i];
          residual += delta * delta; norm += oracle_rhs[i] * oracle_rhs[i];
        }
        max_residual = std::max(max_residual, std::sqrt(residual));
        require(std::sqrt(residual) <= 8 * std::max(policy.absolute_tolerance, policy.relative_tolerance * std::sqrt(norm)),
            "Resident PCG independent physical covariance residual failed");
      }
      ++solves;
      std::cout << "resident_pcg_solve B=" << b << " P=" << p << " reason=" << static_cast<int>(d.stop_reason)
                << " iterations=" << d.iterations << " applications=" << d.operator_applications
                << " warm_uploaded_bytes=" << uploads << " warm_downloaded_bytes=" << downloads
                << " scalar_dot_calls=" << dot_calls << " input_output_scan_calls=" << scan_calls
                << " metric_baseline_download=" << metric_download << " scope=device_wide\n";
    };
    compare(control, Stop::CONVERGED);
    if (b == 5)
    {
      auto bounded = control;
      bounded.maximum_iterations = 0;
      compare(bounded, Stop::ITERATION_LIMIT);
      bounded = control; bounded.maximum_operator_applications = 0;
      compare(bounded, Stop::OPERATOR_APPLICATION_LIMIT);
      bounded = control; bounded.maximum_seconds = 1e-300;
      compare(bounded, Stop::TIME_LIMIT);
    }
    auto good = gpu->solve(input, control);
    auto invalid_control = control; invalid_control.relative_tolerance = -1;
    fails([&] { gpu->solve(input, invalid_control); });
    fails([&] { gpu->downloadIterate(good.iterate, actual); });
    (void)gpu->solve(input, control); // Invalid control did not destroy current RHS.
    auto stale = input;
    input = gpu->uploadRhs(oracle_rhs, version);
    fails([&] { gpu->solve(stale, control); });
    (void)gpu->solve(input, control); // Bad receipt did not destroy good stored RHS.
    for (bool bad_extent : {false, true})
    {
      auto bad = oracle_rhs;
      if (bad_extent) bad.pop_back(); else bad[0] = std::numeric_limits<double>::quiet_NaN();
      fails([&] { gpu->uploadRhs(bad, version); });
      fails([&] { gpu->solve(input, control); });
      input = gpu->uploadRhs(oracle_rhs, version);
    }
    fails([&] { gpu->uploadRhs(oracle_rhs, version + 1); });
    fails([&] { gpu->solve(input, control); });
    auto huge = oracle_rhs;
    huge[0] = std::numeric_limits<double>::max() * 0.75; // Frozen, yet ALL-P finite/raw-square admission applies.
    input = gpu->uploadRhs(huge, version);
    const auto overflow = gpu->solve(input, control);
    require(overflow.diagnostics.stop_reason == Stop::NONFINITE_RESULT, "PCG widened raw-square overflow domain");
    gpu->downloadIterate(overflow.iterate, actual); // Explicit diagnostic iterate, not accepted candidate.
    std::vector<double> zero(p, 0);
    input = gpu->uploadRhs(zero, version);
    const auto empty = gpu->solve(input, control);
    require(empty.diagnostics.stop_reason == Stop::ZERO_RIGHT_HAND_SIDE && empty.diagnostics.operator_applications == 0,
        "PCG zero RHS stop changed");
    gpu->downloadIterate(empty.iterate, actual);
    for (double value : actual) require(value == 0, "PCG zero RHS iterate is not zero");
    std::cout << "PASS resident PCG capacities B=" << b << " device_bytes=" << allocated.device_bytes
              << " host_bytes=" << allocated.host_bytes << " fixed_C=1 frozen_rhs_nonzero=1 retained_rhs=1\n";
  }

  // Retained C3, cross-owner receipts, hard node poisoning and canonical image staleness.
  std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * ne), weights{1}, rhs(p, 0.01), actual(p);
  const auto c1 = Owner::requiredBytes(p, 1, ne, 1, frozen.size());
  auto old = Owner::prepare(*session, positions, weights, frozen, version, damping, c1.device_bytes, c1.host_bytes);
  const auto old_rhs = old->uploadRhs(rhs, version);
  session->prepareScoreContractions(3, version);
  fails([&] { old->solve(old_rhs, control); });
  const auto c3 = Owner::requiredBytes(p, 1, ne, 3, frozen.size());
  require(c3.device_bytes > c1.device_bytes && c3.host_bytes > c1.host_bytes, "PCG cost hid retained C3");
  auto owner = Owner::prepare(*session, positions, weights, frozen, version, damping, c3.device_bytes, c3.host_bytes);
  auto other = Owner::prepare(*session, positions, weights, frozen, version, damping, c3.device_bytes, c3.host_bytes);
  require(owner->allocatedBytes().device_bytes == c3.device_bytes && owner->allocatedBytes().host_bytes == c3.host_bytes,
      "PCG actual C3 capacities differ");
  const auto input = owner->uploadRhs(rhs, version), foreign = other->uploadRhs(rhs, version);
  fails([&] { owner->solve(foreign, control); });
  const auto output = owner->solve(input, control);
  fails([&] { other->downloadIterate(output.iterate, actual); });
  owner->downloadIterate(output.iterate, actual);
  for (double value : actual) near(value, 0.02);
  auto node_positions = positions;
  std::copy_n(node_positions.data(), 3, node_positions.data() + 3); // Same-spin equal rows, existing strict-node law.
  auto node = Owner::prepare(*session, node_positions, weights, frozen, version, damping, c3.device_bytes, c3.host_bytes);
  const auto node_rhs = node->uploadRhs(rhs, version);
  fails([&] { node->solve(node_rhs, control); });
  fails([&] { node->uploadRhs(rhs, version); });
  backend.publishParameters(model.p.flat_values(), version + 1, []() {});
  fails([&] { owner->solve(input, control); });
  fails([&] { owner->downloadIterate(output.iterate, actual); });
  fails([&] { owner->uploadRhs(rhs, version); });
  std::cout << "PASS numerical resident PCG solves=" << solves << " max_cpu_solution=" << max_solution
            << " max_independent_residual=" << max_residual << " fixed_C3_device_bytes=" << c3.device_bytes
            << " fixed_C3_host_bytes=" << c3.host_bytes << " candidate_publication=none\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
