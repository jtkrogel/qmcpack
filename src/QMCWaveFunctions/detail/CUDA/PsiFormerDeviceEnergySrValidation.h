//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ENERGY_SR_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ENERGY_SR_VALIDATION_H
#include "PsiFormerDevicePcgValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Physical ENERGY completion oracle, not a public training/optimizer binding.
inline void validateEnergySr(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  using Owner = PreparedDevicePcg;
  using Metric = PreparedDeviceSrMetric;
  using Slot = wftrain::krylov_detail::PcgSlot;
  using Stop = wftrain::KrylovStopReason;
  const auto require = [](bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
  };
  const auto fails = [&](auto action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "ENERGY SR accepted invalid input/provenance");
  };
  const auto near = [&](double actual, double expected, double tolerance = 3e-10) {
    require(device_math::isFiniteBinary64(actual) && device_math::isFiniteBinary64(expected) &&
        std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)), "ENERGY SR CPU oracle mismatch");
  };
  const auto p = plan.parameterCount(), ne = model.ne, version = image->version();
  constexpr std::size_t b = 5;
  constexpr double damping = 0.5, learning_rate = 0.0017;
  require(p == 3922 && ne == 4 && model.cfg.nup == 2 && model.cfg.nconfig >= 3,
      "ENERGY SR gate requires tiny real/open P3922/Ne4 fixture");
  PreparedDeviceBatchBackend backend(image);
  auto session = backend.prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 4);
  session->prepareScoreContractions(3, version);
  require(session->scoreCapacity() == 4, "ENERGY SR lost score capacity4 tail");
  const std::vector<Owner::FrozenInterval> frozen{{0, 2}, {p - 1, 1}};
  const std::vector<Owner::FrozenInterval> all_frozen{{0, p}};
  std::vector<double> positions(3 * b * ne), weights(b), scores(b * p), gradient(p), rhs(p), candidate(p, 91);
  std::vector<double> masses(ne), drift(3 * ne), imaginary_drift(3 * ne), expected(p);
  const auto parameters = model.p.flat_values();
  for (std::size_t i = 0; i < ne; ++i) masses[i] = 0.7 + 0.13 * i;
  for (std::size_t i = 0; i < drift.size(); ++i) drift[i] = 0.01 * (int(i % 5) - 2);
  pf::DirectScoreExecutor score_cpu(model, plan);
  pf::DirectKineticExecutor kinetic_cpu(model, plan);
  auto score_workspace = score_cpu.makeWorkspace();
  auto kinetic_workspace = kinetic_cpu.makeWorkspace();
  std::vector<double> t(p), eo(p), o(p);
  double weight = 0, energy_sum = 0;
  for (std::size_t s = 0; s < b; ++s)
  {
    auto* position = positions.data() + s * 3 * ne;
    std::copy_n(model.cfg.electrons.x.data() + (s % model.cfg.nconfig) * 3 * ne, 3 * ne, position);
    weights[s] = 0.8 + 0.07 * s;
    const double energy = -2.3 + 0.09 * s;
    weight += weights[s]; energy_sum += weights[s] * energy;
    score_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
    kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
    const auto score = score_cpu.evaluate(*score_workspace);
    const auto kinetic = kinetic_cpu.evaluate(*kinetic_workspace, drift.data(), drift.size(), masses.data(), masses.size());
    for (std::size_t i = 0; i < p; ++i)
    {
      scores[s * p + i] = score.parameter_score[i];
      t[i] += weights[s] * kinetic.kinetic_parameter_response[i];
      eo[i] += weights[s] * energy * score.parameter_score[i];
      o[i] += weights[s] * score.parameter_score[i];
    }
  }
  const double energy = energy_sum / weight;
  double gradient_norm = 0;
  for (std::size_t i = 0; i < p; ++i)
  {
    gradient[i] = 2 * (t[i] + eo[i] - energy * o[i]) / weight;
    gradient_norm = std::hypot(gradient_norm, gradient[i]);
    rhs[i] = i < 2 || i + 1 == p ? 0 : gradient[i];
  }
  require(gradient[0] != 0 || gradient[1] != 0 || gradient.back() != 0,
      "ENERGY SR frozen full-gradient norm fixture is vacuous");

  // The physical producer supplies T/EO/O in its tagged canonical channel order.
  const auto build = [&](ScoreContractionRole role = ScoreContractionRole::ENERGY_T_EO_O, double scale = 1) {
    const auto generation = session->beginScoreContraction(3, version, role);
    std::vector<std::complex<double>> score(3), kinetic(3), imaginary(3);
    for (std::size_t s = 0; s < b; ++s)
    {
      score[1] = scale * weights[s] * (-2.3 + 0.09 * s);
      score[2] = scale * weights[s];
      kinetic[0] = scale * weights[s];
      session->accumulateKineticContraction(positions.data() + s * 3 * ne, drift.data(), imaginary_drift.data(),
          masses.data(), score.data(), kinetic.data(), imaginary.data(), 3, version);
    }
    session->finishScoreContraction(version);
    return generation;
  };
  const auto cost = Owner::energyRequiredBytes(p, b, ne, 3, frozen.size());
  const auto ordinary = Owner::requiredBytes(p, b, ne, 3, frozen.size());
  const auto norm_bytes = 16 * (p / 128 + (p % 128 != 0)) + 8;
  require(cost.device_bytes == ordinary.device_bytes + norm_bytes &&
      cost.host_bytes == ordinary.host_bytes + norm_bytes, "ENERGY SR optional norm cost mismatch");
  auto before = session->transferSnapshot();
  fails([&] { Owner::prepareEnergy(*session, positions, weights, frozen, version, damping, cost.device_bytes - 1, cost.host_bytes); });
  fails([&] { Owner::prepareEnergy(*session, positions, weights, frozen, version, damping, cost.device_bytes, cost.host_bytes - 1); });
  fails([&] { Owner::energyRequiredBytes(std::numeric_limits<std::size_t>::max(), b, ne, 3); });
  fails([&] { Owner::energyRequiredBytes(p, b, ne, 1, frozen.size()); });
  require(session->transferSnapshot().allocation_calls == before.allocation_calls, "ENERGY SR cap preflight allocated");
  std::size_t metric_download = 0;
  {
    const auto metric_cost = Metric::requiredBytes(p, b, ne, 3, frozen.size());
    auto metric = Metric::prepare(*session, positions, weights, frozen, version, metric_cost.device_bytes, metric_cost.host_bytes);
    const auto direction = metric->uploadDirection(rhs);
    (void)metric->apply(direction);
    before = session->transferSnapshot();
    (void)metric->apply(direction);
    const auto after = session->transferSnapshot();
    metric_download = after.downloaded_bytes - before.downloaded_bytes;
    require(after.uploaded_bytes - before.uploaded_bytes == 48 * b * ne + 8 * b,
        "ENERGY SR undamped Metric baseline ingress changed");
  }
  auto owner = Owner::prepareEnergy(*session, positions, weights, frozen, version, damping, cost.device_bytes, cost.host_bytes);
  require(owner->allocatedBytes().device_bytes == cost.device_bytes && owner->allocatedBytes().host_bytes == cost.host_bytes,
      "ENERGY SR actual capacity differs from exact cost");
  wftrain::KrylovSolverControl control;
  control.relative_tolerance = 1e-8;
  control.absolute_tolerance = 1e-13;
  control.maximum_iterations = 20;
  control.maximum_operator_applications = 24;
  const auto consume = [&](std::size_t generation, const wftrain::KrylovSolverControl& policy) {
    return owner->consumeEnergy(version, generation, parameters, weight, energy, learning_rate, policy, candidate);
  };
  fails([&] { consume(build(ScoreContractionRole::GENERIC), control); });
  auto generation = build();
  fails([&] { consume(generation + 1, control); });
  generation = build();
  fails([&] { owner->consumeEnergy(version + 1, generation, parameters, weight, energy, learning_rate, control, candidate); });
  auto wrong_parameters = parameters;
  wrong_parameters[0] += 1;
  generation = build();
  {
    auto wrong_owner = Owner::prepareEnergy(*session, positions, weights, frozen, version, damping,
        cost.device_bytes, cost.host_bytes);
    fails([&] { wrong_owner->consumeEnergy(version, generation, wrong_parameters, weight, energy,
        learning_rate, control, candidate); });
    fails([&] { wrong_owner->consumeEnergy(version, generation, parameters, weight, energy,
        learning_rate, control, candidate); }); // Kernel-discovered mismatch poisons only this owner.
  }
  generation = build();
  fails([&] { owner->consumeEnergy(version, generation, parameters, weight + 1, energy, learning_rate, control, candidate); });

  PcgCpuReference reference(scores, weights, rhs, damping);
  wftrain::KrylovSolveDiagnostics solve;
  wftrain::krylov_detail::runPreconditionedConjugateGradient(reference, control, p, solve);
  require(solve.converged, "ENERGY SR CPU reference did not converge");
  const auto dot_calls = reference.dots, scan_calls = reference.scans, applications = reference.applications;
  reference.applyOperator(Slot::X, Slot::AP);
  double direction_square = 0, contraction = 0, action_square = 0, residual_square = 0;
  for (std::size_t i = 0; i < p; ++i)
  {
    const double x = reference.get(Slot::X)[i], sx = reference.get(Slot::AP)[i] - damping * x;
    direction_square += x * x; action_square += sx * sx; contraction += x * sx;
    const double residual = reference.get(Slot::AP)[i] - rhs[i];
    residual_square += residual * residual;
    volatile double product = -learning_rate * x;
    expected[i] = parameters[i] + product;
  }
  require(contraction > 0 && action_square > 0 && std::sqrt(residual_square) <= 8e-8 * gradient_norm,
      "ENERGY SR undamped oracle/residual fixture is vacuous or inaccurate");
  generation = build();
  (void)consume(generation, control); // Warm every completion path before its measured repeat.
  generation = build();
  before = session->transferSnapshot();
  const auto result = consume(generation, control);
  const auto after = session->transferSnapshot();
  require(result.candidate_ready && result.solve.converged && result.solve.iterations == solve.iterations &&
      result.solve.operator_applications == solve.operator_applications, "ENERGY SR converged diagnostics differ");
  near(result.gradient_norm, gradient_norm);
  near(result.direction_norm, std::sqrt(direction_square));
  near(result.metric_norm, std::sqrt(contraction));
  double max_candidate = 0;
  for (std::size_t i = 0; i < p; ++i)
  {
    near(candidate[i], expected[i]);
    max_candidate = std::max(max_candidate, std::abs(candidate[i] - expected[i]));
    if (i < 2 || i + 1 == p) require(candidate[i] == parameters[i], "ENERGY SR changed frozen canonical bits");
  }
  require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls,
      "ENERGY SR warmed DEVICE allocation/free");
  const auto q = p / 4096 + (p % 4096 != 0);
  const auto ingress_download = 8 + 16 * (p / 128 + (p % 128 != 0));
  const auto uploads = after.uploaded_bytes - before.uploaded_bytes;
  const auto downloads = after.downloaded_bytes - before.downloaded_bytes;
  require(uploads == 8 * p + (applications + 1) * (48 * b * ne + 8 * b) &&
      downloads == ingress_download + (applications + 1) * metric_download + 24 * dot_calls +
          16 * q * scan_calls + 96 + 3 * 16 * q + 8 * p,
      "ENERGY SR terminal trace differs from ingress+PCG+undamped diagnostics+candidate");
  std::cout << "ENERGY SR terminal uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
            << " downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes
            << " cpu_solve_dots=" << dot_calls << " cpu_solve_scans=" << scan_calls
            << " cpu_solve_applications=" << applications << " scope=device_wide\n";
  auto bounded = control;
  bounded.maximum_iterations = 0;
  std::fill(candidate.begin(), candidate.end(), 91);
  generation = build();
  before = session->transferSnapshot();
  const auto stopped = consume(generation, bounded);
  const auto stopped_after = session->transferSnapshot();
  require(!stopped.candidate_ready && stopped.solve.stop_reason == Stop::ITERATION_LIMIT,
      "ENERGY SR nonconverged solve exposed candidate");
  for (double value : candidate) require(value == 91, "ENERGY SR nonconverged completion overwrote caller");
  PcgCpuReference bounded_reference(scores, weights, rhs, damping);
  wftrain::KrylovSolveDiagnostics bounded_solve;
  wftrain::krylov_detail::runPreconditionedConjugateGradient(bounded_reference, bounded, p, bounded_solve);
  require(stopped.solve.stop_reason == bounded_solve.stop_reason &&
      stopped.solve.operator_applications == bounded_solve.operator_applications &&
      stopped_after.uploaded_bytes - before.uploaded_bytes == 8 * p &&
      stopped_after.downloaded_bytes - before.downloaded_bytes == ingress_download +
          24 * bounded_reference.dots + 16 * q * bounded_reference.scans &&
      stopped_after.allocation_calls == before.allocation_calls,
      "ENERGY SR nonconverged terminal transferred candidate or changed common stop trace");

  // Huge frozen gradients are stable-norm admitted, then project to exact zero RHS.
  const auto frozen_cost = Owner::energyRequiredBytes(p, b, ne, 3, all_frozen.size());
  auto frozen_owner = Owner::prepareEnergy(*session, positions, weights, all_frozen, version, damping,
      frozen_cost.device_bytes, frozen_cost.host_bytes);
  generation = build(ScoreContractionRole::ENERGY_T_EO_O, 1e200);
  const auto huge = frozen_owner->consumeEnergy(version, generation, parameters, weight, energy, learning_rate, control, candidate);
  require(huge.candidate_ready && huge.solve.stop_reason == Stop::ZERO_RIGHT_HAND_SIDE &&
      huge.direction_norm == 0 && huge.metric_norm == 0 && candidate == parameters,
      "ENERGY SR huge finite frozen gradient was not projected after stable norm");
  near(huge.gradient_norm * 1e-200, gradient_norm);
  generation = build(ScoreContractionRole::ENERGY_T_EO_O, 1e200);
  fails([&] { frozen_owner->consumeEnergy(version, generation, parameters, weight, 1e200,
      learning_rate, control, candidate); }); // All-P overflow must fail even though every lane is frozen.
  backend.publishParameters(parameters, version + 1, []() {});
  fails([&] { consume(generation, control); });
  std::cout << "PASS numerical ENERGY SR B=" << b << " P=" << p << " C=3 device_bytes=" << cost.device_bytes
            << " host_bytes=" << cost.host_bytes << " full_gradient_norm=" << gradient_norm
            << " direction_norm=" << result.direction_norm << " undamped_metric_norm=" << result.metric_norm
            << " max_cpu_candidate=" << max_candidate << " independent_residual=" << std::sqrt(residual_square)
            << " production_bound=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
