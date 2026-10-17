//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_ENERGY_SR_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_RETAINED_ENERGY_SR_VALIDATION_H
#include "PsiFormerDevicePcg.h"
#include "PsiFormerDeviceBatchBackend.h"
#include <iostream>

namespace qmcplusplus::psiformer::device
{
/// Compare retained lifecycle against qualified ordinary C0 on identical physical inputs.
inline void validateRetainedEnergySr(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& caller_image)
{
  using Owner = PreparedDevicePcg;
  using Stop = wftrain::KrylovStopReason;
  const auto require = [](bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
  };
  const auto fails = [&](auto action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Retained ENERGY SR accepted invalid lifecycle/input");
  };
  const auto near = [&](double actual, double expected) {
    require(device_math::isFiniteBinary64(actual) && device_math::isFiniteBinary64(expected) &&
        std::abs(actual - expected) <= 3e-10 * (1 + std::abs(expected)), "Retained ENERGY SR differs from ordinary C0");
  };
  const auto p = plan.parameterCount(), ne = model.ne, first_version = caller_image->version() + 1;
  constexpr std::size_t b = 5;
  constexpr double damping = 0.5, learning_rate = 0.0017;
  require(p == 3922 && ne == 4 && model.cfg.nup == 2 && model.cfg.nconfig >= 3,
      "Retained ENERGY SR requires tiny real/open P3922/Ne4 fixture");
  auto parameters = model.p.flat_values();
  // Caller-held image0 cannot make this dedicated successor's lifetime test vacuous.
  auto image1 = caller_image->makeUpdatedImage(parameters, first_version);
  std::weak_ptr<const PreparedDeviceModel> weak1 = image1;
  PreparedDeviceBatchBackend backend(image1);
  auto session = backend.prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 4);
  image1.reset();
  session->prepareScoreContractions(3, first_version);
  const std::vector<Owner::FrozenInterval> frozen{{0, 2}, {p - 1, 1}};
  const auto cost = Owner::energyRequiredBytes(p, b, ne, 3, frozen.size());
  std::vector<double> positions(3 * b * ne), weights(b), masses(ne), drift(3 * ne), imaginary(3 * ne);
  std::vector<double> candidate(p, 91), expected(p), zero(p);
  for (std::size_t i = 0; i < ne; ++i) masses[i] = 0.7 + 0.13 * i;
  for (std::size_t i = 0; i < drift.size(); ++i) drift[i] = 0.01 * (int(i % 5) - 2);
  double weight = 0, energy = 0;
  const auto population = [&](std::size_t step) {
    weight = energy = 0;
    for (std::size_t s = 0; s < b; ++s)
    {
      std::copy_n(model.cfg.electrons.x.data() + ((s + step) % model.cfg.nconfig) * 3 * ne,
          3 * ne, positions.data() + s * 3 * ne);
      weights[s] = 0.8 + 0.07 * s + 0.03 * step;
      weight += weights[s]; energy += weights[s] * (-2.3 + 0.09 * s);
    }
    energy /= weight;
  };
  population(0);
  const auto build = [&](std::size_t version, ScoreContractionRole role = ScoreContractionRole::ENERGY_T_EO_O,
                         PreparedDeviceProductSession* selected = nullptr) {
    auto& target = selected ? *selected : *session;
    const auto generation = target.beginScoreContraction(3, version, role);
    std::vector<std::complex<double>> score(3), kinetic(3), imaginary_weights(3);
    for (std::size_t s = 0; s < b; ++s)
    {
      score[1] = weights[s] * (-2.3 + 0.09 * s); score[2] = weights[s]; kinetic[0] = weights[s];
      target.accumulateKineticContraction(positions.data() + s * 3 * ne, drift.data(), imaginary.data(),
          masses.data(), score.data(), kinetic.data(), imaginary_weights.data(), 3, version);
    }
    target.finishScoreContraction(version);
    return generation;
  };
  wftrain::KrylovSolverControl control;
  control.relative_tolerance = 1e-8; control.absolute_tolerance = 1e-13;
  control.maximum_iterations = 20; control.maximum_operator_applications = 24;
  auto before = session->transferSnapshot();
  auto retained = Owner::prepareRetainedEnergy(*session, positions, weights, frozen, first_version, damping,
      cost.device_bytes, cost.host_bytes);
  auto after = session->transferSnapshot();
  require(after.allocation_calls - before.allocation_calls == 15 && after.free_calls == before.free_calls &&
      retained->allocatedBytes().device_bytes == cost.device_bytes && retained->allocatedBytes().host_bytes == cost.host_bytes,
      "Retained ENERGY SR cold15 allocation/exact capacity contract failed");
  fails([&] { retained->uploadRhs(zero, first_version); });
  double max_candidate = 0;
  for (std::size_t step = 0; step < 2; ++step)
  {
    const auto version = first_version + step;
    population(step);
    auto generation = build(version); // Session catches up BEFORE terminal binding.
    if (step == 1) require(weak1.expired(), "Retained ENERGY SR pinned old immutable P after Session rebind");
    auto ordinary = Owner::prepareEnergy(*session, positions, weights, frozen, version, damping,
        cost.device_bytes, cost.host_bytes);
    const auto old_rhs = ordinary->uploadRhs(zero, version);
    const auto old_iterate = ordinary->solve(old_rhs, control).iterate;
    generation = build(version);
    before = session->transferSnapshot();
    const auto oracle = ordinary->consumeEnergy(version, generation, parameters, weight, energy, learning_rate, control, expected);
    const auto ordinary_after = session->transferSnapshot();
    require(oracle.candidate_ready && oracle.solve.converged, "Retained ENERGY SR ordinary oracle did not converge");
    fails([&] { ordinary->solve(old_rhs, control); });
    fails([&] { ordinary->downloadIterate(old_iterate, candidate); });
    const auto oracle_upload = ordinary_after.uploaded_bytes - before.uploaded_bytes;
    const auto oracle_download = ordinary_after.downloaded_bytes - before.downloaded_bytes;
    ordinary.reset(); // Do not let the oracle pin image1 across publication.
    generation = build(version);
    before = session->transferSnapshot();
    const auto result = retained->consumeRetainedEnergy(positions, weights, frozen, version, generation, parameters,
        weight, energy, damping, learning_rate, control, cost.device_bytes, cost.host_bytes, candidate);
    after = session->transferSnapshot();
    require(result.candidate_ready && result.solve.stop_reason == oracle.solve.stop_reason &&
        result.solve.converged == oracle.solve.converged && result.solve.iterations == oracle.solve.iterations &&
        result.solve.operator_applications == oracle.solve.operator_applications, "Retained ENERGY SR diagnostic counts changed");
    near(result.gradient_norm, oracle.gradient_norm); near(result.direction_norm, oracle.direction_norm);
    near(result.metric_norm, oracle.metric_norm); near(result.solve.initial_residual_norm, oracle.solve.initial_residual_norm);
    near(result.solve.final_residual_norm, oracle.solve.final_residual_norm);
    near(result.solve.relative_residual_norm, oracle.solve.relative_residual_norm);
    for (std::size_t i = 0; i < p; ++i)
    {
      near(candidate[i], expected[i]); max_candidate = std::max(max_candidate, std::abs(candidate[i] - expected[i]));
      if (i < 2 || i + 1 == p) require(candidate[i] == parameters[i], "Retained ENERGY SR altered frozen bits");
    }
    require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls &&
        after.uploaded_bytes - before.uploaded_bytes == oracle_upload &&
        after.downloaded_bytes - before.downloaded_bytes == oracle_download &&
        retained->allocatedBytes().device_bytes == cost.device_bytes && retained->allocatedBytes().host_bytes == cost.host_bytes,
        "Retained ENERGY SR terminal changed allocation/traffic/capacity law");
    fails([&] { retained->uploadRhs(zero, version); });
    fails([&] { retained->solve(old_rhs, control); });
    fails([&] { retained->downloadIterate(old_iterate, expected); });
    std::cout << "retained_energy_sr step=" << step << " version=" << version << " iterations=" << result.solve.iterations
              << " applications=" << result.solve.operator_applications << " gradient_norm=" << result.gradient_norm
              << " direction_norm=" << result.direction_norm << " metric_norm=" << result.metric_norm
              << " uploaded_bytes=" << oracle_upload << " downloaded_bytes=" << oracle_download
              << " device_bytes=" << cost.device_bytes << " host_bytes=" << cost.host_bytes << " scope=device_wide\n";
    if (step == 0)
    {
      require(!weak1.expired(), "Dedicated image lifetime fixture lost active Session image early");
      parameters = candidate;
      backend.publishParameters(parameters, version + 1, []() {});
    }
  }

  // Distinct destructive caches: no fallback/recreation/retry after a failed transaction.
  const auto version = first_version + 1;
  auto same_generation = build(version);
  fails([&] { retained->consumeRetainedEnergy(positions, weights, frozen, version, same_generation, parameters,
      weight, energy, damping, learning_rate, control, cost.device_bytes, cost.host_bytes, candidate); });
  for (int fault = 0; fault < 9; ++fault)
  {
    auto cache = Owner::prepareRetainedEnergy(*session, positions, weights, frozen, version, damping,
        cost.device_bytes, cost.host_bytes);
    auto bad_positions = positions, bad_weights = weights;
    auto bad_frozen = frozen;
    auto device_cap = cost.device_bytes, host_cap = cost.host_bytes;
    if (fault == 0) bad_positions.pop_back();
    if (fault == 1) bad_weights.pop_back();
    if (fault == 2) --bad_frozen[0].second;
    if (fault == 3) ++device_cap; // Sufficient but CHANGED admitted cap must not reconfigure storage.
    if (fault == 4) --host_cap;
    if (fault == 8) bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
    const auto generation = build(version, fault == 6 ? ScoreContractionRole::GENERIC : ScoreContractionRole::ENERGY_T_EO_O);
    auto bounded = control;
    if (fault == 7) bounded.maximum_iterations = 0;
    std::fill(candidate.begin(), candidate.end(), 91);
    before = session->transferSnapshot();
    const auto invoke = [&]() {
      return cache->consumeRetainedEnergy(bad_positions, bad_weights, bad_frozen, version,
          generation + (fault == 5), parameters, weight, energy, damping, learning_rate, bounded,
          device_cap, host_cap, candidate);
    };
    if (fault == 7)
    {
      const auto stopped = invoke();
      require(!stopped.candidate_ready && stopped.solve.stop_reason == Stop::ITERATION_LIMIT,
          "Retained ENERGY SR nonconvergence exposed candidate");
    }
    else fails(invoke);
    for (double value : candidate) require(value == 91, "Failed retained ENERGY SR changed candidate");
    after = session->transferSnapshot();
    require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls,
        "Failed retained ENERGY SR transaction allocated/freed storage");
    const auto fresh = build(version);
    fails([&] { cache->consumeRetainedEnergy(positions, weights, frozen, version, fresh, parameters,
        weight, energy, damping, learning_rate, control, cost.device_bytes, cost.host_bytes, candidate); });
  }
  // DEVICE publication is a separate numerical contract; no public driver admission.
  auto device_parameters = model.p.flat_values();
  auto device_backend = std::make_shared<PreparedDeviceBatchBackend>(
      caller_image->makeUpdatedImage(device_parameters, first_version));
  auto device_session = device_backend->prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 4);
  device_session->prepareScoreContractions(3, first_version);
  population(0);
  std::shared_ptr<Owner> cache = Owner::prepareRetainedEnergy(*device_session, positions, weights, frozen,
      first_version, damping, cost.device_bytes, cost.host_bytes);
  pf::DirectScoreExecutor cpu_score(model, plan);
  pf::DirectKineticExecutor cpu_kinetic(model, plan);
  auto score_workspace = cpu_score.makeWorkspace();
  auto kinetic_workspace = cpu_kinetic.makeWorkspace();
  const auto original_parameters = model.p.flat_values();
  for (std::size_t step = 0; step < 3; ++step)
  {
    const auto version = first_version + step;
    population(step);
    model.p.set_flat_values(device_parameters);
    std::vector<double> scores(b * p), t(p), eo(p), o(p), rhs(p), cpu_expected(p), actual(p);
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      const auto* position = positions.data() + sample * 3 * ne;
      score_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
      kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
      const auto score = cpu_score.evaluate(*score_workspace);
      const auto kinetic = cpu_kinetic.evaluate(*kinetic_workspace, drift.data(), drift.size(), masses.data(), masses.size());
      for (std::size_t i = 0; i < p; ++i)
      {
        scores[sample * p + i] = score.parameter_score[i];
        t[i] += weights[sample] * kinetic.kinetic_parameter_response[i];
        eo[i] += weights[sample] * (-2.3 + .09 * sample) * score.parameter_score[i];
        o[i] += weights[sample] * score.parameter_score[i];
      }
    }
    double gradient_norm = 0;
    for (std::size_t i = 0; i < p; ++i)
    {
      const double gradient = 2 * (t[i] + eo[i] - energy * o[i]) / weight;
      gradient_norm = std::hypot(gradient_norm, gradient);
      rhs[i] = i < 2 || i + 1 == p ? 0 : gradient;
    }
    PcgCpuReference cpu_reference(scores, weights, rhs, damping);
    wftrain::KrylovSolveDiagnostics cpu_solve;
    wftrain::krylov_detail::runPreconditionedConjugateGradient(cpu_reference, control, p, cpu_solve);
    require(cpu_solve.converged, "DEVICE SR CPU oracle failed to converge");
    double displacement = 0;
    for (std::size_t i = 0; i < p; ++i)
    {
      volatile double product = -learning_rate * cpu_reference.get(wftrain::krylov_detail::PcgSlot::X)[i];
      cpu_expected[i] = device_parameters[i] + product;
      displacement = std::hypot(displacement, cpu_expected[i] - device_parameters[i]);
    }
    const auto build_device = [&] { return build(version, ScoreContractionRole::ENERGY_T_EO_O, device_session.get()); };
    const auto consume_device = [&](std::size_t generation, std::size_t cap) {
      return cache->consumeRetainedEnergyDevice(cache, positions, weights, frozen, version, generation,
          weight, energy, damping, learning_rate, control, cost.device_bytes, cost.host_bytes, cap);
    };
    auto generation = build_device();
    const auto cap_before = device_session->transferSnapshot();
    fails([&] { consume_device(generation, 8 * p - 1); });
    require(device_session->transferSnapshot().allocation_calls == cap_before.allocation_calls,
        "DEVICE SR denied snapshot allocated");
    // Successful discard is healthy, unlike an actually failed numerical owner.
    {
      auto discarded = consume_device(build_device(), 8 * p);
      auto source = device_backend->currentImage();
      auto successor = source->makeSharedUpdatedImage(*discarded.snapshot);
      cache->validateDeviceAcceptance(*discarded.snapshot, successor);
      { PreparedDeviceBatchBackend::PreparedDevicePublication cancelled(device_backend, source, successor); }
      cache->rejectDeviceCandidate();
      fails([&] { source->makeSharedUpdatedImage(*discarded.snapshot); });
      require(discarded.snapshot->deviceBytes() == 8 * p, "Rejected DEVICE SR bytes lost ownership");
    }
    generation = build_device();
    auto ordinary = Owner::prepareEnergy(*device_session, positions, weights, frozen, version, damping,
        cost.device_bytes, cost.host_bytes);
    const auto oracle_before = device_session->transferSnapshot();
    const auto oracle = ordinary->consumeEnergy(version, generation, device_parameters, weight, energy,
        learning_rate, control, expected);
    const auto oracle_after = device_session->transferSnapshot();
    ordinary.reset();
    generation = build_device();
    const auto device_before = device_session->transferSnapshot();
    auto proposal = consume_device(generation, 8 * p);
    const auto device_after = device_session->transferSnapshot();
    require(device_after.allocation_calls == device_before.allocation_calls + 1 &&
        device_after.live_bytes == device_before.live_bytes + 8 * p &&
        device_after.uploaded_bytes - device_before.uploaded_bytes ==
            oracle_after.uploaded_bytes - oracle_before.uploaded_bytes - 8 * p &&
        device_after.downloaded_bytes - device_before.downloaded_bytes ==
            oracle_after.downloaded_bytes - oracle_before.downloaded_bytes - 8 * p + 64 * (p / 4096 + (p % 4096 != 0)) + 24,
        "DEVICE SR warm traffic/single immutable allocation changed");
    require(proposal.owner == cache && proposal.result.solve.converged &&
        proposal.result.solve.iterations == cpu_solve.iterations &&
        proposal.result.solve.operator_applications == cpu_solve.operator_applications,
        "DEVICE SR solve/strong lease differs from CPU");
    near(proposal.result.gradient_norm, gradient_norm);
    near(proposal.result.direction_norm, oracle.direction_norm);
    near(proposal.result.metric_norm, oracle.metric_norm);
    near(proposal.displacement_norm, displacement);
    auto source = device_backend->currentImage();
    auto successor = source->makeSharedUpdatedImage(*proposal.snapshot);
    auto foreign = std::shared_ptr<Owner>(Owner::prepareRetainedEnergy(*device_session, positions, weights, frozen,
        version, damping, cost.device_bytes, cost.host_bytes));
    fails([&] { foreign->validateDeviceAcceptance(*proposal.snapshot, successor); });
    fails([&] { cache->validateDeviceAcceptance(*proposal.snapshot, source); });
    cache->validateDeviceAcceptance(*proposal.snapshot, successor);
    {
      PreparedDeviceBatchBackend::PreparedDevicePublication publication(device_backend, source, successor);
      const auto commit_before = device_session->transferSnapshot();
      publication.commit();
      cache->acceptPreparedDeviceCandidate(*proposal.snapshot, successor);
      const auto commit_after = device_session->transferSnapshot();
      require(commit_after.allocation_calls == commit_before.allocation_calls &&
          commit_after.free_calls == commit_before.free_calls &&
          commit_after.uploaded_bytes == commit_before.uploaded_bytes &&
          commit_after.downloaded_bytes == commit_before.downloaded_bytes &&
          commit_after.device_copied_bytes == commit_before.device_copied_bytes,
          "DEVICE SR no-fail commit called runtime or retired numeric storage");
      publication.unlock();
    }
    fails([&] { source->makeSharedUpdatedImage(*proposal.snapshot); });
    successor->downloadCanonical(actual); // Explicit cold oracle, outside warm/commit accounting.
    for (std::size_t i = 0; i < p; ++i)
    {
      near(actual[i], cpu_expected[i]); near(actual[i], expected[i]);
      if (i < 2 || i + 1 == p) require(actual[i] == device_parameters[i], "DEVICE SR altered frozen bits");
    }
    device_parameters = actual;
    std::cout << "device_energy_sr step=" << step << " source_version=" << version
              << " cpu_oracle=1 shared_snapshot_bytes=" << 8 * p << " discard_retry=1 nofail_commit=1\n";
  }
  // No candidate escapes a stopped solve, and failed storage requires explicit re-prepare.
  const auto stopped_version = first_version + 3;
  auto stopped_generation = build(stopped_version, ScoreContractionRole::ENERGY_T_EO_O, device_session.get());
  std::shared_ptr<Owner> stopped_owner = Owner::prepareRetainedEnergy(*device_session, positions, weights, frozen,
      stopped_version, damping, cost.device_bytes, cost.host_bytes);
  auto stopped_control = control;
  stopped_control.maximum_iterations = 0;
  fails([&] { stopped_owner->consumeRetainedEnergyDevice(stopped_owner, positions, weights, frozen,
      stopped_version, stopped_generation, weight, energy, damping, learning_rate, stopped_control,
      cost.device_bytes, cost.host_bytes, 8 * p); });
  stopped_generation = build(stopped_version, ScoreContractionRole::ENERGY_T_EO_O, device_session.get());
  fails([&] { stopped_owner->consumeRetainedEnergyDevice(stopped_owner, positions, weights, frozen,
      stopped_version, stopped_generation, weight, energy, damping, learning_rate, control,
      cost.device_bytes, cost.host_bytes, 8 * p); });
  require(device_backend->currentImage()->version() == stopped_version, "Failed DEVICE SR published a model");
  model.p.set_flat_values(original_parameters);
  // Numerical-only complete ECP C3 -> retained PCG. Component/native guards are untouched.
  {
    constexpr std::size_t samples = 2, knots = 9;
    const std::vector<double> w{1, 2}, e{-2.3, -1.9};
    const double total = 3, mean = (w[0] * e[0] + w[1] * e[1]) / total;
    auto values = original_parameters;
    auto ecp_backend = std::make_shared<PreparedDeviceBatchBackend>(caller_image->makeUpdatedImage(values, first_version));
    auto ecp_session = ecp_backend->prepareProductSession(ProductSessionProfile::SCORE_AND_KINETIC, 4);
    ecp_session->prepareScoreContractions(3, first_version);
    std::vector<double> accepted(samples * 3 * ne);
    for (std::size_t s = 0; s < samples; ++s)
      std::copy_n(model.cfg.electrons.x.data() + s * 3 * ne, 3 * ne, accepted.data() + s * 3 * ne);
    require(!std::equal(accepted.begin(), accepted.begin() + 3 * ne, accepted.begin() + 3 * ne),
        "ECP SR oracle requires two distinct accepted configurations");
    const auto bytes = Owner::energyRequiredBytes(p, samples, ne, 3, frozen.size());
    std::shared_ptr<Owner> owner = Owner::prepareRetainedEnergy(*ecp_session, accepted, w, frozen, first_version,
        damping, bytes.device_bytes, bytes.host_bytes);
    for (std::size_t step = 0; step < 2; ++step)
    {
      const auto version = first_version + step;
      model.p.set_flat_values(values);
      std::vector<double> score(samples * p), complete_t(p), rhs(p), delta(p), wanted(p), actual(p);
      std::vector<double> virtual_positions(samples * knots * 3 * ne), complete_weights(samples * knots);
      double squared = 0, dot = 0, gradient_norm = 0, nonzero_ecp = 0;
      for (std::size_t s = 0; s < samples; ++s)
      {
        const auto* position = accepted.data() + s * 3 * ne;
        score_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
        const auto base = cpu_score.evaluate(*score_workspace);
        require(base.sign != 0 && device_math::isFiniteBinary64(base.logabs), "ECP SR accepted node");
        const double sign = base.sign, logabs = base.logabs;
        std::copy_n(base.parameter_score.data, p, score.data() + s * p);
        kinetic_workspace->setPositions(pf::GeometryPositionView::interleaved(position, ne));
        const auto t = cpu_kinetic.evaluate(*kinetic_workspace, drift.data(), drift.size(), masses.data(), masses.size());
        for (std::size_t i = 0; i < p; ++i)
          complete_t[i] += w[s] * t.kinetic_parameter_response[i];
        for (std::size_t q = 0; q < knots; ++q)
        {
          const auto moved = q < 5 ? 0 : 1; // Existing signed ragged5+4 grid, streamed without Q*P storage.
          auto* point = virtual_positions.data() + (s * knots + q) * 3 * ne;
          std::copy_n(position, 3 * ne, point);
          point[3 * moved] += .003 * (q + 1); point[3 * moved + 1] -= .011; point[3 * moved + 2] += .007;
          score_workspace->setPositions(pf::GeometryPositionView::interleaved(point, ne));
          const auto displaced = cpu_score.evaluate(*score_workspace);
          const double ratio = displaced.sign / sign * std::exp(displaced.logabs - logabs);
          const double factor = (q % 2 ? -.13 : .21) / (q + 1) * ratio;
          require(device_math::isFiniteBinary64(factor), "ECP SR complete ratio/weight is nonfinite");
          complete_weights[s * knots + q] = factor;
          for (std::size_t i = 0; i < p; ++i)
          {
            const double contribution = w[s] * factor * (displaced.parameter_score[i] - score[s * p + i]);
            complete_t[i] += contribution;
            nonzero_ecp = std::max(nonzero_ecp, std::abs(contribution));
          }
        }
      }
      require(nonzero_ecp > 0, "ECP SR test silently lost the nonlocal contribution");
      // Independent two-configuration covariance and Sherman-Morrison solve, not GPU PCG replay.
      for (std::size_t i = 0; i < p; ++i)
      {
        const double g = 2 / total * (complete_t[i] + w[0] * (e[0] - mean) * score[i] +
            w[1] * (e[1] - mean) * score[p + i]);
        gradient_norm = std::hypot(gradient_norm, g);
        const bool active = i >= 2 && i + 1 != p;
        rhs[i] = active ? g : 0; delta[i] = active ? score[i] - score[p + i] : 0;
        squared += delta[i] * delta[i]; dot += delta[i] * rhs[i];
      }
      const double a = w[0] * w[1] / (total * total);
      double displacement = 0;
      for (std::size_t i = 0; i < p; ++i)
      {
        const double direction = rhs[i] / damping - a * delta[i] * dot / (damping * (damping + a * squared));
        volatile double product = -learning_rate * direction;
        wanted[i] = values[i] + product;
        displacement = std::hypot(displacement, wanted[i] - values[i]);
      }
      const auto build_complete = [&](bool node = false) {
        const auto generation = ecp_session->beginScoreContraction(3, version, ScoreContractionRole::ENERGY_T_EO_O);
        try
        {
          for (std::size_t s = 0; s < samples; ++s)
          {
            const auto* position = accepted.data() + s * 3 * ne;
            const std::complex<double> scores_c[]{0, w[s] * e[s], w[s]}, kinetic_c[]{w[s], 0, 0}, imaginary_c[3]{};
            ecp_session->accumulateKineticContraction(position, drift.data(), imaginary.data(), masses.data(),
                scores_c, kinetic_c, imaginary_c, 3, version);
            double sum = 0;
            for (std::size_t q = 0; q < knots; ++q)
            {
              const auto* point = virtual_positions.data() + (s * knots + q) * 3 * ne;
              std::vector<double> zero_node;
              if (node && s == 0 && q == 0)
              {
                zero_node.assign(point, point + 3 * ne);
                std::copy_n(zero_node.data() + 3, 3, zero_node.data()); point = zero_node.data();
              }
              const double factor = node && s == 0 && q == 0 ? 0 : complete_weights[s * knots + q];
              const std::complex<double> coefficient[]{w[s] * factor, 0, 0};
              ecp_session->accumulateScoreContraction(point, coefficient, 3, version);
              sum += factor;
            }
            const std::complex<double> reference[]{-w[s] * sum, 0, 0};
            ecp_session->accumulateScoreContraction(position, reference, 3, version);
          }
          ecp_session->finishScoreContraction(version); // ONE finite admission after complete T+ECP.
          return generation;
        }
        catch (...) { ecp_session->abortScoreContraction(); throw; }
      };
      if (step == 0) fails([&] { build_complete(true); }); // Zero weight does not excuse a nodal score.
      auto generation = build_complete();
      auto ordinary = Owner::prepareEnergy(*ecp_session, accepted, w, frozen, version, damping,
          bytes.device_bytes, bytes.host_bytes);
      const auto host_before = ecp_session->transferSnapshot();
      const auto host = ordinary->consumeEnergy(version, generation, values, total, mean, learning_rate, control, actual);
      const auto host_after = ecp_session->transferSnapshot();
      require(host.candidate_ready && host.solve.converged, "ECP SR low-level HOST oracle did not converge");
      for (std::size_t i = 0; i < p; ++i) near(actual[i], wanted[i]);
      ordinary.reset();
      const auto propose = [&] {
        return owner->consumeRetainedEnergyDevice(owner, accepted, w, frozen, version, build_complete(),
            total, mean, damping, learning_rate, control, bytes.device_bytes, bytes.host_bytes, 8 * p);
      };
      { auto discarded = propose(); owner->rejectDeviceCandidate(); }
      generation = build_complete();
      const auto warm_before = ecp_session->transferSnapshot();
      auto proposal = owner->consumeRetainedEnergyDevice(owner, accepted, w, frozen, version, generation,
          total, mean, damping, learning_rate, control, bytes.device_bytes, bytes.host_bytes, 8 * p);
      const auto warm_after = ecp_session->transferSnapshot();
      require(proposal.result.solve.converged && proposal.result.solve.iterations == host.solve.iterations &&
          proposal.result.solve.operator_applications == host.solve.operator_applications &&
          warm_after.live_bytes == warm_before.live_bytes + 8 * p &&
          warm_after.allocation_calls == warm_before.allocation_calls + 1, "ECP SR snapshot ownership/solve changed");
      require(warm_after.uploaded_bytes - warm_before.uploaded_bytes == host_after.uploaded_bytes - host_before.uploaded_bytes - 8 * p &&
          warm_after.downloaded_bytes - warm_before.downloaded_bytes == host_after.downloaded_bytes - host_before.downloaded_bytes - 8 * p +
              64 * (p / 4096 + (p % 4096 != 0)) + 24, "ECP SR DEVICE added a HOST P payload");
      near(proposal.result.gradient_norm, gradient_norm); near(proposal.displacement_norm, displacement);
      auto source = ecp_backend->currentImage();
      auto successor = source->makeSharedUpdatedImage(*proposal.snapshot);
      owner->validateDeviceAcceptance(*proposal.snapshot, successor);
      {
        PreparedDeviceBatchBackend::PreparedDevicePublication publication(ecp_backend, source, successor);
        const auto before_commit = ecp_session->transferSnapshot();
        publication.commit(); owner->acceptPreparedDeviceCandidate(*proposal.snapshot, successor);
        const auto after_commit = ecp_session->transferSnapshot();
        require(after_commit.allocation_calls == before_commit.allocation_calls && after_commit.free_calls == before_commit.free_calls &&
            after_commit.uploaded_bytes == before_commit.uploaded_bytes && after_commit.downloaded_bytes == before_commit.downloaded_bytes &&
            after_commit.device_copied_bytes == before_commit.device_copied_bytes, "ECP SR commit used numerical runtime");
        publication.unlock();
      }
      successor->downloadCanonical(actual); // Explicit cold full-P oracle, not warm traffic.
      for (std::size_t i = 0; i < p; ++i) near(actual[i], wanted[i]);
      require(actual[0] == values[0] && actual[1] == values[1] && actual.back() == values.back(), "ECP SR changed frozen bits");
      values = actual;
    }
    model.p.set_flat_values(original_parameters);
    std::cout << "PASS numerical ECP DEVICE SR two_updates=2 samples=2 Q=18 CPU_rank_one=1 HOST_oracle=1 zero_weight_node_retry=1 component_driver_guards_unchanged=1\n";
  }
  std::cout << "PASS DEVICE retained ENERGY SR three_updates=3 immutable_shared_successor=1 public_authority=0 raw_square_displacement_domain=1\n";
  std::cout << "PASS retained numerical ENERGY SR cold_device_allocations=15 two_successor_steps=2 weak_old_image_expired=1"
            << " max_ordinary_candidate=" << max_candidate << " aggregate_publication_peak_unchanged=0 production_bound=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
