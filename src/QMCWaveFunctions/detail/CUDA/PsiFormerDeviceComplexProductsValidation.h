//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_PRODUCTS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_PRODUCTS_VALIDATION_H
#include "PsiFormerDeviceComplexProducts.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexKineticExecutor.h"
#include <type_traits>

/// Tiny complete CPU weighted allP oracle, tails/finite/lifecycle/cap/transfer boundaries.
void validateComplexProducts(bool selected_two_cell = false)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using Owner = PreparedDeviceComplexProducts;
  using C = Owner::Complex;
  static_assert(!std::is_move_constructible_v<Owner> && !std::is_move_assignable_v<Owner>);
  static_assert(!std::is_copy_constructible_v<Owner> && !std::is_copy_assignable_v<Owner>);
  const auto require = [](bool valid, const char* message) { if (!valid) throw std::runtime_error(message); };
  const auto fails = [&](auto request) {
    bool rejected = false;
    try { request(); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "Complex products failed to reject an invalid request");
  };
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.periodic_axes = {true, true, true};
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  selectPairedTwoCellValidation(environment, selected_two_cell);
  const ModelShape shape{selected_two_cell ? 2u : 3u, selected_two_cell ? 2u : 1u, 2, 4, 16, 2, 2};
  auto configuration = configurations(environment);
  configuration.nup = selected_two_cell ? 2 : 3;
  configuration.ndown = selected_two_cell ? 2 : 1;
  pf::PsiFormer model(smallParameters(environment), configuration, 4, 16, 2, 2);
  const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
  auto parameters = model.p.flat_values();
  for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
  {
    const auto& interval = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
    for (std::size_t i = interval.begin; i < interval.end; ++i)
      parameters[i] = .09 * std::sin(double(i - interval.begin + 1));
  }
  model.p.set_flat_values(parameters);
  const std::array<double, 3> kappa = selected_two_cell ? std::array<double, 3>{} : std::array<double, 3>{.13, -.07, .04};
  pf::DirectComplexScoreExecutor cpu_score(model, plan);
  auto score_work = cpu_score.makeWorkspace();
  pf::DirectComplexKineticExecutor cpu_kinetic(model, plan, kappa);
  auto kinetic_work = cpu_kinetic.makeWorkspace(cpu_kinetic.requiredStorageBytes());
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = selected_two_cell
      ? PreparedDeviceModel::preparePairedTwoCellProfile(plan, parameters, model.cfg.nuclei.x)
      : PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "complex products device");
#else
  checkRuntime(cudaGetDevice(&device), "complex products device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  constexpr std::size_t b = 2, c = 3, targets = 3, chunk = 7;
  const auto p = plan.parameterCount();
  const auto cost = estimateComplexProductsCost(plan, targets, b, c, chunk);
  const auto prepare = [&](std::size_t dev, std::size_t host, std::size_t work) {
    return Owner::prepare(image, targets, b, c, chunk, queue, blas, dev, host, work);
  };
  const auto cold = metrics.snapshot();
  fails([&] { (void)prepare(cost.device_bytes - 1, cost.host_bytes, cost.matrix_work); });
  fails([&] { (void)prepare(cost.device_bytes, cost.host_bytes - 1, cost.matrix_work); });
  fails([&] { (void)prepare(cost.device_bytes, cost.host_bytes, cost.matrix_work - 1); });
  fails([&] { (void)prepare(0, cost.host_bytes, cost.matrix_work); });
  fails([&] { (void)prepare(cost.device_bytes, std::numeric_limits<std::size_t>::max(), cost.matrix_work); });
  require(metrics.snapshot().allocation_calls == cold.allocation_calls, "Complex products cap failure allocated device storage");
  auto owner = prepare(cost.device_bytes, cost.host_bytes, cost.matrix_work);
  require(metrics.snapshot().live_bytes - cold.live_bytes == cost.device_bytes, "Complex products device inventory mismatch");
  const auto fingerprint = owner->storageFingerprint();
  std::vector<double> positions, masses, one_position(12), one_mass(4);
  for (std::size_t sample = 0; sample < b; ++sample)
  {
    for (std::size_t i = 0; i < 12; ++i)
      positions.push_back(model.cfg.electrons.x[i] + .002 * sample * std::sin(double(i + 1)));
    for (double mu : {1., 1 / 1.3, 1 / .8, 1 / 1.7}) masses.push_back(mu * (1 + .01 * sample));
  }
  std::array<std::vector<C>, c> coefficients{{{{.7, -.2}, {-.4, .3}}, {{.8, .1}, {-.3, -.4}}, {{-.2, .5}, {.6, -.1}}}};
  std::vector<Owner::Channel> channels{{Owner::Role::SCORE, coefficients[0].data(), b},
      {Owner::Role::KINETIC, coefficients[1].data(), b}, {Owner::Role::SCORE, coefficients[2].data(), b}};
  std::vector<C> expected(c * p), expected_hermitian(c * p);
  for (std::size_t sample = 0; sample < b; ++sample)
  {
    const auto view = pf::GeometryPositionView::interleaved(positions.data() + 12 * sample, 4);
    score_work->setPositions(view);
    kinetic_work->setPositions(view);
    const auto score = cpu_score.evaluate(*score_work);
    const auto kinetic = cpu_kinetic.evaluate(*kinetic_work, masses.data() + 4 * sample, 4);
    for (std::size_t channel = 0; channel < c; ++channel)
      for (std::size_t i = 0; i < p; ++i)
      {
        const auto response = channels[channel].role == Owner::Role::SCORE ? score.parameter_score[i] : kinetic.parameter_response[i];
        expected[channel * p + i] += coefficients[channel][sample] * response;
        expected_hermitian[channel * p + i] += coefficients[channel][sample] * std::conj(response);
      }
  }
  // Independent numerical-owner transfer baseline excludes all explicit canonical downloads.
  auto score = PreparedDeviceComplexScore::prepare(image, queue, blas, cost.score.device_bytes);
  auto kinetic = PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas,
      cost.kinetic.device_bytes, cost.kinetic.matrix_work);
  const auto baseline_start = metrics.snapshot();
  std::size_t score_downloads = 0, score_uploads = 0;
  for (std::size_t sample = 0; sample < b; ++sample)
  {
    std::copy_n(positions.data() + sample * 12, 12, one_position.data());
    std::copy_n(masses.data() + sample * 4, 4, one_mass.data());
    const auto score_before = metrics.snapshot();
    score->evaluate(one_position);
    score_downloads += metrics.snapshot().downloaded_bytes - score_before.downloaded_bytes;
    score_uploads += metrics.snapshot().uploaded_bytes - score_before.uploaded_bytes;
    kinetic->evaluate(one_position, one_mass, kappa);
  }
  const auto baseline_download = metrics.snapshot().downloaded_bytes - baseline_start.downloaded_bytes;
  double maximum = 0;
  const auto warm = metrics.snapshot();
  owner->evaluate(positions, masses, kappa, channels);
  const auto first = owner->resultView();
  owner->validate(first);
  require(metrics.snapshot().downloaded_bytes - warm.downloaded_bytes ==
      baseline_download + 16 * c * p + sizeof(DerivativeFiniteStatus) * cost.statuses,
      "Complex products downloaded canonical P per sample or omitted final/status transfer");
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    maximum = std::max(maximum, std::abs(first.data[i] - expected[i]));
    require(std::abs(first.data[i] - expected[i]) <= 3e-9 * (1 + std::abs(expected[i])), "Complex products CPU allP mismatch");
  }
  auto invalid = first;
  invalid.owner = nullptr;
  fails([&] { owner->validate(invalid); });
  invalid = first; invalid.image = nullptr;
  fails([&] { owner->validate(invalid); });
  invalid = first; ++invalid.data;
  fails([&] { owner->validate(invalid); });
  owner->evaluate(positions, masses, kappa, channels);
  fails([&] { owner->validate(first); });
  require(owner->resultView().data == first.data && owner->storageFingerprint() == fingerprint,
      "Complex products warm prepared addresses changed");
  require(metrics.snapshot().allocation_calls == warm.allocation_calls && metrics.snapshot().free_calls == warm.free_calls,
      "Complex products warm device allocation/free");
  // A bad LAST coefficient must fail before even the first sample can launch/upload.
  const auto untouched = metrics.snapshot();
  coefficients[2].back() = {std::numeric_limits<double>::quiet_NaN(), 0};
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  fails([&] { (void)owner->resultView(); });
  require(metrics.snapshot().uploaded_bytes == untouched.uploaded_bytes &&
      metrics.snapshot().downloaded_bytes == untouched.downloaded_bytes, "Complex products late-input preflight launched");
  coefficients[2].back() = {.6, -.1};
  const auto saved_mass = masses.back(); masses.back() = -1;
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  masses.back() = saved_mass;
  const auto saved_position = positions.back(); positions.back() = std::numeric_limits<double>::infinity();
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  positions.back() = saved_position;
  auto invalid_kappa = kappa;
  invalid_kappa.back() = std::numeric_limits<double>::infinity();
  fails([&] { owner->evaluate(positions, masses, invalid_kappa, channels); });
  channels[0].role = static_cast<Owner::Role>(99);
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  channels[0].role = Owner::Role::SCORE;
  --channels[0].count;
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  ++channels[0].count;
  const auto saved_coefficient = coefficients[0][0];
  coefficients[0][0] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  fails([&] { owner->evaluate(positions, masses, kappa, channels); });
  fails([&] { (void)owner->resultView(); });
  coefficients[0][0] = saved_coefficient;
  owner->evaluate(positions, masses, kappa, channels);
  owner->validate(owner->resultView());
  owner->evaluate(positions, masses, kappa, channels, true);
  const auto hermitian = owner->resultView();
  for (std::size_t i = 0; i < expected_hermitian.size(); ++i)
    require(std::abs(hermitian.data[i] - expected_hermitian[i]) <= 3e-9 * (1 + std::abs(expected_hermitian[i])),
        "Complex products Hermitian CPU allP mismatch");
  // Native bounded output has identical DEVICE ownership and no retained C*P HOST array.
  const auto chunk_cost = estimateComplexProductsCost(plan, targets, b, c, chunk, ComplexProductsHostOutput::CHUNKED);
  require(chunk_cost.device_bytes == cost.device_bytes && cost.host_bytes - chunk_cost.host_bytes == 16 * (c*p-chunk),
      "Complex products bounded HOST cost did not remove exactly C*P minus one chunk");
  const auto before_chunk_owner = metrics.snapshot();
  fails([&] { Owner::prepare(image, targets, b, c, chunk, queue, blas, chunk_cost.device_bytes,
      chunk_cost.host_bytes-1, chunk_cost.matrix_work, ComplexProductsHostOutput::CHUNKED); });
  require(metrics.snapshot().allocation_calls == before_chunk_owner.allocation_calls,
      "Complex products bounded HOST denial allocated DEVICE storage");
  auto bounded = Owner::prepare(image, targets, b, c, chunk, queue, blas, chunk_cost.device_bytes,
      chunk_cost.host_bytes, chunk_cost.matrix_work, ComplexProductsHostOutput::CHUNKED);
  require(metrics.snapshot().live_bytes-before_chunk_owner.live_bytes == cost.device_bytes,
      "Complex products bounded output changed DEVICE bytes");
  const auto bounded_fingerprint = bounded->storageFingerprint();
  const auto bounded_warm = metrics.snapshot();
  for (bool conjugate : {false, true})
  {
    const auto before_action = metrics.snapshot();
    bounded->evaluate(positions, masses, kappa, channels, conjugate);
    fails([&] { (void)bounded->resultView(); });
    const auto receipt = bounded->completedResult();
    const auto no_transfer = metrics.snapshot();
    fails([&] { bounded->downloadCompletedChunk(Owner::CompletedResult{}, 0, 0, 1); });
    auto invalid_receipt = receipt; invalid_receipt.owner = owner.get();
    fails([&] { bounded->downloadCompletedChunk(invalid_receipt, 0, 0, 1); });
    invalid_receipt = receipt; invalid_receipt.image.reset();
    fails([&] { bounded->downloadCompletedChunk(invalid_receipt, 0, 0, 1); });
    invalid_receipt = receipt; ++invalid_receipt.generation;
    fails([&] { bounded->downloadCompletedChunk(invalid_receipt, 0, 0, 1); });
    fails([&] { bounded->downloadCompletedChunk(receipt, c, 0, 1); });
    fails([&] { bounded->downloadCompletedChunk(receipt, 0, p, 1); });
    fails([&] { bounded->downloadCompletedChunk(receipt, 0, 0, chunk+1); });
    fails([&] { bounded->downloadCompletedChunk(receipt, 0, 0, 0); });
    require(metrics.snapshot().downloaded_bytes == no_transfer.downloaded_bytes,
        "Complex products invalid bounded receipt transferred data");
    for (std::size_t channel = 0; channel < c; ++channel)
      for (std::size_t offset = 0; offset < p; offset += chunk)
      {
        const auto count = std::min(chunk, p-offset);
        const auto* output = bounded->downloadCompletedChunk(receipt, channel, offset, count);
        const auto& oracle = conjugate ? expected_hermitian : expected;
        for (std::size_t i = 0; i < count; ++i)
          require(std::abs(output[i]-oracle[channel*p+offset+i]) <= 3e-9 * (1+std::abs(oracle[channel*p+offset+i])),
              "Complex products bounded output differs from retained/CPU result");
      }
    require(metrics.snapshot().downloaded_bytes-before_action.downloaded_bytes ==
        baseline_download + 16*c*p + sizeof(DerivativeFiniteStatus)*cost.statuses,
        "Complex products bounded output changed total completed-channel D2H traffic");
    bounded->evaluate(positions, masses, kappa, channels, conjugate);
    fails([&] { bounded->downloadCompletedChunk(receipt, 0, 0, 1); });
    const auto valid = bounded->completedResult();
    const auto saved = coefficients[2].back();
    coefficients[2].back() = {std::numeric_limits<double>::quiet_NaN(), 0};
    fails([&] { bounded->evaluate(positions, masses, kappa, channels); });
    fails([&] { bounded->downloadCompletedChunk(valid, 0, 0, 1); });
    coefficients[2].back() = saved;
  }
  require(metrics.snapshot().allocation_calls == bounded_warm.allocation_calls &&
      metrics.snapshot().free_calls == bounded_warm.free_calls && bounded->storageFingerprint() == bounded_fingerprint,
      "Complex products bounded warm allocation/address drift");
  std::vector<C> direction(p), expected_jvp(b);
  const auto dot_warm = metrics.snapshot();
  for (std::size_t i = 0; i < p; ++i) direction[i] = .005 * std::sin(double(i + 1));
  for (std::size_t requested : {std::size_t{1}, std::size_t{2}})
  {
    const auto before_subset = metrics.snapshot();
    bounded->evaluate(positions, masses, kappa, channels);
    const auto receipt = bounded->completedResult();
    for (std::size_t channel = 0; channel < requested; ++channel)
      for (std::size_t offset = 0; offset < p; offset += chunk)
      {
        const auto count = std::min(chunk, p-offset);
        const auto* output = bounded->downloadCompletedChunk(receipt, channel, offset, count);
        for (std::size_t i = 0; i < count; ++i)
          require(std::abs(output[i]-expected[channel*p+offset+i]) <= 3e-9*(1+std::abs(expected[channel*p+offset+i])),
              "Complex products requested subset differs from CPU");
      }
    require(metrics.snapshot().downloaded_bytes-before_subset.downloaded_bytes ==
        baseline_download + 16*requested*p + sizeof(DerivativeFiniteStatus)*cost.statuses,
        "Complex products subset transferred an unrequested HOST channel");
  }
  bounded->evaluate(positions, masses, kappa, channels);
  const auto before_jvp = bounded->completedResult();
  bounded->evaluateScoreJVP(positions, direction.data(), direction.size());
  fails([&] { bounded->downloadCompletedChunk(before_jvp, 0, 0, 1); });
  const auto bounded_jvp = bounded->scoreJvpView();
  for (std::size_t attempt = 0; attempt < 3; ++attempt)
  {
    if (attempt == 1) direction.back() += .017;
    if (attempt == 2) std::fill(direction.begin(), direction.end(), C{});
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      score_work->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * sample, 4));
      const auto row = cpu_score.evaluate(*score_work);
      expected_jvp[sample] = C{};
      for (std::size_t i = 0; i < p; ++i) expected_jvp[sample] += row.parameter_score[i] * direction[i].real();
    }
    const auto dot_before = metrics.snapshot();
    owner->evaluateScoreJVP(positions, direction.data(), direction.size());
    const auto result = owner->scoreJvpView();
    owner->validate(result);
    fails([&] { owner->validate(hermitian); });
    require(result.samples == b, "Complex JVP sample extent differs");
    for (std::size_t sample = 0; sample < b; ++sample)
      require(std::abs(result.data[sample] - expected_jvp[sample]) <= 3e-9 * (1 + std::abs(expected_jvp[sample])),
          "Complex full-score/device-dot JVP differs from CPU score dot");
    if (!attempt)
      for (std::size_t sample = 0; sample < b; ++sample)
        require(std::abs(result.data[sample]-bounded_jvp.data[sample]) <= 3e-9 * (1+std::abs(result.data[sample])),
            "Complex output-storage policy changed score JVP");
    require(metrics.snapshot().uploaded_bytes - dot_before.uploaded_bytes == score_uploads + 8 * p,
        "Complex JVP did not upload exactly one realP direction per request");
    require(metrics.snapshot().downloaded_bytes - dot_before.downloaded_bytes ==
        score_downloads + b * (16 + 2 * sizeof(DerivativeFiniteStatus)), "Complex JVP transferred canonicalP instead of scalar/status results");
  }
  const auto dot_receipt = owner->scoreJvpView();
  auto foreign_dot = dot_receipt;
  foreign_dot.owner = nullptr;
  fails([&] { owner->validate(foreign_dot); });
  std::vector<double> tail_positions(positions.begin(), positions.begin() + 12);
  owner->evaluateScoreJVP(tail_positions, direction.data(), direction.size());
  require(owner->scoreJvpView().samples == 1, "Complex JVP bounded sample tail differs");
  fails([&] { owner->validate(dot_receipt); });
  const auto bad_direction = metrics.snapshot();
  direction.back() = C{0, .01};
  fails([&] { owner->evaluateScoreJVP(positions, direction.data(), direction.size()); });
  fails([&] { (void)owner->scoreJvpView(); });
  direction.back() = C{std::numeric_limits<double>::quiet_NaN(), 0};
  fails([&] { owner->evaluateScoreJVP(positions, direction.data(), direction.size()); });
  require(metrics.snapshot().uploaded_bytes == bad_direction.uploaded_bytes &&
      metrics.snapshot().downloaded_bytes == bad_direction.downloaded_bytes, "Complex JVP invalid final direction launched");
  direction.back() = C{};
  require(owner->storageFingerprint() == fingerprint && metrics.snapshot().allocation_calls == dot_warm.allocation_calls &&
      metrics.snapshot().free_calls == dot_warm.free_calls, "Complex JVP warm owner allocated or changed storage");
  // Required inverse admission cannot be bypassed by all-zero coefficients.
  auto nodal = parameters;
  for (auto part : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
  {
    const auto& interval = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, part);
    std::fill(nodal.begin() + interval.begin, nodal.begin() + interval.end, 0);
  }
  auto node_image = selected_two_cell
      ? PreparedDeviceModel::preparePairedTwoCellProfile(plan, nodal, model.cfg.nuclei.x)
      : PreparedDeviceModel::preparePairedComplexValue(plan, nodal, model.cfg.nuclei.x);
  auto node = Owner::prepare(node_image, targets, b, c, chunk, queue, blas, cost.device_bytes, cost.host_bytes, cost.matrix_work);
  for (auto& row : coefficients) std::fill(row.begin(), row.end(), C{});
  fails([&] { node->evaluate(positions, masses, kappa, channels); });
  fails([&] { (void)node->resultView(); });
  fails([&] { node->evaluateScoreJVP(positions, direction.data(), direction.size()); });
  fails([&] { (void)node->scoreJvpView(); });

  // Complete regular-node nonlocal ratios include learned phase, outer twist and a fixed companion.
  {
    constexpr std::size_t q = 9;
    const auto nc = estimateComplexProductsCost(plan, targets, 2 * b + q, 3, chunk, ComplexProductsHostOutput::CHUNKED);
    const auto before_owner = metrics.snapshot();
    auto nonlocal = Owner::prepare(image, targets, 2 * b + q, 3, chunk, queue, blas,
        nc.device_bytes, nc.host_bytes, nc.matrix_work, ComplexProductsHostOutput::CHUNKED);
    require(metrics.snapshot().live_bytes == before_owner.live_bytes + nc.device_bytes, "Nonlocal owner cost mismatch");
    const auto nf = nonlocal->storageFingerprint();
    std::vector<std::size_t> refs{1, 0, 1, 0, 0, 1, 0, 1, 1};
    std::vector<C> contractions{{.7, -.2}, {-.4, .3}}, weights(q), reference(p), conjugate_reference(p), gpu(p);
    std::vector<double> virtuals(q * 12), bare(q), phase(q), companion(q);
    for (std::size_t point = 0; point < q; ++point)
    {
      std::copy_n(positions.data() + 12 * refs[point], 12, virtuals.data() + 12 * point);
      const auto electron = point % 4;
      for (std::size_t d = 0; d < 3; ++d)
      {
        const double delta = .003 * (point + 1) * (d == 1 ? -1 : 1);
        virtuals[12 * point + 3 * electron + d] += delta;
        phase[point] += kappa[d] * delta;
      }
      bare[point] = (point % 2 ? -.07 : .09) * (point + 1);
      companion[point] = std::exp(.23 * .003 * (point + 1));
    }
    pf::DirectComplexValueExecutor cpu_value(model, plan);
    auto value_work = cpu_value.makeWorkspace();
    const auto amplitude = [&](const double* coordinate) {
      value_work->setPositions(pf::GeometryPositionView::interleaved(coordinate, 4));
      return cpu_value.evaluate(*value_work).amplitude;
    };

    const auto physical_weights = [&] {
      for (std::size_t point = 0; point < q; ++point)
      {
        const auto base = amplitude(positions.data() + 12 * refs[point]);
        const auto moved = amplitude(virtuals.data() + 12 * point);
        weights[point] = bare[point] * companion[point] * std::polar(1., phase[point]) *
            moved.phase * std::conj(base.phase) * std::exp(moved.log_abs - base.log_abs);
      }
    };
    physical_weights();
    const auto replacement = [&](std::size_t point) {
      const auto electron = point % 4;
      return ComplexVirtualReplacement{electron, {virtuals[12 * point + 3 * electron],
          virtuals[12 * point + 3 * electron + 1], virtuals[12 * point + 3 * electron + 2]}};
    };
    std::vector<std::vector<C>> base_scores(b);
    std::vector<C> tangent(p);
    for (std::size_t i = 0; i < p; ++i) tangent[i] = .01 * (int(i % 13) - 6) / std::sqrt(double(p));
    std::array<C, b> cpu_nonlocal_jvp{};
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      score_work->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * sample, 4));
      const auto score = cpu_score.evaluate(*score_work).parameter_score;
      base_scores[sample].assign(score, score + p);
    }
    for (std::size_t point = 0; point < q; ++point)
    {
      score_work->setPositions(pf::GeometryPositionView::interleaved(virtuals.data() + 12 * point, 4));
      const auto score = cpu_score.evaluate(*score_work).parameter_score;
      for (std::size_t i = 0; i < p; ++i)
      {
        const auto physical = weights[point] * (score[i] - base_scores[refs[point]][i]);
        reference[i] += contractions[refs[point]] * physical;
        conjugate_reference[i] += contractions[refs[point]] * std::conj(physical);
        cpu_nonlocal_jvp[refs[point]] += physical * tangent[i].real();
      }
    }
    // Inline capture uses accepted B capacity even when the native grid has Q>B.
    {
      const auto sc = estimateComplexProductsCost(plan, targets, b, 3, chunk, ComplexProductsHostOutput::CHUNKED);
      auto streamed = Owner::prepare(image, targets, b, 3, chunk, queue, blas,
          sc.device_bytes, sc.host_bytes, sc.matrix_work, ComplexProductsHostOutput::CHUNKED);
      const auto fingerprint = streamed->storageFingerprint();
      const auto warm = metrics.snapshot();
      std::array<C, b> sums{};
      for (std::size_t point = 0; point < q; ++point) sums[refs[point]] += weights[point];
      for (bool adjoint : {false, true})
      {
        const auto replay_before = metrics.snapshot();
        nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions, adjoint);
        const auto replay_download = metrics.snapshot().downloaded_bytes - replay_before.downloaded_bytes;
        const auto stream_before = metrics.snapshot();
        streamed->beginStreamingNonlocalVJP(q + b, adjoint);
        fails([&] { (void)streamed->completedResult(); });
        fails([&] { streamed->beginStreamingNonlocalVJP(q + b, adjoint); });
        for (std::size_t point = 0; point < q; ++point)
        {
          const auto move = replacement(point);
          streamed->accumulateStreamingNonlocalScore(positions, refs[point], &move,
              contractions[refs[point]], weights[point]);
        }
        for (std::size_t sample = 0; sample < b; ++sample)
          streamed->accumulateStreamingNonlocalScore(positions, sample, nullptr, contractions[sample], -sums[sample]);
        streamed->finishStreamingNonlocalVJP();
        require(metrics.snapshot().downloaded_bytes - stream_before.downloaded_bytes == replay_download,
            "Streaming response introduced parameter transfers during native quadrature");
        const auto receipt = streamed->completedResult();
        for (std::size_t offset = 0; offset < p; offset += chunk)
        {
          const auto count = std::min(chunk, p - offset);
          const auto* values = streamed->downloadCompletedChunk(receipt, 0, offset, count);
          for (std::size_t i = 0; i < count; ++i)
          {
            const auto expected = (adjoint ? conjugate_reference : reference)[offset + i];
            require(std::abs(values[i] - expected) <= 3e-9 * (1 + std::abs(expected)),
                "Streaming complete physical nonlocal response differs from CPU allP oracle");
          }
          for (std::size_t channel = 1; channel < 3; ++channel)
          {
            const auto* unused = streamed->downloadCompletedChunk(receipt, channel, offset, count);
            for (std::size_t i = 0; i < count; ++i)
              require(unused[i] == C{}, "Streaming C1 response changed an unused channel");
          }
        }
      }
      require(streamed->storageFingerprint() == fingerprint && metrics.snapshot().allocation_calls == warm.allocation_calls &&
          metrics.snapshot().free_calls == warm.free_calls, "Streaming response grew prepared storage");
      streamed->beginStreamingNonlocalVJP(q + b);
      fails([&] { streamed->finishStreamingNonlocalVJP(); });
      fails([&] { (void)streamed->completedResult(); });
      streamed->beginStreamingNonlocalVJP(1);
      fails([&] { streamed->evaluateScoreJVP(positions, tangent.data(), p); });
      streamed->abortStreamingNonlocalVJP();
      fails([&] { (void)streamed->completedResult(); });
      streamed->beginStreamingNonlocalVJP(1);
      streamed->accumulateStreamingNonlocalScore(positions, 0, nullptr, contractions[0], sums[0]);
      fails([&] { streamed->accumulateStreamingNonlocalScore(positions, 0, nullptr, contractions[0], sums[0]); });
      fails([&] { (void)streamed->completedResult(); });
      streamed->beginStreamingNonlocalVJP(1);
      fails([&] { streamed->accumulateStreamingNonlocalScore(positions, b, nullptr, contractions[0], sums[0]); });
      fails([&] { streamed->finishStreamingNonlocalVJP(); });
      streamed->beginStreamingNonlocalVJP(1);
      const ComplexVirtualReplacement node_move{0, {positions[3], positions[4], positions[5]}};
      fails([&] { streamed->accumulateStreamingNonlocalScore(positions, 0, &node_move, C{}, C{}); });
      fails([&] { (void)streamed->completedResult(); });
      streamed->beginStreamingNonlocalVJP(1);
      fails([&] { streamed->accumulateStreamingNonlocalScore(positions, 0, nullptr,
          {std::numeric_limits<double>::quiet_NaN(), 0}, C{}); });
      fails([&] { (void)streamed->completedResult(); });
      std::cout << "PASS complex streaming nonlocal accepted_B_capacity=1 Q_gt_B=1 both_adjoints=1 allP=1"
                << " incomplete_rejected=1 overrun_poisoned=1 interleave_rejected=1 zero_weight_node_rejected=1"
                << " no_parameter_transfers_per_point=1 warm_allocations=0\n";
    }
    const auto nw = metrics.snapshot();
    for (bool adjoint : {false, true})
    {
      nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions, adjoint);
      const auto receipt = nonlocal->completedResult();
      for (std::size_t offset = 0; offset < p; offset += chunk)
      {
        const auto count = std::min(chunk, p - offset);
        const auto* values = nonlocal->downloadCompletedChunk(receipt, 0, offset, count);
        for (std::size_t i = 0; i < count; ++i)
        {
          const auto expected = adjoint ? conjugate_reference[offset + i] : reference[offset + i];
          require(std::abs(values[i] - expected) <= 3e-9 * (1 + std::abs(expected)), "Nonlocal complete CPU allP mismatch");
          if (!adjoint) gpu[offset + i] = values[i];
        }
        for (std::size_t channel = 1; channel < 3; ++channel)
        {
          const auto* unused = nonlocal->downloadCompletedChunk(receipt, channel, offset, count);
          for (std::size_t i = 0; i < count; ++i)
            require(unused[i] == C{}, "Nonlocal C1 left a nonzero unused C3 channel");
        }
      }
    }
    C dot{};
    for (std::size_t i = 0; i < p; ++i)
      dot += gpu[i] * tangent[i];
    auto joined = positions;
    joined.insert(joined.end(), virtuals.begin(), virtuals.end());
    const auto revoked = nonlocal->completedResult();
    const auto score_dot_before = metrics.snapshot();
    nonlocal->evaluateScoreJVP(joined, tangent.data(), p);
    const auto score_dot_after = metrics.snapshot();
    fails([&] { (void)nonlocal->downloadCompletedChunk(revoked, 0, 0, 1); });
    const auto jvp = nonlocal->scoreJvpView();
    C dual{};
    for (std::size_t point = 0; point < q; ++point)
      dual += contractions[refs[point]] * weights[point] * (jvp.data[b + point] - jvp.data[refs[point]]);
    require(std::abs(dot - dual) <= 3e-9 * (1 + std::abs(dot)), "Nonlocal real-direction duality mismatch");
    double maximum_difference = 0;
    std::size_t overflow_point = 0;
    for (std::size_t point = 0; point < q; ++point)
    {
      const auto difference = jvp.data[b + point] - jvp.data[refs[point]];
      const auto largest = std::max(std::abs(difference.real()), std::abs(difference.imag()));
      if (largest > maximum_difference) { maximum_difference = largest; overflow_point = point; }
    }
    const auto nonlocal_dot_before = metrics.snapshot();
    nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p);
    const auto response = nonlocal->nonlocalJvpView();
    nonlocal->validate(response);
    require(response.samples == b, "Nonlocal JVP per-base response extent mismatch");
    fails([&] { nonlocal->validate(jvp); });
    fails([&] { (void)nonlocal->scoreJvpView(); });
    auto foreign_response = response;
    foreign_response.owner = nullptr;
    fails([&] { nonlocal->validate(foreign_response); });
    require(metrics.snapshot().uploaded_bytes - nonlocal_dot_before.uploaded_bytes ==
        score_dot_after.uploaded_bytes - score_dot_before.uploaded_bytes &&
        metrics.snapshot().downloaded_bytes - nonlocal_dot_before.downloaded_bytes ==
        score_dot_after.downloaded_bytes - score_dot_before.downloaded_bytes, "Nonlocal JVP added parameter transfers");
    C contracted{};
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      require(std::abs(response.data[sample] - cpu_nonlocal_jvp[sample]) <=
          3e-9 * (1 + std::abs(cpu_nonlocal_jvp[sample])), "Nonlocal JVP per-base CPU score mismatch");
      contracted += contractions[sample] * response.data[sample];
    }
    require(std::abs(contracted - dot) <= 3e-9 * (1 + std::abs(dot)), "Nonlocal JVP/VJP duality mismatch");
    C sides[2]{};
    std::array<std::array<C, b>, 2> base_sides{};
    constexpr double step = 1e-4;
    for (std::size_t side = 0; side < 2; ++side)
    {
      auto shifted = parameters;
      for (std::size_t i = 0; i < p; ++i) shifted[i] += (side ? -step : step) * tangent[i].real();
      model.p.set_flat_values(shifted);
      physical_weights();
      for (std::size_t point = 0; point < q; ++point)
      {
        sides[side] += contractions[refs[point]] * weights[point];
        base_sides[side][refs[point]] += weights[point];
      }
    }
    model.p.set_flat_values(parameters);
    physical_weights();
    require(std::abs((sides[0] - sides[1]) / (2 * step) - dot) <= 2e-8 * (1 + std::abs(dot)),
        "Nonlocal complete physical-ratio directional FD mismatch");
    for (std::size_t sample = 0; sample < b; ++sample)
      require(std::abs((base_sides[0][sample] - base_sides[1][sample]) / (2 * step) - response.data[sample]) <=
          2e-8 * (1 + std::abs(response.data[sample])), "Nonlocal JVP per-base physical FD mismatch");
    const auto untouched = metrics.snapshot();
    refs.back() = b;
    fails([&] { nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions); });
    fails([&] { nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p); });
    refs.back() = 1;
    weights.back() = {std::numeric_limits<double>::quiet_NaN(), 0};
    fails([&] { nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions); });
    fails([&] { nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p); });
    physical_weights();
    tangent.back() = C{0, .01};
    fails([&] { nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p); });
    tangent.back() = .01 * (int((p - 1) % 13) - 6) / std::sqrt(double(p));
    auto too_many = virtuals;
    too_many.insert(too_many.end(), virtuals.begin(), virtuals.begin() + 12);
    fails([&] { nonlocal->evaluateNonlocalVJP(positions, too_many, refs, weights, contractions); });
    require(metrics.snapshot().uploaded_bytes == untouched.uploaded_bytes &&
        metrics.snapshot().downloaded_bytes == untouched.downloaded_bytes, "Nonlocal late preflight launched");
    require(maximum_difference > 0, "Nonlocal overflow fixture needs a nonconstant directional response");
    const double scale = 8 / maximum_difference;
    for (auto& value : tangent) value *= scale;
    weights[overflow_point] = std::numeric_limits<double>::max();
    const auto before_overflow = metrics.snapshot();
    fails([&] { nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p); });
    require(metrics.snapshot().downloaded_bytes - before_overflow.downloaded_bytes ==
        score_dot_after.downloaded_bytes - score_dot_before.downloaded_bytes, "Nonlocal overflow did not follow complete inner JVP");
    fails([&] { (void)nonlocal->nonlocalJvpView(); });
    fails([&] { (void)nonlocal->scoreJvpView(); });
    fails([&] { nonlocal->evaluateLocalEnergyJVP(joined, b, refs, weights, masses, kappa,
        Owner::LocalTerms::COMPLETE, tangent.data(), p); });
    fails([&] { (void)nonlocal->localEnergyJvpView(); });
    fails([&] { (void)nonlocal->completedResult(); });
    for (std::size_t i = 0; i < p; ++i) tangent[i] = .01 * (int(i % 13) - 6) / std::sqrt(double(p));
    physical_weights();
    std::copy_n(virtuals.data() + 3, 3, virtuals.data()); // Same-spin node in the FIRST virtual, with exactly zero weight.
    weights[0] = C{};
    fails([&] { nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions); });
    fails([&] { nonlocal->evaluateLocalEnergyVJPReplacements(positions, replacement, refs, weights,
        contractions, masses, kappa, Owner::LocalTerms::NONLOCAL_ECP); });
    fails([&] { nonlocal->evaluateLocalEnergyJVPReplacements(positions, replacement, refs, weights,
        masses, kappa, Owner::LocalTerms::NONLOCAL_ECP, tangent.data(), p); });
    auto nodal_joined = positions;
    nodal_joined.insert(nodal_joined.end(), virtuals.begin(), virtuals.end());
    fails([&] { nonlocal->evaluateNonlocalJVP(nodal_joined, b, refs, weights, tangent.data(), p); });
    fails([&] { (void)nonlocal->nonlocalJvpView(); });
    fails([&] { (void)nonlocal->completedResult(); });
    std::copy_n(joined.data() + 12 * b, 12, virtuals.data());
    physical_weights();
    nonlocal->evaluateNonlocalJVP(joined, b, refs, weights, tangent.data(), p);
    nonlocal->validate(nonlocal->nonlocalJvpView());
    fails([&] { nonlocal->validate(response); });
    const auto valid_nonlocal = nonlocal->nonlocalJvpView();
    nonlocal->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions);
    fails([&] { nonlocal->validate(valid_nonlocal); });
    fails([&] { nonlocal->validate(jvp); });
    // Complete responses use the SAME owner and independent CPU kinetic derivatives.
    using Terms = Owner::LocalTerms;
    std::array<std::vector<C>, b> kinetic_rows;
    std::array<C, b> kinetic_dots{};
    for (std::size_t sample = 0; sample < b; ++sample)
    {
      kinetic_work->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * sample, 4));
      const auto value = cpu_kinetic.evaluate(*kinetic_work, masses.data() + 4 * sample, 4);
      kinetic_rows[sample].assign(value.parameter_response, value.parameter_response + p);
      for (std::size_t i = 0; i < p; ++i) kinetic_dots[sample] += kinetic_rows[sample][i] * tangent[i].real();
    }
    std::array<std::array<std::size_t, 2>, 3> local_traffic{};
    std::size_t term_index = 0;
    for (auto terms : {Terms::KINETIC, Terms::NONLOCAL_ECP, Terms::COMPLETE})
    {
      const auto before_jvp = metrics.snapshot();
      nonlocal->evaluateLocalEnergyJVP(joined, b, refs, weights, masses, kappa, terms, tangent.data(), p);
      local_traffic[term_index++] = {metrics.snapshot().uploaded_bytes - before_jvp.uploaded_bytes,
          metrics.snapshot().downloaded_bytes - before_jvp.downloaded_bytes};
      const auto local = nonlocal->localEnergyJvpView();
      nonlocal->validate(local);
      fails([&] { (void)nonlocal->scoreJvpView(); });
      fails([&] { (void)nonlocal->nonlocalJvpView(); });
      auto foreign = local; foreign.owner = nullptr;
      fails([&] { nonlocal->validate(foreign); });
      auto wrong_terms = local; wrong_terms.terms = static_cast<Terms>(0);
      fails([&] { nonlocal->validate(wrong_terms); });
      std::array<C, b> local_dots;
      for (std::size_t sample = 0; sample < b; ++sample)
      {
        local_dots[sample] = local.data[sample];
        const C expected = (terms == Terms::KINETIC ? C{} : cpu_nonlocal_jvp[sample]) +
            (terms == Terms::NONLOCAL_ECP ? C{} : kinetic_dots[sample]);
        require(std::abs(local.data[sample] - expected) <= 3e-9 * (1 + std::abs(expected)),
            "Complete local-energy per-base CPU JVP mismatch");
      }
      const auto before_streamed = metrics.snapshot();
      nonlocal->evaluateLocalEnergyJVPReplacements(positions, replacement, refs, weights, masses, kappa,
          terms, tangent.data(), p);
      const auto streamed = nonlocal->localEnergyJvpView();
      fails([&] { nonlocal->validate(local); });
      require(metrics.snapshot().uploaded_bytes - before_streamed.uploaded_bytes == local_traffic[term_index - 1][0] &&
          metrics.snapshot().downloaded_bytes - before_streamed.downloaded_bytes == local_traffic[term_index - 1][1],
          "Replacement JVP changed direction ingress or scalar transfers");
      for (std::size_t sample = 0; sample < b; ++sample)
        require(streamed.data[sample] == local_dots[sample], "Packed/replacement JVP ordering changed");
      for (bool adjoint : {false, true})
      {
        nonlocal->evaluateLocalEnergyVJP(positions, virtuals, refs, weights, contractions, masses, kappa, terms, adjoint);
        fails([&] { nonlocal->validate(local); });
        const auto packed = nonlocal->completedResult();
        for (std::size_t offset = 0; offset < p; offset += chunk)
        {
          const auto count = std::min(chunk, p - offset);
          const auto* values = nonlocal->downloadCompletedChunk(packed, 0, offset, count);
          std::copy_n(values, count, gpu.data() + offset);
        }
        nonlocal->evaluateLocalEnergyVJPReplacements(positions, replacement, refs, weights, contractions,
            masses, kappa, terms, adjoint);
        fails([&] { nonlocal->downloadCompletedChunk(packed, 0, 0, 1); });
        const auto completed = nonlocal->completedResult();
        C contracted{};
        for (std::size_t offset = 0; offset < p; offset += chunk)
        {
          const auto count = std::min(chunk, p - offset);
          const auto* result = nonlocal->downloadCompletedChunk(completed, 0, offset, count);
          for (std::size_t i = 0; i < count; ++i)
          {
            require(result[i] == gpu[offset + i], "Packed/replacement VJP changed arithmetic order");
            C expected = terms == Terms::KINETIC ? C{} : (adjoint ? conjugate_reference : reference)[offset + i];
            for (std::size_t sample = 0; terms != Terms::NONLOCAL_ECP && sample < b; ++sample)
              expected += contractions[sample] * (adjoint ? std::conj(kinetic_rows[sample][offset + i]) :
                  kinetic_rows[sample][offset + i]);
            require(std::abs(result[i] - expected) <= 3e-9 * (1 + std::abs(expected)),
                "Complete local-energy CPU allP adjoint mismatch");
            contracted += (adjoint ? std::conj(result[i]) : result[i]) * tangent[offset + i].real();
          }
        }
        C dual{};
        for (std::size_t sample = 0; sample < b; ++sample)
          dual += (adjoint ? std::conj(contractions[sample]) : contractions[sample]) * local_dots[sample];
        require(std::abs(contracted - dual) <= 3e-9 * (1 + std::abs(dual)), "Complete local-energy duality mismatch");
      }
    }
    require(local_traffic[2][0] == local_traffic[0][0] + local_traffic[1][0] - 8 * p &&
        local_traffic[2][1] == local_traffic[0][1] + local_traffic[1][1],
        "Complete JVP must reuse one realP direction ingress and only ordinary scalar/status egress");
    std::array<std::array<C, b>, 2> complete_sides{};
    for (std::size_t side = 0; side < 2; ++side)
    {
      auto shifted = parameters;
      for (std::size_t i = 0; i < p; ++i) shifted[i] += (side ? -step : step) * tangent[i].real();
      model.p.set_flat_values(shifted);
      physical_weights();
      for (std::size_t point = 0; point < q; ++point) complete_sides[side][refs[point]] += weights[point];
      for (std::size_t sample = 0; sample < b; ++sample)
      {
        kinetic_work->setPositions(pf::GeometryPositionView::interleaved(positions.data() + 12 * sample, 4));
        complete_sides[side][sample] += cpu_kinetic.evaluate(*kinetic_work, masses.data() + 4 * sample, 4).kinetic;
      }
    }
    model.p.set_flat_values(parameters);
    physical_weights();
    nonlocal->evaluateLocalEnergyJVP(joined, b, refs, weights, masses, kappa, Terms::COMPLETE, tangent.data(), p);
    const auto complete_jvp = nonlocal->localEnergyJvpView();
    for (std::size_t sample = 0; sample < b; ++sample)
      require(std::abs((complete_sides[0][sample] - complete_sides[1][sample]) / (2 * step) - complete_jvp.data[sample]) <=
          2e-8 * (1 + std::abs(complete_jvp.data[sample])), "Complete kinetic/ECP physical directional FD mismatch");
    auto invalid_masses = masses; invalid_masses.back() = 0;
    const auto before_invalid = metrics.snapshot();
    auto extra_positions = joined;
    extra_positions.insert(extra_positions.end(), virtuals.begin(), virtuals.begin() + 12);
    auto extra_refs = refs; extra_refs.push_back(0);
    auto extra_weights = weights; extra_weights.push_back({});
    fails([&] { nonlocal->evaluateLocalEnergyJVP(extra_positions, b, extra_refs, extra_weights, masses, kappa,
        Terms::COMPLETE, tangent.data(), p); }); // 2B+Q=14 exceeds prepared13, despite B+Q fitting.
    fails([&] { nonlocal->evaluateLocalEnergyJVP(joined, b, refs, weights, invalid_masses, kappa,
        Terms::COMPLETE, tangent.data(), p); });
    fails([&] { (void)nonlocal->localEnergyJvpView(); });
    fails([&] { (void)nonlocal->completedResult(); });
    require(metrics.snapshot().uploaded_bytes == before_invalid.uploaded_bytes, "Invalid complete request launched");
    nonlocal->evaluateLocalEnergyJVP(joined, b, refs, weights, masses, kappa, Terms::COMPLETE, tangent.data(), p);
    nonlocal->validate(nonlocal->localEnergyJvpView());
    auto score_coefficients = contractions;
    std::vector<Owner::Channel> complete_channels{{Owner::Role::SCORE, score_coefficients.data(), b},
        {Owner::Role::KINETIC, contractions.data(), b}, {Owner::Role::SCORE, score_coefficients.data(), b}};
    nonlocal->evaluateLocalEnergyVJP(positions, virtuals, refs, weights, contractions, masses, kappa,
        Terms::COMPLETE, false, &complete_channels, 1);
    const auto complete_receipt = nonlocal->completedResult();
    nonlocal->evaluateLocalEnergyVJPReplacements(positions, replacement, refs, weights, contractions,
        masses, kappa, Terms::COMPLETE, false, &complete_channels, 1);
    fails([&] { nonlocal->downloadCompletedChunk(complete_receipt, 1, 0, 1); });
    const auto before_bad_replacement = metrics.snapshot();
    const auto invalid_replacement = [&](std::size_t point) {
      auto move = replacement(point); if (point == q - 1) move.electron = 4; return move;
    };
    fails([&] { nonlocal->evaluateLocalEnergyVJPReplacements(positions, invalid_replacement, refs, weights,
        contractions, masses, kappa, Terms::COMPLETE); });
    fails([&] { (void)nonlocal->completedResult(); });
    fails([&] { nonlocal->evaluateLocalEnergyJVPReplacements(positions, invalid_replacement, refs, weights,
        masses, kappa, Terms::COMPLETE, tangent.data(), p); });
    fails([&] { (void)nonlocal->localEnergyJvpView(); });
    require(metrics.snapshot().uploaded_bytes == before_bad_replacement.uploaded_bytes &&
        metrics.snapshot().downloaded_bytes == before_bad_replacement.downloaded_bytes,
        "Invalid last replacement launched numerical work");
    const auto nonfinite_replacement = [&](std::size_t point) {
      auto move = replacement(point);
      if (point == q - 1) move.xyz[2] = std::numeric_limits<double>::quiet_NaN();
      return move;
    };
    fails([&] { nonlocal->evaluateLocalEnergyJVPReplacements(positions, nonfinite_replacement, refs, weights,
        masses, kappa, Terms::COMPLETE, tangent.data(), p); });
    require(metrics.snapshot().uploaded_bytes == before_bad_replacement.uploaded_bytes,
        "Nonfinite replacement uploaded direction before complete coordinate preflight");
    const auto before_bad_channel = metrics.snapshot();
    score_coefficients.back() = {std::numeric_limits<double>::infinity(), 0};
    fails([&] { nonlocal->evaluateLocalEnergyVJP(positions, virtuals, refs, weights, contractions, masses, kappa,
        Terms::COMPLETE, false, &complete_channels, 1); });
    fails([&] { (void)nonlocal->completedResult(); });
    require(metrics.snapshot().uploaded_bytes == before_bad_channel.uploaded_bytes &&
        metrics.snapshot().downloaded_bytes == before_bad_channel.downloaded_bytes, "Invalid EO/O coefficient launched");
    std::copy(contractions.begin(), contractions.end(), score_coefficients.begin());
    nonlocal->evaluateLocalEnergyVJP(positions, virtuals, refs, weights, contractions, masses, kappa,
        Terms::COMPLETE, false, &complete_channels, 1);
    fails([&] { nonlocal->downloadCompletedChunk(complete_receipt, 1, 0, 1); });
    std::cout << "PASS complex complete C3 energy_slot=1 finite_publication=1 stale_retry=1 same_storage=1\n";
    std::cout << "PASS complex replacement coordinates B2Q9 packed_exact=1 bothadjoints=1 JVP=1 preflight=1 no_new_owner=1\n";
    std::cout << "PASS complex complete response B=2 Q=9 capacity=13 K_ECP_BOTH=1 CPU_allP=1 physical_FD=1 typed=1 retry=1\n";
    require(nonlocal->storageFingerprint() == nf && metrics.snapshot().allocation_calls == nw.allocation_calls &&
        metrics.snapshot().free_calls == nw.free_calls, "Nonlocal warm request changed prepared ownership");
    std::cout << "PASS complex nonlocal C1 B=" << b << " Q=" << q << " allP=1 transpose=1 hermitian=1 physical_ratio_FD=1"
              << " real_direction_duality=1 zero_weight_node_rejected=1 retry=1 warm_allocations=0 storage_C3_unused_zero=1\n";
    std::cout << "PASS complex nonlocal JVP B=" << b << " Q=" << q
              << " per_base_CPU=1 physical_ratio_FD=1 VJP_duality=1 typed_receipts=1 overflow_revoked=1 same_transfers=1\n";

    // The numerical ECP/SCORE consumer must not own an unused kinetic tape.
    constexpr auto scope = ComplexProductsScope::SCORE_ECP_ONLY;
    constexpr auto output = ComplexProductsHostOutput::CHUNKED;
    const auto selective_cost = estimateComplexProductsCost(plan, targets, 2 * b + q, 3, chunk, output, scope);
    require(selective_cost.kinetic.device_bytes == 0 && selective_cost.kinetic.host_bytes == 0 &&
        selective_cost.kinetic.canonical_bytes == 0 && selective_cost.matrix_work == 0 &&
        selective_cost.score_dot_scalar_work > 0, "Selective owner omitted work/kinetic accounting is false");
    require(nc.device_bytes - selective_cost.device_bytes == nc.kinetic.device_bytes &&
        nc.host_bytes - selective_cost.host_bytes == nc.kinetic.host_bytes, "Selective cost saving is not exact");
    const auto prepare_selective = [&](std::size_t dev, std::size_t host) {
      return Owner::prepare(image, targets, 2 * b + q, 3, chunk, queue, blas, dev, host, 1, output, scope);
    };
    const auto selective_cold = metrics.snapshot();
    fails([&] { prepare_selective(selective_cost.device_bytes - 1, selective_cost.host_bytes); });
    fails([&] { prepare_selective(selective_cost.device_bytes, selective_cost.host_bytes - 1); });
    fails([&] { estimateComplexProductsCost(plan, 0, b, 3, chunk, output, scope); });
    fails([&] { estimateComplexProductsCost(plan, 5, b, 3, chunk, output, scope); });
    fails([&] { estimateComplexProductsCost(plan, targets, b, 3, chunk, output, static_cast<ComplexProductsScope>(9)); });
    require(metrics.snapshot().allocation_calls == selective_cold.allocation_calls,
        "Selective cap/option rejection allocated DEVICE storage");
    auto selective = prepare_selective(selective_cost.device_bytes, selective_cost.host_bytes);
    require(metrics.snapshot().live_bytes - selective_cold.live_bytes == selective_cost.device_bytes,
        "Selective numerical allocation still owns a hidden kinetic tape");
    const auto selective_fingerprint = selective->storageFingerprint();
    const auto selective_warm = metrics.snapshot();
    std::vector<Owner::Channel> score_channels(3, {Owner::Role::SCORE, contractions.data(), b});
    selective->evaluate(positions, masses, kappa, score_channels);
    const auto score_receipt = selective->completedResult();
    for (std::size_t offset = 0; offset < p; offset += chunk)
    {
      const auto count = std::min(chunk, p - offset);
      const auto* values = selective->downloadCompletedChunk(score_receipt, 0, offset, count);
      for (std::size_t i = 0; i < count; ++i)
      {
        C expected{};
        for (std::size_t sample = 0; sample < b; ++sample) expected += contractions[sample] * base_scores[sample][offset + i];
        require(std::abs(values[i] - expected) <= 3e-9 * (1 + std::abs(expected)), "Selective SCORE allP differs");
      }
    }
    for (bool adjoint : {false, true})
    {
      selective->evaluateLocalEnergyVJPReplacements(positions, replacement, refs, weights, contractions,
          masses, kappa, Terms::NONLOCAL_ECP, adjoint);
      const auto receipt = selective->completedResult();
      for (std::size_t offset = 0; offset < p; offset += chunk)
      {
        const auto count = std::min(chunk, p - offset);
        const auto* values = selective->downloadCompletedChunk(receipt, 0, offset, count);
        for (std::size_t i = 0; i < count; ++i)
          require(std::abs(values[i] - (adjoint ? conjugate_reference : reference)[offset + i]) <=
              3e-9 * (1 + std::abs(values[i])), "Selective ECP allP/physicalFD reference differs");
      }
    }
    selective->evaluateLocalEnergyJVPReplacements(positions, replacement, refs, weights, masses, kappa,
        Terms::NONLOCAL_ECP, tangent.data(), p);
    const auto selective_receipt = selective->localEnergyJvpView();
    for (std::size_t sample = 0; sample < b; ++sample)
      require(std::abs(selective_receipt.data[sample] - cpu_nonlocal_jvp[sample]) <=
          3e-9 * (1 + std::abs(cpu_nonlocal_jvp[sample])), "Selective per-base physicalFD/duality JVP differs");
    const auto before_refusal = metrics.snapshot();
    for (auto terms : {Terms::KINETIC, Terms::COMPLETE})
    {
      fails([&] { selective->evaluateLocalEnergyVJP(positions, virtuals, refs, weights, contractions, masses, kappa, terms); });
      fails([&] { selective->evaluateLocalEnergyJVP(joined, b, refs, weights, masses, kappa, terms, tangent.data(), p); });
    }
    fails([&] { selective->evaluate(positions, masses, kappa, complete_channels); });
    fails([&] { selective->validate(selective_receipt); });
    fails([&] { (void)selective->completedResult(); });
    require(metrics.snapshot().uploaded_bytes == before_refusal.uploaded_bytes &&
        metrics.snapshot().downloaded_bytes == before_refusal.downloaded_bytes, "Selective kinetic refusal launched work");
    selective->evaluateNonlocalVJP(positions, virtuals, refs, weights, contractions);
    require(selective->storageFingerprint() == selective_fingerprint &&
        metrics.snapshot().allocation_calls == selective_warm.allocation_calls &&
        metrics.snapshot().free_calls == selective_warm.free_calls, "Selective warm refusal/retry changed ownership");
    std::cout << "PASS complex selective SCORE_ECP_ONLY CPU_allP=1 bothadjoints=1 JVP=1 cap_preflight=1 kinetic_refused=1"
              << " warm_allocations=0 device_saved=" << nc.kinetic.device_bytes << " host_saved=" << nc.kinetic.host_bytes << '\n';
  }
  std::cout << "PASS complex products B=" << b << " channels=" << c << " P=" << p
            << " K=" << targets << " chunk=" << chunk << " max_cpu_allP=" << maximum
            << " device_bytes=" << cost.device_bytes << " host_bytes=" << cost.host_bytes
            << " request_matrix_work=" << cost.matrix_work << " final_canonical_D2H=" << 16 * c * p
            << " score_dot_scalar_work=" << cost.score_dot_scalar_work
            << " warm_allocations=0 whole_batch_preflight=1 ordinary_transpose=1 hermitian=1 full_score_device_dot=1\n";
}
#endif
