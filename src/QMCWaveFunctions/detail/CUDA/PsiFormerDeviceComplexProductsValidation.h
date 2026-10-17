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
void validateComplexProducts()
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
  const ModelShape shape{3, 1, 2, 4, 16, 2, 2};
  auto configuration = configurations(environment);
  configuration.nup = 3;
  configuration.ndown = 1;
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
  const std::array<double, 3> kappa{.13, -.07, .04};
  pf::DirectComplexScoreExecutor cpu_score(model, plan);
  auto score_work = cpu_score.makeWorkspace();
  pf::DirectComplexKineticExecutor cpu_kinetic(model, plan, kappa);
  auto kinetic_work = cpu_kinetic.makeWorkspace(cpu_kinetic.requiredStorageBytes());
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, model.cfg.nuclei.x);
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
  auto node_image = PreparedDeviceModel::preparePairedComplexValue(plan, nodal, model.cfg.nuclei.x);
  auto node = Owner::prepare(node_image, targets, b, c, chunk, queue, blas, cost.device_bytes, cost.host_bytes, cost.matrix_work);
  for (auto& row : coefficients) std::fill(row.begin(), row.end(), C{});
  fails([&] { node->evaluate(positions, masses, kappa, channels); });
  fails([&] { (void)node->resultView(); });
  fails([&] { node->evaluateScoreJVP(positions, direction.data(), direction.size()); });
  fails([&] { (void)node->scoreJvpView(); });
  std::cout << "PASS complex products B=" << b << " channels=" << c << " P=" << p
            << " K=" << targets << " chunk=" << chunk << " max_cpu_allP=" << maximum
            << " device_bytes=" << cost.device_bytes << " host_bytes=" << cost.host_bytes
            << " request_matrix_work=" << cost.matrix_work << " final_canonical_D2H=" << 16 * c * p
            << " score_dot_scalar_work=" << cost.score_dot_scalar_work
            << " warm_allocations=0 whole_batch_preflight=1 ordinary_transpose=1 hermitian=1 full_score_device_dot=1\n";
}
#endif
