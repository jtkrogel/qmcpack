//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerDeviceLargeScoreTileValidation.h
 * @brief Large REAL score tile storage, all-P, tail and cold-policy oracles.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_LARGE_SCORE_TILE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_LARGE_SCORE_TILE_VALIDATION_H

#include "PsiFormerDeviceBatchBackend.h"
#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include <numeric>

namespace qmcplusplus::psiformer::device
{
/// One CPU all-P oracle at Ne64; one realistic Ne100 GPU serial/tile oracle.
inline void validateLargeRealScoreTiles(compute::Queue<PlatformKind::CUDA>& queue)
{
  const char* key = "PSIFORMER_GPU_LARGE_SCORE_TILE4";
  const char* saved = std::getenv(key);
  const std::optional<std::string> previous = saved ? std::optional<std::string>(saved) : std::nullopt;
  struct Restore
  {
    const char* key;
    const std::optional<std::string>& previous;
    ~Restore() { if (previous) setenv(key, previous->c_str(), 1); else unsetenv(key); }
  } restore{key, previous};
  unsetenv(key);
  if (configuredLargeScoreTile4())
    throw std::runtime_error("Large score default is not OFF");
  setenv(key, "invalid", 1);
  bool rejected = false;
  try { configuredLargeScoreTile4(); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Large score selector accepted non-0/1");

  double maximum_error = 0.;
  std::size_t comparisons = 0, strict_diagnostic_violations = 0;
  const auto compare = [&](const std::vector<double>& actual, const std::vector<double>& reference) {
    if (actual.size() != reference.size())
      throw std::runtime_error("Large score response extent mismatch");
    for (std::size_t p = 0; p < actual.size(); ++p)
    {
      const double error = std::abs(actual[p] - reference[p]);
      maximum_error = std::max(maximum_error, error);
      ++comparisons;
      strict_diagnostic_violations += error > 2e-12 * (1 + std::abs(reference[p]));
      if (!device_math::isFiniteBinary64(actual[p]) || !device_math::isFiniteBinary64(reference[p]) ||
          error > 2e-11 * (1 + std::abs(reference[p])))
        throw std::runtime_error("Large score exceeded unchanged core all-P contract");
    }
  };
  for (const std::size_t ne : {std::size_t(64), std::size_t(100)})
  {
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{80., .3, .1}, {0., 80., .2}, {.1, 0., 80.}}};
    environment.periodic_axes = {true, true, true};
    const ModelShape shape{ne / 2, ne / 2, ne, ne == 64 ? 1UL : 16UL,
                           ne == 64 ? 64UL : 256UL, 4, ne == 64 ? 1UL : 4UL};
    auto parameters = mixed_periodic_validation::parameters(environment, shape);
    std::vector<double> nuclei(3 * ne), base(3 * ne), positions;
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const auto grid = d == 0 ? e % 5 : d == 1 ? (e / 5) % 5 : e / 25;
        nuclei[3 * e + d] = 1. + 6. * grid;
        base[3 * e + d] = nuclei[3 * e + d] + .06 + .02 * d;
      }
    for (std::size_t b = 0; b < 4; ++b)
    {
      positions.insert(positions.end(), base.begin(), base.end());
      positions[b * 3 * ne + 3 * b] += .003 * b;
    }
    pf::PsiFormer model(std::move(parameters),
        pf::ConfigData(pf::Tensor({1, ne, 3}, base), pf::Tensor({ne, 3}, nuclei),
                       pf::Tensor({ne}, std::vector<double>(ne, 1.)), ne / 2, ne / 2, shape.determinants),
        shape.determinants, shape.feature_dimension, shape.attention_heads, shape.attention_blocks);
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, shape, environment);
    auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), nuclei, 0);
    AcceleratorBlasHandle blas(queue);
    PreparedDeviceInference serial(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
    PreparedDeviceScoreReverse serial_reverse(image, 1, queue, blas);
    PreparedDeviceScoreJvp serial_jvp(image, 1, queue, blas);
    PreparedDeviceInference tile(image, 4, queue, blas, SpatialJetMode::VALUE_ONLY, true);
    PreparedDeviceScoreReverse tile_reverse(image, 4, queue, blas);
    PreparedDeviceScoreJvp tile_jvp(image, 4, queue, blas);
    for (const std::size_t b : {std::size_t(1), std::size_t(4)})
    {
      const auto cost = estimateScoreTileStorage(plan, b);
      const auto device = b == 1 ? serial.deviceWorkspaceBytes() + serial_reverse.deviceWorkspaceBytes() +
          serial_jvp.deviceWorkspaceBytes() : tile.deviceWorkspaceBytes() + tile_reverse.deviceWorkspaceBytes() +
          tile_jvp.deviceWorkspaceBytes();
      const auto host = b == 1 ? serial.hostWorkspaceBytes() + serial_reverse.hostWorkspaceBytes() +
          serial_jvp.hostWorkspaceBytes() : tile.hostWorkspaceBytes() + tile_reverse.hostWorkspaceBytes() +
          tile_jvp.hostWorkspaceBytes();
      if (device != cost.device_bytes || host != cost.host_numeric_bytes)
        throw std::runtime_error("Large score complete graph inventory differs from cold cost");
    }
    const auto np = plan.parameterCount();
    std::vector<double> response(np), one(np), expected(np), direction(np, .001);
    const std::vector<double> signed_weights{.25, -.375, 0., .125};
    if (ne == 64)
    {
      // Independent host all-P reference once, not a redundant shape cross-product.
      pf::DirectScoreExecutor cpu(model, plan);
      auto workspace = cpu.makeWorkspace();
      workspace->setPositions(pf::GeometryPositionView::interleaved(base.data(), ne));
      const auto reference = cpu.evaluate(*workspace);
      std::vector<double> cpu_score(reference.parameter_score.begin(), reference.parameter_score.end());
      serial.evaluate(base);
      serial_reverse.reverse(serial, {1.});
      serial_reverse.downloadResponse(one);
      compare(one, cpu_score);
    }
    // B4 -> B3 -> B1 exercises live extents without changing allocated capacity.
    for (const std::size_t b : {std::size_t(4), std::size_t(3), std::size_t(1)})
    {
      std::fill(expected.begin(), expected.end(), 0.);
      for (std::size_t i = 0; i < b; ++i)
      {
        serial.evaluate(std::vector<double>(positions.begin() + i * 3 * ne,
                                            positions.begin() + (i + 1) * 3 * ne));
        serial_reverse.reverse(serial, {signed_weights[i]});
        serial_reverse.downloadResponse(one);
        for (std::size_t p = 0; p < np; ++p) expected[p] += one[p];
      }
      const std::vector<double> live(positions.begin(), positions.begin() + b * 3 * ne);
      const std::vector<double> weights(signed_weights.begin(), signed_weights.begin() + b);
      tile.evaluate(live);
      tile_reverse.reverse(tile, weights);
      tile_reverse.downloadResponse(response);
      compare(response, expected);
      tile_jvp.evaluate(tile, direction);
      std::vector<double> products(b);
      tile_jvp.downloadProducts(products);
      const auto lhs = std::inner_product(weights.begin(), weights.end(), products.begin(), 0.);
      const auto rhs = std::inner_product(direction.begin(), direction.end(), response.begin(), 0.);
      compare({lhs}, {rhs});
      const auto before = deviceMemoryMetrics(0).snapshot();
      tile.evaluate(live); tile_reverse.reverse(tile, weights); tile_jvp.evaluatePrepared(tile);
      const auto after = deviceMemoryMetrics(0).snapshot();
      if (after.allocation_calls != before.allocation_calls || after.live_bytes != before.live_bytes)
        throw std::runtime_error("Large score warm tail allocated device storage");
    }
    auto bad = positions;
    bad.back() = std::numeric_limits<double>::quiet_NaN();
    bool failed = false, stale = false;
    try { tile.evaluate(bad); } catch (const std::runtime_error&) { failed = true; }
    try { tile_reverse.reverse(tile, signed_weights); } catch (const std::logic_error&) { stale = true; }
    if (!failed || !stale)
      throw std::runtime_error("Large score failed nonzero-walker tile exposed a tape");
    tile.evaluate(positions); tile_reverse.reverse(tile, signed_weights);
    auto changed = model.p.flat_values();
    changed.front() += .001;
    auto successor = image->makeUpdatedImage(changed, 1);
    tile.rebindModel(successor); tile_reverse.rebindModel(successor); tile_jvp.rebindModel(successor);
    tile.evaluate(base); tile_reverse.reverse(tile, {1.});
    std::cout << "PASS large REAL score graph Ne=" << ne << " P=" << np
              << " B1_bytes=" << estimateScoreTileStorage(plan, 1).device_bytes
              << " B4_bytes=" << estimateScoreTileStorage(plan, 4).device_bytes
              << " CPU_allP=" << (ne == 64 ? "once" : "not_repeated") << '\n';

    if (ne == 64)
    {
      setenv(key, "0", 1);
      PreparedDeviceBatchBackend off(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, {}, false);
      auto off_context = off.prepare(1);
      off.noteLargeScorePreparationContext(true);
      if (off.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4)->scoreCapacity() != 1)
        throw std::runtime_error("Large score default changed");
      setenv(key, "1", 1);
      PreparedDeviceBatchBackend on(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, {}, false);
      auto on_context = on.prepare(1);
      setenv(key, "0", 1); // Frozen opt-in must not reread this mutable value.
      on.noteLargeScorePreparationContext(false);
      if (on.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4)->scoreCapacity() != 1)
        throw std::runtime_error("Unknown/multiple driver crowds admitted large score");
      on.noteLargeScorePreparationContext(true);
      auto session = on.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4);
      if (session->scoreCapacity() != 4)
        throw std::runtime_error("Frozen large score opt-in failed finite admitted C1 preparation");
      session.reset();
      // Leave a baseline-fit but candidate-minus1 DEVICE domain. The optional
      // preflight rejects, then a canonical B1 owner must still construct.
      const auto& policy = configuredDeviceMemoryAdmission();
      if (!policy.limit_bytes)
        throw std::runtime_error("Large score qualification requires a finite aggregate DEVICE cap");
      auto& metrics = deviceMemoryMetrics(0);
      const auto prospective = estimateScoreTileStorage(plan, 4).device_bytes +
          PreparedDeviceScoreContraction::requiredBytes(np, 3) + image->deviceStorageBytes();
      const auto available = *policy.limit_bytes - policy.external_reserve_bytes - metrics.snapshot().live_bytes;
      if (available < prospective)
        throw std::runtime_error("Large score oracle cap cannot exercise candidate-minus1");
      {
        auto occupied = metrics.reserveAllocation(available - (prospective - 1));
        if (on.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4)->scoreCapacity() != 1)
          throw std::runtime_error("Large score candidate-minus1 did not preserve baseline");
      }
    }
  }
  std::cout << "PASS large REAL score tiles: core_contract=2e-11 max_abs_error=" << maximum_error
            << " allP_comparisons=" << comparisons << " strict2e-12_diagnostic_violations="
            << strict_diagnostic_violations << " warm_DEVICE_allocations=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
