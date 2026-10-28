//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ECP_OBSERVATION_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ECP_OBSERVATION_VALIDATION_H
#include "PsiFormerDeviceECPObservation.h"

namespace qmcplusplus::psiformer::device
{
/// Synthetic component-only oracle; native physical ECP admission is tested by XML.
inline void validateECPObservationFinals(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image)
{
  std::size_t checks = 0;
  const auto require = [&](bool good) { ++checks; if (!good) throw std::runtime_error("ECP final observation check " + std::to_string(checks)); };
  const auto rejects = [&](const auto& action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
  };
  constexpr std::size_t samples = 3;
  const auto ne = model.ne, stride = 3 * ne, version = image->version();
  require(model.cfg.nconfig >= samples && model.ne <= 8 && plan.parameterCount() < 100000);
  std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + samples * stride);
  std::vector<double> masses(ne), kinetic(samples), energy{-2.1, -2.2, -1.9};
  for (std::size_t e = 0; e < ne; ++e) masses[e] = .8 + .1 * e;
  pf::DirectValueExecutor value(model, plan);
  pf::DirectSpatialExecutor spatial(model, value, plan);
  auto workspace = spatial.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
  for (std::size_t sample = 0; sample < samples; ++sample)
  {
    workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data() + sample * stride, ne));
    const auto result = spatial.evaluateFull(*workspace);
    for (std::size_t e = 0; e < ne; ++e) kinetic[sample] -= .5 * masses[e] * result.lap_ratio[e];
  }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  // No reverse retention: the previous spatialHeadTape dependency would reject this graph.
  PreparedDeviceInference producer(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
  rejects([&] { (void)producer.completedSpatialFinals(); });
  const auto source = std::make_shared<const std::size_t>(91);
  const auto cost = PreparedDeviceECPObservation::estimate(samples, ne);
  rejects([&] { PreparedDeviceECPObservation bad(image, samples, masses, source, "synthetic", cost.device_bytes - 1, cost.host_bytes); });
  auto observation = std::make_shared<PreparedDeviceECPObservation>(image, samples, masses,
      source, "synthetic", cost.device_bytes, cost.host_bytes);
  require(observation->deviceBytes() == cost.device_bytes);
  require(observation->hostBytes() == cost.host_bytes);
  rejects([&] { (void)observation->view(0, 1, version); });
  std::vector<double> tile(positions.begin(), positions.begin() + 2 * stride);
  producer.evaluate(tile);
  rejects([&] { (void)producer.spatialHeadTape(); });
  auto finals = producer.completedSpatialFinals();
  auto foreign = finals; foreign.image = nullptr;
  rejects([&] { observation->append(foreign, queue, tile, 2, 1); });
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "ECP test current device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "ECP test current device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  auto before = metrics.snapshot();
  observation->append(finals, queue, tile, 2, 1);
  auto after = metrics.snapshot();
  std::size_t captured_bytes = after.device_copied_bytes - before.device_copied_bytes;
  require(after.uploaded_bytes == before.uploaded_bytes && after.downloaded_bytes == before.downloaded_bytes);
  rejects([&] { observation->finishCapture(version); });
  rejects([&] { observation->append(finals, queue, tile, 2, 1); });
  tile.assign(positions.begin() + 2 * stride, positions.end());
  producer.evaluate(tile); finals = producer.completedSpatialFinals();
  before = metrics.snapshot(); observation->append(finals, queue, tile, 1, 2); after = metrics.snapshot();
  captured_bytes += after.device_copied_bytes - before.device_copied_bytes;
  require(captured_bytes == sizeof(double) * samples * (7 * ne + 1));
  rejects([&] { observation->finishCapture(version + 1); });
  observation->finishCapture(version);
  rejects([&] { (void)observation->view(0, 1, version); });
  // Overwrite every producer scratch before consuming the retained physical finals.
  for (auto& x : tile) x += .013;
  producer.evaluate(tile);
  auto wrong_positions = positions; wrong_positions[0] += .01;
  rejects([&] { observation->seal(queue, version, 19, "synthetic", wrong_positions, energy, kinetic); });
  rejects([&] { observation->seal(queue, version, 0, "synthetic", positions, energy, kinetic); });
  rejects([&] { observation->seal(queue, version, 19, "foreign", positions, energy, kinetic); });
  before = metrics.snapshot(); observation->seal(queue, version, 19, "synthetic", positions, energy, kinetic); after = metrics.snapshot();
  require(after.uploaded_bytes - before.uploaded_bytes == samples * sizeof(double));
  require(after.downloaded_bytes - before.downloaded_bytes == samples * (sizeof(double) + sizeof(std::uint64_t)));
  observation->validate(version, 19, "synthetic", 2);
  rejects([&] { observation->validate(version + 1, 19, "synthetic", 2); });
  rejects([&] { observation->validate(version, 20, "synthetic", 2); });
  rejects([&] { observation->validate(version, 19, "foreign", 2); });
  rejects([&] { observation->validate(version, 19, "synthetic", 1); });
  rejects([&] { observation->seal(queue, version, 19, "synthetic", positions, energy, kinetic); });
  for (std::size_t i = 0; i < samples; ++i)
    require(std::abs(observation->kinetic()[i] - kinetic[i]) < 1e-9 * (1 + std::abs(kinetic[i])));
  PreparedDeviceInference consumer(image, 2, queue, blas, SpatialJetMode::FULL_VGL);
  consumer.evaluateDeviceCoordinates(observation->view(0, 2, version));
  std::vector<double> phase(2), roots(spatialJetSpanElements(consumer.resultLayout())), ratios(2 * ne);
  consumer.downloadResults(phase, roots, ratios);
  tile.assign(positions.begin(), positions.begin() + 2 * stride); producer.evaluate(tile);
  std::vector<double> reference_phase(2), reference(roots.size()), reference_ratios(ratios.size());
  producer.downloadResults(reference_phase, reference, reference_ratios);
  for (std::size_t i = 0; i < roots.size(); ++i) require(std::abs(roots[i] - reference[i]) < 1e-9 * (1 + std::abs(reference[i])));
  const auto expired = observation->view(0, 1, version);
  observation.reset();
  rejects([&] { consumer.evaluateDeviceCoordinates(expired); });
  std::cout << "PASS ECP immutable observation component checks=" << checks
            << " DEVICE_bytes=" << cost.device_bytes << " HOST_bytes=" << cost.host_bytes
            << " captured_D2D_bytes=" << captured_bytes
            << "; synthetic grid, native ECP energy not qualified by this component test\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
