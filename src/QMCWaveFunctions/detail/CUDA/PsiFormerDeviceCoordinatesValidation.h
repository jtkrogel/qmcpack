//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_VALIDATION_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceScoreReverse.h"
#include "PsiFormerDeviceBatchBackend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace qmcplusplus::psiformer::device
{
/// Qualify one typed coordinate source without binding production sample owners.
template<class Model>
void validateDeviceCoordinates(Model& model, const std::shared_ptr<const PreparedDeviceModel>& image,
                               compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto ne = image->plan().modelShape().electrons();
  const auto stride = 3 * ne;
  std::vector<double> packed(3 * stride);
  for (std::size_t sample = 0; sample < 3; ++sample)
    std::copy_n(model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * stride, stride,
                packed.data() + sample * stride);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "coordinate validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "coordinate validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before_owner = metrics.snapshot();
  const auto exact_bytes = checkedDeviceBytes<double>(packed.size());
  // Budget/overflow rejection must happen before state or device allocation.
  for (const auto request : {std::array<std::size_t, 2>{3, exact_bytes - 1},
                             std::array<std::size_t, 2>{std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()}})
  {
    bool rejected = false;
    try { (void)image->prepareCoordinates(request[0], request[1]); }
    catch (const std::exception&) { rejected = true; }
    const auto after = metrics.snapshot();
    if (!rejected || after.allocation_calls != before_owner.allocation_calls || after.live_bytes != before_owner.live_bytes)
      throw std::runtime_error("Coordinate admission rejected after allocating device storage");
  }
  auto source = image->prepareCoordinates(3, exact_bytes);
  const auto after_owner = metrics.snapshot();
  if (source->deviceWorkspaceBytes() != checkedDeviceBytes<double>(packed.size()) ||
      after_owner.live_bytes - before_owner.live_bytes != source->deviceWorkspaceBytes())
    throw std::runtime_error("Coordinate owner differs from its exact allocation accounting");
  source->upload(packed);
  const auto after_upload = metrics.snapshot();
  if (after_upload.upload_calls - after_owner.upload_calls != 1 ||
      after_upload.uploaded_bytes - after_owner.uploaded_bytes != packed.size() * sizeof(double))
    throw std::runtime_error("Coordinate source did not perform exactly one bounded H2D upload");

  double maximum_error = 0;
  std::size_t measured_copies = 0, measured_copy_bytes = 0;
  // Match ordinary value/score tapes and full VGL across full/nonzero-begin/tail intervals.
  for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::FULL_VGL})
  {
    const bool score_tape = mode == SpatialJetMode::VALUE_ONLY;
    PreparedDeviceInference host(image, 2, queue, blas, mode, score_tape);
    PreparedDeviceInference device(image, 2, queue, blas, mode, score_tape);
    PreparedDeviceScoreReverse score(image, 2, queue, blas);
    std::vector<double> host_score(image->plan().parameterCount()), device_score(host_score.size());
    for (const auto interval : {std::array<std::size_t, 2>{0, 2}, std::array<std::size_t, 2>{1, 2},
                                std::array<std::size_t, 2>{2, 1}, std::array<std::size_t, 2>{0, 2}})
    {
      const auto begin = interval[0], count = interval[1];
      const auto view = source->view(begin, count);
      if (view.batchCount() != count || view.physicalElectrons() != ne || view.elements() != count * stride ||
          view.deviceOrdinal() != ordinal || !view.sourceIdentity() || view.sourceGeneration() == 0)
        throw std::runtime_error("Coordinate view lost its typed source extent or identity");
      std::vector<double> positions(packed.begin() + begin * stride, packed.begin() + (begin + count) * stride);
      host.evaluate(positions);
      device.evaluateDeviceCoordinates(view);
      const auto fixed_bytes = device.deviceWorkspaceBytes();
      const auto before = metrics.snapshot();
      device.evaluateDeviceCoordinates(view);
      const auto after = metrics.snapshot();
      if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
          after.upload_calls != before.upload_calls || after.uploaded_bytes != before.uploaded_bytes ||
          after.device_copy_calls - before.device_copy_calls != 1 ||
          after.device_copied_bytes - before.device_copied_bytes != view.elements() * sizeof(double) ||
          device.deviceWorkspaceBytes() != fixed_bytes)
        throw std::runtime_error("Coordinate reuse allocated, uploaded coordinates or lost exact D2D accounting");
      measured_copies += after.device_copy_calls - before.device_copy_calls;
      measured_copy_bytes += after.device_copied_bytes - before.device_copied_bytes;
      std::vector<double> host_phases(count), device_phases(count),
          host_jets(spatialJetSpanElements(host.resultLayout())), device_jets(host_jets.size()),
          host_ratios(count * host.resultLayout().laplacian_lanes), device_ratios(host_ratios.size());
      host.downloadResults(host_phases, host_jets, host_ratios);
      device.downloadResults(device_phases, device_jets, device_ratios);
      // Compare all published scalars with an explicit finite, scale-aware bound.
      auto compare = [&](const std::vector<double>& actual, const std::vector<double>& expected) {
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
          const auto difference = std::abs(actual[index] - expected[index]);
          if (!device_math::isFiniteBinary64(difference) || difference > 1.e-10 * (1 + std::abs(expected[index])))
            throw std::runtime_error("Host-upload/device-coordinate numerical parity failed");
          maximum_error = std::max(maximum_error, difference);
        }
      };
      compare(device_phases, host_phases);
      compare(device_jets, host_jets);
      compare(device_ratios, host_ratios);
      if (score_tape)
      {
        std::vector<double> weights(count, -.3);
        if (count == 2)
          weights[1] = .2;
        score.reverse(host, weights);
        score.downloadResponse(host_score);
        score.reverse(device, weights);
        score.downloadResponse(device_score);
        compare(device_score, host_score);
      }
    }
  }

  // Every rejected borrow replaces a previously valid result and reverse tape.
  const std::vector<double> one(packed.begin(), packed.begin() + stride);
  PreparedDeviceInference guard(image, 2, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  auto reject = [&](const PackedDeviceCoordinatesView& view) {
    guard.evaluate(one);
    const auto before = metrics.snapshot();
    bool rejected = false, result_missing = false, tape_missing = false;
    try { guard.evaluateDeviceCoordinates(view); }
    catch (const std::invalid_argument&) { rejected = true; }
    const auto after = metrics.snapshot();
    std::vector<double> phases(1), values(1), ratios;
    try { guard.downloadResults(phases, values, ratios); }
    catch (const std::logic_error&) { result_missing = true; }
    try { (void)guard.valueHeadTape(); }
    catch (const std::logic_error&) { tape_missing = true; }
    if (!rejected || !result_missing || !tape_missing || after.device_copy_calls != before.device_copy_calls ||
        after.upload_calls != before.upload_calls || after.allocation_calls != before.allocation_calls)
      throw std::runtime_error("Rejected coordinate borrow copied data or retained prior outputs");
  };
  reject(PackedDeviceCoordinatesView{});
  reject(source->view(0, 3));
  auto stale = source->view(0, 1);
  source->upload(packed);
  reject(stale);
  stale = source->view(0, 1);
  auto invalid = packed;
  invalid.back() = std::numeric_limits<double>::quiet_NaN();
  bool invalid_upload = false;
  try { source->upload(invalid); }
  catch (const std::invalid_argument&) { invalid_upload = true; }
  if (!invalid_upload)
    throw std::runtime_error("Coordinate source accepted nonfinite input");
  reject(stale);
  source->upload(packed);
  auto expired_owner = image->prepareCoordinates(1, checkedDeviceBytes<double>(stride));
  expired_owner->upload(one);
  const auto expired = expired_owner->view(0, 1);
  expired_owner.reset();
  reject(expired);
  auto foreign_image = std::make_shared<PreparedDeviceModel>(image->plan(), model.p.flat_values(), model.cfg.nuclei.x);
  auto foreign = foreign_image->prepareCoordinates(1, checkedDeviceBytes<double>(stride));
  foreign->upload(one);
  reject(foreign->view(0, 1));
  for (const auto count : {std::size_t{0}, std::size_t{4}})
  {
    bool rejected = false;
    try { (void)source->view(0, count); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Coordinate source minted an invalid interval");
  }

  // Parameter publication does not change the source's immutable geometry family.
  const auto successor = image->makeUpdatedImage(model.p.flat_values(), image->version() + 1);
  guard.rebindModel(successor);
  guard.evaluateDeviceCoordinates(source->view(1, 1));
  (void)guard.valueHeadTape();
  int device_count = 0;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDeviceCount(&device_count), "coordinate validation device count");
#else
  checkRuntime(cudaGetDeviceCount(&device_count), "coordinate validation device count");
#endif
  if (device_count > 1)
  {
    // Restore the owning device before destruction even if the rejection fails.
    bool rejected = false;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipSetDevice((ordinal + 1) % device_count), "coordinate validation foreign device");
#else
    checkRuntime(cudaSetDevice((ordinal + 1) % device_count), "coordinate validation foreign device");
#endif
    try { guard.evaluateDeviceCoordinates(source->view(0, 1)); }
    catch (const std::invalid_argument&) { rejected = true; }
#ifdef QMC_CUDA2HIP
    checkRuntime(hipSetDevice(ordinal), "coordinate validation restore device");
#else
    checkRuntime(cudaSetDevice(ordinal), "coordinate validation restore device");
#endif
    if (!rejected)
      throw std::runtime_error("Inference accepted coordinates on a foreign current device");
  }
  std::cout << "PASS typed device coordinates: value/VGL/score full/tail parity, stale/expired/foreign source "
               "rejection, failure invalidation, successor image reuse, zero warmed allocations; max_abs_error="
            << maximum_error << " measured_D2D_calls=" << measured_copies << " measured_D2D_bytes="
            << measured_copy_bytes << " coordinate_H2D_reuse_calls=0 transfer_counter_scope=device_wide "
            << "foreign_device_check=" << (device_count > 1 ? "PASS" : "NOT_RUN_single_device") << '\n';
}

/// Compare host compatibility and retained score-only replays with measured transfers.
template<class Model>
void validateRetainedCoordinateReplays(Model& model, const std::shared_ptr<const PreparedDeviceModel>& image)
{
  if (PreparedDeviceProductSession::admittedScoreCapacity(image->plan().modelShape(), 4) != 4)
    return;
  using Complex = std::complex<double>;
  const auto stride = 3 * image->plan().modelShape().electrons(), parameters = image->plan().parameterCount();
  std::vector<double> positions(7 * stride), direction(parameters), imaginary_direction(parameters), weights(7);
  for (std::size_t sample = 0; sample < 7; ++sample)
  {
    std::copy_n(model.cfg.electrons.x.data() + (sample % model.cfg.nconfig) * stride, stride,
                positions.data() + sample * stride);
    weights[sample] = .2 * (static_cast<int>(sample % 3) - 1);
  }
  for (std::size_t parameter = 0; parameter < parameters; ++parameter)
  {
    direction[parameter] = .003 * (static_cast<int>(parameter % 7) - 3);
    imaginary_direction[parameter] = -.2 * direction[parameter];
  }
  DeviceExecutionPreparation preparation;
  preparation.retained_coordinate_workspace_budget_bytes = checkedDeviceBytes<double>(positions.size());
  PreparedDeviceBatchBackend backend(image, {}, PsiFormerPrecisionPolicy::FP64_REFERENCE, preparation);
  auto session = backend.prepareProductSession(ProductSessionProfile::SCORE_ONLY, 4);
  session->prepareScoreContractions(3, image->version());
  const auto before_source = session->transferSnapshot();
  auto source = session->prepareSampleCoordinates(positions, image->version());
  const auto after_source = session->transferSnapshot();
  if (after_source.upload_calls - before_source.upload_calls != 1 ||
      after_source.uploaded_bytes - before_source.uploaded_bytes != positions.size() * sizeof(double) ||
      after_source.live_bytes - before_source.live_bytes != source->deviceWorkspaceBytes())
    throw std::runtime_error("Retained operator did not prepare exactly one whole-population upload");
  const auto fixed_bytes = session->deviceWorkspaceBytes(), fixed_host_bytes = session->hostWorkspaceBytes();
  std::vector<Complex> expected(3 * parameters), actual(expected.size()), expected_jvp(7), actual_jvp(7);
  double maximum_error = 0;
  std::size_t saved_coordinate_bytes = 0, measured_copies = 0;
  // Exact complex coefficients are already in the caller's selected adjoint convention.
  const auto coefficients = [](std::size_t sample) {
    return std::array<Complex, 3>{Complex(.1 * (sample + 1), -.03 * sample),
        sample % 3 == 0 ? Complex{} : Complex(-.07 * sample, .11), Complex(.02, -.09 * (sample + 1))};
  };
  // Measure the same numerical contraction twice, differing only in input source.
  const auto run_vjp = [&](bool retained, bool weighted, std::size_t population, std::vector<Complex>& output) {
    const auto channels = weighted ? std::size_t{1} : std::size_t{3};
    session->beginScoreContraction(channels, image->version());
    if (weighted)
      for (std::size_t offset = 0; offset < population; offset += 4)
      {
        const auto count = std::min(std::size_t{4}, population - offset);
        if (retained)
          session->accumulateWeightedScoreTileDeviceCoordinates(source->view(offset, count), weights.data() + offset, image->version());
        else
          session->accumulateWeightedScoreTile(positions.data() + offset * stride, weights.data() + offset, count, image->version());
      }
    else
      for (std::size_t sample = 0; sample < population; ++sample)
      {
        const auto values = coefficients(sample);
        if (retained)
          session->accumulateScoreContractionDeviceCoordinates(source->view(sample, 1), values.data(), channels, image->version());
        else
          session->accumulateScoreContraction(positions.data() + sample * stride, values.data(), channels, image->version());
      }
    session->finishScoreContraction(image->version());
    for (std::size_t channel = 0; channel < channels; ++channel)
      session->downloadScoreContraction(channel, output.data() + channel * parameters, parameters, image->version());
  };
  // Keep the one canonical direction resident throughout all tiles of each component.
  const auto run_jvp = [&](bool retained, std::size_t population, std::vector<Complex>& output) {
    std::fill(output.begin(), output.end(), Complex{});
    for (unsigned component = 0; component < 2; ++component)
    {
      session->prepareScoreDirection(component == 0 ? direction : imaginary_direction, image->version());
      for (std::size_t offset = 0; offset < population; offset += 4)
      {
        const auto count = std::min(std::size_t{4}, population - offset);
        const auto& products = retained
            ? session->scoreJvpPreparedTileDeviceCoordinates(source->view(offset, count), image->version())
            : session->scoreJvpPreparedTile(positions.data() + offset * stride, count, image->version());
        for (std::size_t sample = 0; sample < count; ++sample)
          output[offset + sample] += (component == 0 ? Complex(1, 0) : Complex(0, 1)) * products[sample];
      }
    }
  };
  // Check complete scalar/P output vectors without assuming identical reduction order.
  const auto compare = [&](const std::vector<Complex>& lhs, const std::vector<Complex>& rhs, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index)
    {
      const auto error = std::abs(lhs[index] - rhs[index]);
      if (!device_math::isFiniteBinary64(error) || error > 1.e-10 * (1 + std::abs(rhs[index])))
        throw std::runtime_error("Retained coordinate score replay parity failed");
      maximum_error = std::max(maximum_error, error);
    }
  };
  // Compare transfer intervals at this serial validation point, not across crowds.
  const auto measure = [&](const auto& operation, std::size_t copy_count, std::size_t coordinate_bytes) {
    const auto before = session->transferSnapshot();
    operation(false);
    const auto middle = session->transferSnapshot();
    operation(true);
    const auto after = session->transferSnapshot();
    if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
        session->deviceWorkspaceBytes() != fixed_bytes || session->hostWorkspaceBytes() != fixed_host_bytes ||
        middle.uploaded_bytes - before.uploaded_bytes != after.uploaded_bytes - middle.uploaded_bytes + coordinate_bytes ||
        middle.upload_calls - before.upload_calls != after.upload_calls - middle.upload_calls + copy_count ||
        after.device_copy_calls - middle.device_copy_calls != copy_count ||
        after.device_copied_bytes - middle.device_copied_bytes != coordinate_bytes)
      throw std::runtime_error("Retained score replay lost exact coordinate transfers or warmed storage");
    saved_coordinate_bytes += (middle.uploaded_bytes - before.uploaded_bytes) -
        (after.uploaded_bytes - middle.uploaded_bytes);
    measured_copies += after.device_copy_calls - middle.device_copy_calls;
  };
  for (const auto population : {std::size_t{1}, std::size_t{4}, std::size_t{5}, std::size_t{7}})
  {
    for (bool weighted : {false, true})
    {
      const auto channels = weighted ? std::size_t{1} : std::size_t{3};
      measure([&](bool retained) { run_vjp(retained, weighted, population, retained ? actual : expected); },
          weighted ? (population + 3) / 4 : population, population * stride * sizeof(double));
      compare(actual, expected, channels * parameters);
    }
    measure([&](bool retained) { run_jvp(retained, population, retained ? actual_jvp : expected_jvp); },
        2 * ((population + 3) / 4), 2 * population * stride * sizeof(double));
    compare(actual_jvp, expected_jvp, population);
  }
  const auto& host_single = session->score(positions.data(), image->version());
  std::vector<double> single_reference(host_single);
  const auto& device_single = session->scoreDeviceCoordinates(source->view(0, 1), image->version());
  for (std::size_t parameter = 0; parameter < parameters; ++parameter)
    if (std::abs(device_single[parameter] - single_reference[parameter]) > 1.e-10 * (1 + std::abs(single_reference[parameter])))
      throw std::runtime_error("Retained singleton score replay parity failed");

  // A late stale source invalidates all earlier accumulated channels, not just its tile.
  session->beginScoreContraction(3, image->version());
  const auto values = coefficients(0);
  session->accumulateScoreContractionDeviceCoordinates(source->view(0, 1), values.data(), 3, image->version());
  const auto stale = source->view(1, 1);
  source->upload(positions); // Failure injection only; normal operators never mutate their source.
  bool rejected = false, missing = false;
  try { session->accumulateScoreContractionDeviceCoordinates(stale, values.data(), 3, image->version()); }
  catch (const std::invalid_argument&) { rejected = true; }
  try { session->finishScoreContraction(image->version()); }
  catch (const std::exception&) { missing = true; }
  if (!rejected || !missing)
    throw std::runtime_error("Late retained source failure published a partial score contraction");
  session->prepareScoreDirection(direction, image->version());
  const auto& borrowed = session->scoreJvpPreparedTileDeviceCoordinates(source->view(0, 1), image->version());
  rejected = false;
  try { session->scoreJvpPreparedTileDeviceCoordinates(stale, image->version()); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected || !borrowed.empty())
    throw std::runtime_error("Failed retained JVP kept prior scalar products");
  backend.publishParameters(model.p.flat_values(), image->version() + 1, []() {});
  rejected = false;
  try { session->scoreDeviceCoordinates(source->view(0, 1), image->version()); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected)
    throw std::runtime_error("Retained source bypassed stale parameter-version admission");
  session->prepareScoreDirection(direction, image->version() + 1);
  session->scoreJvpPreparedTileDeviceCoordinates(source->view(0, 1), image->version() + 1);
  std::cout << "PASS retained operator coordinates: wholeB7 upload once, score-only VJP/JVP B1/4/5/7 "
               "tails, signed complex channels, singleton, failure/stale/successor reuse, zero warmed allocations; "
            << "max_abs_error=" << maximum_error << " measured_coordinate_H2D_bytes_saved=" << saved_coordinate_bytes
            << " measured_D2D_calls=" << measured_copies << " transfer_counter_scope=device_wide\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
