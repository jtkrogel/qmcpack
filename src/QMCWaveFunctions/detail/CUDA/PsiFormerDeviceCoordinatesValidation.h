//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATES_VALIDATION_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceScoreReverse.h"
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
} // namespace qmcplusplus::psiformer::device
#endif
