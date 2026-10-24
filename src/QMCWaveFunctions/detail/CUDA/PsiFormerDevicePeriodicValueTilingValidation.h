//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_TILING_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_PERIODIC_VALUE_TILING_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "PsiFormerDeviceComplexScore.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Full-buffer scalar GPU parity includes failed partial scratch, padding and recovery.
inline void validatePeriodicValueTiling(compute::Queue<PlatformKind::CUDA>& queue)
{
  const auto cell = periodic_geometry::makeCell({{{9., .3, .1}, {0., 10., .2}, {.1, 0., 11.}}},
                                               {{true, true, true}});
  std::size_t cases = 0;
  double max_host_error = 0.;
  for (const std::size_t ne : {64, 100, 200})
    for (const std::size_t batch : {3, 2, 1})
    {
      const auto mode = SpatialJetMode::VALUE_ONLY;
      const auto feature = makePeriodicFeatureJetLayout(batch, ne, ne, ne / 2, mode,
          5, 5 * ne + 11, 7, ne * (7 * ne + 1) + 13, ne * (7 * ne + 1) + 31);
      const auto cusp = makeOpenCuspJetLayout(batch, ne, ne / 2, mode,
          5, 5 * ne + 11, 3, 19);
      std::vector<double> base(batch * feature.position_configuration_stride, 17.);
      std::vector<double> nuclei(ne * 7, 23.);
      for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t e = 0; e < ne; ++e)
          for (std::size_t d = 0; d < 3; ++d)
            base[b * feature.position_configuration_stride + 5 * e + d] =
                .13 + .037 * e + .17 * d + .03 * b;
      for (std::size_t n = 0; n < ne; ++n)
        for (std::size_t d = 0; d < 3; ++d)
          nuclei[7 * n + d] = .09 + .051 * n + .23 * d;
      DeviceBuffer<double> positions(base.size()), ions(nuclei.size());
      ions.upload(nuclei.data(), nuclei.size());
      const auto scratch_cost = estimatePeriodicValueTilingCost(3, ne, ne);
      DeviceBuffer<double> terms(scratch_cost.term_elements);
      DeviceBuffer<std::uint32_t> flags(scratch_cost.flag_elements);
      const std::vector<double> poisoned_terms(scratch_cost.term_elements, std::numeric_limits<double>::quiet_NaN());
      const std::vector<std::uint32_t> poisoned_flags(scratch_cost.flag_elements, ~std::uint32_t(0));
      terms.upload(poisoned_terms.data(), poisoned_terms.size());
      flags.upload(poisoned_flags.data(), poisoned_flags.size());
      DeviceBuffer<std::size_t> active_device(batch);
      DeviceBuffer<OpenSpatialStatus> old_status(batch), new_status(batch);
      for (bool features : {true, false})
      {
        const auto& output = features ? feature.output : cusp.output;
        const auto span = spatialJetSpanElements(output) + 7;
        std::vector<double> initial(span, 97.), old(span), candidate(span), host(span);
        DeviceBuffer<double> old_output(span), new_output(span);
        std::vector<OpenSpatialStatus> old_codes(batch), new_codes(batch), host_codes(batch);
        for (int variant = 0; variant < 9; ++variant)
        {
          auto coordinates = base;
          std::vector<std::size_t> active(batch, ne - 1);
          double same = .7, opposite = 1.3;
          if (variant == 1) active.back() = ne;
          if (variant == 2) coordinates[(batch - 1) * feature.position_configuration_stride + 5 * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
          if (variant == 3) coordinates[(batch - 1) * feature.position_configuration_stride + 5 * (ne - 1)] = std::numeric_limits<double>::infinity();
          if (variant == 4)
            for (std::size_t d = 0; d < 3; ++d)
              if (features)
                coordinates[d] = nuclei[d]; // Exact EI radial limit.
              else
                coordinates[5 + d] = coordinates[d]; // Exact EE radial limit.
          if (variant == 5) same = std::numeric_limits<double>::infinity();
          if (variant == 6) same = opposite = 1e308;
          if (variant == 7)
            for (std::size_t d = 0; d < 3; ++d)
              coordinates[d] += cell.lattice[0][d];
          // The final good variant proves recovery after every hazard.
          positions.upload(coordinates.data(), coordinates.size());
          active_device.upload(active.data(), active.size());
          old_output.upload(initial.data(), span);
          new_output.upload(initial.data(), span);
          host = initial;
          for (std::size_t b = 0; b < batch; ++b)
            host_codes[b] = features ? open_spatial::buildOpenFeatureJetsForConfiguration(
                feature, coordinates.data(), nuclei.data(), b, active[b], host.data(), &cell) :
                open_spatial::buildOpenCuspJetsForConfiguration(
                    cusp, coordinates.data(), b, active[b], same, opposite, host.data(), &cell);
          if (variant == 0)
          {
            bool rejected = false;
            try
            {
              if (features)
              {
                const auto pairs = ne * ne;
                const auto needed = batch * (pairs / 128 + (pairs % 128 != 0));
                launchPeriodicFeatureValuesTiled(queue.getNative(), feature, cell, positions.data(),
                    ions.data(), active_device.data(), flags.data(), needed - 1, new_output.data(), new_status.data());
              }
              else
                launchPeriodicCuspValuesTiled(queue.getNative(), cusp, cell, positions.data(), active_device.data(),
                    same, opposite, terms.data(), batch * ne * ne - 1, flags.data(), flags.capacity(),
                    new_output.data(), new_status.data());
            }
            catch (const std::length_error&) { rejected = true; }
            if (!rejected)
              throw std::runtime_error("Periodic VALUE tiling accepted undersized declared scratch");
          }
          const auto before = deviceMemoryMetrics(0).snapshot();
          if (features)
          {
            checkRuntime(launchPeriodicFeatureJets(queue.getNative(), feature, cell, positions.data(),
                ions.data(), active_device.data(), old_output.data(), old_status.data()), "periodic scalar features");
            checkRuntime(launchPeriodicFeatureValuesTiled(queue.getNative(), feature, cell, positions.data(),
                ions.data(), active_device.data(), flags.data(), flags.capacity(),
                new_output.data(), new_status.data()), "periodic tiled features");
          }
          else
          {
            checkRuntime(launchPeriodicCuspJets(queue.getNative(), cusp, cell, positions.data(), active_device.data(),
                same, opposite, old_output.data(), old_status.data()), "periodic scalar cusp");
            checkRuntime(launchPeriodicCuspValuesTiled(queue.getNative(), cusp, cell, positions.data(),
                active_device.data(), same, opposite, terms.data(), terms.capacity(), flags.data(), flags.capacity(),
                new_output.data(), new_status.data()), "periodic tiled cusp");
          }
          queue.sync();
          const auto after = deviceMemoryMetrics(0).snapshot();
          if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
              before.live_bytes != after.live_bytes)
            throw std::runtime_error("Periodic VALUE geometry changed tracked owners");
          old_output.download(old.data(), span);
          new_output.download(candidate.data(), span);
          old_status.download(old_codes.data(), batch);
          new_status.download(new_codes.data(), batch);
          if (old_codes != new_codes || old_codes != host_codes)
            throw std::runtime_error("Periodic VALUE geometry status oracle differs");
          if (std::memcmp(old.data(), candidate.data(), span * sizeof(double)))
            throw std::runtime_error("Periodic VALUE geometry scalar GPU bits/padding/partial scratch differ");
          for (std::size_t b = 0; b < batch; ++b)
            if (old_codes[b] == OpenSpatialStatus::REGULAR)
              for (std::size_t e = 0; e < output.element_count; ++e)
              {
                const auto i = output.uncheckedValueOffset(b, e);
                const double error = std::abs(candidate[i] - host[i]);
                max_host_error = std::max(max_host_error, error);
                if (!device_math::isFiniteBinary64(candidate[i]) || error > 3e-12 * (1. + std::abs(host[i])))
                  throw std::runtime_error("Periodic VALUE geometry CPU oracle differs");
              }
          ++cases;
        }
      }
    }
  std::cout << "PASS periodic_VALUE_geometry_tiled cases=" << cases << " CPU_max_abs=" << max_host_error
            << " scalar_GPU_full_buffer_bits_exact padding failure_replay recovery scratch_minus1 warm_alloc0\n";
}
/// Observe explicit owner selection without adding a production runtime counter.
struct PeriodicValueTilingValidationAccess
{
  static void run(compute::Queue<PlatformKind::CUDA>& queue)
  {
    // Test cold parsing without leaving a changed process policy for other fixtures.
    struct RestoreOption
    {
      const char* key = "PSIFORMER_GPU_PERIODIC_VALUE_GEOMETRY_TILED";
      bool present = std::getenv(key) != nullptr;
      std::string value = present ? std::getenv(key) : "";
      ~RestoreOption() { if (present) ::setenv(key, value.c_str(), 1); else ::unsetenv(key); }
    } restore;
    ::unsetenv(restore.key);
    if (configuredPeriodicValueGeometryTiled())
      throw std::runtime_error("Tiled geometry changed default policy");
    ::setenv(restore.key, "0", 1);
    if (configuredPeriodicValueGeometryTiled())
      throw std::runtime_error("Tiled geometry rejected OFF");
    ::setenv(restore.key, "1", 1);
    if (!configuredPeriodicValueGeometryTiled())
      throw std::runtime_error("Tiled geometry rejected ON");
    ::setenv(restore.key, "invalid", 1);
    bool invalid_policy = false;
    try { (void)configuredPeriodicValueGeometryTiled(); }
    catch (const std::invalid_argument&) { invalid_policy = true; }
    if (!invalid_policy) throw std::runtime_error("Tiled geometry accepted a nonboolean option");
    ::setenv(restore.key, "0", 1);
    bool overflow = false;
    try { (void)estimatePeriodicValueTilingCost(3, std::numeric_limits<std::size_t>::max(), 1); }
    catch (const std::length_error&) { overflow = true; }
    if (!overflow) throw std::runtime_error("Tiled geometry unchecked capacity product");
    const auto descriptors = makeTargetCostPlan(64, true, true);
    const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
        PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
    const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
    constexpr std::size_t batch = 3, ne = 64;
    std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(batch * 3 * ne);
    for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 2.0 + 1.7 * std::sin(.37 * (i + 1));
    AcceleratorBlasHandle blas(queue);
    auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
    const auto baseline = estimateComplexValueCost(plan, batch);
    const auto tiled = estimateComplexValueCost(plan, batch, true, false, false, true);
    const auto extra = estimatePeriodicValueTilingCost(batch, ne, plan.modelShape().nuclei);
    if (tiled.device_bytes - baseline.device_bytes != extra.device_bytes ||
        tiled.trunk_bytes - baseline.trunk_bytes != extra.device_bytes ||
        tiled.host_bytes != baseline.host_bytes)
      throw std::runtime_error("Tiled VALUE per-domain cost differs from exact scratch");
    int ordinal;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "tiled VALUE owner device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "tiled VALUE owner device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    const auto before = metrics.snapshot();
    auto fallback = PreparedDeviceComplexValue::prepare(image, batch, queue, blas,
        tiled.device_bytes - 1, false, false, true);
    if (fallback->cost().periodic_geometry.device_bytes || fallback->trunk_->usesTiledPeriodicValueGeometry() ||
        metrics.snapshot().live_bytes - before.live_bytes != baseline.device_bytes)
      throw std::runtime_error("Tiled VALUE optional-minus1 did not prepare exact canonical baseline");
    fallback.reset();
    const auto allocated = metrics.snapshot();
    auto ordinary = PreparedDeviceComplexValue::prepare(image, batch, queue, blas, baseline.device_bytes);
    auto candidate = PreparedDeviceComplexValue::prepare(image, batch, queue, blas,
        tiled.device_bytes, false, false, true);
    if (!candidate->trunk_->usesTiledPeriodicValueGeometry() ||
        metrics.snapshot().live_bytes - allocated.live_bytes != baseline.device_bytes + tiled.device_bytes)
      throw std::runtime_error("Tiled VALUE actual owner inventory/selection differs");
    const auto compare = [&] (const std::vector<double>& coordinates) {
      ordinary->evaluate(coordinates);
      candidate->evaluate(coordinates);
      std::vector<ComplexValueResult> old_result(coordinates.size() / (3 * ne)), new_result(old_result.size());
      ordinary->download(ordinary->resultView(), old_result);
      candidate->download(candidate->resultView(), new_result);
      if (std::memcmp(old_result.data(), new_result.data(), old_result.size() * sizeof(ComplexValueResult)))
        throw std::runtime_error("Tiled VALUE complete owner changed canonical result bits");
    };
    const auto warm = metrics.snapshot();
    for (std::size_t live : {3u, 2u, 1u, 3u})
      compare(std::vector<double>(positions.begin(), positions.begin() + live * 3 * ne));
    if (warm.allocation_calls != metrics.snapshot().allocation_calls ||
        warm.free_calls != metrics.snapshot().free_calls)
      throw std::runtime_error("Tiled VALUE warm tail evaluation allocated DEVICE storage");
    auto stale = candidate->resultView();
    auto invalid = positions;
    invalid[(batch - 1) * 3 * ne + 3 * (ne - 1)] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { candidate->evaluate(invalid); } catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Tiled VALUE failed configuration unexpectedly published");
    std::vector<ComplexValueResult> sink(batch);
    rejected = false;
    try { candidate->download(stale, sink); } catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Tiled VALUE retained a failed result epoch");
    compare(positions);
    auto successor = image->makeUpdatedImage(initialized.values, 1);
    const auto rebind = metrics.snapshot();
    ordinary->rebindModel(successor);
    candidate->rebindModel(successor);
    if (rebind.allocation_calls != metrics.snapshot().allocation_calls ||
        !candidate->trunk_->usesTiledPeriodicValueGeometry())
      throw std::runtime_error("Tiled VALUE rebind changed admitted scratch");
    compare(positions);
    const auto score_cost = estimateComplexScoreCost(plan);
    const auto retained_before = metrics.snapshot();
    auto score = PreparedDeviceComplexScore::prepare(successor, queue, blas, score_cost.device_bytes);
    if (score_cost.forward.periodic_geometry.device_bytes ||
        metrics.snapshot().live_bytes - retained_before.live_bytes != score_cost.device_bytes)
      throw std::runtime_error("Retained score inherited ordinary tiled geometry");
    std::cout << "PASS periodic_VALUE_tiling_owner Ne=64 B3_B2_B1 exact_cost optional_minus1_canonical"
              << " result_bits_exact late_failure recovery rebind warm_alloc0 retained_excluded\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
