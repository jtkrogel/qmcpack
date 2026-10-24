//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_PROJECTION_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_PROJECTION_VALIDATION_H
#include "PsiFormerDeviceOrbitalProjection.h"
#include "PsiFormerDeviceComplexSpatial.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalProjection.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Reuse the exact Ne64 seed/geometry and compare complete physical complex G/L.
inline void validateBatchedHeadFullOwner(bool inverse_columns = false)
{
  const auto descriptors = makeTargetCostPlan(64, true, true);
  const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
  std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(3 * 64);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
  const double irrational[3]{.6180339887498948, .4142135623730951, .7320508075688772};
  for (std::size_t e = 0; e < 64; ++e)
    for (std::size_t d = 0; d < 3; ++d)
    {
      const double fraction = (e + 1) * irrational[d] + .13 * (d + 1);
      positions[3 * e + d] = 8 * (fraction - std::floor(fraction));
    }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto old_cost = estimateComplexSpatialCost(plan, 1, 8, inverse_columns);
  const auto cost = estimateComplexSpatialCost(plan, 1, 8, true);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "FULL head supplement device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "FULL head supplement device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  if (inverse_columns)
  {
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { (void)PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, cost.device_bytes - 1, 8, true, true); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Inverse-column cost-minus-one did not reject before numeric allocation");
  }
  auto ordinary = PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, old_cost.device_bytes, 8, inverse_columns);
  const auto cold = metrics.snapshot();
  auto batched = PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, cost.device_bytes, 8, true, inverse_columns);
  if (metrics.snapshot().live_bytes - cold.live_bytes != cost.device_bytes ||
      cost.device_bytes - old_cost.device_bytes != (inverse_columns ? 0 : cost.projection_pointer_bytes) ||
      cost.host_bytes != old_cost.host_bytes)
    throw std::runtime_error("FULL head supplement owner differs from exact preflight");

  std::size_t comparisons = 0;
  double max_g = 0, max_l = 0, max_ratio = 0, max_amplitude = 0;
  // Existing complete complex FULL validator uses 2e-11*(1+abs(reference)).
  const auto scalar = [&](double reference, double actual, double& maximum) {
    const double error = std::abs(reference - actual);
    maximum = std::max(maximum, error);
    if (!std::isfinite(reference) || !std::isfinite(actual) || error > 2e-11 * (1 + std::abs(reference)))
      throw std::runtime_error("FULL batched-head component exceeds existing spatial FP64 contract");
    ++comparisons;
  };
  const auto complex = [&](cuDoubleComplex reference, cuDoubleComplex actual, double& maximum) {
    scalar(cuCreal(reference), cuCreal(actual), maximum);
    scalar(cuCimag(reference), cuCimag(actual), maximum);
  };
  const auto compare = [&] {
    const auto warm = metrics.snapshot();
    ordinary->evaluate(positions);
    const auto old_evaluated = metrics.snapshot();
    batched->evaluate(positions);
    const auto both_evaluated = metrics.snapshot();
    if (warm.allocation_calls != both_evaluated.allocation_calls || warm.free_calls != both_evaluated.free_calls ||
        old_evaluated.uploaded_bytes - warm.uploaded_bytes != both_evaluated.uploaded_bytes - old_evaluated.uploaded_bytes)
      throw std::runtime_error("FULL batched-head warm path allocated or added HOST uploads");
    std::vector<ComplexValueResult> a(1), b(1);
    std::vector<cuDoubleComplex> ag(192), bg(192), al(64), bl(64), ar(64), br(64);
    ordinary->download(ordinary->resultView(), a, ag, al, ar);
    batched->download(batched->resultView(), b, bg, bl, br);
    if (a[0].status != 0 || b[0].status != 0)
      throw std::runtime_error("FULL head supplement requires two successful physical owners");
    scalar(a[0].log_abs, b[0].log_abs, max_amplitude);
    complex(a[0].phase, b[0].phase, max_amplitude);
    for (std::size_t i = 0; i < ag.size(); ++i) complex(ag[i], bg[i], max_g);
    for (std::size_t i = 0; i < al.size(); ++i)
    {
      complex(al[i], bl[i], max_l);
      complex(ar[i], br[i], max_ratio);
    }
  };
  compare();
  auto parameters = initialized.values;
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT) parameters[tensor.begin] += .0001;
  auto successor = image->makeUpdatedImage(parameters, 1);
  ordinary->rebindModel(successor);
  batched->rebindModel(successor);
  compare();
  if (inverse_columns)
  {
    const auto stale = batched->resultView();
    bool rejected_input = false, rejected_view = false;
    try { batched->evaluate({}); }
    catch (const std::invalid_argument&) { rejected_input = true; }
    std::vector<ComplexValueResult> amplitude(1);
    amplitude[0].status = 777;
    const auto marker = make_cuDoubleComplex(13, -9);
    std::vector<cuDoubleComplex> gradient(192, marker), lap_log(64, marker), lap_ratio(64, marker);
    try { batched->download(stale, amplitude, gradient, lap_log, lap_ratio); }
    catch (const std::invalid_argument&) { rejected_view = true; }
    const auto unchanged = [&](const auto& values) {
      return std::all_of(values.begin(), values.end(), [&](const auto value) {
        return cuCreal(value) == 13 && cuCimag(value) == -9;
      });
    };
    if (!rejected_input || !rejected_view || amplitude[0].status != 777 ||
        !unchanged(gradient) || !unchanged(lap_log) || !unchanged(lap_ratio))
      throw std::runtime_error("Inverse-column failed evaluation exposed stale or partial outputs");
    compare();
  }
  std::cout << std::setprecision(17) << "PASS direct " << (inverse_columns ? "inverse-column" : "batched")
            << " paired Ne64 FULL G/L comparisons=" << comparisons << " inverse_columns=" << inverse_columns
            << " max_G=" << max_g << " max_Llog=" << max_l << " max_Lratio=" << max_ratio
            << " max_amplitude=" << max_amplitude << " active_head_rebind=1 warm_allocations=0 added_host_uploads=0"
            << " pointer_device_bytes=" << cost.projection_pointer_bytes << " pointer_host_bytes=0\n";
}

/// Exercise actual Ne64 paired owners, exact costs, successor refresh and nodes.
inline void validateBatchedHeadOwner()
{
  const auto descriptors = makeTargetCostPlan(64, true, true);
  const auto initialized = initializePsiFormerParameters(descriptors.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
  std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(3 * 64 * 3);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
  const double irrational[3]{.6180339887498948, .4142135623730951, .7320508075688772};
  for (std::size_t b = 0; b < 3; ++b)
    for (std::size_t e = 0; e < 64; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const double fraction = (e + 1) * irrational[d] + .13 * (d + 1) + .017 * b;
        positions[b * 192 + 3 * e + d] = 8 * (fraction - std::floor(fraction));
      }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto old_cost = estimateComplexValueCost(plan, 3);
  const auto cost = estimateComplexValueCost(plan, 3, true, false, true);
  if (cost.device_bytes - old_cost.device_bytes != cost.projection_pointer_bytes ||
      cost.host_bytes != old_cost.host_bytes || !cost.projection_pointer_bytes)
    throw std::runtime_error("Batched head per-domain preflight is inconsistent");
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "batched head owner device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "batched head owner device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  bool rejected = false;
  try { (void)PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes - 1, false, true); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || before.allocation_calls != metrics.snapshot().allocation_calls)
    throw std::runtime_error("Batched head cost-minus-one allocated storage");
  auto ordinary = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, old_cost.device_bytes);
  const auto owner_before = metrics.snapshot();
  auto batched = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes, false, true);
  if (metrics.snapshot().live_bytes - owner_before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Batched head actual owner differs from preflight");
  std::size_t comparisons = 0;
  const auto compare = [&](std::size_t live) {
    const std::vector<double> input(positions.begin(), positions.begin() + live * 192);
    ordinary->evaluate(input);batched->evaluate(input);
    std::vector<ComplexValueResult> a(live), b(live);
    ordinary->download(ordinary->resultView(), a);batched->download(batched->resultView(), b);
    for (std::size_t i = 0; i < live; ++i)
    {
      if (a[i].log_abs == -std::numeric_limits<double>::infinity() &&
          b[i].log_abs == -std::numeric_limits<double>::infinity())
        continue;
      if (!std::isfinite(b[i].log_abs) || std::abs(a[i].log_abs - b[i].log_abs) > 2e-10 ||
          std::hypot(cuCreal(a[i].phase) - cuCreal(b[i].phase),
                     cuCimag(a[i].phase) - cuCimag(b[i].phase)) > 2e-10)
        throw std::runtime_error("Batched paired head changed canonical amplitudes");
      ++comparisons;
    }
  };
  const auto warm = metrics.snapshot();
  for (std::size_t live : {3u, 1u, 2u}) compare(live);
  if (warm.allocation_calls != metrics.snapshot().allocation_calls || warm.free_calls != metrics.snapshot().free_calls)
    throw std::runtime_error("Warm paired batched-head evaluation allocated storage");
  auto parameters = initialized.values;
  // Keep the old image alive and change BOTH active head components: stale
  // pointer banks must disagree with the canonical successor, not merely cusp.
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT)
      parameters[tensor.begin] += .0001;
  auto successor = image->makeUpdatedImage(parameters, 1);
  ordinary->rebindModel(successor);
  batched->rebindModel(successor);
  compare(3);
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      std::fill(parameters.begin() + tensor.begin, parameters.begin() + tensor.end, 0);
  auto node = successor->makeUpdatedImage(parameters, 2);
  ordinary->rebindModel(node);batched->rebindModel(node);compare(1);
  std::cout << "PASS batched paired Ne64 owner comparisons=" << comparisons
            << " cost_minus_one_noalloc=1 liveB=3,1,2 rebind=1 node=1 device_pointer_bytes="
            << cost.projection_pointer_bytes << " host_pointer_bytes=0\n";
}

/// Compare unchanged logical cells and padding with independent CPU and canonical GPU.
inline void validateOrbitalDeterminantBatches()
{
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  std::size_t comparisons = 0;
  double max_cpu = 0, max_old = 0;
  for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::FULL_VGL})
    for (const auto spin_up : {std::size_t(0), std::size_t(3), std::size_t(7)})
    {
      // Ragged spins, non-power-of-two D/B, padded source/weights/output, full/target tails.
      auto capacity = makeOpenOrbitalJetLayout(3, 3, 7, 2, 5, spin_up, mode, 8, 9, 65, 10, 31);
      std::vector<double> features(spatialJetSpanElements(capacity.features));
      std::vector<double> up(openOrbitalBackflowSpan(capacity)), down(up.size());
      const auto output_size = spatialJetSpanElements(capacity.orbitals);
      for (std::size_t i = 0; i < features.size(); ++i) features[i] = .23 * std::sin(.17 * (i + 1));
      for (std::size_t i = 0; i < up.size(); ++i)
      {
        up[i] = .31 * std::cos(.13 * (i + 2));
        down[i] = .29 * std::sin(.19 * (i + 3));
      }
      DeviceBuffer<double> x(features.size()), wu(up.size()), wd(down.size());
      DeviceBuffer<double> old_output(output_size), new_output(output_size);
      x.upload(features.data(), features.size());wu.upload(up.data(), up.size());wd.upload(down.data(), down.size());
      PreparedDeviceOrbitalProjection bank(capacity);
      const auto bytes = orbitalProjectionPointerBytes(3, capacity.features.plane_count, 3,
          (spin_up != 0) + (spin_up != 7));
      if (bank.deviceBytes() != bytes)
        throw std::runtime_error("Batched orbital pointer cost differs from actual capacity");
      auto bad = capacity;
      bad.backflow_feature_stride = std::size_t(std::numeric_limits<int>::max()) + 1;
      auto& admission_metrics = deviceMemoryMetrics(x.deviceOrdinal());
      const auto denied_before = admission_metrics.snapshot();
      bool denied = false;
      try { PreparedDeviceOrbitalProjection invalid(bad); }
      catch (const std::exception&) { denied = true; }
      if (!denied || admission_metrics.snapshot().allocation_calls != denied_before.allocation_calls)
        throw std::runtime_error("Unsupported batched BLAS stride allocated pointer storage");
      for (std::size_t live : {3u, 1u, 2u})
      {
        auto layout = capacity;
        layout.features.configuration_count = layout.orbitals.configuration_count = live;
        if (mode == SpatialJetMode::FULL_VGL && live == 2)
        {
          layout.features = withSpatialTargetRange(layout.features, 4, 3);
          layout.orbitals = withSpatialTargetRange(layout.orbitals, 4, 3);
        }
        std::vector<double> cpu(output_size, -.125), old(cpu), observed(cpu);
        old_output.upload(old.data(), old.size());new_output.upload(observed.data(), observed.size());
        forEachOpenOrbitalProjection(layout, [&](const OpenOrbitalProjectionSlice& slice) {
          const auto& dense = slice.dense;
          const auto& weights = slice.spin_up ? up : down;
          for (std::size_t row = 0; row < dense.rows; ++row)
            for (std::size_t col = 0; col < dense.output_width; ++col)
            {
              double sum = 0;
              for (std::size_t k = 0; k < dense.input_width; ++k)
                sum += features[slice.source_offset + row * dense.source_row_stride + k] *
                    weights[slice.weight_offset + k * dense.weight_row_stride + col];
              cpu[slice.target_offset + row * dense.target_row_stride + col] = sum;
            }
        });
        auto& metrics = deviceMemoryMetrics(x.deviceOrdinal());
        const auto before = metrics.snapshot();
        projectOpenOrbitalJets(blas, layout, x.data(), wu.data(), wd.data(), old_output.data());
        bank.project(blas, queue.getNative(), layout, x.data(), wu.data(), wd.data(), new_output.data());
        queue.sync();
        const auto after = metrics.snapshot();
        if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
            before.upload_calls != after.upload_calls)
          throw std::runtime_error("Warm determinant batching allocated or uploaded pointer storage");
        old_output.download(old.data(), old.size());new_output.download(observed.data(), observed.size());
        for (std::size_t i = 0; i < observed.size(); ++i)
        {
          const auto ce = std::abs(observed[i] - cpu[i]), oe = std::abs(observed[i] - old[i]);
          max_cpu = std::max(max_cpu, ce);max_old = std::max(max_old, oe);
          if (!std::isfinite(observed[i]) || ce > 2e-12 * (1 + std::abs(cpu[i])) ||
              oe > 2e-12 * (1 + std::abs(old[i])))
            throw std::runtime_error("Determinant batching changed logical projection or padding");
          ++comparisons;
        }
      }
    }
  std::cout << "PASS determinant-batched orbital projections comparisons=" << comparisons
            << " max_cpu=" << max_cpu << " max_old_gpu=" << max_old
            << " warm_allocations=0 warm_uploads=0 B_tail=1 target_tail=1 empty_spins=1\n";
  validateBatchedHeadOwner();
  validateBatchedHeadFullOwner();
}
} // namespace qmcplusplus::psiformer::device
#endif
