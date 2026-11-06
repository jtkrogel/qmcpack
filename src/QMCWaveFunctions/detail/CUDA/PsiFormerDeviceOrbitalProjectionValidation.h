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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Compare complete-Pair validity/radius against the native host oracle, including two cells.
inline void validateSharedPairRadiusProducer()
{
  compute::Queue<PlatformKind::CUDA> queue;
  auto layout = makeOpenOrbitalJetLayout(3, 16, 44, 4, 2, 26, SpatialJetMode::VALUE_ONLY,
                                         0, 0, 0, 0, 0, 0, 0, 4, 0, 4);
  const auto cell = periodic_geometry::makeTwoCell(
      {{{12., .4, .2}, {.2, 14., .6}, {.4, .2, 16.}}},
      {{{6., .2, .1}, {.1, 7., .3}, {.2, .1, 8.}}}, {true, true, true});
  std::vector<double> positions(3 * layout.position_configuration_stride), nuclei(16);
  for (std::size_t b = 0; b < 3; ++b)
    for (std::size_t e = 0; e < 44; ++e)
      for (std::size_t d = 0; d < 3; ++d)
        positions[b * layout.position_configuration_stride + 4 * e + d] = .1 + .021 * e + .13 * d + .03 * b;
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t d = 0; d < 3; ++d) nuclei[4 * i + d] = .4 + .17 * i + .11 * d;
  const auto original = positions;
  const auto capacity = estimatePeriodicValuePairCost(3, 44, 4).pair_elements;
  DeviceBuffer<double> point(positions.size()), ions(nuclei.size()), radius(capacity);
  DeviceBuffer<std::uint8_t> valid(capacity);
  ions.upload(nuclei.data(), nuclei.size());
  const auto compare = [&](std::size_t live) {
    layout.features.configuration_count = layout.orbitals.configuration_count = live;
    point.upload(positions.data(), positions.size());
    for (const bool paired : {false, true})
    {
      checkRuntime(paired
          ? launchComplexValuePairs(queue.getNative(), layout, cell, point.data(), ions.data(),
                                    radius.data(), valid.data(), capacity)
          : launchPeriodicValuePairs(queue.getNative(), layout, cell, point.data(), ions.data(),
                                     radius.data(), valid.data(), capacity), "Pair radius producer oracle");
      queue.sync();
      const auto count = live * 44 * 4;
      std::vector<double> r(count);
      std::vector<std::uint8_t> v(count);
      radius.download(r.data(), count);
      valid.download(v.data(), count);
      for (std::size_t index = 0; index < count; ++index)
      {
        const auto b = index / (44 * 4), e = (index / 4) % 44, ion = index % 4;
        periodic_geometry::Pair pair;
        const bool expected = periodic_geometry::evaluatePair(cell,
            positions.data() + b * layout.position_configuration_stride + 4 * e,
            nuclei.data() + 4 * ion, pair, true);
        if (bool(v[index]) != expected ||
            (expected && std::abs(r[index] - pair.radius) > 1e-13 * (1 + std::abs(pair.radius))) ||
            (!expected && r[index] != 0.))
          throw std::runtime_error("Shared Pair producer differs from full host Pair radius/validity");
      }
    }
  };
  compare(3);
  std::copy_n(nuclei.data(), 3, positions.data());compare(1);
  positions[0] += 1e-13;compare(2);
  positions[0] = -0.;compare(3);
  for (std::size_t d = 0; d < 3; ++d) positions[d] += cell.ion_lattice[1][d];
  compare(2);
  for (const double bad : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max()})
  { positions[0] = bad;compare(3); }
  positions[0] = std::numeric_limits<double>::max();
  nuclei[0] = -std::numeric_limits<double>::max();
  ions.upload(nuclei.data(), nuclei.size());compare(3);
  nuclei[0] = .4;ions.upload(nuclei.data(), nuclei.size());
  positions = original;compare(3);
  auto malformed = cell;
  malformed.ion_reciprocal[0][0] += .1;
  bool rejected = false;
  try { (void)launchPeriodicValuePairs(queue.getNative(), layout, malformed, point.data(), ions.data(),
                                       radius.data(), valid.data(), capacity); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Pair producer admitted an inconsistent primitive cell");
  rejected = false;
  try { (void)launchPeriodicValuePairs(queue.getNative(), layout, cell, point.data(), ions.data(),
                                       radius.data(), valid.data(), capacity - 1); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("Pair producer admitted a short live bank");
  std::cout << "PASS shared Pair producer real_and_complex_math N44 ions4 skew_two_cell tails coalescence nonfinite recovery malformed\n";
}

/// Inspect the real envelope and both projections without exposing production views.
struct SharedPairGeometryValidationAccess
{
  static void compare(PreparedDeviceComplexValue& baseline, PreparedDeviceComplexValue& banked, std::size_t live,
                      std::size_t ne, std::size_t determinants)
  {
    const auto elements = baseline.cost_.matrices / baseline.results_.capacity() * live;
    std::vector<double> a(elements), b(elements);
    baseline.envelope_.download(a.data(), elements);
    banked.envelope_.download(b.data(), elements);
    if (std::memcmp(a.data(), b.data(), elements * sizeof(double)))
    {
      const auto bits = [](double value) {
        std::uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
      };
      const auto ordered = [](std::uint64_t value) {
        return value & (UINT64_C(1) << 63) ? ~value : value | (UINT64_C(1) << 63);
      };
      std::size_t first = elements, mismatches = 0;
      std::uint64_t maximum_ulp = 0;
      double maximum_absolute = 0.;
      for (std::size_t i = 0; i < elements; ++i)
        if (bits(a[i]) != bits(b[i]))
        {
          if (first == elements) first = i;
          ++mismatches;
          const auto x = ordered(bits(a[i])), y = ordered(bits(b[i]));
          maximum_ulp = std::max(maximum_ulp, x > y ? x - y : y - x);
          maximum_absolute = std::max(maximum_absolute, std::abs(a[i] - b[i]));
        }
      std::cerr << std::setprecision(17)
                << "PAIR_ENVELOPE_MISMATCH stage=real_envelope Ne=" << ne << " D=" << determinants
                << " live=" << live << " first=" << first << " configuration=" << first / (determinants * ne * ne)
                << " determinant=" << (first / (ne * ne)) % determinants
                << " row=" << (first / ne) % ne << " column=" << first % ne
                << " baseline=" << a[first] << " banked=" << b[first]
                << " absolute=" << std::abs(a[first] - b[first])
                << " baseline_hex=" << std::hexfloat << a[first] << " banked_hex=" << b[first] << std::defaultfloat
                << " baseline_bits=" << std::hex << bits(a[first]) << " banked_bits=" << bits(b[first]) << std::dec
                << " mismatches=" << mismatches << " maximum_ulp=" << maximum_ulp
                << " maximum_absolute=" << maximum_absolute << '\n';
      throw std::runtime_error("Complex Pair bank changed the canonical real envelope bits");
    }
    for (std::size_t component = 0; component < 2; ++component)
    {
      baseline.projections_.download(a.data(), elements, component * baseline.cost_.matrices);
      banked.projections_.download(b.data(), elements, component * banked.cost_.matrices);
      if (std::memcmp(a.data(), b.data(), elements * sizeof(double)))
        throw std::runtime_error("Complex Pair bank changed projected head authority");
    }
  }
};

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
inline void validateBatchedHeadOwner(std::size_t ne = 64, std::size_t determinants = 16, bool pair_geometry = false)
{
  const auto descriptors = makeTargetCostPlan(ne, true, true);
  auto shape = descriptors.modelShape();
  if (ne == 44) { shape.spin_up_electrons = 26; shape.spin_down_electrons = 18; }
  auto initialized = initializePsiFormerParameters(shape, 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptors.environment());
  if (determinants != shape.determinants)
  {
    // Expand the canonical D16 fixture using native tensor roles/layouts.
    // New real/imaginary channels get distinct small perturbations; the trunk
    // and finite envelope coefficients retain their original conditioning.
    const PsiFormerExecutionPlan original(shape, initialized.layouts, initialized.environment);
    const auto old_channels = shape.determinants * ne, channels = determinants * ne;
    std::vector<ParameterLayoutInput> layouts;
    std::vector<double> values;
    for (const auto& tensor : original.parameterTensors())
    {
      auto dimensions = tensor.shape;
      const bool backflow = tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT ||
          tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT;
      const bool envelope = tensor.role == ParameterRole::ENVELOPE_PI_UP ||
          tensor.role == ParameterRole::ENVELOPE_PI_DOWN || tensor.role == ParameterRole::ENVELOPE_ZETA_UP ||
          tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN;
      if (backflow) dimensions[1] = channels;
      if (envelope) dimensions[0] = channels;
      std::size_t count = 1;
      for (auto dimension : dimensions)
        count = spatial_detail::checkedProduct(count, dimension, "Paired head fixture extent overflow");
      const auto begin = values.size();
      for (std::size_t i = 0; i < count; ++i)
      {
        const auto source = backflow ? (i / channels) * old_channels + (i % channels) % old_channels :
            envelope ? ((i / shape.nuclei) % old_channels) * shape.nuclei + i % shape.nuclei : i;
        double value = initialized.values[tensor.begin + source];
        if (backflow && i % channels >= old_channels)
        {
          const double component = tensor.head_component == HeadTensorComponent::IMAG ? .37 : .13;
          const double spin = tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT ? .11 : .29;
          value += 1e-5 * std::sin(.017 * double(i / channels + 1) * double(i % channels + 1) + component + spin);
        }
        values.push_back(value);
      }
      layouts.push_back({tensor.module, tensor.name, dimensions, begin, values.size()});
    }
    initialized.model_shape.determinants = determinants;
    initialized.layouts = std::move(layouts);
    initialized.values = std::move(values);
  }
  const PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
  std::vector<double> nuclei(3 * plan.modelShape().nuclei), positions(3 * ne * 3);
  for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = .23 + .071 * i;
  const double irrational[3]{.6180339887498948, .4142135623730951, .7320508075688772};
  for (std::size_t b = 0; b < 3; ++b)
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
      {
        const double fraction = (e + 1) * irrational[d] + .13 * (d + 1) + .017 * b;
        positions[b * 3 * ne + 3 * e + d] = 8 * (fraction - std::floor(fraction));
      }
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto old_cost = estimateComplexValueCost(plan, 3, true, false, pair_geometry);
  const auto cost = estimateComplexValueCost(plan, 3, true, false, true, false, pair_geometry);
  // Grouping remains restricted to ordinary physical VALUE roles and the
  // admitted fixed legacy torus. Geometry-policy rejection is independent of Ne.
  bool rejected_role = false;
  try { (void)estimateComplexValueCost(plan, 3, false, false, true); }
  catch (const std::invalid_argument&) { rejected_role = true; }
  if (!rejected_role)
    throw std::runtime_error("Grouped paired projection admitted a retained VALUE role");
  for (int policy = 0; policy < 2; ++policy)
  {
    auto environment = initialized.environment;
    if (policy == 0) environment.fixed_nuclei = false;
    else environment.periodic_axes[2] = false;
    bool rejected_policy = false;
    try
    {
      const PsiFormerExecutionPlan invalid(plan.modelShape(), initialized.layouts, environment);
      (void)estimateComplexValueCost(invalid, 3, true, false, true);
    }
    catch (const std::invalid_argument&) { rejected_policy = true; }
    if (!rejected_policy)
      throw std::runtime_error("Grouped paired projection admitted an unsupported geometry policy");
  }
  if (cost.device_bytes - old_cost.device_bytes !=
          (pair_geometry ? cost.envelope_pairs.device_bytes : cost.projection_pointer_bytes) ||
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
  try { (void)PreparedDeviceComplexValue::prepare(image, 3, queue, blas, (pair_geometry ? old_cost.device_bytes : cost.device_bytes) - 1, false, true, false, pair_geometry); }
  catch (const std::length_error&) { rejected = true; }
  if (!rejected || before.allocation_calls != metrics.snapshot().allocation_calls)
    throw std::runtime_error("Batched head cost-minus-one allocated storage");
  auto ordinary = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, old_cost.device_bytes, false, pair_geometry);
  if (pair_geometry)
  {
    const auto fallback_before = metrics.snapshot();
    auto fallback = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes - 1,
                                                       false, true, false, true);
    if (fallback->cost().envelope_pairs.pair_elements || fallback->cost().device_bytes != old_cost.device_bytes ||
        metrics.snapshot().live_bytes - fallback_before.live_bytes != old_cost.device_bytes)
      throw std::runtime_error("Complex Pair optional fallback changed baseline ownership");
  }
  const auto owner_before = metrics.snapshot();
  auto batched = PreparedDeviceComplexValue::prepare(image, 3, queue, blas, cost.device_bytes, false, true, false, pair_geometry);
  if (metrics.snapshot().live_bytes - owner_before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Batched head actual owner differs from preflight");
  std::size_t comparisons = 0;
  const auto compare = [&](std::size_t live) {
    const std::vector<double> input(positions.begin(), positions.begin() + live * 3 * ne);
    ordinary->evaluate(input);batched->evaluate(input);
    if (pair_geometry) SharedPairGeometryValidationAccess::compare(*ordinary, *batched, live, ne, determinants);
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
  if (pair_geometry)
  {
    const auto stale = batched->resultView();
    auto malformed = positions;
    malformed.back() = std::numeric_limits<double>::quiet_NaN();
    bool failed = false;
    try { batched->evaluate(malformed); }
    catch (const std::exception&) { failed = true; }
    if (!failed) throw std::runtime_error("Complex Pair owner accepted nonfinite coordinates");
    std::vector<ComplexValueResult> revoked(3);
    bool stale_rejected = false;
    try { batched->download(stale, revoked); }
    catch (const std::exception&) { stale_rejected = true; }
    if (!stale_rejected) throw std::runtime_error("Complex Pair owner published a failed generation");
    compare(2);
  }
  for (const auto& tensor : plan.parameterTensors())
    if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
      std::fill(parameters.begin() + tensor.begin, parameters.begin() + tensor.end, 0);
  auto node = successor->makeUpdatedImage(parameters, 2);
  ordinary->rebindModel(node);batched->rebindModel(node);compare(1);
  std::cout << "PASS batched paired owner Ne=" << ne << " D=" << determinants << " comparisons=" << comparisons
            << " pair_geometry=" << pair_geometry
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
  struct Case { SpatialJetMode mode; std::size_t ne, determinants, spin_up; };
  std::vector<Case> cases;
  for (const auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::FULL_VGL})
    for (const auto spin_up : {std::size_t(0), std::size_t(3), std::size_t(7)})
      cases.push_back({mode, 7, 3, spin_up});
  for (const std::size_t determinants : {16, 64})
    cases.push_back({SpatialJetMode::VALUE_ONLY, 44, determinants, 26});
  for (const auto& test : cases)
    {
      const auto mode = test.mode;
      const auto spin_up = test.spin_up, ne = test.ne, determinants = test.determinants;
      // Ragged spins, N44 D16/D64, padded source/weights/output, full/target tails.
      auto capacity = makeOpenOrbitalJetLayout(3, determinants, ne, 2, 5, spin_up, mode,
          8, ne + 2, ne * (ne + 2) + 2, ne + 3, determinants * (ne + 3) + 1);
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
      DeviceBuffer<double> old_output(output_size), new_output(output_size), determinant_output(output_size);
      x.upload(features.data(), features.size());wu.upload(up.data(), up.size());wd.upload(down.data(), down.size());
      PreparedDeviceOrbitalProjection bank(capacity);
      const auto bytes = orbitalProjectionPointerBytes(3, capacity.features.plane_count, determinants,
          (spin_up != 0) + (spin_up != ne));
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
        std::vector<double> cpu(output_size, -.125), old(cpu), observed(cpu), determinant_observed(cpu);
        old_output.upload(old.data(), old.size());new_output.upload(observed.data(), observed.size());
        determinant_output.upload(determinant_observed.data(), determinant_observed.size());
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
        bank.project(blas, queue.getNative(), layout, x.data(), wu.data(), wd.data(), determinant_output.data());
        bank.projectAllConfigurations(blas, queue.getNative(), layout, x.data(), wu.data(), wd.data(), new_output.data());
        queue.sync();
        const auto after = metrics.snapshot();
        if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
            before.upload_calls != after.upload_calls)
          throw std::runtime_error("Warm determinant batching allocated or uploaded pointer storage");
        old_output.download(old.data(), old.size());new_output.download(observed.data(), observed.size());
        determinant_output.download(determinant_observed.data(), determinant_observed.size());
        for (std::size_t i = 0; i < observed.size(); ++i)
        {
          const auto ce = std::abs(observed[i] - cpu[i]), oe = std::abs(observed[i] - old[i]);
          const auto de = std::abs(determinant_observed[i] - cpu[i]);
          max_cpu = std::max(max_cpu, ce);max_old = std::max(max_old, oe);
          if (!std::isfinite(observed[i]) || ce > 2e-12 * (1 + std::abs(cpu[i])) ||
              oe > 2e-12 * (1 + std::abs(old[i])) || !std::isfinite(determinant_observed[i]) ||
              de > 2e-12 * (1 + std::abs(cpu[i])))
            throw std::runtime_error("Determinant batching changed logical projection or padding");
          ++comparisons;
        }
      }
    }
  std::cout << "PASS determinant-batched orbital projections comparisons=" << comparisons
            << " max_cpu=" << max_cpu << " max_old_gpu=" << max_old
            << " warm_allocations=0 warm_uploads=0 B_tail=1 target_tail=1 empty_spins=1\n";
  validateBatchedHeadOwner();
  validateBatchedHeadOwner(8, 16);
  validateBatchedHeadOwner(44, 16);
  validateBatchedHeadOwner(44, 64);
  validateBatchedHeadFullOwner();
}
} // namespace qmcplusplus::psiformer::device
#endif
