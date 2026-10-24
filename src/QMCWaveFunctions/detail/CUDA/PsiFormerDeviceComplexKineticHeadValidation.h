//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_HEAD_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_HEAD_VALIDATION_H
#include "PsiFormerDeviceComplexProducts.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace qmcplusplus::psiformer::device
{
/// Compare complete destinations, including failed prefixes and padding, with the unchanged scalar kernel.
inline void validateComplexKineticHeadParallel()
{
  compute::Queue<PlatformKind::CUDA> queue;
  const auto cell = periodic_geometry::makeCell({{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}}, {true, true, true});
  std::size_t comparisons = 0;
  for (const auto dimensions : {std::array<std::size_t, 2>{63, 2}, {64, 2}, {100, 2}, {64, 17}})
    for (std::size_t k : {std::size_t(1), std::size_t(3)})
    {
      const auto n = dimensions[0], d = dimensions[1];
      auto layout = makeOpenOrbitalJetLayout(1, d, n, 2, 3, n / 2, SpatialJetMode::FULL_VGL, 5);
      layout.features = withSpatialTargetRange(layout.features, n - k, k);
      layout.orbitals = withSpatialTargetRange(layout.orbitals, n - k, k);
      const auto w = openOrbitalBackflowSpan(layout), e = openOrbitalEnvelopeSpan(layout);
      const auto pcount = 4 * w + 4 * e + 7;
      const ComplexKineticHeadLayout head{{layout, pcount, 0, w, 4 * w, 4 * w + e,
          4 * w + 2 * e, 4 * w + 3 * e}, 2 * w, 3 * w};
      const auto fcount = layout.orbitals.plane_count * layout.features.element_count;
      const auto hcount = layout.orbitals.plane_count * layout.orbitals.element_count;
      std::vector<double> positions(3 * n), nuclei{0.2, 0.3, 0.4, 1.2, 1.3, 1.4};
      for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 2.0 + 0.003 * i;
      std::vector<double> parameters(pcount), features(fcount), envelope(hcount), real(hcount), imag(hcount);
      std::vector<cuDoubleComplex> roots(hcount);
      for (std::size_t i = 0; i < pcount; ++i) parameters[i] = 0.01 * double(int(i % 13) - 6);
      for (std::size_t i = 0; i < fcount; ++i) features[i] = 0.02 * double(int(i % 11) - 5);
      for (std::size_t i = 0; i < hcount; ++i)
      {
        envelope[i] = 0.03 * double(int(i % 7) - 3);
        real[i] = 0.01 * double(int(i % 5) - 2);
        imag[i] = 0.02 * double(int(i % 3) - 1);
        roots[i] = make_cuDoubleComplex(0.001 * double(int(i % 7) - 3), 0.002 * double(int(i % 5) - 2));
      }
      DeviceBuffer<double> dp(3 * n), dn(6), weights(pcount), inputs(fcount), env(hcount), re(hcount), im(hcount);
      DeviceBuffer<cuDoubleComplex> q(hcount);
      DeviceBuffer<double> old_p(pcount), new_p(pcount), old_f(fcount), new_f(fcount), scratch_p(pcount), scratch_f(fcount);
      DeviceBuffer<std::uint64_t> old_status(1), new_status(1), private_status(1);
      int device = -1;
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&device), "parallel kinetic head device");
#else
      checkRuntime(cudaGetDevice(&device), "parallel kinetic head device");
#endif
      auto& metrics = deviceMemoryMetrics(device);
      for (std::size_t selected = 0; selected < 2; ++selected)
        for (int variant = 0; variant < 10; ++variant)
        {
          auto p = parameters, f = features, en = envelope, x = positions, r = real;
          auto root = roots;
          std::vector<double> initial_p(pcount, -0.0), initial_f(fcount, 0.125);
          const auto nan = std::numeric_limits<double>::quiet_NaN();
          const auto inf = std::numeric_limits<double>::infinity();
          if (variant == 1) std::fill(en.begin(), en.end(), 0.0);
          if (variant == 2)
            for (std::size_t i = 4 * w + 2 * e; i < 4 * w + 4 * e; ++i) p[i] = 0.0;
          if (variant == 3) root.back() = make_cuDoubleComplex(nan, 0);
          if (variant == 4) { x[0] = nuclei[0]; x[1] = nuclei[1]; x[2] = nuclei[2]; }
          if (variant == 5) initial_p[w / 2] = inf;
          if (variant == 6) f[0] = nan;
          if (variant == 7) r.back() = inf;
          if (variant == 8)
          {
            std::fill(root.begin(), root.end(), make_cuDoubleComplex(std::numeric_limits<double>::max(), 0));
            std::fill(en.begin(), en.end(), 2.0); // Finite inputs analytically overflow the unweighted product.
          }
          if (variant == 9) std::fill(en.begin(), en.end(), std::numeric_limits<double>::denorm_min());
          dp.upload(x.data(), x.size()); dn.upload(nuclei.data(), nuclei.size()); weights.upload(p.data(), p.size());
          inputs.upload(f.data(), f.size()); env.upload(en.data(), en.size()); re.upload(r.data(), r.size());
          im.upload(imag.data(), imag.size()); q.upload(root.data(), root.size());
          old_p.upload(initial_p.data(), pcount); new_p.upload(initial_p.data(), pcount);
          old_f.upload(initial_f.data(), fcount); new_f.upload(initial_f.data(), fcount);
          // Two representative cases preserve a pre-existing public failure word on success/replay.
          const bool incoming_status = n == 64 && d == 2 && k == 1 && selected == 0 && (variant == 0 || variant == 3);
          const auto clear = incoming_status ? std::uint64_t(7) : std::numeric_limits<std::uint64_t>::max();
          old_status.upload(&clear, 1); new_status.upload(&clear, 1);
          const auto before = metrics.snapshot();
          checkRuntime(launchComplexKineticHeadReverse(queue.getNative(), head, cell, dp.data(), dn.data(), weights.data(),
              inputs.data(), env.data(), re.data(), im.data(), q.data(), selected, old_f.data(), old_p.data(), old_status.data()),
              "legacy kinetic head oracle");
          checkRuntime(launchComplexKineticHeadReverseParallel(queue.getNative(), head, cell, dp.data(), dn.data(), weights.data(),
              inputs.data(), env.data(), re.data(), im.data(), q.data(), selected, new_f.data(), new_p.data(), new_status.data(),
              scratch_f.data(), fcount, scratch_p.data(), pcount, private_status.data()), "parallel kinetic head candidate");
          queue.sync();
          const auto after = metrics.snapshot();
          if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls)
            throw std::runtime_error("Parallel kinetic head allocated warm storage");
          std::vector<double> a(pcount), b(pcount), fa(fcount), fb(fcount);
          std::uint64_t sa, sb;
          old_p.download(a.data(), pcount); new_p.download(b.data(), pcount);
          old_f.download(fa.data(), fcount); new_f.download(fb.data(), fcount);
          old_status.download(&sa, 1); new_status.download(&sb, 1);
          if (sa != sb || std::memcmp(a.data(), b.data(), 8 * pcount) || std::memcmp(fa.data(), fb.data(), 8 * fcount))
            throw std::runtime_error("Parallel kinetic head changed complete output/failure-prefix bits: Ne=" +
                std::to_string(n) + " K=" + std::to_string(k) + " component=" + std::to_string(selected) +
                " variant=" + std::to_string(variant));
          if ((variant < 3 || variant == 9) && sa != clear)
            throw std::runtime_error("Parallel kinetic head valid fixture was rejected");
          if (variant >= 3 && variant <= 8 && sa == clear)
            throw std::runtime_error("Parallel kinetic head invalid fixture did not fail");
          ++comparisons;
        }
    }
  const auto plan = makeTargetCostPlan(100, true, true);
  for (std::size_t k : {std::size_t(1), std::size_t(4)})
  {
    const auto off = estimateComplexKineticCost(plan, k, false), on = estimateComplexKineticCost(plan, k, true);
    const auto expected = 8 * (plan.parameterCount() + (1 + 4 * k) * 100 * plan.modelShape().feature_dimension) + 8;
    if (on.device_bytes - off.device_bytes != expected || on.host_bytes != off.host_bytes ||
        on.matrix_work != off.matrix_work || off.private_features || off.private_parameters || !on.parallel_head)
      throw std::runtime_error("Parallel kinetic head exact private cost differs");
  }
  std::cout << "PASS parallel kinetic head comparisons=" << comparisons
            << " complete_response_feature_status_bits=exact failed_prefix=exact padded_features=exact warm_allocations=0\n";
}

/// Exercise complete dK/dP ownership, exact preallocation rejection, warm reuse and same-family rebind.
inline void validateComplexKineticHeadOwner(std::size_t targets = 3, bool qualify_factor = false)
{
  const auto metadata = makeTargetCostPlan(64, true, true);
  // The versioned initializer remains fixed at its production architecture.
  // Reduce its canonical layouts deterministically, as the existing small numerical fixtures do.
  // Constant pi/zeta makes the envelope common to orbital columns, so F>=32 is
  // necessary to avoid a structural rank deficit in each 32-electron spin block.
  const ModelShape shape{32, 32, 2, 2, 32, 2, 1}, production_shape{32, 32, 2, 16, 256, 4, 4};
  const auto initialized = initializePsiFormerParameters(production_shape, 1729, PSIFORMER_COMPLEX_PERIODIC_V1, metadata.environment());
  const PsiFormerExecutionPlan production_plan(production_shape, initialized.layouts, metadata.environment());
  std::vector<ParameterLayoutInput> layouts;
  std::vector<double> parameters;
  for (const auto& tensor : production_plan.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= shape.attention_blocks) continue;
    auto dimensions = tensor.shape;
    std::size_t count = 1;
    for (auto& extent : dimensions)
    {
      if (extent == production_shape.feature_dimension) extent = shape.feature_dimension;
      else if (extent == production_shape.determinants * shape.electrons()) extent = shape.determinants * shape.electrons();
      count = spatial_detail::checkedProduct(count, extent, "Small kinetic fixture tensor overflow");
    }
    const auto begin = parameters.size();
    // Match the established small fixture's width-variance rule; physical scalar/envelope
    // constants and embedding input scaling remain exactly the production values.
    const bool rescale = tensor.role != ParameterRole::ELECTRON_EMBEDDING_WEIGHT &&
        (tensor.shape.size() == 1 || (tensor.shape.size() == 2 && tensor.shape[0] == production_shape.feature_dimension));
    const double scale = rescale ? std::sqrt(double(production_shape.feature_dimension) / shape.feature_dimension) : 1.0;
    for (std::size_t index = 0; index < count; ++index)
      parameters.push_back(initialized.values[tensor.begin + index] * scale);
    layouts.push_back({tensor.module, tensor.name, dimensions, begin, parameters.size()});
  }
  const PsiFormerExecutionPlan plan(shape, layouts, metadata.environment());
  std::vector<double> positions(192), nuclei{0.2, 0.3, 0.4, 1.2, 1.3, 1.4}, masses(64, 1.0);
  // Fixed rational low-discrepancy coordinates span three dimensions inside the cell.
  // This is a unit-only fixture, not a seed search or a production physics/work change.
  for (std::size_t i = 0; i < 64; ++i)
  {
    positions[3 * i] = 8 * (double((37 * i) % 67) + 0.5) / 67;
    positions[3 * i + 1] = 8 * (double((53 * i) % 71) + 0.5) / 71;
    positions[3 * i + 2] = 8 * (double((29 * i) % 73) + 0.5) / 73;
  }
  const std::array<double, 3> kappa{0.13, -0.07, 0.04};
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters, nuclei);
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "parallel kinetic owner device");
#else
  checkRuntime(cudaGetDevice(&device), "parallel kinetic owner device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  const auto off = estimateComplexKineticCost(plan, targets, qualify_factor, false);
  const auto on = estimateComplexKineticCost(plan, targets, true, qualify_factor);
  if (qualify_factor)
  {
    if (off.device_bytes != on.device_bytes || off.host_bytes != on.host_bytes ||
        off.canonical_bytes != on.canonical_bytes || off.matrix_work != on.matrix_work ||
        off.cooperative_factor || !on.cooperative_factor)
      throw std::runtime_error("Kinetic factor composition changed its exact zero-storage cost");
    const auto selected = estimateComplexKineticCost(plan, targets, true);
    auto selected_owner = PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas,
        selected.device_bytes, selected.matrix_work, true);
    if (selected_owner->usesCooperativeFactorization() != configuredComplexKineticCooperativeLu())
      throw std::runtime_error("Kinetic factor cold selection did not reach the actual owner");
    for (const auto n : {63, 64, 300, 301})
      if (complexKineticCooperativeLuEligible(makeTargetCostPlan(n, true, true)) != (n >= 64 && n <= 300))
        throw std::runtime_error("Kinetic factor electron range differs from qualification");
    if (complexKineticCooperativeLuEligible(makeTargetCostPlan(64, false)) ||
        complexKineticCooperativeLuEligible(makeTargetCostPlan(64, true)))
      throw std::runtime_error("Kinetic factor admitted open/real metadata");
    const auto layouts_for = [&](std::size_t determinants) {
      std::vector<ParameterLayoutInput> result;
      std::size_t begin = 0;
      for (const auto& tensor : plan.parameterTensors())
      {
        auto dimensions = tensor.shape;
        if (tensor.role == ParameterRole::ENVELOPE_PI_UP || tensor.role == ParameterRole::ENVELOPE_PI_DOWN ||
            tensor.role == ParameterRole::ENVELOPE_ZETA_UP || tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN)
          dimensions[0] = determinants * 64;
        if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
          dimensions[1] = determinants * 64;
        std::size_t count = 1;
        for (auto dimension : dimensions)
          count = spatial_detail::checkedProduct(count, dimension, "Kinetic factor metadata test overflow");
        result.push_back({tensor.module, tensor.name, dimensions, begin, begin + count});
        begin += count;
      }
      return result;
    };
    auto d17 = plan.modelShape();
    d17.determinants = 17;
    if (complexKineticCooperativeLuEligible(PsiFormerExecutionPlan(d17, layouts_for(17), plan.environment())))
      throw std::runtime_error("Kinetic factor admitted D17 metadata");
    for (bool partial_axes : {false, true})
    {
      auto environment = plan.environment();
      if (partial_axes)
        environment.periodic_axes[2] = false;
      else
        environment.fixed_nuclei = false;
      try
      {
        const PsiFormerExecutionPlan excluded(plan.modelShape(), layouts_for(shape.determinants), environment);
        if (complexKineticCooperativeLuEligible(excluded))
          throw std::runtime_error("Kinetic factor admitted moving nuclei/partial periodic axes");
      }
      catch (const std::invalid_argument&) {}
    }
  }
  for (const auto cost : {off, on})
  {
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { (void)PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas, cost.device_bytes - 1,
        cost.matrix_work, cost.parallel_head, cost.cooperative_factor); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected || metrics.snapshot().allocation_calls != before.allocation_calls)
      throw std::runtime_error("Parallel kinetic owner byte-minus-one allocated");
  }
  // Enclosing COMPLETE admission must include exactly the selected kinetic scratch before any allocation.
  const auto outer = estimateComplexProductsCost(plan, targets, 1, 1, plan.parameterCount());
  const auto before_outer = metrics.snapshot();
  bool outer_rejected = false;
  try { (void)PreparedDeviceComplexProducts::prepare(image, targets, 1, 1, plan.parameterCount(), queue, blas,
      outer.device_bytes - 1, outer.host_bytes, outer.matrix_work); }
  catch (const std::length_error&) { outer_rejected = true; }
  if (!outer_rejected || metrics.snapshot().allocation_calls != before_outer.allocation_calls)
    throw std::runtime_error("Parallel kinetic enclosing admission allocated below exact cost");
  const auto selective = estimateComplexProductsCost(plan, targets, 1, 1, plan.parameterCount(),
      ComplexProductsHostOutput::RETAINED, ComplexProductsScope::SCORE_ECP_ONLY);
  if (selective.kinetic.device_bytes || selective.kinetic.private_features || selective.kinetic.private_parameters ||
      selective.kinetic.cooperative_factor)
    throw std::runtime_error("Selective score owner charged kinetic scratch");
  const auto before = metrics.snapshot();
  auto scalar = PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas,
      off.device_bytes, off.matrix_work, off.parallel_head, false);
  auto parallel = PreparedDeviceComplexKinetic::prepare(image, targets, queue, blas,
      on.device_bytes, on.matrix_work, on.parallel_head, on.cooperative_factor);
  if (scalar->usesCooperativeFactorization() || parallel->usesCooperativeFactorization() != qualify_factor)
    throw std::runtime_error("Kinetic owner did not retain its explicit factor policies");
  if (metrics.snapshot().live_bytes - before.live_bytes != off.device_bytes + on.device_bytes)
    throw std::runtime_error("Parallel kinetic actual owner bytes differ");
  const auto compare = [&] {
    std::vector<cuDoubleComplex> a(plan.parameterCount()), b(plan.parameterCount());
    cuDoubleComplex ka, kb;
    ComplexValueResult aa, ab;
    const auto warm = metrics.snapshot();
    // Flushed test-only boundaries distinguish the unchanged scalar owner from the candidate.
    std::cout << "DIAGNOSTIC kinetic owner scalar evaluate begin" << std::endl;
    try { scalar->evaluate(positions, masses, kappa); }
    catch (const std::runtime_error& error)
    { throw std::runtime_error(std::string("Kinetic scalar owner evaluate: ") + error.what()); }
    std::cout << "DIAGNOSTIC kinetic owner scalar evaluate complete; parallel evaluate begin" << std::endl;
    try { parallel->evaluate(positions, masses, kappa); }
    catch (const std::runtime_error& error)
    { throw std::runtime_error(std::string("Kinetic parallel owner evaluate: ") + error.what()); }
    std::cout << "DIAGNOSTIC kinetic owner parallel evaluate complete" << std::endl;
    scalar->download(scalar->resultView(), a, ka, aa); parallel->download(parallel->resultView(), b, kb, ab);
    const auto after = metrics.snapshot();
    if (warm.allocation_calls != after.allocation_calls || warm.free_calls != after.free_calls ||
        std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) || std::memcmp(&ka, &kb, sizeof(ka)) ||
        std::memcmp(&aa, &ab, sizeof(aa)))
      throw std::runtime_error("Parallel kinetic complete owner changed dK/dP, energy, amplitude or warm allocation");
  };
  compare();
  const auto stale = parallel->resultView();
  auto values = parameters;
  values[plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::REAL).begin] += 0.015625;
  auto successor = image->makeUpdatedImage(values, 1);
  scalar->rebindModel(successor); parallel->rebindModel(successor);
  std::vector<cuDoubleComplex> discarded(plan.parameterCount());
  cuDoubleComplex energy; ComplexValueResult amplitude;
  bool stale_rejected = false;
  try { parallel->download(stale, discarded, energy, amplitude); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  if (!stale_rejected) throw std::runtime_error("Parallel kinetic rebind preserved stale authority");
  compare();
  if (qualify_factor)
  {
    auto invalid = positions;
    invalid.back() = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { parallel->evaluate(invalid, masses, kappa); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Kinetic factor accepted a nonfinite request");
    rejected = false;
    try { (void)parallel->resultView(); }
    catch (const std::logic_error&) { rejected = true; }
    if (!rejected)
      throw std::runtime_error("Kinetic factor failed request retained publication authority");
    // Zero BOTH components of every up-spin projection: each determinant has
    // exact zero rows, forcing canonical singular-factor rejection before roots.
    for (auto component : {HeadTensorComponent::REAL, HeadTensorComponent::IMAG})
    {
      const auto& interval = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, component);
      std::fill(values.begin() + interval.begin, values.begin() + interval.end, 0.0);
    }
    auto singular = successor->makeUpdatedImage(values, 2);
    scalar->rebindModel(singular);
    parallel->rebindModel(singular);
    std::string canonical_failure;
    for (auto* owner : {scalar.get(), parallel.get()})
    {
      bool failed = false;
      try { owner->evaluate(positions, masses, kappa); }
      catch (const std::runtime_error& error)
      {
        if (canonical_failure.empty())
          canonical_failure = error.what();
        else if (canonical_failure != error.what())
          throw std::runtime_error("Kinetic factor changed canonical singular failure status");
        failed = true;
      }
      if (!failed || canonical_failure.find("finite/inverse admission failed") == std::string::npos)
        throw std::runtime_error("Kinetic factor did not reject a singular inverse canonically");
      failed = false;
      try { (void)owner->resultView(); }
      catch (const std::logic_error&) { failed = true; }
      if (!failed)
        throw std::runtime_error("Kinetic singular factor published a partial derivative");
    }
    std::cout << "PASS kinetic_factor_owner Ne=64 D=2 F=32 K=" << targets
              << " complete_dK_dP_energy_amplitude_bits=exact changed_weight_rebind=exact stale_rejected=1"
                 " singular_status=canonical nonfinite_publication=rejected warm_allocations=0 minus_one=1"
                 " extra_device_bytes=0 selected=" << configuredComplexKineticCooperativeLu() << '\n';
    return;
  }
  std::cout << "PASS parallel kinetic owner Ne=64 D=2 F=32 K=" << targets << " complete_dK_dP_energy_amplitude_bits=exact"
               " same_family_changed_weight_rebind=exact stale_view=rejected warm_allocations=0 byte_minus_one=preallocation\n";
}

/// Measure only the complete leaf pipeline, including private copy and publish kernels, not a training update.
inline void measureComplexKineticHeadParallel()
{
  compute::Queue<PlatformKind::CUDA> queue;
#ifdef QMC_CUDA2HIP
  using Event = hipEvent_t;
  const auto create = [](Event* e) { return hipEventCreate(e); };
  const auto record = [](Event e, Stream s) { return hipEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return hipEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return hipEventDestroy(e); };
#else
  using Event = cudaEvent_t;
  const auto create = [](Event* e) { return cudaEventCreate(e); };
  const auto record = [](Event e, Stream s) { return cudaEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return cudaEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return cudaEventDestroy(e); };
#endif
  Event begin, end;
  checkRuntime(create(&begin), "kinetic head timing begin");
  checkRuntime(create(&end), "kinetic head timing end");
  const auto cell = periodic_geometry::makeCell({{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}}, {true, true, true});
  for (const auto dimensions : {std::array<std::size_t, 4>{64, 2, 8, 2}, {100, 16, 256, 20}})
  {
    const auto n = dimensions[0], d = dimensions[1], width = dimensions[2], ni = dimensions[3];
    auto layout = makeOpenOrbitalJetLayout(1, d, n, ni, width, n / 2, SpatialJetMode::FULL_VGL);
    layout.features = withSpatialTargetRange(layout.features, n - 1, 1);
    layout.orbitals = withSpatialTargetRange(layout.orbitals, n - 1, 1);
    const auto w = openOrbitalBackflowSpan(layout), e = openOrbitalEnvelopeSpan(layout);
    const auto pcount = 4 * (w + e), fcount = 5 * layout.features.element_count, hcount = 5 * layout.orbitals.element_count;
    const ComplexKineticHeadLayout head{{layout, pcount, 0, w, 4 * w, 4 * w + e,
        4 * w + 2 * e, 4 * w + 3 * e}, 2 * w, 3 * w};
    std::vector<double> positions(3 * n), nuclei(3 * ni), p(pcount, 0.01), f(fcount, 0.02), en(hcount, 0.03);
    std::vector<double> re(hcount, 0.01), im(hcount, -0.01), initial_p(pcount, 0.0), initial_f(fcount, 0.0);
    std::vector<cuDoubleComplex> roots(hcount, make_cuDoubleComplex(0.001, -0.002));
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 2 + 0.003 * i;
    for (std::size_t i = 0; i < nuclei.size(); ++i) nuclei[i] = 0.1 + 0.004 * i;
    DeviceBuffer<double> dp(3 * n), dn(3 * ni), weights(pcount), inputs(fcount), env(hcount), real(hcount), imag(hcount);
    DeviceBuffer<cuDoubleComplex> q(hcount);
    DeviceBuffer<double> response(pcount), feature(fcount), private_p(pcount), private_f(fcount);
    DeviceBuffer<std::uint64_t> status(1), failure(1);
    dp.upload(positions.data(), positions.size()); dn.upload(nuclei.data(), nuclei.size()); weights.upload(p.data(), p.size());
    inputs.upload(f.data(), f.size()); env.upload(en.data(), en.size()); real.upload(re.data(), re.size());
    imag.upload(im.data(), im.size()); q.upload(roots.data(), roots.size());
    for (int repetition = 0; repetition < 4; ++repetition)
      for (bool parallel : {false, true})
      {
        const auto clear = std::numeric_limits<std::uint64_t>::max();
        status.upload(&clear, 1); response.upload(initial_p.data(), pcount); feature.upload(initial_f.data(), fcount);
        checkRuntime(record(begin, queue.getNative()), "kinetic head timing record");
        if (parallel)
          checkRuntime(launchComplexKineticHeadReverseParallel(queue.getNative(), head, cell, dp.data(), dn.data(), weights.data(),
              inputs.data(), env.data(), real.data(), imag.data(), q.data(), 0, feature.data(), response.data(), status.data(),
              private_f.data(), fcount, private_p.data(), pcount, failure.data()), "timed parallel kinetic head");
        else
          checkRuntime(launchComplexKineticHeadReverse(queue.getNative(), head, cell, dp.data(), dn.data(), weights.data(),
              inputs.data(), env.data(), real.data(), imag.data(), q.data(), 0, feature.data(), response.data(), status.data()),
              "timed scalar kinetic head");
        checkRuntime(record(end, queue.getNative()), "kinetic head timing record");
        queue.sync();
        float ms = 0;
        checkRuntime(elapsed(&ms, begin, end), "kinetic head GPU elapsed");
        std::uint64_t code; status.download(&code, 1);
        if (code != clear) throw std::runtime_error("Timed kinetic head failed admission");
        std::cout << std::setprecision(9) << "MEASURE kinetic_head Ne=" << n << " D=" << d << " F=" << width
                  << " Ni=" << ni << " K=1 parallel=" << parallel << " repetition=" << repetition
                  << " measured=" << (repetition != 0) << " gpu_ms=" << ms
                  << " scope=complete_leaf_pipeline uploads_excluded=1 training_fraction=unmeasured\n";
      }
  }
  checkRuntime(destroy(begin), "kinetic head timing destroy");
  checkRuntime(destroy(end), "kinetic head timing destroy");
}
} // namespace qmcplusplus::psiformer::device
#endif
