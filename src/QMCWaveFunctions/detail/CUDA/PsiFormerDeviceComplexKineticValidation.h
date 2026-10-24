//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_VALIDATION_H
#include "PsiFormerDeviceComplexKineticKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexOrbital.h"
#include <array>
#include <cstdlib>
#include <string_view>
#include <type_traits>

/// Compare channel ownership with the existing cooperative leaf, including canonical failure prefixes.
inline void validateComplexKineticChannelRoots()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  const char* cooperative = std::getenv("PSIFORMER_GPU_COOPERATIVE_COMPLEX_KINETIC_ROOTS");
  if (!cooperative || std::string_view(cooperative) != "1")
    throw std::invalid_argument("Channel root oracle requires explicit cooperative matrix roots=1");
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  std::size_t comparisons = 0;
  // Cheap matrix-only cases, not a large-N neural architecture cross-product.
  for (const auto descriptor : {std::array<std::size_t, 3>{64, 2, 3}, {100, 16, 1}})
  {
    const auto n = descriptor[0], d = descriptor[1], k = descriptor[2], nn = n * n, m = d * nn;
    auto layout = makeOpenOrbitalJetLayout(1, d, n, 1, 3, n / 2, SpatialJetMode::FULL_VGL);
    layout.features = withSpatialTargetRange(layout.features, n - k, k);
    layout.orbitals = withSpatialTargetRange(layout.orbitals, n - k, k);
    const auto elements = layout.orbitals.plane_count * m;
    std::vector<cuDoubleComplex> inverse(m), head(elements), gradient(3 * k), laplacian(k), qg(3 * k), ql(k);
    std::vector<cuDoubleComplex> initial(elements), a(elements), b(elements);
    std::vector<ComplexValueChannel> records(d);
    DeviceBuffer<cuDoubleComplex> di(m), dh(elements), dg(3 * k), dl(k), dqg(3 * k), dql(k);
    DeviceBuffer<cuDoubleComplex> old_out(elements), new_out(elements), old_work(3 * nn), new_work(3 * m);
    DeviceBuffer<ComplexValueChannel> dc(d);
    DeviceBuffer<ComplexValueResult> amplitude(1);
    DeviceBuffer<std::uint64_t> old_status(1), new_status(1), private_failure(1);
    for (std::size_t variant = 0; variant < 13; ++variant)
    {
      std::fill(inverse.begin(), inverse.end(), make_cuDoubleComplex(0, 0));
      std::fill(head.begin(), head.end(), make_cuDoubleComplex(0, 0));
      std::fill(gradient.begin(), gradient.end(), make_cuDoubleComplex(0, 0));
      std::fill(laplacian.begin(), laplacian.end(), make_cuDoubleComplex(0, 0));
      for (std::size_t channel = 0; channel < d; ++channel)
      {
        records[channel] = {make_cuDoubleComplex(1, 0), 0, 0};
        for (std::size_t i = 0; i < n; ++i)
        {
          inverse[channel * nn + i * n + i] = make_cuDoubleComplex(1, 0);
          for (std::size_t plane = 1; plane < layout.orbitals.plane_count; ++plane)
            head[plane * m + channel * nn + i * n + i] =
                make_cuDoubleComplex(0.001 * (i + 1), variant == 1 ? -0.0003 * (plane + 1) : 0);
        }
      }
      std::fill(qg.begin(), qg.end(), make_cuDoubleComplex(0.25, -0.125));
      std::fill(ql.begin(), ql.end(), make_cuDoubleComplex(-0.5, 0.125));
      ComplexValueResult result{make_cuDoubleComplex(1, 0), std::log(double(d)), 1, 0, 0};
      if (variant == 2)
        records.back().log_abs = -740;
      if (variant == 3)
        result.phase = make_cuDoubleComplex(0, 0);
      if (variant == 4)
        for (std::size_t channel = 0; channel < d; ++channel)
          records[channel].phase = make_cuDoubleComplex(channel % 2 ? -1 : 1, 0);
      if (variant == 5)
        inverse.back() = make_cuDoubleComplex(std::numeric_limits<double>::quiet_NaN(), 0);
      if (variant == 6 || variant == 8 || variant == 11)
      {
        const auto plane = variant == 8 ? 2 : variant == 11 ? 3 : 1;
        head[plane * m + (d - 1) * nn] = make_cuDoubleComplex(1e200, 0);
      }
      if (variant == 12)
        records.back().status = 1;
      const std::uint64_t incoming = variant == 10 || variant == 11 ? 0 : variant == 7 || variant == 8 ? 7 : clear;
      for (std::size_t i = 0; i < elements; ++i)
        initial[i] = make_cuDoubleComplex(i % 2 ? -0.0 : 0.125, -0.25);
      di.upload(inverse.data(), m); dh.upload(head.data(), elements);
      dg.upload(gradient.data(), gradient.size()); dl.upload(laplacian.data(), k);
      dqg.upload(qg.data(), qg.size()); dql.upload(ql.data(), k);
      dc.upload(records.data(), d); amplitude.upload(&result, 1);
      old_out.upload(initial.data(), elements); new_out.upload(initial.data(), elements);
      old_status.upload(&incoming, 1); new_status.upload(&incoming, 1);
      checkRuntime(launchComplexKineticMatrixRoots(queue.getNative(), layout, dc.data(), amplitude.data(), di.data(),
          dh.data(), dg.data(), dl.data(), dqg.data(), dql.data(), old_out.data(), old_work.data(), old_status.data()),
          "canonical cooperative root oracle");
      checkRuntime(launchComplexKineticChannelRoots(queue.getNative(), layout, dc.data(), amplitude.data(), di.data(),
          dh.data(), dg.data(), dl.data(), dqg.data(), dql.data(), new_out.data(), new_work.data(), new_work.capacity(),
          new_status.data(), private_failure.data()), "channel root candidate");
      queue.sync();
      std::uint64_t sa, sb;
      old_status.download(&sa, 1); new_status.download(&sb, 1);
      old_out.download(a.data(), elements); new_out.download(b.data(), elements);
      if (sa != sb || std::memcmp(a.data(), b.data(), elements * sizeof(a[0])))
        throw std::runtime_error("Channel roots changed output/status/failure-prefix bits variant=" + std::to_string(variant));
      if ((variant == 0 || variant == 1 || variant == 7 || variant == 9 || variant == 10) && sa != incoming)
        throw std::runtime_error("Channel roots valid fixture rejected");
      if (((variant >= 3 && variant <= 6) || variant == 8 || variant == 11 || variant == 12) &&
          sa != std::min(incoming, std::uint64_t(1)))
        throw std::runtime_error("Channel roots failure fixture did not reject canonically");
      const auto unchanged = [&](auto& buffer, const auto& expected) {
        using T = typename std::decay_t<decltype(expected)>::value_type;
        std::vector<T> actual(expected.size());
        buffer.download(actual.data(), actual.size());
        if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)))
          throw std::runtime_error("Channel roots mutated an immutable input");
      };
      unchanged(di, inverse); unchanged(dh, head); unchanged(dg, gradient); unchanged(dl, laplacian);
      unchanged(dqg, qg); unchanged(dql, ql); unchanged(dc, records);
      ComplexValueResult result_after;
      amplitude.download(&result_after, 1);
      if (std::memcmp(&result_after, &result, sizeof(result)))
        throw std::runtime_error("Channel roots mutated amplitude metadata");
      ++comparisons;
    }
    // All invalid calls stop before memset/launch, including partial byte-span aliases.
    const auto invoke = [&](OpenOrbitalJetLayout selected, cuDoubleComplex* work, std::size_t capacity,
                            std::uint64_t* failure) {
      return launchComplexKineticChannelRoots(queue.getNative(), selected, dc.data(), amplitude.data(), di.data(),
          dh.data(), dg.data(), dl.data(), dqg.data(), dql.data(), new_out.data(), work, capacity,
          new_status.data(), failure);
    };
    for (std::size_t guard = 0; guard < 4; ++guard)
    {
      bool rejected = false;
      try
      {
        (void)invoke(layout, guard == 1 ? nullptr : guard == 2 ? di.data() + 1 : new_work.data(),
            guard == 0 ? new_work.capacity() - 1 : new_work.capacity(),
            guard == 3 ? new_status.data() : private_failure.data());
      }
      catch (const std::logic_error&) { rejected = true; }
      if (!rejected)
        throw std::runtime_error("Channel roots invalid scratch/alias admission launched");
    }
    for (const auto unsupported_shape : {std::array<std::size_t, 2>{63, d}, {301, d}, {n, 1}, {n, 17}})
    {
      const auto unsupported = makeOpenOrbitalJetLayout(1, unsupported_shape[1], unsupported_shape[0], 1, 3,
          unsupported_shape[0] / 2, SpatialJetMode::FULL_VGL);
      bool rejected = false;
      try { (void)invoke(unsupported, new_work.data(), new_work.capacity(), private_failure.data()); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected)
        throw std::runtime_error("Channel root primitive admitted unsupported Ne/D");
    }
  }
  std::cout << "PASS kinetic_channel_roots comparisons=" << comparisons
            << " complete_output_status_bits=exact failure_prefix=exact inputs_unchanged=1 private_reset=per_call guards=8\n";
}

/// Independent CPU Wide root oracle and paired-head FD, never a numerical GPU fallback.
void validateComplexKineticLeaves()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  using Wide = std::complex<long double>;
  const auto pack = [](C z) { return make_cuDoubleComplex(z.real(), z.imag()); };
  const auto host = [](cuDoubleComplex z) { return C(cuCreal(z), cuCimag(z)); };
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "complex kinetic leaf metric device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "complex kinetic leaf metric device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  const auto clear = std::numeric_limits<std::uint64_t>::max();
  double max_root = 0, max_fd = 0, max_root_fd = 0, max_tiny = 0;
  std::size_t root_cases = 0, head_lanes = 0;
  const auto root_case = [&](std::size_t n, std::size_t begin, std::size_t k, bool complex, int special) {
    constexpr std::size_t d = 2;
    const auto nn = n * n, m = d * nn, full_s = 1 + 4 * n, s = 1 + 4 * k;
    auto layout = makeOpenOrbitalJetLayout(1, d, n, 1, 3, n / 2, SpatialJetMode::FULL_VGL);
    layout.features = withSpatialTargetRange(layout.features, begin, k);
    layout.orbitals = withSpatialTargetRange(layout.orbitals, begin, k);
    std::vector<C> jets(full_s * m), g(3 * n), l(n), ratio(n), qg(3 * n), ql(n), reference(full_s * m);
    for (std::size_t channel = 0; channel < d; ++channel)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          jets[channel * nn + i * n + j] = C(i == j ? 1.2 + 0.1 * channel + 0.03 * i : 0.02 * std::sin(double(i + j + 1)),
              complex ? 0.07 * std::cos(double(channel * nn + i * n + j + 1)) : 0);
    for (std::size_t i = m; i < jets.size(); ++i)
      jets[i] = C(0.02 * std::sin(double(i + 1)), complex ? 0.013 * std::cos(double(i + 2)) : 0);
    if (special)
    {
      std::fill(jets.begin(), jets.end(), C{});
      jets[0] = jets[3] = 1;
      jets[4] = jets[7] = 1e-200;
      if (special == 1) jets[m + 4] = 1e-50; // A fits, B/C do not, although CPU Wide weighted terms fit.
    }
    for (std::size_t e = begin; e < begin + k; ++e)
    {
      ql[e] = C(-0.4 + 0.03 * e, complex ? 0.11 : 0);
      for (std::size_t mu = 0; mu < 3; ++mu)
        qg[3 * e + mu] = C(0.2 * std::sin(double(3 * e + mu + 1)), complex ? -0.09 * std::cos(double(mu + 1)) : 0);
    }
    determinant::ComplexDeterminantWorkspace oracle(d, n, 0, 3 * n, n);
    const auto amplitude = oracle.evaluateSpatial(jets.data(), jets.data() + m, 3 * n,
        jets.data() + (1 + 3 * n) * m, n, g.data(), l.data(), ratio.data()).amplitude;
    std::vector<Wide> cpu_scratch(3 * nn);
    oracle.fillSpatialRootMatrixAdjoints(jets.data() + m, jets.data() + (1 + 3 * n) * m,
        qg.data(), ql.data(), reference.data(), reference.size(), cpu_scratch.data(), cpu_scratch.size());
    std::vector<cuDoubleComplex> compact(s * m), expected(s * m), matrix(m), gg(3 * k), ll(k), qgg(3 * k), qll(k);
    for (std::size_t plane = 0; plane < s; ++plane)
    {
      const auto global = !plane ? 0 : (plane <= 3 * k ? plane + 3 * begin : plane - 3 * k + 3 * n + begin);
      for (std::size_t i = 0; i < m; ++i)
      {
        compact[plane * m + i] = pack(jets[global * m + i]);
        expected[plane * m + i] = pack(reference[global * m + i]);
      }
    }
    std::copy_n(compact.data(), m, matrix.data());
    for (std::size_t i = 0; i < 3 * k; ++i) { gg[i] = pack(g[3 * begin + i]); qgg[i] = pack(qg[3 * begin + i]); }
    for (std::size_t i = 0; i < k; ++i) { ll[i] = pack(l[begin + i]); qll[i] = pack(ql[begin + i]); }
    DeviceBuffer<cuDoubleComplex> a(m), head(s * m), inv(m), solve(d * n), scratch(3 * nn), out(s * m), dg(3 * k), dl(k), dqg(3 * k), dql(k);
    DeviceBuffer<std::size_t> permutation(d * n);
    DeviceBuffer<ComplexValueChannel> channels(d);
    DeviceBuffer<ComplexValueResult> result(1);
    DeviceBuffer<std::uint64_t> status(1);
    a.upload(matrix.data(), m); head.upload(compact.data(), compact.size());
    dg.upload(gg.data(), gg.size()); dl.upload(ll.data(), ll.size()); dqg.upload(qgg.data(), qgg.size()); dql.upload(qll.data(), qll.size());
    status.upload(&clear, 1);
    checkRuntime(launchComplexValueFactor(queue.getNative(), d, n, a.data(), channels.data(), status.data(), inv.data(), permutation.data(), solve.data()), "complex root leaf factor");
    queue.sync();
    std::uint64_t factor_status;
    status.download(&factor_status, 1);
    if (factor_status != clear) throw std::runtime_error("Complex kinetic leaf factor rejected");
    const ComplexValueResult regular{pack(amplitude.phase), amplitude.log_abs, 1, 0, 0};
    result.upload(&regular, 1);
    std::vector<cuDoubleComplex> original(m), after(m), actual(s * m);
    inv.download(original.data(), m);
    const auto run = [&] {
      status.upload(&clear, 1);
      checkRuntime(launchComplexKineticMatrixRoots(queue.getNative(), layout, channels.data(), result.data(), inv.data(), head.data(),
          dg.data(), dl.data(), dqg.data(), dql.data(), out.data(), scratch.data(), status.data()), "complex kinetic matrix roots");
      queue.sync();
      std::uint64_t value;
      status.download(&value, 1);
      return value == clear;
    };
    if (special == 1)
    {
      double reference_nonzero = 0;
      for (auto z : reference) reference_nonzero = std::max(reference_nonzero, std::abs(z));
      if (!(reference_nonzero > 1e-60) || run())
        throw std::runtime_error("Explicit finite-factor domain did not reject Wide-supported B/C overflow");
      return;
    }
    const auto before = metrics.snapshot();
    for (std::size_t repeat = 0; repeat < 2; ++repeat)
    {
      if (!run()) throw std::runtime_error("Conditioned complex matrix roots rejected");
      out.download(actual.data(), actual.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        max_root = std::max(max_root, std::abs(host(actual[i]) - host(expected[i])));
        near(cuCreal(actual[i]), cuCreal(expected[i]), "complex root CPU Wide real", 2e-11);
        near(cuCimag(actual[i]), cuCimag(expected[i]), "complex root CPU Wide imaginary", 2e-11);
        if (special == 2 && i % m >= nn)
        {
          const auto error = std::abs(host(actual[i]) / 1e-200 - host(expected[i]) / 1e-200);
          max_tiny = std::max(max_tiny, error);
          near(error, 0, "nonzero tiny complex root scaled CPU Wide", 2e-11);
          if (host(expected[i]) != C{} && host(actual[i]) == C{})
            throw std::runtime_error("Underflowed normalized weight erased a representable orbital root");
        }
      }
    }
    if (metrics.snapshot().allocation_calls != before.allocation_calls || metrics.snapshot().free_calls != before.free_calls)
      throw std::runtime_error("Complex matrix root leaf allocated/freed warmed DEVICE storage");
    inv.download(after.data(), m);
    if (std::memcmp(original.data(), after.data(), m * sizeof(cuDoubleComplex)))
      throw std::runtime_error("Complex root leaf consumed original inverse epoch");
    if (!special)
    {
      // Directional FD varies VALUE and selected derivative planes independently.
      std::vector<C> direction(s * m), shifted = jets, sg(3 * n), sl(n), sr(n);
      C dot{};
      for (std::size_t i = 0; i < direction.size(); ++i)
      {
        direction[i] = C(0.013 * std::sin(double(i + 1)), complex ? 0.009 * std::cos(double(i + 2)) : 0);
        dot += host(actual[i]) * direction[i];
      }
      const auto objective = [&](double step) {
        shifted = jets;
        for (std::size_t plane = 0; plane < s; ++plane)
        {
          const auto global = !plane ? 0 : (plane <= 3 * k ? plane + 3 * begin : plane - 3 * k + 3 * n + begin);
          for (std::size_t i = 0; i < m; ++i) shifted[global * m + i] += step * direction[plane * m + i];
        }
        oracle.evaluateSpatial(shifted.data(), shifted.data() + m, 3 * n,
            shifted.data() + (1 + 3 * n) * m, n, sg.data(), sl.data(), sr.data());
        C value{};
        for (std::size_t i = 0; i < 3 * n; ++i) value += qg[i] * sg[i];
        for (std::size_t i = 0; i < n; ++i) value += ql[i] * sl[i];
        return value;
      };
      const auto numerical = (objective(1e-5) - objective(-1e-5)) / 2e-5;
      max_root_fd = std::max(max_root_fd, std::abs(numerical - dot));
      near(numerical.real(), dot.real(), "complex kinetic root directional FD real", 2e-8);
      near(numerical.imag(), dot.imag(), "complex kinetic root directional FD imaginary", 2e-8);
    }
    std::vector<ComplexValueChannel> records(d);
    channels.download(records.data(), d);
    const auto saved = records;
    records[1].status = 1;
    channels.upload(records.data(), d);
    if (run()) throw std::runtime_error("Mixed singular canonical channel admitted roots");
    channels.upload(saved.data(), d);
    auto node = regular;
    node.phase = pack(C{}); node.log_abs = -std::numeric_limits<double>::infinity();
    result.upload(&node, 1);
    if (run()) throw std::runtime_error("Expected amplitude node admitted complex roots");
    result.upload(&regular, 1);
    records = saved;
    records[0].phase = pack(C(1, 0)); records[1].phase = pack(C(-1, 0));
    records[0].log_abs = records[1].log_abs = 0;
    channels.upload(records.data(), d);
    if (run()) throw std::runtime_error("Cancelling denominator admitted complex roots");
    ++root_cases;
  };
  root_case(2, 0, 2, false, 0);
  root_case(2, 1, 1, true, 0);
  root_case(4, 0, 3, true, 0);
  root_case(4, 3, 1, true, 0);
  root_case(2, 0, 1, false, 2);
  root_case(2, 0, 1, false, 1);

  // Independent paired-head objective: vary arbitrary feature jets and canonical real parameters.
  constexpr std::size_t n = 2, d = 2, f = 3, k = 1, s = 5, m = 8, u = 6, ws = 12, es = 4, pcount = 64;
  auto layout = makeOpenOrbitalJetLayout(1, d, n, 1, f, 1, SpatialJetMode::FULL_VGL);
  layout.features = withSpatialTargetRange(layout.features, 1, k);
  layout.orbitals = withSpatialTargetRange(layout.orbitals, 1, k);
  const ComplexKineticHeadLayout pullback{{layout, pcount, 0, ws, 4 * ws, 4 * ws + es, 4 * ws + 2 * es, 4 * ws + 3 * es}, 2 * ws, 3 * ws};
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.lattice_vectors = {{{4.0, 0.4, -0.2}, {0.7, 5.0, 0.3}, {-0.1, 0.8, 6.0}}};
  environment.periodic_axes = {true, true, true};
  const auto cell = periodic_geometry::makeCell(environment.lattice_vectors, environment.periodic_axes);
  std::vector<double> positions{0.4, 0.7, 0.9, -0.2, 1.3, -0.7}, nuclei{0.1, -0.3, 0.2}, parameters(pcount), features(s * u), envelope(s * m);
  std::vector<C> roots(s * m);
  for (std::size_t i = 0; i < pcount; ++i) parameters[i] = i < 4 * ws ? 0.2 * std::sin(double(i + 1)) : 0.8 + 0.03 * (i - 4 * ws);
  parameters[4 * ws] = 0; // Exactly zero E for one up-spin orbital, but dE/dpi survives.
  parameters[4 * ws + 2 * es + 1] = 0; // Existing abs(zeta) zero subgradient, with nonzero pi.
  for (std::size_t i = 0; i < features.size(); ++i) features[i] = 0.3 * std::sin(double(i + 2));
  for (std::size_t i = 0; i < roots.size(); ++i) roots[i] = C(0.05 * std::cos(double(i + 1)), 0.07 * std::sin(double(i + 3)));
  ComplexOrbitalHeadWorkspace cpu_head({n, 1, d, f, ComplexOrbitalStructure::FULL, ComplexOrbitalPhase::NONE, {}}, layout.features, layout.orbitals);
  const auto objective = [&] {
    std::fill(envelope.begin(), envelope.end(), 0);
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t channel = 0; channel < d; ++channel)
        for (std::size_t orbital = 0; orbital < n; ++orbital)
        {
          periodic_geometry::Pair pair;
          if (!periodic_geometry::evaluatePair(cell, positions.data() + 3 * row, nuclei.data(), pair))
            throw std::runtime_error("Head FD pair fixture invalid");
          const auto index = channel * n + orbital, element = channel * n * n + row * n + orbital;
          const auto pi = parameters[4 * ws + (row ? es : 0) + index];
          const auto z = std::abs(parameters[4 * ws + 2 * es + (row ? es : 0) + index]);
          const auto e = pi * std::exp(-z * pair.radius);
          envelope[element] = e;
          if (row == 1)
          {
            for (std::size_t mu = 0; mu < 3; ++mu) envelope[(1 + mu) * m + element] = -z * e * pair.radius_gradient[mu];
            envelope[4 * m + element] = e * (z * z * pair.radius_gradient_norm_squared - z * pair.radius_laplacian);
          }
        }
    ComplexOrbitalHeadParameters heads;
    heads.up = {parameters.data(), parameters.data() + 2 * ws, nullptr, nullptr, ws, 0};
    heads.down = {parameters.data() + ws, parameters.data() + 3 * ws, nullptr, nullptr, ws, 0};
    cpu_head.evaluate(features.data(), features.size(), envelope.data(), envelope.size(), positions.data(), positions.size(), heads);
    Wide value{};
    for (std::size_t i = 0; i < roots.size(); ++i) value += Wide(roots[i]) * Wide(cpu_head.data()[i]);
    return C(value.real(), value.imag());
  };
  (void)objective();
  std::vector<double> real(s * m), imag(s * m);
  std::vector<cuDoubleComplex> root_packed(s * m);
  for (std::size_t i = 0; i < roots.size(); ++i)
  {
    real[i] = cpu_head.affineProjectionData()[i].real(); imag[i] = cpu_head.affineProjectionData()[i].imag(); root_packed[i] = pack(roots[i]);
  }
  DeviceBuffer<double> dp(pcount), dx(3 * n), di(3), df(s * u), de(s * m), br(s * m), bi(s * m), seed(s * u), response(pcount);
  DeviceBuffer<cuDoubleComplex> dr(s * m);
  DeviceBuffer<std::uint64_t> status(1);
  dp.upload(parameters.data(), pcount); dx.upload(positions.data(), 3 * n); di.upload(nuclei.data(), 3);
  df.upload(features.data(), features.size()); de.upload(envelope.data(), envelope.size()); br.upload(real.data(), real.size()); bi.upload(imag.data(), imag.size()); dr.upload(root_packed.data(), root_packed.size());
  std::vector<double> zero_p(pcount), zero_f(s * u), actual_p(pcount), actual_f(s * u);
  std::vector<C> head_response(pcount), feature_response(s * u);
  const auto before_head = metrics.snapshot();
  for (std::size_t component = 0; component < 2; ++component)
  {
    response.upload(zero_p.data(), pcount); seed.upload(zero_f.data(), zero_f.size()); status.upload(&clear, 1);
    checkRuntime(launchComplexKineticHeadReverse(queue.getNative(), pullback, cell, dx.data(), di.data(), dp.data(), df.data(), de.data(), br.data(), bi.data(), dr.data(), component, seed.data(), response.data(), status.data()), "complex kinetic paired head reverse");
    queue.sync();
    std::uint64_t value;
    status.download(&value, 1);
    if (value != clear) throw std::runtime_error("Complex kinetic paired head rejected");
    response.download(actual_p.data(), pcount); seed.download(actual_f.data(), actual_f.size());
    for (std::size_t i = 0; i < pcount; ++i) head_response[i] += component ? C(0, actual_p[i]) : C(actual_p[i], 0);
    for (std::size_t i = 0; i < actual_f.size(); ++i) feature_response[i] += component ? C(0, actual_f[i]) : C(actual_f[i], 0);
  }
  if (metrics.snapshot().allocation_calls != before_head.allocation_calls || metrics.snapshot().free_calls != before_head.free_calls)
    throw std::runtime_error("Complex head leaf allocated/freed DEVICE storage");
  const auto fd = [&](double& coordinate, C expected) {
    const auto center = coordinate, h = 1e-5;
    coordinate = center + h; const auto plus = objective();
    coordinate = center - h; const auto minus = objective();
    coordinate = center;
    const auto numerical = (plus - minus) / (2 * h);
    max_fd = std::max(max_fd, std::abs(numerical - expected));
    near(numerical.real(), expected.real(), "paired kinetic head FD real", 2e-8);
    near(numerical.imag(), expected.imag(), "paired kinetic head FD imaginary", 2e-8);
    ++head_lanes;
  };
  for (std::size_t i = 0; i < pcount; ++i) fd(parameters[i], head_response[i]);
  for (std::size_t i = 0; i < features.size(); ++i) fd(features[i], feature_response[i]);
  if (std::abs(head_response[4 * ws]) < 1e-6 || head_response[4 * ws + 2 * es + 1] != C{})
    throw std::runtime_error("Zero-E pi or zero-zeta subgradient fixture was vacuous");
  std::cout << "PASS complex kinetic leaves: root_cases=" << root_cases << " head_FD_lanes=" << head_lanes
      << " max_CPU_Wide_root_error=" << max_root << " max_head_FD_error=" << max_fd
      << " max_root_FD_error=" << max_root_fd << " max_tiny_root_scaled_error=" << max_tiny
      << " raw_I_unchanged=1 finite_factor_domain_rejection=1 zero_E_pi=1 zeta_zero_subgradient=1 warm_DEVICE_allocations=0 scope=device_wide\n";
}
#endif
