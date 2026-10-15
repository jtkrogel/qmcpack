//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceLiftedDeterminantProductValidation.cu
 * @brief Focused factor-jet parity, independent trace oracle and opt-in timings.
 */

#include "PsiFormerDeviceLiftedDeterminantProducts.h"
#include "PsiFormerDeviceSpatialTraces.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;
namespace det = qmcplusplus::psiformer::device_determinant;

namespace
{

/// Report the first failed invariant without running a broad fixture matrix.
void require(bool condition, const char* what)
{ if (!condition) throw std::runtime_error(what); }

/// Compare well-conditioned nonsymmetric fixtures without promising node accuracy.
void near(double actual, double expected, const char* what)
{
  require(det::isFiniteBinary64(actual) && det::isFiniteBinary64(expected) &&
      ::fabs(actual - expected) <= 1e-10 + 5e-10 * std::max(::fabs(actual), ::fabs(expected)), what);
}

/** Keep synthetic inputs distinct in every matrix/plane/channel. The inverse
 * is nonsymmetric and diagonally dominant; no inverse-as-transpose shortcut works.
 */
struct Fixture
{
  LiftedDeterminantLayout layout;
  std::vector<det::FactorizationMetadata> factors;
  det::CombinationMetadata combination;
  std::vector<double> weights, inverse, jets, gradient, laplacian;

  Fixture(std::size_t n, std::size_t targets, std::size_t channels, bool exact_roots = true)
      : layout{makeSpatialTargetJetLayout(1, channels * n * n, n, n - targets, targets, channels * n * n + 7),
               makeSpatialTargetJetLayout(1, 1, n, n - targets, targets), channels, n},
        factors(channels), weights(channels), inverse(channels * n * n),
        jets(spatialJetSpanElements(layout.matrices), 913), gradient(3 * targets), laplacian(targets)
  {
    combination.status = det::CombinationStatus::REGULAR;
    combination.phase = 1;
    combination.normalized_weights_available = combination.all_required_inverses_available = true;
    const auto nn = n * n;
    for (std::size_t d = 0; d < channels; ++d)
    {
      factors[d].status = det::FactorizationStatus::REGULAR;
      factors[d].inverse_available = true;
      factors[d].phase = factors[d].matrix_scale = factors[d].minimum_scaled_pivot = factors[d].maximum_scaled_pivot = 1;
      weights[d] = channels == 1 ? 1.0 : d == 0 ? 1.25 : -.25 / (channels - 1);
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
        {
          const auto i = d * nn + r * n + c;
          inverse[i] = r == c ? 1 + .001 * r : .003 * (static_cast<int>((r + 2 * c + d) % 7) - 3);
          for (std::size_t p = 0; p < layout.matrices.plane_count; ++p)
            jets[layout.matrices.uncheckedPlaneOffset(0, p, i)] =
                .013 * (static_cast<int>((r + 3 * c + 2 * p + d) % 11) - 5);
        }
    }
    if (exact_roots)
    {
      const auto packed = packedDerivatives();
      std::vector<double> scratch(nn), ratios(targets);
      require(det::combineSpatialTraces(factors.data(), combination, weights.data(), inverse.data(),
          packed.first.data(), packed.second.data(), channels, n, 3 * targets, targets, scratch.data(),
          gradient.data(), ratios.data(), laplacian.data()) == det::DerivativeStatus::AVAILABLE, "CPU trace fixture failed");
    }
  }

  /// Extract contiguous derivative planes for the independent parallel trace API.
  std::pair<std::vector<double>, std::vector<double>> packedDerivatives() const
  {
    const auto& m = layout.matrices;
    std::vector<double> g(m.gradient_lanes * m.element_count), l(m.laplacian_lanes * m.element_count);
    for (std::size_t p = 0; p < m.gradient_lanes; ++p)
      std::copy_n(jets.data() + m.uncheckedGradientOffset(0, p, 0), m.element_count, g.data() + p * m.element_count);
    for (std::size_t p = 0; p < m.laplacian_lanes; ++p)
      std::copy_n(jets.data() + m.uncheckedLaplacianOffset(0, p, 0), m.element_count, l.data() + p * m.element_count);
    return {std::move(g), std::move(l)};
  }
};

/// Own only test inputs/outputs; these bytes are separate from the product owner.
struct DeviceFixture
{
  DeviceBuffer<det::FactorizationMetadata> factors;
  DeviceBuffer<det::CombinationMetadata> combination;
  DeviceBuffer<double> weights, inverse, jets, gradient, laplacian, output;
  explicit DeviceFixture(const Fixture& f)
      : factors(f.factors.size()), combination(1), weights(f.weights.size()), inverse(f.inverse.size()),
        jets(f.jets.size()), gradient(f.gradient.size()), laplacian(f.laplacian.size()), output(f.jets.size() + 2)
  { upload(f); }

  /// Replace test inputs and reset guard/padding slots outside the warmed interval.
  void upload(const Fixture& f)
  {
    factors.upload(f.factors.data(), f.factors.size()); combination.upload(&f.combination, 1);
    weights.upload(f.weights.data(), f.weights.size()); inverse.upload(f.inverse.data(), f.inverse.size());
    jets.upload(f.jets.data(), f.jets.size()); gradient.upload(f.gradient.data(), f.gradient.size());
    laplacian.upload(f.laplacian.data(), f.laplacian.size());
    const std::vector<double> guards(f.jets.size() + 2, 913);
    output.upload(guards.data(), guards.size());
  }

  /// Bind the current borrowed test arrays without packing any device matrices.
  LiftedDeterminantProductInputs inputs(const Fixture& f) const
  { return {f.layout, factors.data(), combination.data(), weights.data(), inverse.data(), jets.data(), gradient.data(), laplacian.data()}; }
};

/// Compare every plane and every padding/guard slot, including whole-output failure.
double compare(const Fixture& f, const std::vector<double>& actual, const std::vector<double>& expected)
{
  require(actual.front() == 913 && actual.back() == 913, "Factor guards changed");
  const auto& m = f.layout.matrices;
  double maximum_error = 0;
  std::vector<bool> touched(f.jets.size(), false);
  for (std::size_t p = 0; p < m.plane_count; ++p)
    for (std::size_t i = 0; i < m.element_count; ++i)
    {
      const auto offset = m.uncheckedPlaneOffset(0, p, i);
      touched[offset] = true;
      near(actual[offset + 1], expected[offset], "Lifted factor plane differs from scalar reference");
      maximum_error = std::max(maximum_error, ::fabs(actual[offset + 1] - expected[offset]));
    }
  for (std::size_t i = 0; i < touched.size(); ++i)
    if (!touched[i]) require(actual[i + 1] == 913, "Factor padding changed");
  return maximum_error;
}

/// Reuse the existing independently qualified compensated trace owner as a second oracle.
void traceOracle(const Fixture& f, const DeviceFixture& gpu, qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue)
{
  const auto packed = f.packedDerivatives();
  DeviceBuffer<double> g(packed.first.size()), l(packed.second.size());
  g.upload(packed.first.data(), packed.first.size()); l.upload(packed.second.data(), packed.second.size());
  const det::SpatialLayout layout{1, f.layout.determinant_count, f.layout.matrix_size,
      f.layout.matrices.gradient_lanes, f.layout.matrices.laplacian_lanes};
  PreparedDeviceSpatialTraces oracle(layout, queue.getNative(), 1024 * 1024);
  oracle.evaluate(layout, {gpu.factors.data(), gpu.combination.data(), gpu.weights.data(), gpu.inverse.data(), g.data(), l.data()});
  std::vector<double> gradient(f.gradient.size()), ratio(f.laplacian.size()), log(f.laplacian.size());
  std::vector<det::DerivativeStatus> status(1);
  oracle.downloadResults(gradient, ratio, log, status);
  require(status[0] == det::DerivativeStatus::AVAILABLE, "Independent trace oracle failed");
  for (std::size_t i = 0; i < gradient.size(); ++i) near(gradient[i], f.gradient[i], "Independent nonsymmetric gradient trace differs");
  for (std::size_t i = 0; i < log.size(); ++i) near(log[i], f.laplacian[i], "Independent signed square trace differs");
}

/// Check one fixture against the original scalar factor algebra without changing the owner.
double parity(Fixture& f, DeviceFixture& gpu, PreparedDeviceLiftedDeterminantProducts& owner)
{
  std::vector<double> expected(f.jets.size(), 913), scratch(3 * f.layout.matrix_size * f.layout.matrix_size);
  const auto reference = det::buildLiftedDeterminantFactors(f.layout.matrices, 0, f.factors.data(), f.combination,
      f.weights.data(), f.inverse.data(), f.jets.data(), f.gradient.data(), f.laplacian.data(),
      f.layout.determinant_count, f.layout.matrix_size, expected.data(), scratch.data());
  const auto actual = owner.evaluate(gpu.inputs(f), gpu.output.data() + 1);
  require(actual == reference, "Lifted factor typed status differs from scalar");
  std::vector<double> result(f.jets.size() + 2);
  gpu.output.download(result.data(), result.size());
  return compare(f, result, expected);
}

/// Constrain masks, channel tails, compact physical targets, overflow and recovery.
void correctness(qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, int ordinal)
{
  Fixture f(5, 2, 5);
  DeviceFixture gpu(f);
  traceOracle(f, gpu, queue);
  const auto plan = planLiftedDeterminantProducts(5, 2, 1024 * 1024);
  const auto before = deviceMemoryMetrics(ordinal).snapshot();
  PreparedDeviceLiftedDeterminantProducts owner(5, 2, plan.device_workspace_bytes, queue, blas);
  require(deviceMemoryMetrics(ordinal).snapshot().live_bytes - before.live_bytes == plan.device_workspace_bytes,
      "Lifted product predicted/actual owned bytes differ");
  const auto max_error = parity(f, gpu, owner);
  const auto warm = deviceMemoryMetrics(ordinal).snapshot();
  parity(f, gpu, owner);
  require(deviceMemoryMetrics(ordinal).snapshot().allocation_calls == warm.allocation_calls, "Warmed factor operation allocated storage");

  // A masked channel may have no inverse and arbitrary poisoned derivative planes.
  const auto baseline = f;
  f.weights.back() = 0;
  f.factors.back().status = det::FactorizationStatus::SINGULAR;
  f.factors.back().inverse_available = false;
  const auto nn = f.layout.matrix_size * f.layout.matrix_size, begin = (f.layout.determinant_count - 1) * nn;
  std::fill(f.inverse.begin() + begin, f.inverse.end(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t p = 0; p < f.layout.matrices.plane_count; ++p)
    std::fill_n(f.jets.data() + f.layout.matrices.uncheckedPlaneOffset(0, p, begin), nn,
        std::numeric_limits<double>::quiet_NaN());
  gpu.upload(f); parity(f, gpu, owner);
  f = baseline;
  f.combination.status = det::CombinationStatus::NODE;
  gpu.upload(f); parity(f, gpu, owner);
  f = baseline;
  f.jets[f.layout.matrices.uncheckedGradientOffset(0, 0, 0)] = std::numeric_limits<double>::max();
  gpu.upload(f); parity(f, gpu, owner);
  f = baseline;
  gpu.upload(f); parity(f, gpu, owner);
  Fixture tail(5, 1, 3);
  DeviceFixture gpu_tail(tail);
  parity(tail, gpu_tail, owner);
  std::cout << "PASS lifted factor products: every plane, independent traces, K2/K1, D5/D3 tails, zero-mask poison, "
               "node/overflow/recovery, padding, exact bytes and warmed reuse; max_abs_difference=" << max_error << '\n';
}

/// Time one scalar GPU baseline and three warmed GEMM repeats, only B1/C1/K1.
void benchmark(std::size_t n, qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  Fixture f(n, 1, 1, false);
  DeviceFixture gpu(f);
  const auto plan = planLiftedDeterminantProducts(n, 1, 64 * 1024 * 1024);
  PreparedDeviceLiftedDeterminantProducts owner(n, 1, plan.device_workspace_bytes, queue, blas);
  DeviceBuffer<double> scalar(f.jets.size()), scratch(3 * n * n);
  DeviceBuffer<det::DerivativeStatus> status(1);
  const std::vector<double> scalar_guards(f.jets.size(), 913);
  scalar.upload(scalar_guards.data(), scalar_guards.size());
  queue.sync();
  const auto start = std::chrono::steady_clock::now();
  checkRuntime(launchDeterminantFactorJets(queue.getNative(), f.layout, gpu.factors.data(), gpu.combination.data(),
      gpu.weights.data(), gpu.inverse.data(), gpu.jets.data(), gpu.gradient.data(), gpu.laplacian.data(),
      scalar.data(), scratch.data(), status.data()), "scalar lifted factor benchmark");
  queue.sync();
  det::DerivativeStatus scalar_status;
  status.download(&scalar_status, 1);
  const auto scalar_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  require(scalar_status == det::DerivativeStatus::AVAILABLE, "Scalar large-N baseline rejected fixture");
  require(owner.evaluate(gpu.inputs(f), gpu.output.data() + 1) == scalar_status, "GEMM large-N status differs");
  std::vector<double> actual(f.jets.size() + 2), expected(f.jets.size());
  gpu.output.download(actual.data(), actual.size()); scalar.download(expected.data(), expected.size());
  const auto max_error = compare(f, actual, expected);
  const auto warmed = std::chrono::steady_clock::now();
  for (std::size_t repeat = 0; repeat < 3; ++repeat)
    require(owner.evaluate(gpu.inputs(f), gpu.output.data() + 1) == scalar_status, "Warmed large-N factor failed");
  const auto gemm_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - warmed).count() / 3;
  std::cout << "Lifted factor timing: N=" << n << " B=1 D=1 K=1 C=1 scalar_samples=1 gemm_samples=3"
            << " scalar_seconds=" << scalar_seconds << " gemm_seconds=" << gemm_seconds
            << " owned_device_bytes=" << owner.deviceWorkspaceBytes()
            << " max_abs_difference=" << max_error << '\n';
}

} // namespace

// Keep this opt-in numerical executable independent of the QMC production driver.
int main(int argc, char** argv)
{
  try
  {
    const bool timing = argc == 2 && std::string(argv[1]) == "--benchmark";
    require(argc == 1 || timing, "Usage: psiformer_device_lifted_products_validation [--benchmark]");
    bool rejected = false;
    try { (void)planLiftedDeterminantProducts(300, 4, 1); }
    catch (const std::length_error&) { rejected = true; }
    require(rejected, "Under-budget product plan admitted storage");
    rejected = false;
    try { (void)planLiftedDeterminantProducts(std::numeric_limits<std::size_t>::max(), 16, std::numeric_limits<std::size_t>::max()); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "Overflowing product plan admitted storage");
    int count = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
    if (hipGetDeviceCount(&count) != success || count == 0) return 77;
    checkRuntime(hipGetDevice(&ordinal), "lifted validation device");
#else
    if (cudaGetDeviceCount(&count) != success || count == 0) return 77;
    checkRuntime(cudaGetDevice(&ordinal), "lifted validation device");
#endif
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle blas(queue);
    correctness(queue, blas, ordinal);
    if (timing) for (const auto n : {50, 100, 200, 300}) benchmark(n, queue, blas);
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
