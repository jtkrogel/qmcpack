//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatialTraceValidation.cu
 * @brief Standalone bounded determinant trace oracle, lifecycle and timing checks.
 */
#include "PsiFormerDeviceSpatialTraces.h"
#include "PsiFormerDeviceDeterminantKernels.h"

#include <algorithm>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;
namespace determinant = qmcplusplus::psiformer::device_determinant;

namespace
{

void require(bool condition, const char* description)
{
  if (!condition)
    throw std::runtime_error(description);
}

void near(double actual, double expected, const char* description)
{
  if (!determinant::isFiniteBinary64(actual) || !determinant::isFiniteBinary64(expected) ||
      std::abs(actual - expected) > 1.e-10 + 5.e-10 * std::max(std::abs(actual), std::abs(expected)))
    throw std::runtime_error(std::string(description) + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

struct Fixture
{
  determinant::SpatialLayout layout;
  std::vector<determinant::FactorizationMetadata> factors;
  std::vector<determinant::CombinationMetadata> combinations;
  std::vector<double> weights, inverses, gradients, laplacians;

  Fixture(std::size_t n, std::size_t targets, std::size_t configurations = 2, std::size_t channels = 2)
      : layout(determinant::resolveSpatialConfigurationStrides({configurations, channels, n, 3 * targets, targets,
            3 * targets * channels * n * n + 11, targets * channels * n * n + 7})),
        factors(configurations * channels), combinations(configurations), weights(factors.size()),
        inverses(configurations * channels * n * n),
        gradients(configurations * layout.gradient_configuration_stride, std::numeric_limits<double>::quiet_NaN()),
        laplacians(configurations * layout.laplacian_configuration_stride, std::numeric_limits<double>::quiet_NaN())
  {
    for (auto& factor : factors)
    {
      factor.status = determinant::FactorizationStatus::REGULAR;
      factor.inverse_available = true;
      factor.phase = factor.matrix_scale = factor.minimum_scaled_pivot = factor.maximum_scaled_pivot = 1;
      factor.log_abs = 0;
    }
    for (auto& combination : combinations)
    {
      combination.status = determinant::CombinationStatus::REGULAR;
      combination.phase = 1;
      combination.log_abs = 0;
      combination.normalized_weights_available = combination.all_required_inverses_available = true;
    }
    for (std::size_t configuration = 0; configuration < configurations; ++configuration)
      for (std::size_t channel = 0; channel < channels; ++channel)
      {
        weights[configuration * channels + channel] = channels == 1 ? 1.0 : channel == 0 ? 1.25 : -.25 / (channels - 1);
        for (std::size_t row = 0; row < n; ++row)
          for (std::size_t column = 0; column < n; ++column)
          {
            const auto element = row * n + column;
            inverses[(configuration * channels + channel) * n * n + element] =
                (row == column ? 1.0 + .01 * row : .003 * (static_cast<int>((row + 2 * column) % 5) - 2));
            for (std::size_t lane = 0; lane < 3 * targets; ++lane)
              gradients[configuration * layout.gradient_configuration_stride +
                  (lane * channels + channel) * n * n + element] =
                  .013 * (static_cast<int>((row + 3 * column + 2 * lane + channel + configuration) % 11) - 5);
            for (std::size_t target = 0; target < targets; ++target)
              laplacians[configuration * layout.laplacian_configuration_stride +
                  (target * channels + channel) * n * n + element] =
                  .017 * (static_cast<int>((3 * row + column + target + channel + configuration) % 13) - 6);
          }
      }
  }

  std::size_t inputBytes() const
  {
    return (weights.size() + inverses.size() + gradients.size() + laplacians.size()) * sizeof(double) +
        factors.size() * sizeof(factors[0]) + combinations.size() * sizeof(combinations[0]);
  }
};

struct Results
{
  std::vector<double> gradient, ratio, log;
  std::vector<determinant::DerivativeStatus> status;

  explicit Results(determinant::SpatialLayout layout)
      : gradient(layout.configuration_count * layout.gradient_lanes),
        ratio(layout.configuration_count * layout.electron_count), log(ratio.size()), status(layout.configuration_count)
  {}
};

Results reference(const Fixture& fixture)
{
  Results result(fixture.layout);
  const auto& l = fixture.layout;
  std::vector<double> scratch(l.matrix_size * l.matrix_size);
  for (std::size_t configuration = 0; configuration < l.configuration_count; ++configuration)
    result.status[configuration] = determinant::combineSpatialTraces(
        fixture.factors.data() + configuration * l.determinant_count, fixture.combinations[configuration],
        fixture.weights.data() + configuration * l.determinant_count,
        fixture.inverses.data() + configuration * l.determinant_count * l.matrix_size * l.matrix_size,
        fixture.gradients.data() + configuration * l.gradient_configuration_stride,
        fixture.laplacians.data() + configuration * l.laplacian_configuration_stride,
        l.determinant_count, l.matrix_size, l.gradient_lanes, l.electron_count, scratch.data(),
        result.gradient.data() + configuration * l.gradient_lanes,
        result.ratio.data() + configuration * l.electron_count, result.log.data() + configuration * l.electron_count);
  return result;
}

void compare(const Results& actual, const Results& expected)
{
  require(actual.status == expected.status, "Parallel/scalar trace typed statuses differ");
  for (std::size_t i = 0; i < actual.gradient.size(); ++i)
    near(actual.gradient[i], expected.gradient[i], "parallel trace gradient parity");
  for (std::size_t i = 0; i < actual.ratio.size(); ++i)
  {
    near(actual.ratio[i], expected.ratio[i], "parallel trace Laplacian ratio parity");
    near(actual.log[i], expected.log[i], "parallel trace Laplacian log parity");
  }
}

// A nearly singular inverse can make t*t-tr(M*M) catastrophically cancelling.
// Qualify that fixture against wider arithmetic and a fixture-scaled allowance;
// a small relative error in the final answer is not promised in this regime.
void compareCancellationScale(const Fixture& fixture, const Results& output)
{
  using Wide = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;
  const auto absolute = [](const Wide& value) -> Wide { return boost::multiprecision::abs(value); };
  const auto& l = fixture.layout;
  const auto n = l.matrix_size;
  const auto matrix_elements = n * n;
  const auto scalar = reference(fixture);
  Wide maximum_ratio_error = 0, maximum_log_error = 0, maximum_ratio_relative_error = 0;
  Wide maximum_scalar_ratio_error = 0, maximum_scalar_log_error = 0, maximum_allowance = 0;
  for (std::size_t b = 0; b < l.configuration_count; ++b)
    for (std::size_t target = 0; target < l.electron_count; ++target)
    {
      require(output.status[b] == determinant::DerivativeStatus::AVAILABLE,
              "Conditioned trace unexpectedly unavailable");
      Wide gradient[3]{}, ratio = 0, scale = 0;
      for (std::size_t d = 0; d < l.determinant_count; ++d)
      {
        const auto* inverse = fixture.inverses.data() + (b * l.determinant_count + d) * matrix_elements;
        const auto* laplacian = fixture.laplacians.data() + b * l.laplacian_configuration_stride +
            (target * l.determinant_count + d) * matrix_elements;
        const Wide weight = fixture.weights[b * l.determinant_count + d];
        Wide bracket = 0, channel_scale = 0;
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
          {
            const Wide value = Wide(inverse[i * n + j]) * laplacian[j * n + i];
            bracket += value;
            channel_scale += absolute(value);
          }
        for (std::size_t dim = 0; dim < 3; ++dim)
        {
          const auto* g = fixture.gradients.data() + b * l.gradient_configuration_stride +
              ((3 * target + dim) * l.determinant_count + d) * matrix_elements;
          std::vector<Wide> product(matrix_elements, 0);
          Wide trace = 0, absolute_trace = 0, square = 0, absolute_square = 0;
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
              const Wide value = Wide(inverse[i * n + j]) * g[j * n + i];
              trace += value;
              absolute_trace += absolute(value);
              for (std::size_t k = 0; k < n; ++k)
                product[i * n + j] += Wide(inverse[i * n + k]) * g[k * n + j];
            }
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
              const auto value = product[i * n + j] * product[j * n + i];
              square += value;
              absolute_square += absolute(value);
            }
          gradient[dim] += weight * trace;
          bracket += trace * trace - square;
          channel_scale += absolute_trace * absolute_trace + absolute_square;
        }
        ratio += weight * bracket;
        scale += absolute(weight) * channel_scale;
      }
      Wide log = ratio;
      for (const auto value : gradient)
      {
        log -= value * value;
        scale += value * value;
      }
      const auto offset = b * l.electron_count + target;
      const Wide ratio_error = absolute(Wide(output.ratio[offset]) - ratio);
      const Wide log_error = absolute(Wide(output.log[offset]) - log);
      // A fixture-scaled sensitivity allowance, NOT a rigorous general bound:
      // in particular this scale omits cancellation inside individual M entries.
      const Wide allowance = Wide(1.e-10) + 128 * (n + 256) * Wide(std::numeric_limits<double>::epsilon()) * scale;
      require(std::max(ratio_error, log_error) <= allowance,
              "Ill-conditioned trace exceeds its fixture-scaled allowance");
      maximum_ratio_error = std::max(maximum_ratio_error, ratio_error);
      maximum_log_error = std::max(maximum_log_error, log_error);
      const Wide relative = ratio_error / std::max(absolute(ratio), Wide(1.e-99));
      maximum_ratio_relative_error = std::max(maximum_ratio_relative_error, relative);
      maximum_scalar_ratio_error = std::max(maximum_scalar_ratio_error,
                                            absolute(Wide(scalar.ratio[offset]) - ratio));
      maximum_scalar_log_error = std::max(maximum_scalar_log_error,
                                          absolute(Wide(scalar.log[offset]) - log));
      maximum_allowance = std::max(maximum_allowance, allowance);
    }
  std::cout << "Cancellation qualification N=" << n
            << " ratio_abs_error=" << static_cast<double>(maximum_ratio_error)
            << " log_abs_error=" << static_cast<double>(maximum_log_error)
            << " ratio_relative_error=" << static_cast<double>(maximum_ratio_relative_error)
            << " scalar_ratio_abs_error=" << static_cast<double>(maximum_scalar_ratio_error)
            << " scalar_log_abs_error=" << static_cast<double>(maximum_scalar_log_error)
            << " fixture_scaled_allowance=" << static_cast<double>(maximum_allowance) << '\n';
}

double maxDifference(const std::vector<double>& left, const std::vector<double>& right)
{
  double difference = 0;
  for (std::size_t i = 0; i < left.size(); ++i)
    difference = std::max(difference, std::abs(left[i] - right[i]));
  return difference;
}

struct StreamOwner
{
  Stream stream{};

  StreamOwner()
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "trace validation stream");
#else
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "trace validation stream");
#endif
  }

  ~StreamOwner()
  {
#ifdef QMC_CUDA2HIP
    (void)hipStreamDestroy(stream);
#else
    (void)cudaStreamDestroy(stream);
#endif
  }
};

struct DeviceFixture
{
  DeviceBuffer<determinant::FactorizationMetadata> factors;
  DeviceBuffer<determinant::CombinationMetadata> combinations;
  DeviceBuffer<double> weights, inverses, gradients, laplacians;

  explicit DeviceFixture(const Fixture& capacity)
      : factors(capacity.factors.size()), combinations(capacity.combinations.size()), weights(capacity.weights.size()),
        inverses(capacity.inverses.size()), gradients(capacity.gradients.size()), laplacians(capacity.laplacians.size())
  {}

  SpatialTraceInputs upload(const Fixture& fixture)
  {
    factors.upload(fixture.factors.data(), fixture.factors.size());
    combinations.upload(fixture.combinations.data(), fixture.combinations.size());
    weights.upload(fixture.weights.data(), fixture.weights.size());
    inverses.upload(fixture.inverses.data(), fixture.inverses.size());
    gradients.upload(fixture.gradients.data(), fixture.gradients.size());
    laplacians.upload(fixture.laplacians.data(), fixture.laplacians.size());
    return {factors.data(), combinations.data(), weights.data(), inverses.data(), gradients.data(), laplacians.data()};
  }
};

Results directTraces(const Fixture& fixture, SpatialTraceInputs input, Stream stream)
{
  Results output(fixture.layout);
  DeviceBuffer<double> scratch(fixture.layout.configuration_count * fixture.layout.matrix_size * fixture.layout.matrix_size),
      gradient(output.gradient.size()), ratio(output.ratio.size()), log(output.log.size());
  DeviceBuffer<determinant::DerivativeStatus> status(output.status.size());
  checkRuntime(launchDeterminantSpatialTraces(stream, fixture.layout, input.factorizations,
      input.combinations, input.normalized_weights, input.inverses, input.gradients, input.laplacians,
      scratch.data(), gradient.data(), ratio.data(), log.data(), status.data()), "direct determinant traces");
#ifdef QMC_CUDA2HIP
  checkRuntime(hipStreamSynchronize(stream), "direct determinant trace completion");
#else
  checkRuntime(cudaStreamSynchronize(stream), "direct determinant trace completion");
#endif
  gradient.download(output.gradient.data(), output.gradient.size());
  ratio.download(output.ratio.data(), output.ratio.size());
  log.download(output.log.data(), output.log.size());
  status.download(output.status.data(), output.status.size());
  return output;
}

void hostChecks()
{
  for (const auto n : {std::size_t{100}, std::size_t{200}, std::size_t{300}})
  {
    const auto plan = planSpatialTraceWorkspace({1, 16, n, 12, 4});
    require(plan.total_device_bytes <= 8 * 1024 * 1024 && plan.product_slots > 0,
            "Large-N trace planner exceeded its complete explicit buffer budget");
    std::cout << "Trace host admission: N=" << n << " K=4 D=16 slots=" << plan.product_slots
              << " product_bytes=" << plan.product_bytes << " metadata_bytes=" << plan.metadata_bytes
              << " total_bytes=" << plan.total_device_bytes << " shared_bytes_per_block=" << plan.shared_bytes_per_block << '\n';
  }
  bool too_small = false, overflow = false, bad_lanes = false, byte_stride_overflow = false;
  try { (void)planSpatialTraceWorkspace({1, 2, 16, 3, 1}, 10); }
  catch (const std::length_error&) { too_small = true; }
  try { (void)planSpatialTraceWorkspace({std::numeric_limits<std::size_t>::max(), 2, 16, 3, 1}); }
  catch (const std::exception&) { overflow = true; }
  try { (void)planSpatialTraceWorkspace({1, 2, 16, 4, 1}); }
  catch (const std::invalid_argument&) { bad_lanes = true; }
  try { (void)planSpatialTraceWorkspace({2, 2, 2, 3, 1,
      std::numeric_limits<std::size_t>::max() / sizeof(double) + 1, 0}); }
  catch (const std::overflow_error&) { byte_stride_overflow = true; }
  require(too_small && overflow && bad_lanes && byte_stride_overflow,
          "Trace workspace admission accepted malformed/overflowing extents");

  Fixture antisymmetric(2, 1, 1, 1);
  antisymmetric.inverses = {1, 0, 0, 1};
  std::fill(antisymmetric.gradients.begin(), antisymmetric.gradients.end(), 0);
  std::fill(antisymmetric.laplacians.begin(), antisymmetric.laplacians.end(), 0);
  antisymmetric.gradients[1] = 2;
  antisymmetric.gradients[2] = -2;
  near(reference(antisymmetric).ratio[0], 8, "signed square trace must not be a Frobenius norm");
  std::cout << "PASS trace host oracle/admission: N100/200/300 budget, overflow, signed square algebra\n";
}

void numericalChecks(Stream stream, DeviceMemoryMetrics& metrics)
{
  for (const auto n : {std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{16}, std::size_t{32}})
  {
    const auto max_targets = std::min(n, std::size_t{3});
    Fixture capacity(n, max_targets);
    DeviceFixture device(capacity);
    const auto metadata = planSpatialTraceWorkspace(capacity.layout).metadata_bytes;
    const auto budget = metadata + 5 * n * n * sizeof(double); // Force multiple chunks and odd tails.
    const auto before_prepare = metrics.snapshot();
    PreparedDeviceSpatialTraces engine(capacity.layout, stream, budget);
    const auto after_prepare = metrics.snapshot();
    require(engine.workspacePlan().product_slots == 5, "Trace product slots ignored the exact prepared budget");
    require(after_prepare.live_bytes - before_prepare.live_bytes == engine.deviceWorkspaceBytes() &&
              engine.deviceWorkspaceBytes() <= budget && engine.hostWorkspaceBytes() == 0,
            "Trace plan did not exactly charge all prepared device buffers");
    const auto owned = engine.deviceWorkspaceBytes();
    for (const auto targets : {max_targets, std::size_t{1}, std::min(n, std::size_t{2}), max_targets})
    {
      Fixture live(n, targets, targets == 1 ? 1 : 2);
      const auto input = device.upload(live);
      Results output(live.layout);
      engine.evaluate(live.layout, input);
      engine.downloadResults(output.gradient, output.ratio, output.log, output.status);
      compare(output, reference(live));
      compare(directTraces(live, input, stream), reference(live));
      const auto before = metrics.snapshot();
      engine.evaluate(live.layout, input);
      engine.downloadResults(output.gradient, output.ratio, output.log, output.status);
      const auto after = metrics.snapshot();
      require(after.allocation_calls == before.allocation_calls && after.free_calls == before.free_calls &&
              after.upload_calls == before.upload_calls && after.download_calls - before.download_calls == 4 &&
              engine.deviceWorkspaceBytes() == owned, "Warmed trace tail reused/gained device allocations or uploads");
    }

    auto check = [&](const Fixture& fixture) {
      Results output(fixture.layout);
      const auto input = device.upload(fixture);
      engine.evaluate(fixture.layout, input);
      engine.downloadResults(output.gradient, output.ratio, output.log, output.status);
      compare(output, reference(fixture));
      compare(directTraces(fixture, input, stream), reference(fixture));
    };
    {
      PreparedDeviceSpatialTraces one_slot(capacity.layout, stream, metadata + n * n * sizeof(double));
      Results output(capacity.layout);
      one_slot.evaluate(capacity.layout, device.upload(capacity));
      one_slot.downloadResults(output.gradient, output.ratio, output.log, output.status);
      compare(output, reference(capacity));
      require(one_slot.workspacePlan().product_slots == 1, "Exact minimum trace budget did not produce one slot");
    }
    Fixture cancellation = capacity;
    cancellation.weights[0] = 1000;
    cancellation.weights[1] = -999;
    std::copy_n(cancellation.inverses.data(), n * n, cancellation.inverses.data() + n * n);
    for (std::size_t lane = 0; lane < cancellation.layout.gradient_lanes; ++lane)
      std::copy_n(cancellation.gradients.data() + lane * 2 * n * n, n * n,
                  cancellation.gradients.data() + (lane * 2 + 1) * n * n);
    for (std::size_t target = 0; target < cancellation.layout.electron_count; ++target)
      std::copy_n(cancellation.laplacians.data() + target * 2 * n * n, n * n,
                  cancellation.laplacians.data() + (target * 2 + 1) * n * n);
    check(cancellation);
    Fixture conditioned = capacity;
    conditioned.inverses[0] = 1.e12;
    conditioned.gradients[0] = 1.e-12;
    Results conditioned_output(conditioned.layout);
    engine.evaluate(conditioned.layout, device.upload(conditioned));
    engine.downloadResults(conditioned_output.gradient, conditioned_output.ratio, conditioned_output.log,
                           conditioned_output.status);
    compareCancellationScale(conditioned, conditioned_output);
    compareCancellationScale(conditioned, directTraces(conditioned, device.upload(conditioned), stream));
    // The same inverse with balanced derivative rows remains well scaled and
    // must also pass the ordinary strict scalar-parity tolerance.
    for (std::size_t lane = 0; lane < conditioned.layout.gradient_lanes; ++lane)
      for (std::size_t column = 0; column < n; ++column)
        conditioned.gradients[lane * 2 * n * n + column] *= 1.e-12;
    for (std::size_t target = 0; target < conditioned.layout.electron_count; ++target)
      for (std::size_t column = 0; column < n; ++column)
        conditioned.laplacians[target * 2 * n * n + column] *= 1.e-12;
    check(conditioned);
    if (n == 2)
    {
      Fixture antisymmetric = capacity;
      antisymmetric.weights[0] = 1;
      antisymmetric.weights[1] = 0;
      std::fill_n(antisymmetric.inverses.data(), n * n, 0.0);
      antisymmetric.inverses[0] = antisymmetric.inverses[3] = 1;
      std::fill_n(antisymmetric.gradients.data(), antisymmetric.layout.gradient_lanes * 2 * n * n, 0.0);
      std::fill_n(antisymmetric.laplacians.data(), antisymmetric.layout.electron_count * 2 * n * n, 0.0);
      antisymmetric.gradients[1] = 2;
      antisymmetric.gradients[2] = -2;
      check(antisymmetric);
    }
    Fixture broken = capacity;
    broken.weights[1] = 0;
    broken.factors[1].status = determinant::FactorizationStatus::SINGULAR;
    broken.factors[1].inverse_available = false;
    std::fill_n(broken.inverses.data() + n * n, n * n, std::numeric_limits<double>::quiet_NaN());
    check(broken); // Zero weight permits an unavailable inverse, not bad G/L.
    broken.factors[1].status = determinant::FactorizationStatus::NONFINITE_INPUT;
    check(broken);
    broken.factors[1].status = determinant::FactorizationStatus::SINGULAR;
    broken.gradients[n * n] = std::numeric_limits<double>::quiet_NaN();
    check(broken);
    broken = capacity;
    broken.weights[1] = 0;
    broken.laplacians[n * n] = std::numeric_limits<double>::quiet_NaN();
    check(broken);
    broken = capacity;
    broken.factors[0].inverse_available = false;
    check(broken);
    broken.combinations[0].phase = 0;
    broken.gradients[0] = std::numeric_limits<double>::quiet_NaN();
    check(broken); // NODE wins over inverse/nonfinite failures.
    broken = capacity;
    broken.inverses[0] = std::numeric_limits<double>::max();
    broken.gradients[0] = std::numeric_limits<double>::max();
    check(broken);
    check(capacity);
    const auto old_generation = engine.result().generation;
    bool replacement_rejected = false, old_result_rejected = false, old_download_rejected = false;
    try { engine.evaluate(capacity.layout, {}); }
    catch (const std::invalid_argument&) { replacement_rejected = true; }
    try { (void)engine.result(); }
    catch (const std::logic_error&) { old_result_rejected = true; }
    Results stale(capacity.layout);
    try { engine.downloadResults(stale.gradient, stale.ratio, stale.log, stale.status); }
    catch (const std::logic_error&) { old_download_rejected = true; }
    require(replacement_rejected && old_result_rejected && old_download_rejected,
            "Trace failed replacement exposed an older result");
    check(capacity);
    require(engine.result().generation > old_generation, "Trace recovery failed to advance its generation");
  }
  {
    Fixture large(100, 1, 1, 2);
    DeviceFixture device(large);
    compare(directTraces(large, device.upload(large), stream), reference(large));
  }
  std::cout << "PASS parallel compensated traces: nonsymmetric/signed channels, B/K/slot tails, padding, "
               "zero-weight inverse/nonfinite rules, node priority, overflow, failure/recovery, zero warm allocations\n";
}

void benchmark(Stream stream, DeviceMemoryMetrics& metrics, const std::string& path, bool large_probes)
{
  std::ofstream report(path);
  if (!report)
    throw std::runtime_error("Cannot open trace benchmark CSV");
  report << "operation,N,D,K,B,repetitions,min_seconds,median_seconds,product_slots,product_bytes,metadata_bytes,"
            "engine_bytes,input_bytes,allocation_calls,free_calls,upload_calls,uploaded_bytes,download_calls,downloaded_bytes,"
            "gradient_max_abs_error,ratio_max_abs_error,log_max_abs_error\n";
  report << std::setprecision(17);
  const std::vector<std::size_t> sizes = large_probes ? std::vector<std::size_t>{32, 100, 200, 300}
                                                    : std::vector<std::size_t>{16, 32};
  for (const auto n : sizes)
  {
    const unsigned repetitions = n >= 100 ? 1 : 3; // Bound expensive serial-oracle qualification time.
    Fixture fixture(n, 1, 1, 2);
    require(fixture.inputBytes() < 16 * 1024 * 1024, "Synthetic trace probe exceeds its input budget");
    const auto planned = planSpatialTraceWorkspace(fixture.layout);
    const auto scalar_bytes = (n * n + 5) * sizeof(double) + sizeof(determinant::DerivativeStatus);
    const auto complete_bytes = fixture.inputBytes() + planned.total_device_bytes + scalar_bytes;
    require(complete_bytes < 32 * 1024 * 1024, "Synthetic trace probe exceeds its complete explicit device budget");
    std::cout << "Trace-only probe admitted: N=" << n << " D=2 K=1 input_bytes=" << fixture.inputBytes()
              << " engine_bytes=" << planned.total_device_bytes << " scalar_bytes=" << scalar_bytes
              << " complete_explicit_bytes=" << complete_bytes << std::endl;
    DeviceFixture device(fixture);
    const auto input = device.upload(fixture);
    PreparedDeviceSpatialTraces engine(fixture.layout, stream);
    DeviceBuffer<double> scalar_scratch(n * n), scalar_gradient(3), scalar_ratio(1), scalar_log(1);
    DeviceBuffer<determinant::DerivativeStatus> scalar_status(1);
    Results parallel(fixture.layout), scalar(fixture.layout);
    auto run_parallel = [&]() {
      engine.evaluate(fixture.layout, input);
      engine.downloadResults(parallel.gradient, parallel.ratio, parallel.log, parallel.status);
    };
    auto run_scalar = [&]() {
      checkRuntime(launchDeterminantSpatialTraces(stream, fixture.layout, input.factorizations,
          input.combinations, input.normalized_weights, input.inverses, input.gradients, input.laplacians,
          scalar_scratch.data(), scalar_gradient.data(), scalar_ratio.data(), scalar_log.data(), scalar_status.data()),
          "trace scalar baseline");
#ifdef QMC_CUDA2HIP
      checkRuntime(hipStreamSynchronize(stream), "trace scalar completion");
#else
      checkRuntime(cudaStreamSynchronize(stream), "trace scalar completion");
#endif
      scalar_gradient.download(scalar.gradient.data(), scalar.gradient.size());
      scalar_ratio.download(scalar.ratio.data(), scalar.ratio.size());
      scalar_log.download(scalar.log.data(), scalar.log.size());
      scalar_status.download(scalar.status.data(), scalar.status.size());
    };
    run_parallel();
    run_scalar();
    compare(parallel, scalar);
    const auto gradient_error = maxDifference(parallel.gradient, scalar.gradient);
    const auto ratio_error = maxDifference(parallel.ratio, scalar.ratio);
    const auto log_error = maxDifference(parallel.log, scalar.log);
    for (const bool custom : {false, true})
    {
      const auto before = metrics.snapshot();
      std::vector<double> timings;
      for (unsigned repeat = 0; repeat < repetitions; ++repeat)
      {
        const auto start = std::chrono::steady_clock::now();
        if (custom)
          run_parallel();
        else
          run_scalar();
        timings.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      }
      const auto after = metrics.snapshot();
      require(before.allocation_calls == after.allocation_calls && before.free_calls == after.free_calls,
              "Trace benchmark allocated in a warmed fixed-capacity operation");
      std::sort(timings.begin(), timings.end());
      const auto& plan = engine.workspacePlan();
      report << (custom ? "parallel_compensated" : "scalar") << ',' << n << ",2,1,1," << repetitions << ','
             << timings.front() << ',' << timings[timings.size() / 2] << ',' << plan.product_slots << ','
             << plan.product_bytes << ',' << plan.metadata_bytes << ','
             << plan.total_device_bytes << ',' << fixture.inputBytes() << ','
             << after.allocation_calls - before.allocation_calls << ',' << after.free_calls - before.free_calls << ','
             << after.upload_calls - before.upload_calls << ',' << after.uploaded_bytes - before.uploaded_bytes << ','
             << after.download_calls - before.download_calls << ',' << after.downloaded_bytes - before.downloaded_bytes << ','
             << gradient_error << ',' << ratio_error << ',' << log_error << '\n';
    }
    std::cout << "PASS trace-only synthetic benchmark N=" << n << " D=2 K=1 input_bytes=" << fixture.inputBytes()
              << " explicit_engine_bytes=" << engine.deviceWorkspaceBytes() << std::endl;
    report.flush();
  }
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    bool host_only = false, large_probes = false;
    std::string report;
    for (int argument = 1; argument < argc; ++argument)
    {
      const std::string option(argv[argument]);
      if (option == "--host-check")
        host_only = true;
      else if (option == "--large-probes")
        large_probes = true;
      else if (option == "--benchmark" && argument + 1 < argc)
        report = argv[++argument];
      else
        throw std::invalid_argument("Usage: psiformer_device_spatial_trace_validation [--host-check] [--benchmark CSV] [--large-probes]");
    }
    hostChecks();
    if (host_only)
      return 0;
    StreamOwner owner;
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "trace metrics device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "trace metrics device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    numericalChecks(owner.stream, metrics);
    if (!report.empty())
      benchmark(owner.stream, metrics, report, large_probes);
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL spatial traces: " << error.what() << '\n';
    return 1;
  }
}
