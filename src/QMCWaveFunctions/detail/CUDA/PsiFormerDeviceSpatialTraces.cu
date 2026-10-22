//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "PsiFormerDeviceSpatialTraces.h"

#include <algorithm>
#include <limits>

namespace qmcplusplus::psiformer::device
{
namespace
{

constexpr unsigned int trace_block_size = 256;
constexpr std::size_t maximum_workspace = 8 * 1024 * 1024;
constexpr unsigned int available = static_cast<unsigned int>(device_determinant::DerivativeStatus::AVAILABLE);
constexpr unsigned int nonfinite =
    static_cast<unsigned int>(device_determinant::DerivativeStatus::NONFINITE_INPUT_OR_RESULT);
using device_determinant::CompensatedSumBinary64;
using device_determinant::SpatialLayout;

std::size_t multiply(std::size_t left, std::size_t right)
{
  return spatial_detail::checkedProduct(left, right, "Spatial trace workspace extent overflow");
}

std::size_t add(std::size_t left, std::size_t right)
{
  return spatial_detail::checkedAdd(left, right, "Spatial trace workspace extent overflow");
}

SpatialLayout checkedLayout(SpatialLayout layout)
{
  if (layout.configuration_count == 0 || layout.determinant_count == 0 || layout.matrix_size == 0 ||
      layout.electron_count == 0 || layout.electron_count > layout.matrix_size ||
      layout.gradient_lanes != multiply(3, layout.electron_count))
    throw std::invalid_argument("Spatial traces require positive dimensions and three lanes per local target");
  const auto channels = multiply(layout.configuration_count, layout.determinant_count);
  const auto channel_elements = multiply(layout.determinant_count, multiply(layout.matrix_size, layout.matrix_size));
  (void)checkedDeviceBytes<double>(multiply(layout.configuration_count, channel_elements));
  (void)checkedDeviceBytes<double>(channels);
  (void)checkedDeviceBytes<device_determinant::FactorizationMetadata>(channels);
  (void)checkedDeviceBytes<device_determinant::CombinationMetadata>(layout.configuration_count);
  layout = device_determinant::resolveSpatialConfigurationStrides(layout);
  (void)checkedDeviceBytes<double>(spatial_detail::checkedSpan(layout.configuration_count,
      layout.gradient_configuration_stride, multiply(layout.gradient_lanes, channel_elements),
      "Spatial trace gradient input span overflow"));
  (void)checkedDeviceBytes<double>(spatial_detail::checkedSpan(layout.configuration_count,
      layout.laplacian_configuration_stride, multiply(layout.electron_count, channel_elements),
      "Spatial trace Laplacian input span overflow"));
  return layout;
}

unsigned int blocks(std::size_t elements)
{
  const auto count = elements / trace_block_size + (elements % trace_block_size != 0);
  if (count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("Spatial trace grid exceeds CUDA/HIP x dimension");
  return static_cast<unsigned int>(count);
}

Error peekError()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

Error synchronize(Stream stream)
{
#ifdef QMC_CUDA2HIP
  return hipStreamSynchronize(stream);
#else
  return cudaStreamSynchronize(stream);
#endif
}

__device__ void markNonfinite(unsigned int* status)
{
  atomicCAS(status, available, nonfinite);
}

__global__ void admitMetadata(SpatialLayout layout, SpatialTraceInputs input, unsigned int* failure,
                              device_determinant::DerivativeStatus* admission)
{
  const auto configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration < layout.configuration_count)
  {
    admission[configuration] = device_determinant::derivativeAvailability(
        input.factorizations + configuration * layout.determinant_count, input.combinations[configuration],
        input.normalized_weights + configuration * layout.determinant_count, layout.determinant_count);
    failure[configuration] = static_cast<unsigned int>(admission[configuration]);
  }
}

/// Only logical live elements are scanned; configuration padding may be unwritten.
__global__ void scanInputs(SpatialLayout layout, SpatialTraceInputs input,
                           std::size_t inverse_count, std::size_t gradient_count,
                           std::size_t total_count,
                           const device_determinant::DerivativeStatus* admission, unsigned int* failure)
{
  auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_count)
    return;
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  const auto channel_elements = layout.determinant_count * matrix_elements;
  std::size_t configuration;
  const double* value;
  if (index < inverse_count)
  {
    configuration = index / channel_elements;
    const auto channel = (index / matrix_elements) % layout.determinant_count;
    if (admission[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
      return;
    if (input.normalized_weights[configuration * layout.determinant_count + channel] == 0.0)
      return; // An unavailable zero-weight inverse must never be read.
    value = input.inverses + index;
  }
  else if ((index -= inverse_count) < gradient_count)
  {
    const auto span = layout.gradient_lanes * channel_elements;
    configuration = index / span;
    value = input.gradients + configuration * layout.gradient_configuration_stride + index % span;
  }
  else
  {
    index -= gradient_count;
    const auto span = layout.electron_count * channel_elements;
    configuration = index / span;
    value = input.laplacians + configuration * layout.laplacian_configuration_stride + index % span;
  }
  if (admission[configuration] == device_determinant::DerivativeStatus::AVAILABLE &&
      !device_determinant::isFiniteBinary64(*value))
    markNonfinite(failure + configuration);
}

/// Every output entry owns its compensated N-term inner loop, with no atomics.
__global__ void matrixProducts(SpatialLayout layout, SpatialTraceInputs input, std::size_t first_task,
                               std::size_t count, double* products,
                               const device_determinant::DerivativeStatus* admission, unsigned int* failure)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  if (index >= count * matrix_elements)
    return;
  const auto task = first_task + index / matrix_elements;
  const auto channel = task % layout.determinant_count;
  const auto lane = (task / layout.determinant_count) % layout.gradient_lanes;
  const auto configuration = task / (layout.determinant_count * layout.gradient_lanes);
  products[index] = 0.0;
  if (admission[configuration] != device_determinant::DerivativeStatus::AVAILABLE ||
      input.normalized_weights[configuration * layout.determinant_count + channel] == 0.0)
    return;
  const auto row = (index % matrix_elements) / layout.matrix_size;
  const auto column = index % layout.matrix_size;
  const auto* inverse = input.inverses + (configuration * layout.determinant_count + channel) * matrix_elements;
  const auto* gradient = input.gradients + configuration * layout.gradient_configuration_stride +
      (lane * layout.determinant_count + channel) * matrix_elements;
  CompensatedSumBinary64 entry;
  for (std::size_t inner = 0; inner < layout.matrix_size; ++inner)
    entry.add(inverse[row * layout.matrix_size + inner] * gradient[inner * layout.matrix_size + column]);
  const auto value = entry.value();
  if (!device_determinant::isFiniteBinary64(value))
    markNonfinite(failure + configuration);
  else
    products[index] = value;
}

/// Direct tr(I*G) and signed tr((I*G)^2), not a Frobenius norm.
__global__ void gradientStatistics(SpatialLayout layout, SpatialTraceInputs input,
                                   std::size_t first_task, const double* products,
                                   double* stats, const device_determinant::DerivativeStatus* admission,
                                   unsigned int* failure)
{
  const auto slot = std::size_t(blockIdx.x);
  const auto task = first_task + slot;
  const auto channel = task % layout.determinant_count;
  const auto lane = (task / layout.determinant_count) % layout.gradient_lanes;
  const auto configuration = task / (layout.determinant_count * layout.gradient_lanes);
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  const bool selected = admission[configuration] == device_determinant::DerivativeStatus::AVAILABLE &&
      input.normalized_weights[configuration * layout.determinant_count + channel] != 0.0;
  const auto* inverse = input.inverses + (configuration * layout.determinant_count + channel) * matrix_elements;
  const auto* gradient = input.gradients + configuration * layout.gradient_configuration_stride +
      (lane * layout.determinant_count + channel) * matrix_elements;
  const auto* product = products + slot * matrix_elements;
  CompensatedSumBinary64 direct, square;
  if (selected)
    for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x)
    {
      const auto row = element / layout.matrix_size;
      const auto column = element % layout.matrix_size;
      const auto transpose = column * layout.matrix_size + row;
      direct.add(inverse[element] * gradient[transpose]);
      square.add(product[element] * product[transpose]);
    }
  __shared__ double partial_direct[trace_block_size], partial_square[trace_block_size];
  const auto direct_value = direct.value(), square_value = square.value();
  if (!device_determinant::isFiniteBinary64(direct_value) ||
      !device_determinant::isFiniteBinary64(square_value))
    markNonfinite(failure + configuration);
  partial_direct[threadIdx.x] = direct_value;
  partial_square[threadIdx.x] = square_value;
  __syncthreads();
  if (threadIdx.x == 0)
  {
    CompensatedSumBinary64 combined_direct, combined_square;
    for (unsigned int part = 0; part < trace_block_size; ++part)
    {
      combined_direct.add(partial_direct[part]);
      combined_square.add(partial_square[part]);
    }
    const auto t = combined_direct.value(), s = combined_square.value();
    if (!device_determinant::isFiniteBinary64(t) || !device_determinant::isFiniteBinary64(s))
      markNonfinite(failure + configuration);
    const auto offset = ((configuration * layout.electron_count + lane / 3) * layout.determinant_count + channel) * 7;
    stats[offset + lane % 3] = device_determinant::isFiniteBinary64(t) ? t : 0.0;
    stats[offset + 3 + lane % 3] = device_determinant::isFiniteBinary64(s) ? s : 0.0;
  }
}

__global__ void laplacianStatistics(SpatialLayout layout, SpatialTraceInputs input,
                                    double* stats, const device_determinant::DerivativeStatus* admission,
                                    unsigned int* failure)
{
  const auto task = std::size_t(blockIdx.x);
  const auto channel = task % layout.determinant_count;
  const auto target = (task / layout.determinant_count) % layout.electron_count;
  const auto configuration = task / (layout.determinant_count * layout.electron_count);
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  const bool selected = admission[configuration] == device_determinant::DerivativeStatus::AVAILABLE &&
      input.normalized_weights[configuration * layout.determinant_count + channel] != 0.0;
  const auto* inverse = input.inverses + (configuration * layout.determinant_count + channel) * matrix_elements;
  const auto* laplacian = input.laplacians + configuration * layout.laplacian_configuration_stride +
      (target * layout.determinant_count + channel) * matrix_elements;
  CompensatedSumBinary64 sum;
  if (selected)
    for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x)
      sum.add(inverse[element] * laplacian[(element % layout.matrix_size) * layout.matrix_size +
                                          element / layout.matrix_size]);
  __shared__ double partials[trace_block_size];
  partials[threadIdx.x] = sum.value();
  if (!device_determinant::isFiniteBinary64(partials[threadIdx.x]))
    markNonfinite(failure + configuration);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    CompensatedSumBinary64 combined;
    for (unsigned int part = 0; part < trace_block_size; ++part)
      combined.add(partials[part]);
    const auto value = combined.value();
    if (!device_determinant::isFiniteBinary64(value))
      markNonfinite(failure + configuration);
    stats[task * 7 + 6] = device_determinant::isFiniteBinary64(value) ? value : 0.0;
  }
}

// Retained derivatives feed optimizer updates: preserve the canonical row-major
// compensated trace order while keeping independent products and outputs parallel.
__global__ void canonicalGradientStatistics(SpatialLayout layout, SpatialTraceInputs input,
                                            std::size_t first_task, const double* products,
                                            double* stats,
                                            const device_determinant::DerivativeStatus* admission,
                                            unsigned int* failure)
{
  const auto slot = std::size_t(blockIdx.x);
  const auto task = first_task + slot;
  const auto channel = task % layout.determinant_count;
  const auto lane = (task / layout.determinant_count) % layout.gradient_lanes;
  const auto configuration = task / (layout.determinant_count * layout.gradient_lanes);
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  double direct = 0.0, square = 0.0;
  if (admission[configuration] == device_determinant::DerivativeStatus::AVAILABLE &&
      input.normalized_weights[configuration * layout.determinant_count + channel] != 0.0)
  {
    const auto* inverse = input.inverses + (configuration * layout.determinant_count + channel) * matrix_elements;
    const auto* gradient = input.gradients + configuration * layout.gradient_configuration_stride +
        (lane * layout.determinant_count + channel) * matrix_elements;
    const auto* product = products + slot * matrix_elements;
    direct = device_determinant::traceProductBinary64(inverse, gradient, layout.matrix_size);
    square = device_determinant::traceProductBinary64(product, product, layout.matrix_size);
  }
  if (!device_determinant::isFiniteBinary64(direct) || !device_determinant::isFiniteBinary64(square))
    markNonfinite(failure + configuration);
  const auto offset = ((configuration * layout.electron_count + lane / 3) * layout.determinant_count + channel) * 7;
  stats[offset + lane % 3] = device_determinant::isFiniteBinary64(direct) ? direct : 0.0;
  stats[offset + 3 + lane % 3] = device_determinant::isFiniteBinary64(square) ? square : 0.0;
}

// Preserve the canonical compensated inverse/Laplacian trace for each output.
__global__ void canonicalLaplacianStatistics(SpatialLayout layout, SpatialTraceInputs input,
                                             double* stats,
                                             const device_determinant::DerivativeStatus* admission,
                                             unsigned int* failure)
{
  const auto task = std::size_t(blockIdx.x);
  const auto channel = task % layout.determinant_count;
  const auto target = (task / layout.determinant_count) % layout.electron_count;
  const auto configuration = task / (layout.determinant_count * layout.electron_count);
  const auto matrix_elements = layout.matrix_size * layout.matrix_size;
  double value = 0.0;
  if (admission[configuration] == device_determinant::DerivativeStatus::AVAILABLE &&
      input.normalized_weights[configuration * layout.determinant_count + channel] != 0.0)
  {
    const auto* inverse = input.inverses + (configuration * layout.determinant_count + channel) * matrix_elements;
    const auto* laplacian = input.laplacians + configuration * layout.laplacian_configuration_stride +
        (target * layout.determinant_count + channel) * matrix_elements;
    value = device_determinant::traceProductBinary64(inverse, laplacian, layout.matrix_size);
  }
  if (!device_determinant::isFiniteBinary64(value))
    markNonfinite(failure + configuration);
  stats[task * 7 + 6] = device_determinant::isFiniteBinary64(value) ? value : 0.0;
}

__global__ void combineStatistics(SpatialLayout layout, SpatialTraceInputs input, const double* stats,
                                  const unsigned int* failure, double* gradient, double* lap_ratio,
                                  double* lap_log, device_determinant::DerivativeStatus* status)
{
  const auto configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= layout.configuration_count)
    return;
  auto admission = static_cast<device_determinant::DerivativeStatus>(failure[configuration]);
  if (admission == device_determinant::DerivativeStatus::AVAILABLE)
    for (std::size_t target = 0; target < layout.electron_count; ++target)
    {
      CompensatedSumBinary64 combined_gradient[3], combined_laplacian;
      for (std::size_t channel = 0; channel < layout.determinant_count; ++channel)
      {
        const auto weight = input.normalized_weights[configuration * layout.determinant_count + channel];
        if (weight == 0.0)
          continue;
        const auto* channel_stats = stats + ((configuration * layout.electron_count + target) *
                                             layout.determinant_count + channel) * 7;
        CompensatedSumBinary64 bracket;
        bracket.add(channel_stats[6]);
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const auto trace = channel_stats[dimension];
          combined_gradient[dimension].add(weight * trace);
          bracket.add(trace * trace);
          bracket.add(-channel_stats[3 + dimension]);
        }
        combined_laplacian.add(weight * bracket.value());
      }
      CompensatedSumBinary64 squared_gradient;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const auto value = combined_gradient[dimension].value();
        gradient[configuration * layout.gradient_lanes + 3 * target + dimension] = value;
        squared_gradient.add(value * value);
        if (!device_determinant::isFiniteBinary64(value))
          admission = device_determinant::DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
      }
      const auto ratio = combined_laplacian.value();
      const auto log = ratio - squared_gradient.value();
      lap_ratio[configuration * layout.electron_count + target] = ratio;
      lap_log[configuration * layout.electron_count + target] = log;
      if (!device_determinant::isFiniteBinary64(ratio) || !device_determinant::isFiniteBinary64(log))
        admission = device_determinant::DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
    }
  if (admission != device_determinant::DerivativeStatus::AVAILABLE)
  {
    for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
      gradient[configuration * layout.gradient_lanes + lane] = 0.0;
    for (std::size_t target = 0; target < layout.electron_count; ++target)
      lap_ratio[configuration * layout.electron_count + target] =
          lap_log[configuration * layout.electron_count + target] = 0.0;
  }
  status[configuration] = admission;
}

} // namespace

SpatialTraceWorkspacePlan planSpatialTraceWorkspace(SpatialLayout capacity, std::size_t budget_bytes)
{
  capacity = checkedLayout(capacity);
  if (budget_bytes == 0 || budget_bytes > maximum_workspace)
    throw std::invalid_argument("Spatial trace explicit workspace budget must be within 8 MiB");
  const auto targets = multiply(capacity.configuration_count, capacity.electron_count);
  const auto stats = checkedDeviceBytes<double>(multiply(multiply(targets, capacity.determinant_count), 7));
  const auto outputs = checkedDeviceBytes<double>(multiply(targets, 5));
  const auto statuses = add(checkedDeviceBytes<unsigned int>(capacity.configuration_count),
      checkedDeviceBytes<device_determinant::DerivativeStatus>(capacity.configuration_count));
  const auto metadata = add(add(stats, outputs), statuses);
  const auto matrix_bytes = checkedDeviceBytes<double>(multiply(capacity.matrix_size, capacity.matrix_size));
  if (metadata >= budget_bytes || matrix_bytes > budget_bytes - metadata)
    throw std::length_error("Spatial trace budget cannot admit one product slot and its metadata");
  const auto tasks = multiply(multiply(capacity.configuration_count, capacity.gradient_lanes), capacity.determinant_count);
  const auto slots = std::min(tasks, (budget_bytes - metadata) / matrix_bytes);
  const auto product_bytes = multiply(slots, matrix_bytes);
  (void)blocks(multiply(slots, multiply(capacity.matrix_size, capacity.matrix_size)));
  const auto inverse_elements = multiply(multiply(capacity.configuration_count, capacity.determinant_count),
      multiply(capacity.matrix_size, capacity.matrix_size));
  (void)blocks(multiply(inverse_elements, add(add(1, capacity.gradient_lanes), capacity.electron_count)));
  if (multiply(targets, capacity.determinant_count) > std::numeric_limits<unsigned int>::max() ||
      slots > std::numeric_limits<unsigned int>::max())
    throw std::length_error("Spatial trace task grid exceeds CUDA/HIP x dimension");
  return {capacity, slots, product_bytes, metadata, add(product_bytes, metadata),
          2 * trace_block_size * sizeof(double)};
}

PreparedDeviceSpatialTraces::PreparedDeviceSpatialTraces(SpatialLayout capacity, Stream stream,
                                                       std::size_t budget_bytes, bool canonical_statistics)
    : plan_(planSpatialTraceWorkspace(capacity, budget_bytes)), stream_(stream),
      canonical_statistics_(canonical_statistics)
{
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device_ordinal_), "spatial trace owning device");
#else
  checkRuntime(cudaGetDevice(&device_ordinal_), "spatial trace owning device");
#endif
  const auto targets = multiply(plan_.capacity.configuration_count, plan_.capacity.electron_count);
  products_.reserveNamed(plan_.product_bytes / sizeof(double), "SpatialTraces::products", __FILE__, __LINE__);
  stats_.reserveNamed(multiply(multiply(targets, plan_.capacity.determinant_count), 7),
      "SpatialTraces::stats", __FILE__, __LINE__);
  gradient_.reserveNamed(multiply(targets, 3), "SpatialTraces::gradient", __FILE__, __LINE__);
  lap_ratio_.reserveNamed(targets, "SpatialTraces::lap_ratio", __FILE__, __LINE__);
  lap_log_.reserveNamed(targets, "SpatialTraces::lap_log", __FILE__, __LINE__);
  failure_.reserveNamed(plan_.capacity.configuration_count, "SpatialTraces::failure", __FILE__, __LINE__);
  status_.reserveNamed(plan_.capacity.configuration_count, "SpatialTraces::status", __FILE__, __LINE__);
}

PreparedDeviceSpatialTraces::~PreparedDeviceSpatialTraces()
{
  // DeviceBuffer destruction follows this drain; the borrowed stream must outlive us.
  (void)synchronize(stream_);
}

void PreparedDeviceSpatialTraces::validateCurrentDevice() const
{
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "spatial trace current device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "spatial trace current device");
#endif
  if (ordinal != device_ordinal_)
    throw std::logic_error("Spatial traces require their owning device to be current");
}

SpatialTraceDeviceResult PreparedDeviceSpatialTraces::evaluate(SpatialLayout layout, SpatialTraceInputs input)
{
  invalidate();
  validateCurrentDevice();
  layout = checkedLayout(layout);
  if (layout.matrix_size != plan_.capacity.matrix_size || layout.determinant_count != plan_.capacity.determinant_count ||
      layout.configuration_count > plan_.capacity.configuration_count || layout.electron_count > plan_.capacity.electron_count)
    throw std::invalid_argument("Spatial trace request exceeds its prepared shape");
  if (!input.factorizations || !input.combinations || !input.normalized_weights ||
      !input.inverses || !input.gradients || !input.laplacians)
    throw std::invalid_argument("Spatial trace request has missing input storage");
  if (generation_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Spatial trace result generation exhausted");
  const auto matrices = multiply(multiply(layout.configuration_count, layout.determinant_count),
      multiply(layout.matrix_size, layout.matrix_size));
  const auto gradient_count = multiply(matrices, layout.gradient_lanes);
  const auto laplacian_count = multiply(matrices, layout.electron_count);
  const auto all_inputs = add(add(matrices, gradient_count), laplacian_count);
  const auto tasks = multiply(multiply(layout.configuration_count, layout.gradient_lanes), layout.determinant_count);
  // status_ is immutable metadata admission through the compute kernels. Only
  // failure_ receives atomic updates; final combination publishes status_ last.
  admitMetadata<<<blocks(layout.configuration_count), trace_block_size, 0, stream_>>>(
      layout, input, failure_.data(), status_.data());
  checkRuntime(peekError(), "spatial trace metadata admission");
  scanInputs<<<blocks(all_inputs), trace_block_size, 0, stream_>>>(
      layout, input, matrices, gradient_count, all_inputs, status_.data(), failure_.data());
  checkRuntime(peekError(), "spatial trace parallel finite admission");
  for (std::size_t first = 0; first < tasks; first += plan_.product_slots)
  {
    const auto count = std::min(plan_.product_slots, tasks - first);
    matrixProducts<<<blocks(multiply(count, multiply(layout.matrix_size, layout.matrix_size))),
                     trace_block_size, 0, stream_>>>(
        layout, input, first, count, products_.data(), status_.data(), failure_.data());
    checkRuntime(peekError(), "spatial trace parallel compensated product");
    if (canonical_statistics_)
      canonicalGradientStatistics<<<static_cast<unsigned int>(count), 1, 0, stream_>>>(
          layout, input, first, products_.data(), stats_.data(), status_.data(), failure_.data());
    else
      gradientStatistics<<<static_cast<unsigned int>(count), trace_block_size, 0, stream_>>>(
          layout, input, first, products_.data(), stats_.data(), status_.data(), failure_.data());
    checkRuntime(peekError(), "spatial trace gradient and signed square reductions");
  }
  const auto laplacian_tasks = multiply(multiply(layout.configuration_count, layout.electron_count), layout.determinant_count);
  if (canonical_statistics_)
    canonicalLaplacianStatistics<<<static_cast<unsigned int>(laplacian_tasks), 1, 0, stream_>>>(
        layout, input, stats_.data(), status_.data(), failure_.data());
  else
    laplacianStatistics<<<static_cast<unsigned int>(laplacian_tasks), trace_block_size, 0, stream_>>>(
        layout, input, stats_.data(), status_.data(), failure_.data());
  checkRuntime(peekError(), "spatial trace Laplacian reduction");
  combineStatistics<<<blocks(layout.configuration_count), trace_block_size, 0, stream_>>>(
      layout, input, stats_.data(), failure_.data(), gradient_.data(), lap_ratio_.data(), lap_log_.data(), status_.data());
  checkRuntime(peekError(), "spatial trace channel combination");
  live_ = layout;
  ++generation_;
  submitted_ = true;
  return result();
}

SpatialTraceDeviceResult PreparedDeviceSpatialTraces::result() const
{
  if (!submitted_)
    throw std::logic_error("Spatial traces have no submitted result");
  validateCurrentDevice();
  return {gradient_.data(), lap_ratio_.data(), lap_log_.data(), status_.data(), live_, generation_};
}

void PreparedDeviceSpatialTraces::downloadResults(std::vector<double>& gradient, std::vector<double>& lap_ratio,
    std::vector<double>& lap_log, std::vector<device_determinant::DerivativeStatus>& status)
{
  if (!submitted_)
    throw std::logic_error("Spatial traces have no submitted result");
  if (gradient.size() != live_.configuration_count * live_.gradient_lanes ||
      lap_ratio.size() != live_.configuration_count * live_.electron_count ||
      lap_log.size() != lap_ratio.size() || status.size() != live_.configuration_count)
    throw std::invalid_argument("Spatial trace result destination extent mismatch");
  try
  {
    validateCurrentDevice();
    checkRuntime(synchronize(stream_), "spatial trace completion");
    gradient_.download(gradient.data(), gradient.size());
    lap_ratio_.download(lap_ratio.data(), lap_ratio.size());
    lap_log_.download(lap_log.data(), lap_log.size());
    status_.download(status.data(), status.size());
  }
  catch (...)
  {
    invalidate();
    throw;
  }
}

} // namespace qmcplusplus::psiformer::device
