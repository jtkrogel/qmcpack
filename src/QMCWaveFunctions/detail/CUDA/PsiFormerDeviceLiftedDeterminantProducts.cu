//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceLiftedDeterminantProducts.cu
 * @brief FP64 batched products with compensated scalar trace contractions.
 */

#include "PsiFormerDeviceLiftedDeterminantProducts.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

namespace qmcplusplus::psiformer::device
{
namespace
{
using Status = device_determinant::DerivativeStatus;
using Sum = device_determinant::CompensatedSumBinary64;
constexpr unsigned int threads = 128;

/// Name each row-major product without changing the reference factor algebra.
enum class ProductStage
{
  INVERSE_GRADIENT,
  GRADIENT_INVERSE,
  GRADIENT_SQUARE,
  GRADIENT_SQUARE_INVERSE,
  INVERSE_LAPLACIAN,
  LAPLACIAN_INVERSE
};

/// Preserve the reference's cross, quadratic-gradient and matrix-Laplacian order.
enum class LaplacianTerm
{
  CROSS_GRADIENT,
  GRADIENT_SQUARE,
  MATRIX_LAPLACIAN
};

/// Check numerical extents before reserving storage or constructing a device grid.
std::size_t mul(std::size_t a, std::size_t b)
{ return spatial_detail::checkedProduct(a, b, "Lifted product extent overflow"); }

/// Check complete byte sums, including pointer arrays and admission records.
std::size_t add(std::size_t a, std::size_t b)
{ return spatial_detail::checkedAdd(a, b, "Lifted product storage overflow"); }

/// Reject a grid exceeding the existing CUDA/HIP launch integer seam.
unsigned int blocks(std::size_t n)
{
  const auto count = n / threads + (n % threads != 0);
  if (count > std::numeric_limits<unsigned int>::max())
    throw std::length_error("Lifted product grid exceeds device x dimension");
  return static_cast<unsigned int>(count);
}

/// Surface launch failures immediately without synchronizing intermediate products.
void checkLaunch()
{
#ifdef QMC_CUDA2HIP
  checkRuntime(hipPeekAtLastError(), "lifted product kernel");
#else
  checkRuntime(cudaPeekAtLastError(), "lifted product kernel");
#endif
}

/// Initialize only the permanent inactive-input matrix during preparation.
__global__ void zeroMatrix(double* matrix, std::size_t n)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) matrix[i] = 0;
}

/// Admit channel metadata and clear the caller's logical output without padding.
__global__ void initialize(LiftedDeterminantProductInputs in, double* out, Status* status, unsigned int* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto& m = in.layout.matrices;
  if (i < m.plane_count * m.element_count)
    out[(i / m.element_count) * m.plane_stride + i % m.element_count] = 0;
  if (i == 0)
  {
    *failure = 0;
    *status = device_determinant::derivativeAvailability(in.factorizations, *in.combination,
        in.weights, in.layout.determinant_count);
    if (*status == Status::AVAILABLE)
    {
      for (std::size_t channel = 0; channel < in.layout.determinant_count; ++channel)
        if (!device_determinant::isFiniteBinary64(in.weights[channel])) *failure = 1;
      for (std::size_t lane = 0; lane < m.gradient_lanes; ++lane)
        if (!device_determinant::isFiniteBinary64(in.log_gradient[lane])) *failure = 1;
      for (std::size_t e = 0; e < m.laplacian_lanes; ++e)
        if (!device_determinant::isFiniteBinary64(in.log_laplacian[e])) *failure = 1;
    }
  }
}

/// Check only active channel inputs; masked unavailable inverses are never read.
__global__ void validateInputs(LiftedDeterminantProductInputs in, std::size_t first, std::size_t count,
    std::size_t electron, const Status* status, unsigned int* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto nn = in.layout.matrix_size * in.layout.matrix_size;
  if (i >= count * nn || *status != Status::AVAILABLE) return;
  const auto channel = first + i / nn, element = i % nn;
  if (in.weights[channel] == 0) return;
  const auto& m = in.layout.matrices;
  bool finite = device_determinant::isFiniteBinary64(in.inverses[channel * nn + element]);
  for (std::size_t a = 0; a < 3; ++a)
    finite = finite && device_determinant::isFiniteBinary64(
        in.orbitals[m.uncheckedGradientOffset(0, 3 * electron + a, channel * nn + element)]);
  finite = finite && device_determinant::isFiniteBinary64(
      in.orbitals[m.uncheckedLaplacianOffset(0, electron, channel * nn + element)]);
  if (!finite) atomicExch(failure, 1u);
}

/// Select row-major operands for A=I*G, Q=A*I, A*A, (A*A)*I, I*L and (I*L)*I.
__global__ void pointers(LiftedDeterminantProductInputs in, std::size_t first, std::size_t count,
    std::size_t electron, std::size_t slots, ProductStage stage, std::size_t dimension, double* matrices,
    const Status* status, const unsigned int* failure, const double** left, const double** right, double** target)
{
  const auto batch = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool gradient_batch = stage == ProductStage::INVERSE_GRADIENT || stage == ProductStage::GRADIENT_INVERSE;
  const auto batches = gradient_batch ? 3 * count : count;
  if (batch >= batches) return;
  const auto slot = batch % count, a = gradient_batch ? batch / count : dimension;
  const auto nn = in.layout.matrix_size * in.layout.matrix_size, channel = first + slot;
  auto* A = matrices + (a * slots + slot) * nn;
  auto* Q = matrices + (3 * slots + a * slots + slot) * nn;
  const auto* I = in.inverses + channel * nn;
  if (stage == ProductStage::INVERSE_GRADIENT)
  {
    left[batch] = I;
    right[batch] = in.orbitals + in.layout.matrices.uncheckedGradientOffset(0, 3 * electron + a, channel * nn);
    target[batch] = A;
  }
  else if (stage == ProductStage::GRADIENT_INVERSE || stage == ProductStage::GRADIENT_SQUARE_INVERSE ||
           stage == ProductStage::LAPLACIAN_INVERSE)
  {
    left[batch] = stage == ProductStage::GRADIENT_INVERSE ? A : Q;
    right[batch] = I;
    target[batch] = stage == ProductStage::GRADIENT_INVERSE ? Q : A;
  }
  else if (stage == ProductStage::GRADIENT_SQUARE)
  {
    left[batch] = A;
    right[batch] = A;
    target[batch] = Q;
  }
  else
  {
    left[batch] = I;
    right[batch] = in.orbitals + in.layout.matrices.uncheckedLaplacianOffset(0, electron, channel * nn);
    target[batch] = Q;
  }
  if (*status != Status::AVAILABLE || *failure || in.weights[channel] == 0)
    left[batch] = right[batch] = matrices + 6 * slots * nn;
}

/// Reduce tr(I*G), signed tr((I*G)^2) and tr(I*L), matching the independent oracle.
__global__ void statistics(LiftedDeterminantProductInputs in, std::size_t first, std::size_t count,
    std::size_t electron, std::size_t slots, const double* matrices, double* stats,
    const Status* status, unsigned int* failure)
{
  const auto slot = std::size_t(blockIdx.x) / 4, a = std::size_t(blockIdx.x) % 4;
  const auto channel = first + slot, n = in.layout.matrix_size, nn = n * n;
  Sum direct, square;
  if (*status == Status::AVAILABLE && in.weights[channel] != 0)
    for (std::size_t i = threadIdx.x; i < nn; i += blockDim.x)
    {
      const auto t = (i % n) * n + i / n;
      const auto offset = a < 3 ? in.layout.matrices.uncheckedGradientOffset(0, 3 * electron + a, channel * nn + t)
                                : in.layout.matrices.uncheckedLaplacianOffset(0, electron, channel * nn + t);
      direct.add(in.inverses[channel * nn + i] * in.orbitals[offset]);
      if (a < 3)
      {
        const auto* A = matrices + (a * slots + slot) * nn;
        square.add(A[i] * A[t]);
      }
    }
  __shared__ double d[threads], q[threads];
  d[threadIdx.x] = direct.value(); q[threadIdx.x] = square.value();
  __syncthreads();
  if (threadIdx.x == 0)
  {
    Sum ds, qs;
    for (unsigned int i = 0; i < threads; ++i) { ds.add(d[i]); qs.add(q[i]); }
    const double dv = ds.value(), qv = qs.value();
    if (!device_determinant::isFiniteBinary64(dv) || !device_determinant::isFiniteBinary64(qv)) atomicExch(failure, 1u);
    stats[7 * slot + (a < 3 ? a : 6)] = dv;
    if (a < 3) stats[7 * slot + 3 + a] = qv;
  }
}

/// Write factor values/gradients and the initial Laplacian scalar-inverse term.
__global__ void factors(LiftedDeterminantProductInputs in, std::size_t first, std::size_t count,
    std::size_t electron, std::size_t slots, const double* matrices, const double* stats,
    const Status* status, const unsigned int* failure, double* out)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto n = in.layout.matrix_size, nn = n * n;
  if (i >= count * nn || *status != Status::AVAILABLE || *failure) return;
  const auto slot = i / nn, element = i % nn, transpose = (element % n) * n + element / n;
  const auto channel = first + slot, output = channel * nn + element;
  const double w = in.weights[channel];
  if (w == 0) return;
  const double inverse = in.inverses[channel * nn + transpose];
  if (electron == 0) out[in.layout.matrices.uncheckedValueOffset(0, output)] = w * inverse;
  double channel_lap = stats[7 * slot + 6], squared = 0;
  for (std::size_t a = 0; a < 3; ++a)
  {
    const double difference = stats[7 * slot + a] - in.log_gradient[3 * electron + a];
    channel_lap -= stats[7 * slot + 3 + a]; squared += difference * difference;
    out[in.layout.matrices.uncheckedGradientOffset(0, 3 * electron + a, output)] =
        w * difference * inverse - w * matrices[(3 * slots + a * slots + slot) * nn + transpose];
  }
  out[in.layout.matrices.uncheckedLaplacianOffset(0, electron, output)] =
      w * (channel_lap - in.log_laplacian[electron] + squared) * inverse;
}

/// Accumulate each dimension's cross and cubic terms in reference order, then I*L*I.
__global__ void updateLaplacian(LiftedDeterminantProductInputs in, std::size_t first, std::size_t count,
    std::size_t electron, std::size_t slots, std::size_t a, LaplacianTerm term, const double* matrices,
    const double* stats, const Status* status, const unsigned int* failure, double* out)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto n = in.layout.matrix_size, nn = n * n;
  if (i >= count * nn || *status != Status::AVAILABLE || *failure) return;
  const auto slot = i / nn, element = i % nn, channel = first + slot;
  const double w = in.weights[channel];
  if (w == 0) return;
  const auto transpose = (element % n) * n + element / n;
  const double value = matrices[((term == LaplacianTerm::CROSS_GRADIENT ? 3 * slots : 0) + a * slots + slot) * nn + transpose];
  auto& result = out[in.layout.matrices.uncheckedLaplacianOffset(0, electron, channel * nn + element)];
  if (term == LaplacianTerm::CROSS_GRADIENT) result -= 2.0 * (w * (stats[7 * slot + a] - in.log_gradient[3 * electron + a])) * value;
  else if (term == LaplacianTerm::GRADIENT_SQUARE) result += 2.0 * w * value;
  else result -= w * value;
}

/// Scan every published logical factor and convert overflow into whole-output failure.
__global__ void scan(LiftedDeterminantLayout layout, const double* out, unsigned int* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < layout.matrices.plane_count * layout.matrices.element_count &&
      !device_determinant::isFiniteBinary64(out[(i / layout.matrices.element_count) * layout.matrices.plane_stride +
          i % layout.matrices.element_count])) atomicExch(failure, 1u);
}

/// Publish one typed status and erase all logical factors after any failed contraction.
__global__ void finish(LiftedDeterminantLayout layout, double* out, Status* status, const unsigned int* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < layout.matrices.plane_count * layout.matrices.element_count && (*failure || *status != Status::AVAILABLE))
    out[(i / layout.matrices.element_count) * layout.matrices.plane_stride + i % layout.matrices.element_count] = 0;
  if (i == 0 && *failure) *status = Status::NONFINITE_INPUT_OR_RESULT;
}

} // namespace

// Include every matrix, pointer, statistic and status allocation in admission.
LiftedDeterminantProductPlan planLiftedDeterminantProducts(std::size_t n, std::size_t slots, std::size_t budget)
{
  if (n == 0 || slots == 0 || slots > 16) throw std::invalid_argument("Lifted product dimensions or channel slots are invalid");
  attention_detail::requireBlasInteger(n, "Lifted product size exceeds BLAS ABI");
  const auto nn = mul(n, n), matrices = mul(add(mul(6, slots), 1), nn);
  auto bytes = checkedDeviceBytes<double>(add(matrices, mul(7, slots)));
  bytes = add(bytes, checkedDeviceBytes<const double*>(mul(6, slots)));
  bytes = add(bytes, checkedDeviceBytes<double*>(mul(3, slots)));
  bytes = add(bytes, add(sizeof(Status), sizeof(unsigned int)));
  if (bytes > budget) throw std::length_error("Lifted product numerical storage exceeds the explicit budget");
  return {n, slots, nn, matrices, bytes};
}

// Reserve fixed scratch once; BLAS library storage remains an explicit exclusion.
PreparedDeviceLiftedDeterminantProducts::PreparedDeviceLiftedDeterminantProducts(std::size_t n, std::size_t slots,
    std::size_t budget, compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
    : plan_(planLiftedDeterminantProducts(n, slots, budget)), queue_(queue), blas_(blas)
{
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device_ordinal_), "lifted product owning device");
#else
  checkRuntime(cudaGetDevice(&device_ordinal_), "lifted product owning device");
#endif
  matrices_.reserve(plan_.matrix_storage_elements); statistics_.reserve(7 * slots);
  left_.reserve(3 * slots); right_.reserve(3 * slots); target_.reserve(3 * slots);
  failure_.reserve(1); status_.reserve(1);
  zeroMatrix<<<blocks(plan_.matrix_elements), threads, 0, queue_.getNative()>>>(
      matrices_.data() + 6 * slots * plan_.matrix_elements, plan_.matrix_elements);
  checkLaunch();
}

// Reverse operand order because row-major C=A*B is column-major C^T=B^T*A^T.
void PreparedDeviceLiftedDeterminantProducts::product(std::size_t batches)
{
  const int n = static_cast<int>(plan_.matrix_size);
  compute::BLAS::gemm_batched(blas_, 'N', 'N', n, n, n, 1.0, right_.data(), n, left_.data(), n,
      0.0, target_.data(), n, static_cast<int>(batches));
}

// Replay targets through fixed channel slots and publish only a fully scanned result.
device_determinant::DerivativeStatus PreparedDeviceLiftedDeterminantProducts::evaluate(
    const LiftedDeterminantProductInputs& in, double* out)
{
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "lifted product current device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "lifted product current device");
#endif
  if (ordinal != device_ordinal_) throw std::logic_error("Lifted product owner changed device");
  const auto& m = in.layout.matrices;
  validateSpatialJetLayout(m); validateSpatialJetLayout(in.layout.roots);
  if (m.configuration_count != 1 || m.mode != SpatialJetMode::FULL_VGL || m.laplacian_lanes == 0 ||
      !haveMatchingSpatialPlanes(m, in.layout.roots) || in.layout.roots.element_count != 1 ||
      in.layout.matrix_size != plan_.matrix_size || m.electron_count != plan_.matrix_size ||
      in.layout.determinant_count == 0 || m.element_count != mul(in.layout.determinant_count, plan_.matrix_elements) ||
      !in.factorizations || !in.combination || !in.weights || !in.inverses || !in.orbitals ||
      !in.log_gradient || !in.log_laplacian || !out || out == in.orbitals || out == in.inverses ||
      out == in.weights || out == in.log_gradient || out == in.log_laplacian)
    throw std::invalid_argument("Lifted product input/output layout or storage is incompatible");
  const auto stream = queue_.getNative();
  const auto slots = plan_.channel_slots;
  const auto logical = mul(m.plane_count, m.element_count);
  initialize<<<blocks(logical), threads, 0, stream>>>(in, out, status_.data(), failure_.data()); checkLaunch();
  for (std::size_t first = 0; first < in.layout.determinant_count; first += slots)
    for (std::size_t e = 0; e < m.laplacian_lanes; ++e)
    {
      const auto count = std::min(slots, in.layout.determinant_count - first);
      const auto grid = blocks(mul(count, plan_.matrix_elements));
      validateInputs<<<grid, threads, 0, stream>>>(in, first, count, e, status_.data(), failure_.data()); checkLaunch();
      const auto multiply = [&](ProductStage stage, std::size_t a) {
        const bool gradient_batch = stage == ProductStage::INVERSE_GRADIENT || stage == ProductStage::GRADIENT_INVERSE;
        const auto batches = gradient_batch ? 3 * count : count;
        pointers<<<blocks(batches), threads, 0, stream>>>(in, first, count, e, slots, stage, a, matrices_.data(),
            status_.data(), failure_.data(), left_.data(), right_.data(), target_.data()); checkLaunch(); product(batches);
      };
      multiply(ProductStage::INVERSE_GRADIENT, 0);
      statistics<<<static_cast<unsigned int>(4 * count), threads, 0, stream>>>(in, first, count, e, slots,
          matrices_.data(), statistics_.data(), status_.data(), failure_.data()); checkLaunch();
      multiply(ProductStage::GRADIENT_INVERSE, 0);
      factors<<<grid, threads, 0, stream>>>(in, first, count, e, slots, matrices_.data(), statistics_.data(),
          status_.data(), failure_.data(), out); checkLaunch();
      const auto update = [&](LaplacianTerm term, std::size_t a) {
        updateLaplacian<<<grid, threads, 0, stream>>>(in, first, count, e, slots, a, term, matrices_.data(),
            statistics_.data(), status_.data(), failure_.data(), out); checkLaunch();
      };
      for (std::size_t a = 0; a < 3; ++a)
      {
        update(LaplacianTerm::CROSS_GRADIENT, a);
        multiply(ProductStage::GRADIENT_SQUARE, a);
        multiply(ProductStage::GRADIENT_SQUARE_INVERSE, a);
        update(LaplacianTerm::GRADIENT_SQUARE, a);
      }
      multiply(ProductStage::INVERSE_LAPLACIAN, 0);
      multiply(ProductStage::LAPLACIAN_INVERSE, 0);
      update(LaplacianTerm::MATRIX_LAPLACIAN, 0);
    }
  scan<<<blocks(logical), threads, 0, stream>>>(in.layout, out, failure_.data()); checkLaunch();
  finish<<<blocks(logical), threads, 0, stream>>>(in.layout, out, status_.data(), failure_.data()); checkLaunch();
  queue_.sync();
  Status result;
  status_.download(&result, 1);
  return result;
}

} // namespace qmcplusplus::psiformer::device
