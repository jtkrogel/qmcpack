//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceInverseCertificate.h"
#include <cstdint>
#include <limits>
namespace qmcplusplus::psiformer::device::inverse_certificate
{
namespace
{
constexpr unsigned int block_size = 128;

/// One block owns one head; threads bound independent rows and max-reduce them.
template<class T, bool complex>
__global__ void certifyKernel(const T* A, const T* B, std::size_t n, Result* output)
{
  const auto head = std::size_t(blockIdx.x), offset = head * n * n;
  __shared__ double maxima[block_size];
  __shared__ unsigned int bad_input, bad_bound;
  if (threadIdx.x == 0)
    bad_input = bad_bound = 0;
  __syncthreads();
  double maximum = 0;
  for (std::size_t row = threadIdx.x; row < n; row += block_size)
  {
    std::uint32_t status = CERTIFIED;
    double bound;
    if constexpr (complex)
      bound = math::complexRow(A + offset, B + offset, n, row, status);
    else
      bound = math::realRow(A + offset, B + offset, n, row, status);
    if (status == NONFINITE_INPUT)
      atomicExch(&bad_input, 1u);
    else if (status != CERTIFIED)
      atomicExch(&bad_bound, 1u);
    maximum = maximum > bound ? maximum : bound;
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned int stride = block_size / 2; stride; stride /= 2)
  {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] = maxima[threadIdx.x] > maxima[threadIdx.x + stride] ?
          maxima[threadIdx.x] : maxima[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[head] = math::finish(maxima[0], bad_input ? NONFINITE_INPUT : bad_bound ? UNCERTIFIABLE : CERTIFIED);
}

/// Detect output/input overlap without overflowing pointer-end arithmetic.
bool overlaps(const void* a, std::size_t an, const void* b, std::size_t bn)
{
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  return x <= y ? y - x < an : x - y < bn;
}

/// Reject malformed layouts before launch; no allocation or certificate authority on error.
template<class T, bool complex>
Error launch(Stream stream, const T* A, const T* B, std::size_t count, std::size_t n, Result* output)
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  if (!A || !B || !output || !count || !n || count > 2147483647u || n > max / n)
    return invalid_value;
  const auto nn = n * n;
  if (count > max / nn || count * nn > max / sizeof(T) || count > max / sizeof(Result))
    return invalid_value;
  const auto bytes = count * nn * sizeof(T), out_bytes = count * sizeof(Result);
  if (overlaps(A, bytes, output, out_bytes) || overlaps(B, bytes, output, out_bytes))
    return invalid_value;
  certifyKernel<T, complex><<<static_cast<unsigned int>(count), block_size, 0, stream>>>(A, B, n, output);
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Certify row-major real FP64 A/B with the fixed AB residual orientation.
Error launchReal(Stream stream, const double* A, const double* B, std::size_t count, std::size_t n, Result* output)
{
  return launch<double, false>(stream, A, B, count, n, output);
}

/// Certify complex FP64 A/B using explicit real/imaginary operation bounds.
Error launchComplex(Stream stream, const cuDoubleComplex* A, const cuDoubleComplex* B,
                    std::size_t count, std::size_t n, Result* output)
{
  return launch<cuDoubleComplex, true>(stream, A, B, count, n, output);
}
} // namespace qmcplusplus::psiformer::device::inverse_certificate
