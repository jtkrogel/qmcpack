//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_INVERSE_CERTIFICATE_MATH_H
#define QMCPLUSPLUS_PSIFORMER_INVERSE_CERTIFICATE_MATH_H
#include "PsiFormerDeviceMath.h"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QMC_PF_INV_HD __host__ __device__
#else
#define QMC_PF_INV_HD
#endif
namespace qmcplusplus::psiformer::device::inverse_certificate
{
inline constexpr double relative_budget = 0x1p-24; // Binary32 unit roundoff; never a tunable tolerance.

/// Distinguish a certificate from rejected, malformed or uncomputed diagnostics.
enum Status : std::uint32_t
{
  UNINITIALIZED = 0,
  CERTIFIED = 1,
  BOUND_EXCEEDED = 2,
  NONFINITE_INPUT = 3,
  UNCERTIFIABLE = 4
};

/// Compact per-head diagnostics; default storage carries no validity authority.
struct Result
{
  double rho_upper = 0, relative_inverse_error_upper = 0;
  std::uint32_t status = UNINITIALIZED, reserved = 0;
};
static_assert(sizeof(Result) == 24 && std::is_trivially_copyable_v<Result>);

/// Admit only recognized, finite fixed-budget inverse diagnostics, never unknown bytes.
QMC_PF_INV_HD inline bool isCertified(const Result& r) noexcept
{
  return r.status == CERTIFIED && r.reserved == 0 && device_math::isFiniteBinary64(r.rho_upper) &&
      device_math::isFiniteBinary64(r.relative_inverse_error_upper) && r.rho_upper >= 0 && r.rho_upper < 1 &&
      r.relative_inverse_error_upper >= r.rho_upper && r.relative_inverse_error_upper <= relative_budget;
}

/// Bound scalar work, not timing. Inner terms cover center products/adds and outward
/// absolute-product sums; entry/row terms cover gamma, residual and bound reductions.
inline std::size_t requiredScalarWork(std::size_t count, std::size_t n, bool complex)
{
  if (!count || !n)
    throw std::invalid_argument("Inverse certificate requires positive matrix extents");
  // Check products before narrowing any work descriptor or allocating its owners.
  const auto mul = [](std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
      throw std::length_error("Inverse certificate work overflow");
    return a * b;
  };
  // Keep the sum checked independently of its already checked terms.
  const auto add = [](std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
      throw std::length_error("Inverse certificate work overflow");
    return a + b;
  };
  const auto nn = mul(n, n), nnn = mul(nn, n);
  return mul(count, add(add(mul(complex ? 512 : 256, nnn), mul(complex ? 512 : 256, nn)), add(mul(64, n), 128)));
}

namespace math
{
/// Materialize positive infinity by bits, independent of fast-math classification.
QMC_PF_INV_HD inline double infinity() noexcept
{
  union
  {
    std::uint64_t bits;
    double value;
  } x{UINT64_C(0x7ff0000000000000)};
  return x.value;
}

/// Round the next bound outward toward positive infinity.
QMC_PF_INV_HD inline double up(double x) noexcept { return ::nextafter(x, infinity()); }

/// Round a positive denominator outward toward negative infinity.
QMC_PF_INV_HD inline double down(double x) noexcept { return ::nextafter(x, -infinity()); }

/// Conservatively reject nonzero subnormal operands, avoiding unmodelled input FTZ.
/// Exact signed zero and finite normal inputs remain supported on both backends.
QMC_PF_INV_HD inline bool normalOrZero(double x) noexcept
{
  union
  {
    double value;
    std::uint64_t bits;
  } b{x};
  return (b.bits & UINT64_C(0x7ff0000000000000)) != 0 ||
      (b.bits & UINT64_C(0x000fffffffffffff)) == 0;
}

/// Bound nonnegative addition, including gradual/FTZ intermediate underflow.
/// The nextafter barriers prevent contraction across bound-path operations.
QMC_PF_INV_HD inline double addUp(double a, double b) noexcept { return up(up(a + b) + DBL_MIN); }

/// Bound a nonnegative product, including its own underflow before accumulation.
QMC_PF_INV_HD inline double mulUp(double a, double b) noexcept { return addUp(up(a * b), DBL_MIN); }

/// Bound division by a separately down-rounded positive denominator.
QMC_PF_INV_HD inline double divUp(double a, double b) noexcept { return addUp(up(a / b), DBL_MIN); }

/// Inflate the standard binary64 operation-count bound without rounding it downward.
QMC_PF_INV_HD inline double gamma(std::size_t operations) noexcept
{
  const double t = mulUp(static_cast<double>(operations), 0x1p-53);
  const double denominator = down(1.0 - t);
  return denominator > 0 ? divUp(t, denominator) : infinity();
}

/// Retain one computed scalar dot and an outward bound on its exact absolute sum.
struct Dot
{
  double center = 0, absolute_upper = 0;
  std::uint32_t status = CERTIFIED;

  /// Account for a signed product in both center and nonnegative error-bound paths.
  QMC_PF_INV_HD void add(double a, double b) noexcept
  {
    if (!device_math::isFiniteBinary64(a) || !device_math::isFiniteBinary64(b))
    {
      status = NONFINITE_INPUT;
      return;
    }
    if (!normalOrZero(a) || !normalOrZero(b))
    {
      if (status != NONFINITE_INPUT)
        status = UNCERTIFIABLE;
      return;
    }
    center += a * b; // Gamma covers FMA and separate multiply/add implementations.
    absolute_upper = addUp(absolute_upper, mulUp(::fabs(a), ::fabs(b)));
    if (!device_math::isFiniteBinary64(center) || !device_math::isFiniteBinary64(absolute_upper))
      if (status != NONFINITE_INPUT)
        status = UNCERTIFIABLE;
  }
};

/// Bound the exact residual component of stored binary64 A,B. Gamma covers the
/// center dot; 1/(1-u64) covers subtraction, with propagated underflow allowances.
QMC_PF_INV_HD inline double residualUpper(const Dot& dot, double identity, std::size_t operations) noexcept
{
  if (dot.status != CERTIFIED)
    return infinity();
  const double r = identity - dot.center;
  if (!device_math::isFiniteBinary64(r))
    return infinity();
  const double g = gamma(operations);
  const double underflow = divUp(mulUp(static_cast<double>(operations), DBL_MIN), down(1.0 - g));
  const double dot_error = addUp(mulUp(g, dot.absolute_upper), underflow);
  const double subtraction = divUp(addUp(::fabs(r), DBL_MIN), down(1.0 - 0x1p-53));
  return addUp(subtraction, dot_error);
}

/// Convert the outward residual norm into a fixed-budget normwise inverse bound.
QMC_PF_INV_HD inline Result finish(double rho, std::uint32_t status) noexcept
{
  Result r;
  r.rho_upper = rho;
  r.relative_inverse_error_upper = infinity();
  r.status = status;
  if (status != CERTIFIED)
    return r;
  if (!device_math::isFiniteBinary64(rho) || rho < 0 || !(rho < 1))
  {
    r.status = UNCERTIFIABLE;
    return r;
  }
  const double denominator = down(1.0 - rho);
  if (!(denominator > 0))
  {
    r.status = UNCERTIFIABLE;
    return r;
  }
  r.relative_inverse_error_upper = divUp(rho, denominator);
  r.status = device_math::isFiniteBinary64(r.relative_inverse_error_upper) &&
      r.relative_inverse_error_upper <= relative_budget ? CERTIFIED : BOUND_EXCEEDED;
  return r;
}

/// Bound one row of FIXED AB only; the caller max-reduces rows without another sum.
QMC_PF_INV_HD inline double realRow(const double* A, const double* B, std::size_t n,
                                  std::size_t row, std::uint32_t& status) noexcept
{
  if (!A || !B || !n || row >= n)
  {
    status = UNCERTIFIABLE;
    return infinity();
  }
  double total = 0;
  for (std::size_t col = 0; col < n; ++col)
  {
    Dot dot;
    for (std::size_t k = 0; k < n; ++k)
      dot.add(A[row * n + k], B[k * n + col]);
    if (dot.status != CERTIFIED)
      status = dot.status == NONFINITE_INPUT ? NONFINITE_INPUT :
          (status == NONFINITE_INPUT ? status : UNCERTIFIABLE);
    total = addUp(total, residualUpper(dot, row == col ? 1.0 : 0.0, 2 * n + 1));
  }
  if (!device_math::isFiniteBinary64(total) && status == CERTIFIED)
    status = UNCERTIFIABLE;
  return total;
}

/// Bound complex AB using explicit Re/Im dots of 2n products and 2n additions.
/// Gamma_(4n+2) covers FMA/non-FMA; |Re|+|Im| bounds each complex residual magnitude.
/// Entries expose binary64 .x/.y; no rounded hypot or extra matrix owner is needed.
template<class Complex>
QMC_PF_INV_HD inline double complexRow(const Complex* A, const Complex* B, std::size_t n,
                                     std::size_t row, std::uint32_t& status) noexcept
{
  if (!A || !B || !n || row >= n)
  {
    status = UNCERTIFIABLE;
    return infinity();
  }
  double total = 0;
  for (std::size_t col = 0; col < n; ++col)
  {
    Dot re, im;
    for (std::size_t k = 0; k < n; ++k)
    {
      const auto a = A[row * n + k], b = B[k * n + col];
      re.add(a.x, b.x);
      re.add(-a.y, b.y);
      im.add(a.x, b.y);
      im.add(a.y, b.x);
    }
    if (re.status != CERTIFIED)
      status = re.status == NONFINITE_INPUT ? NONFINITE_INPUT :
          (status == NONFINITE_INPUT ? status : UNCERTIFIABLE);
    if (im.status != CERTIFIED)
      status = im.status == NONFINITE_INPUT ? NONFINITE_INPUT :
          (status == NONFINITE_INPUT ? status : UNCERTIFIABLE);
    const double bound = addUp(residualUpper(re, row == col ? 1.0 : 0.0, 4 * n + 2),
                               residualUpper(im, 0.0, 4 * n + 2));
    total = addUp(total, bound);
  }
  if (!device_math::isFiniteBinary64(total) && status == CERTIFIED)
    status = UNCERTIFIABLE;
  return total;
}
/// The refinement-only domain puts inputs on a 2^-502 grid, exact products on
/// a 2^-1004 grid. Every nonzero finite center/EFT intermediate is therefore
/// normal (>=2^-1004); no backend FTZ assumption is needed. Zeros are exact.
QMC_PF_INV_HD inline bool refinementOperand(double x) noexcept
{
  if (!device_math::isFiniteBinary64(x) || !normalOrZero(x)) return false;
  const double a = ::fabs(x);
  return a == 0 || (a >= 0x1p-450 && a <= 0x1p450);
}

/// Materialize each rounded operation; strict host/device compilation also
/// prohibits reassociation. Explicit fma below is the only contracted operation.
QMC_PF_INV_HD inline double roundedAdd(double a, double b) noexcept
{ volatile double r = a + b; return r; }
QMC_PF_INV_HD inline double roundedSubtract(double a, double b) noexcept
{ volatile double r = a - b; return r; }
QMC_PF_INV_HD inline double roundedMultiply(double a, double b) noexcept
{ volatile double r = a * b; return r; }

/// Dot2: FMA captures each product error; TwoSum captures each high-sum error.
/// Only the low residual accumulation is rounded without its own EFT. Its
/// ordinary gamma bound scales with captured residuals, not |A|*|B|.
struct CompensatedDot
{
  double high = 0, low = 0, residual_absolute_upper = 0;
  std::uint32_t status = CERTIFIED;

  QMC_PF_INV_HD void add(double a, double b) noexcept
  {
    if (status != CERTIFIED) return;
    if (!refinementOperand(a) || !refinementOperand(b))
    { status = UNCERTIFIABLE; return; }
    const double p = roundedMultiply(a, b);
    const double e = ::fma(a, b, -p);
    const double sum = roundedAdd(high, p);
    const double bb = roundedSubtract(sum, high);
    const double aa = roundedSubtract(sum, bb);
    const double br = roundedSubtract(p, bb);
    const double ar = roundedSubtract(high, aa);
    const double q = roundedAdd(ar, br);
    const double next_low = roundedAdd(low, e);
    const double final_low = roundedAdd(next_low, q);
    const double centers[] = {p, e, sum, bb, aa, br, ar, q, next_low, final_low};
    for (double x : centers)
      if (!device_math::isFiniteBinary64(x))
      { status = UNCERTIFIABLE; return; }
    high = sum; low = final_low;
    residual_absolute_upper = addUp(residual_absolute_upper, addUp(::fabs(e), ::fabs(q)));
    if (!device_math::isFiniteBinary64(residual_absolute_upper)) status = UNCERTIFIABLE;
  }
};

/// Enclose I-(high+exact_low) without first rounding high+low. The two residual
/// subtractions have error <=u*|computed|/(1-u); their centers share the safe grid.
QMC_PF_INV_HD inline double compensatedResidualUpper(const CompensatedDot& dot,
    double identity, std::size_t products) noexcept
{
  if (dot.status != CERTIFIED || products > (std::numeric_limits<std::size_t>::max() - 1) / 2)
    return infinity();
  const double r0 = roundedSubtract(identity, dot.high), r1 = roundedSubtract(r0, dot.low);
  if (!device_math::isFiniteBinary64(r0) || !device_math::isFiniteBinary64(r1)) return infinity();
  const double low_error = mulUp(gamma(2 * products + 1), dot.residual_absolute_upper);
  const double residual = divUp(addUp(::fabs(r1), mulUp(0x1p-53, ::fabs(r0))), down(1.0 - 0x1p-53));
  return addUp(residual, low_error);
}

/// Refine one fixed AB real row; a failed safe-domain check has no authority.
QMC_PF_INV_HD inline double compensatedRealRow(const double* A, const double* B,
    std::size_t n, std::size_t row, std::uint32_t& status) noexcept
{
  if (!A || !B || !n || row >= n)
  { status = UNCERTIFIABLE; return infinity(); }
  double total = 0;
  for (std::size_t col = 0; col < n; ++col)
  {
    CompensatedDot dot;
    for (std::size_t k = 0; k < n; ++k) dot.add(A[row*n+k], B[k*n+col]);
    if (dot.status != CERTIFIED) status = UNCERTIFIABLE;
    total = addUp(total, compensatedResidualUpper(dot, row == col ? 1.0 : 0.0, n));
  }
  if (!device_math::isFiniteBinary64(total)) status = UNCERTIFIABLE;
  return total;
}

/// Preserve explicit real/imaginary signs and the original component-L1 norm.
template<class Complex>
QMC_PF_INV_HD inline double compensatedComplexRow(const Complex* A, const Complex* B,
    std::size_t n, std::size_t row, std::uint32_t& status) noexcept
{
  if (!A || !B || !n || row >= n || n > std::numeric_limits<std::size_t>::max()/2)
  { status = UNCERTIFIABLE; return infinity(); }
  double total = 0;
  for (std::size_t col = 0; col < n; ++col)
  {
    CompensatedDot re, im;
    for (std::size_t k = 0; k < n; ++k)
    {
      const auto a = A[row*n+k], b = B[k*n+col];
      re.add(a.x,b.x); re.add(-a.y,b.y); im.add(a.x,b.y); im.add(a.y,b.x);
    }
    if (re.status != CERTIFIED || im.status != CERTIFIED) status = UNCERTIFIABLE;
    total = addUp(total, addUp(compensatedResidualUpper(re, row == col ? 1.0 : 0.0, 2*n),
        compensatedResidualUpper(im, 0.0, 2*n)));
  }
  if (!device_math::isFiniteBinary64(total)) status = UNCERTIFIABLE;
  return total;
}
} // namespace math
} // namespace qmcplusplus::psiformer::device::inverse_certificate
#undef QMC_PF_INV_HD
#endif
