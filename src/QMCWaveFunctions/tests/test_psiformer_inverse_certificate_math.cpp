//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "QMCWaveFunctions/PsiFormer/PsiFormerInverseCertificateMath.h"
#include <array>
#include <complex>
#include <limits>
namespace cert = qmcplusplus::psiformer::device::inverse_certificate;
namespace
{
using Mat = std::array<double, 4>;

/// Runtime-free complex entries exercise the shared Re/Im scalar core on CPU.
struct C
{
  double x, y;
};
using CMat = std::array<C, 4>;

/// Apply the shared real bound core to the complete two-row fixture.
cert::Result realFast(const Mat& A, const Mat& B)
{
  double rho = 0;
  std::uint32_t status = cert::CERTIFIED;
  for (std::size_t i = 0; i < 2; ++i)
    rho = std::max(rho, cert::math::realRow(A.data(), B.data(), 2, i, status));
  return cert::math::finish(rho, status);
}

/// Apply the shared complex bound core to both fixture rows.
cert::Result pairedFast(const CMat& A, const CMat& B)
{
  double rho = 0;
  std::uint32_t status = cert::CERTIFIED;
  for (std::size_t i = 0; i < 2; ++i)
    rho = std::max(rho, cert::math::complexRow(A.data(), B.data(), 2, i, status));
  return cert::math::finish(rho, status);
}

/// Match the device selection contract while keeping fast bounds observable.
cert::Result real(const Mat& A, const Mat& B)
{
  auto original = realFast(A, B);
  if (original.status != cert::BOUND_EXCEEDED) return original;
  double rho = 0; std::uint32_t status = cert::CERTIFIED;
  for (std::size_t i=0; i<2; ++i)
    rho = std::max(rho, cert::math::compensatedRealRow(A.data(), B.data(), 2, i, status));
  const auto refined = cert::math::finish(rho, status);
  return (refined.status == cert::CERTIFIED || refined.status == cert::BOUND_EXCEEDED) &&
      refined.rho_upper < original.rho_upper ? refined : original;
}
cert::Result paired(const CMat& A, const CMat& B)
{
  auto original = pairedFast(A, B);
  if (original.status != cert::BOUND_EXCEEDED) return original;
  double rho = 0; std::uint32_t status = cert::CERTIFIED;
  for (std::size_t i=0; i<2; ++i)
    rho = std::max(rho, cert::math::compensatedComplexRow(A.data(), B.data(), 2, i, status));
  const auto refined = cert::math::finish(rho, status);
  return (refined.status == cert::CERTIFIED || refined.status == cert::BOUND_EXCEEDED) &&
      refined.rho_upper < original.rho_upper ? refined : original;
}

/// Independently accumulate the real residual at wider precision for enclosure tests.
long double exactResidual(const Mat& A, const Mat& B)
{
  long double rho = 0;
  for (int i = 0; i < 2; ++i)
  {
    long double sum = 0;
    for (int j = 0; j < 2; ++j)
    {
      long double dot = 0;
      for (int k = 0; k < 2; ++k)
        dot += static_cast<long double>(A[2 * i + k]) * B[2 * k + j];
      sum += std::abs((i == j ? 1.L : 0.L) - dot);
    }
    rho = std::max(rho, sum);
  }
  return rho;
}
} // namespace

TEST_CASE("Canonical inverse validity permits accurate small pivots, not approximate raw heads",
          "[psiformer][inverse_certificate]")
{
  const Mat A{0x1p-40, 0, 0, 1}, B{0x1p40, 0, 0, 1};
  const auto r = real(A, B);
  REQUIRE(cert::isCertified(r));
  REQUIRE(r.rho_upper >= exactResidual(A, B));
  REQUIRE(r.relative_inverse_error_upper <= 0x1p-24);
  // No mixed jets were evaluated: an inverse certificate does not certify their accuracy.
  const CMat Z{{{0, 0x1p-40}, {0, 0}, {0, 0}, {0, -1}}};
  const CMat W{{{0, -0x1p40}, {0, 0}, {0, 0}, {0, 1}}};
  REQUIRE(cert::isCertified(paired(Z, W)));
}

TEST_CASE("Canonical inverse certificate rejects corrupt inverse and refines roundoff-dominated dependence",
          "[psiformer][inverse_certificate]")
{
  const Mat I{1, 0, 0, 1}, wrong{1.001, 0, 0, 1};
  const auto r = real(I, wrong);
  REQUIRE_FALSE(cert::isCertified(r));
  REQUIRE(r.rho_upper >= exactResidual(I, wrong));
  const Mat A{1, 1, 1, 1 + 0x1p-30}, B{1 + 0x1p30, -0x1p30, -0x1p30, 0x1p30};
  REQUIRE(exactResidual(A, B) == 0);
  const auto wide = realFast(A, B);
  REQUIRE(wide.status == cert::BOUND_EXCEEDED);
  REQUIRE_FALSE(cert::isCertified(wide));
  const CMat Z{{{1, 0}, {1, 0}, {1, 0}, {1 + 0x1p-30, 0}}};
  const CMat W{{{1 + 0x1p30, 0}, {-0x1p30, 0}, {-0x1p30, 0}, {0x1p30, 0}}};
  REQUIRE(pairedFast(Z, W).status == cert::BOUND_EXCEEDED);
  REQUIRE(cert::isCertified(real(A, B)));
  REQUIRE(cert::isCertified(paired(Z, W)));
}

TEST_CASE("Compensated refinement safe domain is bounded and never repairs corruption",
          "[psiformer][inverse_certificate]")
{
  for (double x : {0., -0., 0x1p-450, -0x1p-450, 0x1p450, -0x1p450})
    REQUIRE(cert::math::refinementOperand(x));
  for (double x : {0x1p-451, -0x1p451, std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    REQUIRE_FALSE(cert::math::refinementOperand(x));
  const Mat A{1,1,1,1+0x1p-30}, B{1+0x1p30,-0x1p30,-0x1p30,0x1p30};
  for (double scale : {0x1p-419, 0x1p420})
  {
    Mat scaledA, scaledB;
    for (int i=0; i<4; ++i) { scaledA[i]=A[i]*scale; scaledB[i]=B[i]/scale; }
    REQUIRE(cert::isCertified(real(scaledA,scaledB)));
  }
  Mat unsafeA, unsafeB;
  for (int i=0; i<4; ++i) { unsafeA[i]=A[i]*0x1p451; unsafeB[i]=B[i]*0x1p-451; }
  const auto rejected=real(unsafeA,unsafeB), fast=realFast(unsafeA,unsafeB);
  REQUIRE(rejected.status==cert::BOUND_EXCEEDED);
  REQUIRE(rejected.rho_upper==fast.rho_upper);
  auto corrupt=B; corrupt[0]+=0.001;
  REQUIRE_FALSE(cert::isCertified(real(A,corrupt)));
  REQUIRE(real(A,corrupt).rho_upper>=exactResidual(A,corrupt));
  cert::math::CompensatedDot dot;
  dot.add(0x1p450,0x1p450); dot.add(-0x1p450,0x1p450);
  REQUIRE(dot.status==cert::CERTIFIED);
  REQUIRE(dot.high==0); REQUIRE(dot.low==0);
  REQUIRE(cert::math::compensatedResidualUpper(dot,0,2)<0x1p-24);
  cert::math::CompensatedDot product_error;
  product_error.add(1+0x1p-27,1-0x1p-27); product_error.add(-1,1);
  REQUIRE(product_error.high==0); REQUIRE(product_error.low==-0x1p-54);
  REQUIRE(cert::math::compensatedResidualUpper(product_error,0,2)>=0x1p-54);
  REQUIRE(cert::math::compensatedResidualUpper(product_error,0,2)<0x1p-53);
  cert::math::CompensatedDot sum_error;
  sum_error.add(1,1); sum_error.add(0x1p-54,1); sum_error.add(-1,1);
  REQUIRE(sum_error.high==0); REQUIRE(sum_error.low==0x1p-54);
  REQUIRE(cert::math::compensatedResidualUpper(sum_error,0,3)>=0x1p-54);
  REQUIRE(cert::math::compensatedResidualUpper(sum_error,0,3)<0x1p-53);
  cert::math::CompensatedDot tiny;
  tiny.add(0x1p-450,0x1p-450); tiny.add(-0x1p-450,0x1p-450);
  REQUIRE(tiny.status==cert::CERTIFIED); REQUIRE(tiny.high==0); REQUIRE(tiny.low==0);
}

TEST_CASE("Compensation captures nonzero product residuals in a roundoff-heavy inverse",
          "[psiformer][inverse_certificate]")
{
  const double x=0x1p26+.5, y=1-0x1p-27, z=1+0x1p26;
  const Mat A{1,x,y,z}, B{z,-x,-y,1};
  REQUIRE(realFast(A,B).status==cert::BOUND_EXCEEDED);
  REQUIRE(exactResidual(A,B)==0x1p-28L);
  const auto refined=real(A,B);
  REQUIRE(cert::isCertified(refined)); REQUIRE(refined.rho_upper>=exactResidual(A,B));
  REQUIRE(refined.rho_upper<0x1p-27);
  CMat Z, W;
  for (std::size_t i=0;i<4;++i) { Z[i]={A[i],A[i]}; W[i]={B[i]/2,-B[i]/2}; }
  REQUIRE(pairedFast(Z,W).status==cert::BOUND_EXCEEDED);
  REQUIRE(cert::isCertified(paired(Z,W)));
}

TEST_CASE("Outward AB residual bounds enclose independent long-double residuals",
          "[psiformer][inverse_certificate]")
{
  const Mat A{1, 0x1p-3, 0x1p-4, 1};
  for (double delta : {-0x1p-30, 0., 0x1p-30})
  {
    const double d = 1 - 0x1p-7;
    const Mat B{1 / d + delta, -0x1p-3 / d, -0x1p-4 / d, 1 / d};
    const auto r = real(A, B);
    REQUIRE(r.rho_upper >= exactResidual(A, B));
    REQUIRE(cert::isCertified(r));
  }
  const CMat Z{{{1, 0}, {0, 0x1p-3}, {0, 0x1p-4}, {1, 0}}};
  const double d = 1 + 0x1p-7;
  const CMat W{{{1 / d, 0}, {0, -0x1p-3 / d}, {0, -0x1p-4 / d}, {1 / d, 0}}};
  const auto r = paired(Z, W);
  REQUIRE(cert::isCertified(r));
  long double rho = 0;
  for (int i = 0; i < 2; ++i)
  {
    long double sum = 0;
    for (int j = 0; j < 2; ++j)
    {
      std::complex<long double> dot{};
      for (int k = 0; k < 2; ++k)
        dot += std::complex<long double>(Z[2 * i + k].x, Z[2 * i + k].y) *
            std::complex<long double>(W[2 * k + j].x, W[2 * k + j].y);
      sum += std::abs(std::complex<long double>(i == j ? 1 : 0, 0) - dot);
    }
    rho = std::max(rho, sum);
  }
  REQUIRE(r.rho_upper >= rho);
}

TEST_CASE("Malformed nonfinite unsupported or unknown certificate inputs never publish validity",
          "[psiformer][inverse_certificate]")
{
  const Mat I{1, 0, 0, 1};
  for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
  {
    auto A = I;
    A[1] = bad;
    REQUIRE(real(A, I).status == cert::NONFINITE_INPUT);
    CMat Z{{{1, 0}, {0, bad}, {0, 0}, {1, 0}}}, W{{{1, 0}, {0, 0}, {0, 0}, {1, 0}}};
    REQUIRE(paired(Z, W).status == cert::NONFINITE_INPUT);
  }
  auto A = I;
  A[1] = std::numeric_limits<double>::denorm_min();
  REQUIRE(real(A, I).status == cert::UNCERTIFIABLE);
  REQUIRE_FALSE(cert::isCertified(cert::Result{}));
  auto unknown = real(I, I);
  unknown.status = 99;
  REQUIRE_FALSE(cert::isCertified(unknown));
  unknown = real(I, I);
  unknown.reserved = 1;
  REQUIRE_FALSE(cert::isCertified(unknown));
  unknown = real(I, I);
  unknown.rho_upper = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(cert::isCertified(unknown));
  REQUIRE_THROWS_AS(cert::requiredScalarWork(0, 2, false), std::invalid_argument);
  REQUIRE_THROWS_AS(cert::requiredScalarWork(1, 0, true), std::invalid_argument);
  REQUIRE_THROWS_AS(cert::requiredScalarWork(std::numeric_limits<std::size_t>::max(), 2, false), std::length_error);
  REQUIRE(cert::requiredScalarWork(1, 100, false) == 258566528);
  REQUIRE(cert::requiredScalarWork(1, 100, true) == 517126528);
}

/// Run only the dependency-light certificate algebra and guard cases.
int main(int argc, char** argv) { return Catch::Session().run(argc, argv); }
