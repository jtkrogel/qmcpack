//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_INVERSE_CERTIFICATE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_INVERSE_CERTIFICATE_VALIDATION_H
#include "PsiFormerDeviceInverseCertificate.h"
#include "PsiFormerDeviceResources.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <vector>
namespace qmcplusplus::psiformer::device
{
/// Exercise coefficient validity on hardware; no mixed derivative/energy proof is implied.
inline void validateCanonicalInverseCertificates()
{
  namespace cert = inverse_certificate;
  compute::Queue<PlatformKind::CUDA> queue;
  // Stop at the first actual device disagreement, preserving the typed diagnostic.
  const auto require = [](bool ok, const char* why) {
    if (!ok)
      throw std::runtime_error(why);
  };
  constexpr std::size_t count = 12, n = 2;
  const std::array<double, 4> I{1, 0, 0, 1};
  std::vector<double> A(count * 4), B(count * 4);
  for (std::size_t h = 0; h < count; ++h)
  {
    std::copy(I.begin(), I.end(), A.begin() + 4 * h);
    std::copy(I.begin(), I.end(), B.begin() + 4 * h);
  }
  A[4] = 0x1p-40;
  B[4] = 0x1p40; // Accurate inverse of a very small pivot.
  // Exact dyadic inverses separate a useful positive from a roundoff-refined positive.
  const auto dependence = [&](std::size_t h, int power) {
    const double e = std::ldexp(1., -power), inv = std::ldexp(1., power);
    const std::array<double, 4> a{1, 1, 1, 1 + e}, b{1 + inv, -inv, -inv, inv};
    std::copy(a.begin(), a.end(), A.begin() + 4 * h);
    std::copy(b.begin(), b.end(), B.begin() + 4 * h);
  };
  dependence(2, 12);
  dependence(3, 30);
  B[16] = 1.001;
  A[20] = std::numeric_limits<double>::quiet_NaN();
  B[24] = std::numeric_limits<double>::infinity();
  A[29] = std::numeric_limits<double>::denorm_min();
  for (std::size_t h = 8; h < 11; ++h)
  {
    dependence(h, 30);
    const double scale = h == 8 ? 0x1p-419 : h == 9 ? 0x1p420 : 0x1p451;
    for (std::size_t i = 0; i < 4; ++i) { A[4*h+i] *= scale; B[4*h+i] /= scale; }
  }
  // Nonzero FMA product residual: exact stored AB differs from I by2^-28.
  const double x=0x1p26+.5, y=1-0x1p-27, z=1+0x1p26;
  const std::array<double,4> productA{1,x,y,z}, productB{z,-x,-y,1};
  std::copy(productA.begin(), productA.end(), A.begin()+44);
  std::copy(productB.begin(), productB.end(), B.begin()+44);
  DeviceBuffer<double> da(A.size()), db(B.size());
  DeviceBuffer<cert::Result> output(count);
  da.upload(A.data(), A.size());
  db.upload(B.data(), B.size());
  require(cert::launchReal(queue.getNative(), da.data(), db.data(), 0, n, output.data()) == invalid_value,
          "Zero inverse certificate count accepted");
  require(cert::launchReal(queue.getNative(), nullptr, db.data(), count, n, output.data()) == invalid_value,
          "Null inverse certificate input accepted");
  require(cert::launchReal(queue.getNative(), da.data(), db.data(), count, 0, output.data()) == invalid_value,
          "Zero inverse certificate matrix accepted");
  require(cert::launchReal(queue.getNative(), da.data(), db.data(), count, n,
                          reinterpret_cast<cert::Result*>(da.data())) == invalid_value,
          "Aliased certificate output accepted");
  checkRuntime(cert::launchReal(queue.getNative(), da.data(), db.data(), count, n, output.data()),
               "REAL inverse certificates");
  queue.sync();
  std::vector<cert::Result> actual(count);
  output.download(actual.data(), count);
  // Check the independent fixture meanings, not equality to a duplicate device algorithm.
  const auto verify = [&](const auto& results) {
    for (std::size_t h = 0; h < 3; ++h)
      require(cert::isCertified(results[h]), "Accurate canonical inverse was rejected by copied pivot heuristic");
    require(cert::isCertified(results[3]), "Exact cancellation-heavy inverse did not receive compensated certification");
    require(!cert::isCertified(results[4]), "Corrupt inverse was certified");
    require(results[5].status == cert::NONFINITE_INPUT && results[6].status == cert::NONFINITE_INPUT,
            "Nonfinite inverse certificate succeeded");
    require(results[7].status == cert::UNCERTIFIABLE, "Unsupported subnormal input was certified");
    require(cert::isCertified(results[8]) && cert::isCertified(results[9]),
            "Safe scaled cancellation-heavy inverse was rejected");
    require(results[10].status == cert::BOUND_EXCEEDED, "Unsafe refinement range changed fast rejection");
    require(cert::isCertified(results[11]) && results[11].rho_upper >= 0x1p-28 && results[11].rho_upper < 0x1p-27,
            "Compensated device product-error enclosure is incorrect");
  };
  verify(actual);
  std::vector<cuDoubleComplex> ca(A.size()), cb(B.size());
  for (std::size_t i = 0; i < A.size(); ++i)
  {
    ca[i] = make_cuDoubleComplex(A[i], 0);
    cb[i] = make_cuDoubleComplex(B[i], 0);
  }
  // Nonzero real/imaginary cancellation exercises both compensated component dots.
  for (std::size_t h : {std::size_t(3), std::size_t(8), std::size_t(9), std::size_t(10), std::size_t(11)})
    for (std::size_t i=0; i<4; ++i)
    {
      ca[4*h+i] = make_cuDoubleComplex(A[4*h+i], A[4*h+i]);
      cb[4*h+i] = make_cuDoubleComplex(B[4*h+i]/2, -B[4*h+i]/2);
    }
  // A non-real accurate low-pivot pair exercises both component dots.
  ca[4] = make_cuDoubleComplex(0, 0x1p-40);
  cb[4] = make_cuDoubleComplex(0, -0x1p40);
  DeviceBuffer<cuDoubleComplex> dca(ca.size()), dcb(cb.size());
  dca.upload(ca.data(), ca.size());
  dcb.upload(cb.data(), cb.size());
  checkRuntime(cert::launchComplex(queue.getNative(), dca.data(), dcb.data(), count, n, output.data()),
               "Paired inverse certificates");
  queue.sync();
  output.download(actual.data(), count);
  verify(actual);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "Inverse certificate accounting device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "Inverse certificate accounting device");
#endif
  const auto before = deviceMemoryMetrics(ordinal).snapshot();
  checkRuntime(cert::launchReal(queue.getNative(), da.data(), db.data(), count, n, output.data()),
               "Warm REAL inverse certificates");
  checkRuntime(cert::launchComplex(queue.getNative(), dca.data(), dcb.data(), count, n, output.data()),
               "Warm paired inverse certificates");
  queue.sync();
  const auto after = deviceMemoryMetrics(ordinal).snapshot();
  require(before.live_bytes == after.live_bytes && before.allocation_calls == after.allocation_calls,
          "Warm inverse certificate allocated numeric storage");
  std::cout << "PASS canonical inverse certificate fixed_AB=1 accurate_low_pivot=allowed corrupt_inverse=reject"
               " roundoff_refinement=pass safe_range_boundaries=1 unsafe_range=reject real_complex=1 nonfinite_unknown=reject warm_DeviceBuffer_delta=0"
               " mixed_jet_accuracy_claim=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
