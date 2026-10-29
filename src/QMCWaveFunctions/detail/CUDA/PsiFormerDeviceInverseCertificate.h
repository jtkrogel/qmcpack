//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_INVERSE_CERTIFICATE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_INVERSE_CERTIFICATE_H
#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInverseCertificateMath.h"
#include <complex>
#include "Platforms/CUDA/CUDATypeMapping.hpp"
namespace qmcplusplus::psiformer::device::inverse_certificate
{
/// Row-major [matrix_count,n,n], one compact Result per matrix; no allocation.
/// FIXED AB normwise inverse coefficient validity only, not determinant-weight,
/// mixed-jet or physical-energy accuracy. Callers separately retain factor/cancellation,
/// image/R/queue/generation authority and original numerical accuracy oracles.
Error launchReal(Stream stream, const double* A, const double* B,
                 std::size_t matrix_count, std::size_t n, Result* output);
/// Certify complex coefficients with explicit Re/Im rounding bounds, never an LU-as-A alias.
Error launchComplex(Stream stream, const cuDoubleComplex* A, const cuDoubleComplex* B,
                    std::size_t matrix_count, std::size_t n, Result* output);
} // namespace qmcplusplus::psiformer::device::inverse_certificate
#endif
