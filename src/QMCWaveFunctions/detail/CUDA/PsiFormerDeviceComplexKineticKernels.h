//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_KINETIC_KERNELS_H
#include "PsiFormerDeviceComplexValueKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenKineticReverse.h"

namespace qmcplusplus::psiformer::device
{
/// Explicit canonical imaginary-head intervals augment the existing real head layout.
struct ComplexKineticHeadLayout
{
  OpenOrbitalKineticReverseLayout real;
  std::size_t imaginary_up, imaginary_down;
};

/// Reverse ordinary complex CORE G/L roots; caller owns three complex N*N scratch matrices.
/// Requires finite unweighted A/B/C/combined terms; no Wide-domain fallback.
Error launchComplexKineticMatrixRoots(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* orbitals, const cuDoubleComplex* core_gradient, const cuDoubleComplex* core_laplacian,
    const cuDoubleComplex* gradient_root, const cuDoubleComplex* laplacian_root,
    cuDoubleComplex* roots, cuDoubleComplex* scratch, std::uint64_t* status);

/// Add one real objective's paired-head/envelope P and feature seeds, never cusp or trunk.
Error launchComplexKineticHeadReverse(Stream stream, ComplexKineticHeadLayout layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const double* parameters,
    const double* features, const double* envelope, const double* real, const double* imag,
    const cuDoubleComplex* roots, std::size_t component, double* feature_seed, double* response,
    std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
