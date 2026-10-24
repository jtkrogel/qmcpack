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

/// Capture the strict, default-off disjoint-output reverse selector once per process.
bool complexKineticHeadParallelRequested();

/// Parallel private destinations publish only after complete numerical admission.
/// Capacities are in doubles; a failed attempt replays onto untouched original outputs.
Error launchComplexKineticHeadReverseParallel(Stream stream, ComplexKineticHeadLayout layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const double* parameters,
    const double* features, const double* envelope, const double* real, const double* imag,
    const cuDoubleComplex* roots, std::size_t component, double* feature_seed, double* response,
    std::uint64_t* status, double* private_features, std::size_t feature_capacity,
    double* private_response, std::size_t response_capacity, std::uint64_t* private_status);

/// Reverse ordinary complex CORE G/L roots; caller owns three complex N*N scratch matrices.
/// Requires finite unweighted A/B/C/combined terms; no Wide-domain fallback.
/// Cold PSIFORMER_GPU_COOPERATIVE_COMPLEX_KINETIC_ROOTS=1 opts into independent-entry work only.
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

/// Prepare physical roots/energy once per tile, with fixed positive masses and common kappa.
Error launchComplexKineticCoefficients(Stream stream, SpatialJetLayout layout,
    const cuDoubleComplex* gradient, const cuDoubleComplex* laplacian, const double* cusp,
    const double* inverse_masses, const double* kappa, cuDoubleComplex* gradient_root,
    cuDoubleComplex* laplacian_root, cuDoubleComplex* energy, std::uint64_t* status);

/// Seed one real cusp objective from retained complex physical roots.
Error launchComplexKineticCuspSeed(Stream stream, std::size_t targets, const cuDoubleComplex* gradient_root,
    const cuDoubleComplex* laplacian_root, std::size_t component, double* seed, std::uint64_t* status);

/// Add the selected completed real response into the sole complexP sum.
Error launchComplexKineticPack(Stream stream, std::size_t parameters, const double* real,
    std::size_t component, cuDoubleComplex* output, std::uint64_t* status);
} // namespace qmcplusplus::psiformer::device
#endif
