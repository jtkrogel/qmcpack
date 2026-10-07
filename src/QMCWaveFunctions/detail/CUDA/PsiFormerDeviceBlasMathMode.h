//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceBlasMathMode.h
 * @brief Scoped CUDA/HIP BLAS math-mode containment for PsiFormer GEMMs.
 *
 * Inclusion is restricted to CUDA/HIP translation units.  The current dense ABI is
 * passed a handle by its future crowd resource, so this adapter saves, sets, and
 * restores its mode around each logical group of GEMMs.  The caller must provide
 * exclusive access to that handle during the call; production orchestration should
 * own one handle per crowd/queue rather than share it between host threads.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_BLAS_MATH_MODE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_BLAS_MATH_MODE_H

#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBlasMathMode.h"

#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer::device
{

#ifdef QMC_CUDA2HIP

/// Adapt the public accelerator handle to hipBLAS get/set math-mode operations.
class NativeBlasMathModeController
{
public:
  using mode_type = hipblasMath_t;

  explicit NativeBlasMathModeController(
      compute::BLASHandle<PlatformKind::CUDA>& handle) noexcept
      : handle_(handle.h_cublas)
  {}

  mode_type currentMode() const
  {
    mode_type mode;
    cublasErrorCheck(hipblasGetMathMode(handle_, &mode),
                     "PsiFormer hipBLAS math-mode query failed");
    return mode;
  }

  void setMode(mode_type mode)
  {
    cublasErrorCheck(hipblasSetMathMode(handle_, mode),
                     "PsiFormer hipBLAS math-mode update failed");
  }

private:
  hipblasHandle_t handle_;
};

/// Translate an already validated backend-neutral HIP plan to its native enum.
inline hipblasMath_t nativeBlasMathMode(const PsiFormerBlasMathModePlan& plan)
{
  if (plan.native != PsiFormerNativeBlasMathMode::HIP_DEFAULT_STRICT)
    throw std::invalid_argument("PsiFormer HIP received an unsupported BLAS math mode");
  return HIPBLAS_DEFAULT_MATH;
}

#else

/// Adapt the public accelerator handle to cuBLAS get/set math-mode operations.
class NativeBlasMathModeController
{
public:
  using mode_type = cublasMath_t;

  explicit NativeBlasMathModeController(
      compute::BLASHandle<PlatformKind::CUDA>& handle) noexcept
      : handle_(handle.h_cublas)
  {}

  mode_type currentMode() const
  {
    mode_type mode;
    cublasErrorCheck(cublasGetMathMode(handle_, &mode),
                     "PsiFormer cuBLAS math-mode query failed");
    return mode;
  }

  void setMode(mode_type mode)
  {
    cublasErrorCheck(cublasSetMathMode(handle_, mode),
                     "PsiFormer cuBLAS math-mode update failed");
  }

private:
  cublasHandle_t handle_;
};

/// Translate an already validated backend-neutral CUDA plan to its native enum.
inline cublasMath_t nativeBlasMathMode(const PsiFormerBlasMathModePlan& plan)
{
  switch (plan.native)
  {
  case PsiFormerNativeBlasMathMode::CUDA_PEDANTIC:
    return CUBLAS_PEDANTIC_MATH;
  case PsiFormerNativeBlasMathMode::CUDA_TF32:
    return CUBLAS_TF32_TENSOR_OP_MATH;
  default:
    throw std::invalid_argument("PsiFormer CUDA received an unsupported BLAS math mode");
  }
}

#endif

/** Execute a logical FP32 GEMM group under its explicit backend mode. */
template<class Operation>
void executeWithDeviceBlasMathMode(compute::BLASHandle<PlatformKind::CUDA>& handle,
                                   const PsiFormerBlasMathModePlan& plan,
                                   Operation&& operation)
{
  if (plan.requested == PsiFormerBackendMathMode::FP64_STRICT)
    throw std::invalid_argument("PsiFormer FP32 GEMM cannot use the FP64 BLAS mode");
#ifdef QMC_CUDA2HIP
  constexpr PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::HIP;
#else
  constexpr PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::CUDA;
#endif
  if (!(plan == makePsiFormerBlasMathModePlan(plan.requested, backend)))
    throw std::invalid_argument(
        "PsiFormer BLAS math-mode plan does not match the compiled accelerator backend");
  NativeBlasMathModeController controller(handle);
  executeWithPsiFormerBlasMathMode(controller, nativeBlasMathMode(plan),
                                   std::forward<Operation>(operation));
}

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_BLAS_MATH_MODE_H
