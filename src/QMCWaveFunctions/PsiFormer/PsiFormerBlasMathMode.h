//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerBlasMathMode.h
 * @brief Backend-neutral PsiFormer BLAS math-mode mapping and scoped transitions.
 *
 * Precision policy and vendor BLAS mode remain distinct identities.  The pure mapping
 * and transition helper can be tested without a GPU runtime; the CUDA/HIP controller
 * that applies these transitions lives in the device-only adapter header.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_BLAS_MATH_MODE_H
#define QMCPLUSPLUS_PSIFORMER_BLAS_MATH_MODE_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{

/// Name the enforceable native behavior without exposing vendor enum values.
enum class PsiFormerNativeBlasMathMode : std::uint8_t
{
  HOST_DEFAULT,
  CUDA_PEDANTIC,
  CUDA_TF32,
  HIP_DEFAULT_STRICT
};

/// Record one validated requested-to-native backend mapping.
struct PsiFormerBlasMathModePlan
{
  PsiFormerBackendMathMode requested = PsiFormerBackendMathMode::FP64_STRICT;
  PsiFormerAcceleratorBackend backend = PsiFormerAcceleratorBackend::CPU;
  PsiFormerNativeBlasMathMode native = PsiFormerNativeBlasMathMode::HOST_DEFAULT;
  bool reduced_multiply = false;

  friend bool operator==(const PsiFormerBlasMathModePlan& lhs,
                         const PsiFormerBlasMathModePlan& rhs) noexcept
  {
    return lhs.requested == rhs.requested && lhs.backend == rhs.backend &&
        lhs.native == rhs.native && lhs.reduced_multiply == rhs.reduced_multiply;
  }
};

/** Map one explicit math mode to behavior enforceable by the selected backend.
 *
 * CUDA strict modes use pedantic math so ambient TF32 cannot leak in.  HIP strict
 * modes explicitly select hipBLAS default math, which disables ambient XF32; the
 * installed rocBLAS backend documents hipBLAS pedantic and TF32 as unsupported.
 */
inline PsiFormerBlasMathModePlan makePsiFormerBlasMathModePlan(
    PsiFormerBackendMathMode requested,
    PsiFormerAcceleratorBackend backend)
{
  switch (backend)
  {
  case PsiFormerAcceleratorBackend::CPU:
    if (requested == PsiFormerBackendMathMode::FP64_STRICT)
      return {requested, backend, PsiFormerNativeBlasMathMode::HOST_DEFAULT, false};
    throw std::invalid_argument("PsiFormer reduced-precision BLAS mode requires an accelerator");
  case PsiFormerAcceleratorBackend::CUDA:
    if (requested == PsiFormerBackendMathMode::CUDA_TF32)
      return {requested, backend, PsiFormerNativeBlasMathMode::CUDA_TF32, true};
    return {requested, backend, PsiFormerNativeBlasMathMode::CUDA_PEDANTIC, false};
  case PsiFormerAcceleratorBackend::HIP:
    if (requested == PsiFormerBackendMathMode::CUDA_TF32)
      throw std::invalid_argument("PsiFormer CUDA TF32 BLAS mode is unavailable on HIP");
    return {requested, backend, PsiFormerNativeBlasMathMode::HIP_DEFAULT_STRICT, false};
  case PsiFormerAcceleratorBackend::OPENMP_TARGET:
  case PsiFormerAcceleratorBackend::SYCL:
    throw std::invalid_argument(
        "PsiFormer BLAS math-mode mapping is not implemented for this backend");
  }
  throw std::invalid_argument("PsiFormer BLAS math-mode mapping received an invalid backend");
}

/// Return the stable diagnostic spelling for one native behavior.
inline const char* psiFormerNativeBlasMathModeName(PsiFormerNativeBlasMathMode mode) noexcept
{
  switch (mode)
  {
  case PsiFormerNativeBlasMathMode::HOST_DEFAULT:
    return "host_default";
  case PsiFormerNativeBlasMathMode::CUDA_PEDANTIC:
    return "cuda_pedantic";
  case PsiFormerNativeBlasMathMode::CUDA_TF32:
    return "cuda_tf32";
  case PsiFormerNativeBlasMathMode::HIP_DEFAULT_STRICT:
    return "hip_default_strict";
  }
  return "unknown";
}

/** Execute an operation under one temporary controller mode and restore it.
 *
 * Controller supplies ``mode_type``, ``currentMode()``, and ``setMode(mode)``.
 * Restoration happens on success and when the operation throws.  A restoration
 * failure is always surfaced; after an operation failure it is nested so callers are
 * never told that shared handle state was restored when it was not.
 */
template<class Controller, class Operation>
void executeWithPsiFormerBlasMathMode(Controller& controller,
                                      typename Controller::mode_type required,
                                      Operation&& operation)
{
  const typename Controller::mode_type previous = controller.currentMode();
  if (previous == required)
  {
    std::forward<Operation>(operation)();
    return;
  }

  controller.setMode(required);
  try
  {
    std::forward<Operation>(operation)();
  }
  catch (...)
  {
    const std::exception_ptr operation_failure = std::current_exception();
    try
    {
      controller.setMode(previous);
    }
    catch (...)
    {
      std::throw_with_nested(std::runtime_error(
          "PsiFormer BLAS math-mode restoration failed after an operation failure"));
    }
    std::rethrow_exception(operation_failure);
  }
  controller.setMode(previous);
}

} // namespace qmcplusplus::psiformer

#endif // QMCPLUSPLUS_PSIFORMER_BLAS_MATH_MODE_H
