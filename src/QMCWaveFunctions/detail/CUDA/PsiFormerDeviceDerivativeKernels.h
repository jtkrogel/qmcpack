//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceDerivativeKernels.h
 * @brief Allocation-free CUDA/HIP leaves for the ordinary FP64 PsiFormer reverse pass.
 *
 * These entry points deliberately stop below executor level.  Every reduction has
 * one output owner and caller-supplied bounded storage; no function allocates or
 * publishes a parameter response.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_KERNELS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_KERNELS_H

#include "PsiFormerDeviceRuntime.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAttention.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qmcplusplus::psiformer::device
{

/** A logical strided FP64 copy interval.  Offsets are measured in elements. */
struct DerivativeVectorLayout
{
  std::size_t element_count      = 0;
  std::size_t source_offset      = 0;
  std::size_t source_stride      = 0;
  std::size_t destination_offset = 0;
  std::size_t destination_stride = 0;
};

/** A real source and an interleaved-complex destination. */
struct RealToComplexAxpyLayout
{
  std::size_t element_count       = 0;
  std::size_t source_offset       = 0;
  std::size_t source_stride       = 0;
  std::size_t accumulator_offset  = 0;
  std::size_t accumulator_stride  = 0;
};

/** Fixed-size pieces for a deterministic response/direction dot product. */
struct ResponseDotLayout
{
  std::size_t element_count        = 0;
  std::size_t elements_per_partial = 0;
  std::size_t partial_capacity     = 0;
  std::size_t response_offset      = 0;
  std::size_t response_stride      = 0;
  std::size_t direction_offset     = 0;
  std::size_t direction_stride     = 0;
};

/** Deterministic finite-scan pieces for a strided FP64 interval. */
struct DerivativeFiniteScanLayout
{
  std::size_t element_count       = 0;
  std::size_t elements_per_status = 0;
  std::size_t status_capacity     = 0;
  std::size_t source_offset       = 0;
  std::size_t source_stride       = 0;
};

/** One caller-owned reduction/scan status record. */
struct DerivativeFiniteStatus
{
  std::size_t first_nonfinite = 0;
  std::size_t nonfinite_count = 0;
};

/** Padded [configuration,determinant,row-electron,orbital] value layout. */
struct OrbitalProductReverseLayout
{
  std::size_t configuration_count       = 0;
  std::size_t determinant_count         = 0;
  std::size_t electron_count            = 0;
  std::size_t row_stride                = 0;
  std::size_t determinant_stride        = 0;
  std::size_t configuration_stride      = 0;
};

/** Output-owned outer-serial ordinary envelope-parameter reverse layout.
 *
 * One call covers one configuration and one spin channel, selected by the
 * half-open electron interval.  The response is exactly one canonical parameter
 * vector; configurations are intentionally serialized outside this leaf.
 * ``pi_response_offset`` and ``zeta_response_offset`` are independent canonical
 * offsets and are never derived from one another.
 */
struct EnvelopeParameterReverseLayout
{
  std::size_t configuration_count          = 0;
  std::size_t determinant_count            = 0;
  std::size_t electron_count               = 0;
  std::size_t nucleus_count                = 0;
  std::size_t electron_begin               = 0;
  std::size_t electron_end                 = 0;
  std::size_t matrix_row_stride            = 0;
  std::size_t matrix_determinant_stride    = 0;
  std::size_t matrix_configuration_stride  = 0;
  std::size_t distance_electron_stride     = 0;
  std::size_t distance_configuration_stride = 0;
  std::size_t parameter_orbital_stride     = 0;
  std::size_t parameter_determinant_stride = 0;
  std::size_t pi_response_offset           = 0;
  std::size_t zeta_response_offset         = 0;
};

/** Output-owned outer-serial ordinary cusp-alpha reverse layout.
 *
 * The response is exactly one canonical parameter vector; configurations are
 * intentionally serialized outside this leaf.
 */
struct CuspAlphaReverseLayout
{
  std::size_t configuration_count          = 0;
  std::size_t pair_count                   = 0;
  std::size_t distance_configuration_stride = 0;
  std::size_t same_response_offset         = 0;
  std::size_t opposite_response_offset     = 0;
  bool has_same_alpha                      = false;
};

static_assert(std::is_standard_layout_v<DerivativeVectorLayout>);
static_assert(std::is_trivially_copyable_v<DerivativeVectorLayout>);
static_assert(std::is_standard_layout_v<RealToComplexAxpyLayout>);
static_assert(std::is_trivially_copyable_v<RealToComplexAxpyLayout>);
static_assert(std::is_standard_layout_v<ResponseDotLayout>);
static_assert(std::is_trivially_copyable_v<ResponseDotLayout>);
static_assert(std::is_standard_layout_v<DerivativeFiniteScanLayout>);
static_assert(std::is_trivially_copyable_v<DerivativeFiniteScanLayout>);
static_assert(std::is_standard_layout_v<DerivativeFiniteStatus>);
static_assert(std::is_trivially_copyable_v<DerivativeFiniteStatus>);
static_assert(std::is_standard_layout_v<OrbitalProductReverseLayout>);
static_assert(std::is_trivially_copyable_v<OrbitalProductReverseLayout>);
static_assert(std::is_standard_layout_v<EnvelopeParameterReverseLayout>);
static_assert(std::is_trivially_copyable_v<EnvelopeParameterReverseLayout>);
static_assert(std::is_standard_layout_v<CuspAlphaReverseLayout>);
static_assert(std::is_trivially_copyable_v<CuspAlphaReverseLayout>);

/// Number of partial/status records written by a checked response-dot layout.
std::size_t responseDotPartialCount(const ResponseDotLayout& layout);

/// Number of status records written by a checked finite-scan layout.
std::size_t derivativeFiniteStatusCount(const DerivativeFiniteScanLayout& layout);

/// Zero the logical destinations without touching stride padding.
Error launchDerivativeZero(Stream stream,
                           const DerivativeVectorLayout& layout,
                           double* destination);

/// Copy the logical source interval without touching stride padding.
Error launchDerivativeCopy(Stream stream,
                           const DerivativeVectorLayout& layout,
                           const double* source,
                           double* destination);

/// Accumulate ``(alpha_real+i*alpha_imag)*source`` into interleaved complex storage.
Error launchRealToComplexAxpy(Stream stream,
                              const RealToComplexAxpyLayout& layout,
                              double alpha_real,
                              double alpha_imag,
                              const double* source,
                              double* accumulator);

/** Produce deterministic caller-bounded partials of ``dot(response,direction)``.
 *
 * Each partial and status record has exactly one thread owner.  A partial containing
 * a non-finite input or product is written as zero and identified by its status.
 */
Error launchResponseDotDirectionPartials(Stream stream,
                                         const ResponseDotLayout& layout,
                                         const double* response,
                                         const double* direction,
                                         double* partials,
                                         DerivativeFiniteStatus* partial_status);

/** Reduce response-dot partials in increasing order and aggregate their status. */
Error launchFinalizeResponseDotDirection(Stream stream,
                                         std::size_t partial_count,
                                         std::size_t logical_element_count,
                                         const double* partials,
                                         const DerivativeFiniteStatus* partial_status,
                                         double* result,
                                         DerivativeFiniteStatus* aggregate_status);

/** Complete response/direction dot product using only the supplied bounded scratch. */
Error launchResponseDotDirection(Stream stream,
                                 const ResponseDotLayout& layout,
                                 const double* response,
                                 const double* direction,
                                 double* partials,
                                 DerivativeFiniteStatus* partial_status,
                                 double* result,
                                 DerivativeFiniteStatus* aggregate_status);

/** Reverse ``output = residual + tanh(preactivation)`` from retained tanh output. */
Error launchTanhOutputResidualReverse(Stream stream,
                                      const BatchedValueLayout& layout,
                                      const double* tanh_output,
                                      const double* output_adjoint,
                                      double* tanh_input_adjoint,
                                      double* residual_adjoint);

/** Reverse context[q,h,f] = sum_k attention[h,q,k] * value[k,h,f]. */
Error launchAttentionContextReverse(Stream stream,
                                    const BatchedAttentionForwardLayout& layout,
                                    const double* attention,
                                    const double* value,
                                    const double* context_adjoint,
                                    double* attention_adjoint,
                                    double* value_adjoint);

/** Reverse row softmax and scaled Q/K products directly into Q and K adjoints. */
Error launchSoftmaxScaledQkReverse(Stream stream,
                                   const BatchedAttentionForwardLayout& layout,
                                   const double* query,
                                   const double* key,
                                   const double* probabilities,
                                   const double* probability_adjoint,
                                   double* query_adjoint,
                                   double* key_adjoint);

/** Prepare Bbar=Abar*E and Ebar=Abar*B for the orbital product A=B*E. */
Error launchOrbitalBbarEbar(Stream stream,
                            const OrbitalProductReverseLayout& layout,
                            const double* backflow,
                            const double* envelope,
                            const double* orbital_adjoint,
                            double* backflow_adjoint,
                            double* envelope_adjoint);

/** Accumulate one spin channel's ordinary envelope pi/zeta response. */
Error launchEnvelopeParameterReverse(Stream stream,
                                     const EnvelopeParameterReverseLayout& layout,
                                     const double* distances,
                                     const double* pi,
                                     const double* zeta,
                                     const double* envelope_adjoint,
                                     double* response);

/** Accumulate ordinary same/opposite-spin cusp-alpha response. */
Error launchCuspAlphaReverse(Stream stream,
                             const CuspAlphaReverseLayout& layout,
                             const double* distances,
                             const std::uint8_t* same_spin,
                             double same_alpha,
                             double opposite_alpha,
                             double* response);

/** Scan a strided interval into deterministic caller-bounded status records. */
Error launchDerivativeFiniteScan(Stream stream,
                                 const DerivativeFiniteScanLayout& layout,
                                 const double* source,
                                 DerivativeFiniteStatus* status);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_KERNELS_H
