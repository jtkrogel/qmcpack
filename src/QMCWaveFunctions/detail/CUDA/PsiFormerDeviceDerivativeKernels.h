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
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicGeometry.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qmcplusplus::psiformer::device
{

/** Compact first-failure record: 16-bit stage, 40-bit index, 8-bit typed status.
 * All-one bits mean success; ordered launches and atomic minimum retain the earliest failure. */
using InferenceStatusWord = unsigned long long;
static_assert(sizeof(InferenceStatusWord) == 8, "Inference diagnostics require a 64-bit status word");

/// Reject records whose stage/index cannot be represented without losing identity.
constexpr std::size_t INFERENCE_STATUS_MAX_STAGE = 0xfffe;
constexpr std::size_t INFERENCE_STATUS_MAX_INDEX = (UINT64_C(1) << 40) - 1;
constexpr unsigned int INFERENCE_STATUS_JET_DOMAIN = 0x80;

/** Aggregate spatial leaf statuses into one caller-owned sticky first-failure word. */
Error launchAggregateOpenInferenceStatus(Stream stream, const OpenSpatialStatus* statuses,
                                         std::size_t count, std::size_t stage,
                                         InferenceStatusWord* sticky);

/** Aggregate softmax leaf statuses without synchronizing or exposing a result. */
Error launchAggregateJetInferenceStatus(Stream stream, const device_math::JetMathStatus* statuses,
                                        std::size_t count, std::size_t stage,
                                        InferenceStatusWord* sticky);

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

/** One outer-serial replacement-score fold with no Q-by-P storage.
 *
 * Coefficients and ratios are interleaved complex FP64.  The wrapper is called once
 * per point in canonical order, so each accumulator scalar and sample weight sum has
 * one launch owner and requires no atomics.
 */
struct ECPWeightedResponseLayout
{
  std::size_t parameter_count = 0;
  std::size_t channel_count = 0;
  std::size_t sample_count = 0;
  std::size_t sample_index = 0;
  std::size_t point_count = 0;
  std::size_t point_index = 0;
  std::size_t response_offset = 0;
  std::size_t response_stride = 0;
  std::size_t coefficient_channel_stride_complex = 0;
  std::size_t accumulator_channel_stride_fp64 = 0;
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

/** Output-owned bounded-batch ordinary envelope-parameter reverse layout.
 *
 * One call covers a configuration tile and one spin channel, selected by the
 * half-open electron interval.  The response is exactly one canonical parameter
 * vector; each parameter owner reduces configurations in increasing order.
 * Sample weights must already be included in the envelope adjoint.
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

/** Output-owned bounded-batch ordinary cusp-alpha reverse layout.
 *
 * The response is exactly one canonical parameter vector; configurations are
 * reduced deterministically inside each alpha owner's thread.
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
static_assert(std::is_standard_layout_v<ECPWeightedResponseLayout>);
static_assert(std::is_trivially_copyable_v<ECPWeightedResponseLayout>);
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

/** In-place real scale; only the destination interval of layout is used.
 * Raw borrowed launch: caller owns device/capacity/lifetime and must finite-scan
 * ALL inputs before mutation and outputs before publication. No allocation/sync.
 * Alpha must be finite even for an empty no-op interval.
 */
Error launchDerivativeScale(Stream stream, const DerivativeVectorLayout& layout,
                            double alpha, double* destination);

/** Real y += alpha*x with separately rounded product/add (no FMA range widening).
 * Same logical span is allowed; other overlapping bounding byte spans reject,
 * including strided bounding overlaps. Caller finite-admits all input/output
 * lanes and owns allocation extents/current device, as for scale above.
 */
Error launchDerivativeAxpy(Stream stream, const DerivativeVectorLayout& layout,
                           double alpha, const double* source, double* destination);

/// Accumulate ``(alpha_real+i*alpha_imag)*source`` into interleaved complex storage.
Error launchRealToComplexAxpy(Stream stream,
                              const RealToComplexAxpyLayout& layout,
                              double alpha_real,
                              double alpha_imag,
                              const double* source,
                              double* accumulator);

/** Fold one complete weighted replacement response and update its sample sum.
 *
 * All point launches in one contraction must use the same ordered stream.  Every
 * input/output buffer must cover the extents described by layout, and destructive
 * outputs must be disjoint from each other and from all inputs.
 */
Error launchECPWeightedResponse(Stream stream,
                                const ECPWeightedResponseLayout& layout,
                                const double* bare_weights,
                                const double* complete_ratios_interleaved,
                                const double* coefficients_interleaved,
                                const double* response,
                                double* sample_weight_sums_interleaved,
                                double* accumulators_interleaved);

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

/// Accumulate the selected periodic pair-bias parameter VJP from softmax roots.
Error launchPeriodicPairAttentionParameterReverse(Stream stream,
    const BatchedAttentionForwardLayout& layout, const periodic_geometry::Cell& cell,
    std::size_t spin_up_count, const double* positions, const double* probabilities,
    const double* probability_adjoint, double* weight_response, double* bias_response);

/** Prepare Bbar=Abar*E and Ebar=Abar*B for the orbital product A=B*E. */
Error launchOrbitalBbarEbar(Stream stream,
                            const OrbitalProductReverseLayout& layout,
                            const double* backflow,
                            const double* envelope,
                            const double* orbital_adjoint,
                            double* backflow_adjoint,
                            double* envelope_adjoint);

/** Weight orbital seeds and pack spin-selected backflow adjoints for dense BLAS.
 * Matrices are contiguous [B,D,N,N]; packed outputs are [B,N,D*N]. Both spin
 * destinations are fully written, with zero rows for the opposite spin.
 */
Error launchWeightedOrbitalProductReverse(Stream stream,
    const OrbitalProductReverseLayout& layout, std::size_t spin_up_count,
    const double* configuration_weights, const double* backflow, const double* envelope,
    const double* orbital_adjoint, double* up_adjoint, double* down_adjoint,
    double* envelope_adjoint);

/** Seed independent channel log derivatives as c[b,d] A[b,d]^{-T}.
 * Zero coefficients write zero without reading an unavailable inverse. The
 * caller admits every nonzero channel; determinant-sum nodes are irrelevant.
 */
Error launchIndependentChannelSeeds(Stream stream, const OrbitalProductReverseLayout& layout,
    const double* coefficients, const double* inverses, double* orbital_adjoint);

/** Build contiguous electron-ion and unordered electron-pair radii for score heads.
 * Pair labels are shared across configurations; inputs use interleaved XYZ.
 * A supplied cell selects the CPU-equivalent torus radius, not a minimum image.
 */
Error launchScoreGeometry(Stream stream, std::size_t configuration_count,
    std::size_t electron_count, std::size_t nucleus_count, std::size_t spin_up_count,
    const double* positions, const double* nuclei, double* ion_distances,
    double* pair_distances, std::uint8_t* same_spin,
    const periodic_geometry::Cell* cell = nullptr);

/// Add two contiguous canonical response vectors into a distinct destination.
Error launchAddParameterResponses(Stream stream, std::size_t count,
    const double* left, const double* right, double* destination);

/** Apply an ordinary parameter-direction tanh rule, including a bias tangent. */
Error launchBiasTanhParameterDirection(Stream stream, const BatchedValueLayout& layout,
    const double* tanh_value, const double* preactivation_direction,
    const double* bias_direction, double* output_direction);

/// Borrow fixed geometry and device parameter tangents until the stream completes.
struct PeriodicPairAttentionDirectionInputs
{
  const periodic_geometry::Cell* cell = nullptr;
  const double* positions = nullptr;
  const double* weight_direction = nullptr;
  const double* bias_direction = nullptr;
  std::size_t spin_up_count = 0;
};

/** Differentiate attention probabilities and context along one parameter direction.
 * Buffers are ordinary values/tangents, never coordinate-jet planes. Probability
 * directions are caller-owned scratch and used to form the context tangent.
 */
Error launchAttentionParameterDirection(Stream stream, const BatchedAttentionForwardLayout& layout,
    const double* query, const double* key, const double* value, const double* probabilities,
    const double* query_direction, const double* key_direction, const double* value_direction,
    double* probability_direction, double* context_direction,
    const PeriodicPairAttentionDirectionInputs* pair_attention = nullptr);

/** Bind the six explicit spin-channel orbital parameter intervals. */
struct OrbitalDirectionParameters
{
  const double* backflow_up;
  const double* backflow_down;
  const double* pi_up;
  const double* pi_down;
  const double* zeta_up;
  const double* zeta_down;
};

/** Differentiate ordinary B*E orbitals along a parameter/feature direction.
 * The compatibility name retains open defaults; a cell selects torus envelopes.
 */
Error launchOpenOrbitalParameterDirection(Stream stream, const OpenOrbitalJetLayout& layout,
    const double* features, const double* feature_direction, const double* positions,
    const double* nuclei, OrbitalDirectionParameters parameters, OrbitalDirectionParameters direction,
    const double* retained_backflow, const double* retained_envelope, double* orbital_direction,
    const periodic_geometry::Cell* cell = nullptr);

/** Contract regular determinant seeds with orbital tangents and add cusp tangents.
 * An optional fixed cell selects the same pair radius as the forward cusp.
 */
Error launchLogScoreParameterDirection(Stream stream, std::size_t batch, std::size_t ne,
    std::size_t determinants, std::size_t spin_up_count, const double* positions,
    const double* matrix_seeds, const double* orbital_direction, double same_alpha, double opposite_alpha,
    double same_alpha_direction, double opposite_alpha_direction, double* score_direction,
    const periodic_geometry::Cell* cell = nullptr);

/** Seed a weighted kinetic lifted reverse with mapped real-space root adjoints.
 * Inverse masses are per electron (null means one); optional total drift is
 * contiguous in physical 3Ne lanes within a caller-specified configuration stride,
 * including for a compact target root. All physical coefficients are validated.
 * Null total drift uses this component's root gradient. Destinations must not alias inputs.
 * Gradient-only mode requires explicit drift and sets every Laplacian seed to
 * zero, providing the imaginary kinetic response of a fixed complex outer phase.
 */
Error launchKineticRootAdjoints(Stream stream, const SpatialJetLayout& root_layout,
    const double* forward_root, const double* configuration_weights,
    const double* inverse_masses, const double* total_log_gradient,
    std::size_t total_gradient_configuration_stride, double* root_adjoint,
    OpenSpatialStatus* status, bool gradient_only = false);

/** Accumulate one spin channel's ordinary envelope pi/zeta response. */
Error launchEnvelopeParameterReverse(Stream stream,
                                     const EnvelopeParameterReverseLayout& layout,
                                     const double* distances,
                                     const double* pi,
                                     const double* zeta,
                                     const double* envelope_adjoint,
                                     double* response);

/** Accumulate ordinary same/opposite-spin cusp-alpha response.
 * Optional configuration weights contract a tile directly into one response;
 * a null weight pointer preserves the unweighted one-configuration API.
 */
Error launchCuspAlphaReverse(Stream stream,
                             const CuspAlphaReverseLayout& layout,
                             const double* distances,
                             const std::uint8_t* same_spin,
                             double same_alpha,
                             double opposite_alpha,
                             double* response,
                             const double* configuration_weights = nullptr);

/** Scan a strided interval into deterministic caller-bounded status records. */
Error launchDerivativeFiniteScan(Stream stream,
                                 const DerivativeFiniteScanLayout& layout,
                                 const double* source,
                                 DerivativeFiniteStatus* status);

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_DERIVATIVE_KERNELS_H
