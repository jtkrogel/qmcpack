//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceSpatial.h
 * @brief CUDA/HIP entry points for the PsiFormer spatial-jet foundation.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H

#include "PsiFormerDeviceDense.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

namespace qmcplusplus::psiformer::device
{

/** Scatter one admitted local target root into separate complete VGL outputs.
 * Does not mark a request complete; the owner must admit all tiles first.
 */
Error launchScatterSpatialTargets(Stream stream, const SpatialJetLayout& source, const SpatialJetLayout& target,
    const double* source_phase, const double* source_jets, const double* source_ratios,
    double* target_phase, double* target_jets, double* target_ratios);

/** Return value-only signed-log results, including exact nodes, without inverses. */
Error launchFinalValueCombination(Stream stream, std::size_t configuration_count,
    const device_determinant::CombinationMetadata* determinant_metadata,
    const double* cusp, double* phase, double* logabs, OpenSpatialStatus* status);

/** Transform batched spatial-logit rows to softmax jets in place. */
Error launchSoftmaxJetRows(Stream stream,
                           const SoftmaxJetRowLayout& layout,
                           double* jets,
                           device_math::JetMathStatus* row_status);

/** Apply one dense weight matrix independently to every B*plane slice. */
void denseJetsForward(AcceleratorBlasHandle& handle,
                      const SpatialDenseJetLayout& layout,
                      const double* source,
                      const double* weight,
                      double* target);

/** Apply three dense projections independently to every B*plane slice. */
void projectQkvJetsForward(AcceleratorBlasHandle& handle,
                           const SpatialDenseJetLayout& layout,
                           const double* source,
                           const double* query_weight,
                           const double* key_weight,
                           const double* value_weight,
                           double* query,
                           double* key,
                           double* value);

/// Borrow host cell metadata and device arrays through the ordered attention launch.
/// The caller retains every allocation until the selected stream has completed.
struct PeriodicPairAttentionInputs
{
  const periodic_geometry::Cell* cell = nullptr;
  const double* positions = nullptr;
  const double* weights = nullptr;
  const double* bias = nullptr;
  const std::size_t* active_electrons = nullptr;
  std::size_t spin_up_count = 0;
};

/** Form Q*K^T jets and then transform every head/query row with jet softmax. */
Error launchAttentionJetWeights(
    Stream stream,
    const SpatialAttentionJetLayout& layout,
    const double* query,
    const double* key,
    double* attention,
    device_math::JetMathStatus* row_status,
    double* retained_logits = nullptr,
    const PeriodicPairAttentionInputs* pair_attention = nullptr);

/** Form context jets from prepared attention probabilities and value jets. */
Error launchAttentionContextJets(Stream stream,
                                 const SpatialAttentionJetLayout& layout,
                                 const double* attention,
                                 const double* value,
                                 double* context);

Error launchOpenFeatureJets(Stream stream,
                            const OpenFeatureJetLayout& layout,
                            const double* positions,
                            const double* nuclei,
                            const std::size_t* active_electrons,
                            double* output,
                            OpenSpatialStatus* status);

/** Explicit OPEN pair-parallel alternative; small Ne keeps the scalar path.
 * Reuses output/status owners and canonical failure replay without new global scratch.
 */
Error launchOpenFeatureJetsCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const double* positions, const double* nuclei, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status);

/// Build smooth-torus feature jets with seven inputs per nucleus plus spin.
Error launchPeriodicFeatureJets(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status);

/// Pair-disjoint FULL/target periodic jets; same owners and scalar failure replay.
Error launchPeriodicFeatureJetsCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status);

/// Pair-disjoint periodic VALUE features with canonical GPU failure replay.
Error launchPeriodicFeatureValuesCooperative(Stream stream, const OpenFeatureJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status);

Error launchBiasTanhJets(Stream stream,
                         const SpatialElementwiseJetLayout& layout,
                         const double* input,
                         const double* bias,
                         double* output,
                         OpenSpatialStatus* status,
                         double* retained_preactivation = nullptr);

Error launchResidualSpatialJets(Stream stream,
                                const SpatialElementwiseJetLayout& layout,
                                const double* left,
                                const double* right,
                                double* output,
                                OpenSpatialStatus* status);

Error launchOpenCuspJets(Stream stream,
                         const OpenCuspJetLayout& layout,
                         const double* positions,
                         const std::size_t* active_electrons,
                         double same_spin_alpha,
                         double opposite_spin_alpha,
                         double* output,
                         OpenSpatialStatus* status);

/// Opt-in OPEN VALUE pair chunks with canonical ordered summation and failure replay.
Error launchOpenCuspValuesCooperative(Stream stream, const OpenCuspJetLayout& layout,
    const double* positions, const std::size_t* active_electrons,
    double same_spin_alpha, double opposite_spin_alpha, double* output, OpenSpatialStatus* status);

/// Build periodic radial cusp jets, including the non-Euclidean radial trace rule.
Error launchPeriodicCuspJets(Stream stream, const OpenCuspJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    double* output, OpenSpatialStatus* status);

/// Bounded periodic VALUE pair chunks, retaining the scalar subtraction order.
Error launchPeriodicCuspValuesCooperative(Stream stream, const OpenCuspJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    double* output, OpenSpatialStatus* status);

/// Gather a compact physical target interval, replaying the canonical cusp on cached failure.
Error launchGatherCuspTargets(Stream stream, const OpenCuspJetLayout& complete,
    const OpenCuspJetLayout& target, const double* positions,
    const std::size_t* active_electrons, double same_spin_alpha, double opposite_spin_alpha,
    const double* cached, const OpenSpatialStatus* cached_status,
    double* output, OpenSpatialStatus* status, const periodic_geometry::Cell* cell = nullptr);

Error launchFinalSpatialCombination(
    Stream stream,
    const FinalSpatialJetLayout& layout,
    const device_determinant::CombinationMetadata* determinant_metadata,
    const device_determinant::DerivativeStatus* determinant_status,
    const double* determinant_gradient,
    const double* determinant_laplacian_log,
    const double* cusp,
    double* phase,
    double* output,
    double* laplacian_ratio,
    OpenSpatialStatus* status);

/** Project feature jets into an already admitted retained backflow destination.
 * Uses existing strided dense GEMM adapters without packing or temporary storage.
 * The caller binds the BLAS handle to the subsequent product-kernel stream.
 */
void projectOpenOrbitalJets(AcceleratorBlasHandle& handle, const OpenOrbitalJetLayout& layout,
    const double* features, const double* backflow_up, const double* backflow_down,
    double* projected_backflow);

/** Build orbital jets, optionally retaining ordinary B and E factors separately.
 * Retained buffers are paired, distinct, and use the complete orbital jet layout
 * [configuration,plane,D,N,N], matching the requested value-only or full-VGL mode.
 * A non-null projected_backflow skips scalar feature contractions; it may equal
 * retained_backflow, but must be distinct from output and retained_envelope.
 */
Error launchOpenOrbitalJets(Stream stream,
                            const OpenOrbitalJetLayout& layout,
                            const double* features,
                            const double* positions,
                            const double* nuclei,
                            const double* backflow_up,
                            const double* backflow_down,
                            const double* pi_up,
                            const double* pi_down,
                            const double* zeta_up,
                            const double* zeta_down,
                            const std::size_t* active_electrons,
                            double* output,
                            OpenSpatialStatus* status,
                            double* retained_backflow = nullptr,
                            double* retained_envelope = nullptr,
                            const double* projected_backflow = nullptr);

/// Tile ordinary OPEN VALUE feature contractions without global projection storage.
Error launchOpenOrbitalValuesTiled(Stream stream, const OpenOrbitalJetLayout& layout,
    const double* features, const double* positions, const double* nuclei,
    const double* backflow_up, const double* backflow_down,
    const double* pi_up, const double* pi_down, const double* zeta_up, const double* zeta_down,
    const std::size_t* active_electrons, double* output, OpenSpatialStatus* status);

/// One immutable reference-zero envelope bank; no neural or determinant reuse.
struct SparseEnvelopeAnchorView
{
  const std::size_t* references = nullptr;
  const std::size_t* changed_electrons = nullptr;
  double* positions = nullptr;
  double* envelope = nullptr;
  unsigned long long* counters = nullptr;
  bool capture = false;
  bool ready = false;
};

/// Build real Gamma-periodic orbital jets and optional retained product factors.
Error launchPeriodicOrbitalJets(Stream stream, const OpenOrbitalJetLayout& layout,
    const periodic_geometry::Cell& cell, const double* features,
    const double* positions, const double* nuclei, const double* backflow_up,
    const double* backflow_down, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const std::size_t* active_electrons,
    double* output, OpenSpatialStatus* status,
    double* retained_backflow = nullptr, double* retained_envelope = nullptr,
    const double* projected_backflow = nullptr,
    SparseEnvelopeAnchorView anchor = {});

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_SPATIAL_H
