//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceInference.h
 * @brief Prepared FP64 forward/VGL graph, optional mixed affine planes and reverse tapes.
 *
 * This chains actual CUDA/HIP leaves for a complete imported model. It is not
 * itself a publication owner: the production crowd adapter binds it to the
 * control-plane lease and wait protocol. Parameters are immutable for this
 * instance. A synchronous debug policy and guarded neural-stage aggregation
 * preserve first-error identity without publishing failed results. Retaining a
 * value tape supplies reverse intermediates, not a qualified score evaluator.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceCoordinates.h"
#include "PsiFormerDeviceSpatial.h"
#include "PsiFormerDeviceSpatialTraces.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceDerivativeKernels.h"
#include "PsiFormerDeviceMixedValueDense.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFoldedDense.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicGeometry.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/// Choose eager diagnostics or defer only neural stages with defined failure outputs.
enum class InferenceStatusPolicy
{
  SYNCHRONOUS_DEBUG,
  DEFERRED_NEURAL
};

/// Read the opt-in once when constructing an explicit preparation descriptor.
inline bool configuredCuspStateReuse()
{
  const char* value = std::getenv("PSIFORMER_GPU_CUSP_STATE_REUSE");
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("PSIFORMER_GPU_CUSP_STATE_REUSE requires 0 or 1");
}

/** Explicit numerical-only target preparation; all-zero keeps the reference graph.
 * Budgets charge this graph's device storage and a deterministic scalar-work
 * proxy. Shared model images/vendor storage and total runtime need separate admission.
 */
struct TargetVglPreparation
{
  std::size_t max_targets = 0;
  std::size_t device_workspace_budget_bytes = 0;
  std::size_t scalar_work_budget = 0;
  /// Default zero preserves scalar traces; positive cap admits an additional bounded real/open leaf.
  std::size_t parallel_trace_workspace_budget_bytes = 0;
  /// Only non-retained explicit FULL_VGL targets may own the O(B*Ne) cusp cache.
  bool reuse_cusp_state = configuredCuspStateReuse();
};

struct TargetVglCost
{
  std::size_t device_workspace_bytes = 0;
  std::size_t host_workspace_bytes = 0;
  std::size_t charged_scalar_work_units = 0;
  std::size_t tile_count = 0;
};

/// Count only the B1 max-K real prefix: no real head, retained tapes or responseP.
inline TargetVglCost estimateHeadFreeTargetPrefixCost(const PsiFormerExecutionPlan& plan, std::size_t targets)
{
  const auto& s = plan.modelShape();
  const auto n = s.electrons(), f = s.feature_dimension;
  if (!targets || targets > n)
    throw std::invalid_argument("Head-free target prefix requires K within physical Ne");
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Target prefix extent overflow"); };
  const auto add = [](auto a, auto b) { return spatial_detail::checkedAdd(a, b, "Target prefix bytes overflow"); };
  const auto planes = add(1, mul(4, targets)), u = mul(n, f), nn = mul(n, n);
  const auto raw = mul(n, plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]);
  const auto host = add(mul(u, sizeof(OpenSpatialStatus)), mul(mul(s.attention_heads, n), sizeof(device_math::JetMathStatus)));
  const auto bytes = add(mul(8, add(mul(3, n), mul(planes, add(add(raw, mul(8, u)), add(mul(s.attention_heads, nn), 1))))),
      add(host, add(sizeof(std::size_t), sizeof(InferenceStatusWord))));
  const auto tiles = n / targets + (n % targets != 0);
  const auto work = add(mul(3, mul(s.determinants, mul(nn, nn))), mul(4, mul(s.determinants, mul(tiles, mul(nn, n)))));
  return {bytes, host, work, tiles};
}

/** Count every non-retained target graph buffer before its first device reserve.
 * Work proxy: 3*B*D*Ne^4 plus 4*B*D*T*Ne^3 for repeated factor/inverse work.
 * Compensated inner sums have several operations/site; this is not a FLOP bound.
 */
inline TargetVglCost estimateTargetVglCost(const PsiFormerExecutionPlan& plan, std::size_t batch,
                                          std::size_t targets)
{
  const auto& shape = plan.modelShape();
  const auto ne = shape.electrons(), dets = shape.determinants, width = shape.feature_dimension;
  if (batch == 0 || targets == 0 || targets > ne)
    throw std::invalid_argument("Target VGL preparation requires positive batch and targets within physical Ne");
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Target VGL cost product overflow");
  };
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Target VGL cost sum overflow");
  };
  const auto planes = add(1, mul(4, targets)), full_planes = add(1, mul(4, ne));
  const auto matrices = mul(batch, mul(dets, mul(ne, ne)));
  const auto features = mul(batch, mul(planes, mul(ne, width)));
  const auto attention = mul(batch, mul(planes, mul(shape.attention_heads, mul(ne, ne))));
  const auto raw = mul(batch, mul(planes,
      mul(ne, plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0])));
  std::size_t doubles = 0;
  for (const auto count : {mul(batch, mul(3, ne)), raw, mul(8, features), attention,
      mul(planes, matrices), mul(3, matrices), mul(batch, mul(dets, ne)), mul(4, mul(batch, dets)),
      mul(batch, mul(ne, ne)), mul(batch, mul(3, targets)), mul(2, mul(batch, targets)),
      mul(batch, planes), batch, mul(batch, planes), mul(batch, targets),
      batch, mul(batch, full_planes), mul(batch, ne)})
    doubles = add(doubles, count);
  TargetVglCost cost;
  cost.device_workspace_bytes = checkedDeviceBytes<double>(doubles);
  const auto bytes = [&](std::size_t amount) {
    cost.device_workspace_bytes = add(cost.device_workspace_bytes, amount);
  };
  bytes(checkedDeviceBytes<std::size_t>(batch));
  bytes(checkedDeviceBytes<std::size_t>(mul(batch, mul(dets, ne))));
  bytes(checkedDeviceBytes<device_determinant::FactorizationMetadata>(mul(batch, dets)));
  bytes(checkedDeviceBytes<device_determinant::CombinationMetadata>(batch));
  bytes(checkedDeviceBytes<device_determinant::DerivativeStatus>(batch));
  const auto open_status = std::max(mul(batch, mul(ne, width)), matrices);
  const auto jet_status = mul(batch, mul(shape.attention_heads, ne));
  bytes(checkedDeviceBytes<OpenSpatialStatus>(open_status));
  bytes(checkedDeviceBytes<device_math::JetMathStatus>(jet_status));
  bytes(checkedDeviceBytes<InferenceStatusWord>(1));
  cost.host_workspace_bytes = add(checkedDeviceBytes<OpenSpatialStatus>(open_status),
      checkedDeviceBytes<device_math::JetMathStatus>(jet_status));
  cost.host_workspace_bytes = add(cost.host_workspace_bytes, checkedDeviceBytes<double>(mul(4, batch)));
  cost.host_workspace_bytes = add(cost.host_workspace_bytes,
      checkedDeviceBytes<device_determinant::CombinationMetadata>(batch));
  cost.tile_count = ne / targets + (ne % targets != 0);
  const auto cube = mul(ne, mul(ne, ne)), bd = mul(batch, dets);
  cost.charged_scalar_work_units = add(mul(3, mul(bd, mul(ne, cube))),
      mul(4, mul(bd, mul(cost.tile_count, cube))));
  return cost;
}

/// Charge the optional bounded trace leaf before any composite graph allocation.
inline TargetVglCost estimatePreparedTargetVglCost(const PsiFormerExecutionPlan& plan, std::size_t batch,
                                                  const TargetVglPreparation& preparation)
{
  auto cost = estimateTargetVglCost(plan, batch, preparation.max_targets);
  if (preparation.parallel_trace_workspace_budget_bytes)
  {
    const auto& environment = plan.environment();
    if ((environment.boundary != BoundaryCondition::OPEN && environment.boundary != BoundaryCondition::PERIODIC) ||
        (environment.boundary == BoundaryCondition::PERIODIC &&
         environment.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1) ||
        environment.amplitude_scalar_domain != ScalarDomain::REAL ||
        environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL || !environment.fixed_nuclei)
      throw std::invalid_argument("Parallel VGL traces require fixed-ion real FP64 open or periodic-torus geometry");
    const auto& shape = plan.modelShape();
    const auto k = preparation.max_targets;
    const auto trace = planSpatialTraceWorkspace({batch, shape.determinants, shape.electrons(),
        spatial_detail::checkedProduct(3, k, "Parallel trace lanes overflow"), k},
        preparation.parallel_trace_workspace_budget_bytes);
    cost.device_workspace_bytes = spatial_detail::checkedAdd(cost.device_workspace_bytes,
        trace.total_device_bytes, "Parallel VGL composite bytes overflow");
  }
  if (preparation.reuse_cusp_state && preparation.max_targets < plan.modelShape().electrons())
  {
    const auto planes = spatial_detail::checkedAdd(1,
        spatial_detail::checkedProduct(4, plan.modelShape().electrons(), "Cusp cache planes overflow"),
        "Cusp cache planes overflow");
    const auto count = spatial_detail::checkedProduct(batch, planes, "Cusp cache extent overflow");
    cost.device_workspace_bytes = spatial_detail::checkedAdd(cost.device_workspace_bytes,
        checkedDeviceBytes<double>(count), "Cusp cache bytes overflow");
    cost.device_workspace_bytes = spatial_detail::checkedAdd(cost.device_workspace_bytes,
        checkedDeviceBytes<OpenSpatialStatus>(batch), "Cusp cache status bytes overflow");
  }
  return cost;
}

/** Retained target tapes add two orbital factors and each layer's lifted tape.
 * This private-owner estimate does not relax the public non-retained VGL contract.
 */
inline TargetVglCost estimateRetainedTargetForwardCost(const PsiFormerExecutionPlan& plan,
                                                       std::size_t targets,
                                                       std::size_t parallel_trace_budget_bytes = 0)
{
  if (parallel_trace_budget_bytes && plan.environment().boundary != BoundaryCondition::OPEN)
    throw std::invalid_argument("Retained parallel traces require fixed-ion real FP64 open geometry");
  // The retained owner never uses the non-retained cusp cache. Reuse the leaf's
  // checked admission and exact storage charge, independently of ordinary VGL.
  auto cost = estimatePreparedTargetVglCost(plan, 1,
      TargetVglPreparation{targets, 0, 0, parallel_trace_budget_bytes, false});
  const auto& shape = plan.modelShape();
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Retained target forward extent overflow");
  };
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Retained target forward storage overflow");
  };
  const auto ne = shape.electrons(), planes = add(1, mul(4, targets));
  const auto features = mul(planes, mul(ne, shape.feature_dimension));
  const auto attention = mul(planes, mul(shape.attention_heads, mul(ne, ne)));
  const auto orbitals = mul(planes, mul(shape.determinants, mul(ne, ne)));
  const auto retained = add(mul(2, orbitals),
      mul(shape.attention_blocks, add(mul(10, features), mul(2, attention))));
  cost.device_workspace_bytes = add(cost.device_workspace_bytes, checkedDeviceBytes<double>(retained));
  return cost;
}

/** Extra device bytes for retaining an ordinary real-amplitude FP64 VALUE graph.
 * Includes inverse/solve storage and the value-only layer/head tapes. Host
 * workspace is unchanged; shared model/vendor allocations are not charged.
 */
inline std::size_t estimateRetainedValueIncrementBytes(const PsiFormerExecutionPlan& plan, std::size_t batch)
{
  if (batch == 0)
    throw std::invalid_argument("Retained value preparation requires positive batch");
  if (plan.environment().amplitude_scalar_domain != ScalarDomain::REAL)
    throw std::invalid_argument("Retained value increment requires ordinary real amplitude metadata");
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Retained value extent overflow");
  };
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Retained value storage overflow");
  };
  const auto& s = plan.modelShape();
  const auto ne = s.electrons(), rows = mul(batch, ne);
  const auto matrices = mul(mul(rows, s.determinants), ne);
  const auto layers = mul(s.attention_blocks, add(mul(8, mul(rows, s.feature_dimension)),
      mul(mul(rows, s.attention_heads), ne)));
  return checkedDeviceBytes<double>(add(add(mul(3, matrices), mul(rows, s.determinants)), layers));
}

/** Conservative cross-tile base admission, not a reference accuracy tolerance.
 * A different sign, severe sum cancellation or a nonmatching rounded base cannot
 * be assembled into one physical root; callers may retry an admitted full graph.
 */
inline void admitTargetVglBase(double phase, double logabs, double reference_phase, double reference_logabs,
                              const device_determinant::CombinationMetadata& combination)
{
  if (!device_math::isFiniteBinary64(phase) || !device_math::isFiniteBinary64(logabs) ||
      !device_math::isFiniteBinary64(reference_phase) || !device_math::isFiniteBinary64(reference_logabs) ||
      (phase != 1.0 && phase != -1.0) || phase != reference_phase || combination.severe_cancellation ||
      combination.status != device_determinant::CombinationStatus::REGULAR)
    throw std::runtime_error("Target VGL tiles have an unavailable or inconsistent base value/phase");
  const double bound = 256 * std::numeric_limits<double>::epsilon() *
      (1 + std::max(std::abs(logabs), std::abs(reference_logabs)));
  if (std::abs(logabs - reference_logabs) > bound)
    throw std::runtime_error("Target VGL tiles disagree on the rounded log amplitude");
}

/// Identify the numerical enum domain stored in a compact first-failure record.
enum class InferenceStatusDomain
{
  SPATIAL,
  SOFTMAX
};

/** Preserve a failed stage's typed status, logical index and request generation. */
class InferenceStatusError final : public std::runtime_error
{
public:
  /// Decode one admitted device diagnostic without exposing any failed output.
  InferenceStatusError(InferenceStatusWord word, std::size_t generation)
      : std::runtime_error("Device inference " +
            std::string((word & INFERENCE_STATUS_JET_DOMAIN) ? "softmax" : "spatial") +
            " stage " + std::to_string(word >> 48) + " has nonregular status " +
            std::to_string(word & 0x7f) + " at logical element " +
            std::to_string((word >> 8) & INFERENCE_STATUS_MAX_INDEX) +
            " (evaluation " + std::to_string(generation) + ")"),
        word_(word), generation_(generation)
  {}

  /// Return the leaf enum family without conflating spatial and softmax status codes.
  InferenceStatusDomain domain() const noexcept
  { return word_ & INFERENCE_STATUS_JET_DOMAIN ? InferenceStatusDomain::SOFTMAX : InferenceStatusDomain::SPATIAL; }

  /// Return the ordered diagnostic stage, including eager physical stages.
  std::size_t stage() const noexcept { return word_ >> 48; }

  /// Return the failing element/row in its stage's live logical prefix.
  std::size_t element() const noexcept { return (word_ >> 8) & INFERENCE_STATUS_MAX_INDEX; }

  /// Return the graph attempt to which this error belongs.
  std::size_t generation() const noexcept { return generation_; }

  /// Decode a spatial status only when that enum family was recorded.
  OpenSpatialStatus spatialStatus() const
  {
    if (domain() != InferenceStatusDomain::SPATIAL)
      throw std::logic_error("Requested spatial status from a softmax inference diagnostic");
    return static_cast<OpenSpatialStatus>(word_ & 0x7f);
  }

  /// Decode a softmax status only when that enum family was recorded.
  device_math::JetMathStatus jetStatus() const
  {
    if (domain() != InferenceStatusDomain::SOFTMAX)
      throw std::logic_error("Requested softmax status from a spatial inference diagnostic");
    return static_cast<device_math::JetMathStatus>(word_ & 0x7f);
  }

private:
  InferenceStatusWord word_;
  std::size_t generation_;
};

/** Read the explicit diagnostic override once during graph preparation. */
inline InferenceStatusPolicy configuredInferenceStatusPolicy()
{
  const char* value = std::getenv("PSIFORMER_GPU_STATUS_CHECKS");
  if (!value || std::string_view(value) == "synchronous")
    return InferenceStatusPolicy::SYNCHRONOUS_DEBUG;
  if (std::string_view(value) == "deferred")
    return InferenceStatusPolicy::DEFERRED_NEURAL;
  throw std::invalid_argument("PSIFORMER_GPU_STATUS_CHECKS must be synchronous or deferred");
}

/** Freeze the experimental VALUE-only LU opt-in before allocating graph owners. */
inline bool configuredCooperativeValueLu()
{
  const char* value = std::getenv("PSIFORMER_GPU_VALUE_COOPERATIVE_LU");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_VALUE_COOPERATIVE_LU must be 0 or 1");
}

/// Freeze the opt-in OPEN VALUE orbital execution map before graph allocation.
inline bool configuredTiledValueOrbitals()
{
  const char* value = std::getenv("PSIFORMER_GPU_ORBITAL_VALUE_TILED");
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("PSIFORMER_GPU_ORBITAL_VALUE_TILED must be 0 or 1");
}

/// Freeze the opt-in OPEN VALUE cusp execution map before graph allocation.
inline bool configuredCooperativeValueCusp()
{
  const char* value = std::getenv("PSIFORMER_GPU_CUSP_VALUE_COOPERATIVE");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_CUSP_VALUE_COOPERATIVE must be 0 or 1");
}

/// Keep ordinary and retained VALUE cusp admission independent and immutable.
inline bool configuredRetainedValueCusp()
{
  const char* value = std::getenv("PSIFORMER_GPU_RETAINED_VALUE_CUSP");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_RETAINED_VALUE_CUSP must be 0 or 1");
}

/// Freeze the opt-in OPEN feature execution map before graph allocation.
inline bool configuredCooperativeOpenFeatures()
{
  const char* value = std::getenv("PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE must be 0 or 1");
}

/// Freeze retained OPEN feature admission independently of ordinary inference.
inline bool configuredRetainedOpenFeatures()
{
  const char* value = std::getenv("PSIFORMER_GPU_RETAINED_OPEN_FEATURES");
  if (!value || std::string_view(value) == "0")
    return false;
  if (std::string_view(value) == "1")
    return true;
  throw std::invalid_argument("PSIFORMER_GPU_RETAINED_OPEN_FEATURES must be 0 or 1");
}

/** Own immutable layout/fixed geometry independently of retiring parameter images.
 * Successor images share this family, keeping prepared plan references valid
 * without retaining every previous O(P) parameter allocation.
 */
class PreparedDeviceModelMetadata
{
public:
  /// Validate and upload fixed geometry once for one layout/device family.
  PreparedDeviceModelMetadata(const PsiFormerExecutionPlan& plan, const std::vector<double>& nuclei)
      : PreparedDeviceModelMetadata(plan, nuclei, false)
  {}

private:
  friend class PreparedDeviceModel;

  /// Named numerical factories admit only their explicitly checked profile metadata.
  PreparedDeviceModelMetadata(const PsiFormerExecutionPlan& plan, const std::vector<double>& nuclei, bool paired_value,
      bool real_two_cell = false)
      : plan_(plan)
  {
    const auto& environment = plan_.environment();
    if (real_two_cell && !(paired_value ? matchesPairedDevicePairMetadata(environment)
                                       : matchesRealDevicePairMetadata(environment)))
      throw std::invalid_argument("Numerical two-cell model requires selected atom-centered log-complement metadata");
    if (paired_value && (environment.boundary != BoundaryCondition::PERIODIC ||
        environment.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
        environment.amplitude_scalar_domain != ScalarDomain::COMPLEX))
      throw std::invalid_argument("Numerical paired VALUE needs periodic paired-real complex amplitude metadata");
    if ((environment.boundary != BoundaryCondition::OPEN && environment.boundary != BoundaryCondition::PERIODIC) ||
        environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL ||
        (environment.amplitude_scalar_domain != ScalarDomain::REAL && !paired_value) || !environment.fixed_nuclei)
      throw std::invalid_argument("Device model supports fixed-ion real FP64 models only");
    if (environment.boundary == BoundaryCondition::PERIODIC)
    {
      if (real_two_cell)
        periodic_cell_ = periodic_geometry::makeTwoCell(environment.lattice_vectors,
            environment.periodic_profile->cells.primitive.lattice_vectors, environment.periodic_axes);
      else if (environment.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1)
        throw std::invalid_argument("Periodic device metadata requires the torus-v1 feature policy");
      else
        periodic_cell_ = periodic_geometry::makeCell(environment.lattice_vectors, environment.periodic_axes);
    }
    if (nuclei.size() != spatial_detail::checkedProduct(3, plan_.modelShape().nuclei, "Device nuclear extent overflow"))
      throw std::invalid_argument("Device nuclear extent mismatch");
    if (std::any_of(nuclei.begin(), nuclei.end(), [](double value) { return !device_math::isFiniteBinary64(value); }))
      throw std::invalid_argument("Device fixed geometry contains nonfinite values");
    if (real_two_cell)
      for (std::size_t ion = 0; ion < environment.periodic_profile->primitive_motif.size(); ++ion)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          if (nuclei[3 * ion + dimension] != environment.periodic_profile->primitive_motif[ion][dimension])
            throw std::invalid_argument("Numerical two-cell model nuclei differ from the declared primitive motif");
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device_), "model device identity");
#else
    checkRuntime(cudaGetDevice(&device_), "model device identity");
#endif
    nuclei_.reserveNamed(nuclei.size(), "Inference::nuclei_", __FILE__, __LINE__);
    nuclei_.upload(nuclei.data(), nuclei.size());
  }

public:
  PreparedDeviceModelMetadata(const PreparedDeviceModelMetadata&) = delete;
  PreparedDeviceModelMetadata& operator=(const PreparedDeviceModelMetadata&) = delete;

  /// Reject accidental cross-device reuse without changing the caller's device.
  void validateCurrentDevice() const
  {
    int current = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&current), "workspace device identity");
#else
    checkRuntime(cudaGetDevice(&current), "workspace device identity");
#endif
    if (current != device_)
      throw std::invalid_argument("Device model belongs to a different current device");
  }

  /// Expose immutable model metadata and canonical uploaded intervals.
  const PsiFormerExecutionPlan& plan() const noexcept { return plan_; }

  /// Return fixed-ion storage shared by all versions in this immutable family.
  const double* nuclei() const noexcept { return nuclei_.data(); }

  /// Return immutable validated torus coefficients, or null for open geometry.
  const periodic_geometry::Cell* periodicCell() const noexcept
  { return periodic_cell_ ? &*periodic_cell_ : nullptr; }

  /// Account shared geometry once, separately from parameter-version allocations.
  std::size_t deviceStorageBytes() const { return checkedDeviceBytes<double>(nuclei_.capacity()); }

private:
  PsiFormerExecutionPlan plan_;
  int device_ = -1;
  DeviceBuffer<double> nuclei_;
  std::optional<periodic_geometry::Cell> periodic_cell_;
};

/** Own one immutable canonical parameter image within a shared metadata family.
 * Workspaces retain an image through submitted work; successors share layout and
 * fixed-ion storage. An admitted candidate may share its immutable payload with
 * its successor; ordinary HOST and copying successors remain independent.
 */
class PreparedDeviceModel
{
  friend class PreparedDeviceAdam;
  friend class PreparedDevicePcg;

  /// Revocable identity only: never retain moment buffers or an old parameter image.
  struct CandidateControl
  {
    std::weak_ptr<const PreparedDeviceModel> source;
    std::size_t generation;
    bool live = true;
  };

public:
  /** Own finite-admitted immutable candidate bytes, not model publication authority.
   * Same-thread optimizer mutation revokes admission, never this buffer's lifetime.
   */
  class CandidateSnapshot
  {
    friend class PreparedDeviceAdam;
    friend class PreparedDevicePcg;
    friend class PreparedDeviceModel;

  public:
    CandidateSnapshot(const CandidateSnapshot&) = delete;
    CandidateSnapshot& operator=(const CandidateSnapshot&) = delete;
    CandidateSnapshot(CandidateSnapshot&&) noexcept = default;

    /// Report retained payload bytes; a shared successor aliases, not duplicates, them.
    std::size_t deviceBytes() const noexcept { return parameters_ ? sizeof(double) * parameters_->capacity() : 0; }

    /// Identify the immutable source version without reading its HOST parameters.
    std::size_t sourceVersion() const noexcept { return source_version_; }

    /// Identify one optimizer proposal, not a reused scratch pointer.
    std::size_t proposalGeneration() const noexcept { return generation_; }

  private:
    /// Allocate private storage now; expose it only after finite admission.
    CandidateSnapshot(const std::shared_ptr<CandidateControl>& control,
        const std::shared_ptr<const PreparedDeviceModel>& source)
        : control_(control), metadata_(source->metadata_), source_version_(source->version_),
          generation_(control->generation),
          parameters_(std::make_shared<DeviceBuffer<double>>(source->plan().parameterCount(),
              "CandidateSnapshot::parameters", __FILE__, __LINE__)) {}

    std::weak_ptr<CandidateControl> control_;
    std::shared_ptr<const PreparedDeviceModelMetadata> metadata_;
    std::size_t source_version_, generation_;
    // Only the proposing optimizer writes before admission; exposed storage is immutable.
    std::shared_ptr<DeviceBuffer<double>> parameters_;
  };

  /// One independent candidate or successor owns exactly one canonical P vector.
  static std::size_t candidateDeviceBytes(std::size_t count) { return checkedDeviceBytes<double>(count); }

  /// Prepare one canonical numerical paired-VALUE image, without public GPU admission.
  static std::shared_ptr<const PreparedDeviceModel> preparePairedComplexValue(
      const PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
      const std::vector<double>& nuclei, std::size_t version = 0)
  {
    auto family = std::shared_ptr<PreparedDeviceModelMetadata>(new PreparedDeviceModelMetadata(plan, nuclei, true));
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(std::move(family), parameters, version));
  }

  /// Prepare a selected numerical profile without changing native builder admission.
  static std::shared_ptr<const PreparedDeviceModel> prepareRealTwoCellProfile(
      const PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
      const std::vector<double>& nuclei, std::size_t version = 0)
  {
    auto family = std::shared_ptr<PreparedDeviceModelMetadata>(
        new PreparedDeviceModelMetadata(plan, nuclei, false, true));
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(std::move(family), parameters, version));
  }

  /// Prepare paired two-cell numerical products without admitting native profiles.
  static std::shared_ptr<const PreparedDeviceModel> preparePairedTwoCellProfile(
      const PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
      const std::vector<double>& nuclei, std::size_t version = 0)
  {
    auto family = std::shared_ptr<PreparedDeviceModelMetadata>(
        new PreparedDeviceModelMetadata(plan, nuclei, true, true));
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(std::move(family), parameters, version));
  }

  /// Upload an initial checked open-real model and its fixed geometry metadata.
  PreparedDeviceModel(const PsiFormerExecutionPlan& plan, const std::vector<double>& parameters,
                      const std::vector<double>& nuclei, std::size_t version = 0)
      : PreparedDeviceModel(std::make_shared<PreparedDeviceModelMetadata>(plan, nuclei), parameters, version)
  {}

  PreparedDeviceModel(const PreparedDeviceModel&) = delete;
  PreparedDeviceModel& operator=(const PreparedDeviceModel&) = delete;
  PreparedDeviceModel(PreparedDeviceModel&&) = delete;
  PreparedDeviceModel& operator=(PreparedDeviceModel&&) = delete;

  /// Validate current device ownership without switching the caller's device.
  void validateCurrentDevice() const { metadata_->validateCurrentDevice(); }

  /// Keep typed parameter/layout metadata independent of one parameter allocation.
  const PsiFormerExecutionPlan& plan() const noexcept { return metadata_->plan(); }

  /// Identify immutable layout/geometry compatibility without comparing pointers to weights.
  const std::shared_ptr<const PreparedDeviceModelMetadata>& metadataFamily() const noexcept { return metadata_; }

  /// Admit exact coordinate bytes before allocating any state or device storage.
  /// The per-owner budget excludes shared metadata and every inference workspace.
  std::unique_ptr<PreparedDeviceCoordinates> prepareCoordinates(std::size_t capacity,
                                                               std::size_t workspace_budget_bytes) const
  {
    validateCurrentDevice();
    const auto ne = plan().modelShape().electrons();
    if (capacity == 0 || ne == 0 || ne > std::numeric_limits<std::size_t>::max() / 3 / capacity)
      throw std::invalid_argument("Device coordinate preparation extent is invalid or overflows");
    const auto bytes = checkedDeviceBytes<double>(3 * ne * capacity);
    if (bytes > workspace_budget_bytes)
      throw std::length_error("Device coordinate owner exceeds its workspace byte budget");
    return std::unique_ptr<PreparedDeviceCoordinates>(
        new PreparedDeviceCoordinates(metadata_, ne, capacity));
  }

  /// Prepare a newer image without changing this model or uploading geometry again.
  std::shared_ptr<const PreparedDeviceModel> makeUpdatedImage(const std::vector<double>& parameters,
                                                            std::size_t version) const
  {
    if (version <= version_)
      throw std::invalid_argument("Successor device model version must increase");
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(metadata_, parameters, version));
  }

  /// Admit one pending snapshot before independently copying a same-family successor.
  std::shared_ptr<const PreparedDeviceModel> makeUpdatedImage(const CandidateSnapshot& candidate,
                                                            std::size_t device_budget) const
  {
    validateCandidate(candidate);
    const auto bytes = candidateDeviceBytes(plan().parameterCount());
    if (!device_budget || device_budget == std::numeric_limits<std::size_t>::max() || bytes > device_budget)
      throw std::length_error("Device candidate successor exceeds its explicit byte budget");
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(metadata_, candidate));
  }

  /// Retain already admitted immutable bytes: zero additional DEVICE allocation/copy.
  /// Construction failure never steals or revokes the receipt's retryable owner.
  std::shared_ptr<const PreparedDeviceModel> makeSharedUpdatedImage(const CandidateSnapshot& candidate) const
  {
    validateCandidate(candidate);
    return std::shared_ptr<const PreparedDeviceModel>(new PreparedDeviceModel(metadata_, candidate, true));
  }

  /// Check a fixed-layout successor before any prepared graph changes ownership.
  void validateSuccessor(const std::shared_ptr<const PreparedDeviceModel>& successor) const
  {
    if (!successor || successor->metadata_.get() != metadata_.get() || successor->version() <= version_)
      throw std::invalid_argument("Device successor has incompatible metadata or a non-increasing version");
  }

  /// Identify the parameter image without inferring publication from a pointer.
  std::size_t version() const noexcept { return version_; }

  /// Return shared parameter storage; callers cannot mutate it through this API.
  const double* parameters() const noexcept { return parameterStorage().data(); }

  /// Cold explicit mirror boundary; the owning image keeps bytes stable through download.
  void downloadCanonical(std::vector<double>& values) const
  {
    validateCurrentDevice();
    if (values.size() != plan().parameterCount())
      throw std::invalid_argument("Canonical mirror download extent mismatch");
    parameterStorage().download(values.data(), values.size());
  }

  /// Return fixed-ion geometry stored once per shared image.
  const double* nuclei() const noexcept { return metadata_->nuclei(); }

  /// Expose immutable torus geometry to periodic numerical leaves; open models return null.
  const periodic_geometry::Cell* periodicCell() const noexcept { return metadata_->periodicCell(); }

  /// Supply the host scalar needed by the cusp launch, not a per-evaluation copy.
  double sameAlpha() const noexcept { return same_alpha_; }

  /// Supply the opposite-spin cusp scalar for this exact model version.
  double oppositeAlpha() const noexcept { return opposite_alpha_; }

  /// Logical image footprint, including shared payload/family. Do not sum aliasing
  /// image/receipt footprints; the aggregate allocation ledger counts physical bytes once.
  std::size_t deviceStorageBytes() const
  {
    return spatial_detail::checkedAdd(checkedDeviceBytes<double>(parameterStorage().capacity()),
        metadata_->deviceStorageBytes(), "Device model storage overflow");
  }

  std::uint64_t memoryTraceIdentity() const noexcept { return parameterStorage().memoryTraceIdentity(); }

private:
  /// Keep identical provenance and device checks for copying and sharing factories.
  void validateCandidate(const CandidateSnapshot& candidate) const
  {
    validateCurrentDevice();
    const auto control = candidate.control_.lock();
    if (!control || !control->live || control->generation != candidate.generation_ ||
        control->source.lock().get() != this || candidate.metadata_ != metadata_ ||
        candidate.source_version_ != version_ || !candidate.parameters_ ||
        candidate.parameters_->capacity() != plan().parameterCount())
      throw std::invalid_argument("Device candidate is foreign, stale, rejected or source-expired");
    if (version_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device candidate successor version exhausted");
  }

  /// Only one payload path is populated; shared storage is read-only after admission.
  const DeviceBuffer<double>& parameterStorage() const noexcept
  {
    return shared_parameters_ ? *shared_parameters_ : parameters_;
  }

  /// Copy only admitted DEVICE values; unchanged metadata never crosses HOST again.
  PreparedDeviceModel(std::shared_ptr<const PreparedDeviceModelMetadata> metadata, const CandidateSnapshot& candidate,
                      bool share_payload = false)
      : metadata_(std::move(metadata)), version_(candidate.source_version_ + 1),
        candidate_control_(candidate.control_), candidate_generation_(candidate.generation_)
  {
    if (share_payload)
      shared_parameters_ = candidate.parameters_;
    else
    {
      parameters_.reserveNamed(plan().parameterCount(), "Inference::parameters_", __FILE__, __LINE__);
      parameters_.copyFromDevice(*candidate.parameters_, plan().parameterCount());
    }
    if (const auto* same = plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA))
      parameterStorage().download(&same_alpha_, 1, same->begin);
    parameterStorage().download(&opposite_alpha_, 1, plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin);
    parameterStorage().traceContext(version_, reinterpret_cast<std::uintptr_t>(this));
  }

  /// Upload only canonical parameters after the immutable family is established.
  PreparedDeviceModel(std::shared_ptr<const PreparedDeviceModelMetadata> metadata,
                      const std::vector<double>& parameters, std::size_t version)
      : metadata_(std::move(metadata)), version_(version)
  {
    validateCurrentDevice();
    if (parameters.size() != plan().parameterCount())
      throw std::invalid_argument("Device model parameter extent mismatch");
    if (std::any_of(parameters.begin(), parameters.end(), [](double value) { return !device_math::isFiniteBinary64(value); }))
      throw std::invalid_argument("Device parameter image contains nonfinite values");
    parameters_.reserveNamed(parameters.size(), "Inference::parameters_", __FILE__, __LINE__);
    parameters_.upload(parameters.data(), parameters.size());
    parameters_.traceContext(version_, reinterpret_cast<std::uintptr_t>(this));
    if (const auto* same = plan().optionalParameter(ParameterRole::CUSP_SAME_ALPHA))
      same_alpha_ = parameters[same->begin];
    opposite_alpha_ = parameters[plan().parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin];
  }

  std::shared_ptr<const PreparedDeviceModelMetadata> metadata_;
  std::size_t version_;
  double same_alpha_ = 1, opposite_alpha_ = 1;
  DeviceBuffer<double> parameters_;
  std::shared_ptr<const DeviceBuffer<double>> shared_parameters_;
  std::weak_ptr<CandidateControl> candidate_control_;
  std::size_t candidate_generation_ = 0;
};

/** Preallocate one model image and a fixed-capacity configuration-major VGL graph.
 *
 * Feature/attention work stays on the GPU. Explicit packing bridges jet tensor
 * strides to the existing determinant ABI. Evaluation allocates no caller-owned
 * device buffers or per-stage host vectors; vendor BLAS internal workspace still
 * requires runtime profiling. Stream/BLAS ownership remains with the caller.
 */
class PreparedDeviceInference
{
  /// Forward-declare the private lifetime guard used by the real head completion.
  struct MixedValueAttempt;

public:
  /** Retain one block's value or full-spatial jets for an explicit reverse pass.
   * Feature geometry/embedding stay separate in the graph's raw/input storage;
   * this tape contains only the remaining attention/residual/MLP intermediates.
   */
  struct RetainedBlockTape
  {
    DeviceBuffer<double> input, query, key, value, probabilities, context;
    DeviceBuffer<double> residual, hidden, projected;
    DeviceBuffer<double> hidden_pre, projected_pre, logits;

    /// Reserve fixed-capacity value storage once, before any graph launch.
    RetainedBlockTape(std::size_t feature_count, std::size_t probability_count, bool spatial)
    {
      input.reserveNamed(feature_count, "RetainedBlockTape::input", __FILE__, __LINE__);
      query.reserveNamed(feature_count, "RetainedBlockTape::query", __FILE__, __LINE__);
      key.reserveNamed(feature_count, "RetainedBlockTape::key", __FILE__, __LINE__);
      value.reserveNamed(feature_count, "RetainedBlockTape::value", __FILE__, __LINE__);
      context.reserveNamed(feature_count, "RetainedBlockTape::context", __FILE__, __LINE__);
      residual.reserveNamed(feature_count, "RetainedBlockTape::residual", __FILE__, __LINE__);
      hidden.reserveNamed(feature_count, "RetainedBlockTape::hidden", __FILE__, __LINE__);
      projected.reserveNamed(feature_count, "RetainedBlockTape::projected", __FILE__, __LINE__);
      probabilities.reserveNamed(probability_count, "Inference::probabilities", __FILE__, __LINE__);
      if (spatial)
      {
        hidden_pre.reserveNamed(feature_count, "Inference::hidden_pre", __FILE__, __LINE__);
        projected_pre.reserveNamed(feature_count, "Inference::projected_pre", __FILE__, __LINE__);
        logits.reserveNamed(probability_count, "Inference::logits", __FILE__, __LINE__);
      }
    }

    /// Account explicit tape storage separately from the shared parameter image.
    std::size_t storageBytes() const
    {
      std::size_t bytes = 0;
      for (const auto* buffer : {&input, &query, &key, &value, &probabilities,
          &context, &residual, &hidden, &projected, &hidden_pre, &projected_pre, &logits})
        bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(buffer->capacity()),
                                           "Device value tape storage overflow");
      return bytes;
    }
  };

  /** Borrow the ordinary orbital factors and determinant seed ingredients.
   * A consumer must check factorization/combination status before using inverses;
   * singular-channel fallback is not implicitly supplied by this view.
   */
  struct ValueHeadTape
  {
    const double* positions;
    const double* backflow;
    const double* envelope;
    const double* matrices;
    const double* inverses;
    const double* normalized_weights;
    const device_determinant::FactorizationMetadata* factorizations;
    const device_determinant::CombinationMetadata* combinations;
  };

  /** Borrow full-VGL head/root storage with explicit configuration/plane layouts.
   * This is preparation for lifted reverse, not a complete kinetic-response API.
   */
  struct SpatialHeadTape
  {
    OpenOrbitalJetLayout orbital_layout;
    OpenCuspJetLayout cusp_layout;
    SpatialJetLayout root_layout;
    const double* positions;
    const double* raw_features;
    const double* features;
    const double* orbitals;
    const double* backflow;
    const double* envelope;
    const double* determinant_log_gradient;
    const double* determinant_log_laplacian;
    const double* cusp;
    const double* root;
    const double* inverses;
    const double* normalized_weights;
    const device_determinant::FactorizationMetadata* factorizations;
    const device_determinant::CombinationMetadata* combinations;
  };

  /// Validate the open-real model and prepare all storage before numerical launches.
  PreparedDeviceInference(const PsiFormerExecutionPlan& plan,
                          const std::vector<double>& parameters,
                          const std::vector<double>& nuclei,
                          std::size_t batch_size,
                          compute::Queue<PlatformKind::CUDA>& queue,
                          AcceleratorBlasHandle& blas,
                          SpatialJetMode mode = SpatialJetMode::FULL_VGL,
                          bool retain_score_tape = false,
                          bool retain_spatial_tape = false,
                          InferenceStatusPolicy status_policy = configuredInferenceStatusPolicy(),
                          TargetVglPreparation target_preparation = {})
      : PreparedDeviceInference(std::make_shared<PreparedDeviceModel>(plan, parameters, nuclei),
                                batch_size, queue, blas, mode, retain_score_tape, retain_spatial_tape,
                                status_policy, target_preparation)
  {}

  /// Prepare independent scratch while retaining an already uploaded model image.
  PreparedDeviceInference(std::shared_ptr<const PreparedDeviceModel> model,
                          std::size_t batch_size,
                          compute::Queue<PlatformKind::CUDA>& queue,
                          AcceleratorBlasHandle& blas,
                          SpatialJetMode mode = SpatialJetMode::FULL_VGL,
                          bool retain_score_tape = false,
                          bool retain_spatial_tape = false,
                          InferenceStatusPolicy status_policy = configuredInferenceStatusPolicy(),
                          TargetVglPreparation target_preparation = {})
      : PreparedDeviceInference(std::move(model), batch_size, queue, blas, mode,
          retain_score_tape, retain_spatial_tape, status_policy, target_preparation, false)
  {}

private:
  friend class PreparedDeviceTargetKinetic;
  friend class PreparedDeviceMetropolis;
  friend class PreparedDeviceKineticReverse;
  friend class PreparedDeviceComplexValue;
  friend class PreparedDeviceComplexSpatial;
  friend struct ComplexSpatialTargetValidationAccess;
  friend class PreparedDeviceComplexScore;
  friend class PreparedDeviceKineticTrunk;
  friend struct KineticTrunkValidationAccess;
  friend struct RetainedOpenFeatureValidationAccess;
  friend struct RetainedTraceValidationAccess;
  friend struct RetainedValueCuspValidationAccess;

  /// Only the target kinetic owner may prepare partial retained spatial tapes.
  PreparedDeviceInference(std::shared_ptr<const PreparedDeviceModel> model,
                          std::size_t batch_size,
                          compute::Queue<PlatformKind::CUDA>& queue,
                          AcceleratorBlasHandle& blas,
                          SpatialJetMode mode, bool retain_score_tape, bool retain_spatial_tape,
                          InferenceStatusPolicy status_policy,
                          TargetVglPreparation target_preparation, bool retained_targets, bool real_trunk_only = false,
                          bool kinetic_trunk_only = false)
      : model_(std::move(model)), plan_(checkedPlan(model_)), shape_(plan_.modelShape()),
        batch_(batch_size), batch_capacity_(batch_size), queue_(queue), blas_(blas), mode_(mode),
        target_preparation_(target_preparation), retained_targets_(retained_targets),
        cooperative_value_lu_requested_(configuredCooperativeValueLu()),
        cooperative_open_features_requested_(configuredCooperativeOpenFeatures()),
        retained_open_features_requested_(configuredRetainedOpenFeatures()),
        cooperative_value_cusp_requested_(configuredCooperativeValueCusp()),
        retained_value_cusp_requested_(configuredRetainedValueCusp()),
        tiled_value_orbitals_requested_(configuredTiledValueOrbitals()), kinetic_trunk_only_(kinetic_trunk_only)
  {
    model_->validateCurrentDevice();
    setStatusPolicy(status_policy);
    if (mode_ != SpatialJetMode::FULL_VGL && mode_ != SpatialJetMode::VALUE_ONLY)
      throw std::invalid_argument("Prepared device inference supports value-only or full VGL");
    if (retain_score_tape && mode_ != SpatialJetMode::VALUE_ONLY)
      throw std::invalid_argument("Ordinary score tape requires a value-only graph");
    if (retain_spatial_tape && mode_ != SpatialJetMode::FULL_VGL)
      throw std::invalid_argument("Spatial reverse tape requires a full-VGL graph");
    if (target_preparation_.parallel_trace_workspace_budget_bytes &&
        (real_trunk_only || retain_score_tape || (retain_spatial_tape && !retained_targets) ||
         mode_ != SpatialJetMode::FULL_VGL || !target_preparation_.max_targets))
      throw std::invalid_argument("Parallel traces reject trunks, score tapes and implicit targets");
    const auto& environment = plan_.environment();
    const bool head_free_targets = real_trunk_only && target_preparation_.max_targets;
    if (kinetic_trunk_only && (!real_trunk_only || batch_size != 1 || !retain_spatial_tape ||
        retain_score_tape || retained_targets || !target_preparation.max_targets ||
        target_preparation.max_targets > shape_.electrons()))
      throw std::invalid_argument("Private kinetic trunk requires one explicitly bounded retained spatial graph");
    if (real_trunk_only && !kinetic_trunk_only && (retain_spatial_tape ||
        retained_targets_ || environment.boundary != BoundaryCondition::PERIODIC ||
        environment.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
        environment.amplitude_scalar_domain != ScalarDomain::COMPLEX))
      throw std::invalid_argument("Private real trunk preparation requires the numerical paired profile");
    if (batch_ == 0 ||
        environment.parameter_scalar_domain != ScalarDomain::REAL ||
        environment.compute_scalar_domain != ScalarDomain::REAL ||
        (environment.amplitude_scalar_domain != ScalarDomain::REAL && !real_trunk_only) || !environment.fixed_nuclei)
      throw std::invalid_argument("Device inference supports fixed-ion real FP64 models only");
    ne_ = shape_.electrons();
    // Only the all-zero option is the legacy full graph. A malformed explicit
    // request must not silently bypass budgets and allocate full-Ne scratch.
    if (target_preparation_.max_targets == 0)
    {
      if (target_preparation_.device_workspace_budget_bytes != 0 || target_preparation_.scalar_work_budget != 0)
        throw std::invalid_argument("Target VGL budgets require an explicit positive target capacity");
    }
    else if (head_free_targets && !kinetic_trunk_only)
    {
      if (batch_size != 1 || mode_ != SpatialJetMode::FULL_VGL || retain_score_tape || retain_spatial_tape ||
          !target_preparation_.device_workspace_budget_bytes || !target_preparation_.scalar_work_budget)
        throw std::invalid_argument("Private forward target prefix requires B1/FULL, no tapes and positive caps");
      target_cost_ = estimateHeadFreeTargetPrefixCost(plan_, target_preparation_.max_targets);
      if (target_cost_.device_workspace_bytes > target_preparation_.device_workspace_budget_bytes ||
          target_cost_.charged_scalar_work_units > target_preparation_.scalar_work_budget)
        throw std::length_error("Private forward target prefix exceeds byte/work caps");
    }
    else if (!kinetic_trunk_only)
    {
      if (mode_ != SpatialJetMode::FULL_VGL || retain_score_tape ||
          (retain_spatial_tape && !retained_targets_))
        throw std::invalid_argument("Target VGL requires an FP64 non-retained full-VGL graph");
      if (target_preparation_.device_workspace_budget_bytes == 0 || target_preparation_.scalar_work_budget == 0)
        throw std::invalid_argument("Target VGL requires positive workspace and whole-request work budgets");
      if (retained_targets_ && (batch_capacity_ != 1 || !retain_spatial_tape))
        throw std::invalid_argument("Retained targets require one configuration and spatial tapes");
      target_cost_ = retained_targets_
          ? estimateRetainedTargetForwardCost(plan_, target_preparation_.max_targets,
                target_preparation_.parallel_trace_workspace_budget_bytes)
          : estimatePreparedTargetVglCost(plan_, batch_capacity_, target_preparation_);
      if (target_cost_.device_workspace_bytes > target_preparation_.device_workspace_budget_bytes ||
          target_cost_.charged_scalar_work_units > target_preparation_.scalar_work_budget)
        throw std::length_error("Target VGL preparation exceeds its explicit workspace or scalar-work budget");
    }
    const std::size_t width = shape_.feature_dimension;
    features_ = model_->periodicCell()
        ? makePeriodicFeatureJetLayout(batch_, ne_, shape_.nuclei, shape_.spin_up_electrons, mode_)
        : makeOpenFeatureJetLayout(batch_, ne_, shape_.nuclei, shape_.spin_up_electrons, mode_);
    const auto& embedding = plan_.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT);
    if (embedding.shape[0] != features_.feature_width)
      throw std::invalid_argument("Device inference feature contract mismatch");
    embedding_ = makeSpatialDenseJetLayout(batch_, ne_, features_.feature_width, width, ne_, mode_);
    dense_ = makeSpatialDenseJetLayout(batch_, ne_, width, width, ne_, mode_);
    elementwise_ = makeSpatialElementwiseJetLayout(batch_, ne_, width, ne_, mode_);
    attention_ = makeSpatialAttentionJetLayout(batch_, ne_, shape_.attention_heads,
                                               width / shape_.attention_heads, ne_, mode_);
    orbitals_ = makeOpenOrbitalJetLayout(batch_, shape_.determinants, ne_, shape_.nuclei,
                                         width, shape_.spin_up_electrons, mode_);
    cusp_ = makeOpenCuspJetLayout(batch_, ne_, shape_.spin_up_electrons, mode_);
    cusp_state_enabled_ = target_preparation_.reuse_cusp_state &&
        target_preparation_.max_targets && target_preparation_.max_targets < ne_ &&
        mode_ == SpatialJetMode::FULL_VGL && !retained_targets_ && !real_trunk_only;
    if (cusp_state_enabled_) cusp_state_layout_ = cusp_;
    final_ = makeFinalSpatialJetLayout(batch_, ne_, mode_);
    result_layout_ = final_.output;
    matrices_per_configuration_ = orbitals_.orbitals.element_count;
    gradient_lanes_ = dense_.source.gradient_lanes;
    if (target_preparation_.max_targets)
      setTargetRange(0, target_preparation_.max_targets);

    positions_.reserveNamed(product(batch_, product(ne_, 3)), "Inference::positions_", __FILE__, __LINE__);
    active_.reserveNamed(batch_, "Inference::active_", __FILE__, __LINE__);
    const std::vector<std::size_t> zero_active(batch_, 0);
    active_.upload(zero_active.data(), batch_);
    raw_.reserveNamed(spatialJetSpanElements(features_.output), "Inference::raw_", __FILE__, __LINE__);
    const auto feature_elements = spatialJetSpanElements(dense_.source);
    a_.reserveNamed(feature_elements, "Inference::a_", __FILE__, __LINE__);
    b_.reserveNamed(feature_elements, "Inference::b_", __FILE__, __LINE__);
    q_.reserveNamed(feature_elements, "Inference::q_", __FILE__, __LINE__);
    k_.reserveNamed(feature_elements, "Inference::k_", __FILE__, __LINE__);
    v_.reserveNamed(feature_elements, "Inference::v_", __FILE__, __LINE__);
    context_.reserveNamed(feature_elements, "Inference::context_", __FILE__, __LINE__);
    hidden_.reserveNamed(feature_elements, "Inference::hidden_", __FILE__, __LINE__);
    projected_.reserveNamed(feature_elements, "Inference::projected_", __FILE__, __LINE__);
    weights_.reserveNamed(spatialJetSpanElements(attention_.attention), "Inference::weights_", __FILE__, __LINE__);
    traceJet(raw_, features_.output, "Inference::raw_");
    for (auto* buffer : {&a_, &b_, &q_, &k_, &v_, &context_, &hidden_, &projected_})
      traceJet(*buffer, dense_.source, "feature_planes");
    traceJet(weights_, attention_.attention, "Inference::weights_");
    if (!real_trunk_only)
    {
      orbital_jets_.reserveNamed(spatialJetSpanElements(orbitals_.orbitals),
          "Inference::orbital_jets_", __FILE__, __LINE__);
      traceJet(orbital_jets_, orbitals_.orbitals, "Inference::orbital_jets_");
      matrices_.reserveNamed(product(batch_, matrices_per_configuration_), "Inference::matrices_", __FILE__, __LINE__);
      lu_.reserveNamed(matrices_.capacity(), "Inference::lu_", __FILE__, __LINE__);
      if (mode_ == SpatialJetMode::FULL_VGL || retain_score_tape)
        inverse_.reserveNamed(matrices_.capacity(), "Inference::inverse_", __FILE__, __LINE__);
      permutations_.reserveNamed(product(product(batch_, shape_.determinants), ne_),
          "Inference::permutations_", __FILE__, __LINE__);
      if (mode_ == SpatialJetMode::FULL_VGL || retain_score_tape)
        solve_.reserveNamed(permutations_.capacity(), "Inference::solve_", __FILE__, __LINE__);
      factors_.reserveNamed(product(batch_, shape_.determinants), "Inference::factors_", __FILE__, __LINE__);
      // Host admission copies complete POD records. Initialize their padding once
      // so those transfers never read unwritten bytes; numerical fields are still
      // set by the factorization kernel for every live determinant.
#ifdef QMC_CUDA2HIP
      checkRuntime(hipMemsetAsync(factors_.data(), 0,
          checkedDeviceBytes<device_determinant::FactorizationMetadata>(factors_.capacity()), queue_.getNative()),
          "factorization metadata storage initialization");
#else
      checkRuntime(cudaMemsetAsync(factors_.data(), 0,
          checkedDeviceBytes<device_determinant::FactorizationMetadata>(factors_.capacity()), queue_.getNative()),
          "factorization metadata storage initialization");
#endif
      combinations_.reserveNamed(batch_, "Inference::combinations_", __FILE__, __LINE__);
      term_phase_.reserveNamed(factors_.capacity(), "Inference::term_phase_", __FILE__, __LINE__);
      term_log_.reserveNamed(factors_.capacity(), "Inference::term_log_", __FILE__, __LINE__);
      scaled_.reserveNamed(factors_.capacity(), "Inference::scaled_", __FILE__, __LINE__);
      normalized_.reserveNamed(factors_.capacity(), "Inference::normalized_", __FILE__, __LINE__);
    }
    if (mode_ == SpatialJetMode::FULL_VGL && !real_trunk_only)
    {
      matrix_product_.reserveNamed(product(batch_, product(ne_, ne_)),
          "Inference::matrix_product_", __FILE__, __LINE__);
      det_gradient_.reserveNamed(product(batch_, gradient_lanes_), "Inference::det_gradient_", __FILE__, __LINE__);
      det_lap_ratio_.reserveNamed(product(batch_, final_.output.laplacian_lanes),
          "Inference::det_lap_ratio_", __FILE__, __LINE__);
      det_lap_log_.reserveNamed(product(batch_, final_.output.laplacian_lanes),
          "Inference::det_lap_log_", __FILE__, __LINE__);
      det_status_.reserveNamed(batch_, "Inference::det_status_", __FILE__, __LINE__);
    }
    cusp_jets_.reserveNamed(spatialJetSpanElements(cusp_.output), "Inference::cusp_jets_", __FILE__, __LINE__);
    if (cusp_state_enabled_)
    {
      cusp_state_.reserveNamed(spatialJetSpanElements(cusp_state_layout_.output),
          "Inference::cusp_state_", __FILE__, __LINE__);
      cusp_state_status_.reserveNamed(batch_capacity_, "Inference::cusp_state_status_", __FILE__, __LINE__);
    }
    if (!real_trunk_only)
    {
      phase_.reserveNamed(batch_,
          "Inference::phase_", __FILE__, __LINE__); output_.reserveNamed(spatialJetSpanElements(final_.output),
          "Inference::output_", __FILE__, __LINE__);
      lap_ratio_.reserveNamed(product(batch_, final_.output.laplacian_lanes),
          "Inference::lap_ratio_", __FILE__, __LINE__);
    }
    const auto status_count = std::max(product(batch_, product(ne_, width)), matrices_.capacity());
    open_status_.reserveNamed(status_count,
        "Inference::open_status_", __FILE__, __LINE__); host_status_.resize(status_count);
    jet_status_.reserveNamed(product(batch_, product(shape_.attention_heads, ne_)),
        "Inference::jet_status_", __FILE__, __LINE__);
    host_jet_status_.resize(jet_status_.capacity());
    sticky_status_.reserveNamed(1, "Inference::sticky_status_", __FILE__, __LINE__);
    if (target_preparation_.max_targets && !head_free_targets)
    {
      full_phase_.reserveNamed(batch_, "Inference::full_phase_", __FILE__, __LINE__);
      full_output_.reserveNamed(spatialJetSpanElements(result_layout_), "Inference::full_output_", __FILE__, __LINE__);
      full_lap_ratio_.reserveNamed(product(batch_, ne_), "Inference::full_lap_ratio_", __FILE__, __LINE__);
      tile_phase_.resize(batch_); tile_log_.resize(batch_);
      canonical_phase_.resize(batch_); canonical_log_.resize(batch_);
      host_combinations_.resize(batch_);
      if (target_preparation_.parallel_trace_workspace_budget_bytes)
        parallel_traces_ = std::make_unique<PreparedDeviceSpatialTraces>(
            device_determinant::SpatialLayout{batch_, shape_.determinants, ne_,
                3 * target_preparation_.max_targets, target_preparation_.max_targets}, queue_.getNative(),
            target_preparation_.parallel_trace_workspace_budget_bytes, retained_targets_);
      // Keep the pre-allocation accounting an executable invariant. Model image
      // and vendor storage are deliberately not charged to this owned workspace.
      if (!retained_targets_ && (deviceWorkspaceBytes() != target_cost_.device_workspace_bytes ||
          hostWorkspaceBytes() != target_cost_.host_workspace_bytes))
        throw std::logic_error("Target VGL prepared storage disagrees with pre-allocation accounting");
    }
    if (retain_score_tape || retain_spatial_tape)
    {
      if (!real_trunk_only)
      {
        retained_backflow_.reserveNamed(spatialJetSpanElements(orbitals_.orbitals),
            "Inference::retained_backflow_", __FILE__, __LINE__);
        retained_envelope_.reserveNamed(spatialJetSpanElements(orbitals_.orbitals),
            "Inference::retained_envelope_", __FILE__, __LINE__);
      }
      for (std::size_t layer = 0; layer < shape_.attention_blocks; ++layer)
        value_tape_.push_back(std::make_unique<RetainedBlockTape>(feature_elements, weights_.capacity(), retain_spatial_tape));
    }
    if (retained_targets_ && (deviceWorkspaceBytes() != target_cost_.device_workspace_bytes ||
        hostWorkspaceBytes() != target_cost_.host_workspace_bytes))
      throw std::logic_error("Retained target forward storage disagrees with pre-allocation accounting");
    if (head_free_targets && !kinetic_trunk_only && (deviceWorkspaceBytes() != target_cost_.device_workspace_bytes ||
        hostWorkspaceBytes() != target_cost_.host_workspace_bytes))
      throw std::logic_error("Head-free target prefix differs from exact preflight");
    if (cooperative_value_lu_requested_)
      std::clog << "PsiFormer VALUE LU preparation: selected="
                << (usesCooperativeValueLu() ? "cooperative" : "scalar")
                << " Ne=" << ne_ << " static_shared_control_bytes<=64; no inverse or extra global owner\n";
    if (cooperative_open_features_requested_)
      std::clog << "PsiFormer OPEN feature preparation: selected="
                << (usesCooperativeOpenFeatures() ? "cooperative" : "scalar")
                << " Ne=" << ne_ << " static_shared_control_bytes=4; no extra global owner\n";
    if (retained_open_features_requested_)
      std::clog << "PsiFormer retained OPEN feature preparation: selected="
                << ((!value_tape_.empty() || retained_targets_) && usesCooperativeOpenFeatures()
                        ? "cooperative" : "scalar")
                << " Ne=" << ne_ << " static_shared_control_bytes=4; no extra global owner\n";
    if (cooperative_value_cusp_requested_)
      std::clog << "PsiFormer VALUE cusp preparation: selected="
                << (usesCooperativeValueCusp() ? "cooperative" : "scalar")
                << " Ne=" << ne_ << " static_shared_payload_bytes=4100; no extra global owner\n";
    if (retained_value_cusp_requested_)
      std::clog << "PsiFormer retained VALUE cusp preparation: selected="
                << (!value_tape_.empty() && usesCooperativeValueCusp() ? "cooperative" : "scalar")
                << " Ne=" << ne_ << " static_shared_payload_bytes=4100; no extra global owner\n";
    if (memory::enabled())
    {
      const auto resource = reinterpret_cast<std::uintptr_t>(this);
      const auto image = model_->memoryTraceIdentity();
      for (const auto* buffer : {&positions_, &raw_, &a_, &b_, &q_, &k_, &v_, &weights_, &context_, &hidden_,
          &projected_, &orbital_jets_, &matrices_, &lu_, &inverse_, &solve_, &term_phase_, &term_log_, &scaled_,
          &normalized_, &matrix_product_, &det_gradient_, &det_lap_ratio_, &det_lap_log_, &cusp_jets_, &phase_,
          &output_, &lap_ratio_, &retained_backflow_, &retained_envelope_, &full_phase_, &full_output_, &full_lap_ratio_,
          &cusp_state_})
        buffer->traceContext(model_->version(), resource, image);
    }
  }

public:
  /// Submit feature layers, remaining network layers, orbital/determinant and cusp VGL.
  void evaluate(const std::vector<double>& positions)
  {
    beginCoordinateEvaluation();
    const auto coordinates_per_configuration = product(ne_, 3);
    if (positions.empty() || positions.size() > positions_.capacity() ||
        positions.size() % coordinates_per_configuration != 0)
      throw std::invalid_argument("Device inference positions exceed capacity or contain an incomplete configuration");
    setBatchCount(positions.size() / coordinates_per_configuration);
    positions_.uploadOnStream(positions.data(), positions.size(), queue_.getNative());
    evaluateOwnedCoordinates();
  }

  /// Consume a checked device population without a coordinate host upload.
  void evaluateDeviceCoordinates(const PackedDeviceCoordinatesView& positions)
  {
    beginCoordinateEvaluation();
    positions.copyTo(positions_, queue_, model_->metadataFamily(), ne_);
    setBatchCount(positions.batchCount());
    evaluateOwnedCoordinates();
  }

private:
  /// Revoke every previous output before validating or copying replacement input.
  void beginCoordinateEvaluation()
  {
    cusp_state_valid_ = false;
    if (parallel_traces_) parallel_traces_->invalidate();
    value_tape_valid_ = false;
    result_valid_ = false;
    target_tape_valid_ = false;
    admitted_target_tiles_ = 0;
    if (retained_targets_)
      throw std::logic_error("Retained target graphs require their private immediate reverse consumer");
    if (evaluation_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device inference evaluation generation exhausted");
    ++evaluation_generation_;
    model_->validateCurrentDevice();
  }

  /// Run the unchanged ordinary/target graph body over inference-owned coordinates.
  void evaluateOwnedCoordinates()
  {
    if (target_preparation_.max_targets)
    {
      for (std::size_t begin = 0; begin < ne_;)
      {
        const auto count = std::min(target_preparation_.max_targets, ne_ - begin);
        // Fresh compact logical prefixes reuse the max-K allocation. Tail planes
        // and configuration strides must shrink together for folded GEMM.
        setTargetRange(begin, count);
        evaluatePreparedTile(begin != 0);
        admitTileBase(begin == 0);
        checkRuntime(launchScatterSpatialTargets(queue_.getNative(), final_.output, result_layout_,
            phase_.data(), output_.data(), lap_ratio_.data(), full_phase_.data(), full_output_.data(),
            full_lap_ratio_.data()), "model target VGL scatter");
        ++admitted_target_tiles_;
        begin += count;
      }
      queue_.sync();
    }
    else
      evaluatePreparedTile();
    // No tile, base check or scatter failure may expose a partial physical root.
    value_tape_valid_ = !value_tape_.empty();
    result_valid_ = true;
  }

private:
  /** A borrow valid only during the current tile's immediate reverse callback. */
  struct SpatialTargetTapeView
  {
    const PreparedDeviceInference* owner;
    const PreparedDeviceModel* image;
    std::size_t generation;
    std::size_t ordinal;
    SpatialHeadTape head;
  };

  /// Reject saved views after another tile, request, failure or model rebind.
  void validateTargetTape(const SpatialTargetTapeView& view) const
  {
    if (!retained_targets_ || !target_tape_valid_ || view.owner != this || view.image != model_.get() ||
        view.generation != evaluation_generation_ || view.ordinal != admitted_target_tiles_ ||
        view.head.root_layout.target_electron_begin != final_.output.target_electron_begin ||
        view.head.root_layout.laplacian_lanes != final_.output.laplacian_lanes)
      throw std::logic_error("Retained target tape view is stale or belongs to another request");
  }

  /// Private head-free borrow, valid only in its immediate physical-target callback.
  struct KineticTrunkTicket
  {
    const PreparedDeviceInference* owner;
    const PreparedDeviceModel* image;
    std::size_t generation, ordinal;
    SpatialJetLayout features, cusp;
    const double *positions, *raw, *final_features, *cusp_jets;
  };

  /// Reject saved, foreign, mutated or successor-image trunk borrows before launch.
  void validateKineticTrunk(const KineticTrunkTicket& ticket) const
  {
    model_->validateCurrentDevice();
    if (!kinetic_trunk_only_ || !target_tape_valid_ || ticket.owner != this || ticket.image != model_.get() ||
        ticket.generation != evaluation_generation_ || ticket.ordinal != admitted_target_tiles_ ||
        ticket.features.target_electron_begin != dense_.source.target_electron_begin ||
        ticket.features.configuration_count != 1 || ticket.features.element_count != dense_.source.element_count ||
        ticket.features.electron_count != ne_ || ticket.features.mode != SpatialJetMode::FULL_VGL ||
        ticket.features.gradient_lanes != dense_.source.gradient_lanes ||
        ticket.features.laplacian_lanes != dense_.source.laplacian_lanes ||
        ticket.features.plane_count != dense_.source.plane_count ||
        ticket.features.plane_stride != dense_.source.plane_stride ||
        ticket.features.configuration_stride != dense_.source.configuration_stride ||
        ticket.cusp.target_electron_begin != cusp_.output.target_electron_begin ||
        ticket.cusp.configuration_count != cusp_.output.configuration_count ||
        ticket.cusp.element_count != cusp_.output.element_count || ticket.cusp.electron_count != ne_ ||
        ticket.cusp.mode != cusp_.output.mode || ticket.cusp.gradient_lanes != cusp_.output.gradient_lanes ||
        ticket.cusp.laplacian_lanes != cusp_.output.laplacian_lanes ||
        ticket.cusp.plane_stride != cusp_.output.plane_stride ||
        ticket.cusp.configuration_stride != cusp_.output.configuration_stride ||
        ticket.cusp.plane_count != cusp_.output.plane_count || ticket.positions != positions_.data() ||
        ticket.raw != raw_.data() || ticket.final_features != a_.data() || ticket.cusp_jets != cusp_jets_.data())
      throw std::logic_error("Kinetic trunk ticket is stale or foreign");
  }

  /// Upload once; publish no head, factors, normalized weights or full-Ne fallback.
  template<class Consumer>
  void evaluateKineticTrunk(const std::vector<double>& positions, Consumer&& consume)
  {
    beginCoordinateEvaluation();
    if (!kinetic_trunk_only_ || positions.size() != product(3, ne_))
      throw std::invalid_argument("Kinetic trunk requires one complete physical configuration");
    positions_.upload(positions.data(), positions.size());
    try
    {
      for (std::size_t begin = 0; begin < ne_;)
      {
        const auto count = std::min(target_preparation_.max_targets, ne_ - begin);
        setTargetRange(begin, count);
        resetStatusAdmission();
        evaluatePreparedTrunk();
        evaluateCusp();
        checkOpen(1);
        target_tape_valid_ = true;
        consume(KineticTrunkTicket{this, model_.get(), evaluation_generation_, admitted_target_tiles_,
            dense_.source, cusp_.output, positions_.data(), raw_.data(), a_.data(), cusp_jets_.data()});
        target_tape_valid_ = false;
        ++admitted_target_tiles_;
        begin += count;
      }
    }
    catch (...)
    {
      target_tape_valid_ = false;
      throw;
    }
  }

  /// Upload once and consume each retained tile before reusing its max-K storage.
  template<class Consumer>
  void evaluateRetainedTargets(const std::vector<double>& positions, Consumer&& consume)
  {
    value_tape_valid_ = result_valid_ = target_tape_valid_ = false;
    if (parallel_traces_) parallel_traces_->invalidate();
    admitted_target_tiles_ = 0;
    if (!retained_targets_ || positions.size() != product(3, ne_))
      throw std::invalid_argument("Retained target request requires one complete physical configuration");
    if (evaluation_generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Retained target evaluation generation exhausted");
    ++evaluation_generation_;
    model_->validateCurrentDevice();
    positions_.upload(positions.data(), positions.size());
    try
    {
      for (std::size_t begin = 0; begin < ne_;)
      {
        const auto count = std::min(target_preparation_.max_targets, ne_ - begin);
        setTargetRange(begin, count);
        evaluatePreparedTile();
        admitTileBase(begin == 0);
        const SpatialHeadTape head{orbitals_, cusp_, final_.output, positions_.data(), raw_.data(), a_.data(),
            orbital_jets_.data(), retained_backflow_.data(), retained_envelope_.data(),
            det_gradient_.data(), det_lap_log_.data(), cusp_jets_.data(), output_.data(), inverse_.data(),
            normalized_.data(), factors_.data(), combinations_.data()};
        target_tape_valid_ = true;
        consume(SpatialTargetTapeView{this, model_.get(), evaluation_generation_, admitted_target_tiles_, head});
        target_tape_valid_ = false;
        if (parallel_traces_) parallel_traces_->invalidate();
        ++admitted_target_tiles_;
        begin += count;
      }
    }
    catch (...)
    {
      target_tape_valid_ = false;
      if (parallel_traces_) parallel_traces_->invalidate();
      throw;
    }
  }

  /// Execute one admitted shape against the geometry uploaded by the logical request.
  void evaluatePreparedTile(bool reuse_target_base = false)
  {
    resetStatusAdmission();
    MixedValueAttempt mixed_attempt(mixed_dense_.get(), model_);
    evaluatePreparedTrunk();
    finishPreparedTile(mixed_attempt, reuse_target_base);
  }

  /// Shared real prefix in its original order; no head or determinant publication.
  void evaluatePreparedTrunk()
  {
    const auto stream = queue_.getNative();

    // Feature layers: physical radial/Cartesian jets followed by electron embedding.
    if (const auto* cell = model_->periodicCell())
      checkRuntime(launchPeriodicFeatureJets(stream, features_, *cell, positions_.data(), model_->nuclei(),
          active_.data(), raw_.data(), open_status_.data()), "model periodic feature VGL");
    else if (usesCooperativeOpenFeatures())
      checkRuntime(launchOpenFeatureJetsCooperative(stream, features_, positions_.data(), model_->nuclei(),
          active_.data(), raw_.data(), open_status_.data()), "model cooperative open feature VGL");
    else
      checkRuntime(launchOpenFeatureJets(stream, features_, positions_.data(), model_->nuclei(),
          active_.data(), raw_.data(), open_status_.data()), "model open feature VGL");
    checkOpen(batch_);
    project(embedding_, raw_.data(), parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT), a_.data());

    // Remaining neural layers: four validated attention/residual/MLP blocks.
    for (std::size_t layer = 0; layer < shape_.attention_blocks; ++layer)
    {
      if (!value_tape_.empty())
        copy(a_, 0, value_tape_[layer]->input, 0, spatialJetSpanElements(dense_.source));
      project(dense_, a_.data(), parameter(ParameterRole::ATTENTION_QUERY_WEIGHT, layer), q_.data());
      project(dense_, a_.data(), parameter(ParameterRole::ATTENTION_KEY_WEIGHT, layer), k_.data());
      project(dense_, a_.data(), parameter(ParameterRole::ATTENTION_VALUE_WEIGHT, layer), v_.data());
      PeriodicPairAttentionInputs pair_attention;
      if (model_->plan().environment().periodic_profile)
        pair_attention = {model_->periodicCell(), positions_.data(),
            parameter(ParameterRole::PAIR_ATTENTION_WEIGHT, layer), parameter(ParameterRole::PAIR_ATTENTION_BIAS, layer),
            active_.data(), shape_.spin_up_electrons};
      checkRuntime(launchAttentionJetWeights(stream, attention_, q_.data(), k_.data(),
          weights_.data(), jet_status_.data(),
          !value_tape_.empty() && mode_ == SpatialJetMode::FULL_VGL ? value_tape_[layer]->logits.data() : nullptr,
          pair_attention.cell ? &pair_attention : nullptr),
          "model attention VGL");
      checkJets();
      checkRuntime(launchAttentionContextJets(stream, attention_, weights_.data(),
                                              v_.data(), context_.data()), "model attention context");
      project(dense_, context_.data(), parameter(ParameterRole::ATTENTION_OUTPUT_WEIGHT, layer), b_.data());
      residual(a_.data(), b_.data(), b_.data());
      project(dense_, b_.data(), parameter(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer), hidden_.data());
      activate(hidden_.data(), parameter(ParameterRole::UPDATE_HIDDEN_BIAS, layer),
          !value_tape_.empty() && mode_ == SpatialJetMode::FULL_VGL ? value_tape_[layer]->hidden_pre.data() : nullptr);
      project(dense_, hidden_.data(), parameter(ParameterRole::UPDATE_OUTPUT_WEIGHT, layer), projected_.data());
      activate(projected_.data(), parameter(ParameterRole::UPDATE_OUTPUT_BIAS, layer),
          !value_tape_.empty() && mode_ == SpatialJetMode::FULL_VGL ? value_tape_[layer]->projected_pre.data() : nullptr);
      if (!value_tape_.empty())
      {
        // Capture before the next block overwrites shared scratch. Copies use the
        // same stream, so reverse consumers observe the exact forward values.
        auto& tape = *value_tape_[layer];
        const auto count = spatialJetSpanElements(dense_.source);
        copy(q_, 0, tape.query, 0, count);
        copy(k_, 0, tape.key, 0, count);
        copy(v_, 0, tape.value, 0, count);
        copy(weights_, 0, tape.probabilities, 0, spatialJetSpanElements(attention_.attention));
        copy(context_, 0, tape.context, 0, count);
        copy(b_, 0, tape.residual, 0, count);
        copy(hidden_, 0, tape.hidden, 0, count);
        copy(projected_, 0, tape.projected, 0, count);
      }
      residual(b_.data(), projected_.data(), a_.data());
    }

    // Every deferred neural result is initialized, but may be numerically invalid.
    // Admit the whole trunk before orbital construction or reverse-tape publication.
    admitStatus();
  }

  /// Complete the unchanged real head after the shared neural prefix is admitted.
  void finishPreparedTile(MixedValueAttempt& mixed_attempt, bool reuse_target_base = false)
  {
    const auto stream = queue_.getNative();
    // Spin-conditioned orbitals and learned exponential envelopes remain device-side.
    if (const auto* cell = model_->periodicCell())
      checkRuntime(launchPeriodicOrbitalJets(stream, orbitals_, *cell, a_.data(), positions_.data(), model_->nuclei(),
          parameter(ParameterRole::BACKFLOW_UP_WEIGHT), parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT),
          parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
          parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
          active_.data(), orbital_jets_.data(), open_status_.data(),
          value_tape_.empty() ? nullptr : retained_backflow_.data(),
          value_tape_.empty() ? nullptr : retained_envelope_.data()), "model periodic orbital VGL");
    else if (usesTiledValueOrbitals())
      checkRuntime(launchOpenOrbitalValuesTiled(stream, orbitals_, a_.data(), positions_.data(), model_->nuclei(),
        parameter(ParameterRole::BACKFLOW_UP_WEIGHT), parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT),
        parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
        active_.data(), orbital_jets_.data(), open_status_.data()), "model tiled OPEN orbital VALUE");
    else
      checkRuntime(launchOpenOrbitalJets(stream, orbitals_, a_.data(), positions_.data(), model_->nuclei(),
        parameter(ParameterRole::BACKFLOW_UP_WEIGHT), parameter(ParameterRole::BACKFLOW_DOWN_WEIGHT),
        parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
        active_.data(), orbital_jets_.data(), open_status_.data(),
        value_tape_.empty() ? nullptr : retained_backflow_.data(),
          value_tape_.empty() ? nullptr : retained_envelope_.data()), "model open orbital VGL");
    checkOpen(product(batch_, matrices_per_configuration_));

    // Only values need compact packing for factorization and the score-head ABI.
    // Spatial traces borrow the existing orbital planes with a configuration stride,
    // avoiding a second B*4Ne*D*Ne^2 derivative tensor and two copies per configuration.
    const char* reuse_flag = std::getenv("PSIFORMER_GPU_TARGET_FACTOR_REUSE");
    const bool request_local_factor_reuse = reuse_flag && reuse_flag[0] == '1' && reuse_flag[1] == '\0' &&
        target_preparation_.max_targets && mode_ == SpatialJetMode::FULL_VGL &&
        !retained_targets_ && value_tape_.empty() && !kinetic_trunk_only_ && !mixed_dense_ &&
        plan_.environment().amplitude_scalar_domain == ScalarDomain::REAL;
    if ((parallel_traces_ && !retained_targets_) || request_local_factor_reuse)
    {
      // Only later tiles of this request may reuse successful bitwise-equal bases.
      // Every public/direct tile call and every new coordinate attempt starts fresh.
      checkRuntime(launchDeterminantTargetFactorization(stream, orbital_jets_.data(), batch_,
          shape_.determinants, ne_, orbitals_.orbitals.configuration_stride, reuse_target_base,
          matrices_.data(), lu_.data(), inverse_.data(), permutations_.data(), solve_.data(),
          factors_.data()), "model target determinants");
    }
    else
    {
      for (std::size_t configuration = 0; configuration < batch_; ++configuration)
      {
        const auto origin = configuration * orbitals_.orbitals.configuration_stride;
        copy(orbital_jets_, origin, matrices_, configuration * matrices_per_configuration_, matrices_per_configuration_);
      }
      if (usesCooperativeValueLu())
        checkRuntime(launchDeterminantValueFactorizationCooperative(stream, matrices_.data(), batch_,
            shape_.determinants, ne_, lu_.data(), permutations_.data(), factors_.data()), "model cooperative VALUE determinants");
      else
        checkRuntime(launchDeterminantFactorization(stream, matrices_.data(), batch_, shape_.determinants, ne_,
            mode_ == SpatialJetMode::FULL_VGL || !value_tape_.empty(), lu_.data(), inverse_.data(),
            permutations_.data(), solve_.data(), factors_.data()), "model determinants");
    }
    checkRuntime(launchDeterminantCombination(stream, factors_.data(), nullptr, batch_, shape_.determinants,
        term_phase_.data(), term_log_.data(), scaled_.data(), normalized_.data(), combinations_.data()), "model determinant sum");
    // Value-only evaluation deliberately avoids inverse/traces and all spatial
    // output planes; nodal values remain defined even when log derivatives are not.
    if (mode_ == SpatialJetMode::VALUE_ONLY)
    {
      evaluateCusp();
      checkOpen(batch_);
      checkRuntime(launchFinalValueCombination(stream, batch_, combinations_.data(), cusp_jets_.data(),
          phase_.data(), output_.data(), open_status_.data()), "model final values");
      checkOpen(batch_);
      if (mixed_dense_)
        mixed_dense_->checkValueConditioning(factors_.data(), combinations_.data(), batch_, shape_.determinants);
      mixed_attempt.finish();
      return;
    }
    device_determinant::SpatialLayout determinant_layout{batch_, shape_.determinants, ne_, gradient_lanes_,
        final_.output.laplacian_lanes,
        orbitals_.orbitals.configuration_stride, orbitals_.orbitals.configuration_stride};
    // These layouts use packed D*Ne*Ne planes. The view starts at the first
    // requested derivative plane, and its stride skips the remaining jet planes.
    const auto* trace_status = det_status_.data();
    const auto* trace_gradient = det_gradient_.data();
    const auto* trace_lap_log = det_lap_log_.data();
    // Ordinary VGL borrows leaf outputs only on this queue. Retained reverse
    // consumers keep their original determinant-buffer ABI and lease authority.
    if (parallel_traces_)
    {
      const auto result = parallel_traces_->evaluate(determinant_layout,
          {factors_.data(), combinations_.data(), normalized_.data(), inverse_.data(),
           orbital_jets_.data() + orbitals_.orbitals.plane_stride,
           orbital_jets_.data() + (1 + gradient_lanes_) * orbitals_.orbitals.plane_stride});
      trace_status = result.status;
      trace_gradient = result.log_gradient;
      trace_lap_log = result.lap_log;
      if (retained_targets_)
      {
        const auto copy_result = [&](auto* destination, const auto* source, std::size_t count) {
          const auto bytes = checkedDeviceBytes<std::remove_pointer_t<decltype(destination)>>(count);
#ifdef QMC_CUDA2HIP
          checkRuntime(hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToDevice, stream),
              "retained trace D2D copy");
#else
          checkRuntime(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream),
              "retained trace D2D copy");
#endif
          deviceMemoryMetrics(det_gradient_.deviceOrdinal()).copied(bytes);
        };
        copy_result(det_gradient_.data(), result.log_gradient, product(batch_, gradient_lanes_));
        copy_result(det_lap_ratio_.data(), result.lap_ratio, product(batch_, final_.output.laplacian_lanes));
        copy_result(det_lap_log_.data(), result.lap_log, product(batch_, final_.output.laplacian_lanes));
        copy_result(det_status_.data(), result.status, batch_);
        trace_status = det_status_.data();
        trace_gradient = det_gradient_.data();
        trace_lap_log = det_lap_log_.data();
      }
    }
    else
      checkRuntime(launchDeterminantSpatialTraces(stream, determinant_layout, factors_.data(), combinations_.data(),
        normalized_.data(), inverse_.data(), orbital_jets_.data() + orbitals_.orbitals.plane_stride,
        orbital_jets_.data() + (1 + gradient_lanes_) * orbitals_.orbitals.plane_stride, matrix_product_.data(),
        det_gradient_.data(), det_lap_ratio_.data(), det_lap_log_.data(), det_status_.data()), "model determinant spatial traces");
    evaluateCusp();
    checkOpen(batch_);
    checkRuntime(launchFinalSpatialCombination(stream, final_, combinations_.data(), trace_status,
        trace_gradient, trace_lap_log, cusp_jets_.data(), phase_.data(), output_.data(),
        lap_ratio_.data(), open_status_.data()), "model final VGL");
    checkOpen(batch_);
    if (mixed_dense_)
      mixed_dense_->checkValueConditioning(factors_.data(), combinations_.data(), batch_, shape_.determinants);
    mixed_attempt.finish();
  }

public:
  /// Select diagnostics between completed evaluations; numerical semantics are unchanged.
  void setStatusPolicy(InferenceStatusPolicy policy)
  {
    if (policy != InferenceStatusPolicy::SYNCHRONOUS_DEBUG && policy != InferenceStatusPolicy::DEFERRED_NEURAL)
      throw std::invalid_argument("Unknown device inference status policy");
    status_policy_ = policy;
  }

  /// Report the graph's preparation-time or explicitly selected diagnostic policy.
  InferenceStatusPolicy statusPolicy() const noexcept { return status_policy_; }

  /** Evaluate once with explicit mixed arithmetic, replaying the same complete mode in FP64 on numerical rejection.
   * Invalid geometry, stale ownership and vendor-runtime errors are never hidden by this retry path. */
  bool evaluateWithFp64Retry(const std::vector<double>& positions, std::string& retry_reason)
  {
    retry_reason.clear();
    try
    {
      evaluate(positions);
      return false;
    }
    catch (const MixedPrecisionRetryRequired& error)
    {
      if (!mixed_dense_)
        throw;
      retry_reason = error.what();
    }
    catch (const InferenceStatusError& error)
    {
      if (!mixed_dense_ || error.stage() == 0)
        throw;
      retry_reason = error.what();
    }
    // Private pointer detachment avoids the public setter's result invalidation
    // after replay. The guard restores policy on both success and reference failure.
    auto adapter = std::move(mixed_dense_);
    try
    {
      evaluate(positions);
    }
    catch (...)
    {
      mixed_dense_ = std::move(adapter);
      throw;
    }
    mixed_dense_ = std::move(adapter);
    return true;
  }

  /** Opt into matched approximate affine value/VGL planes; reverse tapes remain rejected.
   * Production dispatch never attaches this experimental numerical adapter implicitly. */
  void setMixedDense(std::shared_ptr<PreparedDeviceMixedValueDense> adapter)
  {
    if (adapter && target_preparation_.max_targets)
      throw std::invalid_argument("Target VGL does not admit mixed affine arithmetic");
    if (adapter && (!value_tape_.empty() || !supportsMixedRealInferenceEnvironment(plan_.environment())))
      throw std::invalid_argument("Mixed affine inference requires a supported real geometry without reverse tapes");
    if (adapter && adapter->modelImage().get() != model_.get())
      throw std::invalid_argument("Mixed affine adapter has a stale or foreign parameter image");
    if (adapter && adapter->maximumRows() < product(product(batch_capacity_, ne_), dense_.source.plane_count))
      throw std::invalid_argument("Mixed affine adapter cannot hold every requested spatial plane");
    mixed_dense_ = std::move(adapter);
    if (mixed_dense_)
    {
      excludeCooperativeValueLuForMixedRequest();
      excludeCooperativeOpenFeaturesForMixedRequest();
    }
    value_tape_valid_ = false;
    result_valid_ = false;
  }

  /// Retain strict value-only attachment for callers of the original experimental API.
  void setMixedValueDense(std::shared_ptr<PreparedDeviceMixedValueDense> adapter)
  {
    if (adapter && mode_ != SpatialJetMode::VALUE_ONLY)
      throw std::invalid_argument("Value-only mixed adapter cannot attach to a VGL graph");
    setMixedDense(std::move(adapter));
  }

  /// Download signed-log amplitude, complete log-gradient/log-Laplacian jets and lap ratios.
  void downloadResults(std::vector<double>& phases, std::vector<double>& jets, std::vector<double>& ratios) const
  {
    if (!result_valid_)
      throw std::logic_error("Device inference has no completed result for the current image");
    if (phases.size() != batch_ || jets.size() != spatialJetSpanElements(result_layout_) ||
        ratios.size() != product(batch_, result_layout_.laplacian_lanes))
      throw std::invalid_argument("Device inference result buffers have wrong extent");
    (target_preparation_.max_targets ? full_phase_ : phase_).downloadOnStream(phases.data(), phases.size(), queue_.getNative());
    (target_preparation_.max_targets ? full_output_ : output_).downloadOnStream(jets.data(), jets.size(), queue_.getNative());
    (target_preparation_.max_targets ? full_lap_ratio_ : lap_ratio_).downloadOnStream(ratios.data(), ratios.size(), queue_.getNative());
  }

  /// Expose the fixed output layout for result comparison, not internal scratch.
  const SpatialJetLayout& resultLayout() const noexcept { return result_layout_; }

  /// Expose actual dispatch, not merely the requested flag; mixed/retained graphs coexist.
  bool usesCooperativeValueLu() const noexcept
  {
    const auto& environment = plan_.environment();
    return cooperative_value_lu_requested_ && ne_ >= 64 && mode_ == SpatialJetMode::VALUE_ONLY &&
        value_tape_.empty() && !mixed_value_lu_excluded_ && !mixed_dense_ && !kinetic_trunk_only_ &&
        environment.boundary == BoundaryCondition::OPEN &&
        environment.amplitude_scalar_domain == ScalarDomain::REAL &&
        environment.parameter_scalar_domain == ScalarDomain::REAL &&
        environment.compute_scalar_domain == ScalarDomain::REAL;
  }

  /// Permanently preserve scalar VALUE LU for both sides of a mixed request/replay.
  void excludeCooperativeValueLuForMixedRequest()
  {
    if (mixed_value_lu_excluded_)
      return;
    mixed_value_lu_excluded_ = true;
    if (cooperative_value_lu_requested_)
      std::clog << "PsiFormer VALUE LU preparation: selected=scalar (mixed request/replay excluded)\n";
  }

  /// Admit the existing pair map with independent ordinary and retained selectors.
  bool usesCooperativeOpenFeatures() const noexcept
  {
    const auto& environment = plan_.environment();
    const bool retained = !value_tape_.empty() || retained_targets_;
    const bool requested = retained ? retained_open_features_requested_ : cooperative_open_features_requested_;
    return requested && ne_ >= 64 &&
        (mode_ == SpatialJetMode::FULL_VGL || mode_ == SpatialJetMode::VALUE_ONLY) &&
        !kinetic_trunk_only_ &&
        !mixed_open_features_excluded_ && !mixed_dense_ && environment.boundary == BoundaryCondition::OPEN &&
        environment.amplitude_scalar_domain == ScalarDomain::REAL &&
        environment.parameter_scalar_domain == ScalarDomain::REAL &&
        environment.compute_scalar_domain == ScalarDomain::REAL;
  }

  /// Reuse ordered VALUE sums with independent ordinary and retained admission.
  bool usesCooperativeValueCusp() const noexcept
  {
    const auto& environment = plan_.environment();
    const bool requested = value_tape_.empty() ? cooperative_value_cusp_requested_ : retained_value_cusp_requested_;
    return requested && ne_ >= 64 && mode_ == SpatialJetMode::VALUE_ONLY &&
        !retained_targets_ && !kinetic_trunk_only_ &&
        !mixed_open_features_excluded_ && !mixed_dense_ && environment.boundary == BoundaryCondition::OPEN &&
        environment.amplitude_scalar_domain == ScalarDomain::REAL &&
        environment.parameter_scalar_domain == ScalarDomain::REAL &&
        environment.compute_scalar_domain == ScalarDomain::REAL;
  }

  /// Tile only ordinary real OPEN VALUE heads, never retained or mixed paths.
  bool usesTiledValueOrbitals() const noexcept
  {
    const auto& environment = plan_.environment();
    return tiled_value_orbitals_requested_ && ne_ >= 64 && mode_ == SpatialJetMode::VALUE_ONLY &&
        value_tape_.empty() && !retained_targets_ && !kinetic_trunk_only_ &&
        !mixed_open_features_excluded_ && !mixed_dense_ && environment.boundary == BoundaryCondition::OPEN &&
        environment.amplitude_scalar_domain == ScalarDomain::REAL &&
        environment.parameter_scalar_domain == ScalarDomain::REAL &&
        environment.compute_scalar_domain == ScalarDomain::REAL;
  }

  /// Permanently retain the old map for mixed reference, approximate and retry graphs.
  void excludeCooperativeOpenFeaturesForMixedRequest()
  {
    if (mixed_open_features_excluded_)
      return;
    mixed_open_features_excluded_ = true;
    if (cooperative_open_features_requested_ || retained_open_features_requested_)
      std::clog << "PsiFormer OPEN feature preparation: selected=scalar (mixed request/replay excluded)\n";
  }

  /// Report immutable explicit target preparation and its admitted capacity costs.
  const TargetVglPreparation& targetVglPreparation() const noexcept { return target_preparation_; }
  const TargetVglCost& targetVglCost() const noexcept { return target_cost_; }

  /// Diagnose partial progress without making its numerical buffers public.
  std::size_t admittedTargetTiles() const noexcept { return admitted_target_tiles_; }

  /// Expose identity/version ownership for publication binding and reuse tests.
  const std::shared_ptr<const PreparedDeviceModel>& modelImage() const noexcept { return model_; }

  /** Rebind a newer immutable image without reallocating fixed-shape scratch.
   * Caller must hold the owning publication/resource lease and have completed
   * earlier work. Shared metadata identity preserves plan and geometry lifetime;
   * this numerical primitive does not itself publish a control-plane version.
   */
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    validateModelRebind(successor);
    auto retired_image = model_; // Keep ordinary callers within the adoption lifetime contract too.
    adoptPreparedModel(successor);
  }

  /// Validate without revoking active tapes/results; publication retains owner authority.
  void validateModelRebind(const std::shared_ptr<const PreparedDeviceModel>& successor) const
  {
    model_->validateSuccessor(successor);
    if (mixed_dense_ && mixed_dense_->modelImage().get() != successor.get())
      throw std::invalid_argument("Rebind the prepared mixed affine cache before replacing its graph image");
  }

private:
  /** Adopt only after validation, with exclusive owner authority and an external
   * strong old-image owner. No runtime call, allocation or resource retirement.
   */
  void adoptPreparedModel(const std::shared_ptr<const PreparedDeviceModel>& successor) noexcept
  {
    cusp_state_valid_ = false;
    model_ = successor;
    value_tape_valid_ = false;
    result_valid_ = false;
    target_tape_valid_ = false;
    if (parallel_traces_) parallel_traces_->invalidate();
  }

public:
  /// Return prepared capacity independently of the most recent logical batch.
  std::size_t batchCapacity() const noexcept { return batch_capacity_; }

  /// Identify attempted coordinate evaluations independently of immutable parameters.
  std::size_t evaluationGeneration() const noexcept { return evaluation_generation_; }

  /// Expose a completed tape only; callers retain the graph and its model lease.
  const RetainedBlockTape& valueBlockTape(std::size_t layer) const
  {
    if (!value_tape_valid_ || mode_ != SpatialJetMode::VALUE_ONLY)
      throw std::logic_error("Device value tape is not available for a completed evaluation");
    return *value_tape_.at(layer);
  }

  /// Expose full-spatial block planes without allowing ordinary score interpretation.
  const RetainedBlockTape& spatialBlockTape(std::size_t layer) const
  {
    if (retained_targets_ || !value_tape_valid_ || mode_ != SpatialJetMode::FULL_VGL)
      throw std::logic_error("Device spatial tape is not available for a completed full-VGL evaluation");
    return *value_tape_.at(layer);
  }

  /// Borrow completed full-spatial head/root data while retaining the model lease.
  SpatialHeadTape spatialHeadTape() const
  {
    (void)spatialBlockTape(0);
    return {orbitals_, cusp_, final_.output, positions_.data(), raw_.data(), a_.data(),
            orbital_jets_.data(), retained_backflow_.data(), retained_envelope_.data(),
            det_gradient_.data(), det_lap_log_.data(), cusp_jets_.data(), output_.data(), inverse_.data(),
            normalized_.data(), factors_.data(), combinations_.data()};
  }

  /// Copy live full-spatial determinant admission records into prepared host storage.
  void downloadSpatialFactorizations(std::vector<device_determinant::FactorizationMetadata>& destination) const
  {
    (void)spatialBlockTape(0);
    const auto count = product(batch_, shape_.determinants);
    if (destination.size() < count)
      throw std::invalid_argument("Spatial factorization metadata destination is too small");
    factors_.download(destination.data(), count);
  }

  /// Download retained full-spatial orbital product factors for numerical validation.
  void downloadSpatialOrbitalTape(std::vector<double>& backflow, std::vector<double>& envelope,
                                  std::vector<double>& orbitals) const
  {
    (void)spatialBlockTape(0);
    const auto count = spatialJetSpanElements(orbitals_.orbitals);
    if (backflow.size() != count || envelope.size() != count || orbitals.size() != count)
      throw std::invalid_argument("Spatial orbital tape destination extent mismatch");
    retained_backflow_.download(backflow.data(), count);
    retained_envelope_.download(envelope.data(), count);
    orbital_jets_.download(orbitals.data(), count);
  }

  /// Report the live configuration prefix belonging to the completed tape.
  std::size_t valueTapeBatchCount() const
  {
    (void)valueBlockTape(0);
    return batch_;
  }

  /// Borrow geometry-derived feature values from the completed value-only tape.
  const double* valueRawFeatures() const
  {
    (void)valueTapeBatchCount();
    return raw_.data();
  }

  /// Borrow completed orbital/determinant inputs while retaining the graph lease.
  ValueHeadTape valueHeadTape() const
  {
    (void)valueTapeBatchCount();
    return {positions_.data(), retained_backflow_.data(), retained_envelope_.data(),
            matrices_.data(), inverse_.data(), normalized_.data(), factors_.data(), combinations_.data()};
  }

  /// Download the common cusp without subtracting possibly nodal log amplitudes.
  void downloadValueCusp(std::vector<double>& destination) const
  {
    const auto count = valueTapeBatchCount();
    if (destination.size() != count)
      throw std::invalid_argument("Value cusp destination extent mismatch");
    cusp_jets_.download(destination.data(), count);
  }

  /// Copy live determinant metadata into preallocated caller storage for admission.
  void downloadValueFactorizations(std::vector<device_determinant::FactorizationMetadata>& destination) const
  {
    const auto count = product(valueTapeBatchCount(), shape_.determinants);
    if (destination.size() < count)
      throw std::invalid_argument("Factorization metadata destination is too small");
    factors_.download(destination.data(), count);
  }

  /// Download the separately retained product factors for regression validation.
  void downloadValueOrbitalFactors(std::vector<double>& backflow, std::vector<double>& envelope) const
  {
    const auto count = product(valueTapeBatchCount(), matrices_per_configuration_);
    if (backflow.size() != count || envelope.size() != count)
      throw std::invalid_argument("Retained orbital factor destination extent mismatch");
    retained_backflow_.download(backflow.data(), count);
    retained_envelope_.download(envelope.data(), count);
  }

  /// Borrow the final shared neural features before orbital/envelope construction.
  const double* valueFinalFeatures() const
  {
    (void)valueTapeBatchCount();
    return a_.data();
  }

  /// Download a completed final-feature tile into an exactly sized host destination.
  void downloadValueFinalFeatures(std::vector<double>& destination) const
  {
    const auto count = spatial_detail::checkedProduct(valueTapeBatchCount(),
        spatial_detail::checkedProduct(ne_, shape_.feature_dimension, "Value feature extent overflow"),
        "Value feature tile overflow");
    if (destination.size() != count)
      throw std::invalid_argument("Value feature destination extent mismatch");
    a_.download(destination.data(), count);
  }

  /// Account every explicitly owned device allocation, excluding shared images/BLAS.
  /// Count retained host status arrays, separately from vendor allocations.
  std::size_t hostWorkspaceBytes() const
  {
    return checkedDeviceBytes<OpenSpatialStatus>(host_status_.capacity()) +
        checkedDeviceBytes<device_math::JetMathStatus>(host_jet_status_.capacity()) +
        checkedDeviceBytes<double>(tile_phase_.capacity() + tile_log_.capacity() +
            canonical_phase_.capacity() + canonical_log_.capacity()) +
        checkedDeviceBytes<device_determinant::CombinationMetadata>(host_combinations_.capacity());
  }

  /// Count graph-owned device buffers, excluding the shared immutable image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = 0;
    auto add = [&](std::size_t amount) {
      bytes = spatial_detail::checkedAdd(bytes, amount, "Device workspace storage overflow");
    };
    for (const auto* buffer : {&positions_, &raw_, &a_, &b_, &q_, &k_, &v_, &weights_, &context_,
        &hidden_, &projected_, &orbital_jets_, &matrices_, &lu_, &inverse_,
        &solve_, &term_phase_, &term_log_, &scaled_, &normalized_, &matrix_product_, &det_gradient_,
        &det_lap_ratio_, &det_lap_log_, &cusp_jets_, &phase_, &output_, &lap_ratio_,
        &retained_backflow_, &retained_envelope_, &full_phase_, &full_output_, &full_lap_ratio_, &cusp_state_})
      add(checkedDeviceBytes<double>(buffer->capacity()));
    add(checkedDeviceBytes<std::size_t>(active_.capacity()));
    add(checkedDeviceBytes<std::size_t>(permutations_.capacity()));
    add(checkedDeviceBytes<device_determinant::FactorizationMetadata>(factors_.capacity()));
    add(checkedDeviceBytes<device_determinant::CombinationMetadata>(combinations_.capacity()));
    add(checkedDeviceBytes<device_determinant::DerivativeStatus>(det_status_.capacity()));
    add(checkedDeviceBytes<OpenSpatialStatus>(open_status_.capacity()));
    add(checkedDeviceBytes<OpenSpatialStatus>(cusp_state_status_.capacity()));
    add(checkedDeviceBytes<InferenceStatusWord>(sticky_status_.capacity()));
    add(checkedDeviceBytes<device_math::JetMathStatus>(jet_status_.capacity()));
    if (parallel_traces_) add(parallel_traces_->deviceWorkspaceBytes());
    for (const auto& tape : value_tape_)
      add(tape->storageBytes());
    return bytes;
  }

private:
  /** Cancel adapter state on every graph exit, including an early physical failure. */
  struct MixedValueAttempt
  {
    PreparedDeviceMixedValueDense* adapter;

    /// Open the attempt only after graph geometry/capacity admission.
    MixedValueAttempt(PreparedDeviceMixedValueDense* selected,
                      const std::shared_ptr<const PreparedDeviceModel>& image) : adapter(selected)
    {
      if (adapter)
        adapter->beginEvaluation(image);
    }

    /// Failed partial arithmetic cannot keep an adapter transaction open.
    ~MixedValueAttempt()
    {
      if (adapter)
        adapter->cancelEvaluation();
    }

    /// Admit numerical diagnostics before the owning graph publishes its value.
    void finish()
    {
      if (adapter)
        adapter->finishEvaluation();
    }
  };

  /// Restrict all launches to a compact live prefix without resizing any buffers.
  void setBatchCount(std::size_t count)
  {
    batch_ = count;
    for (auto* layout : {&features_.output, &embedding_.source, &embedding_.target,
        &dense_.source, &dense_.target, &elementwise_.jets, &attention_.features,
        &attention_.attention, &orbitals_.features, &orbitals_.orbitals,
        &cusp_.output, &final_.output})
      layout->configuration_count = count;
    result_layout_.configuration_count = count;
    if (cusp_state_enabled_) cusp_state_layout_.output.configuration_count = count;
  }

  /// Select physical targets while all intermediate tensors use compact local planes.
  void setTargetRange(std::size_t begin, std::size_t count)
  {
    for (auto* layout : {&features_.output, &embedding_.source, &embedding_.target,
        &dense_.source, &dense_.target, &elementwise_.jets, &attention_.features,
        &attention_.attention, &orbitals_.features, &orbitals_.orbitals,
        &cusp_.output, &final_.output})
      *layout = withSpatialTargetRange(*layout, begin, count);
    gradient_lanes_ = dense_.source.gradient_lanes;
    final_.determinant_gradient_configuration_stride = gradient_lanes_;
    final_.determinant_laplacian_configuration_stride = count;
    final_.laplacian_ratio_configuration_stride = count;
  }

  /// Admit every tile's base before scattering or resetting its sticky status.
  void admitTileBase(bool first)
  {
    phase_.downloadOnStream(tile_phase_.data(), batch_, queue_.getNative());
    combinations_.downloadOnStream(host_combinations_.data(), batch_, queue_.getNative());
    for (std::size_t configuration = 0; configuration < batch_; ++configuration)
    {
      output_.downloadOnStream(&tile_log_[configuration], 1, queue_.getNative(), final_.output.uncheckedValueOffset(configuration, 0));
      if (first)
      {
        canonical_phase_[configuration] = tile_phase_[configuration];
        canonical_log_[configuration] = tile_log_[configuration];
      }
      admitTargetVglBase(tile_phase_[configuration], tile_log_[configuration],
          canonical_phase_[configuration], canonical_log_[configuration], host_combinations_[configuration]);
    }
  }

  /// Reject a missing shared image before any dependent layout is constructed.
  static const PsiFormerExecutionPlan& checkedPlan(const std::shared_ptr<const PreparedDeviceModel>& model)
  {
    if (!model)
      throw std::invalid_argument("Prepared device inference requires a model image");
    return model->plan();
  }

  /// Keep multiplication overflow checks at preparation boundaries.
  static std::size_t product(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Device inference extent overflow"); }

  /// Bind a canonical device parameter interval without hot-path string lookup.
  const double* parameter(ParameterRole role, std::size_t layer = NO_ATTENTION_BLOCK) const
  { return model_->parameters() + plan_.parameter(role, layer).begin; }

  /** Fold contiguous configurations and derivative planes into one shared-weight GEMM.
   * The same affine weight acts independently on every jet plane. No mathematical
   * coupling crosses configurations; only the BLAS row dimension is enlarged.
   */
  /// Evaluate the same physical cusp geometry used by feature and orbital layers.
  void evaluateCusp()
  {
    if (cusp_state_enabled_)
    {
      // Queue ordering protects the immutable per-request cache; no cross-state reuse.
      if (!cusp_state_valid_ || cusp_state_generation_ != evaluation_generation_ ||
          cusp_state_model_ != model_.get() || cusp_state_batch_ != batch_)
      {
        if (const auto* cell = model_->periodicCell())
          checkRuntime(launchPeriodicCuspJets(queue_.getNative(), cusp_state_layout_, *cell,
              positions_.data(), active_.data(), model_->sameAlpha(), model_->oppositeAlpha(),
              cusp_state_.data(), cusp_state_status_.data()), "model periodic cusp state");
        else
          checkRuntime(launchOpenCuspJets(queue_.getNative(), cusp_state_layout_, positions_.data(),
              active_.data(), model_->sameAlpha(), model_->oppositeAlpha(),
              cusp_state_.data(), cusp_state_status_.data()), "model open cusp state");
        cusp_state_generation_ = evaluation_generation_;
        cusp_state_model_ = model_.get();
        cusp_state_batch_ = batch_;
        cusp_state_valid_ = true;
      }
      checkRuntime(launchGatherCuspTargets(queue_.getNative(), cusp_state_layout_, cusp_,
          positions_.data(), active_.data(), model_->sameAlpha(), model_->oppositeAlpha(),
          cusp_state_.data(), cusp_state_status_.data(), cusp_jets_.data(), open_status_.data(),
          model_->periodicCell()), "model cusp state targets");
      return;
    }
    if (const auto* cell = model_->periodicCell())
      checkRuntime(launchPeriodicCuspJets(queue_.getNative(), cusp_, *cell, positions_.data(), active_.data(),
          model_->sameAlpha(), model_->oppositeAlpha(), cusp_jets_.data(), open_status_.data()),
          "model periodic cusp");
    else if (usesCooperativeValueCusp())
      checkRuntime(launchOpenCuspValuesCooperative(queue_.getNative(), cusp_, positions_.data(), active_.data(),
          model_->sameAlpha(), model_->oppositeAlpha(), cusp_jets_.data(), open_status_.data()),
          "model cooperative open VALUE cusp");
    else
      checkRuntime(launchOpenCuspJets(queue_.getNative(), cusp_, positions_.data(), active_.data(),
          model_->sameAlpha(), model_->oppositeAlpha(), cusp_jets_.data(), open_status_.data()),
          "model open cusp");
  }

  /// Apply a prepared affine projection under the explicitly selected compute policy.
  void project(const SpatialDenseJetLayout& layout, const double* source,
               const double* weight, double* target)
  {
    if (mixed_dense_)
      mixed_dense_->projectSpatial(layout, source, weight, target);
    else
      denseForward(blas_, makeFoldedSpatialDenseLayout(layout), source, weight, target);
  }

  /// Reset ordered sticky diagnostics for this attempt before any numerical launch.
  void resetStatusAdmission()
  {
    status_stage_ = 0;
    status_generation_ = evaluation_generation_;
    status_pending_ = false;
    if (status_policy_ == InferenceStatusPolicy::DEFERRED_NEURAL)
    {
#ifdef QMC_CUDA2HIP
      checkRuntime(hipMemsetAsync(sticky_status_.data(), 0xff, sizeof(InferenceStatusWord), queue_.getNative()),
                   "model sticky status initialization");
#else
      checkRuntime(cudaMemsetAsync(sticky_status_.data(), 0xff, sizeof(InferenceStatusWord), queue_.getNative()),
                   "model sticky status initialization");
#endif
    }
  }

  /// Give every status-producing stage a representable monotonically ordered identity.
  std::size_t nextStatusStage()
  {
    if (status_stage_ > INFERENCE_STATUS_MAX_STAGE)
      throw std::length_error("Device inference status stage count exceeds compact diagnostics");
    return status_stage_++;
  }

  /** Publish no numerical result until the current sticky status is admitted.
   * Physical boundaries call this immediately; only defined in-place neural outputs defer it. */
  void admitStatus()
  {
    if (status_policy_ != InferenceStatusPolicy::DEFERRED_NEURAL || !status_pending_)
      return;
    if (status_generation_ != evaluation_generation_)
      throw std::logic_error("Device inference status belongs to a stale evaluation");
    queue_.sync();
    InferenceStatusWord word;
    sticky_status_.downloadOnStream(&word, 1, queue_.getNative());
    if (word != std::numeric_limits<InferenceStatusWord>::max())
      throw InferenceStatusError(word, evaluation_generation_);
    status_pending_ = false;
  }

  /// Preserve leaf diagnostics before a later launch overwrites its temporary status array.
  void checkOpen(std::size_t count, bool defer_neural = false)
  {
    const auto stage = nextStatusStage();
    if (count == 0 || count - 1 > INFERENCE_STATUS_MAX_INDEX)
      throw std::length_error("Device inference spatial status count is not representable");
    if (status_policy_ == InferenceStatusPolicy::DEFERRED_NEURAL)
    {
      checkRuntime(launchAggregateOpenInferenceStatus(queue_.getNative(), open_status_.data(), count,
          stage, sticky_status_.data()), "model spatial status aggregation");
      status_pending_ = true;
      if (!defer_neural)
        admitStatus();
      return;
    }
    queue_.sync();
    open_status_.downloadOnStream(host_status_.data(), count, queue_.getNative());
    for (std::size_t i = 0; i < count; ++i)
      if (host_status_[i] != OpenSpatialStatus::REGULAR)
        throw InferenceStatusError((InferenceStatusWord(stage) << 48) |
            (InferenceStatusWord(i) << 8) | static_cast<unsigned int>(host_status_[i]), evaluation_generation_);
  }

  /// Softmax mutates fully initialized logits in place, so failures may defer admission.
  void checkJets()
  {
    const auto stage = nextStatusStage();
    const std::size_t count = product(batch_, product(shape_.attention_heads, ne_));
    if (status_policy_ == InferenceStatusPolicy::DEFERRED_NEURAL)
    {
      checkRuntime(launchAggregateJetInferenceStatus(queue_.getNative(), jet_status_.data(), count,
          stage, sticky_status_.data()), "model softmax status aggregation");
      status_pending_ = true;
      return;
    }
    queue_.sync();
    jet_status_.downloadOnStream(host_jet_status_.data(), count, queue_.getNative());
    for (std::size_t row = 0; row < count; ++row)
    {
      const auto status = host_jet_status_[row];
      if (status != device_math::JetMathStatus::REGULAR)
        throw InferenceStatusError((InferenceStatusWord(stage) << 48) |
            (InferenceStatusWord(row) << 8) | INFERENCE_STATUS_JET_DOMAIN |
            static_cast<unsigned int>(status), evaluation_generation_);
    }
  }

  /// Apply a bias/tanh transformation in place with preallocated diagnostics.
  void activate(double* values, const double* bias, double* retained_preactivation = nullptr)
  {
    checkRuntime(launchBiasTanhJets(queue_.getNative(), elementwise_, values, bias, values,
                                    open_status_.data(), retained_preactivation), "model bias/tanh");
    // GEMM initialized every plane before this in-place activation; early-return
    // failures therefore leave defined storage for subsequent deferred kernels.
    checkOpen(batch_ * ne_ * shape_.feature_dimension, true);
  }

  /// Apply a residual sum, permitting exact pointwise input/output aliasing.
  void residual(const double* left, const double* right, double* destination)
  {
    // The attention residual aliases b_. The final MLP residual writes a_,
    // which still contains this block's fully initialized input (embedding for
    // block zero, preceding residual thereafter). Both cases therefore leave
    // defined storage even when the leaf returns before writing every plane.
    if (status_policy_ == InferenceStatusPolicy::DEFERRED_NEURAL &&
        destination != left && destination != right && destination != a_.data())
      throw std::logic_error("Deferred inference residual requires an initialized destination");
    checkRuntime(launchResidualSpatialJets(queue_.getNative(), elementwise_, left, right, destination,
                                           open_status_.data()), "model residual");
    // Admission can defer only for the initialized destinations checked above.
    checkOpen(batch_ * ne_ * shape_.feature_dimension, true);
  }

  /// Describe the already prepared strided slab without charging its views.
  static void traceJet(const DeviceBuffer<double>& buffer, const SpatialJetLayout& layout,
                       const char* name) noexcept
  {
    if (!memory::enabled() || !layout.configuration_count || !layout.element_count)
      return;
    try
    {
      const auto shape = [&](std::size_t planes) {
        return std::to_string(layout.configuration_count) + "," + std::to_string(planes) + "," +
            std::to_string(layout.element_count) + ";strides=" + std::to_string(layout.configuration_stride) +
            "," + std::to_string(layout.plane_stride) + ",1";
      };
      buffer.traceView(name, 0, spatialJetSpanElements(layout), shape(layout.plane_count).c_str());
      // Plane-family views are strided across configurations; shape records that stride.
      const auto base = (layout.configuration_count - 1) * layout.configuration_stride;
      buffer.traceView("value_planes", 0, base + layout.element_count, shape(1).c_str());
      if (layout.gradient_lanes)
        buffer.traceView("gradient_planes", layout.plane_stride,
            base + (layout.gradient_lanes - 1) * layout.plane_stride + layout.element_count,
            shape(layout.gradient_lanes).c_str());
      if (layout.laplacian_lanes)
        buffer.traceView("laplacian_planes", (1 + layout.gradient_lanes) * layout.plane_stride,
            base + (layout.laplacian_lanes - 1) * layout.plane_stride + layout.element_count,
            shape(layout.laplacian_lanes).c_str());
    }
    catch (...)
    {
      memory::emit({"manifest_failed", buffer.memoryTraceIdentity(), name});
    }
  }

  /// Pack one contiguous jet interval into already prepared determinant storage.
  void copy(const DeviceBuffer<double>& source, std::size_t source_offset,
            DeviceBuffer<double>& target, std::size_t target_offset, std::size_t count)
  {
    DerivativeVectorLayout layout{count, source_offset, 1, target_offset, 1};
    checkRuntime(launchDerivativeCopy(queue_.getNative(), layout, source.data(), target.data()), "model jet packing");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::shared_ptr<PreparedDeviceMixedValueDense> mixed_dense_;
  const PsiFormerExecutionPlan& plan_;
  ModelShape shape_;
  std::size_t batch_;
  const std::size_t batch_capacity_;
  std::size_t ne_ = 0, gradient_lanes_ = 0, matrices_per_configuration_ = 0;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  const SpatialJetMode mode_;
  const TargetVglPreparation target_preparation_;
  const bool retained_targets_;
  const bool cooperative_value_lu_requested_;
  bool mixed_value_lu_excluded_ = false;
  const bool cooperative_open_features_requested_;
  const bool retained_open_features_requested_;
  bool mixed_open_features_excluded_ = false;
  const bool cooperative_value_cusp_requested_;
  const bool retained_value_cusp_requested_;
  const bool tiled_value_orbitals_requested_;
  bool target_tape_valid_ = false;
  TargetVglCost target_cost_;
  std::size_t admitted_target_tiles_ = 0;
  OpenFeatureJetLayout features_;
  SpatialDenseJetLayout embedding_, dense_;
  SpatialElementwiseJetLayout elementwise_;
  SpatialAttentionJetLayout attention_;
  OpenOrbitalJetLayout orbitals_;
  OpenCuspJetLayout cusp_;
  OpenCuspJetLayout cusp_state_layout_;
  bool cusp_state_enabled_ = false, cusp_state_valid_ = false;
  std::size_t cusp_state_generation_ = 0, cusp_state_batch_ = 0;
  const PreparedDeviceModel* cusp_state_model_ = nullptr;
  DeviceBuffer<double> cusp_state_;
  DeviceBuffer<OpenSpatialStatus> cusp_state_status_;
  FinalSpatialJetLayout final_;
  SpatialJetLayout result_layout_;
  DeviceBuffer<double> positions_, raw_, a_, b_, q_, k_, v_, weights_, context_, hidden_, projected_;
  DeviceBuffer<double> orbital_jets_, matrices_, lu_, inverse_, solve_;
  DeviceBuffer<double> term_phase_, term_log_, scaled_, normalized_, matrix_product_;
  DeviceBuffer<double> det_gradient_, det_lap_ratio_, det_lap_log_, cusp_jets_, phase_, output_, lap_ratio_;
  DeviceBuffer<std::size_t> active_, permutations_;
  DeviceBuffer<device_determinant::FactorizationMetadata> factors_;
  DeviceBuffer<device_determinant::CombinationMetadata> combinations_;
  DeviceBuffer<device_determinant::DerivativeStatus> det_status_;
  DeviceBuffer<OpenSpatialStatus> open_status_;
  DeviceBuffer<InferenceStatusWord> sticky_status_;
  DeviceBuffer<device_math::JetMathStatus> jet_status_;
  std::vector<OpenSpatialStatus> host_status_;
  std::vector<device_math::JetMathStatus> host_jet_status_;
  InferenceStatusPolicy status_policy_ = InferenceStatusPolicy::SYNCHRONOUS_DEBUG;
  std::size_t status_stage_ = 0;
  std::size_t status_generation_ = 0;
  bool status_pending_ = false;
  std::vector<std::unique_ptr<RetainedBlockTape>> value_tape_;
  DeviceBuffer<double> retained_backflow_, retained_envelope_;
  DeviceBuffer<double> full_phase_, full_output_, full_lap_ratio_;
  std::vector<double> tile_phase_, tile_log_, canonical_phase_, canonical_log_;
  std::vector<device_determinant::CombinationMetadata> host_combinations_;
  bool kinetic_trunk_only_ = false;
  bool value_tape_valid_ = false;
  bool result_valid_ = false;
  std::size_t evaluation_generation_ = 0;
  // Drain the borrowed stream before any earlier-declared orbital input buffers retire.
  std::unique_ptr<PreparedDeviceSpatialTraces> parallel_traces_;
};

/** Explicit matched approximate value/VGL with whole-request FP64 replay.
 * Spatial planes follow the standard mixed-AD convention; reverse tapes remain unavailable. */
class PreparedDeviceMixedInference
{
public:
  /// Prepare both routes up front; parameter conversion hazards may select FP64-only operation.
  PreparedDeviceMixedInference(std::shared_ptr<const PreparedDeviceModel> model, std::size_t capacity,
                                    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
                                    SpatialJetMode mode = SpatialJetMode::VALUE_ONLY,
                                    bool allow_fp64_retry = true)
      : reference_(model, capacity, queue, blas, mode), allow_retry_(allow_fp64_retry)
  {
    // Exclude the reference before conversion can fail and select FP64-only recovery.
    reference_.excludeCooperativeValueLuForMixedRequest();
    reference_.excludeCooperativeOpenFeaturesForMixedRequest();
    try
    {
      parameters_ = std::make_shared<PreparedDeviceMixedDenseParameters>(model, queue);
      adapter_ = std::make_shared<PreparedDeviceMixedValueDense>(parameters_,
          spatial_detail::checkedProduct(spatial_detail::checkedProduct(capacity,
              model->plan().modelShape().electrons(), "Mixed row overflow"),
              reference_.resultLayout().plane_count, "Mixed spatial row overflow"),
          queue, blas);
      approximate_ = std::make_unique<PreparedDeviceInference>(model, capacity, queue, blas, mode);
      approximate_->excludeCooperativeValueLuForMixedRequest();
      approximate_->excludeCooperativeOpenFeaturesForMixedRequest();
      approximate_->setMixedDense(adapter_);
    }
    catch (const MixedPrecisionRetryRequired& error)
    {
      if (!allow_retry_)
        throw;
      preparation_retry_reason_ = error.what();
    }
  }

  /// Preserve the original value-only retry-control constructor without an implicit mode conversion.
  PreparedDeviceMixedInference(std::shared_ptr<const PreparedDeviceModel> model, std::size_t capacity,
                                compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas,
                                bool allow_fp64_retry)
      : PreparedDeviceMixedInference(std::move(model), capacity, queue, blas,
                                    SpatialJetMode::VALUE_ONLY, allow_fp64_retry)
  {}

  /// Admit a complete approximate request or replay the same coordinates wholly in FP64.
  void evaluate(const std::vector<double>& positions)
  {
    selected_ = nullptr;
    used_retry_ = false;
    retry_reason_.clear();
    if (approximate_)
    {
      try
      {
        approximate_->evaluate(positions);
        selected_ = approximate_.get();
        return;
      }
      catch (const MixedPrecisionRetryRequired& error)
      {
        if (!allow_retry_)
          throw;
        retry_reason_ = error.what();
      }
      catch (const InferenceStatusError& error)
      {
        // Geometry failure is not a precision issue. Later numerical failures
        // may be mixed overflow; the reference replay must independently admit them.
        if (!allow_retry_ || error.stage() == 0)
          throw;
        retry_reason_ = error.what();
      }
    }
    else
      retry_reason_ = preparation_retry_reason_;
    used_retry_ = true;
    reference_.evaluate(positions);
    selected_ = &reference_;
  }

  /// Copy only a successfully admitted approximate or reference result.
  void downloadResults(std::vector<double>& phases, std::vector<double>& values, std::vector<double>& ratios) const
  {
    if (!selected_)
      throw std::logic_error("Mixed inference has no admitted result");
    selected_->downloadResults(phases, values, ratios);
  }

  /// Return the live-prefix layout, using prepared reference metadata before the first result.
  const SpatialJetLayout& resultLayout() const noexcept
  { return selected_ ? selected_->resultLayout() : reference_.resultLayout(); }

  /// Report a whole-graph retry explicitly rather than presenting it as mixed throughput.
  bool usedFp64Retry() const noexcept { return used_retry_; }

  /// Expose actual selected LU dispatch, or any eligible prepared route before evaluation.
  bool usesCooperativeValueLu() const noexcept
  {
    return selected_ ? selected_->usesCooperativeValueLu() :
        reference_.usesCooperativeValueLu() || (approximate_ && approximate_->usesCooperativeValueLu());
  }

  /// Report actual feature execution selection, or any eligible prepared route before evaluation.
  bool usesCooperativeOpenFeatures() const noexcept
  {
    return selected_ ? selected_->usesCooperativeOpenFeatures() :
        reference_.usesCooperativeOpenFeatures() || (approximate_ && approximate_->usesCooperativeOpenFeatures());
  }

  /// Describe the last numerical or preparation hazard selecting the reference route.
  const std::string& retryReason() const noexcept { return retry_reason_; }

  /// Expose explicit affine cast traffic for the attempted mixed request.
  const MixedValueDenseStatistics& statistics() const noexcept
  { return adapter_ ? adapter_->statistics() : no_statistics_; }

  /// Include both prepared graphs, FP32 scratch/cache and diagnostics, excluding the shared FP64 image.
  std::size_t deviceWorkspaceBytes() const
  {
    std::size_t bytes = reference_.deviceWorkspaceBytes();
    const auto add = [&](std::size_t amount) {
      if (bytes > std::numeric_limits<std::size_t>::max() - amount)
        throw std::overflow_error("Mixed value workspace byte count overflow");
      bytes += amount;
    };
    if (approximate_) add(approximate_->deviceWorkspaceBytes());
    if (adapter_) add(adapter_->deviceWorkspaceBytes());
    if (parameters_) add(parameters_->deviceStorageBytes());
    return bytes;
  }

private:
  PreparedDeviceInference reference_;
  std::shared_ptr<const PreparedDeviceMixedDenseParameters> parameters_;
  std::shared_ptr<PreparedDeviceMixedValueDense> adapter_;
  std::unique_ptr<PreparedDeviceInference> approximate_;
  PreparedDeviceInference* selected_ = nullptr;
  bool allow_retry_;
  bool used_retry_ = false;
  std::string preparation_retry_reason_;
  std::string retry_reason_;
  MixedValueDenseStatistics no_statistics_{};
};

/// Compatibility spelling retaining the original default value-only constructor.
using PreparedDeviceMixedValueInference = PreparedDeviceMixedInference;

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_INFERENCE_H
