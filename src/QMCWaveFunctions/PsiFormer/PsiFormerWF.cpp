//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWF.cpp
 * @brief QMCPACK integration for the native PsiFormer evaluator.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "Message/Communicate.h"
#include "Particle/MCMultiParticleMoves.h"
#include "Particle/VirtualParticleSet.h"
#include "ResourceCollection.h"
#include "Utilities/BatchResourcePreparation.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace qmcplusplus
{
namespace
{

/// Select migration behavior for calls that require only phase and log amplitude.
enum class DirectBackendMode
{
  DIRECT,
  ORACLE,
  COMPARE
};

/// Parse the developer migration switch once when shared model state is created.
DirectBackendMode configuredDirectBackend(const char* variable)
{
  const char* configured = std::getenv(variable);
  if (!configured || std::string(configured) == "direct")
    return DirectBackendMode::DIRECT;
  if (std::string(configured) == "oracle")
    return DirectBackendMode::ORACLE;
  if (std::string(configured) == "compare")
    return DirectBackendMode::COMPARE;
  throw std::invalid_argument(std::string(variable) + " must be direct, oracle, or compare");
}

/// Return a stable name for logging the selected value-only implementation.
const char* directBackendModeName(DirectBackendMode mode)
{
  switch (mode)
  {
  case DirectBackendMode::DIRECT:
    return "direct";
  case DirectBackendMode::ORACLE:
    return "oracle";
  case DirectBackendMode::COMPARE:
    return "compare";
  }
  return "unknown";
}

/// Convert the implementation switch to the dependency-light memory-policy enum.
psiformer::PsiFormerMemoryBackend memoryPolicyBackend(DirectBackendMode mode)
{
  switch (mode)
  {
  case DirectBackendMode::DIRECT:
    return psiformer::PsiFormerMemoryBackend::DIRECT;
  case DirectBackendMode::ORACLE:
    return psiformer::PsiFormerMemoryBackend::ORACLE;
  case DirectBackendMode::COMPARE:
    return psiformer::PsiFormerMemoryBackend::COMPARE;
  }
  throw std::logic_error("PsiFormer has an unknown direct-backend mode");
}

/// Return whether one component maximum fits inside the aggregate plan envelope.
bool capacitiesFitWithin(const BatchTileCapacities& capacities,
                         const BatchTileCapacities& envelope) noexcept
{
  return capacities.value <= envelope.value &&
      capacities.full_vgl <= envelope.full_vgl &&
      capacities.active_gradient <= envelope.active_gradient &&
      capacities.ecp_outer <= envelope.ecp_outer;
}

/// Reject planned operation families routed through an unaccounted developer backend.
void validatePlannedBackends(
    const psiformer::PsiFormerMemoryPolicyInput& input,
    const BatchExecutionRequirements& requirements)
{
  using Backend = psiformer::PsiFormerMemoryBackend;
  const bool has_active_parameters = input.active_parameter_count != 0;

  if (batchExecutionModeIsRequired(requirements, BatchExecutionMode::VALUE) &&
      input.backends.value != Backend::DIRECT)
    throw std::invalid_argument(
        "PsiFormer planned VALUE execution requires the direct backend");
  if ((requirements.requires(BatchExecutionMode::FULL_VGL) ||
       requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT)) &&
      input.backends.spatial != Backend::DIRECT)
    throw std::invalid_argument(
        "PsiFormer planned spatial execution requires the direct backend");
  if (has_active_parameters &&
      (requirements.requires(BatchExecutionMode::SCORE) ||
       requirements.requires(BatchExecutionMode::ECP_WEIGHTED_SCORE)) &&
      input.backends.score != Backend::DIRECT)
    throw std::invalid_argument(
        "PsiFormer planned score execution requires the direct backend");
  if (has_active_parameters &&
      requirements.requires(BatchExecutionMode::KINETIC) &&
      input.backends.kinetic != Backend::DIRECT)
    throw std::invalid_argument(
        "PsiFormer planned kinetic execution requires the direct backend");
}

/// Initial value for deterministic FNV-1a identities stored in walker buffers.
constexpr std::uint64_t PERSISTENT_FINGERPRINT_OFFSET = UINT64_C(14695981039346656037);

/// Prime used by deterministic FNV-1a identities stored in walker buffers.
constexpr std::uint64_t PERSISTENT_FINGERPRINT_PRIME = UINT64_C(1099511628211);

/// Keep ephemeral selected-team identities disjoint from persistent model/configuration hashes.
constexpr std::uint64_t SELECTED_TEAM_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053464c414e4531); // "PSFLANE1"

/// Keep a descriptor-bound selected transaction disjoint from its owning team identity.
constexpr std::uint64_t SELECTED_TRANSACTION_FINGERPRINT_DOMAIN =
    UINT64_C(0x50534654584e3031); // "PSFTXN01"

/// Identify the fixed scalar record layout used by the Stage-8 walker buffer.
constexpr std::uint64_t WALKER_BUFFER_MAGIC = UINT64_C(0x505349464f524d38);

/// Version the persistent accepted-state layout independently of VP persistence.
constexpr std::uint64_t WALKER_BUFFER_SCHEMA_VERSION = 1;

/// Number of exact scalar slots occupied by one complete accepted-state header.
constexpr std::size_t WALKER_BUFFER_SCALAR_COUNT = 17;

/// Mix one byte into a deterministic persistent-state fingerprint.
void mixPersistentByte(std::uint64_t& hash, std::uint8_t byte) noexcept
{
  hash ^= byte;
  hash *= PERSISTENT_FINGERPRINT_PRIME;
}

/// Mix one fixed-width integer into a deterministic persistent-state fingerprint.
void mixPersistentInteger(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
    mixPersistentByte(hash, static_cast<std::uint8_t>(value >> (8 * byte)));
}

/// Mix one length-delimited string into a deterministic persistent-state fingerprint.
void mixPersistentString(std::uint64_t& hash, const std::string& value) noexcept
{
  mixPersistentInteger(hash, value.size());
  for (unsigned char character : value)
    mixPersistentByte(hash, character);
}

/// Mix the exact binary64 representation of one native evaluator input.
void mixPersistentDouble(std::uint64_t& hash, double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  mixPersistentInteger(hash, bits);
}

/** Build a cross-rank model identity from immutable physics, layout, and initial weights.
 *
 * Including initial weights distinguishes independently loaded models whose local
 * parameter-version counters happen to agree.  Later optimizer changes are tracked
 * separately by the synchronized parameter version.
 */
std::uint64_t persistentModelIdentity(const pf::PsiFormer& model,
                                      const std::string& model_origin,
                                      const std::string& initialization_profile,
                                      std::uint64_t initialization_seed)
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentString(hash, model_origin);
  mixPersistentString(hash, initialization_profile);
  mixPersistentInteger(hash, initialization_seed);
  mixPersistentString(hash, model.p.layout_fingerprint());
  mixPersistentInteger(hash, model.ne);
  mixPersistentInteger(hash, model.cfg.nup);
  mixPersistentInteger(hash, model.cfg.ndown);
  mixPersistentInteger(hash, model.ndet);
  mixPersistentInteger(hash, model.dim);
  mixPersistentInteger(hash, model.heads);
  mixPersistentInteger(hash, model.blocks);
  for (std::size_t extent : model.cfg.nuclei.shape)
    mixPersistentInteger(hash, extent);
  for (double coordinate : model.cfg.nuclei.x)
    mixPersistentDouble(hash, coordinate);
  for (double charge : model.cfg.charges.x)
    mixPersistentDouble(hash, charge);
  for (double parameter : model.p.flat_values())
    mixPersistentDouble(hash, parameter);
  return hash;
}

/** Fingerprint the exact configuration presented to the native double evaluator.
 *
 * A replacement position represents the active proposal before ParticleSet commits
 * it, allowing acceptMove to publish the correct accepted identity immediately.
 */
std::uint64_t configurationIdentity(const ParticleSet& particles,
                                    int replaced_particle = -1,
                                    const ParticleSet::PosType* replacement_position = nullptr) noexcept
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(hash, particles.getTotalNum());
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    mixPersistentInteger(hash, static_cast<std::uint64_t>(particles.GroupID[electron]));
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : particles.activeR(electron))
        : particles.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      mixPersistentDouble(hash, static_cast<double>(position[dimension]));
  }
  return hash;
}

/// Fingerprint a selected-electron proposal without modifying the accepted ParticleSet.
std::uint64_t configurationIdentity(
    const ParticleSet& particles,
    const MCMultiParticleMoves<CoordsType::POS>::Slice& moves)
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(hash, particles.getTotalNum());
  std::size_t selected = 0;
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    mixPersistentInteger(hash, static_cast<std::uint64_t>(particles.GroupID[electron]));
    const bool replaced = selected < moves.size() && moves.particleIndex(selected) == electron;
    const auto& position = replaced ? moves.proposedPosition(selected) : particles.R[electron];
    if (replaced)
      ++selected;
    for (int dimension = 0; dimension < 3; ++dimension)
      mixPersistentDouble(hash, static_cast<double>(position[dimension]));
  }
  return hash;
}

/// Return whether a descriptor leaves one lane bitwise identical to accepted coordinates.
bool selectedCoordinatesExactlyUnchanged(
    const ParticleSet& particles,
    const MCMultiParticleMoves<CoordsType::POS>::Slice& moves)
{
  for (std::size_t selected = 0; selected < moves.size(); ++selected)
  {
    const auto electron = static_cast<std::size_t>(moves.particleIndex(selected));
    const auto& accepted = particles.R[electron];
    const auto& proposed = moves.proposedPosition(selected);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      if (std::memcmp(std::addressof(accepted[dimension]),
                      std::addressof(proposed[dimension]),
                      sizeof(accepted[dimension])) != 0)
        return false;
  }
  return true;
}

/// Store one uint64 as two exactly representable 32-bit limbs in the scalar pool.
void putPersistentInteger(WaveFunctionComponent::WFBufferType& buffer, std::uint64_t value)
{
  double low  = static_cast<double>(value & UINT64_C(0xffffffff));
  double high = static_cast<double>(value >> 32);
  buffer.put(low);
  buffer.put(high);
}

/// Decode and validate one exactly represented 32-bit scalar-pool limb.
std::uint32_t getPersistentLimb(WaveFunctionComponent::WFBufferType& buffer, const char* description)
{
  double encoded;
  buffer.get(encoded);
  if (!psiformer::determinant::isFiniteReal(encoded) || encoded < 0.0 ||
      encoded > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
    throw std::runtime_error(std::string("PsiFormer walker buffer has an invalid ") + description + " limb");
  const auto decoded = static_cast<std::uint32_t>(encoded);
  if (encoded != static_cast<double>(decoded))
    throw std::runtime_error(std::string("PsiFormer walker buffer has a fractional ") + description + " limb");
  return decoded;
}

/// Restore one uint64 from two validated 32-bit scalar-pool limbs.
std::uint64_t getPersistentInteger(WaveFunctionComponent::WFBufferType& buffer, const char* description)
{
  const std::uint64_t low  = getPersistentLimb(buffer, description);
  const std::uint64_t high = getPersistentLimb(buffer, description);
  return low | (high << 32);
}

/// Return true only for the binary64 encoding of negative infinity.
bool isNegativeInfinity(double value) noexcept
{
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == UINT64_C(0xfff0000000000000);
}

/// Couple an in-memory native model to the provenance needed by persistence and diagnostics.
struct InitializedNativeModel
{
  pf::PsiFormer model;
  std::string profile;
  std::uint64_t seed;
};

/** Convert dependency-light initialized parameters and QMCPACK particle metadata
 * into the native evaluator's owning representation. */
InitializedNativeModel makeInitializedNativeModel(
    psiformer::InitializedPsiFormerParameters initialized,
    const ParticleSet& electrons,
    const ParticleSet& ions)
{
  const psiformer::ModelShape& shape = initialized.model_shape;
  if (electrons.groups() != 2 || electrons.groupsize(0) <= 0 || electrons.groupsize(1) <= 0)
    throw std::invalid_argument(
        "Internally initialized PsiFormer requires two nonempty electron spin groups");
  if (shape.spin_up_electrons != static_cast<std::size_t>(electrons.groupsize(0)) ||
      shape.spin_down_electrons != static_cast<std::size_t>(electrons.groupsize(1)) ||
      shape.nuclei != static_cast<std::size_t>(ions.getTotalNum()))
    throw std::invalid_argument(
        "Internally initialized PsiFormer dimensions do not match the QMCPACK particle sets");

  const SpeciesSet& ion_species = ions.getSpeciesSet();
  const int charge_index        = ion_species.findAttribute("charge");
  if (charge_index < 0)
    throw std::invalid_argument(
        "Internally initialized PsiFormer source particle set has no charge attribute");

  std::vector<double> electron_positions;
  electron_positions.reserve(3 * electrons.getTotalNum());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electron_positions.push_back(static_cast<double>(electrons.R[electron][dimension]));

  std::vector<double> nuclear_positions;
  std::vector<double> nuclear_charges;
  nuclear_positions.reserve(3 * ions.getTotalNum());
  nuclear_charges.reserve(ions.getTotalNum());
  for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      nuclear_positions.push_back(static_cast<double>(ions.R[nucleus][dimension]));
    nuclear_charges.push_back(ion_species(charge_index, ions.GroupID[nucleus]));
  }

  std::vector<pf::Layout> native_layouts;
  native_layouts.reserve(initialized.layouts.size());
  for (psiformer::ParameterLayoutInput& layout : initialized.layouts)
    native_layouts.push_back({std::move(layout.module), std::move(layout.name),
                              std::move(layout.shape), layout.begin, layout.end});

  pf::Parameters parameters(std::move(initialized.values), std::move(native_layouts));
  pf::ConfigData configuration(
      pf::Tensor({1, shape.electrons(), 3}, std::move(electron_positions)),
      pf::Tensor({shape.nuclei, 3}, std::move(nuclear_positions)),
      pf::Tensor({shape.nuclei}, std::move(nuclear_charges)), shape.spin_up_electrons,
      shape.spin_down_electrons);
  pf::PsiFormer model(std::move(parameters), std::move(configuration), shape.determinants,
                      shape.feature_dimension, shape.attention_heads, shape.attention_blocks);
  return {std::move(model), std::move(initialized.profile), initialized.seed};
}

/// Convert the native DeepQMC leaf layout into compact tensor-level training metadata.
std::shared_ptr<const wftrain::StructuredParameterSchema> makeStructuredParameterSchema(
    const std::string& component_name,
    const pf::Parameters& parameters)
{
  std::vector<wftrain::ParameterBlockDescriptor> blocks;
  blocks.reserve(parameters.layouts.size());
  for (const pf::Layout& layout : parameters.layouts)
    blocks.push_back({layout.module + "/" + layout.name, layout.shape, layout.begin,
                      layout.end - layout.begin, wftrain::ParameterScalarDomain::REAL64,
                      true, "neural_network"});
  return std::make_shared<const wftrain::StructuredParameterSchema>(
      "psiformer/" + component_name, std::move(blocks));
}

} // namespace

/** Shared native model protected at the optimizer/evaluator synchronization
 * boundary. Component clones retain only walker-local move state. */
class PsiFormerSharedState
{
public:
  /// Load the model that all clones of one PsiFormer component will share.
  PsiFormerSharedState(const std::string& parameters, const std::string& configuration)
      : PsiFormerSharedState(pf::PsiFormer(parameters, configuration), "hdf5", "external_hdf5", 0)
  {}

  /// Materialize a fresh in-memory model and retain its reproducibility provenance.
  PsiFormerSharedState(psiformer::InitializedPsiFormerParameters initialized,
                       const ParticleSet& electrons,
                       const ParticleSet& ions)
      : PsiFormerSharedState(makeInitializedNativeModel(std::move(initialized), electrons, ions))
  {}

  /// Take ownership of a converted initialized model without copying its flat parameters.
  explicit PsiFormerSharedState(InitializedNativeModel initialized)
      : PsiFormerSharedState(std::move(initialized.model), "internal",
                             std::move(initialized.profile), initialized.seed)
  {}

  /// Complete shared executor construction for imported and internally initialized models.
  PsiFormerSharedState(pf::PsiFormer model_input,
                       std::string origin,
                       std::string profile,
                       std::uint64_t seed)
      : model(std::move(model_input)),
        model_origin(std::move(origin)),
        initialization_profile(std::move(profile)),
        initialization_seed(seed),
        persistent_model_identity(
            persistentModelIdentity(model, model_origin, initialization_profile, initialization_seed)),
        execution_plan(psiformer::PsiFormerExecutionPlan::fromParameters(
            model.p,
            {/*spin_up_electrons=*/model.cfg.nup,
             /*spin_down_electrons=*/model.cfg.ndown,
             /*nuclei=*/model.cfg.nuclei.shape[0],
             /*determinants=*/model.ndet,
             /*feature_dimension=*/model.dim,
             /*attention_heads=*/model.heads,
             /*attention_blocks=*/model.blocks})),
        direct_value_executor(model, execution_plan),
        direct_spatial_executor(model, direct_value_executor, execution_plan),
        direct_batch_executor(direct_value_executor, direct_spatial_executor),
        direct_score_executor(model, execution_plan),
        direct_kinetic_executor(model, execution_plan),
        direct_value_mode(configuredDirectBackend("PSIFORMER_VALUE_BACKEND")),
        direct_spatial_mode(configuredDirectBackend("PSIFORMER_SPATIAL_BACKEND")),
        direct_score_mode(configuredDirectBackend("PSIFORMER_SCORE_BACKEND")),
        direct_kinetic_mode(configuredDirectBackend("PSIFORMER_KINETIC_BACKEND"))
  {}

  mutable std::shared_mutex mutex;
  /// Number of pending planned selected crowd transactions sharing this model.
  /// Each crowd checks in once, never once per lane.
  std::atomic<std::size_t> planned_selected_transaction_count{0};
  pf::PsiFormer model;
  /// Whether the initial model came from the legacy HDF5 import or an internal initializer.
  const std::string model_origin;
  /// Versioned initializer name or stable external-HDF5 profile sentinel.
  const std::string initialization_profile;
  /// Explicit component-local initialization seed, zero for imported models.
  const std::uint64_t initialization_seed;
  /// Stable content identity used to reject buffers from another physical model.
  const std::uint64_t persistent_model_identity;
  /// Immutable typed tensor descriptors shared by every component clone.
  const psiformer::PsiFormerExecutionPlan execution_plan;
  /// Immutable direct forward executor sharing the versioned parameter store.
  const pf::DirectValueExecutor direct_value_executor;
  /// Immutable direct trace-jet executor sharing the value parameter layout.
  const pf::DirectSpatialExecutor direct_spatial_executor;
  /// Direct configuration-major batch boundary shared by clone and crowd workspaces.
  const pf::DirectBatchExecutor direct_batch_executor;
  /// Immutable direct score executor sharing typed model metadata and parameters.
  const pf::DirectScoreExecutor direct_score_executor;
  /// Immutable direct combined score/kinetic trace-jet executor.
  const pf::DirectKineticExecutor direct_kinetic_executor;
  /// Development switch selecting direct, oracle, or compare behavior.
  const DirectBackendMode direct_value_mode;
  /// Development switch selecting direct, oracle, or compare spatial behavior.
  const DirectBackendMode direct_spatial_mode;
  /// Development switch selecting direct, oracle, or compare score behavior.
  const DirectBackendMode direct_score_mode;
  /// Development switch selecting direct, oracle, or compare kinetic behavior.
  const DirectBackendMode direct_kinetic_mode;
};

/** Optimizer metadata shared by every clone of one PsiFormer component.
 *
 * VariableSet is retained as the compatibility boundary with the existing
 * QMCPACK optimizer, but it lives here instead of in OptimizableObject::myVars.
 * Its values and local-to-global indices change only at optimizer barriers.
 */
class PsiFormerOptimizationMetadata
{
public:
  /// Take ownership of the canonical selection and its legacy optimizer view.
  PsiFormerOptimizationMetadata(bool enabled,
                                bool optimize_all,
                                std::vector<std::size_t> selected_flat_indices,
                                OptVariables variables)
      : enabled(enabled),
        optimize_all(optimize_all),
        selected_flat_indices(std::move(selected_flat_indices)),
        variables(std::move(variables))
  {}

  /// Serialize mutations of values and global indices at optimizer barriers.
  mutable std::shared_mutex mutex;
  const bool enabled;
  const bool optimize_all;
  const std::vector<std::size_t> selected_flat_indices;
  OptVariables variables;
};

/** Hold the shared-model mutex for one complete logical read operation.
 *
 * The captured version is meaningful only while this transaction remains
 * alive: direct executors retain a pointer into the same protected parameter
 * vector, and native oracle graphs retain its parameter leaves.
 */
class PsiFormerReadTransaction
{
public:
  explicit PsiFormerReadTransaction(PsiFormerSharedState& state)
      : state_(state), lock_(state_.mutex), parameter_version_(state_.model.p.version())
  {}

  const PsiFormerSharedState& state() const noexcept { return state_; }
  const pf::PsiFormer& model() const noexcept { return state_.model; }
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

private:
  PsiFormerSharedState& state_;
  std::shared_lock<std::shared_mutex> lock_;
  const std::size_t parameter_version_;
};

/** Extend one model transaction with the optimizer mapping using state-first order. */
class PsiFormerDerivativeReadTransaction
{
public:
  PsiFormerDerivativeReadTransaction(PsiFormerSharedState& state,
                                     PsiFormerOptimizationMetadata& metadata)
      : model_transaction_(state),
        metadata_(metadata),
        metadata_lock_(metadata_.mutex)
  {}

  const PsiFormerReadTransaction& modelTransaction() const noexcept
  { return model_transaction_; }

  const std::vector<std::size_t>& selectedFlatIndices() const noexcept
  { return metadata_.selected_flat_indices; }

  const OptVariables& variables() const noexcept { return metadata_.variables; }

  const PsiFormerOptimizationMetadata& metadata() const noexcept
  { return metadata_; }

  bool hasActiveParameters() const noexcept
  {
    for (std::size_t local_index = 0; local_index < metadata_.variables.size();
         ++local_index)
      if (metadata_.variables.where(local_index) >= 0)
        return true;
    return false;
  }

private:
  PsiFormerReadTransaction model_transaction_;
  PsiFormerOptimizationMetadata& metadata_;
  std::shared_lock<std::shared_mutex> metadata_lock_;
};

/** Crowd-owned mutable storage.  ResourceCollection cloning recreates scratch
 * against the same immutable/versioned model state without copying buffers. */
struct PsiFormerWF::PsiFormerMultiWalkerResource : public Resource
{
  /** Create an empty template bound to one model and, when selected, one
   * structurally identified participant view. */
  PsiFormerMultiWalkerResource(std::shared_ptr<PsiFormerSharedState> model_state,
                               psiformer::PsiFormerMemoryPolicyInput memory_policy_input,
                               std::string participant_id = {},
                               BatchExecutionParticipantPlan expected_plan = {})
      : Resource("PsiFormerMultiWalkerResource"),
        model_state(std::move(model_state)),
        memory_policy_input(std::move(memory_policy_input)),
        participant_id(std::move(participant_id)),
        expected_plan(std::move(expected_plan)),
        batch_workspace(this->model_state->direct_batch_executor.makeWorkspace())
  {}

  /** Clone template provenance but never prepared or lazily accumulated
   * numeric storage.  Collection provenance separately marks copies derived
   * from a prepared collection as requiring an explicit clear. */
  PsiFormerMultiWalkerResource(const PsiFormerMultiWalkerResource& other)
      : PsiFormerMultiWalkerResource(other.model_state,
                                     other.memory_policy_input,
                                     other.participant_id,
                                     other.expected_plan)
  {}

  /// Recreate an independent resource for a copied crowd resource collection.
  std::unique_ptr<Resource> makeClone() const override
  { return std::make_unique<PsiFormerMultiWalkerResource>(*this); }

  /** Reject invalid or overlapping preparation before ResourceCollection
   * constructs the first transactional candidate. */
  void validateBatchResourcePreparation(
      const BatchResourcePreparationContext& context) const override
  {
    context.validate();
    if (!context.plan)
      return;
    if (prepared_plan)
      throw std::logic_error(
          "PsiFormer nonnull crowd-resource replanning requires an explicit null clear");
    if (retainedNumericBytes() != 0)
      throw std::logic_error(
          "PsiFormer lazy crowd storage must be cleared before applying a batch plan");
    if (participant_id.empty())
      throw std::logic_error(
          "PsiFormer crowd resource was created before its batch participant identity was bound");

    const BatchExecutionParticipantPlan selected =
        makeBatchExecutionParticipantPlan(context.plan, participant_id);
    if (expected_plan && !expected_plan.sameBinding(selected))
      throw std::invalid_argument(
          "PsiFormer crowd resource preparation received the wrong participant plan");
    validateParticipantEvidence(selected);
    const std::vector<psiformer::PsiFormerCrowdMemoryPlan> crowd_plans =
        makeCrowdPlans(selected);
    if (context.crowd_index >= crowd_plans.size())
      throw std::out_of_range(
          "PsiFormer crowd resource preparation index is outside its memory plan");
  }

  /** Materialize one exact candidate, or return it to empty legacy mode for a
   * null context.  Publication into this resource happens only after every
   * candidate allocation and byte check succeeds. */
  void prepareBatchResource(const BatchResourcePreparationContext& context) override
  {
    validateBatchResourcePreparation(context);
    if (!context.plan)
    {
      clearToNoPolicy();
      return;
    }

    BatchExecutionParticipantPlan selected =
        makeBatchExecutionParticipantPlan(context.plan, participant_id);
    std::vector<psiformer::PsiFormerCrowdMemoryPlan> crowd_plans =
        makeCrowdPlans(selected);
    psiformer::PsiFormerCrowdMemoryPlan crowd_plan =
        std::move(crowd_plans.at(context.crowd_index));

    PsiFormerMultiWalkerResource candidate(model_state, memory_policy_input,
                                            participant_id, selected);
    candidate.materializePreparedStorage(selected, context.crowd_index,
                                         std::move(crowd_plan));
    publishPreparedStorage(std::move(candidate));
  }

  std::shared_ptr<PsiFormerSharedState> model_state;
  /// Immutable model/backend/type facts used to reproduce selected evidence.
  psiformer::PsiFormerMemoryPolicyInput memory_policy_input;
  /// Stable structural identity retained across an explicit null clear.
  std::string participant_id;
  /// Template binding captured when the golden ResourceCollection was built.
  BatchExecutionParticipantPlan expected_plan;
  /// Binding published only after exact crowd storage is complete.
  BatchExecutionParticipantPlan prepared_plan;
  /// Exact crowd allocation descriptor paired with the prepared binding.
  std::optional<psiformer::PsiFormerCrowdMemoryPlan> prepared_crowd_plan;
  std::size_t prepared_crowd_index          = 0;
  std::uint64_t prepared_plan_fingerprint   = 0;
  std::size_t prepared_storage_fingerprint = 0;
  /// Exact publication/operation staging identities retained at preparation.
  const std::uint64_t* prepared_configuration_identities_data = nullptr;
  const double* prepared_staged_signs_data                     = nullptr;
  const double* prepared_staged_log_magnitudes_data            = nullptr;
  const std::size_t* prepared_walker_indices_data               = nullptr;
  const unsigned char* prepared_preservation_flags_data         = nullptr;
  const std::size_t* prepared_active_electrons_data             = nullptr;
  const GradType* prepared_staged_gradients_data                = nullptr;
  std::size_t prepared_configuration_identities_capacity       = 0;
  std::size_t prepared_staged_signs_capacity                   = 0;
  std::size_t prepared_staged_log_magnitudes_capacity          = 0;
  std::size_t prepared_walker_indices_capacity                  = 0;
  std::size_t prepared_preservation_flags_capacity              = 0;
  std::size_t prepared_active_electrons_capacity                = 0;
  std::size_t prepared_staged_gradients_capacity                = 0;
  BatchMemoryEstimate actual_resource_storage;
  /// Preparation provenance captured while this exact resource is on loan.
  const BatchExecutionPlan* acquired_plan_identity = nullptr;
  std::size_t acquired_crowd_index                  = 0;
  bool acquired_from_prepared_collection            = false;

  std::unique_ptr<pf::DirectBatchWorkspace> batch_workspace;
  /// Lazily allocated score tape serialized across component-major crowd calls.
  std::unique_ptr<pf::DirectScoreWorkspace> score_workspace;
  /// Lazily allocated kinetic tape serialized across component-major crowd calls.
  std::unique_ptr<pf::DirectKineticWorkspace> kinetic_workspace;
  /// Complete per-walker drift packed immediately before a kinetic reverse pass.
  std::vector<double> total_log_gradient;
  /// Exact identities used to publish accepted/proposed state transactionally.
  std::vector<std::uint64_t> configuration_identities;
  /// Dense-result slots for configurations that require native evaluation.
  std::vector<std::size_t> batch_slots;
  /// Shared phase/sign staging across dense crowd operation families.
  std::vector<double> staged_signs;
  /// Shared logarithmic-amplitude staging across dense crowd operation families.
  std::vector<double> staged_log_magnitudes;
  /// Real/complex wavefunction-value ratios when FULL_VGL is not reachable.
  std::vector<ValueType> staged_value_ratios;
  /// Full-precision complex log ratios, also reused by narrower ratio calls.
  std::vector<LogValue> staged_log_ratios;
  /// Shared gradient publication staging for active and full spatial calls.
  std::vector<GradType> staged_gradients;
  /// Byte-addressable flags retaining accepted spatial products during recompute.
  std::vector<unsigned char> preservation_flags;
  /// Per-configuration active-electron indices consumed by the native batch API.
  std::vector<std::size_t> active_electrons;
  /// Prefix offsets mapping flattened ragged virtual configurations back to walkers.
  std::vector<std::size_t> virtual_offsets;
  /// Compact active-walker order used by flattened sparse virtual batches.
  std::vector<std::size_t> active_virtual_walkers;
  /// Map every descriptor walker to its compact sparse-reference slot.
  std::vector<std::size_t> virtual_reference_indices;
  /// Atomic flattened ratio staging retained across sparse virtual calls.
  std::vector<ValueType> flat_virtual_ratios;
  /// Sum of descriptor weights multiplying each compact walker reference score.
  std::vector<ValueType> virtual_reference_weights;
  /// Global derivative destinations in selected-parameter iteration order.
  std::vector<std::size_t> active_derivative_global_indices;
  /// Atomic compact rows indexed by active virtual walker then active parameter.
  std::vector<ValueType> flat_virtual_weighted_derivatives;
  /// Reusable selected-score contribution gathered after each serialized reverse pass.
  SelectedDerivativeDelta virtual_score_contribution;
  /// Second selected-derivative contribution retained for kinetic responses.
  SelectedDerivativeDelta kinetic_parameter_contribution;
  /// Oracle reference signs retained across flattened virtual calls.
  std::vector<double> virtual_reference_signs;
  /// Oracle reference log magnitudes retained across flattened virtual calls.
  std::vector<double> virtual_reference_logabs;
  /// Selected walker indices for masked recomputation.
  std::vector<std::size_t> walker_indices;
  /// Successful flattened weighted-call diagnostics; failures leave these unchanged.
  std::size_t weighted_reference_configurations   = 0;
  std::size_t weighted_replacement_configurations = 0;
  std::size_t weighted_active_parameters           = 0;

  /** Return the score tape, allocating only under the legacy no-policy
   * contract.  A hard plan must have prepared it already. */
  pf::DirectScoreWorkspace& requireScoreWorkspace()
  {
    if (expected_plan || prepared_plan)
    {
      if (!prepared_plan || !score_workspace)
        throw std::logic_error(
            "PsiFormer planned crowd score workspace was not prepared");
      return *score_workspace;
    }
    if (!score_workspace)
      score_workspace = model_state->direct_score_executor.makeWorkspace();
    return *score_workspace;
  }

  /** Return the kinetic tape, allocating only under the legacy no-policy
   * contract. */
  pf::DirectKineticWorkspace& requireKineticWorkspace()
  {
    if (expected_plan || prepared_plan)
    {
      if (!prepared_plan || !kinetic_workspace)
        throw std::logic_error(
            "PsiFormer planned crowd kinetic workspace was not prepared");
      return *kinetic_workspace;
    }
    if (!kinetic_workspace)
      kinetic_workspace = model_state->direct_kinetic_executor.makeWorkspace();
    return *kinetic_workspace;
  }

  /** Return fixed-size drift storage without growing it under a hard plan. */
  std::vector<double>& requireTotalLogGradient()
  {
    const std::size_t required_size =
        3 * model_state->execution_plan.modelShape().electrons();
    if (expected_plan || prepared_plan)
    {
      if (!prepared_plan || total_log_gradient.capacity() != required_size)
        throw std::logic_error(
            "PsiFormer planned crowd total-drift buffer was not prepared");
      // releaseResource clears logical contents while retaining capacity.
      // Restoring the admitted extent cannot reallocate after the exact check.
      total_log_gradient.resize(required_size);
      return total_log_gradient;
    }
    if (total_log_gradient.empty())
      total_log_gradient.resize(required_size);
    if (total_log_gradient.size() != required_size)
      throw std::logic_error("PsiFormer crowd total-drift buffer has the wrong size");
    return total_log_gradient;
  }

  /** Validate one acquired resource against model, participant, crowd, and
   * live-lane provenance. */
  void validateAcquiredBinding(const PsiFormerSharedState* expected_model,
                               const BatchExecutionParticipantPlan& component_plan,
                               std::optional<std::size_t> collection_crowd,
                               std::size_t live_lanes) const
  {
    if (model_state.get() != expected_model)
      throw std::logic_error(
          "PsiFormer ResourceCollection belongs to a different model");

    if (!component_plan)
    {
      if (expected_plan || prepared_plan || prepared_crowd_plan)
        throw std::logic_error(
            "PsiFormer no-policy component acquired a planned crowd resource");
      return;
    }

    if (!expected_plan.sameBinding(component_plan) ||
        !prepared_plan.sameBinding(component_plan) || !prepared_crowd_plan)
      throw std::logic_error(
          "PsiFormer crowd resource was not prepared for the component batch plan");
    if (participant_id != component_plan.evidence().participant_id)
      throw std::logic_error(
          "PsiFormer crowd resource has the wrong participant identity");
    if (prepared_plan_fingerprint != component_plan.plan().fingerprint())
      throw std::logic_error(
          "PsiFormer crowd resource has stale plan provenance");
    if (collection_crowd && prepared_crowd_index != *collection_crowd)
      throw std::logic_error(
          "PsiFormer crowd resource and ResourceCollection disagree on crowd identity");
    if (live_lanes > prepared_crowd_plan->reserve_walkers)
      throw std::length_error(
          "PsiFormer live crowd exceeds its prepared reserve envelope");
    if (prepared_storage_fingerprint == 0 ||
        prepared_storage_fingerprint != storageFingerprint())
      throw std::logic_error(
          "PsiFormer prepared crowd storage changed after plan publication");
  }

  /// Recompute retained allocation identity without exposing mutable storage.
  std::size_t currentStorageFingerprint() const noexcept
  { return storageFingerprint(); }

  /// Planned resources retain every typed staging vector at its full exact extent.
  bool hasExactPreparedStagingExtents() const noexcept
  {
    const auto exact = [](const auto& values) noexcept {
      return values.size() == values.capacity();
    };
    return exact(total_log_gradient) && exact(configuration_identities) &&
        exact(batch_slots) && exact(staged_signs) &&
        exact(staged_log_magnitudes) && exact(staged_value_ratios) &&
        exact(staged_log_ratios) && exact(staged_gradients) &&
        exact(preservation_flags) && exact(active_electrons) &&
        exact(virtual_offsets) && exact(active_virtual_walkers) &&
        exact(virtual_reference_indices) && exact(flat_virtual_ratios) &&
        exact(virtual_reference_weights) &&
        exact(active_derivative_global_indices) &&
        exact(flat_virtual_weighted_derivatives) &&
        exact(virtual_score_contribution) &&
        exact(kinetic_parameter_contribution) &&
        exact(virtual_reference_signs) && exact(virtual_reference_logabs) &&
        exact(walker_indices);
  }

  /// Report overlap with any allocation retained directly by this resource.
  bool overlapsStagingStorage(const void* data, std::size_t bytes) const noexcept
  {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    if (bytes == 0)
      return false;
    if (data == nullptr || begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
      return true;
    const std::uintptr_t end = begin + bytes;
    const auto overlaps = [begin, end](const auto& values) noexcept {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      if (values.capacity() == 0)
        return false;
      if (values.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Element))
        return true;
      const std::size_t storage_bytes = values.capacity() * sizeof(Element);
      const std::uintptr_t storage_begin =
          reinterpret_cast<std::uintptr_t>(values.data());
      if (values.data() == nullptr ||
          storage_begin > std::numeric_limits<std::uintptr_t>::max() - storage_bytes)
        return true;
      const std::uintptr_t storage_end = storage_begin + storage_bytes;
      return begin < storage_end && storage_begin < end;
    };
    return overlaps(total_log_gradient) || overlaps(configuration_identities) ||
        overlaps(batch_slots) || overlaps(staged_signs) ||
        overlaps(staged_log_magnitudes) || overlaps(staged_value_ratios) ||
        overlaps(staged_log_ratios) || overlaps(staged_gradients) ||
        overlaps(preservation_flags) || overlaps(active_electrons) ||
        overlaps(virtual_offsets) || overlaps(active_virtual_walkers) ||
        overlaps(virtual_reference_indices) || overlaps(flat_virtual_ratios) ||
        overlaps(virtual_reference_weights) ||
        overlaps(active_derivative_global_indices) ||
        overlaps(flat_virtual_weighted_derivatives) ||
        overlaps(virtual_score_contribution) ||
        overlaps(kinetic_parameter_contribution) ||
        overlaps(virtual_reference_signs) ||
        overlaps(virtual_reference_logabs) || overlaps(walker_indices);
  }

  /** Validate the full prepared metadata arrays used through bounded prefixes.
   * Planned release retains physical extents, so runtime never resizes these
   * vectors and any size or allocation-identity drift fails closed.
   */
  void requireFullVGLStaging(std::size_t live_walkers) const
  {
    if (!prepared_plan || !prepared_crowd_plan ||
        prepared_storage_fingerprint == 0 ||
        storageFingerprint() != prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer FULL_VGL staging lost its prepared allocation identity");

    const pf::ResourceStagingStorageRequirement& staging =
        prepared_crowd_plan->publication_storage;
    if (configuration_identities.data() !=
            prepared_configuration_identities_data ||
        staged_signs.data() != prepared_staged_signs_data ||
        staged_log_magnitudes.data() !=
            prepared_staged_log_magnitudes_data ||
        configuration_identities.capacity() !=
            prepared_configuration_identities_capacity ||
        staged_signs.capacity() != prepared_staged_signs_capacity ||
        staged_log_magnitudes.capacity() !=
            prepared_staged_log_magnitudes_capacity ||
        configuration_identities.size() !=
            prepared_configuration_identities_capacity ||
        staged_signs.size() != prepared_staged_signs_capacity ||
        staged_log_magnitudes.size() !=
            prepared_staged_log_magnitudes_capacity ||
        vectorBytes(configuration_identities,
                    "PsiFormer FULL_VGL configuration staging") !=
            staging.configuration_identities ||
        vectorBytes(staged_signs, "PsiFormer FULL_VGL sign staging") !=
            staging.signs ||
        vectorBytes(staged_log_magnitudes,
                    "PsiFormer FULL_VGL log staging") !=
            staging.log_magnitudes)
      throw std::logic_error(
          "PsiFormer FULL_VGL staging capacities differ from the prepared plan");
    if (live_walkers > configuration_identities.size() ||
        live_walkers > staged_signs.size() ||
        live_walkers > staged_log_magnitudes.size())
      throw std::length_error(
          "PsiFormer FULL_VGL staging prefix exceeds prepared capacity");
  }

  /// Validate every exact prepared prefix used by masked value recomputation.
  void requireRecomputeStaging(std::size_t selected_walkers) const
  {
    if (!prepared_plan || !prepared_crowd_plan ||
        prepared_storage_fingerprint == 0 ||
        storageFingerprint() != prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer RECOMPUTE_VALUE staging lost its prepared allocation identity");

    const pf::ResourceStagingStorageRequirement& staging =
        prepared_crowd_plan->publication_storage;
    if (configuration_identities.data() !=
            prepared_configuration_identities_data ||
        staged_signs.data() != prepared_staged_signs_data ||
        staged_log_magnitudes.data() !=
            prepared_staged_log_magnitudes_data ||
        walker_indices.data() != prepared_walker_indices_data ||
        preservation_flags.data() != prepared_preservation_flags_data ||
        configuration_identities.capacity() !=
            prepared_configuration_identities_capacity ||
        staged_signs.capacity() != prepared_staged_signs_capacity ||
        staged_log_magnitudes.capacity() !=
            prepared_staged_log_magnitudes_capacity ||
        walker_indices.capacity() != prepared_walker_indices_capacity ||
        preservation_flags.capacity() !=
            prepared_preservation_flags_capacity ||
        walker_indices.size() != prepared_walker_indices_capacity ||
        configuration_identities.size() !=
            prepared_configuration_identities_capacity ||
        staged_signs.size() != prepared_staged_signs_capacity ||
        staged_log_magnitudes.size() !=
            prepared_staged_log_magnitudes_capacity ||
        preservation_flags.size() !=
            prepared_preservation_flags_capacity ||
        vectorBytes(walker_indices,
                    "PsiFormer RECOMPUTE_VALUE walker staging") !=
            staging.walker_indices ||
        vectorBytes(configuration_identities,
                    "PsiFormer RECOMPUTE_VALUE configuration staging") !=
            staging.configuration_identities ||
        vectorBytes(staged_signs,
                    "PsiFormer RECOMPUTE_VALUE sign staging") !=
            staging.signs ||
        vectorBytes(staged_log_magnitudes,
                    "PsiFormer RECOMPUTE_VALUE log staging") !=
            staging.log_magnitudes ||
        vectorBytes(preservation_flags,
                    "PsiFormer RECOMPUTE_VALUE preservation staging") !=
            staging.preservation_flags)
      throw std::logic_error(
          "PsiFormer RECOMPUTE_VALUE staging extents differ from the prepared plan");
    if (selected_walkers > walker_indices.size() ||
        selected_walkers > configuration_identities.size() ||
        selected_walkers > staged_signs.size() ||
        selected_walkers > staged_log_magnitudes.size() ||
        selected_walkers > preservation_flags.size())
      throw std::length_error(
          "PsiFormer RECOMPUTE_VALUE staging prefix exceeds prepared capacity");
  }

  /// Validate the exact prepared prefixes used by an active-gradient query.
  void requireActiveGradientStaging(std::size_t live_walkers) const
  {
    if (!prepared_plan || !prepared_crowd_plan ||
        prepared_storage_fingerprint == 0 ||
        storageFingerprint() != prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer ACTIVE_GRADIENT staging lost its prepared allocation identity");

    const pf::ResourceStagingStorageRequirement& staging =
        prepared_crowd_plan->publication_storage;
    if (active_electrons.data() != prepared_active_electrons_data ||
        staged_gradients.data() != prepared_staged_gradients_data ||
        active_electrons.capacity() != prepared_active_electrons_capacity ||
        staged_gradients.capacity() != prepared_staged_gradients_capacity ||
        active_electrons.size() != prepared_active_electrons_capacity ||
        staged_gradients.size() != prepared_staged_gradients_capacity ||
        vectorBytes(active_electrons,
                    "PsiFormer ACTIVE_GRADIENT electron staging") !=
            staging.active_electrons ||
        vectorBytes(staged_gradients,
                    "PsiFormer ACTIVE_GRADIENT result staging") !=
            staging.gradients)
      throw std::logic_error(
          "PsiFormer ACTIVE_GRADIENT staging capacities differ from the prepared plan");
    if (live_walkers > active_electrons.size() ||
        live_walkers > staged_gradients.size())
      throw std::length_error(
          "PsiFormer ACTIVE_GRADIENT staging prefix exceeds prepared capacity");
  }

  /// Validate every prepared prefix used by a selected full-VGL proposal.
  void requireSelectedProposalStaging(std::size_t live_walkers) const
  {
    requireFullVGLStaging(live_walkers);
    const pf::ResourceStagingStorageRequirement& staging =
        prepared_crowd_plan->publication_storage;
    if (vectorBytes(batch_slots,
                    "PsiFormer selected batch-slot staging") !=
            staging.batch_slots ||
        vectorBytes(walker_indices,
                    "PsiFormer selected walker-index staging") !=
            staging.walker_indices ||
        vectorBytes(staged_log_ratios,
                    "PsiFormer selected log-ratio staging") != staging.ratios)
      throw std::logic_error(
          "PsiFormer selected staging capacities differ from the prepared plan");
    if (live_walkers > batch_slots.size() ||
        live_walkers > walker_indices.size() ||
        live_walkers > staged_log_ratios.size())
      throw std::length_error(
          "PsiFormer selected staging prefix exceeds prepared capacity");
  }

private:
  /// Allocate exactly one typed vector from a byte requirement.
  template<class T>
  static std::vector<T> makeExactVector(std::size_t bytes, const char* description)
  {
    if (bytes % sizeof(T) != 0)
      throw std::length_error(std::string(description) +
                              " is not divisible by its element width");
    std::vector<T> result(bytes / sizeof(T));
    if (result.capacity() * sizeof(T) != bytes)
      throw std::length_error(std::string(description) +
                              " exceeded its admitted vector capacity");
    return result;
  }

  /// Release one vector's allocation rather than retaining lazy high water.
  template<class T>
  static void releaseVector(std::vector<T>& values)
  { std::vector<T>().swap(values); }

  /// Return exact capacity bytes for one typed vector.
  template<class T>
  static std::size_t vectorBytes(const std::vector<T>& values,
                                 const char* description)
  { return pf::checkedStorageBytes<T>(values.capacity(), description); }

  /// Compare every retained direct-batch byte category.
  static bool sameDirectStorage(const pf::DirectBatchStorageRequirement& left,
                                const pf::DirectBatchStorageRequirement& right) noexcept
  {
    return left.dense_logical == right.dense_logical &&
        left.sparse_logical == right.sparse_logical &&
        left.logical_outputs == right.logical_outputs &&
        left.sparse_tile_positions == right.sparse_tile_positions &&
        left.value_tile == right.value_tile &&
        left.full_vgl_tile == right.full_vgl_tile &&
        left.active_gradient_tile == right.active_gradient_tile &&
        left.shared_spatial_arena == right.shared_spatial_arena &&
        left.replacement_transient == right.replacement_transient;
  }

  /// Compare every resource publication-staging byte category.
  static bool sameStagingStorage(const pf::ResourceStagingStorageRequirement& left,
                                 const pf::ResourceStagingStorageRequirement& right) noexcept
  {
    return left.walker_indices == right.walker_indices &&
        left.active_electrons == right.active_electrons &&
        left.configuration_identities == right.configuration_identities &&
        left.batch_slots == right.batch_slots && left.signs == right.signs &&
        left.log_magnitudes == right.log_magnitudes &&
        left.ratios == right.ratios && left.gradients == right.gradients &&
        left.preservation_flags == right.preservation_flags &&
        left.active_virtual_walkers == right.active_virtual_walkers &&
        left.virtual_reference_indices == right.virtual_reference_indices &&
        left.flattened_virtual_ratios == right.flattened_virtual_ratios &&
        left.virtual_reference_weights == right.virtual_reference_weights &&
        left.active_parameter_indices == right.active_parameter_indices &&
        left.selected_derivative_deltas == right.selected_derivative_deltas &&
        left.weighted_derivatives == right.weighted_derivatives;
  }

  /// Reconstruct the selected rank evidence without trusting mutable runtime state.
  void validateParticipantEvidence(
      const BatchExecutionParticipantPlan& participant_plan) const
  {
    const BatchExecutionPlan& plan = participant_plan.plan();
    const BatchMemoryParticipantEvidence& evidence = participant_plan.evidence();
    if (evidence.participant_id != participant_id ||
        evidence.participant_id.empty())
      throw std::invalid_argument(
          "PsiFormer crowd resource participant evidence has the wrong identity");
    if (plan.topology().serialized_walkers)
      throw std::invalid_argument(
          "PsiFormer crowd resource preparation does not admit serialized walkers");
    if (!plan.requirements().requires(BatchExecutionMode::FULL_VGL))
      throw std::invalid_argument(
          "PsiFormer crowd resource plan omits the component-owned FULL_VGL requirement");
    validatePlannedBackends(memory_policy_input, plan.requirements());

    const BatchExecutionPlanningContext selected_context{
        plan.requirements(), plan.topology(), plan.logicalMaximum(),
        plan.selectedCapacities(), plan.particleCount(), plan.activeParameterCount(),
        plan.parameterDerivativeWidth(), plan.targetCoordinate()};
    const BatchMemoryContribution selected =
        psiformer::estimatePsiFormerBatchMemory(memory_policy_input,
                                                selected_context);
    if (!capacitiesFitWithin(selected.logical_maximum, plan.logicalMaximum()) ||
        !(evidence.logical_maximum == selected.logical_maximum))
      throw std::invalid_argument(
          "PsiFormer crowd resource logical-maximum evidence is stale");
    if (selected.owner_multiplicity != 1 || evidence.owner_multiplicity != 1)
      throw std::invalid_argument(
          "PsiFormer crowd resource requires one exact rank-local owner");
    if (!(evidence.selected_per_owner == selected.per_owner))
      throw std::invalid_argument(
          "PsiFormer crowd resource selected storage evidence is stale");
    if (evidence.fully_accounted != selected.fully_accounted)
      throw std::invalid_argument(
          "PsiFormer crowd resource accounting evidence is stale");
    if (!selected.fully_accounted)
      throw std::invalid_argument(
          "PsiFormer crowd resource plan lacks complete accounting evidence");

    BatchExecutionPlanningContext minimum_context = selected_context;
    minimum_context.candidate_capacities = plan.minimumCapacities();
    const BatchMemoryContribution minimum =
        psiformer::estimatePsiFormerBatchMemory(memory_policy_input,
                                                minimum_context);
    if (!(evidence.fixed_minimum_per_owner == minimum.per_owner))
      throw std::invalid_argument(
          "PsiFormer crowd resource fixed-minimum storage evidence is stale");
  }

  /// Build stable per-crowd allocation records from one validated view.
  std::vector<psiformer::PsiFormerCrowdMemoryPlan> makeCrowdPlans(
      const BatchExecutionParticipantPlan& participant_plan) const
  {
    const BatchExecutionPlan& plan = participant_plan.plan();
    return psiformer::makePsiFormerCrowdMemoryPlans(
        memory_policy_input,
        {plan.requirements(), plan.topology(), plan.logicalMaximum(),
         plan.selectedCapacities(), plan.particleCount(), plan.activeParameterCount(),
         plan.parameterDerivativeWidth(), plan.targetCoordinate()});
  }

  /// Report every retained numeric byte, including legacy-only lazy staging.
  std::size_t retainedNumericBytes() const
  {
    std::size_t bytes = batch_workspace
        ? batch_workspace->actualStorage().executionBytes()
        : 0;
    auto add = [&bytes](std::size_t value, const char* description) {
      pf::addStorageBytes(bytes, value, description);
    };
    if (score_workspace)
      add(score_workspace->vectorStorageBytes(),
          "PsiFormer retained score bytes overflowed");
    if (kinetic_workspace)
      add(kinetic_workspace->vectorStorageBytes(),
          "PsiFormer retained kinetic bytes overflowed");
    add(actualStagingStorage().totalBytes(),
        "PsiFormer retained staging bytes overflowed");
    add(vectorBytes(total_log_gradient, "PsiFormer retained drift bytes overflowed"),
        "PsiFormer retained drift total overflowed");
    add(vectorBytes(virtual_offsets, "PsiFormer retained virtual-offset bytes overflowed"),
        "PsiFormer retained legacy bytes overflowed");
    add(vectorBytes(virtual_reference_signs,
                    "PsiFormer retained oracle-sign bytes overflowed"),
        "PsiFormer retained legacy bytes overflowed");
    add(vectorBytes(virtual_reference_logabs,
                    "PsiFormer retained oracle-log bytes overflowed"),
        "PsiFormer retained legacy bytes overflowed");
    return bytes;
  }

  /// Measure publication staging directly from typed vector capacities.
  pf::ResourceStagingStorageRequirement actualStagingStorage() const
  {
    pf::ResourceStagingStorageRequirement result;
    result.walker_indices = vectorBytes(
        walker_indices, "PsiFormer walker-index bytes overflowed");
    result.active_electrons = vectorBytes(
        active_electrons, "PsiFormer active-electron bytes overflowed");
    result.configuration_identities = vectorBytes(
        configuration_identities,
        "PsiFormer configuration-identity bytes overflowed");
    result.batch_slots = vectorBytes(
        batch_slots, "PsiFormer batch-slot bytes overflowed");
    result.signs = vectorBytes(staged_signs,
                               "PsiFormer sign bytes overflowed");
    result.log_magnitudes = vectorBytes(
        staged_log_magnitudes, "PsiFormer log-magnitude bytes overflowed");
    result.ratios = pf::checkedStorageSum(
        vectorBytes(staged_value_ratios,
                    "PsiFormer value-ratio bytes overflowed"),
        vectorBytes(staged_log_ratios,
                    "PsiFormer log-ratio bytes overflowed"),
        "PsiFormer ratio bytes overflowed");
    result.gradients = vectorBytes(
        staged_gradients, "PsiFormer gradient bytes overflowed");
    result.preservation_flags = vectorBytes(
        preservation_flags, "PsiFormer preservation bytes overflowed");
    result.active_virtual_walkers = vectorBytes(
        active_virtual_walkers,
        "PsiFormer active-virtual-walker bytes overflowed");
    result.virtual_reference_indices = vectorBytes(
        virtual_reference_indices,
        "PsiFormer virtual-reference-index bytes overflowed");
    result.flattened_virtual_ratios = vectorBytes(
        flat_virtual_ratios,
        "PsiFormer flattened-ratio bytes overflowed");
    result.virtual_reference_weights = vectorBytes(
        virtual_reference_weights,
        "PsiFormer virtual-reference-weight bytes overflowed");
    result.active_parameter_indices = vectorBytes(
        active_derivative_global_indices,
        "PsiFormer active-parameter-index bytes overflowed");
    result.selected_derivative_deltas = pf::checkedStorageSum(
        vectorBytes(virtual_score_contribution,
                    "PsiFormer score-delta bytes overflowed"),
        vectorBytes(kinetic_parameter_contribution,
                    "PsiFormer kinetic-delta bytes overflowed"),
        "PsiFormer selected-delta bytes overflowed");
    result.weighted_derivatives = vectorBytes(
        flat_virtual_weighted_derivatives,
        "PsiFormer weighted-derivative bytes overflowed");
    return result;
  }

  /// Categorize measured resource-owned storage independently of policy.
  BatchMemoryEstimate measureActualResourceStorage() const
  {
    BatchMemoryEstimate actual;
    const pf::DirectBatchStorageRequirement direct =
        batch_workspace->actualStorage();
    std::size_t logical = pf::checkedStorageSum(
        direct.dense_logical, direct.sparse_logical,
        "PsiFormer actual logical storage overflowed");
    logical = pf::checkedStorageSum(
        logical, direct.logical_outputs,
        "PsiFormer actual logical storage overflowed");
    std::size_t inner_tile = 0;
    for (const std::size_t bytes : {
             direct.sparse_tile_positions, direct.value_tile,
             direct.full_vgl_tile, direct.active_gradient_tile,
             direct.shared_spatial_arena})
      inner_tile = pf::checkedStorageSum(
          inner_tile, bytes,
          "PsiFormer actual inner-tile storage overflowed");

    actual.add(BatchMemoryCategory::LOGICAL_INPUT_OUTPUT, {logical, 0},
               "PsiFormer actual logical storage");
    actual.add(BatchMemoryCategory::INNER_TILE_SCRATCH, {inner_tile, 0},
               "PsiFormer actual inner-tile storage");
    actual.add(BatchMemoryCategory::REALLOCATION_TRANSIENT,
               {direct.replacementTransientBytes(), 0},
               "PsiFormer actual replacement transient");
    actual.add(BatchMemoryCategory::PUBLICATION_STAGING,
               {actualStagingStorage().totalBytes(), 0},
               "PsiFormer actual publication staging");
    if (score_workspace)
      actual.add(BatchMemoryCategory::SCORE_TAPE,
                 {score_workspace->vectorStorageBytes(), 0},
                 "PsiFormer actual score tape");

    std::size_t kinetic_bytes = vectorBytes(
        total_log_gradient, "PsiFormer actual drift bytes overflowed");
    if (kinetic_workspace)
      kinetic_bytes = pf::checkedStorageSum(
          kinetic_bytes, kinetic_workspace->vectorStorageBytes(),
          "PsiFormer actual kinetic storage overflowed");
    actual.add(BatchMemoryCategory::KINETIC_TAPE, {kinetic_bytes, 0},
               "PsiFormer actual kinetic tape");
    return actual;
  }

  /// Fill an unpublished candidate with every byte selected for one crowd.
  void materializePreparedStorage(
      BatchExecutionParticipantPlan selected,
      std::size_t crowd_index,
      psiformer::PsiFormerCrowdMemoryPlan crowd_plan)
  {
    batch_workspace->prepare(crowd_plan.direct_batch);
    if (!sameDirectStorage(batch_workspace->actualStorage(),
                           crowd_plan.direct_storage))
      throw std::length_error(
          "PsiFormer prepared crowd batch storage differs from policy");

    if (crowd_plan.score_workspace_bytes != 0)
    {
      score_workspace = model_state->direct_score_executor.makeWorkspace();
      if (score_workspace->vectorStorageBytes() !=
          crowd_plan.score_workspace_bytes)
        throw std::length_error(
            "PsiFormer prepared crowd score storage differs from policy");
    }
    if (crowd_plan.kinetic_workspace_bytes != 0)
    {
      kinetic_workspace = model_state->direct_kinetic_executor.makeWorkspace();
      if (kinetic_workspace->vectorStorageBytes() !=
          crowd_plan.kinetic_workspace_bytes)
        throw std::length_error(
            "PsiFormer prepared crowd kinetic storage differs from policy");
    }

    const pf::ResourceStagingStorageRequirement& staging =
        crowd_plan.publication_storage;
    walker_indices = makeExactVector<std::size_t>(
        staging.walker_indices, "PsiFormer walker-index staging");
    active_electrons = makeExactVector<std::size_t>(
        staging.active_electrons, "PsiFormer active-electron staging");
    configuration_identities = makeExactVector<std::uint64_t>(
        staging.configuration_identities,
        "PsiFormer configuration-identity staging");
    batch_slots = makeExactVector<std::size_t>(
        staging.batch_slots, "PsiFormer batch-slot staging");
    staged_signs = makeExactVector<double>(staging.signs,
                                           "PsiFormer sign staging");
    staged_log_magnitudes = makeExactVector<double>(
        staging.log_magnitudes, "PsiFormer log-magnitude staging");
    if (crowd_plan.publication_staging.full_vgl)
      staged_log_ratios = makeExactVector<LogValue>(
          staging.ratios, "PsiFormer log-ratio staging");
    else
      staged_value_ratios = makeExactVector<ValueType>(
          staging.ratios, "PsiFormer value-ratio staging");
    staged_gradients = makeExactVector<GradType>(
        staging.gradients, "PsiFormer gradient staging");
    preservation_flags = makeExactVector<unsigned char>(
        staging.preservation_flags, "PsiFormer preservation staging");
    active_virtual_walkers = makeExactVector<std::size_t>(
        staging.active_virtual_walkers,
        "PsiFormer active-virtual-walker staging");
    virtual_reference_indices = makeExactVector<std::size_t>(
        staging.virtual_reference_indices,
        "PsiFormer virtual-reference-index staging");
    flat_virtual_ratios = makeExactVector<ValueType>(
        staging.flattened_virtual_ratios,
        "PsiFormer flattened-ratio staging");
    virtual_reference_weights = makeExactVector<ValueType>(
        staging.virtual_reference_weights,
        "PsiFormer virtual-reference-weight staging");
    active_derivative_global_indices = makeExactVector<std::size_t>(
        staging.active_parameter_indices,
        "PsiFormer active-parameter-index staging");

    const bool first_delta = crowd_plan.publication_staging.weighted_ecp_score ||
        crowd_plan.publication_staging.score ||
        crowd_plan.publication_staging.kinetic;
    const std::size_t one_delta_bytes = first_delta
        ? pf::checkedStorageBytes<SelectedDerivativeDelta::value_type>(
              crowd_plan.publication_staging.active_parameters,
              "PsiFormer selected-delta staging overflowed")
        : 0;
    virtual_score_contribution =
        makeExactVector<SelectedDerivativeDelta::value_type>(
            one_delta_bytes, "PsiFormer score-delta staging");
    kinetic_parameter_contribution =
        makeExactVector<SelectedDerivativeDelta::value_type>(
            crowd_plan.publication_staging.kinetic ? one_delta_bytes : 0,
            "PsiFormer kinetic-delta staging");
    flat_virtual_weighted_derivatives = makeExactVector<ValueType>(
        staging.weighted_derivatives,
        "PsiFormer weighted-derivative staging");
    total_log_gradient = makeExactVector<double>(
        crowd_plan.total_log_gradient_bytes,
        "PsiFormer total-drift staging");

    if (!sameStagingStorage(actualStagingStorage(), staging))
      throw std::length_error(
          "PsiFormer prepared crowd publication storage differs from policy");
    if (!virtual_offsets.empty() || !virtual_reference_signs.empty() ||
        !virtual_reference_logabs.empty())
      throw std::logic_error(
          "PsiFormer planned crowd retained legacy ragged or oracle staging");

    BatchMemoryEstimate measured_storage = measureActualResourceStorage();
    if (!(measured_storage == crowd_plan.expected_resource_storage))
      throw std::length_error(
          "PsiFormer prepared crowd categorized storage differs from policy");

    prepared_configuration_identities_data = configuration_identities.data();
    prepared_staged_signs_data              = staged_signs.data();
    prepared_staged_log_magnitudes_data     = staged_log_magnitudes.data();
    prepared_walker_indices_data            = walker_indices.data();
    prepared_preservation_flags_data        = preservation_flags.data();
    prepared_active_electrons_data           = active_electrons.data();
    prepared_staged_gradients_data           = staged_gradients.data();
    prepared_configuration_identities_capacity =
        configuration_identities.capacity();
    prepared_staged_signs_capacity = staged_signs.capacity();
    prepared_staged_log_magnitudes_capacity =
        staged_log_magnitudes.capacity();
    prepared_walker_indices_capacity     = walker_indices.capacity();
    prepared_preservation_flags_capacity = preservation_flags.capacity();
    prepared_active_electrons_capacity   = active_electrons.capacity();
    prepared_staged_gradients_capacity   = staged_gradients.capacity();
    expected_plan                = selected;
    prepared_plan                = std::move(selected);
    prepared_crowd_index         = crowd_index;
    prepared_plan_fingerprint    = prepared_plan.plan().fingerprint();
    actual_resource_storage      = std::move(measured_storage);
    prepared_crowd_plan          = std::move(crowd_plan);
    prepared_storage_fingerprint = storageFingerprint();
    if (prepared_storage_fingerprint == 0)
      throw std::logic_error(
          "PsiFormer prepared crowd produced an invalid storage fingerprint");
  }

  /// Publish a complete candidate using only ownership transfers.
  void publishPreparedStorage(PsiFormerMultiWalkerResource&& candidate) noexcept
  {
    static_assert(std::is_nothrow_move_assignable_v<BatchExecutionParticipantPlan>);
    static_assert(
        std::is_nothrow_move_assignable_v<std::optional<psiformer::PsiFormerCrowdMemoryPlan>>);
    static_assert(std::is_nothrow_copy_assignable_v<BatchMemoryEstimate>);
    static_assert(std::is_nothrow_move_assignable_v<std::unique_ptr<pf::DirectBatchWorkspace>>);
    static_assert(std::is_nothrow_move_assignable_v<
                  std::vector<SelectedDerivativeDelta::value_type>>);

    expected_plan                  = std::move(candidate.expected_plan);
    prepared_plan                  = std::move(candidate.prepared_plan);
    prepared_crowd_plan            = std::move(candidate.prepared_crowd_plan);
    prepared_crowd_index           = candidate.prepared_crowd_index;
    prepared_plan_fingerprint      = candidate.prepared_plan_fingerprint;
    prepared_storage_fingerprint   = candidate.prepared_storage_fingerprint;
    actual_resource_storage        = candidate.actual_resource_storage;
    prepared_configuration_identities_data =
        candidate.prepared_configuration_identities_data;
    prepared_staged_signs_data = candidate.prepared_staged_signs_data;
    prepared_staged_log_magnitudes_data =
        candidate.prepared_staged_log_magnitudes_data;
    prepared_walker_indices_data = candidate.prepared_walker_indices_data;
    prepared_preservation_flags_data =
        candidate.prepared_preservation_flags_data;
    prepared_active_electrons_data =
        candidate.prepared_active_electrons_data;
    prepared_staged_gradients_data =
        candidate.prepared_staged_gradients_data;
    prepared_configuration_identities_capacity =
        candidate.prepared_configuration_identities_capacity;
    prepared_staged_signs_capacity =
        candidate.prepared_staged_signs_capacity;
    prepared_staged_log_magnitudes_capacity =
        candidate.prepared_staged_log_magnitudes_capacity;
    prepared_walker_indices_capacity =
        candidate.prepared_walker_indices_capacity;
    prepared_preservation_flags_capacity =
        candidate.prepared_preservation_flags_capacity;
    prepared_active_electrons_capacity =
        candidate.prepared_active_electrons_capacity;
    prepared_staged_gradients_capacity =
        candidate.prepared_staged_gradients_capacity;
    batch_workspace                = std::move(candidate.batch_workspace);
    score_workspace                = std::move(candidate.score_workspace);
    kinetic_workspace              = std::move(candidate.kinetic_workspace);
    total_log_gradient             = std::move(candidate.total_log_gradient);
    configuration_identities       = std::move(candidate.configuration_identities);
    batch_slots                    = std::move(candidate.batch_slots);
    staged_signs                   = std::move(candidate.staged_signs);
    staged_log_magnitudes          = std::move(candidate.staged_log_magnitudes);
    staged_value_ratios            = std::move(candidate.staged_value_ratios);
    staged_log_ratios              = std::move(candidate.staged_log_ratios);
    staged_gradients               = std::move(candidate.staged_gradients);
    preservation_flags             = std::move(candidate.preservation_flags);
    active_electrons               = std::move(candidate.active_electrons);
    virtual_offsets                = std::move(candidate.virtual_offsets);
    active_virtual_walkers         = std::move(candidate.active_virtual_walkers);
    virtual_reference_indices      = std::move(candidate.virtual_reference_indices);
    flat_virtual_ratios            = std::move(candidate.flat_virtual_ratios);
    virtual_reference_weights      = std::move(candidate.virtual_reference_weights);
    active_derivative_global_indices =
        std::move(candidate.active_derivative_global_indices);
    flat_virtual_weighted_derivatives =
        std::move(candidate.flat_virtual_weighted_derivatives);
    virtual_score_contribution =
        std::move(candidate.virtual_score_contribution);
    kinetic_parameter_contribution =
        std::move(candidate.kinetic_parameter_contribution);
    virtual_reference_signs =
        std::move(candidate.virtual_reference_signs);
    virtual_reference_logabs =
        std::move(candidate.virtual_reference_logabs);
    walker_indices = std::move(candidate.walker_indices);
  }

  /// Rebuild empty no-policy storage and clear every hard-plan marker.
  void clearToNoPolicy()
  {
    std::unique_ptr<pf::DirectBatchWorkspace> empty_batch =
        model_state->direct_batch_executor.makeWorkspace();
    batch_workspace = std::move(empty_batch);
    score_workspace.reset();
    kinetic_workspace.reset();
    releaseVector(total_log_gradient);
    releaseVector(configuration_identities);
    releaseVector(batch_slots);
    releaseVector(staged_signs);
    releaseVector(staged_log_magnitudes);
    releaseVector(staged_value_ratios);
    releaseVector(staged_log_ratios);
    releaseVector(staged_gradients);
    releaseVector(preservation_flags);
    releaseVector(active_electrons);
    releaseVector(virtual_offsets);
    releaseVector(active_virtual_walkers);
    releaseVector(virtual_reference_indices);
    releaseVector(flat_virtual_ratios);
    releaseVector(virtual_reference_weights);
    releaseVector(active_derivative_global_indices);
    releaseVector(flat_virtual_weighted_derivatives);
    releaseVector(virtual_score_contribution);
    releaseVector(kinetic_parameter_contribution);
    releaseVector(virtual_reference_signs);
    releaseVector(virtual_reference_logabs);
    releaseVector(walker_indices);
    expected_plan                 = {};
    prepared_plan                 = {};
    prepared_crowd_plan.reset();
    prepared_crowd_index         = 0;
    prepared_plan_fingerprint    = 0;
    prepared_storage_fingerprint = 0;
    prepared_configuration_identities_data = nullptr;
    prepared_staged_signs_data              = nullptr;
    prepared_staged_log_magnitudes_data     = nullptr;
    prepared_walker_indices_data            = nullptr;
    prepared_preservation_flags_data        = nullptr;
    prepared_active_electrons_data           = nullptr;
    prepared_staged_gradients_data           = nullptr;
    prepared_configuration_identities_capacity = 0;
    prepared_staged_signs_capacity             = 0;
    prepared_staged_log_magnitudes_capacity    = 0;
    prepared_walker_indices_capacity           = 0;
    prepared_preservation_flags_capacity       = 0;
    prepared_active_electrons_capacity         = 0;
    prepared_staged_gradients_capacity         = 0;
    actual_resource_storage      = {};
    weighted_reference_configurations   = 0;
    weighted_replacement_configurations = 0;
    weighted_active_parameters           = 0;
  }

  /// Hash all retained allocation identities and capacities.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uintptr_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    auto mix_vector = [&mix](const auto& values) {
      mix(reinterpret_cast<std::uintptr_t>(values.data()));
      mix(values.capacity());
    };
    if (batch_workspace)
    {
      mix(reinterpret_cast<std::uintptr_t>(batch_workspace.get()));
      mix(batch_workspace->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY));
      mix(batch_workspace->storageFingerprint(pf::DirectBatchMode::FULL_VGL));
      mix(batch_workspace->storageFingerprint(
          pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT));
    }
    if (score_workspace)
    {
      mix(reinterpret_cast<std::uintptr_t>(score_workspace.get()));
      mix(score_workspace->vectorStorageBytes());
    }
    if (kinetic_workspace)
    {
      mix(reinterpret_cast<std::uintptr_t>(kinetic_workspace.get()));
      mix(kinetic_workspace->storageFingerprint());
    }
    mix_vector(total_log_gradient);
    mix_vector(configuration_identities);
    mix_vector(batch_slots);
    mix_vector(staged_signs);
    mix_vector(staged_log_magnitudes);
    mix_vector(staged_value_ratios);
    mix_vector(staged_log_ratios);
    mix_vector(staged_gradients);
    mix_vector(preservation_flags);
    mix_vector(active_electrons);
    mix_vector(virtual_offsets);
    mix_vector(active_virtual_walkers);
    mix_vector(virtual_reference_indices);
    mix_vector(flat_virtual_ratios);
    mix_vector(virtual_reference_weights);
    mix_vector(active_derivative_global_indices);
    mix_vector(flat_virtual_weighted_derivatives);
    mix_vector(virtual_score_contribution);
    mix_vector(kinetic_parameter_contribution);
    mix_vector(virtual_reference_signs);
    mix_vector(virtual_reference_logabs);
    mix_vector(walker_indices);
    return hash;
  }
};

namespace
{
constexpr std::array<int, 3> PERSISTENCE_VERSION{1, 2, 0};

/// Hash construction provenance, architecture, layout, and physical-system metadata.
std::string modelFingerprint(const pf::PsiFormer& model,
                             const std::string& model_origin,
                             const std::string& initialization_profile,
                             std::uint64_t initialization_seed)
{
  std::uint64_t hash = 14695981039346656037ULL;
  auto mix_byte      = [&hash](std::uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  };
  auto mix_integer = [&mix_byte](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte)
      mix_byte(static_cast<std::uint8_t>(value >> (8 * byte)));
  };
  auto mix_string = [&mix_byte, &mix_integer](const std::string& value) {
    mix_integer(value.size());
    for (unsigned char character : value)
      mix_byte(character);
  };
  auto mix_double = [&mix_integer](double value) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    mix_integer(bits);
  };

  mix_string(model_origin);
  mix_string(initialization_profile);
  mix_integer(initialization_seed);
  mix_string(model.p.layout_fingerprint());
  mix_integer(model.cfg.nup);
  mix_integer(model.cfg.ndown);
  mix_integer(model.ndet);
  mix_integer(model.dim);
  mix_integer(model.heads);
  mix_integer(model.blocks);
  for (double coordinate : model.cfg.nuclei.x)
    mix_double(coordinate);
  for (double charge : model.cfg.charges.x)
    mix_double(charge);

  std::ostringstream fingerprint;
  fingerprint << std::hex << std::setfill('0') << std::setw(16) << hash;
  return fingerprint.str();
}

/// Build a stable, compact VariableSet name from a component and canonical flat index.
std::string makeParameterName(const std::string& component_name, std::size_t flat_index, std::size_t width)
{
  std::ostringstream name;
  name << component_name << "_pf_" << std::setfill('0') << std::setw(width) << flat_index;
  return name.str();
}

/// Convert canonical native indices to an explicitly sized HDF5 integer type.
std::vector<std::uint64_t> persistIndices(const std::vector<std::size_t>& indices)
{
  return std::vector<std::uint64_t>(indices.begin(), indices.end());
}

/// Read a rank-one HDF5 dataset after determining its on-disk extent.
template<class T>
std::vector<T> readVector(hdf_archive& input, const std::string& name)
{
  std::vector<int> shape;
  if (!input.getShape<T>(name, shape) || shape.size() != 1 || shape[0] < 0)
    throw std::runtime_error("PsiFormer VP dataset " + name + " is not a rank-one array");

  std::vector<T> values(shape[0]);
  input.read(values, name);
  return values;
}

/// Require exact immutable metadata equality before accepting a model payload.
template<class T>
void requireEqual(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& description)
{
  if (actual != expected)
    throw std::runtime_error("PsiFormer VP " + description + " does not match the configured model");
}

/// Pack one accepted or explicitly replaced configuration into a selected batch slot.
void packBatchConfiguration(pf::DirectBatchWorkspace& workspace,
                            std::size_t configuration,
                            const ParticleSet& particles,
                            int replaced_particle = -1,
                            const ParticleSet::PosType* replacement_position = nullptr)
{
  if (particles.getTotalNum() != static_cast<int>(workspace.electronCount()))
    throw std::invalid_argument("PsiFormer batch electron count differs from the exported model");
  if (replaced_particle < -1 ||
      (replaced_particle >= 0 && replaced_particle >= static_cast<int>(particles.getTotalNum())))
    throw std::out_of_range("PsiFormer batch replacement electron is out of range");

  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : particles.activeR(electron))
        : particles.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      workspace.setPosition(configuration, electron, dimension, position[dimension]);
  }
}

/// Pack accepted coordinates followed by descriptor-owned absolute replacements.
void packBatchConfiguration(
    pf::DirectBatchWorkspace& workspace,
    std::size_t configuration,
    const ParticleSet& particles,
    const MCMultiParticleMoves<CoordsType::POS>::Slice& moves)
{
  packBatchConfiguration(workspace, configuration, particles);
  for (std::size_t selected = 0; selected < moves.size(); ++selected)
  {
    const auto electron = static_cast<std::size_t>(moves.particleIndex(selected));
    const auto& position = moves.proposedPosition(selected);
    for (int dimension = 0; dimension < 3; ++dimension)
      workspace.setPosition(configuration, electron, dimension, position[dimension]);
  }
}

/// Convert the real sign/log-magnitude result to QMCPACK's complex-log convention.
WaveFunctionComponent::LogValue makeLogValue(double sign, double logabs) noexcept
{ return WaveFunctionComponent::LogValue(logabs, sign < 0 ? M_PI : 0.0); }

/// Validate the exact real-wavefunction sign/phase convention of an accepted value.
bool isCoherentAcceptedValue(double sign,
                             const WaveFunctionComponent::LogValue& log_value) noexcept
{
  return (sign == -1.0 || sign == 1.0) &&
      psiformer::determinant::isFiniteReal(std::real(log_value)) &&
      psiformer::determinant::isFiniteReal(std::imag(log_value)) &&
      log_value == makeLogValue(sign, std::real(log_value));
}

/// Checked non-owning byte range used by planned output-alias preflight.
struct CheckedMemoryRange
{
  std::uintptr_t begin = 0;
  std::uintptr_t end   = 0;
};

template<class T>
CheckedMemoryRange checkedMemoryRange(const T* data,
                                      std::size_t count,
                                      const char* overflow_message)
{
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::length_error(overflow_message);
  const std::size_t bytes = count * sizeof(T);
  if (bytes == 0)
    return {};
  if (data == nullptr)
    throw std::invalid_argument("PsiFormer planned output storage is null");
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    throw std::length_error(overflow_message);
  return {begin, begin + bytes};
}

bool memoryRangesOverlap(CheckedMemoryRange left,
                         CheckedMemoryRange right) noexcept
{
  return left.begin < right.end && right.begin < left.end;
}

bool sameMemoryRange(CheckedMemoryRange left, CheckedMemoryRange right) noexcept
{ return left.begin == right.begin && left.end == right.end; }

template<class T>
bool isFiniteWavefunctionValue(const T& value) noexcept
{
  return psiformer::determinant::isFiniteReal(
             static_cast<double>(std::real(value))) &&
      psiformer::determinant::isFiniteReal(
             static_cast<double>(std::imag(value)));
}

/// Form a finite real ratio without materializing either wavefunction amplitude.
WaveFunctionComponent::PsiValue makeRatio(double proposed_sign,
                                          double proposed_logabs,
                                          double reference_sign,
                                          double reference_logabs)
{
  const double ratio = (proposed_sign / reference_sign) * std::exp(proposed_logabs - reference_logabs);
  if (!psiformer::determinant::isFiniteReal(ratio))
    throw std::runtime_error("PsiFormer batch ratio is non-finite");
  return WaveFunctionComponent::PsiValue(ratio);
}

/// Require the unit electron masses assumed by the native kinetic-response formula.
void requireUnitElectronMasses(const ParticleSet& particles)
{
  if (particles.getTotalNum() == 0)
    return;

  const auto& masses = particles.get_mass_by_group();
  if (masses.size() < static_cast<std::size_t>(particles.groups()))
    throw std::invalid_argument(
        "PsiFormer kinetic parameter derivatives require initialized unit electron masses");

  constexpr double tolerance = 64 * std::numeric_limits<double>::epsilon();
  for (int group = 0; group < particles.groups(); ++group)
  {
    if (particles.groupsize(group) == 0)
      continue;
    const double mass = masses[group];
    if (!psiformer::determinant::isFiniteReal(mass) || std::abs(mass - 1.0) > tolerance)
      throw std::invalid_argument(
          "PsiFormer kinetic parameter derivatives currently require unit electron masses");
  }
}

/// Reject complex total drifts until kinetic reverse kernels support complex arithmetic.
void requireRealTotalWavefunctionDrift(const ParticleSet& particles)
{
#ifdef QMC_COMPLEX
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      if (std::imag(particles.G[electron][dimension]) != 0.0)
        throw std::invalid_argument(
            "PsiFormer kinetic parameter derivatives require a real total wavefunction drift; "
            "genuinely complex kinetic response is not implemented");
#else
  static_cast<void>(particles);
#endif
}
} // namespace

// Load an exported model before entering the common optimizer-registration path.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::string parameters,
                         std::string configuration,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export)
    : PsiFormerWF(std::move(name),
                  std::make_shared<PsiFormerSharedState>(parameters, configuration),
                  enable_optimization, std::move(selected_flat_indices), optimize_all,
                  std::move(optimized_parameter_export))
{}

// Construct a native model directly from initialized parameters and QMCPACK system metadata.
PsiFormerWF::PsiFormerWF(std::string name,
                         psiformer::InitializedPsiFormerParameters initialized_parameters,
                         const ParticleSet& electrons,
                         const ParticleSet& ions,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export)
    : PsiFormerWF(std::move(name),
                  std::make_shared<PsiFormerSharedState>(std::move(initialized_parameters), electrons, ions),
                  enable_optimization, std::move(selected_flat_indices), optimize_all,
                  std::move(optimized_parameter_export))
{
  bound_particle_set_ = &electrons;
}

// Register selected or complete parameters after either model-construction route.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::shared_ptr<PsiFormerSharedState> model_state,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export)
    : WaveFunctionComponent(name),
      OptimizableObject(name),
      model_state_(std::move(model_state)),
      structured_parameter_schema_(
          makeStructuredParameterSchema(name, model_state_->model.p)),
      optimized_parameter_export_(std::move(optimized_parameter_export))
{
  if (!enable_optimization && !selected_flat_indices.empty())
    throw std::invalid_argument("PsiFormer optimize_indices requires optimize=yes");
  if (optimize_all && !enable_optimization)
    throw std::invalid_argument("PsiFormer optimize_scope=all requires optimize=yes");
  if (optimize_all && !selected_flat_indices.empty())
    throw std::invalid_argument("PsiFormer optimize_scope=all cannot be combined with optimize_indices");

  pf::Parameters& parameters_ref = model_state_->model.p;
  if (parameters_ref.size() == 0)
    throw std::invalid_argument("PsiFormer parameter export contains no scalar values");
  if (optimize_all)
  {
    selected_flat_indices.resize(parameters_ref.size());
    std::iota(selected_flat_indices.begin(), selected_flat_indices.end(), std::size_t{0});
  }
  if (enable_optimization && selected_flat_indices.empty())
    throw std::invalid_argument("PsiFormer selected-parameter optimization requires at least one flat index");
  if (!optimized_parameter_export_.empty() && !enable_optimization)
    throw std::invalid_argument("PsiFormer export_parameters requires optimize=yes");

  if (!optimize_all)
  {
    std::sort(selected_flat_indices.begin(), selected_flat_indices.end());
    if (std::adjacent_find(selected_flat_indices.begin(), selected_flat_indices.end()) !=
        selected_flat_indices.end())
      throw std::invalid_argument("PsiFormer optimize_indices contains a duplicate flat index");
  }

  observed_parameter_version_ = parameters_ref.version();
  accepted_parameter_version_ = observed_parameter_version_;
  resizeAcceptedSpatialStorage(model_state_->model.ne);
  const std::size_t name_width = std::to_string(parameters_ref.size() - 1).size();
  std::vector<OptVariables::pair_type> selected_parameters;
  selected_parameters.reserve(selected_flat_indices.size());
  for (std::size_t flat_index : selected_flat_indices)
  {
    if (!optimize_all)
      parameters_ref.layout_for_flat_index(flat_index);
    selected_parameters.emplace_back(makeParameterName(WaveFunctionComponent::getName(), flat_index, name_width),
                                     parameters_ref.flat_values()[flat_index]);
  }
  OptVariables variables;
  variables.insertBulk(std::move(selected_parameters), true, optimize::OTHER_P);
  optimization_metadata_ = std::make_shared<PsiFormerOptimizationMetadata>(
      enable_optimization, optimize_all, std::move(selected_flat_indices), std::move(variables));

  std::set<std::string> selected_tensors;
  for (std::size_t flat_index : optimization_metadata_->selected_flat_indices)
  {
    const pf::Layout& layout = parameters_ref.layout_for_flat_index(flat_index);
    selected_tensors.insert(layout.module + "/" + layout.name);
  }
  app_log() << "  PsiFormer " << WaveFunctionComponent::getName() << ": model="
            << modelFingerprint(model_state_->model, model_state_->model_origin,
                                model_state_->initialization_profile,
                                model_state_->initialization_seed)
            << ", origin=" << model_state_->model_origin;
  if (!model_state_->initialization_profile.empty())
    app_log() << ", initialization=" << model_state_->initialization_profile
              << ", initialization_seed=" << model_state_->initialization_seed;
  app_log() << ", parameters=" << parameters_ref.size()
            << ", active=" << optimization_metadata_->selected_flat_indices.size()
            << ", tensors=" << selected_tensors.size()
            << ", parameter_version=" << parameters_ref.version() << ", derivative_mode="
            << (enable_optimization ? "score+kinetic+nonlocal-ratio" : "fixed")
            << ", value_backend=" << directBackendModeName(model_state_->direct_value_mode)
            << ", spatial_backend=" << directBackendModeName(model_state_->direct_spatial_mode)
            << ", score_backend=" << directBackendModeName(model_state_->direct_score_mode)
            << ", kinetic_backend=" << directBackendModeName(model_state_->direct_kinetic_mode)
            << std::endl;
  if (!selected_tensors.empty())
  {
    app_log() << "    selected tensors:";
    std::size_t reported = 0;
    for (const std::string& tensor : selected_tensors)
    {
      if (reported++ == 6)
      {
        app_log() << " ...";
        break;
      }
      app_log() << " " << tensor;
    }
    app_log() << std::endl;
  }
  if (!optimized_parameter_export_.empty())
    app_log() << "    DeepQMC parameter export destination: " << optimized_parameter_export_ << std::endl;
}

// Copy clone-local accepted state while deliberately discarding an in-flight proposal.
PsiFormerWF::PsiFormerWF(const PsiFormerWF& other)
    : WaveFunctionComponent(other),
      OptimizableObject(other),
      model_state_(other.model_state_),
      optimization_metadata_(other.optimization_metadata_),
      structured_parameter_schema_(other.structured_parameter_schema_),
      batch_execution_plan_(other.batch_execution_plan_),
      complete_batch_memory_accounting_for_testing_(
          other.complete_batch_memory_accounting_for_testing_),
      system_kind_(other.system_kind_),
      bound_particle_set_(other.bound_particle_set_),
      optimized_parameter_export_(other.optimized_parameter_export_),
      observed_parameter_version_(other.observed_parameter_version_),
      restore_validation_pending_(other.restore_validation_pending_),
      accepted_value_valid_(other.accepted_value_valid_),
      accepted_gradient_(other.accepted_gradient_),
      accepted_laplacian_(other.accepted_laplacian_),
      accepted_configuration_identity_(other.accepted_configuration_identity_),
      accepted_parameter_version_(other.accepted_parameter_version_),
      accepted_state_requirement_(other.accepted_state_requirement_),
      current_sign_(other.current_sign_)
{
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  clearProposalState();
}

PsiFormerWF::~PsiFormerWF() = default;

// Translate immutable component facts into one allocation-free policy input.
psiformer::PsiFormerMemoryPolicyInput PsiFormerWF::makeBatchMemoryPolicyInput() const
{
  const psiformer::ModelShape& model_shape = model_state_->execution_plan.modelShape();
  const auto& value_layout                = model_state_->direct_value_executor.layout();
  if (!value_layout)
    throw std::logic_error("PsiFormer direct value layout is not initialized");

  psiformer::PsiFormerMemoryPolicyInput input;
  input.storage_shape.electrons        = model_shape.electrons();
  input.storage_shape.nuclei           = model_shape.nuclei;
  input.storage_shape.determinants     = model_shape.determinants;
  input.storage_shape.feature_width    = model_shape.feature_dimension;
  input.storage_shape.attention_heads  = model_shape.attention_heads;
  input.storage_shape.input_width      = value_layout->inputWidth();
  input.storage_shape.attention_blocks = model_shape.attention_blocks;
  input.storage_shape.parameter_count  = model_state_->execution_plan.parameterCount();
  input.type_sizes = psiformer::makePsiFormerMemoryTypeSizes<
      ValueType, LogValue, GradType, SelectedDerivativeDelta::value_type>();
  input.backends.value   = memoryPolicyBackend(model_state_->direct_value_mode);
  input.backends.spatial = memoryPolicyBackend(model_state_->direct_spatial_mode);
  input.backends.score   = memoryPolicyBackend(model_state_->direct_score_mode);
  input.backends.kinetic = memoryPolicyBackend(model_state_->direct_kinetic_mode);

  // The selected flat-index vector is immutable after construction.  It is the
  // component-local derivative width, whereas the planning context carries the
  // aggregate active width across every wavefunction component.
  input.active_parameter_count = optimization_metadata_->selected_flat_indices.size();
  input.scalar_value_logical_maximum =
      checkedBatchMemoryAdd(model_shape.electrons(), 1,
                            "PsiFormer all-to-one logical envelope");
  input.flattened_ecp = true;

  // Clone and crowd resources now have exact preparation boundaries, but
  // aggregate TrialWaveFunction scratch and every planned runtime path are not
  // yet allocation-free. Keep the production gate closed until those owners
  // and guards land; the friend-only override exercises this completed owner
  // boundary without widening the public API.
  input.accounting_claims = complete_batch_memory_accounting_for_testing_
      ? psiformer::PsiFormerMemoryAccountingClaims::complete()
      : psiformer::PsiFormerMemoryAccountingClaims{};
  return input;
}

// Full VGL initialization is always reachable; phase-specific drivers add all other modes.
void PsiFormerWF::contributeBatchExecutionRequirements(
    BatchExecutionRequirements& requirements) const
{
  requirements.require(BatchExecutionMode::FULL_VGL);
}

// Discover this component's shape-derived maxima before candidate selection.
BatchTileCapacities PsiFormerWF::batchExecutionLogicalMaximum(
    const BatchExecutionWorkloadContext& context) const
{
  return psiformer::psiFormerBatchLogicalMaximum(
      makeBatchMemoryPolicyInput(), context);
}

// Delegate exact mixed clone/crowd ownership accounting to the pure policy.
BatchMemoryContribution PsiFormerWF::estimateBatchExecutionMemory(
    const BatchExecutionPlanningContext& context) const
{
  return psiformer::estimatePsiFormerBatchMemory(
      makeBatchMemoryPolicyInput(), context);
}

// Check the retained participant identity without rebuilding plan evidence.
bool PsiFormerWF::hasBatchExecutionPlanBinding(
    const BatchExecutionParticipantPlan& plan) const noexcept
{
  return batch_execution_plan_.sameBinding(plan);
}

// Check exact clone-local preparation provenance without touching storage.
bool PsiFormerWF::hasPreparedBatchExecutionClone(
    const BatchExecutionParticipantPlan& plan) const noexcept
{
  return prepared_clone_batch_execution_plan_.sameBinding(plan) &&
      !accepted_gradient_.isAttached() && !accepted_laplacian_.isAttached() &&
      !proposed_gradient_.isAttached() && !proposed_laplacian_.isAttached() &&
      accepted_gradient_.data() == prepared_accepted_gradient_data_ &&
      accepted_laplacian_.data() == prepared_accepted_laplacian_data_ &&
      proposed_gradient_.data() == prepared_proposed_gradient_data_ &&
      proposed_laplacian_.data() == prepared_proposed_laplacian_data_ &&
      accepted_gradient_.size() == prepared_accepted_gradient_capacity_ &&
      accepted_gradient_.capacity() == prepared_accepted_gradient_capacity_ &&
      accepted_laplacian_.size() == prepared_accepted_laplacian_capacity_ &&
      accepted_laplacian_.capacity() == prepared_accepted_laplacian_capacity_ &&
      proposed_gradient_.size() == prepared_proposed_gradient_capacity_ &&
      proposed_gradient_.capacity() == prepared_proposed_gradient_capacity_ &&
      proposed_laplacian_.size() == prepared_proposed_laplacian_capacity_ &&
      proposed_laplacian_.capacity() == prepared_proposed_laplacian_capacity_;
}

// Check a prospective participant view completely before noexcept publication.
void PsiFormerWF::validateBatchExecutionPlanBinding(
    const BatchExecutionParticipantPlan& participant_plan) const
{
  const bool changes_binding = !batch_execution_plan_.sameBinding(participant_plan);
  if (changes_binding && has_proposal_)
    throw std::logic_error(
        "Cannot change a PsiFormer batch execution plan while a proposal is pending");
  if (participant_plan && batch_execution_plan_ && changes_binding)
    throw std::logic_error(
        "PsiFormer nonempty batch replanning requires an explicit null-plan clear");

  if (!participant_plan)
  {
    if (mw_resource_handle_ && batch_execution_plan_)
      throw std::logic_error(
          "Cannot clear a PsiFormer batch execution plan while its resource is acquired");
    return;
  }

  if (mw_resource_handle_ &&
      !batch_execution_plan_.sameBinding(participant_plan))
    throw std::logic_error(
        "Cannot rebind a PsiFormer batch execution plan while its resource is acquired");

  const BatchExecutionPlan& plan = participant_plan.plan();
  const BatchMemoryParticipantEvidence& evidence = participant_plan.evidence();
  if (evidence.participant_id.empty())
    throw std::invalid_argument(
        "PsiFormer batch execution participant evidence has an empty identity");

  BatchExecutionRequirements component_requirements;
  contributeBatchExecutionRequirements(component_requirements);
  if ((plan.requirements().mask() & component_requirements.mask()) !=
      component_requirements.mask())
    throw std::invalid_argument(
        "PsiFormer batch execution plan omits a component-owned requirement");

  const psiformer::PsiFormerMemoryPolicyInput input = makeBatchMemoryPolicyInput();
  validatePlannedBackends(input, plan.requirements());

  BatchExecutionPlanningContext selected_context{
      plan.requirements(), plan.topology(), plan.logicalMaximum(),
      plan.selectedCapacities(), plan.particleCount(), plan.activeParameterCount(),
      plan.parameterDerivativeWidth(), plan.targetCoordinate()};
  const BatchMemoryContribution selected =
      psiformer::estimatePsiFormerBatchMemory(input, selected_context);

  if (!capacitiesFitWithin(selected.logical_maximum, plan.logicalMaximum()))
    throw std::invalid_argument(
        "PsiFormer logical maximum exceeds the aggregate batch plan envelope");
  if (!(evidence.logical_maximum == selected.logical_maximum))
    throw std::invalid_argument(
        "PsiFormer batch participant logical-maximum evidence is stale");
  if (selected.owner_multiplicity != 1 ||
      evidence.owner_multiplicity != selected.owner_multiplicity)
    throw std::invalid_argument(
        "PsiFormer batch participant must have one exact rank-local owner");
  if (!(evidence.selected_per_owner == selected.per_owner))
    throw std::invalid_argument(
        "PsiFormer selected batch-memory evidence does not match the current model and topology");

  BatchExecutionPlanningContext minimum_context = selected_context;
  minimum_context.candidate_capacities = plan.minimumCapacities();
  const BatchMemoryContribution minimum =
      psiformer::estimatePsiFormerBatchMemory(input, minimum_context);
  if (!(evidence.fixed_minimum_per_owner == minimum.per_owner))
    throw std::invalid_argument(
        "PsiFormer fixed-minimum batch-memory evidence does not match the current model and topology");
  if (evidence.fully_accounted != selected.fully_accounted)
    throw std::invalid_argument(
        "PsiFormer batch participant accounting evidence is stale");

  // Subsequent stages will turn on claims only after preparing every reachable
  // owner and guarding each live call.  Until then, a nonempty binding fails closed.
  if (!selected.fully_accounted)
    throw std::logic_error(
        "PsiFormer planned execution is not yet fully storage-accounted");
}

// Publish only views that the aggregate two-phase binding boundary already validated.
void PsiFormerWF::bindBatchExecutionPlan(
    BatchExecutionParticipantPlan participant_plan) noexcept
{
  if (!participant_plan)
  {
    // A null binding returns the component to its historical lazy-allocation
    // contract.  In particular, discard the bounded batch workspace so a
    // later unplanned scalar request is not constrained by stale capacities.
    batch_execution_plan_ = {};
    prepared_clone_batch_execution_plan_ = {};
    prepared_accepted_gradient_data_ = nullptr;
    prepared_accepted_laplacian_data_ = nullptr;
    prepared_proposed_gradient_data_ = nullptr;
    prepared_proposed_laplacian_data_ = nullptr;
    prepared_accepted_gradient_capacity_ = 0;
    prepared_accepted_laplacian_capacity_ = 0;
    prepared_proposed_gradient_capacity_ = 0;
    prepared_proposed_laplacian_capacity_ = 0;
    direct_value_workspace_.reset();
    direct_score_workspace_.reset();
    direct_kinetic_workspace_.reset();
    direct_full_spatial_workspace_.reset();
    direct_active_spatial_workspace_.reset();
    direct_batch_workspace_.reset();
    std::vector<double>().swap(direct_total_log_gradient_);
    std::vector<ValueType>().swap(scalar_value_publication_);
    return;
  }

  if (!batch_execution_plan_)
  {
    // The selected estimate describes a canonical owner, not legacy lazy high
    // water.  Drop recomputable scratch before publishing the first hard plan;
    // subsequent exact preparation can then allocate without overlap.
    prepared_clone_batch_execution_plan_ = {};
    prepared_accepted_gradient_data_ = nullptr;
    prepared_accepted_laplacian_data_ = nullptr;
    prepared_proposed_gradient_data_ = nullptr;
    prepared_proposed_laplacian_data_ = nullptr;
    prepared_accepted_gradient_capacity_ = 0;
    prepared_accepted_laplacian_capacity_ = 0;
    prepared_proposed_gradient_capacity_ = 0;
    prepared_proposed_laplacian_capacity_ = 0;
    direct_value_workspace_.reset();
    direct_score_workspace_.reset();
    direct_kinetic_workspace_.reset();
    direct_full_spatial_workspace_.reset();
    direct_active_spatial_workspace_.reset();
    direct_batch_workspace_.reset();
    std::vector<double>().swap(direct_total_log_gradient_);
    std::vector<ValueType>().swap(scalar_value_publication_);
  }
  batch_execution_plan_ = std::move(participant_plan);
}

// Materialize every clone-owned byte admitted by one already-bound participant view.
void PsiFormerWF::prepareBatchExecutionClone(
    const BatchExecutionParticipantPlan& participant_plan)
{
  if (!participant_plan ||
      !batch_execution_plan_.sameBinding(participant_plan))
    throw std::logic_error(
        "PsiFormer clone preparation received the wrong batch execution plan");
  if (mw_resource_handle_)
    throw std::logic_error(
        "Cannot prepare PsiFormer clone storage while a crowd resource is acquired");
  if (has_proposal_)
    throw std::logic_error(
        "Cannot prepare PsiFormer clone storage while a proposal is pending");

  const BatchExecutionPlan& plan = participant_plan.plan();
  const psiformer::PsiFormerMemoryPolicyInput input =
      makeBatchMemoryPolicyInput();
  const std::size_t electron_count = input.storage_shape.electrons;
  const auto exact_storage_bytes = [electron_count](const auto& storage,
                                                     std::size_t element_bytes,
                                                     const char* description) {
    if (storage.isAttached())
      throw std::logic_error(
          "PsiFormer prepared clone spatial storage must be owned");
    if (storage.size() != electron_count ||
        storage.capacity() != electron_count)
      throw std::length_error(description);
    return checkedBatchMemoryMultiply(storage.capacity(), element_bytes,
                                      description);
  };
  const auto validate_exact_storage = [&]() {
    const pf::CloneStateStorageRequirement clone_state =
        pf::cloneStateStorageRequirement(
            electron_count, input.type_sizes.value_type,
            input.type_sizes.gradient_type);
    std::size_t actual_clone_bytes = 0;
    for (const std::size_t bytes : {
             exact_storage_bytes(
                 accepted_gradient_, sizeof(GradType),
                 "PsiFormer accepted gradient does not have its exact admitted capacity"),
             exact_storage_bytes(
                 proposed_gradient_, sizeof(GradType),
                 "PsiFormer proposed gradient does not have its exact admitted capacity"),
             exact_storage_bytes(
                 accepted_laplacian_, sizeof(ValueType),
                 "PsiFormer accepted Laplacian does not have its exact admitted capacity"),
             exact_storage_bytes(
                 proposed_laplacian_, sizeof(ValueType),
                 "PsiFormer proposed Laplacian does not have its exact admitted capacity")})
      actual_clone_bytes = checkedBatchMemoryAdd(
          actual_clone_bytes, bytes,
          "PsiFormer prepared clone-state bytes");
    if (actual_clone_bytes != clone_state.totalBytes())
      throw std::length_error(
          "PsiFormer prepared clone state does not match its admitted storage");
  };

  if (prepared_clone_batch_execution_plan_)
  {
    if (!prepared_clone_batch_execution_plan_.sameBinding(participant_plan))
      throw std::logic_error(
          "Cannot replace a prepared PsiFormer clone batch execution plan");
    validate_exact_storage();
    if (!hasPreparedBatchExecutionClone(participant_plan))
      throw std::logic_error(
          "PsiFormer prepared clone allocation identity changed");
    return;
  }

  if (plan.topology().serialized_walkers)
    throw std::invalid_argument(
        "PsiFormer clone preparation does not admit serialized-walker execution");

  BatchExecutionRequirements component_requirements;
  contributeBatchExecutionRequirements(component_requirements);
  if ((plan.requirements().mask() & component_requirements.mask()) !=
      component_requirements.mask())
    throw std::invalid_argument(
        "PsiFormer clone preparation plan omits a component-owned requirement");

  validatePlannedBackends(input, plan.requirements());
  const pf::DirectBatchCapacityPlan scalar_plan = psiformer::makePsiFormerScalarValueCapacityPlan(
      input, plan.requirements(), plan.selectedCapacities());
  const bool scalar_value = plan.requirements().requires(
      BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);

  // Construct scalar scratch off to the side.  A failed allocation therefore
  // preserves both the old workspace and the unpublished preparation marker.
  std::unique_ptr<pf::DirectBatchWorkspace> prepared_batch_workspace;
  std::vector<ValueType> prepared_scalar_publication;
  if (scalar_value)
  {
    prepared_batch_workspace = model_state_->direct_batch_executor.makeWorkspace();
    prepared_batch_workspace->prepare(scalar_plan);
    const pf::DirectBatchStorageRequirement expected =
        pf::directBatchStorageRequirement(input.storage_shape, scalar_plan);
    if (prepared_batch_workspace->vectorStorageBytes() !=
        expected.executionBytes())
      throw std::length_error(
          "PsiFormer prepared scalar workspace does not match its admitted storage");

    prepared_scalar_publication.resize(input.scalar_value_logical_maximum);
    if (prepared_scalar_publication.capacity() !=
        input.scalar_value_logical_maximum)
      throw std::length_error(
          "PsiFormer scalar publication storage exceeds its admitted capacity");
  }
  if (fail_clone_preparation_before_publish_for_testing_)
    throw std::bad_alloc();

  // Accepted nonempty state may represent a physical configuration and cannot
  // be canonicalized by discarding it.  Empty accepted storage and all
  // proposal storage are safe to materialize because proposals are absent.
  const auto validate_accepted_storage = [electron_count](const auto& storage) {
    if (storage.isAttached())
      throw std::logic_error(
          "PsiFormer clone preparation requires owned accepted spatial storage");
    if (storage.size() != 0 &&
        (storage.size() != electron_count ||
         storage.capacity() != electron_count))
      throw std::length_error(
          "PsiFormer accepted spatial storage retains a noncanonical capacity");
  };
  validate_accepted_storage(accepted_gradient_);
  validate_accepted_storage(accepted_laplacian_);
  if (proposed_gradient_.isAttached() || proposed_laplacian_.isAttached())
    throw std::logic_error(
        "PsiFormer clone preparation requires owned proposed spatial storage");

  const bool materialize_accepted = accepted_gradient_.size() == 0 ||
      accepted_laplacian_.size() == 0;
  if (accepted_gradient_.size() == 0)
  {
    if (accepted_gradient_.capacity() != electron_count)
      accepted_gradient_.free();
    accepted_gradient_.resize(electron_count);
  }
  if (accepted_laplacian_.size() == 0)
  {
    if (accepted_laplacian_.capacity() != electron_count)
      accepted_laplacian_.free();
    accepted_laplacian_.resize(electron_count);
  }
  if (materialize_accepted)
  {
    accepted_gradient_          = ValueType(0);
    accepted_laplacian_         = ValueType(0);
    accepted_value_valid_       = false;
    accepted_state_requirement_ = AcceptedStateRequirement::INVALID;
  }
  if (proposed_gradient_.size() != electron_count ||
      proposed_gradient_.capacity() != electron_count)
  {
    proposed_gradient_.free();
    proposed_gradient_.resize(electron_count);
  }
  if (proposed_laplacian_.size() != electron_count ||
      proposed_laplacian_.capacity() != electron_count)
  {
    proposed_laplacian_.free();
    proposed_laplacian_.resize(electron_count);
  }
  validate_exact_storage();

  // No other clone-local evaluator tape is admitted by the first hard-plan
  // implementation.  Publish the fully staged scalar owner with no throwing
  // allocation after this point.
  direct_value_workspace_.reset();
  direct_score_workspace_.reset();
  direct_kinetic_workspace_.reset();
  direct_full_spatial_workspace_.reset();
  direct_active_spatial_workspace_.reset();
  std::vector<double>().swap(direct_total_log_gradient_);
  direct_batch_workspace_ = std::move(prepared_batch_workspace);
  scalar_value_publication_ = std::move(prepared_scalar_publication);
  prepared_accepted_gradient_data_  = accepted_gradient_.data();
  prepared_accepted_laplacian_data_ = accepted_laplacian_.data();
  prepared_proposed_gradient_data_  = proposed_gradient_.data();
  prepared_proposed_laplacian_data_ = proposed_laplacian_.data();
  prepared_accepted_gradient_capacity_ = accepted_gradient_.capacity();
  prepared_accepted_laplacian_capacity_ = accepted_laplacian_.capacity();
  prepared_proposed_gradient_capacity_ = proposed_gradient_.capacity();
  prepared_proposed_laplacian_capacity_ = proposed_laplacian_.capacity();
  static_assert(std::is_nothrow_copy_assignable_v<BatchExecutionParticipantPlan>);
  // The plan view is the validity marker and is deliberately published last.
  prepared_clone_batch_execution_plan_ = participant_plan;
}

// Report the immutable optimization mode shared by the complete clone family.
bool PsiFormerWF::isOptimizable() const
{
  return optimization_metadata_->enabled;
}

const wftrain::StructuredParameterSchema& PsiFormerWF::parameterSchema() const noexcept
{
  return *structured_parameter_schema_;
}

wftrain::StructuredParameterSnapshot PsiFormerWF::snapshotParameters() const
{
  std::shared_lock state_lock(model_state_->mutex);
  const pf::Parameters& parameters = model_state_->model.p;
  return {structured_parameter_schema_->fingerprint(), parameters.version(),
          parameters.flat_values()};
}

std::size_t PsiFormerWF::publishParameters(
    const wftrain::StructuredParameterSnapshot& candidate,
    std::size_t expected_version)
{
  requireNoPlannedSelectedProposalMutation("structured parameter publication");
  if (candidate.schema_fingerprint != structured_parameter_schema_->fingerprint())
    throw std::invalid_argument("PsiFormer structured update has an incompatible schema fingerprint");
  if (candidate.values.size() != structured_parameter_schema_->parameterCount())
    throw std::invalid_argument("PsiFormer structured update has the wrong parameter count");
  if (candidate.version != expected_version)
    throw std::invalid_argument("PsiFormer structured update candidate has the wrong source version");

  std::unique_lock state_lock(model_state_->mutex);
  requireNoPlannedSelectedProposalMutation("structured parameter publication");
  pf::Parameters& parameters = model_state_->model.p;
  if (parameters.version() != expected_version)
    throw std::runtime_error("PsiFormer structured update rejected a stale parameter version");

  // set_flat_values validates every value before mutating the native leaves.
  parameters.set_flat_values(candidate.values);
  const std::size_t committed_version = parameters.version();
  invalidateParameterCaches(committed_version);
  return committed_version;
}

// Add one empty resource template after the component's selected participant
// view is known.  Crowd copies retain this provenance but no numeric scratch.
void PsiFormerWF::createResource(ResourceCollection& collection) const
{
  const std::string participant_id = batch_execution_plan_
      ? batch_execution_plan_.evidence().participant_id
      : std::string{};
  collection.addResource(std::make_unique<PsiFormerMultiWalkerResource>(
      model_state_, makeBatchMemoryPolicyInput(), participant_id,
      batch_execution_plan_));
}

// Lend the crowd workspace to the leader after validating homogeneous clone identity.
void PsiFormerWF::acquireResource(
    ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto* leader_ptr = dynamic_cast<PsiFormerWF*>(&wfc_list.getLeader());
  if (leader_ptr == nullptr)
    throw std::invalid_argument(
        "PsiFormer multiwalker resource acquisition received a foreign leader type");
  auto& leader = *leader_ptr;
  if (this != &leader)
    throw std::logic_error("PsiFormer multiwalker resource acquisition must be invoked on the leader");
  if (leader.mw_resource_handle_)
    throw std::logic_error("PsiFormer multiwalker resource is already acquired");
  if (!wfc_list.empty() && &wfc_list[0] != &leader)
    throw std::invalid_argument(
        "PsiFormer multiwalker resource acquisition requires its leader in lane zero");
  if (leader.acquired_crowd_leader_ != nullptr ||
      leader.acquired_crowd_size_ != 0)
    throw std::logic_error(
        "PsiFormer multiwalker leader retained stale acquisition-lane state");
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto* component_ptr = dynamic_cast<PsiFormerWF*>(&wfc_list[walker]);
    if (component_ptr == nullptr)
      throw std::invalid_argument(
          "PsiFormer multiwalker resource acquisition contains a foreign component type");
    const auto& component = *component_ptr;
    for (std::size_t prior = 0; prior < walker; ++prior)
      if (&wfc_list[walker] == &wfc_list[prior])
        throw std::invalid_argument(
            "PsiFormer multiwalker resource acquisition contains a duplicate component");
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer multiwalker list contains components from different models");
    if (!component.batch_execution_plan_.sameBinding(
            leader.batch_execution_plan_))
      throw std::invalid_argument(
          "PsiFormer multiwalker list contains components with different batch plans");
    if (component.acquired_crowd_leader_ != nullptr ||
        component.acquired_crowd_size_ != 0)
      throw std::logic_error(
          "PsiFormer multiwalker component retained stale acquisition-lane state");
  }

  const BatchResourcePreparationProvenance& collection_provenance =
      collection.getBatchResourcePreparationProvenance();
  if (leader.batch_execution_plan_)
  {
    if (collection_provenance.state !=
            BatchResourcePreparationState::PREPARED ||
        collection_provenance.plan.get() !=
            &leader.batch_execution_plan_.plan())
      throw std::logic_error(
          "PsiFormer hard-plan acquisition requires a matching prepared ResourceCollection");
  }
  else if (collection_provenance.state !=
               BatchResourcePreparationState::UNPREPARED ||
           collection_provenance.plan)
    throw std::logic_error(
        "PsiFormer no-policy acquisition received planned ResourceCollection storage");

  const auto entry_cursor = collection.getCursor();
  auto candidate_handle = collection.lendResource<PsiFormerMultiWalkerResource>();
  try
  {
    candidate_handle.getResource().validateAcquiredBinding(
        leader.model_state_.get(), leader.batch_execution_plan_,
        leader.batch_execution_plan_
            ? std::optional<std::size_t>(collection_provenance.crowd_index)
            : std::nullopt,
        wfc_list.size());
  }
  catch (...)
  {
    const std::exception_ptr failure = std::current_exception();
    // ResourceCollection takeback traverses from the cursor at which the
    // candidate was lent.  Return the loan before restoring the entry cursor.
    collection.rewind(entry_cursor);
    collection.takebackResource(candidate_handle);
    collection.rewind(entry_cursor);
    std::rethrow_exception(failure);
  }
  leader.mw_resource_handle_ = std::move(candidate_handle);
  auto& acquired_resource = leader.mw_resource_handle_.getResource();
  if (leader.batch_execution_plan_)
  {
    acquired_resource.acquired_plan_identity            = collection_provenance.plan.get();
    acquired_resource.acquired_crowd_index              = collection_provenance.crowd_index;
    acquired_resource.acquired_from_prepared_collection = true;
  }
  leader.acquired_resource_collection_        = &collection;
  leader.acquired_resource_cursor_            = collection.getCursor();
  leader.acquired_resource_outstanding_loans_ = collection.getOutstandingLoanCount();
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
    component.acquired_crowd_leader_ = &leader;
    component.acquired_lane_index_   = lane;
    component.acquired_crowd_size_   = wfc_list.size();
  }
}

// Return the exact handle previously lent to this crowd leader.
void PsiFormerWF::releaseResource(
    ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto* leader_ptr = dynamic_cast<PsiFormerWF*>(&wfc_list.getLeader());
  if (leader_ptr == nullptr)
    throw std::invalid_argument(
        "PsiFormer multiwalker resource release received a foreign leader type");
  auto& leader = *leader_ptr;
  if (this != &leader || !leader.mw_resource_handle_)
    throw std::logic_error("PsiFormer multiwalker resource release has no acquired leader handle");
  if (leader.acquired_resource_collection_ != &collection)
    throw std::logic_error(
        "PsiFormer multiwalker resource release received a different collection");
  if (collection.getOutstandingLoanCount() !=
      leader.acquired_resource_outstanding_loans_)
    throw std::logic_error(
        "PsiFormer multiwalker resource release observed changed loan ownership");
  if (!wfc_list.empty() && &wfc_list[0] != &leader)
    throw std::invalid_argument(
        "PsiFormer multiwalker resource release requires its leader in lane zero");
  if (wfc_list.empty() &&
      (leader.acquired_crowd_leader_ != nullptr ||
       leader.acquired_crowd_size_ != 0))
    throw std::invalid_argument(
        "PsiFormer multiwalker empty release does not match its acquired crowd");
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    auto* component_ptr = dynamic_cast<PsiFormerWF*>(&wfc_list[lane]);
    if (component_ptr == nullptr)
      throw std::invalid_argument(
          "PsiFormer multiwalker resource release contains a foreign component type");
    const auto& component = *component_ptr;
    for (std::size_t prior = 0; prior < lane; ++prior)
      if (&wfc_list[lane] == &wfc_list[prior])
        throw std::invalid_argument(
            "PsiFormer multiwalker resource release contains a duplicate component");
    if (component.acquired_crowd_leader_ != &leader ||
        component.acquired_lane_index_ != lane ||
        component.acquired_crowd_size_ != wfc_list.size())
      throw std::invalid_argument(
          "PsiFormer multiwalker resource release order differs from acquisition");
  }
  auto& resource = leader.mw_resource_handle_.getResource();
  if (resource.prepared_plan)
  {
    for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    {
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      if (component.has_proposal_ || component.proposal_origin_ != ProposalOrigin::NONE)
        throw std::logic_error(
            "Cannot release a planned PsiFormer crowd resource while a proposal is pending");
    }
    if (!resource.hasExactPreparedStagingExtents() ||
        resource.currentStorageFingerprint() !=
            resource.prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer planned resource storage changed before release");
    resource.requireFullVGLStaging(wfc_list.size());
  }
  // Return ownership first.  If ResourceCollection rejects the ordering, the
  // handle, logical extents, and provenance remain intact for a retry.
  collection.takebackResource(leader.mw_resource_handle_);
  // A prepared resource retains every exact physical extent.  Planned calls
  // consume bounded prefixes and fail closed if an unmigrated path changes a
  // size.  Legacy resources preserve their historical clear-on-release behavior.
  if (!resource.prepared_plan)
  {
    resource.configuration_identities.clear();
    resource.batch_slots.clear();
    resource.staged_signs.clear();
    resource.staged_log_magnitudes.clear();
    resource.staged_value_ratios.clear();
    resource.staged_log_ratios.clear();
    resource.staged_gradients.clear();
    resource.preservation_flags.clear();
    resource.active_electrons.clear();
    resource.virtual_offsets.clear();
    resource.active_virtual_walkers.clear();
    resource.virtual_reference_indices.clear();
    resource.flat_virtual_ratios.clear();
    resource.virtual_reference_weights.clear();
    resource.active_derivative_global_indices.clear();
    resource.flat_virtual_weighted_derivatives.clear();
    resource.virtual_score_contribution.clear();
    resource.kinetic_parameter_contribution.clear();
    resource.virtual_reference_signs.clear();
    resource.virtual_reference_logabs.clear();
    resource.walker_indices.clear();
  }
  resource.acquired_plan_identity            = nullptr;
  resource.acquired_crowd_index              = 0;
  resource.acquired_from_prepared_collection = false;
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
    component.acquired_crowd_leader_ = nullptr;
    component.acquired_lane_index_   = 0;
    component.acquired_crowd_size_   = 0;
  }
  leader.acquired_resource_collection_        = nullptr;
  leader.acquired_resource_cursor_            = 0;
  leader.acquired_resource_outstanding_loans_ = 0;
}

// Validate a crowd call before dereferencing mutable shared scratch.
PsiFormerWF::PsiFormerMultiWalkerResource& PsiFormerWF::requireMultiWalkerResource(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  if (this != &leader)
    throw std::logic_error("PsiFormer multiwalker method must be invoked on the crowd leader");
  if (!leader.mw_resource_handle_)
    throw std::logic_error("PsiFormer multiwalker method requires an acquired ResourceCollection");

  auto& resource = leader.mw_resource_handle_.getResource();
  resource.validateAcquiredBinding(leader.model_state_.get(),
                                   leader.batch_execution_plan_, std::nullopt,
                                   wfc_list.size());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer multiwalker list contains components from different models");
    if (!component.batch_execution_plan_.sameBinding(
            leader.batch_execution_plan_))
      throw std::invalid_argument(
          "PsiFormer multiwalker list contains components with different batch plans");
  }
  return resource;
}

// Fingerprint the exact ephemeral component/PSet team established at acquisition.
std::uint64_t PsiFormerWF::selectedTeamFingerprint(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list) const noexcept
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  const auto mix_pointer = [&hash](const void* pointer) noexcept {
    mixPersistentInteger(
        hash, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)));
  };
  mixPersistentInteger(hash, SELECTED_TEAM_FINGERPRINT_DOMAIN);
  mixPersistentInteger(hash, wfc_list.size());
  mix_pointer(this);
  mix_pointer(model_state_.get());
  mixPersistentInteger(hash, model_state_->persistent_model_identity);
  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  mix_pointer(&plan);
  mixPersistentInteger(hash, plan.fingerprint());
  mix_pointer(acquired_resource_collection_);
  mixPersistentInteger(hash, acquired_resource_cursor_);
  mixPersistentInteger(hash, acquired_resource_outstanding_loans_);
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    mix_pointer(&component);
    mix_pointer(component.acquired_crowd_leader_);
    mixPersistentInteger(hash, component.acquired_lane_index_);
    mixPersistentInteger(hash, component.acquired_crowd_size_);
    mix_pointer(component.bound_particle_set_);
    mix_pointer(&p_list[lane]);
  }
  return hash;
}

// Bind a selected descriptor to one exact acquired component/PSet team.
std::uint64_t PsiFormerWF::selectedTransactionFingerprint(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    std::uint64_t descriptor_fingerprint) const noexcept
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(hash, SELECTED_TRANSACTION_FINGERPRINT_DOMAIN);
  mixPersistentInteger(hash, descriptor_fingerprint);
  mixPersistentInteger(hash, selectedTeamFingerprint(wfc_list, p_list));
  return hash;
}

// Map each typed operation to the explicit modes that authorize its runtime path.
BatchExecutionRequirements PsiFormerWF::plannedOperationRequiredModes(
    PlannedOperation operation) noexcept
{
  BatchExecutionRequirements modes;
  switch (operation)
  {
  case PlannedOperation::FULL_VGL:
  case PlannedOperation::SELECTED_PROPOSE:
    modes.require(BatchExecutionMode::FULL_VGL);
    break;
  case PlannedOperation::RECOMPUTE_VALUE:
  case PlannedOperation::CALC_RATIO:
    modes.require(BatchExecutionMode::VALUE);
    break;
  case PlannedOperation::ACTIVE_GRADIENT:
    modes.require(BatchExecutionMode::ACTIVE_GRADIENT);
    break;
  case PlannedOperation::RATIO_GRADIENT:
    modes.require(BatchExecutionMode::ACTIVE_GRADIENT);
    break;
  case PlannedOperation::ECP_VALUE:
    modes.require(BatchExecutionMode::VALUE);
    modes.require(BatchExecutionMode::ECP_OUTER);
    break;
  case PlannedOperation::ECP_WEIGHTED_SCORE:
    modes.require(BatchExecutionMode::VALUE);
    modes.require(BatchExecutionMode::ECP_OUTER);
    modes.require(BatchExecutionMode::ECP_WEIGHTED_SCORE);
    break;
  case PlannedOperation::SCORE_DERIVATIVES:
    modes.require(BatchExecutionMode::SCORE);
    break;
  case PlannedOperation::KINETIC_DERIVATIVES:
    modes.require(BatchExecutionMode::KINETIC);
    break;
  case PlannedOperation::SCALAR_VALUE_COMPATIBILITY:
    modes.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
    break;
  case PlannedOperation::ACCEPT_REJECT_VALUE:
  case PlannedOperation::SELECTED_RESOLVE:
  case PlannedOperation::SELECTED_CANCEL:
  case PlannedOperation::BUFFER_READ:
  case PlannedOperation::BUFFER_WRITE:
  case PlannedOperation::PREPARE_GROUP:
  case PlannedOperation::COMPLETE_UPDATES:
    break;
  }
  return modes;
}

// Map each typed operation to the proposal state required before it may begin.
PsiFormerWF::ProposalRequirement PsiFormerWF::plannedOperationProposalRequirement(
    PlannedOperation operation) noexcept
{
  switch (operation)
  {
  case PlannedOperation::ACCEPT_REJECT_VALUE:
    return ProposalRequirement::SINGLE_PENDING;
  case PlannedOperation::SELECTED_RESOLVE:
  case PlannedOperation::SELECTED_CANCEL:
    return ProposalRequirement::SELECTED_PENDING;
  case PlannedOperation::FULL_VGL:
  case PlannedOperation::RECOMPUTE_VALUE:
  case PlannedOperation::CALC_RATIO:
  case PlannedOperation::ACTIVE_GRADIENT:
  case PlannedOperation::RATIO_GRADIENT:
  case PlannedOperation::SELECTED_PROPOSE:
  case PlannedOperation::ECP_VALUE:
  case PlannedOperation::ECP_WEIGHTED_SCORE:
  case PlannedOperation::SCORE_DERIVATIVES:
  case PlannedOperation::KINETIC_DERIVATIVES:
  case PlannedOperation::SCALAR_VALUE_COMPATIBILITY:
  case PlannedOperation::BUFFER_READ:
  case PlannedOperation::BUFFER_WRITE:
  case PlannedOperation::PREPARE_GROUP:
  case PlannedOperation::COMPLETE_UPDATES:
    return ProposalRequirement::ABSENT;
  }
  return ProposalRequirement::NONE;
}

// Prove all common planned-runtime facts without synchronizing or publishing state.
PsiFormerWF::PlannedRuntimeAccess PsiFormerWF::requirePlannedMultiWalkerOperation(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const PlannedRuntimeRequest& request) const
{
  if (wfc_list.empty() || p_list.empty())
    throw std::invalid_argument("PsiFormer planned operation requires a nonempty crowd");
  if (wfc_list.size() != p_list.size() ||
      request.live_walkers != wfc_list.size())
    throw std::invalid_argument(
        "PsiFormer planned operation has inconsistent live-lane counts");
  if (&wfc_list.getLeader() != &wfc_list[0] ||
      &p_list.getLeader() != &p_list[0])
    throw std::invalid_argument(
        "PsiFormer planned operation lists must place their leaders in lane zero");

  auto* component_leader = dynamic_cast<PsiFormerWF*>(&wfc_list.getLeader());
  if (component_leader == nullptr || component_leader != this)
    throw std::invalid_argument(
        "PsiFormer planned operation was not invoked on its component leader");

  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    if (dynamic_cast<PsiFormerWF*>(&wfc_list[lane]) == nullptr)
      throw std::invalid_argument(
          "PsiFormer planned operation contains a foreign component type");
    for (std::size_t prior = 0; prior < lane; ++prior)
    {
      if (&wfc_list[lane] == &wfc_list[prior])
        throw std::invalid_argument(
            "PsiFormer planned operation contains a duplicate component");
      if (&p_list[lane] == &p_list[prior])
        throw std::invalid_argument(
            "PsiFormer planned operation contains a duplicate ParticleSet");
    }
  }

  if (!batch_execution_plan_ ||
      !hasPreparedBatchExecutionClone(batch_execution_plan_))
    throw std::logic_error(
        "PsiFormer planned operation requires an exactly prepared leader clone");

  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  const psiformer::ModelShape& model_shape =
      model_state_->execution_plan.modelShape();
  const std::size_t electron_count = model_shape.electrons();
  if (plan.targetCoordinate() != BatchExecutionTargetCoordinate::POS_ONLY)
    throw std::invalid_argument(
        "PsiFormer planned operation requires explicit POS-only target evidence");
  if (plan.particleCount() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned operation particle count differs from its model");
  const bool requires_active_move =
      request.operation == PlannedOperation::CALC_RATIO ||
      request.operation == PlannedOperation::RATIO_GRADIENT ||
      request.operation == PlannedOperation::ACCEPT_REJECT_VALUE;

  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const ParticleSet& particles = p_list[lane];
    if (component.model_state_.get() != model_state_.get())
      throw std::invalid_argument(
          "PsiFormer planned operation mixes distinct shared models");
    if (component.optimization_metadata_.get() != optimization_metadata_.get())
      throw std::invalid_argument(
          "PsiFormer planned operation mixes distinct optimizer metadata");
    if (component.bound_particle_set_ != &particles)
      throw std::invalid_argument(
          "PsiFormer planned operation component and ParticleSet lanes are not identically bound");
    if (component.acquired_crowd_leader_ != component_leader ||
        component.acquired_lane_index_ != lane ||
        component.acquired_crowd_size_ != wfc_list.size())
      throw std::invalid_argument(
          "PsiFormer planned operation lane order differs from resource acquisition");
    if (!component.batch_execution_plan_.sameBinding(batch_execution_plan_) ||
        !component.hasPreparedBatchExecutionClone(batch_execution_plan_))
      throw std::logic_error(
          "PsiFormer planned operation contains an unprepared or differently bound clone");
    if (particles.isSpinor())
      throw std::invalid_argument(
          "PsiFormer planned POS-only operation received a spinor ParticleSet");
    if (static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
        particles.R.size() != electron_count || particles.G.size() != electron_count ||
        particles.L.size() != electron_count || particles.GroupID.size() != electron_count ||
        particles.spins.size() != electron_count ||
        particles.getCoordinates().getAllParticlePos().size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer planned operation received incompatible ParticleSet extents");
    if (particles.groups() != 2 ||
        particles.groupsize(0) != static_cast<int>(model_shape.spin_up_electrons) ||
        particles.groupsize(1) != static_cast<int>(model_shape.spin_down_electrons))
      throw std::invalid_argument(
          "PsiFormer planned operation received an incompatible spin partition");
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      const int expected_group =
          electron < model_shape.spin_up_electrons ? 0 : 1;
      if (particles.GroupID[electron] != expected_group)
        throw std::invalid_argument(
            "PsiFormer planned operation received noncanonical spin ordering");
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        if (!psiformer::determinant::isFiniteReal(
                static_cast<double>(particles.R[electron][dimension])))
          throw std::invalid_argument(
              "PsiFormer planned operation received a non-finite position");
        if (particles.getCoordinates().getAllParticlePos()[electron][dimension] !=
            particles.R[electron][dimension])
          throw std::invalid_argument(
              "PsiFormer planned operation has inconsistent AoS and SoA positions");
      }
    }
    if ((requires_active_move &&
         (!request.active_electron || particles.getActivePtcl() !=
              static_cast<int>(*request.active_electron))) ||
        (!requires_active_move && particles.getActivePtcl() != -1))
      throw std::invalid_argument(
          "PsiFormer planned operation has incompatible ParticleSet active-move state");
    if (requires_active_move)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!psiformer::determinant::isFiniteReal(
                static_cast<double>(particles.getActivePos()[dimension])))
          throw std::invalid_argument(
              "PsiFormer planned operation received a non-finite active position");
    if (component.prepared_accepted_gradient_capacity_ != electron_count ||
        component.prepared_accepted_laplacian_capacity_ != electron_count ||
        component.prepared_proposed_gradient_capacity_ != electron_count ||
        component.prepared_proposed_laplacian_capacity_ != electron_count)
      throw std::logic_error(
          "PsiFormer planned operation clone storage differs from its exact prepared extent");
  }

  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  if (acquired_resource_collection_ == nullptr ||
      acquired_resource_cursor_ == 0 ||
      acquired_resource_outstanding_loans_ == 0)
    throw std::logic_error(
        "PsiFormer planned operation lacks borrowed collection provenance");
  const BatchResourcePreparationProvenance& collection_provenance =
      acquired_resource_collection_->getBatchResourcePreparationProvenance();
  if (collection_provenance.state != BatchResourcePreparationState::PREPARED ||
      collection_provenance.plan.get() != &plan ||
      collection_provenance.crowd_index != resource.prepared_crowd_index ||
      acquired_resource_collection_->getCursor() != acquired_resource_cursor_ ||
      acquired_resource_collection_->getOutstandingLoanCount() !=
          acquired_resource_outstanding_loans_)
    throw std::logic_error(
        "PsiFormer planned operation borrowed collection changed after acquisition");
  if (!resource.acquired_from_prepared_collection ||
      resource.acquired_plan_identity != &plan ||
      resource.acquired_crowd_index != resource.prepared_crowd_index)
    throw std::logic_error(
        "PsiFormer planned operation lacks exact prepared-collection provenance");
  if (!resource.prepared_plan.sameBinding(batch_execution_plan_) ||
      !resource.prepared_crowd_plan ||
      resource.prepared_plan_fingerprint != plan.fingerprint())
    throw std::logic_error(
        "PsiFormer planned operation has stale resource plan provenance");

  const std::size_t crowd_index = resource.prepared_crowd_index;
  const BatchExecutionTopology& topology = plan.topology();
  const std::vector<std::size_t>& reserve_walkers =
      psiformer::psiFormerReserveWalkersPerCrowd(topology);
  if (crowd_index >= topology.initial_walkers_per_crowd.size() ||
      crowd_index >= reserve_walkers.size())
    throw std::logic_error(
        "PsiFormer planned operation resource crowd index is outside the topology");

  const psiformer::PsiFormerCrowdMemoryPlan& crowd =
      *resource.prepared_crowd_plan;
  if (crowd.initial_walkers != topology.initial_walkers_per_crowd[crowd_index] ||
      crowd.reserve_walkers != reserve_walkers[crowd_index] ||
      request.live_walkers > crowd.reserve_walkers)
    throw std::length_error(
        "PsiFormer planned operation exceeds or mismatches its crowd envelope");
  if (!resource.batch_workspace || !resource.batch_workspace->hasCapacityPlan())
    throw std::logic_error(
        "PsiFormer planned operation has no prepared direct-batch workspace");

  const pf::DirectBatchCapacityPlan& workspace_plan =
      resource.batch_workspace->capacityPlan();
  const auto same_direct_capacity = [](const pf::DirectBatchCapacityPlan& left,
                                       const pf::DirectBatchCapacityPlan& right) {
    return left.logical.value_dense == right.logical.value_dense &&
        left.logical.full_vgl == right.logical.full_vgl &&
        left.logical.active_gradient == right.logical.active_gradient &&
        left.logical.sparse_references == right.logical.sparse_references &&
        left.logical.sparse_replacements == right.logical.sparse_replacements &&
        left.tile.value == right.tile.value &&
        left.tile.full_vgl == right.tile.full_vgl &&
        left.tile.active_gradient == right.tile.active_gradient;
  };
  if (!same_direct_capacity(workspace_plan, crowd.direct_batch))
    throw std::logic_error(
        "PsiFormer planned operation direct-batch capacity changed after preparation");

  const std::size_t storage_fingerprint = resource.currentStorageFingerprint();
  if (storage_fingerprint == 0 ||
      storage_fingerprint != resource.prepared_storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned operation resource storage changed after preparation");
  if (!resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned operation resource staging extent changed after preparation");

  if (topology.serialized_walkers || topology.backend_id != "cpu" ||
      topology.device_id)
    throw std::invalid_argument(
        "PsiFormer planned operation requires direct nonserialized CPU execution");
  const BatchExecutionRequirements operation_modes =
      plannedOperationRequiredModes(request.operation);
  for (const BatchExecutionMode mode : {
           BatchExecutionMode::VALUE, BatchExecutionMode::FULL_VGL,
           BatchExecutionMode::ACTIVE_GRADIENT, BatchExecutionMode::SCORE,
           BatchExecutionMode::KINETIC, BatchExecutionMode::ECP_OUTER,
           BatchExecutionMode::ECP_WEIGHTED_SCORE,
           BatchExecutionMode::ECP_TMOVE_CANDIDATES,
           BatchExecutionMode::ECP_LISTENER_OUTPUT,
           BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY})
    if (operation_modes.requires(mode) &&
        !plan.requirements().requires(mode))
      throw std::invalid_argument(
          "PsiFormer planned operation lacks an explicitly required mode");
  validatePlannedBackends(makeBatchMemoryPolicyInput(), operation_modes);

  const bool sparse_operation = request.operation == PlannedOperation::ECP_VALUE ||
      request.operation == PlannedOperation::ECP_WEIGHTED_SCORE;
  if (sparse_operation)
  {
    if (request.dense_configurations != 0 ||
        request.sparse_references > crowd.direct_batch.logical.sparse_references ||
        request.sparse_references > request.live_walkers ||
        request.sparse_replacements > crowd.direct_batch.logical.sparse_replacements)
      throw std::length_error(
          "PsiFormer planned sparse operation exceeds its prepared extents");
  }
  else if (request.sparse_references != 0 || request.sparse_replacements != 0)
    throw std::invalid_argument(
        "PsiFormer planned dense operation received sparse extents");

  std::size_t dense_capacity = 0;
  enum class DenseExtentRule
  {
    ZERO,
    EXACT_LIVE,
    AT_MOST_LIVE,
    SCALAR_CLONE
  };
  DenseExtentRule dense_rule = DenseExtentRule::ZERO;
  switch (request.operation)
  {
  case PlannedOperation::FULL_VGL:
    dense_capacity    = crowd.direct_batch.logical.full_vgl;
    dense_rule        = DenseExtentRule::EXACT_LIVE;
    break;
  case PlannedOperation::SELECTED_PROPOSE:
    dense_capacity = crowd.direct_batch.logical.full_vgl;
    dense_rule     = DenseExtentRule::EXACT_LIVE;
    break;
  case PlannedOperation::RECOMPUTE_VALUE:
    dense_capacity    = crowd.direct_batch.logical.value_dense;
    dense_rule        = DenseExtentRule::AT_MOST_LIVE;
    break;
  case PlannedOperation::CALC_RATIO:
    dense_capacity = crowd.direct_batch.logical.value_dense;
    dense_rule     = DenseExtentRule::EXACT_LIVE;
    break;
  case PlannedOperation::ACTIVE_GRADIENT:
  case PlannedOperation::RATIO_GRADIENT:
    dense_capacity    = crowd.direct_batch.logical.active_gradient;
    dense_rule        = DenseExtentRule::EXACT_LIVE;
    break;
  case PlannedOperation::SCORE_DERIVATIVES:
  case PlannedOperation::KINETIC_DERIVATIVES:
    dense_capacity    = crowd.reserve_walkers;
    dense_rule        = DenseExtentRule::EXACT_LIVE;
    break;
  case PlannedOperation::SCALAR_VALUE_COMPATIBILITY:
    dense_rule = DenseExtentRule::SCALAR_CLONE;
    break;
  case PlannedOperation::ACCEPT_REJECT_VALUE:
  case PlannedOperation::SELECTED_RESOLVE:
  case PlannedOperation::SELECTED_CANCEL:
  case PlannedOperation::ECP_VALUE:
  case PlannedOperation::ECP_WEIGHTED_SCORE:
  case PlannedOperation::BUFFER_READ:
  case PlannedOperation::BUFFER_WRITE:
  case PlannedOperation::PREPARE_GROUP:
  case PlannedOperation::COMPLETE_UPDATES:
    break;
  }
  bool invalid_dense_extent = false;
  switch (dense_rule)
  {
  case DenseExtentRule::ZERO:
    invalid_dense_extent = request.dense_configurations != 0;
    break;
  case DenseExtentRule::EXACT_LIVE:
    invalid_dense_extent = request.dense_configurations != request.live_walkers ||
        request.dense_configurations > dense_capacity;
    break;
  case DenseExtentRule::AT_MOST_LIVE:
    invalid_dense_extent = request.dense_configurations > request.live_walkers ||
        request.dense_configurations > dense_capacity;
    break;
  case DenseExtentRule::SCALAR_CLONE:
    for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    {
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      if (!component.direct_batch_workspace_ ||
          !component.direct_batch_workspace_->hasCapacityPlan() ||
          request.dense_configurations >
              component.direct_batch_workspace_->capacityPlan().logical.value_dense)
      {
        invalid_dense_extent = true;
        break;
      }
    }
    break;
  }
  if (invalid_dense_extent)
    throw std::length_error(
        "PsiFormer planned operation has incompatible dense extents");

  const bool derivative_operation =
      request.operation == PlannedOperation::ECP_WEIGHTED_SCORE ||
      request.operation == PlannedOperation::SCORE_DERIVATIVES ||
      request.operation == PlannedOperation::KINETIC_DERIVATIVES;
  if (derivative_operation)
  {
    if (request.selected_parameters != crowd.publication_staging.active_parameters ||
        request.derivative_width != plan.parameterDerivativeWidth())
      throw std::invalid_argument(
          "PsiFormer planned derivative operation has incompatible parameter extents");
  }
  else if (request.selected_parameters != 0 || request.derivative_width != 0)
    throw std::invalid_argument(
        "PsiFormer planned nonderivative operation received derivative extents");

  const bool single_particle_operation =
      request.operation == PlannedOperation::CALC_RATIO ||
      request.operation == PlannedOperation::ACTIVE_GRADIENT ||
      request.operation == PlannedOperation::RATIO_GRADIENT ||
      request.operation == PlannedOperation::ACCEPT_REJECT_VALUE;
  if (single_particle_operation)
  {
    if (!request.active_electron || *request.active_electron >= electron_count)
      throw std::out_of_range(
          "PsiFormer planned one-electron operation has an invalid active electron");
  }
  else if (request.active_electron)
    throw std::invalid_argument(
        "PsiFormer planned operation received an unexpected active electron");

  const bool selected_operation =
      request.operation == PlannedOperation::SELECTED_PROPOSE ||
      request.operation == PlannedOperation::SELECTED_RESOLVE ||
      request.operation == PlannedOperation::SELECTED_CANCEL;
  if (selected_operation != request.descriptor_fingerprint.has_value())
    throw std::invalid_argument(
        "PsiFormer selected-operation descriptor identity is absent or unexpected");

  const bool consumes_selected_proposal =
      request.operation == PlannedOperation::SELECTED_RESOLVE ||
      request.operation == PlannedOperation::SELECTED_CANCEL;
  if (consumes_selected_proposal !=
      request.expected_proposal_version.has_value())
    throw std::invalid_argument(
        "PsiFormer selected proposal-version token is absent or unexpected");

  const std::optional<std::uint64_t> selected_transaction_fingerprint =
      selected_operation
      ? std::optional<std::uint64_t>(selectedTransactionFingerprint(
            wfc_list, p_list, *request.descriptor_fingerprint))
      : std::nullopt;

  if (selected_operation)
  {
    const std::size_t pending_selected_transactions =
        model_state_->planned_selected_transaction_count.load(
            std::memory_order_acquire);
    if (consumes_selected_proposal && pending_selected_transactions == 0)
      throw std::logic_error(
          "PsiFormer planned selected operation has inconsistent shared transaction state");
  }

  const ProposalRequirement expected_proposal =
      plannedOperationProposalRequirement(request.operation);
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    switch (expected_proposal)
    {
    case ProposalRequirement::NONE:
      break;
    case ProposalRequirement::ABSENT:
      if (component.has_proposal_ ||
          component.proposal_origin_ != ProposalOrigin::NONE)
        throw std::logic_error(
            "PsiFormer planned operation requires absent proposal state");
      break;
    case ProposalRequirement::SINGLE_PENDING:
      if (!component.has_proposal_ ||
          (component.proposal_origin_ != ProposalOrigin::MW_CALC_RATIO_VALUE &&
           component.proposal_origin_ != ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE) ||
          component.proposed_particle_ != static_cast<int>(*request.active_electron) ||
          component.proposed_parameter_version_ !=
              component.observed_parameter_version_)
        throw std::logic_error(
            "PsiFormer planned operation requires one exact single-particle proposal");
      break;
    case ProposalRequirement::SELECTED_PENDING:
      if (!component.has_proposal_ ||
          component.proposal_origin_ != ProposalOrigin::MW_SELECTED_FULL_VGL ||
          component.proposed_descriptor_fingerprint_ !=
              *selected_transaction_fingerprint ||
          component.proposed_parameter_version_ !=
              *request.expected_proposal_version ||
          component.proposed_parameter_version_ !=
              component.observed_parameter_version_)
        throw std::logic_error(
            "PsiFormer planned operation requires one exact selected-particle proposal");
      break;
    }
  }

  if (storage_fingerprint != resource.currentStorageFingerprint())
    throw std::logic_error(
        "PsiFormer planned operation resource storage changed during preflight");
  return {resource, batch_execution_plan_, crowd, storage_fingerprint,
          selected_transaction_fingerprint};
}

// Reserve one model-wide transaction count before nonthrowing clone publication.
bool PsiFormerWF::tryRegisterPlannedSelectedTransaction() const noexcept
{
  std::size_t pending_transactions =
      model_state_->planned_selected_transaction_count.load(
          std::memory_order_acquire);
  while (pending_transactions != std::numeric_limits<std::size_t>::max())
    if (model_state_->planned_selected_transaction_count.compare_exchange_weak(
            pending_transactions, pending_transactions + 1,
            std::memory_order_acq_rel, std::memory_order_acquire))
      return true;
  return false;
}

// Withdraw one crowd transaction only after its mechanically nonthrowing commit.
void PsiFormerWF::unregisterPlannedSelectedTransaction() const noexcept
{
  std::size_t pending_transactions =
      model_state_->planned_selected_transaction_count.load(
          std::memory_order_acquire);
  for (;;)
  {
    assert(pending_transactions != 0);
    if (pending_transactions == 0)
      std::terminate();
    if (model_state_->planned_selected_transaction_count.compare_exchange_weak(
            pending_transactions, pending_transactions - 1,
            std::memory_order_release, std::memory_order_acquire))
      return;
  }
}

// Publish only selected lifecycle provenance while holding the model read barrier.
PsiFormerWF::PlannedSelectedProposalEvidence
PsiFormerWF::publishPlannedSelectedProposalMetadata(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    std::uint64_t descriptor_fingerprint) const
{
  PlannedRuntimeRequest request;
  request.operation              = PlannedOperation::SELECTED_PROPOSE;
  request.live_walkers           = wfc_list.size();
  request.dense_configurations   = wfc_list.size();
  request.descriptor_fingerprint = descriptor_fingerprint;
  const PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);

  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t proposal_version = transaction.parameterVersion();
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    if (static_cast<const PsiFormerWF&>(wfc_list[lane])
            .observed_parameter_version_ != proposal_version)
      throw std::logic_error(
          "PsiFormer planned selected proposal has stale clone parameter state");

  if (!tryRegisterPlannedSelectedTransaction())
    throw std::overflow_error(
        "PsiFormer planned selected transaction count overflow");

  const std::uint64_t transaction_fingerprint =
      *access.selected_transaction_fingerprint;
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
    component.proposed_descriptor_fingerprint_ = transaction_fingerprint;
    component.proposed_parameter_version_      = proposal_version;
    component.proposed_particle_               = -1;
    component.proposal_origin_                 = ProposalOrigin::MW_SELECTED_FULL_VGL;
  }
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = true;
  return {transaction_fingerprint, proposal_version};
}

// Abandon a selected planned transaction only after proving its full provenance.
void PsiFormerWF::cancelPlannedSelectedProposal(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    std::size_t expected_proposal_version) const
{
  moves.validateFor(p_list);
  PlannedRuntimeRequest request;
  request.operation                 = PlannedOperation::SELECTED_CANCEL;
  request.live_walkers              = wfc_list.size();
  request.descriptor_fingerprint    = moves.fingerprint();
  request.expected_proposal_version = expected_proposal_version;
  requirePlannedMultiWalkerOperation(wfc_list, p_list, request);

  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    static_cast<PsiFormerWF&>(wfc_list[lane]).resetProposalMetadata();
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
    static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = false;
  unregisterPlannedSelectedTransaction();
}

// Expose only ownership counts needed to prove bounded crowd scratch in a regression test.
std::array<std::size_t, 2> PsiFormerWF::directKineticWorkspaceOwnershipForTesting(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  std::size_t clone_workspaces = 0;
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    if (wfc_list.getCastedElement<PsiFormerWF>(walker).direct_kinetic_workspace_)
      ++clone_workspaces;

  const PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  return {clone_workspaces, resource.kinetic_workspace ? std::size_t{1} : std::size_t{0}};
}

// Report evaluator scratch and fixed clone state separately; accountedBytes()
// deliberately excludes the latter to preserve its evaluator-scratch contract.
testing::PsiFormerWorkspaceDiagnostics PsiFormerWF::directWorkspaceDiagnosticsForTesting() const
{
  testing::PsiFormerWorkspaceDiagnostics diagnostics;
  diagnostics.owns_value_workspace          = static_cast<bool>(direct_value_workspace_);
  diagnostics.owns_full_spatial_workspace   = static_cast<bool>(direct_full_spatial_workspace_);
  diagnostics.owns_active_spatial_workspace = static_cast<bool>(direct_active_spatial_workspace_);
  diagnostics.owns_batch_workspace          = static_cast<bool>(direct_batch_workspace_);
  diagnostics.owns_score_workspace          = static_cast<bool>(direct_score_workspace_);
  diagnostics.owns_kinetic_workspace        = static_cast<bool>(direct_kinetic_workspace_);
  diagnostics.has_prepared_clone_plan = static_cast<bool>(prepared_clone_batch_execution_plan_);

  if (direct_value_workspace_)
    diagnostics.value_bytes = direct_value_workspace_->vectorStorageBytes();
  if (direct_full_spatial_workspace_)
    diagnostics.full_spatial_bytes = direct_full_spatial_workspace_->vectorStorageBytes();
  if (direct_active_spatial_workspace_)
    diagnostics.active_spatial_bytes = direct_active_spatial_workspace_->vectorStorageBytes();
  if (direct_batch_workspace_)
  {
    diagnostics.batch_bytes = direct_batch_workspace_->vectorStorageBytes();
    diagnostics.batch_storage_fingerprint =
        direct_batch_workspace_->storageFingerprint(pf::DirectBatchMode::VALUE_ONLY);
    diagnostics.batch_workspace_identity = direct_batch_workspace_.get();
  }
  if (direct_score_workspace_)
    diagnostics.score_bytes = direct_score_workspace_->vectorStorageBytes();
  if (direct_kinetic_workspace_)
    diagnostics.kinetic_bytes = direct_kinetic_workspace_->vectorStorageBytes();
  diagnostics.total_log_gradient_bytes = direct_total_log_gradient_.capacity() * sizeof(double);
  diagnostics.scalar_value_publication_bytes =
      scalar_value_publication_.capacity() * sizeof(ValueType);
  diagnostics.accepted_spatial_bytes =
      accepted_gradient_.capacity() * sizeof(GradType) +
      accepted_laplacian_.capacity() * sizeof(ValueType);
  diagnostics.proposed_spatial_bytes =
      proposed_gradient_.capacity() * sizeof(GradType) +
      proposed_laplacian_.capacity() * sizeof(ValueType);
  return diagnostics;
}

// Expose only opaque identities and numeric capacities needed by the
// non-gating production-thread benchmark.  The read transaction binds every
// model field in this snapshot to one published parameter version.
testing::PsiFormerCrowdWorkspaceDiagnostics
PsiFormerWF::crowdWorkspaceDiagnosticsForTesting(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  PsiFormerReadTransaction transaction(*model_state_);
  const PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);

  testing::PsiFormerCrowdWorkspaceDiagnostics diagnostics;
  diagnostics.shared_model_identity      = &transaction.state();
  diagnostics.resource_identity          = &resource;
  diagnostics.batch_workspace_identity   = resource.batch_workspace.get();
  diagnostics.score_workspace_identity   = resource.score_workspace.get();
  diagnostics.kinetic_workspace_identity = resource.kinetic_workspace.get();
  diagnostics.persistent_model_identity  = transaction.state().persistent_model_identity;
  diagnostics.parameter_version          = transaction.parameterVersion();
  diagnostics.batch_bytes                = resource.batch_workspace->vectorStorageBytes();
  if (resource.score_workspace)
    diagnostics.score_bytes = resource.score_workspace->vectorStorageBytes();
  if (resource.kinetic_workspace)
    diagnostics.kinetic_bytes = resource.kinetic_workspace->vectorStorageBytes();
  diagnostics.transient_bytes =
      resource.total_log_gradient.capacity() * sizeof(double) +
      resource.configuration_identities.capacity() * sizeof(std::uint64_t) +
      resource.batch_slots.capacity() * sizeof(std::size_t) +
      resource.staged_signs.capacity() * sizeof(double) +
      resource.staged_log_magnitudes.capacity() * sizeof(double) +
      resource.staged_value_ratios.capacity() * sizeof(ValueType) +
      resource.staged_log_ratios.capacity() * sizeof(LogValue) +
      resource.staged_gradients.capacity() * sizeof(GradType) +
      resource.preservation_flags.capacity() * sizeof(unsigned char) +
      resource.active_electrons.capacity() * sizeof(std::size_t) +
      resource.virtual_offsets.capacity() * sizeof(std::size_t) +
      resource.active_virtual_walkers.capacity() * sizeof(std::size_t) +
      resource.virtual_reference_indices.capacity() * sizeof(std::size_t) +
      resource.flat_virtual_ratios.capacity() * sizeof(ValueType) +
      resource.virtual_reference_weights.capacity() * sizeof(ValueType) +
      resource.active_derivative_global_indices.capacity() * sizeof(std::size_t) +
      resource.flat_virtual_weighted_derivatives.capacity() * sizeof(ValueType) +
      resource.virtual_score_contribution.capacity() * sizeof(SelectedDerivativeDelta::value_type) +
      resource.kinetic_parameter_contribution.capacity() * sizeof(SelectedDerivativeDelta::value_type) +
      resource.virtual_reference_signs.capacity() * sizeof(double) +
      resource.virtual_reference_logabs.capacity() * sizeof(double) +
      resource.walker_indices.capacity() * sizeof(std::size_t);
  const pf::DirectBatchExecutionStatistics& batch_statistics =
      resource.batch_workspace->executionStatistics();
  diagnostics.reference_configurations =
      batch_statistics.reference_configurations;
  diagnostics.replacement_configurations =
      batch_statistics.replacement_configurations;
  diagnostics.reference_evaluations =
      batch_statistics.reference_evaluations;
  diagnostics.dense_coordinate_bytes_avoided =
      batch_statistics.dense_coordinate_bytes_avoided;
  diagnostics.weighted_reference_configurations =
      resource.weighted_reference_configurations;
  diagnostics.weighted_replacement_configurations =
      resource.weighted_replacement_configurations;
  diagnostics.weighted_active_parameters = resource.weighted_active_parameters;
  diagnostics.weighted_derivative_staging_bytes =
      resource.flat_virtual_weighted_derivatives.capacity() * sizeof(ValueType);
  diagnostics.has_expected_plan  = static_cast<bool>(resource.expected_plan);
  diagnostics.has_prepared_plan  = static_cast<bool>(resource.prepared_plan);
  diagnostics.participant_id     = resource.participant_id;
  if (resource.prepared_plan)
  {
    diagnostics.prepared_plan_identity = &resource.prepared_plan.plan();
    diagnostics.prepared_plan_fingerprint =
        resource.prepared_plan_fingerprint;
  }
  if (resource.prepared_crowd_plan)
  {
    diagnostics.prepared_crowd_index = resource.prepared_crowd_index;
    diagnostics.initial_walker_capacity =
        resource.prepared_crowd_plan->initial_walkers;
    diagnostics.reserve_walker_capacity =
        resource.prepared_crowd_plan->reserve_walkers;
    diagnostics.expected_resource_storage =
        resource.prepared_crowd_plan->expected_resource_storage;
  }
  diagnostics.actual_resource_storage = resource.actual_resource_storage;
  diagnostics.prepared_storage_fingerprint =
      resource.prepared_storage_fingerprint;
  diagnostics.current_storage_fingerprint = resource.currentStorageFingerprint();
  diagnostics.logical_sizes = {
      resource.total_log_gradient.size(),
      resource.configuration_identities.size(),
      resource.batch_slots.size(),
      resource.staged_signs.size(),
      resource.staged_log_magnitudes.size(),
      resource.staged_value_ratios.size(),
      resource.staged_log_ratios.size(),
      resource.staged_gradients.size(),
      resource.preservation_flags.size(),
      resource.active_electrons.size(),
      resource.virtual_offsets.size(),
      resource.active_virtual_walkers.size(),
      resource.virtual_reference_indices.size(),
      resource.flat_virtual_ratios.size(),
      resource.virtual_reference_weights.size(),
      resource.active_derivative_global_indices.size(),
      resource.flat_virtual_weighted_derivatives.size(),
      resource.virtual_score_contribution.size(),
      resource.kinetic_parameter_contribution.size(),
      resource.virtual_reference_signs.size(),
      resource.virtual_reference_logabs.size(),
      resource.walker_indices.size()};
  diagnostics.backend_modes = {
      directBackendModeName(transaction.state().direct_value_mode),
      directBackendModeName(transaction.state().direct_spatial_mode),
      directBackendModeName(transaction.state().direct_score_mode),
      directBackendModeName(transaction.state().direct_kinetic_mode)};
  return diagnostics;
}

// Copy only the bounded compaction prefixes needed by deterministic tests.
testing::PsiFormerSelectedProposalMapDiagnostics
PsiFormerWF::selectedProposalMapDiagnosticsForTesting(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    std::size_t live_walkers,
    std::size_t evaluated_rows) const
{
  const PsiFormerMultiWalkerResource& resource =
      requireMultiWalkerResource(wfc_list);
  if (live_walkers > resource.batch_slots.size() ||
      evaluated_rows > resource.walker_indices.size())
    throw std::out_of_range(
        "PsiFormer selected compact-map diagnostic prefix is too large");
  return {{resource.batch_slots.begin(),
           resource.batch_slots.begin() + live_walkers},
          {resource.walker_indices.begin(),
           resource.walker_indices.begin() + evaluated_rows}};
}

// Expose the scalar guard count without granting a production lifecycle API.
std::size_t PsiFormerWF::plannedSelectedTransactionCountForTesting() const noexcept
{
  return model_state_->planned_selected_transaction_count.load(
      std::memory_order_acquire);
}

// Deliberately create model-version drift behind a pending proposal for testing.
std::size_t PsiFormerWF::advanceParameterVersionForTesting()
{
  std::unique_lock state_lock(model_state_->mutex);
  pf::Parameters& parameters = model_state_->model.p;
  const std::vector<double> unchanged_values = parameters.flat_values();
  parameters.set_flat_values(unchanged_values);
  return parameters.version();
}

// Expose metadata identity and cardinalities without copying the shared vectors.
testing::PsiFormerOptimizationMetadataDiagnostics
PsiFormerWF::optimizationMetadataDiagnosticsForTesting() const
{
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  testing::PsiFormerOptimizationMetadataDiagnostics diagnostics;
  diagnostics.identity                 = optimization_metadata_.get();
  diagnostics.shared_owner_count       = optimization_metadata_.use_count();
  diagnostics.selected_index_count     = optimization_metadata_->selected_flat_indices.size();
  diagnostics.shared_variable_count    = optimization_metadata_->variables.size();
  diagnostics.inherited_variable_count = myVars.size();
  for (std::size_t local_index = 0; local_index < optimization_metadata_->variables.size(); ++local_index)
    if (optimization_metadata_->variables.where(local_index) >= 0)
      ++diagnostics.mapped_variable_count;
  return diagnostics;
}

// Register this object only when the input explicitly enabled optimization.
void PsiFormerWF::extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs)
{
  if (optimization_metadata_->enabled)
    opt_obj_refs.push_back(*this);
}

// Append selected local values to the optimizer's global variable collection.
void PsiFormerWF::checkInVariablesExclusive(OptVariables& active)
{
  if (!optimization_metadata_->enabled)
    return;

  std::shared_lock state_lock(model_state_->mutex);
  std::unique_lock metadata_lock(optimization_metadata_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  for (std::size_t local_index = 0;
       local_index < optimization_metadata_->selected_flat_indices.size(); ++local_index)
    optimization_metadata_->variables[local_index] =
        model_state_->model.p.flat_values()[optimization_metadata_->selected_flat_indices[local_index]];
  active.insertFrom(optimization_metadata_->variables);
}

// Cache the global active index corresponding to each selected local parameter.
void PsiFormerWF::checkOutVariables(const OptVariables& active)
{
  if (optimization_metadata_->enabled)
  {
    std::unique_lock metadata_lock(optimization_metadata_->mutex);
    optimization_metadata_->variables.getIndex(active);
  }
}

// Clear all cached values derived from an older parameter vector.
void PsiFormerWF::invalidateParameterCaches(std::size_t parameter_version)
{
  current_sign_                     = 1.0;
  log_value_                        = LogValue(0);
  accepted_value_valid_             = false;
  accepted_configuration_identity_ = 0;
  accepted_parameter_version_      = parameter_version;
  accepted_state_requirement_      = AcceptedStateRequirement::INVALID;
  accepted_gradient_               = ValueType(0);
  accepted_laplacian_              = ValueType(0);
  clearProposalState();
  observed_parameter_version_      = parameter_version;
}

// Lazily invalidate clone-local caches after another clone updates the model.
void PsiFormerWF::synchronizeParameterVersion(std::size_t parameter_version)
{
  if (observed_parameter_version_ != parameter_version)
  {
    if (batch_execution_plan_ && has_proposal_ &&
        proposal_origin_ == ProposalOrigin::MW_SELECTED_FULL_VGL)
      throw std::logic_error(
          "PsiFormer planned selected proposal cannot be cleared by parameter-version synchronization");
    invalidateParameterCaches(parameter_version);
  }
}

// Reset proposal metadata without withdrawing the externally visible pending marker.
void PsiFormerWF::resetProposalMetadata() noexcept
{
  proposed_sign_                   = 1.0;
  proposed_log_value_              = LogValue(0);
  proposed_configuration_identity_ = 0;
  proposed_descriptor_fingerprint_ = 0;
  proposed_parameter_version_      = 0;
  proposed_particle_               = -1;
  proposal_origin_                 = ProposalOrigin::NONE;
}

// Drop all proposal discriminators while preserving reusable complete-VGL storage.
void PsiFormerWF::clearProposalState() noexcept
{
  resetProposalMetadata();
  has_proposal_ = false;
}

// Retain a one-electron proposal behind its exact scalar or crowd origin.
void PsiFormerWF::cacheSingleParticleProposal(double sign,
                                              double logabs,
                                              std::uint64_t configuration_identity,
                                              int particle,
                                              std::size_t parameter_version,
                                              ProposalOrigin origin)
{
  if (origin != ProposalOrigin::SCALAR_RATIO_VALUE &&
      origin != ProposalOrigin::SCALAR_RATIO_GRADIENT_ACTIVE &&
      origin != ProposalOrigin::MW_CALC_RATIO_VALUE &&
      origin != ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE)
    throw std::invalid_argument(
        "PsiFormer one-electron proposal has an incompatible origin");
  proposed_sign_                   = sign;
  proposed_log_value_              = makeLogValue(sign, logabs);
  proposed_configuration_identity_ = configuration_identity;
  proposed_descriptor_fingerprint_ = 0;
  proposed_parameter_version_      = parameter_version;
  proposed_particle_               = particle;
  proposal_origin_                 = origin;
  has_proposal_                    = true;
}

// Selected transactions must be resolved rather than silently replaced by another lifecycle call.
void PsiFormerWF::requireNoSelectedParticleProposal(const char* operation) const
{
  if (has_proposal_ && proposal_origin_ == ProposalOrigin::MW_SELECTED_FULL_VGL)
    throw std::logic_error(std::string("PsiFormer ") + operation +
                           " cannot run while a selected-electron proposal is pending");
}

// Planned selected state is crowd-owned and cannot be cleared by one parameter publisher.
void PsiFormerWF::requireNoPlannedSelectedProposalMutation(
    const char* operation) const
{
  if (model_state_->planned_selected_transaction_count.load(
          std::memory_order_acquire) != 0 ||
      (batch_execution_plan_ && has_proposal_ &&
       proposal_origin_ == ProposalOrigin::MW_SELECTED_FULL_VGL))
    throw std::logic_error(
        std::string("PsiFormer ") + operation +
        " cannot mutate parameters while a planned selected-electron proposal is pending");
}

// Prepare clone-local complete proposal products before the transaction is published.
void PsiFormerWF::resizeProposedSpatialStorage(std::size_t electron_count)
{
  if (proposed_gradient_.size() != electron_count)
    proposed_gradient_.resize(electron_count);
  if (proposed_laplacian_.size() != electron_count)
    proposed_laplacian_.resize(electron_count);
}

// Allocate fixed-size accepted derivative storage without changing a warmed cache.
void PsiFormerWF::resizeAcceptedSpatialStorage(std::size_t electron_count)
{
  if (accepted_gradient_.size() == electron_count && accepted_laplacian_.size() == electron_count)
    return;
  accepted_gradient_.resize(electron_count);
  accepted_laplacian_.resize(electron_count);
  accepted_gradient_          = ValueType(0);
  accepted_laplacian_         = ValueType(0);
  accepted_value_valid_       = false;
  accepted_state_requirement_ = AcceptedStateRequirement::INVALID;
}

// Check every mutable part of the accepted-state key before cache reuse.
bool PsiFormerWF::acceptedStateMatches(const ParticleSet& particles,
                                       std::size_t parameter_version,
                                       AcceptedStateRequirement requirement) const
{
  return accepted_value_valid_ && accepted_parameter_version_ == parameter_version &&
      accepted_configuration_identity_ == configurationIdentity(particles) &&
      static_cast<std::uint64_t>(accepted_state_requirement_) >= static_cast<std::uint64_t>(requirement);
}

// Accumulate only this component's persisted derivatives into the TWF totals.
void PsiFormerWF::accumulateAcceptedSpatial(ParticleSet::ParticleGradient& gradient,
                                            ParticleSet::ParticleLaplacian& laplacian) const
{
  if (accepted_state_requirement_ != AcceptedStateRequirement::FULL_SPATIAL)
    throw std::logic_error("PsiFormer accepted spatial cache is not live");
  if (gradient.size() != accepted_gradient_.size() || laplacian.size() != accepted_laplacian_.size())
    throw std::invalid_argument("PsiFormer accepted spatial cache has the wrong electron count");
  for (std::size_t electron = 0; electron < accepted_gradient_.size(); ++electron)
  {
    gradient[electron] += accepted_gradient_[electron];
    laplacian[electron] += accepted_laplacian_[electron];
  }
}

// Serialize bulk G/L products followed by an exact, self-identifying scalar header.
void PsiFormerWF::putAcceptedState(WFBufferType& buffer) const
{
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  if (!accepted_value_valid_ || accepted_state_requirement_ != AcceptedStateRequirement::FULL_SPATIAL)
    throw std::logic_error("PsiFormer cannot persist an incomplete accepted state");

  buffer.put(accepted_gradient_.data(), accepted_gradient_.data() + accepted_gradient_.size());
  buffer.put(accepted_laplacian_.data(), accepted_laplacian_.data() + accepted_laplacian_.size());
  putPersistentInteger(buffer, WALKER_BUFFER_MAGIC);
  putPersistentInteger(buffer, WALKER_BUFFER_SCHEMA_VERSION);
  putPersistentInteger(buffer, static_cast<std::uint64_t>(accepted_state_requirement_));
  putPersistentInteger(buffer, model_state_->persistent_model_identity);
  putPersistentInteger(buffer, static_cast<std::uint64_t>(accepted_parameter_version_));
  putPersistentInteger(buffer, accepted_configuration_identity_);
  putPersistentInteger(buffer, accepted_gradient_.size());
  double sign   = current_sign_;
  double logabs = std::real(log_value_);
  double phase  = std::imag(log_value_);
  buffer.put(sign);
  buffer.put(logabs);
  buffer.put(phase);
}

// Consume one complete record, rejecting corruption and invalidating stale keys.
void PsiFormerWF::getAcceptedState(const ParticleSet& particles,
                                   WFBufferType& buffer,
                                   std::size_t parameter_version)
{
  const std::size_t model_electrons =
      model_state_->execution_plan.modelShape().electrons();
  if (particles.getTotalNum() < 0 ||
      static_cast<std::size_t>(particles.getTotalNum()) != model_electrons)
    throw std::invalid_argument(
        "PsiFormer walker buffer electron count differs from the model");

  resizeAcceptedSpatialStorage(particles.getTotalNum());
  buffer.get(accepted_gradient_.data(), accepted_gradient_.data() + accepted_gradient_.size());
  buffer.get(accepted_laplacian_.data(), accepted_laplacian_.data() + accepted_laplacian_.size());
  const std::uint64_t magic       = getPersistentInteger(buffer, "magic");
  const std::uint64_t schema      = getPersistentInteger(buffer, "schema");
  const std::uint64_t requirement = getPersistentInteger(buffer, "evaluation requirement");
  const std::uint64_t model       = getPersistentInteger(buffer, "model identity");
  const std::uint64_t version     = getPersistentInteger(buffer, "parameter version");
  const std::uint64_t configuration = getPersistentInteger(buffer, "configuration identity");
  const std::uint64_t electron_count = getPersistentInteger(buffer, "electron count");
  double sign;
  double logabs;
  double phase;
  buffer.get(sign);
  buffer.get(logabs);
  buffer.get(phase);

  // A freshly allocated PooledMemory record is zero-filled.  It is deliberately
  // consumed as an invalid sentinel during the driver's initialization sequence,
  // but a partially initialized record is corruption rather than that sentinel.
  if (magic == 0)
  {
    const bool zero_sentinel = schema == 0 && requirement == 0 && model == 0 && version == 0 &&
        configuration == 0 && electron_count == 0 && sign == 0.0 && logabs == 0.0 && phase == 0.0;
    invalidateParameterCaches(parameter_version);
    if (!zero_sentinel)
      throw std::runtime_error("PsiFormer walker buffer has a malformed zero sentinel");
    return;
  }

  if (magic != WALKER_BUFFER_MAGIC)
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer magic does not match");
  }
  if (schema != WALKER_BUFFER_SCHEMA_VERSION)
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer schema is unsupported");
  }
  if (model != model_state_->persistent_model_identity)
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer belongs to a different physical model");
  }
  if (electron_count != static_cast<std::uint64_t>(particles.getTotalNum()))
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer electron count does not match");
  }
  if (requirement > static_cast<std::uint64_t>(AcceptedStateRequirement::FULL_SPATIAL))
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer has an unknown evaluation requirement");
  }
  if (version > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer parameter version is out of range");
  }
  if (!psiformer::determinant::isFiniteReal(sign) ||
      (!psiformer::determinant::isFiniteReal(logabs) && !(sign == 0.0 && isNegativeInfinity(logabs))) ||
      !psiformer::determinant::isFiniteReal(phase) ||
      (sign != -1.0 && sign != 0.0 && sign != 1.0) ||
      phase != (sign < 0.0 ? M_PI : 0.0))
  {
    invalidateParameterCaches(parameter_version);
    throw std::runtime_error("PsiFormer walker buffer amplitude is invalid");
  }

  const auto decoded_requirement = static_cast<AcceptedStateRequirement>(requirement);
  if (decoded_requirement != AcceptedStateRequirement::FULL_SPATIAL ||
      version != static_cast<std::uint64_t>(parameter_version) ||
      configuration != configurationIdentity(particles))
  {
    invalidateParameterCaches(parameter_version);
    return;
  }

  current_sign_                   = sign;
  log_value_                      = LogValue(logabs, phase);
  accepted_configuration_identity_ = configuration;
  accepted_parameter_version_      = static_cast<std::size_t>(version);
  accepted_state_requirement_      = decoded_requirement;
  accepted_value_valid_            = true;
  clearProposalState();
  observed_parameter_version_       = parameter_version;
}

// Apply a validated selected-parameter or complete-vector update at an exclusive model barrier.
void PsiFormerWF::resetParametersExclusive(const OptVariables& active)
{
  if (!optimization_metadata_->enabled)
    return;
  requireNoPlannedSelectedProposalMutation("optimizer reset");

  std::unique_lock state_lock(model_state_->mutex);
  requireNoPlannedSelectedProposalMutation("optimizer reset");
  std::unique_lock metadata_lock(optimization_metadata_->mutex);
  pf::Parameters& parameters = model_state_->model.p;
  const auto& selected_flat_indices = optimization_metadata_->selected_flat_indices;
  OptVariables& variables           = optimization_metadata_->variables;

  // Full scope is already in canonical flat order. Avoid materializing and
  // sorting redundant local/flat index vectors for every optimizer step.
  if (optimization_metadata_->optimize_all)
  {
    std::vector<double> active_values;
    active_values.reserve(selected_flat_indices.size());
    for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
    {
      const int global_index = variables.where(local_index);
      if (global_index < 0)
        throw std::runtime_error("PsiFormer optimize_scope=all requires every model parameter to remain active");
      if (global_index >= active.size())
        throw std::out_of_range("PsiFormer global optimization index is out of range");
      active_values.push_back(std::real(active[global_index]));
    }

    if (restore_validation_pending_ && active_values != parameters.flat_values())
      throw std::runtime_error(
          "PsiFormer generic full-network values disagree with the authoritative VP model payload");
    restore_validation_pending_ = false;
    const bool model_changed    = active_values != parameters.flat_values();
    if (model_changed)
      parameters.set_flat_values(active_values);
    const std::size_t parameter_version = parameters.version();

    for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
      variables[parameter] = active_values[parameter];
    if (model_changed || observed_parameter_version_ != parameter_version)
      invalidateParameterCaches(parameter_version);
    return;
  }

  std::vector<std::size_t> active_local_indices;
  std::vector<std::size_t> active_flat_indices;
  std::vector<double> active_values;
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= active.size())
      throw std::out_of_range("PsiFormer global optimization index is out of range");

    active_local_indices.push_back(local_index);
    active_flat_indices.push_back(selected_flat_indices[local_index]);
    active_values.push_back(std::real(active[global_index]));
  }

  if (active_values.empty())
    return;

  bool model_changed = false;
  // The complete object-specific payload is authoritative on restart. The
  // duplicated generic scalar list must agree before the normal reset path
  // is allowed to continue.
  if (restore_validation_pending_)
  {
    for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
      if (active_values[parameter] != parameters.flat_values()[active_flat_indices[parameter]])
        throw std::runtime_error(
            "PsiFormer generic selected values disagree with the authoritative VP model payload");
    restore_validation_pending_ = false;
  }

  for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
    model_changed =
        model_changed || active_values[parameter] != parameters.flat_values()[active_flat_indices[parameter]];

  if (model_changed)
    parameters.set_flat_values(active_flat_indices, active_values);
  const std::size_t parameter_version = parameters.version();

  for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
    variables[active_local_indices[parameter]] = active_values[parameter];

  if (model_changed || observed_parameter_version_ != parameter_version)
    invalidateParameterCaches(parameter_version);
}

// Store a complete, self-identifying model payload in the optimizer VP file.
void PsiFormerWF::writeVariationalParameters(hdf_archive& output)
{
  if (!optimization_metadata_->enabled)
    return;

  std::shared_lock state_lock(model_state_->mutex);
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const pf::PsiFormer& model = model_state_->model;

  output.push("PsiFormer");
  output.push(OptimizableObject::getName());

  const std::vector<int> format_version(PERSISTENCE_VERSION.begin(), PERSISTENCE_VERSION.end());
  const std::vector<std::uint64_t> parameter_count{model.p.size()};
  const std::vector<std::uint64_t> spin_counts{model.cfg.nup, model.cfg.ndown};
  const std::vector<std::uint64_t> architecture{model.ndet, model.dim, model.heads, model.blocks};
  const std::vector<std::uint64_t> initialization_seed{model_state_->initialization_seed};
  const std::vector<std::uint64_t> nuclear_shape(model.cfg.nuclei.shape.begin(), model.cfg.nuclei.shape.end());
  const std::vector<std::uint64_t> selected_indices =
      persistIndices(optimization_metadata_->selected_flat_indices);
  const std::string layout_fingerprint              = model.p.layout_fingerprint();
  const std::string model_fingerprint =
      modelFingerprint(model, model_state_->model_origin, model_state_->initialization_profile,
                       model_state_->initialization_seed);

  output.write(format_version, "format_version");
  output.write(parameter_count, "parameter_count");
  output.write(layout_fingerprint, "layout_fingerprint");
  output.write(model_fingerprint, "model_fingerprint");
  output.write(model_state_->model_origin, "model_origin");
  output.write(model_state_->initialization_profile, "initialization_profile");
  output.write(initialization_seed, "initialization_seed");
  output.write(system_kind_, "system_kind");
  output.write(spin_counts, "spin_counts");
  output.write(architecture, "architecture");
  output.write(nuclear_shape, "nuclear_shape");
  output.write(model.cfg.nuclei.x, "nuclear_positions");
  output.write(model.cfg.charges.x, "nuclear_charges");
  output.write(selected_indices, "selected_flat_indices");
  output.write(model.p.flat_values(), "flat_values");

  output.pop();
  output.pop();

  // reportParameters() invokes this hook on rank zero after the final reset.
  // Use a sibling temporary file so readers never observe a partial HDF5 export.
  if (!optimized_parameter_export_.empty())
  {
    const std::filesystem::path destination(optimized_parameter_export_);
    const std::filesystem::path temporary = destination.string() + ".tmp";
    model.p.write(temporary.string());
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error)
    {
      std::filesystem::remove(temporary);
      throw std::runtime_error("Unable to publish PsiFormer DeepQMC parameter export " + destination.string() +
                               ": " + error.message());
    }
    app_log() << "  Wrote optimized PsiFormer parameters to " << destination.string() << std::endl;
  }
}

// Restore a complete model only after validating all identifying metadata.
void PsiFormerWF::readVariationalParameters(hdf_archive& input)
{
  if (!optimization_metadata_->enabled)
    return;
  requireNoPlannedSelectedProposalMutation("variational-parameter restore");
  if (!input.is_group("PsiFormer"))
    throw std::runtime_error("PsiFormer VP file has no PsiFormer object group");

  input.push("PsiFormer", false);
  if (!input.is_group(OptimizableObject::getName()))
    throw std::runtime_error("PsiFormer VP file has no group for component " + OptimizableObject::getName());
  input.push(OptimizableObject::getName(), false);

  const std::vector<int> format_version             = readVector<int>(input, "format_version");
  const std::vector<int> expected_version(PERSISTENCE_VERSION.begin(), PERSISTENCE_VERSION.end());
  if (format_version != expected_version)
  {
    input.pop();
    input.pop();
    throw std::runtime_error(
        "PsiFormer VP format version is incompatible; this build requires version 1.2.0");
  }
  const std::vector<std::uint64_t> parameter_count  = readVector<std::uint64_t>(input, "parameter_count");
  const std::vector<std::uint64_t> spin_counts      = readVector<std::uint64_t>(input, "spin_counts");
  const std::vector<std::uint64_t> architecture     = readVector<std::uint64_t>(input, "architecture");
  const std::vector<std::uint64_t> initialization_seed =
      readVector<std::uint64_t>(input, "initialization_seed");
  const std::vector<std::uint64_t> nuclear_shape    = readVector<std::uint64_t>(input, "nuclear_shape");
  const std::vector<double> nuclear_positions       = readVector<double>(input, "nuclear_positions");
  const std::vector<double> nuclear_charges         = readVector<double>(input, "nuclear_charges");
  const std::vector<std::uint64_t> selected_indices =
      readVector<std::uint64_t>(input, "selected_flat_indices");
  const std::vector<double> flat_values = readVector<double>(input, "flat_values");
  std::string layout_fingerprint;
  std::string model_fingerprint;
  std::string model_origin;
  std::string initialization_profile;
  std::string system_kind;
  input.read(layout_fingerprint, "layout_fingerprint");
  input.read(model_fingerprint, "model_fingerprint");
  input.read(model_origin, "model_origin");
  input.read(initialization_profile, "initialization_profile");
  input.read(system_kind, "system_kind");

  input.pop();
  input.pop();

  std::unique_lock state_lock(model_state_->mutex);
  requireNoPlannedSelectedProposalMutation("variational-parameter restore");
  std::unique_lock metadata_lock(optimization_metadata_->mutex);
  pf::PsiFormer& model = model_state_->model;

    requireEqual(parameter_count, std::vector<std::uint64_t>{model.p.size()}, "parameter count");
    if (layout_fingerprint != model.p.layout_fingerprint())
      throw std::runtime_error("PsiFormer VP layout fingerprint does not match the configured model");
    if (model_origin != model_state_->model_origin)
      throw std::runtime_error("PsiFormer VP construction origin does not match the configured model");
    if (initialization_profile != model_state_->initialization_profile)
      throw std::runtime_error("PsiFormer VP initialization profile does not match the configured model");
    requireEqual(initialization_seed,
                 std::vector<std::uint64_t>{model_state_->initialization_seed},
                 "initialization seed");
    if (model_fingerprint !=
        modelFingerprint(model, model_state_->model_origin, model_state_->initialization_profile,
                         model_state_->initialization_seed))
      throw std::runtime_error("PsiFormer VP model fingerprint does not match the configured model");
    if (system_kind != system_kind_)
      throw std::runtime_error("PsiFormer VP system declaration does not match the configured model");
    requireEqual(spin_counts, std::vector<std::uint64_t>{model.cfg.nup, model.cfg.ndown}, "spin populations");
    requireEqual(architecture,
                 std::vector<std::uint64_t>{model.ndet, model.dim, model.heads, model.blocks},
                 "architecture");
    requireEqual(nuclear_shape,
                 std::vector<std::uint64_t>(model.cfg.nuclei.shape.begin(), model.cfg.nuclei.shape.end()),
                 "nuclear-position shape");
    requireEqual(nuclear_positions, model.cfg.nuclei.x, "nuclear positions");
    requireEqual(nuclear_charges, model.cfg.charges.x, "nuclear charges");
    requireEqual(selected_indices, persistIndices(optimization_metadata_->selected_flat_indices),
                 "selected flat indices");

    if (flat_values != model.p.flat_values())
      model.p.set_flat_values(flat_values);
    const std::size_t parameter_version = model.p.version();

    for (std::size_t local_index = 0;
         local_index < optimization_metadata_->selected_flat_indices.size(); ++local_index)
      optimization_metadata_->variables[local_index] =
          model.p.flat_values()[optimization_metadata_->selected_flat_indices[local_index]];
  invalidateParameterCaches(parameter_version);
  restore_validation_pending_ = true;
}

// Return the parameter version shared by this component and all of its clones.
std::size_t PsiFormerWF::parameterVersion() const
{
  std::shared_lock state_lock(model_state_->mutex);
  return model_state_->model.p.version();
}

// Write a standalone flat parameter file without exposing mutable native storage.
void PsiFormerWF::exportParameters(const std::string& path) const
{
  std::shared_lock state_lock(model_state_->mutex);
  model_state_->model.p.write(path);
}

// Check that the runtime particle sets describe exactly the exported physical system.
void PsiFormerWF::validateSystem(const ParticleSet& electrons,
                                const ParticleSet& ions,
                                const std::string& system_kind)
{
  if (system_kind != "all_electron" && system_kind != "pseudopotential")
    throw std::invalid_argument("PsiFormer system must be all_electron or pseudopotential");
  if (electrons.isSpinor())
    throw std::invalid_argument("PsiFormer does not support spinor electrons or spin-orbit pseudopotentials");
  if (electrons.getLattice().getSuperCellEnum() != SUPERCELL_OPEN)
    throw std::invalid_argument(
        "PsiFormer supports only open-boundary electron particle sets; periodic execution is not implemented");
  if (ions.getLattice().getSuperCellEnum() != SUPERCELL_OPEN)
    throw std::invalid_argument(
        "PsiFormer supports only open-boundary source-ion particle sets; periodic execution is not implemented");

  std::shared_lock state_lock(model_state_->mutex);
  const pf::PsiFormer& model = model_state_->model;
  if (electrons.getTotalNum() != static_cast<int>(model.ne))
    throw std::invalid_argument("PsiFormer runtime electron count does not match the exported model");
  if (electrons.groups() != 2 || electrons.groupsize(0) != static_cast<int>(model.cfg.nup) ||
      electrons.groupsize(1) != static_cast<int>(model.cfg.ndown))
    throw std::invalid_argument("PsiFormer runtime spin populations do not match the exported model");
  if (ions.getTotalNum() != static_cast<int>(model.cfg.nuclei.shape[0]))
    throw std::invalid_argument("PsiFormer runtime nucleus count does not match the exported model");

  const SpeciesSet& species = ions.getSpeciesSet();
  const int charge_index     = species.findAttribute("charge");
  if (charge_index < 0)
    throw std::invalid_argument("PsiFormer source particle set has no charge attribute");
  constexpr double tolerance = 1e-10;
  for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      if (std::abs(ions.R[nucleus][dimension] - model.cfg.nuclei.x[3 * nucleus + dimension]) > tolerance)
        throw std::invalid_argument("PsiFormer runtime nuclear positions do not match the exported model");
    const double runtime_charge = species(charge_index, ions.GroupID[nucleus]);
    if (std::abs(runtime_charge - model.cfg.charges.x[nucleus]) > tolerance)
      throw std::invalid_argument(
          "PsiFormer runtime ionic/effective charges do not match the exported model; use an export trained for this pseudopotential system");
  }

  if ((bound_particle_set_ != &electrons || system_kind_ != system_kind) &&
      (has_proposal_ || proposal_origin_ != ProposalOrigin::NONE))
    throw std::logic_error(
        "Cannot rebind a PsiFormer system while a proposal is pending");
  system_kind_ = system_kind;
  bound_particle_set_ = &electrons;
  app_log() << "  PsiFormer " << WaveFunctionComponent::getName() << ": validated " << system_kind_
            << " system metadata against electron and ion particle sets" << std::endl;
}

// Translate one configuration while the caller retains the model read lock.
pf::Result PsiFormerWF::evaluatePositionsUnderRead(
    const PsiFormerReadTransaction& transaction,
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position,
    EvaluationPurpose purpose,
    int active_gradient_particle)
{
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer read transaction belongs to a different model");
  const pf::PsiFormer& model = transaction.model();
  const PsiFormerSharedState& state = transaction.state();

  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  // Value-only calls use clone-local fixed storage. Compare mode evaluates
  // both implementations and is intended for migration/debug validation.
  std::optional<pf::DirectValueResult> direct_result;
  if (purpose == EvaluationPurpose::VALUE_ONLY && state.direct_value_mode != DirectBackendMode::ORACLE)
  {
    pf::DirectValueWorkspace& workspace = requireDirectValueWorkspace();
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      const auto& position = electron == replaced_particle
          ? (replacement_position ? *replacement_position : p.activeR(electron))
          : p.R[electron];
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension, position[dimension]);
    }
    direct_result = state.direct_value_executor.evaluate(workspace);
    if (direct_result->parameter_version != transaction.parameterVersion())
      throw std::logic_error("PsiFormer direct value observed inconsistent parameters");
    if (state.direct_value_mode == DirectBackendMode::DIRECT)
    {
      pf::Result result;
      result.sign   = direct_result->sign;
      result.logabs = direct_result->logabs;
      result.value  = direct_result->value;
      return result;
    }
  }

  // Score-only calls use a clone-local forward tape and explicit reverse pass.
  // The owning result copy is retained here for compatibility with the current
  // scalar adapter; later direct-output sinks will scatter without this copy.
  std::optional<pf::DirectScoreResult> direct_score_result;
  if (purpose == EvaluationPurpose::SCORE_ONLY &&
      state.direct_score_mode != DirectBackendMode::ORACLE)
  {
    pf::DirectScoreWorkspace& score_workspace = requireDirectScoreWorkspace();
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      const auto& position = electron == replaced_particle
          ? (replacement_position ? *replacement_position : p.activeR(electron))
          : p.R[electron];
      for (int dimension = 0; dimension < 3; ++dimension)
        score_workspace.setPosition(electron, dimension, position[dimension]);
    }
    direct_score_result = state.direct_score_executor.evaluate(score_workspace);
    if (direct_score_result->parameter_version != transaction.parameterVersion())
      throw std::logic_error("PsiFormer direct score observed inconsistent parameters");
    if (state.direct_score_mode == DirectBackendMode::DIRECT)
    {
      pf::Result result;
      result.sign   = direct_score_result->sign;
      result.logabs = direct_score_result->logabs;
      result.value  = direct_score_result->value;
      result.param_gradient.assign(direct_score_result->parameter_score.begin(),
                                   direct_score_result->parameter_score.end());
      return result;
    }
  }

  // Spatial calls propagate only the requested Cartesian lanes.  FULL_VGL
  // carries one gradient vector and one already-contracted Laplacian per
  // electron; the active path carries only three first-derivative lanes.
  std::optional<pf::DirectSpatialResultView> direct_spatial_result;
  if ((purpose == EvaluationPurpose::FULL_SPATIAL ||
       purpose == EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT) &&
      state.direct_spatial_mode != DirectBackendMode::ORACLE)
  {
    pf::DirectSpatialWorkspace& workspace = requireDirectSpatialWorkspace(purpose);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      const auto& position = electron == replaced_particle
          ? (replacement_position ? *replacement_position : p.activeR(electron))
          : p.R[electron];
      for (int dimension = 0; dimension < 3; ++dimension)
        workspace.setPosition(electron, dimension, position[dimension]);
    }

    if (purpose == EvaluationPurpose::FULL_SPATIAL)
      direct_spatial_result = state.direct_spatial_executor.evaluateFull(workspace);
    else
      direct_spatial_result = state.direct_spatial_executor.evaluateActive(
          workspace, static_cast<std::size_t>(active_gradient_particle));
    if (direct_spatial_result->parameter_version != transaction.parameterVersion())
      throw std::logic_error("PsiFormer direct spatial evaluation observed inconsistent parameters");

    if (state.direct_spatial_mode == DirectBackendMode::DIRECT)
    {
      pf::Result result;
      result.sign   = direct_spatial_result->sign;
      result.logabs = direct_spatial_result->logabs;
      result.value  = direct_spatial_result->value;
      if (purpose == EvaluationPurpose::FULL_SPATIAL)
      {
        result.gradient.assign(direct_spatial_result->gradient.begin(),
                               direct_spatial_result->gradient.end());
        result.lap_log.assign(direct_spatial_result->lap_log.begin(),
                              direct_spatial_result->lap_log.end());
        result.lap_ratio.assign(direct_spatial_result->lap_ratio.begin(),
                                direct_spatial_result->lap_ratio.end());
      }
      else
        result.active_gradient.assign(direct_spatial_result->gradient.begin(),
                                      direct_spatial_result->gradient.end());
      return result;
    }
  }

  // For a particle-by-particle proposal, substitute only the active position.
  // All other positions remain at the last accepted ParticleSet coordinates.
  pf::Tensor positions({static_cast<std::size_t>(p.getTotalNum()), 3});
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : p.activeR(electron))
        : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      positions.x[3 * electron + dimension] = position[dimension];
  }

  pf::EvaluationRequest request;
  request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
  switch (purpose)
  {
  case EvaluationPurpose::VALUE_ONLY:
    request.spatial_derivatives = pf::SpatialDerivativeRequest::NONE;
    break;
  case EvaluationPurpose::FULL_SPATIAL:
    request.spatial_derivatives = pf::SpatialDerivativeRequest::FULL_VGL;
    break;
  case EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT:
    request.spatial_derivatives = pf::SpatialDerivativeRequest::ACTIVE_ELECTRON_GRADIENT;
    request.active_electron     = active_gradient_particle;
    break;
  case EvaluationPurpose::SCORE_ONLY:
    request.spatial_derivatives   = pf::SpatialDerivativeRequest::NONE;
    request.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_ONLY;
    break;
  case EvaluationPurpose::SCORE_AND_KINETIC:
    request.spatial_derivatives   = pf::SpatialDerivativeRequest::FULL_VGL;
    request.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_AND_KINETIC;
    break;
  }

  std::vector<double> total_log_gradient;
  if (purpose == EvaluationPurpose::SCORE_AND_KINETIC)
  {
    total_log_gradient.reserve(3 * p.getTotalNum());
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        total_log_gradient.push_back(std::real(p.G[electron][dimension]));
    request.total_log_gradient = &total_log_gradient;
  }

  pf::Result result = model.evaluate(positions, request);
  if (direct_result)
  {
    const double scale = std::max(std::abs(result.logabs), std::abs(direct_result->logabs));
    if (result.sign != direct_result->sign ||
        std::abs(result.logabs - direct_result->logabs) > 2.0e-11 * (1.0 + scale))
      throw std::runtime_error("PsiFormer direct value result differs from the native oracle");
  }
  if (direct_score_result)
  {
    const double scale = std::max(std::abs(result.logabs), std::abs(direct_score_result->logabs));
    if (result.sign != direct_score_result->sign ||
        std::abs(result.logabs - direct_score_result->logabs) > 2.0e-11 * (1.0 + scale) ||
        result.param_gradient.size() != direct_score_result->parameter_score.size)
      throw std::runtime_error("PsiFormer direct score result differs from the native oracle");
    for (std::size_t parameter = 0; parameter < result.param_gradient.size(); ++parameter)
      if (std::abs(result.param_gradient[parameter] - direct_score_result->parameter_score[parameter]) > 2.0e-8)
        throw std::runtime_error("PsiFormer direct parameter score differs from the native oracle");
  }
  if (direct_spatial_result)
  {
    const double scale = std::max(std::abs(result.logabs), std::abs(direct_spatial_result->logabs));
    if (result.sign != direct_spatial_result->sign ||
        std::abs(result.logabs - direct_spatial_result->logabs) > 3.0e-10 * (1.0 + scale))
      throw std::runtime_error("PsiFormer direct spatial value differs from the native oracle");

    const std::vector<double>& oracle_gradient = purpose == EvaluationPurpose::FULL_SPATIAL
        ? result.gradient
        : result.active_gradient;
    if (oracle_gradient.size() != direct_spatial_result->gradient.size())
      throw std::runtime_error("PsiFormer direct spatial gradient has the wrong size");
    for (std::size_t coordinate = 0; coordinate < oracle_gradient.size(); ++coordinate)
      if (std::abs(oracle_gradient[coordinate] - direct_spatial_result->gradient[coordinate]) > 2.0e-7)
        throw std::runtime_error("PsiFormer direct spatial gradient differs from the native oracle");

    if (purpose == EvaluationPurpose::FULL_SPATIAL)
    {
      if (result.lap_log.size() != direct_spatial_result->lap_log.size() ||
          result.lap_ratio.size() != direct_spatial_result->lap_ratio.size())
        throw std::runtime_error("PsiFormer direct spatial Laplacian has the wrong size");
      for (std::size_t electron = 0; electron < result.lap_log.size(); ++electron)
        if (std::abs(result.lap_log[electron] - direct_spatial_result->lap_log[electron]) > 3.0e-7 ||
            std::abs(result.lap_ratio[electron] - direct_spatial_result->lap_ratio[electron]) > 3.0e-7)
          throw std::runtime_error("PsiFormer direct spatial Laplacian differs from the native oracle");
    }
  }
  return result;
}

pf::DirectValueResult PsiFormerWF::evaluateDirectValuePositionsUnderRead(
    const PsiFormerReadTransaction& transaction,
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position)
{
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer value transaction belongs to a different model");
  const pf::PsiFormer& model = transaction.model();
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  pf::DirectValueWorkspace& workspace = requireDirectValueWorkspace();
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : p.activeR(electron))
        : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      workspace.setPosition(electron, dimension, position[dimension]);
  }
  const pf::DirectValueResult result =
      transaction.state().direct_value_executor.evaluate(workspace);
  if (result.parameter_version != transaction.parameterVersion())
    throw std::logic_error("PsiFormer direct value observed inconsistent parameters");
  return result;
}

pf::DirectSpatialResultView PsiFormerWF::evaluateDirectSpatialPositionsUnderRead(
    const PsiFormerReadTransaction& transaction,
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position,
    EvaluationPurpose purpose,
    int active_gradient_particle)
{
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer spatial transaction belongs to a different model");
  if (purpose != EvaluationPurpose::FULL_SPATIAL &&
      purpose != EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT)
    throw std::logic_error("PsiFormer direct spatial adapter received a non-spatial request");

  const pf::PsiFormer& model = transaction.model();
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  pf::DirectSpatialWorkspace& workspace = requireDirectSpatialWorkspace(purpose);
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : p.activeR(electron))
        : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      workspace.setPosition(electron, dimension, position[dimension]);
  }

  const pf::DirectSpatialResultView result = purpose == EvaluationPurpose::FULL_SPATIAL
      ? transaction.state().direct_spatial_executor.evaluateFull(workspace)
      : transaction.state().direct_spatial_executor.evaluateActive(
            workspace, static_cast<std::size_t>(active_gradient_particle));
  if (result.parameter_version != transaction.parameterVersion())
    throw std::logic_error("PsiFormer direct spatial evaluation observed inconsistent parameters");
  return result;
}

// Delay scalar forward-buffer construction until a value path actually runs.
pf::DirectValueWorkspace& PsiFormerWF::requireDirectValueWorkspace()
{
  if (batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer scalar value evaluation is not admitted by the explicit batch plan");
  if (!direct_value_workspace_)
    direct_value_workspace_ = model_state_->direct_value_executor.makeWorkspace();
  return *direct_value_workspace_;
}

// Keep the much larger full-VGL tape independent from the compact active-gradient tape.
pf::DirectSpatialWorkspace& PsiFormerWF::requireDirectSpatialWorkspace(EvaluationPurpose purpose)
{
  if (batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer scalar spatial evaluation is not admitted by the explicit batch plan");
  switch (purpose)
  {
  case EvaluationPurpose::FULL_SPATIAL:
    if (!direct_full_spatial_workspace_)
      direct_full_spatial_workspace_ = model_state_->direct_spatial_executor.makeWorkspace(
          pf::DirectSpatialMode::FULL_VGL);
    return *direct_full_spatial_workspace_;
  case EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT:
    if (!direct_active_spatial_workspace_)
      direct_active_spatial_workspace_ = model_state_->direct_spatial_executor.makeWorkspace(
          pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT);
    return *direct_active_spatial_workspace_;
  default:
    throw std::logic_error("PsiFormer spatial workspace requires a spatial evaluation purpose");
  }
}

// Batch scratch is needed only by scalar APIs that evaluate several related configurations.
pf::DirectBatchWorkspace& PsiFormerWF::requireDirectBatchWorkspace()
{
  if (batch_execution_plan_)
  {
    if (!batch_execution_plan_.plan().requirements().requires(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY))
      throw std::logic_error(
          "PsiFormer scalar VALUE compatibility is not admitted by the explicit batch plan");
    if (!prepared_clone_batch_execution_plan_.sameBinding(batch_execution_plan_) || !direct_batch_workspace_ ||
        !direct_batch_workspace_->hasCapacityPlan())
      throw std::logic_error(
          "PsiFormer scalar VALUE workspace was not prepared for the bound batch plan");
  }
  else if (!direct_batch_workspace_)
    direct_batch_workspace_ = model_state_->direct_batch_executor.makeWorkspace();
  return *direct_batch_workspace_;
}

// Validate planned scalar extents before any output or workspace state changes.
PsiFormerWF::ValueType* PsiFormerWF::requirePlannedScalarValuePublication(
    std::size_t configuration_count,
    std::size_t output_count,
    const char* operation)
{
  if (!batch_execution_plan_)
    return nullptr;

  pf::DirectBatchWorkspace& workspace = requireDirectBatchWorkspace();
  const pf::DirectBatchCapacityPlan& capacity = workspace.capacityPlan();
  if (configuration_count > capacity.logical.value_dense ||
      output_count > scalar_value_publication_.size())
    throw std::length_error(std::string("PsiFormer ") + operation +
                            " exceeds the planned scalar VALUE envelope");
  return scalar_value_publication_.data();
}

// Scalar direct value and spatial tapes are not part of the hard-plan owner set.
void PsiFormerWF::requireUnplannedScalarEvaluation(const char* operation) const
{
  if (batch_execution_plan_)
    throw std::logic_error(std::string("PsiFormer ") + operation +
                           " is not admitted as a scalar operation by the explicit batch plan");
}

// Planned nonlocal work must use flattened crowd descriptors and bounded resource storage.
void PsiFormerWF::requireNoPlannedEcpScalarDispatch(const char* operation) const
{
  if (batch_execution_plan_ &&
      batchExecutionModeIsRequired(batch_execution_plan_.plan().requirements(),
                                   BatchExecutionMode::ECP_OUTER))
    throw std::logic_error(std::string("PsiFormer ") + operation +
                           " must use flattened multiwalker ECP dispatch under the explicit batch plan");
}

// Score and kinetic reverse tapes remain crowd-owned under an explicit plan.
void PsiFormerWF::requireUnplannedScalarDerivative(const char* operation) const
{
  if (batch_execution_plan_)
    throw std::logic_error(std::string("PsiFormer ") + operation +
                           " is not admitted as a clone-local operation by the explicit batch plan");
}

// Lazily allocate score scratch for scalar calls, keeping inference-only clones lightweight.
pf::DirectScoreWorkspace& PsiFormerWF::requireDirectScoreWorkspace()
{
  requireUnplannedScalarDerivative("parameter-score evaluation");
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer direct score workspace requires an optimizable component");
  if (!direct_score_workspace_)
    direct_score_workspace_ = model_state_->direct_score_executor.makeWorkspace();
  return *direct_score_workspace_;
}

// Lazily allocate the much larger kinetic tape only for an actual scalar reverse call.
pf::DirectKineticWorkspace& PsiFormerWF::requireDirectKineticWorkspace()
{
  requireUnplannedScalarDerivative("kinetic-parameter evaluation");
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer direct kinetic workspace requires an optimizable component");
  if (!direct_kinetic_workspace_)
    direct_kinetic_workspace_ = model_state_->direct_kinetic_executor.makeWorkspace();
  return *direct_kinetic_workspace_;
}

// Lazily allocate the scalar adapter's complete TrialWaveFunction drift buffer.
std::vector<double>& PsiFormerWF::requireDirectTotalLogGradient()
{
  requireUnplannedScalarDerivative("kinetic total-drift preparation");
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer total-drift storage requires an optimizable component");
  const std::size_t required_size =
      3 * model_state_->execution_plan.modelShape().electrons();
  if (direct_total_log_gradient_.empty())
    direct_total_log_gradient_.resize(required_size);
  if (direct_total_log_gradient_.size() != required_size)
    throw std::logic_error("PsiFormer direct total-drift buffer has the wrong size");
  return direct_total_log_gradient_;
}

pf::DirectScoreResult PsiFormerWF::evaluateDirectScorePositionsUnderRead(
    const PsiFormerReadTransaction& transaction,
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position,
    pf::DirectScoreWorkspace& score_workspace)
{
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer score transaction belongs to a different model");
  const pf::PsiFormer& model = transaction.model();
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer direct score evaluation requires an optimizable component");
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : p.activeR(electron))
        : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      score_workspace.setPosition(electron, dimension, position[dimension]);
  }
  const pf::DirectScoreResult result =
      transaction.state().direct_score_executor.evaluate(score_workspace);
  if (result.parameter_version != transaction.parameterVersion())
    throw std::logic_error("PsiFormer direct score observed inconsistent parameters");
  return result;
}

// Evaluate a full accepted configuration and accumulate its spatial derivatives.
PsiFormerWF::LogValue PsiFormerWF::evaluateLog(const ParticleSet& p,
                                               ParticleSet::ParticleGradient& g,
                                               ParticleSet::ParticleLaplacian& l)
{
  requireUnplannedScalarEvaluation("evaluateLog");
  requireNoSelectedParticleProposal("evaluateLog");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  return evaluateLogUnderRead(transaction, p, g, l);
}

PsiFormerWF::LogValue PsiFormerWF::evaluateLogUnderRead(
    const PsiFormerReadTransaction& transaction,
    const ParticleSet& p,
    ParticleSet::ParticleGradient& g,
    ParticleSet::ParticleLaplacian& l)
{
  if (g.size() != static_cast<std::size_t>(p.getTotalNum()) ||
      l.size() != static_cast<std::size_t>(p.getTotalNum()))
    throw std::invalid_argument("PsiFormer evaluateLog output arrays have the wrong size");

  // Keep the scatter independent of result ownership.  Production direct mode
  // passes workspace-backed views; oracle and compare retain pf::Result.
  auto scatter = [&](double sign, double logabs, const auto& gradient, const auto& lap_log) {
    const std::size_t electrons = static_cast<std::size_t>(p.getTotalNum());
    if (gradient.size() != 3 * electrons || lap_log.size() != electrons)
      throw std::logic_error("PsiFormer full-spatial result has the wrong shape");
    resizeAcceptedSpatialStorage(p.getTotalNum());
    current_sign_ = sign;
    // QMCPACK represents a negative real wavefunction by adding pi to its complex
    // phase.
    log_value_ = LogValue(logabs, sign < 0 ? M_PI : 0.0);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        accepted_gradient_[electron][dimension] = gradient[3 * electron + dimension];
      accepted_laplacian_[electron] = lap_log[electron];
    }
    accepted_configuration_identity_ = configurationIdentity(p);
    accepted_parameter_version_      = transaction.parameterVersion();
    accepted_state_requirement_      = AcceptedStateRequirement::FULL_SPATIAL;
    accepted_value_valid_            = true;
    clearProposalState();
    accumulateAcceptedSpatial(g, l);
    return log_value_;
  };

  if (transaction.state().direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositionsUnderRead(
        transaction, p, -1, nullptr, EvaluationPurpose::FULL_SPATIAL, -1);
    return scatter(result.sign, result.logabs, result.gradient, result.lap_log);
  }

  const pf::Result result = evaluatePositionsUnderRead(
      transaction, p, -1, nullptr, EvaluationPurpose::FULL_SPATIAL, -1);
  return scatter(result.sign, result.logabs, result.gradient, result.lap_log);
}

// Evaluate all accepted crowd configurations under one parameter read lock and
// scatter complete spatial outputs directly into caller-owned accumulators.
void PsiFormerWF::mw_evaluateLog(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVector<ParticleSet::ParticleGradient>& gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list) const
{
  // Preserve the historical lazy/oracle implementation byte-for-byte behind
  // an explicit no-policy branch.
  if (!batch_execution_plan_)
  {
    if (wfc_list.size() != p_list.size() || wfc_list.size() != gradient_list.size() ||
        wfc_list.size() != laplacian_list.size())
      throw std::invalid_argument("PsiFormer mw_evaluateLog list sizes do not match");
    if (wfc_list.empty())
      return;

    const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
    auto& resource     = requireMultiWalkerResource(wfc_list);
    PsiFormerReadTransaction transaction(*leader.model_state_);
    const std::size_t parameter_version = transaction.parameterVersion();
    const std::size_t electrons =
        transaction.state().execution_plan.modelShape().electrons();
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      wfc_list.getCastedElement<PsiFormerWF>(walker).requireNoSelectedParticleProposal(
          "mw_evaluateLog");
      component.synchronizeParameterVersion(parameter_version);
      if (gradient_list[walker].get().size() != electrons ||
          laplacian_list[walker].get().size() != electrons)
        throw std::invalid_argument("PsiFormer mw_evaluateLog output arrays have the wrong size");
      component.resizeAcceptedSpatialStorage(electrons);
    }

    // Preserve oracle and compare validation while binding the complete crowd to
    // the same model transaction.  Owning VGL results stage all rows before the
    // no-fail publication pass.
    if (transaction.state().direct_spatial_mode != DirectBackendMode::DIRECT)
    {
      std::vector<pf::Result> staged_results;
      staged_results.reserve(wfc_list.size());
      for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      {
        auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
        staged_results.push_back(component.evaluatePositionsUnderRead(
            transaction, p_list[walker], -1, nullptr,
            EvaluationPurpose::FULL_SPATIAL, -1));
        if (staged_results.back().gradient.size() != 3 * electrons ||
            staged_results.back().lap_log.size() != electrons)
          throw std::logic_error("PsiFormer multiwalker oracle VGL result has the wrong shape");
      }

      for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      {
        auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
        const pf::Result& result = staged_results[walker];
        component.current_sign_ = result.sign;
        component.log_value_    = makeLogValue(result.sign, result.logabs);
        for (std::size_t electron = 0; electron < electrons; ++electron)
        {
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            component.accepted_gradient_[electron][dimension] =
                result.gradient[3 * electron + dimension];
          component.accepted_laplacian_[electron] = result.lap_log[electron];
        }
        component.accepted_configuration_identity_ = configurationIdentity(p_list[walker]);
        component.accepted_parameter_version_      = parameter_version;
        component.accepted_state_requirement_      = AcceptedStateRequirement::FULL_SPATIAL;
        component.accepted_value_valid_            = true;
        component.clearProposalState();
        component.accumulateAcceptedSpatial(gradient_list[walker].get(),
                                            laplacian_list[walker].get());
      }
      return;
    }

    auto& batch = *resource.batch_workspace;
    batch.resize(pf::DirectBatchMode::FULL_VGL, wfc_list.size());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      packBatchConfiguration(batch, walker, p_list[walker]);
    }

    const pf::DirectBatchSpatialResultView result =
        transaction.state().direct_batch_executor.evaluateFull(batch);
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      if (result.parameter_version[walker] != parameter_version)
        throw std::logic_error("PsiFormer full-spatial batch observed inconsistent parameters");

    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      component.current_sign_ = result.sign[walker];
      component.log_value_ = makeLogValue(result.sign[walker], result.logabs[walker]);
      auto& gradient = gradient_list[walker].get();
      auto& laplacian = laplacian_list[walker].get();
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          component.accepted_gradient_[electron][dimension] =
              result.gradient[walker * result.gradient_stride + 3 * electron + dimension];
        component.accepted_laplacian_[electron] =
            result.lap_log[walker * result.laplacian_stride + electron];
      }
      component.accepted_configuration_identity_ = configurationIdentity(p_list[walker]);
      component.accepted_parameter_version_      = parameter_version;
      component.accepted_state_requirement_      = AcceptedStateRequirement::FULL_SPATIAL;
      component.accepted_value_valid_            = true;
      component.clearProposalState();
      component.accumulateAcceptedSpatial(gradient, laplacian);
    }
    return;
  }

  const std::size_t walker_count = wfc_list.size();
  PlannedRuntimeRequest request;
  request.operation            = PlannedOperation::FULL_VGL;
  request.live_walkers         = walker_count;
  request.dense_configurations = walker_count;
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (gradient_list.size() != walker_count ||
      laplacian_list.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned FULL_VGL output-list sizes do not match the crowd");

  const std::size_t electrons = access.participant.plan().particleCount();
  PsiFormerMultiWalkerResource& resource = access.resource;
  resource.requireFullVGLStaging(walker_count);
  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;

  // Phase A: prove caller shapes, finite additive seeds, and complete storage
  // separation before changing workspace contents or opening a model read.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& gradient  = gradient_list[lane].get();
    const auto& laplacian = laplacian_list[lane].get();
    if (gradient.size() != electrons || laplacian.size() != electrons)
      throw std::invalid_argument(
          "PsiFormer planned FULL_VGL output arrays have the wrong shape");
    for (std::size_t electron = 0; electron < electrons; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!isFiniteWavefunctionValue(gradient[electron][dimension]))
          throw std::invalid_argument(
              "PsiFormer planned FULL_VGL gradient seed is non-finite");
      if (!isFiniteWavefunctionValue(laplacian[electron]))
        throw std::invalid_argument(
            "PsiFormer planned FULL_VGL Laplacian seed is non-finite");
    }
  }

  const auto output_range = [&](std::size_t lane, bool gradient) {
    return gradient
        ? checkedMemoryRange(gradient_list[lane].get().data(), electrons,
                             "PsiFormer planned FULL_VGL gradient range overflowed")
        : checkedMemoryRange(laplacian_list[lane].get().data(), electrons,
                             "PsiFormer planned FULL_VGL Laplacian range overflowed");
  };
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (int output_kind = 0; output_kind < 2; ++output_kind)
    {
      const bool is_gradient = output_kind == 0;
      const CheckedMemoryRange output = output_range(lane, is_gradient);
      const void* output_data = is_gradient
          ? static_cast<const void*>(gradient_list[lane].get().data())
          : static_cast<const void*>(laplacian_list[lane].get().data());
      const std::size_t output_bytes = output.end - output.begin;

      for (std::size_t prior_lane = 0; prior_lane <= lane; ++prior_lane)
        for (int prior_kind = 0; prior_kind < 2; ++prior_kind)
        {
          if (prior_lane == lane && prior_kind >= output_kind)
            continue;
          if (memoryRangesOverlap(
                  output, output_range(prior_lane, prior_kind == 0)))
            throw std::invalid_argument(
                "PsiFormer planned FULL_VGL caller outputs overlap");
        }

      for (std::size_t particle_lane = 0; particle_lane < walker_count;
           ++particle_lane)
      {
        const auto& soa_positions =
            p_list[particle_lane].getCoordinates().getAllParticlePos();
        if (soa_positions.capacity() >
            std::numeric_limits<std::size_t>::max() / 3)
          throw std::length_error(
              "PsiFormer ParticleSet SoA position range overflowed");
        const CheckedMemoryRange position_range = checkedMemoryRange(
            p_list[particle_lane].R.data(), electrons,
            "PsiFormer ParticleSet position range overflowed");
        const CheckedMemoryRange soa_position_range = checkedMemoryRange(
            soa_positions.data(), 3 * soa_positions.capacity(),
            "PsiFormer ParticleSet SoA position range overflowed");
        const CheckedMemoryRange active_position_range = checkedMemoryRange(
            std::addressof(p_list[particle_lane].getActivePos()), std::size_t{1},
            "PsiFormer ParticleSet active-position range overflowed");
        if (memoryRangesOverlap(output, position_range) ||
            memoryRangesOverlap(output, soa_position_range) ||
            memoryRangesOverlap(output, active_position_range))
          throw std::invalid_argument(
              "PsiFormer planned FULL_VGL output aliases ParticleSet position storage");
        for (int particle_kind = 0; particle_kind < 2; ++particle_kind)
        {
          const bool particle_gradient = particle_kind == 0;
          const CheckedMemoryRange particle_range = particle_gradient
              ? checkedMemoryRange(p_list[particle_lane].G.data(), electrons,
                                   "PsiFormer ParticleSet gradient range overflowed")
              : checkedMemoryRange(p_list[particle_lane].L.data(), electrons,
                                   "PsiFormer ParticleSet Laplacian range overflowed");
          if (!memoryRangesOverlap(output, particle_range))
            continue;
          const bool exact_same_lane = particle_lane == lane &&
              particle_gradient == is_gradient &&
              sameMemoryRange(output, particle_range);
          if (!exact_same_lane)
            throw std::invalid_argument(
                "PsiFormer planned FULL_VGL output aliases incompatible ParticleSet storage");
        }
      }

      for (std::size_t component_lane = 0; component_lane < walker_count;
           ++component_lane)
      {
        const auto& component =
            static_cast<const PsiFormerWF&>(wfc_list[component_lane]);
        for (const CheckedMemoryRange internal : {
                 checkedMemoryRange(
                     component.accepted_gradient_.data(), electrons,
                     "PsiFormer accepted gradient range overflowed"),
                 checkedMemoryRange(
                     component.accepted_laplacian_.data(), electrons,
                     "PsiFormer accepted Laplacian range overflowed"),
                 checkedMemoryRange(
                     component.proposed_gradient_.data(), electrons,
                     "PsiFormer proposed gradient range overflowed"),
                 checkedMemoryRange(
                     component.proposed_laplacian_.data(), electrons,
                     "PsiFormer proposed Laplacian range overflowed")})
          if (memoryRangesOverlap(output, internal))
            throw std::invalid_argument(
                "PsiFormer planned FULL_VGL output aliases component state");
      }
      if (resource.overlapsStagingStorage(output_data, output_bytes) ||
          batch.overlapsStorage(output_data, output_bytes))
        throw std::invalid_argument(
            "PsiFormer planned FULL_VGL output aliases prepared scratch");
    }

  for (std::size_t lane = 0; lane < walker_count; ++lane)
    resource.configuration_identities[lane] = configurationIdentity(p_list[lane]);

  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_spatial_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned FULL_VGL requires the direct spatial backend");
  const std::size_t parameter_version = transaction.parameterVersion();
  batch.resize(pf::DirectBatchMode::FULL_VGL, walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    packBatchConfiguration(batch, lane, p_list[lane]);

  const pf::DirectBatchSpatialResultView result =
      transaction.state().direct_batch_executor.evaluateFull(batch);
  if (!batch.ownsSpatialResult(result, pf::DirectSpatialMode::FULL_VGL,
                               walker_count))
    throw std::logic_error(
        "PsiFormer planned FULL_VGL result is not the exact workspace-owned view");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    if (result.parameter_version[lane] != parameter_version)
      throw std::logic_error(
          "PsiFormer planned FULL_VGL observed inconsistent parameters");
    if ((result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]))
      throw std::runtime_error(
          "PsiFormer planned FULL_VGL produced an invalid sign or log magnitude");
    resource.staged_signs[lane] = result.sign[lane];
    resource.staged_log_magnitudes[lane] = result.logabs[lane];
    for (std::size_t electron = 0; electron < electrons; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!psiformer::determinant::isFiniteReal(
                result.gradient[lane * result.gradient_stride +
                                3 * electron + dimension]))
          throw std::runtime_error(
              "PsiFormer planned FULL_VGL produced a non-finite gradient");
      if (!psiformer::determinant::isFiniteReal(
              result.lap_log[lane * result.laplacian_stride + electron]) ||
          !psiformer::determinant::isFiniteReal(
              result.lap_ratio[lane * result.laplacian_stride + electron]))
        throw std::runtime_error(
            "PsiFormer planned FULL_VGL produced a non-finite Laplacian");
    }
  }

  // Phase B: revalidate every borrowed identity and result immediately before
  // the last possibly failing work.  Stale/invalid accepted caches are allowed:
  // this successful operation authoritatively replaces them at the current version.
  if (!batch.ownsSpatialResult(result, pf::DirectSpatialMode::FULL_VGL,
                               walker_count) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned FULL_VGL storage changed during evaluation");
  resource.requireFullVGLStaging(walker_count);
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned FULL_VGL runtime evidence changed during evaluation");
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (component.model_state_.get() != model_state_.get() ||
        component.optimization_metadata_.get() != optimization_metadata_.get() ||
        component.bound_particle_set_ != &p_list[lane] ||
        component.acquired_crowd_leader_ != this ||
        component.acquired_lane_index_ != lane ||
        component.acquired_crowd_size_ != walker_count ||
        !component.batch_execution_plan_.sameBinding(access.participant) ||
        !component.hasPreparedBatchExecutionClone(access.participant) ||
        result.parameter_version[lane] != parameter_version ||
        configurationIdentity(p_list[lane]) !=
            resource.configuration_identities[lane])
      throw std::logic_error(
          "PsiFormer planned FULL_VGL lane identity changed during evaluation");
  }

  // Friend-only regression seam for the strong guarantee at the latest
  // throwing boundary: evaluator scratch is live, but nothing is published.
  if (fail_planned_full_vgl_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned FULL_VGL pre-publication failure");

  // Dry-run every additive operation last.  Nothing after this loop may fail.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (std::size_t electron = 0; electron < electrons; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const ValueType future = gradient_list[lane].get()[electron][dimension] +
            static_cast<ValueType>(
                result.gradient[lane * result.gradient_stride +
                                3 * electron + dimension]);
        if (!isFiniteWavefunctionValue(future))
          throw std::overflow_error(
              "PsiFormer planned FULL_VGL gradient sum is non-finite");
      }
      const ValueType future = laplacian_list[lane].get()[electron] +
          static_cast<ValueType>(
              result.lap_log[lane * result.laplacian_stride + electron]);
      if (!isFiniteWavefunctionValue(future))
        throw std::overflow_error(
            "PsiFormer planned FULL_VGL Laplacian sum is non-finite");
    }

  // Dedicated mechanically nonthrowing publication: accepted arrays, caller
  // additions, metadata, then the accepted-valid marker for every lane.
  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          component.accepted_gradient_[electron][dimension] =
              static_cast<ValueType>(
                  result.gradient[lane * result.gradient_stride +
                                  3 * electron + dimension]);
        component.accepted_laplacian_[electron] = static_cast<ValueType>(
            result.lap_log[lane * result.laplacian_stride + electron]);
      }
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_list[lane].get()[electron][dimension] +=
              static_cast<ValueType>(
                  result.gradient[lane * result.gradient_stride +
                                  3 * electron + dimension]);
        laplacian_list[lane].get()[electron] += static_cast<ValueType>(
            result.lap_log[lane * result.laplacian_stride + electron]);
      }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      component.current_sign_ = resource.staged_signs[lane];
      component.log_value_ = makeLogValue(
          resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane]);
      component.accepted_configuration_identity_ =
          resource.configuration_identities[lane];
      component.accepted_parameter_version_ = parameter_version;
      component.accepted_state_requirement_ =
          AcceptedStateRequirement::FULL_SPATIAL;
      component.observed_parameter_version_ = parameter_version;
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).accepted_value_valid_ = true;
  };
  static_assert(noexcept(publish()));
  publish();
}

void PsiFormerWF::mw_evaluateGL(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVector<ParticleSet::ParticleGradient>& gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list,
    bool) const
{ mw_evaluateLog(wfc_list, p_list, gradient_list, laplacian_list); }

// Evaluate complete selected-electron proposals without modifying accepted ParticleSets.
void PsiFormerWF::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    std::vector<LogValue>& log_ratios,
    const RefVector<ParticleSet::ParticleGradient>& proposed_gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& proposed_laplacian_list) const
{
  const std::size_t walker_count = wfc_list.size();
  if (!batch_execution_plan_)
  {
    if (p_list.size() != walker_count || moves.walkerCount() != walker_count ||
        log_ratios.size() != walker_count || proposed_gradient_list.size() != walker_count ||
        proposed_laplacian_list.size() != walker_count)
      throw std::invalid_argument(
          "PsiFormer selected-electron proposal has inconsistent walker counts");
    moves.validateFor(p_list);
    if (walker_count == 0)
      return;

    const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
    auto& resource     = requireMultiWalkerResource(wfc_list);
    PsiFormerReadTransaction transaction(*leader.model_state_);
    const std::size_t parameter_version = transaction.parameterVersion();
    const std::size_t electron_count =
        transaction.state().execution_plan.modelShape().electrons();
    const std::uint64_t descriptor_fingerprint = moves.fingerprint();
    constexpr std::size_t no_batch_slot = std::numeric_limits<std::size_t>::max();

    std::vector<std::uint64_t> proposed_identities(walker_count);
    std::vector<std::size_t> batch_slots(walker_count, no_batch_slot);
    std::vector<double> proposed_signs(walker_count);
    std::vector<double> proposed_logabs(walker_count);
    std::vector<LogValue> staged_log_ratios(walker_count);
    resource.walker_indices.clear();

    // Validate every walker and allocate all persistent proposal storage before
    // evaluating or publishing any pending state.
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      component.synchronizeParameterVersion(parameter_version);
      if (component.has_proposal_)
        throw std::logic_error(
            "PsiFormer cannot start a selected-electron proposal before resolving the previous proposal");
      if (!component.acceptedStateMatches(
              p_list[walker], parameter_version, AcceptedStateRequirement::VALUE_ONLY))
        throw std::logic_error(
            "PsiFormer selected-electron proposal requested before evaluateLog");
      if (proposed_gradient_list[walker].get().size() < electron_count ||
          proposed_laplacian_list[walker].get().size() < electron_count)
        throw std::invalid_argument(
            "PsiFormer selected-electron proposal output arrays are too small");

      component.resizeProposedSpatialStorage(electron_count);
      proposed_identities[walker] = configurationIdentity(p_list[walker], moves.slice(walker));
      const bool reuse_accepted = proposed_identities[walker] == component.accepted_configuration_identity_ &&
          component.acceptedStateMatches(
              p_list[walker], parameter_version, AcceptedStateRequirement::FULL_SPATIAL);
      if (reuse_accepted)
      {
        proposed_signs[walker]  = component.current_sign_;
        proposed_logabs[walker] = std::real(component.log_value_);
      }
      else
      {
        batch_slots[walker] = resource.walker_indices.size();
        resource.walker_indices.push_back(walker);
      }
    }

    auto& batch = *resource.batch_workspace;
    pf::DirectBatchSpatialResultView batch_result;
    const DirectBackendMode spatial_mode = transaction.state().direct_spatial_mode;
    if (!resource.walker_indices.empty() && spatial_mode != DirectBackendMode::ORACLE)
    {
      batch.resize(pf::DirectBatchMode::FULL_VGL, resource.walker_indices.size());
      for (std::size_t slot = 0; slot < resource.walker_indices.size(); ++slot)
      {
        const std::size_t walker = resource.walker_indices[slot];
        packBatchConfiguration(batch, slot, p_list[walker], moves.slice(walker));
      }
      batch_result = transaction.state().direct_batch_executor.evaluateFull(batch);
    }

    std::vector<pf::Result> oracle_results;
    if (!resource.walker_indices.empty() && spatial_mode != DirectBackendMode::DIRECT)
    {
      oracle_results.reserve(resource.walker_indices.size());
      for (const std::size_t walker : resource.walker_indices)
      {
        const auto selected_moves = moves.slice(walker);
        pf::Tensor positions({electron_count, 3});
        std::size_t selected = 0;
        for (std::size_t electron = 0; electron < electron_count; ++electron)
        {
          const bool replaced = selected < selected_moves.size() &&
              static_cast<std::size_t>(selected_moves.particleIndex(selected)) == electron;
          const auto& position = replaced
              ? selected_moves.proposedPosition(selected)
              : p_list[walker].R[electron];
          if (replaced)
            ++selected;
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            positions.x[3 * electron + dimension] = position[dimension];
        }

        pf::EvaluationRequest request;
        request.spatial_derivatives = pf::SpatialDerivativeRequest::FULL_VGL;
        request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
        oracle_results.push_back(transaction.model().evaluate(positions, request));
        const pf::Result& oracle = oracle_results.back();
        if (oracle.gradient.size() != 3 * electron_count ||
            oracle.lap_log.size() != electron_count ||
            oracle.lap_ratio.size() != electron_count)
          throw std::logic_error(
              "PsiFormer selected-electron oracle VGL result has the wrong shape");
      }
    }

    // Validate the complete native result before any clone advertises a pending transaction.
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const std::size_t slot = batch_slots[walker];
      if (slot != no_batch_slot)
      {
        if (spatial_mode != DirectBackendMode::ORACLE)
        {
          if (batch_result.parameter_version[slot] != parameter_version)
            throw std::logic_error(
                "PsiFormer selected-electron batch observed inconsistent parameters");
          proposed_signs[walker]  = batch_result.sign[slot];
          proposed_logabs[walker] = batch_result.logabs[slot];
        }
        if (spatial_mode != DirectBackendMode::DIRECT)
        {
          const pf::Result& oracle = oracle_results[slot];
          if (spatial_mode == DirectBackendMode::COMPARE)
          {
            const double scale = std::max(std::abs(oracle.logabs),
                                          std::abs(batch_result.logabs[slot]));
            if (oracle.sign != batch_result.sign[slot] ||
                std::abs(oracle.logabs - batch_result.logabs[slot]) >
                    3.0e-10 * (1.0 + scale))
              throw std::runtime_error(
                  "PsiFormer selected-electron direct value differs from the native oracle");
            for (std::size_t coordinate = 0; coordinate < 3 * electron_count;
                 ++coordinate)
              if (std::abs(oracle.gradient[coordinate] -
                           batch_result.gradient[slot * batch_result.gradient_stride +
                                                 coordinate]) > 2.0e-7)
                throw std::runtime_error(
                    "PsiFormer selected-electron direct gradient differs from the native oracle");
            for (std::size_t electron = 0; electron < electron_count; ++electron)
              if (std::abs(oracle.lap_log[electron] -
                           batch_result.lap_log[slot * batch_result.laplacian_stride +
                                                electron]) > 3.0e-7 ||
                  std::abs(oracle.lap_ratio[electron] -
                           batch_result.lap_ratio[slot * batch_result.laplacian_stride +
                                                  electron]) > 3.0e-7)
                throw std::runtime_error(
                    "PsiFormer selected-electron direct Laplacian differs from the native oracle");
          }
          proposed_signs[walker]  = oracle.sign;
          proposed_logabs[walker] = oracle.logabs;
        }
      }
      if (!psiformer::determinant::isFiniteReal(proposed_signs[walker]) ||
          !psiformer::determinant::isFiniteReal(proposed_logabs[walker]) ||
          proposed_signs[walker] == 0.0)
        throw std::runtime_error(
            "PsiFormer selected-electron proposal produced a non-finite value");
      staged_log_ratios[walker] =
          makeLogValue(proposed_signs[walker], proposed_logabs[walker]) - component.log_value_;
      if (!psiformer::determinant::isFiniteReal(std::real(staged_log_ratios[walker])) ||
          !psiformer::determinant::isFiniteReal(std::imag(staged_log_ratios[walker])))
        throw std::runtime_error(
            "PsiFormer selected-electron proposal produced a non-finite log ratio");
    }

    // All storage and evaluator checks have completed. Publish every clone-local
    // proposal and add its complete component VGL contribution without further allocation.
    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const std::size_t slot = batch_slots[walker];
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        if (slot == no_batch_slot)
        {
          component.proposed_gradient_[electron]  = component.accepted_gradient_[electron];
          component.proposed_laplacian_[electron] = component.accepted_laplacian_[electron];
        }
        else
        {
          if (spatial_mode == DirectBackendMode::DIRECT)
          {
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              component.proposed_gradient_[electron][dimension] =
                  batch_result.gradient[slot * batch_result.gradient_stride +
                                        3 * electron + dimension];
            component.proposed_laplacian_[electron] =
                batch_result.lap_log[slot * batch_result.laplacian_stride + electron];
          }
          else
          {
            const pf::Result& oracle = oracle_results[slot];
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              component.proposed_gradient_[electron][dimension] =
                  oracle.gradient[3 * electron + dimension];
            component.proposed_laplacian_[electron] = oracle.lap_log[electron];
          }
        }
      }

      component.proposed_sign_                   = proposed_signs[walker];
      component.proposed_log_value_              =
          makeLogValue(proposed_signs[walker], proposed_logabs[walker]);
      component.proposed_configuration_identity_ = proposed_identities[walker];
      component.proposed_descriptor_fingerprint_ = descriptor_fingerprint;
      component.proposed_parameter_version_      = parameter_version;
      component.proposed_particle_               = -1;
      component.proposal_origin_                 = ProposalOrigin::MW_SELECTED_FULL_VGL;
      component.has_proposal_                    = true;
      log_ratios[walker]                         = staged_log_ratios[walker];

      auto& proposed_gradient  = proposed_gradient_list[walker].get();
      auto& proposed_laplacian = proposed_laplacian_list[walker].get();
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        proposed_gradient[electron] += component.proposed_gradient_[electron];
        proposed_laplacian[electron] += component.proposed_laplacian_[electron];
      }
    }
    return;
  }

  if (p_list.size() != walker_count || moves.walkerCount() != walker_count ||
      log_ratios.size() != walker_count ||
      proposed_gradient_list.size() != walker_count ||
      proposed_laplacian_list.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned selected proposal has inconsistent walker counts");
  moves.validateFor(p_list);

  const std::uint64_t descriptor_fingerprint = moves.fingerprint();
  PlannedRuntimeRequest request;
  request.operation              = PlannedOperation::SELECTED_PROPOSE;
  request.live_walkers           = walker_count;
  request.dense_configurations   = walker_count;
  request.descriptor_fingerprint = descriptor_fingerprint;
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);

  const std::size_t electron_count = access.participant.plan().particleCount();
  const std::size_t reserve_walkers = access.crowd.reserve_walkers;
  if (electron_count != 0 &&
      reserve_walkers > std::numeric_limits<std::size_t>::max() / electron_count)
    throw std::length_error(
        "PsiFormer planned selected descriptor capacity overflowed");
  const std::size_t selected_capacity = reserve_walkers * electron_count;
  if (electron_count != 0 &&
      walker_count > std::numeric_limits<std::size_t>::max() / electron_count)
    throw std::length_error(
        "PsiFormer planned selected live descriptor extent overflowed");
  if (moves.size() > walker_count * electron_count ||
      moves.size() > selected_capacity)
    throw std::length_error(
        "PsiFormer planned selected descriptor exceeds prepared capacity");

  PsiFormerMultiWalkerResource& resource = access.resource;
  resource.requireSelectedProposalStaging(walker_count);
  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;

  // Phase A proves every caller destination and all persistent clone storage
  // before scratch is changed or the shared model read barrier is acquired.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const auto& gradient = proposed_gradient_list[lane].get();
    const auto& laplacian = proposed_laplacian_list[lane].get();
    if (gradient.size() != electron_count ||
        laplacian.size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer planned selected proposal outputs have the wrong shape");
    if (!component.acceptedStateMatches(
            p_list[lane], component.observed_parameter_version_,
            AcceptedStateRequirement::FULL_SPATIAL) ||
        component.accepted_parameter_version_ !=
            component.observed_parameter_version_)
      throw std::logic_error(
          "PsiFormer planned selected proposal requires current accepted full-VGL state");
    if ((component.current_sign_ != 1.0 && component.current_sign_ != -1.0) ||
        !isFiniteWavefunctionValue(component.log_value_))
      throw std::logic_error(
          "PsiFormer planned selected proposal has invalid accepted value state");

    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        if (!isFiniteWavefunctionValue(
                component.accepted_gradient_[electron][dimension]) ||
            !isFiniteWavefunctionValue(gradient[electron][dimension]))
          throw std::invalid_argument(
              "PsiFormer planned selected gradient state is non-finite");
      }
      if (!isFiniteWavefunctionValue(component.accepted_laplacian_[electron]) ||
          !isFiniteWavefunctionValue(laplacian[electron]))
        throw std::invalid_argument(
            "PsiFormer planned selected Laplacian state is non-finite");
    }
  }

  const CheckedMemoryRange descriptor_offsets = checkedMemoryRange(
      moves.walkerOffsets().data(), moves.walkerOffsets().capacity(),
      "PsiFormer selected descriptor-offset range overflowed");
  const CheckedMemoryRange descriptor_indices = checkedMemoryRange(
      moves.particleIndices().data(), moves.particleIndices().capacity(),
      "PsiFormer selected descriptor-index range overflowed");
  const CheckedMemoryRange descriptor_positions = checkedMemoryRange(
      moves.proposedPositions().data(), moves.proposedPositions().capacity(),
      "PsiFormer selected descriptor-position range overflowed");
  const auto descriptor_overlaps = [&](const CheckedMemoryRange& range) noexcept {
    return memoryRangesOverlap(range, descriptor_offsets) ||
        memoryRangesOverlap(range, descriptor_indices) ||
        memoryRangesOverlap(range, descriptor_positions);
  };
  const auto output_range = [&](std::size_t lane, int kind) {
    if (kind == 0)
      return checkedMemoryRange(
          proposed_gradient_list[lane].get().data(), electron_count,
          "PsiFormer planned selected gradient range overflowed");
    if (kind == 1)
      return checkedMemoryRange(
          proposed_laplacian_list[lane].get().data(), electron_count,
          "PsiFormer planned selected Laplacian range overflowed");
    return checkedMemoryRange(
        log_ratios.data() + lane, std::size_t{1},
        "PsiFormer planned selected ratio range overflowed");
  };

  for (std::size_t lane = 0; lane < walker_count; ++lane)
    for (int kind = 0; kind < 3; ++kind)
    {
      const CheckedMemoryRange output = output_range(lane, kind);
      const void* output_data = kind == 0
          ? static_cast<const void*>(proposed_gradient_list[lane].get().data())
          : kind == 1
          ? static_cast<const void*>(proposed_laplacian_list[lane].get().data())
          : static_cast<const void*>(log_ratios.data() + lane);
      const std::size_t output_bytes = output.end - output.begin;

      if (descriptor_overlaps(output))
        throw std::invalid_argument(
            "PsiFormer planned selected output aliases its move descriptor");
      for (std::size_t prior_lane = 0; prior_lane <= lane; ++prior_lane)
        for (int prior_kind = 0; prior_kind < 3; ++prior_kind)
        {
          if (prior_lane == lane && prior_kind >= kind)
            continue;
          if (memoryRangesOverlap(output, output_range(prior_lane, prior_kind)))
            throw std::invalid_argument(
                "PsiFormer planned selected caller outputs overlap");
        }

      for (std::size_t particle_lane = 0; particle_lane < walker_count;
           ++particle_lane)
      {
        const auto& soa_positions =
            p_list[particle_lane].getCoordinates().getAllParticlePos();
        if (soa_positions.capacity() >
            std::numeric_limits<std::size_t>::max() / 3)
          throw std::length_error(
              "PsiFormer ParticleSet SoA position range overflowed");
        for (const CheckedMemoryRange position_range : {
                 checkedMemoryRange(
                     p_list[particle_lane].R.data(), electron_count,
                     "PsiFormer ParticleSet position range overflowed"),
                 checkedMemoryRange(
                     soa_positions.data(), 3 * soa_positions.capacity(),
                     "PsiFormer ParticleSet SoA position range overflowed"),
                 checkedMemoryRange(
                     std::addressof(p_list[particle_lane].getActivePos()),
                     std::size_t{1},
                     "PsiFormer ParticleSet active-position range overflowed")})
          if (memoryRangesOverlap(output, position_range))
            throw std::invalid_argument(
                "PsiFormer planned selected output aliases ParticleSet position storage");

        for (int particle_kind = 0; particle_kind < 2; ++particle_kind)
        {
          const CheckedMemoryRange particle_range = particle_kind == 0
              ? checkedMemoryRange(
                    p_list[particle_lane].G.data(), electron_count,
                    "PsiFormer ParticleSet gradient range overflowed")
              : checkedMemoryRange(
                    p_list[particle_lane].L.data(), electron_count,
                    "PsiFormer ParticleSet Laplacian range overflowed");
          if (!memoryRangesOverlap(output, particle_range))
            continue;
          const bool exact_same_lane = particle_lane == lane &&
              particle_kind == kind && sameMemoryRange(output, particle_range);
          if (!exact_same_lane)
            throw std::invalid_argument(
                "PsiFormer planned selected output aliases incompatible ParticleSet storage");
        }
      }

      for (std::size_t component_lane = 0; component_lane < walker_count;
           ++component_lane)
      {
        const auto& component =
            static_cast<const PsiFormerWF&>(wfc_list[component_lane]);
        for (const CheckedMemoryRange internal : {
                 checkedMemoryRange(
                     component.accepted_gradient_.data(), electron_count,
                     "PsiFormer accepted gradient range overflowed"),
                 checkedMemoryRange(
                     component.accepted_laplacian_.data(), electron_count,
                     "PsiFormer accepted Laplacian range overflowed"),
                 checkedMemoryRange(
                     component.proposed_gradient_.data(), electron_count,
                     "PsiFormer proposed gradient range overflowed"),
                 checkedMemoryRange(
                     component.proposed_laplacian_.data(), electron_count,
                     "PsiFormer proposed Laplacian range overflowed")})
          if (memoryRangesOverlap(output, internal))
            throw std::invalid_argument(
                "PsiFormer planned selected output aliases component state");
      }
      if (resource.overlapsStagingStorage(output_data, output_bytes) ||
          batch.overlapsStorage(output_data, output_bytes))
        throw std::invalid_argument(
            "PsiFormer planned selected output aliases prepared scratch");
    }

  // Internal allocations and immutable descriptor backing must themselves be
  // disjoint from mutable prepared scratch and from one another.
  for (const CheckedMemoryRange descriptor_range : {
           descriptor_offsets, descriptor_indices, descriptor_positions})
    if (resource.overlapsStagingStorage(
            reinterpret_cast<const void*>(descriptor_range.begin),
            descriptor_range.end - descriptor_range.begin) ||
        batch.overlapsStorage(
            reinterpret_cast<const void*>(descriptor_range.begin),
            descriptor_range.end - descriptor_range.begin))
      throw std::invalid_argument(
          "PsiFormer planned selected descriptor aliases prepared scratch");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const std::array<CheckedMemoryRange, 4> internal_ranges{
        checkedMemoryRange(component.accepted_gradient_.data(), electron_count,
                           "PsiFormer accepted gradient range overflowed"),
        checkedMemoryRange(component.accepted_laplacian_.data(), electron_count,
                           "PsiFormer accepted Laplacian range overflowed"),
        checkedMemoryRange(component.proposed_gradient_.data(), electron_count,
                           "PsiFormer proposed gradient range overflowed"),
        checkedMemoryRange(component.proposed_laplacian_.data(), electron_count,
                           "PsiFormer proposed Laplacian range overflowed")};
    for (std::size_t kind = 0; kind < internal_ranges.size(); ++kind)
    {
      const CheckedMemoryRange internal = internal_ranges[kind];
      for (std::size_t prior_lane = 0; prior_lane <= lane; ++prior_lane)
      {
        const auto& prior = static_cast<const PsiFormerWF&>(wfc_list[prior_lane]);
        const std::array<CheckedMemoryRange, 4> prior_ranges{
            checkedMemoryRange(prior.accepted_gradient_.data(), electron_count,
                               "PsiFormer accepted gradient range overflowed"),
            checkedMemoryRange(prior.accepted_laplacian_.data(), electron_count,
                               "PsiFormer accepted Laplacian range overflowed"),
            checkedMemoryRange(prior.proposed_gradient_.data(), electron_count,
                               "PsiFormer proposed gradient range overflowed"),
            checkedMemoryRange(prior.proposed_laplacian_.data(), electron_count,
                               "PsiFormer proposed Laplacian range overflowed")};
        for (std::size_t prior_kind = 0; prior_kind < prior_ranges.size();
             ++prior_kind)
        {
          if (prior_lane == lane && prior_kind >= kind)
            continue;
          if (memoryRangesOverlap(internal, prior_ranges[prior_kind]))
            throw std::logic_error(
                "PsiFormer planned selected component storage overlaps");
        }
      }
      const void* internal_data = reinterpret_cast<const void*>(internal.begin);
      const std::size_t internal_bytes = internal.end - internal.begin;
      if (resource.overlapsStagingStorage(internal_data, internal_bytes) ||
          batch.overlapsStorage(internal_data, internal_bytes))
        throw std::logic_error(
            "PsiFormer planned selected component storage aliases prepared scratch");
    }
  }

  constexpr std::size_t no_batch_slot =
      std::numeric_limits<std::size_t>::max();
  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_spatial_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned selected proposal requires the direct spatial backend");
  const std::size_t parameter_version = transaction.parameterVersion();
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (component.observed_parameter_version_ != parameter_version ||
        component.accepted_parameter_version_ != parameter_version)
      throw std::logic_error(
          "PsiFormer planned selected proposal has stale accepted parameters");
  }

  std::size_t evaluated_rows = 0;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const auto lane_moves = moves.slice(lane);
    resource.configuration_identities[lane] =
        configurationIdentity(p_list[lane], lane_moves);
    const bool reuse_accepted =
        selectedCoordinatesExactlyUnchanged(p_list[lane], lane_moves) &&
        resource.configuration_identities[lane] ==
            component.accepted_configuration_identity_;
    if (reuse_accepted)
    {
      resource.batch_slots[lane] = no_batch_slot;
      resource.staged_signs[lane] = component.current_sign_;
      resource.staged_log_magnitudes[lane] = std::real(component.log_value_);
      resource.staged_log_ratios[lane] = LogValue(0);
    }
    else
    {
      resource.batch_slots[lane] = evaluated_rows;
      resource.walker_indices[evaluated_rows++] = lane;
    }
  }

  batch.resize(pf::DirectBatchMode::FULL_VGL, evaluated_rows);
  for (std::size_t slot = 0; slot < evaluated_rows; ++slot)
  {
    const std::size_t lane = resource.walker_indices[slot];
    packBatchConfiguration(batch, slot, p_list[lane], moves.slice(lane));
  }
  const pf::DirectBatchSpatialResultView result =
      transaction.state().direct_batch_executor.evaluateFull(batch);
  if (!batch.ownsSpatialResult(result, pf::DirectSpatialMode::FULL_VGL,
                               evaluated_rows))
    throw std::logic_error(
        "PsiFormer planned selected result is not the exact workspace-owned view");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const std::size_t slot = resource.batch_slots[lane];
    if (slot != no_batch_slot)
    {
      if (slot >= evaluated_rows || result.parameter_version[slot] != parameter_version)
        throw std::logic_error(
            "PsiFormer planned selected proposal observed inconsistent parameters");
      resource.staged_signs[lane] = result.sign[slot];
      resource.staged_log_magnitudes[lane] = result.logabs[slot];
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          if (!psiformer::determinant::isFiniteReal(
                  result.gradient[slot * result.gradient_stride +
                                  3 * electron + dimension]))
            throw std::runtime_error(
                "PsiFormer planned selected proposal produced a non-finite gradient");
        if (!psiformer::determinant::isFiniteReal(
                result.lap_log[slot * result.laplacian_stride + electron]) ||
            !psiformer::determinant::isFiniteReal(
                result.lap_ratio[slot * result.laplacian_stride + electron]))
          throw std::runtime_error(
              "PsiFormer planned selected proposal produced a non-finite Laplacian");
      }
    }
    if ((resource.staged_signs[lane] != 1.0 &&
         resource.staged_signs[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(
            resource.staged_log_magnitudes[lane]))
      throw std::runtime_error(
          "PsiFormer planned selected proposal produced an invalid value");
    if (slot != no_batch_slot)
      resource.staged_log_ratios[lane] = makeLogValue(
          resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane]) - component.log_value_;
    if (!isFiniteWavefunctionValue(resource.staged_log_ratios[lane]))
      throw std::runtime_error(
          "PsiFormer planned selected proposal produced a non-finite ratio");
  }

  // Phase B repeats all borrowed identity and input-state checks immediately
  // before the final dry sums and mechanically nonthrowing publication.
  if (!batch.ownsSpatialResult(result, pf::DirectSpatialMode::FULL_VGL,
                               evaluated_rows) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents() ||
      moves.fingerprint() != descriptor_fingerprint)
    throw std::logic_error(
        "PsiFormer planned selected proposal evidence changed during evaluation");
  resource.requireSelectedProposalStaging(walker_count);
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint ||
      final_access.selected_transaction_fingerprint !=
          access.selected_transaction_fingerprint)
    throw std::logic_error(
        "PsiFormer planned selected runtime evidence changed during evaluation");
  std::size_t expected_slot = 0;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const auto lane_moves = moves.slice(lane);
    if (component.model_state_.get() != model_state_.get() ||
        component.optimization_metadata_.get() != optimization_metadata_.get() ||
        component.bound_particle_set_ != &p_list[lane] ||
        component.acquired_crowd_leader_ != this ||
        component.acquired_lane_index_ != lane ||
        component.acquired_crowd_size_ != walker_count ||
        !component.batch_execution_plan_.sameBinding(access.participant) ||
        !component.hasPreparedBatchExecutionClone(access.participant) ||
        !component.acceptedStateMatches(
            p_list[lane], parameter_version,
            AcceptedStateRequirement::FULL_SPATIAL) ||
        resource.configuration_identities[lane] !=
            configurationIdentity(p_list[lane], lane_moves) ||
        (component.current_sign_ != 1.0 && component.current_sign_ != -1.0) ||
        !isFiniteWavefunctionValue(component.log_value_))
      throw std::logic_error(
          "PsiFormer planned selected lane identity changed during evaluation");
    const std::size_t slot = resource.batch_slots[lane];
    const bool reuse_accepted =
        selectedCoordinatesExactlyUnchanged(p_list[lane], lane_moves) &&
        resource.configuration_identities[lane] ==
            component.accepted_configuration_identity_;
    if (reuse_accepted != (slot == no_batch_slot))
      throw std::logic_error(
          "PsiFormer planned selected reuse classification changed during evaluation");
    if (slot == no_batch_slot)
    {
      if (resource.staged_signs[lane] != component.current_sign_ ||
          resource.staged_log_magnitudes[lane] !=
              std::real(component.log_value_) ||
          resource.staged_log_ratios[lane] != LogValue(0))
        throw std::logic_error(
            "PsiFormer planned selected reused value staging changed during evaluation");
    }
    else
    {
      if (slot != expected_slot || expected_slot >= evaluated_rows ||
          resource.walker_indices[expected_slot] != lane ||
          result.parameter_version[expected_slot] != parameter_version)
        throw std::logic_error(
            "PsiFormer planned selected compact mapping changed during evaluation");
      if ((result.sign[slot] != 1.0 && result.sign[slot] != -1.0) ||
          !psiformer::determinant::isFiniteReal(result.logabs[slot]) ||
          resource.staged_signs[lane] != result.sign[slot] ||
          resource.staged_log_magnitudes[lane] != result.logabs[slot] ||
          resource.staged_log_ratios[lane] !=
              makeLogValue(result.sign[slot], result.logabs[slot]) -
                  component.log_value_)
        throw std::logic_error(
            "PsiFormer planned selected native value staging changed during evaluation");
      ++expected_slot;
    }

    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        if (!isFiniteWavefunctionValue(
                component.accepted_gradient_[electron][dimension]))
          throw std::logic_error(
              "PsiFormer planned selected accepted gradient changed during evaluation");
        if (slot != no_batch_slot &&
            !psiformer::determinant::isFiniteReal(
                result.gradient[slot * result.gradient_stride +
                                3 * electron + dimension]))
          throw std::logic_error(
              "PsiFormer planned selected native gradient changed during evaluation");
      }
      if (!isFiniteWavefunctionValue(component.accepted_laplacian_[electron]))
        throw std::logic_error(
            "PsiFormer planned selected accepted Laplacian changed during evaluation");
      if (slot != no_batch_slot &&
          (!psiformer::determinant::isFiniteReal(
               result.lap_log[slot * result.laplacian_stride + electron]) ||
           !psiformer::determinant::isFiniteReal(
               result.lap_ratio[slot * result.laplacian_stride + electron])))
        throw std::logic_error(
            "PsiFormer planned selected native Laplacian changed during evaluation");
    }
    if (!isFiniteWavefunctionValue(resource.staged_log_ratios[lane]))
      throw std::logic_error(
          "PsiFormer planned selected ratio staging changed during evaluation");
  }
  if (expected_slot != evaluated_rows)
    throw std::logic_error(
        "PsiFormer planned selected compact row count changed during evaluation");

  if (fail_planned_selected_proposal_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned selected pre-publication failure");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const std::size_t slot = resource.batch_slots[lane];
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const ValueType contribution = slot == no_batch_slot
            ? component.accepted_gradient_[electron][dimension]
            : static_cast<ValueType>(
                  result.gradient[slot * result.gradient_stride +
                                  3 * electron + dimension]);
        const ValueType future =
            proposed_gradient_list[lane].get()[electron][dimension] +
            contribution;
        if (!isFiniteWavefunctionValue(future))
          throw std::overflow_error(
              "PsiFormer planned selected gradient sum is non-finite");
      }
      const ValueType contribution = slot == no_batch_slot
          ? component.accepted_laplacian_[electron]
          : static_cast<ValueType>(
                result.lap_log[slot * result.laplacian_stride + electron]);
      const ValueType future =
          proposed_laplacian_list[lane].get()[electron] + contribution;
      if (!isFiniteWavefunctionValue(future))
        throw std::overflow_error(
            "PsiFormer planned selected Laplacian sum is non-finite");
    }
  }

  if (!tryRegisterPlannedSelectedTransaction())
    throw std::overflow_error(
        "PsiFormer planned selected transaction count overflow");

  const std::uint64_t transaction_fingerprint =
      *access.selected_transaction_fingerprint;
  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      const std::size_t slot = resource.batch_slots[lane];
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          component.proposed_gradient_[electron][dimension] =
              slot == no_batch_slot
              ? component.accepted_gradient_[electron][dimension]
              : static_cast<ValueType>(
                    result.gradient[slot * result.gradient_stride +
                                    3 * electron + dimension]);
        component.proposed_laplacian_[electron] = slot == no_batch_slot
            ? component.accepted_laplacian_[electron]
            : static_cast<ValueType>(
                  result.lap_log[slot * result.laplacian_stride + electron]);
      }
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      log_ratios[lane] = resource.staged_log_ratios[lane];
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        proposed_gradient_list[lane].get()[electron] +=
            component.proposed_gradient_[electron];
        proposed_laplacian_list[lane].get()[electron] +=
            component.proposed_laplacian_[electron];
      }
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      component.proposed_sign_ = resource.staged_signs[lane];
      component.proposed_log_value_ = makeLogValue(
          resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane]);
      component.proposed_configuration_identity_ =
          resource.configuration_identities[lane];
      component.proposed_descriptor_fingerprint_ = transaction_fingerprint;
      component.proposed_parameter_version_ = parameter_version;
      component.proposed_particle_ = -1;
      component.proposal_origin_ = ProposalOrigin::MW_SELECTED_FULL_VGL;
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = true;
  };
  static_assert(noexcept(publish()));
  publish();
}

// Resolve a selected-electron transaction only after validating the complete crowd.
void PsiFormerWF::mw_accept_rejectMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    const std::vector<bool>& accepted) const
{
  const std::size_t walker_count = wfc_list.size();
  if (!batch_execution_plan_)
  {
    if (p_list.size() != walker_count || moves.walkerCount() != walker_count ||
        accepted.size() != walker_count)
      throw std::invalid_argument(
          "PsiFormer selected-electron resolution has inconsistent walker counts");
    moves.validateFor(p_list);
    if (walker_count == 0)
      return;

    const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
    requireMultiWalkerResource(wfc_list);
    PsiFormerReadTransaction transaction(*leader.model_state_);
    const std::size_t parameter_version = transaction.parameterVersion();
    const std::uint64_t descriptor_fingerprint = moves.fingerprint();
    const std::size_t electron_count =
        transaction.state().execution_plan.modelShape().electrons();

    // Observe a concurrent publication across the complete crowd before reporting
    // any stale proposal; synchronization clears every clone's pending state.
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
          parameter_version);

    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      if (!component.has_proposal_ ||
          component.proposal_origin_ != ProposalOrigin::MW_SELECTED_FULL_VGL)
        throw std::logic_error(
            "PsiFormer selected-electron resolution has no matching pending proposal");
      if (component.proposed_descriptor_fingerprint_ != descriptor_fingerprint)
        throw std::logic_error(
            "PsiFormer selected-electron resolution descriptor does not match the pending proposal");
      if (component.proposed_parameter_version_ != parameter_version)
        throw std::logic_error(
            "PsiFormer parameters changed during a selected-electron transaction");
      if (component.proposed_gradient_.size() != electron_count ||
          component.proposed_laplacian_.size() != electron_count)
        throw std::logic_error(
            "PsiFormer selected-electron proposal has incomplete spatial state");
      if (accepted[walker] &&
          (component.accepted_gradient_.size() != electron_count ||
           component.accepted_laplacian_.size() != electron_count))
        throw std::logic_error(
            "PsiFormer selected-electron resolution has invalid accepted spatial storage");
      // Rejection is also used to unwind an earlier component when a later TWF
      // component fails, before ParticleSet installs proposed coordinates.
      if (accepted[walker] &&
          component.proposed_configuration_identity_ != configurationIdentity(p_list[walker]))
        throw std::logic_error(
            "PsiFormer accepted selected-electron coordinates do not match the pending proposal");
    }

    for (std::size_t walker = 0; walker < walker_count; ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      if (accepted[walker])
      {
        for (std::size_t electron = 0; electron < electron_count; ++electron)
        {
          component.accepted_gradient_[electron]  = component.proposed_gradient_[electron];
          component.accepted_laplacian_[electron] = component.proposed_laplacian_[electron];
        }
        component.current_sign_                   = component.proposed_sign_;
        component.log_value_                      = component.proposed_log_value_;
        component.accepted_configuration_identity_ = component.proposed_configuration_identity_;
        component.accepted_parameter_version_      = parameter_version;
        component.accepted_state_requirement_      = AcceptedStateRequirement::FULL_SPATIAL;
        component.accepted_value_valid_            = true;
      }
    }
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      wfc_list.getCastedElement<PsiFormerWF>(walker).resetProposalMetadata();
    for (std::size_t walker = 0; walker < walker_count; ++walker)
      wfc_list.getCastedElement<PsiFormerWF>(walker).has_proposal_ = false;
    return;
  }

  if (p_list.size() != walker_count || moves.walkerCount() != walker_count ||
      accepted.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned selected-electron resolution has inconsistent walker counts");
  moves.validateFor(p_list);

  const std::uint64_t descriptor_fingerprint = moves.fingerprint();
  PlannedRuntimeRequest request;
  request.operation                 = PlannedOperation::SELECTED_RESOLVE;
  request.live_walkers              = walker_count;
  request.descriptor_fingerprint    = descriptor_fingerprint;
  request.expected_proposal_version = proposed_parameter_version_;
  const PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);

  const std::size_t electron_count = access.participant.plan().particleCount();
  const std::size_t reserve_walkers = access.crowd.reserve_walkers;
  if (electron_count != 0 &&
      reserve_walkers > std::numeric_limits<std::size_t>::max() / electron_count)
    throw std::length_error(
        "PsiFormer planned selected resolution descriptor capacity overflowed");
  const std::size_t selected_capacity = reserve_walkers * electron_count;
  if (electron_count != 0 &&
      walker_count > std::numeric_limits<std::size_t>::max() / electron_count)
    throw std::length_error(
        "PsiFormer planned selected resolution live descriptor extent overflowed");
  if (moves.size() > walker_count * electron_count ||
      moves.size() > selected_capacity)
    throw std::length_error(
        "PsiFormer planned selected resolution descriptor exceeds prepared capacity");

  const std::size_t proposal_version = *request.expected_proposal_version;
  const std::uint64_t transaction_fingerprint =
      *access.selected_transaction_fingerprint;
  const auto validate_lane_state = [&](std::optional<std::size_t> authoritative_version) {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      if (!component.has_proposal_ ||
          component.proposal_origin_ != ProposalOrigin::MW_SELECTED_FULL_VGL ||
          component.proposed_particle_ != -1 ||
          component.proposed_descriptor_fingerprint_ != transaction_fingerprint ||
          component.proposed_configuration_identity_ !=
              access.resource.configuration_identities[lane] ||
          component.proposed_sign_ != access.resource.staged_signs[lane] ||
          component.proposed_log_value_ != makeLogValue(
              access.resource.staged_signs[lane],
              access.resource.staged_log_magnitudes[lane]))
        throw std::logic_error(
            "PsiFormer planned selected resolution has incomplete proposal provenance");
      if (!component.accepted_value_valid_ ||
          component.accepted_state_requirement_ !=
              AcceptedStateRequirement::FULL_SPATIAL ||
          component.accepted_parameter_version_ != proposal_version ||
          component.observed_parameter_version_ != proposal_version ||
          component.proposed_parameter_version_ != proposal_version)
        throw std::logic_error(
            "PsiFormer planned selected resolution has incoherent cached versions");
      if (authoritative_version &&
          (*authoritative_version != proposal_version ||
           component.accepted_parameter_version_ != *authoritative_version ||
           component.observed_parameter_version_ != *authoritative_version ||
           component.proposed_parameter_version_ != *authoritative_version))
        throw std::logic_error(
            "PsiFormer parameters changed during a planned selected transaction");
      if (component.accepted_gradient_.size() != electron_count ||
          component.accepted_laplacian_.size() != electron_count ||
          component.proposed_gradient_.size() != electron_count ||
          component.proposed_laplacian_.size() != electron_count)
        throw std::logic_error(
            "PsiFormer planned selected resolution has incomplete spatial state");
      if ((component.current_sign_ != 1.0 && component.current_sign_ != -1.0) ||
          (component.proposed_sign_ != 1.0 && component.proposed_sign_ != -1.0) ||
          !isFiniteWavefunctionValue(component.log_value_) ||
          !isFiniteWavefunctionValue(component.proposed_log_value_))
        throw std::logic_error(
            "PsiFormer planned selected resolution has invalid value state");
      for (std::size_t electron = 0; electron < electron_count; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          if (!isFiniteWavefunctionValue(
                  component.accepted_gradient_[electron][dimension]) ||
              !isFiniteWavefunctionValue(
                  component.proposed_gradient_[electron][dimension]))
            throw std::logic_error(
                "PsiFormer planned selected resolution has non-finite gradient state");
        if (!isFiniteWavefunctionValue(component.accepted_laplacian_[electron]) ||
            !isFiniteWavefunctionValue(component.proposed_laplacian_[electron]))
          throw std::logic_error(
              "PsiFormer planned selected resolution has non-finite Laplacian state");
      }

      // Rejected lanes deliberately have no live-coordinate identity
      // prerequisite: the outer ParticleSet transaction may not have rolled
      // them back yet. Accepted lanes must already expose the proposed state.
      if (accepted[lane] &&
          component.proposed_configuration_identity_ !=
              configurationIdentity(p_list[lane]))
        throw std::logic_error(
            "PsiFormer planned accepted selected coordinates do not match the proposal");
    }
  };

  // Phase A is strictly read only and deliberately precedes model-lock
  // acquisition so malformed non-version evidence cannot open a transaction.
  validate_lane_state(std::nullopt);

  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  if (transaction.state().execution_plan.modelShape().electrons() != electron_count)
    throw std::logic_error(
        "PsiFormer planned selected resolution model shape changed");
  validate_lane_state(parameter_version);

  // Phase B repeats the exact descriptor, resource, allocation, team, and lane
  // evidence under the authoritative model read transaction.
  if (accepted.size() != walker_count || moves.walkerCount() != walker_count)
    throw std::logic_error(
        "PsiFormer planned selected resolution counts changed during preflight");
  moves.validateFor(p_list);
  if (moves.fingerprint() != descriptor_fingerprint ||
      moves.size() > walker_count * electron_count ||
      moves.size() > selected_capacity)
    throw std::logic_error(
        "PsiFormer planned selected resolution descriptor changed during preflight");
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &access.resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint ||
      final_access.selected_transaction_fingerprint !=
          access.selected_transaction_fingerprint)
    throw std::logic_error(
        "PsiFormer planned selected resolution runtime evidence changed during preflight");
  validate_lane_state(parameter_version);

  if (fail_planned_selected_resolution_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned selected resolution pre-publication failure");

  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (accepted[lane])
      {
        auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
        for (std::size_t electron = 0; electron < electron_count; ++electron)
        {
          component.accepted_gradient_[electron] =
              component.proposed_gradient_[electron];
          component.accepted_laplacian_[electron] =
              component.proposed_laplacian_[electron];
        }
      }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (accepted[lane])
      {
        auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
        component.current_sign_ = component.proposed_sign_;
        component.log_value_ = component.proposed_log_value_;
        component.accepted_configuration_identity_ =
            component.proposed_configuration_identity_;
        component.accepted_parameter_version_ = parameter_version;
        component.accepted_state_requirement_ =
            AcceptedStateRequirement::FULL_SPATIAL;
        component.observed_parameter_version_ = parameter_version;
      }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (accepted[lane])
        static_cast<PsiFormerWF&>(wfc_list[lane]).accepted_value_valid_ = true;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).resetProposalMetadata();
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = false;
  };
  static_assert(noexcept(publish()));
  publish();

  // The count remains conservatively nonzero until every lane has completed
  // the mechanically nonthrowing state publication above.
  unregisterPlannedSelectedTransaction();
}

// Retain the allocating inherited scalar fallback only in legacy no-plan mode.
void PsiFormerWF::recompute(const ParticleSet& particles)
{
  requireUnplannedScalarEvaluation("recompute");
  WaveFunctionComponent::recompute(particles);
}

// Refresh a selected set of accepted values through the legacy or planned path.
void PsiFormerWF::mw_recompute(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const std::vector<bool>& recompute_mask) const
{
  // Preserve the historical lazy/oracle implementation behind the explicit
  // no-policy branch.
  if (!batch_execution_plan_)
  {
    if (wfc_list.size() != p_list.size() || wfc_list.size() != recompute_mask.size())
      throw std::invalid_argument("PsiFormer mw_recompute list sizes do not match");
    if (wfc_list.empty())
      return;

    const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
    auto& resource     = requireMultiWalkerResource(wfc_list);
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      wfc_list.getCastedElement<PsiFormerWF>(walker).requireNoSelectedParticleProposal(
          "mw_recompute");

    resource.walker_indices.clear();
    for (std::size_t walker = 0; walker < recompute_mask.size(); ++walker)
      if (recompute_mask[walker])
        resource.walker_indices.push_back(walker);
    if (resource.walker_indices.empty())
      return;

    PsiFormerReadTransaction transaction(*leader.model_state_);
    const std::size_t parameter_version = transaction.parameterVersion();
    const std::size_t selected_count = resource.walker_indices.size();
    std::vector<double> staged_sign(selected_count);
    std::vector<double> staged_logabs(selected_count);
    std::vector<std::uint64_t> staged_configuration(selected_count);
    std::vector<bool> preserve_spatial(selected_count);
    for (std::size_t selected = 0; selected < resource.walker_indices.size(); ++selected)
    {
      const std::size_t walker = resource.walker_indices[selected];
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      component.synchronizeParameterVersion(parameter_version);
      staged_configuration[selected] = configurationIdentity(p_list[walker]);
      preserve_spatial[selected] = component.acceptedStateMatches(
          p_list[walker], parameter_version,
          AcceptedStateRequirement::FULL_SPATIAL);
    }

    if (transaction.state().direct_value_mode == DirectBackendMode::DIRECT)
    {
      auto& batch = *resource.batch_workspace;
      batch.resize(pf::DirectBatchMode::VALUE_ONLY, selected_count);
      for (std::size_t selected = 0; selected < selected_count; ++selected)
        packBatchConfiguration(batch, selected,
                               p_list[resource.walker_indices[selected]]);

      const pf::DirectBatchValueResultView result =
          transaction.state().direct_batch_executor.evaluateValues(batch);
      for (std::size_t selected = 0; selected < selected_count; ++selected)
      {
        if (result.parameter_version[selected] != parameter_version)
          throw std::logic_error("PsiFormer recompute batch observed inconsistent parameters");
        staged_sign[selected]   = result.sign[selected];
        staged_logabs[selected] = result.logabs[selected];
      }
    }
    else
    {
      for (std::size_t selected = 0; selected < selected_count; ++selected)
      {
        const std::size_t walker = resource.walker_indices[selected];
        auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
        const pf::Result result = component.evaluatePositionsUnderRead(
            transaction, p_list[walker], -1, nullptr,
            EvaluationPurpose::VALUE_ONLY);
        staged_sign[selected]   = result.sign;
        staged_logabs[selected] = result.logabs;
      }
    }

    for (std::size_t selected = 0; selected < selected_count; ++selected)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(resource.walker_indices[selected]);
      component.current_sign_ = staged_sign[selected];
      component.log_value_ = makeLogValue(staged_sign[selected], staged_logabs[selected]);
      component.accepted_configuration_identity_ = staged_configuration[selected];
      component.accepted_parameter_version_      = parameter_version;
      component.accepted_state_requirement_ = preserve_spatial[selected]
          ? AcceptedStateRequirement::FULL_SPATIAL
          : AcceptedStateRequirement::VALUE_ONLY;
      component.accepted_value_valid_ = true;
      component.clearProposalState();
    }
    return;
  }

  const std::size_t walker_count = wfc_list.size();
  if (recompute_mask.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned RECOMPUTE_VALUE mask size does not match the crowd");
  std::size_t selected_count = 0;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    if (recompute_mask[lane])
      ++selected_count;

  PlannedRuntimeRequest request;
  request.operation            = PlannedOperation::RECOMPUTE_VALUE;
  request.live_walkers         = walker_count;
  request.dense_configurations = selected_count;
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  PsiFormerMultiWalkerResource& resource = access.resource;
  resource.requireRecomputeStaging(selected_count);
  if (selected_count == 0)
    return;

  std::size_t selected = 0;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    if (recompute_mask[lane])
    {
      resource.walker_indices[selected] = lane;
      resource.configuration_identities[selected] =
          configurationIdentity(p_list[lane]);
      ++selected;
    }
  if (selected != selected_count)
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE mask changed while indexing");

  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE requires the direct value backend");
  const std::size_t parameter_version = transaction.parameterVersion();
  const std::size_t electron_count = access.participant.plan().particleCount();

  const auto has_finite_full_state = [&](const PsiFormerWF& component) noexcept {
    if (component.accepted_state_requirement_ !=
            AcceptedStateRequirement::FULL_SPATIAL ||
        !isCoherentAcceptedValue(component.current_sign_,
                                 component.log_value_))
      return false;
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!isFiniteWavefunctionValue(
                component.accepted_gradient_[electron][dimension]))
          return false;
      if (!isFiniteWavefunctionValue(component.accepted_laplacian_[electron]))
        return false;
    }
    return true;
  };

  for (std::size_t slot = 0; slot < selected_count; ++slot)
  {
    const std::size_t lane = resource.walker_indices[slot];
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (resource.configuration_identities[slot] !=
        configurationIdentity(p_list[lane]))
      throw std::logic_error(
          "PsiFormer planned RECOMPUTE_VALUE configuration changed before evaluation");
    const bool preserve_full =
        component.observed_parameter_version_ == parameter_version &&
        component.acceptedStateMatches(
            p_list[lane], parameter_version,
            AcceptedStateRequirement::FULL_SPATIAL) &&
        has_finite_full_state(component);
    resource.preservation_flags[slot] =
        static_cast<unsigned char>(preserve_full);
  }

  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, selected_count);
  for (std::size_t slot = 0; slot < selected_count; ++slot)
    packBatchConfiguration(
        batch, slot, p_list[resource.walker_indices[slot]]);

  const pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(batch);
  if (!batch.ownsValueResult(
          result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
          selected_count))
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE result is not the exact workspace-owned view");

  const auto valid_value = [](double sign, double logabs) noexcept {
    return (sign == -1.0 || sign == 1.0) &&
        psiformer::determinant::isFiniteReal(logabs);
  };
  for (std::size_t slot = 0; slot < selected_count; ++slot)
  {
    if (result.parameter_version[slot] != parameter_version)
      throw std::logic_error(
          "PsiFormer planned RECOMPUTE_VALUE observed inconsistent parameters");
    if (!valid_value(result.sign[slot], result.logabs[slot]))
      throw std::runtime_error(
          "PsiFormer planned RECOMPUTE_VALUE produced an invalid value");
    resource.staged_signs[slot] = result.sign[slot];
    resource.staged_log_magnitudes[slot] = result.logabs[slot];
  }

  // Phase B replays the exact mask order, accepted preservation decision,
  // resource provenance, and native result evidence immediately before commit.
  if (!batch.ownsValueResult(
          result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
          selected_count) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE storage changed during evaluation");
  resource.requireRecomputeStaging(selected_count);
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE runtime evidence changed during evaluation");
  if (recompute_mask.size() != walker_count)
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE mask size changed during evaluation");

  selected = 0;
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    if (recompute_mask[lane])
    {
      if (selected >= selected_count ||
          resource.walker_indices[selected] != lane)
        throw std::logic_error(
            "PsiFormer planned RECOMPUTE_VALUE mask order changed during evaluation");
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      const bool preserve_full =
          component.observed_parameter_version_ == parameter_version &&
          component.acceptedStateMatches(
              p_list[lane], parameter_version,
              AcceptedStateRequirement::FULL_SPATIAL) &&
          has_finite_full_state(component);
      if (component.model_state_.get() != model_state_.get() ||
          component.optimization_metadata_.get() != optimization_metadata_.get() ||
          component.bound_particle_set_ != &p_list[lane] ||
          component.acquired_crowd_leader_ != this ||
          component.acquired_lane_index_ != lane ||
          component.acquired_crowd_size_ != walker_count ||
          !component.batch_execution_plan_.sameBinding(access.participant) ||
          !component.hasPreparedBatchExecutionClone(access.participant) ||
          resource.configuration_identities[selected] !=
              configurationIdentity(p_list[lane]) ||
          resource.preservation_flags[selected] !=
              static_cast<unsigned char>(preserve_full) ||
          result.parameter_version[selected] != parameter_version ||
          !valid_value(result.sign[selected], result.logabs[selected]) ||
          resource.staged_signs[selected] != result.sign[selected] ||
          resource.staged_log_magnitudes[selected] != result.logabs[selected])
        throw std::logic_error(
            "PsiFormer planned RECOMPUTE_VALUE lane evidence changed during evaluation");
      ++selected;
    }
  if (selected != selected_count)
    throw std::logic_error(
        "PsiFormer planned RECOMPUTE_VALUE selected count changed during evaluation");

  if (fail_planned_recompute_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned RECOMPUTE_VALUE pre-publication failure");

  const auto publish = [&]() noexcept {
    for (std::size_t slot = 0; slot < selected_count; ++slot)
    {
      auto& component = static_cast<PsiFormerWF&>(
          wfc_list[resource.walker_indices[slot]]);
      component.current_sign_ = resource.staged_signs[slot];
      component.log_value_ = makeLogValue(
          resource.staged_signs[slot],
          resource.staged_log_magnitudes[slot]);
      component.accepted_configuration_identity_ =
          resource.configuration_identities[slot];
      component.accepted_parameter_version_ = parameter_version;
      component.accepted_state_requirement_ =
          resource.preservation_flags[slot] != 0
          ? AcceptedStateRequirement::FULL_SPATIAL
          : AcceptedStateRequirement::VALUE_ONLY;
      component.observed_parameter_version_ = parameter_version;
    }
    for (std::size_t slot = 0; slot < selected_count; ++slot)
      static_cast<PsiFormerWF&>(
          wfc_list[resource.walker_indices[slot]]).accepted_value_valid_ = true;
  };
  static_assert(noexcept(publish()));
  publish();
}

// Retain inherited scalar group preparation only for legacy no-plan execution.
void PsiFormerWF::prepareGroup(ParticleSet& particles, int group_index)
{
  requireUnplannedScalarEvaluation("prepareGroup");
  WaveFunctionComponent::prepareGroup(particles, group_index);
}

// Retain inherited crowd group preparation only for legacy no-plan execution.
void PsiFormerWF::mw_prepareGroup(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int group_index) const
{
  requireUnplannedScalarEvaluation("mw_prepareGroup");
  WaveFunctionComponent::mw_prepareGroup(wfc_list, p_list, group_index);
}

// Evaluate and cache the wavefunction ratio for one proposed electron position.
PsiFormerWF::PsiValue PsiFormerWF::ratio(ParticleSet& p, int iat)
{
  requireUnplannedScalarEvaluation("single-particle ratio");
  requireNoSelectedParticleProposal("ratio");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  auto cache_and_form_ratio = [&](double sign, double logabs) {
    if (!acceptedStateMatches(p, transaction.parameterVersion(), AcceptedStateRequirement::VALUE_ONLY))
    {
      invalidateParameterCaches(transaction.parameterVersion());
      throw std::logic_error("PsiFormer ratio requested before evaluateLog for the current parameter version");
    }

    // Cache proposal state so acceptMove can commit it without reevaluating the
    // network.
    cacheSingleParticleProposal(
        sign, logabs, configurationIdentity(p, iat), iat,
        transaction.parameterVersion(), ProposalOrigin::SCALAR_RATIO_VALUE);
    return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
  };

  if (transaction.state().direct_value_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectValueResult result = evaluateDirectValuePositionsUnderRead(
        transaction, p, iat, nullptr);
    return cache_and_form_ratio(result.sign, result.logabs);
  }

  const pf::Result result = evaluatePositionsUnderRead(
      transaction, p, iat, nullptr, EvaluationPurpose::VALUE_ONLY);
  return cache_and_form_ratio(result.sign, result.logabs);
}

// Batch one common particle index across a crowd while retaining proposal state
// independently on each clone.
void PsiFormerWF::mw_calcRatio(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<PsiValue>& ratios) const
{
  requireUnplannedScalarEvaluation("mw_calcRatio");
  if (wfc_list.size() != p_list.size())
    throw std::invalid_argument("PsiFormer mw_calcRatio list sizes do not match");
  if (wfc_list.empty())
  {
    ratios.clear();
    return;
  }

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).requireNoSelectedParticleProposal(
        "mw_calcRatio");

  PsiFormerReadTransaction transaction(*leader.model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  std::vector<double> staged_sign(wfc_list.size());
  std::vector<double> staged_logabs(wfc_list.size());
  std::vector<std::uint64_t> staged_configuration(wfc_list.size());
  std::vector<PsiValue> staged_ratios(wfc_list.size());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    if (!component.acceptedStateMatches(
            p_list[walker], parameter_version, AcceptedStateRequirement::VALUE_ONLY))
    {
      component.invalidateParameterCaches(parameter_version);
      throw std::logic_error("PsiFormer mw_calcRatio requested before mw_evaluateLog");
    }
    staged_configuration[walker] =
        configurationIdentity(p_list[walker], particle_index);
  }

  if (transaction.state().direct_value_mode == DirectBackendMode::DIRECT)
  {
    auto& batch = *resource.batch_workspace;
    batch.resize(pf::DirectBatchMode::VALUE_ONLY, wfc_list.size());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      packBatchConfiguration(batch, walker, p_list[walker], particle_index);

    const pf::DirectBatchValueResultView result =
        transaction.state().direct_batch_executor.evaluateValues(batch);
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      if (result.parameter_version[walker] != parameter_version)
        throw std::logic_error("PsiFormer ratio batch observed inconsistent parameters");
      staged_sign[walker]   = result.sign[walker];
      staged_logabs[walker] = result.logabs[walker];
    }
  }
  else
  {
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const pf::Result result = component.evaluatePositionsUnderRead(
          transaction, p_list[walker], particle_index, nullptr,
          EvaluationPurpose::VALUE_ONLY);
      staged_sign[walker]   = result.sign;
      staged_logabs[walker] = result.logabs;
    }
  }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    staged_ratios[walker] = makeRatio(
        staged_sign[walker], staged_logabs[walker], component.current_sign_,
        std::real(component.log_value_));
  }

  ratios.swap(staged_ratios);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.cacheSingleParticleProposal(
        staged_sign[walker], staged_logabs[walker],
        staged_configuration[walker], particle_index, parameter_version,
        ProposalOrigin::MW_CALC_RATIO_VALUE);
  }
}

// Return one accepted electron logarithmic gradient.
PsiFormerWF::GradType PsiFormerWF::evalGrad(ParticleSet& p, int iat)
{
  requireUnplannedScalarEvaluation("active-electron gradient");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  auto scatter = [](const auto& source) {
    GradType gradient;
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] = source[dimension];
    return gradient;
  };

  if (transaction.state().direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositionsUnderRead(
        transaction, p, -1, nullptr,
        EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
    return scatter(result.gradient);
  }

  const pf::Result result = evaluatePositionsUnderRead(
      transaction, p, -1, nullptr,
      EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
  return scatter(result.active_gradient);
}

// Retain the spin-independent scalar wrapper only for legacy no-plan execution.
PsiFormerWF::GradType PsiFormerWF::evalGradWithSpin(
    ParticleSet& particles,
    int particle_index,
    ComplexType& spin_gradient)
{
  requireUnplannedScalarEvaluation("evalGradWithSpin");
  return WaveFunctionComponent::evalGradWithSpin(
      particles, particle_index, spin_gradient);
}

// Imported PsiFormer models keep their nuclear coordinates fixed and cannot
// supply the ionic derivatives required by force estimators.
PsiFormerWF::GradType PsiFormerWF::evalGradSource(ParticleSet&, ParticleSet&, int)
{
  throw std::runtime_error(
      "PsiFormer uses fixed nuclei and does not support source-ion gradients or force evaluation");
}

PsiFormerWF::GradType PsiFormerWF::evalGradSource(
    ParticleSet&,
    ParticleSet&,
    int,
    TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>&,
    TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>&)
{
  throw std::runtime_error(
      "PsiFormer uses fixed nuclei and does not support source-ion gradients or force evaluation");
}

// Evaluate accepted active-electron gradients through the legacy or planned path.
void PsiFormerWF::mw_evalGrad(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<GradType>& gradients) const
{
  // Preserve the historical lazy/oracle implementation behind the explicit
  // no-policy branch.
  if (!batch_execution_plan_)
  {
    if (wfc_list.size() != p_list.size() || wfc_list.size() != gradients.size())
      throw std::invalid_argument("PsiFormer mw_evalGrad list sizes do not match");
    if (wfc_list.empty())
      return;

    const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
    auto& resource     = requireMultiWalkerResource(wfc_list);
    PsiFormerReadTransaction transaction(*leader.model_state_);
    const std::size_t parameter_version = transaction.parameterVersion();
    std::vector<GradType> staged_gradients(wfc_list.size());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
          parameter_version);

    if (transaction.state().direct_spatial_mode == DirectBackendMode::DIRECT)
    {
      auto& batch = *resource.batch_workspace;
      batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT,
                   wfc_list.size());
      resource.active_electrons.assign(
          wfc_list.size(), static_cast<std::size_t>(particle_index));
      for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
        packBatchConfiguration(batch, walker, p_list[walker]);

      const pf::DirectBatchSpatialResultView result =
          transaction.state().direct_batch_executor.evaluateActive(
              batch, resource.active_electrons.data());
      for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      {
        if (result.parameter_version[walker] != parameter_version)
          throw std::logic_error("PsiFormer active-gradient batch observed inconsistent parameters");
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          staged_gradients[walker][dimension] =
              result.gradient[walker * result.gradient_stride + dimension];
      }
    }
    else
      for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      {
        auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
        const pf::Result result = component.evaluatePositionsUnderRead(
            transaction, p_list[walker], -1, nullptr,
            EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, particle_index);
        if (result.active_gradient.size() != 3)
          throw std::logic_error("PsiFormer multiwalker active gradient has the wrong shape");
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          staged_gradients[walker][dimension] = result.active_gradient[dimension];
      }

    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      gradients[walker] = staged_gradients[walker];
    return;
  }

  // A negative signed index must never acquire meaning through conversion to
  // the unsigned request/evaluator representation.
  if (particle_index < 0)
    throw std::out_of_range(
        "PsiFormer planned ACTIVE_GRADIENT has a negative active electron");

  const std::size_t walker_count = wfc_list.size();
  PlannedRuntimeRequest request;
  request.operation            = PlannedOperation::ACTIVE_GRADIENT;
  request.live_walkers         = walker_count;
  request.dense_configurations = walker_count;
  request.active_electron      = static_cast<std::size_t>(particle_index);
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (gradients.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned ACTIVE_GRADIENT output size does not match the crowd");

  GradType* const output_data           = gradients.data();
  const std::size_t output_capacity     = gradients.capacity();
  PsiFormerMultiWalkerResource& resource = access.resource;
  resource.requireActiveGradientStaging(walker_count);
  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;
  const std::size_t electron_count =
      access.participant.plan().particleCount();

  const CheckedMemoryRange output_range = checkedMemoryRange(
      output_data, walker_count,
      "PsiFormer planned ACTIVE_GRADIENT output range overflowed");
  const std::size_t output_bytes = output_range.end - output_range.begin;
  const auto require_output_nonoverlap = [&]() {
    if (resource.overlapsStagingStorage(output_data, output_bytes) ||
        batch.overlapsStorage(output_data, output_bytes))
      throw std::invalid_argument(
          "PsiFormer planned ACTIVE_GRADIENT output aliases prepared scratch");

    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
      for (const CheckedMemoryRange internal : {
               checkedMemoryRange(
                   component.accepted_gradient_.data(), electron_count,
                   "PsiFormer accepted gradient range overflowed"),
               checkedMemoryRange(
                   component.accepted_laplacian_.data(), electron_count,
                   "PsiFormer accepted Laplacian range overflowed"),
               checkedMemoryRange(
                   component.proposed_gradient_.data(), electron_count,
                   "PsiFormer proposed gradient range overflowed"),
               checkedMemoryRange(
                   component.proposed_laplacian_.data(), electron_count,
                   "PsiFormer proposed Laplacian range overflowed")})
        if (memoryRangesOverlap(output_range, internal))
          throw std::invalid_argument(
              "PsiFormer planned ACTIVE_GRADIENT output aliases component state");

      const ParticleSet& particles = p_list[lane];
      const auto& soa_positions =
          particles.getCoordinates().getAllParticlePos();
      if (soa_positions.capacity() >
          std::numeric_limits<std::size_t>::max() / 3)
        throw std::length_error(
            "PsiFormer ParticleSet SoA position range overflowed");
      for (const CheckedMemoryRange particle_storage : {
               checkedMemoryRange(
                   particles.R.data(), electron_count,
                   "PsiFormer ParticleSet position range overflowed"),
               checkedMemoryRange(
                   soa_positions.data(), 3 * soa_positions.capacity(),
                   "PsiFormer ParticleSet SoA position range overflowed"),
               checkedMemoryRange(
                   std::addressof(particles.getActivePos()), std::size_t{1},
                   "PsiFormer ParticleSet active-position range overflowed"),
               checkedMemoryRange(
                   particles.G.data(), electron_count,
                   "PsiFormer ParticleSet gradient range overflowed"),
               checkedMemoryRange(
                   particles.L.data(), electron_count,
                   "PsiFormer ParticleSet Laplacian range overflowed")})
        if (memoryRangesOverlap(output_range, particle_storage))
          throw std::invalid_argument(
              "PsiFormer planned ACTIVE_GRADIENT output aliases ParticleSet storage");
    }
  };
  require_output_nonoverlap();

  // Phase A proves the accepted value/configuration key without synchronizing
  // stale clones or changing caller/resource state.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const bool valid_requirement =
        component.accepted_state_requirement_ ==
            AcceptedStateRequirement::VALUE_ONLY ||
        component.accepted_state_requirement_ ==
            AcceptedStateRequirement::FULL_SPATIAL;
    if (!component.acceptedStateMatches(
            p_list[lane], component.observed_parameter_version_,
            AcceptedStateRequirement::VALUE_ONLY) ||
        component.accepted_parameter_version_ !=
            component.observed_parameter_version_)
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT requires current accepted value state");
    if (!valid_requirement ||
        !isCoherentAcceptedValue(component.current_sign_,
                                 component.log_value_))
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT has invalid accepted value state");
  }

  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_spatial_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned ACTIVE_GRADIENT requires the direct spatial backend");
  const std::size_t parameter_version = transaction.parameterVersion();
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (!component.acceptedStateMatches(
            p_list[lane], parameter_version,
            AcceptedStateRequirement::VALUE_ONLY) ||
        component.observed_parameter_version_ != parameter_version ||
        (component.accepted_state_requirement_ !=
             AcceptedStateRequirement::VALUE_ONLY &&
         component.accepted_state_requirement_ !=
             AcceptedStateRequirement::FULL_SPATIAL) ||
        !isCoherentAcceptedValue(component.current_sign_,
                                 component.log_value_))
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT has stale accepted parameters");
  }

  batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    resource.active_electrons[lane] =
        static_cast<std::size_t>(particle_index);
    packBatchConfiguration(batch, lane, p_list[lane]);
  }

  const pf::DirectBatchSpatialResultView result =
      transaction.state().direct_batch_executor.evaluateActive(
          batch, resource.active_electrons.data());
  if (!batch.ownsSpatialResult(
          result, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
          walker_count))
    throw std::logic_error(
        "PsiFormer planned ACTIVE_GRADIENT result is not the exact workspace-owned view");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    if (result.parameter_version[lane] != parameter_version)
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT observed inconsistent parameters");
    if ((result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]))
      throw std::runtime_error(
          "PsiFormer planned ACTIVE_GRADIENT produced an invalid value");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double native =
          result.gradient[lane * result.gradient_stride + dimension];
      if (!psiformer::determinant::isFiniteReal(native))
        throw std::runtime_error(
            "PsiFormer planned ACTIVE_GRADIENT produced a non-finite gradient");
      resource.staged_gradients[lane][dimension] =
          static_cast<ValueType>(native);
      if (!isFiniteWavefunctionValue(
              resource.staged_gradients[lane][dimension]))
        throw std::overflow_error(
            "PsiFormer planned ACTIVE_GRADIENT gradient conversion is non-finite");
    }
  }

  // Phase B repeats the complete borrowed/state/result evidence while the
  // version read lock remains held, immediately before the failure seam and
  // mechanically nonthrowing caller publication.
  if (!batch.ownsSpatialResult(
          result, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
          walker_count) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned ACTIVE_GRADIENT storage changed during evaluation");
  resource.requireActiveGradientStaging(walker_count);
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned ACTIVE_GRADIENT runtime evidence changed during evaluation");
  if (gradients.size() != walker_count || gradients.data() != output_data ||
      gradients.capacity() != output_capacity)
    throw std::logic_error(
        "PsiFormer planned ACTIVE_GRADIENT caller output changed during evaluation");
  require_output_nonoverlap();

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (component.model_state_.get() != model_state_.get() ||
        component.optimization_metadata_.get() != optimization_metadata_.get() ||
        component.bound_particle_set_ != &p_list[lane] ||
        component.acquired_crowd_leader_ != this ||
        component.acquired_lane_index_ != lane ||
        component.acquired_crowd_size_ != walker_count ||
        !component.batch_execution_plan_.sameBinding(access.participant) ||
        !component.hasPreparedBatchExecutionClone(access.participant) ||
        !component.acceptedStateMatches(
            p_list[lane], parameter_version,
            AcceptedStateRequirement::VALUE_ONLY) ||
        component.observed_parameter_version_ != parameter_version ||
        (component.accepted_state_requirement_ !=
             AcceptedStateRequirement::VALUE_ONLY &&
         component.accepted_state_requirement_ !=
             AcceptedStateRequirement::FULL_SPATIAL) ||
        !isCoherentAcceptedValue(component.current_sign_,
                                 component.log_value_) ||
        resource.active_electrons[lane] !=
            static_cast<std::size_t>(particle_index) ||
        result.parameter_version[lane] != parameter_version ||
        (result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]))
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT lane evidence changed during evaluation");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double native =
          result.gradient[lane * result.gradient_stride + dimension];
      const ValueType staged = static_cast<ValueType>(native);
      if (!psiformer::determinant::isFiniteReal(native) ||
          !isFiniteWavefunctionValue(staged) ||
          resource.staged_gradients[lane][dimension] != staged)
        throw std::logic_error(
            "PsiFormer planned ACTIVE_GRADIENT result staging changed during evaluation");
    }
  }

  if (fail_planned_active_gradient_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned ACTIVE_GRADIENT pre-publication failure");

  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      gradients[lane] = resource.staged_gradients[lane];
  };
  static_assert(noexcept(publish()));
  publish();
}

// Retain the spin-independent crowd wrapper only for legacy no-plan execution.
void PsiFormerWF::mw_evalGradWithSpin(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<GradType>& gradients,
    std::vector<ComplexType>& spin_gradients) const
{
  requireUnplannedScalarEvaluation("mw_evalGradWithSpin");
  WaveFunctionComponent::mw_evalGradWithSpin(
      wfc_list, p_list, particle_index, gradients, spin_gradients);
}

// Evaluate a proposed ratio and gradient in one native-model traversal.
PsiFormerWF::PsiValue PsiFormerWF::ratioGrad(ParticleSet& p, int iat, GradType& gradient)
{
  requireUnplannedScalarEvaluation("single-particle ratio-gradient");
  requireNoSelectedParticleProposal("ratioGrad");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  auto scatter = [&](double sign, double logabs, const auto& active_gradient) {
    if (!acceptedStateMatches(p, transaction.parameterVersion(), AcceptedStateRequirement::VALUE_ONLY))
    {
      invalidateParameterCaches(transaction.parameterVersion());
      throw std::logic_error("PsiFormer ratioGrad requested before evaluateLog for the current parameter version");
    }

    // Evaluate the proposal once and return both its ratio and active-electron
    // gradient.
    cacheSingleParticleProposal(
        sign, logabs, configurationIdentity(p, iat), iat,
        transaction.parameterVersion(),
        ProposalOrigin::SCALAR_RATIO_GRADIENT_ACTIVE);
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] += active_gradient[dimension];
    return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
  };

  if (transaction.state().direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositionsUnderRead(
        transaction, p, iat, nullptr,
        EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
    return scatter(result.sign, result.logabs, result.gradient);
  }

  const pf::Result result = evaluatePositionsUnderRead(
      transaction, p, iat, nullptr,
      EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
  return scatter(result.sign, result.logabs, result.active_gradient);
}

// Retain the scalar POS_SPIN ratio-gradient wrapper only in legacy mode.
PsiFormerWF::PsiValue PsiFormerWF::ratioGradWithSpin(
    ParticleSet& particles,
    int particle_index,
    GradType& gradient,
    ComplexType& spin_gradient)
{
  requireUnplannedScalarEvaluation("ratioGradWithSpin");
  return WaveFunctionComponent::ratioGradWithSpin(
      particles, particle_index, gradient, spin_gradient);
}

// Evaluate and cache crowd proposal ratios and gradients in legacy no-plan mode.
void PsiFormerWF::mw_ratioGrad(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<PsiValue>& ratios,
    std::vector<GradType>& gradients) const
{
  requireUnplannedScalarEvaluation("mw_ratioGrad");
  if (wfc_list.size() != p_list.size() || wfc_list.size() != gradients.size())
    throw std::invalid_argument("PsiFormer mw_ratioGrad list sizes do not match");
  if (wfc_list.empty())
  {
    ratios.clear();
    return;
  }

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).requireNoSelectedParticleProposal(
        "mw_ratioGrad");

  PsiFormerReadTransaction transaction(*leader.model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  std::vector<double> staged_sign(wfc_list.size());
  std::vector<double> staged_logabs(wfc_list.size());
  std::vector<std::uint64_t> staged_configuration(wfc_list.size());
  std::vector<PsiValue> staged_ratios(wfc_list.size());
  std::vector<GradType> staged_gradients(wfc_list.size());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    if (!component.acceptedStateMatches(
            p_list[walker], parameter_version, AcceptedStateRequirement::VALUE_ONLY))
    {
      component.invalidateParameterCaches(parameter_version);
      throw std::logic_error("PsiFormer mw_ratioGrad requested before mw_evaluateLog");
    }
    staged_configuration[walker] =
        configurationIdentity(p_list[walker], particle_index);
  }

  if (transaction.state().direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    auto& batch = *resource.batch_workspace;
    batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT,
                 wfc_list.size());
    resource.active_electrons.assign(
        wfc_list.size(), static_cast<std::size_t>(particle_index));
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      packBatchConfiguration(batch, walker, p_list[walker], particle_index);

    const pf::DirectBatchSpatialResultView result =
        transaction.state().direct_batch_executor.evaluateActive(
            batch, resource.active_electrons.data());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      if (result.parameter_version[walker] != parameter_version)
        throw std::logic_error("PsiFormer ratio-gradient batch observed inconsistent parameters");
      staged_sign[walker]   = result.sign[walker];
      staged_logabs[walker] = result.logabs[walker];
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        staged_gradients[walker][dimension] =
            result.gradient[walker * result.gradient_stride + dimension];
    }
  }
  else
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const pf::Result result = component.evaluatePositionsUnderRead(
          transaction, p_list[walker], particle_index, nullptr,
          EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, particle_index);
      if (result.active_gradient.size() != 3)
        throw std::logic_error("PsiFormer multiwalker ratio gradient has the wrong shape");
      staged_sign[walker]   = result.sign;
      staged_logabs[walker] = result.logabs;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        staged_gradients[walker][dimension] = result.active_gradient[dimension];
    }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    staged_ratios[walker] = makeRatio(
        staged_sign[walker], staged_logabs[walker], component.current_sign_,
        std::real(component.log_value_));
  }

  ratios.swap(staged_ratios);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.cacheSingleParticleProposal(
        staged_sign[walker], staged_logabs[walker],
        staged_configuration[walker], particle_index, parameter_version,
        ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE);
    gradients[walker] += staged_gradients[walker];
  }
}

// Retain the crowd POS_SPIN ratio-gradient wrapper only in legacy mode.
void PsiFormerWF::mw_ratioGradWithSpin(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<PsiValue>& ratios,
    std::vector<GradType>& gradients,
    std::vector<ComplexType>& spin_gradients) const
{
  requireUnplannedScalarEvaluation("mw_ratioGradWithSpin");
  WaveFunctionComponent::mw_ratioGradWithSpin(
      wfc_list, p_list, particle_index, ratios, gradients, spin_gradients);
}

// Promote cached proposal state to accepted state after a successful move.
void PsiFormerWF::acceptMove(ParticleSet& particles, int particle_index, bool)
{
  requireUnplannedScalarEvaluation("acceptMove");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  if (has_proposal_)
  {
    if (proposal_origin_ != ProposalOrigin::SCALAR_RATIO_VALUE &&
        proposal_origin_ != ProposalOrigin::SCALAR_RATIO_GRADIENT_ACTIVE)
      throw std::logic_error(
          "PsiFormer scalar single-electron accept cannot resolve a proposal from a different origin");
    if (proposed_parameter_version_ != observed_parameter_version_ ||
        particle_index != proposed_particle_ ||
        proposed_configuration_identity_ != configurationIdentity(particles, particle_index))
    {
      clearProposalState();
      throw std::logic_error("PsiFormer accepted move does not match the cached proposal");
    }
    log_value_            = proposed_log_value_;
    current_sign_         = proposed_sign_;
    accepted_value_valid_ = true;
    accepted_configuration_identity_ = proposed_configuration_identity_;
    accepted_parameter_version_      = observed_parameter_version_;
    accepted_state_requirement_      = AcceptedStateRequirement::VALUE_ONLY;
  }
  clearProposalState();
}

// Forget cached proposal state after a rejected move.
void PsiFormerWF::restore(int particle_index)
{
  requireUnplannedScalarEvaluation("restore");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  if (has_proposal_)
  {
    if (proposal_origin_ != ProposalOrigin::SCALAR_RATIO_VALUE &&
        proposal_origin_ != ProposalOrigin::SCALAR_RATIO_GRADIENT_ACTIVE)
      throw std::logic_error(
          "PsiFormer scalar single-electron restore cannot resolve a proposal from a different origin");
    if (proposed_parameter_version_ != observed_parameter_version_ ||
        particle_index != proposed_particle_)
      throw std::logic_error("PsiFormer restored move does not match the cached proposal");
  }
  clearProposalState();
}

void PsiFormerWF::mw_accept_rejectMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    const std::vector<bool>& is_accepted,
    bool) const
{
  requireUnplannedScalarEvaluation("mw_accept_rejectMove");
  if (wfc_list.size() != p_list.size() || wfc_list.size() != is_accepted.size())
    throw std::invalid_argument("PsiFormer mw_accept_rejectMove list sizes do not match");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  if (this != &leader)
    throw std::logic_error("PsiFormer mw_accept_rejectMove must be invoked on the crowd leader");
  PsiFormerReadTransaction transaction(*leader.model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer accept/reject list contains components from different models");
    component.synchronizeParameterVersion(parameter_version);
    if (component.has_proposal_ &&
        component.proposal_origin_ != ProposalOrigin::MW_CALC_RATIO_VALUE &&
        component.proposal_origin_ != ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE)
      throw std::logic_error(
          "PsiFormer crowd single-electron resolution cannot resolve a proposal from a different origin");
    if (component.has_proposal_ &&
        (component.proposed_parameter_version_ != parameter_version ||
         component.proposed_particle_ != particle_index))
      throw std::logic_error("PsiFormer crowd accept/reject does not match the cached proposal");
    if (is_accepted[walker] && component.has_proposal_ &&
        component.proposed_configuration_identity_ !=
            configurationIdentity(p_list[walker], particle_index))
      throw std::logic_error("PsiFormer crowd accepted move does not match the cached proposal");
  }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (is_accepted[walker] && component.has_proposal_)
    {
      component.log_value_ = component.proposed_log_value_;
      component.current_sign_ = component.proposed_sign_;
      component.accepted_value_valid_ = true;
      component.accepted_configuration_identity_ = component.proposed_configuration_identity_;
      component.accepted_parameter_version_      = parameter_version;
      component.accepted_state_requirement_      = AcceptedStateRequirement::VALUE_ONLY;
    }
  }
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).resetProposalMetadata();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).has_proposal_ = false;
}

// Retain inherited scalar update completion only for legacy no-plan execution.
void PsiFormerWF::completeUpdates()
{
  requireUnplannedScalarEvaluation("completeUpdates");
  WaveFunctionComponent::completeUpdates();
}

// Retain inherited crowd completion only for legacy no-plan execution.
void PsiFormerWF::mw_completeUpdates(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  requireUnplannedScalarEvaluation("mw_completeUpdates");
  WaveFunctionComponent::mw_completeUpdates(wfc_list);
}

// Reserve fixed bulk and scalar slots without assuming that registration follows evaluation.
void PsiFormerWF::registerData(ParticleSet& particles, WFBufferType& buffer)
{
  requireUnplannedScalarEvaluation("registerData");
  requireNoSelectedParticleProposal("registerData");
  static_assert(std::numeric_limits<FullPrecRealType>::digits >= 32,
                "PsiFormer walker metadata requires exact 32-bit scalar limbs");
  if (particles.getTotalNum() !=
      static_cast<int>(model_state_->execution_plan.modelShape().electrons()))
    throw std::invalid_argument("PsiFormer walker buffer electron count differs from the model");
  resizeAcceptedSpatialStorage(particles.getTotalNum());
  buffer.add(accepted_gradient_.data(), accepted_gradient_.data() + accepted_gradient_.size());
  buffer.add(accepted_laplacian_.data(), accepted_laplacian_.data() + accepted_laplacian_.size());
  double placeholder = 0.0;
  for (std::size_t scalar = 0; scalar < WALKER_BUFFER_SCALAR_COUNT; ++scalar)
    buffer.add(placeholder);
}

// Reuse complete accepted products when every validity-key field still matches.
PsiFormerWF::LogValue PsiFormerWF::updateBuffer(ParticleSet& particles,
                                                WFBufferType& buffer,
                                                bool from_scratch)
{
  requireUnplannedScalarEvaluation("updateBuffer");
  requireNoSelectedParticleProposal("updateBuffer");
  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  const bool can_reuse = !from_scratch && acceptedStateMatches(
      particles, parameter_version, AcceptedStateRequirement::FULL_SPATIAL);
  if (!can_reuse)
    requireUnplannedScalarEvaluation("updateBuffer refresh");
  synchronizeParameterVersion(parameter_version);
  if (can_reuse)
    accumulateAcceptedSpatial(particles.G, particles.L);
  else
    evaluateLogUnderRead(transaction, particles, particles.G, particles.L);

  if (!acceptedStateMatches(particles, parameter_version,
                            AcceptedStateRequirement::FULL_SPATIAL))
    throw std::runtime_error("PsiFormer failed to refresh a coherent walker buffer");
  putAcceptedState(buffer);
  return log_value_;
}

// Restore only records whose model, parameter, configuration, and products match.
void PsiFormerWF::copyFromBuffer(ParticleSet& particles, WFBufferType& buffer)
{
  requireUnplannedScalarEvaluation("copyFromBuffer");
  requireNoSelectedParticleProposal("copyFromBuffer");
  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  synchronizeParameterVersion(parameter_version);
  try
  {
    getAcceptedState(particles, buffer, parameter_version);
  }
  catch (...)
  {
    invalidateParameterCaches(parameter_version);
    throw;
  }
}

// Copy selected score entries while both the model and optimizer mapping are immutable.
void PsiFormerWF::gatherSelectedGradientUnderRead(
    const PsiFormerDerivativeReadTransaction& transaction,
    const double* flat_gradient,
    std::size_t gradient_size,
    ValueType scale,
    std::size_t destination_size,
    SelectedDerivativeDelta& output) const
{
  if (&transaction.modelTransaction().state() != model_state_.get() ||
      &transaction.metadata() != optimization_metadata_.get())
    throw std::logic_error("PsiFormer derivative transaction belongs to a different clone family");
  if (gradient_size != 0 && flat_gradient == nullptr)
    throw std::invalid_argument("PsiFormer native derivative buffer is null");

  const auto& selected_flat_indices = transaction.selectedFlatIndices();
  const OptVariables& variables     = transaction.variables();
  if (variables.size() != selected_flat_indices.size())
    throw std::logic_error("PsiFormer optimizer mapping has the wrong size");

  output.clear();
  output.reserve(static_cast<std::size_t>(variables.size_of_active()));
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (static_cast<std::size_t>(global_index) >= destination_size)
      throw std::out_of_range("PsiFormer derivative destination index is out of range");

    const std::size_t flat_index = selected_flat_indices[local_index];
    if (flat_index >= gradient_size)
      throw std::out_of_range("PsiFormer native derivative is missing a selected flat index");
    if (!psiformer::determinant::isFiniteReal(flat_gradient[flat_index]))
      throw std::runtime_error("PsiFormer native parameter derivative is non-finite");
    const ValueType value = scale * ValueType(flat_gradient[flat_index]);
    if (!psiformer::determinant::isFiniteReal(std::real(value)) ||
        !psiformer::determinant::isFiniteReal(std::imag(value)))
      throw std::runtime_error("PsiFormer scaled parameter derivative is non-finite");
    output.emplace_back(static_cast<std::size_t>(global_index), value);
  }
}

// Replace each electron by the common virtual position in one state-isolated batch.
void PsiFormerWF::evaluateRatiosAlltoOne(ParticleSet& particles, std::vector<ValueType>& ratios)
{
  if (particles.isSpinor())
    throw std::invalid_argument("PsiFormer all-to-one ratios do not support spinor virtual moves");
  if (ratios.size() != static_cast<std::size_t>(particles.getTotalNum()))
    throw std::invalid_argument("PsiFormer all-to-one ratio output has the wrong size");

  const std::size_t configurations = checkedBatchMemoryAdd(
      static_cast<std::size_t>(particles.getTotalNum()), 1,
      "PsiFormer all-to-one configuration count");
  ValueType* staged_values = requirePlannedScalarValuePublication(
      configurations, ratios.size(), "all-to-one ratios");
  std::vector<ValueType> legacy_staged_ratios;
  if (!staged_values)
  {
    legacy_staged_ratios.resize(ratios.size());
    staged_values = legacy_staged_ratios.data();
  }
  const auto publish_ratios = [&]() {
    if (batch_execution_plan_)
      std::copy_n(staged_values, ratios.size(), ratios.begin());
    else
      ratios.swap(legacy_staged_ratios);
  };

  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());

  // Oracle and compare modes retain the scalar validation path but use the explicit
  // common position; the WaveFunctionComponent default incorrectly calls activeR().
  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
  {
    const pf::Result reference = evaluatePositionsUnderRead(
        transaction, particles, -1, nullptr, EvaluationPurpose::VALUE_ONLY);
    for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    {
      const pf::Result moved = evaluatePositionsUnderRead(
          transaction, particles, electron, &particles.getActivePos(),
          EvaluationPurpose::VALUE_ONLY);
      staged_values[electron] =
          makeRatio(moved.sign, moved.logabs, reference.sign, reference.logabs);
    }
    publish_ratios();
    return;
  }

  const std::size_t parameter_version = transaction.parameterVersion();
  auto& batch = requireDirectBatchWorkspace();
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, configurations);
  packBatchConfiguration(batch, 0, particles);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    packBatchConfiguration(batch, static_cast<std::size_t>(electron) + 1, particles, electron,
                           &particles.getActivePos());

  const pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(batch);
  if (result.parameter_version[0] != parameter_version)
    throw std::logic_error("PsiFormer all-to-one reference observed inconsistent parameters");
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    const std::size_t configuration = static_cast<std::size_t>(electron) + 1;
    if (result.parameter_version[configuration] != parameter_version)
      throw std::logic_error("PsiFormer all-to-one batch observed inconsistent parameters");
    staged_values[electron] = makeRatio(
        result.sign[configuration], result.logabs[configuration], result.sign[0],
        result.logabs[0]);
  }
  publish_ratios();
}

// Evaluate independent full-network ratios for all quadrature positions without mutating walker state.
void PsiFormerWF::evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios)
{
  if (ratios.size() !=
      static_cast<std::size_t>(virtual_particles.getTotalNum()))
    throw std::invalid_argument(
        "PsiFormer virtual-particle ratio output has the wrong size");
  requireNoPlannedEcpScalarDispatch("scalar virtual-ratio evaluation");
  const std::size_t configurations = checkedBatchMemoryAdd(
      ratios.size(), 1, "PsiFormer virtual-ratio configuration count");
  (void)requirePlannedScalarValuePublication(
      configurations, ratios.size(), "virtual ratios");

  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  evaluateRatiosUnderRead(transaction, virtual_particles, ratios);
}

void PsiFormerWF::evaluateRatiosUnderRead(
    const PsiFormerReadTransaction& transaction,
    const VirtualParticleSet& virtual_particles,
    std::vector<ValueType>& ratios)
{
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer virtual-ratio transaction belongs to a different model");
  if (virtual_particles.getRefPS().isSpinor())
    throw std::invalid_argument("PsiFormer nonlocal ratios do not support spinor virtual moves");
  if (ratios.size() != static_cast<std::size_t>(virtual_particles.getTotalNum()))
    throw std::invalid_argument("PsiFormer virtual-particle ratio output has the wrong size");

  const ParticleSet& reference = virtual_particles.getRefPS();
  const int electron = virtual_particles.refPtcl;
  if (electron < 0 || electron >= reference.getTotalNum())
    throw std::out_of_range("PsiFormer virtual-particle reference electron is invalid");

  const std::size_t configurations = checkedBatchMemoryAdd(
      ratios.size(), 1, "PsiFormer virtual-ratio configuration count");
  ValueType* staged_values = requirePlannedScalarValuePublication(
      configurations, ratios.size(), "virtual ratios");
  std::vector<ValueType> legacy_staged_ratios;
  if (!staged_values)
  {
    legacy_staged_ratios.resize(ratios.size());
    staged_values = legacy_staged_ratios.data();
  }
  const auto publish_ratios = [&]() {
    if (batch_execution_plan_)
      std::copy_n(staged_values, ratios.size(), ratios.begin());
    else
      ratios.swap(legacy_staged_ratios);
  };

  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
  {
    const pf::Result reference_result =
        evaluatePositionsUnderRead(transaction, reference, -1, nullptr,
                                   EvaluationPurpose::VALUE_ONLY);
    for (std::size_t move = 0; move < ratios.size(); ++move)
    {
      const pf::Result virtual_result = evaluatePositionsUnderRead(
          transaction, reference, electron, &virtual_particles.R[move],
          EvaluationPurpose::VALUE_ONLY);
      staged_values[move] = makeRatio(
          virtual_result.sign, virtual_result.logabs, reference_result.sign,
          reference_result.logabs);
    }
    publish_ratios();
    return;
  }

  const std::size_t parameter_version = transaction.parameterVersion();
  auto& batch = requireDirectBatchWorkspace();
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, configurations);
  packBatchConfiguration(batch, 0, reference);
  for (std::size_t move = 0; move < ratios.size(); ++move)
    packBatchConfiguration(batch, move + 1, reference, electron, &virtual_particles.R[move]);

  const pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(batch);
  if (result.parameter_version[0] != parameter_version)
    throw std::logic_error("PsiFormer virtual-ratio reference observed inconsistent parameters");
  for (std::size_t move = 0; move < ratios.size(); ++move)
  {
    if (result.parameter_version[move + 1] != parameter_version)
      throw std::logic_error("PsiFormer virtual-ratio batch observed inconsistent parameters");
    staged_values[move] = makeRatio(
        result.sign[move + 1], result.logabs[move + 1], result.sign[0],
        result.logabs[0]);
  }
  publish_ratios();
}

// Evaluate a descriptor-ordered virtual batch from one sparse reference per active walker.
WaveFunctionComponent::EvaluationStamp PsiFormerWF::mw_evaluateVirtualRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& virtual_batch,
    std::vector<ValueType>& ratios) const
{
  if (this != std::addressof(wfc_list.getLeader()))
    throw std::invalid_argument(
        "PsiFormer mw_evaluateVirtualRatios must be invoked on the component-list leader");
  if (wfc_list.size() != virtual_batch.walkerCount() ||
      p_list.size() != virtual_batch.walkerCount() ||
      vp_scratch_list.size() != virtual_batch.walkerCount())
    throw std::invalid_argument(
        "PsiFormer mw_evaluateVirtualRatios list sizes do not match the descriptor walker count");

  virtual_batch.validateOutputExtent(ratios.size());
  virtual_batch.validateFor(p_list);

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  const std::size_t electron_count =
      leader.model_state_->execution_plan.modelShape().electrons();

  // Validate the complete crowd before changing resource scratch or lazily
  // invalidating clone-local state after a genuine parameter publication.
  for (std::size_t walker = 0; walker < virtual_batch.walkerCount(); ++walker)
  {
    if (typeid(wfc_list[walker]) != typeid(leader))
      throw std::invalid_argument(
          "PsiFormer mw_evaluateVirtualRatios component clones have different dynamic types");
    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wfc_list[walker]) == std::addressof(wfc_list[other]))
        throw std::invalid_argument(
            "PsiFormer mw_evaluateVirtualRatios requires one distinct component clone per walker");
      if (std::addressof(vp_scratch_list[walker]) ==
          std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "PsiFormer mw_evaluateVirtualRatios requires one distinct scratch object per walker");
    }

    const ParticleSet* scratch_as_particles =
        static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < virtual_batch.walkerCount();
         ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "PsiFormer mw_evaluateVirtualRatios scratch objects must not alias reference walkers");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "PsiFormer mw_evaluateVirtualRatios reference and scratch spinor modes do not match");
    if (p_list[walker].isSpinor())
      throw std::invalid_argument(
          "PsiFormer flattened nonlocal ratios do not support spinor virtual moves");
    if (static_cast<std::size_t>(p_list[walker].getTotalNum()) != electron_count)
      throw std::invalid_argument(
          "PsiFormer flattened virtual-ratio electron count differs from the model");

    const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument(
          "PsiFormer flattened virtual-ratio crowd contains components from different models");
  }

  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  constexpr std::size_t no_reference = std::numeric_limits<std::size_t>::max();
  resource.active_virtual_walkers.clear();
  resource.virtual_reference_indices.assign(virtual_batch.walkerCount(), no_reference);

  // Compact only walkers named by at least one descriptor segment.  Multiple
  // segments, including different moved electrons, share this reference slot.
  for (const VirtualParticleBatch::Segment& segment : virtual_batch.segments())
  {
    const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
    if (resource.virtual_reference_indices[walker] == no_reference)
    {
      resource.virtual_reference_indices[walker] =
          resource.active_virtual_walkers.size();
      resource.active_virtual_walkers.push_back(walker);
    }
  }

  resource.flat_virtual_ratios.resize(virtual_batch.size());
  PsiFormerReadTransaction transaction(*leader.model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  for (std::size_t walker = 0; walker < virtual_batch.walkerCount(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
        parameter_version);

  const EvaluationStamp evaluation_stamp = EvaluationStamp::versioned(
      leader.model_state_.get(), static_cast<std::uint64_t>(parameter_version));

  const DirectBackendMode value_mode = transaction.state().direct_value_mode;
  const std::size_t reference_count = resource.active_virtual_walkers.size();
  pf::DirectBatchValueResultView sparse_result;
  if (value_mode != DirectBackendMode::ORACLE)
  {
    pf::DirectBatchWorkspace& workspace = *resource.batch_workspace;
    workspace.resizeSparseValues(reference_count, virtual_batch.size());

    // Pack accepted references once without assuming a ParticleSet precision or
    // storage layout.  The sparse workspace owns the only native double copy.
    for (std::size_t reference = 0; reference < reference_count; ++reference)
    {
      const ParticleSet& particles =
          p_list[resource.active_virtual_walkers[reference]];
      for (std::size_t electron = 0; electron < electron_count; ++electron)
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          workspace.setReferencePosition(
              reference, electron, dimension, particles.R[electron][dimension]);
    }

    // Preserve descriptor flat order exactly in the sparse replacement table.
    for (std::size_t segment_index = 0;
         segment_index < virtual_batch.segmentCount(); ++segment_index)
    {
      const VirtualParticleBatch::Slice slice = virtual_batch.slice(segment_index);
      const std::size_t reference = resource.virtual_reference_indices[
          static_cast<std::size_t>(slice.walkerId())];
      for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
      {
        pf::GeometryPosition position{};
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          position[dimension] = slice.absolutePosition(local_index)[dimension];
        workspace.setVirtualReplacement(
            slice.flatOffset() + local_index, reference,
            static_cast<std::size_t>(slice.electronId()), position);
      }
    }

    sparse_result =
        transaction.state().direct_batch_executor.evaluateValues(workspace);
    if (sparse_result.size != reference_count + virtual_batch.size())
      throw std::logic_error(
          "PsiFormer sparse virtual batch returned the wrong result extent");
    for (std::size_t configuration = 0; configuration < sparse_result.size;
         ++configuration)
      if (sparse_result.parameter_version[configuration] != parameter_version)
        throw std::logic_error(
            "PsiFormer sparse virtual batch observed inconsistent parameters");

    for (std::size_t segment_index = 0;
         segment_index < virtual_batch.segmentCount(); ++segment_index)
    {
      const VirtualParticleBatch::Slice slice = virtual_batch.slice(segment_index);
      const std::size_t reference = resource.virtual_reference_indices[
          static_cast<std::size_t>(slice.walkerId())];
      for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
      {
        const std::size_t flat_index = slice.flatOffset() + local_index;
        const std::size_t replacement = reference_count + flat_index;
        resource.flat_virtual_ratios[flat_index] = makeRatio(
            sparse_result.sign[replacement], sparse_result.logabs[replacement],
            sparse_result.sign[reference], sparse_result.logabs[reference]);
      }
    }
  }

  if (value_mode != DirectBackendMode::DIRECT)
  {
    // Oracle mode evaluates only the native graph.  Compare mode also checks
    // every sparse reference/replacement against that established oracle.
    auto requireSparseMatch = [](double sparse_sign,
                                 double sparse_logabs,
                                 double oracle_sign,
                                 double oracle_logabs,
                                 const char* description) {
      if (sparse_sign != oracle_sign)
        throw std::runtime_error(std::string("PsiFormer sparse ") + description +
                                 " sign differs from the native oracle");
      if (sparse_logabs == oracle_logabs)
        return;
      if (!psiformer::determinant::isFiniteReal(sparse_logabs) ||
          !psiformer::determinant::isFiniteReal(oracle_logabs))
        throw std::runtime_error(std::string("PsiFormer sparse ") + description +
                                 " log amplitude differs from the native oracle");
      const double scale = std::max(std::abs(sparse_logabs),
                                    std::abs(oracle_logabs));
      if (std::abs(sparse_logabs - oracle_logabs) >
          2.0e-11 * (1.0 + scale))
        throw std::runtime_error(std::string("PsiFormer sparse ") + description +
                                 " log amplitude differs from the native oracle");
    };

    resource.virtual_reference_signs.resize(reference_count);
    resource.virtual_reference_logabs.resize(reference_count);
    for (std::size_t reference = 0; reference < reference_count; ++reference)
    {
      const std::size_t walker = resource.active_virtual_walkers[reference];
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const pf::Result reference_result = component.evaluatePositionsUnderRead(
          transaction, p_list[walker], -1, nullptr, EvaluationPurpose::VALUE_ONLY);
      if (value_mode == DirectBackendMode::COMPARE)
        requireSparseMatch(
            sparse_result.sign[reference], sparse_result.logabs[reference],
            reference_result.sign, reference_result.logabs, "reference");
      resource.virtual_reference_signs[reference] = reference_result.sign;
      resource.virtual_reference_logabs[reference] = reference_result.logabs;
    }

    for (std::size_t segment_index = 0;
         segment_index < virtual_batch.segmentCount(); ++segment_index)
    {
      const VirtualParticleBatch::Slice slice = virtual_batch.slice(segment_index);
      const std::size_t walker = static_cast<std::size_t>(slice.walkerId());
      const std::size_t reference = resource.virtual_reference_indices[walker];
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
      {
        const pf::Result virtual_result = component.evaluatePositionsUnderRead(
            transaction, p_list[walker], slice.electronId(),
            std::addressof(slice.absolutePosition(local_index)),
            EvaluationPurpose::VALUE_ONLY);
        const std::size_t flat_index = slice.flatOffset() + local_index;
        if (value_mode == DirectBackendMode::COMPARE)
        {
          const std::size_t replacement = reference_count + flat_index;
          requireSparseMatch(
              sparse_result.sign[replacement], sparse_result.logabs[replacement],
              virtual_result.sign, virtual_result.logabs, "replacement");
        }
        resource.flat_virtual_ratios[flat_index] = makeRatio(
            virtual_result.sign, virtual_result.logabs,
            resource.virtual_reference_signs[reference],
            resource.virtual_reference_logabs[reference]);
      }
    }
  }

  // ValueType copies cannot fail; publish only after all sparse/oracle results,
  // versions, and ratios have been validated.
  std::copy(resource.flat_virtual_ratios.begin(),
            resource.flat_virtual_ratios.end(), ratios.begin());
  return evaluation_stamp;
}

// Flatten one reference plus a ragged walker-major sequence of virtual moves.
void PsiFormerWF::mw_evaluateRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& virtual_particle_list,
    std::vector<std::vector<ValueType>>& ratios) const
{
  if (wfc_list.size() != virtual_particle_list.size() || wfc_list.size() != ratios.size())
    throw std::invalid_argument("PsiFormer mw_evaluateRatios list sizes do not match");
  if (wfc_list.empty())
    return;

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& virtual_particles = virtual_particle_list[walker];
    if (virtual_particles.getRefPS().isSpinor())
      throw std::invalid_argument("PsiFormer nonlocal ratios do not support spinor virtual moves");
    if (ratios[walker].size() != static_cast<std::size_t>(virtual_particles.getTotalNum()))
      throw std::invalid_argument("PsiFormer ragged virtual-ratio output has the wrong size");
    if (virtual_particles.refPtcl < 0 ||
        virtual_particles.refPtcl >= virtual_particles.getRefPS().getTotalNum())
      throw std::out_of_range("PsiFormer virtual-particle reference electron is invalid");
  }

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);

  resource.virtual_offsets.resize(wfc_list.size() + 1);
  resource.virtual_offsets[0] = 0;
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    resource.virtual_offsets[walker + 1] = resource.virtual_offsets[walker] + ratios[walker].size() + 1;

  PsiFormerReadTransaction transaction(*leader.model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  std::vector<std::vector<ValueType>> staged_ratios(wfc_list.size());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    staged_ratios[walker].resize(ratios[walker].size());
  }

  if (transaction.state().direct_value_mode == DirectBackendMode::DIRECT)
  {
    auto& batch = *resource.batch_workspace;
    batch.resize(pf::DirectBatchMode::VALUE_ONLY, resource.virtual_offsets.back());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      const auto& virtual_particles = virtual_particle_list[walker];
      const ParticleSet& reference = virtual_particles.getRefPS();
      const std::size_t begin = resource.virtual_offsets[walker];
      packBatchConfiguration(batch, begin, reference);
      for (std::size_t move = 0; move < ratios[walker].size(); ++move)
        packBatchConfiguration(batch, begin + move + 1, reference,
                               virtual_particles.refPtcl,
                               &virtual_particles.R[move]);
    }

    const pf::DirectBatchValueResultView result =
        transaction.state().direct_batch_executor.evaluateValues(batch);
    for (std::size_t configuration = 0;
         configuration < resource.virtual_offsets.back(); ++configuration)
      if (result.parameter_version[configuration] != parameter_version)
        throw std::logic_error("PsiFormer ragged virtual batch observed inconsistent parameters");

    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      const std::size_t reference = resource.virtual_offsets[walker];
      for (std::size_t move = 0; move < ratios[walker].size(); ++move)
      {
        const std::size_t configuration = reference + move + 1;
        staged_ratios[walker][move] = makeRatio(
            result.sign[configuration], result.logabs[configuration],
            result.sign[reference], result.logabs[reference]);
      }
    }
  }
  else
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const auto& virtual_particles = virtual_particle_list[walker];
      const ParticleSet& reference = virtual_particles.getRefPS();
      const pf::Result reference_result = component.evaluatePositionsUnderRead(
          transaction, reference, -1, nullptr, EvaluationPurpose::VALUE_ONLY);
      for (std::size_t move = 0; move < ratios[walker].size(); ++move)
      {
        const pf::Result virtual_result = component.evaluatePositionsUnderRead(
            transaction, reference, virtual_particles.refPtcl,
            &virtual_particles.R[move], EvaluationPurpose::VALUE_ONLY);
        staged_ratios[walker][move] = makeRatio(
            virtual_result.sign, virtual_result.logabs, reference_result.sign,
            reference_result.logabs);
      }
    }

  ratios.swap(staged_ratios);
}

void PsiFormerWF::mw_evaluateSpinorRatios(
    const RefVectorWithLeader<WaveFunctionComponent>&,
    const RefVectorWithLeader<const VirtualParticleSet>&,
    const RefVector<std::pair<ValueVector, ValueVector>>&,
    std::vector<std::vector<ValueType>>&) const
{
  throw std::invalid_argument("PsiFormer does not implement spinor or spin-orbit ECP ratios");
}

// Evaluate ratios and their logarithmic parameter-derivative changes for nonlocal ECP optimization.
void PsiFormerWF::evaluateDerivRatios(const VirtualParticleSet& virtual_particles,
                                      const OptVariables&,
                                      std::vector<ValueType>& ratios,
                                      Matrix<ValueType>& derivative_ratios)
{
  requireNoPlannedEcpScalarDispatch("scalar derivative-ratio evaluation");
  if (optimization_metadata_->enabled)
    requireUnplannedScalarDerivative("derivative-ratio evaluation");
  else
  {
    if (ratios.size() !=
        static_cast<std::size_t>(virtual_particles.getTotalNum()))
      throw std::invalid_argument(
          "PsiFormer virtual derivative-ratio output has the wrong shape");
    const std::size_t configurations = checkedBatchMemoryAdd(
        ratios.size(), 1,
        "PsiFormer virtual derivative-ratio configuration count");
    (void)requirePlannedScalarValuePublication(
        configurations, ratios.size(), "virtual derivative ratios");
  }
  PsiFormerDerivativeReadTransaction transaction(*model_state_, *optimization_metadata_);
  synchronizeParameterVersion(transaction.modelTransaction().parameterVersion());
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
  {
    evaluateRatiosUnderRead(transaction.modelTransaction(), virtual_particles, ratios);
    return;
  }
  if (virtual_particles.getRefPS().isSpinor())
    throw std::invalid_argument("PsiFormer nonlocal derivative ratios do not support spinor virtual moves");
  if (ratios.size() != static_cast<std::size_t>(virtual_particles.getTotalNum()) ||
      derivative_ratios.rows() != ratios.size())
    throw std::invalid_argument("PsiFormer virtual derivative-ratio output has the wrong shape");

  const ParticleSet& reference = virtual_particles.getRefPS();
  const int electron           = virtual_particles.refPtcl;
  if (electron < 0 || electron >= reference.getTotalNum())
    throw std::out_of_range("PsiFormer virtual-particle reference electron is invalid");

  const std::size_t destination_size =
      static_cast<std::size_t>(derivative_ratios.cols());
  const DirectBackendMode score_mode =
      transaction.modelTransaction().state().direct_score_mode;
  pf::DirectScoreWorkspace* score_workspace = score_mode == DirectBackendMode::DIRECT
      ? &requireDirectScoreWorkspace()
      : nullptr;
  SelectedDerivativeDelta requested_reference;
  SelectedDerivativeDelta requested_virtual;

  double reference_sign;
  double reference_logabs;
  if (score_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectScoreResult reference_result =
        evaluateDirectScorePositionsUnderRead(transaction.modelTransaction(),
                                              reference, -1, nullptr,
                                              *score_workspace);
    reference_sign   = reference_result.sign;
    reference_logabs = reference_result.logabs;
    gatherSelectedGradientUnderRead(
        transaction, reference_result.parameter_score.data,
        reference_result.parameter_score.size, ValueType(1), destination_size,
        requested_reference);
  }
  else
  {
    const pf::Result reference_result = evaluatePositionsUnderRead(
        transaction.modelTransaction(), reference, -1, nullptr,
        EvaluationPurpose::SCORE_ONLY);
    reference_sign   = reference_result.sign;
    reference_logabs = reference_result.logabs;
    gatherSelectedGradientUnderRead(
        transaction, reference_result.param_gradient.data(),
        reference_result.param_gradient.size(), ValueType(1), destination_size,
        requested_reference);
  }

  // Publish one complete compatibility row only after its ratio, mapping, and
  // selected score difference are valid. This keeps storage O(active) without
  // duplicating the caller's Q x parameter matrix.
  for (std::size_t move = 0; move < ratios.size(); ++move)
  {
    double virtual_sign;
    double virtual_logabs;
    if (score_mode == DirectBackendMode::DIRECT)
    {
      const pf::DirectScoreResult virtual_result =
          evaluateDirectScorePositionsUnderRead(
              transaction.modelTransaction(), reference, electron,
              &virtual_particles.R[move], *score_workspace);
      virtual_sign   = virtual_result.sign;
      virtual_logabs = virtual_result.logabs;
      gatherSelectedGradientUnderRead(
          transaction, virtual_result.parameter_score.data,
          virtual_result.parameter_score.size, ValueType(1), destination_size,
          requested_virtual);
    }
    else
    {
      const pf::Result virtual_result = evaluatePositionsUnderRead(
          transaction.modelTransaction(), reference, electron,
          &virtual_particles.R[move], EvaluationPurpose::SCORE_ONLY);
      virtual_sign   = virtual_result.sign;
      virtual_logabs = virtual_result.logabs;
      gatherSelectedGradientUnderRead(
          transaction, virtual_result.param_gradient.data(),
          virtual_result.param_gradient.size(), ValueType(1), destination_size,
          requested_virtual);
    }

    const ValueType staged_ratio = makeRatio(
        virtual_sign, virtual_logabs, reference_sign, reference_logabs);
    if (requested_virtual.size() != requested_reference.size())
      throw std::logic_error("PsiFormer virtual score mapping changed");
    for (std::size_t selected = 0; selected < requested_virtual.size(); ++selected)
    {
      if (requested_virtual[selected].first != requested_reference[selected].first)
        throw std::logic_error("PsiFormer virtual score mapping changed");
      const ValueType difference = requested_virtual[selected].second -
          requested_reference[selected].second;
      if (!psiformer::determinant::isFiniteReal(std::real(difference)) ||
          !psiformer::determinant::isFiniteReal(std::imag(difference)))
        throw std::runtime_error("PsiFormer virtual score difference is non-finite");
    }
    for (std::size_t selected = 0; selected < requested_virtual.size(); ++selected)
      derivative_ratios(move, requested_virtual[selected].first) +=
          requested_virtual[selected].second - requested_reference[selected].second;
    ratios[move] = staged_ratio;
  }
}

// Contract virtual score differences with total-wavefunction quadrature weights.
void PsiFormerWF::evaluateDerivRatiosWeighted(const VirtualParticleSet& virtual_particles,
                                              const OptVariables& optvars,
                                              const std::vector<ValueType>& total_weights,
                                              ParameterDerivativeView weighted_derivatives)
{
  if (weighted_derivatives.size != 0 && weighted_derivatives.data == nullptr)
    throw std::invalid_argument("PsiFormer weighted derivative destination is null");
  requireNoPlannedEcpScalarDispatch(
      "scalar weighted derivative-ratio evaluation");
  if (optimization_metadata_->enabled)
    requireUnplannedScalarDerivative("weighted derivative-ratio evaluation");
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  synchronizeParameterVersion(transaction.modelTransaction().parameterVersion());
  SelectedDerivativeDelta delta;
  evaluateDerivRatiosWeightedImpl(
      transaction, virtual_particles, optvars, total_weights,
      weighted_derivatives.size, nullptr, delta);
  for (const auto& [global_index, value] : delta)
    weighted_derivatives[global_index] += value;
}

// Contract one flattened virtual batch without materializing a Q-by-parameter matrix.
WaveFunctionComponent::EvaluationStamp PsiFormerWF::mw_evaluateVirtualDerivRatiosWeighted(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& virtual_batch,
    const OptVariables& optvars,
    const std::vector<ValueType>& total_weights,
    const std::vector<ParameterDerivativeView>& weighted_derivatives) const
{
  if (this != std::addressof(wfc_list.getLeader()))
    throw std::invalid_argument(
        "PsiFormer flattened weighted reduction must be invoked on the component-list leader");
  if (wfc_list.size() != virtual_batch.walkerCount() ||
      p_list.size() != virtual_batch.walkerCount() ||
      vp_scratch_list.size() != virtual_batch.walkerCount() ||
      weighted_derivatives.size() != virtual_batch.walkerCount())
    throw std::invalid_argument(
        "PsiFormer flattened weighted-reduction lists do not match the descriptor walker count");

  virtual_batch.validateOutputExtent(total_weights.size());
  virtual_batch.validateFor(p_list);

  std::size_t required_destination_extent = 0;
  for (std::size_t local_index = 0; local_index < optvars.size(); ++local_index)
  {
    const int global_index = optvars.where(local_index);
    if (global_index >= 0)
      required_destination_extent =
          std::max(required_destination_extent,
                   static_cast<std::size_t>(global_index) + 1);
  }

  const std::size_t destination_size =
      weighted_derivatives.empty() ? 0 : weighted_derivatives.front().size;
  if (!weighted_derivatives.empty() &&
      destination_size < required_destination_extent)
    throw std::invalid_argument(
        "PsiFormer flattened weighted derivative rows are too short");

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  const std::size_t electron_count =
      leader.model_state_->execution_plan.modelShape().electrons();

  // Validate all topology and caller storage before synchronizing clone caches
  // or lazily constructing the resource-owned score tape.
  for (std::size_t walker = 0; walker < virtual_batch.walkerCount(); ++walker)
  {
    if (weighted_derivatives[walker].size != destination_size ||
        (destination_size != 0 && weighted_derivatives[walker].data == nullptr))
      throw std::invalid_argument(
          "PsiFormer flattened weighted derivative rows have inconsistent shapes");
    if (typeid(wfc_list[walker]) != typeid(leader))
      throw std::invalid_argument(
          "PsiFormer flattened weighted crowd contains different component types");
    for (std::size_t other = 0; other < walker; ++other)
    {
      if (std::addressof(wfc_list[walker]) == std::addressof(wfc_list[other]))
        throw std::invalid_argument(
            "PsiFormer flattened weighted reduction requires distinct component clones");
      if (std::addressof(vp_scratch_list[walker]) ==
          std::addressof(vp_scratch_list[other]))
        throw std::invalid_argument(
            "PsiFormer flattened weighted reduction requires distinct scratch objects");
    }

    const ParticleSet* scratch_as_particles =
        static_cast<const ParticleSet*>(std::addressof(vp_scratch_list[walker]));
    for (std::size_t reference = 0; reference < virtual_batch.walkerCount();
         ++reference)
      if (scratch_as_particles == std::addressof(p_list[reference]))
        throw std::invalid_argument(
            "PsiFormer flattened weighted scratch objects must not alias reference walkers");
    if (vp_scratch_list[walker].isSpinor() != p_list[walker].isSpinor())
      throw std::invalid_argument(
          "PsiFormer flattened weighted reference and scratch spinor modes differ");
    if (p_list[walker].isSpinor())
      throw std::invalid_argument(
          "PsiFormer flattened weighted derivatives do not support spinor virtual moves");
    if (static_cast<std::size_t>(p_list[walker].getTotalNum()) != electron_count)
      throw std::invalid_argument(
          "PsiFormer flattened weighted walker has the wrong electron count");

    const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument(
          "PsiFormer flattened weighted crowd contains different models");
    if (component.optimization_metadata_.get() !=
        leader.optimization_metadata_.get())
      throw std::invalid_argument(
          "PsiFormer flattened weighted crowd contains different optimizer mappings");
  }
  for (ValueType weight : total_weights)
    if (!psiformer::determinant::isFiniteReal(std::real(weight)) ||
        !psiformer::determinant::isFiniteReal(std::imag(weight)))
      throw std::invalid_argument(
          "PsiFormer flattened weighted reduction received a non-finite weight");

  PsiFormerDerivativeReadTransaction transaction(
      *leader.model_state_, *leader.optimization_metadata_);
  const std::size_t parameter_version =
      transaction.modelTransaction().parameterVersion();
  const EvaluationStamp evaluation_stamp = EvaluationStamp::versioned(
      leader.model_state_.get(), static_cast<std::uint64_t>(parameter_version));

  // RefVectorWithLeader retains a leader even when its element list is empty.
  // Such a no-walker request still participates in the outer version protocol.
  if (virtual_batch.walkerCount() == 0)
    return evaluation_stamp;

  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  std::vector<std::size_t>& active_global_indices =
      resource.active_derivative_global_indices;
  active_global_indices.clear();
  active_global_indices.reserve(
      static_cast<std::size_t>(transaction.variables().size_of_active()));
  if (transaction.variables().size() != transaction.selectedFlatIndices().size())
    throw std::logic_error("PsiFormer flattened weighted optimizer mapping has the wrong size");
  const std::size_t parameter_count =
      transaction.modelTransaction().model().p.flat_values().size();
  for (std::size_t local_index = 0; local_index < transaction.variables().size();
       ++local_index)
  {
    const std::size_t flat_index = transaction.selectedFlatIndices()[local_index];
    if (flat_index >= parameter_count)
      throw std::out_of_range(
          "PsiFormer flattened weighted selected parameter is out of range");
    const int global_index = transaction.variables().where(local_index);
    if (global_index !=
        optvars.getIndex(transaction.variables().name(local_index)))
      throw std::invalid_argument(
          "PsiFormer flattened weighted optimizer mapping is stale");
    if (global_index < 0)
      continue;
    if (static_cast<std::size_t>(global_index) >= destination_size)
      throw std::out_of_range(
          "PsiFormer flattened weighted derivative destination is out of range");
    active_global_indices.push_back(static_cast<std::size_t>(global_index));
  }

  // A successful no-work call still observes and propagates a genuine shared
  // parameter publication so no clone retains a stale accepted-state cache.
  for (std::size_t walker = 0; walker < virtual_batch.walkerCount(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
        parameter_version);

  constexpr std::size_t no_reference = std::numeric_limits<std::size_t>::max();
  resource.active_virtual_walkers.clear();
  resource.virtual_reference_indices.assign(virtual_batch.walkerCount(),
                                             no_reference);
  for (const VirtualParticleBatch::Segment& segment : virtual_batch.segments())
  {
    const std::size_t walker = static_cast<std::size_t>(segment.walkerId());
    if (resource.virtual_reference_indices[walker] == no_reference)
    {
      resource.virtual_reference_indices[walker] =
          resource.active_virtual_walkers.size();
      resource.active_virtual_walkers.push_back(walker);
    }
  }

  const std::size_t reference_count = resource.active_virtual_walkers.size();
  const std::size_t active_count    = active_global_indices.size();
  if (active_count != 0 &&
      reference_count > std::numeric_limits<std::size_t>::max() / active_count)
    throw std::length_error(
        "PsiFormer flattened weighted derivative staging extent overflows");

  resource.virtual_reference_weights.assign(reference_count, ValueType(0));
  for (std::size_t segment_index = 0;
       segment_index < virtual_batch.segmentCount(); ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = virtual_batch.slice(segment_index);
    const std::size_t reference = resource.virtual_reference_indices[
        static_cast<std::size_t>(slice.walkerId())];
    for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
    {
      ValueType& sum = resource.virtual_reference_weights[reference];
      sum += total_weights[slice.flatOffset() + local_index];
      if (!psiformer::determinant::isFiniteReal(std::real(sum)) ||
          !psiformer::determinant::isFiniteReal(std::imag(sum)))
        throw std::runtime_error(
            "PsiFormer flattened weighted reference coefficient is non-finite");
    }
  }
  resource.flat_virtual_weighted_derivatives.assign(reference_count * active_count,
                                                     ValueType(0));

  // Empty, fixed, and inactive calls still certify the model version observed
  // by the value phase, but they do not construct a large score tape.
  if (!transaction.metadata().enabled || active_count == 0 || reference_count == 0)
  {
    resource.weighted_reference_configurations   = 0;
    resource.weighted_replacement_configurations = 0;
    resource.weighted_active_parameters           = active_count;
    return evaluation_stamp;
  }

  const DirectBackendMode score_mode =
      transaction.modelTransaction().state().direct_score_mode;
  pf::DirectScoreWorkspace* score_workspace =
      score_mode == DirectBackendMode::ORACLE
      ? nullptr
      : &resource.requireScoreWorkspace();

  // Gather one selected score and immediately fold it into its compact row;
  // direct result views alias the serialized workspace and cannot be retained.
  auto accumulate_score = [&](PsiFormerWF& component,
                              const ParticleSet& particles,
                              int replaced_particle,
                              const PosType* replacement_position,
                              ValueType scale,
                              std::size_t reference) {
    SelectedDerivativeDelta& contribution = resource.virtual_score_contribution;
    if (score_mode == DirectBackendMode::ORACLE)
    {
      const pf::Result oracle = component.evaluatePositionsUnderRead(
          transaction.modelTransaction(), particles, replaced_particle,
          replacement_position, EvaluationPurpose::SCORE_ONLY);
      component.gatherSelectedGradientUnderRead(
          transaction, oracle.param_gradient.data(), oracle.param_gradient.size(),
          scale, destination_size, contribution);
    }
    else
    {
      const pf::DirectScoreResult direct =
          component.evaluateDirectScorePositionsUnderRead(
              transaction.modelTransaction(), particles, replaced_particle,
              replacement_position, *score_workspace);

      if (score_mode == DirectBackendMode::COMPARE)
      {
        pf::Tensor positions({static_cast<std::size_t>(particles.getTotalNum()), 3});
        for (int electron = 0; electron < particles.getTotalNum(); ++electron)
        {
          const PosType& position = electron == replaced_particle
              ? (replacement_position ? *replacement_position
                                      : particles.activeR(electron))
              : particles.R[electron];
          for (int dimension = 0; dimension < 3; ++dimension)
            positions.x[3 * electron + dimension] = position[dimension];
        }
        pf::EvaluationRequest request;
        request.spatial_derivatives   = pf::SpatialDerivativeRequest::NONE;
        request.parameter_derivatives = pf::ParameterDerivativeRequest::LOG_ONLY;
        request.validation_hamiltonian = pf::ValidationHamiltonianRequest::NONE;
        const pf::Result oracle =
            transaction.modelTransaction().model().evaluate(positions, request);
        const double log_scale =
            std::max(std::abs(oracle.logabs), std::abs(direct.logabs));
        if (oracle.sign != direct.sign ||
            std::abs(oracle.logabs - direct.logabs) >
                2.0e-11 * (1.0 + log_scale) ||
            oracle.param_gradient.size() != direct.parameter_score.size)
          throw std::runtime_error(
              "PsiFormer flattened direct score differs from the native oracle");
        for (std::size_t parameter = 0;
             parameter < oracle.param_gradient.size(); ++parameter)
          if (std::abs(oracle.param_gradient[parameter] -
                       direct.parameter_score[parameter]) > 2.0e-8)
            throw std::runtime_error(
                "PsiFormer flattened direct parameter score differs from the native oracle");

        // Compare mode follows the established migration convention: validate
        // the direct engine, then publish the native oracle result.
        component.gatherSelectedGradientUnderRead(
            transaction, oracle.param_gradient.data(),
            oracle.param_gradient.size(), scale, destination_size,
            contribution);
      }
      else
        component.gatherSelectedGradientUnderRead(
            transaction, direct.parameter_score.data,
            direct.parameter_score.size, scale, destination_size,
            contribution);
    }

    if (contribution.size() != active_count)
      throw std::logic_error(
          "PsiFormer flattened weighted score mapping changed");
    const std::size_t row_offset = reference * active_count;
    for (std::size_t selected = 0; selected < active_count; ++selected)
    {
      if (contribution[selected].first !=
          resource.active_derivative_global_indices[selected])
        throw std::logic_error(
            "PsiFormer flattened weighted score mapping changed");
      ValueType& destination =
          resource.flat_virtual_weighted_derivatives[row_offset + selected];
      destination += contribution[selected].second;
      if (!psiformer::determinant::isFiniteReal(std::real(destination)) ||
          !psiformer::determinant::isFiniteReal(std::imag(destination)))
        throw std::runtime_error(
            "PsiFormer flattened weighted score reduction is non-finite");
    }
  };

  for (std::size_t reference = 0; reference < reference_count; ++reference)
  {
    const std::size_t walker = resource.active_virtual_walkers[reference];
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    accumulate_score(component, p_list[walker], -1, nullptr,
                     -resource.virtual_reference_weights[reference], reference);
  }

  for (std::size_t segment_index = 0;
       segment_index < virtual_batch.segmentCount(); ++segment_index)
  {
    const VirtualParticleBatch::Slice slice = virtual_batch.slice(segment_index);
    const std::size_t walker = static_cast<std::size_t>(slice.walkerId());
    const std::size_t reference = resource.virtual_reference_indices[walker];
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
    {
      const std::size_t flat_index = slice.flatOffset() + local_index;
      accumulate_score(component, p_list[walker], slice.electronId(),
                       std::addressof(slice.absolutePosition(local_index)),
                       total_weights[flat_index], reference);
    }
  }

  // Caller rows change only after every reference and replacement score has
  // succeeded, preserving the component-level whole-crowd transaction.
  for (std::size_t reference = 0; reference < reference_count; ++reference)
  {
    const std::size_t walker = resource.active_virtual_walkers[reference];
    const std::size_t row_offset = reference * active_count;
    for (std::size_t selected = 0; selected < active_count; ++selected)
      weighted_derivatives[walker][resource.active_derivative_global_indices[selected]] +=
          resource.flat_virtual_weighted_derivatives[row_offset + selected];
  }

  resource.weighted_reference_configurations   = reference_count;
  resource.weighted_replacement_configurations = virtual_batch.size();
  resource.weighted_active_parameters           = active_count;
  return evaluation_stamp;
}

// Validate and reduce one weighted virtual-move set, optionally in crowd-owned scratch.
void PsiFormerWF::evaluateDerivRatiosWeightedImpl(
    const PsiFormerDerivativeReadTransaction& transaction,
    const VirtualParticleSet& virtual_particles,
    const OptVariables& optvars,
    const std::vector<ValueType>& total_weights,
    std::size_t destination_size,
    pf::DirectScoreWorkspace* crowd_workspace,
    SelectedDerivativeDelta& output)
{
  const std::size_t virtual_count = virtual_particles.getTotalNum();
  if (total_weights.size() != virtual_count ||
      destination_size < static_cast<std::size_t>(optvars.size_of_active()))
    throw std::invalid_argument("PsiFormer weighted derivative-ratio outputs have the wrong shape");
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
  {
    output.clear();
    return;
  }
  if (virtual_particles.getRefPS().isSpinor())
    throw std::invalid_argument("PsiFormer weighted nonlocal derivatives do not support spinor virtual moves");

  const ParticleSet& reference = virtual_particles.getRefPS();
  const int electron           = virtual_particles.refPtcl;
  if (electron < 0 || electron >= reference.getTotalNum())
    throw std::out_of_range("PsiFormer weighted derivative-ratio reference electron is invalid");

  const ValueType reference_weight =
      std::accumulate(total_weights.begin(), total_weights.end(), ValueType(0));

  const DirectBackendMode requested_mode =
      transaction.modelTransaction().state().direct_score_mode;
  pf::DirectScoreWorkspace* score_workspace = requested_mode == DirectBackendMode::DIRECT
      ? (crowd_workspace ? crowd_workspace : &requireDirectScoreWorkspace())
      : nullptr;
  if (requested_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectScoreResult reference_result =
        evaluateDirectScorePositionsUnderRead(
            transaction.modelTransaction(), reference, -1, nullptr,
            *score_workspace);
    gatherSelectedGradientUnderRead(
        transaction, reference_result.parameter_score.data,
        reference_result.parameter_score.size, -reference_weight,
        destination_size, output);
  }
  else
  {
    const pf::Result reference_result = evaluatePositionsUnderRead(
        transaction.modelTransaction(), reference, -1, nullptr,
        EvaluationPurpose::SCORE_ONLY);
    gatherSelectedGradientUnderRead(
        transaction, reference_result.param_gradient.data(),
        reference_result.param_gradient.size(), -reference_weight,
        destination_size, output);
  }

  SelectedDerivativeDelta contribution;
  contribution.reserve(output.size());
  for (std::size_t move = 0; move < virtual_count; ++move)
  {
    if (requested_mode == DirectBackendMode::DIRECT)
    {
      const pf::DirectScoreResult virtual_result =
          evaluateDirectScorePositionsUnderRead(
              transaction.modelTransaction(), reference, electron,
              &virtual_particles.R[move], *score_workspace);
      gatherSelectedGradientUnderRead(
          transaction, virtual_result.parameter_score.data,
          virtual_result.parameter_score.size, total_weights[move],
          destination_size, contribution);
    }
    else
    {
      const pf::Result virtual_result = evaluatePositionsUnderRead(
          transaction.modelTransaction(), reference, electron,
          &virtual_particles.R[move], EvaluationPurpose::SCORE_ONLY);
      gatherSelectedGradientUnderRead(
          transaction, virtual_result.param_gradient.data(),
          virtual_result.param_gradient.size(), total_weights[move],
          destination_size, contribution);
    }

    if (contribution.size() != output.size())
      throw std::logic_error("PsiFormer weighted score mapping changed");
    for (std::size_t selected = 0; selected < output.size(); ++selected)
    {
      if (contribution[selected].first != output[selected].first)
        throw std::logic_error("PsiFormer weighted score mapping changed");
      output[selected].second += contribution[selected].second;
      if (!psiformer::determinant::isFiniteReal(std::real(output[selected].second)) ||
          !psiformer::determinant::isFiniteReal(std::imag(output[selected].second)))
        throw std::runtime_error("PsiFormer weighted score reduction is non-finite");
    }
  }
}

// Reuse one crowd score tape while preserving ragged walker ownership and output rows.
void PsiFormerWF::mw_evaluateDerivRatiosWeighted(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const OptVariables& optvars,
    const RefVector<const std::vector<ValueType>>& total_weights,
    const std::vector<ParameterDerivativeView>& weighted_derivatives) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != vp_list.size() || total_weights.size() != wfc_list.size() ||
      weighted_derivatives.size() != wfc_list.size())
    throw std::invalid_argument("PsiFormer batched weighted reductions have inconsistent walker counts");
  if (wfc_list.empty())
    return;

  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  const std::size_t parameter_version =
      transaction.modelTransaction().parameterVersion();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    if (weighted_derivatives[walker].size != 0 &&
        weighted_derivatives[walker].data == nullptr)
      throw std::invalid_argument("PsiFormer weighted derivative destination is null");
    if (total_weights[walker].get().size() !=
            static_cast<std::size_t>(vp_list[walker].getTotalNum()) ||
        weighted_derivatives[walker].size <
            static_cast<std::size_t>(optvars.size_of_active()))
      throw std::invalid_argument(
          "PsiFormer weighted derivative-ratio outputs have the wrong shape");
    if (vp_list[walker].getRefPS().isSpinor())
      throw std::invalid_argument(
          "PsiFormer weighted nonlocal derivatives do not support spinor virtual moves");
    if (vp_list[walker].refPtcl < 0 ||
        vp_list[walker].refPtcl >= vp_list[walker].getRefPS().getTotalNum())
      throw std::out_of_range(
          "PsiFormer weighted derivative-ratio reference electron is invalid");
    if (static_cast<std::size_t>(vp_list[walker].getRefPS().getTotalNum()) !=
        transaction.modelTransaction().model().ne)
      throw std::invalid_argument(
          "PsiFormer weighted derivative-ratio walker has the wrong electron count");
    for (std::size_t local_index = 0;
         local_index < transaction.variables().size(); ++local_index)
    {
      const int global_index = transaction.variables().where(local_index);
      if (global_index >= 0 && static_cast<std::size_t>(global_index) >=
              weighted_derivatives[walker].size)
        throw std::out_of_range(
            "PsiFormer weighted derivative destination index is out of range");
    }
  }

  pf::DirectScoreWorkspace* score_workspace =
      transaction.modelTransaction().state().direct_score_mode ==
          DirectBackendMode::DIRECT
      ? &resource.requireScoreWorkspace()
      : nullptr;
  SelectedDerivativeDelta scratch;
  // Each walker reduction is staged in one O(active) row and published only
  // after all of that walker's virtual positions succeed.
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.evaluateDerivRatiosWeightedImpl(
        transaction, vp_list[walker], optvars, total_weights[walker].get(),
        weighted_derivatives[walker].size, score_workspace, scratch);
    for (const auto& [global_index, value] : scratch)
      weighted_derivatives[walker][global_index] += value;
  }
}

// Fail before a spin-orbit ECP can silently discard its spin quadrature multipliers.
void PsiFormerWF::evaluateSpinorRatios(const VirtualParticleSet&,
                                       const std::pair<ValueVector, ValueVector>&,
                                       std::vector<ValueType>&)
{
  throw std::invalid_argument("PsiFormer does not implement spinor or spin-orbit ECP ratios");
}

// Fail before unsupported spin-orbit parameter derivatives reach the nonlocal operator.
void PsiFormerWF::evaluateSpinorDerivRatios(const VirtualParticleSet&,
                                            const std::pair<ValueVector, ValueVector>&,
                                            const OptVariables&,
                                            std::vector<ValueType>&,
                                            Matrix<ValueType>&)
{
  throw std::invalid_argument("PsiFormer does not implement spinor or spin-orbit ECP derivative ratios");
}

// Add only score derivatives, avoiding the mixed coordinate-jet reverse used for kinetic derivatives.
void PsiFormerWF::evaluateDerivativesWF(ParticleSet& p, const OptVariables&, Vector<ValueType>& dlogpsi)
{
  if (optimization_metadata_->enabled)
    requireUnplannedScalarDerivative("parameter-score evaluation");
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  synchronizeParameterVersion(transaction.modelTransaction().parameterVersion());
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
    return;

  SelectedDerivativeDelta delta;
  if (transaction.modelTransaction().state().direct_score_mode ==
      DirectBackendMode::DIRECT)
  {
    pf::DirectScoreWorkspace& workspace = requireDirectScoreWorkspace();
    const pf::DirectScoreResult result = evaluateDirectScorePositionsUnderRead(
        transaction.modelTransaction(), p, -1, nullptr, workspace);
    gatherSelectedGradientUnderRead(
        transaction, result.parameter_score.data, result.parameter_score.size,
        ValueType(1), static_cast<std::size_t>(dlogpsi.size()), delta);
  }
  else
  {
    const pf::Result result = evaluatePositionsUnderRead(
        transaction.modelTransaction(), p, -1, nullptr,
        EvaluationPurpose::SCORE_ONLY);
    gatherSelectedGradientUnderRead(
        transaction, result.param_gradient.data(), result.param_gradient.size(),
        ValueType(1), static_cast<std::size_t>(dlogpsi.size()), delta);
  }
  for (const auto& [global_index, value] : delta)
    dlogpsi[global_index] += value;
}

// Fill score rows using one serialized crowd tape in direct mode.
void PsiFormerWF::mw_evaluateParameterDerivativesWF(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables&,
    RecordArray<ValueType>& dlogpsi) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wfc_list.size())
    throw std::invalid_argument("PsiFormer batched score outputs have inconsistent shapes");
  if (wfc_list.empty())
    return;

  const int parameter_count = dlogpsi.getNumOfParams();
  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  const std::size_t parameter_version =
      transaction.modelTransaction().parameterVersion();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
        parameter_version);
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
    return;

  for (std::size_t local_index = 0;
       local_index < transaction.variables().size(); ++local_index)
  {
    const int global_index = transaction.variables().where(local_index);
    if (global_index >= parameter_count)
      throw std::out_of_range("PsiFormer score destination index is out of range");
  }
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    if (static_cast<std::size_t>(p_list[walker].getTotalNum()) !=
        transaction.modelTransaction().model().ne)
      throw std::invalid_argument("PsiFormer score walker has the wrong electron count");

  const DirectBackendMode score_mode =
      transaction.modelTransaction().state().direct_score_mode;
  pf::DirectScoreWorkspace* score_workspace = score_mode == DirectBackendMode::DIRECT
      ? &resource.requireScoreWorkspace()
      : nullptr;
  SelectedDerivativeDelta scratch;
  auto evaluate_row = [&](PsiFormerWF& component, std::size_t walker) {
    if (score_mode == DirectBackendMode::DIRECT)
    {
      const pf::DirectScoreResult result =
          component.evaluateDirectScorePositionsUnderRead(
              transaction.modelTransaction(), p_list[walker], -1, nullptr,
              *score_workspace);
      component.gatherSelectedGradientUnderRead(
          transaction, result.parameter_score.data, result.parameter_score.size,
          ValueType(1), static_cast<std::size_t>(parameter_count), scratch);
    }
    else
    {
      const pf::Result result = component.evaluatePositionsUnderRead(
          transaction.modelTransaction(), p_list[walker], -1, nullptr,
          EvaluationPurpose::SCORE_ONLY);
      component.gatherSelectedGradientUnderRead(
          transaction, result.param_gradient.data(), result.param_gradient.size(),
          ValueType(1), static_cast<std::size_t>(parameter_count), scratch);
    }
  };

  // Stage and publish one complete walker row at a time.
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    evaluate_row(component, walker);
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    for (const auto& [global_index, value] : scratch)
      score[global_index] += value;
  }
}

// Add score and component kinetic derivatives using the complete TrialWaveFunction gradient in P.G.
void PsiFormerWF::evaluateDerivatives(ParticleSet& p,
                                      const OptVariables& optvars,
                                      Vector<ValueType>& dlogpsi,
                                      Vector<ValueType>& dhpsioverpsi)
{
  if (optimization_metadata_->enabled)
    requireUnplannedScalarDerivative("kinetic-parameter evaluation");
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  synchronizeParameterVersion(transaction.modelTransaction().parameterVersion());
  if (optimization_metadata_->enabled && transaction.hasActiveParameters())
  {
    requireUnitElectronMasses(p);
    requireRealTotalWavefunctionDrift(p);
  }
  SelectedDerivativeDelta score_delta;
  SelectedDerivativeDelta kinetic_delta;
  evaluateDerivativesImpl(
      transaction, p, optvars, static_cast<std::size_t>(dlogpsi.size()),
      static_cast<std::size_t>(dhpsioverpsi.size()), nullptr, nullptr,
      score_delta, kinetic_delta);
  for (const auto& [global_index, value] : score_delta)
    dlogpsi[global_index] += value;
  for (const auto& [global_index, value] : kinetic_delta)
    dhpsioverpsi[global_index] += value;
}

// Evaluate one kinetic response using either lazy scalar scratch or an acquired crowd tape.
void PsiFormerWF::evaluateDerivativesImpl(
    const PsiFormerDerivativeReadTransaction& transaction,
    ParticleSet& p,
    const OptVariables&,
    std::size_t score_destination_size,
    std::size_t kinetic_destination_size,
    pf::DirectKineticWorkspace* crowd_workspace,
    std::vector<double>* crowd_total_log_gradient,
    SelectedDerivativeDelta& score_output,
    SelectedDerivativeDelta& kinetic_output)
{
  score_output.clear();
  kinetic_output.clear();
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
    return;
  if ((crowd_workspace == nullptr) != (crowd_total_log_gradient == nullptr))
    throw std::invalid_argument("PsiFormer kinetic scratch requires both tape and total-drift storage");

  const DirectBackendMode kinetic_mode =
      transaction.modelTransaction().state().direct_kinetic_mode;
  std::optional<pf::Result> oracle;
  if (kinetic_mode != DirectBackendMode::DIRECT)
  {
    oracle = evaluatePositionsUnderRead(
        transaction.modelTransaction(), p, -1, nullptr,
        EvaluationPurpose::SCORE_AND_KINETIC);
    gatherSelectedGradientUnderRead(
        transaction, oracle->param_gradient.data(), oracle->param_gradient.size(),
        ValueType(1), score_destination_size, score_output);
    gatherSelectedGradientUnderRead(
        transaction, oracle->local_energy_param_gradient.data(),
        oracle->local_energy_param_gradient.size(), ValueType(1),
        kinetic_destination_size, kinetic_output);
    if (kinetic_mode == DirectBackendMode::ORACLE)
    {
      return;
    }
  }

  const pf::PsiFormer& model = transaction.modelTransaction().model();
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  pf::DirectKineticWorkspace& kinetic_workspace =
      crowd_workspace ? *crowd_workspace : requireDirectKineticWorkspace();
  std::vector<double>& total_log_gradient =
      crowd_total_log_gradient ? *crowd_total_log_gradient
                               : requireDirectTotalLogGradient();
  if (total_log_gradient.size() != 3 * model.ne)
    throw std::logic_error("PsiFormer direct total-drift buffer has the wrong size");

  // ParticleSet::G is the complete TrialWaveFunction drift, not merely the
  // PsiFormer contribution. Repack it for every walker before tape reuse.
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      kinetic_workspace.setPosition(electron, dimension,
                                    std::real(p.R[electron][dimension]));
      total_log_gradient[3 * electron + dimension] =
          std::real(p.G[electron][dimension]);
    }

  const pf::DirectKineticResultView direct =
      transaction.modelTransaction().state().direct_kinetic_executor.evaluate(
          kinetic_workspace, total_log_gradient.data(),
          total_log_gradient.size());
  if (direct.parameter_version != transaction.modelTransaction().parameterVersion())
    throw std::logic_error("PsiFormer direct kinetic evaluation observed inconsistent parameters");

  if (oracle)
  {
    if (direct.parameter_score.size() != oracle->param_gradient.size() ||
        direct.kinetic_parameter_response.size() != oracle->local_energy_param_gradient.size())
      throw std::runtime_error("PsiFormer direct kinetic response has the wrong size");
    for (std::size_t parameter = 0; parameter < direct.parameter_score.size(); ++parameter)
    {
      if (std::abs(direct.parameter_score[parameter] - oracle->param_gradient[parameter]) > 3.0e-8)
        throw std::runtime_error("PsiFormer direct kinetic score differs from the native oracle");
      if (std::abs(direct.kinetic_parameter_response[parameter] -
                   oracle->local_energy_param_gradient[parameter]) > 3.0e-6)
        throw std::runtime_error("PsiFormer direct kinetic response differs from the native oracle");
    }
  }

  gatherSelectedGradientUnderRead(
      transaction, direct.parameter_score.data(), direct.parameter_score.size(),
      ValueType(1), score_destination_size, score_output);
  gatherSelectedGradientUnderRead(
      transaction, direct.kinetic_parameter_response.data(),
      direct.kinetic_parameter_response.size(), ValueType(1),
      kinetic_destination_size, kinetic_output);
}

// Fill score and kinetic rows serially through one resource-owned direct kinetic tape.
void PsiFormerWF::mw_evaluateParameterDerivatives(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dlogpsi,
    RecordArray<ValueType>& dhpsioverpsi) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wfc_list.size() ||
      dhpsioverpsi.getNumOfEntries() != wfc_list.size() ||
      dlogpsi.getNumOfParams() != dhpsioverpsi.getNumOfParams())
    throw std::invalid_argument("PsiFormer batched derivative outputs have inconsistent shapes");
  if (wfc_list.empty())
    return;

  PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  const std::size_t parameter_version =
      transaction.modelTransaction().parameterVersion();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(
        parameter_version);
  if (!optimization_metadata_->enabled || !transaction.hasActiveParameters())
    return;

  // Validate every active walker before allocating or mutating the shared
  // kinetic tape so a heterogeneous-mass crowd fails as one atomic request.
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    if (static_cast<std::size_t>(p_list[walker].getTotalNum()) !=
        transaction.modelTransaction().model().ne)
      throw std::invalid_argument(
          "PsiFormer kinetic derivative walker has the wrong electron count");
    requireUnitElectronMasses(p_list[walker]);
    requireRealTotalWavefunctionDrift(p_list[walker]);
  }

  const int parameter_count = dlogpsi.getNumOfParams();
  for (std::size_t local_index = 0;
       local_index < transaction.variables().size(); ++local_index)
  {
    const int global_index = transaction.variables().where(local_index);
    if (global_index >= parameter_count)
      throw std::out_of_range("PsiFormer kinetic derivative destination index is out of range");
  }
  const DirectBackendMode kinetic_mode =
      transaction.modelTransaction().state().direct_kinetic_mode;
  pf::DirectKineticWorkspace* kinetic_workspace = nullptr;
  std::vector<double>* total_log_gradient        = nullptr;
  if (kinetic_mode != DirectBackendMode::ORACLE)
  {
    kinetic_workspace  = &resource.requireKineticWorkspace();
    total_log_gradient = &resource.requireTotalLogGradient();
  }
  SelectedDerivativeDelta score_delta;
  SelectedDerivativeDelta kinetic_delta;
  // Stage and publish one complete score/kinetic walker row at a time.
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.evaluateDerivativesImpl(
        transaction, p_list[walker], optvars,
        static_cast<std::size_t>(parameter_count),
        static_cast<std::size_t>(parameter_count), kinetic_workspace,
        total_log_gradient, score_delta, kinetic_delta);
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    Vector<ValueType> kinetic_response(dhpsioverpsi[walker], parameter_count);
    for (const auto& [global_index, value] : score_delta)
      score[global_index] += value;
    for (const auto& [global_index, value] : kinetic_delta)
      kinetic_response[global_index] += value;
  }
}

// Copy optimizer mapping and accepted state while sharing the synchronized native model.
std::unique_ptr<WaveFunctionComponent> PsiFormerWF::makeClone(ParticleSet& particles) const
{
  auto clone = std::make_unique<PsiFormerWF>(*this);
  clone->bound_particle_set_ = &particles;
  return clone;
}

} // namespace qmcplusplus
