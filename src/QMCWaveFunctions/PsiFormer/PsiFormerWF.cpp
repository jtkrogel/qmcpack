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
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#include "Message/Communicate.h"
#include "Particle/VirtualParticleSet.h"
#include "ResourceCollection.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

/// Initial value for deterministic FNV-1a identities stored in walker buffers.
constexpr std::uint64_t PERSISTENT_FINGERPRINT_OFFSET = UINT64_C(14695981039346656037);

/// Prime used by deterministic FNV-1a identities stored in walker buffers.
constexpr std::uint64_t PERSISTENT_FINGERPRINT_PRIME = UINT64_C(1099511628211);

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

/** Crowd-owned mutable storage.  ResourceCollection cloning recreates scratch
 * against the same immutable/versioned model state without copying buffers. */
struct PsiFormerWF::PsiFormerMultiWalkerResource : public Resource
{
  /// Create empty crowd scratch bound to one shared immutable model state.
  explicit PsiFormerMultiWalkerResource(std::shared_ptr<PsiFormerSharedState> model_state)
      : Resource("PsiFormerMultiWalkerResource"),
        model_state(std::move(model_state)),
        batch_workspace(this->model_state->direct_batch_executor.makeWorkspace())
  {}

  /// Clone only model identity; ResourceCollection copies never duplicate live scratch.
  PsiFormerMultiWalkerResource(const PsiFormerMultiWalkerResource& other)
      : PsiFormerMultiWalkerResource(other.model_state)
  {}

  /// Recreate an independent resource for a copied crowd resource collection.
  std::unique_ptr<Resource> makeClone() const override
  { return std::make_unique<PsiFormerMultiWalkerResource>(*this); }

  std::shared_ptr<PsiFormerSharedState> model_state;
  std::unique_ptr<pf::DirectBatchWorkspace> batch_workspace;
  /// Lazily allocated score tape serialized across component-major crowd calls.
  std::unique_ptr<pf::DirectScoreWorkspace> score_workspace;
  /// Lazily allocated kinetic tape serialized across component-major crowd calls.
  std::unique_ptr<pf::DirectKineticWorkspace> kinetic_workspace;
  /// Complete per-walker drift packed immediately before a kinetic reverse pass.
  std::vector<double> total_log_gradient;
  /// Per-configuration active-electron indices consumed by the native batch API.
  std::vector<std::size_t> active_electrons;
  /// Prefix offsets mapping flattened ragged virtual configurations back to walkers.
  std::vector<std::size_t> virtual_offsets;
  /// Selected walker indices for masked recomputation.
  std::vector<std::size_t> walker_indices;

  /// Create the single reusable score tape only when an optimizer path requests it.
  pf::DirectScoreWorkspace& requireScoreWorkspace()
  {
    if (!score_workspace)
      score_workspace = model_state->direct_score_executor.makeWorkspace();
    return *score_workspace;
  }

  /// Create the single reusable kinetic tape only when an optimizer path requests it.
  pf::DirectKineticWorkspace& requireKineticWorkspace()
  {
    if (!kinetic_workspace)
      kinetic_workspace = model_state->direct_kinetic_executor.makeWorkspace();
    return *kinetic_workspace;
  }

  /// Return fixed-size crowd drift storage paired with the reusable kinetic tape.
  std::vector<double>& requireTotalLogGradient()
  {
    const std::size_t required_size =
        3 * model_state->execution_plan.modelShape().electrons();
    if (total_log_gradient.empty())
      total_log_gradient.resize(required_size);
    if (total_log_gradient.size() != required_size)
      throw std::logic_error("PsiFormer crowd total-drift buffer has the wrong size");
    return total_log_gradient;
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

/// Convert the real sign/log-magnitude result to QMCPACK's complex-log convention.
WaveFunctionComponent::LogValue makeLogValue(double sign, double logabs)
{ return WaveFunctionComponent::LogValue(logabs, sign < 0 ? M_PI : 0.0); }

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
{}

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
      system_kind_(other.system_kind_),
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

  proposed_sign_      = 1.0;
  proposed_log_value_ = LogValue(0);
  proposed_configuration_identity_ = 0;
  proposed_particle_               = -1;
  has_proposal_       = false;
}

PsiFormerWF::~PsiFormerWF() = default;

// Report the immutable optimization mode shared by the complete clone family.
bool PsiFormerWF::isOptimizable() const
{
  return optimization_metadata_->enabled;
}

// Add one cloneable workspace.  A copied ResourceCollection reconstructs empty
// batch scratch while retaining the same shared model/plan identity.
void PsiFormerWF::createResource(ResourceCollection& collection) const
{ collection.addResource(std::make_unique<PsiFormerMultiWalkerResource>(model_state_)); }

// Lend the crowd workspace to the leader after validating homogeneous clone identity.
void PsiFormerWF::acquireResource(
    ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  if (this != &leader)
    throw std::logic_error("PsiFormer multiwalker resource acquisition must be invoked on the leader");
  if (leader.mw_resource_handle_)
    throw std::logic_error("PsiFormer multiwalker resource is already acquired");
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    if (wfc_list.getCastedElement<PsiFormerWF>(walker).model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer multiwalker list contains components from different models");

  leader.mw_resource_handle_ = collection.lendResource<PsiFormerMultiWalkerResource>();
  if (leader.mw_resource_handle_.getResource().model_state.get() != leader.model_state_.get())
    throw std::logic_error("PsiFormer ResourceCollection belongs to a different model");
}

// Return the exact handle previously lent to this crowd leader.
void PsiFormerWF::releaseResource(
    ResourceCollection& collection,
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  if (this != &leader || !leader.mw_resource_handle_)
    throw std::logic_error("PsiFormer multiwalker resource release has no acquired leader handle");
  auto& resource = leader.mw_resource_handle_.getResource();
  resource.active_electrons.clear();
  resource.virtual_offsets.clear();
  resource.walker_indices.clear();
  collection.takebackResource(leader.mw_resource_handle_);
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
  if (resource.model_state.get() != leader.model_state_.get())
    throw std::logic_error("PsiFormer acquired workspace belongs to a different model");
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    if (wfc_list.getCastedElement<PsiFormerWF>(walker).model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer multiwalker list contains components from different models");
  return resource;
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

// Account only explicitly owned numeric buffers; immutable shared model state and
// persistent accepted-state vectors intentionally remain outside this diagnostic.
testing::PsiFormerWorkspaceDiagnostics PsiFormerWF::directWorkspaceDiagnosticsForTesting() const
{
  testing::PsiFormerWorkspaceDiagnostics diagnostics;
  diagnostics.owns_value_workspace          = static_cast<bool>(direct_value_workspace_);
  diagnostics.owns_full_spatial_workspace   = static_cast<bool>(direct_full_spatial_workspace_);
  diagnostics.owns_active_spatial_workspace = static_cast<bool>(direct_active_spatial_workspace_);
  diagnostics.owns_batch_workspace          = static_cast<bool>(direct_batch_workspace_);
  diagnostics.owns_score_workspace          = static_cast<bool>(direct_score_workspace_);
  diagnostics.owns_kinetic_workspace        = static_cast<bool>(direct_kinetic_workspace_);

  if (direct_value_workspace_)
    diagnostics.value_bytes = direct_value_workspace_->vectorStorageBytes();
  if (direct_full_spatial_workspace_)
    diagnostics.full_spatial_bytes = direct_full_spatial_workspace_->vectorStorageBytes();
  if (direct_active_spatial_workspace_)
    diagnostics.active_spatial_bytes = direct_active_spatial_workspace_->vectorStorageBytes();
  if (direct_batch_workspace_)
    diagnostics.batch_bytes = direct_batch_workspace_->vectorStorageBytes();
  if (direct_score_workspace_)
    diagnostics.score_bytes = direct_score_workspace_->vectorStorageBytes();
  if (direct_kinetic_workspace_)
    diagnostics.kinetic_bytes = direct_kinetic_workspace_->vectorStorageBytes();
  diagnostics.total_log_gradient_bytes = direct_total_log_gradient_.capacity() * sizeof(double);
  return diagnostics;
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
  proposed_sign_                    = 1.0;
  log_value_                        = LogValue(0);
  proposed_log_value_               = LogValue(0);
  has_proposal_                     = false;
  accepted_value_valid_             = false;
  accepted_configuration_identity_ = 0;
  proposed_configuration_identity_ = 0;
  accepted_parameter_version_      = parameter_version;
  accepted_state_requirement_      = AcceptedStateRequirement::INVALID;
  proposed_particle_               = -1;
  accepted_gradient_               = ValueType(0);
  accepted_laplacian_              = ValueType(0);
  observed_parameter_version_      = parameter_version;
}

// Lazily invalidate clone-local caches after another clone updates the model.
void PsiFormerWF::synchronizeParameterVersion(std::size_t parameter_version)
{
  if (observed_parameter_version_ != parameter_version)
    invalidateParameterCaches(parameter_version);
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
  proposed_sign_                   = 1.0;
  proposed_log_value_              = LogValue(0);
  proposed_configuration_identity_ = 0;
  proposed_particle_               = -1;
  has_proposal_                     = false;
  observed_parameter_version_       = parameter_version;
}

// Apply a validated selected-parameter or complete-vector update at an exclusive model barrier.
void PsiFormerWF::resetParametersExclusive(const OptVariables& active)
{
  if (!optimization_metadata_->enabled)
    return;

  std::unique_lock state_lock(model_state_->mutex);
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

  std::size_t parameter_version;
  {
    std::unique_lock state_lock(model_state_->mutex);
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
    parameter_version = model.p.version();

    for (std::size_t local_index = 0;
         local_index < optimization_metadata_->selected_flat_indices.size(); ++local_index)
      optimization_metadata_->variables[local_index] =
          model.p.flat_values()[optimization_metadata_->selected_flat_indices[local_index]];
  }

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

  system_kind_ = system_kind;
  app_log() << "  PsiFormer " << WaveFunctionComponent::getName() << ": validated " << system_kind_
            << " system metadata against electron and ion particle sets" << std::endl;
}

// Translate QMCPACK particle coordinates and the public-call purpose into a native request.
pf::Result PsiFormerWF::evaluate(const ParticleSet& p,
                                 int replaced_particle,
                                 EvaluationPurpose purpose,
                                 int active_gradient_particle)
{
  return evaluatePositions(p, replaced_particle, nullptr, purpose, active_gradient_particle);
}

// Translate a full or one-electron-replaced configuration into a native request.
pf::Result PsiFormerWF::evaluatePositions(const ParticleSet& p,
                                          int replaced_particle,
                                          const PosType* replacement_position,
                                          EvaluationPurpose purpose,
                                          int active_gradient_particle)
{
  std::shared_lock state_lock(model_state_->mutex);
  pf::PsiFormer& model = model_state_->model;
  synchronizeParameterVersion(model.p.version());

  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  // Value-only calls use clone-local fixed storage. Compare mode evaluates
  // both implementations and is intended for migration/debug validation.
  std::optional<pf::DirectValueResult> direct_result;
  if (purpose == EvaluationPurpose::VALUE_ONLY && model_state_->direct_value_mode != DirectBackendMode::ORACLE)
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
    direct_result = model_state_->direct_value_executor.evaluate(workspace);
    if (model_state_->direct_value_mode == DirectBackendMode::DIRECT)
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
      model_state_->direct_score_mode != DirectBackendMode::ORACLE)
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
    direct_score_result = model_state_->direct_score_executor.evaluate(score_workspace);
    if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
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
      model_state_->direct_spatial_mode != DirectBackendMode::ORACLE)
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
      direct_spatial_result = model_state_->direct_spatial_executor.evaluateFull(workspace);
    else
      direct_spatial_result = model_state_->direct_spatial_executor.evaluateActive(
          workspace, static_cast<std::size_t>(active_gradient_particle));

    if (model_state_->direct_spatial_mode == DirectBackendMode::DIRECT)
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

// Run one value-only request in clone-local fixed storage.  Returning the small
// scalar record does not materialize any of the owning arrays in pf::Result.
pf::DirectValueResult PsiFormerWF::evaluateDirectValuePositions(
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position)
{
  std::shared_lock state_lock(model_state_->mutex);
  pf::PsiFormer& model = model_state_->model;
  synchronizeParameterVersion(model.p.version());
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
  return model_state_->direct_value_executor.evaluate(workspace);
}

// Run one spatial request in clone-local fixed storage.  The returned views
// remain valid until the same component's corresponding workspace is reused.
pf::DirectSpatialResultView PsiFormerWF::evaluateDirectSpatialPositions(
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position,
    EvaluationPurpose purpose,
    int active_gradient_particle)
{
  if (purpose != EvaluationPurpose::FULL_SPATIAL &&
      purpose != EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT)
    throw std::logic_error("PsiFormer direct spatial adapter received a non-spatial request");

  std::shared_lock state_lock(model_state_->mutex);
  pf::PsiFormer& model = model_state_->model;
  synchronizeParameterVersion(model.p.version());
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

  if (purpose == EvaluationPurpose::FULL_SPATIAL)
    return model_state_->direct_spatial_executor.evaluateFull(workspace);
  return model_state_->direct_spatial_executor.evaluateActive(
      workspace, static_cast<std::size_t>(active_gradient_particle));
}

// Delay scalar forward-buffer construction until a value path actually runs.
pf::DirectValueWorkspace& PsiFormerWF::requireDirectValueWorkspace()
{
  if (!direct_value_workspace_)
    direct_value_workspace_ = model_state_->direct_value_executor.makeWorkspace();
  return *direct_value_workspace_;
}

// Keep the much larger full-VGL tape independent from the compact active-gradient tape.
pf::DirectSpatialWorkspace& PsiFormerWF::requireDirectSpatialWorkspace(EvaluationPurpose purpose)
{
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
  if (!direct_batch_workspace_)
    direct_batch_workspace_ = model_state_->direct_batch_executor.makeWorkspace();
  return *direct_batch_workspace_;
}

// Lazily allocate score scratch for scalar calls, keeping inference-only clones lightweight.
pf::DirectScoreWorkspace& PsiFormerWF::requireDirectScoreWorkspace()
{
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer direct score workspace requires an optimizable component");
  if (!direct_score_workspace_)
    direct_score_workspace_ = model_state_->direct_score_executor.makeWorkspace();
  return *direct_score_workspace_;
}

// Lazily allocate the much larger kinetic tape only for an actual scalar reverse call.
pf::DirectKineticWorkspace& PsiFormerWF::requireDirectKineticWorkspace()
{
  if (!optimization_metadata_->enabled)
    throw std::logic_error("PsiFormer direct kinetic workspace requires an optimizable component");
  if (!direct_kinetic_workspace_)
    direct_kinetic_workspace_ = model_state_->direct_kinetic_executor.makeWorkspace();
  return *direct_kinetic_workspace_;
}

// Lazily allocate the scalar adapter's complete TrialWaveFunction drift buffer.
std::vector<double>& PsiFormerWF::requireDirectTotalLogGradient()
{
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

// Run the graph-free score pass and expose its clone-local non-owning output.
pf::DirectScoreResult PsiFormerWF::evaluateDirectScorePositions(
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position)
{
  return evaluateDirectScorePositions(
      p, replaced_particle, replacement_position, requireDirectScoreWorkspace());
}

// Run a graph-free score pass in explicitly supplied clone- or crowd-owned scratch.
pf::DirectScoreResult PsiFormerWF::evaluateDirectScorePositions(
    const ParticleSet& p,
    int replaced_particle,
    const PosType* replacement_position,
    pf::DirectScoreWorkspace& score_workspace)
{
  std::shared_lock state_lock(model_state_->mutex);
  pf::PsiFormer& model = model_state_->model;
  synchronizeParameterVersion(model.p.version());
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
  return model_state_->direct_score_executor.evaluate(score_workspace);
}

// Evaluate a full accepted configuration and accumulate its spatial derivatives.
PsiFormerWF::LogValue PsiFormerWF::evaluateLog(const ParticleSet& p,
                                               ParticleSet::ParticleGradient& g,
                                               ParticleSet::ParticleLaplacian& l)
{
  // Keep the scatter independent of result ownership.  Production direct mode
  // passes workspace-backed views; oracle and compare retain pf::Result.
  auto scatter = [&](double sign, double logabs, const auto& gradient, const auto& lap_log) {
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
    accepted_parameter_version_      = observed_parameter_version_;
    accepted_state_requirement_      = AcceptedStateRequirement::FULL_SPATIAL;
    accepted_value_valid_            = true;
    proposed_sign_                   = 1.0;
    proposed_log_value_              = LogValue(0);
    proposed_configuration_identity_ = 0;
    proposed_particle_               = -1;
    has_proposal_                     = false;
    accumulateAcceptedSpatial(g, l);
    return log_value_;
  };

  if (model_state_->direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositions(
        p, -1, nullptr, EvaluationPurpose::FULL_SPATIAL, -1);
    return scatter(result.sign, result.logabs, result.gradient, result.lap_log);
  }

  const pf::Result result = evaluate(p, -1, EvaluationPurpose::FULL_SPATIAL);
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
  if (wfc_list.size() != p_list.size() || wfc_list.size() != gradient_list.size() ||
      wfc_list.size() != laplacian_list.size())
    throw std::invalid_argument("PsiFormer mw_evaluateLog list sizes do not match");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  // Preserve the developer oracle/compare switches.  Production direct mode never
  // enters the serialized component fallback.
  if (leader.model_state_->direct_spatial_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_evaluateLog(wfc_list, p_list, gradient_list, laplacian_list);
    return;
  }

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::FULL_VGL, wfc_list.size());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    packBatchConfiguration(batch, walker, p_list[walker]);
  }

  const pf::DirectBatchSpatialResultView result =
      leader.model_state_->direct_batch_executor.evaluateFull(batch);
  const std::size_t electrons = batch.electronCount();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    if (result.parameter_version[walker] != parameter_version)
      throw std::logic_error("PsiFormer full-spatial batch observed inconsistent parameters");
    if (gradient_list[walker].get().size() < electrons ||
        laplacian_list[walker].get().size() < electrons)
      throw std::invalid_argument("PsiFormer mw_evaluateLog output arrays are too small");

    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.resizeAcceptedSpatialStorage(electrons);
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
    component.proposed_sign_                   = 1.0;
    component.proposed_log_value_              = LogValue(0);
    component.proposed_configuration_identity_ = 0;
    component.proposed_particle_               = -1;
    component.has_proposal_                     = false;
    component.accumulateAcceptedSpatial(gradient, laplacian);
  }
}

void PsiFormerWF::mw_evaluateGL(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVector<ParticleSet::ParticleGradient>& gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list,
    bool) const
{ mw_evaluateLog(wfc_list, p_list, gradient_list, laplacian_list); }

void PsiFormerWF::mw_recompute(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const std::vector<bool>& recompute_mask) const
{
  if (wfc_list.size() != p_list.size() || wfc_list.size() != recompute_mask.size())
    throw std::invalid_argument("PsiFormer mw_recompute list sizes do not match");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  if (leader.model_state_->direct_value_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_recompute(wfc_list, p_list, recompute_mask);
    return;
  }

  resource.walker_indices.clear();
  for (std::size_t walker = 0; walker < recompute_mask.size(); ++walker)
    if (recompute_mask[walker])
      resource.walker_indices.push_back(walker);
  if (resource.walker_indices.empty())
    return;

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, resource.walker_indices.size());
  for (std::size_t selected = 0; selected < resource.walker_indices.size(); ++selected)
  {
    const std::size_t walker = resource.walker_indices[selected];
    wfc_list.getCastedElement<PsiFormerWF>(walker).synchronizeParameterVersion(parameter_version);
    packBatchConfiguration(batch, selected, p_list[walker]);
  }

  const pf::DirectBatchValueResultView result =
      leader.model_state_->direct_batch_executor.evaluateValues(batch);
  for (std::size_t selected = 0; selected < resource.walker_indices.size(); ++selected)
  {
    if (result.parameter_version[selected] != parameter_version)
      throw std::logic_error("PsiFormer recompute batch observed inconsistent parameters");
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(resource.walker_indices[selected]);
    const auto& particles = p_list[resource.walker_indices[selected]];
    const std::uint64_t configuration = configurationIdentity(particles);
    const bool preserve_spatial = component.acceptedStateMatches(
        particles, parameter_version, AcceptedStateRequirement::FULL_SPATIAL);
    component.current_sign_ = result.sign[selected];
    component.log_value_ = makeLogValue(result.sign[selected], result.logabs[selected]);
    component.accepted_configuration_identity_ = configuration;
    component.accepted_parameter_version_      = parameter_version;
    component.accepted_state_requirement_ = preserve_spatial
        ? AcceptedStateRequirement::FULL_SPATIAL
        : AcceptedStateRequirement::VALUE_ONLY;
    component.accepted_value_valid_ = true;
    component.proposed_sign_                   = 1.0;
    component.proposed_log_value_              = LogValue(0);
    component.proposed_configuration_identity_ = 0;
    component.proposed_particle_               = -1;
    component.has_proposal_                     = false;
  }
}

// Evaluate and cache the wavefunction ratio for one proposed electron position.
PsiFormerWF::PsiValue PsiFormerWF::ratio(ParticleSet& p, int iat)
{
  auto cache_and_form_ratio = [&](double sign, double logabs) {
    if (!acceptedStateMatches(p, observed_parameter_version_, AcceptedStateRequirement::VALUE_ONLY))
    {
      invalidateParameterCaches(observed_parameter_version_);
      throw std::logic_error("PsiFormer ratio requested before evaluateLog for the current parameter version");
    }

    // Cache proposal state so acceptMove can commit it without reevaluating the
    // network.
    proposed_sign_      = sign;
    proposed_log_value_ = LogValue(logabs, sign < 0 ? M_PI : 0.0);
    proposed_configuration_identity_ = configurationIdentity(p, iat);
    proposed_particle_               = iat;
    has_proposal_                    = true;
    return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
  };

  if (model_state_->direct_value_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectValueResult result = evaluateDirectValuePositions(p, iat, nullptr);
    return cache_and_form_ratio(result.sign, result.logabs);
  }

  const pf::Result result = evaluate(p, iat, EvaluationPurpose::VALUE_ONLY);
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
  if (wfc_list.size() != p_list.size())
    throw std::invalid_argument("PsiFormer mw_calcRatio list sizes do not match");
  ratios.resize(wfc_list.size());
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  if (leader.model_state_->direct_value_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_calcRatio(wfc_list, p_list, particle_index, ratios);
    return;
  }

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, wfc_list.size());
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
    packBatchConfiguration(batch, walker, p_list[walker], particle_index);
  }

  const pf::DirectBatchValueResultView result =
      leader.model_state_->direct_batch_executor.evaluateValues(batch);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    if (result.parameter_version[walker] != parameter_version)
      throw std::logic_error("PsiFormer ratio batch observed inconsistent parameters");
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.proposed_sign_ = result.sign[walker];
    component.proposed_log_value_ = makeLogValue(result.sign[walker], result.logabs[walker]);
    component.proposed_configuration_identity_ =
        configurationIdentity(p_list[walker], particle_index);
    component.proposed_particle_ = particle_index;
    component.has_proposal_ = true;
    ratios[walker] = makeRatio(result.sign[walker], result.logabs[walker],
                               component.current_sign_, std::real(component.log_value_));
  }
}

// Return one accepted electron logarithmic gradient.
PsiFormerWF::GradType PsiFormerWF::evalGrad(ParticleSet& p, int iat)
{
  auto scatter = [](const auto& source) {
    GradType gradient;
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] = source[dimension];
    return gradient;
  };

  if (model_state_->direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositions(
        p, -1, nullptr, EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
    return scatter(result.gradient);
  }

  const pf::Result result = evaluate(p, -1, EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
  return scatter(result.active_gradient);
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

void PsiFormerWF::mw_evalGrad(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<GradType>& gradients) const
{
  if (wfc_list.size() != p_list.size() || wfc_list.size() != gradients.size())
    throw std::invalid_argument("PsiFormer mw_evalGrad list sizes do not match");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  if (leader.model_state_->direct_spatial_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_evalGrad(wfc_list, p_list, particle_index, gradients);
    return;
  }

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, wfc_list.size());
  resource.active_electrons.assign(wfc_list.size(), static_cast<std::size_t>(particle_index));
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    packBatchConfiguration(batch, walker, p_list[walker]);
  }

  const pf::DirectBatchSpatialResultView result =
      leader.model_state_->direct_batch_executor.evaluateActive(batch, resource.active_electrons.data());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    if (result.parameter_version[walker] != parameter_version)
      throw std::logic_error("PsiFormer active-gradient batch observed inconsistent parameters");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      gradients[walker][dimension] = result.gradient[walker * result.gradient_stride + dimension];
  }
}

// Evaluate a proposed ratio and gradient in one native-model traversal.
PsiFormerWF::PsiValue PsiFormerWF::ratioGrad(ParticleSet& p, int iat, GradType& gradient)
{
  auto scatter = [&](double sign, double logabs, const auto& active_gradient) {
    if (!acceptedStateMatches(p, observed_parameter_version_, AcceptedStateRequirement::VALUE_ONLY))
    {
      invalidateParameterCaches(observed_parameter_version_);
      throw std::logic_error("PsiFormer ratioGrad requested before evaluateLog for the current parameter version");
    }

    // Evaluate the proposal once and return both its ratio and active-electron
    // gradient.
    proposed_sign_      = sign;
    proposed_log_value_ = LogValue(logabs, sign < 0 ? M_PI : 0.0);
    proposed_configuration_identity_ = configurationIdentity(p, iat);
    proposed_particle_               = iat;
    has_proposal_                    = true;
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] += active_gradient[dimension];
    return (proposed_sign_ / current_sign_) * std::exp(std::real(proposed_log_value_ - log_value_));
  };

  if (model_state_->direct_spatial_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectSpatialResultView result = evaluateDirectSpatialPositions(
        p, iat, nullptr, EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
    return scatter(result.sign, result.logabs, result.gradient);
  }

  const pf::Result result = evaluate(p, iat, EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT, iat);
  return scatter(result.sign, result.logabs, result.active_gradient);
}

void PsiFormerWF::mw_ratioGrad(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<PsiValue>& ratios,
    std::vector<GradType>& gradients) const
{
  if (wfc_list.size() != p_list.size() || wfc_list.size() != gradients.size())
    throw std::invalid_argument("PsiFormer mw_ratioGrad list sizes do not match");
  ratios.resize(wfc_list.size());
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  auto& resource     = requireMultiWalkerResource(wfc_list);
  if (leader.model_state_->direct_spatial_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_ratioGrad(wfc_list, p_list, particle_index, ratios, gradients);
    return;
  }

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, wfc_list.size());
  resource.active_electrons.assign(wfc_list.size(), static_cast<std::size_t>(particle_index));
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
    packBatchConfiguration(batch, walker, p_list[walker], particle_index);
  }

  const pf::DirectBatchSpatialResultView result =
      leader.model_state_->direct_batch_executor.evaluateActive(batch, resource.active_electrons.data());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    if (result.parameter_version[walker] != parameter_version)
      throw std::logic_error("PsiFormer ratio-gradient batch observed inconsistent parameters");
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.proposed_sign_ = result.sign[walker];
    component.proposed_log_value_ = makeLogValue(result.sign[walker], result.logabs[walker]);
    component.proposed_configuration_identity_ =
        configurationIdentity(p_list[walker], particle_index);
    component.proposed_particle_ = particle_index;
    component.has_proposal_ = true;
    ratios[walker] = makeRatio(result.sign[walker], result.logabs[walker],
                               component.current_sign_, std::real(component.log_value_));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      gradients[walker][dimension] +=
          result.gradient[walker * result.gradient_stride + dimension];
  }
}

// Promote cached proposal state to accepted state after a successful move.
void PsiFormerWF::acceptMove(ParticleSet& particles, int particle_index, bool)
{
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  if (has_proposal_)
  {
    if (particle_index != proposed_particle_ ||
        proposed_configuration_identity_ != configurationIdentity(particles, particle_index))
    {
      proposed_sign_                   = 1.0;
      proposed_log_value_              = LogValue(0);
      proposed_configuration_identity_ = 0;
      proposed_particle_               = -1;
      has_proposal_                     = false;
      throw std::logic_error("PsiFormer accepted move does not match the cached proposal");
    }
    log_value_            = proposed_log_value_;
    current_sign_         = proposed_sign_;
    accepted_value_valid_ = true;
    accepted_configuration_identity_ = proposed_configuration_identity_;
    accepted_parameter_version_      = observed_parameter_version_;
    accepted_state_requirement_      = AcceptedStateRequirement::VALUE_ONLY;
  }
  proposed_sign_                   = 1.0;
  proposed_log_value_              = LogValue(0);
  proposed_configuration_identity_ = 0;
  proposed_particle_               = -1;
  has_proposal_                     = false;
}

// Forget cached proposal state after a rejected move.
void PsiFormerWF::restore(int particle_index)
{
  std::shared_lock state_lock(model_state_->mutex);
  synchronizeParameterVersion(model_state_->model.p.version());
  if (has_proposal_ && particle_index != proposed_particle_)
    throw std::logic_error("PsiFormer restored move does not match the cached proposal");
  proposed_sign_                   = 1.0;
  proposed_log_value_              = LogValue(0);
  proposed_configuration_identity_ = 0;
  proposed_particle_               = -1;
  has_proposal_                     = false;
}

void PsiFormerWF::mw_accept_rejectMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    const std::vector<bool>& is_accepted,
    bool) const
{
  if (wfc_list.size() != p_list.size() || wfc_list.size() != is_accepted.size())
    throw std::invalid_argument("PsiFormer mw_accept_rejectMove list sizes do not match");
  if (wfc_list.empty())
    return;

  const auto& leader = wfc_list.getCastedLeader<PsiFormerWF>();
  if (this != &leader)
    throw std::logic_error("PsiFormer mw_accept_rejectMove must be invoked on the crowd leader");
  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    if (component.model_state_.get() != leader.model_state_.get())
      throw std::invalid_argument("PsiFormer accept/reject list contains components from different models");
    component.synchronizeParameterVersion(parameter_version);
    if (component.has_proposal_ && component.proposed_particle_ != particle_index)
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
    component.proposed_sign_                   = 1.0;
    component.proposed_log_value_              = LogValue(0);
    component.proposed_configuration_identity_ = 0;
    component.proposed_particle_               = -1;
    component.has_proposal_                     = false;
  }
}

// Reserve fixed bulk and scalar slots without assuming that registration follows evaluation.
void PsiFormerWF::registerData(ParticleSet& particles, WFBufferType& buffer)
{
  static_assert(std::numeric_limits<FullPrecRealType>::digits >= 32,
                "PsiFormer walker metadata requires exact 32-bit scalar limbs");
  if (particles.getTotalNum() != static_cast<int>(model_state_->model.ne))
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
  std::size_t parameter_version;
  bool can_reuse;
  {
    std::shared_lock state_lock(model_state_->mutex);
    parameter_version = model_state_->model.p.version();
    synchronizeParameterVersion(parameter_version);
    can_reuse = !from_scratch && acceptedStateMatches(
        particles, parameter_version, AcceptedStateRequirement::FULL_SPATIAL);
    if (can_reuse)
    {
      accumulateAcceptedSpatial(particles.G, particles.L);
      putAcceptedState(buffer);
      return log_value_;
    }
  }

  evaluateLog(particles, particles.G, particles.L);
  {
    std::shared_lock state_lock(model_state_->mutex);
    parameter_version = model_state_->model.p.version();
    synchronizeParameterVersion(parameter_version);
    if (!acceptedStateMatches(particles, parameter_version, AcceptedStateRequirement::FULL_SPATIAL))
      throw std::runtime_error("PsiFormer parameters changed while refreshing a walker buffer");
    putAcceptedState(buffer);
  }
  return log_value_;
}

// Restore only records whose model, parameter, configuration, and products match.
void PsiFormerWF::copyFromBuffer(ParticleSet& particles, WFBufferType& buffer)
{
  std::shared_lock state_lock(model_state_->mutex);
  const std::size_t parameter_version = model_state_->model.p.version();
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

// Return whether a selected local parameter is present in the global active set.
bool PsiFormerWF::hasActiveParameters() const
{
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const OptVariables& variables = optimization_metadata_->variables;
  for (std::size_t local_index = 0; local_index < variables.size(); ++local_index)
    if (variables.where(local_index) >= 0)
      return true;
  return false;
}

// Scatter selected flat derivatives into their global QMCPACK entries.
void PsiFormerWF::addSelectedGradient(const std::vector<double>& flat_gradient, Vector<ValueType>& output) const
{
  addSelectedGradient(flat_gradient.data(), flat_gradient.size(), output);
}

// Scatter directly from a non-owning canonical score view into global active entries.
void PsiFormerWF::addSelectedGradient(const double* flat_gradient,
                                      std::size_t gradient_size,
                                      Vector<ValueType>& output) const
{
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const auto& selected_flat_indices = optimization_metadata_->selected_flat_indices;
  const OptVariables& variables     = optimization_metadata_->variables;
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= output.size())
      throw std::out_of_range("PsiFormer derivative output index is out of range");

    const std::size_t flat_index = selected_flat_indices[local_index];
    if (flat_index >= gradient_size)
      throw std::out_of_range("PsiFormer native derivative is missing a selected flat index");
    if (!psiformer::determinant::isFiniteReal(flat_gradient[flat_index]))
      throw std::runtime_error("PsiFormer native parameter derivative is non-finite");
    output[global_index] += ValueType(flat_gradient[flat_index]);
  }
}

// Contract a canonical score into a caller-owned active-parameter row.
void PsiFormerWF::addSelectedGradientScaled(const double* flat_gradient,
                                            std::size_t gradient_size,
                                            ValueType scale,
                                            ParameterDerivativeView output) const
{
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const auto& selected_flat_indices = optimization_metadata_->selected_flat_indices;
  const OptVariables& variables     = optimization_metadata_->variables;
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (static_cast<std::size_t>(global_index) >= output.size)
      throw std::out_of_range("PsiFormer weighted derivative destination is out of range");

    const std::size_t flat_index = selected_flat_indices[local_index];
    if (flat_index >= gradient_size)
      throw std::out_of_range("PsiFormer weighted native score is incomplete");
    if (!psiformer::determinant::isFiniteReal(flat_gradient[flat_index]))
      throw std::runtime_error("PsiFormer weighted native score is non-finite");
    output[global_index] += scale * ValueType(flat_gradient[flat_index]);
  }
}

// Contract a canonical score into one row of the public matrix compatibility API.
void PsiFormerWF::addSelectedGradientScaled(const double* flat_gradient,
                                            std::size_t gradient_size,
                                            ValueType scale,
                                            Matrix<ValueType>& output,
                                            std::size_t row) const
{
  if (row >= output.rows())
    throw std::out_of_range("PsiFormer derivative-ratio row is out of range");
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const auto& selected_flat_indices = optimization_metadata_->selected_flat_indices;
  const OptVariables& variables     = optimization_metadata_->variables;
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= output.cols())
      throw std::out_of_range("PsiFormer derivative-ratio column is out of range");

    const std::size_t flat_index = selected_flat_indices[local_index];
    if (flat_index >= gradient_size)
      throw std::out_of_range("PsiFormer native derivative-ratio input is incomplete");
    if (!psiformer::determinant::isFiniteReal(flat_gradient[flat_index]))
      throw std::runtime_error("PsiFormer native derivative-ratio score is non-finite");
    output(row, global_index) += scale * ValueType(flat_gradient[flat_index]);
  }
}

// Scatter O_theta(virtual)-O_theta(reference), the contract consumed by NonLocalECPComponent.
void PsiFormerWF::addSelectedGradientDifference(const std::vector<double>& reference_gradient,
                                                const std::vector<double>& virtual_gradient,
                                                Matrix<ValueType>& output,
                                                std::size_t row) const
{
  if (row >= output.rows())
    throw std::out_of_range("PsiFormer derivative-ratio row is out of range");
  std::shared_lock metadata_lock(optimization_metadata_->mutex);
  const auto& selected_flat_indices = optimization_metadata_->selected_flat_indices;
  const OptVariables& variables     = optimization_metadata_->variables;
  for (std::size_t local_index = 0; local_index < selected_flat_indices.size(); ++local_index)
  {
    const int global_index = variables.where(local_index);
    if (global_index < 0)
      continue;
    if (global_index >= output.cols())
      throw std::out_of_range("PsiFormer derivative-ratio column is out of range");

    const std::size_t flat_index = selected_flat_indices[local_index];
    if (flat_index >= reference_gradient.size() || flat_index >= virtual_gradient.size())
      throw std::out_of_range("PsiFormer native derivative-ratio input is incomplete");
    const double difference = virtual_gradient[flat_index] - reference_gradient[flat_index];
    if (!psiformer::determinant::isFiniteReal(difference))
      throw std::runtime_error("PsiFormer native derivative ratio is non-finite");
    output(row, global_index) += ValueType(difference);
  }
}

// Replace each electron by the common virtual position in one state-isolated batch.
void PsiFormerWF::evaluateRatiosAlltoOne(ParticleSet& particles, std::vector<ValueType>& ratios)
{
  if (particles.isSpinor())
    throw std::invalid_argument("PsiFormer all-to-one ratios do not support spinor virtual moves");
  if (ratios.size() != static_cast<std::size_t>(particles.getTotalNum()))
    throw std::invalid_argument("PsiFormer all-to-one ratio output has the wrong size");

  // Oracle and compare modes retain the scalar validation path but use the explicit
  // common position; the WaveFunctionComponent default incorrectly calls activeR().
  if (model_state_->direct_value_mode != DirectBackendMode::DIRECT)
  {
    const pf::Result reference = evaluatePositions(
        particles, -1, nullptr, EvaluationPurpose::VALUE_ONLY);
    for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    {
      const pf::Result moved = evaluatePositions(
          particles, electron, &particles.getActivePos(), EvaluationPurpose::VALUE_ONLY);
      ratios[electron] = makeRatio(moved.sign, moved.logabs, reference.sign, reference.logabs);
    }
    return;
  }

  std::shared_lock state_lock(model_state_->mutex);
  const std::size_t parameter_version = model_state_->model.p.version();
  synchronizeParameterVersion(parameter_version);
  auto& batch = requireDirectBatchWorkspace();
  const std::size_t configurations = static_cast<std::size_t>(particles.getTotalNum()) + 1;
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, configurations);
  packBatchConfiguration(batch, 0, particles);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    packBatchConfiguration(batch, static_cast<std::size_t>(electron) + 1, particles, electron,
                           &particles.getActivePos());

  const pf::DirectBatchValueResultView result =
      model_state_->direct_batch_executor.evaluateValues(batch);
  if (result.parameter_version[0] != parameter_version)
    throw std::logic_error("PsiFormer all-to-one reference observed inconsistent parameters");
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    const std::size_t configuration = static_cast<std::size_t>(electron) + 1;
    if (result.parameter_version[configuration] != parameter_version)
      throw std::logic_error("PsiFormer all-to-one batch observed inconsistent parameters");
    ratios[electron] = makeRatio(result.sign[configuration], result.logabs[configuration],
                                 result.sign[0], result.logabs[0]);
  }
}

// Evaluate independent full-network ratios for all quadrature positions without mutating walker state.
void PsiFormerWF::evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios)
{
  if (virtual_particles.getRefPS().isSpinor())
    throw std::invalid_argument("PsiFormer nonlocal ratios do not support spinor virtual moves");
  if (ratios.size() != static_cast<std::size_t>(virtual_particles.getTotalNum()))
    throw std::invalid_argument("PsiFormer virtual-particle ratio output has the wrong size");

  const ParticleSet& reference = virtual_particles.getRefPS();
  const int electron = virtual_particles.refPtcl;
  if (electron < 0 || electron >= reference.getTotalNum())
    throw std::out_of_range("PsiFormer virtual-particle reference electron is invalid");

  if (model_state_->direct_value_mode != DirectBackendMode::DIRECT)
  {
    const pf::Result reference_result =
        evaluatePositions(reference, -1, nullptr, EvaluationPurpose::VALUE_ONLY);
    for (std::size_t move = 0; move < ratios.size(); ++move)
    {
      const pf::Result virtual_result = evaluatePositions(
          reference, electron, &virtual_particles.R[move], EvaluationPurpose::VALUE_ONLY);
      ratios[move] = makeRatio(virtual_result.sign, virtual_result.logabs,
                               reference_result.sign, reference_result.logabs);
    }
    return;
  }

  std::shared_lock state_lock(model_state_->mutex);
  const std::size_t parameter_version = model_state_->model.p.version();
  synchronizeParameterVersion(parameter_version);
  auto& batch = requireDirectBatchWorkspace();
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, ratios.size() + 1);
  packBatchConfiguration(batch, 0, reference);
  for (std::size_t move = 0; move < ratios.size(); ++move)
    packBatchConfiguration(batch, move + 1, reference, electron, &virtual_particles.R[move]);

  const pf::DirectBatchValueResultView result =
      model_state_->direct_batch_executor.evaluateValues(batch);
  if (result.parameter_version[0] != parameter_version)
    throw std::logic_error("PsiFormer virtual-ratio reference observed inconsistent parameters");
  for (std::size_t move = 0; move < ratios.size(); ++move)
  {
    if (result.parameter_version[move + 1] != parameter_version)
      throw std::logic_error("PsiFormer virtual-ratio batch observed inconsistent parameters");
    ratios[move] = makeRatio(result.sign[move + 1], result.logabs[move + 1],
                             result.sign[0], result.logabs[0]);
  }
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
  if (leader.model_state_->direct_value_mode != DirectBackendMode::DIRECT)
  {
    WaveFunctionComponent::mw_evaluateRatios(wfc_list, virtual_particle_list, ratios);
    return;
  }

  resource.virtual_offsets.resize(wfc_list.size() + 1);
  resource.virtual_offsets[0] = 0;
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    resource.virtual_offsets[walker + 1] = resource.virtual_offsets[walker] + ratios[walker].size() + 1;

  std::shared_lock state_lock(leader.model_state_->mutex);
  const std::size_t parameter_version = leader.model_state_->model.p.version();
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, resource.virtual_offsets.back());
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.synchronizeParameterVersion(parameter_version);
    const auto& virtual_particles = virtual_particle_list[walker];
    const ParticleSet& reference = virtual_particles.getRefPS();
    const std::size_t begin = resource.virtual_offsets[walker];
    packBatchConfiguration(batch, begin, reference);
    for (std::size_t move = 0; move < ratios[walker].size(); ++move)
      packBatchConfiguration(batch, begin + move + 1, reference, virtual_particles.refPtcl,
                             &virtual_particles.R[move]);
  }

  const pf::DirectBatchValueResultView result =
      leader.model_state_->direct_batch_executor.evaluateValues(batch);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const std::size_t reference = resource.virtual_offsets[walker];
    if (result.parameter_version[reference] != parameter_version)
      throw std::logic_error("PsiFormer ragged virtual batch observed inconsistent parameters");
    for (std::size_t move = 0; move < ratios[walker].size(); ++move)
    {
      const std::size_t configuration = reference + move + 1;
      if (result.parameter_version[configuration] != parameter_version)
        throw std::logic_error("PsiFormer ragged virtual batch observed inconsistent parameters");
      ratios[walker][move] = makeRatio(result.sign[configuration], result.logabs[configuration],
                                       result.sign[reference], result.logabs[reference]);
    }
  }
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
  if (!optimization_metadata_->enabled || !hasActiveParameters())
  {
    evaluateRatios(virtual_particles, ratios);
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

  // Seed every compatibility row with -O(reference) before the shared score
  // tape is overwritten, then add O(virtual) as each move is evaluated.
  if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectScoreResult reference_result =
        evaluateDirectScorePositions(reference, -1, nullptr);
    const std::size_t reference_parameter_version = reference_result.parameter_version;
    for (std::size_t move = 0; move < ratios.size(); ++move)
      addSelectedGradientScaled(reference_result.parameter_score.data,
                                reference_result.parameter_score.size, ValueType(-1),
                                derivative_ratios, move);

    for (std::size_t move = 0; move < ratios.size(); ++move)
    {
      const pf::DirectScoreResult virtual_result =
          evaluateDirectScorePositions(reference, electron, &virtual_particles.R[move]);
      if (virtual_result.parameter_version != reference_parameter_version)
        throw std::runtime_error("PsiFormer parameters changed during virtual score evaluation");
      const double ratio = (virtual_result.sign / reference_result.sign) *
          std::exp(virtual_result.logabs - reference_result.logabs);
      if (!psiformer::determinant::isFiniteReal(ratio))
        throw std::runtime_error("PsiFormer virtual-particle ratio is non-finite");
      ratios[move] = ValueType(ratio);
      addSelectedGradientScaled(virtual_result.parameter_score.data,
                                virtual_result.parameter_score.size, ValueType(1),
                                derivative_ratios, move);
    }
    return;
  }

  const pf::Result reference_result =
      evaluatePositions(reference, -1, nullptr, EvaluationPurpose::SCORE_ONLY);
  for (std::size_t move = 0; move < ratios.size(); ++move)
  {
    const pf::Result virtual_result =
        evaluatePositions(reference, electron, &virtual_particles.R[move], EvaluationPurpose::SCORE_ONLY);
    const double ratio = (virtual_result.sign / reference_result.sign) *
        std::exp(virtual_result.logabs - reference_result.logabs);
    if (!psiformer::determinant::isFiniteReal(ratio))
      throw std::runtime_error("PsiFormer virtual-particle ratio is non-finite");
    ratios[move] = ValueType(ratio);
    addSelectedGradientDifference(reference_result.param_gradient, virtual_result.param_gradient,
                                  derivative_ratios, move);
  }
}

// Contract virtual score differences with total-wavefunction quadrature weights.
void PsiFormerWF::evaluateDerivRatiosWeighted(const VirtualParticleSet& virtual_particles,
                                              const OptVariables& optvars,
                                              const std::vector<ValueType>& total_weights,
                                              ParameterDerivativeView weighted_derivatives)
{
  evaluateDerivRatiosWeightedImpl(
      virtual_particles, optvars, total_weights, weighted_derivatives, nullptr);
}

// Validate and reduce one weighted virtual-move set, optionally in crowd-owned scratch.
void PsiFormerWF::evaluateDerivRatiosWeightedImpl(
    const VirtualParticleSet& virtual_particles,
    const OptVariables& optvars,
    const std::vector<ValueType>& total_weights,
    ParameterDerivativeView weighted_derivatives,
    pf::DirectScoreWorkspace* crowd_workspace)
{
  const std::size_t virtual_count = virtual_particles.getTotalNum();
  if (total_weights.size() != virtual_count || weighted_derivatives.size < optvars.size_of_active() ||
      (weighted_derivatives.size != 0 && weighted_derivatives.data == nullptr))
    throw std::invalid_argument("PsiFormer weighted derivative-ratio outputs have the wrong shape");
  if (!optimization_metadata_->enabled || !hasActiveParameters())
    return;
  if (virtual_particles.getRefPS().isSpinor())
    throw std::invalid_argument("PsiFormer weighted nonlocal derivatives do not support spinor virtual moves");

  const ParticleSet& reference = virtual_particles.getRefPS();
  const int electron           = virtual_particles.refPtcl;
  if (electron < 0 || electron >= reference.getTotalNum())
    throw std::out_of_range("PsiFormer weighted derivative-ratio reference electron is invalid");

  const ValueType reference_weight =
      std::accumulate(total_weights.begin(), total_weights.end(), ValueType(0));

  if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
  {
    pf::DirectScoreWorkspace& score_workspace =
        crowd_workspace ? *crowd_workspace : requireDirectScoreWorkspace();
    const pf::DirectScoreResult reference_result =
        evaluateDirectScorePositions(reference, -1, nullptr, score_workspace);
    const std::size_t reference_parameter_version = reference_result.parameter_version;
    addSelectedGradientScaled(reference_result.parameter_score.data,
                              reference_result.parameter_score.size, -reference_weight,
                              weighted_derivatives);

    for (std::size_t move = 0; move < virtual_count; ++move)
    {
      const pf::DirectScoreResult virtual_result =
          evaluateDirectScorePositions(reference, electron, &virtual_particles.R[move], score_workspace);
      if (virtual_result.parameter_version != reference_parameter_version)
        throw std::runtime_error("PsiFormer parameters changed during a weighted virtual score reduction");
      addSelectedGradientScaled(virtual_result.parameter_score.data,
                                virtual_result.parameter_score.size, total_weights[move],
                                weighted_derivatives);
    }
    return;
  }

  const pf::Result reference_result = evaluatePositions(reference, -1, nullptr, EvaluationPurpose::SCORE_ONLY);
  addSelectedGradientScaled(reference_result.param_gradient.data(), reference_result.param_gradient.size(),
                            -reference_weight, weighted_derivatives);

  for (std::size_t move = 0; move < virtual_count; ++move)
  {
    const pf::Result virtual_result =
        evaluatePositions(reference, electron, &virtual_particles.R[move], EvaluationPurpose::SCORE_ONLY);
    addSelectedGradientScaled(virtual_result.param_gradient.data(), virtual_result.param_gradient.size(),
                              total_weights[move], weighted_derivatives);
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

  if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
  {
    pf::DirectScoreWorkspace& score_workspace =
        requireMultiWalkerResource(wfc_list).requireScoreWorkspace();
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = dynamic_cast<PsiFormerWF&>(wfc_list[walker]);
      component.evaluateDerivRatiosWeightedImpl(vp_list[walker], optvars, total_weights[walker].get(),
                                                weighted_derivatives[walker], &score_workspace);
    }
    return;
  }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    auto& component = dynamic_cast<PsiFormerWF&>(wfc_list[walker]);
    component.evaluateDerivRatiosWeighted(vp_list[walker], optvars, total_weights[walker].get(),
                                          weighted_derivatives[walker]);
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
  if (!optimization_metadata_->enabled || !hasActiveParameters())
    return;

  if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
  {
    const pf::DirectScoreResult result = evaluateDirectScorePositions(p, -1, nullptr);
    addSelectedGradient(result.parameter_score.data, result.parameter_score.size, dlogpsi);
  }
  else
  {
    const pf::Result result = evaluate(p, -1, EvaluationPurpose::SCORE_ONLY);
    addSelectedGradient(result.param_gradient, dlogpsi);
  }
}

// Fill score rows using one serialized crowd tape in direct mode.
void PsiFormerWF::mw_evaluateParameterDerivativesWF(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dlogpsi) const
{
  assert(this == &wfc_list.getLeader());
  if (wfc_list.size() != p_list.size() || dlogpsi.getNumOfEntries() != wfc_list.size())
    throw std::invalid_argument("PsiFormer batched score outputs have inconsistent shapes");

  const int parameter_count = dlogpsi.getNumOfParams();
  if (model_state_->direct_score_mode == DirectBackendMode::DIRECT)
  {
    pf::DirectScoreWorkspace& score_workspace =
        requireMultiWalkerResource(wfc_list).requireScoreWorkspace();
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = dynamic_cast<PsiFormerWF&>(wfc_list[walker]);
      if (!component.optimization_metadata_->enabled || !component.hasActiveParameters())
        continue;
      Vector<ValueType> score(dlogpsi[walker], parameter_count);
      const pf::DirectScoreResult result =
          component.evaluateDirectScorePositions(p_list[walker], -1, nullptr, score_workspace);
      component.addSelectedGradient(result.parameter_score.data, result.parameter_score.size, score);
    }
    return;
  }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    dynamic_cast<PsiFormerWF&>(wfc_list[walker]).evaluateDerivativesWF(p_list[walker], optvars, score);
  }
}

// Add score and component kinetic derivatives using the complete TrialWaveFunction gradient in P.G.
void PsiFormerWF::evaluateDerivatives(ParticleSet& p,
                                      const OptVariables& optvars,
                                      Vector<ValueType>& dlogpsi,
                                      Vector<ValueType>& dhpsioverpsi)
{
  if (optimization_metadata_->enabled && hasActiveParameters())
  {
    requireUnitElectronMasses(p);
    requireRealTotalWavefunctionDrift(p);
  }
  evaluateDerivativesImpl(p, optvars, dlogpsi, dhpsioverpsi, nullptr, nullptr);
}

// Evaluate one kinetic response using either lazy scalar scratch or an acquired crowd tape.
void PsiFormerWF::evaluateDerivativesImpl(
    ParticleSet& p,
    const OptVariables&,
    Vector<ValueType>& dlogpsi,
    Vector<ValueType>& dhpsioverpsi,
    pf::DirectKineticWorkspace* crowd_workspace,
    std::vector<double>* crowd_total_log_gradient)
{
  if (!optimization_metadata_->enabled || !hasActiveParameters())
    return;
  if ((crowd_workspace == nullptr) != (crowd_total_log_gradient == nullptr))
    throw std::invalid_argument("PsiFormer kinetic scratch requires both tape and total-drift storage");

  std::optional<pf::Result> oracle;
  if (model_state_->direct_kinetic_mode != DirectBackendMode::DIRECT)
  {
    oracle = evaluate(p, -1, EvaluationPurpose::SCORE_AND_KINETIC);
    if (model_state_->direct_kinetic_mode == DirectBackendMode::ORACLE)
    {
      addSelectedGradient(oracle->param_gradient, dlogpsi);
      addSelectedGradient(oracle->local_energy_param_gradient, dhpsioverpsi);
      return;
    }
  }

  pf::DirectKineticResultView direct;
  {
    std::shared_lock state_lock(model_state_->mutex);
    pf::PsiFormer& model = model_state_->model;
    synchronizeParameterVersion(model.p.version());
    if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
      throw std::runtime_error("PsiFormerWF electron count differs from exported model");

    pf::DirectKineticWorkspace& kinetic_workspace =
        crowd_workspace ? *crowd_workspace : requireDirectKineticWorkspace();
    std::vector<double>& total_log_gradient =
        crowd_total_log_gradient ? *crowd_total_log_gradient : requireDirectTotalLogGradient();
    if (total_log_gradient.size() != 3 * model.ne)
      throw std::logic_error("PsiFormer direct total-drift buffer has the wrong size");

    // ParticleSet::G is the complete TrialWaveFunction drift, not merely the
    // PsiFormer contribution. Repack it for every walker before tape reuse.
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        kinetic_workspace.setPosition(electron, dimension, std::real(p.R[electron][dimension]));
        total_log_gradient[3 * electron + dimension] =
            std::real(p.G[electron][dimension]);
      }

    direct = model_state_->direct_kinetic_executor.evaluate(
        kinetic_workspace, total_log_gradient.data(), total_log_gradient.size());
  }

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

  addSelectedGradient(direct.parameter_score.data(), direct.parameter_score.size(), dlogpsi);
  addSelectedGradient(direct.kinetic_parameter_response.data(),
                      direct.kinetic_parameter_response.size(), dhpsioverpsi);
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

  // Validate every active walker before allocating or mutating the shared
  // kinetic tape so a heterogeneous-mass crowd fails as one atomic request.
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& component = dynamic_cast<const PsiFormerWF&>(wfc_list[walker]);
    if (component.optimization_metadata_->enabled && component.hasActiveParameters())
    {
      requireUnitElectronMasses(p_list[walker]);
      requireRealTotalWavefunctionDrift(p_list[walker]);
    }
  }

  const int parameter_count = dlogpsi.getNumOfParams();
  if (model_state_->direct_kinetic_mode != DirectBackendMode::ORACLE)
  {
    PsiFormerMultiWalkerResource& resource = requireMultiWalkerResource(wfc_list);
    pf::DirectKineticWorkspace& kinetic_workspace = resource.requireKineticWorkspace();
    std::vector<double>& total_log_gradient       = resource.requireTotalLogGradient();
    assert(resource.kinetic_workspace);
    assert(total_log_gradient.size() ==
           3 * model_state_->execution_plan.modelShape().electrons());
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      Vector<ValueType> score(dlogpsi[walker], parameter_count);
      Vector<ValueType> kinetic_response(dhpsioverpsi[walker], parameter_count);
      dynamic_cast<PsiFormerWF&>(wfc_list[walker])
          .evaluateDerivativesImpl(p_list[walker], optvars, score, kinetic_response,
                                   &kinetic_workspace, &total_log_gradient);
    }
    return;
  }

  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    Vector<ValueType> score(dlogpsi[walker], parameter_count);
    Vector<ValueType> kinetic_response(dhpsioverpsi[walker], parameter_count);
    dynamic_cast<PsiFormerWF&>(wfc_list[walker])
        .evaluateDerivatives(p_list[walker], optvars, score, kinetic_response);
  }
}

// Copy optimizer mapping and accepted state while sharing the synchronized native model.
std::unique_ptr<WaveFunctionComponent> PsiFormerWF::makeClone(ParticleSet&) const
{
  return std::make_unique<PsiFormerWF>(*this);
}

} // namespace qmcplusplus
