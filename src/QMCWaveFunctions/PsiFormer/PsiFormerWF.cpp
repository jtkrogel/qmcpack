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
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#define PSIFORMER_LIBRARY
#include "QMCWaveFunctions/PsiFormer/PsiFormerNative.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFourierHead.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerMemoryPolicy.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminantChannelBatch.h"
#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"
#include "QMCDrivers/WFTrain/ResidentEnergySRDiagnostics.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include <cstdio>
#include "QMCWaveFunctions/PsiFormer/PsiFormerScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerAffineObservation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerBatchExecutor.h"
#ifdef QMC_COMPLEX
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexValueExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexSpatialExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexScoreExecutor.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexKineticExecutor.h"
#endif
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceBatchBackend.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDevicePcg.h"
#ifdef QMC_COMPLEX
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexSpatial.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexProducts.h"
#endif
#endif
#include "QMCWaveFunctions/TrialWaveFunction.h"
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

/// Compare every logical and tiled bound in two direct-batch capacity plans.
bool sameDirectBatchCapacityPlan(const pf::DirectBatchCapacityPlan& left,
                                 const pf::DirectBatchCapacityPlan& right) noexcept
{
  return left.logical.value_dense == right.logical.value_dense &&
      left.logical.full_vgl == right.logical.full_vgl &&
      left.logical.active_gradient == right.logical.active_gradient &&
      left.logical.sparse_references == right.logical.sparse_references &&
      left.logical.sparse_replacements == right.logical.sparse_replacements &&
      left.tile.value == right.tile.value &&
      left.tile.full_vgl == right.tile.full_vgl &&
      left.tile.active_gradient == right.tile.active_gradient;
}

/// Compare retained direct-batch storage without setup-only replacement overlap.
bool sameDirectBatchExecutionStorage(
    const pf::DirectBatchStorageRequirement& left,
    const pf::DirectBatchStorageRequirement& right) noexcept
{
  return left.dense_logical == right.dense_logical &&
      left.sparse_logical == right.sparse_logical &&
      left.logical_outputs == right.logical_outputs &&
      left.sparse_tile_positions == right.sparse_tile_positions &&
      left.value_tile == right.value_tile &&
      left.full_vgl_tile == right.full_vgl_tile &&
      left.active_gradient_tile == right.active_gradient_tile &&
      left.shared_spatial_arena == right.shared_spatial_arena;
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

/// Keep a one-electron transaction disjoint from selected and persistent identities.
constexpr std::uint64_t SINGLE_TRANSACTION_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053463154583031); // "PSF1TX01"

/// Keep all-to-one scalar observations disjoint from other transaction identities.
constexpr std::uint64_t ALL_TO_ONE_VALUE_FINGERPRINT_DOMAIN =
    UINT64_C(0x50534641544f3031); // "PSFATO01"

/// Keep virtual-particle scalar observations disjoint from other input identities.
constexpr std::uint64_t VIRTUAL_VALUE_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053465650543031); // "PSFVPT01"

/// Keep walker-buffer storage evidence disjoint from record content identities.
constexpr std::uint64_t WALKER_BUFFER_STORAGE_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053465742533031); // "PSFWBS01"

/// Keep walker-buffer component-input evidence disjoint from storage metadata.
constexpr std::uint64_t WALKER_BUFFER_INPUT_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053465742493031); // "PSFWBI01"

/// Keep persistent record-content identities disjoint from prepared layouts.
constexpr std::uint64_t WALKER_BUFFER_CONTENT_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053465742433031); // "PSFWBC01"

/// Keep cache-reuse inputs disjoint from storage and persistent record hashes.
constexpr std::uint64_t WALKER_BUFFER_REFRESH_FINGERPRINT_DOMAIN =
    UINT64_C(0x5053465742523031); // "PSFWBR01"

/// Identify the fixed scalar record layout used by the Stage-8 walker buffer.
constexpr std::uint64_t WALKER_BUFFER_MAGIC = UINT64_C(0x505349464f524d38);

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

/// Extend identity only for opt-in two-cell profiles, preserving legacy fingerprints.
void mixPeriodicProfile(std::uint64_t& hash, const psiformer::ExecutionEnvironment& environment)
{
  const auto& profile = environment.periodic_profile;
  if (!profile) return;
  mixPersistentString(hash, "periodic_two_cell_profile_v1");
  mixPersistentInteger(hash, profile->version);
  mixPersistentInteger(hash, static_cast<std::uint64_t>(profile->feature_transform));
  mixPersistentInteger(hash, profile->pair_feature_dimension);
  mixPersistentInteger(hash, profile->electron_electron_cusp);
  for (const auto* cell : {&profile->cells.primitive, &profile->cells.simulation})
    for (const auto& row : cell->lattice_vectors)
      for (double value : row) mixPersistentDouble(hash, value);
  for (const auto& row : profile->cells.tiling)
    for (int value : row) mixPersistentInteger(hash, value);
  mixPersistentInteger(hash, profile->primitive_motif.size());
  for (const auto& position : profile->primitive_motif)
    for (double value : position) mixPersistentDouble(hash, value);
  for (int species : profile->motif_species) mixPersistentInteger(hash, species);
  mixPersistentInteger(hash, profile->occupied_columns.size());
  for (const auto& column : profile->occupied_columns)
  {
    for (int value : column.reciprocal) mixPersistentInteger(hash, value);
    mixPersistentInteger(hash, column.band);
    mixPersistentInteger(hash, column.spin);
    mixPersistentDouble(hash, column.occupation);
  }
  if (profile->envelope_policy == psiformer::PeriodicEnvelopePolicy::FOURIER)
  {
    mixPersistentString(hash, "fourier_replacement_envelope_v1");
    mixPersistentInteger(hash, profile->fourier_modes.size());
    for (const auto& mode : profile->fourier_modes)
      for (int value : mode) mixPersistentInteger(hash, value);
    for (double value : profile->fourier_origin) mixPersistentDouble(hash, value);
  }
}

/** Build a cross-rank identity from immutable physics, layout, and initial weights.
 * Later optimizer changes are tracked separately by the synchronized parameter version. */
std::uint64_t persistentModelIdentity(const pf::PsiFormer& model,
                                      const std::string& model_origin,
                                      const std::string& initialization_profile,
                                      std::uint64_t initialization_seed,
                                      const psiformer::ExecutionEnvironment& environment,
                                      const std::array<double, 3>& reduced_twist,
                                      const std::array<double, 3>& cartesian_twist)
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentString(hash, model_origin);
  mixPersistentString(hash, initialization_profile);
  mixPersistentInteger(hash, initialization_seed);
  mixPeriodicProfile(hash, environment);
  // Preserve the established OPEN identity exactly.  Periodic models add the
  // complete feature-policy and cell contract because identical parameters in
  // distinct cells do not represent the same physical wavefunction.
  if (environment.boundary == psiformer::BoundaryCondition::PERIODIC)
  {
    mixPersistentInteger(hash, static_cast<std::uint64_t>(environment.boundary));
    mixPersistentInteger(hash, static_cast<std::uint64_t>(environment.geometry_feature_policy));
    for (bool periodic : environment.periodic_axes)
      mixPersistentInteger(hash, periodic ? 1 : 0);
    for (const auto& vector : environment.lattice_vectors)
      for (double component : vector)
        mixPersistentDouble(hash, component);
  }
  if (std::any_of(reduced_twist.begin(), reduced_twist.end(),
                  [](double component) { return component != 0.0; }))
  {
    mixPersistentString(hash, "outer_twist_phase_v1");
    for (double component : reduced_twist)
      mixPersistentDouble(hash, component);
    for (double component : cartesian_twist)
      mixPersistentDouble(hash, component);
  }
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

/// Run one no-throw restoration action when a private fault scope exits.
template<class Function>
class PsiFormerScopeExit
{
public:
  static_assert(std::is_nothrow_invocable_v<Function&>);

  /// Take ownership of the restoration action without allocating.
  explicit PsiFormerScopeExit(Function function) noexcept(
      std::is_nothrow_move_constructible_v<Function>)
      : function_(std::move(function))
  {}

  /// Prevent two guards from restoring the same mutation.
  PsiFormerScopeExit(const PsiFormerScopeExit&) = delete;

  /// Keep restoration ownership fixed for the complete lexical scope.
  PsiFormerScopeExit& operator=(const PsiFormerScopeExit&) = delete;

  /// Restore the transient mutation on normal and exceptional exits.
  ~PsiFormerScopeExit() noexcept { function_(); }

private:
  Function function_;
};

/// Deduce the private restoration action type while retaining stack ownership.
template<class Function>
auto makePsiFormerScopeExit(Function&& function)
{
  return PsiFormerScopeExit<std::decay_t<Function>>(
      std::forward<Function>(function));
}

/// Return one checked numeric address end without performing pointer arithmetic.
std::uintptr_t checkedAddressEnd(const void* data,
                                 std::size_t bytes,
                                 const char* description)
{
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
    throw std::length_error(description);
  return begin + bytes;
}

/// Return whether two checked nonempty byte ranges overlap.
bool checkedByteRangesOverlap(const void* left,
                              std::size_t left_bytes,
                              const void* right,
                              std::size_t right_bytes,
                              const char* description)
{
  if (left_bytes == 0 || right_bytes == 0)
    return false;
  if (!left || !right)
    throw std::invalid_argument(description);
  const std::uintptr_t left_begin  = reinterpret_cast<std::uintptr_t>(left);
  const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
  const std::uintptr_t left_end =
      checkedAddressEnd(left, left_bytes, description);
  const std::uintptr_t right_end =
      checkedAddressEnd(right, right_bytes, description);
  return left_begin < right_end && right_begin < left_end;
}

/// Load one trivially copyable value from validated character storage.
template<class T>
T loadWalkerBufferValue(const char* source) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(std::addressof(value), source, sizeof(T));
  return value;
}

/// Store one trivially copyable value in an already validated character range.
template<class T>
void storeWalkerBufferValue(char* destination, const T& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(destination, std::addressof(value), sizeof(T));
}

/// Encode one uint64 as two exact full-precision scalar limbs without a cursor.
void storePersistentInteger(char* scalar_data,
                            std::size_t scalar,
                            std::uint64_t value) noexcept
{
  using FullPrecRealType = QMCTraits::FullPrecRealType;
  static_assert(std::numeric_limits<FullPrecRealType>::digits >= 32);
  const FullPrecRealType low = static_cast<FullPrecRealType>(
      value & UINT64_C(0xffffffff));
  const FullPrecRealType high =
      static_cast<FullPrecRealType>(value >> 32);
  storeWalkerBufferValue(
      scalar_data + scalar * sizeof(FullPrecRealType), low);
  storeWalkerBufferValue(
      scalar_data + (scalar + 1) * sizeof(FullPrecRealType), high);
}

/// Mix one trivially copyable object's exact representation into a fingerprint.
template<class T>
void mixPersistentObject(std::uint64_t& hash, const T& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(
      std::addressof(value));
  for (std::size_t byte = 0; byte < sizeof(T); ++byte)
    mixPersistentByte(hash, bytes[byte]);
}

/// Return whether every byte in one validated range is the canonical zero byte.
bool walkerBufferBytesAreZero(const char* data, std::size_t bytes) noexcept
{
  for (std::size_t byte = 0; byte < bytes; ++byte)
    if (static_cast<unsigned char>(data[byte]) != 0U)
      return false;
  return true;
}

/// Return whether one real or complex wavefunction scalar is finite.
template<class T>
bool walkerBufferScalarIsFinite(const T& value) noexcept
{
  if constexpr (std::is_arithmetic_v<T>)
    return psiformer::determinant::isFiniteReal(static_cast<double>(value));
  else
    return psiformer::determinant::isFiniteReal(
               static_cast<double>(std::real(value))) &&
        psiformer::determinant::isFiniteReal(
               static_cast<double>(std::imag(value)));
}

/// Return true only for the binary64 encoding of negative zero.
bool isNegativeZero(double value) noexcept
{
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == UINT64_C(0x8000000000000000);
}

/// Decode one exact uint32 limb from a validated full-precision scalar slot.
std::uint32_t decodePersistentLimb(const char* scalar_data,
                                   std::size_t scalar,
                                   const char* description)
{
  using FullPrecRealType = QMCTraits::FullPrecRealType;
  static_assert(std::numeric_limits<FullPrecRealType>::digits >= 32);
  const FullPrecRealType stored = loadWalkerBufferValue<FullPrecRealType>(
      scalar_data + scalar * sizeof(FullPrecRealType));
  const double encoded = static_cast<double>(stored);
  if (!psiformer::determinant::isFiniteReal(encoded) || encoded < 0.0 ||
      isNegativeZero(encoded) ||
      encoded >
          static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
    throw std::runtime_error(
        std::string("PsiFormer planned walker buffer has an invalid ") +
        description + " limb");
  const auto decoded = static_cast<std::uint32_t>(encoded);
  if (encoded != static_cast<double>(decoded))
    throw std::runtime_error(
        std::string("PsiFormer planned walker buffer has a fractional ") +
        description + " limb");
  return decoded;
}

/// Decode one exact uint64 from two validated full-precision scalar slots.
std::uint64_t decodePersistentInteger(const char* scalar_data,
                                      std::size_t scalar,
                                      const char* description)
{
  const std::uint64_t low =
      decodePersistentLimb(scalar_data, scalar, description);
  const std::uint64_t high =
      decodePersistentLimb(scalar_data, scalar + 1, description);
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

/// Snapshot all physical ions without replacing them by feature-only centers.
std::vector<double> physicalSourceSnapshot(const ParticleSet& ions)
{
  const auto& species = ions.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  if (charge < 0 || ions.GroupID.size() != ions.R.size())
    throw std::invalid_argument("PsiFormer physical source metadata is incomplete");
  std::vector<double> result;
  result.reserve(pf::checkedStorageProduct(5, ions.getTotalNum(), "physical source size overflow"));
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    const auto group = ions.GroupID[ion];
    if (group < 0 || group >= ions.groups())
      throw std::invalid_argument("PsiFormer physical source species is invalid");
    for (int d = 0; d < 3; ++d) result.push_back(ions.R[ion][d]);
    result.push_back(species(charge, group));
    result.push_back(group);
  }
  if (!std::all_of(result.begin(), result.end(), [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("PsiFormer physical source must be finite");
  return result;
}

/// Retain physical species names independently of mutable source metadata.
std::vector<std::string> physicalSpeciesSnapshot(const ParticleSet& ions)
{
  std::vector<std::string> result;
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
    result.push_back(ions.getSpeciesSet().getSpeciesName(ions.GroupID[ion]));
  return result;
}

/// Couple an in-memory native model to its immutable feature and physical source metadata.
struct InitializedNativeModel
{
  pf::PsiFormer model;
  std::string profile;
  std::uint64_t seed;
  psiformer::ExecutionEnvironment environment;
  std::vector<double> physical_source;
  std::vector<std::string> physical_species;
  const ParticleSet* source = nullptr;
};

/** Immutable outer Bloch phase applied by the QMCPACK adapter.
 * The native network and all trainable parameters remain real.
 */
struct PsiFormerTwistPhase
{
  std::array<double, 3> reduced{};
  std::array<double, 3> cartesian{};
  bool enabled = false;

  bool active() const noexcept { return enabled; }
};

/// Convert a reduced twist to the Cartesian wavevector used by the outer phase.
PsiFormerTwistPhase makeTwistPhase(const ParticleSet& electrons,
                                   std::array<double, 3> reduced)
{
  PsiFormerTwistPhase phase;
  phase.reduced = reduced;
  phase.enabled = std::any_of(
      reduced.begin(), reduced.end(),
      [](double component) { return component != 0.0; });
  ParticleSet::PosType reduced_position;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    if (!psiformer::determinant::isFiniteReal(reduced[dimension]))
      throw std::invalid_argument("PsiFormer reduced twist must be finite");
    reduced_position[dimension] = reduced[dimension];
  }
  const auto cartesian = electrons.getLattice().k_cart(reduced_position);
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    if (!psiformer::determinant::isFiniteReal(cartesian[dimension]))
      throw std::invalid_argument(
          "PsiFormer Cartesian twist must be finite for the simulation cell");
    phase.cartesian[dimension] = cartesian[dimension];
  }
  return phase;
}

/// Translate a QMCPACK simulation cell into immutable PsiFormer geometry metadata.
psiformer::ExecutionEnvironment makeExecutionEnvironment(const ParticleSet& electrons)
{
  psiformer::ExecutionEnvironment environment;
  const auto& lattice = electrons.getLattice();
  if (lattice.getSuperCellEnum() == SUPERCELL_OPEN)
    return environment;
  if (lattice.getSuperCellEnum() != SUPERCELL_BULK)
    throw std::invalid_argument(
        "Real Gamma PsiFormer currently supports only fully periodic 3D bulk cells");

  environment.boundary = psiformer::BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy =
      psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.periodic_axes = {true, true, true};
  for (std::size_t axis = 0; axis < 3; ++axis)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      environment.lattice_vectors[axis][dimension] = lattice.R(axis, dimension);
  return environment;
}

/// Require one runtime particle set to match the model's boundary and lattice.
void validateRuntimeLattice(const ParticleSet& particles,
                            const psiformer::ExecutionEnvironment& environment,
                            const char* context)
{
  const auto& lattice = particles.getLattice();
  if (environment.boundary == psiformer::BoundaryCondition::OPEN)
  {
    if (lattice.getSuperCellEnum() != SUPERCELL_OPEN)
      throw std::invalid_argument(std::string(context) +
                                  " boundary differs from the open model feature policy");
    return;
  }

  if (lattice.getSuperCellEnum() != SUPERCELL_BULK)
    throw std::invalid_argument(std::string(context) +
                                " requires the model's 3D bulk periodic cell");
  constexpr double tolerance = 128 * std::numeric_limits<double>::epsilon();
  for (std::size_t axis = 0; axis < 3; ++axis)
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double expected = environment.lattice_vectors[axis][dimension];
      if (std::abs(lattice.R(axis, dimension) - expected) >
          tolerance * (1 + std::abs(expected)))
        throw std::invalid_argument(std::string(context) +
                                    " lattice differs from the model lattice");
    }
}

/// Require compatible boundaries and an identical cell when features are periodic.
void validateMatchingLattices(const ParticleSet& electrons, const ParticleSet& ions)
{
  const auto& electron_lattice = electrons.getLattice();
  const auto& ion_lattice      = ions.getLattice();
  if (electron_lattice.getSuperCellEnum() != ion_lattice.getSuperCellEnum())
    throw std::invalid_argument("PsiFormer electron and source-ion boundary conditions differ");
  // The lattice matrix is not part of the open-boundary feature policy.  Open
  // ParticleSets may therefore carry different finite bounding boxes without
  // changing any PsiFormer input or observable.
  if (electron_lattice.getSuperCellEnum() == SUPERCELL_OPEN)
    return;
  constexpr double tolerance = 128 * std::numeric_limits<double>::epsilon();
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t column = 0; column < 3; ++column)
      if (std::abs(electron_lattice.R(row, column) - ion_lattice.R(row, column)) >
          tolerance * (1 + std::abs(electron_lattice.R(row, column))))
        throw std::invalid_argument("PsiFormer electron and source-ion lattices differ");
}

/** Bind imported periodic nuclei to runtime positions after proving that order is
 * preserved up to one common translation and individual lattice images. */
void retargetPeriodicNuclei(pf::PsiFormer& model,
                            const ParticleSet& ions,
                            const psiformer::ExecutionEnvironment& environment)
{
  if (environment.boundary != psiformer::BoundaryCondition::PERIODIC)
    return;
  if (static_cast<std::size_t>(ions.getTotalNum()) != model.cfg.nuclei.shape[0])
    throw std::invalid_argument("Periodic PsiFormer runtime nucleus count differs from the export");

  ParticleSet::PosType common_translation;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    common_translation[dimension] =
        ions.R[0][dimension] - model.cfg.nuclei.x[dimension];
  const auto& lattice = ions.getLattice();
  constexpr double image_tolerance = 2.0e-10;
  for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
  {
    ParticleSet::PosType residual;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      residual[dimension] = ions.R[nucleus][dimension] -
          model.cfg.nuclei.x[3 * nucleus + dimension] - common_translation[dimension];
    const auto fractional = lattice.toUnit(residual);
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (std::abs(fractional[axis] - std::nearbyint(fractional[axis])) > image_tolerance)
        throw std::invalid_argument(
            "Periodic PsiFormer runtime ions differ from exported ordering/geometry");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      model.cfg.nuclei.x[3 * nucleus + dimension] = ions.R[nucleus][dimension];
  }
}

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
      (!initialized.environment.periodic_profile &&
       shape.nuclei != static_cast<std::size_t>(ions.getTotalNum())))
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

  if (const auto& profile = initialized.environment.periodic_profile)
  {
    validateRuntimeLattice(electrons, initialized.environment, "Periodic PsiFormer initialization");
    validateRuntimeLattice(ions, initialized.environment, "Periodic PsiFormer source");
    if (profile->envelope_policy == psiformer::PeriodicEnvelopePolicy::FOURIER)
      psiformer::validateFourierOrigin(profile->fourier_origin,
          pf::GeometryPositionView::interleaved(nuclear_positions.data(), ions.getTotalNum()));
    if (shape.nuclei != profile->primitive_motif.size())
      throw std::invalid_argument("Periodic PsiFormer feature count differs from its primitive motif");
    std::vector<double> motif_positions;
    for (const auto& position : profile->primitive_motif)
      motif_positions.insert(motif_positions.end(), position.begin(), position.end());
    std::vector<int> physical_species;
    for (int ion = 0; ion < ions.getTotalNum(); ++ion)
      physical_species.push_back(ions.GroupID[ion]);
    const auto mapping = pf::mapPeriodicIonMotif(profile->cells,
        pf::GeometryPositionView::interleaved(motif_positions.data(), shape.nuclei), profile->motif_species,
        pf::GeometryPositionView::interleaved(nuclear_positions.data(), ions.getTotalNum()), physical_species);
    std::vector<double> motif_charges(shape.nuclei, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t ion = 0; ion < mapping.size(); ++ion)
    {
      auto& charge = motif_charges[mapping[ion]];
      if (std::isfinite(charge) && charge != nuclear_charges[ion])
        throw std::invalid_argument("Periodic PsiFormer motif replicas have different charges");
      charge = nuclear_charges[ion];
    }
    nuclear_positions = std::move(motif_positions);
    nuclear_charges = std::move(motif_charges);
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
  const bool profile = static_cast<bool>(initialized.environment.periodic_profile);
  return {std::move(model), std::move(initialized.profile), initialized.seed, std::move(initialized.environment),
          profile ? physicalSourceSnapshot(ions) : std::vector<double>{},
          profile ? physicalSpeciesSnapshot(ions) : std::vector<std::string>{}, profile ? &ions : nullptr};
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

/** Validate the currently supported production route before model materialization.
 *
 * A CPU result is either an explicit request or one structurally valid automatic
 * fallback. A device route needs explicit validated inference coverage and a
 * matching compiled backend; bare forged readiness records remain insufficient.
 */
psiformer::PsiFormerAcceleratorResolution validateProductionAcceleratorResolution(
    psiformer::PsiFormerAcceleratorResolution resolution)
{
  using Backend = psiformer::PsiFormerAcceleratorBackend;
  using Reason  = psiformer::PsiFormerAcceleratorResolutionReason;

  if (resolution.backend != Backend::CPU)
  {
    const auto required = psiformer::psiFormerAcceleratorInferenceOperations();
    if (resolution.reason != Reason::ACCELERATOR_READY || resolution.fallback_count != 0 ||
        resolution.readiness_records_consulted == 0 || resolution.readiness_records_consulted > 2 ||
        (resolution.validated_operations & required) != required ||
        !psiformer::compiledPsiFormerAcceleratorSupport().supports(resolution.backend))
      throw std::invalid_argument("PsiFormer numerical dispatch requires a compiled, coverage-validated accelerator resolution");
    return resolution;
  }

  if (resolution.reason == Reason::CPU_REQUESTED)
  {
    if (resolution.readiness_records_consulted != 0 || resolution.fallback_count != 0)
      throw std::invalid_argument("PsiFormer CPU accelerator resolution has inconsistent diagnostics");
    return resolution;
  }

  const bool valid_fallback_reason =
      resolution.reason == Reason::NOT_COMPILED ||
      resolution.reason == Reason::NO_USABLE_DEVICE ||
      resolution.reason == Reason::INCOMPLETE_OPERATION_COVERAGE ||
      resolution.reason == Reason::RUNTIME_VALIDATION_NOT_RUN ||
      resolution.reason == Reason::RUNTIME_VALIDATION_FAILED;
  if (!valid_fallback_reason || resolution.fallback_count != 1 ||
      resolution.readiness_records_consulted == 0 ||
      resolution.readiness_records_consulted > 2)
    throw std::invalid_argument("PsiFormer automatic CPU fallback has inconsistent diagnostics");
  return resolution;
}

} // namespace

struct PsiFormerDerivativeScratch;

/// Describe an explicitly bounded CPU numerical owner, independently of output storage.
struct CpuTargetVGLPreparation
{
  std::size_t targets = 0, owner_budget = 0, owner_bytes = 0;
};

/// Parse only explicit CPU target controls; unset is off, present values must be positive/finite.
std::size_t positiveCpuTargetControl(const char* name)
{
  const char* value = std::getenv(name);
  if (!value)
    return 0;
  std::size_t result = 0;
  for (const char* digit = value; *digit; ++digit)
  {
    if (*digit < '0' || *digit > '9' ||
        result > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
      throw std::invalid_argument(std::string(name) + " must be a positive base-10 size_t");
    result = 10 * result + (*digit - '0');
  }
  if (result == 0 || result == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(std::string(name) + " must be positive and finite");
  return result;
}

/// Read CPU controls only on the resolved CPU route; admit before clone buffers exist.
CpuTargetVGLPreparation configuredCpuTargetVGL(
    const psiformer::PsiFormerAcceleratorResolution& resolution,
    const psiformer::PsiFormerExecutionPlan& plan, const pf::DirectValueParameterLayout& layout)
{
  if (resolution.backend != psiformer::PsiFormerAcceleratorBackend::CPU)
    return {};
  CpuTargetVGLPreparation result;
  result.targets = positiveCpuTargetControl("PSIFORMER_CPU_VGL_TARGETS");
  result.owner_budget = positiveCpuTargetControl("PSIFORMER_CPU_TARGET_VGL_OWNER_BYTES_PER_CLONE");
  if (!result.targets && !result.owner_budget)
    return result;
  if (!result.targets || !result.owner_budget)
    throw std::invalid_argument("PsiFormer CPU target VGL requires both targets and owner bytes per clone");
  const auto& shape = plan.modelShape();
  result.owner_bytes = pf::targetSpatialWorkspaceStorageRequirement(
      {shape.electrons(), shape.nuclei, shape.determinants, shape.feature_dimension,
       shape.attention_heads, layout.inputWidth(), 0, 0}, result.targets);
  if (result.owner_bytes > result.owner_budget)
    throw std::length_error("PsiFormer CPU target VGL owner exceeds its per-clone byte cap");
  return result;
}

/// Bound one shared CPU derivative tape, independently of objective and inference storage.
struct CpuTargetKineticPreparation
{
  std::size_t targets = 0, owner_budget = 0, owner_bytes = 0;
};

/// Read the CPU streaming-product profile once and admit its exact owner before allocation.
CpuTargetKineticPreparation configuredCpuTargetKinetic(
    const psiformer::PsiFormerAcceleratorResolution& resolution,
    const psiformer::PsiFormerExecutionPlan& plan, const pf::DirectValueParameterLayout& layout)
{
  if (resolution.backend != psiformer::PsiFormerAcceleratorBackend::CPU)
    return {};
  CpuTargetKineticPreparation result;
  result.targets = positiveCpuTargetControl("PSIFORMER_CPU_KINETIC_TARGETS");
  result.owner_budget = positiveCpuTargetControl("PSIFORMER_CPU_TARGET_KINETIC_OWNER_BYTES_PER_LANE");
  if (!result.targets && !result.owner_budget)
    return result;
  if (!result.targets || !result.owner_budget)
    throw std::invalid_argument("PsiFormer CPU target kinetic requires both targets and per-lane owner bytes");
  const auto& s = plan.modelShape();
  result.owner_bytes = pf::targetKineticWorkspaceStorageRequirement({s.electrons(), s.nuclei, s.determinants,
      s.feature_dimension, s.attention_heads, layout.inputWidth(), s.attention_blocks, plan.parameterCount()}, result.targets);
  if (result.owner_bytes > result.owner_budget)
    throw std::length_error("PsiFormer CPU target kinetic owner exceeds its per-lane byte cap");
  return result;
}

#ifdef ENABLE_CUDA
/** Read each developer GPU control once at shared-backend construction.
 * Target limits apply to each owner/logical request; retained-coordinate limits
 * apply to each provider. Neither changes batch_memory or admits aggregate
 * process storage, shared images or vendor reserves.
 */
psiformer::device::DeviceExecutionPreparation configuredGpuTargetPreparation(
    const psiformer::PsiFormerExecutionPlan& plan, psiformer::PsiFormerPrecisionPolicy precision)
{
  const auto [select_vgl, select_kinetic] = psiformer::device::parseDeviceTargetSelection(
      std::getenv("PSIFORMER_GPU_TARGET_SELECTION"));
  const auto positive = [](const char* name) -> std::size_t {
    const char* value = std::getenv(name);
    if (!value)
      return 0;
    std::size_t result = 0;
    if (*value == '\0')
      throw std::invalid_argument(std::string(name) + " must be a positive base-10 integer when set");
    for (const char* digit = value; *digit; ++digit)
    {
      if (*digit < '0' || *digit > '9')
        throw std::invalid_argument(std::string(name) + " must be a positive base-10 integer when set");
      const auto number = static_cast<std::size_t>(*digit - '0');
      if (result > (std::numeric_limits<std::size_t>::max() - number) / 10)
        throw std::invalid_argument(std::string(name) + " exceeds size_t capacity");
      result = 10 * result + number;
    }
    if (result == 0)
      throw std::invalid_argument(std::string(name) + " must be positive when set");
    return result;
  };
  const auto vgl = positive("PSIFORMER_GPU_VGL_TARGETS");
  const auto parallel_trace_bytes = positive("PSIFORMER_GPU_VGL_PARALLEL_TRACE_BYTES");
  const auto kinetic = positive("PSIFORMER_GPU_KINETIC_TARGETS");
  const auto lifted = positive("PSIFORMER_GPU_LIFTED_CHANNELS");
  const auto bytes = positive("PSIFORMER_GPU_TARGET_WORKSPACE_BYTES");
  const auto work = positive("PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET");
  const auto retained_coordinates = positive("PSIFORMER_GPU_RETAINED_COORDINATE_BYTES");
  const auto metropolis = positive("PSIFORMER_GPU_METROPOLIS_BYTES");
  const auto metropolis_drift = positive("PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES");
  const auto active_inference_crowds = positive("PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS");
  if (metropolis_drift && !metropolis)
    throw std::invalid_argument("PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES requires PSIFORMER_GPU_METROPOLIS_BYTES");
  const auto adam = [&]() -> std::size_t {
    const char* value = std::getenv("PSIFORMER_GPU_ADAM_BYTES");
    if (!value || !*value)
      return 0;
    bool enabled = false;
    for (const char* digit = value; *digit; ++digit)
    {
      if (*digit < '0' || *digit > '9')
        throw std::invalid_argument("PSIFORMER_GPU_ADAM_BYTES must be a base-10 byte limit or zero/off");
      enabled = enabled || *digit != '0';
    }
    return enabled ? positive("PSIFORMER_GPU_ADAM_BYTES") : 0;
  }();
  if ((select_vgl && vgl) || (select_kinetic && kinetic))
    throw std::invalid_argument("GPU target selection cannot override explicit targets for the same family");
  if (parallel_trace_bytes && ((!vgl && !select_vgl) || select_kinetic || active_inference_crowds > 1))
    throw std::invalid_argument("Parallel GPU VGL traces require VGL-only targets/selection and one active inference crowd");
  if (lifted && !kinetic && !select_kinetic)
    throw std::invalid_argument("PSIFORMER_GPU_LIFTED_CHANNELS requires PSIFORMER_GPU_KINETIC_TARGETS");
  if ((vgl || kinetic || select_vgl || select_kinetic) ? (!bytes || !work) : (bytes || work))
    throw std::invalid_argument("GPU target settings require targets and both per-owner workspace/request work limits");
  if ((vgl || kinetic || select_vgl || select_kinetic || retained_coordinates || metropolis || adam) &&
      precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE)
    throw std::invalid_argument("GPU target/retained-coordinate preparation requires FP64 reference precision");
  psiformer::device::DeviceExecutionPreparation preparation;
  preparation.active_inference_crowds = active_inference_crowds ? active_inference_crowds : 1;
  preparation.retained_coordinate_workspace_budget_bytes = retained_coordinates;
  preparation.metropolis_workspace_budget_bytes = metropolis;
  preparation.metropolis_drift_budget_bytes = metropolis_drift;
  preparation.adam_workspace_budget_bytes = adam;
  preparation.select_vgl_targets = select_vgl;
  preparation.select_kinetic_targets = select_kinetic;
  if (metropolis && plan.environment().boundary != psiformer::BoundaryCondition::OPEN)
    throw std::invalid_argument("PSIFORMER_GPU_METROPOLIS_BYTES requires real/open sampling");
  if (vgl || select_vgl)
    preparation.vgl = {vgl, bytes, work, parallel_trace_bytes};
  if (kinetic || select_kinetic)
    preparation.kinetic = {kinetic, bytes, work, lifted};
  preparation = psiformer::device::resolveDeviceTargetSelection(plan, preparation);
  psiformer::device::validateDeviceExecutionPreparation(plan, preparation);
  if (parallel_trace_bytes)
    app_log() << "  PsiFormer parallel VGL B1 preallocation: trace_workspace_limit_bytes=" << parallel_trace_bytes
              << ", composite_owner_device_bytes="
              << psiformer::device::estimatePreparedTargetVglCost(plan, 1, preparation.vgl).device_workspace_bytes
              << "; excludes shared image and vendor/private reserve" << std::endl;
  if (select_vgl)
    psiformer::device::reportDeviceTargetSelection(app_log(), "VGL_B1_preflight", plan, 1,
        preparation.vgl.max_targets, psiformer::device::estimatePreparedTargetVglCost(plan, 1, preparation.vgl),
        bytes, work);
  if (select_kinetic)
    psiformer::device::reportDeviceTargetSelection(app_log(), "KINETIC", plan, 1,
        preparation.kinetic.max_targets, psiformer::device::estimateTargetKineticCost(plan,
            preparation.kinetic.max_targets, lifted), bytes, work);
  app_log() << "  PsiFormer GPU target preparation: "
            << (select_vgl ? "vgl_B1_preflight_targets=" : "vgl_targets=") << preparation.vgl.max_targets
            << ", kinetic_targets=" << preparation.kinetic.max_targets
            << ", lifted_channel_slots=" << lifted
            << ", per_owner_workspace_limit_bytes=" << bytes << ", per_request_scalar_work_limit=" << work
            << ", per_provider_retained_coordinate_limit_bytes=" << retained_coordinates
            << ", per_crowd_metropolis_limit_bytes=" << metropolis
            << ", per_optimizer_adam_limit_bytes=" << adam
            << ", maximum_active_inference_crowds=" << preparation.active_inference_crowds
            << ", aggregate_process_budget=no" << std::endl;
  return preparation;
}
#endif

/** Select numerical batch tiles only for an explicitly resolved device route.
 * CPU models never construct vendor handles or upload parameter images.
 */
std::shared_ptr<const pf::DirectBatchTileBackend> makePsiFormerNumericalTiles(
    const pf::PsiFormer& model, const psiformer::PsiFormerExecutionPlan& plan,
    const psiformer::PsiFormerAcceleratorResolution& resolution,
    psiformer::PsiFormerPrecisionPolicy precision)
{
  if (precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE &&
      (precision != psiformer::PsiFormerPrecisionPolicy::FP32_DENSE_FP64_ISLANDS ||
       resolution.backend == psiformer::PsiFormerAcceleratorBackend::CPU ||
       plan.environment().boundary != psiformer::BoundaryCondition::OPEN))
    throw std::invalid_argument("Explicit mixed PsiFormer precision requires an open-real GPU inference model");
  if (resolution.backend == psiformer::PsiFormerAcceleratorBackend::CPU)
    return {};
#ifdef ENABLE_CUDA
  using namespace psiformer::device;
  if (resolution.backend != PreparedDeviceCrowdHooks::backend())
    throw std::invalid_argument("PsiFormer numerical tile request does not match the compiled device backend");
  const auto preparation = configuredGpuTargetPreparation(plan, precision);
  reportCurrentDeviceMemoryAdmission(app_log());
  auto image = std::make_shared<PreparedDeviceModel>(plan, model.p.flat_values(), model.cfg.nuclei.x, model.p.version());
  const auto backend = resolution.backend;
  const auto version = model.p.version();
  return std::make_shared<PreparedDeviceBatchBackend>(std::move(image), [backend, version, preparation]() {
    app_log() << "  PsiFormer numerical device tile completed: backend="
              << psiformer::psiFormerAcceleratorBackendName(backend)
              << ", parameter_version=" << version
              << (preparation.select_vgl_targets ? ", vgl_B1_preflight_targets=" : ", vgl_targets=")
              << preparation.vgl.max_targets
              << ", maximum_active_inference_crowds=" << preparation.active_inference_crowds
              << ", CPU fallback=no" << std::endl;
  }, precision, preparation);
#else
  throw std::invalid_argument("PsiFormer numerical device tiles are not compiled in this executable");
#endif
}

/** Shared native model protected at the optimizer/evaluator synchronization
 * boundary. Component clones retain only walker-local move state. */
class PsiFormerSharedState
{
public:
  /// Load the model that all clones of one PsiFormer component will share.
  PsiFormerSharedState(
      const std::string& parameters,
      const std::string& configuration,
      psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
      psiformer::PsiFormerPrecisionPolicy precision)
      : PsiFormerSharedState([&]() {
          validateProductionAcceleratorResolution(accelerator_resolution);
          return pf::PsiFormer(parameters, configuration);
        }(), "hdf5", "external_hdf5", 0, {}, {}, accelerator_resolution, precision)
  {}

  /// Load and retarget a periodic import against explicit runtime particle sets.
  PsiFormerSharedState(const std::string& parameters,
                       const std::string& configuration,
                       const ParticleSet& electrons,
                       const ParticleSet& ions,
                       std::array<double, 3> reduced_twist,
                       psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                       psiformer::PsiFormerPrecisionPolicy precision)
      : PsiFormerSharedState([&]() {
          validateProductionAcceleratorResolution(accelerator_resolution);
          validateMatchingLattices(electrons, ions);
          pf::PsiFormer model(parameters, configuration);
          const auto environment = makeExecutionEnvironment(electrons);
          retargetPeriodicNuclei(model, ions, environment);
          return model;
        }(), "hdf5", "external_hdf5", 0, makeExecutionEnvironment(electrons),
            makeTwistPhase(electrons, reduced_twist), accelerator_resolution, precision)
  {}

  /// Materialize a fresh in-memory model and retain its reproducibility provenance.
  PsiFormerSharedState(psiformer::InitializedPsiFormerParameters initialized,
                       const ParticleSet& electrons,
                       const ParticleSet& ions,
                       std::array<double, 3> reduced_twist,
                       psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                       psiformer::PsiFormerPrecisionPolicy precision)
      : PsiFormerSharedState([&]() {
          validateProductionAcceleratorResolution(accelerator_resolution);
          return makeInitializedNativeModel(std::move(initialized), electrons, ions);
        }(), makeExecutionEnvironment(electrons),
            makeTwistPhase(electrons, reduced_twist), accelerator_resolution, precision)
  {}

  /// Take ownership of a converted initialized model without copying its flat parameters.
  PsiFormerSharedState(InitializedNativeModel initialized,
                       psiformer::ExecutionEnvironment environment,
                       PsiFormerTwistPhase twist_phase,
                       psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                       psiformer::PsiFormerPrecisionPolicy precision)
      : PsiFormerSharedState(std::move(initialized.model), "internal",
                             std::move(initialized.profile), initialized.seed,
                             initialized.environment.periodic_profile ? std::move(initialized.environment) : std::move(environment),
                             std::move(twist_phase), accelerator_resolution, precision,
                             std::move(initialized.physical_source), std::move(initialized.physical_species), initialized.source)
  {}

  /// Complete shared executor construction for imported and internally initialized models.
  PsiFormerSharedState(pf::PsiFormer model_input,
                       std::string origin,
                       std::string profile,
                       std::uint64_t seed,
                       psiformer::ExecutionEnvironment environment,
                       PsiFormerTwistPhase twist_phase,
                       psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                       psiformer::PsiFormerPrecisionPolicy precision,
                       std::vector<double> physical_source = {},
                       std::vector<std::string> physical_species = {}, const ParticleSet* source = nullptr)
      : accelerator_resolution(validateProductionAcceleratorResolution(accelerator_resolution)),
        precision(precision),
        model(std::move(model_input)),
        model_origin(std::move(origin)),
        initialization_profile(std::move(profile)),
        initialization_seed(seed),
        twist_phase(std::move(twist_phase)),
        physical_source(std::move(physical_source)), physical_species(std::move(physical_species)), source(source),
        persistent_model_identity(
            [&]() {
              auto identity = persistentModelIdentity(model, model_origin, initialization_profile,
                                    initialization_seed, environment,
                                    this->twist_phase.reduced,
                                    this->twist_phase.cartesian);
              if (environment.periodic_profile)
              {
                for (double value : this->physical_source) mixPersistentDouble(identity, value);
                for (const auto& name : this->physical_species) mixPersistentString(identity, name);
              }
              return identity;
            }()),
        execution_plan(psiformer::PsiFormerExecutionPlan::fromParameters(
            model.p,
            {/*spin_up_electrons=*/model.cfg.nup,
             /*spin_down_electrons=*/model.cfg.ndown,
             /*nuclei=*/model.cfg.nuclei.shape[0],
             /*determinants=*/model.ndet,
             /*feature_dimension=*/model.dim,
             /*attention_heads=*/model.heads,
             /*attention_blocks=*/model.blocks}, std::move(environment))),
        direct_value_executor(model, execution_plan),
        direct_spatial_executor(model, direct_value_executor, execution_plan),
        numerical_tile_backend(makePsiFormerNumericalTiles(model, execution_plan, this->accelerator_resolution, precision)),
        direct_batch_executor(direct_value_executor, direct_spatial_executor, numerical_tile_backend),
        direct_score_executor(model, execution_plan),
        direct_kinetic_executor(model, execution_plan),
        direct_value_mode(configuredDirectBackend("PSIFORMER_VALUE_BACKEND")),
        direct_spatial_mode(configuredDirectBackend("PSIFORMER_SPATIAL_BACKEND")),
        direct_score_mode(configuredDirectBackend("PSIFORMER_SCORE_BACKEND")),
        direct_kinetic_mode(configuredDirectBackend("PSIFORMER_KINETIC_BACKEND")),
        cpu_target_vgl(configuredCpuTargetVGL(this->accelerator_resolution, execution_plan,
                                            *direct_value_executor.layout())),
        cpu_target_kinetic(configuredCpuTargetKinetic(this->accelerator_resolution, execution_plan,
                                                    *direct_value_executor.layout()))
  {
    if (execution_plan.environment().periodic_profile &&
        (numerical_tile_backend || direct_value_mode != DirectBackendMode::DIRECT ||
         direct_spatial_mode != DirectBackendMode::DIRECT || direct_score_mode != DirectBackendMode::DIRECT ||
         direct_kinetic_mode != DirectBackendMode::DIRECT))
      throw std::invalid_argument("Periodic real profile requires direct CPU value, spatial, score and kinetic products");
    if (cpu_target_kinetic.targets &&
        (direct_score_mode != DirectBackendMode::DIRECT || direct_kinetic_mode != DirectBackendMode::DIRECT))
      throw std::invalid_argument("PsiFormer CPU target streaming products require DIRECT score/kinetic backends");
    if (cpu_target_vgl.targets && direct_spatial_mode != DirectBackendMode::DIRECT)
      throw std::invalid_argument("PsiFormer CPU target VGL requires the direct spatial backend");
    if (numerical_tile_backend && (direct_value_mode != DirectBackendMode::DIRECT ||
        direct_spatial_mode != DirectBackendMode::DIRECT))
      throw std::invalid_argument("PsiFormer GPU inference requires direct value/spatial modes, not CPU oracle or compare modes");
    if (numerical_tile_backend)
      if (const char* flag = std::getenv("PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY"))
      {
        if (std::string(flag) != "0" && std::string(flag) != "1")
          throw std::invalid_argument("PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY must be 0 or 1");
        resident_device_authority_requested = std::string(flag) == "1";
      }
    host_mirror_version = model.p.version();
    if (twist_phase.active() &&
        execution_plan.environment().boundary != psiformer::BoundaryCondition::PERIODIC)
      throw std::invalid_argument(
          "A nonzero PsiFormer twist requires a fully periodic model");
#ifndef QMC_COMPLEX
    if (twist_phase.active())
      throw std::invalid_argument(
          "A nonzero PsiFormer twist requires a complex QMCPACK build");
#endif
    if (execution_plan.environment().boundary == psiformer::BoundaryCondition::PERIODIC &&
        (direct_value_mode != DirectBackendMode::DIRECT ||
         direct_spatial_mode != DirectBackendMode::DIRECT ||
         direct_score_mode != DirectBackendMode::DIRECT ||
         direct_kinetic_mode != DirectBackendMode::DIRECT))
      throw std::invalid_argument(
          "Periodic PsiFormer requires direct value/spatial/score/kinetic backends; "
          "the molecular native oracle is not periodic");
  }

  /// Publish under the caller's exclusive model lock, including restart/rollback.
  /// Device upload and native validation precede the shared version transition.
  void setParameterValues(const std::vector<double>& values)
  {
    auto& parameters = model.p;
    if (parameters.version() == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PsiFormer parameter version exhausted");
    if (numerical_tile_backend)
      numerical_tile_backend->publishParameters(values, parameters.version() + 1,
                                                [&]() { parameters.set_flat_values(values); });
    else
      parameters.set_flat_values(values);
    host_mirror_version = parameters.version();
  }

  /// Validate fixed physical ions centrally before any new-profile model read.
  void validatePhysicalSource() const
  {
    if (!execution_plan.environment().periodic_profile)
      return;
    if (!source || source->getTotalNum() != static_cast<int>(physical_source.size() / 5) ||
        physical_source.size() % 5 != 0 || physical_source.empty() ||
        source->R.size() != physical_source.size() / 5 || source->GroupID.size() != source->R.size() ||
        physical_species.size() != source->R.size())
      throw std::invalid_argument("Periodic real profile physical source extent changed");
    validateRuntimeLattice(*source, execution_plan.environment(), "Periodic real fixed source");
    const auto& species = source->getSpeciesSet();
    const int charge = species.findAttribute("charge");
    if (charge < 0)
      throw std::invalid_argument("Periodic real profile source charges are unavailable");
    for (std::size_t ion = 0; ion < source->R.size(); ++ion)
    {
      const int group = source->GroupID[ion];
      if (group < 0 || group >= source->groups() || group != physical_source[5 * ion + 4] ||
          species(charge, group) != physical_source[5 * ion + 3] ||
          species.getSpeciesName(group) != physical_species[ion])
        throw std::invalid_argument("Periodic real profile physical source species or charge changed");
      for (std::size_t d = 0; d < 3; ++d)
        if (source->R[ion][d] != physical_source[5 * ion + d])
          throw std::invalid_argument("Periodic real profile physical source positions changed");
    }
  }

  /// Refresh cached native leaves without upgrading a held shared model lock.
  void ensureHostMirror()
  {
#ifdef ENABLE_CUDA
    std::lock_guard refresh_lock(host_refresh_mutex);
    for (;;)
    {
      std::shared_ptr<const psiformer::device::PreparedDeviceModel> image;
      std::size_t version;
      {
        std::shared_lock read(mutex);
        version = model.p.version();
        if (host_mirror_version == version)
          return;
        const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(numerical_tile_backend.get());
        if (!resident_device_authority_enabled || !backend)
          throw std::logic_error("PsiFormer stale HOST mirror without DEVICE authority");
        image = backend->currentImage();
        if (image->version() != version)
          throw std::logic_error("PsiFormer DEVICE image/native metadata version mismatch");
      }
      app_log() << "PsiFormer cold HOST mirror refresh: attempted_bytes="
                << 8 * image->plan().parameterCount() << ", version=" << version << std::endl;
      std::vector<double> values(image->plan().parameterCount());
      image->downloadCanonical(values);
      std::unique_lock write(mutex);
      const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(numerical_tile_backend.get());
      if (model.p.version() != version || !backend || backend->currentImage() != image)
        continue;
      auto install = model.p.prepare_same_version_flat_values(std::move(values), version);
      install.commit();
      host_mirror_version = version;
      return;
    }
#endif
  }

  /// Only direct GPU VALUE dispatch may use stale native leaves as metadata.
  bool deviceOnlyValueReads() const noexcept
  {
    return resident_device_authority_enabled && numerical_tile_backend &&
        precision == psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE &&
        direct_value_mode == DirectBackendMode::DIRECT &&
        execution_plan.environment().boundary == psiformer::BoundaryCondition::OPEN &&
        execution_plan.environment().amplitude_scalar_domain == psiformer::ScalarDomain::REAL;
  }

  /// Partial HOST writers must reacquire and recheck after the cold fence.
  std::unique_lock<std::shared_mutex> acquireHostWrite()
  {
    for (;;)
    {
      ensureHostMirror();
      std::unique_lock write(mutex);
      if (host_mirror_version == model.p.version())
        return write;
    }
  }

  mutable std::shared_mutex mutex;
  std::mutex host_refresh_mutex;
  std::size_t host_mirror_version = 0;
  bool resident_device_authority_requested = false;
  bool resident_device_authority_enabled = false;
  /// Factory lock is separate from the scratch execution lease and model lock.
  mutable std::mutex derivative_scratch_factory_mutex;
  std::weak_ptr<PsiFormerDerivativeScratch> derivative_scratch;
  /// Number of pending planned one-electron crowd transactions sharing this model.
  /// Each crowd checks in once, never once per lane.
  std::atomic<std::size_t> planned_single_transaction_count{0};
  /// Number of pending planned selected crowd transactions sharing this model.
  /// Each crowd checks in once, never once per lane.
  std::atomic<std::size_t> planned_selected_transaction_count{0};
  /// Immutable resolved production route shared by every component clone.
  const psiformer::PsiFormerAcceleratorResolution accelerator_resolution;
  /// Immutable numerical policy, preserved across clones and candidate isolation.
  const psiformer::PsiFormerPrecisionPolicy precision;
  pf::PsiFormer model;
  /// Whether the initial model came from the legacy HDF5 import or an internal initializer.
  const std::string model_origin;
  /// Versioned initializer name or stable external-HDF5 profile sentinel.
  const std::string initialization_profile;
  /// Explicit component-local initialization seed, zero for imported models.
  const std::uint64_t initialization_seed;
  /// Fixed outer Bloch phase; zero keeps the exact Gamma/open adapter path.
  const PsiFormerTwistPhase twist_phase;
  /// Full physical supercell source, distinct from feature-only primitive centers.
  const std::vector<double> physical_source;
  const std::vector<std::string> physical_species;
  const ParticleSet* const source;
  /// Stable content identity used to reject buffers from another physical model.
  const std::uint64_t persistent_model_identity;
  /// Immutable typed tensor descriptors shared by every component clone.
  const psiformer::PsiFormerExecutionPlan execution_plan;
  /// Immutable direct forward executor sharing the versioned parameter store.
  const pf::DirectValueExecutor direct_value_executor;
  /// Immutable direct trace-jet executor sharing the value parameter layout.
  const pf::DirectSpatialExecutor direct_spatial_executor;
  /// Immutable numerical route; device scratch is owned by individual batch workspaces.
  const std::shared_ptr<const pf::DirectBatchTileBackend> numerical_tile_backend;
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
  /// Per-clone target numerical payload only; other products/owners are excluded.
  const CpuTargetVGLPreparation cpu_target_vgl;
  /// Objective-independent streaming SCORE/LOCAL_ENERGY preparation, per shared execution lane.
  const CpuTargetKineticPreparation cpu_target_kinetic;
  /// Pretraining has no retained streaming snapshot to keep its lane alive.
  /// Cache one lane across orbital-loss calls; never one tape per walker/epoch.
  std::shared_ptr<PsiFormerDerivativeScratch> orbital_pretraining_scratch;
  /// Explicit resident ENERGY keeps one existing lane/tapes alive across iterations.
  std::shared_ptr<PsiFormerDerivativeScratch> resident_energy_scratch;
#ifdef ENABLE_CUDA
  psiformer::device::DeviceMemorySnapshot resident_transfer_baseline;
  bool resident_transfer_window_active = false;
  /// Optimizer workspace dies BEFORE its anchored Session; source diagnostics exclude it.
  std::shared_ptr<psiformer::device::PreparedDevicePcg> resident_energy_sr_workspace;
#endif
  bool resident_energy_sr_reuse = false;
  bool resident_energy_sr_workspace_started = false;
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
  explicit PsiFormerReadTransaction(PsiFormerSharedState& state, bool device_only = false)
      : state_(state), lock_(state_.mutex)
  {
    state_.validatePhysicalSource();
    while (!device_only && state_.host_mirror_version != state_.model.p.version())
    {
      lock_.unlock();
      state_.ensureHostMirror();
      lock_.lock();
    }
    parameter_version_ = state_.model.p.version();
  }

  const PsiFormerSharedState& state() const noexcept { return state_; }
  const pf::PsiFormer& model() const
  {
    if (state_.host_mirror_version != parameter_version_)
      throw std::logic_error("DEVICE-only PsiFormer transaction cannot consume stale HOST leaves");
    return state_.model;
  }
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

private:
  PsiFormerSharedState& state_;
  std::shared_lock<std::shared_mutex> lock_;
  std::size_t parameter_version_ = 0;
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

namespace
{
/// Convert explicit real/imaginary parts to the build-selected scalar type.
template<class Target>
Target makePsiFormerScalar(double real_part, double imaginary_part)
{
  if constexpr (IsComplex_t<Target>::value)
    return Target(real_part, imaginary_part);
  else
  {
    if (imaginary_part != 0.0)
      throw std::logic_error(
          "A nonzero PsiFormer twist cannot publish through a real scalar build");
    return static_cast<Target>(real_part);
  }
}

/// Return the fixed outer-phase angle for one complete electron configuration.
double twistPhaseAngle(const PsiFormerSharedState& state,
                       const ParticleSet& particles,
                       int replaced_particle = -1,
                       const ParticleSet::PosType* replacement_position = nullptr) noexcept
{
  if (!state.twist_phase.active())
    return 0.0;
  double phase = 0.0;
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : particles.activeR(electron))
        : particles.R[electron];
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      phase += state.twist_phase.cartesian[dimension] * position[dimension];
  }
  return phase;
}

/// Return the outer-phase change for replacing one electron position.
double twistDisplacementPhase(const PsiFormerSharedState& state,
                              const ParticleSet::PosType& accepted,
                              const ParticleSet::PosType& proposed) noexcept
{
  if (!state.twist_phase.active())
    return 0.0;
  double phase = 0.0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    phase += state.twist_phase.cartesian[dimension] *
        (proposed[dimension] - accepted[dimension]);
  return phase;
}

/// Return the phase for an atomic selected-electron descriptor.
double twistPhaseAngle(const PsiFormerSharedState& state,
                       const ParticleSet& particles,
                       const MCMultiParticleMoves<CoordsType::POS>::Slice& moves) noexcept
{
  if (!state.twist_phase.active())
    return 0.0;
  double phase = twistPhaseAngle(state, particles);
  for (std::size_t selected = 0; selected < moves.size(); ++selected)
  {
    const std::size_t electron =
        static_cast<std::size_t>(moves.particleIndex(selected));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      phase += state.twist_phase.cartesian[dimension] *
          (moves.proposedPosition(selected)[dimension] -
           particles.R[electron][dimension]);
  }
  return phase;
}

/// Combine the real native amplitude with the immutable outer twist phase.
WaveFunctionComponent::LogValue makePhysicalLogValue(
    const PsiFormerSharedState& state,
    double sign,
    double logabs,
    double twist_phase) noexcept
{
  return {logabs, (sign < 0.0 ? M_PI : 0.0) +
          (state.twist_phase.active() ? twist_phase : 0.0)};
}

/// Form a physical ratio from real amplitude data and an outer-phase change.
WaveFunctionComponent::PsiValue makePhysicalRatio(
    const PsiFormerSharedState& state,
    double proposed_sign,
    double proposed_logabs,
    double reference_sign,
    double reference_logabs,
    double phase_change)
{
  const double real_ratio = (proposed_sign / reference_sign) *
      std::exp(proposed_logabs - reference_logabs);
  if (!psiformer::determinant::isFiniteReal(real_ratio) ||
      !psiformer::determinant::isFiniteReal(phase_change))
    throw std::runtime_error("PsiFormer batch ratio is non-finite");
  if (!state.twist_phase.active())
    return WaveFunctionComponent::PsiValue(real_ratio);
  return makePsiFormerScalar<WaveFunctionComponent::PsiValue>(
      real_ratio * std::cos(phase_change),
      real_ratio * std::sin(phase_change));
}

/// Convert one real native log-gradient component to its physical complex value.
WaveFunctionComponent::ValueType makePhysicalGradient(
    const PsiFormerSharedState& state,
    double native,
    std::size_t dimension) noexcept
{
  if (!state.twist_phase.active())
    return static_cast<WaveFunctionComponent::ValueType>(native);
  return makePsiFormerScalar<WaveFunctionComponent::ValueType>(
      native, state.twist_phase.cartesian[dimension]);
}

/// Validate sign, magnitude, and phase against the immutable policy and coordinates.
bool isCoherentPhysicalValue(const PsiFormerSharedState& state,
                             double sign,
                             const WaveFunctionComponent::LogValue& log_value,
                             const ParticleSet& particles,
                             int replaced_particle = -1,
                             const ParticleSet::PosType* replacement_position = nullptr) noexcept
{
  if ((sign != -1.0 && sign != 1.0) ||
      !psiformer::determinant::isFiniteReal(std::real(log_value)) ||
      !psiformer::determinant::isFiniteReal(std::imag(log_value)))
    return false;
  return log_value == makePhysicalLogValue(
      state, sign, std::real(log_value),
      twistPhaseAngle(state, particles, replaced_particle,
                      replacement_position));
}
} // namespace

/** Own copied configurations and reusable scratch for overlap-channel evaluation. */
class PsiFormerDeterminantChannelBatch final
    : public psiformer::DeterminantChannelBatch
{
public:
  PsiFormerDeterminantChannelBatch(
      std::shared_ptr<PsiFormerSharedState> model_state,
      std::shared_ptr<const wftrain::StructuredParameterSchema> schema,
      std::size_t parameter_version,
      std::size_t sample_count,
      std::vector<double> positions)
      : model_state_(std::move(model_state)),
        schema_(std::move(schema)),
        parameter_version_(parameter_version),
        sample_count_(sample_count),
        electron_count_(model_state_->execution_plan.modelShape().electrons()),
        channel_count_(model_state_->execution_plan.modelShape().determinants),
        positions_(std::move(positions)),
        phases_(channel_count_),
        log_abs_(channel_count_),
        workspace_(model_state_->numerical_tile_backend ? nullptr : model_state_->direct_score_executor.makeWorkspace())
  {
    const std::size_t expected_positions = checkedBatchMemoryMultiply(
        checkedBatchMemoryMultiply(sample_count_, electron_count_,
                                   "PsiFormer overlap sample positions"),
        std::size_t{3}, "PsiFormer overlap Cartesian positions");
    if (positions_.size() != expected_positions || channel_count_ == 0 ||
        schema_->parameterCount() != model_state_->execution_plan.parameterCount() ||
        (workspace_ && workspace_->scoreSize() != schema_->parameterCount()))
      throw std::invalid_argument(
          "PsiFormer determinant-channel batch has inconsistent shape or schema");
    std::size_t workspace_bytes = 0;
#ifdef ENABLE_CUDA
    if (model_state_->numerical_tile_backend)
    {
      const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
          model_state_->numerical_tile_backend.get());
      if (!backend)
        throw std::logic_error("PsiFormer independent channels require the numerical device backend");
      device_products_ = backend->prepareProductSession(
          psiformer::device::ProductSessionProfile::SCORE_ONLY);
      device_positions_.resize(3 * electron_count_);
      workspace_bytes = checkedBatchMemoryAdd(device_products_->deviceWorkspaceBytes(),
          device_products_->hostWorkspaceBytes(), "PsiFormer overlap device scratch");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(device_positions_.capacity(), sizeof(double), "PsiFormer overlap staging"),
          "PsiFormer overlap device workspace");
    }
    else
#endif
      workspace_bytes = checkedBatchMemoryAdd(workspace_->vectorStorageBytes(), workspace_->scoreStorageBytes(),
                                               "PsiFormer overlap score workspace");
    retained_numeric_bytes_ = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(
            checkedBatchMemoryMultiply(positions_.capacity(), sizeof(double),
                                       "PsiFormer overlap position storage"),
            checkedBatchMemoryMultiply(
                checkedBatchMemoryAdd(phases_.capacity(), log_abs_.capacity(),
                                      "PsiFormer overlap channel scratch"),
                sizeof(double), "PsiFormer overlap channel storage"),
            "PsiFormer overlap sample and channel storage"),
        workspace_bytes,
        "PsiFormer overlap retained storage");
  }

  const std::string& providerId() const noexcept override
  {
    return schema_->providerId();
  }

  const std::string& schemaFingerprint() const noexcept override
  {
    return schema_->fingerprint();
  }

  std::size_t parameterVersion() const noexcept override { return parameter_version_; }
  std::size_t sampleCount() const noexcept override { return sample_count_; }
  std::size_t channelCount() const noexcept override { return channel_count_; }
  std::size_t parameterCount() const noexcept override { return schema_->parameterCount(); }

  void evaluateAmplitudes(
      std::size_t sample,
      psiformer::RealSignedLogAmplitude* output,
      std::size_t output_count) const override
  {
    if (sample >= sample_count_ || !output || output_count != channel_count_)
      throw std::invalid_argument(
          "PsiFormer determinant-channel amplitude request has invalid extent");
    PsiFormerReadTransaction state_lock(*model_state_);
    requireBoundVersion();
    setWorkspacePositions(sample);
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      device_products_->channelAmplitudes(device_positions_.data(), phases_.data(), log_abs_.data(),
                                           channel_count_, parameter_version_);
      for (std::size_t channel = 0; channel < channel_count_; ++channel)
        output[channel] = {phases_[channel], log_abs_[channel]};
      return;
    }
#endif
    const pf::DirectChannelAmplitudeResult result =
        model_state_->direct_score_executor.evaluateChannelAmplitudes(
            *workspace_, phases_.data(), log_abs_.data(), channel_count_);
    if (result.parameter_version != parameter_version_ ||
        result.channel_count != channel_count_)
      throw std::logic_error(
          "PsiFormer determinant-channel amplitude result has the wrong identity");
    for (std::size_t channel = 0; channel < channel_count_; ++channel)
      output[channel] = {phases_[channel], log_abs_[channel]};
  }

  void evaluateWeightedScore(std::size_t sample,
                             const double* channel_weights,
                             std::size_t channel_count,
                             double* output,
                             std::size_t output_count) const override
  {
    if (sample >= sample_count_ || !channel_weights || channel_count != channel_count_ ||
        !output || output_count != schema_->parameterCount())
      throw std::invalid_argument(
          "PsiFormer determinant-channel score request has invalid extent");
    PsiFormerReadTransaction state_lock(*model_state_);
    requireBoundVersion();
    setWorkspacePositions(sample);
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      const auto& response = device_products_->channelScore(device_positions_.data(), channel_weights,
                                                            channel_count_, parameter_version_);
      std::copy(response.begin(), response.end(), output);
      if (!reported_device_channels_)
      {
        app_log() << "  PsiFormer numerical device determinant-channel product completed: parameter_version="
                  << parameter_version_ << ", CPU derivative fallback=no" << std::endl;
        reported_device_channels_ = true;
      }
      return;
    }
#endif
    const pf::DirectWeightedChannelScoreResult result =
        model_state_->direct_score_executor.evaluateWeightedChannelScore(
            *workspace_, channel_weights, channel_count);
    if (result.parameter_version != parameter_version_ ||
        result.parameter_score.size != output_count)
      throw std::logic_error(
          "PsiFormer weighted-channel score result has the wrong identity");
    std::copy(result.parameter_score.begin(), result.parameter_score.end(), output);
  }

  std::size_t retainedNumericBytes() const noexcept override
  {
    return retained_numeric_bytes_;
  }

private:
  void setWorkspacePositions(std::size_t sample) const
  {
    const double* positions =
        positions_.data() + sample * electron_count_ * std::size_t{3};
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      std::copy_n(positions, device_positions_.size(), device_positions_.begin());
      return;
    }
#endif
    workspace_->setPositions(
        pf::GeometryPositionView::interleaved(positions, electron_count_));
  }

  void requireBoundVersion() const
  {
    if (model_state_->model.p.version() != parameter_version_)
      throw std::runtime_error(
          "PsiFormer determinant-channel batch parameter version is stale");
  }

  std::shared_ptr<PsiFormerSharedState> model_state_;
  std::shared_ptr<const wftrain::StructuredParameterSchema> schema_;
  std::size_t parameter_version_ = 0;
  std::size_t sample_count_ = 0;
  std::size_t electron_count_ = 0;
  std::size_t channel_count_ = 0;
  std::vector<double> positions_;
  mutable std::vector<double> phases_;
  mutable std::vector<double> log_abs_;
  mutable std::unique_ptr<pf::DirectScoreWorkspace> workspace_;
#ifdef ENABLE_CUDA
  std::unique_ptr<psiformer::device::PreparedDeviceProductSession> device_products_;
  mutable std::vector<double> device_positions_;
  mutable bool reported_device_channels_ = false;
#endif
  std::size_t retained_numeric_bytes_ = 0;
};

/** Own an exact accepted ordinary-locality ECP traversal for repeated JVP actions. */
class PsiFormerNonLocalECPReplay final : public wftrain::NonLocalECPJVPReplay
{
public:
  PsiFormerNonLocalECPReplay(
      std::string provider_id,
      std::string schema_fingerprint,
      std::size_t parameter_version,
      std::size_t batch_ordinal,
      std::size_t sample_offset,
      std::size_t sample_count,
      std::size_t electron_count,
      std::uint64_t grid_fingerprint,
      std::vector<double> reference_positions,
      std::vector<wftrain::NonLocalECPReplayPoint> points,
      std::vector<wftrain::NonLocalECPReplaySegment> segments,
      std::vector<wftrain::NonLocalECPReplayTile> tiles,
      std::vector<WaveFunctionComponent::EvaluationStamp> evaluation_stamps)
      : provider_id_(std::move(provider_id)),
        schema_fingerprint_(std::move(schema_fingerprint)),
        parameter_version_(parameter_version),
        batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset),
        sample_count_(sample_count),
        electron_count_(electron_count),
        grid_fingerprint_(grid_fingerprint),
        reference_positions_(std::move(reference_positions)),
        points_(std::move(points)),
        segments_(std::move(segments)),
        tiles_(std::move(tiles)),
        evaluation_stamps_(std::move(evaluation_stamps))
  {
    const std::size_t expected_positions = checkedBatchMemoryMultiply(
        checkedBatchMemoryMultiply(sample_count_, electron_count_,
                                   "PsiFormer ECP replay references"),
        std::size_t{3}, "PsiFormer ECP replay Cartesian references");
    if (provider_id_.empty() || schema_fingerprint_.empty() ||
        grid_fingerprint_ == 0 || reference_positions_.size() != expected_positions)
      throw std::invalid_argument("PsiFormer ECP replay identity or reference extent is invalid");

    for (double coordinate : reference_positions_)
      requireFinite(coordinate, "reference coordinate");
    for (const auto& point : points_)
    {
      if (point.sample >= sample_count_ || point.electron >= electron_count_)
        throw std::out_of_range("PsiFormer ECP replay point identity is out of range");
      for (double coordinate : point.replacement_position)
        requireFinite(coordinate, "replacement coordinate");
      requireFinite(point.bare_weight, "bare weight");
      requireFinite(point.complete_ratio, "complete ratio");
      if (point.bare_weight.imag() != 0.0)
        throw std::invalid_argument("PsiFormer ECP replay bare weight must be real");
    }

    std::size_t expected_point_offset = 0;
    std::size_t expected_segment_offset = 0;
    std::size_t expected_stamp_offset = 0;
    for (std::size_t ordinal = 0; ordinal < tiles_.size(); ++ordinal)
    {
      const auto& tile = tiles_[ordinal];
      if (tile.tile_ordinal != ordinal || tile.point_offset != expected_point_offset ||
          tile.segment_offset != expected_segment_offset ||
          tile.stamp_offset != expected_stamp_offset ||
          tile.point_count > points_.size() - tile.point_offset ||
          tile.segment_count > segments_.size() - tile.segment_offset ||
          tile.stamp_count > evaluation_stamps_.size() - tile.stamp_offset ||
          (tile.point_count != 0 && tile.stamp_count == 0) ||
          tile.virtual_batch_fingerprint == 0)
        throw std::invalid_argument("PsiFormer ECP replay tile provenance is noncanonical");

      std::size_t tile_segment_points = 0;
      for (std::size_t segment_index = tile.segment_offset;
           segment_index < tile.segment_offset + tile.segment_count;
           ++segment_index)
      {
        const auto& segment = segments_[segment_index];
        if (segment.point_offset < tile.point_offset ||
            segment.point_offset > tile.point_offset + tile.point_count ||
            segment.point_count >
                tile.point_offset + tile.point_count - segment.point_offset)
          throw std::invalid_argument(
              "PsiFormer ECP replay segment escapes its source tile");
        tile_segment_points = checkedBatchMemoryAdd(
            tile_segment_points, segment.point_count,
            "PsiFormer ECP replay tile segment coverage");
      }
      if (tile_segment_points != tile.point_count)
        throw std::invalid_argument(
            "PsiFormer ECP replay tile segment coverage is incomplete");
      expected_point_offset += tile.point_count;
      expected_segment_offset += tile.segment_count;
      expected_stamp_offset += tile.stamp_count;
    }
    if (expected_point_offset != points_.size() ||
        expected_segment_offset != segments_.size() ||
        expected_stamp_offset != evaluation_stamps_.size())
      throw std::invalid_argument("PsiFormer ECP replay provenance is incomplete");

    std::size_t expected_segment_point = 0;
    for (const auto& segment : segments_)
    {
      if (segment.sample >= sample_count_ || segment.electron >= electron_count_ ||
          segment.point_offset != expected_segment_point ||
          segment.point_offset > points_.size() ||
          segment.point_count > points_.size() - segment.point_offset)
        throw std::invalid_argument("PsiFormer ECP replay segment is invalid");
      for (std::size_t point = segment.point_offset;
           point < segment.point_offset + segment.point_count; ++point)
        if (points_[point].sample != segment.sample ||
            points_[point].electron != segment.electron)
          throw std::invalid_argument(
              "PsiFormer ECP replay point disagrees with segment provenance");
      expected_segment_point = checkedBatchMemoryAdd(
          expected_segment_point, segment.point_count,
          "PsiFormer ECP replay segment point coverage");
    }
    if (expected_segment_point != points_.size())
      throw std::invalid_argument(
          "PsiFormer ECP replay segments do not cover every point");
    diagnostics_ = measureStorageDiagnostics();
  }

  std::string_view providerId() const noexcept override { return provider_id_; }
  std::string_view schemaFingerprint() const noexcept override { return schema_fingerprint_; }
  std::size_t parameterVersion() const noexcept override { return parameter_version_; }
  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }
  std::size_t sampleOffset() const noexcept override { return sample_offset_; }
  std::size_t sampleCount() const noexcept override { return sample_count_; }
  std::size_t electronCount() const noexcept override { return electron_count_; }
  std::uint32_t localEnergyTermMask() const noexcept override
  {
    return wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
  }
  std::uint64_t gridFingerprint() const noexcept override { return grid_fingerprint_; }
  wftrain::DerivativeArrayView<const double> referencePositions() const noexcept override
  {
    return {reference_positions_.data(), reference_positions_.size()};
  }
  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayPoint>
  points() const noexcept override
  {
    return {points_.data(), points_.size()};
  }
  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplaySegment>
  segments() const noexcept override
  {
    return {segments_.data(), segments_.size()};
  }
  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayTile>
  tiles() const noexcept override
  {
    return {tiles_.data(), tiles_.size()};
  }
  wftrain::DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp>
  evaluationStamps() const noexcept override
  {
    return {evaluation_stamps_.data(), evaluation_stamps_.size()};
  }
  wftrain::NonLocalECPReplayStorageDiagnostics storageDiagnostics() const noexcept override
  {
    return diagnostics_;
  }

private:
  static void requireFinite(double value, const char* description)
  {
    if (!psiformer::determinant::isFiniteReal(value))
      throw std::invalid_argument(std::string("PsiFormer ECP replay ") + description +
                                  " must be finite");
  }

  static void requireFinite(wftrain::DerivativeValue value,
                            const char* description)
  {
    requireFinite(value.real(), description);
    requireFinite(value.imag(), description);
  }

  static void mixFingerprint(std::size_t& fingerprint, std::size_t value) noexcept
  {
    fingerprint ^= value + std::size_t{0x9e3779b9U} + (fingerprint << 6) +
        (fingerprint >> 2);
  }

  wftrain::NonLocalECPReplayStorageDiagnostics measureStorageDiagnostics() const noexcept
  {
    wftrain::NonLocalECPReplayStorageDiagnostics result;
    result.point_count = points_.size();
    result.segment_count = segments_.size();
    result.tile_count = tiles_.size();
    result.evaluation_stamp_count = evaluation_stamps_.size();
    result.reference_position_bytes = reference_positions_.capacity() * sizeof(double);
    result.point_bytes = points_.capacity() * sizeof(wftrain::NonLocalECPReplayPoint);
    result.provenance_bytes =
        segments_.capacity() * sizeof(wftrain::NonLocalECPReplaySegment) +
        tiles_.capacity() * sizeof(wftrain::NonLocalECPReplayTile) +
        evaluation_stamps_.capacity() * sizeof(WaveFunctionComponent::EvaluationStamp);
    result.retained_numeric_bytes = result.reference_position_bytes +
        result.point_bytes + result.provenance_bytes;
    std::size_t fingerprint = std::size_t{1469598103U};
    mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(reference_positions_.data()));
    mixFingerprint(fingerprint, reference_positions_.capacity());
    mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(points_.data()));
    mixFingerprint(fingerprint, points_.capacity());
    mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(segments_.data()));
    mixFingerprint(fingerprint, segments_.capacity());
    mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(tiles_.data()));
    mixFingerprint(fingerprint, tiles_.capacity());
    mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(evaluation_stamps_.data()));
    mixFingerprint(fingerprint, evaluation_stamps_.capacity());
    result.storage_fingerprint = fingerprint == 0 ? 1 : fingerprint;
    return result;
  }

  std::string provider_id_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  std::size_t batch_ordinal_ = 0;
  std::size_t sample_offset_ = 0;
  std::size_t sample_count_ = 0;
  std::size_t electron_count_ = 0;
  std::uint64_t grid_fingerprint_ = 0;
  std::vector<double> reference_positions_;
  std::vector<wftrain::NonLocalECPReplayPoint> points_;
  std::vector<wftrain::NonLocalECPReplaySegment> segments_;
  std::vector<wftrain::NonLocalECPReplayTile> tiles_;
  std::vector<WaveFunctionComponent::EvaluationStamp> evaluation_stamps_;
  wftrain::NonLocalECPReplayStorageDiagnostics diagnostics_;
};

/** Apply bounded score and kinetic products without a sample-by-parameter matrix.
 *
 * Operators retain distinct positions, complete TrialWaveFunction drift and
 * inverse masses, but lease the O(P) score/kinetic tapes and contraction vectors.
 * Storage diagnostics include the shared bytes and an identity so aggregate
 * owners count a lane once rather than once per retained snapshot.
 */
/** Reusable O(P) execution scratch, independent of immutable retained samples.
 * One exclusive lease serializes products sharing this lane; separate models and
 * MPI processes have separate lanes. Scratch is released with its last owner.
 */
struct PsiFormerDerivativeScratch
{
  PsiFormerDerivativeScratch(PsiFormerSharedState& state, std::size_t parameters,
                             std::size_t score_capacity = 1)
      : score(state.numerical_tile_backend ? nullptr : state.direct_score_executor.makeWorkspace()),
        kinetic(state.numerical_tile_backend ? nullptr : state.cpu_target_kinetic.targets
            ? state.direct_kinetic_executor.makeTargetWorkspace(state.cpu_target_kinetic.targets, state.cpu_target_kinetic.owner_budget)
            : state.direct_kinetic_executor.makeWorkspace())
  {
    if (state.cpu_target_kinetic.targets && (!kinetic || kinetic->vectorStorageBytes() != state.cpu_target_kinetic.owner_bytes))
      throw std::logic_error("PsiFormer CPU target kinetic owner differs from admitted numeric bytes");
    checkedBatchMemoryMultiply(parameters, sizeof(wftrain::DerivativeValue),
                               "PsiFormer shared derivative channels");
    bool full_host_channels = true;
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    full_host_channels = !state.numerical_tile_backend;
#endif
    if (full_host_channels)
      for (auto& channel : channels)
        channel.resize(parameters);
#ifdef ENABLE_CUDA
    if (state.numerical_tile_backend)
    {
      const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
          state.numerical_tile_backend.get());
      if (!backend)
        throw std::logic_error("PsiFormer device products require the prepared numerical backend");
      device_products = backend->prepareProductSession(
          psiformer::device::ProductSessionProfile::SCORE_AND_KINETIC, score_capacity);
      device_positions.resize(3 * state.execution_plan.modelShape().electrons());
      device_direction.resize(parameters);
    }
#endif
  }

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  /// Cold numeric preparation under the exclusive lease, before any model read.
  void prepareHostOutputs(std::size_t parameters, std::size_t chunk, bool fallback)
  {
    const auto allowance = checkedBatchMemoryMultiply(checkedBatchMemoryMultiply(parameters,
        std::size_t{3}, "GPU HOST channel allowance"), sizeof(wftrain::DerivativeValue), "GPU HOST channel bytes");
    const auto chunk_bytes = checkedBatchMemoryMultiply(chunk, sizeof(wftrain::DerivativeValue), "GPU HOST output chunk");
    const bool grow = device_ecp_chunk.size() < chunk || (fallback && channels[0].size() < parameters);
    if (grow)
      app_log() << "  PsiFormer GPU HOST outputs before allocation: bounded_chunk_bytes=" << chunk_bytes
                << ", fallback_channel_allowance_bytes=" << allowance << ", materialize_fallback=" << fallback
                << "; numeric HOST only; model/vendor/metadata excluded" << std::endl;
    if (device_ecp_chunk.size() < chunk) device_ecp_chunk.resize(chunk);
    if (fallback)
      for (auto& channel : channels)
        if (channel.size() < parameters) channel.resize(parameters);
  }
#endif

  std::recursive_mutex execution_mutex;
  bool active = false;
  std::unique_ptr<pf::DirectScoreWorkspace> score;
  std::unique_ptr<pf::DirectKineticWorkspace> kinetic;
  bool reported_cpu_target_products = false;
#ifdef ENABLE_CUDA
  std::unique_ptr<psiformer::device::PreparedDeviceProductSession> device_products;
  std::vector<double> device_positions;
  std::vector<double> device_direction;
  std::vector<wftrain::DerivativeValue> device_ecp_chunk;
  std::vector<double> device_ecp_positions, device_ecp_weights;
  bool reported_device_products = false;
  bool reported_device_score_contraction = false;
  bool reported_device_score_tiles = false;
  bool reported_device_jvp_tiles = false;
#endif
  std::array<std::vector<wftrain::DerivativeValue>, 3> channels;
};

/** Serialize competing calls, but reject re-entry from a sink callback.
 * Recursive acquisition must fail rather than deadlock or overwrite the outer
 * operation's still-live channels/sample products.
 */
class PsiFormerDerivativeLease
{
public:
  explicit PsiFormerDerivativeLease(PsiFormerDerivativeScratch& scratch)
      : lock_(scratch.execution_mutex), scratch_(scratch)
  {
    if (scratch_.active)
      throw std::logic_error("PsiFormer shared derivative scratch cannot be re-entered");
    scratch_.active = true;
  }

  ~PsiFormerDerivativeLease() { scratch_.active = false; }

private:
  std::unique_lock<std::recursive_mutex> lock_;
  PsiFormerDerivativeScratch& scratch_;
};

/** Obtain the prepared lane without retaining scratch permanently on the model. */
std::shared_ptr<PsiFormerDerivativeScratch> acquireDerivativeScratch(
    PsiFormerSharedState& state, std::size_t parameters, std::size_t score_capacity = 1)
{
  std::lock_guard factory_lock(state.derivative_scratch_factory_mutex);
  auto scratch = state.derivative_scratch.lock();
  if (!scratch)
  {
    scratch = std::make_shared<PsiFormerDerivativeScratch>(state, parameters, score_capacity);
    state.derivative_scratch = scratch;
  }
  return scratch;
}

/** Bind immutable samples to one model version and exclusively leased scratch. */
class PsiFormerStreamingDerivativeOperator final : public wftrain::StreamingDerivativeOperator
{
  friend class PsiFormerWF;
  psiformer::PsiFormerTimers phase_timers_;
public:
  static constexpr std::size_t MAXIMUM_VJP_CHANNELS = 3;

  /** Bind an immutable schema/version and take ownership of the ordered positions.
   * Checked byte arithmetic is completed before any parameter-channel allocation.
   */
  PsiFormerStreamingDerivativeOperator(
      std::shared_ptr<PsiFormerSharedState> model_state,
      std::shared_ptr<const wftrain::StructuredParameterSchema> schema,
      std::size_t parameter_version,
      std::size_t batch_ordinal,
      std::size_t sample_offset,
      std::size_t sample_count,
      std::vector<double> positions,
      std::vector<double> total_log_gradients_real,
      std::vector<double> total_log_gradients_imaginary,
      std::vector<double> inverse_masses,
      std::size_t maximum_parameter_chunk_size)
      : model_state_(std::move(model_state)),
        schema_(std::move(schema)),
        parameter_version_(parameter_version),
        batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset),
        sample_count_(sample_count),
        electron_count_(model_state_->execution_plan.modelShape().electrons()),
        positions_(std::move(positions)),
        total_log_gradients_real_(std::move(total_log_gradients_real)),
        total_log_gradients_imaginary_(std::move(total_log_gradients_imaginary)),
        inverse_masses_(std::move(inverse_masses)),
        chunk_plan_(*schema_, parameter_version_, maximum_parameter_chunk_size),
        // A retained pretraining capacity1 lane remains serial; never duplicate
        // its P-sized resources or grow it implicitly during an evaluation.
        shared_scratch_(acquireDerivativeScratch(*model_state_, schema_->parameterCount(), 4)),
        score_workspace_(shared_scratch_->score),
        kinetic_workspace_(shared_scratch_->kinetic),
        channel_accumulators_(shared_scratch_->channels),
        sample_products_(makeSampleProductVector(sample_count_)),
        reference_score_products_(makeSampleProductVector(sample_count_)),
        replay_weight_sums_(makeSampleProductVector(sample_count_))
  {
    const std::size_t expected_positions = checkedPositionCount(sample_count_, electron_count_);
    if (positions_.size() != expected_positions)
      throw std::invalid_argument("PsiFormer streaming derivative received the wrong position extent");
    if (total_log_gradients_real_.size() != expected_positions ||
        total_log_gradients_imaginary_.size() != expected_positions)
      throw std::invalid_argument(
          "PsiFormer streaming derivative received the wrong total-drift extent");
    const std::size_t expected_masses = checkedMassCount(sample_count_, electron_count_);
    if (inverse_masses_.size() != expected_masses)
      throw std::invalid_argument(
          "PsiFormer streaming derivative received the wrong inverse-mass extent");
    if (schema_->parameterCount() != model_state_->execution_plan.parameterCount() ||
        (score_workspace_ && score_workspace_->scoreSize() != schema_->parameterCount()) ||
        (kinetic_workspace_ && (kinetic_workspace_->scoreSize() != schema_->parameterCount() ||
                               kinetic_workspace_->kineticSize() != schema_->parameterCount())))
      throw std::invalid_argument("PsiFormer streaming derivative schema does not match the execution plan");
    checkedBatchMemoryAdd(sample_offset_, sample_count_,
                          "PsiFormer streaming derivative sample interval");
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    std::optional<PsiFormerDerivativeLease> preparation_lease;
    std::optional<PsiFormerReadTransaction> preparation_read;
    if (hasDeviceProducts())
    {
      preparation_lease.emplace(*shared_scratch_);
      shared_scratch_->prepareHostOutputs(schema_->parameterCount(), chunk_plan_.maximumChunkSize(),
          sample_count_ <= 2 * MAXIMUM_VJP_CHANNELS);
      preparation_read.emplace(*model_state_, model_state_->resident_device_authority_enabled);
      requireBoundVersion();
    }
#endif
#ifdef ENABLE_CUDA
    // Immutable samples belong to this operator, not its shared derivative lane.
    // Empty operators need no allocation/upload. Enabled admission failures throw;
    // they must not silently return to coordinate host uploads.
    if (hasDeviceProducts() && shared_scratch_->device_products->retainedCoordinateBudget() && sample_count_ != 0)
    {
      std::lock_guard scratch_lock(shared_scratch_->execution_mutex);
      auto& session = *shared_scratch_->device_products;
      const auto before = session.transferSnapshot();
      retained_coordinates_ = session.prepareSampleCoordinates(positions_, parameter_version_);
      const auto after = session.transferSnapshot();
      app_log() << "  PsiFormer retained device sample coordinates: configurations=" << sample_count_
                << ", host_position_bytes=" << checkedBatchMemoryMultiply(positions_.capacity(), sizeof(double),
                    "PsiFormer retained host coordinate bytes")
                << ", device_position_bytes=" << retained_coordinates_->deviceWorkspaceBytes()
                << ", per_provider_limit_bytes=" << session.retainedCoordinateBudget()
                << ", measured_upload_calls=" << after.upload_calls - before.upload_calls
                << ", measured_uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
                << ", transfer_counter_scope=device_wide, aggregate_process_budget=no" << std::endl;
    }
#endif
    initializeStorageDiagnostics();
  }

  /// Advertise the conservative full-channel fallback O(P) bound.
  wftrain::StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    wftrain::StreamingDerivativeCapabilities result;
    result.product_mask =
        wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_VJP) |
        wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_JVP) |
        wftrain::derivativeProductBit(
            wftrain::DerivativeProduct::LOCAL_ENERGY_JVP) |
        wftrain::derivativeProductBit(wftrain::DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask =
        wftrain::derivativeAdjointBit(wftrain::DerivativeAdjoint::TRANSPOSE) |
        wftrain::derivativeAdjointBit(wftrain::DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = wftrain::ParameterScalarDomain::REAL64;
    result.result_scalar_domain    = wftrain::ParameterScalarDomain::COMPLEX128;
    result.reduction_domain        = wftrain::ReductionDomain::CROWD_LOCAL;
    // This field describes sink-visible storage, not the kernel device. The
    // Numerical products emit host complex chunks after their device contractions.
    result.execution_domain        = wftrain::DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask =
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
    if (ecp_replay_)
      result.local_energy_term_mask |= ecp_replay_->localEnergyTermMask();
    result.maximum_vjp_channels              = MAXIMUM_VJP_CHANNELS;
    result.maximum_parameter_chunk_size      = chunk_plan_.maximumChunkSize();
    result.maximum_sample_tile_size          = deviceScoreCapacity();
    result.fixed_parameter_scratch_vectors   =
        (hasDeviceProducts() ? 8 + 2 * deviceScoreContractionCapacity() : 3) + MAXIMUM_VJP_CHANNELS;
    result.block_streaming                   = true;
    return result;
  }

  /// Return the immutable component schema retained by shared ownership.
  const wftrain::StructuredParameterSchema& parameterSchema() const noexcept override
  {
    return *schema_;
  }

  /// Return the model version captured at factory time.
  std::size_t parameterVersion() const noexcept override { return parameter_version_; }

  /// Return the caller-supplied ordered batch identity.
  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }

  /// Return the caller-supplied first sample offset.
  std::size_t sampleOffset() const noexcept override { return sample_offset_; }

  /// Return the number of copied sample configurations.
  std::size_t sampleCount() const noexcept override { return sample_count_; }

  /// Return the canonical trainable-parameter chunk partition.
  const wftrain::ParameterChunkPlan& parameterChunkPlan() const noexcept override
  {
    return chunk_plan_;
  }

  /// Return exact prepared numeric bytes and allocation-identity evidence.
  wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    // Optional contraction preparation can grow shared scratch on another
    // retained provider. Snapshot its bytes/capacity under the same outer lease.
    std::lock_guard scratch_lock(shared_scratch_->execution_mutex);
    wftrain::StreamingDerivativeStorageDiagnostics current =
        measureStorageDiagnostics();
    if (storage_diagnostics_.storage_fingerprint != 0 &&
        current.storage_fingerprint !=
            storage_diagnostics_.storage_fingerprint)
      current.allocation_generation = std::max(current.allocation_generation, checkedBatchMemoryAdd(
          storage_diagnostics_.allocation_generation, std::size_t{1},
          "PsiFormer streaming allocation generation"));
    return current;
  }

  /// Bind one exact accepted-grid ECP replay before derivative products are used.
  void bindNonLocalECPJVPReplay(
      std::unique_ptr<wftrain::NonLocalECPJVPReplay> replay) override
  {
    if (!replay)
      throw std::invalid_argument("PsiFormer cannot bind a null ECP JVP replay");
    if (ecp_replay_)
      throw std::logic_error("PsiFormer ECP JVP replay is already bound");
    if (has_evaluated_.load(std::memory_order_acquire))
      throw std::logic_error(
          "PsiFormer ECP JVP replay must be bound before the first derivative action");
    if (replay->providerId() != schema_->providerId() ||
        replay->schemaFingerprint() != schema_->fingerprint() ||
        replay->parameterVersion() != parameter_version_ ||
        replay->batchOrdinal() != batch_ordinal_ ||
        replay->sampleOffset() != sample_offset_ ||
        replay->sampleCount() != sample_count_ ||
        replay->electronCount() != electron_count_ ||
        replay->localEnergyTermMask() !=
            wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP) ||
        replay->gridFingerprint() == 0)
      throw std::invalid_argument("PsiFormer ECP JVP replay identity is incompatible");
    const auto references = replay->referencePositions();
    if (references.size() != positions_.size() ||
        !std::equal(references.begin(), references.end(), positions_.begin()))
      throw std::invalid_argument(
          "PsiFormer ECP JVP replay references do not match the derivative batch");
    const auto replay_points = replay->points();
    const auto replay_segments = replay->segments();
    const auto replay_tiles = replay->tiles();
    const auto replay_stamps = replay->evaluationStamps();
    const auto diagnostics = replay->storageDiagnostics();
    if (diagnostics.point_count != replay_points.size() ||
        diagnostics.segment_count != replay_segments.size() ||
        diagnostics.tile_count != replay_tiles.size() ||
        diagnostics.evaluation_stamp_count != replay_stamps.size() ||
        diagnostics.storage_fingerprint == 0)
      throw std::invalid_argument(
          "PsiFormer ECP JVP replay diagnostics disagree with its retained traversal");
    const auto current_storage = measureStorageDiagnostics();
    checkedBatchMemoryAdd(
        current_storage.retained_numeric_bytes,
        diagnostics.retained_numeric_bytes,
        "PsiFormer streaming retained storage with accepted-grid replay");
    for (const auto& tile : replay_tiles)
    {
      if (tile.stamp_offset > replay_stamps.size() ||
          tile.stamp_count > replay_stamps.size() - tile.stamp_offset)
        throw std::invalid_argument(
            "PsiFormer ECP JVP replay has an invalid stamp range");
      bool matching_stamp = false;
      for (std::size_t stamp = tile.stamp_offset;
           stamp < tile.stamp_offset + tile.stamp_count; ++stamp)
        matching_stamp = matching_stamp || replay_stamps[stamp].matches(
            model_state_.get(), static_cast<std::uint64_t>(parameter_version_));
      if (!matching_stamp)
        throw std::invalid_argument(
            "PsiFormer ECP JVP replay lacks bound value-evaluation provenance");
    }
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    if (hasDeviceProducts())
    {
      PsiFormerDerivativeLease scratch_lease(*shared_scratch_);
      // Large resident HOST-C3 may append before its single finish. Other
      // requests materialize the admitted fallback before their model read.
      shared_scratch_->prepareHostOutputs(schema_->parameterCount(), chunk_plan_.maximumChunkSize(),
          sample_count_ <= 2 * MAXIMUM_VJP_CHANNELS);
      PsiFormerReadTransaction state_lock(*model_state_, model_state_->deviceOnlyValueReads());
      requireBoundVersion();
      const auto bytes = psiformer::device::PreparedDeviceScoreContraction::requiredBytes(
          schema_->parameterCount(), MAXIMUM_VJP_CHANNELS);
      const auto chunk = std::min(schema_->parameterCount(), chunk_plan_.maximumChunkSize());
      const auto capacity = replay->points().empty() || deviceScoreCapacity() == 1 ? 0 : deviceScoreCapacity();
      const auto coordinates = checkedBatchMemoryMultiply(capacity,
          checkedPositionCount(1, electron_count_), "GPU ECP coordinate tile");
      const auto tile_bytes = checkedBatchMemoryMultiply(
          checkedBatchMemoryAdd(coordinates, capacity, "GPU ECP HOST tile elements"),
          sizeof(double), "GPU ECP HOST tile bytes");
      app_log() << "  PsiFormer regular-node real GPU ECP replay before allocation: DEVICE_channels="
                << MAXIMUM_VJP_CHANNELS << ", DEVICE_bytes=" << bytes
                << ", HOST_chunk_bytes=" << chunk * sizeof(wftrain::DerivativeValue)
                << ", HOST_virtual_tile_bytes=" << tile_bytes << ", virtual_tile_capacity=" << capacity
                << "; native aggregate DEVICE admission applies; metadata/vendor excluded" << std::endl;
      shared_scratch_->device_products->prepareScoreContractions(MAXIMUM_VJP_CHANNELS, parameter_version_, bytes);
      if (shared_scratch_->device_ecp_chunk.size() < chunk)
        shared_scratch_->device_ecp_chunk.resize(chunk);
      if (shared_scratch_->device_ecp_positions.size() < coordinates)
        shared_scratch_->device_ecp_positions.resize(coordinates);
      if (shared_scratch_->device_ecp_weights.size() < capacity)
        shared_scratch_->device_ecp_weights.resize(capacity);
    }
#endif
    ecp_replay_ = std::move(replay);
    storage_diagnostics_ = measureStorageDiagnostics();
  }

protected:
  /// Accumulate fused score/kinetic VJPs, then emit canonical bounded chunks.
  void evaluateVJPs(
      wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
      wftrain::DerivativeAdjoint adjoint,
      wftrain::ParameterReductionSink& sink) const override
  {
    evaluateVJPsImpl(channels, adjoint, &sink);
  }

  /// A terminal consumer shares physical arithmetic but bypasses host P emission.
  void evaluateVJPsImpl(wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
      wftrain::DerivativeAdjoint adjoint, wftrain::ParameterReductionSink* sink,
      const std::function<void(std::size_t)>& terminal = {}, bool device_authority_terminal = false) const
  {
    NativeTimerScope timer(phase_timers_.parameter_vjp);
    PsiFormerDerivativeLease scratch_lease(*shared_scratch_);
    has_evaluated_.store(true, std::memory_order_release);
    bool append_ecp_host = false;
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    std::optional<psiformer::device::PreparedDeviceProductSession::CompletedScoreContraction> completed;
    std::array<std::size_t, MAXIMUM_VJP_CHANNELS> completed_slots{};
    completed_slots.fill(MAXIMUM_VJP_CHANNELS);
    if (hasDeviceProducts() && !terminal)
    {
      const auto complete_energy = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) |
          wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
      append_ecp_host = ecp_replay_ && sample_count_ > 2 * MAXIMUM_VJP_CHANNELS && channels.size() == 3 &&
          model_state_->precision == psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE &&
          model_state_->execution_plan.environment().boundary == psiformer::BoundaryCondition::OPEN &&
          model_state_->execution_plan.environment().amplitude_scalar_domain == psiformer::ScalarDomain::REAL &&
          channels[0].product == wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
          channels[0].local_energy_term_mask == complete_energy &&
          channels[1].product == wftrain::DerivativeProduct::SCORE_VJP &&
          channels[2].product == wftrain::DerivativeProduct::SCORE_VJP;
      const bool requests_ecp = std::any_of(channels.begin(), channels.end(), [](const auto& channel) {
        return channel.local_energy_term_mask & wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
      });
      const bool has_contracted_response = std::any_of(channels.begin(), channels.end(), [](const auto& channel) {
        return channel.product == wftrain::DerivativeProduct::SCORE_VJP ||
            (channel.local_energy_term_mask & wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC));
      });
      shared_scratch_->prepareHostOutputs(schema_->parameterCount(), chunk_plan_.maximumChunkSize(),
          sample_count_ <= 2 * MAXIMUM_VJP_CHANNELS || (requests_ecp && !append_ecp_host) || !has_contracted_response);
    }
#endif
    {
      PsiFormerReadTransaction state_lock(*model_state_, device_authority_terminal && model_state_->resident_device_authority_enabled);
      requireBoundVersion();
      if (!terminal)
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
          std::fill(channel_accumulators_[channel].begin(),
                    channel_accumulators_[channel].end(),
                    wftrain::DerivativeValue{});

      const bool needs_kinetic = std::any_of(
          channels.begin(), channels.end(), [](const auto& channel) {
            return channel.product ==
                    wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
                (channel.local_energy_term_mask & wftrain::localEnergyTermBit(
                    wftrain::LocalEnergyTerm::KINETIC)) != 0;
          });
      const bool needs_ecp = std::any_of(
          channels.begin(), channels.end(), [](const auto& channel) {
            return channel.product ==
                    wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
                (channel.local_energy_term_mask & wftrain::localEnergyTermBit(
                    wftrain::LocalEnergyTerm::NONLOCAL_ECP)) != 0;
          });

#ifdef ENABLE_CUDA
      std::array<std::size_t, MAXIMUM_VJP_CHANNELS> score_channels{};
      std::size_t score_channel_count = 0;
      std::array<std::size_t, MAXIMUM_VJP_CHANNELS> device_channels{};
      std::size_t device_channel_count = 0;
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        if (channels[channel].product == wftrain::DerivativeProduct::SCORE_VJP)
          score_channels[score_channel_count++] = channel;
        if (channels[channel].product == wftrain::DerivativeProduct::SCORE_VJP ||
            (channels[channel].product == wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
             (channels[channel].local_energy_term_mask & wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC)) != 0))
          device_channels[device_channel_count++] = channel;
      }
      // Real per-sample downloads cost 8*B*P; complex contracted outputs cost
      // 16*C*P. Admit optional storage only above that transfer break-even.
      const bool resident_scores = hasDeviceProducts() && !needs_kinetic &&
          score_channel_count != 0 && sample_count_ > 2 * score_channel_count;
      const bool resident_kinetic = hasDeviceProducts() && needs_kinetic &&
          device_channel_count != 0 && (sample_count_ > 2 * device_channel_count ||
              (device_authority_terminal && terminal && sample_count_ > 0 && model_state_->resident_device_authority_enabled));
      const bool resident_products = resident_scores || resident_kinetic;
      // Mixed kinetic/ECP requests retain their existing host coordinate boundary.
      const bool retained_score_input = retained_coordinates_ && !needs_kinetic && !needs_ecp;
      const auto& contracted_channels = resident_kinetic ? device_channels : score_channels;
      const auto contracted_count = resident_kinetic ? device_channel_count : score_channel_count;
      if (terminal && !resident_kinetic)
        throw std::invalid_argument("Resident ENERGY requires the qualified fused kinetic traversal");
      std::size_t contraction_generation = 0;
      psiformer::device::DeviceMemorySnapshot transfer_before;
      // The existing weighted reverse contracts one real channel. Exact API
      // coefficients determine eligibility; real parameters do not imply real
      // optimizer coefficients. Finite coefficients were admitted by applyVJPs.
      const bool batched_scores = resident_scores && channels.size() == 1 &&
          !needs_ecp && deviceScoreCapacity() > 1 &&
          std::all_of(channels[0].coefficients.values.begin(), channels[0].coefficients.values.end(),
                      [](const auto& coefficient) { return coefficient.imag() == 0.0; });
      std::size_t completed_score_tiles = 0;
      std::size_t final_score_tile_count = 0;
      std::size_t weighted_configurations = 0;
      if (resident_products)
      {
        auto& session = *shared_scratch_->device_products;
        transfer_before = session.transferSnapshot();
        // Explicit cold preparation may grow capacity. Repeated requests with
        // the same admitted channel count reuse it without allocation.
        session.prepareScoreContractions(contracted_count, parameter_version_);
        contraction_generation = session.beginScoreContraction(contracted_count, parameter_version_,
            terminal ? psiformer::device::ScoreContractionRole::ENERGY_T_EO_O
                     : psiformer::device::ScoreContractionRole::GENERIC);
      }
#endif

#ifdef ENABLE_CUDA
      if (batched_scores)
      {
        std::array<double, 4> weights{};
        const auto capacity = deviceScoreCapacity();
        const auto before_tiles = shared_scratch_->device_products->weightedScoreTiles();
        const auto before_configurations = shared_scratch_->device_products->weightedScoreConfigurations();
        for (std::size_t offset = 0; offset < sample_count_; offset += capacity)
        {
          const auto count = std::min(capacity, sample_count_ - offset);
          for (std::size_t sample = 0; sample < count; ++sample)
            weights[sample] = channels[0].coefficients.values[offset + sample].real();
          auto& session = *shared_scratch_->device_products;
          if (retained_score_input)
            session.accumulateWeightedScoreTileDeviceCoordinates(retained_coordinates_->view(offset, count),
                weights.data(), parameter_version_);
          else
            session.accumulateWeightedScoreTile(positions_.data() + offset * 3 * electron_count_,
                weights.data(), count, parameter_version_);
          final_score_tile_count = count;
        }
        completed_score_tiles = shared_scratch_->device_products->weightedScoreTiles() - before_tiles;
        weighted_configurations = shared_scratch_->device_products->weightedScoreConfigurations() - before_configurations;
        reportDeviceProducts();
      }
      else
#endif
      for (std::size_t sample = 0; sample < sample_count_; ++sample)
      {
        if (needs_kinetic)
        {
          setKineticWorkspacePositions(sample);
          const double* total_drift = total_log_gradients_real_.data() +
              sample * electron_count_ * std::size_t{3};
          const double* imaginary_drift =
              total_log_gradients_imaginary_.data() +
              sample * electron_count_ * std::size_t{3};
          const double* inverse_mass = inverse_masses_.data() +
              sample * electron_count_;
#ifdef ENABLE_CUDA
          if (resident_kinetic)
          {
            std::array<wftrain::DerivativeValue, MAXIMUM_VJP_CHANNELS> score_coefficients{},
                kinetic_coefficients{}, imaginary_coefficients{};
            const wftrain::DerivativeValue imaginary_unit = adjoint == wftrain::DerivativeAdjoint::HERMITIAN
                ? wftrain::DerivativeValue(0.0, -1.0) : wftrain::DerivativeValue(0.0, 1.0);
            for (std::size_t slot = 0; slot < contracted_count; ++slot)
            {
              const auto& channel = channels[contracted_channels[slot]];
              const auto coefficient = channel.coefficients.values[sample];
              if (channel.product == wftrain::DerivativeProduct::SCORE_VJP)
                score_coefficients[slot] = coefficient;
              else
              {
                kinetic_coefficients[slot] = coefficient;
                imaginary_coefficients[slot] = coefficient * imaginary_unit;
              }
            }
            shared_scratch_->device_products->accumulateKineticContraction(
                shared_scratch_->device_positions.data(), total_drift, imaginary_drift, inverse_mass,
                score_coefficients.data(), kinetic_coefficients.data(), imaginary_coefficients.data(),
                contracted_count, parameter_version_);
            reportDeviceProducts();
            continue;
          }
#endif
          const pf::DirectKineticResultView result = evaluateKineticResponse(total_drift, inverse_mass);
          requireResultVersion(result);
          for (std::size_t channel = 0; channel < channels.size(); ++channel)
          {
            if (channels[channel].product ==
                wftrain::DerivativeProduct::SCORE_VJP)
              accumulateResponse(channel,
                                 channels[channel].coefficients.values[sample],
                                 result.parameter_score);
            else if ((channels[channel].local_energy_term_mask &
                      wftrain::localEnergyTermBit(
                          wftrain::LocalEnergyTerm::KINETIC)) != 0)
              accumulateResponse(channel,
                                 channels[channel].coefficients.values[sample],
                                 result.kinetic_parameter_response);
          }
          const bool has_imaginary_drift = std::any_of(
              imaginary_drift,
              imaginary_drift + electron_count_ * std::size_t{3},
              [](double value) { return value != 0.0; });
          if (has_imaginary_drift)
          {
            const pf::DirectKineticConstView imaginary_response =
                evaluateGradientResponse(imaginary_drift, inverse_mass);
            const wftrain::DerivativeValue imaginary_unit =
                adjoint == wftrain::DerivativeAdjoint::HERMITIAN
                ? wftrain::DerivativeValue(0.0, -1.0)
                : wftrain::DerivativeValue(0.0, 1.0);
            for (std::size_t channel = 0; channel < channels.size(); ++channel)
              if (channels[channel].product ==
                      wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
                  (channels[channel].local_energy_term_mask &
                   wftrain::localEnergyTermBit(
                       wftrain::LocalEnergyTerm::KINETIC)) != 0)
                accumulateResponse(
                    channel,
                    channels[channel].coefficients.values[sample] *
                        imaginary_unit,
                    imaginary_response);
          }
        }
        else
        {
#ifdef ENABLE_CUDA
          if (!retained_score_input)
#endif
            setScoreWorkspacePositions(sample);
#ifdef ENABLE_CUDA
          if (resident_scores)
          {
            std::array<wftrain::DerivativeValue, MAXIMUM_VJP_CHANNELS> coefficients{};
            for (std::size_t score_channel = 0; score_channel < score_channel_count; ++score_channel)
              coefficients[score_channel] = channels[score_channels[score_channel]].coefficients.values[sample];
            auto& session = *shared_scratch_->device_products;
            if (retained_score_input)
              session.accumulateScoreContractionDeviceCoordinates(retained_coordinates_->view(sample, 1),
                  coefficients.data(), score_channel_count, parameter_version_);
            else
              session.accumulateScoreContraction(shared_scratch_->device_positions.data(), coefficients.data(),
                  score_channel_count, parameter_version_);
            reportDeviceProducts();
          }
          else
#endif
          {
            const pf::DirectScoreResult score =
#ifdef ENABLE_CUDA
                retained_score_input ? evaluateRetainedScoreResponse(sample) :
#endif
                evaluateScoreResponse();
            requireResultVersion(score);
            for (std::size_t channel = 0; channel < channels.size(); ++channel)
              if (channels[channel].product ==
                  wftrain::DerivativeProduct::SCORE_VJP)
                accumulateResponse(
                    channel, channels[channel].coefficients.values[sample],
                    score.parameter_score);
          }
        }
      }
#ifdef ENABLE_CUDA
      if (resident_products)
      {
        auto& session = *shared_scratch_->device_products;
        // Append physical ECP to the tagged ENERGY sum before its single finite admission.
        // Native DEVICE-authority ECP admission remains independently controlled by the driver.
        try
        {
          // Ordered HOST reuse keeps the existing GENERIC transaction; only
          // the independently admitted driver terminal uses the ENERGY role.
          if ((terminal || append_ecp_host) && needs_ecp)
            accumulateECPReplayVJPs(channels, true);
          session.finishScoreContraction(parameter_version_);
        }
        catch (...)
        {
          session.abortScoreContraction();
          throw;
        }
        if (terminal)
        {
          terminal(contraction_generation);
          return; // RAII releases BOTH source leases before caller publication.
        }
#if !defined(QMC_COMPLEX)
        if ((!needs_ecp || append_ecp_host) && sample_count_ > 2 * MAXIMUM_VJP_CHANNELS)
        {
          completed = session.completedScoreContraction(parameter_version_);
          for (std::size_t slot = 0; slot < contracted_count; ++slot)
            completed_slots[contracted_channels[slot]] = slot;
          app_log() << "  PsiFormer resident HOST-VJP output: channels=" << contracted_count
                    << ", parameters=" << schema_->parameterCount() << ", chunk_capacity="
                    << shared_scratch_->device_ecp_chunk.capacity() << ", retained_full_HOST_channel_bytes="
                    << (channel_accumulators_[0].capacity() + channel_accumulators_[1].capacity() +
                        channel_accumulators_[2].capacity()) * sizeof(wftrain::DerivativeValue)
                    << ", complete_ECP_appended=" << append_ecp_host
                    << "; completed generation pinned through sink callbacks" << std::endl;
        }
        else
#endif
        for (std::size_t slot = 0; slot < contracted_count; ++slot)
        {
          auto& destination = channel_accumulators_[contracted_channels[slot]];
          session.downloadScoreContraction(slot, destination.data(), destination.size(), parameter_version_);
        }
        if (resident_kinetic)
        {
          const auto transfer_after = session.transferSnapshot();
          app_log() << "  PsiFormer resident score/kinetic contraction completed: configurations=" << sample_count_
                    << ", channels=" << contracted_count << ", canonical_parameters=" << schema_->parameterCount()
                    << ", parameter_version=" << parameter_version_
                    << ", measured_download_calls=" << transfer_after.download_calls - transfer_before.download_calls
                    << ", measured_downloaded_bytes=" << transfer_after.downloaded_bytes - transfer_before.downloaded_bytes
                    << ", status_transfers_included=yes, transfer_counter_scope=device_wide" << std::endl;
        }
        if (resident_scores && !shared_scratch_->reported_device_score_contraction)
        {
          app_log() << "  PsiFormer resident score contraction completed: configurations=" << sample_count_
                    << ", channels=" << score_channel_count << ", canonical_parameters=" << schema_->parameterCount()
                    << ", parameter_version=" << parameter_version_ << std::endl;
          shared_scratch_->reported_device_score_contraction = true;
        }
        if (batched_scores && !shared_scratch_->reported_device_score_tiles)
        {
          app_log() << "  PsiFormer batched score contraction completed: configurations=" << sample_count_
                    << ", prepared_capacity=" << deviceScoreCapacity() << ", weighted_tiles=" << completed_score_tiles
                    << ", weighted_configurations=" << weighted_configurations
                    << ", final_tile_count=" << final_score_tile_count << ", channels=1, parameter_version="
                    << parameter_version_ << std::endl;
          shared_scratch_->reported_device_score_tiles = true;
        }
      }
#endif
      if (needs_ecp && !append_ecp_host)
        accumulateECPReplayVJPs(channels);
    }

    // Never invoke caller-controlled sink code while holding the model lock.
    try
    {
      for (const wftrain::ParameterChunkDescriptor& chunk : chunk_plan_.chunks())
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
          if (completed)
          {
            auto& output = shared_scratch_->device_ecp_chunk;
            if (completed_slots[channel] == MAXIMUM_VJP_CHANNELS)
              std::fill_n(output.data(), chunk.count, wftrain::DerivativeValue{});
            else
              shared_scratch_->device_products->downloadScoreContractionChunk(*completed,
                  completed_slots[channel], chunk.parameter_offset, output.data(), chunk.count);
            sink->add(channel, {chunk, {output.data(), chunk.count}});
            continue;
          }
#endif
          const auto* values = channel_accumulators_[channel].data() + chunk.parameter_offset;
          sink->add(channel, {chunk, {values, chunk.count}});
        }
    }
    catch (...)
    {
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
      if (completed) shared_scratch_->device_products->abortScoreContraction();
#endif
      throw;
    }
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    if (completed) shared_scratch_->device_products->abortScoreContraction();
#endif
  }

  /// Contract scores with one direction; device execution retains the uploaded direction.
  void evaluateScoreJVP(const wftrain::StructuredParameterVectorConstView& direction,
                        wftrain::SampleProductSink& sink) const override
  {
    NativeTimerScope timer(phase_timers_.score_jvp);
    PsiFormerDerivativeLease scratch_lease(*shared_scratch_);
    has_evaluated_.store(true, std::memory_order_release);
    {
      PsiFormerReadTransaction state_lock(*model_state_);
      requireBoundVersion();
#ifdef ENABLE_CUDA
      if (hasDeviceProducts())
      {
        // Upload each real direction once for the whole retained population.
        // Numerical staging separates real/imaginary components; the public
        // REAL64 facade currently admits real tangent directions only.
        std::fill(sample_products_.begin(), sample_products_.end(), wftrain::DerivativeValue{});
        const auto capacity = deviceScoreCapacity();
        std::size_t completed_tiles = 0, components = 0, final_count = 0;
        for (unsigned component = 0; component < 2; ++component)
        {
          if (!prepareDeviceDirection(direction, component))
            continue;
          ++components;
          const wftrain::DerivativeValue factor = component == 0
              ? wftrain::DerivativeValue(1.0, 0.0) : wftrain::DerivativeValue(0.0, 1.0);
          for (std::size_t offset = 0; offset < sample_count_; offset += capacity)
          {
            const auto count = std::min(capacity, sample_count_ - offset);
            auto& session = *shared_scratch_->device_products;
            const auto& products = retained_coordinates_
                ? session.scoreJvpPreparedTileDeviceCoordinates(retained_coordinates_->view(offset, count), parameter_version_)
                : session.scoreJvpPreparedTile(positions_.data() + offset * 3 * electron_count_, count, parameter_version_);
            for (std::size_t sample = 0; sample < count; ++sample)
              sample_products_[offset + sample] += factor * products[sample];
            ++completed_tiles;
            final_count = count;
          }
        }
        if (completed_tiles != 0)
          reportDeviceProducts();
        if (completed_tiles != 0 && !shared_scratch_->reported_device_jvp_tiles)
        {
          app_log() << "  PsiFormer device score JVP tiles: configurations=" << sample_count_
                    << ", prepared_capacity=" << capacity << ", completed_tiles=" << completed_tiles
                    << ", final_tile_count=" << final_count << ", direction_uploads=" << components
                    << ", retained_coordinate_input=" << (retained_coordinates_ ? "device" : "host")
                    << ", parameter_version=" << parameter_version_ << ", CPU derivative fallback=no" << std::endl;
          shared_scratch_->reported_device_jvp_tiles = true;
        }
      }
      else
#endif
      {
        for (std::size_t sample = 0; sample < sample_count_; ++sample)
        {
          setScoreWorkspacePositions(sample);
          const pf::DirectScoreResult score = evaluateScoreResponse();
          requireResultVersion(score);

          wftrain::DerivativeValue& product = sample_products_[sample];
          product                           = {};
          for (std::size_t parameter = 0;
               parameter < score.parameter_score.size; ++parameter)
            product += score.parameter_score[parameter] *
                direction.values()[parameter];
        }
      }
    }

    // Emit only after the complete batch has been computed and the model unlocked.
    for (std::size_t sample = 0; sample < sample_count_; ++sample)
    {
      const wftrain::SampleProductTileDescriptor descriptor{
          schema_->providerId(), schema_->fingerprint(), parameter_version_,
          batch_ordinal_, sample_offset_ + sample, 1, sample};
      sink.add({descriptor, {sample_products_.data() + sample, 1}});
    }
  }

  /// Contract exact kinetic and accepted-grid ECP responses with one direction.
  void evaluateLocalEnergyJVP(
      const wftrain::StructuredParameterVectorConstView& direction,
      std::uint32_t local_energy_term_mask,
      wftrain::SampleProductSink& sink) const override
  {
    NativeTimerScope timer(phase_timers_.energy_jvp);
    PsiFormerDerivativeLease scratch_lease(*shared_scratch_);
    has_evaluated_.store(true, std::memory_order_release);
    const std::uint32_t kinetic_mask =
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC);
    const std::uint32_t ecp_mask =
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP);
    if (local_energy_term_mask == 0 ||
        (local_energy_term_mask & ~(kinetic_mask | ecp_mask)) != 0 ||
        ((local_energy_term_mask & ecp_mask) != 0 && !ecp_replay_))
      throw std::invalid_argument(
          "PsiFormer streaming local-energy JVP term coverage is unavailable");

    std::fill(sample_products_.begin(), sample_products_.end(),
              wftrain::DerivativeValue{});

    if ((local_energy_term_mask & kinetic_mask) != 0)
    {
      PsiFormerReadTransaction state_lock(*model_state_);
      requireBoundVersion();
      for (std::size_t sample = 0; sample < sample_count_; ++sample)
      {
        setKineticWorkspacePositions(sample);
        const double* total_drift = total_log_gradients_real_.data() +
            sample * electron_count_ * std::size_t{3};
        const double* imaginary_drift =
            total_log_gradients_imaginary_.data() +
            sample * electron_count_ * std::size_t{3};
        const double* inverse_mass =
            inverse_masses_.data() + sample * electron_count_;
        const pf::DirectKineticResultView result = evaluateKineticResponse(total_drift, inverse_mass);
        requireResultVersion(result);

        wftrain::DerivativeValue& product = sample_products_[sample];
        for (std::size_t parameter = 0;
             parameter < result.kinetic_parameter_response.size(); ++parameter)
          product += result.kinetic_parameter_response[parameter] *
              direction.values()[parameter];

        const bool has_imaginary_drift = std::any_of(
            imaginary_drift,
            imaginary_drift + electron_count_ * std::size_t{3},
            [](double value) { return value != 0.0; });
        if (has_imaginary_drift)
        {
          const pf::DirectKineticConstView imaginary_response =
              evaluateGradientResponse(imaginary_drift, inverse_mass);
          for (std::size_t parameter = 0;
               parameter < imaginary_response.size(); ++parameter)
            product += wftrain::DerivativeValue(0.0,
                                                imaginary_response[parameter]) *
                direction.values()[parameter];
        }
      }
    }

    if ((local_energy_term_mask & ecp_mask) != 0)
      accumulateECPReplayJVP(direction);

    // Emit only after every sample is complete and the shared model lock is released.
    for (std::size_t sample = 0; sample < sample_count_; ++sample)
    {
      const wftrain::SampleProductTileDescriptor descriptor{
          schema_->providerId(), schema_->fingerprint(), parameter_version_,
          batch_ordinal_, sample_offset_ + sample, 1, sample};
      sink.add({descriptor, {sample_products_.data() + sample, 1}});
    }
  }

private:
  /// Validate the packed B-by-Ne-by-3 input extent with checked arithmetic.
  static std::size_t checkedPositionCount(std::size_t samples,
                                          std::size_t electrons)
  {
    return checkedBatchMemoryMultiply(
        checkedBatchMemoryMultiply(samples, electrons,
                                   "PsiFormer streaming sample positions"),
        std::size_t{3}, "PsiFormer streaming Cartesian positions");
  }

  /// Validate one sample-major inverse-mass extent with checked arithmetic.
  static std::size_t checkedMassCount(std::size_t samples,
                                      std::size_t electrons)
  {
    return checkedBatchMemoryMultiply(
        samples, electrons, "PsiFormer streaming inverse masses");
  }

  /// Validate one complex P-vector byte extent before allocating it.
  static std::vector<wftrain::DerivativeValue> makeParameterVector(
      std::size_t parameter_count)
  {
    checkedBatchMemoryMultiply(parameter_count,
                               sizeof(wftrain::DerivativeValue),
                               "PsiFormer streaming parameter channel");
    return std::vector<wftrain::DerivativeValue>(parameter_count);
  }

  /// Validate the O(B) JVP result extent before allocating it.
  static std::vector<wftrain::DerivativeValue> makeSampleProductVector(
      std::size_t sample_count)
  {
    checkedBatchMemoryMultiply(sample_count, sizeof(wftrain::DerivativeValue),
                               "PsiFormer streaming sample products");
    return std::vector<wftrain::DerivativeValue>(sample_count);
  }

  /// Keep vendor types private while reporting the selected numerical provider.
  bool hasDeviceProducts() const noexcept
  {
#ifdef ENABLE_CUDA
    return shared_scratch_->device_products != nullptr;
#else
    return false;
#endif
  }

  /// Optional preparation changes the live O(P) scratch bound explicitly.
  std::size_t deviceScoreContractionCapacity() const noexcept
  {
#ifdef ENABLE_CUDA
    return hasDeviceProducts() ? shared_scratch_->device_products->scoreContractionCapacity() : 0;
#else
    return 0;
#endif
  }

  std::size_t deviceScoreCapacity() const noexcept
  {
#ifdef ENABLE_CUDA
    return hasDeviceProducts() ? shared_scratch_->device_products->scoreCapacity() : 1;
#else
    return 1;
#endif
  }

  std::size_t deviceScoreContractionGeneration() const noexcept
  {
#ifdef ENABLE_CUDA
    return hasDeviceProducts() ? shared_scratch_->device_products->scoreContractionGeneration() : 0;
#else
    return 0;
#endif
  }

#ifdef ENABLE_CUDA
  /// Stage one algebraic component once; even a zero real direction admits the graph.
  bool prepareDeviceDirection(const wftrain::StructuredParameterVectorConstView& direction,
                              unsigned component) const
  {
    auto& values = shared_scratch_->device_direction;
    bool nonzero = false;
    for (std::size_t parameter = 0; parameter < values.size(); ++parameter)
    {
      values[parameter] = component == 0 ? direction.values()[parameter].real()
                                         : direction.values()[parameter].imag();
      nonzero = nonzero || values[parameter] != 0.0;
    }
    if (component != 0 && !nonzero)
      return false;
    shared_scratch_->device_products->prepareScoreDirection(values, parameter_version_);
    return true;
  }

  /// Return only the scalar contraction, keeping the P-vector response off the host.
  double evaluatePreparedDeviceJVP() const
  {
    const double result = shared_scratch_->device_products->scoreJvpPrepared(
        shared_scratch_->device_positions.data(), parameter_version_);
    reportDeviceProducts();
    return result;
  }

  /// Keep retained score inputs separate from host-modified ECP/kinetic workspaces.
  pf::DirectScoreResult evaluateRetainedScoreResponse(std::size_t sample) const
  {
    NativeTimerScope host_timer(phase_timers_.gpu_call);
    const auto& response = shared_scratch_->device_products->scoreDeviceCoordinates(
        retained_coordinates_->view(sample, 1), parameter_version_);
    reportDeviceProducts();
    pf::DirectScoreResult result;
    result.parameter_score = {response.data(), response.size()};
    result.parameter_version = parameter_version_;
    return result;
  }
#endif

  /// Evaluate a real canonical score through the selected, exclusively leased lane.
  pf::DirectScoreResult evaluateScoreResponse() const
  {
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      NativeTimerScope host_timer(phase_timers_.gpu_call);
      const auto& response = shared_scratch_->device_products->score(
          shared_scratch_->device_positions.data(), parameter_version_);
      reportDeviceProducts();
      pf::DirectScoreResult result;
      result.parameter_score = {response.data(), response.size()};
      result.parameter_version = parameter_version_;
      return result;
    }
#endif
    NativeTimerScope numerical_timer(phase_timers_.cpu_reverse);
    return model_state_->direct_score_executor.evaluate(*score_workspace_);
  }

  /// Preserve complete drift and sample-specific masses in the device kinetic root.
  pf::DirectKineticResultView evaluateKineticResponse(const double* drift, const double* masses) const
  {
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      NativeTimerScope host_timer(phase_timers_.gpu_call);
      auto& session = *shared_scratch_->device_products;
      const auto* positions = shared_scratch_->device_positions.data();
      const auto& response = session.kinetic(positions, drift, masses, parameter_version_);
      const auto& score = session.score(positions, parameter_version_);
      reportDeviceProducts();
      pf::DirectKineticResultView result;
      result.parameter_score = {score.data(), score.size()};
      result.kinetic_parameter_response = {response.data(), response.size()};
      result.parameter_version = parameter_version_;
      return result;
    }
#endif
    NativeTimerScope numerical_timer(phase_timers_.cpu_kinetic);
    if (model_state_->cpu_target_kinetic.targets)
    {
      const auto result = model_state_->direct_kinetic_executor.evaluateTargets(*kinetic_workspace_, drift,
          3 * electron_count_, masses, electron_count_);
      if (!shared_scratch_->reported_cpu_target_products)
      {
        app_log() << "  PsiFormer CPU target streaming kinetic completed: maxK=" << model_state_->cpu_target_kinetic.targets
                  << ", canonical_parameters=" << schema_->parameterCount() << ", parameter_version=" << parameter_version_
                  << ", numerical_owner_bytes_per_shared_lane=" << kinetic_workspace_->vectorStorageBytes()
                  << ", objective_independent=yes; ECP coverage follows existing request/replay masks" << std::endl;
        shared_scratch_->reported_cpu_target_products = true;
      }
      return result;
    }
    return model_state_->direct_kinetic_executor.evaluate(*kinetic_workspace_, drift,
        3 * electron_count_, masses, electron_count_);
  }

  /// Differentiate the gradient-only cross term for an imaginary complete drift.
  pf::DirectKineticConstView evaluateGradientResponse(const double* drift, const double* masses) const
  {
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      NativeTimerScope host_timer(phase_timers_.gpu_call);
      const auto& response = shared_scratch_->device_products->gradientResponse(
          shared_scratch_->device_positions.data(), drift, masses, parameter_version_);
      reportDeviceProducts();
      return {response.data(), response.size()};
    }
#endif
    NativeTimerScope numerical_timer(phase_timers_.cpu_kinetic);
    if (model_state_->cpu_target_kinetic.targets)
      return model_state_->direct_kinetic_executor.evaluateTargetGradientParameterResponse(
          *kinetic_workspace_, drift, 3 * electron_count_, masses, electron_count_);
    return model_state_->direct_kinetic_executor.evaluateGradientParameterResponse(
        *kinetic_workspace_, drift, 3 * electron_count_, masses, electron_count_);
  }

  /// Log real completed numerical work once, not merely device preparation.
  void reportDeviceProducts() const
  {
#ifdef ENABLE_CUDA
    if (!shared_scratch_->reported_device_products)
    {
      app_log() << "  PsiFormer numerical device parameter product completed: parameter_version="
                << parameter_version_ << ", kinetic_targets=" << shared_scratch_->device_products->kineticTargetCapacity()
                << ", CPU derivative fallback=no" << std::endl;
      shared_scratch_->reported_device_products = true;
    }
#endif
  }

  /// Copy one packed configuration into the reusable score workspace.
  void setScoreWorkspacePositions(std::size_t sample) const
  {
    const double* positions =
        positions_.data() + sample * electron_count_ * std::size_t{3};
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      std::copy_n(positions, shared_scratch_->device_positions.size(),
                  shared_scratch_->device_positions.begin());
      return;
    }
#endif
    score_workspace_->setPositions(pf::GeometryPositionView::interleaved(positions, electron_count_));
  }

  /// Copy one packed configuration into the reusable kinetic workspace.
  void setKineticWorkspacePositions(std::size_t sample) const
  {
    const double* positions =
        positions_.data() + sample * electron_count_ * std::size_t{3};
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      std::copy_n(positions, shared_scratch_->device_positions.size(),
                  shared_scratch_->device_positions.begin());
      return;
    }
#endif
    kinetic_workspace_->setPositions(pf::GeometryPositionView::interleaved(positions, electron_count_));
  }

  /// Copy one accepted replacement into the reusable score workspace.
  void setReplayPointPositions(const wftrain::NonLocalECPReplayPoint& point) const
  {
    setScoreWorkspacePositions(point.sample);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
#ifdef ENABLE_CUDA
      if (hasDeviceProducts())
      {
        shared_scratch_->device_positions[3 * point.electron + dimension] = point.replacement_position[dimension];
        continue;
      }
#endif
      score_workspace_->setPosition(point.electron, dimension,
                                    point.replacement_position[dimension]);
    }
  }

  /// Add the exact fixed-grid ratio derivative without drawing or rebuilding a grid.
  void accumulateECPReplayJVP(
      const wftrain::StructuredParameterVectorConstView& direction) const
  {
    if (!ecp_replay_)
      throw std::logic_error("PsiFormer ECP JVP replay is not bound");
    PsiFormerReadTransaction state_lock(*model_state_);
    requireBoundVersion();
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      // Replay exactly the accepted grid. Only scalar JVPs cross the device boundary.
      const auto capacity = deviceScoreCapacity();
      const auto points = ecp_replay_->points();
#ifndef QMC_COMPLEX
      const bool tile_virtuals = capacity > 1 && !points.empty();
#else
      const bool tile_virtuals = false; // Complex-build replay keeps its existing scalar qualification.
#endif
      if (tile_virtuals && shared_scratch_->device_ecp_positions.size() <
          checkedPositionCount(capacity, electron_count_))
        throw std::logic_error("GPU ECP JVP exceeds its cold HOST tile preparation");
      std::size_t components = 0, completed_tiles = 0, configurations = 0, last_count = 0;
      for (unsigned component = 0; component < 2; ++component)
      {
        if (!prepareDeviceDirection(direction, component))
          continue;
        ++components;
        const wftrain::DerivativeValue factor = component == 0
            ? wftrain::DerivativeValue(1.0, 0.0) : wftrain::DerivativeValue(0.0, 1.0);
        for (std::size_t sample = 0; sample < sample_count_; ++sample)
        {
          setScoreWorkspacePositions(sample);
          reference_score_products_[sample] = evaluatePreparedDeviceJVP();
        }
        if (tile_virtuals)
        {
          for (std::size_t begin = 0; begin < points.size(); begin += capacity)
          {
            const auto count = std::min(capacity, points.size() - begin);
            const auto elements = 3 * electron_count_;
            for (std::size_t local = 0; local < count; ++local)
            {
              const auto& point = points[begin + local];
              auto* destination = shared_scratch_->device_ecp_positions.data() + local * elements;
              std::copy_n(positions_.data() + point.sample * elements, elements, destination);
              std::copy_n(point.replacement_position.data(), 3, destination + 3 * point.electron);
            }
            const auto& products = shared_scratch_->device_products->scoreJvpPreparedTile(
                shared_scratch_->device_ecp_positions.data(), count, parameter_version_);
            // Consume the borrowed O(B) scalars before any subsequent session action.
            for (std::size_t local = 0; local < count; ++local)
            {
              const auto& point = points[begin + local];
              const wftrain::DerivativeValue weight = point.bare_weight * point.complete_ratio;
              requireFinite(weight, "accepted-grid weight");
              const wftrain::DerivativeValue contribution = factor * weight *
                  (products[local] - reference_score_products_[point.sample]);
              requireFinite(contribution, "accepted-grid JVP contribution");
              sample_products_[point.sample] += contribution;
              requireFinite(sample_products_[point.sample], "accepted-grid sample JVP");
            }
            ++completed_tiles;
            configurations += count;
            last_count = count;
          }
        }
        else
          for (const auto& point : points)
          {
            setReplayPointPositions(point);
            const wftrain::DerivativeValue weight = point.bare_weight * point.complete_ratio;
            requireFinite(weight, "accepted-grid weight");
            const wftrain::DerivativeValue contribution = factor * weight *
                (evaluatePreparedDeviceJVP() - reference_score_products_[point.sample]);
            requireFinite(contribution, "accepted-grid JVP contribution");
            sample_products_[point.sample] += contribution;
            requireFinite(sample_products_[point.sample], "accepted-grid sample JVP");
          }
      }
      app_log() << "  PsiFormer real GPU accepted-grid ECP JVP: points=" << points.size()
                << ", prepared_score_capacity=" << capacity << ", direction_components=" << components
                << ", virtual_jvp_tiles=" << completed_tiles << ", virtual_jvp_configurations=" << configurations
                << ", last_virtual_tile_count=" << last_count << "; CPU derivative fallback=no" << std::endl;
      return;
    }
#endif
    for (std::size_t sample = 0; sample < sample_count_; ++sample)
    {
      setScoreWorkspacePositions(sample);
      const pf::DirectScoreResult score = evaluateScoreResponse();
      requireResultVersion(score);
      wftrain::DerivativeValue product{};
      for (std::size_t parameter = 0; parameter < score.parameter_score.size;
           ++parameter)
        product += score.parameter_score[parameter] * direction.values()[parameter];
      requireFinite(product, "reference score JVP");
      reference_score_products_[sample] = product;
    }

    for (const auto& point : ecp_replay_->points())
    {
      setReplayPointPositions(point);
      const pf::DirectScoreResult score = evaluateScoreResponse();
      requireResultVersion(score);
      wftrain::DerivativeValue virtual_product{};
      for (std::size_t parameter = 0; parameter < score.parameter_score.size;
           ++parameter)
        virtual_product +=
            score.parameter_score[parameter] * direction.values()[parameter];
      const wftrain::DerivativeValue weight =
          point.bare_weight * point.complete_ratio;
      requireFinite(virtual_product, "virtual score JVP");
      requireFinite(weight, "accepted-grid weight");
      const wftrain::DerivativeValue contribution =
          weight * (virtual_product - reference_score_products_[point.sample]);
      requireFinite(contribution, "accepted-grid JVP contribution");
      sample_products_[point.sample] += contribution;
      requireFinite(sample_products_[point.sample],
                    "accepted-grid sample JVP");
    }
  }

  /// Add replayed ECP VJPs when the combined owner advertises ECP term coverage.
  void accumulateECPReplayVJPs(
      wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel>
          channels, bool append_energy = false) const
  {
    if (!ecp_replay_)
      throw std::logic_error("PsiFormer ECP VJP replay is not bound");
    std::fill(replay_weight_sums_.begin(), replay_weight_sums_.end(),
              wftrain::DerivativeValue{});
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
    if (hasDeviceProducts())
    {
      std::array<std::size_t, MAXIMUM_VJP_CHANNELS> active{};
      std::size_t count = 0;
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        if (channels[channel].product == wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
            (channels[channel].local_energy_term_mask & ecp_replay_->localEnergyTermMask()) != 0)
          active[count++] = channel;
      if (count == 0 || sample_count_ == 0)
        return;
      if (append_energy && (count != 1 || active[0] != 0 || channels.size() != 3))
        throw std::invalid_argument("Resident ECP append requires the first ordered ENERGY C3 channel");
      auto& session = *shared_scratch_->device_products;
      const auto transfer_before = session.transferSnapshot();
      const auto before_tiles = session.weightedScoreTiles();
      const auto before_configurations = session.weightedScoreConfigurations();
      const auto capacity = session.scoreCapacity();
      const auto& coefficient_values = channels[active[0]].coefficients.values;
      const bool tile_virtuals = count == 1 && capacity > 1 && !ecp_replay_->points().empty() &&
          std::all_of(coefficient_values.begin(), coefficient_values.end(),
              [](const auto& value) { return value.imag() == 0.0; }) &&
          std::all_of(ecp_replay_->points().begin(), ecp_replay_->points().end(),
              [](const auto& point) { return point.bare_weight.imag() == 0.0 && point.complete_ratio.imag() == 0.0; });
      std::size_t tile_count = 0, last_tile_count = 0;
      if (tile_virtuals && (shared_scratch_->device_ecp_weights.size() < capacity ||
          shared_scratch_->device_ecp_positions.size() < checkedPositionCount(capacity, electron_count_)))
        throw std::logic_error("GPU ECP virtual tile exceeds its cold HOST preparation");
      try
      {
        if (!append_energy)
          session.beginScoreContraction(count, parameter_version_);
        std::array<wftrain::DerivativeValue, MAXIMUM_VJP_CHANNELS> coefficients{};
        const std::array<wftrain::DerivativeValue, 3> energy_slot{{{1, 0}, {}, {}}};
        const auto contraction_count = append_energy ? 3 : count;
        const auto flush = [&]() {
          session.accumulateWeightedScoreTile(shared_scratch_->device_ecp_positions.data(),
              shared_scratch_->device_ecp_weights.data(), tile_count, parameter_version_,
              append_energy ? energy_slot.data() : nullptr, append_energy ? 3 : 1);
          last_tile_count = tile_count;
          tile_count = 0;
        };
        for (const auto& point : ecp_replay_->points())
        {
          const auto weight = point.bare_weight * point.complete_ratio;
          requireFinite(weight, "accepted-grid VJP weight");
          replay_weight_sums_[point.sample] += weight;
          requireFinite(replay_weight_sums_[point.sample], "accepted-grid sample weight sum");
          for (std::size_t slot = 0; slot < count; ++slot)
            coefficients[slot] = channels[active[slot]].coefficients.values[point.sample] * weight;
          if (tile_virtuals)
          {
            // Pack immutable accepted references, not coordinates from a previous virtual point.
            const auto elements = 3 * electron_count_;
            auto* destination = shared_scratch_->device_ecp_positions.data() + tile_count * elements;
            std::copy_n(positions_.data() + point.sample * elements, elements, destination);
            std::copy_n(point.replacement_position.data(), 3, destination + 3 * point.electron);
            shared_scratch_->device_ecp_weights[tile_count++] = coefficients[0].real();
            // The existing weighted reverse still validates zero-weight configurations.
            if (tile_count == capacity)
              flush();
          }
          else
          {
            setReplayPointPositions(point);
            session.accumulateScoreContraction(shared_scratch_->device_positions.data(), coefficients.data(),
                                               contraction_count, parameter_version_);
          }
        }
        if (tile_count)
          flush();
        for (std::size_t sample = 0; sample < sample_count_; ++sample)
        {
          setScoreWorkspacePositions(sample);
          for (std::size_t slot = 0; slot < count; ++slot)
            coefficients[slot] = -channels[active[slot]].coefficients.values[sample] * replay_weight_sums_[sample];
          session.accumulateScoreContraction(shared_scratch_->device_positions.data(), coefficients.data(),
                                             contraction_count, parameter_version_);
        }
        if (append_energy)
        {
          app_log() << "  Resident ENERGY accepted-grid ECP appended: points=" << ecp_replay_->points().size()
                    << ", prepared_score_capacity=" << capacity << ", weighted_virtual_tiles="
                    << session.weightedScoreTiles() - before_tiles << ", weighted_virtual_configurations="
                    << session.weightedScoreConfigurations() - before_configurations
                    << ", last_virtual_tile_count=" << last_tile_count
                    << ", final_parameter_download_bytes=0; complete C3 finite admission pending" << std::endl;
          return; // The outer ENERGY transaction owns finish, consumption and revocation.
        }
        session.finishScoreContraction(parameter_version_);
        auto& output = shared_scratch_->device_ecp_chunk;
        for (const auto& chunk : chunk_plan_.chunks())
          for (std::size_t slot = 0; slot < count; ++slot)
          {
            if (chunk.count > output.size())
              throw std::logic_error("GPU ECP replay chunk exceeds cold preparation");
            session.downloadScoreContractionChunk(slot, chunk.parameter_offset, output.data(), chunk.count,
                                                  parameter_version_);
            auto* destination = channel_accumulators_[active[slot]].data() + chunk.parameter_offset;
            for (std::size_t i = 0; i < chunk.count; ++i)
            {
              destination[i] += output[i]; // Preserve existing kinetic/score contributions.
              requireFinite(destination[i], "complete ECP VJP");
            }
          }
        const auto after = session.transferSnapshot();
        app_log() << "  PsiFormer regular-node real GPU accepted-grid ECP replay: points="
                  << ecp_replay_->points().size() << ", channels=" << count
                  << ", prepared_score_capacity=" << capacity
                  << ", weighted_virtual_tiles=" << session.weightedScoreTiles() - before_tiles
                  << ", weighted_virtual_configurations=" << session.weightedScoreConfigurations() - before_configurations
                  << ", last_virtual_tile_count=" << last_tile_count
                  << ", full_parameter_downloads_per_point=0, final_channel_parameter_bytes="
                  << count * schema_->parameterCount() * sizeof(wftrain::DerivativeValue)
                  << ", measured_downloaded_bytes=" << after.downloaded_bytes - transfer_before.downloaded_bytes
                  << "; status/factor transfers included, CPU derivative fallback=no" << std::endl;
        reportDeviceProducts();
        // HOST channels now own the result; revoke DEVICE readiness before any sink callback can fail.
        session.abortScoreContraction();
      }
      catch (...)
      {
        session.abortScoreContraction();
        throw;
      }
      return;
    }
#endif
    if (append_energy)
      throw std::invalid_argument("Resident ECP append requires real GPU products");
    for (const auto& point : ecp_replay_->points())
    {
      setReplayPointPositions(point);
      const pf::DirectScoreResult score = evaluateScoreResponse();
      requireResultVersion(score);
      const wftrain::DerivativeValue weight =
          point.bare_weight * point.complete_ratio;
      requireFinite(weight, "accepted-grid VJP weight");
      replay_weight_sums_[point.sample] += weight;
      requireFinite(replay_weight_sums_[point.sample],
                    "accepted-grid sample weight sum");
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        if (channels[channel].product ==
                wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
            (channels[channel].local_energy_term_mask &
             wftrain::localEnergyTermBit(
                 wftrain::LocalEnergyTerm::NONLOCAL_ECP)) != 0)
          accumulateResponse(
              channel,
              channels[channel].coefficients.values[point.sample] * weight,
              score.parameter_score);
    }
    for (std::size_t sample = 0; sample < sample_count_; ++sample)
    {
      setScoreWorkspacePositions(sample);
      const pf::DirectScoreResult score = evaluateScoreResponse();
      requireResultVersion(score);
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        if (channels[channel].product ==
                wftrain::DerivativeProduct::LOCAL_ENERGY_VJP &&
            (channels[channel].local_energy_term_mask &
             wftrain::localEnergyTermBit(
                 wftrain::LocalEnergyTerm::NONLOCAL_ECP)) != 0)
          accumulateResponse(
              channel,
              -channels[channel].coefficients.values[sample] *
                  replay_weight_sums_[sample],
              score.parameter_score);
    }
  }

  /// Reject non-finite replay arithmetic in fast-math builds.
  static void requireFinite(wftrain::DerivativeValue value,
                            const char* description)
  {
    if (!psiformer::determinant::isFiniteReal(value.real()) ||
        !psiformer::determinant::isFiniteReal(value.imag()))
      throw std::invalid_argument(std::string("PsiFormer ") + description +
                                  " must be finite");
  }

  /// Fold one real canonical response into a complex coefficient channel.
  template<class ResponseView>
  void accumulateResponse(std::size_t channel,
                          wftrain::DerivativeValue coefficient,
                          const ResponseView& response) const
  {
    const std::size_t response_size = responseExtent(response);
    if (response_size != schema_->parameterCount())
      throw std::logic_error(
          "PsiFormer streaming response has the wrong parameter extent");
    std::vector<wftrain::DerivativeValue>& destination =
        channel_accumulators_[channel];
    for (std::size_t parameter = 0; parameter < response_size; ++parameter)
      destination[parameter] += coefficient * response[parameter];
  }

  /// Return the extent of the score executor's dependency-light response view.
  static std::size_t responseExtent(
      const pf::DirectParameterScoreView& response) noexcept
  {
    return response.size;
  }

  /// Return the extent of the kinetic executor's span-like response view.
  static std::size_t responseExtent(
      const pf::DirectKineticConstView& response) noexcept
  {
    return response.size();
  }

  /// Reject publication that occurred after this operator was prepared.
  void requireBoundVersion() const
  {
    if (model_state_->model.p.version() != parameter_version_)
      throw std::runtime_error(
          "PsiFormer streaming derivative parameter version is stale");
  }

  /// Reject a direct result that was not produced from the bound read transaction.
  void requireResultVersion(const pf::DirectScoreResult& result) const
  {
    if (result.parameter_version != parameter_version_)
      throw std::logic_error(
          "PsiFormer streaming derivative result has the wrong parameter version");
  }

  /// Reject a kinetic result not produced from the bound read transaction.
  void requireResultVersion(const pf::DirectKineticResultView& result) const
  {
    if (result.parameter_version != parameter_version_)
      throw std::logic_error(
          "PsiFormer streaming kinetic result has the wrong parameter version");
  }

  /// Mix one allocation identity into the stable prepared-storage fingerprint.
  static void mixFingerprint(std::size_t& fingerprint, std::size_t value) noexcept
  {
    fingerprint ^= value + std::size_t{0x9e3779b9U} + (fingerprint << 6) +
        (fingerprint >> 2);
  }

  /// Cache the construction-time identity used to detect later storage changes.
  void initializeStorageDiagnostics()
  {
    storage_diagnostics_ = measureStorageDiagnostics();
  }

  /// Re-measure every retained numeric owner for live warmed-call diagnostics.
  wftrain::StreamingDerivativeStorageDiagnostics measureStorageDiagnostics() const
  {
    std::lock_guard scratch_lock(shared_scratch_->execution_mutex);
    const std::size_t score_bytes = hasDeviceProducts()
        ? checkedBatchMemoryMultiply(checkedBatchMemoryMultiply(
              std::size_t{8} + 2 * deviceScoreContractionCapacity(), schema_->parameterCount(),
              "PsiFormer device product parameter vectors"), sizeof(double), "PsiFormer device parameter bytes")
        : score_workspace_->scoreStorageBytes();
    const std::size_t kinetic_parameter_bytes =
        hasDeviceProducts() ? 0 : kinetic_workspace_->parameterStorageBytes();
    const std::size_t expected_kinetic_parameter_bytes =
        checkedBatchMemoryMultiply(
            checkedBatchMemoryMultiply(
                std::size_t{2}, schema_->parameterCount(),
                "PsiFormer streaming kinetic parameter vectors"),
            sizeof(double), "PsiFormer streaming kinetic parameter bytes");
    if (!hasDeviceProducts() && kinetic_parameter_bytes != expected_kinetic_parameter_bytes)
      throw std::length_error(
          "PsiFormer streaming kinetic parameter storage is not exact");
    std::size_t channel_bytes     = 0;
    for (const auto& channel : channel_accumulators_)
      channel_bytes = checkedBatchMemoryAdd(
          channel_bytes,
          checkedBatchMemoryMultiply(
              channel.capacity(), sizeof(wftrain::DerivativeValue),
              "PsiFormer streaming channel capacity"),
          "PsiFormer streaming channel storage");
    const std::size_t position_bytes = checkedBatchMemoryMultiply(
        positions_.capacity(), sizeof(double),
        "PsiFormer streaming position capacity");
    std::size_t device_position_bytes = 0;
#ifdef ENABLE_CUDA
    if (retained_coordinates_)
      device_position_bytes = retained_coordinates_->deviceWorkspaceBytes();
#endif
    const std::size_t gradient_bytes = checkedBatchMemoryMultiply(
        total_log_gradients_real_.capacity(), sizeof(double),
        "PsiFormer streaming real total-drift capacity");
    const std::size_t imaginary_gradient_bytes = checkedBatchMemoryMultiply(
        total_log_gradients_imaginary_.capacity(), sizeof(double),
        "PsiFormer streaming imaginary total-drift capacity");
    const std::size_t inverse_mass_bytes = checkedBatchMemoryMultiply(
        inverse_masses_.capacity(), sizeof(double),
        "PsiFormer streaming inverse-mass capacity");
    const std::size_t auxiliary_bytes = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(gradient_bytes, imaginary_gradient_bytes,
                              "PsiFormer streaming complex drift storage"),
        inverse_mass_bytes,
        "PsiFormer streaming sample auxiliary storage");
    const std::size_t sample_product_elements = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(sample_products_.capacity(),
                              reference_score_products_.capacity(),
                              "PsiFormer streaming JVP sample products"),
        replay_weight_sums_.capacity(),
        "PsiFormer streaming sample-product elements");
    const std::size_t sample_product_bytes = checkedBatchMemoryMultiply(
        sample_product_elements,
        sizeof(wftrain::DerivativeValue),
        "PsiFormer streaming sample-product capacity");
    const std::size_t replay_bytes = ecp_replay_
        ? ecp_replay_->storageDiagnostics().retained_numeric_bytes
        : 0;
    std::size_t workspace_bytes = 0;
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      const auto& session = *shared_scratch_->device_products;
      workspace_bytes = checkedBatchMemoryAdd(session.deviceWorkspaceBytes(), session.hostWorkspaceBytes(),
                                               "PsiFormer device product host/device storage");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(shared_scratch_->device_positions.capacity(), sizeof(double),
                                      "PsiFormer device provider coordinate staging"),
          "PsiFormer device provider scratch");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(shared_scratch_->device_direction.capacity(), sizeof(double),
                                      "PsiFormer device provider direction staging"),
          "PsiFormer device provider scratch");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(shared_scratch_->device_ecp_chunk.capacity(), sizeof(wftrain::DerivativeValue),
                                      "PsiFormer shared ECP chunk bytes"), "PsiFormer device provider scratch");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(checkedBatchMemoryAdd(shared_scratch_->device_ecp_positions.capacity(),
              shared_scratch_->device_ecp_weights.capacity(), "GPU ECP HOST tile capacity"), sizeof(double),
              "GPU ECP HOST tile bytes"), "PsiFormer device provider scratch");
    }
    else
#endif
      workspace_bytes = checkedBatchMemoryAdd(score_workspace_->vectorStorageBytes(),
          kinetic_workspace_->vectorStorageBytes(), "PsiFormer streaming evaluator workspaces");
    const std::size_t parameter_bytes = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(score_bytes, kinetic_parameter_bytes,
                              "PsiFormer streaming real parameter scratch"),
        channel_bytes,
        "PsiFormer streaming parameter scratch");
    const std::size_t retained_bytes = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(
            checkedBatchMemoryAdd(workspace_bytes, channel_bytes,
                                  "PsiFormer streaming evaluator and channels"),
            checkedBatchMemoryAdd(position_bytes, device_position_bytes, "PsiFormer host/device sample positions"),
            "PsiFormer streaming positions"),
        checkedBatchMemoryAdd(
            checkedBatchMemoryAdd(auxiliary_bytes, replay_bytes,
                                  "PsiFormer streaming ECP replay storage"),
            sample_product_bytes,
            "PsiFormer streaming sample storage"),
        "PsiFormer streaming retained storage");

    std::size_t fingerprint = std::size_t{1469598103U};
    if (score_workspace_)
      mixFingerprint(fingerprint, score_workspace_->storageFingerprint());
    mixFingerprint(fingerprint, workspace_bytes);
    mixFingerprint(fingerprint, device_position_bytes);
#ifdef ENABLE_CUDA
    if (retained_coordinates_)
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(retained_coordinates_->view(0, sample_count_).sourceIdentity()));
#endif
    if (kinetic_workspace_)
      mixFingerprint(fingerprint, kinetic_workspace_->storageFingerprint());
#ifdef ENABLE_CUDA
    if (hasDeviceProducts())
    {
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_products.get()));
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_positions.data()));
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_direction.data()));
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_ecp_chunk.data()));
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_ecp_positions.data()));
      mixFingerprint(fingerprint, shared_scratch_->device_ecp_positions.capacity());
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(shared_scratch_->device_ecp_weights.data()));
      mixFingerprint(fingerprint, shared_scratch_->device_ecp_weights.capacity());
    }
#endif
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(positions_.data()));
    mixFingerprint(fingerprint, positions_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(total_log_gradients_real_.data()));
    mixFingerprint(fingerprint, total_log_gradients_real_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(total_log_gradients_imaginary_.data()));
    mixFingerprint(fingerprint, total_log_gradients_imaginary_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(inverse_masses_.data()));
    mixFingerprint(fingerprint, inverse_masses_.capacity());
    for (const auto& channel : channel_accumulators_)
    {
      mixFingerprint(fingerprint,
                     reinterpret_cast<std::uintptr_t>(channel.data()));
      mixFingerprint(fingerprint, channel.capacity());
    }
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(sample_products_.data()));
    mixFingerprint(fingerprint, sample_products_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(reference_score_products_.data()));
    mixFingerprint(fingerprint, reference_score_products_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(replay_weight_sums_.data()));
    mixFingerprint(fingerprint, replay_weight_sums_.capacity());
    if (ecp_replay_)
      mixFingerprint(fingerprint,
                     ecp_replay_->storageDiagnostics().storage_fingerprint);
    if (fingerprint == 0)
      fingerprint = 1;

    const auto fallback_allowance = hasDeviceProducts() ? checkedBatchMemoryMultiply(schema_->parameterCount(),
        MAXIMUM_VJP_CHANNELS * sizeof(wftrain::DerivativeValue), "GPU HOST fallback allowance") : 0;
    return {schema_->parameterCount(),
            sample_count_,
            hasDeviceProducts() ? std::size_t{8} + 2 * deviceScoreContractionCapacity() : std::size_t{3},
            static_cast<std::size_t>(std::count_if(channel_accumulators_.begin(), channel_accumulators_.end(),
                [](const auto& channel) { return !channel.empty(); })),
            parameter_bytes,
            workspace_bytes,
            position_bytes,
            auxiliary_bytes,
            sample_product_bytes,
            retained_bytes,
            1 + deviceScoreContractionGeneration(),
            fingerprint,
            reinterpret_cast<std::uintptr_t>(shared_scratch_.get()),
            checkedBatchMemoryAdd(workspace_bytes, channel_bytes,
                                  "PsiFormer shared scratch retained bytes"),
            device_position_bytes,
            fallback_allowance > channel_bytes ? fallback_allowance - channel_bytes : 0};
  }

  std::shared_ptr<PsiFormerSharedState> model_state_;
  std::shared_ptr<const wftrain::StructuredParameterSchema> schema_;
  std::size_t parameter_version_ = 0;
  std::size_t batch_ordinal_     = 0;
  std::size_t sample_offset_     = 0;
  std::size_t sample_count_      = 0;
  std::size_t electron_count_    = 0;
  std::vector<double> positions_;
#ifdef ENABLE_CUDA
  std::unique_ptr<psiformer::device::PreparedDeviceCoordinates> retained_coordinates_;
#endif
  std::vector<double> total_log_gradients_real_;
  std::vector<double> total_log_gradients_imaginary_;
  std::vector<double> inverse_masses_;
  wftrain::ParameterChunkPlan chunk_plan_;
  std::shared_ptr<PsiFormerDerivativeScratch> shared_scratch_;
  const std::unique_ptr<pf::DirectScoreWorkspace>& score_workspace_;
  const std::unique_ptr<pf::DirectKineticWorkspace>& kinetic_workspace_;
  std::array<std::vector<wftrain::DerivativeValue>,
                     MAXIMUM_VJP_CHANNELS>
      &channel_accumulators_;
  mutable std::vector<wftrain::DerivativeValue> sample_products_;
  mutable std::vector<wftrain::DerivativeValue> reference_score_products_;
  mutable std::vector<wftrain::DerivativeValue> replay_weight_sums_;
  std::unique_ptr<wftrain::NonLocalECPJVPReplay> ecp_replay_;
  mutable std::atomic<bool> has_evaluated_{false};
  wftrain::StreamingDerivativeStorageDiagnostics storage_diagnostics_;
};

/** Contract live ECP tiles and optionally freeze their traversal for exact replay.
 *
 * The consumer retains one direct score tape, three complex P-vectors, copied
 * reference positions, three B-sized coefficient rows, and one B-sized weight
 * sum.  Only an explicitly requested JVP replay adds exact O(Q)
 * positions/weights/provenance; neither mode constructs a Q-by-P object.
 */
class PsiFormerNonLocalECPDerivativeConsumer final
    : public wftrain::NonLocalECPDerivativeConsumer
{
public:
  static constexpr std::size_t MAXIMUM_CHANNELS = 3;

  /// Bind one immutable model version and copy only the reference configurations.
  PsiFormerNonLocalECPDerivativeConsumer(
      std::shared_ptr<PsiFormerSharedState> model_state,
      std::shared_ptr<const wftrain::StructuredParameterSchema> schema,
      std::size_t parameter_version,
      std::size_t sample_count,
      std::vector<double> positions,
      std::size_t maximum_parameter_chunk_size,
      bool capture_jvp_replay,
      bool replay_only)
      : model_state_(std::move(model_state)),
        schema_(std::move(schema)),
        parameter_version_(parameter_version),
        capture_jvp_replay_(capture_jvp_replay),
        replay_only_(replay_only),
        sample_count_(sample_count),
        electron_count_(model_state_->execution_plan.modelShape().electrons()),
        positions_(std::move(positions)),
        chunk_plan_(*schema_, parameter_version_, maximum_parameter_chunk_size),
        score_workspace_(replay_only_ || model_state_->numerical_tile_backend ? nullptr : model_state_->direct_score_executor.makeWorkspace()),
        channel_accumulators_{makeParameterVector(replay_only_ ? 0 : schema_->parameterCount()),
                              makeParameterVector(replay_only_ ? 0 : schema_->parameterCount()),
                              makeParameterVector(replay_only_ ? 0 : schema_->parameterCount())},
        coefficients_{makeSampleVector(sample_count_),
                      makeSampleVector(sample_count_),
                      makeSampleVector(sample_count_)},
        sample_weight_sums_(makeSampleVector(sample_count_))
  {
    if (replay_only_ && !capture_jvp_replay_)
      throw std::invalid_argument("PsiFormer replay-only ECP consumer requires replay capture");
    const std::size_t expected_positions = checkedBatchMemoryMultiply(
        checkedBatchMemoryMultiply(sample_count_, electron_count_,
                                   "PsiFormer ECP reference positions"),
        std::size_t{3}, "PsiFormer ECP Cartesian positions");
    if (positions_.size() != expected_positions ||
        (score_workspace_ && score_workspace_->scoreSize() != schema_->parameterCount()))
      throw std::invalid_argument(
          "PsiFormer ECP consumer has inconsistent model or reference extents");
#ifdef ENABLE_CUDA
    if (model_state_->numerical_tile_backend && !replay_only_)
    {
      const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
          model_state_->numerical_tile_backend.get());
      if (!backend)
        throw std::logic_error("PsiFormer ECP products require the numerical device backend");
      device_products_ = backend->prepareProductSession(
          psiformer::device::ProductSessionProfile::SCORE_ONLY);
      device_positions_.resize(3 * electron_count_);
    }
#endif
    baseline_storage_ = measureStorageDiagnostics();
  }

  /// Reject foreign identity, unsupported channels, and stale model state before RNG use.
  void preflight(
      const wftrain::NonLocalECPDerivativeContext& context,
      wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
      const wftrain::ParameterReductionSink& sink) const override
  {
    if (state_ != State::IDLE)
      throw std::logic_error(
          "PsiFormer ECP consumer can preflight only from the idle state");
    if (context.provider_id != schema_->providerId() ||
        context.schema_fingerprint != schema_->fingerprint() ||
        context.parameter_version != parameter_version_ ||
        context.sample_count != sample_count_)
      throw std::invalid_argument(
          "PsiFormer ECP consumer context has a foreign schema, version, or batch");
    if (context.localization !=
            wftrain::NonLocalECPLocalization::ORDINARY_LOCALITY ||
        !context.uses_virtual_particles || !context.scalar_relativistic ||
        context.used_dense_derivative_fallback)
      throw std::invalid_argument(
          "PsiFormer ECP consumer requires bounded scalar ordinary locality");
    if (channels.empty() || channels.size() > MAXIMUM_CHANNELS)
      throw std::invalid_argument(
          "PsiFormer ECP consumer requires one to three coefficient channels");
    if (sink.state() != wftrain::DerivativeSinkState::IDLE)
      throw std::invalid_argument(
          "PsiFormer ECP consumer requires an idle parameter sink");

    {
      PsiFormerReadTransaction state_lock(*model_state_, replay_only_ && model_state_->deviceOnlyValueReads());
      requireBoundVersion();
    }
    for (const auto& input : channels)
    {
      if (input.product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
          input.local_energy_term_mask != wftrain::localEnergyTermBit(
                                              wftrain::LocalEnergyTerm::NONLOCAL_ECP))
        throw std::invalid_argument(
            "PsiFormer ECP consumer accepts only isolated nonlocal-ECP VJP channels");
      if (input.coefficients.provider_id != schema_->providerId() ||
          input.coefficients.schema_fingerprint != schema_->fingerprint() ||
          input.coefficients.parameter_version != parameter_version_ ||
          input.coefficients.batch_ordinal != context.batch_ordinal ||
          input.coefficients.sample_offset != context.sample_offset ||
          input.coefficients.values.size() != sample_count_)
        throw std::invalid_argument(
            "PsiFormer ECP coefficient channel has incompatible identity or extent");
      for (std::size_t sample = 0; sample < sample_count_; ++sample)
        requireFinite(input.coefficients.values[sample], "coefficient");
    }
  }

  /// Validate the ordinary-locality transaction and activate its checked sink.
  void begin(
      const wftrain::NonLocalECPDerivativeContext& context,
      wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
      wftrain::ParameterReductionSink& sink) override
  {
    preflight(context, channels, sink);
    if (context.grid_fingerprint == 0)
      throw std::invalid_argument(
          "PsiFormer ECP consumer requires a nonzero accepted-grid fingerprint");
    for (std::size_t channel = 0; !replay_only_ && channel < channels.size(); ++channel)
    {
      const auto& input = channels[channel];
      for (std::size_t sample = 0; sample < sample_count_; ++sample)
      {
        const auto coefficient = input.coefficients.values[sample];
        coefficients_[channel][sample] = coefficient;
      }
      std::fill(channel_accumulators_[channel].begin(),
                channel_accumulators_[channel].end(),
                wftrain::DerivativeValue{});
    }
    std::fill(sample_weight_sums_.begin(), sample_weight_sums_.end(),
              wftrain::DerivativeValue{});

    if (capture_jvp_replay_)
    {
      checkedBatchMemoryMultiply(context.expected_point_count,
                                 sizeof(wftrain::NonLocalECPReplayPoint),
                                 "PsiFormer ECP replay point bytes");
      replay_points_.assign(context.expected_point_count, {});
      replay_segments_.clear();
      replay_segments_.reserve(context.expected_point_count);
      replay_tiles_.clear();
      replay_tiles_.reserve(context.expected_point_count);
      replay_evaluation_stamps_.clear();
      replay_evaluation_stamps_.reserve(context.expected_point_count);
    }

    channel_count_             = channels.size();
    expected_tile_ordinal_     = 0;
    expected_point_count_      = context.expected_point_count;
    consumed_point_count_      = 0;
    grid_fingerprint_          = context.grid_fingerprint;
    batch_ordinal_             = context.batch_ordinal;
    sample_offset_             = context.sample_offset;
    replay_released_           = false;
    stream_descriptor_         = {
        schema_->providerId(), schema_->fingerprint(), parameter_version_,
        wftrain::derivativeProductBit(
            wftrain::DerivativeProduct::LOCAL_ENERGY_VJP),
        wftrain::DerivativeAdjoint::TRANSPOSE,
        wftrain::ParameterScalarDomain::REAL64,
        wftrain::ParameterScalarDomain::COMPLEX128,
        context.batch_ordinal, context.sample_offset, sample_count_,
        wftrain::ReductionDomain::CROWD_LOCAL,
        wftrain::DerivativeExecutionDomain::HOST, MAXIMUM_CHANNELS,
        chunk_plan_.maximumChunkSize(), 1};
    sink_ = &sink;
    try
    {
      // Replay-only preparation validates the grid but performs no derivative
      // contraction and leaves the supplied response sink idle.
      if (!replay_only_)
        sink_->begin(stream_descriptor_, chunk_plan_, channels);
      state_ = State::ACTIVE;
    }
    catch (...)
    {
      sink_ = nullptr;
      state_ = State::POISONED;
      throw;
    }
  }

  /// Consume every replacement score in one live bounded tile.
  void consume(const wftrain::NonLocalECPDerivativeTile& tile) override
  {
    try
    {
      requireActive("consume");
      if (tile.tile_ordinal != expected_tile_ordinal_ ||
          tile.grid_fingerprint != grid_fingerprint_)
        throw std::invalid_argument(
            "PsiFormer ECP tile has a stale grid identity or traversal ordinal");
      if (tile.virtual_particles.walkerCount() != sample_count_ ||
          tile.bare_weights.size() != tile.virtual_particles.size() ||
          tile.complete_ratios.size() != tile.virtual_particles.size())
        throw std::invalid_argument(
            "PsiFormer ECP tile has inconsistent walker or knot extents");
      if (consumed_point_count_ > expected_point_count_ ||
          tile.virtual_particles.size() >
              expected_point_count_ - consumed_point_count_)
        throw std::invalid_argument(
            "PsiFormer ECP tile exceeds the declared traversal point count");

      bool matching_stamp = false;
      for (const auto& stamp : tile.evaluation_stamps)
        matching_stamp = matching_stamp || stamp.matches(
            model_state_.get(), static_cast<std::uint64_t>(parameter_version_));
      if (!matching_stamp)
        throw std::runtime_error(
            "PsiFormer ECP tile lacks the bound value-evaluation stamp");

      const std::size_t segment_offset = replay_segments_.size();
      const std::size_t stamp_offset = replay_evaluation_stamps_.size();
      if (capture_jvp_replay_)
        replay_evaluation_stamps_.insert(replay_evaluation_stamps_.end(),
                                         tile.evaluation_stamps.begin(),
                                         tile.evaluation_stamps.end());

      PsiFormerReadTransaction state_lock(*model_state_, replay_only_ && model_state_->deviceOnlyValueReads());
      requireBoundVersion();
      for (std::size_t segment_index = 0;
           segment_index < tile.virtual_particles.segmentCount(); ++segment_index)
      {
        const VirtualParticleBatch::Slice slice =
            tile.virtual_particles.slice(segment_index);
        if (slice.walkerId() < 0 || slice.electronId() < 0 ||
            static_cast<std::size_t>(slice.walkerId()) >= sample_count_ ||
            static_cast<std::size_t>(slice.electronId()) >= electron_count_)
          throw std::out_of_range(
              "PsiFormer ECP tile names an invalid walker or electron");
        const std::size_t sample = static_cast<std::size_t>(slice.walkerId());
        if (capture_jvp_replay_)
          replay_segments_.push_back({
              consumed_point_count_ + slice.flatOffset(), slice.size(), sample,
              static_cast<std::size_t>(slice.electronId()), slice.isOnSphere(),
              slice.sourceCenterId() == VirtualParticleBatch::NO_SOURCE
                  ? static_cast<std::size_t>(-1)
                  : static_cast<std::size_t>(slice.sourceCenterId())});
        for (std::size_t local_point = 0; local_point < slice.size(); ++local_point)
        {
          const std::size_t point = slice.flatOffset() + local_point;
          const auto bare         = tile.bare_weights[point];
          const auto ratio        = tile.complete_ratios[point];
          requireFiniteValue(bare, "bare weight");
          requireFiniteValue(ratio, "complete ratio");
          if (std::imag(bare) != 0)
            throw std::invalid_argument(
                "PsiFormer ECP bare quadrature weight must be real");
          const auto weighted_value = bare * ratio;
          const wftrain::DerivativeValue weight{
              static_cast<double>(std::real(weighted_value)),
              static_cast<double>(std::imag(weighted_value))};
          requireFinite(weight, "complete weight");
          sample_weight_sums_[sample] += weight;

          if (capture_jvp_replay_)
          {
            wftrain::NonLocalECPReplayPoint& replay_point =
                replay_points_[consumed_point_count_ + point];
            replay_point.sample = sample;
            replay_point.electron = static_cast<std::size_t>(slice.electronId());
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              replay_point.replacement_position[dimension] = static_cast<double>(
                  slice.absolutePosition(local_point)[dimension]);
            replay_point.bare_weight = {
                static_cast<double>(std::real(bare)),
                static_cast<double>(std::imag(bare))};
            replay_point.complete_ratio = {
                static_cast<double>(std::real(ratio)),
                static_cast<double>(std::imag(ratio))};
          }

          if (replay_only_)
            continue;
          setVirtualPositions(sample, static_cast<std::size_t>(slice.electronId()),
                              slice.absolutePosition(local_point));
          const pf::DirectScoreResult score = evaluateScoreResponse();
          if (score.parameter_version != parameter_version_)
            throw std::logic_error(
                "PsiFormer ECP virtual score has the wrong parameter version");
          for (std::size_t channel = 0; channel < channel_count_; ++channel)
            accumulateResponse(
                channel, coefficients_[channel][sample] * weight,
                score.parameter_score);
        }
      }
      if (capture_jvp_replay_)
        replay_tiles_.push_back({
            tile.tile_ordinal, consumed_point_count_, tile.virtual_particles.size(),
            segment_offset, replay_segments_.size() - segment_offset,
            stamp_offset, replay_evaluation_stamps_.size() - stamp_offset,
            tile.virtual_particles.fingerprint()});
      consumed_point_count_ = checkedBatchMemoryAdd(
          consumed_point_count_, tile.virtual_particles.size(),
          "PsiFormer ECP consumed point count");
      ++expected_tile_ordinal_;
    }
    catch (...)
    {
      abort();
      throw;
    }
  }

  /// Fold the reference subtraction and publish canonical chunks outside the model lock.
  void end() override
  {
    try
    {
      requireActive("end");
      if (consumed_point_count_ != expected_point_count_)
        throw std::logic_error(
            "PsiFormer ECP traversal ended before consuming every declared point");
      if (capture_jvp_replay_ &&
          (replay_points_.size() != expected_point_count_ ||
           (expected_point_count_ != 0 && replay_tiles_.empty())))
        throw std::logic_error(
            "PsiFormer ECP replay did not capture the complete traversal");
      {
        PsiFormerReadTransaction state_lock(*model_state_, replay_only_ && model_state_->deviceOnlyValueReads());
        requireBoundVersion();
        for (std::size_t sample = 0; !replay_only_ && sample < sample_count_; ++sample)
        {
          setReferencePositions(sample);
          const pf::DirectScoreResult score = evaluateScoreResponse();
          if (score.parameter_version != parameter_version_)
            throw std::logic_error(
                "PsiFormer ECP reference score has the wrong parameter version");
          for (std::size_t channel = 0; channel < channel_count_; ++channel)
            accumulateResponse(
                channel,
                -coefficients_[channel][sample] * sample_weight_sums_[sample],
                score.parameter_score);
        }
      }

      if (replay_only_)
      {} // Only releaseJVPReplay publishes this transaction's compact result.
      else if (sample_count_ == 0)
        sink_->endEmptyBatch();
      else
      {
        for (const auto& chunk : chunk_plan_.chunks())
          for (std::size_t channel = 0; channel < channel_count_; ++channel)
          {
            const auto* values = channel_accumulators_[channel].data() +
                chunk.parameter_offset;
            sink_->add(channel, {chunk, {values, chunk.count}});
          }
        sink_->end();
      }
      sink_  = nullptr;
      state_ = State::COMPLETE;
    }
    catch (...)
    {
      abort();
      throw;
    }
  }

  /// Poison an active transaction while retaining reusable allocated storage.
  void abort() noexcept override
  {
    if (sink_ && sink_->state() == wftrain::DerivativeSinkState::ACTIVE)
      sink_->abort();
    sink_  = nullptr;
    state_ = State::POISONED;
  }

  /// Transfer the complete immutable accepted-grid replay exactly once.
  std::unique_ptr<wftrain::NonLocalECPJVPReplay> releaseJVPReplay() override
  {
    if (state_ != State::COMPLETE)
      throw std::logic_error("PsiFormer ECP replay requires a complete transaction");
    if (!capture_jvp_replay_)
      throw std::logic_error("PsiFormer ECP replay capture was not requested");
    if (replay_released_)
      throw std::logic_error("PsiFormer ECP replay was already released");
    replay_released_ = true;
    return std::make_unique<PsiFormerNonLocalECPReplay>(
        schema_->providerId(), schema_->fingerprint(), parameter_version_,
        batch_ordinal_, sample_offset_, sample_count_, electron_count_,
        grid_fingerprint_, std::move(positions_), std::move(replay_points_),
        std::move(replay_segments_), std::move(replay_tiles_),
        std::move(replay_evaluation_stamps_));
  }

  /// Report the exact O(P)+O(B) retained capacities and live allocation identity.
  wftrain::StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    auto current = measureStorageDiagnostics();
    if (current.storage_fingerprint != baseline_storage_.storage_fingerprint)
      current.allocation_generation = checkedBatchMemoryAdd(
          baseline_storage_.allocation_generation, std::size_t{1},
          "PsiFormer ECP allocation generation");
    return current;
  }

private:
  enum class State
  {
    IDLE,
    ACTIVE,
    COMPLETE,
    POISONED
  };

  /// Allocate one checked complex canonical parameter vector.
  static std::vector<wftrain::DerivativeValue> makeParameterVector(
      std::size_t parameter_count)
  {
    checkedBatchMemoryMultiply(parameter_count,
                               sizeof(wftrain::DerivativeValue),
                               "PsiFormer ECP parameter channel");
    return std::vector<wftrain::DerivativeValue>(parameter_count);
  }

  /// Allocate one checked B-sized complex scalar row.
  static std::vector<wftrain::DerivativeValue> makeSampleVector(
      std::size_t sample_count)
  {
    checkedBatchMemoryMultiply(sample_count,
                               sizeof(wftrain::DerivativeValue),
                               "PsiFormer ECP sample scalars");
    return std::vector<wftrain::DerivativeValue>(sample_count);
  }

  /// Reject nonfinite complex full-precision input before arithmetic.
  static void requireFinite(wftrain::DerivativeValue value,
                            const char* description)
  {
    if (!psiformer::determinant::isFiniteReal(value.real()) ||
        !psiformer::determinant::isFiniteReal(value.imag()))
      throw std::invalid_argument(std::string("PsiFormer ECP ") + description +
                                  " must be finite");
  }

  /// Convert and validate the build-selected QMCPACK scalar domain.
  static void requireFiniteValue(QMCTraits::ValueType value,
                                 const char* description)
  {
    requireFinite({static_cast<double>(std::real(value)),
                   static_cast<double>(std::imag(value))},
                  description);
  }

  /// Require the only legal in-flight lifecycle state.
  void requireActive(const char* operation) const
  {
    if (state_ != State::ACTIVE || sink_ == nullptr)
      throw std::logic_error(std::string("PsiFormer ECP ") + operation +
                             " requires an active transaction");
  }

  /// Reject parameter publication after the consumer was prepared.
  void requireBoundVersion() const
  {
    if (model_state_->model.p.version() != parameter_version_)
      throw std::runtime_error(
          "PsiFormer ECP consumer parameter version is stale");
  }

  /// Copy one immutable reference configuration into the reusable score tape.
  void setReferencePositions(std::size_t sample) const
  {
    const double* positions =
        positions_.data() + sample * electron_count_ * std::size_t{3};
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      std::copy_n(positions, device_positions_.size(), device_positions_.begin());
      return;
    }
#endif
    score_workspace_->setPositions(
        pf::GeometryPositionView::interleaved(positions, electron_count_));
  }

  /// Use the actual accepted quadrature positions in a leased numerical score call.
  pf::DirectScoreResult evaluateScoreResponse() const
  {
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      const auto& score = device_products_->score(device_positions_.data(), parameter_version_);
      pf::DirectScoreResult result;
      result.parameter_score = {score.data(), score.size()};
      result.parameter_version = parameter_version_;
      return result;
    }
#endif
    return model_state_->direct_score_executor.evaluate(*score_workspace_);
  }

  /// Copy the reference and then overwrite exactly one electron position.
  void setVirtualPositions(std::size_t sample,
                           std::size_t electron,
                           const QMCTraits::PosType& position) const
  {
    setReferencePositions(sample);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double coordinate = static_cast<double>(position[dimension]);
      if (!psiformer::determinant::isFiniteReal(coordinate))
        throw std::invalid_argument(
            "PsiFormer ECP virtual position must be finite");
#ifdef ENABLE_CUDA
      if (device_products_)
      {
        device_positions_[3 * electron + dimension] = coordinate;
        continue;
      }
#endif
      score_workspace_->setPosition(electron, dimension, coordinate);
    }
  }

  /// Fold one real score vector into one complex contraction channel.
  void accumulateResponse(std::size_t channel,
                          wftrain::DerivativeValue coefficient,
                          const pf::DirectParameterScoreView& response) const
  {
    if (response.size != schema_->parameterCount())
      throw std::logic_error(
          "PsiFormer ECP score response has the wrong parameter extent");
    auto& destination = channel_accumulators_[channel];
    for (std::size_t parameter = 0; parameter < response.size; ++parameter)
      destination[parameter] += coefficient * response[parameter];
  }

  /// Mix one allocation identity into a deterministic process-local fingerprint.
  static void mixFingerprint(std::size_t& fingerprint, std::size_t value) noexcept
  {
    fingerprint ^= value + std::size_t{0x9e3779b9U} + (fingerprint << 6) +
        (fingerprint >> 2);
  }

  /// Measure all retained numeric owners without changing their capacities.
  wftrain::StreamingDerivativeStorageDiagnostics measureStorageDiagnostics() const
  {
    const std::size_t parameter_count = schema_->parameterCount();
    const std::size_t score_bytes = replay_only_ ? 0 : model_state_->numerical_tile_backend
        ? checkedBatchMemoryMultiply(checkedBatchMemoryMultiply(std::size_t{7}, parameter_count,
            "PsiFormer ECP device parameter vectors"), sizeof(double), "PsiFormer ECP device parameter bytes")
        : score_workspace_->scoreStorageBytes();
    std::size_t channel_bytes         = 0;
    for (const auto& channel : channel_accumulators_)
      channel_bytes = checkedBatchMemoryAdd(
          channel_bytes,
          checkedBatchMemoryMultiply(channel.capacity(),
                                     sizeof(wftrain::DerivativeValue),
                                     "PsiFormer ECP channel capacity"),
          "PsiFormer ECP channel storage");
    const std::size_t position_bytes = checkedBatchMemoryMultiply(
        positions_.capacity(), sizeof(double),
        "PsiFormer ECP reference-position capacity");
    std::size_t auxiliary_bytes = checkedBatchMemoryMultiply(
        sample_weight_sums_.capacity(), sizeof(wftrain::DerivativeValue),
        "PsiFormer ECP weight-sum capacity");
    for (const auto& coefficients : coefficients_)
      auxiliary_bytes = checkedBatchMemoryAdd(
          auxiliary_bytes,
          checkedBatchMemoryMultiply(coefficients.capacity(),
                                     sizeof(wftrain::DerivativeValue),
                                     "PsiFormer ECP coefficient capacity"),
          "PsiFormer ECP sample auxiliary storage");
    const std::size_t replay_bytes = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(
            checkedBatchMemoryMultiply(
                replay_points_.capacity(),
                sizeof(wftrain::NonLocalECPReplayPoint),
                "PsiFormer ECP replay point capacity"),
            checkedBatchMemoryMultiply(
                replay_segments_.capacity(),
                sizeof(wftrain::NonLocalECPReplaySegment),
                "PsiFormer ECP replay segment capacity"),
            "PsiFormer ECP replay point and segment storage"),
        checkedBatchMemoryAdd(
            checkedBatchMemoryMultiply(
                replay_tiles_.capacity(),
                sizeof(wftrain::NonLocalECPReplayTile),
                "PsiFormer ECP replay tile capacity"),
            checkedBatchMemoryMultiply(
                replay_evaluation_stamps_.capacity(),
                sizeof(WaveFunctionComponent::EvaluationStamp),
                "PsiFormer ECP replay stamp capacity"),
            "PsiFormer ECP replay provenance storage"),
        "PsiFormer ECP replay storage");
    auxiliary_bytes = checkedBatchMemoryAdd(
        auxiliary_bytes, replay_bytes, "PsiFormer ECP auxiliary and replay storage");
    std::size_t workspace_bytes = 0;
#ifdef ENABLE_CUDA
    if (device_products_)
    {
      workspace_bytes = checkedBatchMemoryAdd(device_products_->deviceWorkspaceBytes(),
          device_products_->hostWorkspaceBytes(), "PsiFormer ECP device product storage");
      workspace_bytes = checkedBatchMemoryAdd(workspace_bytes,
          checkedBatchMemoryMultiply(device_positions_.capacity(), sizeof(double), "PsiFormer ECP device coordinates"),
          "PsiFormer ECP device workspace");
    }
    else
#endif
      workspace_bytes = score_workspace_ ? score_workspace_->vectorStorageBytes() : 0;
    const std::size_t parameter_bytes = checkedBatchMemoryAdd(
        score_bytes, channel_bytes, "PsiFormer ECP parameter scratch");
    const std::size_t retained_bytes = checkedBatchMemoryAdd(
        checkedBatchMemoryAdd(workspace_bytes, channel_bytes,
                              "PsiFormer ECP evaluator and channels"),
        checkedBatchMemoryAdd(position_bytes, auxiliary_bytes,
                              "PsiFormer ECP sample storage"),
        "PsiFormer ECP retained storage");

    std::size_t fingerprint = std::size_t{1469598103U};
    if (score_workspace_)
      mixFingerprint(fingerprint, score_workspace_->storageFingerprint());
#ifdef ENABLE_CUDA
    if (device_products_)
      mixFingerprint(fingerprint, reinterpret_cast<std::uintptr_t>(device_products_.get()));
#endif
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(positions_.data()));
    mixFingerprint(fingerprint, positions_.capacity());
    for (const auto& channel : channel_accumulators_)
    {
      mixFingerprint(fingerprint,
                     reinterpret_cast<std::uintptr_t>(channel.data()));
      mixFingerprint(fingerprint, channel.capacity());
    }
    for (const auto& coefficients : coefficients_)
    {
      mixFingerprint(fingerprint,
                     reinterpret_cast<std::uintptr_t>(coefficients.data()));
      mixFingerprint(fingerprint, coefficients.capacity());
    }
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(sample_weight_sums_.data()));
    mixFingerprint(fingerprint, sample_weight_sums_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(replay_points_.data()));
    mixFingerprint(fingerprint, replay_points_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(replay_segments_.data()));
    mixFingerprint(fingerprint, replay_segments_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(replay_tiles_.data()));
    mixFingerprint(fingerprint, replay_tiles_.capacity());
    mixFingerprint(fingerprint,
                   reinterpret_cast<std::uintptr_t>(replay_evaluation_stamps_.data()));
    mixFingerprint(fingerprint, replay_evaluation_stamps_.capacity());
    if (fingerprint == 0)
      fingerprint = 1;

    return {parameter_count, sample_count_, model_state_->numerical_tile_backend ? std::size_t{7} : std::size_t{1}, MAXIMUM_CHANNELS,
            parameter_bytes, workspace_bytes, position_bytes,
            auxiliary_bytes, 0, retained_bytes, 1, fingerprint};
  }

  std::shared_ptr<PsiFormerSharedState> model_state_;
  std::shared_ptr<const wftrain::StructuredParameterSchema> schema_;
  std::size_t parameter_version_      = 0;
  bool capture_jvp_replay_            = false;
  bool replay_only_                   = false;
  std::size_t batch_ordinal_          = 0;
  std::size_t sample_offset_          = 0;
  std::size_t sample_count_           = 0;
  std::size_t electron_count_         = 0;
  std::vector<double> positions_;
  wftrain::ParameterChunkPlan chunk_plan_;
  mutable std::unique_ptr<pf::DirectScoreWorkspace> score_workspace_;
#ifdef ENABLE_CUDA
  std::unique_ptr<psiformer::device::PreparedDeviceProductSession> device_products_;
  mutable std::vector<double> device_positions_;
#endif
  mutable std::array<std::vector<wftrain::DerivativeValue>, MAXIMUM_CHANNELS>
      channel_accumulators_;
  std::array<std::vector<wftrain::DerivativeValue>, MAXIMUM_CHANNELS>
      coefficients_;
  std::vector<wftrain::DerivativeValue> sample_weight_sums_;
  std::vector<wftrain::NonLocalECPReplayPoint> replay_points_;
  std::vector<wftrain::NonLocalECPReplaySegment> replay_segments_;
  std::vector<wftrain::NonLocalECPReplayTile> replay_tiles_;
  std::vector<WaveFunctionComponent::EvaluationStamp> replay_evaluation_stamps_;
  wftrain::DerivativeStreamDescriptor stream_descriptor_;
  wftrain::ParameterReductionSink* sink_ = nullptr;
  std::size_t channel_count_             = 0;
  std::size_t expected_tile_ordinal_     = 0;
  std::size_t expected_point_count_      = 0;
  std::size_t consumed_point_count_      = 0;
  std::uint64_t grid_fingerprint_        = 0;
  State state_                           = State::IDLE;
  bool replay_released_                  = false;
  wftrain::StreamingDerivativeStorageDiagnostics baseline_storage_;
};

/** Crowd-owned mutable storage.  ResourceCollection cloning recreates scratch
 * against the same immutable/versioned model state without copying buffers. */
struct PsiFormerWF::PsiFormerMultiWalkerResource : public Resource
{
  /// Identify which of the two statically typed ratio vectors owns the arena.
  enum class RatioArenaKind : std::uint8_t
  {
    NONE,
    PSI_VALUE,
    LOG_VALUE
  };

  /** Provide checked scalar access without reinterpreting one vector's bytes as
   * the other scalar type.  LOG_VALUE is a storage representation here: its
   * entries still contain ordinary wavefunction ratios, not logarithms. */
  struct RatioArenaView
  {
    RatioArenaKind kind       = RatioArenaKind::NONE;
    PsiValue* psi_value_data  = nullptr;
    LogValue* log_value_data  = nullptr;
    std::size_t extent        = 0;

    /// Load one public ratio, rejecting a non-real value in a real build.
    PsiValue load(std::size_t index) const
    {
      if (index >= extent)
        throw std::out_of_range("PsiFormer ratio-arena read exceeds its prepared extent");
      switch (kind)
      {
      case RatioArenaKind::PSI_VALUE:
        if (!psi_value_data || log_value_data)
          throw std::logic_error("PsiFormer PsiValue ratio arena has invalid typed storage");
        return psi_value_data[index];
      case RatioArenaKind::LOG_VALUE:
        if (!log_value_data || psi_value_data)
          throw std::logic_error("PsiFormer LogValue ratio arena has invalid typed storage");
        return fromLogValue(log_value_data[index]);
      case RatioArenaKind::NONE:
        break;
      }
      throw std::logic_error("PsiFormer ratio arena has no active typed storage");
    }

    /** Load after checked Phase-B validation without any throwing operation.
     * The caller must have proved the arena kind, pointers, extent, and (for a
     * real build using LOG_VALUE storage) zero imaginary component. */
    PsiValue loadUnchecked(std::size_t index) const noexcept
    {
      switch (kind)
      {
      case RatioArenaKind::PSI_VALUE:
        return psi_value_data[index];
      case RatioArenaKind::LOG_VALUE:
        return fromLogValueUnchecked(log_value_data[index]);
      case RatioArenaKind::NONE:
        return PsiValue{};
      }
      return PsiValue{};
    }

    /// Store one public ratio in the selected typed vector without allocation.
    void store(std::size_t index, PsiValue value) const
    {
      if (index >= extent)
        throw std::out_of_range("PsiFormer ratio-arena write exceeds its prepared extent");
      switch (kind)
      {
      case RatioArenaKind::PSI_VALUE:
        if (!psi_value_data || log_value_data)
          throw std::logic_error("PsiFormer PsiValue ratio arena has invalid typed storage");
        psi_value_data[index] = value;
        return;
      case RatioArenaKind::LOG_VALUE:
        if (!log_value_data || psi_value_data)
          throw std::logic_error("PsiFormer LogValue ratio arena has invalid typed storage");
        log_value_data[index] = toLogValue(value);
        return;
      case RatioArenaKind::NONE:
        break;
      }
      throw std::logic_error("PsiFormer ratio arena has no active typed storage");
    }

  private:
    /// Convert through full-precision real and imaginary components explicitly.
    static LogValue toLogValue(PsiValue value) noexcept
    {
      return {static_cast<FullPrecRealType>(std::real(value)),
              static_cast<FullPrecRealType>(std::imag(value))};
    }

    /// Preserve complex ratios and reject accidental phase loss in real builds.
    template<class Target = PsiValue>
    static Target fromLogValue(LogValue value)
    {
      if constexpr (IsComplex_t<Target>::value)
        return Target(static_cast<typename Target::value_type>(std::real(value)),
                      static_cast<typename Target::value_type>(std::imag(value)));
      else
      {
        if (std::imag(value) != FullPrecRealType{0})
          throw std::domain_error(
              "PsiFormer complex ratio cannot be published by a real build");
        return static_cast<Target>(std::real(value));
      }
    }

    /// Convert data already proved representable by the checked Phase-B load.
    template<class Target = PsiValue>
    static Target fromLogValueUnchecked(LogValue value) noexcept
    {
      if constexpr (IsComplex_t<Target>::value)
        return Target(static_cast<typename Target::value_type>(std::real(value)),
                      static_cast<typename Target::value_type>(std::imag(value)));
      else
        return static_cast<Target>(std::real(value));
    }
  };

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
    if (context.plan && model_state->cpu_target_vgl.targets)
      throw std::invalid_argument("PsiFormer CPU target VGL does not support planned crowd resources");
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
  const PsiValue* prepared_staged_value_ratios_data             = nullptr;
  const LogValue* prepared_staged_log_ratios_data               = nullptr;
  std::size_t prepared_configuration_identities_capacity       = 0;
  std::size_t prepared_staged_signs_capacity                   = 0;
  std::size_t prepared_staged_log_magnitudes_capacity          = 0;
  std::size_t prepared_walker_indices_capacity                  = 0;
  std::size_t prepared_preservation_flags_capacity              = 0;
  std::size_t prepared_active_electrons_capacity                = 0;
  std::size_t prepared_staged_gradients_capacity                = 0;
  std::size_t prepared_staged_value_ratios_size                 = 0;
  std::size_t prepared_staged_log_ratios_size                   = 0;
  std::size_t prepared_staged_value_ratios_capacity             = 0;
  std::size_t prepared_staged_log_ratios_capacity               = 0;
  RatioArenaKind prepared_ratio_arena_kind = RatioArenaKind::NONE;
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
  std::vector<PsiValue> staged_value_ratios;
  /// Full-precision complex log ratios, also reused by narrower ratio calls.
  std::vector<LogValue> staged_log_ratios;
  /// Typed arena selected once during exact resource materialization.
  RatioArenaKind ratio_arena_kind = RatioArenaKind::NONE;
  /// Shared gradient publication staging for active and full spatial calls.
  std::vector<GradType> staged_gradients;
  /// Report CPU target crowd multiplicity once, without claiming an aggregate cap.
  bool cpu_target_vgl_reported = false;
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
    if (model_state->cpu_target_kinetic.targets)
      throw std::invalid_argument("CPU target kinetic profile supports streaming products, not legacy/planned crowd derivatives");
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

  /** Validate the common value-result metadata used through bounded prefixes.
   * Planned release retains physical extents, so runtime never resizes these
   * vectors and any size or allocation-identity drift fails closed. */
  void requireValueMetadataStaging(std::size_t live_walkers) const
  {
    if (!prepared_plan || !prepared_crowd_plan ||
        prepared_storage_fingerprint == 0 ||
        storageFingerprint() != prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer value metadata lost its prepared allocation identity");

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
                    "PsiFormer value-metadata log staging") !=
            staging.log_magnitudes)
      throw std::logic_error(
          "PsiFormer value-metadata extents differ from the prepared plan");
    if (live_walkers > configuration_identities.size() ||
        live_walkers > staged_signs.size() ||
        live_walkers > staged_log_magnitudes.size())
      throw std::length_error(
          "PsiFormer value-metadata prefix exceeds prepared capacity");
  }

  /// Validate value metadata needed by a complete spatial publication.
  void requireFullVGLStaging(std::size_t live_walkers) const
  { requireValueMetadataStaging(live_walkers); }

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

  /** Validate both candidate ratio vectors against their exact prepared
   * identities and the single active scalar representation selected by policy. */
  void requireRatioArenaStaging(std::size_t live_walkers) const
  {
    if (!prepared_plan || !prepared_crowd_plan ||
        prepared_storage_fingerprint == 0 ||
        storageFingerprint() != prepared_storage_fingerprint)
      throw std::logic_error(
          "PsiFormer ratio arena lost its prepared allocation identity");

    const auto& staging = prepared_crowd_plan->publication_storage;
    const auto& capacity = prepared_crowd_plan->publication_staging;
    const RatioArenaKind expected_kind = staging.ratios == 0
        ? RatioArenaKind::NONE
        : (capacity.full_vgl ? RatioArenaKind::LOG_VALUE
                             : RatioArenaKind::PSI_VALUE);
    const std::size_t psi_value_bytes = vectorBytes(
        staged_value_ratios, "PsiFormer PsiValue ratio staging");
    const std::size_t log_value_bytes = vectorBytes(
        staged_log_ratios, "PsiFormer LogValue ratio staging");

    if (ratio_arena_kind != expected_kind ||
        prepared_ratio_arena_kind != expected_kind ||
        staged_value_ratios.data() !=
            prepared_staged_value_ratios_data ||
        staged_log_ratios.data() != prepared_staged_log_ratios_data ||
        staged_value_ratios.size() !=
            prepared_staged_value_ratios_size ||
        staged_log_ratios.size() != prepared_staged_log_ratios_size ||
        staged_value_ratios.capacity() !=
            prepared_staged_value_ratios_capacity ||
        staged_log_ratios.capacity() !=
            prepared_staged_log_ratios_capacity ||
        prepared_staged_value_ratios_size !=
            prepared_staged_value_ratios_capacity ||
        prepared_staged_log_ratios_size !=
            prepared_staged_log_ratios_capacity ||
        pf::checkedStorageSum(
            psi_value_bytes, log_value_bytes,
            "PsiFormer prepared ratio staging overflowed") != staging.ratios)
      throw std::logic_error(
          "PsiFormer ratio-arena identity differs from the prepared plan");

    std::size_t active_extent = 0;
    switch (expected_kind)
    {
    case RatioArenaKind::NONE:
      if (psi_value_bytes != 0 || log_value_bytes != 0)
        throw std::logic_error(
            "PsiFormer empty ratio arena retained typed storage");
      break;
    case RatioArenaKind::PSI_VALUE:
      if (psi_value_bytes != staging.ratios || log_value_bytes != 0)
        throw std::logic_error(
            "PsiFormer PsiValue ratio arena has mixed typed storage");
      active_extent = staged_value_ratios.size();
      break;
    case RatioArenaKind::LOG_VALUE:
      if (log_value_bytes != staging.ratios || psi_value_bytes != 0)
        throw std::logic_error(
            "PsiFormer LogValue ratio arena has mixed typed storage");
      active_extent = staged_log_ratios.size();
      break;
    }
    if (expected_kind != RatioArenaKind::NONE &&
        active_extent != capacity.reserve_walkers)
      throw std::logic_error(
          "PsiFormer ratio arena does not contain exactly one entry per reserve walker");
    if (live_walkers > active_extent)
      throw std::length_error(
          "PsiFormer ratio-arena prefix exceeds prepared capacity");
  }

  /// Return mutable typed access after exact arena validation has succeeded.
  RatioArenaView ratioArenaView() noexcept
  {
    switch (ratio_arena_kind)
    {
    case RatioArenaKind::PSI_VALUE:
      return {ratio_arena_kind, staged_value_ratios.data(), nullptr,
              staged_value_ratios.size()};
    case RatioArenaKind::LOG_VALUE:
      return {ratio_arena_kind, nullptr, staged_log_ratios.data(),
              staged_log_ratios.size()};
    case RatioArenaKind::NONE:
      return {};
    }
    return {};
  }

  /// Validate all scratch consumed by a planned value-only ratio producer.
  RatioArenaView requireCalcRatioStaging(std::size_t live_walkers)
  {
    requireValueMetadataStaging(live_walkers);
    requireRatioArenaStaging(live_walkers);
    return ratioArenaView();
  }

  /// Validate all scratch consumed by a planned ratio-and-gradient producer.
  RatioArenaView requireRatioGradientStaging(std::size_t live_walkers)
  {
    requireValueMetadataStaging(live_walkers);
    requireActiveGradientStaging(live_walkers);
    requireRatioArenaStaging(live_walkers);
    return ratioArenaView();
  }

  /// Revalidate retained proposal scratch before a planned resolution commit.
  RatioArenaView requireSingleResolutionStaging(std::size_t live_walkers)
  {
    requireValueMetadataStaging(live_walkers);
    requireRatioArenaStaging(live_walkers);
    return ratioArenaView();
  }

  /// Validate every prepared prefix used by a selected full-VGL proposal.
  void requireSelectedProposalStaging(std::size_t live_walkers) const
  {
    requireFullVGLStaging(live_walkers);
    requireRatioArenaStaging(live_walkers);
    const pf::ResourceStagingStorageRequirement& staging =
        prepared_crowd_plan->publication_storage;
    if (vectorBytes(batch_slots,
                    "PsiFormer selected batch-slot staging") !=
            staging.batch_slots ||
        vectorBytes(walker_indices,
                    "PsiFormer selected walker-index staging") !=
            staging.walker_indices ||
        ratio_arena_kind != RatioArenaKind::LOG_VALUE)
      throw std::logic_error(
          "PsiFormer selected staging capacities differ from the prepared plan");
    if (live_walkers > batch_slots.size() ||
        live_walkers > walker_indices.size())
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
      if (model_state->cpu_target_kinetic.targets)
        throw std::invalid_argument("CPU target kinetic profile does not admit a planned FULL kinetic tape");
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
    if (staging.ratios == 0)
      ratio_arena_kind = RatioArenaKind::NONE;
    else if (crowd_plan.publication_staging.full_vgl)
    {
      const std::size_t exact_ratio_bytes = pf::checkedStorageBytes<LogValue>(
          crowd_plan.publication_staging.reserve_walkers,
          "PsiFormer LogValue ratio-arena extent overflowed");
      if (staging.ratios != exact_ratio_bytes)
        throw std::length_error(
            "PsiFormer LogValue ratio arena is not exactly one entry per reserve walker");
      ratio_arena_kind = RatioArenaKind::LOG_VALUE;
      staged_log_ratios = makeExactVector<LogValue>(
          exact_ratio_bytes, "PsiFormer log-ratio staging");
    }
    else
    {
      const std::size_t exact_ratio_bytes = pf::checkedStorageBytes<PsiValue>(
          crowd_plan.publication_staging.reserve_walkers,
          "PsiFormer PsiValue ratio-arena extent overflowed");
      if (staging.ratios != exact_ratio_bytes)
        throw std::length_error(
            "PsiFormer PsiValue ratio arena is not exactly one entry per reserve walker");
      ratio_arena_kind = RatioArenaKind::PSI_VALUE;
      staged_value_ratios = makeExactVector<PsiValue>(
          exact_ratio_bytes, "PsiFormer value-ratio staging");
    }
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
    prepared_staged_value_ratios_data        = staged_value_ratios.data();
    prepared_staged_log_ratios_data          = staged_log_ratios.data();
    prepared_configuration_identities_capacity =
        configuration_identities.capacity();
    prepared_staged_signs_capacity = staged_signs.capacity();
    prepared_staged_log_magnitudes_capacity =
        staged_log_magnitudes.capacity();
    prepared_walker_indices_capacity     = walker_indices.capacity();
    prepared_preservation_flags_capacity = preservation_flags.capacity();
    prepared_active_electrons_capacity   = active_electrons.capacity();
    prepared_staged_gradients_capacity   = staged_gradients.capacity();
    prepared_staged_value_ratios_size     = staged_value_ratios.size();
    prepared_staged_log_ratios_size       = staged_log_ratios.size();
    prepared_staged_value_ratios_capacity = staged_value_ratios.capacity();
    prepared_staged_log_ratios_capacity   = staged_log_ratios.capacity();
    prepared_ratio_arena_kind             = ratio_arena_kind;
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
    prepared_staged_value_ratios_data =
        candidate.prepared_staged_value_ratios_data;
    prepared_staged_log_ratios_data =
        candidate.prepared_staged_log_ratios_data;
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
    prepared_staged_value_ratios_size =
        candidate.prepared_staged_value_ratios_size;
    prepared_staged_log_ratios_size =
        candidate.prepared_staged_log_ratios_size;
    prepared_staged_value_ratios_capacity =
        candidate.prepared_staged_value_ratios_capacity;
    prepared_staged_log_ratios_capacity =
        candidate.prepared_staged_log_ratios_capacity;
    prepared_ratio_arena_kind = candidate.prepared_ratio_arena_kind;
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
    ratio_arena_kind               = candidate.ratio_arena_kind;
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
    prepared_staged_value_ratios_data        = nullptr;
    prepared_staged_log_ratios_data          = nullptr;
    prepared_configuration_identities_capacity = 0;
    prepared_staged_signs_capacity             = 0;
    prepared_staged_log_magnitudes_capacity    = 0;
    prepared_walker_indices_capacity           = 0;
    prepared_preservation_flags_capacity       = 0;
    prepared_active_electrons_capacity         = 0;
    prepared_staged_gradients_capacity         = 0;
    prepared_staged_value_ratios_size           = 0;
    prepared_staged_log_ratios_size             = 0;
    prepared_staged_value_ratios_capacity       = 0;
    prepared_staged_log_ratios_capacity         = 0;
    prepared_ratio_arena_kind                   = RatioArenaKind::NONE;
    ratio_arena_kind                            = RatioArenaKind::NONE;
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
    mix(static_cast<std::uintptr_t>(ratio_arena_kind));
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
                             std::uint64_t initialization_seed,
                             const psiformer::ExecutionEnvironment& environment,
                             const std::array<double, 3>& reduced_twist,
                             const std::array<double, 3>& cartesian_twist,
                             const std::vector<double>& physical_source,
                             const std::vector<std::string>& physical_species)
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
  mixPeriodicProfile(hash, environment);
  if (environment.periodic_profile)
  {
    for (double value : physical_source) mix_double(value);
    for (const auto& name : physical_species) mix_string(name);
  }
  // Keep OPEN restart fingerprints backward compatible while binding periodic
  // state to the exact feature policy and simulation cell.
  if (environment.boundary == psiformer::BoundaryCondition::PERIODIC)
  {
    mix_integer(static_cast<std::uint64_t>(environment.boundary));
    mix_integer(static_cast<std::uint64_t>(environment.geometry_feature_policy));
    for (bool periodic : environment.periodic_axes)
      mix_integer(periodic ? 1 : 0);
    for (const auto& vector : environment.lattice_vectors)
      for (double component : vector)
        mix_double(component);
  }
  if (std::any_of(reduced_twist.begin(), reduced_twist.end(),
                  [](double component) { return component != 0.0; }))
  {
    mix_string("outer_twist_phase_v1");
    for (double component : reduced_twist)
      mix_double(component);
    for (double component : cartesian_twist)
      mix_double(component);
  }
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

/// Stream complete physical target slices into an output sink without a full tape.
template<class Sink>
pf::DirectValueResult streamCpuTargetVGL(const pf::DirectSpatialExecutor& executor,
    pf::DirectSpatialWorkspace& workspace, std::size_t max_targets, std::size_t version,
    const ParticleSet& particles, int replaced_particle, const ParticleSet::PosType* replacement,
    Sink&& sink)
{
  const std::size_t n = executor.layout()->electronCount();
  if (static_cast<std::size_t>(particles.getTotalNum()) != n || replaced_particle < -1 ||
      (replaced_particle >= 0 && static_cast<std::size_t>(replaced_particle) >= n))
    throw std::invalid_argument("PsiFormer CPU target VGL received incompatible physical positions");
  for (std::size_t e = 0; e < n; ++e)
  {
    const auto& position = static_cast<int>(e) == replaced_particle
        ? (replacement ? *replacement : particles.activeR(e)) : particles.R[e];
    for (std::size_t d = 0; d < 3; ++d)
      workspace.setPosition(e, d, position[d]);
  }
  pf::DirectValueResult value;
  for (std::size_t begin = 0; begin < n; begin += max_targets)
  {
    const std::size_t count = std::min(max_targets, n - begin);
    const auto tile = executor.evaluateTarget(workspace, begin, count);
    if (tile.parameter_version != version || tile.target_begin != begin || tile.target_count != count ||
        tile.gradient.size() != 3 * count || tile.lap_log.size() != count || tile.lap_ratio.size() != count ||
        tile.evaluation_generation != workspace.evaluationGeneration())
      throw std::logic_error("PsiFormer CPU target VGL returned inconsistent provenance");
    if (begin == 0)
      value = {tile.sign, tile.logabs, tile.value, tile.parameter_version};
    else if (tile.sign != value.sign || tile.logabs != value.logabs || tile.value != value.value)
      throw std::logic_error("PsiFormer CPU target VGL tiles disagree on the complete value");
    sink(tile);
  }
  return value;
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

} // namespace

// Load an exported model before entering the common optimizer-registration path.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::string parameters,
                         std::string configuration,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export,
                         psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                         psiformer::PsiFormerPrecisionPolicy precision)
    : PsiFormerWF(std::move(name),
                  std::make_shared<PsiFormerSharedState>(parameters, configuration,
                                                         accelerator_resolution, precision),
                  enable_optimization, std::move(selected_flat_indices), optimize_all,
                  std::move(optimized_parameter_export))
{}

// Load a periodic model only after binding its cell and ordered runtime ions.
PsiFormerWF::PsiFormerWF(std::string name,
                         std::string parameters,
                         std::string configuration,
                         const ParticleSet& electrons,
                         const ParticleSet& ions,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export,
                         std::array<double, 3> reduced_twist,
                         psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                         psiformer::PsiFormerPrecisionPolicy precision)
    : PsiFormerWF(std::move(name),
                  std::make_shared<PsiFormerSharedState>(parameters, configuration,
                                                        electrons, ions, reduced_twist,
                                                        accelerator_resolution, precision),
                  enable_optimization, std::move(selected_flat_indices), optimize_all,
                  std::move(optimized_parameter_export))
{
  bound_particle_set_ = &electrons;
}

// Construct a native model directly from initialized parameters and QMCPACK system metadata.
PsiFormerWF::PsiFormerWF(std::string name,
                         psiformer::InitializedPsiFormerParameters initialized_parameters,
                         const ParticleSet& electrons,
                         const ParticleSet& ions,
                         bool enable_optimization,
                         std::vector<std::size_t> selected_flat_indices,
                         bool optimize_all,
                         std::string optimized_parameter_export,
                         std::array<double, 3> reduced_twist,
                         psiformer::PsiFormerAcceleratorResolution accelerator_resolution,
                         psiformer::PsiFormerPrecisionPolicy precision)
    : PsiFormerWF(std::move(name),
                  std::make_shared<PsiFormerSharedState>(std::move(initialized_parameters), electrons, ions,
                                                        reduced_twist, accelerator_resolution, precision),
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
  if (model_state_->cpu_target_vgl.targets && enable_optimization)
    throw std::invalid_argument("PsiFormer CPU target VGL is inference-only and requires optimize=no");
  if (model_state_->precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE && enable_optimization)
    throw std::invalid_argument("Mixed PsiFormer precision is inference-only; optimize=yes is unsupported");
  if (!enable_optimization && !selected_flat_indices.empty())
    throw std::invalid_argument("PsiFormer optimize_indices requires optimize=yes");
  if (enable_optimization && model_state_->numerical_tile_backend &&
      (model_state_->accelerator_resolution.validated_operations & psiformer::allPsiFormerAcceleratorOperations()) !=
          psiformer::allPsiFormerAcceleratorOperations())
    throw std::invalid_argument("PsiFormer GPU optimization requires validated products and parameter publication");
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
                                model_state_->initialization_seed,
                                model_state_->execution_plan.environment(),
                                model_state_->twist_phase.reduced,
                                model_state_->twist_phase.cartesian,
                                model_state_->physical_source, model_state_->physical_species)
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
            << ", accelerator_backend="
            << psiformer::psiFormerAcceleratorBackendName(
                   model_state_->accelerator_resolution.backend)
            << ", accelerator_reason="
            << psiformer::psiFormerAcceleratorResolutionReasonName(
                   model_state_->accelerator_resolution.reason)
            << ", accelerator_fallbacks="
            << model_state_->accelerator_resolution.fallback_count
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
  if (model_state_->cpu_target_vgl.targets)
    app_log() << "    CPU target VGL: maxK=" << model_state_->cpu_target_vgl.targets
              << ", numerical_owner_bytes_per_clone=" << model_state_->cpu_target_vgl.owner_bytes
              << ", owner_cap_bytes_per_clone=" << model_state_->cpu_target_vgl.owner_budget
              << "; excludes typed G/L outputs, VALUE/ACTIVE/ECP/score/kinetic owners, model and allocator overhead"
              << "; no aggregate feasibility claim" << std::endl;
  if (model_state_->cpu_target_kinetic.targets)
    app_log() << "    CPU target streaming products: maxK=" << model_state_->cpu_target_kinetic.targets
              << ", numerical_kinetic_owner_bytes_per_shared_lane=" << model_state_->cpu_target_kinetic.owner_bytes
              << ", owner_cap_bytes_per_lane=" << model_state_->cpu_target_kinetic.owner_budget
              << "; objective-independent SCORE/LOCAL_ENERGY; inference unchanged; excludes score/channel/sample/ECP/model/allocator storage"
              << "; no aggregate feasibility claim" << std::endl;
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
  PsiFormerReadTransaction state_lock(*model_state_);
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
  input.storage_shape.pair_attention = static_cast<bool>(model_state_->execution_plan.environment().periodic_profile);
  if (const auto& profile = model_state_->execution_plan.environment().periodic_profile)
    input.storage_shape.pair_feature_width = profile->pair_feature_dimension;
  input.type_sizes = psiformer::makePsiFormerMemoryTypeSizes<
      ValueType, PsiValue, LogValue, GradType,
      SelectedDerivativeDelta::value_type, FullPrecRealType>();
  input.walker_buffer_alignment = QMC_SIMD_ALIGNMENT;
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
  if (!plan || !prepared_clone_batch_execution_plan_.sameBinding(plan))
    return false;
  if (!bound_particle_set_ ||
      prepared_bound_particle_set_ != bound_particle_set_)
    return false;

  const std::size_t electron_count =
      model_state_->execution_plan.modelShape().electrons();
  if (plan.plan().particleCount() != electron_count ||
      prepared_accepted_gradient_capacity_ != electron_count ||
      prepared_accepted_laplacian_capacity_ != electron_count ||
      prepared_proposed_gradient_capacity_ != electron_count ||
      prepared_proposed_laplacian_capacity_ != electron_count ||
      accepted_gradient_.isAttached() || accepted_laplacian_.isAttached() ||
      proposed_gradient_.isAttached() || proposed_laplacian_.isAttached() ||
      accepted_gradient_.data() != prepared_accepted_gradient_data_ ||
      accepted_laplacian_.data() != prepared_accepted_laplacian_data_ ||
      proposed_gradient_.data() != prepared_proposed_gradient_data_ ||
      proposed_laplacian_.data() != prepared_proposed_laplacian_data_ ||
      accepted_gradient_.size() != electron_count ||
      accepted_gradient_.capacity() != electron_count ||
      accepted_laplacian_.size() != electron_count ||
      accepted_laplacian_.capacity() != electron_count ||
      proposed_gradient_.size() != electron_count ||
      proposed_gradient_.capacity() != electron_count ||
      proposed_laplacian_.size() != electron_count ||
      proposed_laplacian_.capacity() != electron_count)
    return false;

  // The public preparation predicate is noexcept.  Treat malformed byte or
  // capacity evidence as an unprepared clone rather than allowing an
  // accounting helper's checked arithmetic to escape.
  try
  {
    const BatchExecutionRequirements& requirements =
        plan.plan().requirements();
    const bool scalar_selected = requirements.requires(
        BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
    const bool walker_buffer_selected =
        requirements.requires(BatchExecutionMode::BUFFER_READ) ||
        requirements.requires(BatchExecutionMode::BUFFER_WRITE);
    const psiformer::PsiFormerMemoryPolicyInput input =
        makeBatchMemoryPolicyInput();
    const pf::WalkerBufferLayout expected_walker_buffer_layout =
        walker_buffer_selected
        ? psiformer::makePsiFormerWalkerBufferLayout(input)
        : pf::WalkerBufferLayout{};
    if (prepared_walker_buffer_layout_ != expected_walker_buffer_layout ||
        (walker_buffer_selected &&
         prepared_walker_buffer_layout_.fingerprint == 0))
      return false;

    if (!scalar_selected)
      return !prepared_scalar_value_compatibility_ &&
          !direct_batch_workspace_ && scalar_value_publication_.empty() &&
          scalar_value_publication_.capacity() == 0 &&
          prepared_batch_workspace_identity_ == nullptr &&
          prepared_batch_storage_fingerprint_ == 0 &&
          prepared_batch_bytes_ == 0 &&
          prepared_scalar_value_publication_data_ == nullptr &&
          prepared_scalar_value_publication_size_ == 0 &&
          prepared_scalar_value_publication_capacity_ == 0;

    if (!prepared_scalar_value_compatibility_ || !direct_batch_workspace_ ||
        direct_batch_workspace_.get() !=
            prepared_batch_workspace_identity_ ||
        !direct_batch_workspace_->hasCapacityPlan())
      return false;

    const pf::DirectBatchCapacityPlan expected_capacity =
        psiformer::makePsiFormerScalarValueCapacityPlan(
            input, requirements,
            plan.plan().selectedCapacities());
    const pf::DirectBatchStorageRequirement expected_storage =
        pf::directBatchStorageRequirement(input.storage_shape,
                                          expected_capacity);
    const pf::DirectBatchStorageRequirement actual_storage =
        direct_batch_workspace_->actualStorage();
    const std::size_t expected_bytes = expected_storage.executionBytes();
    const std::size_t actual_bytes   = actual_storage.executionBytes();
    const std::size_t scalar_extent =
        input.scalar_value_logical_maximum;

    return sameDirectBatchCapacityPlan(
               direct_batch_workspace_->capacityPlan(), expected_capacity) &&
        sameDirectBatchExecutionStorage(actual_storage, expected_storage) &&
        actual_bytes == expected_bytes &&
        direct_batch_workspace_->vectorStorageBytes() == actual_bytes &&
        prepared_batch_bytes_ == actual_bytes &&
        direct_batch_workspace_->storageFingerprint(
            pf::DirectBatchMode::VALUE_ONLY) ==
            prepared_batch_storage_fingerprint_ &&
        prepared_batch_storage_fingerprint_ != 0 &&
        scalar_value_publication_.data() ==
            prepared_scalar_value_publication_data_ &&
        scalar_value_publication_.size() == scalar_extent &&
        scalar_value_publication_.capacity() == scalar_extent &&
        prepared_scalar_value_publication_size_ == scalar_extent &&
        prepared_scalar_value_publication_capacity_ == scalar_extent;
  }
  catch (...)
  {
    return false;
  }
}

// Check a prospective participant view completely before noexcept publication.
void PsiFormerWF::validateBatchExecutionPlanBinding(
    const BatchExecutionParticipantPlan& participant_plan) const
{
  if (participant_plan && model_state_->cpu_target_vgl.targets)
    throw std::invalid_argument("PsiFormer CPU target VGL does not support explicit batch plans");
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
  if (model_state_->cpu_target_kinetic.targets && plan.requirements().requires(BatchExecutionMode::KINETIC))
    throw std::invalid_argument("CPU target kinetic profile supports streaming products, not planned FULL kinetic derivatives");
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
    prepared_bound_particle_set_ = nullptr;
    prepared_walker_buffer_layout_ = {};
    prepared_accepted_gradient_data_ = nullptr;
    prepared_accepted_laplacian_data_ = nullptr;
    prepared_proposed_gradient_data_ = nullptr;
    prepared_proposed_laplacian_data_ = nullptr;
    prepared_accepted_gradient_capacity_ = 0;
    prepared_accepted_laplacian_capacity_ = 0;
    prepared_proposed_gradient_capacity_ = 0;
    prepared_proposed_laplacian_capacity_ = 0;
    prepared_scalar_value_compatibility_ = false;
    prepared_batch_workspace_identity_ = nullptr;
    prepared_batch_storage_fingerprint_ = 0;
    prepared_batch_bytes_ = 0;
    prepared_scalar_value_publication_data_ = nullptr;
    prepared_scalar_value_publication_size_ = 0;
    prepared_scalar_value_publication_capacity_ = 0;
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
    prepared_bound_particle_set_ = nullptr;
    prepared_walker_buffer_layout_ = {};
    prepared_accepted_gradient_data_ = nullptr;
    prepared_accepted_laplacian_data_ = nullptr;
    prepared_proposed_gradient_data_ = nullptr;
    prepared_proposed_laplacian_data_ = nullptr;
    prepared_accepted_gradient_capacity_ = 0;
    prepared_accepted_laplacian_capacity_ = 0;
    prepared_proposed_gradient_capacity_ = 0;
    prepared_proposed_laplacian_capacity_ = 0;
    prepared_scalar_value_compatibility_ = false;
    prepared_batch_workspace_identity_ = nullptr;
    prepared_batch_storage_fingerprint_ = 0;
    prepared_batch_bytes_ = 0;
    prepared_scalar_value_publication_data_ = nullptr;
    prepared_scalar_value_publication_size_ = 0;
    prepared_scalar_value_publication_capacity_ = 0;
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
  NativeTimerScope timer(phase_timers_.preparation);
  if (model_state_->cpu_target_vgl.targets)
    throw std::invalid_argument("PsiFormer CPU target VGL does not support planned clone storage");
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
  if (!bound_particle_set_)
    throw std::logic_error(
        "PsiFormer clone preparation requires a bound ParticleSet");
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

  const pf::DirectBatchCapacityPlan scalar_plan =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, plan.requirements(), plan.selectedCapacities());
  const bool scalar_value = plan.requirements().requires(
      BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const bool walker_buffer_selected =
      plan.requirements().requires(BatchExecutionMode::BUFFER_READ) ||
      plan.requirements().requires(BatchExecutionMode::BUFFER_WRITE);
  const pf::WalkerBufferLayout prepared_walker_buffer_layout =
      walker_buffer_selected
      ? psiformer::makePsiFormerWalkerBufferLayout(input)
      : pf::WalkerBufferLayout{};

  if (prepared_clone_batch_execution_plan_)
  {
    if (!prepared_clone_batch_execution_plan_.sameBinding(participant_plan))
      throw std::logic_error(
          "Cannot replace a prepared PsiFormer clone batch execution plan");
    validate_exact_storage();
    if (!hasPreparedBatchExecutionClone(participant_plan))
      throw std::logic_error(
          "PsiFormer prepared clone allocation identity changed");
    if (scalar_value)
    {
      const pf::DirectBatchStorageRequirement expected =
          pf::directBatchStorageRequirement(input.storage_shape, scalar_plan);
      if (!sameDirectBatchCapacityPlan(
              direct_batch_workspace_->capacityPlan(), scalar_plan) ||
          !sameDirectBatchExecutionStorage(
              direct_batch_workspace_->actualStorage(), expected) ||
          direct_batch_workspace_->vectorStorageBytes() !=
              expected.executionBytes() ||
          prepared_batch_bytes_ != expected.executionBytes() ||
          scalar_value_publication_.size() !=
              input.scalar_value_logical_maximum ||
          scalar_value_publication_.capacity() !=
              input.scalar_value_logical_maximum)
        throw std::logic_error(
            "PsiFormer prepared scalar owner differs from its admitted storage");
    }
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
    if (!sameDirectBatchExecutionStorage(
            prepared_batch_workspace->actualStorage(), expected) ||
        prepared_batch_workspace->vectorStorageBytes() !=
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

  const pf::DirectBatchWorkspace* const prepared_batch_identity =
      prepared_batch_workspace.get();
  const std::size_t prepared_batch_fingerprint = scalar_value
      ? prepared_batch_workspace->storageFingerprint(
            pf::DirectBatchMode::VALUE_ONLY)
      : 0;
  const std::size_t prepared_batch_bytes = scalar_value
      ? prepared_batch_workspace->vectorStorageBytes()
      : 0;
  const ValueType* const prepared_publication_data = scalar_value
      ? prepared_scalar_publication.data()
      : nullptr;
  const std::size_t prepared_publication_size = scalar_value
      ? prepared_scalar_publication.size()
      : 0;
  const std::size_t prepared_publication_capacity = scalar_value
      ? prepared_scalar_publication.capacity()
      : 0;

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
  prepared_scalar_value_compatibility_ = scalar_value;
  prepared_batch_workspace_identity_ = prepared_batch_identity;
  prepared_batch_storage_fingerprint_ = prepared_batch_fingerprint;
  prepared_batch_bytes_ = prepared_batch_bytes;
  prepared_scalar_value_publication_data_ = prepared_publication_data;
  prepared_scalar_value_publication_size_ = prepared_publication_size;
  prepared_scalar_value_publication_capacity_ =
      prepared_publication_capacity;
  static_assert(std::is_nothrow_copy_assignable_v<pf::WalkerBufferLayout>);
  prepared_walker_buffer_layout_ = prepared_walker_buffer_layout;
  prepared_bound_particle_set_ = bound_particle_set_;
  static_assert(std::is_nothrow_copy_assignable_v<BatchExecutionParticipantPlan>);
  // The plan view is the validity marker and is deliberately published last.
  prepared_clone_batch_execution_plan_ = participant_plan;
}

// Report the immutable optimization mode shared by the complete clone family.
bool PsiFormerWF::isOptimizable() const
{
  return optimization_metadata_->enabled;
}

// Distinguish actual device math from host pointers emitted by chunk sinks.
bool PsiFormerWF::usesAcceleratorParameterProducts() const noexcept
{
  return model_state_->numerical_tile_backend != nullptr;
}

const wftrain::StructuredParameterSchema& PsiFormerWF::parameterSchema() const noexcept
{
  return *structured_parameter_schema_;
}

// Prepare a version-bound score-product owner from one homogeneous clone batch.
std::unique_ptr<wftrain::StreamingDerivativeOperator>
PsiFormerWF::makeStreamingDerivativeOperator(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t maximum_parameter_chunk_size) const
{
  NativeTimerScope timer(phase_timers_.preparation);
  if (this != &wfc_list.getLeader())
    throw std::invalid_argument(
        "PsiFormer streaming derivative factory must be called on the batch leader");
  if (wfc_list.size() != p_list.size())
    throw std::invalid_argument(
        "PsiFormer streaming derivative component and particle batches differ in size");
  if (wfc_list.size() != 0 &&
      (&wfc_list[0] != &wfc_list.getLeader() ||
       &p_list[0] != &p_list.getLeader()))
    throw std::invalid_argument(
        "PsiFormer streaming derivative nonempty batches must begin with their leaders");
  if (maximum_parameter_chunk_size == 0)
    throw std::invalid_argument(
        "PsiFormer streaming derivative requires a positive parameter chunk size");

  // Capture positions/drifts/version coherently. GPU scratch preparation waits
  // outside this read; stale operators reject before numerical execution.
  std::optional<PsiFormerReadTransaction> state_lock;
  state_lock.emplace(*model_state_, model_state_->resident_device_authority_enabled);
  if (model_state_->direct_score_mode != DirectBackendMode::DIRECT)
    throw std::runtime_error(
        "PsiFormer streaming derivative requires the direct score backend");
  if (model_state_->direct_kinetic_mode != DirectBackendMode::DIRECT)
    throw std::runtime_error(
        "PsiFormer streaming derivative requires the direct kinetic backend");

  const std::size_t sample_count = wfc_list.size();
  const psiformer::ModelShape& model_shape =
      model_state_->execution_plan.modelShape();
  const std::size_t electron_count = model_shape.electrons();
  const std::size_t position_count = checkedBatchMemoryMultiply(
      checkedBatchMemoryMultiply(sample_count, electron_count,
                                 "PsiFormer streaming sample positions"),
      std::size_t{3}, "PsiFormer streaming Cartesian positions");
  checkedBatchMemoryMultiply(position_count, sizeof(double),
                             "PsiFormer streaming position bytes");
  const std::size_t mass_count = checkedBatchMemoryMultiply(
      sample_count, electron_count,
      "PsiFormer streaming inverse-mass count");
  checkedBatchMemoryMultiply(mass_count, sizeof(double),
                             "PsiFormer streaming inverse-mass bytes");
  std::vector<double> positions(position_count);
  std::vector<double> total_log_gradients_real(position_count);
  std::vector<double> total_log_gradients_imaginary(position_count);
  std::vector<double> inverse_masses(mass_count);

  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const auto* component = dynamic_cast<const PsiFormerWF*>(&wfc_list[sample]);
    if (!component || component->model_state_.get() != model_state_.get() ||
        component->structured_parameter_schema_->fingerprint() !=
            structured_parameter_schema_->fingerprint())
      throw std::invalid_argument(
          "PsiFormer streaming derivative batch contains a foreign component");
    const ParticleSet& particles = p_list[sample];
    if (particles.isSpinor() || particles.getTotalNum() < 0 ||
        static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
        particles.R.size() != electron_count ||
        particles.GroupID.size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer streaming derivative requires boundary-compatible POS-only samples with the model electron count");
    validateRuntimeLattice(particles, model_state_->execution_plan.environment(),
                           "PsiFormer streaming derivative sample");
    if (particles.groups() != 2 ||
        particles.groupsize(0) !=
            static_cast<int>(model_shape.spin_up_electrons) ||
        particles.groupsize(1) !=
            static_cast<int>(model_shape.spin_down_electrons))
      throw std::invalid_argument(
          "PsiFormer streaming derivative sample has an incompatible spin partition");
    if (particles.G.size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer streaming derivative requires one current total TrialWaveFunction drift per electron");
    const auto& masses = particles.get_mass_by_group();
    if (masses.size() < static_cast<std::size_t>(particles.groups()))
      throw std::invalid_argument(
          "PsiFormer streaming derivative requires initialized electron masses");
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      const int expected_group =
          electron < model_shape.spin_up_electrons ? 0 : 1;
      if (particles.GroupID[electron] != expected_group)
        throw std::invalid_argument(
            "PsiFormer streaming derivative sample has noncanonical spin ordering");
      const double mass = masses[expected_group];
      if (!psiformer::determinant::isFiniteReal(mass) || mass <= 0.0)
        throw std::invalid_argument(
            "PsiFormer streaming derivative electron masses must be finite and positive");
      const double inverse_mass = 1.0 / mass;
      if (!psiformer::determinant::isFiniteReal(inverse_mass))
        throw std::invalid_argument(
            "PsiFormer streaming derivative inverse electron masses must be finite");
      inverse_masses[sample * electron_count + electron] = inverse_mass;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double coordinate = particles.R[electron][dimension];
        if (!psiformer::determinant::isFiniteReal(coordinate))
          throw std::invalid_argument(
              "PsiFormer streaming derivative sample positions must be finite");
        positions[(sample * electron_count + electron) * 3 + dimension] =
            coordinate;
        const auto drift_value = particles.G[electron][dimension];
        const double drift_real = static_cast<double>(std::real(drift_value));
        const double drift_imaginary = static_cast<double>(std::imag(drift_value));
        if (!psiformer::determinant::isFiniteReal(drift_real) ||
            !psiformer::determinant::isFiniteReal(drift_imaginary))
          throw std::invalid_argument(
              "PsiFormer streaming derivative requires a finite total TrialWaveFunction drift");
        total_log_gradients_real[
            (sample * electron_count + electron) * 3 + dimension] = drift_real;
        total_log_gradients_imaginary[
            (sample * electron_count + electron) * 3 + dimension] =
                drift_imaginary;
      }
    }
  }

  const std::size_t parameter_version = model_state_->model.p.version();
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  if (model_state_->numerical_tile_backend) state_lock.reset();
#endif
  return std::make_unique<PsiFormerStreamingDerivativeOperator>(
      model_state_, structured_parameter_schema_, parameter_version, batch_ordinal,
      sample_offset, sample_count, std::move(positions),
      std::move(total_log_gradients_real),
      std::move(total_log_gradients_imaginary), std::move(inverse_masses),
      maximum_parameter_chunk_size);
}

// Prepare copied configurations and one reusable M-channel/P-score workspace.
std::unique_ptr<psiformer::DeterminantChannelBatch>
PsiFormerWF::makeDeterminantChannelBatch(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list) const
{
  if (this != &wfc_list.getLeader() || wfc_list.empty() ||
      wfc_list.size() != p_list.size() || &wfc_list[0] != &wfc_list.getLeader() ||
      &p_list[0] != &p_list.getLeader())
    throw std::invalid_argument(
        "PsiFormer determinant-channel factory requires matching nonempty leader-first batches");

  PsiFormerReadTransaction state_lock(*model_state_);
  if (model_state_->direct_score_mode != DirectBackendMode::DIRECT)
    throw std::runtime_error(
        "PsiFormer determinant-channel evaluation requires the direct score backend");
  const psiformer::ModelShape& shape = model_state_->execution_plan.modelShape();
  const std::size_t sample_count = wfc_list.size();
  const std::size_t electron_count = shape.electrons();
  const std::size_t position_count = checkedBatchMemoryMultiply(
      checkedBatchMemoryMultiply(sample_count, electron_count,
                                 "PsiFormer overlap sample positions"),
      std::size_t{3}, "PsiFormer overlap Cartesian positions");
  checkedBatchMemoryMultiply(position_count, sizeof(double),
                             "PsiFormer overlap position bytes");
  std::vector<double> positions(position_count);

  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const auto* component = dynamic_cast<const PsiFormerWF*>(&wfc_list[sample]);
    if (!component || component->model_state_.get() != model_state_.get() ||
        component->structured_parameter_schema_->fingerprint() !=
            structured_parameter_schema_->fingerprint())
      throw std::invalid_argument(
          "PsiFormer determinant-channel batch contains a foreign component");
    const ParticleSet& particles = p_list[sample];
    if (particles.isSpinor() || particles.getTotalNum() < 0 ||
        static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
        particles.R.size() != electron_count || particles.GroupID.size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer determinant-channel batch has an incompatible electron extent");
    validateRuntimeLattice(particles, model_state_->execution_plan.environment(),
                           "PsiFormer determinant-channel sample");
    if (particles.groups() != 2 ||
        particles.groupsize(0) != static_cast<int>(shape.spin_up_electrons) ||
        particles.groupsize(1) != static_cast<int>(shape.spin_down_electrons))
      throw std::invalid_argument(
          "PsiFormer determinant-channel sample has an incompatible spin partition");
    for (std::size_t electron = 0; electron < electron_count; ++electron)
    {
      const int expected_group = electron < shape.spin_up_electrons ? 0 : 1;
      if (particles.GroupID[electron] != expected_group)
        throw std::invalid_argument(
            "PsiFormer determinant-channel sample has noncanonical spin ordering");
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double coordinate = particles.R[electron][dimension];
        if (!psiformer::determinant::isFiniteReal(coordinate))
          throw std::invalid_argument(
              "PsiFormer determinant-channel sample positions must be finite");
        positions[(sample * electron_count + electron) * 3 + dimension] = coordinate;
      }
    }
  }

  return std::make_unique<PsiFormerDeterminantChannelBatch>(
      model_state_, structured_parameter_schema_, model_state_->model.p.version(),
      sample_count, std::move(positions));
}

// Prepare an O(P)+O(B) consumer for live ordinary-locality ECP quadrature tiles.
std::unique_ptr<wftrain::NonLocalECPDerivativeConsumer>
PsiFormerWF::makeNonLocalECPDerivativeConsumer(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    std::size_t maximum_parameter_chunk_size,
    bool capture_jvp_replay,
    bool replay_only) const
{
  if (this != &wfc_list.getLeader())
    throw std::invalid_argument(
        "PsiFormer ECP consumer factory must be called on the batch leader");
  if (wfc_list.size() != p_list.size())
    throw std::invalid_argument(
        "PsiFormer ECP consumer component and particle batches differ in size");
  if (wfc_list.size() != 0 &&
      (&wfc_list[0] != &wfc_list.getLeader() ||
       &p_list[0] != &p_list.getLeader()))
    throw std::invalid_argument(
        "PsiFormer ECP consumer nonempty batches must begin with their leaders");
  if (maximum_parameter_chunk_size == 0)
    throw std::invalid_argument(
        "PsiFormer ECP consumer requires a positive parameter chunk size");

  // Replay-only capture copies coordinates/metadata; CPU score consumers still fence.
  PsiFormerReadTransaction state_lock(*model_state_, replay_only && model_state_->deviceOnlyValueReads());
  if (model_state_->direct_score_mode != DirectBackendMode::DIRECT)
    throw std::runtime_error(
        "PsiFormer ECP consumer requires the direct score backend");

  const std::size_t sample_count = wfc_list.size();
  const std::size_t electron_count =
      model_state_->execution_plan.modelShape().electrons();
  const std::size_t position_count = checkedBatchMemoryMultiply(
      checkedBatchMemoryMultiply(sample_count, electron_count,
                                 "PsiFormer ECP reference positions"),
      std::size_t{3}, "PsiFormer ECP Cartesian positions");
  checkedBatchMemoryMultiply(position_count, sizeof(double),
                             "PsiFormer ECP reference-position bytes");
  std::vector<double> positions(position_count);

  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const auto* component = dynamic_cast<const PsiFormerWF*>(&wfc_list[sample]);
    if (!component || component->model_state_.get() != model_state_.get() ||
        component->structured_parameter_schema_->fingerprint() !=
            structured_parameter_schema_->fingerprint())
      throw std::invalid_argument(
          "PsiFormer ECP consumer batch contains a foreign component");
    const ParticleSet& particles = p_list[sample];
    if (particles.isSpinor() || particles.getTotalNum() < 0 ||
        static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
        particles.R.size() != electron_count)
      throw std::invalid_argument(
          "PsiFormer ECP consumer requires boundary-compatible POS-only samples with the model electron count");
    validateRuntimeLattice(particles, model_state_->execution_plan.environment(),
                           "PsiFormer ECP consumer sample");
    for (std::size_t electron = 0; electron < electron_count; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double coordinate = particles.R[electron][dimension];
        if (!psiformer::determinant::isFiniteReal(coordinate))
          throw std::invalid_argument(
              "PsiFormer ECP consumer reference positions must be finite");
        positions[(sample * electron_count + electron) * 3 + dimension] =
            coordinate;
      }
  }

  const std::size_t parameter_version = model_state_->model.p.version();
  return std::make_unique<PsiFormerNonLocalECPDerivativeConsumer>(
      model_state_, structured_parameter_schema_, parameter_version,
      sample_count, std::move(positions), maximum_parameter_chunk_size,
      capture_jvp_replay, replay_only);
}

void PsiFormerWF::isolateParameterState()
{
  requireNoPlannedProposalMutation("candidate model isolation");
  auto original = model_state_;
  PsiFormerReadTransaction lock(*original);
  // Parameters' graph leaves are shared pointers: rebuilding the owning flat
  // store also recreates leaves, preventing oracle trials from mutating live nodes.
  pf::Parameters independent_parameters(original->model.p.flat_values(), original->model.p.layouts);
  independent_parameters.parameter_version = original->model.p.version();
  pf::PsiFormer independent_model(std::move(independent_parameters), original->model.cfg,
      original->model.ndet, original->model.dim, original->model.heads, original->model.blocks);
  model_state_ = std::make_shared<PsiFormerSharedState>(
      std::move(independent_model), original->model_origin, original->initialization_profile,
      original->initialization_seed, original->execution_plan.environment(),
      original->twist_phase, original->accelerator_resolution, original->precision,
      original->physical_source, original->physical_species, original->source);
  invalidateParameterCaches(model_state_->model.p.version());
}

std::vector<std::string> PsiFormerWF::neuralWeightBlocks() const
{
  using Role = psiformer::ParameterRole;
  std::vector<std::string> result;
  for (const auto& tensor : model_state_->execution_plan.parameterTensors())
    switch (tensor.role)
    {
    case Role::ELECTRON_EMBEDDING_WEIGHT:
    case Role::ATTENTION_QUERY_WEIGHT:
    case Role::ATTENTION_KEY_WEIGHT:
    case Role::ATTENTION_VALUE_WEIGHT:
    case Role::ATTENTION_OUTPUT_WEIGHT:
    case Role::UPDATE_HIDDEN_WEIGHT:
    case Role::UPDATE_OUTPUT_WEIGHT:
    case Role::PAIR_ATTENTION_WEIGHT:
    case Role::BACKFLOW_UP_WEIGHT:
    case Role::BACKFLOW_DOWN_WEIGHT:
      result.push_back(tensor.module + "/" + tensor.name);
      break;
    default: break;
    }
  return result;
}

namespace
{
/// Classify native affine roles against exact canonical schema intervals.
std::vector<wftrain::AffineParameterBlock> nativeAffineParameterBlocks(
    const psiformer::PsiFormerExecutionPlan& plan,
    const wftrain::StructuredParameterSchema& schema)
{
  using Role = psiformer::ParameterRole;
  if (schema.parameterCount() != plan.parameterCount())
    throw std::invalid_argument("PsiFormer affine schema has the wrong parameter count");
  const auto schema_block = [&](Role role, std::size_t attention_block) {
    const auto& tensor = plan.parameter(role, attention_block);
    for (std::size_t block = 0; block < schema.blocks().size(); ++block)
    {
      const auto& candidate = schema.blocks()[block];
      if (candidate.id == tensor.module + "/" + tensor.name && candidate.offset == tensor.begin &&
          candidate.count == tensor.size() && candidate.shape == tensor.shape &&
          candidate.scalar_domain == wftrain::ParameterScalarDomain::REAL64)
        return block;
    }
    throw std::invalid_argument("PsiFormer affine schema does not match its execution plan");
  };
  std::vector<wftrain::AffineParameterBlock> result;
  const auto& shape = plan.modelShape();
  result.reserve(3 + 6 * shape.attention_blocks);
  const auto append = [&](Role role, std::size_t attention_block,
                          std::size_t input_width, std::size_t output_width,
                          Role bias_role = Role::COUNT) {
    const auto weight = schema_block(role, attention_block);
    auto bias = wftrain::NO_AFFINE_BIAS_BLOCK;
    if (bias_role != Role::COUNT)
    {
      bias = schema_block(bias_role, attention_block);
      if (schema.blocks()[weight].trainable != schema.blocks()[bias].trainable)
        throw std::invalid_argument("PsiFormer affine weight and bias must share trainability");
    }
    if (!schema.blocks()[weight].trainable)
      return;
    std::string id = psiformer::parameterRoleName(role);
    if (attention_block != psiformer::NO_ATTENTION_BLOCK)
      id += "/block_" + std::to_string(attention_block);
    result.push_back({std::move(id), weight, bias, input_width, output_width});
  };
  const auto width = shape.feature_dimension;
  append(Role::ELECTRON_EMBEDDING_WEIGHT, psiformer::NO_ATTENTION_BLOCK,
         plan.parameter(Role::ELECTRON_EMBEDDING_WEIGHT).shape[0], width);
  for (std::size_t block = 0; block < shape.attention_blocks; ++block)
  {
    for (Role role : {Role::ATTENTION_QUERY_WEIGHT, Role::ATTENTION_KEY_WEIGHT,
                       Role::ATTENTION_VALUE_WEIGHT, Role::ATTENTION_OUTPUT_WEIGHT})
      append(role, block, width, width);
    append(Role::UPDATE_HIDDEN_WEIGHT, block, width, width, Role::UPDATE_HIDDEN_BIAS);
    append(Role::UPDATE_OUTPUT_WEIGHT, block, width, width, Role::UPDATE_OUTPUT_BIAS);
    if (plan.environment().periodic_profile)
      append(Role::PAIR_ATTENTION_WEIGHT, block, plan.parameter(Role::PAIR_ATTENTION_WEIGHT, block).shape[0],
             shape.attention_heads, Role::PAIR_ATTENTION_BIAS);
  }
  append(Role::BACKFLOW_UP_WEIGHT, psiformer::NO_ATTENTION_BLOCK,
         width, shape.determinants * shape.electrons());
  append(Role::BACKFLOW_DOWN_WEIGHT, psiformer::NO_ATTENTION_BLOCK,
         width, shape.determinants * shape.electrons());
  return result;
}

/// Translate native typed observations to provider metadata ordinals.
class NativeAffineScoreObserver final : public pf::DirectAffineObservationSink
{
public:
  NativeAffineScoreObserver(const psiformer::PsiFormerExecutionPlan& plan,
                           const wftrain::StructuredParameterSchema& schema,
                           const std::vector<wftrain::AffineParameterBlock>& blocks,
                           wftrain::AffineScoreObserver& observer)
      : blocks_(blocks), observer_(observer)
  {
    bindings_.reserve(blocks.size());
    for (std::size_t ordinal = 0; ordinal < blocks.size(); ++ordinal)
    {
      const auto& weight = schema.blocks()[blocks[ordinal].weight_block];
      const auto& tensors = plan.parameterTensors();
      const auto tensor = std::find_if(tensors.begin(), tensors.end(), [&](const auto& candidate) {
        return candidate.begin == weight.offset && candidate.size() == weight.count &&
            candidate.module + "/" + candidate.name == weight.id;
      });
      if (tensor == tensors.end())
        throw std::logic_error("PsiFormer affine observation has no native tensor binding");
      bindings_.push_back({tensor->role, tensor->attention_block, ordinal});
    }
  }

  void observe(const pf::DirectAffineObservation& observation) override
  {
    const auto found = std::find_if(bindings_.begin(), bindings_.end(), [&](const auto& binding) {
      return binding.role == observation.role && binding.attention_block == observation.attention_block;
    });
    // The native tape also emits frozen affine layers, which have no trainable boundary.
    if (found == bindings_.end())
      return;
    const auto& block = blocks_[found->ordinal];
    if (observation.input_width != block.input_width || observation.output_width != block.output_width)
      throw std::logic_error("PsiFormer affine observation has incompatible widths");
    observer_.observe({found->ordinal, observation.activations, observation.sensitivities, observation.row_count});
  }

private:
  struct Binding
  {
    psiformer::ParameterRole role;
    std::size_t attention_block;
    std::size_t ordinal;
  };
  const std::vector<wftrain::AffineParameterBlock>& blocks_;
  wftrain::AffineScoreObserver& observer_;
  std::vector<Binding> bindings_;
};
} // namespace

std::vector<wftrain::AffineParameterBlock> PsiFormerWF::affineParameterBlocks() const
{
  PsiFormerReadTransaction state_lock(*model_state_);
  return nativeAffineParameterBlocks(model_state_->execution_plan, *structured_parameter_schema_);
}

std::vector<std::size_t> PsiFormerWF::genericKFACParameterBlocks() const
{
  PsiFormerReadTransaction state_lock(*model_state_);
  const auto& plan = model_state_->execution_plan;
  const auto& blocks = structured_parameter_schema_->blocks();
  std::vector<std::size_t> result;
  for (const auto role : {psiformer::ParameterRole::CUSP_SAME_ALPHA,
                          psiformer::ParameterRole::CUSP_OPPOSITE_ALPHA})
    if (plan.hasParameter(role))
    {
      const auto& tensor = plan.parameter(role);
      const auto found = std::find_if(blocks.begin(), blocks.end(), [&](const auto& block) {
        return block.id == tensor.module + "/" + tensor.name && block.offset == tensor.begin &&
            block.count == tensor.size() && block.shape == tensor.shape &&
            block.scalar_domain == wftrain::ParameterScalarDomain::REAL64;
      });
      if (found == blocks.end())
        throw std::logic_error("PsiFormer generic curvature schema differs from its execution plan");
      if (found->trainable)
        result.push_back(static_cast<std::size_t>(found - blocks.begin()));
    }
  return result;
}

std::size_t PsiFormerWF::affineScoreStorageBytes() const
{
  PsiFormerReadTransaction transaction(*model_state_);
  return pf::scoreWorkspaceStorageRequirement(makeBatchMemoryPolicyInput().storage_shape);
}

/// Inspect the already validated physical model kind without rebinding any source.
bool PsiFormerWF::isPseudopotentialSystem() const
{
  PsiFormerReadTransaction transaction(*model_state_, true);
  return system_kind_ == "pseudopotential";
}

/// Check immutable pair geometry against a native target without copying parameter leaves.
bool PsiFormerWF::matchesRealDevicePairTarget(const ParticleSet& particles) const
{
  PsiFormerReadTransaction transaction(*model_state_, true);
  const auto& state = transaction.state();
  const auto& environment = state.execution_plan.environment();
  if (!psiformer::matchesRealDevicePairMetadata(environment) || state.twist_phase.active() ||
      particles.getLattice().getSuperCellEnum() != SUPERCELL_BULK)
    return false;
  constexpr double tolerance = 128 * std::numeric_limits<double>::epsilon();
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    if (!std::isfinite(particles.getTwist()[axis]) || particles.getTwist()[axis] != 0.0)
      return false;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double actual = particles.getLattice().R(axis, dimension);
      const double expected = environment.lattice_vectors[axis][dimension];
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
        return false;
    }
  }
  return true;
}

/// Reuse persistent physical identity without downloading a device-authoritative parameter image.
std::string PsiFormerWF::trainingConfigurationFingerprint() const
{
  PsiFormerReadTransaction transaction(*model_state_, true);
  const auto& state = transaction.state();
  return "real-training-model-v1/" + system_kind_ + "/" + parameterSchema().fingerprint() + "/" +
      modelFingerprint(state.model, state.model_origin, state.initialization_profile, state.initialization_seed,
                       state.execution_plan.environment(), state.twist_phase.reduced, state.twist_phase.cartesian,
                       state.physical_source, state.physical_species);
}

void PsiFormerWF::observeAffineScores(const ParticleSet& particles,
                                     wftrain::AffineScoreObserver& observer,
                                     std::vector<double>& score) const
{
  PsiFormerReadTransaction transaction(*model_state_);
  const auto& state = transaction.state();
  if (state.direct_score_mode != DirectBackendMode::DIRECT)
    throw std::runtime_error("PsiFormer affine observations require the direct score backend");
  const auto& shape = state.execution_plan.modelShape();
  const auto electrons = shape.electrons();
  if (particles.R.getUnit() != PosUnit::Cartesian)
    throw std::invalid_argument("PsiFormer affine observation positions must use Cartesian units");
  if (particles.isSpinor() || particles.getTotalNum() < 0 ||
      static_cast<std::size_t>(particles.getTotalNum()) != electrons ||
      particles.R.size() != electrons || particles.GroupID.size() != electrons)
    throw std::invalid_argument("PsiFormer affine observations require model-compatible POS-only samples");
  validateRuntimeLattice(particles, state.execution_plan.environment(), "PsiFormer affine observation sample");
  if (particles.groups() != 2 || particles.groupsize(0) != static_cast<int>(shape.spin_up_electrons) ||
      particles.groupsize(1) != static_cast<int>(shape.spin_down_electrons))
    throw std::invalid_argument("PsiFormer affine observation sample has an incompatible spin partition");
  auto workspace = state.direct_score_executor.makeWorkspace();
  for (std::size_t electron = 0; electron < electrons; ++electron)
  {
    if (particles.GroupID[electron] != (electron < shape.spin_up_electrons ? 0 : 1))
      throw std::invalid_argument("PsiFormer affine observation sample has noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double coordinate = particles.R[electron][dimension];
      if (!psiformer::determinant::isFiniteReal(coordinate))
        throw std::invalid_argument("PsiFormer affine observation sample positions must be finite");
      workspace->setPosition(electron, dimension, coordinate);
    }
  }
  const auto blocks = nativeAffineParameterBlocks(state.execution_plan, *structured_parameter_schema_);
  NativeAffineScoreObserver sink(state.execution_plan, *structured_parameter_schema_, blocks, observer);
  const auto result = state.direct_score_executor.evaluateWithAffineObservations(*workspace, sink);
  if (result.parameter_version != transaction.parameterVersion() ||
      result.parameter_score.size != structured_parameter_schema_->parameterCount())
    throw std::logic_error("PsiFormer affine score result has an incompatible parameter identity");
  if (!std::all_of(result.parameter_score.begin(), result.parameter_score.end(),
                   [](double value) { return psiformer::determinant::isFiniteReal(value); }))
    throw std::runtime_error("PsiFormer affine parameter score is non-finite");
  score.assign(result.parameter_score.begin(), result.parameter_score.end());
}

wftrain::StructuredParameterSnapshot PsiFormerWF::snapshotParameters() const
{
  PsiFormerReadTransaction state_lock(*model_state_);
  const pf::Parameters& parameters = model_state_->model.p;
  return {structured_parameter_schema_->fingerprint(), parameters.version(),
          parameters.flat_values()};
}

// Copy current CPU/GPU raw orbital values without a reverse or parameter-response download.
void PsiFormerWF::evaluateOrbitalValues(const ParticleSet& particles,
                                       std::vector<double>& orbitals,
                                       std::size_t expected_version) const
{
  auto scratch = acquireDerivativeScratch(*model_state_, structured_parameter_schema_->parameterCount());
  {
    std::lock_guard factory_lock(model_state_->derivative_scratch_factory_mutex);
    model_state_->orbital_pretraining_scratch = scratch;
  }
  PsiFormerDerivativeLease scratch_lease(*scratch);
  PsiFormerReadTransaction state_lock(*model_state_);
  if (model_state_->model.p.version() != expected_version)
    throw std::runtime_error("PsiFormer orbital forward has a stale parameter version");
  const auto& shape = model_state_->execution_plan.modelShape();
  const auto ne = shape.electrons();
  if (particles.isSpinor() || particles.R.size() != ne || particles.GroupID.size() != ne)
    throw std::invalid_argument("PsiFormer orbital forward particle extent mismatch");
  validateRuntimeLattice(particles, model_state_->execution_plan.environment(), "PsiFormer orbital forward sample");
  for (std::size_t electron = 0; electron < ne; ++electron)
  {
    if (particles.GroupID[electron] != (electron < shape.spin_up_electrons ? 0 : 1))
      throw std::invalid_argument("PsiFormer orbital forward has noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double coordinate = particles.R[electron][dimension];
      if (!psiformer::determinant::isFiniteReal(coordinate))
        throw std::invalid_argument("PsiFormer orbital forward coordinates must be finite");
#ifdef ENABLE_CUDA
      if (scratch->device_products)
      {
        scratch->device_positions[3 * electron + dimension] = coordinate;
        continue;
      }
#endif
      scratch->score->setPosition(electron, dimension, coordinate);
    }
  }
#ifdef ENABLE_CUDA
  if (scratch->device_products)
  {
    const auto count = checkedBatchMemoryMultiply(shape.determinants,
        checkedBatchMemoryMultiply(ne, ne, "PsiFormer orbital forward matrix"), "PsiFormer orbital forward extent");
    orbitals.resize(count);
    scratch->device_products->rawOrbitalValues(scratch->device_positions.data(), orbitals.data(), count,
                                               expected_version);
    return;
  }
#endif
  const auto result = model_state_->direct_score_executor.evaluateOrbitalValues(*scratch->score);
  orbitals.assign(result.values, result.values + result.size);
}

// Evaluate one supplied conventional orbital target with a reusable CPU/GPU lane.
double PsiFormerWF::evaluateOrbitalMSE(const ParticleSet& particles,
                                     const std::vector<double>& target,
                                     std::vector<double>& gradient,
                                     std::size_t expected_version) const
{
  auto scratch = acquireDerivativeScratch(*model_state_, structured_parameter_schema_->parameterCount());
  {
    std::lock_guard factory_lock(model_state_->derivative_scratch_factory_mutex);
    model_state_->orbital_pretraining_scratch = scratch;
  }
  PsiFormerDerivativeLease scratch_lease(*scratch);
  PsiFormerReadTransaction state_lock(*model_state_);
  if (model_state_->model.p.version() != expected_version)
    throw std::runtime_error("PsiFormer orbital-MSE target batch has a stale parameter version");
  const auto& shape = model_state_->execution_plan.modelShape();
  const auto ne = shape.electrons();
  const auto orbital_count = checkedBatchMemoryMultiply(shape.determinants,
      checkedBatchMemoryMultiply(ne, ne, "PsiFormer orbital-MSE target matrix"), "PsiFormer orbital-MSE target extent");
  if (target.size() != orbital_count || gradient.size() != structured_parameter_schema_->parameterCount() ||
      particles.isSpinor() || particles.R.size() != ne || particles.GroupID.size() != ne)
    throw std::invalid_argument("PsiFormer orbital-MSE target, gradient or particle extent mismatch");
  validateRuntimeLattice(particles, model_state_->execution_plan.environment(), "PsiFormer orbital-MSE sample");
  for (std::size_t electron = 0; electron < ne; ++electron)
  {
    if (particles.GroupID[electron] != (electron < shape.spin_up_electrons ? 0 : 1))
      throw std::invalid_argument("PsiFormer orbital-MSE sample has noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double coordinate = particles.R[electron][dimension];
      if (!psiformer::determinant::isFiniteReal(coordinate))
        throw std::invalid_argument("PsiFormer orbital-MSE coordinates must be finite");
#ifdef ENABLE_CUDA
      if (scratch->device_products)
      {
        scratch->device_positions[3 * electron + dimension] = coordinate;
        continue;
      }
#endif
      scratch->score->setPosition(electron, dimension, coordinate);
    }
  }

#ifdef ENABLE_CUDA
  if (scratch->device_products)
  {
    const double loss = scratch->device_products->orbitalMse(scratch->device_positions.data(),
        target.data(), target.size(), gradient.data(), gradient.size(), expected_version);
    if (!scratch->reported_device_products)
    {
      app_log() << "  PsiFormer numerical device orbital-MSE product completed: parameter_version="
                << expected_version << ", CPU derivative fallback=no" << std::endl;
      scratch->reported_device_products = true;
    }
    return loss;
  }
#endif
  const auto result = model_state_->direct_score_executor.evaluateOrbitalMSE(*scratch->score,
      {target.data(), target.size()});
  std::copy(result.parameter_gradient.begin(), result.parameter_gradient.end(), gradient.begin());
  return result.loss;
}

std::size_t PsiFormerWF::publishParameters(
    const wftrain::StructuredParameterSnapshot& candidate,
    std::size_t expected_version)
{
  NativeTimerScope timer(phase_timers_.publication);
  requireNoPlannedProposalMutation("structured parameter publication");
  if (candidate.schema_fingerprint != structured_parameter_schema_->fingerprint())
    throw std::invalid_argument("PsiFormer structured update has an incompatible schema fingerprint");
  if (candidate.values.size() != structured_parameter_schema_->parameterCount())
    throw std::invalid_argument("PsiFormer structured update has the wrong parameter count");
  if (candidate.version != expected_version)
    throw std::invalid_argument("PsiFormer structured update candidate has the wrong source version");

  std::unique_lock state_lock(model_state_->mutex);
  requireNoPlannedProposalMutation("structured parameter publication");
  pf::Parameters& parameters = model_state_->model.p;
  if (parameters.version() != expected_version)
    throw std::runtime_error("PsiFormer structured update rejected a stale parameter version");

  // set_flat_values validates every value before mutating the native leaves.
  model_state_->setParameterValues(candidate.values);
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
  NativeTimerScope timer(phase_timers_.resource);
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
  // Unplanned TrialWaveFunction acquisition visits components in declaration
  // order, so later components may legitimately add loans after PsiFormer
  // records its own.  Fewer loans means an owner was removed prematurely;
  // extra loans are validated by this component's saved cursor and handle.
  if (collection.getOutstandingLoanCount() <
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

// Bind a one-electron proposal to its exact producer and ordered crowd state.
std::uint64_t PsiFormerWF::singleTransactionFingerprint(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    ProposalOrigin origin,
    std::size_t active_electron,
    std::size_t proposal_version) const noexcept
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(hash, SINGLE_TRANSACTION_FINGERPRINT_DOMAIN);
  mixPersistentInteger(hash, selectedTeamFingerprint(wfc_list, p_list));
  mixPersistentInteger(hash, static_cast<std::uint8_t>(origin));
  mixPersistentInteger(hash, active_electron);
  mixPersistentInteger(hash, proposal_version);
  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    mixPersistentInteger(hash, component.accepted_configuration_identity_);
    mixPersistentInteger(
        hash, configurationIdentity(p_list[lane], static_cast<int>(active_electron)));
  }
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
  case PlannedOperation::BUFFER_READ:
    modes.require(BatchExecutionMode::BUFFER_READ);
    break;
  case PlannedOperation::BUFFER_WRITE:
    modes.require(BatchExecutionMode::BUFFER_WRITE);
    break;
  case PlannedOperation::PREPARE_GROUP:
    modes.require(BatchExecutionMode::PREPARE_GROUP);
    break;
  case PlannedOperation::COMPLETE_UPDATES:
    modes.require(BatchExecutionMode::COMPLETE_UPDATES);
    break;
  case PlannedOperation::ACCEPT_REJECT_VALUE:
  case PlannedOperation::SINGLE_CANCEL:
  case PlannedOperation::SELECTED_RESOLVE:
  case PlannedOperation::SELECTED_CANCEL:
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
  case PlannedOperation::SINGLE_CANCEL:
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

// Check lifecycle preparation without depending on unrelated numeric scratch.
bool PsiFormerWF::hasPreparedLifecycleClone(
    const BatchExecutionParticipantPlan& plan) const noexcept
{
  return plan && batch_execution_plan_.sameBinding(plan) &&
      prepared_clone_batch_execution_plan_.sameBinding(plan) &&
      bound_particle_set_ &&
      prepared_bound_particle_set_ == bound_particle_set_ &&
      plan.plan().particleCount() ==
          model_state_->execution_plan.modelShape().electrons();
}

// Validate one bound ParticleSet without evaluating the model or touching caches.
void PsiFormerWF::requirePlannedLifecycleParticleSet(
    const ParticleSet& particles,
    const BatchExecutionPlan& plan,
    std::optional<int> group_index) const
{
  if (plan.targetCoordinate() != BatchExecutionTargetCoordinate::POS_ONLY)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation requires explicit POS-only target evidence");

  const psiformer::ModelShape& model_shape =
      model_state_->execution_plan.modelShape();
  const std::size_t electron_count = model_shape.electrons();
  const auto& soa_positions = particles.getCoordinates().getAllParticlePos();
  if (plan.particleCount() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle particle count differs from its model");
  if (particles.isSpinor())
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation received a spinor ParticleSet");
  if (particles.getTotalNum() < 0 ||
      static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
      particles.R.size() != electron_count ||
      soa_positions.size() != electron_count ||
      soa_positions.capacity() < electron_count ||
      particles.G.size() != electron_count ||
      particles.L.size() != electron_count ||
      particles.GroupID.size() != electron_count ||
      particles.spins.size() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation received incompatible ParticleSet extents");
  if (electron_count != 0 &&
      (!particles.R.data() || !soa_positions.data() || !particles.G.data() ||
       !particles.L.data() || !particles.GroupID.data() ||
       !particles.spins.data()))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation received null ParticleSet storage");
  if (particles.groups() != 2 ||
      particles.groupsize(0) !=
          static_cast<int>(model_shape.spin_up_electrons) ||
      particles.groupsize(1) !=
          static_cast<int>(model_shape.spin_down_electrons))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation received an incompatible spin partition");
  if (group_index &&
      (*group_index < 0 || *group_index >= particles.groups()))
    throw std::out_of_range(
        "PsiFormer planned lifecycle operation received an invalid group index");
  if (particles.getActivePtcl() != -1)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation requires an inactive ParticleSet move");

  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const auto soa_position = soa_positions[electron];
    const int expected_group =
        electron < model_shape.spin_up_electrons ? 0 : 1;
    if (particles.GroupID[electron] != expected_group)
      throw std::invalid_argument(
          "PsiFormer planned lifecycle operation received noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      const auto& aos_coordinate = particles.R[electron][dimension];
      const auto& soa_coordinate = soa_position[dimension];
      if (!psiformer::determinant::isFiniteReal(
              static_cast<double>(aos_coordinate)))
        throw std::invalid_argument(
            "PsiFormer planned lifecycle operation received a non-finite position");
      if (std::memcmp(std::addressof(aos_coordinate),
                      std::addressof(soa_coordinate),
                      sizeof(aos_coordinate)) != 0)
        throw std::invalid_argument(
            "PsiFormer planned lifecycle operation has inconsistent AoS and SoA positions");
    }
  }
}

// Prove one scalar lifecycle no-op through its independent explicit mode.
void PsiFormerWF::requirePlannedScalarLifecycleOperation(
    PlannedOperation operation,
    const ParticleSet& particles,
    std::optional<int> group_index) const
{
  const bool prepares_group = operation == PlannedOperation::PREPARE_GROUP;
  const bool completes_updates =
      operation == PlannedOperation::COMPLETE_UPDATES;
  if ((!prepares_group && !completes_updates) ||
      prepares_group != group_index.has_value())
    throw std::invalid_argument(
        "PsiFormer scalar lifecycle preflight received an incompatible operation");
  if (!batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer planned lifecycle operation has no bound batch plan");
  if (!hasPreparedLifecycleClone(batch_execution_plan_))
    throw std::logic_error(
        "PsiFormer planned lifecycle operation requires an exactly prepared clone");
  if (bound_particle_set_ != std::addressof(particles))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation received a foreign ParticleSet");
  if (has_proposal_ || proposal_origin_ != ProposalOrigin::NONE)
    throw std::logic_error(
        "PsiFormer planned lifecycle operation requires absent proposal state");

  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  const BatchExecutionRequirements required_modes =
      plannedOperationRequiredModes(operation);
  const BatchExecutionMode required_mode = prepares_group
      ? BatchExecutionMode::PREPARE_GROUP
      : BatchExecutionMode::COMPLETE_UPDATES;
  if (!required_modes.requires(required_mode) ||
      !plan.requirements().requires(required_mode))
    throw std::logic_error(
        "PsiFormer lifecycle operation is not admitted by its explicit batch mode");
  const BatchExecutionTopology& topology = plan.topology();
  if (topology.serialized_walkers || topology.backend_id != "cpu" ||
      topology.device_id)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation requires direct nonserialized CPU execution");
  requirePlannedLifecycleParticleSet(particles, plan, group_index);
}

// Prove one acquired component team without consulting numerical workspaces.
void PsiFormerWF::requirePlannedMultiWalkerLifecycleOperation(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>* p_list,
    PlannedOperation operation,
    std::optional<int> group_index) const
{
  const bool prepares_group = operation == PlannedOperation::PREPARE_GROUP;
  const bool completes_updates =
      operation == PlannedOperation::COMPLETE_UPDATES;
  if ((!prepares_group && !completes_updates) ||
      prepares_group != group_index.has_value() ||
      prepares_group != (p_list != nullptr))
    throw std::invalid_argument(
        "PsiFormer crowd lifecycle preflight received incompatible inputs");
  if (wfc_list.empty())
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation requires a nonempty component team");
  if (p_list && (p_list->empty() || p_list->size() != wfc_list.size()))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation has inconsistent live-lane counts");
  if (std::addressof(wfc_list.getLeader()) !=
      std::addressof(wfc_list[0]))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle component leader must occupy lane zero");
  if (p_list && std::addressof(p_list->getLeader()) !=
          std::addressof((*p_list)[0]))
    throw std::invalid_argument(
        "PsiFormer planned lifecycle ParticleSet leader must occupy lane zero");

  auto* component_leader =
      dynamic_cast<PsiFormerWF*>(std::addressof(wfc_list.getLeader()));
  if (!component_leader || component_leader != this)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation was not invoked on its component leader");
  if (!batch_execution_plan_ ||
      !hasPreparedLifecycleClone(batch_execution_plan_))
    throw std::logic_error(
        "PsiFormer planned lifecycle operation requires an exactly prepared leader clone");

  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  const BatchExecutionRequirements required_modes =
      plannedOperationRequiredModes(operation);
  const BatchExecutionMode required_mode = prepares_group
      ? BatchExecutionMode::PREPARE_GROUP
      : BatchExecutionMode::COMPLETE_UPDATES;
  if (!required_modes.requires(required_mode) ||
      !plan.requirements().requires(required_mode))
    throw std::logic_error(
        "PsiFormer lifecycle operation is not admitted by its explicit batch mode");
  const BatchExecutionTopology& topology = plan.topology();
  if (topology.serialized_walkers || topology.backend_id != "cpu" ||
      topology.device_id)
    throw std::invalid_argument(
        "PsiFormer planned lifecycle operation requires direct nonserialized CPU execution");

  if (!mw_resource_handle_ || !acquired_resource_collection_ ||
      acquired_resource_cursor_ == 0 ||
      acquired_resource_outstanding_loans_ == 0)
    throw std::logic_error(
        "PsiFormer planned lifecycle operation requires an acquired crowd resource");
  const PsiFormerMultiWalkerResource& resource =
      mw_resource_handle_.getResource();
  if (resource.model_state.get() != model_state_.get() ||
      !resource.expected_plan.sameBinding(batch_execution_plan_) ||
      !resource.prepared_plan.sameBinding(batch_execution_plan_) ||
      !resource.prepared_crowd_plan ||
      resource.participant_id !=
          batch_execution_plan_.evidence().participant_id ||
      resource.prepared_plan_fingerprint != plan.fingerprint())
    throw std::logic_error(
        "PsiFormer planned lifecycle resource has stale preparation provenance");

  const BatchResourcePreparationProvenance& collection_provenance =
      acquired_resource_collection_->getBatchResourcePreparationProvenance();
  if (collection_provenance.state != BatchResourcePreparationState::PREPARED ||
      collection_provenance.plan.get() != std::addressof(plan) ||
      collection_provenance.crowd_index !=
          resource.prepared_crowd_index ||
      acquired_resource_collection_->getCursor() !=
          acquired_resource_cursor_ ||
      acquired_resource_collection_->getOutstandingLoanCount() !=
          acquired_resource_outstanding_loans_ ||
      !resource.acquired_from_prepared_collection ||
      resource.acquired_plan_identity != std::addressof(plan) ||
      resource.acquired_crowd_index != resource.prepared_crowd_index)
    throw std::logic_error(
        "PsiFormer planned lifecycle resource acquisition provenance changed");

  const std::vector<std::size_t>& reserve_walkers =
      psiformer::psiFormerReserveWalkersPerCrowd(topology);
  const std::size_t crowd_index = resource.prepared_crowd_index;
  if (crowd_index >= topology.initial_walkers_per_crowd.size() ||
      crowd_index >= reserve_walkers.size())
    throw std::logic_error(
        "PsiFormer planned lifecycle crowd index is outside the topology");
  const psiformer::PsiFormerCrowdMemoryPlan& crowd =
      *resource.prepared_crowd_plan;
  if (crowd.initial_walkers !=
          topology.initial_walkers_per_crowd[crowd_index] ||
      crowd.reserve_walkers != reserve_walkers[crowd_index] ||
      wfc_list.size() > crowd.reserve_walkers)
    throw std::length_error(
        "PsiFormer planned lifecycle team exceeds or mismatches its crowd envelope");

  for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
  {
    auto* component =
        dynamic_cast<PsiFormerWF*>(std::addressof(wfc_list[lane]));
    if (!component)
      throw std::invalid_argument(
          "PsiFormer planned lifecycle team contains a foreign component type");
    const ParticleSet* particles = p_list
        ? std::addressof((*p_list)[lane])
        : component->bound_particle_set_;
    if (!particles || component->bound_particle_set_ != particles)
      throw std::invalid_argument(
          "PsiFormer planned lifecycle component and ParticleSet lanes are not identically bound");

    for (std::size_t prior = 0; prior < lane; ++prior)
    {
      if (std::addressof(wfc_list[lane]) ==
          std::addressof(wfc_list[prior]))
        throw std::invalid_argument(
            "PsiFormer planned lifecycle team contains a duplicate component");
      const auto* prior_component =
          dynamic_cast<const PsiFormerWF*>(
              std::addressof(wfc_list[prior]));
      const ParticleSet* prior_particles = p_list
          ? std::addressof((*p_list)[prior])
          : (prior_component ? prior_component->bound_particle_set_ : nullptr);
      if (particles == prior_particles)
        throw std::invalid_argument(
            "PsiFormer planned lifecycle team contains a duplicate ParticleSet");
    }

    if (component->model_state_.get() != model_state_.get() ||
        component->optimization_metadata_.get() !=
            optimization_metadata_.get())
      throw std::invalid_argument(
          "PsiFormer planned lifecycle team mixes distinct shared state");
    if (!component->batch_execution_plan_.sameBinding(
            batch_execution_plan_) ||
        !component->hasPreparedLifecycleClone(
            batch_execution_plan_))
      throw std::logic_error(
          "PsiFormer planned lifecycle team contains an unprepared or differently bound clone");
    if (component->acquired_crowd_leader_ != component_leader ||
        component->acquired_lane_index_ != lane ||
        component->acquired_crowd_size_ != wfc_list.size())
      throw std::invalid_argument(
          "PsiFormer planned lifecycle lane order differs from resource acquisition");
    if (component->has_proposal_ ||
        component->proposal_origin_ != ProposalOrigin::NONE)
      throw std::logic_error(
          "PsiFormer planned lifecycle operation requires absent proposal state");
    requirePlannedLifecycleParticleSet(*particles, plan, group_index);
  }
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
      request.operation == PlannedOperation::ACCEPT_REJECT_VALUE ||
      request.operation == PlannedOperation::SINGLE_CANCEL;

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
  if (!sameDirectBatchCapacityPlan(workspace_plan, crowd.direct_batch))
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
           BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY,
           BatchExecutionMode::BUFFER_READ,
           BatchExecutionMode::BUFFER_WRITE,
           BatchExecutionMode::PREPARE_GROUP,
           BatchExecutionMode::COMPLETE_UPDATES})
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
  case PlannedOperation::SINGLE_CANCEL:
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
      request.operation == PlannedOperation::ACCEPT_REJECT_VALUE ||
      request.operation == PlannedOperation::SINGLE_CANCEL;
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
  const bool consumes_single_proposal =
      request.operation == PlannedOperation::ACCEPT_REJECT_VALUE ||
      request.operation == PlannedOperation::SINGLE_CANCEL;
  if ((consumes_selected_proposal || consumes_single_proposal) !=
      request.expected_proposal_version.has_value())
    throw std::invalid_argument(
        "PsiFormer proposal-version token is absent or unexpected");
  if (consumes_single_proposal != request.expected_proposal_origin.has_value() ||
      consumes_single_proposal !=
          request.expected_single_transaction_fingerprint.has_value())
    throw std::invalid_argument(
        "PsiFormer single-particle proposal identity token is absent or unexpected");
  if (request.expected_proposal_origin &&
      *request.expected_proposal_origin != ProposalOrigin::MW_CALC_RATIO_VALUE &&
      *request.expected_proposal_origin != ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE)
    throw std::invalid_argument(
        "PsiFormer planned single-particle operation has an incompatible origin");

  const std::optional<std::uint64_t> selected_transaction_fingerprint =
      selected_operation
      ? std::optional<std::uint64_t>(selectedTransactionFingerprint(
            wfc_list, p_list, *request.descriptor_fingerprint))
      : std::nullopt;
  const std::optional<std::uint64_t> single_transaction_fingerprint =
      consumes_single_proposal
      ? std::optional<std::uint64_t>(singleTransactionFingerprint(
            wfc_list, p_list, *request.expected_proposal_origin,
            *request.active_electron, *request.expected_proposal_version))
      : std::nullopt;
  if (consumes_single_proposal &&
      (*request.expected_single_transaction_fingerprint == 0 ||
       *single_transaction_fingerprint == 0))
    throw std::invalid_argument(
        "PsiFormer planned single-particle transaction fingerprint is zero");

  if (selected_operation)
  {
    const std::size_t pending_selected_transactions =
        model_state_->planned_selected_transaction_count.load(
            std::memory_order_acquire);
    if (consumes_selected_proposal && pending_selected_transactions == 0)
      throw std::logic_error(
          "PsiFormer planned selected operation has inconsistent shared transaction state");
  }
  if (consumes_single_proposal &&
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire) == 0)
    throw std::logic_error(
        "PsiFormer planned single-particle operation has inconsistent shared transaction state");

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
          component.proposal_origin_ != *request.expected_proposal_origin ||
          component.proposed_particle_ != static_cast<int>(*request.active_electron) ||
          component.proposed_descriptor_fingerprint_ !=
              *request.expected_single_transaction_fingerprint ||
          component.proposed_descriptor_fingerprint_ !=
              *single_transaction_fingerprint ||
          component.proposed_parameter_version_ !=
              *request.expected_proposal_version ||
          component.proposed_parameter_version_ !=
              component.observed_parameter_version_ ||
          !component.accepted_value_valid_ ||
          (component.accepted_state_requirement_ !=
               AcceptedStateRequirement::VALUE_ONLY &&
           component.accepted_state_requirement_ !=
               AcceptedStateRequirement::FULL_SPATIAL) ||
          component.accepted_parameter_version_ !=
              *request.expected_proposal_version ||
          component.accepted_configuration_identity_ !=
              configurationIdentity(p_list[lane]) ||
          component.proposed_configuration_identity_ !=
              configurationIdentity(
                  p_list[lane], static_cast<int>(*request.active_electron)) ||
          !isCoherentPhysicalValue(*component.model_state_,
                                   component.current_sign_, component.log_value_,
                                   p_list[lane]) ||
          !isCoherentPhysicalValue(
              *component.model_state_, component.proposed_sign_,
              component.proposed_log_value_, p_list[lane],
              static_cast<int>(*request.active_electron), nullptr))
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
          single_transaction_fingerprint, selected_transaction_fingerprint};
}

// Reserve one model-wide one-electron transaction count before publication.
bool PsiFormerWF::tryRegisterPlannedSingleTransaction() const noexcept
{
  std::size_t pending_transactions =
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire);
  while (pending_transactions != std::numeric_limits<std::size_t>::max())
    if (model_state_->planned_single_transaction_count.compare_exchange_weak(
            pending_transactions, pending_transactions + 1,
            std::memory_order_acq_rel, std::memory_order_acquire))
      return true;
  return false;
}

// Withdraw one one-electron crowd transaction after a nonthrowing state clear.
void PsiFormerWF::unregisterPlannedSingleTransaction() const noexcept
{
  std::size_t pending_transactions =
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire);
  for (;;)
  {
    assert(pending_transactions != 0);
    if (pending_transactions == 0)
      std::terminate();
    if (model_state_->planned_single_transaction_count.compare_exchange_weak(
            pending_transactions, pending_transactions - 1,
            std::memory_order_release, std::memory_order_acquire))
      return;
  }
}

// Abandon an exact planned one-electron transaction without consulting model version.
void PsiFormerWF::cancelPlannedSingleProposal(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    std::size_t active_electron,
    ProposalOrigin expected_origin,
    std::size_t expected_proposal_version,
    std::uint64_t expected_transaction_fingerprint) const
{
  PlannedRuntimeRequest request;
  request.operation                = PlannedOperation::SINGLE_CANCEL;
  request.live_walkers             = wfc_list.size();
  request.active_electron          = active_electron;
  request.expected_proposal_origin = expected_origin;
  request.expected_proposal_version = expected_proposal_version;
  request.expected_single_transaction_fingerprint =
      expected_transaction_fingerprint;
  const PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  const std::size_t pending_transactions =
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire);
  if (pending_transactions == 0)
    throw std::logic_error(
        "PsiFormer planned single-particle cancellation has no registered transaction");
  if (!access.single_transaction_fingerprint ||
      *access.single_transaction_fingerprint != expected_transaction_fingerprint)
    throw std::logic_error(
        "PsiFormer planned single-particle cancellation has inconsistent Phase-A evidence");

  // Re-run the complete nonmutating boundary immediately before the marker-last
  // commit so a concurrent lifecycle change can only reject the cancellation.
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &access.resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint ||
      final_access.single_transaction_fingerprint !=
          access.single_transaction_fingerprint ||
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire) == 0)
    throw std::logic_error(
        "PsiFormer planned single-particle cancellation evidence changed during preflight");
  if (fail_planned_single_cancellation_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned single-particle cancellation failure");

  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).resetProposalMetadata();
    for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = false;
    unregisterPlannedSingleTransaction();
  };
  static_assert(noexcept(publish()));
  publish();
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
  {
    diagnostics.full_spatial_bytes = direct_full_spatial_workspace_->vectorStorageBytes();
    diagnostics.full_spatial_is_target = direct_full_spatial_workspace_->mode() == pf::DirectSpatialMode::TARGET_VGL;
    diagnostics.full_spatial_fingerprint = direct_full_spatial_workspace_->storageFingerprint();
  }
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
  diagnostics.scalar_value_publication_identity =
      scalar_value_publication_.data();
  diagnostics.scalar_value_publication_size =
      scalar_value_publication_.size();
  diagnostics.scalar_value_publication_capacity =
      scalar_value_publication_.capacity();
  diagnostics.prepared_scalar_value_compatibility =
      prepared_scalar_value_compatibility_;
  diagnostics.prepared_batch_workspace_identity =
      prepared_batch_workspace_identity_;
  diagnostics.prepared_batch_storage_fingerprint =
      prepared_batch_storage_fingerprint_;
  diagnostics.prepared_batch_bytes = prepared_batch_bytes_;
  diagnostics.prepared_scalar_value_publication_identity =
      prepared_scalar_value_publication_data_;
  diagnostics.prepared_scalar_value_publication_size =
      prepared_scalar_value_publication_size_;
  diagnostics.prepared_scalar_value_publication_capacity =
      prepared_scalar_value_publication_capacity_;
  diagnostics.prepared_walker_buffer_layout =
      prepared_walker_buffer_layout_;
  diagnostics.accepted_spatial_bytes =
      accepted_gradient_.capacity() * sizeof(GradType) +
      accepted_laplacian_.capacity() * sizeof(ValueType);
  diagnostics.proposed_spatial_bytes =
      proposed_gradient_.capacity() * sizeof(GradType) +
      proposed_laplacian_.capacity() * sizeof(ValueType);
  return diagnostics;
}

// Rebind only the prepared scalar capacity descriptor for focused corruption tests.
void PsiFormerWF::setPreparedScalarWorkspaceFaultForTesting(
    PreparedScalarWorkspaceFaultForTesting fault)
{
  if (!batch_execution_plan_ || !prepared_clone_batch_execution_plan_ ||
      !prepared_clone_batch_execution_plan_.sameBinding(batch_execution_plan_) ||
      !prepared_scalar_value_compatibility_ || !direct_batch_workspace_ ||
      direct_batch_workspace_.get() != prepared_batch_workspace_identity_ ||
      !direct_batch_workspace_->hasCapacityPlan() ||
      direct_batch_workspace_->storageFingerprint(
          pf::DirectBatchMode::VALUE_ONLY) !=
          prepared_batch_storage_fingerprint_ ||
      direct_batch_workspace_->vectorStorageBytes() != prepared_batch_bytes_ ||
      scalar_value_publication_.data() !=
          prepared_scalar_value_publication_data_ ||
      scalar_value_publication_.size() !=
          prepared_scalar_value_publication_size_ ||
      scalar_value_publication_.capacity() !=
          prepared_scalar_value_publication_capacity_)
    throw std::logic_error(
        "PsiFormer scalar workspace fault injection requires an exactly prepared owner");

  const psiformer::PsiFormerMemoryPolicyInput input =
      makeBatchMemoryPolicyInput();
  const pf::DirectBatchCapacityPlan canonical_capacity =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, batch_execution_plan_.plan().requirements(),
          batch_execution_plan_.plan().selectedCapacities());
  pf::DirectBatchCapacityPlan selected_capacity = canonical_capacity;

  switch (fault)
  {
  case PreparedScalarWorkspaceFaultForTesting::NONE:
    break;
  case PreparedScalarWorkspaceFaultForTesting::LOGICAL_CAPACITY:
    if (selected_capacity.logical.value_dense == 0)
      throw std::logic_error(
          "PsiFormer scalar logical capacity cannot be corrupted");
    --selected_capacity.logical.value_dense;
    break;
  case PreparedScalarWorkspaceFaultForTesting::TILE_CAPACITY:
    if (selected_capacity.tile.value > 1)
      --selected_capacity.tile.value;
    else if (selected_capacity.tile.full_vgl == 0)
      selected_capacity.tile.full_vgl = 1;
    else
      throw std::logic_error(
          "PsiFormer scalar tile capacity cannot be corrupted");
    break;
  }

  direct_batch_workspace_->prepare(selected_capacity);
  if (fault != PreparedScalarWorkspaceFaultForTesting::NONE)
    return;

  const pf::DirectBatchStorageRequirement expected_storage =
      pf::directBatchStorageRequirement(input.storage_shape,
                                        canonical_capacity);
  const pf::DirectBatchStorageRequirement actual_storage =
      direct_batch_workspace_->actualStorage();
  const std::size_t expected_bytes = expected_storage.executionBytes();
  if (!sameDirectBatchCapacityPlan(direct_batch_workspace_->capacityPlan(),
                                   canonical_capacity) ||
      !sameDirectBatchExecutionStorage(actual_storage, expected_storage) ||
      actual_storage.executionBytes() != expected_bytes ||
      direct_batch_workspace_->vectorStorageBytes() != expected_bytes ||
      scalar_value_publication_.size() !=
          input.scalar_value_logical_maximum ||
      scalar_value_publication_.capacity() !=
          input.scalar_value_logical_maximum)
    throw std::logic_error(
        "PsiFormer scalar workspace fault restoration is not canonical");

  // Publish refreshed evidence only after the restored owner has been proved
  // canonical.  No allocation identity changes during this test-only rebind.
  prepared_scalar_value_compatibility_ = true;
  prepared_batch_workspace_identity_   = direct_batch_workspace_.get();
  prepared_batch_storage_fingerprint_ =
      direct_batch_workspace_->storageFingerprint(
          pf::DirectBatchMode::VALUE_ONLY);
  prepared_batch_bytes_ = expected_bytes;
  prepared_scalar_value_publication_data_ =
      scalar_value_publication_.data();
  prepared_scalar_value_publication_size_ =
      scalar_value_publication_.size();
  prepared_scalar_value_publication_capacity_ =
      scalar_value_publication_.capacity();
}

// Exercise both typed adapter branches without borrowing or modifying a resource.
PsiFormerWF::PsiValue PsiFormerWF::ratioArenaRoundTripForTesting(
    PsiValue value, bool use_log_value_arena)
{
  PsiValue psi_value{};
  LogValue log_value{};
  PsiFormerMultiWalkerResource::RatioArenaView arena;
  arena.extent = 1;
  if (use_log_value_arena)
  {
    arena.kind = PsiFormerMultiWalkerResource::RatioArenaKind::LOG_VALUE;
    arena.log_value_data = &log_value;
  }
  else
  {
    arena.kind = PsiFormerMultiWalkerResource::RatioArenaKind::PSI_VALUE;
    arena.psi_value_data = &psi_value;
  }
  arena.store(0, value);
  const PsiValue checked = arena.load(0);
  const PsiValue unchecked = arena.loadUnchecked(0);
  if (checked != unchecked)
    throw std::logic_error(
        "PsiFormer checked and nonthrowing ratio-arena loads disagree");
  return unchecked;
}

// Exercise malformed adapter/resource evidence without exposing mutable arena
// storage to tests or adding any branch to a production execution path.
PsiFormerWF::PsiValue PsiFormerWF::ratioArenaFaultForTesting(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    RatioArenaFaultForTesting fault) const
{
  using Resource  = PsiFormerMultiWalkerResource;
  using ArenaKind = Resource::RatioArenaKind;

  if (fault == RatioArenaFaultForTesting::NONZERO_LOG_IMAGINARY)
  {
    LogValue log_value{FullPrecRealType{1.25}, FullPrecRealType{-0.375}};
    Resource::RatioArenaView arena{ArenaKind::LOG_VALUE, nullptr, &log_value,
                                   1};
    return arena.load(0);
  }

  Resource& resource = requireMultiWalkerResource(wfc_list);
  if (!resource.prepared_crowd_plan)
    throw std::logic_error(
        "PsiFormer ratio-arena fault test requires a prepared resource");

  /** Restore every directly corrupted field and any swapped vector allocation
   * during stack unwinding.  Moving the original vector into this guard keeps
   * its exact data pointer, size, capacity, and contents intact. */
  class RatioArenaRestorer
  {
  public:
    explicit RatioArenaRestorer(Resource& resource) noexcept
        : resource_(resource),
          ratio_arena_kind_(resource.ratio_arena_kind),
          prepared_ratio_arena_kind_(resource.prepared_ratio_arena_kind),
          prepared_storage_fingerprint_(resource.prepared_storage_fingerprint),
          prepared_psi_value_data_(resource.prepared_staged_value_ratios_data),
          prepared_log_value_data_(resource.prepared_staged_log_ratios_data),
          prepared_psi_value_size_(resource.prepared_staged_value_ratios_size),
          prepared_log_value_size_(resource.prepared_staged_log_ratios_size),
          prepared_psi_value_capacity_(
              resource.prepared_staged_value_ratios_capacity),
          prepared_log_value_capacity_(
              resource.prepared_staged_log_ratios_capacity)
    {}

    RatioArenaRestorer(const RatioArenaRestorer&)            = delete;
    RatioArenaRestorer& operator=(const RatioArenaRestorer&) = delete;

    ~RatioArenaRestorer() noexcept
    {
      if (restore_psi_value_vector_)
        resource_.staged_value_ratios.swap(saved_psi_value_vector_);
      if (restore_log_value_vector_)
        resource_.staged_log_ratios.swap(saved_log_value_vector_);
      resource_.ratio_arena_kind = ratio_arena_kind_;
      resource_.prepared_ratio_arena_kind = prepared_ratio_arena_kind_;
      resource_.prepared_storage_fingerprint = prepared_storage_fingerprint_;
      resource_.prepared_staged_value_ratios_data = prepared_psi_value_data_;
      resource_.prepared_staged_log_ratios_data = prepared_log_value_data_;
      resource_.prepared_staged_value_ratios_size = prepared_psi_value_size_;
      resource_.prepared_staged_log_ratios_size = prepared_log_value_size_;
      resource_.prepared_staged_value_ratios_capacity =
          prepared_psi_value_capacity_;
      resource_.prepared_staged_log_ratios_capacity =
          prepared_log_value_capacity_;
    }

    void replacePsiValueVector(std::vector<PsiValue> replacement) noexcept
    {
      saved_psi_value_vector_.swap(resource_.staged_value_ratios);
      resource_.staged_value_ratios.swap(replacement);
      restore_psi_value_vector_ = true;
    }

    void replaceLogValueVector(std::vector<LogValue> replacement) noexcept
    {
      saved_log_value_vector_.swap(resource_.staged_log_ratios);
      resource_.staged_log_ratios.swap(replacement);
      restore_log_value_vector_ = true;
    }

  private:
    Resource& resource_;
    ArenaKind ratio_arena_kind_;
    ArenaKind prepared_ratio_arena_kind_;
    std::size_t prepared_storage_fingerprint_;
    const PsiValue* prepared_psi_value_data_;
    const LogValue* prepared_log_value_data_;
    std::size_t prepared_psi_value_size_;
    std::size_t prepared_log_value_size_;
    std::size_t prepared_psi_value_capacity_;
    std::size_t prepared_log_value_capacity_;
    std::vector<PsiValue> saved_psi_value_vector_;
    std::vector<LogValue> saved_log_value_vector_;
    bool restore_psi_value_vector_ = false;
    bool restore_log_value_vector_ = false;
  } restore(resource);

  const auto require_active_arena = [&resource]() {
    if (resource.ratio_arena_kind == ArenaKind::NONE)
      throw std::logic_error(
          "PsiFormer ratio-arena fault test requires an active arena");
  };
  const auto change_extent = [](std::size_t& extent) noexcept {
    extent = extent == std::numeric_limits<std::size_t>::max() ? extent - 1
                                                               : extent + 1;
  };
  const std::size_t valid_prefix =
      resource.prepared_crowd_plan->publication_staging.reserve_walkers;
  std::size_t tested_prefix = valid_prefix;

  switch (fault)
  {
  case RatioArenaFaultForTesting::WRONG_KIND:
    require_active_arena();
    resource.ratio_arena_kind =
        resource.ratio_arena_kind == ArenaKind::LOG_VALUE
        ? ArenaKind::PSI_VALUE
        : ArenaKind::LOG_VALUE;
    resource.prepared_storage_fingerprint =
        resource.currentStorageFingerprint();
    break;
  case RatioArenaFaultForTesting::WRONG_PREFIX:
    if (valid_prefix == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error(
          "PsiFormer ratio-arena fault prefix cannot be incremented");
    tested_prefix = valid_prefix + 1;
    break;
  case RatioArenaFaultForTesting::CHANGED_POINTER:
    require_active_arena();
    if (resource.ratio_arena_kind == ArenaKind::LOG_VALUE)
    {
      static const LogValue foreign_log_value{};
      resource.prepared_staged_log_ratios_data = &foreign_log_value;
    }
    else
    {
      static const PsiValue foreign_psi_value{};
      resource.prepared_staged_value_ratios_data = &foreign_psi_value;
    }
    break;
  case RatioArenaFaultForTesting::CHANGED_SIZE:
    require_active_arena();
    if (resource.ratio_arena_kind == ArenaKind::LOG_VALUE)
      change_extent(resource.prepared_staged_log_ratios_size);
    else
      change_extent(resource.prepared_staged_value_ratios_size);
    break;
  case RatioArenaFaultForTesting::CHANGED_CAPACITY:
    require_active_arena();
    if (resource.ratio_arena_kind == ArenaKind::LOG_VALUE)
      change_extent(resource.prepared_staged_log_ratios_capacity);
    else
      change_extent(resource.prepared_staged_value_ratios_capacity);
    break;
  case RatioArenaFaultForTesting::DUAL_ARENAS:
    require_active_arena();
    if (resource.ratio_arena_kind == ArenaKind::LOG_VALUE)
    {
      restore.replacePsiValueVector(
          std::vector<PsiValue>(valid_prefix, PsiValue{}));
      resource.prepared_staged_value_ratios_data =
          resource.staged_value_ratios.data();
      resource.prepared_staged_value_ratios_size =
          resource.staged_value_ratios.size();
      resource.prepared_staged_value_ratios_capacity =
          resource.staged_value_ratios.capacity();
    }
    else
    {
      restore.replaceLogValueVector(
          std::vector<LogValue>(valid_prefix, LogValue{}));
      resource.prepared_staged_log_ratios_data =
          resource.staged_log_ratios.data();
      resource.prepared_staged_log_ratios_size =
          resource.staged_log_ratios.size();
      resource.prepared_staged_log_ratios_capacity =
          resource.staged_log_ratios.capacity();
    }
    resource.prepared_storage_fingerprint =
        resource.currentStorageFingerprint();
    break;
  case RatioArenaFaultForTesting::NO_ARENA:
    require_active_arena();
    if (resource.ratio_arena_kind == ArenaKind::LOG_VALUE)
    {
      restore.replaceLogValueVector({});
      resource.prepared_staged_log_ratios_data =
          resource.staged_log_ratios.data();
      resource.prepared_staged_log_ratios_size =
          resource.staged_log_ratios.size();
      resource.prepared_staged_log_ratios_capacity =
          resource.staged_log_ratios.capacity();
    }
    else
    {
      restore.replacePsiValueVector({});
      resource.prepared_staged_value_ratios_data =
          resource.staged_value_ratios.data();
      resource.prepared_staged_value_ratios_size =
          resource.staged_value_ratios.size();
      resource.prepared_staged_value_ratios_capacity =
          resource.staged_value_ratios.capacity();
    }
    resource.prepared_storage_fingerprint =
        resource.currentStorageFingerprint();
    break;
  case RatioArenaFaultForTesting::NONZERO_LOG_IMAGINARY:
    break;
  }

  resource.requireRatioArenaStaging(tested_prefix);
  return {};
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
  diagnostics.successful_batch_generation =
      resource.batch_workspace->successfulGeneration();
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
      resource.staged_value_ratios.capacity() * sizeof(PsiValue) +
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
  const auto diagnostic_kind = [](PsiFormerMultiWalkerResource::RatioArenaKind kind) {
    switch (kind)
    {
    case PsiFormerMultiWalkerResource::RatioArenaKind::NONE:
      return testing::PsiFormerRatioArenaKind::NONE;
    case PsiFormerMultiWalkerResource::RatioArenaKind::PSI_VALUE:
      return testing::PsiFormerRatioArenaKind::PSI_VALUE;
    case PsiFormerMultiWalkerResource::RatioArenaKind::LOG_VALUE:
      return testing::PsiFormerRatioArenaKind::LOG_VALUE;
    }
    return testing::PsiFormerRatioArenaKind::NONE;
  };
  diagnostics.ratio_arena.kind = diagnostic_kind(resource.ratio_arena_kind);
  diagnostics.ratio_arena.prepared_kind =
      diagnostic_kind(resource.prepared_ratio_arena_kind);
  diagnostics.ratio_arena.psi_value_data =
      resource.staged_value_ratios.data();
  diagnostics.ratio_arena.log_value_data = resource.staged_log_ratios.data();
  diagnostics.ratio_arena.prepared_psi_value_data =
      resource.prepared_staged_value_ratios_data;
  diagnostics.ratio_arena.prepared_log_value_data =
      resource.prepared_staged_log_ratios_data;
  diagnostics.ratio_arena.psi_value_size =
      resource.staged_value_ratios.size();
  diagnostics.ratio_arena.log_value_size = resource.staged_log_ratios.size();
  diagnostics.ratio_arena.psi_value_capacity =
      resource.staged_value_ratios.capacity();
  diagnostics.ratio_arena.log_value_capacity =
      resource.staged_log_ratios.capacity();
  diagnostics.ratio_arena.prepared_psi_value_size =
      resource.prepared_staged_value_ratios_size;
  diagnostics.ratio_arena.prepared_log_value_size =
      resource.prepared_staged_log_ratios_size;
  diagnostics.ratio_arena.prepared_psi_value_capacity =
      resource.prepared_staged_value_ratios_capacity;
  diagnostics.ratio_arena.prepared_log_value_capacity =
      resource.prepared_staged_log_ratios_capacity;
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

// Expose the one-electron guard count without granting a production lifecycle API.
std::size_t PsiFormerWF::plannedSingleTransactionCountForTesting() const noexcept
{
  return model_state_->planned_single_transaction_count.load(
      std::memory_order_acquire);
}

// Deliberately create model-version drift behind a pending proposal for testing.
std::size_t PsiFormerWF::advanceParameterVersionForTesting()
{
  auto state_lock = model_state_->acquireHostWrite();
  pf::Parameters& parameters = model_state_->model.p;
  const std::vector<double> unchanged_values = parameters.flat_values();
  parameters.set_flat_values(unchanged_values);
  model_state_->host_mirror_version = parameters.version();
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

// Expose only immutable route evidence and clone-family ownership to tests.
testing::PsiFormerAcceleratorDiagnostics
PsiFormerWF::acceleratorDiagnosticsForTesting() const noexcept
{
  const psiformer::PsiFormerAcceleratorResolution& resolution =
      model_state_->accelerator_resolution;
  return {resolution.backend,
          resolution.reason,
          resolution.readiness_records_consulted,
          resolution.fallback_count,
          model_state_.get(),
          static_cast<std::size_t>(model_state_.use_count())};
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

  PsiFormerReadTransaction state_lock(*model_state_);
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
    // Legacy multiwalker proposals use the same origins but do not carry a
    // hard-plan binding.  Only a plan-bound proposal participates in the
    // model-wide planned transaction whose state must not be invalidated.
    const bool has_planned_proposal = batch_execution_plan_ && has_proposal_ &&
        (proposal_origin_ == ProposalOrigin::MW_CALC_RATIO_VALUE ||
         proposal_origin_ == ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE ||
         proposal_origin_ == ProposalOrigin::MW_SELECTED_FULL_VGL);
    if (has_planned_proposal)
      throw std::logic_error(
          "PsiFormer planned proposal cannot be cleared by parameter-version synchronization");
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
                                              LogValue physical_log_value,
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
  proposed_log_value_              = physical_log_value;
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

// Planned crowd state cannot be stranded by a shared-parameter publisher.
void PsiFormerWF::requireNoPlannedProposalMutation(const char* operation) const
{
  // The MW origins are also used by the preserved no-plan compatibility path.
  // Only a bound hard plan makes lane-local metadata a planned transaction;
  // model-wide counters protect mutations invoked through detached siblings.
  const bool has_planned_proposal = batch_execution_plan_ && has_proposal_ &&
      (proposal_origin_ == ProposalOrigin::MW_CALC_RATIO_VALUE ||
       proposal_origin_ == ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE ||
       proposal_origin_ == ProposalOrigin::MW_SELECTED_FULL_VGL);
  if (model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire) != 0 ||
      model_state_->planned_selected_transaction_count.load(
          std::memory_order_acquire) != 0 ||
      has_planned_proposal)
    throw std::logic_error(
        std::string("PsiFormer ") + operation +
        " cannot mutate parameters while a planned crowd proposal is pending");
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

// Validate replay equality without evaluating, synchronizing, or publishing any state.
bool PsiFormerWF::canReuseFullSpatialReplay(const ParticleSet& live,
                                           const ParticleSet& saved,
                                           std::size_t expected_version,
                                           double expected_log,
                                           double expected_phase) const
{
  PsiFormerReadTransaction transaction(*model_state_, model_state_->resident_device_authority_enabled);
  const auto count = static_cast<std::size_t>(live.getTotalNum());
  if (live.R.size() != count || saved.R.size() != count ||
      live.GroupID.size() != count || saved.GroupID.size() != count)
    return false;
  if (transaction.parameterVersion() != expected_version || observed_parameter_version_ != expected_version ||
      has_proposal_ || proposal_origin_ != ProposalOrigin::NONE || restore_validation_pending_ ||
      model_state_->planned_single_transaction_count.load(std::memory_order_acquire) != 0 ||
      model_state_->planned_selected_transaction_count.load(std::memory_order_acquire) != 0 ||
      !acceptedStateMatches(live, expected_version, AcceptedStateRequirement::FULL_SPATIAL) ||
      !isCoherentPhysicalValue(transaction.state(), current_sign_, log_value_, live) ||
      !walkerBufferScalarIsFinite(expected_log) || !walkerBufferScalarIsFinite(expected_phase) ||
      std::real(log_value_) != expected_log || std::imag(log_value_) != expected_phase ||
      live.isSpinor() || saved.isSpinor() || live.getTotalNum() != saved.getTotalNum() ||
      live.groups() != saved.groups() || live.isSameMass() != saved.isSameMass() ||
      live.get_mass_by_group().size() != saved.get_mass_by_group().size() ||
      live.G.size() != count || saved.G.size() != count || live.L.size() != count || saved.L.size() != count ||
      live.spins.size() != count || saved.spins.size() != count ||
      live.Z.size() != count || saved.Z.size() != count ||
      accepted_gradient_.size() != count || accepted_laplacian_.size() != count)
    return false;
  for (int group = 0; group < live.groups(); ++group)
    if (live.first(group) != saved.first(group) || live.last(group) != saved.last(group))
      return false;
  for (std::size_t group = 0; group < live.get_mass_by_group().size(); ++group)
    if (!walkerBufferScalarIsFinite(live.get_mass_by_group()[group]) ||
        !walkerBufferScalarIsFinite(saved.get_mass_by_group()[group]) ||
        live.get_mass_by_group()[group] != saved.get_mass_by_group()[group])
      return false;
  for (std::size_t electron = 0; electron < count; ++electron)
  {
    if (!walkerBufferScalarIsFinite(live.spins[electron]) || !walkerBufferScalarIsFinite(saved.spins[electron]) ||
        live.spins[electron] != saved.spins[electron] || live.GroupID[electron] != saved.GroupID[electron] ||
        !walkerBufferScalarIsFinite(live.Z[electron]) || !walkerBufferScalarIsFinite(saved.Z[electron]) ||
        live.Z[electron] != saved.Z[electron] || !walkerBufferScalarIsFinite(live.L[electron]) ||
        !walkerBufferScalarIsFinite(saved.L[electron]) || !walkerBufferScalarIsFinite(accepted_laplacian_[electron]) ||
        live.L[electron] != saved.L[electron] || live.L[electron] != accepted_laplacian_[electron])
      return false;
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      if (!walkerBufferScalarIsFinite(live.R[electron][dimension]) ||
          !walkerBufferScalarIsFinite(saved.R[electron][dimension]) ||
          live.R[electron][dimension] != saved.R[electron][dimension] ||
          !walkerBufferScalarIsFinite(live.G[electron][dimension]) ||
          !walkerBufferScalarIsFinite(saved.G[electron][dimension]) ||
          !walkerBufferScalarIsFinite(accepted_gradient_[electron][dimension]) ||
          live.G[electron][dimension] != saved.G[electron][dimension] ||
          live.G[electron][dimension] != accepted_gradient_[electron][dimension])
        return false;
  }
  return true;
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

// Prove one planned walker-buffer lane without borrowing crowd-owned resources.
PsiFormerWF::WalkerBufferCursorSnapshot
PsiFormerWF::requirePlannedWalkerBufferOperation(
    PlannedWalkerBufferOperation operation,
    const ParticleSet& particles,
    const WFBufferType& buffer) const
{
  if (!batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer planned walker-buffer operation has no bound batch plan");
  if (!hasPreparedBatchExecutionClone(batch_execution_plan_))
    throw std::logic_error(
        "PsiFormer planned walker-buffer operation requires an exactly prepared clone");
  if (bound_particle_set_ != std::addressof(particles))
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation received a foreign ParticleSet");
  if (has_proposal_ || proposal_origin_ != ProposalOrigin::NONE)
    throw std::logic_error(
        "PsiFormer planned walker-buffer operation requires absent proposal state");

  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  const BatchExecutionMode required_mode =
      operation == PlannedWalkerBufferOperation::READ
      ? BatchExecutionMode::BUFFER_READ
      : BatchExecutionMode::BUFFER_WRITE;
  if (!plan.requirements().requires(required_mode))
    throw std::logic_error(
        "PsiFormer walker-buffer operation is not admitted by its explicit batch mode");
  if (plan.targetCoordinate() != BatchExecutionTargetCoordinate::POS_ONLY)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation requires explicit POS-only target evidence");
  const BatchExecutionTopology& topology = plan.topology();
  if (topology.serialized_walkers || topology.backend_id != "cpu" ||
      topology.device_id)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation requires direct nonserialized CPU execution");

  const psiformer::ModelShape& model_shape =
      model_state_->execution_plan.modelShape();
  const std::size_t electron_count = model_shape.electrons();
  const auto& soa_positions =
      particles.getCoordinates().getAllParticlePos();
  if (plan.particleCount() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer particle count differs from its model");
  if (particles.isSpinor())
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation received a spinor ParticleSet");
  if (particles.getTotalNum() < 0 ||
      static_cast<std::size_t>(particles.getTotalNum()) != electron_count ||
      particles.R.size() != electron_count ||
      soa_positions.size() != electron_count ||
      soa_positions.capacity() < electron_count ||
      particles.G.size() != electron_count ||
      particles.L.size() != electron_count ||
      particles.GroupID.size() != electron_count ||
      particles.spins.size() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation received incompatible ParticleSet extents");
  if (electron_count != 0 &&
      (!particles.R.data() || !soa_positions.data() || !particles.G.data() ||
       !particles.L.data() || !particles.GroupID.data() ||
       !particles.spins.data()))
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation received null ParticleSet storage");
  if (particles.groups() != 2 ||
      particles.groupsize(0) !=
          static_cast<int>(model_shape.spin_up_electrons) ||
      particles.groupsize(1) !=
          static_cast<int>(model_shape.spin_down_electrons))
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation received an incompatible spin partition");
  if (particles.getActivePtcl() != -1)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer operation requires an inactive ParticleSet move");

  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const auto soa_position = soa_positions[electron];
    const int expected_group =
        electron < model_shape.spin_up_electrons ? 0 : 1;
    if (particles.GroupID[electron] != expected_group)
      throw std::invalid_argument(
          "PsiFormer planned walker-buffer operation received noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      const auto& aos_coordinate = particles.R[electron][dimension];
      const auto& soa_coordinate = soa_position[dimension];
      if (!psiformer::determinant::isFiniteReal(
              static_cast<double>(aos_coordinate)))
        throw std::invalid_argument(
            "PsiFormer planned walker-buffer operation received a non-finite position");
      if (std::memcmp(std::addressof(aos_coordinate),
                      std::addressof(soa_coordinate),
                      sizeof(aos_coordinate)) != 0)
        throw std::invalid_argument(
            "PsiFormer planned walker-buffer operation has inconsistent AoS and SoA positions");
    }
  }

  const pf::WalkerBufferLayout expected_layout =
      psiformer::makePsiFormerWalkerBufferLayout(makeBatchMemoryPolicyInput());
  if (prepared_walker_buffer_layout_ != expected_layout ||
      prepared_walker_buffer_layout_.fingerprint == 0)
    throw std::logic_error(
        "PsiFormer planned walker-buffer layout differs from its prepared evidence");

  WalkerBufferCursorSnapshot snapshot;
  snapshot.operation              = operation;
  snapshot.buffer                 = std::addressof(buffer);
  snapshot.plan_identity          = std::addressof(plan);
  snapshot.data                   = buffer.myData.data();
  snapshot.scalar_data            = buffer.Scalar_ptr;
  snapshot.size                   = buffer.myData.size();
  snapshot.capacity               = buffer.myData.capacity();
  snapshot.bulk_cursor            = buffer.current();
  snapshot.scalar_cursor          = buffer.current_scalar();
  snapshot.attached_storage       = buffer.myData.isAttached();
  snapshot.plan_fingerprint       = plan.fingerprint();
  snapshot.layout_fingerprint     = prepared_walker_buffer_layout_.fingerprint;
  snapshot.configuration_identity = configurationIdentity(particles);

  if (operation == PlannedWalkerBufferOperation::REGISTER)
  {
    if (snapshot.size != 0 || snapshot.scalar_data != nullptr ||
        snapshot.attached_storage)
      throw std::invalid_argument(
          "PsiFormer planned walker-buffer registration requires sizing-phase storage");
    static_cast<void>(pf::checkedStorageSum(
        snapshot.bulk_cursor, prepared_walker_buffer_layout_.bulk_bytes,
        "PsiFormer planned walker-buffer bulk cursor overflowed"));
    static_cast<void>(pf::checkedStorageSum(
        snapshot.scalar_cursor, prepared_walker_buffer_layout_.scalar_count,
        "PsiFormer planned walker-buffer scalar cursor overflowed"));
    if (snapshot.bulk_cursor % prepared_walker_buffer_layout_.alignment != 0)
      throw std::invalid_argument(
          "PsiFormer planned walker-buffer registration cursor is misaligned");
    snapshot = fingerprintWalkerBufferCursorSnapshot(snapshot);
  }
  else
    snapshot = validateWalkerBufferCursorSnapshot(snapshot);

  std::uint64_t input_hash = PERSISTENT_FINGERPRINT_OFFSET;
  const auto mix_pointer = [&input_hash](const void* pointer) noexcept {
    mixPersistentInteger(
        input_hash,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)));
  };
  mixPersistentInteger(input_hash, WALKER_BUFFER_INPUT_FINGERPRINT_DOMAIN);
  mixPersistentInteger(input_hash, snapshot.plan_fingerprint);
  mixPersistentInteger(input_hash, snapshot.layout_fingerprint);
  mixPersistentInteger(input_hash, snapshot.storage_fingerprint);
  mixPersistentInteger(input_hash, snapshot.configuration_identity);
  mix_pointer(this);
  mix_pointer(snapshot.plan_identity);
  mix_pointer(model_state_.get());
  mix_pointer(prepared_accepted_gradient_data_);
  mix_pointer(prepared_accepted_laplacian_data_);
  mix_pointer(prepared_proposed_gradient_data_);
  mix_pointer(prepared_proposed_laplacian_data_);
  mixPersistentInteger(input_hash, prepared_accepted_gradient_capacity_);
  mixPersistentInteger(input_hash, prepared_accepted_laplacian_capacity_);
  mixPersistentInteger(input_hash, prepared_proposed_gradient_capacity_);
  mixPersistentInteger(input_hash, prepared_proposed_laplacian_capacity_);
  mix_pointer(std::addressof(particles));
  mix_pointer(particles.R.data());
  mixPersistentInteger(input_hash,
                       static_cast<std::uint64_t>(particles.R.capacity()));
  mix_pointer(soa_positions.data());
  mixPersistentInteger(input_hash,
                       static_cast<std::uint64_t>(soa_positions.capacity()));
  mix_pointer(particles.G.data());
  mixPersistentInteger(input_hash,
                       static_cast<std::uint64_t>(particles.G.capacity()));
  mix_pointer(particles.L.data());
  mixPersistentInteger(input_hash,
                       static_cast<std::uint64_t>(particles.L.capacity()));
  mix_pointer(particles.GroupID.data());
  mixPersistentInteger(
      input_hash, static_cast<std::uint64_t>(particles.GroupID.capacity()));
  mix_pointer(particles.spins.data());
  mixPersistentInteger(input_hash,
                       static_cast<std::uint64_t>(particles.spins.capacity()));
  snapshot.input_fingerprint = input_hash == 0 ? 1 : input_hash;
  return snapshot;
}

// Validate the two discontiguous allocated PooledMemory address spaces numerically.
PsiFormerWF::WalkerBufferCursorSnapshot
PsiFormerWF::validateWalkerBufferCursorSnapshot(
    WalkerBufferCursorSnapshot snapshot) const
{
  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;
  if (!snapshot.buffer)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer snapshot has no buffer identity");
  if (snapshot.attached_storage)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer storage must own its backing allocation");
  if (snapshot.size == 0 || !snapshot.data)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer storage has no allocated backing");
  if (snapshot.size > snapshot.capacity)
    throw std::length_error(
        "PsiFormer planned walker-buffer logical size exceeds retained capacity");
  if (!snapshot.scalar_data)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer storage has no scalar region");
  if (layout.empty() || layout.alignment == 0 ||
      layout.scalar_count != pf::WALKER_BUFFER_SCALAR_COUNT)
    throw std::logic_error(
        "PsiFormer planned walker-buffer prepared layout is incomplete");

  const std::uintptr_t data_address =
      reinterpret_cast<std::uintptr_t>(snapshot.data);
  const std::uintptr_t data_end = checkedAddressEnd(
      snapshot.data, snapshot.size,
      "PsiFormer planned walker-buffer backing address overflowed");
  const std::uintptr_t scalar_address =
      reinterpret_cast<std::uintptr_t>(snapshot.scalar_data);
  if (data_address % layout.alignment != 0)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer backing is misaligned");
  if (scalar_address < data_address || scalar_address > data_end)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer scalar pointer is outside its backing");
  if (scalar_address % alignof(FullPrecRealType) != 0)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer scalar pointer is misaligned");

  snapshot.scalar_offset =
      static_cast<std::size_t>(scalar_address - data_address);
  const std::size_t scalar_tail_bytes = snapshot.size - snapshot.scalar_offset;
  if (scalar_tail_bytes % sizeof(FullPrecRealType) != 0)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer scalar tail has a fractional element");
  snapshot.scalar_capacity = scalar_tail_bytes / sizeof(FullPrecRealType);

  const std::size_t next_bulk = pf::checkedStorageSum(
      snapshot.bulk_cursor, layout.bulk_bytes,
      "PsiFormer planned walker-buffer bulk cursor overflowed");
  const std::size_t next_scalar = pf::checkedStorageSum(
      snapshot.scalar_cursor, layout.scalar_count,
      "PsiFormer planned walker-buffer scalar cursor overflowed");
  if (snapshot.bulk_cursor > snapshot.scalar_offset ||
      next_bulk > snapshot.scalar_offset)
    throw std::length_error(
        "PsiFormer planned walker-buffer bulk record is truncated");
  if (snapshot.scalar_cursor > snapshot.scalar_capacity ||
      next_scalar > snapshot.scalar_capacity)
    throw std::length_error(
        "PsiFormer planned walker-buffer scalar record is truncated");
  if (snapshot.bulk_cursor % layout.alignment != 0)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer bulk cursor is misaligned");

  const std::size_t laplacian_cursor = pf::checkedStorageSum(
      snapshot.bulk_cursor, layout.laplacian_offset,
      "PsiFormer planned walker-buffer Laplacian cursor overflowed");
  const std::uintptr_t gradient_address = data_address + snapshot.bulk_cursor;
  const std::uintptr_t laplacian_address = data_address + laplacian_cursor;
  if (gradient_address % alignof(GradType) != 0 ||
      laplacian_address % alignof(ValueType) != 0)
    throw std::invalid_argument(
        "PsiFormer planned walker-buffer typed bulk view is misaligned");

  return fingerprintWalkerBufferCursorSnapshot(snapshot);
}

// Hash the exact operation and sizing or allocated PooledMemory observation.
PsiFormerWF::WalkerBufferCursorSnapshot
PsiFormerWF::fingerprintWalkerBufferCursorSnapshot(
    WalkerBufferCursorSnapshot snapshot) noexcept
{
  std::uint64_t storage_hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(storage_hash,
                       WALKER_BUFFER_STORAGE_FINGERPRINT_DOMAIN);
  for (const std::uint64_t value : {
           static_cast<std::uint64_t>(snapshot.operation),
           static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
               snapshot.buffer)),
           static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
               snapshot.data)),
           static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
               snapshot.scalar_data)),
           static_cast<std::uint64_t>(snapshot.size),
           static_cast<std::uint64_t>(snapshot.capacity),
           static_cast<std::uint64_t>(snapshot.bulk_cursor),
           static_cast<std::uint64_t>(snapshot.scalar_cursor),
           static_cast<std::uint64_t>(snapshot.scalar_offset),
           static_cast<std::uint64_t>(snapshot.scalar_capacity),
           static_cast<std::uint64_t>(snapshot.attached_storage),
           snapshot.plan_fingerprint, snapshot.layout_fingerprint})
    mixPersistentInteger(storage_hash, value);
  snapshot.storage_fingerprint = storage_hash == 0 ? 1 : storage_hash;
  return snapshot;
}

// Reject any component record range that aliases its parse or publication inputs.
void PsiFormerWF::requireDisjointWalkerBufferRecord(
    const ParticleSet& particles,
    const WalkerBufferRecordView& record) const
{
  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;
  const auto reject_overlap = [&](const void* destination,
                                  std::size_t destination_bytes,
                                  const char* description) {
    for (const auto& source : {
             std::pair<const void*, std::size_t>{record.gradient_data,
                                                  layout.gradient_bytes},
             {record.laplacian_data, layout.laplacian_bytes},
             {record.scalar_data, record.scalar_payload_bytes}})
      if (checkedByteRangesOverlap(source.first, source.second, destination,
                                   destination_bytes, description))
        throw std::invalid_argument(description);
  };
  const auto checked_bytes = [](std::size_t elements, std::size_t width,
                                const char* description) {
    return pf::checkedStorageProduct(elements, width, description);
  };

  reject_overlap(
      accepted_gradient_.data(),
      checked_bytes(accepted_gradient_.size(), sizeof(GradType),
                    "PsiFormer accepted-gradient range overflowed"),
      "PsiFormer planned walker buffer aliases accepted gradient storage");
  reject_overlap(
      accepted_laplacian_.data(),
      checked_bytes(accepted_laplacian_.size(), sizeof(ValueType),
                    "PsiFormer accepted-Laplacian range overflowed"),
      "PsiFormer planned walker buffer aliases accepted Laplacian storage");
  reject_overlap(
      proposed_gradient_.data(),
      checked_bytes(proposed_gradient_.size(), sizeof(GradType),
                    "PsiFormer proposed-gradient range overflowed"),
      "PsiFormer planned walker buffer aliases proposed gradient storage");
  reject_overlap(
      proposed_laplacian_.data(),
      checked_bytes(proposed_laplacian_.size(), sizeof(ValueType),
                    "PsiFormer proposed-Laplacian range overflowed"),
      "PsiFormer planned walker buffer aliases proposed Laplacian storage");
  reject_overlap(
      scalar_value_publication_.data(),
      checked_bytes(scalar_value_publication_.size(), sizeof(ValueType),
                    "PsiFormer scalar-publication range overflowed"),
      "PsiFormer planned walker buffer aliases scalar publication storage");

  const auto& soa_positions =
      particles.getCoordinates().getAllParticlePos();
  reject_overlap(
      particles.R.data(),
      checked_bytes(particles.R.size(), sizeof(ParticleSet::PosType),
                    "PsiFormer ParticleSet position range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet positions");
  reject_overlap(
      soa_positions.data(),
      checked_bytes(
          pf::checkedStorageProduct(
              soa_positions.capacity(), std::size_t{OHMMS_DIM},
              "PsiFormer ParticleSet SoA extent overflowed"),
          sizeof(soa_positions.data()[0]),
          "PsiFormer ParticleSet SoA range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet SoA positions");
  reject_overlap(
      particles.G.data(),
      checked_bytes(particles.G.size(),
                    sizeof(ParticleSet::ParticleGradient::value_type),
                    "PsiFormer ParticleSet gradient range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet gradients");
  reject_overlap(
      particles.L.data(),
      checked_bytes(particles.L.size(),
                    sizeof(ParticleSet::ParticleLaplacian::value_type),
                    "PsiFormer ParticleSet Laplacian range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet Laplacians");
  reject_overlap(
      particles.GroupID.data(),
      checked_bytes(particles.GroupID.size(),
                    sizeof(ParticleSet::ParticleIndex::value_type),
                    "PsiFormer ParticleSet group range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet group storage");
  reject_overlap(
      particles.spins.data(),
      checked_bytes(particles.spins.size(),
                    sizeof(ParticleSet::ParticleScalar::value_type),
                    "PsiFormer ParticleSet spin range overflowed"),
      "PsiFormer planned walker buffer aliases ParticleSet spin storage");
}

// Form exact nonowning record ranges without requiring old destination bytes to be valid.
PsiFormerWF::WalkerBufferRecordView
PsiFormerWF::makePlannedWalkerBufferRecordView(
    const ParticleSet& particles,
    WalkerBufferCursorSnapshot snapshot) const
{
  snapshot = validateWalkerBufferCursorSnapshot(snapshot);
  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;

  WalkerBufferRecordView record;
  record.cursor = snapshot;
  record.gradient_payload_bytes = pf::checkedStorageProduct(
      layout.electrons, sizeof(GradType),
      "PsiFormer planned walker-buffer gradient payload overflowed");
  record.laplacian_payload_bytes = pf::checkedStorageProduct(
      layout.electrons, sizeof(ValueType),
      "PsiFormer planned walker-buffer Laplacian payload overflowed");
  record.scalar_payload_bytes = pf::checkedStorageProduct(
      layout.scalar_count, sizeof(FullPrecRealType),
      "PsiFormer planned walker-buffer scalar payload overflowed");
  record.next_bulk_cursor = pf::checkedStorageSum(
      snapshot.bulk_cursor, layout.bulk_bytes,
      "PsiFormer planned walker-buffer bulk cursor overflowed");
  record.next_scalar_cursor = pf::checkedStorageSum(
      snapshot.scalar_cursor, layout.scalar_count,
      "PsiFormer planned walker-buffer scalar cursor overflowed");
  const std::size_t laplacian_cursor = pf::checkedStorageSum(
      snapshot.bulk_cursor, layout.laplacian_offset,
      "PsiFormer planned walker-buffer Laplacian cursor overflowed");
  const std::size_t scalar_byte_cursor = pf::checkedStorageProduct(
      snapshot.scalar_cursor, sizeof(FullPrecRealType),
      "PsiFormer planned walker-buffer scalar byte cursor overflowed");
  const std::size_t scalar_record_offset = pf::checkedStorageSum(
      snapshot.scalar_offset, scalar_byte_cursor,
      "PsiFormer planned walker-buffer scalar record offset overflowed");
  record.gradient_data  = snapshot.data + snapshot.bulk_cursor;
  record.laplacian_data = snapshot.data + laplacian_cursor;
  record.scalar_data     = snapshot.data + scalar_record_offset;

  requireDisjointWalkerBufferRecord(particles, record);

  // Content is part of Phase-B destination identity even for WRITE, but no
  // WRITE preflight interprets or classifies these freely overwritable bytes.
  std::uint64_t content_hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(content_hash,
                       WALKER_BUFFER_CONTENT_FINGERPRINT_DOMAIN);
  mixPersistentInteger(content_hash, layout.fingerprint);
  for (std::size_t byte = 0; byte < layout.bulk_bytes; ++byte)
    mixPersistentByte(content_hash,
                      static_cast<unsigned char>(record.gradient_data[byte]));
  for (std::size_t byte = 0; byte < record.scalar_payload_bytes; ++byte)
    mixPersistentByte(content_hash,
                      static_cast<unsigned char>(record.scalar_data[byte]));
  record.content_fingerprint = content_hash == 0 ? 1 : content_hash;
  return record;
}

// Decode and classify one complete record without touching either PooledMemory cursor.
PsiFormerWF::WalkerBufferRecordView
PsiFormerWF::parsePlannedWalkerBufferRecord(
    const ParticleSet& particles,
    WalkerBufferCursorSnapshot snapshot) const
{
  WalkerBufferRecordView record =
      makePlannedWalkerBufferRecordView(particles, std::move(snapshot));
  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;

  const bool zero_bulk = walkerBufferBytesAreZero(
      record.gradient_data, layout.bulk_bytes);
  const bool zero_scalars = walkerBufferBytesAreZero(
      record.scalar_data, record.scalar_payload_bytes);
  if (zero_bulk && zero_scalars)
  {
    record.classification = WalkerBufferRecordClassification::VALID_ZERO;
    return record;
  }

  for (std::size_t electron = 0; electron < layout.electrons; ++electron)
  {
    const GradType gradient = loadWalkerBufferValue<GradType>(
        record.gradient_data + electron * sizeof(GradType));
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      if (!walkerBufferScalarIsFinite(gradient[dimension]))
        throw std::runtime_error(
            "PsiFormer planned walker buffer contains a non-finite gradient");
    const ValueType laplacian = loadWalkerBufferValue<ValueType>(
        record.laplacian_data + electron * sizeof(ValueType));
    if (!walkerBufferScalarIsFinite(laplacian))
      throw std::runtime_error(
          "PsiFormer planned walker buffer contains a non-finite Laplacian");
  }

  record.magic = decodePersistentInteger(record.scalar_data, 0, "magic");
  record.schema = decodePersistentInteger(record.scalar_data, 2, "schema");
  record.requirement =
      decodePersistentInteger(record.scalar_data, 4, "evaluation requirement");
  record.model_identity =
      decodePersistentInteger(record.scalar_data, 6, "model identity");
  record.parameter_version =
      decodePersistentInteger(record.scalar_data, 8, "parameter version");
  record.configuration_identity =
      decodePersistentInteger(record.scalar_data, 10,
                              "configuration identity");
  record.electron_count =
      decodePersistentInteger(record.scalar_data, 12, "electron count");
  record.sign = static_cast<double>(loadWalkerBufferValue<FullPrecRealType>(
      record.scalar_data + 14 * sizeof(FullPrecRealType)));
  record.log_magnitude =
      static_cast<double>(loadWalkerBufferValue<FullPrecRealType>(
          record.scalar_data + 15 * sizeof(FullPrecRealType)));
  record.phase = static_cast<double>(loadWalkerBufferValue<FullPrecRealType>(
      record.scalar_data + 16 * sizeof(FullPrecRealType)));

  if (record.magic == 0)
    throw std::runtime_error(
        "PsiFormer planned walker buffer has a partial zero sentinel");
  if (record.magic != WALKER_BUFFER_MAGIC)
    throw std::runtime_error(
        "PsiFormer planned walker buffer magic does not match");
  if (record.schema != pf::WALKER_BUFFER_LAYOUT_SCHEMA_VERSION)
    throw std::runtime_error(
        "PsiFormer planned walker buffer schema is unsupported");
  if (record.model_identity != model_state_->persistent_model_identity)
    throw std::runtime_error(
        "PsiFormer planned walker buffer belongs to a different physical model");
  if (record.electron_count != layout.electrons)
    throw std::runtime_error(
        "PsiFormer planned walker buffer electron count does not match");
  if (record.requirement >
      static_cast<std::uint64_t>(AcceptedStateRequirement::FULL_SPATIAL))
    throw std::runtime_error(
        "PsiFormer planned walker buffer has an unknown evaluation requirement");
  if (record.parameter_version >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::runtime_error(
        "PsiFormer planned walker buffer parameter version is out of range");

  const bool positive_zero_sign =
      record.sign == 0.0 && !isNegativeZero(record.sign);
  const bool positive_zero_phase =
      record.phase == 0.0 && !isNegativeZero(record.phase);
  const bool zero_amplitude = positive_zero_sign &&
      isNegativeInfinity(record.log_magnitude) && positive_zero_phase;
  const bool finite_nonzero_amplitude =
      isCoherentPhysicalValue(
          *model_state_, record.sign,
          LogValue(record.log_magnitude, record.phase), particles);
  if (!zero_amplitude && !finite_nonzero_amplitude)
    throw std::runtime_error(
        "PsiFormer planned walker buffer amplitude is invalid");

  record.classification = WalkerBufferRecordClassification::VALID_STALE;
  return record;
}

// Combine a structurally valid record with one locked model-version observation.
PsiFormerWF::WalkerBufferRecordView
PsiFormerWF::classifyPlannedWalkerBufferRecord(
    WalkerBufferRecordView record,
    std::size_t authoritative_parameter_version)
{
  if (record.classification == WalkerBufferRecordClassification::VALID_ZERO)
    return record;
  if (record.classification != WalkerBufferRecordClassification::VALID_STALE)
    throw std::logic_error(
        "PsiFormer planned walker-buffer classification received an invalid record");

  const bool positive_zero_sign =
      record.sign == 0.0 && !isNegativeZero(record.sign);
  const bool positive_zero_phase =
      record.phase == 0.0 && !isNegativeZero(record.phase);
  const bool zero_amplitude = positive_zero_sign &&
      isNegativeInfinity(record.log_magnitude) && positive_zero_phase;
  const bool current_full_state =
      record.requirement ==
          static_cast<std::uint64_t>(AcceptedStateRequirement::FULL_SPATIAL) &&
      record.parameter_version ==
          static_cast<std::uint64_t>(authoritative_parameter_version) &&
      record.configuration_identity == record.cursor.configuration_identity &&
      !zero_amplitude;
  if (current_full_state)
    record.classification = WalkerBufferRecordClassification::RESTORABLE;
  return record;
}

// Require a repeat cursor observation to preserve every Phase-A identity.
void PsiFormerWF::requireSameWalkerBufferCursorObservation(
    const WalkerBufferCursorSnapshot& left,
    const WalkerBufferCursorSnapshot& right)
{
  if (left.operation != right.operation || left.buffer != right.buffer ||
      left.plan_identity != right.plan_identity || left.data != right.data ||
      left.scalar_data != right.scalar_data || left.size != right.size ||
      left.capacity != right.capacity ||
      left.bulk_cursor != right.bulk_cursor ||
      left.scalar_cursor != right.scalar_cursor ||
      left.scalar_offset != right.scalar_offset ||
      left.scalar_capacity != right.scalar_capacity ||
      left.attached_storage != right.attached_storage ||
      left.plan_fingerprint != right.plan_fingerprint ||
      left.layout_fingerprint != right.layout_fingerprint ||
      left.storage_fingerprint != right.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned walker-buffer storage evidence changed after Phase A");
  if (left.configuration_identity != right.configuration_identity ||
      left.input_fingerprint != right.input_fingerprint)
    throw std::logic_error(
        "PsiFormer planned walker-buffer input evidence changed after Phase A");
}

// Require a repeat observation to preserve every Phase-A identity and record bit.
void PsiFormerWF::requireSameWalkerBufferObservation(
    const WalkerBufferRecordView& expected,
    const WalkerBufferRecordView& observed)
{
  requireSameWalkerBufferCursorObservation(expected.cursor, observed.cursor);
  if (expected.gradient_data != observed.gradient_data ||
      expected.laplacian_data != observed.laplacian_data ||
      expected.scalar_data != observed.scalar_data ||
      expected.next_bulk_cursor != observed.next_bulk_cursor ||
      expected.next_scalar_cursor != observed.next_scalar_cursor ||
      expected.content_fingerprint != observed.content_fingerprint)
    throw std::logic_error(
        "PsiFormer planned walker-buffer record content changed after Phase A");
}

// Reject refresh inputs that overlap and could mutate a source during publication.
void PsiFormerWF::requireDisjointPlannedWalkerBufferRefresh(
    const ParticleSet& particles) const
{
  const auto checked_bytes = [](std::size_t elements, std::size_t width,
                                const char* description) {
    return pf::checkedStorageProduct(elements, width, description);
  };
  const std::size_t particle_gradient_bytes = checked_bytes(
      particles.G.size(), sizeof(ParticleSet::ParticleGradient::value_type),
      "PsiFormer planned refresh ParticleSet gradient range overflowed");
  const std::size_t particle_laplacian_bytes = checked_bytes(
      particles.L.size(), sizeof(ParticleSet::ParticleLaplacian::value_type),
      "PsiFormer planned refresh ParticleSet Laplacian range overflowed");
  const std::size_t accepted_gradient_bytes = checked_bytes(
      accepted_gradient_.size(), sizeof(GradType),
      "PsiFormer planned refresh accepted-gradient range overflowed");
  const std::size_t accepted_laplacian_bytes = checked_bytes(
      accepted_laplacian_.size(), sizeof(ValueType),
      "PsiFormer planned refresh accepted-Laplacian range overflowed");
  const std::size_t proposed_gradient_bytes = checked_bytes(
      proposed_gradient_.size(), sizeof(GradType),
      "PsiFormer planned refresh proposed-gradient range overflowed");
  const std::size_t proposed_laplacian_bytes = checked_bytes(
      proposed_laplacian_.size(), sizeof(ValueType),
      "PsiFormer planned refresh proposed-Laplacian range overflowed");

  const auto reject_overlap = [](const void* left, std::size_t left_bytes,
                                 const void* right, std::size_t right_bytes,
                                 const char* description) {
    if (checkedByteRangesOverlap(left, left_bytes, right, right_bytes,
                                 description))
      throw std::invalid_argument(description);
  };
  reject_overlap(particles.G.data(), particle_gradient_bytes,
                 particles.L.data(), particle_laplacian_bytes,
                 "PsiFormer planned refresh ParticleSet G/L storage overlaps");
  for (const auto& source : {
           std::pair<const void*, std::size_t>{accepted_gradient_.data(),
                                               accepted_gradient_bytes},
           {accepted_laplacian_.data(), accepted_laplacian_bytes},
           {proposed_gradient_.data(), proposed_gradient_bytes},
           {proposed_laplacian_.data(), proposed_laplacian_bytes}})
  {
    reject_overlap(
        particles.G.data(), particle_gradient_bytes, source.first,
        source.second,
        "PsiFormer planned refresh ParticleSet gradient aliases component storage");
    reject_overlap(
        particles.L.data(), particle_laplacian_bytes, source.first,
        source.second,
        "PsiFormer planned refresh ParticleSet Laplacian aliases component storage");
  }
}

// Prove one accepted FULL_VGL cache and all prospective ParticleSet additions.
PsiFormerWF::WalkerBufferRefreshEvidence
PsiFormerWF::requirePlannedWalkerBufferRefreshInputs(
    const ParticleSet& particles,
    std::size_t parameter_version) const
{
  const std::size_t electron_count =
      model_state_->execution_plan.modelShape().electrons();
  if (!accepted_value_valid_ ||
      accepted_state_requirement_ != AcceptedStateRequirement::FULL_SPATIAL ||
      accepted_configuration_identity_ != configurationIdentity(particles) ||
      accepted_parameter_version_ != parameter_version ||
      observed_parameter_version_ != parameter_version ||
      accepted_gradient_.size() != electron_count ||
      accepted_laplacian_.size() != electron_count)
    throw std::logic_error(
        "PsiFormer planned walker-buffer refresh requires a current FULL_VGL accepted state");

  if (!isCoherentPhysicalValue(*model_state_, current_sign_, log_value_, particles) ||
      (!model_state_->twist_phase.active() && current_sign_ > 0.0 &&
       isNegativeZero(std::imag(log_value_))))
    throw std::logic_error(
        "PsiFormer planned walker-buffer refresh has an invalid accepted amplitude");

  requireDisjointPlannedWalkerBufferRefresh(particles);

  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  mixPersistentInteger(hash, WALKER_BUFFER_REFRESH_FINGERPRINT_DOMAIN);
  mixPersistentInteger(hash, static_cast<std::uint64_t>(parameter_version));
  mixPersistentInteger(hash, accepted_configuration_identity_);
  mixPersistentInteger(
      hash, static_cast<std::uint64_t>(accepted_state_requirement_));
  mixPersistentInteger(hash, static_cast<std::uint64_t>(accepted_value_valid_));
  mixPersistentInteger(
      hash, static_cast<std::uint64_t>(restore_validation_pending_));
  mixPersistentObject(hash, current_sign_);
  mixPersistentObject(hash, log_value_);

  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    {
      const ValueType accepted = accepted_gradient_[electron][dimension];
      const ValueType particle = particles.G[electron][dimension];
      const ValueType prospective = particle + accepted;
      if (!walkerBufferScalarIsFinite(accepted) ||
          !walkerBufferScalarIsFinite(particle) ||
          !walkerBufferScalarIsFinite(prospective))
        throw std::logic_error(
            "PsiFormer planned walker-buffer refresh has a non-finite gradient input or sum");
      mixPersistentObject(hash, accepted);
      mixPersistentObject(hash, particle);
      mixPersistentObject(hash, prospective);
    }

    const ValueType accepted = accepted_laplacian_[electron];
    const ValueType particle = particles.L[electron];
    const ValueType prospective = particle + accepted;
    if (!walkerBufferScalarIsFinite(accepted) ||
        !walkerBufferScalarIsFinite(particle) ||
        !walkerBufferScalarIsFinite(prospective))
      throw std::logic_error(
          "PsiFormer planned walker-buffer refresh has a non-finite Laplacian input or sum");
    mixPersistentObject(hash, accepted);
    mixPersistentObject(hash, particle);
    mixPersistentObject(hash, prospective);
  }

  return {hash == 0 ? 1 : hash, parameter_version, log_value_};
}

// Inspect one typed preflight without parsing or publishing state.
PsiFormerWF::WalkerBufferCursorSnapshot
PsiFormerWF::inspectPlannedWalkerBufferPreflightForTesting(
    PlannedWalkerBufferOperation operation,
    const ParticleSet& particles,
    const WFBufferType& buffer) const
{
  return requirePlannedWalkerBufferOperation(operation, particles, buffer);
}

// Inspect one record under an authoritative model read without publishing state.
PsiFormerWF::WalkerBufferRecordView
PsiFormerWF::inspectPlannedWalkerBufferForTesting(
    const ParticleSet& particles,
    const WFBufferType& buffer) const
{
  const WalkerBufferCursorSnapshot phase_a_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  const WalkerBufferRecordView phase_a =
      parsePlannedWalkerBufferRecord(particles, phase_a_snapshot);

  PsiFormerReadTransaction transaction(*model_state_);
  const WalkerBufferCursorSnapshot phase_b_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  WalkerBufferRecordView phase_b =
      parsePlannedWalkerBufferRecord(particles, phase_b_snapshot);
  requireSameWalkerBufferObservation(phase_a, phase_b);
  return classifyPlannedWalkerBufferRecord(
      std::move(phase_b), transaction.parameterVersion());
}

// Mutate one test fixture only between the real Phase-A and Phase-B captures.
void PsiFormerWF::probePlannedWalkerBufferBetweenPhaseFaultForTesting(
    ParticleSet& particles,
    WFBufferType& buffer,
    PlannedWalkerBufferBetweenPhaseFaultForTesting fault)
{
  const WalkerBufferCursorSnapshot phase_a_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  const WalkerBufferRecordView phase_a =
      parsePlannedWalkerBufferRecord(particles, phase_a_snapshot);

  static_assert(std::is_nothrow_copy_constructible_v<
                BatchExecutionParticipantPlan>);
  static_assert(std::is_nothrow_copy_assignable_v<
                BatchExecutionParticipantPlan>);
  const BatchExecutionParticipantPlan saved_plan = batch_execution_plan_;
  const pf::WalkerBufferLayout saved_layout = prepared_walker_buffer_layout_;
  const GradType* const saved_prepared_gradient =
      prepared_accepted_gradient_data_;
  FullPrecRealType* const saved_scalar_pointer = buffer.Scalar_ptr;
  const std::size_t saved_bulk_cursor = buffer.Current;
  std::uint64_t saved_content_bits = 0;
  char* content_address = nullptr;
  ParticleSet::RealType saved_aos_coordinate{};
  ParticleSet::RealType saved_soa_coordinate{};
  ParticleSet::RealType* soa_coordinate_address = nullptr;
  bool fault_applied = false;

  const auto restore_fault = [&]() noexcept {
    if (!fault_applied)
      return;
    switch (fault)
    {
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::NONE:
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::STORAGE_POINTER:
      buffer.Scalar_ptr = saved_scalar_pointer;
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::BULK_CURSOR:
      buffer.Current = saved_bulk_cursor;
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::RECORD_CONTENT:
      std::memcpy(content_address, &saved_content_bits,
                  sizeof(saved_content_bits));
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::PARTICLE_INPUT:
      particles.R[0][0]       = saved_aos_coordinate;
      *soa_coordinate_address = saved_soa_coordinate;
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::PLAN_BINDING:
      batch_execution_plan_ = saved_plan;
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::LAYOUT_EVIDENCE:
      prepared_walker_buffer_layout_ = saved_layout;
      break;
    case PlannedWalkerBufferBetweenPhaseFaultForTesting::PREPARED_STORAGE_EVIDENCE:
      prepared_accepted_gradient_data_ = saved_prepared_gradient;
      break;
    }
  };
  [[maybe_unused]] const auto restore_guard =
      makePsiFormerScopeExit(restore_fault);

  switch (fault)
  {
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::NONE:
    break;
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::STORAGE_POINTER:
    buffer.Scalar_ptr = nullptr;
    fault_applied = true;
    break;
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::BULK_CURSOR:
    buffer.Current = std::numeric_limits<std::size_t>::max();
    fault_applied = true;
    break;
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::RECORD_CONTENT:
  {
    static_assert(sizeof(saved_content_bits) == sizeof(FullPrecRealType));
    content_address = const_cast<char*>(
        phase_a.scalar_data + 15 * sizeof(FullPrecRealType));
    std::memcpy(&saved_content_bits, content_address,
                sizeof(saved_content_bits));
    const std::uint64_t changed_content_bits = saved_content_bits ^ UINT64_C(1);
    std::memcpy(content_address, &changed_content_bits,
                sizeof(changed_content_bits));
    fault_applied = true;
    break;
  }
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::PARTICLE_INPUT:
  {
    if (particles.getTotalNum() == 0)
      throw std::logic_error(
          "PsiFormer planned walker-buffer input fault requires one particle");
    auto& soa_positions =
        particles.getCoordinates().getAllParticlePos();
    saved_aos_coordinate = particles.R[0][0];
    saved_soa_coordinate = soa_positions.data()[0];
    const ParticleSet::RealType changed_coordinate =
        saved_aos_coordinate + ParticleSet::RealType(0.125);
    if (!psiformer::determinant::isFiniteReal(
            static_cast<double>(changed_coordinate)) ||
        std::memcmp(std::addressof(changed_coordinate),
                    std::addressof(saved_aos_coordinate),
                    sizeof(changed_coordinate)) == 0)
      throw std::logic_error(
          "PsiFormer planned walker-buffer input fault cannot change the coordinate");
    soa_coordinate_address =
        const_cast<ParticleSet::RealType*>(soa_positions.data());
    particles.R[0][0]       = changed_coordinate;
    *soa_coordinate_address = changed_coordinate;
    fault_applied = true;
    break;
  }
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::PLAN_BINDING:
    batch_execution_plan_ = {};
    fault_applied = true;
    break;
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::LAYOUT_EVIDENCE:
    prepared_walker_buffer_layout_.fingerprint ^= UINT64_C(1);
    fault_applied = true;
    break;
  case PlannedWalkerBufferBetweenPhaseFaultForTesting::PREPARED_STORAGE_EVIDENCE:
    if (!saved_prepared_gradient)
      throw std::logic_error(
          "PsiFormer planned walker-buffer prepared-storage fault requires storage");
    prepared_accepted_gradient_data_ = nullptr;
    fault_applied = true;
    break;
  }

  PsiFormerReadTransaction transaction(*model_state_);
  const WalkerBufferCursorSnapshot phase_b_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  WalkerBufferRecordView phase_b =
      parsePlannedWalkerBufferRecord(particles, phase_b_snapshot);
  requireSameWalkerBufferObservation(phase_a, phase_b);
  static_cast<void>(classifyPlannedWalkerBufferRecord(
      std::move(phase_b), transaction.parameterVersion()));
  if (fault == PlannedWalkerBufferBetweenPhaseFaultForTesting::NONE)
    return;
  throw std::logic_error(
      "PsiFormer planned walker-buffer between-phase fault was not rejected");
}

// Select one deterministic failure immediately before a buffer publication.
void PsiFormerWF::setPlannedWalkerBufferLateFaultForTesting(
    PlannedWalkerBufferLateFaultForTesting fault) noexcept
{
  planned_walker_buffer_late_fault_for_testing_ = fault;
}

// Exercise structural and alias failures without changing the live PooledMemory object.
void PsiFormerWF::probePlannedWalkerBufferFaultForTesting(
    const ParticleSet& particles,
    const WFBufferType& buffer,
    PlannedWalkerBufferFaultForTesting fault) const
{
  WalkerBufferCursorSnapshot snapshot = requirePlannedWalkerBufferOperation(
      PlannedWalkerBufferOperation::READ, particles, buffer);
  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;

  switch (fault)
  {
  case PlannedWalkerBufferFaultForTesting::NULL_BACKING:
    snapshot.data = nullptr;
    break;
  case PlannedWalkerBufferFaultForTesting::NULL_SCALAR:
    snapshot.scalar_data = nullptr;
    break;
  case PlannedWalkerBufferFaultForTesting::SIZE_EXCEEDS_CAPACITY:
    snapshot.capacity = snapshot.size - 1;
    break;
  case PlannedWalkerBufferFaultForTesting::ATTACHED_STORAGE:
    snapshot.attached_storage = true;
    break;
  case PlannedWalkerBufferFaultForTesting::MISALIGNED_BACKING:
    snapshot.data = reinterpret_cast<const char*>(
        reinterpret_cast<std::uintptr_t>(snapshot.data) + 1);
    break;
  case PlannedWalkerBufferFaultForTesting::SCALAR_BEFORE_BACKING:
  {
    const std::uintptr_t backing_address =
        reinterpret_cast<std::uintptr_t>(snapshot.data);
    if (backing_address < alignof(FullPrecRealType))
      throw std::logic_error(
          "PsiFormer planned walker-buffer test cannot form a preceding address");
    snapshot.scalar_data = reinterpret_cast<const FullPrecRealType*>(
        backing_address - alignof(FullPrecRealType));
    break;
  }
  case PlannedWalkerBufferFaultForTesting::SCALAR_AFTER_BACKING:
  {
    const std::uintptr_t backing_end = checkedAddressEnd(
        snapshot.data, snapshot.size,
        "PsiFormer planned walker-buffer test backing address overflowed");
    if (backing_end > std::numeric_limits<std::uintptr_t>::max() -
            alignof(FullPrecRealType))
      throw std::logic_error(
          "PsiFormer planned walker-buffer test cannot form a following address");
    snapshot.scalar_data = reinterpret_cast<const FullPrecRealType*>(
        backing_end + alignof(FullPrecRealType));
    break;
  }
  case PlannedWalkerBufferFaultForTesting::MISALIGNED_SCALAR:
    snapshot.scalar_data = reinterpret_cast<const FullPrecRealType*>(
        reinterpret_cast<std::uintptr_t>(snapshot.scalar_data) + 1);
    break;
  case PlannedWalkerBufferFaultForTesting::MISALIGNED_BULK_CURSOR:
    ++snapshot.bulk_cursor;
    break;
  case PlannedWalkerBufferFaultForTesting::BULK_CURSOR_BEYOND_DOMAIN:
    snapshot.bulk_cursor = pf::checkedStorageSum(
        snapshot.scalar_offset, layout.alignment,
        "PsiFormer planned walker-buffer test bulk cursor overflowed");
    break;
  case PlannedWalkerBufferFaultForTesting::SCALAR_CURSOR_BEYOND_DOMAIN:
    snapshot.scalar_cursor = pf::checkedStorageSum(
        snapshot.scalar_capacity, std::size_t{1},
        "PsiFormer planned walker-buffer test scalar cursor overflowed");
    break;
  case PlannedWalkerBufferFaultForTesting::BULK_CURSOR_OVERFLOW:
    snapshot.bulk_cursor = std::numeric_limits<std::size_t>::max() -
        layout.bulk_bytes + 1;
    break;
  case PlannedWalkerBufferFaultForTesting::SCALAR_CURSOR_OVERFLOW:
    snapshot.scalar_cursor = std::numeric_limits<std::size_t>::max() -
        layout.scalar_count + 1;
    break;
  case PlannedWalkerBufferFaultForTesting::TRUNCATED_BULK:
  {
    if (layout.bulk_bytes < layout.alignment)
      throw std::logic_error(
          "PsiFormer planned walker-buffer test cannot truncate an empty bulk record");
    const std::size_t truncated_offset = pf::checkedStorageSum(
        snapshot.bulk_cursor, layout.bulk_bytes - layout.alignment,
        "PsiFormer planned walker-buffer truncated test offset overflowed");
    snapshot.scalar_data = reinterpret_cast<const FullPrecRealType*>(
        snapshot.data + truncated_offset);
    break;
  }
  case PlannedWalkerBufferFaultForTesting::TRUNCATED_SCALAR:
  {
    const std::size_t truncated_elements = pf::checkedStorageSum(
        snapshot.scalar_cursor, layout.scalar_count - 1,
        "PsiFormer planned walker-buffer truncated test scalar overflowed");
    snapshot.size = pf::checkedStorageSum(
        snapshot.scalar_offset,
        pf::checkedStorageProduct(
            truncated_elements, sizeof(FullPrecRealType),
            "PsiFormer planned walker-buffer truncated test bytes overflowed"),
        "PsiFormer planned walker-buffer truncated test size overflowed");
    break;
  }
  case PlannedWalkerBufferFaultForTesting::ACCEPTED_STORAGE_ALIAS:
  case PlannedWalkerBufferFaultForTesting::PROPOSED_STORAGE_ALIAS:
  case PlannedWalkerBufferFaultForTesting::PARTICLE_STORAGE_ALIAS:
  {
    WalkerBufferRecordView record =
        parsePlannedWalkerBufferRecord(particles, snapshot);
    if (fault == PlannedWalkerBufferFaultForTesting::ACCEPTED_STORAGE_ALIAS)
      record.gradient_data = reinterpret_cast<const char*>(
          accepted_gradient_.data());
    else if (fault ==
             PlannedWalkerBufferFaultForTesting::PROPOSED_STORAGE_ALIAS)
      record.laplacian_data = reinterpret_cast<const char*>(
          proposed_laplacian_.data());
    else
      record.gradient_data =
          reinterpret_cast<const char*>(particles.R.data());
    requireDisjointWalkerBufferRecord(particles, record);
    return;
  }
  }

  static_cast<void>(validateWalkerBufferCursorSnapshot(snapshot));
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
  putPersistentInteger(buffer, pf::WALKER_BUFFER_LAYOUT_SCHEMA_VERSION);
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
  if (schema != pf::WALKER_BUFFER_LAYOUT_SCHEMA_VERSION)
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
  const bool zero_amplitude = sign == 0.0 && isNegativeInfinity(logabs) &&
      phase == 0.0;
  const bool finite_nonzero_amplitude = isCoherentPhysicalValue(
      *model_state_, sign, LogValue(logabs, phase), particles);
  if (!psiformer::determinant::isFiniteReal(sign) ||
      (!psiformer::determinant::isFiniteReal(logabs) && !(sign == 0.0 && isNegativeInfinity(logabs))) ||
      !psiformer::determinant::isFiniteReal(phase) ||
      (!zero_amplitude && !finite_nonzero_amplitude))
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
  requireNoPlannedProposalMutation("optimizer reset");

  auto state_lock = model_state_->acquireHostWrite();
  requireNoPlannedProposalMutation("optimizer reset");
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
      model_state_->setParameterValues(active_values);
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
  {
    if (model_state_->numerical_tile_backend)
    {
      // A selected legacy update publishes one complete canonical device image.
      auto complete_values = parameters.flat_values();
      for (std::size_t parameter = 0; parameter < active_values.size(); ++parameter)
        complete_values[active_flat_indices[parameter]] = active_values[parameter];
      model_state_->setParameterValues(complete_values);
    }
    else
    {
      parameters.set_flat_values(active_flat_indices, active_values);
      model_state_->host_mirror_version = parameters.version();
    }
  }
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

  PsiFormerReadTransaction state_lock(*model_state_);
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
                       model_state_->initialization_seed,
                       model_state_->execution_plan.environment(),
                       model_state_->twist_phase.reduced,
                       model_state_->twist_phase.cartesian,
                       model_state_->physical_source, model_state_->physical_species);

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
  if (model_state_->twist_phase.active())
  {
    output.write(std::vector<double>(model_state_->twist_phase.reduced.begin(),
                                     model_state_->twist_phase.reduced.end()),
                 "reduced_twist");
    output.write(std::vector<double>(model_state_->twist_phase.cartesian.begin(),
                                     model_state_->twist_phase.cartesian.end()),
                 "cartesian_twist");
  }
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
  requireNoPlannedProposalMutation("variational-parameter restore");
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
  std::vector<double> reduced_twist;
  std::vector<double> cartesian_twist;
  if (model_state_->twist_phase.active())
  {
    reduced_twist   = readVector<double>(input, "reduced_twist");
    cartesian_twist = readVector<double>(input, "cartesian_twist");
  }
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

  auto state_lock = model_state_->acquireHostWrite();
  requireNoPlannedProposalMutation("variational-parameter restore");
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
                         model_state_->initialization_seed,
                         model_state_->execution_plan.environment(),
                         model_state_->twist_phase.reduced,
                         model_state_->twist_phase.cartesian,
                         model_state_->physical_source, model_state_->physical_species))
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
    if (model_state_->twist_phase.active())
    {
      requireEqual(reduced_twist,
                   std::vector<double>(model_state_->twist_phase.reduced.begin(),
                                       model_state_->twist_phase.reduced.end()),
                   "reduced twist");
      requireEqual(cartesian_twist,
                   std::vector<double>(model_state_->twist_phase.cartesian.begin(),
                                       model_state_->twist_phase.cartesian.end()),
                   "Cartesian twist");
    }

    if (flat_values != model.p.flat_values())
      model_state_->setParameterValues(flat_values);
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
  PsiFormerReadTransaction state_lock(*model_state_, true);
  return model_state_->model.p.version();
}

// Write a standalone flat parameter file without exposing mutable native storage.
namespace
{
void writeStandaloneModel(const std::string& path, const pf::PsiFormer& model,
                          const psiformer::ExecutionEnvironment& environment,
                          const std::string& profile, std::uint64_t seed,
                          const std::string& system, const std::array<double, 3>& twist,
                          const std::vector<double>& physical_source = {})
{
  const bool versioned = static_cast<bool>(environment.periodic_profile);
  const bool fourier = versioned &&
      environment.periodic_profile->envelope_policy == psiformer::PeriodicEnvelopePolicy::FOURIER;
  if (versioned && (profile != psiformer::PSIFORMER_PERIODIC_PAIR_V1 || physical_source.empty() ||
                   physical_source.size() % 5 != 0))
    throw std::invalid_argument("Periodic profile export requires complete physical source metadata");
  if (fourier)
    psiformer::validateFourierOrigin(environment.periodic_profile->fourier_origin,
        pf::GeometryPositionView::interleaved(physical_source.data(), physical_source.size() / 5, 5));
  model.p.write(path);
  hdf_archive output;
  if (!output.open(path, H5F_ACC_RDWR))
    throw std::runtime_error("Unable to append PsiFormer model metadata");
  output.push("psiformer_model");
  output.write(fourier ? std::vector<int>{1, 2, 0} :
      versioned ? std::vector<int>{1, 1, 0} : std::vector<int>{1, 0, 0}, "format_version");
  output.write(std::string("fp64_reference"), "precision");
  output.write(std::vector<std::uint64_t>{model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0],
      model.ndet, model.dim, model.heads, model.blocks}, "shape");
  output.write(profile, "profile");
  output.write(seed, "seed");
  output.write(system, "system");
  output.write(static_cast<int>(environment.head_representation == psiformer::HeadRepresentation::PAIRED_REAL_COMPLEX), "paired_complex");
  output.write(static_cast<int>(environment.boundary == psiformer::BoundaryCondition::PERIODIC), "periodic");
  output.write(std::string(versioned ? "periodic_two_cell_v1" :
      environment.boundary == psiformer::BoundaryCondition::PERIODIC ?
      "periodic_torus_v1" : "open_euclidean_v1"), "feature_policy");
  std::vector<double> lattice;
  for (const auto& row : environment.lattice_vectors)
    lattice.insert(lattice.end(), row.begin(), row.end());
  output.write(lattice, "lattice");
  output.write(std::vector<double>(twist.begin(), twist.end()), "twist");
  output.write(model.cfg.nuclei.x, "nuclear_positions");
  output.write(model.cfg.charges.x, "nuclear_charges");
  if (versioned)
  {
    const auto& metadata = *environment.periodic_profile;
    output.write(physical_source, "physical_source");
    output.push("periodic_profile");
    output.write(metadata.version, "version");
    output.write(static_cast<int>(metadata.feature_transform), "feature_transform");
    output.write(static_cast<std::uint64_t>(metadata.pair_feature_dimension), "pair_feature_dimension");
    output.write(static_cast<int>(metadata.electron_electron_cusp), "electron_electron_cusp");
    std::vector<double> primitive, motif, occupations;
    std::vector<int> tiling, momenta, spins;
    std::vector<std::uint64_t> bands;
    for (const auto& row : metadata.cells.primitive.lattice_vectors)
      primitive.insert(primitive.end(), row.begin(), row.end());
    for (const auto& row : metadata.cells.tiling)
      tiling.insert(tiling.end(), row.begin(), row.end());
    for (const auto& atom : metadata.primitive_motif)
      motif.insert(motif.end(), atom.begin(), atom.end());
    for (const auto& column : metadata.occupied_columns)
    {
      momenta.insert(momenta.end(), column.reciprocal.begin(), column.reciprocal.end());
      bands.push_back(column.band);
      spins.push_back(column.spin);
      occupations.push_back(column.occupation);
    }
    output.write(primitive, "primitive_lattice");
    output.write(tiling, "tiling");
    output.write(motif, "primitive_motif");
    output.write(metadata.motif_species, "motif_species");
    output.write(momenta, "occupied_reciprocal");
    output.write(bands, "occupied_bands");
    output.write(spins, "occupied_spins");
    output.write(occupations, "occupied_occupations");
    if (fourier)
    {
      std::vector<int> modes;
      for (const auto& mode : metadata.fourier_modes)
        modes.insert(modes.end(), mode.begin(), mode.end());
      output.write(static_cast<int>(metadata.envelope_policy), "envelope_policy");
      output.write(modes, "fourier_modes");
      output.write(std::vector<double>(metadata.fourier_origin.begin(), metadata.fourier_origin.end()), "fourier_origin");
    }
    output.pop();
  }
  output.write(model.p.layout_fingerprint(), "layout_fingerprint");
  output.write(wftrain::TrainingCheckpoint::valuesFingerprint(model.p.flat_values()), "values_fingerprint");
  output.write(static_cast<std::uint64_t>(model.p.version()), "parameter_version");
  output.write(std::string("postdecision"), "parameter_timing");
  output.write(std::string("psiformer-model-complete-v1"), "completion_cookie");
  output.pop();
  output.flush();
}
}

psiformer::PsiFormerModelFile psiformer::readPsiFormerModelFile(const std::string& path)
{
  hdf_archive input;
  if (!input.open(path, H5F_ACC_RDONLY) || !input.is_group("psiformer_model"))
    throw std::runtime_error("PsiFormer parameters-only import requires standalone model metadata");
  input.push("psiformer_model", false);
  std::vector<int> version;
  std::vector<std::uint64_t> shape;
  std::vector<double> twist;
  std::string precision, cookie, layout, feature, values_fingerprint;
  std::uint64_t parameter_version;
  int paired = -1, periodic = -1;
  PsiFormerModelFile result;
  input.read(version, "format_version");
  input.read(shape, "shape");
  input.read(precision, "precision");
  input.read(cookie, "completion_cookie");
  input.read(layout, "layout_fingerprint");
  input.read(values_fingerprint, "values_fingerprint");
  input.read(parameter_version, "parameter_version");
  input.read(feature, "feature_policy");
  input.read(paired, "paired_complex");
  input.read(periodic, "periodic");
  input.read(result.parameters.profile, "profile");
  input.read(result.parameters.seed, "seed");
  input.read(result.system, "system");
  input.read(twist, "twist");
  const bool versioned = result.parameters.profile == PSIFORMER_PERIODIC_PAIR_V1;
  if (versioned)
  {
    if ((version != std::vector<int>{1, 1, 0} && version != std::vector<int>{1, 2, 0}) || shape.size() != 7 ||
        std::any_of(shape.begin(), shape.end(), [](auto n) {
          return n == 0 || n > std::numeric_limits<std::size_t>::max(); }) ||
        precision != "fp64_reference" || cookie != "psiformer-model-complete-v1" ||
        (paired != 0 && paired != 1) || periodic != 1 || twist.size() != 3 ||
        feature != "periodic_two_cell_v1" ||
        (result.system != "all_electron" && result.system != "pseudopotential"))
      throw std::runtime_error("Incompatible or incomplete periodic profile model metadata");
  }
  else
  if (version != std::vector<int>{1, 0, 0} || shape.size() != 7 ||
      std::any_of(shape.begin(), shape.end(), [](auto n) { return n == 0 || n > std::numeric_limits<std::size_t>::max(); }) ||
      precision != "fp64_reference" || cookie != "psiformer-model-complete-v1" ||
      (paired != 0 && paired != 1) || (periodic != 0 && periodic != 1) || twist.size() != 3 ||
      feature != (periodic ? "periodic_torus_v1" : "open_euclidean_v1") ||
      (paired && (!periodic || result.parameters.profile != PSIFORMER_COMPLEX_PERIODIC_V1)) ||
      (!paired && result.parameters.profile != DEEPQMC_PSIFORMER_V1 && result.parameters.profile != "external_hdf5") ||
      (result.system != "all_electron" && result.system != "pseudopotential"))
    throw std::runtime_error("Incompatible or incomplete standalone PsiFormer model metadata");
  result.paired_complex = paired;
  result.periodic = periodic;
  std::copy(twist.begin(), twist.end(), result.twist.begin());
  result.parameters.model_shape = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]};
  if (versioned)
  {
    auto metadata = std::make_shared<PeriodicProfileMetadata>();
    auto& environment = result.parameters.environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    environment.periodic_axes = {true, true, true};
    environment.head_representation = paired ? HeadRepresentation::PAIRED_REAL_COMPLEX : HeadRepresentation::REAL;
    environment.amplitude_scalar_domain = paired ? ScalarDomain::COMPLEX : ScalarDomain::REAL;
    std::vector<double> lattice, primitive, motif, occupations;
    std::vector<int> tiling, momenta, spins;
    std::vector<std::uint64_t> bands;
    std::uint64_t pair_dimension = 0;
    int transform = -1, cusp = -1;
    if (!input.readEntry(lattice, "lattice") || lattice.size() != 9 || !input.is_group("periodic_profile"))
      throw std::runtime_error("Periodic profile model is missing cell metadata");
    input.push("periodic_profile", false);
    const bool complete = input.readEntry(metadata->version, "version") &&
        input.readEntry(transform, "feature_transform") && input.readEntry(pair_dimension, "pair_feature_dimension") &&
        input.readEntry(cusp, "electron_electron_cusp") && input.readEntry(primitive, "primitive_lattice") &&
        input.readEntry(tiling, "tiling") && input.readEntry(motif, "primitive_motif") &&
        input.readEntry(metadata->motif_species, "motif_species") && input.readEntry(momenta, "occupied_reciprocal") &&
        input.readEntry(bands, "occupied_bands") && input.readEntry(spins, "occupied_spins") &&
        input.readEntry(occupations, "occupied_occupations");
    if (version == std::vector<int>{1, 2, 0})
    {
      int policy = -1;
      std::vector<int> modes;
      std::vector<double> origin;
      if (!input.readEntry(policy, "envelope_policy") || policy != static_cast<int>(PeriodicEnvelopePolicy::FOURIER) ||
          !input.readEntry(modes, "fourier_modes") || modes.empty() || modes.size() % 3 ||
          modes.size() / 3 > FOURIER_MODE_LIMIT || !input.readEntry(origin, "fourier_origin") || origin.size() != 3)
        throw std::runtime_error("Incomplete Fourier replacement-envelope model metadata");
      metadata->envelope_policy = PeriodicEnvelopePolicy::FOURIER;
      std::copy(origin.begin(), origin.end(), metadata->fourier_origin.begin());
      for (std::size_t mode = 0; mode < modes.size() / 3; ++mode)
        metadata->fourier_modes.push_back({modes[3 * mode], modes[3 * mode + 1], modes[3 * mode + 2]});
    }
    input.pop();
    if (!complete || primitive.size() != 9 || tiling.size() != 9 ||
        motif.size() != 3 * shape[2] || bands.size() != shape[0] + shape[1] ||
        momenta.size() != 3 * bands.size() || spins.size() != bands.size() || occupations.size() != bands.size() ||
        (cusp != 0 && cusp != 1) ||
        (transform != static_cast<int>(PeriodicFeatureTransform::LOG_COMPLEMENT_V1) &&
         transform != static_cast<int>(PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1) &&
         transform != static_cast<int>(PeriodicFeatureTransform::POLYNOMIAL_V1)) ||
        pair_dimension != periodicPairFeatureArity(static_cast<PeriodicFeatureTransform>(transform)))
      throw std::runtime_error("Incomplete or invalid periodic profile feature and occupied metadata");
    metadata->feature_transform = static_cast<PeriodicFeatureTransform>(transform);
    metadata->pair_feature_dimension = pair_dimension;
    metadata->electron_electron_cusp = cusp;
    metadata->cells.primitive.kind = metadata->cells.simulation.kind = pf::GeometryBoundaryKind::PERIODIC;
    metadata->cells.primitive.periodic_axes = metadata->cells.simulation.periodic_axes = {true, true, true};
    for (std::size_t a = 0; a < 3; ++a)
      for (std::size_t d = 0; d < 3; ++d)
      {
        environment.lattice_vectors[a][d] = lattice[3 * a + d];
        metadata->cells.simulation.lattice_vectors[a][d] = lattice[3 * a + d];
        metadata->cells.primitive.lattice_vectors[a][d] = primitive[3 * a + d];
        metadata->cells.tiling[a][d] = tiling[3 * a + d];
      }
    for (std::size_t atom = 0; atom < shape[2]; ++atom)
      metadata->primitive_motif.push_back({motif[3 * atom], motif[3 * atom + 1], motif[3 * atom + 2]});
    for (std::size_t column = 0; column < bands.size(); ++column)
    {
      if (bands[column] > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Periodic profile occupied band index overflows native size");
      metadata->occupied_columns.push_back({{momenta[3 * column], momenta[3 * column + 1],
          momenta[3 * column + 2]}, static_cast<std::size_t>(bands[column]), spins[column], occupations[column]});
    }
    environment.periodic_profile = std::move(metadata);
    validatePeriodicProfileMetadata(result.parameters.model_shape, environment);
    if (environment.periodic_profile->envelope_policy == PeriodicEnvelopePolicy::FOURIER)
    {
      std::vector<double> physical;
      if (!input.readEntry(physical, "physical_source") || physical.empty() || physical.size() % 5)
        throw std::runtime_error("Fourier model lacks its complete physical source snapshot");
      validateFourierOrigin(environment.periodic_profile->fourier_origin,
          pf::GeometryPositionView::interleaved(physical.data(), physical.size() / 5, 5));
    }
  }
  pf::Parameters parameters(path);
  if (parameters.layout_fingerprint() != layout ||
      wftrain::TrainingCheckpoint::valuesFingerprint(parameters.flat_values()) != values_fingerprint)
    throw std::runtime_error("Standalone PsiFormer model layout fingerprint mismatch");
  result.parameters.values = parameters.flat_values();
  for (const auto& tensor : parameters.layouts)
    result.parameters.layouts.push_back({tensor.module, tensor.name, tensor.shape, tensor.begin, tensor.end});
  return result;
}

void psiformer::validatePsiFormerModelFile(const PsiFormerModelFile& model, const std::string& path,
    const ParticleSet& electrons, const ParticleSet& ions, const std::string& system,
    const std::array<double, 3>& twist)
{
  const auto& shape = model.parameters.model_shape;
  const bool versioned = static_cast<bool>(model.parameters.environment.periodic_profile);
  if (system != model.system || electrons.groups() != 2 ||
      electrons.groupsize(0) != shape.spin_up_electrons || electrons.groupsize(1) != shape.spin_down_electrons ||
      (!versioned && ions.getTotalNum() != shape.nuclei) ||
      model.periodic != (electrons.getLattice().getSuperCellEnum() == SUPERCELL_BULK) ||
      model.twist != twist)
    throw std::invalid_argument("Standalone PsiFormer model disagrees with XML system/spins/boundary/twist");
  hdf_archive input;
  if (!input.open(path, H5F_ACC_RDONLY))
    throw std::runtime_error("Unable to reopen standalone PsiFormer metadata");
  input.push("psiformer_model", false);
  std::vector<double> positions, charges, lattice;
  input.read(positions, "nuclear_positions");
  input.read(charges, "nuclear_charges");
  input.read(lattice, "lattice");
  if (positions.size() != 3 * shape.nuclei || charges.size() != shape.nuclei || lattice.size() != 9)
    throw std::invalid_argument("Standalone PsiFormer physical metadata is incomplete");
  const auto& species = ions.getSpeciesSet();
  const int charge = species.findAttribute("charge");
  if (charge < 0)
    throw std::invalid_argument("Standalone PsiFormer XML ions require charges");
  if (versioned)
  {
    const auto& metadata = *model.parameters.environment.periodic_profile;
    std::vector<double> physical;
    if (!input.readEntry(physical, "physical_source") || physical.empty() || physical.size() % 5 != 0 ||
        physical.size() / 5 != static_cast<std::size_t>(ions.getTotalNum()))
      throw std::invalid_argument("Periodic profile physical source snapshot is incomplete");
    for (std::size_t atom = 0; atom < physical.size() / 5; ++atom)
    {
      if (physical[5 * atom + 3] != species(charge, ions.GroupID[atom]) ||
          physical[5 * atom + 4] != ions.GroupID[atom])
        throw std::invalid_argument("Periodic profile physical source charge or species differs from XML");
      for (std::size_t d = 0; d < 3; ++d)
        if (physical[5 * atom + d] != ions.R[atom][d])
          throw std::invalid_argument("Periodic profile physical source positions differ from XML");
    }
    if (metadata.envelope_policy == PeriodicEnvelopePolicy::FOURIER)
      validateFourierOrigin(metadata.fourier_origin,
          pf::GeometryPositionView::interleaved(physical.data(), physical.size() / 5, 5));
    std::vector<double> motif_positions;
    for (const auto& atom : metadata.primitive_motif)
      motif_positions.insert(motif_positions.end(), atom.begin(), atom.end());
    std::vector<int> runtime_species;
    for (std::size_t atom = 0; atom < physical.size() / 5; ++atom)
      runtime_species.push_back(ions.GroupID[atom]);
    const auto mapping = pf::mapPeriodicIonMotif(metadata.cells,
        pf::GeometryPositionView::interleaved(motif_positions.data(), shape.nuclei), metadata.motif_species,
        pf::GeometryPositionView::interleaved(physical.data(), physical.size() / 5, 5), runtime_species);
    for (std::size_t atom = 0; atom < shape.nuclei; ++atom)
      for (std::size_t d = 0; d < 3; ++d)
        if (positions[3 * atom + d] != metadata.primitive_motif[atom][d])
          throw std::invalid_argument("Periodic profile feature centers disagree with its motif");
    for (std::size_t atom = 0; atom < mapping.size(); ++atom)
      if (charges[mapping[atom]] != physical[5 * atom + 3])
        throw std::invalid_argument("Periodic profile feature charges disagree with physical source");
  }
  else for (std::size_t atom = 0; atom < shape.nuclei; ++atom)
  {
    if (charges[atom] != species(charge, ions.GroupID[atom]))
      throw std::invalid_argument("Standalone PsiFormer ionic charges differ from XML");
    for (int d = 0; d < 3; ++d)
      if (positions[3 * atom + d] != ions.R[atom][d])
        throw std::invalid_argument("Standalone PsiFormer ordered ionic geometry differs from XML");
  }
  if (model.periodic)
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d)
        if (lattice[3 * a + d] != electrons.getLattice().R(a, d) ||
            lattice[3 * a + d] != ions.getLattice().R(a, d))
          throw std::invalid_argument("Standalone PsiFormer lattice differs from XML");
}

void PsiFormerWF::exportModel(const std::string& path) const
{
  PsiFormerReadTransaction state_lock(*model_state_);
  writeStandaloneModel(path, model_state_->model, model_state_->execution_plan.environment(),
      model_state_->initialization_profile, model_state_->initialization_seed, system_kind_,
      model_state_->twist_phase.reduced, model_state_->physical_source);
}

void PsiFormerWF::exportParameters(const std::string& path) const
{
  PsiFormerReadTransaction state_lock(*model_state_);
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
  validateMatchingLattices(electrons, ions);

  PsiFormerReadTransaction state_lock(*model_state_);
  const auto& environment = model_state_->execution_plan.environment();
  validateRuntimeLattice(electrons, environment, "PsiFormer runtime electron set");
  const pf::PsiFormer& model = model_state_->model;
  if (electrons.getTotalNum() != static_cast<int>(model.ne))
    throw std::invalid_argument("PsiFormer runtime electron count does not match the exported model");
  if (electrons.groups() != 2 || electrons.groupsize(0) != static_cast<int>(model.cfg.nup) ||
      electrons.groupsize(1) != static_cast<int>(model.cfg.ndown))
    throw std::invalid_argument("PsiFormer runtime spin populations do not match the exported model");
  if (!environment.periodic_profile && ions.getTotalNum() != static_cast<int>(model.cfg.nuclei.shape[0]))
    throw std::invalid_argument("PsiFormer runtime nucleus count does not match the exported model");

  const SpeciesSet& species = ions.getSpeciesSet();
  const int charge_index     = species.findAttribute("charge");
  if (charge_index < 0)
    throw std::invalid_argument("PsiFormer source particle set has no charge attribute");
  constexpr double tolerance = 1e-10;
  if (environment.periodic_profile)
  {
    if (model_state_->source != &ions)
      throw std::invalid_argument("Periodic real profile cannot rebind its fixed physical source");
    model_state_->validatePhysicalSource();
  }
  else for (int nucleus = 0; nucleus < ions.getTotalNum(); ++nucleus)
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

  if (state.numerical_tile_backend)
  {
    // Preserve the owning compatibility result without falling back to native
    // CPU mathematics for an explicitly requested device evaluator.
    pf::Result result;
    if (purpose == EvaluationPurpose::VALUE_ONLY)
    {
      const auto values = evaluateDirectValuePositionsUnderRead(transaction, p, replaced_particle, replacement_position);
      result.sign = values.sign;
      result.logabs = values.logabs;
      result.value = values.value;
    }
    else if (purpose == EvaluationPurpose::FULL_SPATIAL || purpose == EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT)
    {
      const auto values = evaluateDirectSpatialPositionsUnderRead(transaction, p, replaced_particle,
          replacement_position, purpose, active_gradient_particle);
      result.sign = values.sign;
      result.logabs = values.logabs;
      result.value = values.value;
      if (purpose == EvaluationPurpose::FULL_SPATIAL)
      {
        result.gradient.assign(values.gradient.data(), values.gradient.data() + values.gradient.size());
        result.lap_log.assign(values.lap_log.data(), values.lap_log.data() + values.lap_log.size());
        result.lap_ratio.assign(values.lap_ratio.data(), values.lap_ratio.data() + values.lap_ratio.size());
      }
      else
        result.active_gradient.assign(values.gradient.data(), values.gradient.data() + values.gradient.size());
    }
    else
      throw std::runtime_error("PsiFormer GPU parameter score/kinetic response is not implemented yet");
    return result;
  }

  // Compatibility outputs are O(N) owning products, outside the numerical-owner cap.
  NativeTimerScope timer(purpose == EvaluationPurpose::VALUE_ONLY ? phase_timers_.value :
      purpose == EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT ? phase_timers_.gradient :
      purpose == EvaluationPurpose::FULL_SPATIAL ? phase_timers_.vgl : phase_timers_.parameter_vjp);
  if (purpose == EvaluationPurpose::FULL_SPATIAL && state.cpu_target_vgl.targets)
  {
    pf::Result result;
    result.gradient.resize(3 * model.ne);
    result.lap_log.resize(model.ne);
    result.lap_ratio.resize(model.ne);
    auto& workspace = requireDirectSpatialWorkspace(purpose);
    const auto value = streamCpuTargetVGL(state.direct_spatial_executor, workspace,
        state.cpu_target_vgl.targets, transaction.parameterVersion(), p, replaced_particle, replacement_position,
        [&](const pf::DirectSpatialResultView& tile) {
          std::copy(tile.gradient.begin(), tile.gradient.end(), result.gradient.begin() + 3 * tile.target_begin);
          std::copy(tile.lap_log.begin(), tile.lap_log.end(), result.lap_log.begin() + tile.target_begin);
          std::copy(tile.lap_ratio.begin(), tile.lap_ratio.end(), result.lap_ratio.begin() + tile.target_begin);
        });
    result.sign = value.sign;
    result.logabs = value.logabs;
    result.value = value.value;
    return result;
  }

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
  NativeTimerScope timer(phase_timers_.value);
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer value transaction belongs to a different model");
  const pf::PsiFormer& model = transaction.model();
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  if (transaction.state().numerical_tile_backend)
  {
    NativeTimerScope host_timer(phase_timers_.gpu_call);
    auto& batch = requireDirectBatchWorkspace();
    batch.resize(pf::DirectBatchMode::VALUE_ONLY, 1);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      const auto& position = electron == replaced_particle
          ? (replacement_position ? *replacement_position : p.activeR(electron)) : p.R[electron];
      for (int dimension = 0; dimension < 3; ++dimension)
        batch.setPosition(0, electron, dimension, position[dimension]);
    }
    const auto result = transaction.state().direct_batch_executor.evaluateValues(batch);
    return {result.sign[0], result.logabs[0], result.value[0], result.parameter_version[0]};
  }

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
  NativeTimerScope timer(purpose == EvaluationPurpose::FULL_SPATIAL ? phase_timers_.vgl : phase_timers_.gradient);
  if (&transaction.state() != model_state_.get())
    throw std::logic_error("PsiFormer spatial transaction belongs to a different model");
  if (purpose != EvaluationPurpose::FULL_SPATIAL &&
      purpose != EvaluationPurpose::ACTIVE_ELECTRON_GRADIENT)
    throw std::logic_error("PsiFormer direct spatial adapter received a non-spatial request");
  if (purpose == EvaluationPurpose::FULL_SPATIAL && transaction.state().cpu_target_vgl.targets)
    throw std::logic_error("PsiFormer CPU target FULL requires an output sink, not a borrowed full view");

  const pf::PsiFormer& model = transaction.model();
  if (static_cast<std::size_t>(p.getTotalNum()) != model.ne)
    throw std::runtime_error("PsiFormerWF electron count differs from exported model");

  if (transaction.state().numerical_tile_backend)
  {
    NativeTimerScope host_timer(phase_timers_.gpu_call);
    auto& batch = requireDirectBatchWorkspace();
    const bool full = purpose == EvaluationPurpose::FULL_SPATIAL;
    batch.resize(full ? pf::DirectBatchMode::FULL_VGL : pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, 1);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      const auto& position = electron == replaced_particle
          ? (replacement_position ? *replacement_position : p.activeR(electron)) : p.R[electron];
      for (int dimension = 0; dimension < 3; ++dimension)
        batch.setPosition(0, electron, dimension, position[dimension]);
    }
    const std::size_t active = static_cast<std::size_t>(active_gradient_particle);
    const auto result = full ? transaction.state().direct_batch_executor.evaluateFull(batch)
        : transaction.state().direct_batch_executor.evaluateActive(batch, &active);
    return {result.sign[0], result.logabs[0], result.value[0], result.mode,
            {result.gradient, result.gradient_stride}, {result.lap_log, result.laplacian_stride},
            {result.lap_ratio, result.laplacian_stride}, result.parameter_version[0]};
  }

  pf::DirectSpatialWorkspace& workspace = requireDirectSpatialWorkspace(purpose);
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
  {
    const auto& position = electron == replaced_particle
        ? (replacement_position ? *replacement_position : p.activeR(electron))
        : p.R[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      workspace.setPosition(electron, dimension, position[dimension]);
  }

  NativeTimerScope numerical_timer(phase_timers_.cpu_spatial);
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
      direct_full_spatial_workspace_ = model_state_->cpu_target_vgl.targets
          ? model_state_->direct_spatial_executor.makeTargetWorkspace(model_state_->cpu_target_vgl.targets,
                model_state_->cpu_target_vgl.owner_budget)
          : model_state_->direct_spatial_executor.makeWorkspace(pf::DirectSpatialMode::FULL_VGL);
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

// Lazy batch scratch belongs only to legacy scalar APIs. Planned scalar VALUE
// operations must use the exact workspace published during clone preparation.
pf::DirectBatchWorkspace& PsiFormerWF::requireDirectBatchWorkspace()
{
  if (batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer legacy scalar batch workspace is not admitted by the explicit batch plan");
  if (!direct_batch_workspace_)
    direct_batch_workspace_ = model_state_->direct_batch_executor.makeWorkspace();
  return *direct_batch_workspace_;
}

// Stage physical G/L in existing proposal products while keeping accepted state untouched.
void PsiFormerWF::stageCpuTargetFullVGL(const PsiFormerReadTransaction& transaction, const ParticleSet& particles)
{
  NativeTimerScope timer(phase_timers_.vgl);
  NativeTimerScope numerical_timer(phase_timers_.cpu_spatial);
  if (&transaction.state() != model_state_.get() || !model_state_->cpu_target_vgl.targets || has_proposal_)
    throw std::logic_error("PsiFormer CPU target FULL requires its own model and no pending proposal");
  auto& workspace = requireDirectSpatialWorkspace(EvaluationPurpose::FULL_SPATIAL);
  resizeProposedSpatialStorage(model_state_->model.ne);
  const auto value = streamCpuTargetVGL(model_state_->direct_spatial_executor, workspace,
      model_state_->cpu_target_vgl.targets, transaction.parameterVersion(), particles, -1, nullptr,
      [&](const pf::DirectSpatialResultView& tile) {
        for (std::size_t t = 0; t < tile.target_count; ++t)
        {
          for (std::size_t d = 0; d < 3; ++d)
            proposed_gradient_[tile.target_begin + t][d] =
                makePhysicalGradient(transaction.state(), tile.gradient[3 * t + d], d);
          proposed_laplacian_[tile.target_begin + t] = static_cast<ValueType>(tile.lap_log[t]);
        }
      });
  proposed_sign_ = value.sign;
  proposed_log_value_ = makePhysicalLogValue(transaction.state(), value.sign, value.logabs,
                                           twistPhaseAngle(transaction.state(), particles));
  proposed_configuration_identity_ = configurationIdentity(particles);
  proposed_parameter_version_ = transaction.parameterVersion();
}

// Dry-run every caller addition and exclude aliases that could invalidate that proof.
void PsiFormerWF::validateCpuTargetFullVGLPublication(const ParticleSet::ParticleGradient& gradients,
    const ParticleSet::ParticleLaplacian& laplacians) const
{
  const std::size_t n = model_state_->model.ne;
  if (gradients.size() != n || laplacians.size() != n || accepted_gradient_.size() != n ||
      accepted_laplacian_.size() != n || proposed_gradient_.size() != n || proposed_laplacian_.size() != n)
    throw std::invalid_argument("PsiFormer CPU target FULL publication has incompatible output shapes");
  const auto g = checkedMemoryRange(gradients.data(), n, "PsiFormer target gradient range overflow");
  const auto l = checkedMemoryRange(laplacians.data(), n, "PsiFormer target Laplacian range overflow");
  for (const auto owned : {checkedMemoryRange(accepted_gradient_.data(), n, "PsiFormer target range overflow"),
       checkedMemoryRange(accepted_laplacian_.data(), n, "PsiFormer target range overflow"),
       checkedMemoryRange(proposed_gradient_.data(), n, "PsiFormer target range overflow"),
       checkedMemoryRange(proposed_laplacian_.data(), n, "PsiFormer target range overflow")})
    if (memoryRangesOverlap(g, owned) || memoryRangesOverlap(l, owned))
      throw std::invalid_argument("PsiFormer CPU target FULL outputs alias retained products");
  if (memoryRangesOverlap(g, l) || !isFiniteWavefunctionValue(proposed_log_value_))
    throw std::invalid_argument("PsiFormer CPU target FULL outputs overlap or have a nonfinite log value");
  for (std::size_t e = 0; e < n; ++e)
  {
    for (std::size_t d = 0; d < 3; ++d)
      if (!isFiniteWavefunctionValue(ValueType(gradients[e][d] + proposed_gradient_[e][d])))
        throw std::overflow_error("PsiFormer CPU target FULL gradient addition is nonfinite");
    if (!isFiniteWavefunctionValue(ValueType(laplacians[e] + proposed_laplacian_[e])))
      throw std::overflow_error("PsiFormer CPU target FULL Laplacian addition is nonfinite");
  }
}

// Promote only completely staged and validated products; every array already has its final size.
void PsiFormerWF::publishCpuTargetFullVGL(ParticleSet::ParticleGradient& gradients,
    ParticleSet::ParticleLaplacian& laplacians) noexcept
{
  for (std::size_t e = 0; e < accepted_gradient_.size(); ++e)
  {
    accepted_gradient_[e] = proposed_gradient_[e];
    accepted_laplacian_[e] = proposed_laplacian_[e];
    gradients[e] += accepted_gradient_[e];
    laplacians[e] += accepted_laplacian_[e];
  }
  current_sign_ = proposed_sign_;
  log_value_ = proposed_log_value_;
  accepted_configuration_identity_ = proposed_configuration_identity_;
  accepted_parameter_version_ = proposed_parameter_version_;
  accepted_state_requirement_ = AcceptedStateRequirement::FULL_SPATIAL;
  accepted_value_valid_ = true;
  clearProposalState();
}

// Fingerprint exact scalar VALUE input provenance without touching prepared scratch.
std::uint64_t PsiFormerWF::scalarValueInputFingerprint(
    const PlannedScalarValueRequest& request) const noexcept
{
  std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
  const auto mix_pointer = [&hash](const void* pointer) noexcept {
    mixPersistentInteger(
        hash,
        static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(pointer)));
  };

  mixPersistentInteger(
      hash,
      request.operation == PlannedScalarValueOperation::ALL_TO_ONE
          ? ALL_TO_ONE_VALUE_FINGERPRINT_DOMAIN
          : VIRTUAL_VALUE_FINGERPRINT_DOMAIN);
  mix_pointer(this);
  mix_pointer(model_state_.get());
  mix_pointer(request.reference);
  mix_pointer(request.virtual_particles);
  mix_pointer(&batch_execution_plan_.plan());
  mixPersistentInteger(hash, batch_execution_plan_.plan().fingerprint());
  mixPersistentInteger(hash, request.configuration_count);
  mixPersistentInteger(hash, request.output_size);
  mixPersistentInteger(hash, configurationIdentity(*request.reference));

  if (request.operation == PlannedScalarValueOperation::ALL_TO_ONE)
  {
    mixPersistentInteger(
        hash,
        static_cast<std::uint64_t>(request.reference->getActivePtcl()));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      mixPersistentDouble(
          hash,
          static_cast<double>(
              request.reference->getActivePos()[dimension]));
  }
  else
  {
    mixPersistentInteger(
        hash,
        static_cast<std::uint64_t>(request.virtual_particles->refPtcl));
    mixPersistentInteger(
        hash, request.virtual_particles->getTotalNum());
    for (const auto& position : request.virtual_particles->R)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        mixPersistentDouble(
            hash, static_cast<double>(position[dimension]));
  }
  return hash;
}

// Prove exact scalar input, output, plan, and prepared-owner evidence.
PsiFormerWF::PlannedScalarValueAccess
PsiFormerWF::requirePlannedScalarValueOperation(
    const PlannedScalarValueRequest& request)
{
  if (!batch_execution_plan_)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE operation has no bound batch plan");
  if (!request.reference)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation has no reference ParticleSet");
  if (!hasPreparedBatchExecutionClone(batch_execution_plan_))
    throw std::logic_error(
        "PsiFormer scalar VALUE workspace was not prepared for the bound batch plan");
  if (bound_particle_set_ != request.reference)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation received a foreign reference ParticleSet");
  if (has_proposal_ || proposal_origin_ != ProposalOrigin::NONE)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE operation requires absent proposal state");

  const BatchExecutionPlan& plan = batch_execution_plan_.plan();
  if (!plan.requirements().requires(
          BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY))
    throw std::logic_error(
        "PsiFormer scalar VALUE compatibility is not admitted by the explicit batch plan");
  if (plan.targetCoordinate() != BatchExecutionTargetCoordinate::POS_ONLY)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation requires explicit POS-only target evidence");
  const BatchExecutionTopology& topology = plan.topology();
  if (topology.serialized_walkers || topology.backend_id != "cpu" ||
      topology.device_id)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation requires direct nonserialized CPU execution");
  if (model_state_->direct_value_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE operation requires the direct VALUE backend");

  const psiformer::PsiFormerMemoryPolicyInput input =
      makeBatchMemoryPolicyInput();
  const psiformer::ModelShape& model_shape =
      model_state_->execution_plan.modelShape();
  const std::size_t electron_count = model_shape.electrons();
  if (plan.particleCount() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE particle count differs from its model");
  if (prepared_accepted_gradient_capacity_ != electron_count ||
      prepared_accepted_laplacian_capacity_ != electron_count ||
      prepared_proposed_gradient_capacity_ != electron_count ||
      prepared_proposed_laplacian_capacity_ != electron_count)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE clone storage differs from its exact prepared extent");

  // Prove the exact accepted configuration and spin ordering before deriving
  // either operation-specific replacement sequence.
  const ParticleSet& reference = *request.reference;
  const auto& soa_positions =
      reference.getCoordinates().getAllParticlePos();
  if (reference.isSpinor())
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation received a spinor ParticleSet");
  if (static_cast<std::size_t>(reference.getTotalNum()) != electron_count ||
      reference.R.size() != electron_count ||
      reference.G.size() != electron_count ||
      reference.L.size() != electron_count ||
      reference.GroupID.size() != electron_count ||
      reference.spins.size() != electron_count ||
      soa_positions.size() != electron_count)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation received incompatible ParticleSet extents");
  if (reference.groups() != 2 ||
      reference.groupsize(0) !=
          static_cast<int>(model_shape.spin_up_electrons) ||
      reference.groupsize(1) !=
          static_cast<int>(model_shape.spin_down_electrons))
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation received an incompatible spin partition");
  if (reference.getActivePtcl() != -1)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE operation requires an inactive ParticleSet move");
  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const int expected_group =
        electron < model_shape.spin_up_electrons ? 0 : 1;
    if (reference.GroupID[electron] != expected_group)
      throw std::invalid_argument(
          "PsiFormer planned scalar VALUE operation received noncanonical spin ordering");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      if (!psiformer::determinant::isFiniteReal(
              static_cast<double>(reference.R[electron][dimension])))
        throw std::invalid_argument(
            "PsiFormer planned scalar VALUE operation received a non-finite accepted position");
      if (soa_positions[electron][dimension] !=
          reference.R[electron][dimension])
        throw std::invalid_argument(
            "PsiFormer planned scalar VALUE operation has inconsistent AoS and SoA positions");
    }
  }

  std::size_t expected_configurations = 0;
  std::size_t expected_outputs        = 0;
  if (request.operation == PlannedScalarValueOperation::ALL_TO_ONE)
  {
    if (request.virtual_particles)
      throw std::invalid_argument(
          "PsiFormer all-to-one scalar request has unexpected virtual-particle input");
    expected_outputs = electron_count;
    expected_configurations = checkedBatchMemoryAdd(
        electron_count, 1,
        "PsiFormer planned all-to-one configuration count");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      if (!psiformer::determinant::isFiniteReal(
              static_cast<double>(
                  reference.getActivePos()[dimension])))
        throw std::invalid_argument(
            "PsiFormer planned all-to-one position is non-finite");
  }
  else
  {
    if (!request.virtual_particles)
      throw std::invalid_argument(
          "PsiFormer planned virtual scalar request has no VirtualParticleSet");
    const VirtualParticleSet& virtual_particles =
        *request.virtual_particles;
    if (&virtual_particles.getRefPS() != &reference)
      throw std::invalid_argument(
          "PsiFormer planned virtual scalar request has a foreign reference ParticleSet");
    if (virtual_particles.isSpinor() != reference.isSpinor())
      throw std::invalid_argument(
          "PsiFormer planned virtual scalar request has an incompatible coordinate mode");
    if (virtual_particles.refPtcl < 0 ||
        virtual_particles.refPtcl >= reference.getTotalNum())
      throw std::out_of_range(
          "PsiFormer planned virtual scalar reference electron is invalid");
    expected_outputs = virtual_particles.getTotalNum();
    if (virtual_particles.R.size() != expected_outputs)
      throw std::invalid_argument(
          "PsiFormer planned virtual scalar positions have the wrong extent");
    expected_configurations = checkedBatchMemoryAdd(
        expected_outputs, 1,
        "PsiFormer planned virtual scalar configuration count");
    if (expected_configurations >
        input.scalar_value_logical_maximum)
      throw std::length_error(
          "PsiFormer planned virtual scalar request exceeds the planned scalar VALUE envelope");
    for (const auto& position : virtual_particles.R)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        if (!psiformer::determinant::isFiniteReal(
                static_cast<double>(position[dimension])))
          throw std::invalid_argument(
              "PsiFormer planned virtual scalar position is non-finite");
    if (batchExecutionModeIsRequired(
            plan.requirements(), BatchExecutionMode::ECP_OUTER))
      throw std::logic_error(
          "PsiFormer scalar virtual-ratio evaluation must use flattened multiwalker ECP dispatch under the explicit batch plan");
  }

  if (request.configuration_count != expected_configurations ||
      request.output_size != expected_outputs ||
      request.output_size > request.output_capacity)
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE request has incompatible output extents");
  if (request.configuration_count >
      input.scalar_value_logical_maximum)
    throw std::length_error(
        "PsiFormer planned scalar VALUE request exceeds the planned scalar VALUE envelope");

  // Reconcile the live owner with both its preparation record and the
  // independently reconstructed canonical scalar capacity plan.
  if (!prepared_scalar_value_compatibility_ ||
      !direct_batch_workspace_ ||
      direct_batch_workspace_.get() !=
          prepared_batch_workspace_identity_ ||
      !direct_batch_workspace_->hasCapacityPlan())
    throw std::logic_error(
        "PsiFormer planned scalar VALUE workspace lacks exact preparation evidence");
  pf::DirectBatchWorkspace& workspace = *direct_batch_workspace_;
  const pf::DirectBatchCapacityPlan expected_capacity =
      psiformer::makePsiFormerScalarValueCapacityPlan(
          input, plan.requirements(), plan.selectedCapacities());
  const pf::DirectBatchStorageRequirement expected_storage =
      pf::directBatchStorageRequirement(
          input.storage_shape, expected_capacity);
  const pf::DirectBatchStorageRequirement actual_storage =
      workspace.actualStorage();
  const std::size_t workspace_bytes = actual_storage.executionBytes();
  const std::size_t workspace_fingerprint = workspace.storageFingerprint(
      pf::DirectBatchMode::VALUE_ONLY);
  if (!sameDirectBatchCapacityPlan(
          workspace.capacityPlan(), expected_capacity) ||
      !sameDirectBatchExecutionStorage(
          actual_storage, expected_storage) ||
      workspace_bytes != expected_storage.executionBytes() ||
      workspace_bytes != workspace.vectorStorageBytes() ||
      workspace_bytes != prepared_batch_bytes_ ||
      workspace_fingerprint == 0 ||
      workspace_fingerprint !=
          prepared_batch_storage_fingerprint_)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE workspace changed after preparation");

  if (scalar_value_publication_.data() !=
          prepared_scalar_value_publication_data_ ||
      scalar_value_publication_.size() !=
          prepared_scalar_value_publication_size_ ||
      scalar_value_publication_.capacity() !=
          prepared_scalar_value_publication_capacity_ ||
      scalar_value_publication_.size() !=
          input.scalar_value_logical_maximum ||
      scalar_value_publication_.capacity() !=
          input.scalar_value_logical_maximum)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE publication storage changed after preparation");

  // The caller's complete retained range must be disjoint from all scalar
  // inputs and every component-owned state or scratch allocation.
  const CheckedMemoryRange output_range = checkedMemoryRange(
      request.output_data, request.output_capacity,
      "PsiFormer planned scalar VALUE output range overflowed");
  const std::size_t output_bytes = output_range.end - output_range.begin;
  const CheckedMemoryRange publication_range = checkedMemoryRange(
      scalar_value_publication_.data(),
      scalar_value_publication_.capacity(),
      "PsiFormer planned scalar VALUE publication range overflowed");
  if (memoryRangesOverlap(output_range, publication_range) ||
      workspace.overlapsStorage(request.output_data, output_bytes))
    throw std::invalid_argument(
        "PsiFormer planned scalar VALUE output aliases prepared scratch");

  for (const CheckedMemoryRange internal : {
           checkedMemoryRange(
               accepted_gradient_.data(), electron_count,
               "PsiFormer accepted gradient range overflowed"),
           checkedMemoryRange(
               accepted_laplacian_.data(), electron_count,
               "PsiFormer accepted Laplacian range overflowed"),
           checkedMemoryRange(
               proposed_gradient_.data(), electron_count,
               "PsiFormer proposed gradient range overflowed"),
           checkedMemoryRange(
               proposed_laplacian_.data(), electron_count,
               "PsiFormer proposed Laplacian range overflowed"),
           checkedMemoryRange(
               std::addressof(current_sign_), std::size_t{1},
               "PsiFormer accepted sign range overflowed"),
           checkedMemoryRange(
               std::addressof(log_value_), std::size_t{1},
               "PsiFormer accepted log-value range overflowed"),
           checkedMemoryRange(
               std::addressof(accepted_value_valid_), std::size_t{1},
               "PsiFormer accepted-validity range overflowed"),
           checkedMemoryRange(
               std::addressof(accepted_configuration_identity_),
               std::size_t{1},
               "PsiFormer accepted-configuration range overflowed"),
           checkedMemoryRange(
               std::addressof(accepted_parameter_version_),
               std::size_t{1},
               "PsiFormer accepted-version range overflowed"),
           checkedMemoryRange(
               std::addressof(accepted_state_requirement_),
               std::size_t{1},
               "PsiFormer accepted-requirement range overflowed"),
           checkedMemoryRange(
               std::addressof(observed_parameter_version_),
               std::size_t{1},
               "PsiFormer observed-version range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_sign_), std::size_t{1},
               "PsiFormer proposed sign range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_log_value_), std::size_t{1},
               "PsiFormer proposed log-value range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_configuration_identity_),
               std::size_t{1},
               "PsiFormer proposed-configuration range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_descriptor_fingerprint_),
               std::size_t{1},
               "PsiFormer proposed-fingerprint range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_parameter_version_),
               std::size_t{1},
               "PsiFormer proposed-version range overflowed"),
           checkedMemoryRange(
               std::addressof(proposed_particle_), std::size_t{1},
               "PsiFormer proposed-particle range overflowed"),
           checkedMemoryRange(
               std::addressof(proposal_origin_), std::size_t{1},
               "PsiFormer proposal-origin range overflowed"),
           checkedMemoryRange(
               std::addressof(has_proposal_), std::size_t{1},
               "PsiFormer proposal marker range overflowed")})
    if (memoryRangesOverlap(output_range, internal))
      throw std::invalid_argument(
          "PsiFormer planned scalar VALUE output aliases component state");

  if (soa_positions.capacity() >
      std::numeric_limits<std::size_t>::max() / 3)
    throw std::length_error(
        "PsiFormer ParticleSet SoA position range overflowed");
  for (const CheckedMemoryRange particle_storage : {
           checkedMemoryRange(
               reference.R.data(), electron_count,
               "PsiFormer ParticleSet position range overflowed"),
           checkedMemoryRange(
               soa_positions.data(), 3 * soa_positions.capacity(),
               "PsiFormer ParticleSet SoA position range overflowed"),
           checkedMemoryRange(
               std::addressof(reference.getActivePos()), std::size_t{1},
               "PsiFormer ParticleSet active-position range overflowed"),
           checkedMemoryRange(
               reference.G.data(), electron_count,
               "PsiFormer ParticleSet gradient range overflowed"),
           checkedMemoryRange(
               reference.L.data(), electron_count,
               "PsiFormer ParticleSet Laplacian range overflowed"),
           checkedMemoryRange(
               reference.GroupID.data(), electron_count,
               "PsiFormer ParticleSet group range overflowed"),
           checkedMemoryRange(
               reference.spins.data(), electron_count,
               "PsiFormer ParticleSet spin range overflowed")})
    if (memoryRangesOverlap(output_range, particle_storage))
      throw std::invalid_argument(
          "PsiFormer planned scalar VALUE output aliases ParticleSet state");

  if (request.virtual_particles)
  {
    const VirtualParticleSet& virtual_particles =
        *request.virtual_particles;
    const auto& virtual_soa =
        virtual_particles.getCoordinates().getAllParticlePos();
    if (virtual_soa.capacity() >
        std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error(
          "PsiFormer VirtualParticleSet SoA range overflowed");
    for (const CheckedMemoryRange virtual_storage : {
             checkedMemoryRange(
                 virtual_particles.R.data(),
                 virtual_particles.R.size(),
                 "PsiFormer VirtualParticleSet position range overflowed"),
             checkedMemoryRange(
                 virtual_soa.data(), 3 * virtual_soa.capacity(),
                 "PsiFormer VirtualParticleSet SoA range overflowed")})
      if (memoryRangesOverlap(output_range, virtual_storage))
        throw std::invalid_argument(
            "PsiFormer planned scalar VALUE output aliases VirtualParticleSet state");
  }

  return {workspace,
          scalar_value_publication_.data(),
          batch_execution_plan_,
          workspace_fingerprint,
          workspace_bytes,
          scalarValueInputFingerprint(request),
          request.output_data,
          request.output_size,
          request.output_capacity};
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

// Deferred crowd owners must not borrow legacy resources under a hard plan.
void PsiFormerWF::requireUnplannedMultiWalkerOperation(
    const char* operation) const
{
  if (batch_execution_plan_)
    throw std::logic_error(
        std::string("PsiFormer ") + operation +
        " is not admitted as a multi-walker operation by the explicit batch plan");
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
  if (model_state_->cpu_target_kinetic.targets)
    throw std::invalid_argument("CPU target kinetic profile supports streaming products, not scalar FULL kinetic derivatives");
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
  NativeTimerScope timer(phase_timers_.cpu_reverse);
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
  if (model_state_->cpu_target_vgl.targets && has_proposal_)
    throw std::logic_error("PsiFormer CPU target FULL cannot overwrite a pending proposal");
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
  if (transaction.state().cpu_target_vgl.targets)
  {
    stageCpuTargetFullVGL(transaction, p);
    validateCpuTargetFullVGLPublication(g, l);
    publishCpuTargetFullVGL(g, l);
    return log_value_;
  }

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
    log_value_ = makePhysicalLogValue(
        transaction.state(), sign, logabs,
        twistPhaseAngle(transaction.state(), p));
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        accepted_gradient_[electron][dimension] = makePhysicalGradient(
            transaction.state(), gradient[3 * electron + dimension], dimension);
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

/// Parse only on an actual GPU provider; explicit zero/MAX are not an opt-in cap.
std::size_t PsiFormerWF::residentEnergyAdamBudget() const
{
#ifdef ENABLE_CUDA
  if (!model_state_->numerical_tile_backend)
    return 0;
  const char* value = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES");
  if (!value) return 0;
  std::size_t bytes = 0;
  for (const char* digit = value; *digit; ++digit)
  {
    if (*digit < '0' || *digit > '9' || bytes > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
      throw std::invalid_argument("Resident ENERGY Adam cap must be positive finite decimal bytes");
    bytes = 10 * bytes + (*digit - '0');
  }
  if (!bytes || bytes == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Resident ENERGY Adam cap must be positive finite decimal bytes");
  return bytes;
#else
  return 0;
#endif
}

/// Admit C3+norm before either optional allocation, then retain exactly one lane.
std::size_t PsiFormerWF::prepareResidentEnergyAdam(std::size_t budget) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  std::shared_ptr<PsiFormerDerivativeScratch> scratch;
  std::size_t p = 0, c3 = 0, norm = 0;
  {
    PsiFormerReadTransaction state_lock(*model_state_);
    const auto& environment = model_state_->execution_plan.environment();
    if (!model_state_->numerical_tile_backend ||
        model_state_->precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE ||
        environment.amplitude_scalar_domain != psiformer::ScalarDomain::REAL ||
        environment.boundary != psiformer::BoundaryCondition::OPEN ||
        std::any_of(environment.periodic_axes.begin(), environment.periodic_axes.end(), [](bool axis) { return axis; }))
      throw std::invalid_argument("Resident ENERGY Adam requires ordinary real/open/FP64 GPU products");
    p = structured_parameter_schema_->parameterCount();
    c3 = psiformer::device::PreparedDeviceScoreContraction::requiredBytes(p, 3);
    norm = psiformer::device::PreparedDeviceAdam::energyGradientRequiredBytes(p);
    if (!budget || budget == std::numeric_limits<std::size_t>::max() ||
        checkedBatchMemoryAdd(c3, norm, "Resident ENERGY bridge") > budget)
      throw std::length_error("Resident ENERGY C3+norm exceeds its explicit device bridge cap");
    scratch = acquireDerivativeScratch(*model_state_, p, 4); // Factory never waits for execution.
  }
  // Match evaluation lock order: never wait for the lane while holding a model read.
  PsiFormerDerivativeLease lease(*scratch);
  PsiFormerReadTransaction state_lock(*model_state_);
  scratch->device_products->prepareScoreContractions(3, model_state_->model.p.version(), c3);
  app_log() << "Resident ENERGY bridge admitted: C3 device bytes=" << c3 << ", norm device bytes=" << norm
            << ", C3 status host bytes=" << c3 - 48 * p << ", norm host bytes=" << norm
            << "; cap covers C3+norm device payload ONLY; existing Adam, model/tapes, host compatibility/canonical/coefficients, sampling, vendor and checkpoints excluded" << std::endl;
  std::lock_guard factory_lock(model_state_->derivative_scratch_factory_mutex);
  model_state_->resident_energy_scratch = std::move(scratch);
  return norm;
#else
  throw std::invalid_argument("Resident ENERGY Adam requires a real GPU build");
#endif
}

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
namespace
{
/// Own the borrowed session/backend and immutable image receipts, not native caches.
struct ResidentOpenBatch final : PsiFormerWF::ResidentOpenObservationBatch
{
  std::shared_ptr<PsiFormerSharedState> state;
  std::shared_ptr<PsiFormerDerivativeScratch> scratch;
  std::shared_ptr<const pf::DirectBatchTileBackend> backend;
  std::vector<psiformer::device::PreparedDeviceProductSession::ResidentOpenEnergyObservation> observations;
  std::size_t version = 0;
  std::size_t sourceVersion() const noexcept override { return version; }
  std::size_t sampleCount() const noexcept override { return observations.size(); }
  const std::array<double, 5>& energy(std::size_t sample) const override
  { return observations.at(sample).energy; }
};
}
#endif

PsiFormerWF::ResidentOpenObservationRequirements PsiFormerWF::residentOpenObservationRequirements(
    std::size_t samples) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  PsiFormerReadTransaction read(*model_state_, model_state_->resident_device_authority_enabled);
  const auto& shape = read.state().execution_plan.modelShape();
  if (samples != 2 || system_kind_ != "all_electron")
    throw std::invalid_argument("Resident open observation requires the admitted sole AE W2 profile");
  const auto cost = psiformer::device::PreparedDeviceProductSession::estimateResidentOpenEnergyCost(
      shape.electrons(), shape.nuclei);
  const auto receipts = checkedBatchMemoryAdd(sizeof(ResidentOpenBatch), checkedBatchMemoryMultiply(samples,
      sizeof(psiformer::device::PreparedDeviceProductSession::ResidentOpenEnergyObservation),
      "Resident observation receipts"), "Resident observation owner");
  // Cold preparation temporarily holds the caller's mass vector alongside the
  // retained session copy. This is a cold HOST peak, not retained inventory.
  return {cost.device_bytes, checkedBatchMemoryAdd(cost.host_bytes,
      checkedBatchMemoryMultiply(shape.electrons(), sizeof(double), "Resident cold masses"),
      "Resident cold HOST peak"), receipts};
#else
  throw std::invalid_argument("Resident open observation requires a real GPU build");
#endif
}

void PsiFormerWF::prepareResidentOpenObservation(const ParticleSet& electrons, const ParticleSet& ions,
    std::size_t device_cap, std::size_t host_cap) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto scratch = model_state_->resident_energy_scratch;
  if (!scratch || system_kind_ != "all_electron" || !deviceSamplingEnabled())
    throw std::invalid_argument("Resident open observation requires prepared sole AE sampling/C3");
  PsiFormerDerivativeLease lease(*scratch);
  PsiFormerReadTransaction read(*model_state_, model_state_->resident_device_authority_enabled);
  const auto ne = read.state().execution_plan.modelShape().electrons();
  if (electrons.getTotalNum() != ne || electrons.isSpinor() || electrons.getActivePtcl() != -1)
    throw std::invalid_argument("Resident open observation particle shape is incompatible");
  const auto& model = read.state().model;
  if (ions.getTotalNum() != model.cfg.charges.x.size())
    throw std::invalid_argument("Resident open ion source extent differs from its immutable model");
  for (int ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    if (!wftrain::isFiniteTrainingReal(ions.Z[ion]) || ions.Z[ion] != model.cfg.charges.x[ion])
      throw std::invalid_argument("Resident open actual ion charge differs from its immutable model");
    for (std::size_t d = 0; d < 3; ++d)
      if (!wftrain::isFiniteTrainingReal(ions.R[ion][d]) ||
          ions.R[ion][d] != model.cfg.nuclei.x[3 * ion + d])
        throw std::invalid_argument("Resident open actual ion position differs from its immutable model");
  }
  std::vector<double> inverse_masses(ne);
  for (std::size_t e = 0; e < ne; ++e)
  {
    const auto group = electrons.getGroupID(e);
    const auto& masses = electrons.get_mass_by_group();
    if (group < 0 || static_cast<std::size_t>(group) >= masses.size() ||
        !wftrain::isFiniteTrainingReal(masses[group]) || masses[group] <= 0)
      throw std::invalid_argument("Resident open observation mass group is invalid");
    inverse_masses[e] = 1.0 / masses[group];
  }
  const auto temporary_bytes = checkedBatchMemoryMultiply(inverse_masses.capacity(), sizeof(double),
      "Resident temporary masses");
  if (host_cap <= temporary_bytes)
    throw std::length_error("Resident open cold HOST cap does not cover temporary mass staging");
  scratch->device_products->prepareResidentOpenEnergy(inverse_masses,
      read.state().model.cfg.charges.x, device_cap, host_cap - temporary_bytes);
#else
  throw std::invalid_argument("Resident open observation requires a real GPU build");
#endif
}

std::unique_ptr<PsiFormerWF::ResidentOpenObservationBatch> PsiFormerWF::observeResidentOpenAccepted(
    const RefVectorWithLeader<WaveFunctionComponent>& components) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto scratch = model_state_->resident_energy_scratch;
  if (!scratch || !model_state_->resident_device_authority_enabled || components.size() != 2 ||
      system_kind_ != "all_electron")
    throw std::invalid_argument("Resident observation requires active sole AE DEVICE authority");
  auto& resource = requireMultiWalkerResource(components);
  PsiFormerDerivativeLease lease(*scratch);
  PsiFormerReadTransaction read(*model_state_, true);
  auto result = std::make_unique<ResidentOpenBatch>();
  result->state = model_state_;
  result->scratch = scratch;
  result->backend = model_state_->numerical_tile_backend;
  result->version = read.parameterVersion();
  const auto& backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend&>(*result->backend);
  result->observations.reserve(components.size());
  if (result->observations.capacity() != components.size())
    throw std::length_error("Resident observation receipt capacity exceeds its admitted W2 elements");
  for (std::size_t sample = 0; sample < components.size(); ++sample)
  {
    const auto& component = components.getCastedElement<PsiFormerWF>(sample);
    component.requireNoSelectedParticleProposal("resident open observation");
    if (component.model_state_ != model_state_ || component.system_kind_ != "all_electron")
      throw std::invalid_argument("Resident observation contains a foreign component");
    const auto coordinates = backend.residentAcceptedCoordinates(
        resource.batch_workspace->numericalSamplingWorkspace(), sample, result->version);
    result->observations.push_back(scratch->device_products->observeResidentOpenEnergy(coordinates, result->version));
  }
  return result;
#else
  throw std::invalid_argument("Resident open observation requires a real GPU build");
#endif
}

void PsiFormerWF::materializeResidentOpenAccepted(
    const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& particles) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  if (!model_state_->resident_device_authority_enabled || components.empty() || components.size() != particles.size())
    throw std::invalid_argument("Resident cold fence requires the admitted nonempty crowd");
  auto& resource = requireMultiWalkerResource(components);
  PsiFormerReadTransaction read(*model_state_, true);
  const auto ne = read.state().execution_plan.modelShape().electrons();
  for (std::size_t sample = 0; sample < components.size(); ++sample)
    if (components.getCastedElement<PsiFormerWF>(sample).model_state_ != model_state_ ||
        particles[sample].getTotalNum() != ne || particles[sample].getActivePtcl() != -1)
      throw std::invalid_argument("Resident cold fence contains foreign particles/components");
  std::vector<double> positions(checkedBatchMemoryMultiply(components.size(), 3 * ne, "Resident cold coordinates"));
  const auto& backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend&>(
      *model_state_->numerical_tile_backend);
  backend.materializeResidentAccepted(resource.batch_workspace->numericalSamplingWorkspace(), positions,
      read.parameterVersion());
  // This publishes ONLY coordinates. Native tables, FULL, walkers and H must be refreshed by the caller.
  for (std::size_t sample = 0; sample < particles.size(); ++sample)
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
        particles[sample].R[e][d] = positions[(sample * ne + e) * 3 + d];
#else
  throw std::invalid_argument("Resident cold fence requires a real GPU build");
#endif
}

/// Keep the concrete source and both read leases scoped through Adam completion.
double PsiFormerWF::consumeResidentEnergyAdam(const wftrain::StreamingDerivativeOperator& derivative,
    wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
    psiformer::device::PreparedDeviceAdam& adam, const std::vector<double>& parameters,
    double weight, double energy, double beta1, double beta2, double correction1, double correction2,
    double epsilon, std::vector<double>& candidate) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto* source = dynamic_cast<const PsiFormerStreamingDerivativeOperator*>(&derivative);
  if (!source || source->model_state_.get() != model_state_.get() ||
      source->shared_scratch_ != model_state_->resident_energy_scratch || derivative.sampleCount() < 7)
    throw std::invalid_argument("Resident ENERGY requires this provider's current prepared batch");
  (void)source->validateVJPRequest(channels, wftrain::DerivativeAdjoint::TRANSPOSE);
  if (channels.size() != 3 || channels[0].product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
      channels[0].local_energy_term_mask != wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) ||
      channels[1].product != wftrain::DerivativeProduct::SCORE_VJP || channels[2].product != wftrain::DerivativeProduct::SCORE_VJP)
    throw std::invalid_argument("Resident ENERGY requires this provider's current T/EO/O kinetic batch");
  double norm = 0;
  source->evaluateVJPsImpl(channels, wftrain::DerivativeAdjoint::TRANSPOSE, nullptr, [&](std::size_t generation) {
    auto& session = *source->shared_scratch_->device_products;
    const auto before = session.transferSnapshot(); // Seed/producer/finish/checkpoint traffic is OUTSIDE.
    norm = session.consumeEnergyAdam(adam, derivative.parameterVersion(), generation,
        parameters, weight, energy, beta1, beta2, correction1, correction2, epsilon, candidate);
    const auto after = session.transferSnapshot();
    app_log() << "Resident ENERGY terminal consumer: P=" << parameters.size()
              << ", parameter_version=" << derivative.parameterVersion()
              << ", uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
              << ", downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes
              << ", allocation_calls=" << after.allocation_calls - before.allocation_calls
              << ", free_calls=" << after.free_calls - before.free_calls
              << ", expected_uploaded_bytes=" << 8 * parameters.size()
              << ", expected_downloaded_bytes=" << 8 * parameters.size() + adam.energyGradientDeviceBytes() + 16
              << ", counter_scope=device_wide, seed_producer_checkpoint_excluded=yes" << std::endl;
  });
  return norm;
#else
  throw std::invalid_argument("Resident ENERGY Adam requires a real GPU build");
#endif
}

#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
namespace
{
/// Keep Adam and its admitted immutable candidate alive through publication readiness.
struct ResidentAuthorityProposal final : PsiFormerWF::ResidentDeviceAdamProposal
{
  std::shared_ptr<PsiFormerSharedState> state;
  std::shared_ptr<psiformer::device::PreparedDeviceAdam> adam;
  psiformer::device::PreparedDeviceAdam::DeviceEnergyCandidate candidate;

  std::size_t sourceVersion() const noexcept override { return candidate.snapshot->sourceVersion(); }

  double gradientNorm() const noexcept override { return candidate.gradient_norm; }

  double displacementNorm() const noexcept override { return candidate.displacement_norm; }
};

/// Hold final model/backend authority; retire all numeric owners only after unlocking.
struct ResidentAuthorityPublication final : PsiFormerWF::PreparedResidentDeviceAdamPublication
{
  std::unique_ptr<ResidentAuthorityProposal> proposal;
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> successor;
  std::unique_ptr<psiformer::device::PreparedDeviceBatchBackend::PreparedDevicePublication> backend;
  std::unique_lock<std::shared_mutex> authority;

  std::size_t commit() noexcept override
  {
    if (!authority.owns_lock())
      std::terminate();
    backend->commit();
    const auto version = proposal->sourceVersion() + 1;
    proposal->state->model.p.parameter_version = version;
    proposal->adam->acceptPreparedDeviceCandidate(*proposal->candidate.snapshot, successor);
    backend->unlock();
    authority.unlock();
    return version;
  }
};

/// The solver borrows Session: retain the complete anchored resource chain, not merely solver storage.
struct ResidentSRAuthorityProposal final : PsiFormerWF::ResidentDeviceSRProposal
{
  std::shared_ptr<PsiFormerSharedState> state;
  std::shared_ptr<const psiformer::device::PreparedDeviceBatchBackend> backend;
  std::shared_ptr<PsiFormerDerivativeScratch> scratch;
  psiformer::device::PreparedDevicePcg::DeviceEnergyCandidate candidate{};
  wftrain::ResidentEnergySRDiagnostics completed;

  std::size_t sourceVersion() const noexcept override { return candidate.snapshot->sourceVersion(); }

  const wftrain::ResidentEnergySRDiagnostics& diagnostics() const noexcept override { return completed; }

  double displacementNorm() const noexcept override { return candidate.displacement_norm; }
};

/// Match Adam's authority/retirement order while accepting only the prevalidated SR numerical control.
struct ResidentSRAuthorityPublication final : PsiFormerWF::PreparedResidentDeviceSRPublication
{
  std::unique_ptr<ResidentSRAuthorityProposal> proposal;
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> successor;
  std::unique_ptr<psiformer::device::PreparedDeviceBatchBackend::PreparedDevicePublication> backend;
  std::unique_lock<std::shared_mutex> authority;

  std::size_t commit() noexcept override
  {
    if (!authority.owns_lock())
      std::terminate();
    backend->commit();
    const auto version = proposal->sourceVersion() + 1;
    proposal->state->model.p.parameter_version = version;
    proposal->candidate.owner->acceptPreparedDeviceCandidate(*proposal->candidate.snapshot, successor);
    backend->unlock();
    authority.unlock();
    return version;
  }
};
}
#endif

/// The strict GPU-only opt-in was frozen when the shared numerical owner was constructed.
bool PsiFormerWF::residentDeviceAuthorityRequested() const noexcept
{
  return model_state_->resident_device_authority_requested;
}

/// Observe a whole iteration without resetting shared metrics or synchronizing the device.
void PsiFormerWF::residentDeviceAuthorityTransferWindow(const char* label, bool begin) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  PsiFormerReadTransaction read(*model_state_, true);
  const auto& scratch = model_state_->resident_energy_scratch;
  if (!model_state_->resident_device_authority_enabled || !scratch || !label)
    throw std::logic_error("Resident DEVICE transfer window requires active authority");
  const auto now = scratch->device_products->transferSnapshot();
  if (begin)
  {
    if (model_state_->resident_transfer_window_active)
      throw std::logic_error("Resident DEVICE transfer window already active");
    model_state_->resident_transfer_baseline = now;
    model_state_->resident_transfer_window_active = true;
    return;
  }
  if (!model_state_->resident_transfer_window_active)
    throw std::logic_error("Resident DEVICE transfer window has no baseline");
  const auto& before = model_state_->resident_transfer_baseline;
  app_log() << "Resident DEVICE authority window: label=" << label
            << ", parameter_version=" << read.parameterVersion()
            << ", uploaded_bytes=" << now.uploaded_bytes - before.uploaded_bytes
            << ", downloaded_bytes=" << now.downloaded_bytes - before.downloaded_bytes
            << ", device_copied_bytes=" << now.device_copied_bytes - before.device_copied_bytes
            << ", counter_scope=device_wide, cold_checkpoint_excluded=yes" << std::endl;
  model_state_->resident_transfer_window_active = false;
#else
  throw std::invalid_argument("Resident DEVICE authority requires a real GPU build");
#endif
}

/// Activate only the admitted real/open family; allocations remain separately budgeted.
PsiFormerWF::ResidentDeviceAuthorityRequirements PsiFormerWF::prepareResidentDeviceAuthority() const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  auto lock = model_state_->acquireHostWrite();
  const auto& environment = model_state_->execution_plan.environment();
  if (!residentDeviceAuthorityRequested() || !model_state_->resident_energy_scratch ||
      !model_state_->numerical_tile_backend || batch_execution_plan_ ||
      model_state_->precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE ||
      environment.boundary != psiformer::BoundaryCondition::OPEN ||
      environment.amplitude_scalar_domain != psiformer::ScalarDomain::REAL)
    throw std::invalid_argument("DEVICE authority requires explicitly prepared real/open resident ENERGY");
  const auto bytes = checkedBatchMemoryMultiply(structured_parameter_schema_->parameterCount(),
      sizeof(double), "Resident DEVICE authority canonical image");
  model_state_->resident_device_authority_enabled = true;
  // The finite snapshot and immutable successor share ONE physical P allocation.
  return {bytes, 0, bytes};
#else
  throw std::invalid_argument("Resident DEVICE authority requires a real GPU build");
#endif
}

/// Produce a finite owning snapshot without exposing or downloading a HOST P vector.
std::unique_ptr<PsiFormerWF::ResidentDeviceAdamProposal> PsiFormerWF::consumeResidentEnergyAdamDevice(
    const wftrain::StreamingDerivativeOperator& derivative,
    wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
    const std::shared_ptr<psiformer::device::PreparedDeviceAdam>& adam,
    double weight, double energy, double beta1, double beta2, double correction1, double correction2,
    double epsilon, std::size_t snapshot_device_cap) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto* source = dynamic_cast<const PsiFormerStreamingDerivativeOperator*>(&derivative);
  if (!adam || !model_state_->resident_device_authority_enabled || !source ||
      source->model_state_.get() != model_state_.get() || derivative.sampleCount() == 0 ||
      source->shared_scratch_ != model_state_->resident_energy_scratch || channels.size() != 3 ||
      channels[0].product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
      (channels[0].local_energy_term_mask != wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) &&
       channels[0].local_energy_term_mask != (wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) |
                                             wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP))) ||
      channels[1].product != wftrain::DerivativeProduct::SCORE_VJP ||
      channels[2].product != wftrain::DerivativeProduct::SCORE_VJP)
    throw std::invalid_argument("DEVICE authority requires this provider's prepared T/EO/O batch");
  (void)source->validateVJPRequest(channels, wftrain::DerivativeAdjoint::TRANSPOSE);
  auto proposal = std::make_unique<ResidentAuthorityProposal>();
  proposal->state = model_state_;
  proposal->adam = adam;
  source->evaluateVJPsImpl(channels, wftrain::DerivativeAdjoint::TRANSPOSE, nullptr,
      [&](std::size_t generation) {
        proposal->candidate = source->shared_scratch_->device_products->consumeEnergyAdamDeviceCandidate(
            *adam, derivative.parameterVersion(), generation, weight, energy, beta1, beta2,
            correction1, correction2, epsilon, snapshot_device_cap, true);
      }, true);
  return proposal;
#else
  throw std::invalid_argument("Resident DEVICE authority requires a real GPU build");
#endif
}

/// Preserve finite-sample mean(dK): SCORE-only covariance is not this estimator.
std::unique_ptr<PsiFormerWF::ResidentDeviceAdamProposal> PsiFormerWF::consumeResidentOpenAdamDevice(
    const ResidentOpenObservationBatch& opaque,
    const std::shared_ptr<psiformer::device::PreparedDeviceAdam>& adam,
    double beta1, double beta2, double correction1, double correction2, double epsilon,
    std::size_t snapshot_device_cap) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto* batch = dynamic_cast<const ResidentOpenBatch*>(&opaque);
  if (!batch || batch->state != model_state_ || batch->scratch != model_state_->resident_energy_scratch ||
      !model_state_->resident_device_authority_enabled || system_kind_ != "all_electron" || !adam ||
      batch->sampleCount() != 2)
    throw std::invalid_argument("Resident open Adam requires this provider's admitted equal-weight W2 observations");
  PsiFormerDerivativeLease lease(*batch->scratch);
  PsiFormerReadTransaction read(*model_state_, true);
  if (read.parameterVersion() != batch->version || batch->backend != model_state_->numerical_tile_backend)
    throw std::invalid_argument("Resident open Adam observations are stale");
  auto proposal = std::make_unique<ResidentAuthorityProposal>();
  proposal->state = model_state_;
  proposal->adam = adam;
  auto& session = *batch->scratch->device_products;
  const auto generation = session.beginScoreContraction(3, batch->version,
      psiformer::device::ScoreContractionRole::ENERGY_T_EO_O);
  double energy = 0;
  try
  {
    for (const auto& observation : batch->observations)
    {
      const double local_energy = observation.energy[4];
      if (!wftrain::isFiniteTrainingReal(local_energy))
        throw std::invalid_argument("Resident open Adam energy is nonfinite");
      const std::array<std::complex<double>, 3> score{{0.0, local_energy, 1.0}};
      const std::array<std::complex<double>, 3> kinetic{{1.0, 0.0, 0.0}};
      session.accumulateResidentKineticContraction(observation, score.data(), kinetic.data(), 3, batch->version);
      energy += local_energy;
    }
    session.finishScoreContraction(batch->version);
    proposal->candidate = session.consumeEnergyAdamDeviceCandidate(*adam, batch->version, generation,
        2.0, energy / 2.0, beta1, beta2, correction1, correction2, epsilon, snapshot_device_cap, true);
  }
  catch (...)
  {
    session.abortScoreContraction();
    throw;
  }
  return proposal;
#else
  throw std::invalid_argument("Resident open Adam requires a real GPU build");
#endif
}

/// Finish all fallible successor/receipt work before the caller's collective readiness.
std::unique_ptr<PsiFormerWF::PreparedResidentDeviceAdamPublication>
PsiFormerWF::prepareResidentDeviceAdamPublication(std::unique_ptr<ResidentDeviceAdamProposal> opaque,
    std::size_t expected_version, std::size_t successor_device_cap)
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  auto* proposal = dynamic_cast<ResidentAuthorityProposal*>(opaque.get());
  if (!proposal || proposal->state != model_state_ || proposal->sourceVersion() != expected_version)
    throw std::invalid_argument("Resident DEVICE proposal is foreign or stale");
  const auto backend = std::dynamic_pointer_cast<const psiformer::device::PreparedDeviceBatchBackend>(
      model_state_->numerical_tile_backend);
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> source;
  {
    PsiFormerReadTransaction read(*model_state_, true);
    if (read.parameterVersion() != expected_version || !backend)
      throw std::invalid_argument("Resident DEVICE source version changed");
    source = backend->currentImage();
  }
  if (successor_device_cap != 0)
    throw std::invalid_argument("Shared DEVICE successor requires zero additional numeric allocation allowance");
  auto successor = source->makeSharedUpdatedImage(*proposal->candidate.snapshot);
  auto publication = std::make_unique<ResidentAuthorityPublication>();
  publication->authority = std::unique_lock<std::shared_mutex>(model_state_->mutex);
  requireNoPlannedProposalMutation("resident DEVICE publication");
  if (model_state_->model.p.version() != expected_version)
    throw std::invalid_argument("Resident DEVICE publication version changed");
  proposal->adam->validateDeviceAcceptance(*proposal->candidate.snapshot, successor);
  publication->backend = std::make_unique<psiformer::device::PreparedDeviceBatchBackend::PreparedDevicePublication>(
      backend, source, successor);
  publication->successor = std::move(successor);
  publication->proposal.reset(static_cast<ResidentAuthorityProposal*>(opaque.release()));
  return publication;
#else
  throw std::invalid_argument("Resident DEVICE authority requires a real GPU build");
#endif
}

/// Consume one admitted real/open complete C3 into owning DEVICE bytes, never a HOST candidate.
std::unique_ptr<PsiFormerWF::ResidentDeviceSRProposal> PsiFormerWF::consumeResidentEnergySRDevice(
    const wftrain::StreamingDerivativeOperator& derivative,
    wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
    const std::vector<double>& weights, double weight, double energy, double damping, double learning_rate,
    const wftrain::KrylovSolverControl& control, std::pair<std::size_t, std::size_t> budget,
    std::size_t snapshot_device_cap) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto* source = dynamic_cast<const PsiFormerStreamingDerivativeOperator*>(&derivative);
  const bool has_ecp = source && source->ecp_replay_;
  const auto complete_mask = wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) |
      (has_ecp ? wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP) : 0);
  if (system_kind_ != (has_ecp ? "pseudopotential" : "all_electron") || !model_state_->resident_device_authority_enabled ||
      !model_state_->resident_energy_sr_reuse || !source || source->model_state_ != model_state_ ||
      source->shared_scratch_ != model_state_->resident_energy_scratch ||
      !derivative.sampleCount() || weights.size() != derivative.sampleCount() || channels.size() != 3 ||
      channels[0].product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
      channels[0].local_energy_term_mask != complete_mask ||
      channels[1].product != wftrain::DerivativeProduct::SCORE_VJP || channels[2].product != wftrain::DerivativeProduct::SCORE_VJP)
    throw std::invalid_argument("DEVICE SR authority requires this provider's retained real/open complete T/EO/O batch");
  (void)source->validateVJPRequest(channels, wftrain::DerivativeAdjoint::TRANSPOSE);
  using Owner = psiformer::device::PreparedDevicePcg;
  std::vector<Owner::FrozenInterval> frozen;
  for (const auto& block : structured_parameter_schema_->blocks())
    if (!block.trainable)
      frozen.emplace_back(block.offset, block.count);
  auto proposal = std::make_unique<ResidentSRAuthorityProposal>();
  proposal->state = model_state_;
  proposal->scratch = source->shared_scratch_;
  proposal->backend = std::dynamic_pointer_cast<const psiformer::device::PreparedDeviceBatchBackend>(
      model_state_->numerical_tile_backend);
  if (!proposal->backend)
    throw std::invalid_argument("DEVICE SR authority requires the admitted GPU backend");
  source->evaluateVJPsImpl(channels, wftrain::DerivativeAdjoint::TRANSPOSE, nullptr, [&](std::size_t generation) {
    auto& session = *proposal->scratch->device_products;
    if (!model_state_->resident_energy_sr_workspace)
    {
      if (model_state_->resident_energy_sr_workspace_started)
        throw std::logic_error("DEVICE SR failed cold preparation requires explicit stage re-preparation");
      model_state_->resident_energy_sr_workspace_started = true;
      model_state_->resident_energy_sr_workspace = Owner::prepareRetainedEnergy(session, source->positions_, weights,
          frozen, derivative.parameterVersion(), damping, budget.first, budget.second);
    }
    const auto before = session.transferSnapshot();
    const auto& owner = model_state_->resident_energy_sr_workspace;
    proposal->candidate = owner->consumeRetainedEnergyDevice(owner, source->positions_, weights, frozen,
        derivative.parameterVersion(), generation, weight, energy, damping, learning_rate, control,
        budget.first, budget.second, snapshot_device_cap);
    const auto& result = proposal->candidate.result;
    proposal->completed = {result.solve, result.gradient_norm, result.direction_norm, result.metric_norm};
    const auto after = session.transferSnapshot();
    app_log() << "Resident DEVICE SR terminal: source_version=" << derivative.parameterVersion()
              << ", samples=" << derivative.sampleCount() << ", snapshot_device_bytes=" << proposal->candidate.snapshot->deviceBytes()
              << ", uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
              << ", downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes
              << ", allocation_calls=" << after.allocation_calls - before.allocation_calls
              << "; raw_square_displacement_domain=yes; component_terminal_only=yes; caller_owns_profile_admission=yes" << std::endl;
  }, true);
  return proposal;
#else
  throw std::invalid_argument("Resident DEVICE SR authority requires a real GPU build");
#endif
}

/// Pin and prevalidate the exact retained solver/session before final same-thread publication authority.
std::unique_ptr<PsiFormerWF::PreparedResidentDeviceSRPublication>
PsiFormerWF::prepareResidentDeviceSRPublication(std::unique_ptr<ResidentDeviceSRProposal> opaque,
    std::size_t expected_version, std::size_t successor_device_cap)
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  auto* proposal = dynamic_cast<ResidentSRAuthorityProposal*>(opaque.get());
  if (!proposal || proposal->state != model_state_ || proposal->sourceVersion() != expected_version ||
      proposal->scratch != model_state_->resident_energy_scratch ||
      proposal->candidate.owner != model_state_->resident_energy_sr_workspace)
    throw std::invalid_argument("Resident DEVICE SR proposal is foreign, stale or no longer anchored");
  std::shared_ptr<const psiformer::device::PreparedDeviceModel> source;
  {
    PsiFormerReadTransaction read(*model_state_, true);
    if (read.parameterVersion() != expected_version)
      throw std::invalid_argument("Resident DEVICE SR source version changed");
    source = proposal->backend->currentImage();
  }
  if (successor_device_cap != 0)
    throw std::invalid_argument("Shared DEVICE SR successor requires zero additional numeric allocation allowance");
  auto successor = source->makeSharedUpdatedImage(*proposal->candidate.snapshot);
  auto publication = std::make_unique<ResidentSRAuthorityPublication>();
  publication->authority = std::unique_lock<std::shared_mutex>(model_state_->mutex);
  requireNoPlannedProposalMutation("resident DEVICE SR publication");
  if (model_state_->model.p.version() != expected_version || proposal->scratch != model_state_->resident_energy_scratch ||
      proposal->candidate.owner != model_state_->resident_energy_sr_workspace)
    throw std::invalid_argument("Resident DEVICE SR publication source/anchor changed");
  proposal->candidate.owner->validateDeviceAcceptance(*proposal->candidate.snapshot, successor);
  publication->backend = std::make_unique<psiformer::device::PreparedDeviceBatchBackend::PreparedDevicePublication>(
      proposal->backend, source, successor);
  publication->successor = std::move(successor);
  publication->proposal.reset(static_cast<ResidentSRAuthorityProposal*>(opaque.release()));
  return publication;
#else
  throw std::invalid_argument("Resident DEVICE SR authority requires a real GPU build");
#endif
}

/// Resolve both explicit SR caps together; CPU never parses GPU-only controls.
std::pair<std::size_t, std::size_t> PsiFormerWF::residentEnergySRBudget() const
{
#ifdef ENABLE_CUDA
  if (!model_state_->numerical_tile_backend)
    return {0, 0};
  const char* device = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES");
  const char* host = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES");
  if (!device && !host)
  {
    if (const char* reuse = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE"); reuse && std::string(reuse) != "off")
      throw std::invalid_argument("Resident ENERGY SR reuse=on requires both caps; reuse must be on or off");
    return {0, 0};
  }
  const auto parse = [](const char* value) {
    std::size_t bytes = 0;
    if (!value)
      throw std::invalid_argument("Resident ENERGY SR requires both explicit caps");
    for (const char* digit = value; *digit; ++digit)
    {
      if (*digit < '0' || *digit > '9' || bytes > (std::numeric_limits<std::size_t>::max() - (*digit - '0')) / 10)
        throw std::invalid_argument("Resident ENERGY SR cap must be positive finite decimal bytes");
      bytes = 10 * bytes + (*digit - '0');
    }
    if (!bytes || bytes == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Resident ENERGY SR cap must be positive finite decimal bytes");
    return bytes;
  };
  return {parse(device), parse(host)};
#else
  return {0, 0};
#endif
}

/// Admit complete C0 payload before optional C3; exclude the shared C3 once in aggregate sums.
std::size_t PsiFormerWF::prepareResidentEnergySR(std::size_t population,
    std::pair<std::size_t, std::size_t> budget) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  using Owner = psiformer::device::PreparedDevicePcg;
  const char* reuse_control = std::getenv("PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE");
  const bool reuse = reuse_control && std::string(reuse_control) == "on";
  if (reuse_control && !reuse && std::string(reuse_control) != "off")
    throw std::invalid_argument("PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE must be on or off");
  std::shared_ptr<PsiFormerDerivativeScratch> scratch;
  std::size_t incremental = 0, c3 = 0;
  {
    PsiFormerReadTransaction state_lock(*model_state_);
    const auto& environment = model_state_->execution_plan.environment();
    if (!model_state_->numerical_tile_backend || model_state_->precision != psiformer::PsiFormerPrecisionPolicy::FP64_REFERENCE ||
        environment.amplitude_scalar_domain != psiformer::ScalarDomain::REAL || environment.boundary != psiformer::BoundaryCondition::OPEN ||
        std::any_of(environment.periodic_axes.begin(), environment.periodic_axes.end(), [](bool axis) { return axis; }))
      throw std::invalid_argument("Resident ENERGY SR requires ordinary real/open/FP64 GPU products");
    std::size_t frozen = 0;
    for (const auto& block : structured_parameter_schema_->blocks())
      if (!block.trainable)
        ++frozen;
    const auto p = structured_parameter_schema_->parameterCount();
    const auto cost = Owner::energyRequiredBytes(p, population, model_state_->execution_plan.modelShape().electrons(), 3, frozen);
    if (!budget.first || !budget.second || budget.first == std::numeric_limits<std::size_t>::max() ||
        budget.second == std::numeric_limits<std::size_t>::max() || cost.device_bytes > budget.first || cost.host_bytes > budget.second)
      throw std::length_error("Resident ENERGY SR complete numeric payload exceeds explicit device/host caps");
    c3 = psiformer::device::PreparedDeviceScoreContraction::requiredBytes(p, 3);
    const auto shared = checkedBatchMemoryAdd(c3, c3 - 48 * p, "Resident ENERGY SR shared C3");
    incremental = checkedBatchMemoryAdd(cost.device_bytes, cost.host_bytes, "Resident ENERGY SR payload") - shared;
    scratch = acquireDerivativeScratch(*model_state_, p, 4);
    app_log() << "Resident ENERGY SR admitted device_bytes=" << cost.device_bytes << " host_bytes=" << cost.host_bytes
              << " shared_c3_bytes=" << shared << " incremental_peak_bytes=" << incremental
              << "; named caps exclude existing model/tapes/compatibility/source/coefficients/sampling/vendor/checkpoints" << std::endl;
  }
  PsiFormerDerivativeLease lease(*scratch); // Never wait execution while holding Model read.
  PsiFormerReadTransaction state_lock(*model_state_);
  model_state_->resident_energy_sr_workspace.reset(); // Explicit stage boundary, never runtime retry.
  model_state_->resident_energy_sr_workspace_started = false;
  model_state_->resident_energy_sr_reuse = reuse;
  if (reuse)
    app_log() << "Resident ENERGY SR reuse=on: persistent workspace overlaps successor image publication; aggregate peak may increase beyond named caps" << std::endl;
  scratch->device_products->prepareScoreContractions(3, model_state_->model.p.version(), c3);
  std::lock_guard factory_lock(model_state_->derivative_scratch_factory_mutex);
  model_state_->resident_energy_scratch = std::move(scratch);
  return incremental;
#else
  throw std::invalid_argument("Resident ENERGY SR requires a real GPU build");
#endif
}

/// End-run cleanup waits execution without Model read, after all iteration leases unwind.
void PsiFormerWF::releaseResidentEnergySRWorkspace() const noexcept
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  try
  {
    std::shared_ptr<PsiFormerDerivativeScratch> scratch;
    {
      std::lock_guard factory_lock(model_state_->derivative_scratch_factory_mutex);
      scratch = model_state_->resident_energy_scratch;
    }
    if (!scratch)
      return;
    std::lock_guard execution_lock(scratch->execution_mutex);
    if (!model_state_->resident_energy_sr_workspace)
      return;
    const auto before = scratch->device_products->transferSnapshot();
    model_state_->resident_energy_sr_workspace.reset();
    const auto after = scratch->device_products->transferSnapshot();
    app_log() << "Resident ENERGY SR workspace release cached_owner=1 free_calls=" << after.free_calls - before.free_calls
              << " allocation_calls=" << after.allocation_calls - before.allocation_calls << std::endl;
  }
  catch (...)
  {
    std::fputs("Resident ENERGY SR workspace release_failed\n", stderr);
  }
#endif
}

/// Bind sampled positions/weights to the qualified ENERGY owner, under producer leases.
wftrain::ResidentEnergySRDiagnostics PsiFormerWF::consumeResidentEnergySR(
    const wftrain::StreamingDerivativeOperator& derivative,
    wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels,
    const std::vector<double>& weights, const std::vector<double>& parameters, double weight, double energy,
    double damping, double learning_rate, const wftrain::KrylovSolverControl& control,
    std::pair<std::size_t, std::size_t> budget, std::vector<double>& candidate) const
{
#if defined(ENABLE_CUDA) && !defined(QMC_COMPLEX)
  const auto* source = dynamic_cast<const PsiFormerStreamingDerivativeOperator*>(&derivative);
  if (!source || source->model_state_.get() != model_state_.get() || source->shared_scratch_ != model_state_->resident_energy_scratch ||
      derivative.sampleCount() < 7 || weights.size() != derivative.sampleCount())
    throw std::invalid_argument("Resident ENERGY SR requires this provider's current immutable physical batch");
  (void)source->validateVJPRequest(channels, wftrain::DerivativeAdjoint::TRANSPOSE);
  if (channels.size() != 3 || channels[0].product != wftrain::DerivativeProduct::LOCAL_ENERGY_VJP ||
      channels[0].local_energy_term_mask != wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC) ||
      channels[1].product != wftrain::DerivativeProduct::SCORE_VJP || channels[2].product != wftrain::DerivativeProduct::SCORE_VJP)
    throw std::invalid_argument("Resident ENERGY SR requires ordered physical T/EO/O channels");
  std::vector<psiformer::device::PreparedDevicePcg::FrozenInterval> frozen;
  for (const auto& block : structured_parameter_schema_->blocks())
    if (!block.trainable)
      frozen.emplace_back(block.offset, block.count);
  wftrain::ResidentEnergySRDiagnostics diagnostics;
  source->evaluateVJPsImpl(channels, wftrain::DerivativeAdjoint::TRANSPOSE, nullptr, [&](std::size_t generation) {
    auto& session = *source->shared_scratch_->device_products;
    const auto cold = session.transferSnapshot();
    using Owner = psiformer::device::PreparedDevicePcg;
    const bool reuse = model_state_->resident_energy_sr_reuse;
    const bool prepared = !reuse || !model_state_->resident_energy_sr_workspace;
    std::unique_ptr<Owner> ordinary;
    if (!reuse)
      ordinary = Owner::prepareEnergy(session, source->positions_, weights, frozen,
          derivative.parameterVersion(), damping, budget.first, budget.second);
    else if (!model_state_->resident_energy_sr_workspace)
    {
      if (model_state_->resident_energy_sr_workspace_started)
        throw std::logic_error("Retained ENERGY SR cold preparation failed; explicit stage preparation required");
      model_state_->resident_energy_sr_workspace_started = true;
      model_state_->resident_energy_sr_workspace = Owner::prepareRetainedEnergy(session, source->positions_, weights, frozen,
          derivative.parameterVersion(), damping, budget.first, budget.second);
    }
    auto* owner = reuse ? model_state_->resident_energy_sr_workspace.get() : ordinary.get();
    const auto before = session.transferSnapshot();
    if (reuse)
      app_log() << "Resident ENERGY SR workspace cold_preparation=" << static_cast<int>(prepared)
                << " storage_reused=" << static_cast<int>(!prepared) << " ";
    else
      app_log() << "Resident ENERGY SR cold owner ";
    app_log() << "device_bytes=" << owner->cost().device_bytes
              << " host_bytes=" << owner->cost().host_bytes << " P=" << parameters.size()
              << " B=" << weights.size() << " frozen_intervals=" << frozen.size()
              << " uploaded_bytes=" << before.uploaded_bytes - cold.uploaded_bytes
              << " downloaded_bytes=" << before.downloaded_bytes - cold.downloaded_bytes
              << " allocation_calls=" << before.allocation_calls - cold.allocation_calls << std::endl;
    const auto result = reuse ? owner->consumeRetainedEnergy(source->positions_, weights, frozen,
        derivative.parameterVersion(), generation, parameters, weight, energy, damping, learning_rate, control,
        budget.first, budget.second, candidate) : owner->consumeEnergy(derivative.parameterVersion(), generation,
        parameters, weight, energy, learning_rate, control, candidate);
    const auto after = session.transferSnapshot();
    app_log() << "Resident ENERGY SR terminal uploaded_bytes=" << after.uploaded_bytes - before.uploaded_bytes
              << " downloaded_bytes=" << after.downloaded_bytes - before.downloaded_bytes
              << " allocation_calls=" << after.allocation_calls - before.allocation_calls
              << " free_calls=" << after.free_calls - before.free_calls
              << "; scope=device_wide cold_owner_producer_checkpoint_excluded=yes" << std::endl;
    if (!result.candidate_ready || !result.solve.converged)
      throw std::runtime_error(std::string("Resident ENERGY SR did not converge: ") + wftrain::krylovStopReasonName(result.solve.stop_reason));
    diagnostics = {result.solve, result.gradient_norm, result.direction_norm, result.metric_norm};
  });
  return diagnostics;
#else
  throw std::invalid_argument("Resident ENERGY SR requires a real GPU build");
#endif
}

/// Expose only the resolved budget; CPU never reads GPU optimizer controls.
std::size_t PsiFormerWF::deviceAdamBudget() const
{
#ifdef ENABLE_CUDA
  const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
      model_state_->numerical_tile_backend.get());
  if (backend && backend->adamBudget())
  {
#ifdef QMC_COMPLEX
    throw std::invalid_argument("Explicit device Adam currently requires real parameters/build");
#else
    return backend->adamBudget();
#endif
  }
#endif
  return 0;
}

/// Query only the resolved GPU profile; CPU never reads GPU sampler controls.
bool PsiFormerWF::deviceSamplingEnabled() const
{
#ifdef ENABLE_CUDA
  const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
      model_state_->numerical_tile_backend.get());
  if (backend && backend->metropolisBudget())
  {
    if (batch_execution_plan_)
      throw std::invalid_argument("Explicit device sampling does not support a hard batch memory plan");
#ifdef QMC_COMPLEX
    throw std::invalid_argument("Explicit device sampling currently requires a real build");
#else
    return true;
#endif
  }
#endif
  return false;
}

/// Report only the resolved GPU drift profile; CPU never reads GPU environment controls.
bool PsiFormerWF::deviceSamplingUsesDrift() const
{
#ifdef ENABLE_CUDA
  const auto* backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend*>(
      model_state_->numerical_tile_backend.get());
  return backend && backend->metropolisDriftBudget();
#else
  return false;
#endif
}

/// Pass the actual driver drift flag and configured native callback into the leased bridge.
std::pair<std::size_t, std::size_t> PsiFormerWF::mw_advanceDeviceSampling(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list, double tau, std::size_t substeps,
    const std::function<void(std::vector<double>&)>& gaussian_draw,
    const std::function<double()>& uniform_draw, bool use_drift,
    const std::function<void(double, const double*, double*)>& drift,
    bool retain_device_coordinates) const
{
#ifdef ENABLE_CUDA
  if (!deviceSamplingEnabled() || wfc_list.size() != p_list.size() || wfc_list.empty() ||
      use_drift != deviceSamplingUsesDrift() || (use_drift && !drift))
    throw std::invalid_argument("Device sampling needs an enabled, nonempty homogeneous crowd");
  auto& resource = requireMultiWalkerResource(wfc_list);
  PsiFormerReadTransaction transaction(*model_state_, model_state_->resident_device_authority_enabled);
  if (retain_device_coordinates && (!model_state_->resident_device_authority_enabled ||
      !model_state_->resident_energy_scratch || system_kind_ != "all_electron" || use_drift))
    throw std::invalid_argument("Resident coordinates require admitted sole real/open AE DEVICE authority without drift");
  const auto ne = transaction.state().execution_plan.modelShape().electrons();
  const auto extent = psiformer::spatial_detail::checkedProduct(wfc_list.size(),
      psiformer::spatial_detail::checkedProduct(3, ne, "Sampling physical stride overflow"),
      "Sampling physical population overflow");
  std::vector<double> positions(extent), masses(ne), accepted(extent);
  for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
  {
    const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
    component.requireNoSelectedParticleProposal("device sampling");
    if (component.model_state_.get() != model_state_.get() || component.batch_execution_plan_ ||
        p_list[walker].getTotalNum() != ne || p_list[walker].isSpinor() || p_list[walker].getActivePtcl() != -1)
      throw std::invalid_argument("Device sampling received incompatible crowd components or particles");
    for (std::size_t electron = 0; electron < ne; ++electron)
    {
      const double inverse_mass = 1.0 / p_list[walker].get_mass_by_group()[p_list[walker].getGroupID(electron)];
      if (walker && masses[electron] != inverse_mass)
        throw std::invalid_argument("Device sampling crowd physical masses disagree");
      masses[electron] = inverse_mass;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        positions[(walker * ne + electron) * 3 + dimension] = p_list[walker].R[electron][dimension];
    }
  }
  auto& batch = *resource.batch_workspace;
  batch.resize(pf::DirectBatchMode::FULL_VGL, wfc_list.size());
  const auto& backend = dynamic_cast<const psiformer::device::PreparedDeviceBatchBackend&>(
      *transaction.state().numerical_tile_backend);
  const auto counts = backend.advanceMetropolis(batch.numericalSamplingWorkspace(), positions, masses, tau,
      substeps, transaction.parameterVersion(), gaussian_draw, uniform_draw, accepted, {}, use_drift, drift,
      retain_device_coordinates);
  if (retain_device_coordinates)
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
      // Sampler DEVICE coordinates stay authoritative; native FULL is dirty until the cold fence.
      wfc_list.getCastedElement<PsiFormerWF>(walker).accepted_value_valid_ = false;
  // All device work succeeded. Native refresh is immediate on the default path;
  // the retained path keeps native caches dirty until its explicit cold fence.
  for (std::size_t walker = 0; !retain_device_coordinates && walker < wfc_list.size(); ++walker)
    for (std::size_t electron = 0; electron < ne; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        p_list[walker].R[electron][dimension] = accepted[(walker * ne + electron) * 3 + dimension];
  return counts;
#else
  throw std::logic_error("Device sampling is not compiled");
#endif
}

// Evaluate all accepted crowd configurations under one parameter read lock and
// scatter complete spatial outputs directly into caller-owned accumulators.
void PsiFormerWF::mw_evaluateLog(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVector<ParticleSet::ParticleGradient>& gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& laplacian_list) const
{
  NativeTimerScope timer(phase_timers_.batch_vgl);
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
    PsiFormerReadTransaction transaction(*leader.model_state_, leader.model_state_->resident_device_authority_enabled);
    const std::size_t parameter_version = transaction.parameterVersion();
    const std::size_t electrons =
        transaction.state().execution_plan.modelShape().electrons();
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      wfc_list.getCastedElement<PsiFormerWF>(walker).requireNoSelectedParticleProposal(
          "mw_evaluateLog");
      if (transaction.state().cpu_target_vgl.targets && component.has_proposal_)
        throw std::logic_error("PsiFormer CPU target FULL cannot overwrite a pending crowd proposal");
      component.synchronizeParameterVersion(parameter_version);
      if (gradient_list[walker].get().size() != electrons ||
          laplacian_list[walker].get().size() != electrons)
        throw std::invalid_argument("PsiFormer mw_evaluateLog output arrays have the wrong size");
      component.resizeAcceptedSpatialStorage(electrons);
    }

    if (transaction.state().cpu_target_vgl.targets)
    {
      // The resource batch remains free of FULL tapes; every clone owns one target tape.
      const auto owner_bytes = pf::checkedStorageProduct(wfc_list.size(),
          transaction.state().cpu_target_vgl.owner_bytes, "PsiFormer CPU target crowd byte extent overflow");
      if (!resource.cpu_target_vgl_reported)
      {
        app_log() << "    CPU target VGL crowd: live_clones=" << wfc_list.size()
                  << ", W*target_owner_bytes=" << owner_bytes
                  << "; owner cap is per clone, not an aggregate feasibility claim"
                  << "; typed G/L outputs and other lazy owners are excluded" << std::endl;
        resource.cpu_target_vgl_reported = true;
      }
      // Exclude shared output lanes before dry-running additive publication.
      for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
        for (std::size_t prior = 0; prior < lane; ++prior)
        {
          const auto g = checkedMemoryRange(gradient_list[lane].get().data(), electrons,
                                             "PsiFormer CPU target crowd range overflow");
          const auto l = checkedMemoryRange(laplacian_list[lane].get().data(), electrons,
                                             "PsiFormer CPU target crowd range overflow");
          const auto pg = checkedMemoryRange(gradient_list[prior].get().data(), electrons,
                                              "PsiFormer CPU target crowd range overflow");
          const auto pl = checkedMemoryRange(laplacian_list[prior].get().data(), electrons,
                                              "PsiFormer CPU target crowd range overflow");
          if (memoryRangesOverlap(g, pg) || memoryRangesOverlap(g, pl) ||
              memoryRangesOverlap(l, pg) || memoryRangesOverlap(l, pl))
            throw std::invalid_argument("PsiFormer CPU target crowd outputs overlap between walkers");
        }
      for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
        wfc_list.getCastedElement<PsiFormerWF>(lane).stageCpuTargetFullVGL(transaction, p_list[lane]);
      for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
        wfc_list.getCastedElement<PsiFormerWF>(lane).validateCpuTargetFullVGLPublication(
            gradient_list[lane].get(), laplacian_list[lane].get());
      for (std::size_t lane = 0; lane < wfc_list.size(); ++lane)
        wfc_list.getCastedElement<PsiFormerWF>(lane).publishCpuTargetFullVGL(
            gradient_list[lane].get(), laplacian_list[lane].get());
      return;
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
        component.log_value_ = makePhysicalLogValue(
            transaction.state(), result.sign, result.logabs,
            twistPhaseAngle(transaction.state(), p_list[walker]));
        for (std::size_t electron = 0; electron < electrons; ++electron)
        {
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
            component.accepted_gradient_[electron][dimension] =
                makePhysicalGradient(transaction.state(),
                                     result.gradient[3 * electron + dimension],
                                     dimension);
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
      component.log_value_ = makePhysicalLogValue(
          transaction.state(), result.sign[walker], result.logabs[walker],
          twistPhaseAngle(transaction.state(), p_list[walker]));
      auto& gradient = gradient_list[walker].get();
      auto& laplacian = laplacian_list[walker].get();
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          component.accepted_gradient_[electron][dimension] =
              makePhysicalGradient(
                  transaction.state(),
                  result.gradient[walker * result.gradient_stride +
                                  3 * electron + dimension],
                  dimension);
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
            makePhysicalGradient(transaction.state(),
                result.gradient[lane * result.gradient_stride +
                                3 * electron + dimension], dimension);
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
              makePhysicalGradient(transaction.state(),
                  result.gradient[lane * result.gradient_stride +
                                  3 * electron + dimension], dimension);
        component.accepted_laplacian_[electron] = static_cast<ValueType>(
            result.lap_log[lane * result.laplacian_stride + electron]);
      }
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      for (std::size_t electron = 0; electron < electrons; ++electron)
      {
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_list[lane].get()[electron][dimension] +=
              makePhysicalGradient(transaction.state(),
                  result.gradient[lane * result.gradient_stride +
                                  3 * electron + dimension], dimension);
        laplacian_list[lane].get()[electron] += static_cast<ValueType>(
            result.lap_log[lane * result.laplacian_stride + electron]);
      }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      component.current_sign_ = resource.staged_signs[lane];
      component.log_value_ = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane],
          twistPhaseAngle(transaction.state(), p_list[lane]));
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

// Advertise PBYP-only CPU target inference before the driver draws selected moves.
bool PsiFormerWF::supportsMultiParticleMoves() const noexcept
{ return model_state_->cpu_target_vgl.targets == 0; }

// Evaluate complete selected-electron proposals without modifying accepted ParticleSets.
void PsiFormerWF::mw_evaluateMultiParticleMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const MCMultiParticleMoves<CoordsType::POS>& moves,
    std::vector<LogValue>& log_ratios,
    const RefVector<ParticleSet::ParticleGradient>& proposed_gradient_list,
    const RefVector<ParticleSet::ParticleLaplacian>& proposed_laplacian_list) const
{
  NativeTimerScope timer(phase_timers_.batch_vgl);
  if (model_state_->cpu_target_vgl.targets)
    throw std::logic_error("PsiFormer CPU target VGL supports PBYP inference, not selected-electron proposals");
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
          makePhysicalLogValue(
              transaction.state(), proposed_signs[walker],
              proposed_logabs[walker],
              twistPhaseAngle(transaction.state(), p_list[walker],
                              moves.slice(walker))) - component.log_value_;
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
                  makePhysicalGradient(
                      transaction.state(),
                      batch_result.gradient[slot * batch_result.gradient_stride +
                                            3 * electron + dimension],
                      dimension);
            component.proposed_laplacian_[electron] =
                batch_result.lap_log[slot * batch_result.laplacian_stride + electron];
          }
          else
          {
            const pf::Result& oracle = oracle_results[slot];
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
              component.proposed_gradient_[electron][dimension] =
                  makePhysicalGradient(
                      transaction.state(),
                      oracle.gradient[3 * electron + dimension], dimension);
            component.proposed_laplacian_[electron] = oracle.lap_log[electron];
          }
        }
      }

      component.proposed_sign_                   = proposed_signs[walker];
      component.proposed_log_value_              =
          makePhysicalLogValue(
              transaction.state(), proposed_signs[walker],
              proposed_logabs[walker],
              twistPhaseAngle(transaction.state(), p_list[walker],
                              moves.slice(walker)));
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
      resource.staged_log_ratios[lane] = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane],
          twistPhaseAngle(transaction.state(), p_list[lane],
                          moves.slice(lane))) - component.log_value_;
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
              makePhysicalLogValue(
                  transaction.state(), result.sign[slot], result.logabs[slot],
                  twistPhaseAngle(transaction.state(), p_list[lane],
                                  moves.slice(lane))) -
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
            : makePhysicalGradient(transaction.state(),
                  result.gradient[slot * result.gradient_stride +
                                  3 * electron + dimension], dimension);
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
              : makePhysicalGradient(transaction.state(),
                    result.gradient[slot * result.gradient_stride +
                                    3 * electron + dimension], dimension);
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
      component.proposed_log_value_ = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane],
          twistPhaseAngle(transaction.state(), p_list[lane], moves.slice(lane)));
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
  if (model_state_->cpu_target_vgl.targets)
    throw std::logic_error("PsiFormer CPU target VGL does not support selected-electron resolution");
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
          component.proposed_log_value_ != makePhysicalLogValue(
              *component.model_state_, access.resource.staged_signs[lane],
              access.resource.staged_log_magnitudes[lane],
              twistPhaseAngle(*component.model_state_, p_list[lane],
                              moves.slice(lane))))
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
      const std::size_t walker = resource.walker_indices[selected];
      component.log_value_ = makePhysicalLogValue(
          transaction.state(), staged_sign[selected], staged_logabs[selected],
          twistPhaseAngle(transaction.state(), p_list[walker]));
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

  const auto has_finite_full_state = [&](const PsiFormerWF& component,
                                         const ParticleSet& particles) noexcept {
    if (component.accepted_state_requirement_ !=
            AcceptedStateRequirement::FULL_SPATIAL ||
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, particles))
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
        has_finite_full_state(component, p_list[lane]);
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
          has_finite_full_state(component, p_list[lane]);
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
      const std::size_t lane = resource.walker_indices[slot];
      component.log_value_ = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[slot],
          resource.staged_log_magnitudes[slot],
          twistPhaseAngle(transaction.state(), p_list[lane]));
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

// Validate planned group preparation as a pure no-op after exact preflight.
void PsiFormerWF::prepareGroup(ParticleSet& particles, int group_index)
{
  if (!batch_execution_plan_)
  {
    WaveFunctionComponent::prepareGroup(particles, group_index);
    return;
  }
  requirePlannedScalarLifecycleOperation(
      PlannedOperation::PREPARE_GROUP, particles, group_index);
}

// Validate one acquired planned team without inherited serialized dispatch.
void PsiFormerWF::mw_prepareGroup(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int group_index) const
{
  if (!batch_execution_plan_)
  {
    WaveFunctionComponent::mw_prepareGroup(wfc_list, p_list, group_index);
    return;
  }
  requirePlannedMultiWalkerLifecycleOperation(
      wfc_list, std::addressof(p_list), PlannedOperation::PREPARE_GROUP,
      group_index);
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

    // Range-check the public result before proposal publication.  A caller
    // receiving an exception must not be able to commit an unseen proposal.
    const LogValue proposed_log = makePhysicalLogValue(
        transaction.state(), sign, logabs,
        twistPhaseAngle(transaction.state(), p, iat));
    const double phase_change = twistDisplacementPhase(
        transaction.state(), p.R[iat], p.activeR(iat));
    const PsiValue ratio_value = makePhysicalRatio(
        transaction.state(), sign, logabs, current_sign_,
        std::real(log_value_), phase_change);
    if (!isFiniteWavefunctionValue(ratio_value))
      throw std::runtime_error("PsiFormer single-particle ratio is non-finite");

    // Cache proposal state so acceptMove can commit it without reevaluating the
    // network.
    cacheSingleParticleProposal(
        sign, proposed_log, configurationIdentity(p, iat), iat,
        transaction.parameterVersion(), ProposalOrigin::SCALAR_RATIO_VALUE);
    return ratio_value;
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
  NativeTimerScope timer(phase_timers_.batch_value);
  // Preserve the historical lazy/oracle implementation behind the explicit
  // no-policy branch.  Planned execution below never enters an allocating or
  // serialized fallback after its runtime contract has been selected.
  if (!batch_execution_plan_)
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
      const double phase_change = twistDisplacementPhase(
          transaction.state(), p_list[walker].R[particle_index],
          p_list[walker].activeR(particle_index));
      staged_ratios[walker] = makePhysicalRatio(
          transaction.state(), staged_sign[walker], staged_logabs[walker],
          component.current_sign_, std::real(component.log_value_), phase_change);
    }

    ratios.swap(staged_ratios);
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const LogValue physical_log = makePhysicalLogValue(
          transaction.state(), staged_sign[walker], staged_logabs[walker],
          twistPhaseAngle(transaction.state(), p_list[walker], particle_index));
      component.cacheSingleParticleProposal(
          staged_sign[walker], physical_log,
          staged_configuration[walker], particle_index, parameter_version,
          ProposalOrigin::MW_CALC_RATIO_VALUE);
    }
    return;
  }

  if (particle_index < 0)
    throw std::out_of_range(
        "PsiFormer planned CALC_RATIO has a negative active electron");

  const std::size_t walker_count = wfc_list.size();
  PlannedRuntimeRequest request;
  request.operation            = PlannedOperation::CALC_RATIO;
  request.live_walkers         = walker_count;
  request.dense_configurations = walker_count;
  request.active_electron      = static_cast<std::size_t>(particle_index);
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);

  if (ratios.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned CALC_RATIO output size does not match the crowd");
  PsiValue* const output_data       = ratios.data();
  const std::size_t output_capacity = ratios.capacity();
  const CheckedMemoryRange output_range = checkedMemoryRange(
      output_data, walker_count,
      "PsiFormer planned CALC_RATIO output range overflowed");
  const std::size_t output_bytes = output_range.end - output_range.begin;

  PsiFormerMultiWalkerResource& resource = access.resource;
  auto ratio_arena = resource.requireCalcRatioStaging(walker_count);
  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;
  const std::size_t electron_count = access.participant.plan().particleCount();

  // Caller storage must remain completely disjoint from every state or scratch
  // range read or written by the transaction.  This check is repeated in
  // Phase B after native evaluation and before publication.
  const auto require_output_nonoverlap = [&]() {
    if (resource.overlapsStagingStorage(output_data, output_bytes) ||
        batch.overlapsStorage(output_data, output_bytes))
      throw std::invalid_argument(
          "PsiFormer planned CALC_RATIO output aliases prepared scratch");

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
                   "PsiFormer proposed Laplacian range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.current_sign_), std::size_t{1},
                   "PsiFormer accepted sign range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.log_value_), std::size_t{1},
                   "PsiFormer accepted log-value range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.accepted_value_valid_),
                   std::size_t{1},
                   "PsiFormer accepted-validity range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.accepted_configuration_identity_),
                   std::size_t{1},
                   "PsiFormer accepted-configuration range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.accepted_parameter_version_),
                   std::size_t{1},
                   "PsiFormer accepted-version range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.accepted_state_requirement_),
                   std::size_t{1},
                   "PsiFormer accepted-requirement range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.observed_parameter_version_),
                   std::size_t{1},
                   "PsiFormer observed-version range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_sign_), std::size_t{1},
                   "PsiFormer proposed-sign range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_log_value_),
                   std::size_t{1},
                   "PsiFormer proposed-log-value range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_configuration_identity_),
                   std::size_t{1},
                   "PsiFormer proposed-configuration range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_descriptor_fingerprint_),
                   std::size_t{1},
                   "PsiFormer proposed-fingerprint range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_parameter_version_),
                   std::size_t{1},
                   "PsiFormer proposed-version range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposed_particle_),
                   std::size_t{1},
                   "PsiFormer proposed-particle range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.proposal_origin_),
                   std::size_t{1},
                   "PsiFormer proposal-origin range overflowed"),
               checkedMemoryRange(
                   std::addressof(component.has_proposal_), std::size_t{1},
                   "PsiFormer proposal-marker range overflowed")})
        if (memoryRangesOverlap(output_range, internal))
          throw std::invalid_argument(
              "PsiFormer planned CALC_RATIO output aliases component state");

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
                   "PsiFormer ParticleSet Laplacian range overflowed"),
               checkedMemoryRange(
                   particles.GroupID.data(), electron_count,
                   "PsiFormer ParticleSet group range overflowed"),
               checkedMemoryRange(
                   particles.spins.data(), electron_count,
                   "PsiFormer ParticleSet spin range overflowed")})
        if (memoryRangesOverlap(output_range, particle_storage))
          throw std::invalid_argument(
              "PsiFormer planned CALC_RATIO output aliases ParticleSet state");
    }
  };
  require_output_nonoverlap();

  // Phase A proves every accepted value and base-configuration key without
  // synchronizing a stale clone or modifying any caller-visible state.
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
            AcceptedStateRequirement::VALUE_ONLY))
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO requires current accepted value state");
    if (!valid_requirement ||
        !isCoherentPhysicalValue(*component.model_state_, component.current_sign_,
                                 component.log_value_, p_list[lane]))
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO has invalid accepted value state");
  }
  // Record the proposed-coordinate keys only after every accepted lane has
  // passed Phase A; this prepared scratch is not externally visible state.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    resource.configuration_identities[lane] =
        configurationIdentity(p_list[lane], particle_index);

  // One authoritative model read transaction covers accepted-state validation,
  // native evaluation, numerical staging, final revalidation, and publication.
  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO requires the direct value backend");
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
        !isCoherentPhysicalValue(*component.model_state_, component.current_sign_,
                                 component.log_value_, p_list[lane]))
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO has stale accepted parameters");
  }

  // Capacity preparation has already fixed every retained allocation.  This
  // logical activation is therefore bounded and cannot grow the workspace.
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
    packBatchConfiguration(batch, lane, p_list[lane], particle_index);

  const pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(batch);
  if (!batch.ownsValueResult(
          result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
          walker_count))
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO result is not the exact workspace-owned view");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    if (result.parameter_version[lane] != parameter_version)
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO observed inconsistent parameters");
    if ((result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]))
      throw std::runtime_error(
          "PsiFormer planned CALC_RATIO produced an invalid value");

    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const double phase_change = twistDisplacementPhase(
        transaction.state(), p_list[lane].R[particle_index],
        p_list[lane].activeR(particle_index));
    const PsiValue ratio = makePhysicalRatio(
        transaction.state(), result.sign[lane], result.logabs[lane],
        component.current_sign_, std::real(component.log_value_), phase_change);
    if (!isFiniteWavefunctionValue(ratio))
      throw std::overflow_error(
          "PsiFormer planned CALC_RATIO ratio conversion is non-finite");

    resource.staged_signs[lane]          = result.sign[lane];
    resource.staged_log_magnitudes[lane] = result.logabs[lane];
    ratio_arena.store(lane, ratio);
    if (ratio_arena.load(lane) != ratio)
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO ratio arena failed an exact typed round trip");
  }

  const std::uint64_t transaction_fingerprint = singleTransactionFingerprint(
      wfc_list, p_list, ProposalOrigin::MW_CALC_RATIO_VALUE,
      static_cast<std::size_t>(particle_index), parameter_version);
  if (transaction_fingerprint == 0)
    throw std::runtime_error(
        "PsiFormer planned CALC_RATIO produced a zero transaction fingerprint");

  // Phase B repeats every externally relevant identity while the authoritative
  // read lock remains held.  All work after registration is mechanically
  // nonthrowing and publishes the pending marker only after complete metadata.
  if (!batch.ownsValueResult(
          result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
          walker_count) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO storage changed during evaluation");
  const auto final_ratio_arena =
      resource.requireCalcRatioStaging(walker_count);
  if (final_ratio_arena.kind != ratio_arena.kind ||
      final_ratio_arena.psi_value_data != ratio_arena.psi_value_data ||
      final_ratio_arena.log_value_data != ratio_arena.log_value_data ||
      final_ratio_arena.extent != ratio_arena.extent)
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO typed ratio arena changed during evaluation");

  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO runtime evidence changed during evaluation");
  if (ratios.size() != walker_count || ratios.data() != output_data ||
      ratios.capacity() != output_capacity)
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO caller output changed during evaluation");
  require_output_nonoverlap();

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    const std::uint64_t proposed_configuration =
        configurationIdentity(p_list[lane], particle_index);
    const double phase_change = twistDisplacementPhase(
        transaction.state(), p_list[lane].R[particle_index],
        p_list[lane].activeR(particle_index));
    const PsiValue expected_ratio = makePhysicalRatio(
        transaction.state(), result.sign[lane], result.logabs[lane],
        component.current_sign_, std::real(component.log_value_), phase_change);
    const PsiValue staged_ratio = final_ratio_arena.load(lane);
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
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, p_list[lane]) ||
        resource.configuration_identities[lane] != proposed_configuration ||
        result.parameter_version[lane] != parameter_version ||
        (result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]) ||
        resource.staged_signs[lane] != result.sign[lane] ||
        resource.staged_log_magnitudes[lane] != result.logabs[lane] ||
        !isFiniteWavefunctionValue(expected_ratio) ||
        !isFiniteWavefunctionValue(staged_ratio) ||
        staged_ratio != expected_ratio)
      throw std::logic_error(
          "PsiFormer planned CALC_RATIO lane evidence changed during evaluation");
  }
  if (singleTransactionFingerprint(
          wfc_list, p_list, ProposalOrigin::MW_CALC_RATIO_VALUE,
          static_cast<std::size_t>(particle_index), parameter_version) !=
      transaction_fingerprint)
    throw std::logic_error(
        "PsiFormer planned CALC_RATIO transaction identity changed during evaluation");

  if (fail_planned_single_proposal_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned single-particle proposal failure");
  if (!tryRegisterPlannedSingleTransaction())
    throw std::overflow_error(
        "PsiFormer planned single-particle transaction count overflowed");

  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      ratios[lane] = final_ratio_arena.loadUnchecked(lane);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      component.proposed_sign_ = resource.staged_signs[lane];
      component.proposed_log_value_ = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane],
          twistPhaseAngle(transaction.state(), p_list[lane], particle_index));
      component.proposed_configuration_identity_ =
          resource.configuration_identities[lane];
      component.proposed_descriptor_fingerprint_ = transaction_fingerprint;
      component.proposed_parameter_version_      = parameter_version;
      component.proposed_particle_               = particle_index;
      component.proposal_origin_ = ProposalOrigin::MW_CALC_RATIO_VALUE;
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = true;
  };
  static_assert(noexcept(publish()));
  publish();
}

// Return one accepted electron logarithmic gradient.
PsiFormerWF::GradType PsiFormerWF::evalGrad(ParticleSet& p, int iat)
{
  requireUnplannedScalarEvaluation("active-electron gradient");
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());
  auto scatter = [&](const auto& source) {
    GradType gradient;
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] = makePhysicalGradient(
          transaction.state(), source[dimension], dimension);
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
  NativeTimerScope timer(phase_timers_.batch_gradient);
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
              makePhysicalGradient(
                  transaction.state(),
                  result.gradient[walker * result.gradient_stride + dimension],
                  dimension);
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
          staged_gradients[walker][dimension] = makePhysicalGradient(
              transaction.state(), result.active_gradient[dimension], dimension);
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
            AcceptedStateRequirement::VALUE_ONLY))
      throw std::logic_error(
          "PsiFormer planned ACTIVE_GRADIENT requires current accepted value state");
    if (!valid_requirement ||
        !isCoherentPhysicalValue(*component.model_state_, component.current_sign_,
                                 component.log_value_, p_list[lane]))
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
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, p_list[lane]))
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
          makePhysicalGradient(transaction.state(), native, dimension);
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
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, p_list[lane]) ||
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
      const ValueType staged =
          makePhysicalGradient(transaction.state(), native, dimension);
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
    const LogValue physical_log = makePhysicalLogValue(
        transaction.state(), sign, logabs,
        twistPhaseAngle(transaction.state(), p, iat));
    cacheSingleParticleProposal(
        sign, physical_log, configurationIdentity(p, iat), iat,
        transaction.parameterVersion(),
        ProposalOrigin::SCALAR_RATIO_GRADIENT_ACTIVE);
    for (int dimension = 0; dimension < 3; ++dimension)
      gradient[dimension] += makePhysicalGradient(
          transaction.state(), active_gradient[dimension], dimension);
    const double phase_change = twistDisplacementPhase(
        transaction.state(), p.R[iat], p.activeR(iat));
    const PsiValue ratio_value = makePhysicalRatio(
        transaction.state(), sign, logabs, current_sign_,
        std::real(log_value_), phase_change);
    if (!isFiniteWavefunctionValue(ratio_value))
      throw std::runtime_error("PsiFormer single-particle ratio-gradient ratio is non-finite");
    return ratio_value;
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

// Evaluate and cache crowd proposal ratios and gradients through the legacy or planned path.
void PsiFormerWF::mw_ratioGrad(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    std::vector<PsiValue>& ratios,
    std::vector<GradType>& gradients) const
{
  NativeTimerScope timer(phase_timers_.batch_gradient);
  // Preserve the historical lazy/oracle implementation behind the explicit
  // no-policy branch.
  if (!batch_execution_plan_)
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
              makePhysicalGradient(
                  transaction.state(),
                  result.gradient[walker * result.gradient_stride + dimension],
                  dimension);
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
          staged_gradients[walker][dimension] = makePhysicalGradient(
              transaction.state(), result.active_gradient[dimension], dimension);
      }

    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      const auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const double phase_change = twistDisplacementPhase(
          transaction.state(), p_list[walker].R[particle_index],
          p_list[walker].activeR(particle_index));
      staged_ratios[walker] = makePhysicalRatio(
          transaction.state(), staged_sign[walker], staged_logabs[walker],
          component.current_sign_, std::real(component.log_value_), phase_change);
    }

    ratios.swap(staged_ratios);
    for (std::size_t walker = 0; walker < wfc_list.size(); ++walker)
    {
      auto& component = wfc_list.getCastedElement<PsiFormerWF>(walker);
      const LogValue physical_log = makePhysicalLogValue(
          transaction.state(), staged_sign[walker], staged_logabs[walker],
          twistPhaseAngle(transaction.state(), p_list[walker], particle_index));
      component.cacheSingleParticleProposal(
          staged_sign[walker], physical_log,
          staged_configuration[walker], particle_index, parameter_version,
          ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE);
      gradients[walker] += staged_gradients[walker];
    }
    return;
  }

  if (particle_index < 0)
    throw std::out_of_range(
        "PsiFormer planned RATIO_GRADIENT has a negative active electron");

  const std::size_t walker_count = wfc_list.size();
  PlannedRuntimeRequest request;
  request.operation            = PlannedOperation::RATIO_GRADIENT;
  request.live_walkers         = walker_count;
  request.dense_configurations = walker_count;
  request.active_electron      = static_cast<std::size_t>(particle_index);
  PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (ratios.size() != walker_count || gradients.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned RATIO_GRADIENT output sizes do not match the crowd");

  PsiValue* const ratio_output_data                = ratios.data();
  GradType* const gradient_output_data             = gradients.data();
  const std::size_t ratio_output_capacity          = ratios.capacity();
  const std::size_t gradient_output_capacity       = gradients.capacity();

  PsiFormerMultiWalkerResource& resource = access.resource;
  PsiFormerMultiWalkerResource::RatioArenaView ratio_arena =
      resource.requireRatioGradientStaging(walker_count);
  pf::DirectBatchWorkspace& batch = *resource.batch_workspace;
  const std::size_t electron_count =
      access.participant.plan().particleCount();

  const CheckedMemoryRange ratio_output_range = checkedMemoryRange(
      ratio_output_data, walker_count,
      "PsiFormer planned RATIO_GRADIENT ratio range overflowed");
  const CheckedMemoryRange gradient_output_range = checkedMemoryRange(
      gradient_output_data, walker_count,
      "PsiFormer planned RATIO_GRADIENT gradient range overflowed");
  const auto require_output_nonoverlap = [&]() {
    if (memoryRangesOverlap(ratio_output_range, gradient_output_range))
      throw std::invalid_argument(
          "PsiFormer planned RATIO_GRADIENT caller outputs overlap");

    for (const CheckedMemoryRange output : {ratio_output_range,
                                            gradient_output_range})
    {
      const void* output_data =
          reinterpret_cast<const void*>(output.begin);
      const std::size_t output_bytes = output.end - output.begin;
      if (resource.overlapsStagingStorage(output_data, output_bytes) ||
          batch.overlapsStorage(output_data, output_bytes))
        throw std::invalid_argument(
            "PsiFormer planned RATIO_GRADIENT output aliases prepared scratch");

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
                     "PsiFormer proposed Laplacian range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.current_sign_), std::size_t{1},
                     "PsiFormer accepted sign range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.log_value_), std::size_t{1},
                     "PsiFormer accepted log-value range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.accepted_value_valid_),
                     std::size_t{1},
                     "PsiFormer accepted-validity range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.accepted_configuration_identity_),
                     std::size_t{1},
                     "PsiFormer accepted-configuration range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.accepted_parameter_version_),
                     std::size_t{1},
                     "PsiFormer accepted-version range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.accepted_state_requirement_),
                     std::size_t{1},
                     "PsiFormer accepted-requirement range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.observed_parameter_version_),
                     std::size_t{1},
                     "PsiFormer observed-version range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_sign_), std::size_t{1},
                     "PsiFormer proposed-sign range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_log_value_),
                     std::size_t{1},
                     "PsiFormer proposed-log-value range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_configuration_identity_),
                     std::size_t{1},
                     "PsiFormer proposed-configuration range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_descriptor_fingerprint_),
                     std::size_t{1},
                     "PsiFormer proposed-fingerprint range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_parameter_version_),
                     std::size_t{1},
                     "PsiFormer proposed-version range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposed_particle_),
                     std::size_t{1},
                     "PsiFormer proposed-particle range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.proposal_origin_),
                     std::size_t{1},
                     "PsiFormer proposal-origin range overflowed"),
                 checkedMemoryRange(
                     std::addressof(component.has_proposal_), std::size_t{1},
                     "PsiFormer proposal-marker range overflowed")})
          if (memoryRangesOverlap(output, internal))
            throw std::invalid_argument(
                "PsiFormer planned RATIO_GRADIENT output aliases component state");

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
                     "PsiFormer ParticleSet Laplacian range overflowed"),
                 checkedMemoryRange(
                     particles.GroupID.data(), electron_count,
                     "PsiFormer ParticleSet group range overflowed"),
                 checkedMemoryRange(
                     particles.spins.data(), electron_count,
                     "PsiFormer ParticleSet spin range overflowed")})
          if (memoryRangesOverlap(output, particle_storage))
            throw std::invalid_argument(
                "PsiFormer planned RATIO_GRADIENT output aliases ParticleSet storage");
      }
    }
  };
  require_output_nonoverlap();

  // Phase A validates every additive seed and accepted-state key before the
  // workspace or prepared staging prefixes are modified.
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (!component.acceptedStateMatches(
            p_list[lane], component.observed_parameter_version_,
            AcceptedStateRequirement::VALUE_ONLY) ||
        (component.accepted_state_requirement_ !=
             AcceptedStateRequirement::VALUE_ONLY &&
         component.accepted_state_requirement_ !=
             AcceptedStateRequirement::FULL_SPATIAL) ||
        !isCoherentPhysicalValue(*component.model_state_, component.current_sign_,
                                 component.log_value_, p_list[lane]))
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT requires current coherent accepted state");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
      if (!isFiniteWavefunctionValue(gradients[lane][dimension]))
        throw std::invalid_argument(
            "PsiFormer planned RATIO_GRADIENT gradient seed is non-finite");
  }

  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_spatial_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT requires the direct spatial backend");
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
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, p_list[lane]))
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT has stale accepted parameters");
    resource.configuration_identities[lane] =
        configurationIdentity(p_list[lane], particle_index);
  }

  batch.resize(pf::DirectBatchMode::ACTIVE_ELECTRON_GRADIENT, walker_count);
  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    resource.active_electrons[lane] =
        static_cast<std::size_t>(particle_index);
    packBatchConfiguration(batch, lane, p_list[lane], particle_index);
  }

  const pf::DirectBatchSpatialResultView result =
      transaction.state().direct_batch_executor.evaluateActive(
          batch, resource.active_electrons.data());
  if (!batch.ownsSpatialResult(
          result, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
          walker_count))
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT result is not the exact workspace-owned view");

  for (std::size_t lane = 0; lane < walker_count; ++lane)
  {
    const auto& component = static_cast<const PsiFormerWF&>(wfc_list[lane]);
    if (result.parameter_version[lane] != parameter_version)
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT observed inconsistent parameters");
    if ((result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]))
      throw std::runtime_error(
          "PsiFormer planned RATIO_GRADIENT produced an invalid value");

    resource.staged_signs[lane]          = result.sign[lane];
    resource.staged_log_magnitudes[lane] = result.logabs[lane];
    const double phase_change = twistDisplacementPhase(
        transaction.state(), p_list[lane].R[particle_index],
        p_list[lane].activeR(particle_index));
    const PsiValue ratio = makePhysicalRatio(
        transaction.state(), result.sign[lane], result.logabs[lane],
        component.current_sign_, std::real(component.log_value_), phase_change);
    if (!isFiniteWavefunctionValue(ratio))
      throw std::runtime_error(
          "PsiFormer planned RATIO_GRADIENT produced a non-finite ratio");
    ratio_arena.store(lane, ratio);
    if (ratio_arena.load(lane) != ratio)
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT ratio arena did not round-trip exactly");

    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double native = force_planned_ratio_gradient_overflow_for_testing_
          ? static_cast<double>(std::numeric_limits<RealType>::max())
          : result.gradient[lane * result.gradient_stride + dimension];
      if (!psiformer::determinant::isFiniteReal(native))
        throw std::runtime_error(
            "PsiFormer planned RATIO_GRADIENT produced a non-finite gradient");
      const ValueType contribution =
          makePhysicalGradient(transaction.state(), native, dimension);
      if (!isFiniteWavefunctionValue(contribution))
        throw std::overflow_error(
            "PsiFormer planned RATIO_GRADIENT gradient conversion is non-finite");
      const ValueType future = gradients[lane][dimension] + contribution;
      if (!isFiniteWavefunctionValue(future))
        throw std::overflow_error(
            "PsiFormer planned RATIO_GRADIENT gradient sum is non-finite");
      resource.staged_gradients[lane][dimension] = future;
    }
  }

  const std::uint64_t transaction_fingerprint = singleTransactionFingerprint(
      wfc_list, p_list, ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE,
      static_cast<std::size_t>(particle_index), parameter_version);
  if (transaction_fingerprint == 0)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT transaction fingerprint is zero");

  // Phase B repeats all borrowed, caller, accepted-state, active-position,
  // workspace-result, and typed-arena evidence before the late failure seam.
  if (!batch.ownsSpatialResult(
          result, pf::DirectSpatialMode::ACTIVE_ELECTRON_GRADIENT,
          walker_count) ||
      resource.currentStorageFingerprint() != access.storage_fingerprint ||
      !resource.hasExactPreparedStagingExtents())
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT storage changed during evaluation");
  const PsiFormerMultiWalkerResource::RatioArenaView final_ratio_arena =
      resource.requireRatioGradientStaging(walker_count);
  if (final_ratio_arena.kind != ratio_arena.kind ||
      final_ratio_arena.psi_value_data != ratio_arena.psi_value_data ||
      final_ratio_arena.log_value_data != ratio_arena.log_value_data ||
      final_ratio_arena.extent != ratio_arena.extent)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT ratio arena changed during evaluation");
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  if (&final_access.resource != &resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT runtime evidence changed during evaluation");
  if (ratios.size() != walker_count || ratios.data() != ratio_output_data ||
      ratios.capacity() != ratio_output_capacity ||
      gradients.size() != walker_count ||
      gradients.data() != gradient_output_data ||
      gradients.capacity() != gradient_output_capacity)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT caller output changed during evaluation");
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
        !isCoherentPhysicalValue(transaction.state(), component.current_sign_,
                                 component.log_value_, p_list[lane]) ||
        resource.active_electrons[lane] !=
            static_cast<std::size_t>(particle_index) ||
        resource.configuration_identities[lane] !=
            configurationIdentity(p_list[lane], particle_index) ||
        result.parameter_version[lane] != parameter_version ||
        (result.sign[lane] != 1.0 && result.sign[lane] != -1.0) ||
        !psiformer::determinant::isFiniteReal(result.logabs[lane]) ||
        resource.staged_signs[lane] != result.sign[lane] ||
        resource.staged_log_magnitudes[lane] != result.logabs[lane])
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT lane evidence changed during evaluation");

    const double phase_change = twistDisplacementPhase(
        transaction.state(), p_list[lane].R[particle_index],
        p_list[lane].activeR(particle_index));
    const PsiValue expected_ratio = makePhysicalRatio(
        transaction.state(), result.sign[lane], result.logabs[lane],
        component.current_sign_, std::real(component.log_value_), phase_change);
    if (!isFiniteWavefunctionValue(expected_ratio) ||
        final_ratio_arena.load(lane) != expected_ratio)
      throw std::logic_error(
          "PsiFormer planned RATIO_GRADIENT ratio staging changed during evaluation");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const double native =
          result.gradient[lane * result.gradient_stride + dimension];
      const ValueType contribution =
          makePhysicalGradient(transaction.state(), native, dimension);
      if (!psiformer::determinant::isFiniteReal(native) ||
          !isFiniteWavefunctionValue(contribution) ||
          !isFiniteWavefunctionValue(gradients[lane][dimension]))
        throw std::logic_error(
            "PsiFormer planned RATIO_GRADIENT gradient evidence changed during evaluation");
      const ValueType future = gradients[lane][dimension] + contribution;
      if (!isFiniteWavefunctionValue(future) ||
          resource.staged_gradients[lane][dimension] != future)
        throw std::logic_error(
            "PsiFormer planned RATIO_GRADIENT final gradient staging changed during evaluation");
    }
  }
  if (singleTransactionFingerprint(
          wfc_list, p_list, ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE,
          static_cast<std::size_t>(particle_index), parameter_version) !=
      transaction_fingerprint)
    throw std::logic_error(
        "PsiFormer planned RATIO_GRADIENT transaction identity changed during evaluation");

  if (fail_planned_single_proposal_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned single-particle pre-publication failure");
  if (!tryRegisterPlannedSingleTransaction())
    throw std::overflow_error(
        "PsiFormer planned single-particle transaction count overflow");

  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      ratios[lane] = final_ratio_arena.loadUnchecked(lane);
      gradients[lane] = resource.staged_gradients[lane];
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
    {
      auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
      component.proposed_sign_ = resource.staged_signs[lane];
      component.proposed_log_value_ = makePhysicalLogValue(
          transaction.state(), resource.staged_signs[lane],
          resource.staged_log_magnitudes[lane],
          twistPhaseAngle(transaction.state(), p_list[lane], particle_index));
      component.proposed_configuration_identity_ =
          resource.configuration_identities[lane];
      component.proposed_descriptor_fingerprint_ = transaction_fingerprint;
      component.proposed_parameter_version_      = parameter_version;
      component.proposed_particle_               = particle_index;
      component.proposal_origin_ =
          ProposalOrigin::MW_RATIO_GRADIENT_ACTIVE;
    }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = true;
  };
  static_assert(noexcept(publish()));
  publish();
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

// Resolve a one-electron crowd transaction through the legacy or planned path.
void PsiFormerWF::mw_accept_rejectMove(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int particle_index,
    const std::vector<bool>& is_accepted,
    bool) const
{
  // Preserve the existing scalar-compatible resolver as the exact no-policy
  // branch.  Planned proposals must never fall back to this synchronizing path.
  if (!batch_execution_plan_)
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
    return;
  }

  const std::size_t walker_count = wfc_list.size();
  if (walker_count == 0 || p_list.size() != walker_count ||
      is_accepted.size() != walker_count)
    throw std::invalid_argument(
        "PsiFormer planned mw_accept_rejectMove requires one acceptance decision per live lane");
  if (particle_index < 0)
    throw std::out_of_range(
        "PsiFormer planned mw_accept_rejectMove received a negative electron index");

  // The leader's stored token is only a request input.  Common preflight below
  // independently recomputes the team-bound transaction fingerprint and proves
  // that every lane carries the same complete proposal.
  PlannedRuntimeRequest request;
  request.operation                 = PlannedOperation::ACCEPT_REJECT_VALUE;
  request.live_walkers              = walker_count;
  request.active_electron           = static_cast<std::size_t>(particle_index);
  request.expected_proposal_version = proposed_parameter_version_;
  request.expected_proposal_origin  = proposal_origin_;
  request.expected_single_transaction_fingerprint =
      proposed_descriptor_fingerprint_;
  const PlannedRuntimeAccess access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  auto ratio_arena =
      access.resource.requireSingleResolutionStaging(walker_count);

  const auto ratio_arena_kind                      = ratio_arena.kind;
  const PsiValue* const ratio_psi_value_data       = ratio_arena.psi_value_data;
  const LogValue* const ratio_log_value_data       = ratio_arena.log_value_data;
  const std::size_t ratio_arena_extent             = ratio_arena.extent;
  const std::size_t proposal_version               = *request.expected_proposal_version;
  const ProposalOrigin proposal_origin             = *request.expected_proposal_origin;
  const std::uint64_t transaction_fingerprint =
      *access.single_transaction_fingerprint;

  // A compact read-only digest detects any acceptance-mask change between the
  // two validation phases without allocating temporary vector<bool> storage.
  const auto compute_acceptance_fingerprint = [&]() noexcept {
    std::uint64_t hash = PERSISTENT_FINGERPRINT_OFFSET;
    mixPersistentInteger(hash, walker_count);
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      mixPersistentInteger(hash, is_accepted[lane] ? 1 : 0);
    return hash;
  };
  const std::uint64_t acceptance_fingerprint =
      compute_acceptance_fingerprint();

  const auto validate_lane_state =
      [&](std::optional<std::size_t> authoritative_version) {
        for (std::size_t lane = 0; lane < walker_count; ++lane)
        {
          const auto& component =
              static_cast<const PsiFormerWF&>(wfc_list[lane]);
          const std::uint64_t base_identity = configurationIdentity(p_list[lane]);
          const std::uint64_t proposed_identity =
              configurationIdentity(p_list[lane], particle_index);

          if (!component.has_proposal_ ||
              component.proposal_origin_ != proposal_origin ||
              component.proposed_particle_ != particle_index ||
              component.proposed_descriptor_fingerprint_ !=
                  transaction_fingerprint ||
              component.proposed_parameter_version_ != proposal_version ||
              component.observed_parameter_version_ != proposal_version ||
              component.accepted_parameter_version_ != proposal_version ||
              !component.accepted_value_valid_ ||
              (component.accepted_state_requirement_ !=
                   AcceptedStateRequirement::VALUE_ONLY &&
               component.accepted_state_requirement_ !=
                   AcceptedStateRequirement::FULL_SPATIAL) ||
              component.accepted_configuration_identity_ != base_identity ||
              component.proposed_configuration_identity_ != proposed_identity)
            throw std::logic_error(
                "PsiFormer planned single-particle resolution has incomplete proposal provenance");

          if (!isCoherentPhysicalValue(*component.model_state_,
                                       component.current_sign_,
                                       component.log_value_, p_list[lane]) ||
              !isCoherentPhysicalValue(
                  *component.model_state_, component.proposed_sign_,
                  component.proposed_log_value_, p_list[lane], particle_index,
                  nullptr))
            throw std::logic_error(
                "PsiFormer planned single-particle resolution has incoherent value state");

          if (access.resource.configuration_identities[lane] !=
                  proposed_identity ||
              access.resource.staged_signs[lane] !=
                  component.proposed_sign_ ||
              access.resource.staged_log_magnitudes[lane] !=
                  std::real(component.proposed_log_value_) ||
              component.proposed_log_value_ !=
                  makePhysicalLogValue(
                      *component.model_state_,
                      access.resource.staged_signs[lane],
                      access.resource.staged_log_magnitudes[lane],
                      twistPhaseAngle(*component.model_state_, p_list[lane],
                                      particle_index)))
            throw std::logic_error(
                "PsiFormer planned single-particle resolution has changed value staging");

          const PsiValue staged_ratio = ratio_arena.load(lane);
          const double phase_change = twistDisplacementPhase(
              *component.model_state_, p_list[lane].R[particle_index],
              p_list[lane].activeR(particle_index));
          const PsiValue expected_ratio = makePhysicalRatio(
              *component.model_state_, component.proposed_sign_,
              std::real(component.proposed_log_value_),
              component.current_sign_, std::real(component.log_value_),
              phase_change);
          if (!isFiniteWavefunctionValue(staged_ratio) ||
              staged_ratio != expected_ratio)
            throw std::logic_error(
                "PsiFormer planned single-particle resolution has incoherent ratio staging");

          if (authoritative_version &&
              *authoritative_version != proposal_version)
            throw std::logic_error(
                "PsiFormer parameters changed during a planned single-particle transaction");
        }
      };

  // Phase A is read only and validates all lane and staging payload before the
  // authoritative model read transaction is opened.
  validate_lane_state(std::nullopt);

  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  if (transaction.state().execution_plan.modelShape().electrons() !=
      access.participant.plan().particleCount())
    throw std::logic_error(
        "PsiFormer planned single-particle resolution model shape changed");
  validate_lane_state(parameter_version);

  // Phase B repeats the complete typed preflight and exact retained allocation
  // identities under the model read barrier.  Normal resolution deliberately
  // leaves a stale-version proposal intact for the explicit cancellation path.
  if (is_accepted.size() != walker_count ||
      compute_acceptance_fingerprint() != acceptance_fingerprint ||
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire) == 0)
    throw std::logic_error(
        "PsiFormer planned single-particle resolution inputs changed during preflight");
  const PlannedRuntimeAccess final_access =
      requirePlannedMultiWalkerOperation(wfc_list, p_list, request);
  auto final_ratio_arena =
      final_access.resource.requireSingleResolutionStaging(walker_count);
  if (&final_access.resource != &access.resource ||
      !final_access.participant.sameBinding(access.participant) ||
      &final_access.crowd != &access.crowd ||
      final_access.storage_fingerprint != access.storage_fingerprint ||
      final_access.single_transaction_fingerprint !=
          access.single_transaction_fingerprint ||
      final_ratio_arena.kind != ratio_arena_kind ||
      final_ratio_arena.psi_value_data != ratio_psi_value_data ||
      final_ratio_arena.log_value_data != ratio_log_value_data ||
      final_ratio_arena.extent != ratio_arena_extent)
    throw std::logic_error(
        "PsiFormer planned single-particle resolution runtime evidence changed during preflight");
  ratio_arena = final_ratio_arena;
  validate_lane_state(parameter_version);
  if (compute_acceptance_fingerprint() != acceptance_fingerprint ||
      model_state_->planned_single_transaction_count.load(
          std::memory_order_acquire) == 0)
    throw std::logic_error(
        "PsiFormer planned single-particle resolution evidence changed before publication");

  if (fail_planned_single_resolution_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned single-particle resolution pre-publication failure");

  // All throwing work ends above.  Accepted lanes publish VALUE_ONLY records;
  // rejected accepted-state bytes remain untouched.  Proposal metadata and its
  // externally visible marker are cleared crowd-wide only after promotion.
  const auto publish = [&]() noexcept {
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (is_accepted[lane])
        static_cast<PsiFormerWF&>(wfc_list[lane]).accepted_value_valid_ = false;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (is_accepted[lane])
      {
        auto& component = static_cast<PsiFormerWF&>(wfc_list[lane]);
        component.current_sign_ = component.proposed_sign_;
        component.log_value_ = component.proposed_log_value_;
        component.accepted_configuration_identity_ =
            component.proposed_configuration_identity_;
        component.accepted_parameter_version_ = proposal_version;
        component.accepted_state_requirement_ =
            AcceptedStateRequirement::VALUE_ONLY;
      }
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      if (is_accepted[lane])
        static_cast<PsiFormerWF&>(wfc_list[lane]).accepted_value_valid_ = true;
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).resetProposalMetadata();
    for (std::size_t lane = 0; lane < walker_count; ++lane)
      static_cast<PsiFormerWF&>(wfc_list[lane]).has_proposal_ = false;
  };
  static_assert(noexcept(publish()));
  publish();

  // One counter entry represents this whole crowd and is withdrawn only after
  // every lane has completed marker-last publication.
  unregisterPlannedSingleTransaction();
}

// Validate planned scalar completion against the retained bound ParticleSet.
void PsiFormerWF::completeUpdates()
{
  if (!batch_execution_plan_)
  {
    WaveFunctionComponent::completeUpdates();
    return;
  }
  if (!bound_particle_set_)
    throw std::logic_error(
        "PsiFormer planned completion has no bound ParticleSet");
  requirePlannedScalarLifecycleOperation(
      PlannedOperation::COMPLETE_UPDATES, *bound_particle_set_, std::nullopt);
}

// Validate planned crowd completion from retained lane bindings only.
void PsiFormerWF::mw_completeUpdates(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  if (!batch_execution_plan_)
  {
    WaveFunctionComponent::mw_completeUpdates(wfc_list);
    return;
  }
  requirePlannedMultiWalkerLifecycleOperation(
      wfc_list, nullptr, PlannedOperation::COMPLETE_UPDATES, std::nullopt);
}

// Reserve one planned record only after two exact sizing-cursor observations.
void PsiFormerWF::registerDataPlanned(ParticleSet& particles,
                                      WFBufferType& buffer)
{
  static_assert(std::numeric_limits<FullPrecRealType>::digits >= 32,
                "PsiFormer walker metadata requires exact 32-bit scalar limbs");
  const WalkerBufferCursorSnapshot phase_a =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::REGISTER, particles, buffer);
  const std::size_t next_bulk_cursor = pf::checkedStorageSum(
      phase_a.bulk_cursor, prepared_walker_buffer_layout_.bulk_bytes,
      "PsiFormer planned walker-buffer bulk cursor overflowed");
  const std::size_t next_scalar_cursor = pf::checkedStorageSum(
      phase_a.scalar_cursor, prepared_walker_buffer_layout_.scalar_count,
      "PsiFormer planned walker-buffer scalar cursor overflowed");

  const WalkerBufferCursorSnapshot phase_b =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::REGISTER, particles, buffer);
  requireSameWalkerBufferCursorObservation(phase_a, phase_b);
  if (planned_walker_buffer_late_fault_for_testing_ ==
      PlannedWalkerBufferLateFaultForTesting::REGISTER)
    throw std::runtime_error(
        "PsiFormer injected late planned walker-buffer registration failure");

  const auto publish = [&]() noexcept {
    buffer.Current        = next_bulk_cursor;
    buffer.Current_scalar = next_scalar_cursor;
  };
  static_assert(noexcept(publish()));
  publish();
}

// Restore one planned record under a single authoritative parameter read.
void PsiFormerWF::copyFromBufferPlanned(ParticleSet& particles,
                                        WFBufferType& buffer)
{
  const WalkerBufferCursorSnapshot phase_a_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  const WalkerBufferRecordView phase_a =
      parsePlannedWalkerBufferRecord(particles, phase_a_snapshot);

  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  const WalkerBufferCursorSnapshot phase_b_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::READ, particles, buffer);
  WalkerBufferRecordView phase_b =
      parsePlannedWalkerBufferRecord(particles, phase_b_snapshot);
  requireSameWalkerBufferObservation(phase_a, phase_b);
  phase_b = classifyPlannedWalkerBufferRecord(std::move(phase_b),
                                               parameter_version);
  if (planned_walker_buffer_late_fault_for_testing_ ==
      PlannedWalkerBufferLateFaultForTesting::RESTORE)
    throw std::runtime_error(
        "PsiFormer injected late planned walker-buffer restoration failure");

  const auto publish = [&]() noexcept {
    // The live marker is withdrawn before any accepted payload changes.
    accepted_value_valid_ = false;
    if (phase_b.classification ==
        WalkerBufferRecordClassification::RESTORABLE)
    {
      static_assert(std::is_trivially_copyable_v<GradType>);
      static_assert(std::is_trivially_copyable_v<ValueType>);
      std::memcpy(accepted_gradient_.data(), phase_b.gradient_data,
                  phase_b.gradient_payload_bytes);
      std::memcpy(accepted_laplacian_.data(), phase_b.laplacian_data,
                  phase_b.laplacian_payload_bytes);
      current_sign_ = phase_b.sign;
      log_value_ = LogValue(phase_b.log_magnitude, phase_b.phase);
      accepted_configuration_identity_ = phase_b.configuration_identity;
      accepted_parameter_version_ = parameter_version;
      accepted_state_requirement_ = AcceptedStateRequirement::FULL_SPATIAL;
      observed_parameter_version_ = parameter_version;
      // Publish validity only after the complete payload and key are live.
      accepted_value_valid_ = true;
    }
    else
    {
      // Stale and zero records consume successfully without exposing their
      // spatial payload or disturbing reusable proposal allocations.
      current_sign_ = 1.0;
      log_value_ = LogValue(0);
      accepted_configuration_identity_ = 0;
      accepted_parameter_version_ = parameter_version;
      accepted_state_requirement_ = AcceptedStateRequirement::INVALID;
      observed_parameter_version_ = parameter_version;
    }

    // Both discontiguous cursors advance only after component publication.
    buffer.Current        = phase_b.next_bulk_cursor;
    buffer.Current_scalar = phase_b.next_scalar_cursor;
  };
  static_assert(noexcept(publish()));
  publish();
}

// Refresh one planned record solely from a current accepted FULL_VGL cache.
PsiFormerWF::LogValue PsiFormerWF::updateBufferPlanned(
    ParticleSet& particles,
    WFBufferType& buffer,
    bool from_scratch)
{
  if (from_scratch)
    throw std::logic_error(
        "PsiFormer planned walker-buffer update does not support from_scratch; a planned FULL_VGL refresh is required");

  const WalkerBufferCursorSnapshot phase_a_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::WRITE, particles, buffer);
  const WalkerBufferRecordView phase_a =
      makePlannedWalkerBufferRecordView(particles, phase_a_snapshot);
  const WalkerBufferRefreshEvidence phase_a_inputs =
      requirePlannedWalkerBufferRefreshInputs(
          particles, accepted_parameter_version_);

  PsiFormerReadTransaction transaction(*model_state_);
  const std::size_t parameter_version = transaction.parameterVersion();
  const WalkerBufferCursorSnapshot phase_b_snapshot =
      requirePlannedWalkerBufferOperation(
          PlannedWalkerBufferOperation::WRITE, particles, buffer);
  const WalkerBufferRecordView phase_b =
      makePlannedWalkerBufferRecordView(particles, phase_b_snapshot);
  requireSameWalkerBufferObservation(phase_a, phase_b);
  const WalkerBufferRefreshEvidence phase_b_inputs =
      requirePlannedWalkerBufferRefreshInputs(particles, parameter_version);
  if (phase_a_inputs.input_fingerprint != phase_b_inputs.input_fingerprint ||
      phase_a_inputs.parameter_version != phase_b_inputs.parameter_version ||
      std::memcmp(std::addressof(phase_a_inputs.log_value),
                  std::addressof(phase_b_inputs.log_value),
                  sizeof(LogValue)) != 0)
    throw std::logic_error(
        "PsiFormer planned walker-buffer refresh inputs changed after Phase A");

  std::array<char, pf::WALKER_BUFFER_SCALAR_COUNT *
                       sizeof(FullPrecRealType)>
      scalar_record{};
  storePersistentInteger(scalar_record.data(), 0, WALKER_BUFFER_MAGIC);
  storePersistentInteger(scalar_record.data(), 2,
                         pf::WALKER_BUFFER_LAYOUT_SCHEMA_VERSION);
  storePersistentInteger(
      scalar_record.data(), 4,
      static_cast<std::uint64_t>(AcceptedStateRequirement::FULL_SPATIAL));
  storePersistentInteger(scalar_record.data(), 6,
                         model_state_->persistent_model_identity);
  storePersistentInteger(scalar_record.data(), 8,
                         static_cast<std::uint64_t>(parameter_version));
  storePersistentInteger(scalar_record.data(), 10,
                         accepted_configuration_identity_);
  storePersistentInteger(scalar_record.data(), 12,
                         accepted_gradient_.size());
  const FullPrecRealType sign =
      static_cast<FullPrecRealType>(current_sign_);
  const FullPrecRealType log_magnitude =
      static_cast<FullPrecRealType>(std::real(log_value_));
  const FullPrecRealType phase =
      static_cast<FullPrecRealType>(std::imag(log_value_));
  storeWalkerBufferValue(
      scalar_record.data() + 14 * sizeof(FullPrecRealType), sign);
  storeWalkerBufferValue(
      scalar_record.data() + 15 * sizeof(FullPrecRealType), log_magnitude);
  storeWalkerBufferValue(
      scalar_record.data() + 16 * sizeof(FullPrecRealType), phase);

  if (planned_walker_buffer_late_fault_for_testing_ ==
      PlannedWalkerBufferLateFaultForTesting::REFRESH)
    throw std::runtime_error(
        "PsiFormer injected late planned walker-buffer refresh failure");

  const pf::WalkerBufferLayout& layout = prepared_walker_buffer_layout_;
  const LogValue result = phase_b_inputs.log_value;
  const auto publish = [&]() noexcept {
    char* const gradient_destination =
        const_cast<char*>(phase_b.gradient_data);
    char* const laplacian_destination =
        const_cast<char*>(phase_b.laplacian_data);
    char* const scalar_destination = const_cast<char*>(phase_b.scalar_data);

    // Withdraw any old magic marker before replacing a reused destination.
    std::memset(scalar_destination, 0, phase_b.scalar_payload_bytes);

    // Canonicalize aligned padding before copying the exact typed payloads.
    std::memset(gradient_destination, 0, layout.bulk_bytes);
    std::memcpy(gradient_destination, accepted_gradient_.data(),
                phase_b.gradient_payload_bytes);
    std::memcpy(laplacian_destination, accepted_laplacian_.data(),
                phase_b.laplacian_payload_bytes);

    // Keep the magic marker zero until all remaining record fields are live.
    std::memcpy(scalar_destination + 2 * sizeof(FullPrecRealType),
                scalar_record.data() + 2 * sizeof(FullPrecRealType),
                phase_b.scalar_payload_bytes -
                    2 * sizeof(FullPrecRealType));
    std::memcpy(scalar_destination, scalar_record.data(),
                2 * sizeof(FullPrecRealType));

    for (std::size_t electron = 0; electron < accepted_gradient_.size();
         ++electron)
    {
      particles.G[electron] += accepted_gradient_[electron];
      particles.L[electron] += accepted_laplacian_[electron];
    }

    // Cursor movement is the final visibility marker for the whole operation.
    buffer.Current        = phase_b.next_bulk_cursor;
    buffer.Current_scalar = phase_b.next_scalar_cursor;
  };
  static_assert(noexcept(publish()));
  publish();
  return result;
}

// Reserve fixed bulk and scalar slots without assuming that registration follows evaluation.
void PsiFormerWF::registerData(ParticleSet& particles, WFBufferType& buffer)
{
  if (batch_execution_plan_)
  {
    registerDataPlanned(particles, buffer);
    return;
  }

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
  for (std::size_t scalar = 0; scalar < pf::WALKER_BUFFER_SCALAR_COUNT; ++scalar)
    buffer.add(placeholder);
}

// Reuse complete accepted products when every validity-key field still matches.
PsiFormerWF::LogValue PsiFormerWF::updateBuffer(ParticleSet& particles,
                                                WFBufferType& buffer,
                                                bool from_scratch)
{
  if (batch_execution_plan_)
    return updateBufferPlanned(particles, buffer, from_scratch);

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
  if (batch_execution_plan_)
  {
    copyFromBufferPlanned(particles, buffer);
    return;
  }

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

// Execute one prepared scalar VALUE batch and publish only after a full recheck.
void PsiFormerWF::evaluatePlannedScalarValue(
    PlannedScalarValueOperation operation,
    const ParticleSet& reference,
    const VirtualParticleSet* virtual_particles,
    std::vector<ValueType>& ratios)
{
  NativeTimerScope timer(phase_timers_.value);
  const std::size_t configurations = checkedBatchMemoryAdd(
      ratios.size(), 1,
      "PsiFormer planned scalar VALUE configuration count");
  PlannedScalarValueRequest request{
      operation, &reference, virtual_particles, configurations,
      ratios.data(), ratios.size(), ratios.capacity()};
  PlannedScalarValueAccess access =
      requirePlannedScalarValueOperation(request);

  PsiFormerReadTransaction transaction(*model_state_);
  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE operation requires the direct VALUE backend");
  const std::size_t parameter_version = transaction.parameterVersion();
  pf::DirectBatchWorkspace& workspace = access.workspace;
  workspace.resize(pf::DirectBatchMode::VALUE_ONLY, configurations);
  packBatchConfiguration(workspace, 0, reference);
  if (operation == PlannedScalarValueOperation::ALL_TO_ONE)
    for (std::size_t electron = 0; electron < access.output_size; ++electron)
      packBatchConfiguration(
          workspace, electron + 1, reference,
          static_cast<int>(electron), &reference.getActivePos());
  else
    for (std::size_t move = 0; move < access.output_size; ++move)
      packBatchConfiguration(
          workspace, move + 1, reference,
          virtual_particles->refPtcl, &virtual_particles->R[move]);

  pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(workspace);
  if (!workspace.ownsValueResult(
          result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
          configurations))
    throw std::logic_error(
        "PsiFormer planned scalar VALUE result is not the exact workspace-owned view");

  for (std::size_t configuration = 0;
       configuration < configurations; ++configuration)
    if (result.parameter_version[configuration] != parameter_version ||
        (result.sign[configuration] != -1.0 &&
         result.sign[configuration] != 1.0) ||
        !psiformer::determinant::isFiniteReal(
            result.logabs[configuration]))
      throw std::runtime_error(
          "PsiFormer planned scalar VALUE produced invalid version, sign, or log evidence");

  for (std::size_t output = 0; output < access.output_size; ++output)
  {
    const int electron = operation == PlannedScalarValueOperation::ALL_TO_ONE
        ? static_cast<int>(output)
        : virtual_particles->refPtcl;
    const auto& proposed = operation == PlannedScalarValueOperation::ALL_TO_ONE
        ? reference.getActivePos()
        : virtual_particles->R[output];
    const double phase_change = twistDisplacementPhase(
        transaction.state(), reference.R[electron], proposed);
    const ValueType ratio = makePhysicalRatio(
        transaction.state(),
        result.sign[output + 1], result.logabs[output + 1],
        result.sign[0], result.logabs[0], phase_change);
    if (!isFiniteWavefunctionValue(ratio))
      throw std::runtime_error(
          "PsiFormer planned scalar VALUE produced a non-finite ratio");
    access.publication[output] = ratio;
  }

  // A single friend-only selector makes otherwise unreachable Phase-B
  // corruptions deterministic.  Any mutation of live input or owner evidence
  // is restored before the injected validation failure escapes.
  std::size_t* mutated_version = nullptr;
  double* mutated_sign        = nullptr;
  double* mutated_log_zero    = nullptr;
  double* mutated_log_one     = nullptr;
  std::size_t saved_version   = 0;
  double saved_sign           = 0.0;
  double saved_log_zero       = 0.0;
  double saved_log_one        = 0.0;
  ParticleSet* mutated_reference = nullptr;
  using MutableSoAPositions = std::remove_const_t<
      std::remove_reference_t<decltype(
          reference.getCoordinates().getAllParticlePos())>>;
  ParticleSet::RealType saved_aos_coordinate{};
  ParticleSet::RealType saved_soa_coordinate{};
  bool mutated_workspace_evidence   = false;
  bool mutated_publication_evidence = false;
  const std::size_t saved_prepared_workspace_fingerprint =
      prepared_batch_storage_fingerprint_;
  const ValueType* const saved_prepared_publication_data =
      prepared_scalar_value_publication_data_;

  switch (planned_scalar_value_fault_for_testing_)
  {
  case PlannedScalarValueFaultForTesting::NONE:
    break;
  case PlannedScalarValueFaultForTesting::RESULT_OWNER:
    result.owner = nullptr;
    break;
  case PlannedScalarValueFaultForTesting::RESULT_GENERATION:
    result.generation ^= std::size_t{1};
    break;
  case PlannedScalarValueFaultForTesting::RESULT_SIZE:
    ++result.size;
    break;
  case PlannedScalarValueFaultForTesting::RESULT_VERSION:
    mutated_version = const_cast<std::size_t*>(result.parameter_version);
    saved_version = mutated_version[0];
    mutated_version[0] ^= std::size_t{1};
    break;
  case PlannedScalarValueFaultForTesting::RESULT_SIGN:
    mutated_sign = const_cast<double*>(result.sign);
    saved_sign = mutated_sign[0];
    mutated_sign[0] = 0.0;
    break;
  case PlannedScalarValueFaultForTesting::RESULT_LOG_MAGNITUDE:
    mutated_log_zero = const_cast<double*>(result.logabs);
    saved_log_zero = mutated_log_zero[0];
    mutated_log_zero[0] =
        std::numeric_limits<double>::infinity();
    break;
  case PlannedScalarValueFaultForTesting::RESULT_RATIO:
    mutated_log_zero = const_cast<double*>(result.logabs);
    saved_log_zero = mutated_log_zero[0];
    if (configurations > 1)
    {
      mutated_log_one = mutated_log_zero + 1;
      saved_log_one = mutated_log_one[0];
      mutated_log_zero[0] = -std::numeric_limits<double>::max();
      mutated_log_one[0]  = std::numeric_limits<double>::max();
    }
    else
      mutated_log_zero[0] =
          std::numeric_limits<double>::infinity();
    break;
  case PlannedScalarValueFaultForTesting::INPUT_FINGERPRINT:
  {
    mutated_reference = const_cast<ParticleSet*>(&reference);
    auto& mutable_soa = const_cast<MutableSoAPositions&>(
        mutated_reference->getCoordinates().getAllParticlePos());
    saved_aos_coordinate = mutated_reference->R[0][0];
    saved_soa_coordinate = mutable_soa[0][0];
    const ParticleSet::RealType changed = std::nextafter(
        saved_aos_coordinate,
        std::numeric_limits<ParticleSet::RealType>::infinity());
    mutated_reference->R[0][0] = changed;
    mutable_soa.getNonConstData()[0] = changed;
    break;
  }
  case PlannedScalarValueFaultForTesting::OUTPUT_IDENTITY:
    break;
  case PlannedScalarValueFaultForTesting::WORKSPACE_EVIDENCE:
    mutated_workspace_evidence = true;
    prepared_batch_storage_fingerprint_ ^= std::size_t{1};
    break;
  case PlannedScalarValueFaultForTesting::PUBLICATION_EVIDENCE:
    mutated_publication_evidence = true;
    prepared_scalar_value_publication_data_ = nullptr;
    break;
  }

  const auto restore_fault = [&]() noexcept {
    if (mutated_version)
      mutated_version[0] = saved_version;
    if (mutated_sign)
      mutated_sign[0] = saved_sign;
    if (mutated_log_zero)
      mutated_log_zero[0] = saved_log_zero;
    if (mutated_log_one)
      mutated_log_one[0] = saved_log_one;
    if (mutated_reference)
    {
      mutated_reference->R[0][0] = saved_aos_coordinate;
      auto& mutable_soa = const_cast<MutableSoAPositions&>(
          mutated_reference->getCoordinates().getAllParticlePos());
      mutable_soa.getNonConstData()[0] = saved_soa_coordinate;
    }
    if (mutated_workspace_evidence)
      prepared_batch_storage_fingerprint_ =
          saved_prepared_workspace_fingerprint;
    if (mutated_publication_evidence)
      prepared_scalar_value_publication_data_ =
          saved_prepared_publication_data;
  };

  // Phase B reconstructs the request from live caller evidence and repeats
  // every throwing storage/input check without resetting workspace state.
  PlannedScalarValueRequest final_request{
      operation, &reference, virtual_particles, configurations,
      ratios.data(), ratios.size(), ratios.capacity()};
  if (planned_scalar_value_fault_for_testing_ ==
      PlannedScalarValueFaultForTesting::OUTPUT_IDENTITY)
    final_request.output_data = nullptr;
  try
  {
    PlannedScalarValueAccess final_access =
        requirePlannedScalarValueOperation(final_request);
    if (&final_access.workspace != &workspace ||
        final_access.publication != access.publication ||
        !final_access.participant.sameBinding(access.participant) ||
        final_access.workspace_fingerprint !=
            access.workspace_fingerprint ||
        final_access.workspace_bytes != access.workspace_bytes ||
        final_access.input_fingerprint != access.input_fingerprint ||
        final_access.output_data != access.output_data ||
        final_access.output_size != access.output_size ||
        final_access.output_capacity != access.output_capacity ||
        !workspace.ownsValueResult(
            result, pf::DirectBatchValueInput::DENSE_CONFIGURATIONS,
            configurations))
      throw std::logic_error(
          "PsiFormer planned scalar VALUE evidence changed during evaluation");

    for (std::size_t configuration = 0;
         configuration < configurations; ++configuration)
      if (result.parameter_version[configuration] != parameter_version ||
          (result.sign[configuration] != -1.0 &&
           result.sign[configuration] != 1.0) ||
          !psiformer::determinant::isFiniteReal(
              result.logabs[configuration]))
        throw std::runtime_error(
            "PsiFormer planned scalar VALUE result changed before publication");
    for (std::size_t output = 0; output < access.output_size; ++output)
    {
      const int electron = operation == PlannedScalarValueOperation::ALL_TO_ONE
          ? static_cast<int>(output)
          : virtual_particles->refPtcl;
      const auto& proposed = operation == PlannedScalarValueOperation::ALL_TO_ONE
          ? reference.getActivePos()
          : virtual_particles->R[output];
      const double phase_change = twistDisplacementPhase(
          transaction.state(), reference.R[electron], proposed);
      const ValueType expected = makePhysicalRatio(
          transaction.state(),
          result.sign[output + 1], result.logabs[output + 1],
          result.sign[0], result.logabs[0], phase_change);
      if (!isFiniteWavefunctionValue(access.publication[output]) ||
          access.publication[output] != expected)
        throw std::runtime_error(
            "PsiFormer planned scalar VALUE staging changed before publication");
    }
  }
  catch (...)
  {
    restore_fault();
    throw;
  }
  restore_fault();

  if (planned_scalar_value_fault_for_testing_ !=
      PlannedScalarValueFaultForTesting::NONE)
    throw std::logic_error(
        "PsiFormer planned scalar VALUE test fault was not detected");

  for (std::size_t output = 0; output < access.output_size; ++output)
    if (!isFiniteWavefunctionValue(access.publication[output]))
      throw std::runtime_error(
          "PsiFormer planned scalar VALUE publication became non-finite");

  if (fail_planned_scalar_value_before_publish_for_testing_)
    throw std::overflow_error(
        "Injected PsiFormer planned scalar VALUE pre-publication failure");

  static_assert(std::is_nothrow_copy_assignable_v<ValueType>);
  const auto publish = [&]() noexcept {
    for (std::size_t output = 0; output < access.output_size; ++output)
      access.output_data[output] = access.publication[output];
  };
  publish();
}

// Replace each electron by the common virtual position in one state-isolated batch.
void PsiFormerWF::evaluateRatiosAlltoOne(ParticleSet& particles,
                                         std::vector<ValueType>& ratios)
{
  if (batch_execution_plan_)
  {
    evaluatePlannedScalarValue(
        PlannedScalarValueOperation::ALL_TO_ONE, particles, nullptr, ratios);
    return;
  }

  if (particles.isSpinor())
    throw std::invalid_argument(
        "PsiFormer all-to-one ratios do not support spinor virtual moves");
  if (ratios.size() !=
      static_cast<std::size_t>(particles.getTotalNum()))
    throw std::invalid_argument(
        "PsiFormer all-to-one ratio output has the wrong size");

  const std::size_t configurations = checkedBatchMemoryAdd(
      static_cast<std::size_t>(particles.getTotalNum()), 1,
      "PsiFormer all-to-one configuration count");
  std::vector<ValueType> staged_ratios(ratios.size());
  PsiFormerReadTransaction transaction(*model_state_);
  synchronizeParameterVersion(transaction.parameterVersion());

  // Oracle and compare modes retain the scalar validation path but use the
  // explicit common position; the inherited default incorrectly calls activeR().
  if (transaction.state().direct_value_mode != DirectBackendMode::DIRECT)
  {
    const pf::Result reference = evaluatePositionsUnderRead(
        transaction, particles, -1, nullptr, EvaluationPurpose::VALUE_ONLY);
    for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    {
      const pf::Result moved = evaluatePositionsUnderRead(
          transaction, particles, electron, &particles.getActivePos(),
          EvaluationPurpose::VALUE_ONLY);
      staged_ratios[electron] = makePhysicalRatio(
          transaction.state(), moved.sign, moved.logabs, reference.sign,
          reference.logabs, twistDisplacementPhase(
              transaction.state(), particles.R[electron],
              particles.getActivePos()));
    }
    ratios.swap(staged_ratios);
    return;
  }

  const std::size_t parameter_version = transaction.parameterVersion();
  auto& batch = requireDirectBatchWorkspace();
  batch.resize(pf::DirectBatchMode::VALUE_ONLY, configurations);
  packBatchConfiguration(batch, 0, particles);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    packBatchConfiguration(
        batch, static_cast<std::size_t>(electron) + 1, particles,
        electron, &particles.getActivePos());

  const pf::DirectBatchValueResultView result =
      transaction.state().direct_batch_executor.evaluateValues(batch);
  if (result.parameter_version[0] != parameter_version)
    throw std::logic_error(
        "PsiFormer all-to-one reference observed inconsistent parameters");
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    const std::size_t configuration =
        static_cast<std::size_t>(electron) + 1;
    if (result.parameter_version[configuration] != parameter_version)
      throw std::logic_error(
          "PsiFormer all-to-one batch observed inconsistent parameters");
    staged_ratios[electron] = makePhysicalRatio(
        transaction.state(),
        result.sign[configuration], result.logabs[configuration],
        result.sign[0], result.logabs[0], twistDisplacementPhase(
            transaction.state(), particles.R[electron],
            particles.getActivePos()));
  }
  ratios.swap(staged_ratios);
}

// Evaluate independent full-network ratios for all quadrature positions without mutating walker state.
void PsiFormerWF::evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios)
{
  if (batch_execution_plan_)
  {
    requireNoPlannedEcpScalarDispatch(
        "scalar virtual-ratio evaluation");
    evaluatePlannedScalarValue(
        PlannedScalarValueOperation::VIRTUAL_PARTICLE_VALUE,
        virtual_particles.getRefPS(), &virtual_particles, ratios);
    return;
  }

  if (ratios.size() !=
      static_cast<std::size_t>(virtual_particles.getTotalNum()))
    throw std::invalid_argument(
        "PsiFormer virtual-particle ratio output has the wrong size");
  requireNoPlannedEcpScalarDispatch("scalar virtual-ratio evaluation");
  PsiFormerReadTransaction transaction(*model_state_, model_state_->deviceOnlyValueReads());
  synchronizeParameterVersion(transaction.parameterVersion());
  evaluateRatiosUnderRead(transaction, virtual_particles, ratios);
}

void PsiFormerWF::evaluateRatiosUnderRead(
    const PsiFormerReadTransaction& transaction,
    const VirtualParticleSet& virtual_particles,
    std::vector<ValueType>& ratios)
{
  requireUnplannedScalarEvaluation("virtual-ratio evaluation");
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
  std::vector<ValueType> staged_ratios(ratios.size());

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
      staged_ratios[move] = makePhysicalRatio(
          transaction.state(),
          virtual_result.sign, virtual_result.logabs, reference_result.sign,
          reference_result.logabs, twistDisplacementPhase(
              transaction.state(), reference.R[electron],
              virtual_particles.R[move]));
    }
    ratios.swap(staged_ratios);
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
    staged_ratios[move] = makePhysicalRatio(
        transaction.state(),
        result.sign[move + 1], result.logabs[move + 1], result.sign[0],
        result.logabs[0], twistDisplacementPhase(
            transaction.state(), reference.R[electron],
            virtual_particles.R[move]));
  }
  ratios.swap(staged_ratios);
}

// Evaluate a descriptor-ordered virtual batch from one sparse reference per active walker.
WaveFunctionComponent::EvaluationStamp PsiFormerWF::mw_evaluateVirtualRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
    const VirtualParticleBatch& virtual_batch,
    std::vector<ValueType>& ratios) const
{
  requireUnplannedMultiWalkerOperation(
      "flattened virtual-ratio evaluation");
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
  PsiFormerReadTransaction transaction(*leader.model_state_, leader.model_state_->deviceOnlyValueReads());
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
        const ParticleSet& particles =
            p_list[static_cast<std::size_t>(slice.walkerId())];
        resource.flat_virtual_ratios[flat_index] = makePhysicalRatio(
            transaction.state(),
            sparse_result.sign[replacement], sparse_result.logabs[replacement],
            sparse_result.sign[reference], sparse_result.logabs[reference],
            twistDisplacementPhase(
                transaction.state(), particles.R[slice.electronId()],
                slice.absolutePosition(local_index)));
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
        resource.flat_virtual_ratios[flat_index] = makePhysicalRatio(
            transaction.state(),
            virtual_result.sign, virtual_result.logabs,
            resource.virtual_reference_signs[reference],
            resource.virtual_reference_logabs[reference],
            twistDisplacementPhase(
                transaction.state(), p_list[walker].R[slice.electronId()],
                slice.absolutePosition(local_index)));
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
  requireUnplannedMultiWalkerOperation("ragged virtual-ratio evaluation");
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

  PsiFormerReadTransaction transaction(*leader.model_state_, leader.model_state_->deviceOnlyValueReads());
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
      const auto& virtual_particles = virtual_particle_list[walker];
      const std::size_t reference = resource.virtual_offsets[walker];
      for (std::size_t move = 0; move < ratios[walker].size(); ++move)
      {
        const std::size_t configuration = reference + move + 1;
        staged_ratios[walker][move] = makePhysicalRatio(
            transaction.state(),
            result.sign[configuration], result.logabs[configuration],
            result.sign[reference], result.logabs[reference],
            twistDisplacementPhase(
                transaction.state(),
                virtual_particles.getRefPS().R[virtual_particles.refPtcl],
                virtual_particles.R[move]));
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
        staged_ratios[walker][move] = makePhysicalRatio(
            transaction.state(),
            virtual_result.sign, virtual_result.logabs, reference_result.sign,
            reference_result.logabs, twistDisplacementPhase(
                transaction.state(), reference.R[virtual_particles.refPtcl],
                virtual_particles.R[move]));
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
  requireUnplannedScalarDerivative("derivative-ratio evaluation");
  requireNoPlannedEcpScalarDispatch("scalar derivative-ratio evaluation");
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

    const ValueType staged_ratio = makePhysicalRatio(
        transaction.modelTransaction().state(), virtual_sign, virtual_logabs,
        reference_sign, reference_logabs, twistDisplacementPhase(
            transaction.modelTransaction().state(), reference.R[electron],
            virtual_particles.R[move]));
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
  requireUnplannedScalarDerivative("weighted derivative-ratio evaluation");
  if (weighted_derivatives.size != 0 && weighted_derivatives.data == nullptr)
    throw std::invalid_argument("PsiFormer weighted derivative destination is null");
  requireNoPlannedEcpScalarDispatch(
      "scalar weighted derivative-ratio evaluation");
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
  requireUnplannedMultiWalkerOperation(
      "flattened weighted derivative-ratio evaluation");
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
  requireUnplannedMultiWalkerOperation(
      "batched weighted derivative-ratio evaluation");
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
  requireUnplannedMultiWalkerOperation(
      "batched parameter-score evaluation");
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
  requireUnplannedScalarDerivative("kinetic-parameter evaluation");
  PsiFormerDerivativeReadTransaction transaction(*model_state_,
                                                  *optimization_metadata_);
  synchronizeParameterVersion(transaction.modelTransaction().parameterVersion());
  if (optimization_metadata_->enabled && transaction.hasActiveParameters())
    requireUnitElectronMasses(p);
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
  bool has_imaginary_drift = false;
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      has_imaginary_drift = has_imaginary_drift ||
          std::imag(p.G[electron][dimension]) != 0.0;
  if (has_imaginary_drift && kinetic_mode != DirectBackendMode::DIRECT)
    throw std::invalid_argument(
        "Complex PsiFormer kinetic parameter response requires the direct backend");
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

  if (has_imaginary_drift)
  {
    // The score has already been gathered.  Reuse its O(P) workspace vector for
    // the gradient-only imaginary reverse while preserving the real kinetic row.
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        total_log_gradient[3 * electron + dimension] =
            std::imag(p.G[electron][dimension]);
    const pf::DirectKineticConstView imaginary_response =
        transaction.modelTransaction().state().direct_kinetic_executor
            .evaluateGradientParameterResponse(
                kinetic_workspace, total_log_gradient.data(),
                total_log_gradient.size());
    // Accumulate directly into the already-materialized selected response.  A
    // temporary selected-parameter row would add an avoidable O(P) allocation
    // to every complex kinetic-derivative call.
    const auto& selected_flat_indices = transaction.selectedFlatIndices();
    const OptVariables& variables     = transaction.variables();
    std::size_t active_parameter      = 0;
    for (std::size_t local_index = 0;
         local_index < selected_flat_indices.size(); ++local_index)
    {
      const int global_index = variables.where(local_index);
      if (global_index < 0)
        continue;
      if (static_cast<std::size_t>(global_index) >= kinetic_destination_size)
        throw std::out_of_range(
            "PsiFormer complex kinetic destination index is out of range");
      const std::size_t flat_index = selected_flat_indices[local_index];
      if (flat_index >= imaginary_response.size() ||
          !psiformer::determinant::isFiniteReal(
              imaginary_response[flat_index]))
        throw std::runtime_error(
            "PsiFormer imaginary kinetic parameter derivative is invalid");
      if (active_parameter >= kinetic_output.size() ||
          kinetic_output[active_parameter].first !=
              static_cast<std::size_t>(global_index))
        throw std::logic_error(
            "PsiFormer complex kinetic response index changed between reverses");
      kinetic_output[active_parameter].second +=
          makePsiFormerScalar<ValueType>(
              0.0, imaginary_response[flat_index]);
      ++active_parameter;
    }
    if (active_parameter != kinetic_output.size())
      throw std::logic_error(
          "PsiFormer complex kinetic response mapping changed between reverses");
  }
}

// Fill score and kinetic rows serially through one resource-owned direct kinetic tape.
void PsiFormerWF::mw_evaluateParameterDerivatives(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const OptVariables& optvars,
    RecordArray<ValueType>& dlogpsi,
    RecordArray<ValueType>& dhpsioverpsi) const
{
  requireUnplannedMultiWalkerOperation(
      "batched kinetic-parameter evaluation");
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

// The explicit complex sibling shares this Native.h-owning TU, not a second native object.
#define QMC_PSIFORMER_COMPLEX_WF_IMPL_OWNER_TU
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWFImpl.h"
#undef QMC_PSIFORMER_COMPLEX_WF_IMPL_OWNER_TU

} // namespace qmcplusplus
