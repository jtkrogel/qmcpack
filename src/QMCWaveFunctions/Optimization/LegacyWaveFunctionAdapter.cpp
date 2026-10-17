//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file LegacyWaveFunctionAdapter.cpp
 * @brief Implementation of the bounded legacy-variable training bridge.
 */

#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/JeeIOrbitalSoA.h"
#include "QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h"
#include "QMCWaveFunctions/RotatedSPOs.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::wftrain
{
/// One shared lane per adapter, independent of the number of retained block views.
struct LegacyDerivativeWorkspace
{
  explicit LegacyDerivativeWorkspace(std::size_t count)
      : score_row(count), kinetic_row(count),
        channels{std::vector<DerivativeValue>(count), std::vector<DerivativeValue>(count),
                 std::vector<DerivativeValue>(count)}
  {}
  std::mutex mutex;
  Vector<QMCTraits::ValueType> score_row;
  Vector<QMCTraits::ValueType> kinetic_row;
  std::array<std::vector<DerivativeValue>, 3> channels;
};

namespace
{

constexpr std::size_t MAXIMUM_VJP_CHANNELS = 3;

/// Test IEEE-754 finiteness without fast-math assumptions.
bool isFiniteDouble(double value) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559,
                "Legacy adapter requires IEEE-754 binary64 parameters");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

/// Multiply storage extents without allowing size_t wraparound.
std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs, const char* context)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::length_error(std::string(context) + " exceeds addressable storage");
  return lhs * rhs;
}

/// Add storage extents without allowing size_t wraparound.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* context)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::length_error(std::string(context) + " exceeds addressable storage");
  return lhs + rhs;
}

/// Return the conservative two-row plus three-channel derivative scratch bound.
std::size_t requiredScratchBytes(std::size_t parameter_count)
{
  const std::size_t legacy_rows = checkedMultiply(
      checkedMultiply(2, parameter_count, "Legacy derivative rows"), sizeof(QMCTraits::ValueType),
      "Legacy derivative row bytes");
  const std::size_t channels = checkedMultiply(
      checkedMultiply(MAXIMUM_VJP_CHANNELS, parameter_count, "Legacy derivative channels"),
      sizeof(DerivativeValue), "Legacy derivative channel bytes");
  return checkedAdd(legacy_rows, channels, "Legacy derivative scratch");
}

/// Map a legacy parameter category to a stable, human-readable update group.
std::string categoryName(int category)
{
  switch (category)
  {
  case optimize::OTHER_P:
    return "legacy_other";
  case optimize::LOGLINEAR_P:
    return "legacy_loglinear";
  case optimize::LOGLINEAR_K:
    return "legacy_kspace_loglinear";
  case optimize::LINEAR_P:
    return "legacy_linear";
  case optimize::SPO_P:
    return "legacy_spo";
  case optimize::BACKFLOW_P:
    return "legacy_backflow";
  default:
    return "legacy_category_" + std::to_string(category);
  }
}

/// Mix bytes into a deterministic FNV-1a digest independent of std::hash.
void mixDigest(std::uint64_t& digest, const void* data, std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index)
  {
    digest ^= bytes[index];
    digest *= UINT64_C(1099511628211);
  }
}

/// Form a stable block identifier from the ordered legacy names and category.
std::string blockId(const optimize::VariableSet& variables,
                    const std::vector<std::size_t>& active_locations,
                    std::size_t first,
                    std::size_t count,
                    int category)
{
  std::uint64_t digest = UINT64_C(14695981039346656037);
  const std::string category_identity = std::to_string(category);
  mixDigest(digest, category_identity.data(), category_identity.size());
  for (std::size_t index = first; index < first + count; ++index)
  {
    const std::string& name = variables.name(active_locations[index]);
    mixDigest(digest, name.data(), name.size());
    const unsigned char separator = 0;
    mixDigest(digest, &separator, sizeof(separator));
  }

  std::ostringstream id;
  id << categoryName(category) << '/' << std::hex << std::setfill('0') << std::setw(16) << digest;
  return id.str();
}

/// Build compact category runs and the canonical active-index lookup.
std::pair<std::vector<ParameterBlockDescriptor>, std::vector<std::size_t>> makeBlocks(
    const optimize::VariableSet& variables)
{
  const std::size_t active_count = static_cast<std::size_t>(variables.size_of_active());
  std::vector<std::size_t> active_locations(active_count, variables.size());
  for (std::size_t location = 0; location < variables.size(); ++location)
  {
    const int active_index = variables.where(static_cast<int>(location));
    if (active_index >= 0)
    {
      if (static_cast<std::size_t>(active_index) >= active_count ||
          active_locations[active_index] != variables.size())
        throw std::logic_error("Legacy variables do not have unique dense active indices");
      active_locations[active_index] = location;
    }
  }
  if (std::find(active_locations.begin(), active_locations.end(), variables.size()) != active_locations.end())
    throw std::logic_error("Legacy variables have an incomplete active-index ordering");

  std::vector<ParameterBlockDescriptor> blocks;
  for (std::size_t first = 0; first < active_count;)
  {
    const int category = variables.getType(active_locations[first]);
    std::size_t last   = first + 1;
    while (last < active_count && variables.getType(active_locations[last]) == category)
      ++last;
    const std::size_t count = last - first;
    blocks.push_back({blockId(variables, active_locations, first, count, category),
                      {count},
                      first,
                      count,
                      ParameterScalarDomain::REAL64,
                      true,
                      categoryName(category)});
    first = last;
  }
  return {std::move(blocks), std::move(active_locations)};
}

/// Verify that a disposable evaluator registers exactly the source variable layout.
optimize::VariableSet validateEvaluatorRegistration(TrialWaveFunction& evaluator,
                                                    const optimize::VariableSet& expected)
{
  if (!evaluator.extractStructuredParameterProviders().empty())
    throw std::invalid_argument("Legacy adapter evaluators must not contain structured parameter providers");

  optimize::VariableSet actual;
  evaluator.checkInVariables(actual);
  actual.resetIndex();
  if (actual.size() != expected.size() || actual.size_of_active() != expected.size_of_active())
    throw std::invalid_argument("Legacy adapter evaluator variable extent does not match registration source");
  for (std::size_t index = 0; index < actual.size(); ++index)
    if (actual.name(index) != expected.name(index) || actual.getType(index) != expected.getType(index) ||
        actual.where(static_cast<int>(index)) != expected.where(static_cast<int>(index)) ||
        (actual.where(static_cast<int>(index)) < 0 && actual[static_cast<int>(index)] !=
             expected[static_cast<int>(index)]))
      throw std::invalid_argument("Legacy adapter evaluator variable registration does not match source");
  return actual;
}

/// Install one detached snapshot in private component state without touching particles.
void resetEvaluator(TrialWaveFunction& evaluator,
                      const optimize::VariableSet& expected,
                      const std::vector<double>& values)
{
  optimize::VariableSet active = validateEvaluatorRegistration(evaluator, expected);
  // J1/J2 clone their scalar coordinates but deliberately share the physical
  // B-spline coefficient bank. Detach each unique functor before a private reset,
  // including rejected candidate preparation; repeated replay resets are unique.
  auto objects = evaluator.extractOptimizableObjectRefs();
  for (OptimizableObject& object : objects)
    if (auto* spline = dynamic_cast<BsplineFunctor<QMCTraits::RealType>*>(&object))
      spline->isolateOptimizationState();
  for (std::size_t location = 0; location < active.size(); ++location)
  {
    const int active_index = active.where(static_cast<int>(location));
    if (active_index >= 0)
      active[static_cast<int>(location)] = values[active_index];
  }
  evaluator.checkOutVariables(active);
  evaluator.resetParameters(active);
}

void validateSnapshot(const StructuredParameterSnapshot& snapshot, const StructuredParameterSchema& schema)
{
  if (snapshot.schema_fingerprint != schema.fingerprint() || snapshot.values.size() != schema.parameterCount())
    throw std::invalid_argument("Legacy adapter snapshot schema or parameter extent does not match");
  if (std::any_of(snapshot.values.begin(), snapshot.values.end(), [](double value) { return !isFiniteDouble(value); }))
    throw std::invalid_argument("Legacy adapter snapshot contains a non-finite value");
}

void validateMasses(const ParticleSet& particles)
{
  const auto& masses = particles.get_mass_by_group();
  if (masses.size() == 0 || std::any_of(masses.begin(), masses.end(), [](auto mass) { return mass != 1.0; }))
    throw std::invalid_argument("Legacy adapter supports only initialized unit electron masses");
}

DerivativeValue asDerivativeValue(QMCTraits::ValueType value) noexcept;

/// Distinguish nodal-safe physical-ratio rows from the fixed-Slater score path.
enum class LegacyECPProfile { FIXED_SLATER, ORBITAL, DETS };

/// Keep the independently qualified real CPU ECP component profiles explicit.
LegacyECPProfile validateECPProfile(const LegacyWaveFunctionReplaySample& sample)
{
#ifdef QMC_COMPLEX
  throw std::invalid_argument("Legacy ECP products require a real CPU build");
#endif
  if (sample.particles.isSpinor() ||
      sample.particles.getCoordinates().getKind() != DynamicCoordinateKind::DC_POS)
    throw std::invalid_argument("Legacy ECP products require CPU scalar electron coordinates");
  std::size_t slaters = 0;
  bool orbital = false;
  bool dets = false;
  bool polynomial_j3 = false;
  for (const auto& component : sample.wavefunction.getOrbitals())
    if (const auto* msd = dynamic_cast<const MultiSlaterDetTableMethod*>(component.get()))
    {
      ++slaters;
      dets = true;
      if (msd->getDetSize() != 2 || sample.particles.groups() != 2 ||
          sample.particles.getLattice().SuperCellEnum != SUPERCELL_OPEN)
        throw std::invalid_argument("Legacy coefficient ECP products require open fixed-orbital two-spin DETS/CSF");
      msd->validatePreparedCoefficientState(*msd);
      for (int spin = 0; spin < msd->getDetSize(); ++spin)
        if (msd->getDet(spin).getPhi()->isOMPoffload() || msd->getDet(spin).getPhi()->isOptimizable())
          throw std::invalid_argument("Legacy coefficient ECP products require fixed CPU orbitals");
    }
    else if (const auto* slater = dynamic_cast<const SlaterDet*>(component.get()))
    {
      ++slaters;
      orbital = slater->isOptimizable();
      if (orbital && (slater->getNumDets() != 2 || sample.particles.groups() != 2 ||
          sample.particles.groupsize(0) != 1 || sample.particles.groupsize(1) != 1 ||
          sample.particles.getLattice().SuperCellEnum != SUPERCELL_OPEN))
        throw std::invalid_argument("Legacy orbital ECP products require open one-occupied spin determinants");
      for (int determinant = 0; determinant < slater->getNumDets(); ++determinant)
      {
        const auto& phi = slater->getPhi(determinant);
        const auto* rotated = dynamic_cast<const RotatedSPOs*>(&phi);
        const auto* lcao = rotated ? dynamic_cast<const LCAOrbitalSet*>(rotated->Phi_.get()) : nullptr;
        if (phi.isOMPoffload() || (!orbital && phi.isOptimizable()) || (orbital && (!phi.isOptimizable() || !lcao || lcao->isOMPoffload() ||
            lcao->fileImageIdentity() || rotated->nel_major_ != 1 || phi.getOrbitalSetSize() != 3 ||
            rotated->m_act_rot_inds_ != RotatedSPOs::RotationIndices{{0, 1}, {0, 2}})))
          throw std::invalid_argument("Legacy orbital ECP products require two CPU inline three-orbital (0,q) owners");
      }
    }
    else if (dynamic_cast<const JeeIOrbitalSoA<PolynomialFunctor3D>*>(component.get()))
      polynomial_j3 = true;
    else if (component->getClassName() != "J1OrbitalSoA" && component->getClassName() != "TwoBodyJastrow")
      throw std::invalid_argument("Legacy ECP products admit only qualified Slater/DETS and J1/J2 components");
  if (polynomial_j3 && (dets || orbital))
    throw std::invalid_argument("Legacy polynomial J3 ECP products require a fixed single Slater without active orbitals");
  if (slaters != 1)
    throw std::invalid_argument("Legacy ECP products require exactly one fixed Slater component");
  return dets ? LegacyECPProfile::DETS : orbital ? LegacyECPProfile::ORBITAL : LegacyECPProfile::FIXED_SLATER;
}

/// Check finite inputs without permitting fast-math to remove the check.
void requireFiniteECP(DerivativeValue value)
{
  if (!isFiniteDouble(value.real()) || !isFiniteDouble(value.imag()))
    throw std::invalid_argument("Legacy ECP replay contains a nonfinite value");
}

/// Validate complete canonical coverage, including legal empty stamp lists.
void validateECPRecord(const NonLocalECPJVPReplay& record)
{
  if (!record.gridFingerprint() || record.referencePositions().size() !=
      checkedMultiply(checkedMultiply(record.sampleCount(), record.electronCount(), "ECP references"), 3,
                      "ECP Cartesian references"))
    throw std::invalid_argument("Legacy ECP replay grid or reference extent is invalid");
  for (double coordinate : record.referencePositions())
    requireFiniteECP({coordinate, 0});
  const auto points = record.points();
  const auto segments = record.segments();
  std::size_t next_point = 0;
  std::size_t next_segment = 0;
  std::size_t next_stamp = 0;
  for (const auto& tile : record.tiles())
  {
    if (tile.tile_ordinal != &tile - record.tiles().data() || !tile.virtual_batch_fingerprint ||
        tile.point_offset != next_point || tile.segment_offset != next_segment || tile.stamp_offset != next_stamp ||
        tile.point_count > points.size() - next_point || tile.segment_count > segments.size() - next_segment ||
        tile.stamp_count > record.evaluationStamps().size() - next_stamp)
      throw std::invalid_argument("Legacy ECP replay has invalid tile provenance");
    const auto tile_end = next_point + tile.point_count;
    for (std::size_t i = next_segment; i < next_segment + tile.segment_count; ++i)
    {
      const auto& segment = segments[i];
      if (segment.point_offset != next_point || segment.point_count > tile_end - next_point ||
          segment.sample >= record.sampleCount() || segment.electron >= record.electronCount())
        throw std::invalid_argument("Legacy ECP replay has invalid segment coverage");
      for (std::size_t p = next_point; p < next_point + segment.point_count; ++p)
        if (points[p].sample != segment.sample || points[p].electron != segment.electron)
          throw std::invalid_argument("Legacy ECP point disagrees with its segment");
      next_point += segment.point_count;
    }
    if (next_point != tile_end)
      throw std::invalid_argument("Legacy ECP replay tile coverage is incomplete");
    next_segment += tile.segment_count;
    next_stamp += tile.stamp_count;
  }
  if (next_point != points.size() || next_segment != segments.size() ||
      next_stamp != record.evaluationStamps().size())
    throw std::invalid_argument("Legacy ECP replay coverage is incomplete");
  for (const auto& point : points)
  {
    for (double coordinate : point.replacement_position)
      requireFiniteECP({coordinate, 0});
    requireFiniteECP(point.bare_weight);
    requireFiniteECP(point.complete_ratio);
    requireFiniteECP(point.bare_weight * point.complete_ratio);
    if (point.bare_weight.imag() != 0 || point.complete_ratio.imag() != 0)
      throw std::invalid_argument("Legacy ECP replay requires real weights and ratios");
  }
}

/** Own accepted grid data only; never retain parameter-sized buffers per block. */
class LegacyECPRecord final : public NonLocalECPJVPReplay
{
public:
  std::string provider;
  std::string schema;
  std::size_t version = 0;
  std::size_t batch = 0;
  std::size_t offset = 0;
  std::size_t samples = 0;
  std::size_t electrons = 0;
  std::uint64_t grid = 0;
  std::vector<double> references;
  std::vector<NonLocalECPReplayPoint> point_data;
  std::vector<NonLocalECPReplaySegment> segment_data;
  std::vector<NonLocalECPReplayTile> tile_data;
  std::vector<WaveFunctionComponent::EvaluationStamp> stamps;

  /// Expose immutable model, batch and accepted-grid identity to product binding.
  std::string_view providerId() const noexcept override { return provider; }

  std::string_view schemaFingerprint() const noexcept override { return schema; }

  std::size_t parameterVersion() const noexcept override { return version; }

  std::size_t batchOrdinal() const noexcept override { return batch; }

  std::size_t sampleOffset() const noexcept override { return offset; }

  std::size_t sampleCount() const noexcept override { return samples; }

  std::size_t electronCount() const noexcept override { return electrons; }

  std::uint32_t localEnergyTermMask() const noexcept override
  {
    return localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  }

  std::uint64_t gridFingerprint() const noexcept override { return grid; }

  /// Return nonowning views whose lifetime is the accepted record's lifetime.
  DerivativeArrayView<const double> referencePositions() const noexcept override
  {
    return {references.data(), references.size()};
  }

  DerivativeArrayView<const NonLocalECPReplayPoint> points() const noexcept override
  {
    return {point_data.data(), point_data.size()};
  }

  DerivativeArrayView<const NonLocalECPReplaySegment> segments() const noexcept override
  {
    return {segment_data.data(), segment_data.size()};
  }

  DerivativeArrayView<const NonLocalECPReplayTile> tiles() const noexcept override
  {
    return {tile_data.data(), tile_data.size()};
  }

  DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp> evaluationStamps() const noexcept override
  {
    return {stamps.data(), stamps.size()};
  }

  /// Account allocated reference, grid and provenance capacities without P rows.
  NonLocalECPReplayStorageDiagnostics storageDiagnostics() const noexcept override
  {
    NonLocalECPReplayStorageDiagnostics result;
    result.point_count = point_data.size();
    result.segment_count = segment_data.size();
    result.tile_count = tile_data.size();
    result.evaluation_stamp_count = stamps.size();
    result.reference_position_bytes = references.capacity() * sizeof(double);
    result.point_bytes = point_data.capacity() * sizeof(NonLocalECPReplayPoint);
    result.provenance_bytes = segment_data.capacity() * sizeof(NonLocalECPReplaySegment) +
        tile_data.capacity() * sizeof(NonLocalECPReplayTile) +
        stamps.capacity() * sizeof(WaveFunctionComponent::EvaluationStamp);
    result.retained_numeric_bytes = result.reference_position_bytes + result.point_bytes + result.provenance_bytes;
    result.storage_fingerprint = reinterpret_cast<std::size_t>(references.data()) ^
        reinterpret_cast<std::size_t>(point_data.data()) ^ reinterpret_cast<std::size_t>(segment_data.data()) ^
        reinterpret_cast<std::size_t>(tile_data.data()) ^ reinterpret_cast<std::size_t>(stamps.data());
    return result;
  }
};

/** Capture the exact live energy traversal; the supplied parameter sink stays idle. */
class LegacyECPCapture final : public NonLocalECPDerivativeConsumer
{
public:
  /// Take the private reference configurations before any quadrature RNG use.
  explicit LegacyECPCapture(std::unique_ptr<LegacyECPRecord> record) : record_(std::move(record)) {}

  /// Reject incompatible requests without mutating state or activating the sink.
  void preflight(const NonLocalECPDerivativeContext& context,
                 DerivativeArrayView<const VJPCoefficientChannel> channels,
                 const ParameterReductionSink& sink) const override
  {
    if (state_ != State::IDLE || !record_)
      throw std::logic_error("Legacy ECP capture is a one-shot transaction");
    if (context.provider_id != record_->provider || context.schema_fingerprint != record_->schema ||
        context.parameter_version != record_->version || context.batch_ordinal != record_->batch ||
        context.sample_offset != record_->offset || context.sample_count != record_->samples ||
        context.localization != NonLocalECPLocalization::ORDINARY_LOCALITY || !context.uses_virtual_particles ||
        !context.scalar_relativistic || context.used_dense_derivative_fallback)
      throw std::invalid_argument("Legacy ECP capture requires matching scalar ordinary-locality context");
    if (channels.empty() || channels.size() > MAXIMUM_VJP_CHANNELS || sink.state() != DerivativeSinkState::IDLE)
      throw std::invalid_argument("Legacy ECP capture requires bounded channels and an idle sink");
    for (const auto& channel : channels)
    {
      const auto& coefficients = channel.coefficients;
      if (channel.product != DerivativeProduct::LOCAL_ENERGY_VJP ||
          channel.local_energy_term_mask != localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP) ||
          coefficients.provider_id != record_->provider || coefficients.schema_fingerprint != record_->schema ||
          coefficients.parameter_version != record_->version || coefficients.batch_ordinal != record_->batch ||
          coefficients.sample_offset != record_->offset || coefficients.values.size() != record_->samples)
        throw std::invalid_argument("Legacy ECP capture channel identity is incompatible");
      for (const auto& coefficient : coefficients.values)
        requireFiniteECP(coefficient);
    }
  }

  /// Bind one accepted traversal and reserve only point-sized capture storage.
  void begin(const NonLocalECPDerivativeContext& context,
             DerivativeArrayView<const VJPCoefficientChannel> channels, ParameterReductionSink& sink) override
  {
    preflight(context, channels, sink);
    if (!context.grid_fingerprint)
      throw std::invalid_argument("Legacy ECP capture requires an accepted grid fingerprint");
    checkedMultiply(context.expected_point_count, sizeof(NonLocalECPReplayPoint), "Legacy ECP capture points");
    record_->point_data.reserve(context.expected_point_count);
    expected_points_ = context.expected_point_count;
    record_->grid = context.grid_fingerprint;
    state_ = State::ACTIVE;
  }

  /// Copy one transient tile before its Hamiltonian backing storage is reused.
  void consume(const NonLocalECPDerivativeTile& tile) override
  {
    try
    {
      if (state_ != State::ACTIVE || tile.grid_fingerprint != record_->grid ||
          tile.tile_ordinal != record_->tile_data.size() || tile.virtual_particles.walkerCount() != record_->samples ||
          tile.bare_weights.size() != tile.virtual_particles.size() ||
          tile.complete_ratios.size() != tile.virtual_particles.size() ||
          tile.virtual_particles.size() > expected_points_ - record_->point_data.size())
        throw std::invalid_argument("Legacy ECP capture tile identity or extent is invalid");
      const auto point_offset = record_->point_data.size();
      const auto segment_offset = record_->segment_data.size();
      const auto stamp_offset = record_->stamps.size();
      for (std::size_t i = 0; i < tile.virtual_particles.segmentCount(); ++i)
      {
        const auto slice = tile.virtual_particles.slice(i);
        if (slice.walkerId() < 0 || static_cast<std::size_t>(slice.walkerId()) >= record_->samples ||
            slice.electronId() < 0 || static_cast<std::size_t>(slice.electronId()) >= record_->electrons)
          throw std::invalid_argument("Legacy ECP capture segment index is invalid");
        record_->segment_data.push_back({point_offset + slice.flatOffset(), slice.size(),
            static_cast<std::size_t>(slice.walkerId()), static_cast<std::size_t>(slice.electronId()), slice.isOnSphere(),
            slice.sourceCenterId() == VirtualParticleBatch::NO_SOURCE ? static_cast<std::size_t>(-1) :
                static_cast<std::size_t>(slice.sourceCenterId())});
        for (std::size_t p = 0; p < slice.size(); ++p)
        {
          NonLocalECPReplayPoint point;
          point.sample = slice.walkerId();
          point.electron = slice.electronId();
          for (int dimension = 0; dimension < 3; ++dimension)
            point.replacement_position[dimension] = slice.absolutePosition(p)[dimension];
          point.bare_weight = asDerivativeValue(tile.bare_weights[slice.flatOffset() + p]);
          point.complete_ratio = asDerivativeValue(tile.complete_ratios[slice.flatOffset() + p]);
          record_->point_data.push_back(point);
        }
      }
      record_->stamps.insert(record_->stamps.end(), tile.evaluation_stamps.begin(), tile.evaluation_stamps.end());
      record_->tile_data.push_back({tile.tile_ordinal, point_offset, tile.virtual_particles.size(), segment_offset,
          record_->segment_data.size() - segment_offset, stamp_offset, tile.evaluation_stamps.size(),
          tile.virtual_particles.fingerprint()});
      // Validate only this new tile here; end/bind perform the complete audit.
      for (std::size_t p = point_offset; p < record_->point_data.size(); ++p)
      {
        const auto& point = record_->point_data[p];
        for (double coordinate : point.replacement_position)
          requireFiniteECP({coordinate, 0});
        requireFiniteECP(point.bare_weight);
        requireFiniteECP(point.complete_ratio);
        requireFiniteECP(point.bare_weight * point.complete_ratio);
        if (point.bare_weight.imag() != 0 || point.complete_ratio.imag() != 0)
          throw std::invalid_argument("Legacy ECP capture requires real weights and ratios");
      }
    }
    catch (...)
    {
      abort();
      throw;
    }
  }

  /// Audit complete coverage once, after the final incremental tile check.
  void end() override
  {
    if (state_ != State::ACTIVE || record_->point_data.size() != expected_points_)
    {
      abort();
      throw std::logic_error("Legacy ECP capture traversal is incomplete");
    }
    validateECPRecord(*record_);
    state_ = State::COMPLETE;
  }

  /// Poison partial capture without publishing or changing the idle sink.
  void abort() noexcept override { state_ = State::POISONED; }

  /// Transfer complete accepted-grid ownership exactly once.
  std::unique_ptr<NonLocalECPJVPReplay> releaseJVPReplay() override
  {
    if (state_ != State::COMPLETE || !record_)
      throw std::logic_error("Legacy ECP capture has no completed unreleased replay");
    return std::move(record_);
  }

  /// Report capture-only storage separately from the adapter's shared P workspace.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    StreamingDerivativeStorageDiagnostics result;
    if (record_)
    {
      const auto storage = record_->storageDiagnostics();
      result.sample_count = record_->samples;
      result.sample_position_bytes = storage.reference_position_bytes;
      result.sample_auxiliary_bytes = storage.point_bytes + storage.provenance_bytes;
      result.retained_numeric_bytes = storage.retained_numeric_bytes;
      result.storage_fingerprint = storage.storage_fingerprint;
    }
    return result;
  }

private:
  enum class State
  {
    IDLE,
    ACTIVE,
    COMPLETE,
    POISONED
  };
  State state_ = State::IDLE;
  std::unique_ptr<LegacyECPRecord> record_;
  std::size_t expected_points_ = 0;
};

class ReferenceReplay final : public LegacyWaveFunctionReplay
{
public:
  ReferenceReplay(const RefVector<TrialWaveFunction>& evaluators, const RefVector<ParticleSet>& particles)
      : evaluators_(evaluators), particles_(particles)
  {}
  std::size_t sampleCount() const noexcept override { return evaluators_.size(); }
  LegacyWaveFunctionReplaySample prepareSample(std::size_t sample) override
  {
    return {evaluators_[sample], particles_[sample]};
  }
private:
  RefVector<TrialWaveFunction> evaluators_;
  RefVector<ParticleSet> particles_;
};

/// Convert either real- or complex-build wavefunction scalars without projection.
DerivativeValue asDerivativeValue(QMCTraits::ValueType value) noexcept
{
  return {static_cast<DerivativeReal>(std::real(value)), static_cast<DerivativeReal>(std::imag(value))};
}

/** Contract scalar legacy derivative rows immediately into bounded accumulators.
 *
 * The replay worker is exclusive to one adapter lane. All retained views share
 * the two legacy rows and three complex accumulators, plus an iteration snapshot.
 */
class LegacyStreamingDerivativeOperator final : public StreamingDerivativeOperator
{
public:
  LegacyStreamingDerivativeOperator(std::shared_ptr<const StructuredParameterSchema> schema,
                                    std::shared_ptr<const optimize::VariableSet> active,
                                    std::shared_ptr<const StructuredParameterSnapshot> snapshot,
                                    std::shared_ptr<LegacyWaveFunctionReplay> replay,
                                    std::shared_ptr<LegacyDerivativeWorkspace> workspace,
                                    std::size_t batch_ordinal,
                                    std::size_t sample_offset,
                                    std::size_t maximum_chunk_size,
                                    std::size_t maximum_scratch_bytes)
      : schema_(std::move(schema)),
        active_(std::move(active)),
        snapshot_(std::move(snapshot)),
        parameter_version_(snapshot_->version),
        replay_(std::move(replay)),
        workspace_(std::move(workspace)),
        batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset),
        chunk_plan_(*schema_, parameter_version_, maximum_chunk_size)
  {
    // Measure allocator capacities before invoking any fallible legacy callback.
    initializeStorageDiagnostics();
    if (storage_diagnostics_.retained_numeric_bytes > maximum_scratch_bytes)
      throw std::length_error("Legacy adapter allocated derivative scratch beyond its mandatory byte ceiling");
  }

  /// Advertise only the bounded host products implemented by the scalar bridge.
  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain       = ParameterScalarDomain::REAL64;
    result.result_scalar_domain          = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain              = ReductionDomain::CROWD_LOCAL;
    result.execution_domain              = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask        = localEnergyTermBit(LocalEnergyTerm::KINETIC) |
        (ecp_replay_ ? localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP) : 0);
    result.maximum_vjp_channels          = MAXIMUM_VJP_CHANNELS;
    result.maximum_parameter_chunk_size  = chunk_plan_.maximumChunkSize();
    result.maximum_sample_tile_size      = 1;
    result.fixed_parameter_scratch_vectors = 2 + MAXIMUM_VJP_CHANNELS;
    result.block_streaming               = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return *schema_; }
  std::size_t parameterVersion() const noexcept override { return parameter_version_; }
  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }
  std::size_t sampleOffset() const noexcept override { return sample_offset_; }
  std::size_t sampleCount() const noexcept override { return replay_->sampleCount(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return chunk_plan_; }

  /// Bind only a complete grid captured at this immutable model and sample state.
  void bindNonLocalECPJVPReplay(std::unique_ptr<NonLocalECPJVPReplay> replay) override
  {
    std::lock_guard lock(workspace_->mutex);
    if (ecp_replay_ || !replay || replay->providerId() != schema_->providerId() ||
        replay->schemaFingerprint() != schema_->fingerprint() || replay->parameterVersion() != parameter_version_ ||
        replay->batchOrdinal() != batch_ordinal_ || replay->sampleOffset() != sample_offset_ ||
        replay->sampleCount() != sampleCount() ||
        replay->localEnergyTermMask() != localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP))
      throw std::invalid_argument("Legacy ECP replay has incompatible identity or was already bound");
    validateECPRecord(*replay);
    for (std::size_t sample = 0; sample < sampleCount(); ++sample)
    {
      auto state = replay_->prepareSample(sample);
      const auto profile = validateECPProfile(state);
      if (sample > 0 && ecp_profile_ != profile)
        throw std::invalid_argument("Legacy ECP replay mixes fixed and active orbital profiles");
      ecp_profile_ = profile;
      if (profile == LegacyECPProfile::DETS &&
          !state.wavefunction.findMSD().front().get().usesAcceptedReferenceVirtualRatios())
        throw std::invalid_argument("Legacy coefficient ECP replay requires an enabled private evaluator");
      if (replay->electronCount() != state.particles.getTotalNum())
        throw std::invalid_argument("Legacy ECP replay electron extent differs from private evaluator");
      for (std::size_t electron = 0; electron < replay->electronCount(); ++electron)
        for (int dimension = 0; dimension < 3; ++dimension)
          if (state.particles.R[electron][dimension] !=
              replay->referencePositions()[(sample * replay->electronCount() + electron) * 3 + dimension])
            throw std::invalid_argument("Legacy ECP replay reference coordinates differ from private evaluator");
    }
    // Index segment ownership once: each action visits only its sample's knots.
    std::vector<std::size_t> offsets(sampleCount() + 1, 0);
    std::vector<std::size_t> indices(replay->segments().size());
    for (const auto& segment : replay->segments())
      ++offsets[segment.sample + 1];
    for (std::size_t sample = 1; sample < offsets.size(); ++sample)
      offsets[sample] += offsets[sample - 1];
    auto cursor = offsets;
    for (std::size_t segment = 0; segment < replay->segments().size(); ++segment)
      indices[cursor[replay->segments()[segment].sample]++] = segment;
    ecp_segment_offsets_ = std::move(offsets);
    ecp_segment_indices_ = std::move(indices);
    ecp_replay_ = std::move(replay);
  }

  /// Report the exact numeric buffers retained by the compatibility operator.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    auto result = storage_diagnostics_;
    if (ecp_replay_)
    {
      const auto grid = ecp_replay_->storageDiagnostics();
      result.sample_position_bytes = grid.reference_position_bytes;
      result.sample_auxiliary_bytes = grid.point_bytes + grid.provenance_bytes;
      result.retained_numeric_bytes = checkedAdd(result.retained_numeric_bytes, grid.retained_numeric_bytes,
                                                "Legacy ECP retained bytes");
      const auto index_bytes = checkedMultiply(checkedAdd(ecp_segment_offsets_.capacity(), ecp_segment_indices_.capacity(),
          "Legacy ECP segment indices"), sizeof(std::size_t), "Legacy ECP segment index bytes");
      result.sample_auxiliary_bytes = checkedAdd(result.sample_auxiliary_bytes, index_bytes, "Legacy ECP auxiliary bytes");
      result.retained_numeric_bytes = checkedAdd(result.retained_numeric_bytes, index_bytes, "Legacy ECP retained indices");
      result.storage_fingerprint ^= grid.storage_fingerprint;
      result.storage_fingerprint ^= reinterpret_cast<std::size_t>(ecp_segment_offsets_.data()) ^
          reinterpret_cast<std::size_t>(ecp_segment_indices_.data());
    }
    return result;
  }

protected:
  /// Evaluate one scalar row at a time and contract it into at most three channels.
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    std::lock_guard lock(workspace_->mutex);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
      std::fill(workspace_->channels[channel].begin(), workspace_->channels[channel].end(), DerivativeValue{});
    const bool requested_ecp = std::any_of(channels.begin(), channels.end(), [](const auto& channel) {
      return channel.product == DerivativeProduct::LOCAL_ENERGY_VJP &&
          (channel.local_energy_term_mask & localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP));
    });

    for (std::size_t sample = 0; sample < sampleCount(); ++sample)
    {
      evaluateRows(sample);
      for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
          DerivativeValue derivative = channels[channel].product == DerivativeProduct::SCORE_VJP
              ? asDerivativeValue(workspace_->score_row[parameter])
              : ((channels[channel].local_energy_term_mask & localEnergyTermBit(LocalEnergyTerm::KINETIC))
                     ? asDerivativeValue(workspace_->kinetic_row[parameter]) : DerivativeValue{});
          if (adjoint == DerivativeAdjoint::HERMITIAN)
            derivative = std::conj(derivative);
          workspace_->channels[channel][parameter] +=
              derivative * channels[channel].coefficients.values[sample];
        }
      if (requested_ecp)
      {
        if (ecp_profile_ != LegacyECPProfile::FIXED_SLATER)
        {
          forEachECPPoint(sample, [&](const auto& point) {
            evaluateECPRatioRow(sample, point);
            accumulateECPRow(channels, adjoint, sample, point.bare_weight);
          });
          continue;
        }
        DerivativeValue weight_sum{};
        forEachECPPoint(sample, [&](const auto& point) { weight_sum += point.bare_weight * point.complete_ratio; });
        // Subtract the reference score before reusing its row at virtual points.
        accumulateECPRow(channels, adjoint, sample, -weight_sum);
        forEachECPPoint(sample, [&](const auto& point) {
          if (point.complete_ratio != DerivativeValue{})
          {
            evaluateRows(sample, &point);
            accumulateECPRow(channels, adjoint, sample, point.bare_weight * point.complete_ratio);
          }
        });
      }
    }

    for (const ParameterChunkDescriptor& chunk : chunk_plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        sink.add(channel,
                 {chunk,
                  {workspace_->channels[channel].data() + chunk.parameter_offset, chunk.count}});
  }

  /// Evaluate and emit one scalar score-direction product per sample.
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    evaluateJVP(direction, sink, 0);
  }

  void evaluateLocalEnergyJVP(const StructuredParameterVectorConstView& direction,
                              std::uint32_t mask,
                              SampleProductSink& sink) const override
  {
    evaluateJVP(direction, sink, mask);
  }

private:
  void evaluateJVP(const StructuredParameterVectorConstView& direction,
                    SampleProductSink& sink, std::uint32_t mask) const
  {
    std::lock_guard lock(workspace_->mutex);
    for (std::size_t sample = 0; sample < sampleCount(); ++sample)
    {
      evaluateRows(sample);
      DerivativeValue product{}, reference{};
      for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
      {
        reference += asDerivativeValue(workspace_->score_row[parameter]) * direction.values()[parameter];
        if (mask & localEnergyTermBit(LocalEnergyTerm::KINETIC))
          product += asDerivativeValue(workspace_->kinetic_row[parameter]) * direction.values()[parameter];
      }
      if (mask == 0) product = reference;
      if (mask & localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP))
        forEachECPPoint(sample, [&](const auto& point) {
          if (ecp_profile_ != LegacyECPProfile::FIXED_SLATER)
          {
            evaluateECPRatioRow(sample, point);
            for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
              product += point.bare_weight * asDerivativeValue(workspace_->score_row[parameter]) *
                  direction.values()[parameter];
          }
          else if (point.complete_ratio != DerivativeValue{})
          {
            evaluateRows(sample, &point);
            DerivativeValue replacement{};
            for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
              replacement += asDerivativeValue(workspace_->score_row[parameter]) * direction.values()[parameter];
            product += point.bare_weight * point.complete_ratio * (replacement - reference);
          }
        });
      requireFiniteECP(product);
      const SampleProductTileDescriptor descriptor{schema_->providerId(),
                                                   schema_->fingerprint(),
                                                   parameter_version_,
                                                   batch_ordinal_,
                                                   sample_offset_ + sample,
                                                   1,
                                                   sample};
      sink.add({descriptor, {&product, 1}});
    }
  }

  /** Form d(Psi_q/Psi_ref) without ever evaluating a nodal full virtual TWF.
   * Baseline products have already consumed both P rows. First consume the
   * Jastrow difference, then reuse its source row for the unnormalized orbital
   * or accepted-reference CI derivative, including zero total virtual ratios.
   * Only six stack orbital values accompany the two existing parameter rows.
   */
  void evaluateECPRatioRow(std::size_t sample, const NonLocalECPReplayPoint& point) const
  {
    auto state = replay_->prepareSample(sample);
    validateMasses(state.particles);
    resetEvaluator(state.wavefunction, *active_, snapshot_->values);
    state.wavefunction.evaluateLog(state.particles);
    RotatedSPOs* rotated = ecp_profile_ == LegacyECPProfile::ORBITAL
        ? &dynamic_cast<RotatedSPOs&>(state.wavefunction.findSD().front().get().getPhi(
              state.particles.GroupID[point.electron])) : nullptr;
    std::array<QMCTraits::ValueType, 3> reference_storage, replacement_storage;
    SPOSet::ValueVector reference(reference_storage.data(), 3), replacement(replacement_storage.data(), 3);
    if (rotated) rotated->Phi_->evaluateValue(state.particles, point.electron, reference);
    workspace_->score_row = QMCTraits::ValueType{};
    double reference_jastrow_log = 0;
    for (const auto& component : state.wavefunction.getOrbitals())
      if (!component->isFermionic())
      {
        reference_jastrow_log += component->get_log_value().real();
        component->evaluateDerivativesWF(state.particles, *active_, workspace_->score_row);
      }
    for (int dimension = 0; dimension < 3; ++dimension)
      state.particles.R[point.electron][dimension] = point.replacement_position[dimension];
    state.particles.update();
    if (rotated) rotated->Phi_->evaluateValue(state.particles, point.electron, replacement);
    state.particles.G = QMCTraits::GradType{};
    state.particles.L = QMCTraits::ValueType{};
    workspace_->kinetic_row = QMCTraits::ValueType{};
    double replacement_jastrow_log = 0;
    for (const auto& component : state.wavefunction.getOrbitals())
      if (!component->isFermionic())
      {
        replacement_jastrow_log += component->evaluateLog(state.particles, state.particles.G, state.particles.L).real();
        component->evaluateDerivativesWF(state.particles, *active_, workspace_->kinetic_row);
      }
    requireFiniteECP({reference_jastrow_log, 0});
    requireFiniteECP({replacement_jastrow_log, 0});
    const double jastrow_ratio = std::exp(replacement_jastrow_log - reference_jastrow_log);
    requireFiniteECP({jastrow_ratio, 0});
    for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
      workspace_->kinetic_row[parameter] = point.complete_ratio.real() *
          (workspace_->kinetic_row[parameter] - workspace_->score_row[parameter]);
    workspace_->score_row = QMCTraits::ValueType{};
    if (rotated)
      rotated->evaluateOneOccupiedRatioDerivatives(reference, replacement, *active_, workspace_->score_row);
    else
      state.wavefunction.findMSD().front().get().evaluateAcceptedReferenceRatioDerivatives(
          state.particles, point.electron, point.electron, *active_,
          {workspace_->score_row.data(), workspace_->score_row.size()});
    for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
    {
      workspace_->score_row[parameter] = jastrow_ratio * workspace_->score_row[parameter] +
          workspace_->kinetic_row[parameter];
      requireFiniteECP(asDerivativeValue(workspace_->score_row[parameter]));
    }
  }

  /// Reuse the two legacy full-P rows for one evaluator configuration.
  void evaluateRows(std::size_t sample, const NonLocalECPReplayPoint* point = nullptr) const
  {
    auto state = replay_->prepareSample(sample);
    if (point)
    {
      for (int dimension = 0; dimension < 3; ++dimension)
        state.particles.R[point->electron][dimension] = point->replacement_position[dimension];
      state.particles.update();
    }
    validateMasses(state.particles);
    resetEvaluator(state.wavefunction, *active_, snapshot_->values);
    state.wavefunction.evaluateLog(state.particles);
    workspace_->score_row   = QMCTraits::ValueType{};
    workspace_->kinetic_row = QMCTraits::ValueType{};
    if (point)
      state.wavefunction.evaluateDerivativesWF(state.particles, *active_, workspace_->score_row);
    else
      state.wavefunction.evaluateDerivatives(state.particles, *active_, workspace_->score_row, workspace_->kinetic_row);
  }

  /// Indexed traversal is O(owned segments + owned points), not O(samples*Q).
  template<class Action>
  void forEachECPPoint(std::size_t sample, Action action) const
  {
    for (std::size_t index = ecp_segment_offsets_[sample]; index < ecp_segment_offsets_[sample + 1]; ++index)
    {
      const auto& segment = ecp_replay_->segments()[ecp_segment_indices_[index]];
      for (std::size_t p = segment.point_offset; p < segment.point_offset + segment.point_count; ++p)
        action(ecp_replay_->points()[p]);
    }
  }

  /// Contract only explicitly requested ECP channels into the existing P buffers.
  void accumulateECPRow(DerivativeArrayView<const VJPCoefficientChannel> channels,
                        DerivativeAdjoint adjoint, std::size_t sample, DerivativeValue weight) const
  {
    requireFiniteECP(weight);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
      if (channels[channel].product == DerivativeProduct::LOCAL_ENERGY_VJP &&
          (channels[channel].local_energy_term_mask & localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)))
        for (std::size_t parameter = 0; parameter < schema_->parameterCount(); ++parameter)
        {
          auto derivative = weight * asDerivativeValue(workspace_->score_row[parameter]);
          requireFiniteECP(derivative);
          if (adjoint == DerivativeAdjoint::HERMITIAN) derivative = std::conj(derivative);
          workspace_->channels[channel][parameter] += derivative * channels[channel].coefficients.values[sample];
        }
  }

  /// Capture the retained-buffer byte count after all construction allocations.
  void initializeStorageDiagnostics()
  {
    storage_diagnostics_.parameter_count = schema_->parameterCount();
    storage_diagnostics_.sample_count    = sampleCount();
#ifdef QMC_COMPLEX
    storage_diagnostics_.complex_parameter_vectors = 2 + MAXIMUM_VJP_CHANNELS;
#else
    storage_diagnostics_.real_parameter_vectors    = 2;
    storage_diagnostics_.complex_parameter_vectors = MAXIMUM_VJP_CHANNELS;
#endif
    const std::size_t legacy_row_entries = checkedAdd(workspace_->score_row.capacity(), workspace_->kinetic_row.capacity(),
                                                      "Legacy derivative row capacity");
    const std::size_t legacy_row_bytes = checkedMultiply(
        legacy_row_entries, sizeof(QMCTraits::ValueType), "Legacy derivative row capacity bytes");
    std::size_t channel_entries = 0;
    for (const auto& channel : workspace_->channels)
      channel_entries = checkedAdd(channel_entries, channel.capacity(), "Legacy derivative channel capacity");
    const std::size_t channel_bytes = checkedMultiply(
        channel_entries, sizeof(DerivativeValue), "Legacy derivative channel capacity bytes");
    storage_diagnostics_.parameter_scratch_bytes = checkedAdd(
        legacy_row_bytes, channel_bytes, "Legacy retained derivative scratch");
    storage_diagnostics_.retained_numeric_bytes  = storage_diagnostics_.parameter_scratch_bytes;
    storage_diagnostics_.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(workspace_.get());
    storage_diagnostics_.shared_workspace_bytes = storage_diagnostics_.parameter_scratch_bytes;
    std::size_t fingerprint = reinterpret_cast<std::size_t>(workspace_->score_row.data());
    fingerprint ^= reinterpret_cast<std::size_t>(workspace_->kinetic_row.data()) + std::size_t{0x9e3779b9U};
    for (const auto& channel : workspace_->channels)
      fingerprint ^= reinterpret_cast<std::size_t>(channel.data()) + (fingerprint << 6) + (fingerprint >> 2);
    storage_diagnostics_.storage_fingerprint = fingerprint == 0 ? 1 : fingerprint;
  }

  std::shared_ptr<const StructuredParameterSchema> schema_;
  std::shared_ptr<const optimize::VariableSet> active_;
  std::shared_ptr<const StructuredParameterSnapshot> snapshot_;
  std::unique_ptr<NonLocalECPJVPReplay> ecp_replay_;
  LegacyECPProfile ecp_profile_ = LegacyECPProfile::FIXED_SLATER;
  std::vector<std::size_t> ecp_segment_offsets_, ecp_segment_indices_;
  std::size_t parameter_version_ = 0;
  std::shared_ptr<LegacyWaveFunctionReplay> replay_;
  std::shared_ptr<LegacyDerivativeWorkspace> workspace_;
  std::size_t batch_ordinal_ = 0;
  std::size_t sample_offset_ = 0;
  ParameterChunkPlan chunk_plan_;
  StreamingDerivativeStorageDiagnostics storage_diagnostics_;
};

} // namespace

LegacyWaveFunctionAdapter::LegacyWaveFunctionAdapter(std::string provider_id,
                                                     TrialWaveFunction& registration_source,
                                                     LegacyWaveFunctionAdapterOptions options)
    : options_(options)
{
  if (provider_id.empty())
    throw std::invalid_argument("Legacy adapter provider identity must not be empty");
  if (options_.maximum_parameter_count == 0 || options_.maximum_derivative_scratch_bytes == 0 ||
      options_.maximum_parameter_chunk_size == 0)
    throw std::invalid_argument("Legacy adapter requires positive parameter, scratch, and chunk limits");
  if (!registration_source.extractStructuredParameterProviders().empty())
    throw std::invalid_argument("Legacy adapter cannot aggregate existing structured parameter providers");

  optimize::VariableSet registration_template;
  registration_source.checkInVariables(registration_template);
  registration_template.resetIndex();
  if (registration_template.size_of_active() <= 0)
    throw std::invalid_argument("Legacy adapter requires at least one active parameter");
  for (std::size_t location = 0; location < registration_template.size(); ++location)
    if (!isFiniteDouble(registration_template[static_cast<int>(location)]))
      throw std::invalid_argument("Legacy adapter registration contains a non-finite parameter");

  const auto [blocks, active_locations] = makeBlocks(registration_template);
  const std::size_t parameter_count     = active_locations.size();
  if (parameter_count > options_.maximum_parameter_count)
    throw std::length_error("Legacy adapter parameter count exceeds its mandatory ceiling");
  if (requiredScratchBytes(parameter_count) > options_.maximum_derivative_scratch_bytes)
    throw std::length_error("Legacy adapter derivative scratch exceeds its mandatory byte ceiling");

  parameter_values_.resize(parameter_count);
  for (std::size_t index = 0; index < parameter_count; ++index)
    parameter_values_[index] = registration_template[static_cast<int>(active_locations[index])];
  registration_template_ =
      std::make_shared<const optimize::VariableSet>(std::move(registration_template));
  schema_ = std::make_shared<const StructuredParameterSchema>(std::move(provider_id), blocks);
  workspace_ = std::make_shared<LegacyDerivativeWorkspace>(parameter_count);
}

StructuredParameterSnapshot LegacyWaveFunctionAdapter::snapshotParameters() const
{
  std::lock_guard lock(state_mutex_);
  return {schema_->fingerprint(), parameter_version_, parameter_values_};
}

std::size_t LegacyWaveFunctionAdapter::publishParameters(const StructuredParameterSnapshot& candidate,
                                                         std::size_t expected_version)
{
  if (candidate.schema_fingerprint != schema_->fingerprint())
    throw std::invalid_argument("Legacy adapter candidate has the wrong schema fingerprint");
  if (candidate.values.size() != schema_->parameterCount())
    throw std::invalid_argument("Legacy adapter candidate has the wrong parameter extent");
  if (std::any_of(candidate.values.begin(), candidate.values.end(),
                  [](double value) { return !isFiniteDouble(value); }))
    throw std::invalid_argument("Legacy adapter candidate contains a non-finite value");

  // Copy before locking so allocation failure cannot affect the authoritative state.
  std::vector<double> replacement(candidate.values);
  std::lock_guard lock(state_mutex_);
  if (expected_version != parameter_version_ || candidate.version != expected_version)
    throw std::runtime_error("Legacy adapter candidate parameter version is stale");
  if (parameter_version_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Legacy adapter parameter version overflow");
  parameter_values_.swap(replacement);
  return ++parameter_version_;
}

std::size_t LegacyWaveFunctionAdapter::commitPreparedParameters(StructuredParameterSnapshot&& candidate) noexcept
{
  std::lock_guard lock(state_mutex_);
  assert(candidate.schema_fingerprint == schema_->fingerprint());
  assert(candidate.values.size() == schema_->parameterCount());
  assert(candidate.version == parameter_version_ && parameter_version_ < std::numeric_limits<std::size_t>::max());
  parameter_values_.swap(candidate.values);
  return ++parameter_version_;
}

std::unique_ptr<StreamingDerivativeOperator> LegacyWaveFunctionAdapter::makeDerivativeOperator(
    const RefVector<TrialWaveFunction>& evaluators,
    const RefVector<ParticleSet>& particle_sets,
    std::size_t batch_ordinal,
    std::size_t sample_offset) const
{
  if (evaluators.size() != particle_sets.size())
    throw std::invalid_argument("Legacy adapter evaluator and particle batches have different sizes");
  if (sample_offset > std::numeric_limits<std::size_t>::max() - evaluators.size())
    throw std::overflow_error("Legacy adapter sample interval overflow");

  // Capability and resource failures occur before any legacy evaluator callback.
  if (schema_->parameterCount() > options_.maximum_parameter_count ||
      requiredScratchBytes(schema_->parameterCount()) > options_.maximum_derivative_scratch_bytes)
    throw std::length_error("Legacy adapter derivative request exceeds configured resource ceilings");
  auto snapshot = std::make_shared<const StructuredParameterSnapshot>(snapshotParameters());
  // Preserve eager validation for the original exclusive-reference API.
  for (std::size_t sample = 0; sample < evaluators.size(); ++sample)
    installSnapshot(evaluators[sample], particle_sets[sample], *snapshot);
  return makeDerivativeOperator(std::make_shared<ReferenceReplay>(evaluators, particle_sets),
                                std::move(snapshot), batch_ordinal, sample_offset);
}

void LegacyWaveFunctionAdapter::resetPrivateParameters(TrialWaveFunction& evaluator,
                                                       const StructuredParameterSnapshot& snapshot) const
{
  validateSnapshot(snapshot, *schema_);
  resetEvaluator(evaluator, *registration_template_, snapshot.values);
}

void LegacyWaveFunctionAdapter::installSnapshot(TrialWaveFunction& evaluator,
                                               ParticleSet& particles,
                                               const StructuredParameterSnapshot& snapshot) const
{
  validateMasses(particles);
  resetPrivateParameters(evaluator, snapshot);
  evaluator.evaluateLog(particles);
}

std::unique_ptr<StreamingDerivativeOperator> LegacyWaveFunctionAdapter::makeDerivativeOperator(
    std::shared_ptr<LegacyWaveFunctionReplay> replay,
    std::shared_ptr<const StructuredParameterSnapshot> snapshot,
    std::size_t batch_ordinal,
    std::size_t sample_offset) const
{
  if (!replay || !snapshot)
    throw std::invalid_argument("Legacy adapter replay and snapshot must not be null");
  validateSnapshot(*snapshot, *schema_);
  if (sample_offset > std::numeric_limits<std::size_t>::max() - replay->sampleCount())
    throw std::overflow_error("Legacy adapter sample interval overflow");
  return std::make_unique<LegacyStreamingDerivativeOperator>(
      schema_, registration_template_, std::move(snapshot), std::move(replay), workspace_, batch_ordinal, sample_offset,
      std::min(options_.maximum_parameter_chunk_size, schema_->parameterCount()),
      options_.maximum_derivative_scratch_bytes);
}

void LegacyWaveFunctionAdapter::validateNonLocalECPProfile(TrialWaveFunction& evaluator, ParticleSet& particles) const
{
  validateECPProfile({evaluator, particles});
}

std::unique_ptr<NonLocalECPDerivativeConsumer> LegacyWaveFunctionAdapter::makeNonLocalECPDerivativeConsumer(
    std::shared_ptr<LegacyWaveFunctionReplay> replay,
    std::shared_ptr<const StructuredParameterSnapshot> snapshot,
    std::size_t batch_ordinal,
    std::size_t sample_offset) const
{
  if (!replay || !snapshot)
    throw std::invalid_argument("Legacy ECP capture requires replay and snapshot");
  validateSnapshot(*snapshot, *schema_);
  checkedAdd(sample_offset, replay->sampleCount(), "Legacy ECP sample interval");
  auto record = std::make_unique<LegacyECPRecord>();
  record->provider = schema_->providerId();
  record->schema = schema_->fingerprint();
  record->version = snapshot->version;
  record->batch = batch_ordinal;
  record->offset = sample_offset;
  record->samples = replay->sampleCount();
  std::lock_guard lock(workspace_->mutex);
  for (std::size_t sample = 0; sample < replay->sampleCount(); ++sample)
  {
    auto state = replay->prepareSample(sample);
    if (validateECPProfile(state) == LegacyECPProfile::DETS &&
        !state.wavefunction.findMSD().front().get().usesAcceptedReferenceVirtualRatios())
      throw std::invalid_argument("Legacy coefficient ECP capture requires an enabled private evaluator");
    validateMasses(state.particles);
    if (sample == 0)
    {
      record->electrons = state.particles.getTotalNum();
      record->references.reserve(checkedMultiply(checkedMultiply(record->samples, record->electrons,
                                                                "Legacy ECP references"), 3,
                                                 "Legacy ECP Cartesian references"));
    }
    if (record->electrons != state.particles.getTotalNum())
      throw std::invalid_argument("Legacy ECP capture requires equal electron extents");
    // Registration checking excludes active variables outside admitted Jastrows/rotations.
    validateEvaluatorRegistration(state.wavefunction, *registration_template_);
    for (const auto& position : state.particles.R)
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        requireFiniteECP({position[dimension], 0});
        record->references.push_back(position[dimension]);
      }
  }
  return std::make_unique<LegacyECPCapture>(std::move(record));
}

} // namespace qmcplusplus::wftrain
