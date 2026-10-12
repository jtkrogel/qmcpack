//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerECPContraction.cpp
 * @brief Checked flattened-tile copy and serial bounded ECP contraction.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerECPContraction.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDerivativeMath.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace qmcplusplus::psiformer
{
namespace
{

std::size_t checkedAdd(std::size_t left,
                       std::size_t right,
                       const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(description);
  return left + right;
}

std::size_t checkedProduct(std::size_t left,
                           std::size_t right,
                           const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(description);
  return left * right;
}

struct AddressRange
{
  std::uintptr_t begin;
  std::uintptr_t end;
};

AddressRange makeAddressRange(const void* pointer,
                              std::size_t bytes,
                              const char* description)
{
  if (bytes == 0)
    return {0, 0};
  if (pointer == nullptr)
    throw std::invalid_argument(std::string("PsiFormer ECP null ") + description);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
    throw std::overflow_error(std::string("PsiFormer ECP address overflow in ") +
                              description);
  return {begin, begin + bytes};
}

bool overlaps(AddressRange left, AddressRange right) noexcept
{
  return left.begin != left.end && right.begin != right.end &&
      left.begin < right.end && right.begin < left.end;
}

void requireDisjoint(const void* left,
                     std::size_t left_bytes,
                     const void* right,
                     std::size_t right_bytes,
                     const char* description)
{
  if (overlaps(makeAddressRange(left, left_bytes, description),
               makeAddressRange(right, right_bytes, description)))
    throw std::invalid_argument(std::string("PsiFormer ECP storage aliases ") +
                                description);
}

template<class T>
void storeSerialized(std::byte* storage,
                     std::size_t offset,
                     std::size_t index,
                     const T& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(storage + offset + index * sizeof(T), &value, sizeof(T));
}

constexpr std::size_t requiredECPStorageAlignment() noexcept
{
  std::size_t alignment = alignof(PsiFormerECPTileMetadata);
  alignment = alignment < alignof(std::size_t) ? alignof(std::size_t) : alignment;
  alignment = alignment < alignof(std::uint8_t) ? alignof(std::uint8_t) : alignment;
  alignment = alignment < alignof(double) ? alignof(double) : alignment;
  alignment = alignment < alignof(PsiFormerECPEvaluationStampRecord)
      ? alignof(PsiFormerECPEvaluationStampRecord)
      : alignment;
  alignment = alignment < alignof(std::uint32_t) ? alignof(std::uint32_t) : alignment;
  return alignment;
}

void requireFinite(double value, const char* description)
{
  if (!device_math::isFiniteBinary64(value))
    throw std::invalid_argument(std::string("PsiFormer ECP ") + description +
                                " must be finite");
}

void requireFinite(wftrain::DerivativeValue value, const char* description)
{
  requireFinite(value.real(), description);
  requireFinite(value.imag(), description);
}

void storeSerializedComplex(std::byte* storage,
                            std::size_t offset,
                            std::size_t complex_index,
                            wftrain::DerivativeValue value) noexcept
{
  storeSerialized(storage, offset, 2 * complex_index, value.real());
  storeSerialized(storage, offset, 2 * complex_index + 1, value.imag());
}

std::uint32_t requiredECPFlags() noexcept
{
  return ECP_ORDINARY_LOCALITY | ECP_USES_VIRTUAL_PARTICLES |
      ECP_SCALAR_RELATIVISTIC;
}

void validateStorage(const PsiFormerDerivativePlan& plan,
                     const void* storage,
                     std::size_t storage_bytes)
{
  validatePsiFormerDerivativePlan(plan);
  if (plan.ecp_tile.total_bytes == 0)
    throw std::invalid_argument("PsiFormer ECP contraction requires a prepared ECP tile");
  if (storage == nullptr || storage_bytes < plan.ecp_tile.total_bytes)
    throw std::invalid_argument("PsiFormer ECP tile storage is too small");
  if (reinterpret_cast<std::uintptr_t>(storage) %
          requiredECPStorageAlignment() !=
      0)
    throw std::invalid_argument(
        "PsiFormer ECP tile storage does not meet its maximum typed alignment");
}

} // namespace

void PsiFormerCopiedECPTileView::copyAt(void* destination,
                                        std::size_t bytes,
                                        std::size_t offset,
                                        const char* description) const
{
  if (destination == nullptr || offset > layout_.total_bytes ||
      bytes > layout_.total_bytes - offset)
    throw std::out_of_range(std::string("PsiFormer copied ECP ") + description +
                            " is out of range");
  std::memcpy(destination, storage_ + offset, bytes);
}

PsiFormerECPTileMetadata PsiFormerCopiedECPTileView::metadata() const
{
  PsiFormerECPTileMetadata value;
  copyAt(&value, sizeof(value), layout_.metadata_offset_bytes, "metadata");
  return value;
}

std::size_t PsiFormerCopiedECPTileView::segmentOffset(std::size_t index) const
{
  if (index > layout_.maximum_segment_count)
    throw std::out_of_range("PsiFormer copied ECP segment offset is out of range");
  std::size_t value;
  copyAt(&value, sizeof(value),
         layout_.segment_offsets_offset_bytes + index * sizeof(value),
         "segment offset");
  return value;
}

std::size_t PsiFormerCopiedECPTileView::segmentWalkerId(std::size_t index) const
{
  if (index >= layout_.maximum_segment_count)
    throw std::out_of_range("PsiFormer copied ECP segment walker is out of range");
  std::size_t value;
  copyAt(&value, sizeof(value),
         layout_.segment_walker_ids_offset_bytes + index * sizeof(value),
         "segment walker");
  return value;
}

std::size_t PsiFormerCopiedECPTileView::segmentElectronId(std::size_t index) const
{
  if (index >= layout_.maximum_segment_count)
    throw std::out_of_range("PsiFormer copied ECP segment electron is out of range");
  std::size_t value;
  copyAt(&value, sizeof(value),
         layout_.segment_electron_ids_offset_bytes + index * sizeof(value),
         "segment electron");
  return value;
}

std::size_t PsiFormerCopiedECPTileView::segmentSourceCenterId(
    std::size_t index) const
{
  if (index >= layout_.maximum_segment_count)
    throw std::out_of_range("PsiFormer copied ECP source center is out of range");
  std::size_t value;
  copyAt(&value, sizeof(value),
         layout_.segment_source_center_ids_offset_bytes + index * sizeof(value),
         "source center");
  return value;
}

std::uint8_t PsiFormerCopiedECPTileView::segmentFlags(std::size_t index) const
{
  if (index >= layout_.maximum_segment_count)
    throw std::out_of_range("PsiFormer copied ECP segment flags are out of range");
  std::uint8_t value;
  copyAt(&value, sizeof(value), layout_.segment_flags_offset_bytes + index,
         "segment flags");
  return value;
}

double PsiFormerCopiedECPTileView::referencePosition(std::size_t index) const
{
  const std::size_t capacity =
      layout_.maximum_sample_count * layout_.electron_count * std::size_t{3};
  if (index >= capacity)
    throw std::out_of_range("PsiFormer copied ECP reference position is out of range");
  double value;
  copyAt(&value, sizeof(value),
         layout_.reference_positions_offset_bytes + index * sizeof(value),
         "reference position");
  return value;
}

std::array<double, 3> PsiFormerCopiedECPTileView::replacementPosition(
    std::size_t point) const
{
  if (point >= layout_.maximum_point_count)
    throw std::out_of_range("PsiFormer copied ECP replacement position is out of range");
  std::array<double, 3> value;
  copyAt(value.data(), sizeof(value),
         layout_.replacement_positions_offset_bytes +
             3 * point * sizeof(double),
         "replacement position");
  return value;
}

double PsiFormerCopiedECPTileView::bareWeight(std::size_t point) const
{
  if (point >= layout_.maximum_point_count)
    throw std::out_of_range("PsiFormer copied ECP bare weight is out of range");
  double value;
  copyAt(&value, sizeof(value),
         layout_.bare_weights_offset_bytes + point * sizeof(value),
         "bare weight");
  return value;
}

wftrain::DerivativeValue PsiFormerCopiedECPTileView::completeRatio(
    std::size_t point) const
{
  if (point >= layout_.maximum_point_count)
    throw std::out_of_range("PsiFormer copied ECP complete ratio is out of range");
  std::array<double, 2> value;
  copyAt(value.data(), sizeof(value),
         layout_.complete_ratios_offset_bytes + 2 * point * sizeof(double),
         "complete ratio");
  return {value[0], value[1]};
}

PsiFormerECPEvaluationStampRecord
PsiFormerCopiedECPTileView::evaluationStamp(std::size_t index) const
{
  if (index >= layout_.maximum_evaluation_stamp_count)
    throw std::out_of_range("PsiFormer copied ECP stamp is out of range");
  PsiFormerECPEvaluationStampRecord value;
  copyAt(&value, sizeof(value),
         layout_.evaluation_stamps_offset_bytes + index * sizeof(value),
         "evaluation stamp");
  return value;
}

wftrain::DerivativeValue PsiFormerCopiedECPTileView::coefficient(
    std::size_t index) const
{
  const std::size_t capacity =
      layout_.maximum_channel_count * layout_.maximum_sample_count;
  if (index >= capacity)
    throw std::out_of_range("PsiFormer copied ECP coefficient is out of range");
  std::array<double, 2> value;
  copyAt(value.data(), sizeof(value),
         layout_.coefficients_offset_bytes + 2 * index * sizeof(double),
         "coefficient");
  return {value[0], value[1]};
}

wftrain::DerivativeValue PsiFormerCopiedECPTileView::sampleWeightSum(
    std::size_t sample) const
{
  if (sample >= layout_.maximum_sample_count)
    throw std::out_of_range("PsiFormer copied ECP sample sum is out of range");
  std::array<double, 2> value;
  copyAt(value.data(), sizeof(value),
         layout_.sample_weight_sums_offset_bytes +
             2 * sample * sizeof(double),
         "sample weight sum");
  return {value[0], value[1]};
}

std::uint32_t PsiFormerCopiedECPTileView::pointStatus(std::size_t point) const
{
  if (point >= layout_.maximum_point_count)
    throw std::out_of_range("PsiFormer copied ECP point status is out of range");
  std::uint32_t value;
  copyAt(&value, sizeof(value),
         layout_.point_status_offset_bytes + point * sizeof(value),
         "point status");
  return value;
}

PsiFormerCopiedECPTileView PsiFormerECPContraction::copiedTileView() const
{
  return {ecp_tile_storage_, plan_.ecp_tile};
}

PsiFormerECPContraction::PsiFormerECPContraction(
    const PsiFormerDerivativePlan& plan,
    void* ecp_tile_storage,
    std::size_t ecp_tile_storage_bytes,
    double* response,
    std::size_t response_count,
    double* accumulators,
    std::size_t accumulator_fp64_elements,
    PsiFormerECPScoreEvaluator& evaluator)
    : plan_(plan),
      ecp_tile_storage_(static_cast<std::byte*>(ecp_tile_storage)),
      ecp_tile_storage_bytes_(ecp_tile_storage_bytes),
      response_(response),
      response_count_(response_count),
      accumulators_(accumulators),
      accumulator_fp64_elements_(accumulator_fp64_elements),
      evaluator_(evaluator)
{
  validateStorage(plan_, ecp_tile_storage_, ecp_tile_storage_bytes_);
  if (response_ == nullptr || response_count_ != plan_.parameter_count)
    throw std::invalid_argument("PsiFormer ECP contraction requires one canonical response[P]");
  if (accumulators_ == nullptr ||
      accumulator_fp64_elements_ < plan_.accumulators.total_fp64_elements)
    throw std::invalid_argument("PsiFormer ECP contraction accumulator storage is too small");
  const std::size_t response_bytes = checkedProduct(
      response_count_, sizeof(double), "PsiFormer ECP response bytes overflowed");
  const std::size_t accumulator_bytes = checkedProduct(
      plan_.accumulators.total_fp64_elements, sizeof(double),
      "PsiFormer ECP accumulator bytes overflowed");
  requireDisjoint(ecp_tile_storage_, plan_.ecp_tile.total_bytes,
                  response_, response_bytes, "ECP_TILE and response");
  requireDisjoint(ecp_tile_storage_, plan_.ecp_tile.total_bytes,
                  accumulators_, accumulator_bytes,
                  "ECP_TILE and accumulators");
  requireDisjoint(response_, response_bytes, accumulators_, accumulator_bytes,
                  "response and accumulators");
  reference_positions_.resize(checkedProduct(
      checkedProduct(plan_.ecp_tile.maximum_sample_count,
                     plan_.electron_count,
                     "PsiFormer ECP prepared reference extent overflowed"),
      std::size_t{3},
      "PsiFormer ECP prepared reference extent overflowed"));
  coefficients_.resize(checkedProduct(
      plan_.ecp_tile.maximum_channel_count,
      plan_.ecp_tile.maximum_sample_count,
      "PsiFormer ECP prepared coefficient extent overflowed"));
  sample_weight_sums_.resize(plan_.ecp_tile.maximum_sample_count);
  std::memset(ecp_tile_storage_, 0, plan_.ecp_tile.total_bytes);
}

void PsiFormerECPContraction::begin(
    const PsiFormerECPContractionContext& context,
    wftrain::DerivativeArrayView<const double> reference_positions,
    wftrain::DerivativeArrayView<const wftrain::DerivativeValue> coefficients)
{
  if (state_ != State::IDLE)
    throw std::logic_error("PsiFormer ECP contraction can begin only from idle");
  if (context.grid_fingerprint == 0 || context.producer_fingerprint == 0)
    throw std::invalid_argument("PsiFormer ECP contraction fingerprints must be nonzero");
  if (context.sample_count == 0 ||
      context.sample_count > plan_.ecp_tile.maximum_sample_count ||
      context.channel_count == 0 || context.channel_count > 3 ||
      context.channel_count > plan_.ecp_tile.maximum_channel_count ||
      context.adjoint != wftrain::DerivativeAdjoint::TRANSPOSE)
    throw std::invalid_argument("PsiFormer ECP contraction context exceeds the ordinary-locality plan");

  const std::size_t reference_count = checkedProduct(
      checkedProduct(context.sample_count, plan_.electron_count,
                     "PsiFormer ECP reference extent overflowed"),
      3, "PsiFormer ECP reference extent overflowed");
  const std::size_t coefficient_count = checkedProduct(
      context.channel_count, context.sample_count,
      "PsiFormer ECP coefficient extent overflowed");
  if (reference_positions.size() != reference_count ||
      (reference_count != 0 && reference_positions.data() == nullptr) ||
      coefficients.size() != coefficient_count ||
      (coefficient_count != 0 && coefficients.data() == nullptr))
    throw std::invalid_argument("PsiFormer ECP contraction inputs have inconsistent extents");

  const std::size_t response_bytes = response_count_ * sizeof(double);
  const std::size_t accumulator_bytes =
      plan_.accumulators.total_fp64_elements * sizeof(double);
  const std::size_t reference_bytes = reference_count * sizeof(double);
  const std::size_t coefficient_bytes =
      coefficient_count * sizeof(wftrain::DerivativeValue);
  for (const auto& output : std::array<AddressRange, 3>{
           makeAddressRange(ecp_tile_storage_, plan_.ecp_tile.total_bytes,
                            "begin outputs"),
           makeAddressRange(response_, response_bytes, "begin outputs"),
           makeAddressRange(accumulators_, accumulator_bytes, "begin outputs")})
  {
    if (overlaps(makeAddressRange(reference_positions.data(), reference_bytes,
                                  "reference input"), output) ||
        overlaps(makeAddressRange(coefficients.data(), coefficient_bytes,
                                  "coefficient input"), output))
      throw std::invalid_argument("PsiFormer ECP begin input aliases contraction storage");
  }

  // Preflight before the first write.
  for (double coordinate : reference_positions)
    requireFinite(coordinate, "reference coordinate");
  for (wftrain::DerivativeValue coefficient : coefficients)
    requireFinite(coefficient, "coefficient");

  std::copy(reference_positions.begin(), reference_positions.end(),
            reference_positions_.begin());
  for (std::size_t index = 0; index < reference_positions.size(); ++index)
    storeSerialized(ecp_tile_storage_,
                    plan_.ecp_tile.reference_positions_offset_bytes, index,
                    reference_positions[index]);
  std::copy(coefficients.begin(), coefficients.end(), coefficients_.begin());
  for (std::size_t index = 0; index < coefficients.size(); ++index)
    storeSerializedComplex(
        ecp_tile_storage_, plan_.ecp_tile.coefficients_offset_bytes, index,
        coefficients[index]);
  std::fill(sample_weight_sums_.begin(), sample_weight_sums_.end(),
            wftrain::DerivativeValue{});
  std::memset(ecp_tile_storage_ +
                  plan_.ecp_tile.sample_weight_sums_offset_bytes,
              0, 2 * plan_.ecp_tile.maximum_sample_count * sizeof(double));
  std::memset(ecp_tile_storage_ + plan_.ecp_tile.point_status_offset_bytes, 0,
              plan_.ecp_tile.maximum_point_count * sizeof(std::uint32_t));
  std::fill(accumulators_,
            accumulators_ + plan_.accumulators.total_fp64_elements,
            0.0);

  context_ = context;
  expected_tile_ordinal_ = 0;
  consumed_point_count_ = 0;
  state_ = State::ACTIVE;
}

void PsiFormerECPContraction::requireActive(const char* operation) const
{
  if (state_ != State::ACTIVE)
    throw std::logic_error(std::string("PsiFormer ECP ") + operation +
                           " requires an active contraction");
}

PsiFormerECPTileLaunchRecord PsiFormerECPContraction::consume(
    const wftrain::NonLocalECPDerivativeTile& tile,
    wftrain::DerivativeArrayView<const PsiFormerECPEvaluationStampRecord>
        validated_stamp_records)
{
  try
  {
    requireActive("consume");
    const VirtualParticleBatch& batch = tile.virtual_particles;
    if (tile.grid_fingerprint != context_.grid_fingerprint ||
        tile.tile_ordinal != expected_tile_ordinal_)
      throw std::invalid_argument("PsiFormer ECP tile identity or ordinal is stale");
    if (batch.walkerCount() != context_.sample_count ||
        batch.segmentCount() > plan_.ecp_tile.maximum_segment_count ||
        batch.size() > plan_.ecp_tile.maximum_point_count ||
        tile.bare_weights.size() != batch.size() ||
        tile.complete_ratios.size() != batch.size() ||
        tile.evaluation_stamps.size() != validated_stamp_records.size() ||
        validated_stamp_records.size() >
            plan_.ecp_tile.maximum_evaluation_stamp_count)
      throw std::invalid_argument("PsiFormer ECP live tile exceeds the prepared extents");
    if (consumed_point_count_ > context_.expected_point_count ||
        batch.size() > context_.expected_point_count - consumed_point_count_)
      throw std::invalid_argument("PsiFormer ECP live tile exceeds the declared traversal");
    if (batch.size() != 0 &&
        (batch.segmentCount() == 0 || validated_stamp_records.empty()))
      throw std::invalid_argument("PsiFormer ECP nonempty tile lacks segments or stamps");

    const std::size_t response_bytes = response_count_ * sizeof(double);
    const std::size_t accumulator_bytes =
        plan_.accumulators.total_fp64_elements * sizeof(double);
    const std::array<AddressRange, 3> outputs{
        makeAddressRange(ecp_tile_storage_, plan_.ecp_tile.total_bytes,
                         "consume outputs"),
        makeAddressRange(response_, response_bytes, "consume outputs"),
        makeAddressRange(accumulators_, accumulator_bytes, "consume outputs")};
    const std::array<AddressRange, 7> inputs{
        makeAddressRange(batch.segmentOffsets().data(),
                         batch.segmentOffsets().size() * sizeof(std::size_t),
                         "segment offsets"),
        makeAddressRange(batch.segments().data(),
                         batch.segments().size() *
                             sizeof(VirtualParticleBatch::Segment),
                         "segments"),
        makeAddressRange(batch.absolutePositions().data(),
                         batch.absolutePositions().size() *
                             sizeof(VirtualParticleBatch::PosType),
                         "replacement positions"),
        makeAddressRange(tile.bare_weights.data(),
                         tile.bare_weights.size() * sizeof(QMCTraits::ValueType),
                         "bare weights"),
        makeAddressRange(tile.complete_ratios.data(),
                         tile.complete_ratios.size() * sizeof(QMCTraits::ValueType),
                         "complete ratios"),
        makeAddressRange(tile.evaluation_stamps.data(),
                         tile.evaluation_stamps.size() *
                             sizeof(WaveFunctionComponent::EvaluationStamp),
                         "evaluation stamps"),
        makeAddressRange(validated_stamp_records.data(),
                         validated_stamp_records.size() *
                             sizeof(PsiFormerECPEvaluationStampRecord),
                         "translated stamps")};
    for (const AddressRange input : inputs)
      for (const AddressRange output : outputs)
        if (overlaps(input, output))
          throw std::invalid_argument(
              "PsiFormer ECP live tile input aliases contraction storage");

    bool found_bound_stamp = false;
    for (std::size_t stamp = 0; stamp < validated_stamp_records.size(); ++stamp)
    {
      const auto& record = validated_stamp_records[stamp];
      if (!tile.evaluation_stamps[stamp].isVersioned() ||
          record.producer_fingerprint == 0)
        throw std::invalid_argument("PsiFormer ECP stamp translation is incomplete");
      found_bound_stamp = found_bound_stamp ||
          (record.producer_fingerprint == context_.producer_fingerprint &&
           record.parameter_version == plan_.parameter_version);
    }
    if (!found_bound_stamp)
      throw std::invalid_argument("PsiFormer ECP tile lacks the bound producer/version record");

    for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
    {
      const auto& source = batch.segment(segment);
      if (source.walkerId() < 0 ||
          static_cast<std::size_t>(source.walkerId()) >= context_.sample_count ||
          source.electronId() < 0 ||
          static_cast<std::size_t>(source.electronId()) >= plan_.electron_count ||
          (source.isOnSphere() &&
           (source.sourceCenterId() < 0 ||
            static_cast<std::size_t>(source.sourceCenterId()) >=
                plan_.nucleus_count)))
        throw std::out_of_range(
            "PsiFormer ECP segment reference, electron, or source center is out of range");
    }
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
      const auto& position = batch.absolutePositions()[point];
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        requireFinite(static_cast<double>(position[dimension]),
                      "replacement coordinate");
      const double bare = static_cast<double>(std::real(tile.bare_weights[point]));
      const double bare_imaginary =
          static_cast<double>(std::imag(tile.bare_weights[point]));
      const double ratio_real =
          static_cast<double>(std::real(tile.complete_ratios[point]));
      const double ratio_imaginary =
          static_cast<double>(std::imag(tile.complete_ratios[point]));
      requireFinite(bare, "bare weight");
      requireFinite(bare_imaginary, "bare weight");
      requireFinite(ratio_real, "complete ratio");
      requireFinite(ratio_imaginary, "complete ratio");
      if (bare_imaginary != 0.0)
        throw std::invalid_argument("PsiFormer ECP bare weight must be real");
    }

    const std::uint64_t tile_fingerprint = batch.fingerprint();
    if (tile_fingerprint == 0)
      throw std::invalid_argument("PsiFormer ECP descriptor fingerprint must be nonzero");
    PsiFormerECPTileMetadata metadata;
    metadata.grid_fingerprint = context_.grid_fingerprint;
    metadata.tile_fingerprint = tile_fingerprint;
    metadata.parameter_version = plan_.parameter_version;
    metadata.tile_ordinal = tile.tile_ordinal;
    metadata.flat_point_begin = consumed_point_count_;
    metadata.sample_count = context_.sample_count;
    metadata.segment_count = batch.segmentCount();
    metadata.point_count = batch.size();
    metadata.evaluation_stamp_count = validated_stamp_records.size();
    metadata.flags = requiredECPFlags();
    const PsiFormerECPTileLaunchRecord launch =
        makePsiFormerECPTileLaunchRecordFromPrevalidatedPlan(
            plan_, metadata, context_.channel_count);

    // Serialize the complete live descriptor before score evaluation.  The host
    // oracle traverses only this byte copy below.  The producer contract still keeps
    // its transient backing arrays live until consume returns.
    std::memcpy(ecp_tile_storage_ + plan_.ecp_tile.metadata_offset_bytes,
                &metadata, sizeof(metadata));
    for (std::size_t offset = 0; offset < batch.segmentOffsets().size(); ++offset)
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.segment_offsets_offset_bytes, offset,
                      batch.segmentOffsets()[offset]);
    for (std::size_t segment = 0; segment < batch.segmentCount(); ++segment)
    {
      const auto& source = batch.segment(segment);
      const std::size_t walker_id = static_cast<std::size_t>(source.walkerId());
      const std::size_t electron_id =
          static_cast<std::size_t>(source.electronId());
      const std::size_t source_id = source.isOnSphere()
          ? static_cast<std::size_t>(source.sourceCenterId())
          : PSIFORMER_ECP_NO_SOURCE;
      const std::uint8_t flags =
          source.isOnSphere() ? ECP_SEGMENT_ON_SPHERE : 0;
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.segment_walker_ids_offset_bytes, segment,
                      walker_id);
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.segment_electron_ids_offset_bytes, segment,
                      electron_id);
      storeSerialized(
          ecp_tile_storage_,
          plan_.ecp_tile.segment_source_center_ids_offset_bytes, segment,
          source_id);
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.segment_flags_offset_bytes, segment,
                      flags);
    }
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
      const auto& position = batch.absolutePositions()[point];
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double coordinate = static_cast<double>(position[dimension]);
        storeSerialized(
            ecp_tile_storage_,
            plan_.ecp_tile.replacement_positions_offset_bytes,
            3 * point + dimension, coordinate);
      }
      const double bare =
          static_cast<double>(std::real(tile.bare_weights[point]));
      const wftrain::DerivativeValue ratio{
          static_cast<double>(std::real(tile.complete_ratios[point])),
          static_cast<double>(std::imag(tile.complete_ratios[point]))};
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.bare_weights_offset_bytes, point, bare);
      storeSerializedComplex(
          ecp_tile_storage_, plan_.ecp_tile.complete_ratios_offset_bytes,
          point, ratio);
    }
    for (std::size_t stamp = 0; stamp < validated_stamp_records.size(); ++stamp)
      storeSerialized(ecp_tile_storage_,
                      plan_.ecp_tile.evaluation_stamps_offset_bytes, stamp,
                      validated_stamp_records[stamp]);
    std::memset(ecp_tile_storage_ + plan_.ecp_tile.point_status_offset_bytes,
                0, batch.size() * sizeof(std::uint32_t));

    // Strict segment-major, flat-point order from the completed byte copy.  This
    // also isolates later points from evaluator callbacks that touch producer state.
    // Each score overwrites the same P response and is folded before the next one.
    const PsiFormerCopiedECPTileView copied = copiedTileView();
    for (std::size_t segment = 0; segment < metadata.segment_count; ++segment)
    {
      const std::size_t sample = copied.segmentWalkerId(segment);
      const std::size_t electron = copied.segmentElectronId(segment);
      const std::size_t point_begin = copied.segmentOffset(segment);
      const std::size_t point_end = copied.segmentOffset(segment + 1);
      for (std::size_t point = point_begin; point < point_end; ++point)
      {
        const double bare = copied.bareWeight(point);
        const wftrain::DerivativeValue ratio = copied.completeRatio(point);
        const auto weight_components = device_derivative::ecpCompleteWeight(
            bare, {ratio.real(), ratio.imag()});
        const wftrain::DerivativeValue weight{weight_components.real,
                                              weight_components.imaginary};
        requireFinite(weight, "complete weight");
        const std::array<double, 3> replacement =
            copied.replacementPosition(point);
        evaluateAndFold(sample, electron, replacement.data(), weight);
        const wftrain::DerivativeValue updated =
            sample_weight_sums_[sample] + weight;
        requireFinite(updated, "sample weight sum");
        sample_weight_sums_[sample] = updated;
        storeSerializedComplex(
            ecp_tile_storage_,
            plan_.ecp_tile.sample_weight_sums_offset_bytes, sample, updated);
      }
    }

    consumed_point_count_ = checkedAdd(
        consumed_point_count_, batch.size(),
        "PsiFormer ECP consumed point count overflowed");
    expected_tile_ordinal_ = checkedAdd(
        expected_tile_ordinal_, std::size_t{1},
        "PsiFormer ECP tile ordinal overflowed");
    return launch;
  }
  catch (...)
  {
    poison();
    throw;
  }
}

void PsiFormerECPContraction::evaluateAndFold(
    std::size_t sample,
    std::size_t replaced_electron,
    const double* replacement_position,
    wftrain::DerivativeValue weight)
{
  if (sample >= context_.sample_count ||
      (replacement_position != nullptr && replaced_electron >= plan_.electron_count))
    throw std::out_of_range("PsiFormer ECP score mapping is out of range");
  const double* configuration =
      reference_positions_.data() +
      sample * plan_.electron_count * std::size_t{3};
  evaluator_.evaluate(configuration, plan_.electron_count, replaced_electron,
                      replacement_position, response_, response_count_);

  std::array<wftrain::DerivativeValue, 3> weighted_coefficients{};
  for (std::size_t channel = 0; channel < context_.channel_count; ++channel)
  {
    const std::size_t coefficient_index =
        channel * context_.sample_count + sample;
    weighted_coefficients[channel] = coefficients_[coefficient_index] * weight;
    requireFinite(weighted_coefficients[channel], "weighted coefficient");
  }
  foldPsiFormerRealResponse(
      plan_, response_, response_count_,
      {weighted_coefficients.data(), context_.channel_count},
      PsiFormerDerivativeResponsePart::REAL, context_.adjoint, accumulators_,
      accumulator_fp64_elements_);
}

PsiFormerDerivativeCompletionToken PsiFormerECPContraction::finish()
{
  try
  {
    requireActive("finish");
    if (consumed_point_count_ != context_.expected_point_count)
      throw std::logic_error("PsiFormer ECP traversal ended before its declared point count");
    for (std::size_t sample = 0; sample < context_.sample_count; ++sample)
      requireFinite(sample_weight_sums_[sample], "sample weight sum");
    for (std::size_t sample = 0; sample < context_.sample_count; ++sample)
      evaluateAndFold(sample, plan_.electron_count, nullptr,
                      -sample_weight_sums_[sample]);
    for (std::size_t channel = 0; channel < context_.channel_count; ++channel)
    {
      const double* accumulator = accumulators_ +
          channel * plan_.accumulators.complex_accumulator_stride_fp64_elements;
      for (std::size_t scalar = 0; scalar < 2 * plan_.parameter_count; ++scalar)
        requireFinite(accumulator[scalar], "completed accumulator");
    }
    state_ = State::COMPLETE;
    return makePsiFormerVJPCompletionTokenFromPrevalidatedPlan(
        plan_, context_.adjoint, context_.channel_count, 0);
  }
  catch (...)
  {
    poison();
    throw;
  }
}

} // namespace qmcplusplus::psiformer
