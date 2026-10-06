//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file StreamingDerivative.cpp
 * @brief Validation and lifecycle enforcement for bounded derivative streams.
 */

#include "QMCWaveFunctions/Optimization/StreamingDerivative.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{

/** Report whether a real value has a finite IEEE exponent bit pattern.
 *
 * QMCPACK release builds use `-ffast-math`, under which compiler builtins may assume
 * values are finite.  Inspecting the representation keeps input validation effective.
 */
bool isFiniteReal(DerivativeReal value) noexcept
{
  static_assert(std::numeric_limits<DerivativeReal>::is_iec559,
                "Streaming derivative validation requires IEEE floating-point values");
  static_assert(sizeof(DerivativeReal) == sizeof(std::uint64_t) ||
                    sizeof(DerivativeReal) == sizeof(std::uint32_t),
                "Streaming derivative validation supports IEEE float and double");
  if constexpr (sizeof(DerivativeReal) == sizeof(std::uint64_t))
  {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  }
  else
  {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
  }
}

/// Report whether a complex contraction value is finite in both components.
bool isFinite(DerivativeValue value) noexcept
{
  return isFiniteReal(value.real()) && isFiniteReal(value.imag());
}

/// Report whether an adjoint value names exactly one supported convention.
bool isValidAdjoint(DerivativeAdjoint adjoint) noexcept
{
  return adjoint == DerivativeAdjoint::TRANSPOSE || adjoint == DerivativeAdjoint::HERMITIAN;
}

/// Validate finite direction values and the real-parameter tangent restriction.
void validateDirectionValues(const StructuredParameterSchema& schema,
                             DerivativeArrayView<const DerivativeValue> values)
{
  for (const ParameterBlockDescriptor& block : schema.blocks())
    for (std::size_t index = 0; index < block.count; ++index)
    {
      const DerivativeValue value = values[block.offset + index];
      if (!isFinite(value))
        throw std::invalid_argument("Score JVP direction contains a non-finite value");
      if (block.scalar_domain == ParameterScalarDomain::REAL64 && value.imag() != DerivativeReal{})
        throw std::invalid_argument("Score JVP direction for real parameters has a nonzero imaginary part");
    }
}

/// Compare all identity and interval fields of two canonical chunk descriptors.
bool sameChunk(const ParameterChunkDescriptor& left, const ParameterChunkDescriptor& right) noexcept
{
  return left.provider_id == right.provider_id && left.schema_fingerprint == right.schema_fingerprint &&
      left.parameter_version == right.parameter_version && left.block_index == right.block_index &&
      left.block_id == right.block_id && left.block_offset == right.block_offset &&
      left.parameter_offset == right.parameter_offset && left.count == right.count &&
      left.scalar_domain == right.scalar_domain && left.trainable == right.trainable &&
      left.ordinal == right.ordinal;
}

/// Validate that a nonempty view has storage and an empty view does not require it.
template<class T>
bool hasValidStorage(DerivativeArrayView<T> view) noexcept
{
  return view.empty() || view.data() != nullptr;
}

/// Return the exclusive sample end while rejecting size_t overflow.
std::size_t checkedSampleEnd(std::size_t offset, std::size_t count)
{
  if (offset > std::numeric_limits<std::size_t>::max() - count)
    throw std::overflow_error("Derivative sample interval overflows size_t");
  return offset + count;
}

} // namespace

ParameterChunkPlan::ParameterChunkPlan(const StructuredParameterSchema& schema,
                                       std::size_t parameter_version,
                                       std::size_t maximum_chunk_size,
                                       FrozenBlockPolicy frozen_policy)
    : provider_id_(schema.providerId()),
      schema_fingerprint_(schema.fingerprint()),
      parameter_version_(parameter_version),
      scalar_domain_(schema.blocks().front().scalar_domain),
      maximum_chunk_size_(maximum_chunk_size)
{
  if (maximum_chunk_size_ == 0)
    throw std::invalid_argument("Parameter chunk size must be greater than zero");

  // One stream has one scalar domain.  Mixed schemas need a future typed-block transport
  // and are rejected here rather than being silently converted.
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.scalar_domain != scalar_domain_)
      throw std::invalid_argument("Streaming derivative chunk plans require a homogeneous parameter scalar domain");

  for (std::size_t block_index = 0; block_index < schema.blocks().size(); ++block_index)
  {
    const ParameterBlockDescriptor& block = schema.blocks()[block_index];
    if (!block.trainable && frozen_policy == FrozenBlockPolicy::EXCLUDE)
      continue;

    std::size_t block_offset = 0;
    while (block_offset < block.count)
    {
      const std::size_t count = std::min(maximum_chunk_size_, block.count - block_offset);
      chunks_.push_back({provider_id_,
                         schema_fingerprint_,
                         parameter_version_,
                         block_index,
                         block.id,
                         block_offset,
                         block.offset + block_offset,
                         count,
                         block.scalar_domain,
                         block.trainable,
                         chunks_.size()});
      block_offset += count;
    }

    // Schema validation guarantees that the selected-block sum cannot exceed the full
    // parameter count, but retain an explicit guard at this independently reusable API.
    if (selected_parameter_count_ > std::numeric_limits<std::size_t>::max() - block.count)
      throw std::overflow_error("Selected streaming parameter count overflows size_t");
    selected_parameter_count_ += block.count;
  }
}

StructuredParameterVectorConstView::StructuredParameterVectorConstView(
    const StructuredParameterSchema& schema,
    std::size_t parameter_version,
    DerivativeArrayView<const DerivativeValue> values)
    : schema_(&schema), parameter_version_(parameter_version), values_(values)
{
  if (!hasValidStorage(values_))
    throw std::invalid_argument("Structured parameter vector has null nonempty storage");
  if (values_.size() != schema.parameterCount())
    throw std::invalid_argument("Structured parameter vector extent does not match its schema");
  validateDirectionValues(schema, values_);
}

DerivativeArrayView<const DerivativeValue> StructuredParameterVectorConstView::block(std::size_t block_index) const
{
  if (block_index >= schema_->blocks().size())
    throw std::out_of_range("Structured parameter block index is out of range");
  const ParameterBlockDescriptor& descriptor = schema_->blocks()[block_index];
  return {values_.data() + descriptor.offset, descriptor.count};
}

ParameterChunkConstView StructuredParameterVectorConstView::chunk(
    const ParameterChunkDescriptor& descriptor) const
{
  if (descriptor.provider_id != schema_->providerId() ||
      descriptor.schema_fingerprint != schema_->fingerprint() || descriptor.parameter_version != parameter_version_)
    throw std::invalid_argument("Parameter chunk does not match the structured vector identity");
  if (descriptor.block_index >= schema_->blocks().size())
    throw std::out_of_range("Parameter chunk block index is out of range");

  const ParameterBlockDescriptor& block_descriptor = schema_->blocks()[descriptor.block_index];
  if (descriptor.block_id != block_descriptor.id || descriptor.scalar_domain != block_descriptor.scalar_domain ||
      descriptor.block_offset > block_descriptor.count ||
      descriptor.count > block_descriptor.count - descriptor.block_offset ||
      descriptor.parameter_offset != block_descriptor.offset + descriptor.block_offset)
    throw std::invalid_argument("Parameter chunk is inconsistent with its structured block");

  return {descriptor, {values_.data() + descriptor.parameter_offset, descriptor.count}};
}

[[noreturn]] void ParameterReductionSink::poisonAndThrow(const std::string& message)
{
  if (state_ == DerivativeSinkState::ACTIVE)
    onAbort();
  state_ = DerivativeSinkState::POISONED;
  clearTransaction();
  throw std::logic_error(message);
}

void ParameterReductionSink::clearTransaction() noexcept
{
  descriptor_       = nullptr;
  chunk_plan_       = nullptr;
  channel_count_    = 0;
  next_record_      = 0;
  expected_records_ = 0;
}

void ParameterReductionSink::begin(const DerivativeStreamDescriptor& descriptor,
                                   const ParameterChunkPlan& chunk_plan,
                                   DerivativeArrayView<const VJPCoefficientChannel> channels)
{
  if (state_ != DerivativeSinkState::IDLE)
    poisonAndThrow("Parameter derivative sink begin requires idle state");

  if (descriptor.provider_id != chunk_plan.providerId() ||
      descriptor.schema_fingerprint != chunk_plan.schemaFingerprint() ||
      descriptor.parameter_version != chunk_plan.parameterVersion())
    poisonAndThrow("Parameter derivative stream and chunk plan identities differ");
  if (descriptor.execution_domain != DerivativeExecutionDomain::HOST)
    poisonAndThrow("Parameter derivative sinks currently accept host-accessible values only");
  if (!isValidAdjoint(descriptor.adjoint))
    poisonAndThrow("Parameter derivative stream has an invalid adjoint convention");
  if (descriptor.parameter_scalar_domain != chunk_plan.scalarDomain())
    poisonAndThrow("Parameter derivative stream and chunk plan scalar domains differ");
  if (descriptor.maximum_parameter_chunk_size == 0 ||
      chunk_plan.maximumChunkSize() > descriptor.maximum_parameter_chunk_size)
    poisonAndThrow("Parameter derivative chunk plan exceeds the stream capacity");
  if (channels.empty() || !hasValidStorage(channels))
    poisonAndThrow("Parameter derivative stream requires at least one valid VJP channel");
  if (descriptor.maximum_vjp_channels == 0 || channels.size() > descriptor.maximum_vjp_channels)
    poisonAndThrow("Parameter derivative channel count exceeds the stream capacity");
  if (descriptor.sample_offset > std::numeric_limits<std::size_t>::max() - descriptor.sample_count)
    poisonAndThrow("Parameter derivative sample interval overflows size_t");
  if (chunk_plan.chunks().size() > std::numeric_limits<std::size_t>::max() / channels.size())
    poisonAndThrow("Parameter derivative coverage count overflows size_t");

  std::uint32_t product_mask = 0;
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
  {
    if (channels[channel].id.empty())
      poisonAndThrow("VJP channel identity must not be empty");
    for (std::size_t previous = 0; previous < channel; ++previous)
      if (channels[previous].id == channels[channel].id)
        poisonAndThrow("VJP channel identities must be unique");

    const DerivativeProduct product = channels[channel].product;
    if (product != DerivativeProduct::SCORE_VJP && product != DerivativeProduct::LOCAL_ENERGY_VJP)
      poisonAndThrow("Parameter reduction sink accepts only VJP channel products");
    if (product == DerivativeProduct::SCORE_VJP && channels[channel].local_energy_term_mask != 0)
      poisonAndThrow("Score VJP channels must not declare local-energy term coverage");
    if (product == DerivativeProduct::LOCAL_ENERGY_VJP && channels[channel].local_energy_term_mask == 0)
      poisonAndThrow("Local-energy VJP channels must name their Hamiltonian-term coverage");

    const CoefficientView& coefficients = channels[channel].coefficients;
    if (coefficients.provider_id != descriptor.provider_id ||
        coefficients.schema_fingerprint != descriptor.schema_fingerprint ||
        coefficients.parameter_version != descriptor.parameter_version ||
        coefficients.batch_ordinal != descriptor.batch_ordinal ||
        coefficients.sample_offset != descriptor.sample_offset ||
        coefficients.values.size() != descriptor.sample_count || !hasValidStorage(coefficients.values))
      poisonAndThrow("VJP coefficients do not match the parameter stream identity and sample interval");
    for (const DerivativeValue value : coefficients.values)
      if (!isFinite(value))
        poisonAndThrow("VJP coefficients contain a non-finite value");
    product_mask |= derivativeProductBit(product);
  }
  if (product_mask != descriptor.product_mask)
    poisonAndThrow("Parameter derivative stream product mask disagrees with its channels");

  descriptor_       = &descriptor;
  chunk_plan_       = &chunk_plan;
  channel_count_    = channels.size();
  next_record_      = 0;
  expected_records_ = chunk_plan.chunks().size() * channels.size();
  state_            = DerivativeSinkState::ACTIVE;
  try
  {
    onBegin(descriptor, chunk_plan, channels);
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
}

void ParameterReductionSink::add(std::size_t channel_ordinal, const ParameterChunkConstView& chunk)
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Parameter derivative add requires an active transaction");
  if (descriptor_->sample_count == 0)
    poisonAndThrow("A zero-sample parameter derivative stream must not emit chunks");

  if (next_record_ >= expected_records_)
    poisonAndThrow("Parameter derivative stream emitted excess chunks");

  const std::size_t expected_chunk   = next_record_ / channel_count_;
  const std::size_t expected_channel = next_record_ % channel_count_;
  if (channel_ordinal != expected_channel)
    poisonAndThrow("Parameter derivative channels are not in canonical order");
  if (!sameChunk(chunk.descriptor(), chunk_plan_->chunks()[expected_chunk]))
    poisonAndThrow("Parameter derivative chunk is not the next canonical descriptor");
  if (chunk.values().size() != chunk.descriptor().count || !hasValidStorage(chunk.values()))
    poisonAndThrow("Parameter derivative chunk value extent is invalid");
  for (const DerivativeValue value : chunk.values())
    if (!isFinite(value))
      poisonAndThrow("Parameter derivative chunk contains a non-finite value");

  try
  {
    consume(channel_ordinal, chunk);
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  ++next_record_;
}

void ParameterReductionSink::end()
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Parameter derivative end requires an active transaction");
  if (descriptor_->sample_count == 0)
    poisonAndThrow("A zero-sample parameter derivative stream requires endEmptyBatch");
  if (next_record_ != expected_records_)
    poisonAndThrow("Parameter derivative stream ended before complete canonical coverage");

  try
  {
    onEnd();
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  clearTransaction();
  state_ = DerivativeSinkState::COMPLETE;
}

void ParameterReductionSink::endEmptyBatch()
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Parameter derivative endEmptyBatch requires an active transaction");
  if (descriptor_->sample_count != 0 || next_record_ != 0)
    poisonAndThrow("Parameter derivative endEmptyBatch is legal only for an untouched zero-sample stream");

  try
  {
    onEnd();
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  clearTransaction();
  state_ = DerivativeSinkState::COMPLETE;
}

void ParameterReductionSink::abort() noexcept
{
  onAbort();
  clearTransaction();
  state_ = DerivativeSinkState::POISONED;
}

void ParameterReductionSink::reset()
{
  if (state_ == DerivativeSinkState::ACTIVE)
    poisonAndThrow("An active parameter derivative sink must be aborted before reset");
  onReset();
  clearTransaction();
  state_ = DerivativeSinkState::IDLE;
}

[[noreturn]] void SampleProductSink::poisonAndThrow(const std::string& message)
{
  if (state_ == DerivativeSinkState::ACTIVE)
    onAbort();
  state_ = DerivativeSinkState::POISONED;
  clearTransaction();
  throw std::logic_error(message);
}

void SampleProductSink::clearTransaction() noexcept
{
  descriptor_          = nullptr;
  next_sample_offset_  = 0;
  next_tile_ordinal_   = 0;
}

void SampleProductSink::begin(const DerivativeStreamDescriptor& descriptor)
{
  if (state_ != DerivativeSinkState::IDLE)
    poisonAndThrow("Sample-product sink begin requires idle state");
  if (descriptor.provider_id.empty() || descriptor.schema_fingerprint.empty())
    poisonAndThrow("Sample-product stream identity must not be empty");
  if (descriptor.product_mask != derivativeProductBit(DerivativeProduct::SCORE_JVP))
    poisonAndThrow("Sample-product sink accepts only score JVP streams");
  if (descriptor.execution_domain != DerivativeExecutionDomain::HOST)
    poisonAndThrow("Sample-product sinks currently accept host-accessible values only");
  if (descriptor.adjoint != DerivativeAdjoint::TRANSPOSE)
    poisonAndThrow("Score JVP streams require the transpose direction convention");
  if (descriptor.maximum_sample_tile_size == 0)
    poisonAndThrow("Sample-product stream tile capacity must be greater than zero");
  if (descriptor.sample_offset > std::numeric_limits<std::size_t>::max() - descriptor.sample_count)
    poisonAndThrow("Sample-product interval overflows size_t");

  descriptor_         = &descriptor;
  next_sample_offset_ = descriptor.sample_offset;
  next_tile_ordinal_  = 0;
  state_              = DerivativeSinkState::ACTIVE;
  try
  {
    onBegin(descriptor);
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
}

void SampleProductSink::add(const SampleProductTileConstView& tile)
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Sample-product add requires an active transaction");
  if (descriptor_->sample_count == 0)
    poisonAndThrow("A zero-sample JVP stream must not emit tiles");

  const SampleProductTileDescriptor& tile_descriptor = tile.descriptor();
  if (tile_descriptor.provider_id != descriptor_->provider_id ||
      tile_descriptor.schema_fingerprint != descriptor_->schema_fingerprint ||
      tile_descriptor.parameter_version != descriptor_->parameter_version ||
      tile_descriptor.batch_ordinal != descriptor_->batch_ordinal)
    poisonAndThrow("Sample-product tile identity differs from its stream");
  if (tile_descriptor.ordinal != next_tile_ordinal_ || tile_descriptor.sample_offset != next_sample_offset_)
    poisonAndThrow("Sample-product tiles are not in canonical contiguous order");
  if (tile_descriptor.count == 0 || tile_descriptor.count > descriptor_->maximum_sample_tile_size ||
      tile.values().size() != tile_descriptor.count || !hasValidStorage(tile.values()))
    poisonAndThrow("Sample-product tile extent is invalid");

  const std::size_t stream_end = descriptor_->sample_offset + descriptor_->sample_count;
  if (tile_descriptor.sample_offset > std::numeric_limits<std::size_t>::max() - tile_descriptor.count)
    poisonAndThrow("Sample-product tile interval overflows size_t");
  const std::size_t tile_end = tile_descriptor.sample_offset + tile_descriptor.count;
  if (tile_end > stream_end)
    poisonAndThrow("Sample-product tile exceeds the declared sample interval");
  for (const DerivativeValue value : tile.values())
    if (!isFinite(value))
      poisonAndThrow("Sample-product tile contains a non-finite value");

  try
  {
    consume(tile);
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  next_sample_offset_ = tile_end;
  ++next_tile_ordinal_;
}

void SampleProductSink::end()
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Sample-product end requires an active transaction");
  if (descriptor_->sample_count == 0)
    poisonAndThrow("A zero-sample JVP stream requires endEmptyBatch");
  if (next_sample_offset_ != checkedSampleEnd(descriptor_->sample_offset, descriptor_->sample_count))
    poisonAndThrow("Sample-product stream ended before complete sample coverage");

  try
  {
    onEnd();
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  clearTransaction();
  state_ = DerivativeSinkState::COMPLETE;
}

void SampleProductSink::endEmptyBatch()
{
  if (state_ != DerivativeSinkState::ACTIVE)
    poisonAndThrow("Sample-product endEmptyBatch requires an active transaction");
  if (descriptor_->sample_count != 0 || next_tile_ordinal_ != 0)
    poisonAndThrow("Sample-product endEmptyBatch is legal only for an untouched zero-sample stream");

  try
  {
    onEnd();
  }
  catch (...)
  {
    onAbort();
    state_ = DerivativeSinkState::POISONED;
    clearTransaction();
    throw;
  }
  clearTransaction();
  state_ = DerivativeSinkState::COMPLETE;
}

void SampleProductSink::abort() noexcept
{
  onAbort();
  clearTransaction();
  state_ = DerivativeSinkState::POISONED;
}

void SampleProductSink::reset()
{
  if (state_ == DerivativeSinkState::ACTIVE)
    poisonAndThrow("An active sample-product sink must be aborted before reset");
  onReset();
  clearTransaction();
  state_ = DerivativeSinkState::IDLE;
}

bool StreamingDerivativeCapabilities::supports(DerivativeProduct product) const noexcept
{
  return (product_mask & derivativeProductBit(product)) != 0;
}

bool StreamingDerivativeCapabilities::supports(DerivativeAdjoint adjoint) const noexcept
{
  switch (adjoint)
  {
  case DerivativeAdjoint::TRANSPOSE:
  case DerivativeAdjoint::HERMITIAN:
    return (adjoint_mask & derivativeAdjointBit(adjoint)) != 0;
  }
  return false;
}

bool StreamingDerivativeCapabilities::coversLocalEnergyTerms(std::uint32_t requested_terms) const noexcept
{
  return (local_energy_term_mask & requested_terms) == requested_terms;
}

void StreamingDerivativeOperator::applyVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                                             DerivativeAdjoint adjoint,
                                             ParameterReductionSink& sink) const
{
  if (channels.empty() || !hasValidStorage(channels))
    throw std::invalid_argument("Streaming VJP requires at least one valid coefficient channel");

  const StreamingDerivativeCapabilities support = capabilities();
  const StructuredParameterSchema& schema        = parameterSchema();
  const ParameterChunkPlan& plan                  = parameterChunkPlan();
  if (support.execution_domain != DerivativeExecutionDomain::HOST)
    throw std::runtime_error("Streaming VJP requires a device-aware sink for device execution");
  if (!support.block_streaming || support.maximum_parameter_chunk_size == 0 || support.maximum_vjp_channels == 0)
    throw std::runtime_error("Streaming VJP producer does not advertise bounded block streaming");
  if (channels.size() > support.maximum_vjp_channels)
    throw std::invalid_argument("Streaming VJP channel count exceeds the producer capacity");
  if (!support.supports(adjoint))
    throw std::runtime_error("Streaming VJP producer does not support the requested adjoint convention");
  if (plan.providerId() != schema.providerId() || plan.schemaFingerprint() != schema.fingerprint() ||
      plan.parameterVersion() != parameterVersion() || plan.scalarDomain() != support.parameter_scalar_domain ||
      plan.maximumChunkSize() > support.maximum_parameter_chunk_size)
    throw std::runtime_error("Streaming VJP producer schema, version, capability, and chunk plan are inconsistent");

  std::uint32_t product_mask = 0;
  for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index)
  {
    const VJPCoefficientChannel& channel = channels[channel_index];
    if (channel.id.empty())
      throw std::invalid_argument("Streaming VJP channel identity must not be empty");
    for (std::size_t previous = 0; previous < channel_index; ++previous)
      if (channels[previous].id == channel.id)
        throw std::invalid_argument("Streaming VJP channel identities must be unique");
    if (channel.product != DerivativeProduct::SCORE_VJP &&
        channel.product != DerivativeProduct::LOCAL_ENERGY_VJP)
      throw std::invalid_argument("Streaming multi-channel traversal accepts only VJP products");
    if (!support.supports(channel.product))
      throw std::runtime_error("Streaming VJP producer does not support one requested channel product");
    if (channel.product == DerivativeProduct::SCORE_VJP && channel.local_energy_term_mask != 0)
      throw std::invalid_argument("Score VJP channels must not request local-energy terms");
    if (channel.product == DerivativeProduct::LOCAL_ENERGY_VJP &&
        (channel.local_energy_term_mask == 0 ||
         !support.coversLocalEnergyTerms(channel.local_energy_term_mask)))
      throw std::runtime_error("Streaming VJP producer lacks requested local-energy term coverage");

    const CoefficientView& coefficients = channel.coefficients;
    if (coefficients.provider_id != schema.providerId() ||
        coefficients.schema_fingerprint != schema.fingerprint() ||
        coefficients.parameter_version != parameterVersion() || coefficients.batch_ordinal != batchOrdinal() ||
        coefficients.sample_offset != sampleOffset() || coefficients.values.size() != sampleCount() ||
        !hasValidStorage(coefficients.values))
      throw std::invalid_argument("Streaming VJP coefficients do not match the producer identity and sample batch");
    for (const DerivativeValue value : coefficients.values)
      if (!isFinite(value))
        throw std::invalid_argument("Streaming VJP coefficients contain a non-finite value");
    product_mask |= derivativeProductBit(channel.product);
  }

  DerivativeStreamDescriptor descriptor{schema.providerId(),
                                        schema.fingerprint(),
                                        parameterVersion(),
                                        product_mask,
                                        adjoint,
                                        support.parameter_scalar_domain,
                                        support.result_scalar_domain,
                                        batchOrdinal(),
                                        sampleOffset(),
                                        sampleCount(),
                                        support.reduction_domain,
                                        support.execution_domain,
                                        support.maximum_vjp_channels,
                                        support.maximum_parameter_chunk_size,
                                        support.maximum_sample_tile_size};

  sink.begin(descriptor, plan, channels);
  try
  {
    if (sampleCount() == 0)
      sink.endEmptyBatch();
    else
    {
      evaluateVJPs(channels, adjoint, sink);
      sink.end();
    }
  }
  catch (...)
  {
    if (sink.state() == DerivativeSinkState::ACTIVE)
      sink.abort();
    throw;
  }
}

void StreamingDerivativeOperator::applyScoreJVP(const StructuredParameterVectorConstView& direction,
                                                 SampleProductSink& sink) const
{
  const StreamingDerivativeCapabilities support = capabilities();
  const StructuredParameterSchema& schema        = parameterSchema();
  if (support.execution_domain != DerivativeExecutionDomain::HOST)
    throw std::runtime_error("Score JVP requires a device-aware sink for device execution");
  if (!support.supports(DerivativeProduct::SCORE_JVP) || support.maximum_sample_tile_size == 0)
    throw std::runtime_error("Streaming derivative producer does not support bounded score JVP tiles");
  for (const ParameterBlockDescriptor& block : schema.blocks())
    if (block.scalar_domain != support.parameter_scalar_domain)
      throw std::runtime_error("Score JVP producer scalar domain is inconsistent with its parameter schema");
  if (direction.providerId() != schema.providerId() ||
      direction.schemaFingerprint() != schema.fingerprint() || direction.parameterVersion() != parameterVersion())
    throw std::invalid_argument("Score JVP direction does not match the producer schema and version");
  validateDirectionValues(schema, direction.values());

  DerivativeStreamDescriptor descriptor{schema.providerId(),
                                        schema.fingerprint(),
                                        parameterVersion(),
                                        derivativeProductBit(DerivativeProduct::SCORE_JVP),
                                        DerivativeAdjoint::TRANSPOSE,
                                        support.parameter_scalar_domain,
                                        support.result_scalar_domain,
                                        batchOrdinal(),
                                        sampleOffset(),
                                        sampleCount(),
                                        support.reduction_domain,
                                        support.execution_domain,
                                        support.maximum_vjp_channels,
                                        support.maximum_parameter_chunk_size,
                                        support.maximum_sample_tile_size};

  sink.begin(descriptor);
  try
  {
    if (sampleCount() == 0)
      sink.endEmptyBatch();
    else
    {
      evaluateScoreJVP(direction, sink);
      sink.end();
    }
  }
  catch (...)
  {
    if (sink.state() == DerivativeSinkState::ACTIVE)
      sink.abort();
    throw;
  }
}

} // namespace qmcplusplus::wftrain
