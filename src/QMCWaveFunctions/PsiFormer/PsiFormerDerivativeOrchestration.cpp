//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDerivativeOrchestration.cpp
 * @brief Prepared folding and synchronized host publication.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerDerivativeOrchestration.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{
namespace
{

bool finite(wftrain::DerivativeValue value) noexcept
{
  return device_math::isFiniteBinary64(value.real()) &&
      device_math::isFiniteBinary64(value.imag());
}

bool sameChunk(const wftrain::ParameterChunkDescriptor& left,
               const wftrain::ParameterChunkDescriptor& right) noexcept
{
  return left.provider_id == right.provider_id &&
      left.schema_fingerprint == right.schema_fingerprint &&
      left.parameter_version == right.parameter_version &&
      left.block_index == right.block_index && left.block_id == right.block_id &&
      left.block_offset == right.block_offset &&
      left.parameter_offset == right.parameter_offset && left.count == right.count &&
      left.scalar_domain == right.scalar_domain && left.trainable == right.trainable &&
      left.ordinal == right.ordinal;
}

bool sameChunkPlan(const wftrain::ParameterChunkPlan& left,
                   const wftrain::ParameterChunkPlan& right) noexcept
{
  if (left.providerId() != right.providerId() ||
      left.schemaFingerprint() != right.schemaFingerprint() ||
      left.parameterVersion() != right.parameterVersion() ||
      left.scalarDomain() != right.scalarDomain() ||
      left.maximumChunkSize() != right.maximumChunkSize() ||
      left.selectedParameterCount() != right.selectedParameterCount() ||
      left.chunks().size() != right.chunks().size())
    return false;
  for (std::size_t index = 0; index < left.chunks().size(); ++index)
    if (!sameChunk(left.chunks()[index], right.chunks()[index]))
      return false;
  return true;
}

bool isVJPProductMask(std::uint32_t product_mask) noexcept
{
  constexpr std::uint32_t VJP_PRODUCTS =
      wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_VJP) |
      wftrain::derivativeProductBit(wftrain::DerivativeProduct::LOCAL_ENERGY_VJP);
  return product_mask != 0 && (product_mask & ~VJP_PRODUCTS) == 0;
}

} // namespace

PsiFormerComplexAxpyCoefficient makePsiFormerComplexAxpyCoefficient(
    wftrain::DerivativeValue coefficient,
    PsiFormerDerivativeResponsePart response_part,
    wftrain::DerivativeAdjoint adjoint)
{
  if (!finite(coefficient))
    throw std::invalid_argument("PsiFormer derivative fold coefficient must be finite");

  switch (adjoint)
  {
  case wftrain::DerivativeAdjoint::TRANSPOSE:
    break;
  case wftrain::DerivativeAdjoint::HERMITIAN:
    break;
  default:
    throw std::invalid_argument("PsiFormer derivative fold received an invalid adjoint convention");
  }

  switch (response_part)
  {
  case PsiFormerDerivativeResponsePart::REAL:
    return {coefficient.real(), coefficient.imag()};
  case PsiFormerDerivativeResponsePart::IMAGINARY:
    // coefficient*(+i) for transpose and coefficient*(-i) for Hermitian.
    return adjoint == wftrain::DerivativeAdjoint::TRANSPOSE
        ? PsiFormerComplexAxpyCoefficient{-coefficient.imag(), coefficient.real()}
        : PsiFormerComplexAxpyCoefficient{coefficient.imag(), -coefficient.real()};
  }
  throw std::invalid_argument("PsiFormer derivative fold received an invalid response part");
}

void foldPsiFormerRealResponse(
    const PsiFormerDerivativePlan& plan,
    const double* response,
    std::size_t response_count,
    wftrain::DerivativeArrayView<const wftrain::DerivativeValue> coefficients,
    PsiFormerDerivativeResponsePart response_part,
    wftrain::DerivativeAdjoint adjoint,
    double* accumulators,
    std::size_t accumulator_fp64_elements)
{
  if (response_count != plan.parameter_count || response_count == 0 || response == nullptr)
    throw std::invalid_argument("PsiFormer derivative fold response extent is not canonical P");
  if (coefficients.empty() || coefficients.data() == nullptr ||
      coefficients.size() > 3 ||
      coefficients.size() > plan.accumulators.complex_accumulator_count)
    throw std::invalid_argument("PsiFormer derivative fold requires one to three prepared channels");
  if (accumulators == nullptr ||
      accumulator_fp64_elements < plan.accumulators.total_fp64_elements)
    throw std::invalid_argument("PsiFormer derivative fold accumulator storage is too small");
  if (plan.accumulators.parameter_count != plan.parameter_count ||
      plan.accumulators.complex_accumulator_stride_fp64_elements / 2 <
          plan.parameter_count)
    throw std::invalid_argument("PsiFormer derivative fold accumulator layout is inconsistent");

  // Match the device finite-scan contract before changing any accumulator.
  for (std::size_t parameter = 0; parameter < response_count; ++parameter)
    if (!device_math::isFiniteBinary64(response[parameter]))
      throw std::invalid_argument("PsiFormer derivative fold response contains a non-finite value");

  // Preflight every channel before the first write so an invalid later coefficient
  // cannot leave an earlier accumulator partially updated.
  std::array<PsiFormerComplexAxpyCoefficient, 3> alphas{};
  for (std::size_t channel = 0; channel < coefficients.size(); ++channel)
    alphas[channel] = makePsiFormerComplexAxpyCoefficient(
        coefficients[channel], response_part, adjoint);

  for (std::size_t channel = 0; channel < coefficients.size(); ++channel)
  {
    const PsiFormerComplexAxpyCoefficient alpha = alphas[channel];
    double* destination = accumulators +
        channel * plan.accumulators.complex_accumulator_stride_fp64_elements;
    for (std::size_t parameter = 0; parameter < plan.parameter_count; ++parameter)
    {
      destination[2 * parameter] += alpha.real * response[parameter];
      destination[2 * parameter + 1] += alpha.imaginary * response[parameter];
    }
  }
}

PsiFormerDerivativeCompletionToken makePsiFormerVJPCompletionToken(
    const PsiFormerDerivativePlan& plan,
    wftrain::DerivativeAdjoint adjoint,
    std::size_t channel_count,
    std::uint32_t status)
{
  validatePsiFormerDerivativePlan(plan);
  return makePsiFormerVJPCompletionTokenFromPrevalidatedPlan(
      plan, adjoint, channel_count, status);
}

PsiFormerDerivativeCompletionToken
makePsiFormerVJPCompletionTokenFromPrevalidatedPlan(
    const PsiFormerDerivativePlan& plan,
    wftrain::DerivativeAdjoint adjoint,
    std::size_t channel_count,
    std::uint32_t status)
{
  if (adjoint != wftrain::DerivativeAdjoint::TRANSPOSE &&
      adjoint != wftrain::DerivativeAdjoint::HERMITIAN)
    throw std::invalid_argument("PsiFormer VJP completion has an invalid adjoint convention");
  if (channel_count == 0 || channel_count > 3 ||
      channel_count > plan.accumulators.complex_accumulator_count)
    throw std::invalid_argument("PsiFormer VJP completion channel count exceeds the plan");

  PsiFormerDerivativeCompletionToken token;
  token.plan_fingerprint = plan.fingerprint;
  token.device_layout_fingerprint = plan.device_layout_fingerprint;
  token.device_arena_fingerprint = plan.device_arena.fingerprint;
  token.parameter_version = plan.parameter_version;
  token.kind = PsiFormerDerivativePublicationKind::VJP;
  token.adjoint = adjoint;
  token.channel_count = channel_count;
  token.status = status;
  return token;
}

PsiFormerDerivativeCompletionToken makePsiFormerJVPCompletionToken(
    const PsiFormerDerivativePlan& plan,
    std::size_t batch_ordinal,
    std::size_t sample_offset,
    std::size_t tile_ordinal,
    std::uint32_t status)
{
  validatePsiFormerDerivativePlan(plan);
  if (plan.host_staging.maximum_jvp_output_complex_elements < 1)
    throw std::invalid_argument("PsiFormer score JVP completion lacks scalar staging");

  PsiFormerDerivativeCompletionToken token;
  token.plan_fingerprint = plan.fingerprint;
  token.device_layout_fingerprint = plan.device_layout_fingerprint;
  token.device_arena_fingerprint = plan.device_arena.fingerprint;
  token.parameter_version = plan.parameter_version;
  token.kind = PsiFormerDerivativePublicationKind::SCORE_JVP;
  token.adjoint = wftrain::DerivativeAdjoint::TRANSPOSE;
  token.batch_ordinal = batch_ordinal;
  token.sample_offset = sample_offset;
  token.tile_ordinal = tile_ordinal;
  token.status = status;
  return token;
}

PsiFormerDerivativeHostPublisher::PsiFormerDerivativeHostPublisher(
    const PsiFormerDerivativePlan& plan,
    wftrain::ParameterChunkPlan chunk_plan)
    : chunk_plan_(std::move(chunk_plan)),
      plan_fingerprint_(plan.fingerprint),
      device_layout_fingerprint_(plan.device_layout_fingerprint),
      device_arena_fingerprint_(plan.device_arena.fingerprint),
      parameter_version_(plan.parameter_version),
      parameter_count_(plan.parameter_count),
      accumulator_count_(plan.accumulators.complex_accumulator_count),
      accumulator_stride_fp64_elements_(
          plan.accumulators.complex_accumulator_stride_fp64_elements),
      accumulator_total_fp64_elements_(plan.accumulators.total_fp64_elements),
      host_chunk_capacity_(plan.host_staging.maximum_parameter_chunk_elements),
      host_jvp_capacity_(plan.host_staging.maximum_jvp_output_complex_elements)
{
  // This is the allocation-bearing preparation boundary.  Revalidation is not
  // repeated by warmed publication calls.
  validatePsiFormerDerivativePlan(plan);
  if (chunk_plan_.parameterVersion() != parameter_version_)
    throw std::invalid_argument("PsiFormer derivative publication chunk version is stale");
  if (chunk_plan_.scalarDomain() != wftrain::ParameterScalarDomain::REAL64)
    throw std::invalid_argument("PsiFormer derivative publication requires a real parameter schema");
  if (chunk_plan_.maximumChunkSize() == 0 ||
      chunk_plan_.maximumChunkSize() > host_chunk_capacity_)
    throw std::invalid_argument("PsiFormer derivative publication chunk plan exceeds host staging");

  const auto& chunks = chunk_plan_.chunks();
  std::size_t selected_count = 0;
  std::size_t prior_tensor = 0;
  std::size_t expected_block_offset = 0;
  std::size_t active_tensor_begin = 0;
  std::size_t active_tensor_end = 0;
  bool first_chunk = true;
  for (std::size_t ordinal = 0; ordinal < chunks.size(); ++ordinal)
  {
    const wftrain::ParameterChunkDescriptor& chunk = chunks[ordinal];
    if (chunk.ordinal != ordinal || chunk.parameter_version != parameter_version_ ||
        chunk.provider_id != chunk_plan_.providerId() ||
        chunk.schema_fingerprint != chunk_plan_.schemaFingerprint() ||
        chunk.scalar_domain != wftrain::ParameterScalarDomain::REAL64 ||
        chunk.count == 0 || chunk.count > host_chunk_capacity_ ||
        chunk.parameter_offset > parameter_count_ ||
        chunk.count > parameter_count_ - chunk.parameter_offset)
      throw std::invalid_argument("PsiFormer derivative publication chunk plan is noncanonical");

    const bool new_tensor = first_chunk || chunk.block_index != prior_tensor;
    if (new_tensor)
    {
      if (!first_chunk && (chunk.block_index <= prior_tensor ||
                           expected_block_offset !=
                               active_tensor_end - active_tensor_begin))
        throw std::invalid_argument(
            "PsiFormer derivative publication blocks are incomplete or reordered");
      if (chunk.block_index >= plan.parameter_tensor_count)
        throw std::invalid_argument(
            "PsiFormer derivative publication block does not name a device tensor");

      bool found_tensor = false;
      active_tensor_begin = 0;
      active_tensor_end = 0;
      for (const PsiFormerParameterChunkRecord& planned : plan.parameter_chunks.chunks)
        if (planned.tensor_ordinal == chunk.block_index)
        {
          if (!found_tensor)
            active_tensor_begin = planned.canonical_begin_fp64_elements;
          active_tensor_end = planned.canonicalEndFp64Elements();
          found_tensor = true;
        }
      if (!found_tensor || chunk.block_offset != 0 ||
          chunk.parameter_offset != active_tensor_begin)
        throw std::invalid_argument(
            "PsiFormer derivative publication block is relabeled or misaligned");

      expected_block_offset = 0;
      prior_tensor = chunk.block_index;
      first_chunk = false;
    }

    if (chunk.block_index != prior_tensor ||
        chunk.block_offset != expected_block_offset ||
        chunk.parameter_offset != active_tensor_begin + expected_block_offset ||
        chunk.parameter_offset > active_tensor_end ||
        chunk.count > active_tensor_end - chunk.parameter_offset)
      throw std::invalid_argument(
          "PsiFormer derivative publication chunk crosses or gaps a device tensor");

    expected_block_offset += chunk.count;
    selected_count += chunk.count;
  }
  if ((!chunks.empty() && expected_block_offset !=
                            active_tensor_end - active_tensor_begin) ||
      selected_count != chunk_plan_.selectedParameterCount())
    throw std::invalid_argument(
        "PsiFormer derivative publication selected tensor coverage is incomplete");
}

void PsiFormerDerivativeHostPublisher::validateCompletion(
    const PsiFormerDerivativeCompletionToken& completion) const
{
  if (completion.status != 0)
    throw std::runtime_error("PsiFormer derivative completion reported a device failure");
  if (completion.plan_fingerprint != plan_fingerprint_ ||
      completion.device_layout_fingerprint != device_layout_fingerprint_ ||
      completion.device_arena_fingerprint != device_arena_fingerprint_ ||
      completion.parameter_version != parameter_version_)
    throw std::runtime_error("PsiFormer derivative completion identity is stale or incompatible");
}

void PsiFormerDerivativeHostPublisher::validateParameterSink(
    const PsiFormerDerivativeCompletionToken& completion,
    const wftrain::ParameterReductionSink& sink) const
{
  const wftrain::DerivativeStreamDescriptor* descriptor = sink.activeDescriptor();
  const wftrain::ParameterChunkPlan* sink_plan = sink.activeChunkPlan();
  if (descriptor == nullptr || sink_plan == nullptr)
    throw std::logic_error("PsiFormer derivative publication requires an active parameter sink");
  if (completion.kind != PsiFormerDerivativePublicationKind::VJP)
    throw std::invalid_argument("PsiFormer parameter publication requires a VJP completion");
  if (!sameChunkPlan(chunk_plan_, *sink_plan))
    throw std::invalid_argument("PsiFormer parameter publication sink chunk plan differs from the prepared plan");
  if (completion.channel_count == 0 || completion.channel_count > 3 ||
      completion.channel_count > accumulator_count_ ||
      sink.activeChannelCount() != completion.channel_count)
    throw std::invalid_argument("PsiFormer parameter publication sink channel count differs from completion");
  if (descriptor->provider_id != chunk_plan_.providerId() ||
      descriptor->schema_fingerprint != chunk_plan_.schemaFingerprint() ||
      descriptor->parameter_version != parameter_version_ ||
      descriptor->adjoint != completion.adjoint ||
      descriptor->parameter_scalar_domain != wftrain::ParameterScalarDomain::REAL64 ||
      descriptor->result_scalar_domain != wftrain::ParameterScalarDomain::COMPLEX128 ||
      descriptor->reduction_domain != wftrain::ReductionDomain::CROWD_LOCAL ||
      descriptor->execution_domain != wftrain::DerivativeExecutionDomain::HOST ||
      descriptor->maximum_vjp_channels != accumulator_count_ ||
      descriptor->maximum_parameter_chunk_size != host_chunk_capacity_ ||
      descriptor->maximum_sample_tile_size != 1 || descriptor->sample_count == 0 ||
      !isVJPProductMask(descriptor->product_mask))
    throw std::invalid_argument("PsiFormer parameter publication sink descriptor differs from completion");
}

void PsiFormerDerivativeHostPublisher::validateSampleSink(
    const PsiFormerDerivativeCompletionToken& completion,
    const wftrain::SampleProductSink& sink) const
{
  const wftrain::DerivativeStreamDescriptor* descriptor = sink.activeDescriptor();
  if (descriptor == nullptr)
    throw std::logic_error("PsiFormer score JVP publication requires an active sample sink");
  if (completion.kind != PsiFormerDerivativePublicationKind::SCORE_JVP ||
      completion.adjoint != wftrain::DerivativeAdjoint::TRANSPOSE ||
      completion.channel_count != 0)
    throw std::invalid_argument("PsiFormer sample publication requires a score-JVP completion");
  if (descriptor->provider_id != chunk_plan_.providerId() ||
      descriptor->schema_fingerprint != chunk_plan_.schemaFingerprint() ||
      descriptor->parameter_version != parameter_version_ ||
      descriptor->product_mask !=
          wftrain::derivativeProductBit(wftrain::DerivativeProduct::SCORE_JVP) ||
      descriptor->adjoint != wftrain::DerivativeAdjoint::TRANSPOSE ||
      descriptor->parameter_scalar_domain != wftrain::ParameterScalarDomain::REAL64 ||
      descriptor->result_scalar_domain != wftrain::ParameterScalarDomain::COMPLEX128 ||
      descriptor->batch_ordinal != completion.batch_ordinal ||
      descriptor->reduction_domain != wftrain::ReductionDomain::CROWD_LOCAL ||
      descriptor->execution_domain != wftrain::DerivativeExecutionDomain::HOST ||
      descriptor->maximum_parameter_chunk_size != host_chunk_capacity_ ||
      descriptor->maximum_sample_tile_size != 1 ||
      completion.sample_offset < descriptor->sample_offset ||
      completion.sample_offset - descriptor->sample_offset >= descriptor->sample_count ||
      completion.tile_ordinal !=
          completion.sample_offset - descriptor->sample_offset)
    throw std::invalid_argument("PsiFormer score JVP sink descriptor differs from completion");
}

void PsiFormerDerivativeHostPublisher::publishParameterChunks(
    const PsiFormerDerivativeCompletionToken& completion,
    const double* accumulators,
    std::size_t accumulator_fp64_elements,
    wftrain::DerivativeValue* host_staging,
    std::size_t host_staging_capacity,
    PsiFormerDerivativeHostStager& stager,
    wftrain::ParameterReductionSink& sink) const
{
  validateCompletion(completion);
  validateParameterSink(completion, sink);
  if (accumulators == nullptr || accumulator_fp64_elements < accumulator_total_fp64_elements_)
    throw std::invalid_argument("PsiFormer derivative publication accumulator storage is too small");
  if (!chunk_plan_.chunks().empty() &&
      (host_staging == nullptr || host_staging_capacity < host_chunk_capacity_))
    throw std::invalid_argument("PsiFormer derivative publication host staging is too small");

  for (const wftrain::ParameterChunkDescriptor& chunk : chunk_plan_.chunks())
    for (std::size_t channel = 0; channel < completion.channel_count; ++channel)
    {
      const double* source = accumulators +
          channel * accumulator_stride_fp64_elements_ +
          2 * chunk.parameter_offset;
      stager.copyComplexChunk(source, chunk.count, host_staging);
      stager.synchronize();
      sink.add(channel, {chunk, {host_staging, chunk.count}});
      // sink.add is synchronous.  Only now may the single staging interval be reused.
    }
}

void PsiFormerDerivativeHostPublisher::publishJVPScalar(
    const PsiFormerDerivativeCompletionToken& completion,
    const double* device_result,
    wftrain::DerivativeValue* host_staging,
    std::size_t host_staging_capacity,
    PsiFormerDerivativeHostStager& stager,
    wftrain::SampleProductSink& sink) const
{
  validateCompletion(completion);
  validateSampleSink(completion, sink);
  if (host_jvp_capacity_ < 1 || device_result == nullptr || host_staging == nullptr ||
      host_staging_capacity < 1)
    throw std::invalid_argument("PsiFormer score JVP publication lacks scalar staging");

  stager.copyRealJVPScalar(device_result, host_staging);
  stager.synchronize();
  const wftrain::SampleProductTileDescriptor descriptor{
      chunk_plan_.providerId(), chunk_plan_.schemaFingerprint(),
      parameter_version_, completion.batch_ordinal, completion.sample_offset, 1,
      completion.tile_ordinal};
  sink.add({descriptor, {host_staging, 1}});
}

} // namespace qmcplusplus::psiformer
