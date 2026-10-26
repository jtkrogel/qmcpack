//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file FrozenReferenceResidualDerivative.cpp
 * Bounded CPU score products for a sum with an entirely frozen reference branch.
 */
#include "FrozenReferenceResidualWF.h"
#include "Optimization/StreamingDerivative.h"
#include <array>
#include <limits>
#include <mutex>

namespace qmcplusplus
{

/** One clone-family evaluator and O(P) reductions, reused across retained sample blocks.
 * Operators retain coordinates only; neither evaluators nor score rows grow with block count.
 */
struct FrozenResidualDerivativeWorkspace
{
  std::mutex mutex;
  std::unique_ptr<ParticleSet> particles;
  std::unique_ptr<FrozenReferenceResidualWF> worker;
  std::vector<double> score;
  std::array<std::vector<wftrain::DerivativeValue>, 3> reductions;
};

namespace
{
using namespace wftrain;

/** Replay immutable coordinates through one serialized private evaluator.
 * Only real score products are admitted: physical local energies are sampled separately.
 */
class FrozenResidualScoreOperator final : public StreamingDerivativeOperator
{
public:
  /// Capture batch coordinates and retain the shared evaluator without a sample-by-P matrix.
  FrozenResidualScoreOperator(std::shared_ptr<FrozenResidualDerivativeWorkspace> workspace,
      const StructuredParameterSchema& schema, std::size_t version,
      const RefVectorWithLeader<ParticleSet>& particles, std::size_t batch, std::size_t offset,
      std::size_t chunk_size)
      : workspace_(std::move(workspace)), schema_(schema), version_(version), batch_(batch), offset_(offset),
        plan_(schema_, version_, chunk_size)
  {
    if (offset > std::numeric_limits<std::size_t>::max() - particles.size())
      throw std::overflow_error("Frozen reference sum sample interval overflows");
    workspace_->score.resize(schema_.parameterCount());
    for (auto& reduction : workspace_->reductions)
      reduction.resize(schema_.parameterCount());
    positions_.reserve(particles.size());
    for (const ParticleSet& sample : particles)
    {
      if (sample.getTotalNum() != workspace_->particles->getTotalNum())
        throw std::invalid_argument("Frozen reference sum score batch electron extent differs");
      positions_.emplace_back(sample.R.begin(), sample.R.end());
    }
  }

  /// Advertise only transpose/Hermitian products of real parent scores.
  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.maximum_vjp_channels = 3;
    result.maximum_parameter_chunk_size = plan_.maximumChunkSize();
    result.maximum_sample_tile_size = 1;
    result.fixed_parameter_scratch_vectors = 7;
    result.block_streaming = true;
    return result;
  }

  /// Report numeric coordinate/reduction storage; the shared evaluator is never duplicated per block.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    std::lock_guard lock(workspace_->mutex);
    StreamingDerivativeStorageDiagnostics result;
    result.parameter_count = schema_.parameterCount();
    result.sample_count = positions_.size();
    result.real_parameter_vectors = 1;
    result.complex_parameter_vectors = 3;
    result.parameter_scratch_bytes = workspace_->score.capacity() * sizeof(double);
    for (const auto& reduction : workspace_->reductions)
      result.parameter_scratch_bytes += reduction.capacity() * sizeof(DerivativeValue);
    for (const auto& positions : positions_)
      result.sample_position_bytes += positions.capacity() * sizeof(ParticleSet::PosType);
    result.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(workspace_.get());
    result.shared_workspace_bytes = result.parameter_scratch_bytes;
    result.retained_numeric_bytes = result.parameter_scratch_bytes + result.sample_position_bytes;
    result.storage_fingerprint = result.shared_workspace_identity;
    return result;
  }

  /// Return the immutable parent tensor schema.
  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  /// Identify the single model version for all products of this batch.
  std::size_t parameterVersion() const noexcept override { return version_; }

  /// Preserve the caller's ordered batch identity.
  std::size_t batchOrdinal() const noexcept override { return batch_; }

  /// Preserve the first sample's global/rank interval offset.
  std::size_t sampleOffset() const noexcept override { return offset_; }

  /// Return the number of retained physical configurations.
  std::size_t sampleCount() const noexcept override { return positions_.size(); }

  /// Expose the exact bounded canonical chunk layout used by sinks.
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

protected:
  /// Contract each complete real score once into at most three complex weighted reductions.
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint, ParameterReductionSink& sink) const override
  {
    std::lock_guard lock(workspace_->mutex);
    requireVersion();
    for (std::size_t c = 0; c < channels.size(); ++c)
      workspace_->reductions[c].assign(schema_.parameterCount(), DerivativeValue{});
    for (std::size_t sample = 0; sample < positions_.size(); ++sample)
    {
      evaluateSample(sample);
      for (std::size_t c = 0; c < channels.size(); ++c)
        for (std::size_t p = 0; p < workspace_->score.size(); ++p)
          workspace_->reductions[c][p] += channels[c].coefficients.values[sample] * workspace_->score[p];
    }
    requireVersion();
    for (const auto& chunk : plan_.chunks())
      for (std::size_t c = 0; c < channels.size(); ++c)
        sink.add(c, {chunk, {workspace_->reductions[c].data() + chunk.parameter_offset, chunk.count}});
  }

  /// Contract one direction against each score and deliver a one-sample tile immediately.
  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                        SampleProductSink& sink) const override
  {
    std::lock_guard lock(workspace_->mutex);
    requireVersion();
    for (std::size_t sample = 0; sample < positions_.size(); ++sample)
    {
      evaluateSample(sample);
      DerivativeValue value{};
      for (std::size_t p = 0; p < workspace_->score.size(); ++p)
        value += direction.values()[p] * workspace_->score[p];
      const SampleProductTileDescriptor tile{schema_.providerId(), schema_.fingerprint(), version_,
          batch_, offset_ + sample, 1, sample};
      sink.add({tile, {&value, 1}});
    }
    requireVersion();
  }

private:
  /// Refuse replay after publication instead of mixing old samples with new parameter products.
  void requireVersion() const
  {
    if (workspace_->worker->currentParameterVersion() != version_)
      throw std::runtime_error("Frozen reference sum score stream has a stale parameter version");
  }

  /// Restore only positions; the component refreshes its private branch distance tables itself.
  void evaluateSample(std::size_t sample) const
  {
    for (std::size_t e = 0; e < positions_[sample].size(); ++e)
      workspace_->particles->R[e] = positions_[sample][e];
    workspace_->worker->evaluateParameterScore(*workspace_->particles, workspace_->score);
  }

  std::shared_ptr<FrozenResidualDerivativeWorkspace> workspace_;
  StructuredParameterSchema schema_;
  std::size_t version_, batch_, offset_;
  ParameterChunkPlan plan_;
  std::vector<std::vector<ParticleSet::PosType>> positions_;
};
} // namespace

/// Create one empty owner before walker clones start sharing the model family.
std::shared_ptr<FrozenResidualDerivativeWorkspace> FrozenReferenceResidualWF::createDerivativeWorkspace()
{
  return std::make_shared<FrozenResidualDerivativeWorkspace>();
}

/// Reuse one private evaluator across all immutable batches of the clone family.
std::unique_ptr<wftrain::StreamingDerivativeOperator> FrozenReferenceResidualWF::makeStreamingDerivativeOperator(
    const RefVectorWithLeader<WaveFunctionComponent>& components,
    const RefVectorWithLeader<ParticleSet>& particles, std::size_t batch_ordinal,
    std::size_t sample_offset, std::size_t maximum_parameter_chunk_size) const
{
  if (!training_enabled_ || components.size() != particles.size() || particles.empty())
    throw std::invalid_argument("Frozen reference sum score stream requires a selected nonempty matching batch");
  std::lock_guard lock(derivative_workspace_->mutex);
  const auto version = currentParameterVersion();
  for (WaveFunctionComponent& component : components)
  {
    const auto* parent = dynamic_cast<const FrozenReferenceResidualWF*>(&component);
    if (!parent || parent->derivative_workspace_ != derivative_workspace_ ||
        parent->currentParameterVersion() != version)
      throw std::invalid_argument("Frozen reference sum score batch mixes model families or versions");
  }
  if (!derivative_workspace_->worker)
  {
    derivative_workspace_->particles = std::make_unique<ParticleSet>(particles.getLeader());
    auto worker = makeClone(*derivative_workspace_->particles);
    derivative_workspace_->worker.reset(static_cast<FrozenReferenceResidualWF*>(worker.release()));
    // Break the ownership cycle: this private evaluator never creates further derivative streams.
    derivative_workspace_->worker->derivative_workspace_.reset();
  }
  return std::make_unique<FrozenResidualScoreOperator>(derivative_workspace_, *parameter_schema_, version,
      particles, batch_ordinal, sample_offset, maximum_parameter_chunk_size);
}

} // namespace qmcplusplus
