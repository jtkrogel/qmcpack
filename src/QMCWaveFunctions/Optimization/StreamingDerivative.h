//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file StreamingDerivative.h
 * @brief Bounded, schema-aware derivative products for high-parameter wave functions.
 *
 * The contract deliberately exposes only vector products and bounded one-dimensional
 * tiles.  It has no dense sample-by-parameter representation or compatibility fallback.
 * Producers bind every stream to an immutable parameter schema/version and an ordered
 * sample batch; checked sinks enforce canonical, complete delivery before results become
 * usable.
 */

#ifndef QMCPLUSPLUS_STREAMING_DERIVATIVE_H
#define QMCPLUSPLUS_STREAMING_DERIVATIVE_H

#include "Configuration.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace qmcplusplus::wftrain
{

/// Full-precision real scalar used for derivative reductions.
using DerivativeReal = QMCTraits::FullPrecRealType;

/** Full-precision contraction scalar retained even in a real QMCPACK build.
 *
 * Keeping complex intermediates makes transpose and Hermitian products explicit and
 * avoids embedding an irreversible real projection in the producer/sink interface.
 */
using DerivativeValue = std::complex<DerivativeReal>;

/// Identify the matrix-free derivative product carried by a stream or channel.
enum class DerivativeProduct : std::uint32_t
{
  SCORE_VJP        = UINT32_C(1) << 0,
  LOCAL_ENERGY_VJP = UINT32_C(1) << 1,
  SCORE_JVP        = UINT32_C(1) << 2
};

/// Select transpose or conjugate-transpose contraction of a complex derivative.
enum class DerivativeAdjoint : std::uint32_t
{
  TRANSPOSE = UINT32_C(1) << 0,
  HERMITIAN = UINT32_C(1) << 1
};

/// State where a completed reduction is valid and may be consumed.
enum class ReductionDomain
{
  CROWD_LOCAL,
  RANK_LOCAL,
  GLOBAL
};

/// Execution memory/domain used by the derivative producer.
enum class DerivativeExecutionDomain
{
  HOST,
  DEVICE
};

/// Hamiltonian contribution included by a local-energy parameter derivative.
enum class LocalEnergyTerm : std::uint32_t
{
  KINETIC      = UINT32_C(1) << 0,
  NONLOCAL_ECP = UINT32_C(1) << 1
};

/// Observable state of a checked streaming sink.
enum class DerivativeSinkState
{
  IDLE,
  ACTIVE,
  COMPLETE,
  POISONED
};

/// Return the capability-mask bit associated with one derivative product.
constexpr std::uint32_t derivativeProductBit(DerivativeProduct product) noexcept
{
  return static_cast<std::uint32_t>(product);
}

/// Return the capability-mask bit associated with one adjoint convention.
constexpr std::uint32_t derivativeAdjointBit(DerivativeAdjoint adjoint) noexcept
{
  return static_cast<std::uint32_t>(adjoint);
}

/// Return the coverage-mask bit associated with one local-energy term.
constexpr std::uint32_t localEnergyTermBit(LocalEnergyTerm term) noexcept
{
  return static_cast<std::uint32_t>(term);
}

/** Minimal const-correct C++17 view over one contiguous interval.
 *
 * The view has no ownership or two-dimensional shape, so production interfaces cannot
 * use it to communicate a sample-by-parameter matrix accidentally.
 */
template<class T>
class DerivativeArrayView
{
public:
  using value_type = std::remove_cv_t<T>;

  /// Construct an empty view.
  constexpr DerivativeArrayView() noexcept = default;

  /// Construct a view over `size` values beginning at `data`.
  constexpr DerivativeArrayView(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

  /// Convert a mutable view to the corresponding const view.
  template<class U, std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>, int> = 0>
  constexpr DerivativeArrayView(const DerivativeArrayView<U>& other) noexcept
      : data_(other.data()), size_(other.size())
  {}

  /// Return the first element address, or nullptr for an empty view.
  constexpr T* data() const noexcept { return data_; }

  /// Return the number of values in the interval.
  constexpr std::size_t size() const noexcept { return size_; }

  /// Report whether the interval contains no values.
  constexpr bool empty() const noexcept { return size_ == 0; }

  /// Access one value without bounds checking, matching span semantics.
  constexpr T& operator[](std::size_t index) const noexcept { return data_[index]; }

  /// Return an iterator to the first value.
  constexpr T* begin() const noexcept { return data_; }

  /// Return an iterator one past the last value.
  constexpr T* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }

private:
  T* data_          = nullptr;
  std::size_t size_ = 0;
};

/// Select whether a diagnostic chunk plan also exposes non-trainable blocks.
enum class FrozenBlockPolicy
{
  EXCLUDE,
  INCLUDE_FOR_DIAGNOSTICS
};

/** Describe one canonical parameter tile without per-scalar metadata.
 *
 * A descriptor belongs to one immutable chunk plan and never crosses a schema block.
 */
struct ParameterChunkDescriptor
{
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t block_index       = 0;
  std::string block_id;
  std::size_t block_offset     = 0;
  std::size_t parameter_offset = 0;
  std::size_t count            = 0;
  ParameterScalarDomain scalar_domain = ParameterScalarDomain::REAL64;
  bool trainable                       = true;
  std::size_t ordinal                  = 0;
};

/** Immutable canonical partition of selected structured parameter blocks.
 *
 * Construction performs all layout and overflow validation.  The public API exposes no
 * mutator, so producers and sinks share stable ordinals for the lifetime of the plan.
 */
class ParameterChunkPlan
{
public:
  /// Partition each selected block into tiles no larger than `maximum_chunk_size`.
  ParameterChunkPlan(const StructuredParameterSchema& schema,
                     std::size_t parameter_version,
                     std::size_t maximum_chunk_size,
                     FrozenBlockPolicy frozen_policy = FrozenBlockPolicy::EXCLUDE);

  /// Return the provider identity copied from the source schema.
  const std::string& providerId() const noexcept { return provider_id_; }

  /// Return the source schema fingerprint.
  const std::string& schemaFingerprint() const noexcept { return schema_fingerprint_; }

  /// Return the parameter version to which every chunk is bound.
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

  /// Return the homogeneous scalar domain of the source schema.
  ParameterScalarDomain scalarDomain() const noexcept { return scalar_domain_; }

  /// Return the requested upper bound on a chunk's scalar count.
  std::size_t maximumChunkSize() const noexcept { return maximum_chunk_size_; }

  /// Return the immutable canonical chunks.
  const std::vector<ParameterChunkDescriptor>& chunks() const noexcept { return chunks_; }

  /// Return the total number of scalar entries selected by this plan.
  std::size_t selectedParameterCount() const noexcept { return selected_parameter_count_; }

private:
  std::string provider_id_;
  std::string schema_fingerprint_;
  std::size_t parameter_version_ = 0;
  ParameterScalarDomain scalar_domain_ = ParameterScalarDomain::REAL64;
  std::size_t maximum_chunk_size_       = 0;
  std::size_t selected_parameter_count_ = 0;
  std::vector<ParameterChunkDescriptor> chunks_;
};

/// Associate one bounded nonowning parameter interval with its canonical descriptor.
template<class T>
class BasicParameterChunkView
{
public:
  /// Construct a descriptor-bound interval.
  BasicParameterChunkView(const ParameterChunkDescriptor& descriptor, DerivativeArrayView<T> values) noexcept
      : descriptor_(&descriptor), values_(values)
  {}

  /// Return the canonical metadata for this interval.
  const ParameterChunkDescriptor& descriptor() const noexcept { return *descriptor_; }

  /// Return the nonowning values.
  DerivativeArrayView<T> values() const noexcept { return values_; }

private:
  const ParameterChunkDescriptor* descriptor_ = nullptr;
  DerivativeArrayView<T> values_;
};

using ParameterChunkView      = BasicParameterChunkView<DerivativeValue>;
using ParameterChunkConstView = BasicParameterChunkView<const DerivativeValue>;

/** Validate and expose one full canonical parameter vector without ownership.
 *
 * The vector remains one-dimensional.  Block and chunk access derives only bounded
 * intervals from schema metadata and never adds a sample dimension.
 */
class StructuredParameterVectorConstView
{
public:
  /// Bind a complete vector to one schema and version after validating its extent.
  StructuredParameterVectorConstView(const StructuredParameterSchema& schema,
                                     std::size_t parameter_version,
                                     DerivativeArrayView<const DerivativeValue> values);

  /// Prevent a nonowning view from retaining a pointer to a temporary schema.
  StructuredParameterVectorConstView(StructuredParameterSchema&&,
                                     std::size_t,
                                     DerivativeArrayView<const DerivativeValue>) = delete;

  /// Prevent the same dangling-schema error for a const-qualified temporary.
  StructuredParameterVectorConstView(const StructuredParameterSchema&&,
                                     std::size_t,
                                     DerivativeArrayView<const DerivativeValue>) = delete;

  /// Return the bound provider identity.
  const std::string& providerId() const noexcept { return schema_->providerId(); }

  /// Return the bound schema fingerprint.
  const std::string& schemaFingerprint() const noexcept { return schema_->fingerprint(); }

  /// Return the bound parameter version.
  std::size_t parameterVersion() const noexcept { return parameter_version_; }

  /// Return the complete nonowning canonical vector.
  DerivativeArrayView<const DerivativeValue> values() const noexcept { return values_; }

  /// Return the interval occupied by one schema block.
  DerivativeArrayView<const DerivativeValue> block(std::size_t block_index) const;

  /// Return the interval named by one compatible chunk descriptor.
  ParameterChunkConstView chunk(const ParameterChunkDescriptor& descriptor) const;

private:
  const StructuredParameterSchema* schema_ = nullptr;
  std::size_t parameter_version_            = 0;
  DerivativeArrayView<const DerivativeValue> values_;
};

/// Bind one coefficient interval to the exact provider version and ordered sample batch.
struct CoefficientView
{
  std::string_view provider_id;
  std::string_view schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t batch_ordinal     = 0;
  std::size_t sample_offset     = 0;
  DerivativeArrayView<const DerivativeValue> values;
};

/// Describe one independently weighted channel in a fused VJP traversal.
struct VJPCoefficientChannel
{
  std::string_view id;
  DerivativeProduct product = DerivativeProduct::SCORE_VJP;
  CoefficientView coefficients;
  std::uint32_t local_energy_term_mask = 0;
};

/** Carry immutable identity and execution metadata for one sink transaction.
 *
 * A VJP transaction may have several channel products; `product_mask` is their union.
 */
struct DerivativeStreamDescriptor
{
  std::string_view provider_id;
  std::string_view schema_fingerprint;
  std::size_t parameter_version = 0;
  std::uint32_t product_mask    = 0;
  DerivativeAdjoint adjoint     = DerivativeAdjoint::TRANSPOSE;
  ParameterScalarDomain parameter_scalar_domain = ParameterScalarDomain::REAL64;
  ParameterScalarDomain result_scalar_domain    = ParameterScalarDomain::COMPLEX128;
  std::size_t batch_ordinal = 0;
  std::size_t sample_offset = 0;
  std::size_t sample_count  = 0;
  ReductionDomain reduction_domain             = ReductionDomain::CROWD_LOCAL;
  DerivativeExecutionDomain execution_domain   = DerivativeExecutionDomain::HOST;
  std::size_t maximum_vjp_channels             = 0;
  std::size_t maximum_parameter_chunk_size     = 0;
  std::size_t maximum_sample_tile_size         = 0;
};

/// Describe one bounded, ordered interval of JVP sample results.
struct SampleProductTileDescriptor
{
  std::string_view provider_id;
  std::string_view schema_fingerprint;
  std::size_t parameter_version = 0;
  std::size_t batch_ordinal     = 0;
  std::size_t sample_offset     = 0;
  std::size_t count             = 0;
  std::size_t ordinal           = 0;
};

/// Associate a JVP sample-tile descriptor with its nonowning values.
class SampleProductTileConstView
{
public:
  /// Construct a descriptor-bound sample interval.
  SampleProductTileConstView(const SampleProductTileDescriptor& descriptor,
                             DerivativeArrayView<const DerivativeValue> values) noexcept
      : descriptor_(&descriptor), values_(values)
  {}

  /// Return the ordered tile metadata.
  const SampleProductTileDescriptor& descriptor() const noexcept { return *descriptor_; }

  /// Return the nonowning JVP values.
  DerivativeArrayView<const DerivativeValue> values() const noexcept { return values_; }

private:
  const SampleProductTileDescriptor* descriptor_ = nullptr;
  DerivativeArrayView<const DerivativeValue> values_;
};

/** Checked consumer of canonical multi-channel VJP parameter chunks.
 *
 * Delivery order is chunk-major, then channel ordinal.  Any validation or derived-sink
 * failure poisons the transaction; only `reset()` makes the sink reusable.
 */
class ParameterReductionSink
{
public:
  ParameterReductionSink() = default;
  ParameterReductionSink(const ParameterReductionSink&) = delete;
  ParameterReductionSink& operator=(const ParameterReductionSink&) = delete;
  ParameterReductionSink(ParameterReductionSink&&) = delete;
  ParameterReductionSink& operator=(ParameterReductionSink&&) = delete;
  virtual ~ParameterReductionSink() = default;

  /// Start a checked transaction and expose immutable metadata to the derived sink.
  void begin(const DerivativeStreamDescriptor& descriptor,
             const ParameterChunkPlan& chunk_plan,
             DerivativeArrayView<const VJPCoefficientChannel> channels);

  /// Consume the next canonical channel/chunk pair.
  void add(std::size_t channel_ordinal, const ParameterChunkConstView& chunk);

  /// Complete a nonempty-sample transaction after exact canonical coverage.
  void end();

  /// Complete a zero-sample transaction without accepting parameter chunks.
  void endEmptyBatch();

  /// Abandon the active transaction and leave the sink poisoned until reset.
  void abort() noexcept;

  /// Discard completed or poisoned derived state and return to idle.
  void reset();

  /// Return the current checked lifecycle state.
  DerivativeSinkState state() const noexcept { return state_; }

protected:
  /// Prepare derived storage for a validated transaction.
  virtual void onBegin(const DerivativeStreamDescriptor&,
                       const ParameterChunkPlan&,
                       DerivativeArrayView<const VJPCoefficientChannel>)
  {}

  /// Accumulate one validated channel/chunk pair.
  virtual void consume(std::size_t channel_ordinal, const ParameterChunkConstView& chunk) = 0;

  /// Publish derived results after complete validated coverage.
  virtual void onEnd() {}

  /// Drop any partial derived result without throwing.
  virtual void onAbort() noexcept {}

  /// Reinitialize derived storage deterministically before the next transaction.
  virtual void onReset() noexcept {}

private:
  [[noreturn]] void poisonAndThrow(const std::string& message);
  void clearTransaction() noexcept;

  DerivativeSinkState state_ = DerivativeSinkState::IDLE;
  const DerivativeStreamDescriptor* descriptor_ = nullptr;
  const ParameterChunkPlan* chunk_plan_          = nullptr;
  std::size_t channel_count_                     = 0;
  std::size_t next_record_                       = 0;
  std::size_t expected_records_                  = 0;
};

/** Checked consumer of monotonically ordered, bounded JVP sample tiles.
 *
 * The sink retains only scalar coverage counters; the derived implementation chooses
 * whether to reduce or store the one-dimensional sample result.
 */
class SampleProductSink
{
public:
  SampleProductSink() = default;
  SampleProductSink(const SampleProductSink&) = delete;
  SampleProductSink& operator=(const SampleProductSink&) = delete;
  SampleProductSink(SampleProductSink&&) = delete;
  SampleProductSink& operator=(SampleProductSink&&) = delete;
  virtual ~SampleProductSink() = default;

  /// Start one checked JVP transaction.
  void begin(const DerivativeStreamDescriptor& descriptor);

  /// Consume the next contiguous sample tile.
  void add(const SampleProductTileConstView& tile);

  /// Complete a nonempty-sample transaction after exact sample coverage.
  void end();

  /// Complete a zero-sample transaction without accepting tiles.
  void endEmptyBatch();

  /// Abandon the active transaction and leave the sink poisoned until reset.
  void abort() noexcept;

  /// Discard completed or poisoned derived state and return to idle.
  void reset();

  /// Return the current checked lifecycle state.
  DerivativeSinkState state() const noexcept { return state_; }

protected:
  /// Prepare derived storage for a validated JVP transaction.
  virtual void onBegin(const DerivativeStreamDescriptor&) {}

  /// Accumulate one validated sample tile.
  virtual void consume(const SampleProductTileConstView& tile) = 0;

  /// Publish derived results after complete validated coverage.
  virtual void onEnd() {}

  /// Drop any partial derived result without throwing.
  virtual void onAbort() noexcept {}

  /// Reinitialize derived storage deterministically before the next transaction.
  virtual void onReset() noexcept {}

private:
  [[noreturn]] void poisonAndThrow(const std::string& message);
  void clearTransaction() noexcept;

  DerivativeSinkState state_ = DerivativeSinkState::IDLE;
  const DerivativeStreamDescriptor* descriptor_ = nullptr;
  std::size_t next_sample_offset_                = 0;
  std::size_t next_tile_ordinal_                 = 0;
};

/** Advertise bounded products supported by one concrete derivative operator.
 *
 * Local-energy coverage is independent of the product bit: a kinetic-only producer
 * cannot satisfy a request that also contains nonlocal pseudopotential derivatives.
 */
struct StreamingDerivativeCapabilities
{
  std::uint32_t product_mask = 0;
  std::uint32_t adjoint_mask = 0;
  ParameterScalarDomain parameter_scalar_domain = ParameterScalarDomain::REAL64;
  ParameterScalarDomain result_scalar_domain    = ParameterScalarDomain::COMPLEX128;
  ReductionDomain reduction_domain             = ReductionDomain::CROWD_LOCAL;
  DerivativeExecutionDomain execution_domain   = DerivativeExecutionDomain::HOST;
  std::uint32_t local_energy_term_mask          = 0;
  std::size_t maximum_vjp_channels              = 0;
  std::size_t maximum_parameter_chunk_size      = 0;
  std::size_t maximum_sample_tile_size          = 0;
  /// Fixed number of full-P scratch vectors retained by the producer.
  std::size_t fixed_parameter_scratch_vectors   = 0;
  bool block_streaming                          = false;

  /// Report whether one product is implemented.
  bool supports(DerivativeProduct product) const noexcept;

  /// Report whether one adjoint convention is implemented.
  bool supports(DerivativeAdjoint adjoint) const noexcept;

  /// Report whether all requested local-energy contributions are implemented.
  bool coversLocalEnergyTerms(std::uint32_t requested_terms) const noexcept;
};

/** Exact retained numeric-storage diagnostics for a prepared derivative producer.
 *
 * Metadata owned by schemas and chunk descriptors is deliberately excluded.  The
 * reported bytes cover every numeric buffer retained by the producer and therefore
 * provide the quantity that a future driver must add to its aggregate memory budget.
 */
struct StreamingDerivativeStorageDiagnostics
{
  std::size_t parameter_count                 = 0;
  std::size_t sample_count                    = 0;
  std::size_t real_parameter_vectors          = 0;
  std::size_t complex_parameter_vectors       = 0;
  std::size_t parameter_scratch_bytes         = 0;
  std::size_t evaluator_workspace_bytes       = 0;
  std::size_t sample_position_bytes           = 0;
  std::size_t sample_product_bytes            = 0;
  std::size_t retained_numeric_bytes          = 0;
  std::size_t allocation_generation           = 0;
  std::size_t storage_fingerprint              = 0;
};

/** Common checked front end for bounded multi-channel VJP and score JVP kernels.
 *
 * The public wrappers preflight complete capabilities and metadata before beginning a
 * sink transaction.  Concrete kernels emit only bounded chunks/tiles through the sink.
 */
class StreamingDerivativeOperator
{
public:
  virtual ~StreamingDerivativeOperator() = default;

  /// Return the complete immutable capability declaration.
  virtual StreamingDerivativeCapabilities capabilities() const noexcept = 0;

  /// Return the exact schema bound to this operator.
  virtual const StructuredParameterSchema& parameterSchema() const noexcept = 0;

  /// Return the parameter version bound to this operator.
  virtual std::size_t parameterVersion() const noexcept = 0;

  /// Return the ordered sample-batch identity.
  virtual std::size_t batchOrdinal() const noexcept = 0;

  /// Return the global/local offset of the first sample in this batch.
  virtual std::size_t sampleOffset() const noexcept = 0;

  /// Return the number of samples in this batch.
  virtual std::size_t sampleCount() const noexcept = 0;

  /// Return the immutable chunk plan used by all VJP products.
  virtual const ParameterChunkPlan& parameterChunkPlan() const noexcept = 0;

  /** Return exact prepared storage evidence.
   *
   * The compatibility default describes no retained storage.  Production
   * high-parameter providers override this method so their independent owner can be
   * incorporated into the driver-wide budget before execution.
   */
  virtual StreamingDerivativeStorageDiagnostics storageDiagnostics() const noexcept { return {}; }

  /// Apply several independently weighted score/local-energy VJPs in one traversal.
  void applyVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                 DerivativeAdjoint adjoint,
                 ParameterReductionSink& sink) const;

  /// Apply the score Jacobian to one schema/version-bound parameter direction.
  void applyScoreJVP(const StructuredParameterVectorConstView& direction, SampleProductSink& sink) const;

protected:
  /// Emit chunk-major/channel-minor VJP results after common preflight succeeds.
  virtual void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                            DerivativeAdjoint adjoint,
                            ParameterReductionSink& sink) const = 0;

  /// Emit monotonically ordered bounded JVP sample tiles after common preflight succeeds.
  virtual void evaluateScoreJVP(const StructuredParameterVectorConstView& direction,
                                SampleProductSink& sink) const = 0;
};

} // namespace qmcplusplus::wftrain

#endif
