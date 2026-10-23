//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceAffineStatistics.h
 * @brief Bounded real/open affine sufficient statistics with sample transactions.
 *
 * Numerical owner only. Registry/fallback statistics, rank reduction and optimizer
 * recurrence belong to the provider/trainer. No sample-by-parameter rows persist.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_AFFINE_STATISTICS_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_AFFINE_STATISTICS_H

#include "PsiFormerDeviceInference.h"

namespace qmcplusplus::psiformer::device
{

namespace affine_statistics_detail
{
/// Gram triangles may differ by BLAS rounding; publish the lower triangle once.
/// The caller has already finite-admitted the complete source matrix.
inline void copySymmetricGram(const double* source, std::size_t width,
    double* destination, std::size_t destination_stride)
{
  for (std::size_t row = 0; row < width; ++row)
    for (std::size_t column = 0; column <= row; ++column)
    {
      const double value = source[row * width + column];
      destination[row * destination_stride + column] = value;
      destination[column * destination_stride + row] = value;
    }
}
} // namespace affine_statistics_detail

/// Bind one registered affine weight and optional bias to its canonical tensor.
struct DeviceAffineFactorDescriptor
{
  ParameterRole role;
  std::size_t attention_block;
  std::size_t input_width;
  std::size_t output_width;
  bool bias_augmented;
  HeadTensorComponent head_component = HeadTensorComponent::REAL;
};

/** Borrow rows only for the callback; consumers enqueue work on the same stream. */
class DeviceAffineObservationSink
{
public:
  virtual ~DeviceAffineObservationSink() = default;
  virtual void observe(ParameterRole role, std::size_t layer, const double* activations,
      const double* sensitivities, std::size_t rows, std::size_t input_width,
      std::size_t output_width) = 0;
};

/** Caller-sized HOST destinations retain raw sums, never row-normalized factors. */
struct DeviceAffineFactorSums
{
  std::vector<double> activation_outer_sum;
  std::vector<double> sensitivity_outer_sum;
  std::size_t row_count = 0;
  double sample_weight_sum = 0;
  double row_weight_sum = 0;
};

/** Accumulate raw A/G factors without retaining sample rows.
 * The pending bank is finite-admitted before a HOST fallback transaction; a
 * nonthrowing receipt then publishes it by swapping the two prepared banks.
 */
class PreparedDeviceAffineStatistics final : public DeviceAffineObservationSink
{
public:
  /** Finite-admitted pending factors; the owner must outlive this receipt.
   * Destruction discards a prepared sample if its HOST fallback commit fails.
   */
  class SampleCommit
  {
  public:
    SampleCommit(const SampleCommit&) = delete;
    SampleCommit& operator=(const SampleCommit&) = delete;
    SampleCommit(SampleCommit&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), generation_(other.generation_) {}
    ~SampleCommit()
    {
      if (owner_ && owner_->generation_ == generation_)
        owner_->abortSample();
    }
    void commit() noexcept
    {
      if (owner_ && owner_->generation_ == generation_ && owner_->prepared_)
      {
        owner_->accepted_.swap(owner_->pending_);
        ++owner_->samples_;
        owner_->prepared_ = false;
      }
      owner_ = nullptr;
    }
  private:
    friend class PreparedDeviceAffineStatistics;
    SampleCommit(PreparedDeviceAffineStatistics& owner, std::size_t generation) noexcept
        : owner_(&owner), generation_(generation) {}
    PreparedDeviceAffineStatistics* owner_;
    std::size_t generation_;
  };

  /// Raw A and G plus input sums for bias; two banks preserve sample atomicity.
  static std::size_t requiredDeviceBytes(const std::vector<DeviceAffineFactorDescriptor>& blocks)
  {
    const auto elements = requiredElements(blocks);
    return add(mul(2, checkedDeviceBytes<double>(elements)),
        checkedDeviceBytes<DerivativeFiniteStatus>(statusCount(elements)));
  }

  /// Packed factor download and finite-status storage, excluding caller outputs.
  static std::size_t requiredHostBytes(const std::vector<DeviceAffineFactorDescriptor>& blocks)
  {
    const auto elements = requiredElements(blocks);
    return add(checkedDeviceBytes<double>(elements),
        checkedDeviceBytes<DerivativeFiniteStatus>(statusCount(elements)));
  }

  /// Admit exact block shapes and both numeric budgets before allocating banks.
  PreparedDeviceAffineStatistics(std::shared_ptr<const PreparedDeviceModel> model,
      std::vector<DeviceAffineFactorDescriptor> blocks, bool gaussian_curvature,
      std::size_t device_budget, std::size_t host_budget,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
      : model_(std::move(model)), blocks_(std::move(blocks)), gaussian_(gaussian_curvature),
        queue_(queue), blas_(blas)
  {
    if (!model_ || blocks_.empty() || !device_budget || !host_budget ||
        device_budget == std::numeric_limits<std::size_t>::max() ||
        host_budget == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Affine statistics require a model, blocks and finite positive budgets");
    model_->validateCurrentDevice();
    const auto& plan = model_->plan();
    const auto& environment = plan.environment();
    paired_ = environment.amplitude_scalar_domain == ScalarDomain::COMPLEX;
    if (environment.periodic_profile || (paired_ &&
        (environment.boundary != BoundaryCondition::PERIODIC || !environment.fixed_nuclei ||
         environment.geometry_feature_policy != GeometryFeaturePolicy::PERIODIC_TORUS_V1 ||
         environment.head_representation != HeadRepresentation::PAIRED_REAL_COMPLEX ||
         environment.parameter_scalar_domain != ScalarDomain::REAL ||
         environment.compute_scalar_domain != ScalarDomain::REAL)) ||
        (!paired_ && environment.amplitude_scalar_domain != ScalarDomain::REAL))
      throw std::invalid_argument("Device affine statistics require real open or legacy paired torus metadata");
    if (requiredDeviceBytes(blocks_) > device_budget || requiredHostBytes(blocks_) > host_budget)
      throw std::length_error("Device affine statistics exceed prepared numeric budgets");
    const auto& shape = plan.modelShape();
    std::size_t offset = 0;
    for (std::size_t index = 0; index < blocks_.size(); ++index)
    {
      const auto& block = blocks_[index];
      if (block.role != ParameterRole::ELECTRON_EMBEDDING_WEIGHT &&
          block.role != ParameterRole::ATTENTION_QUERY_WEIGHT &&
          block.role != ParameterRole::ATTENTION_KEY_WEIGHT &&
          block.role != ParameterRole::ATTENTION_VALUE_WEIGHT &&
          block.role != ParameterRole::ATTENTION_OUTPUT_WEIGHT &&
          block.role != ParameterRole::UPDATE_HIDDEN_WEIGHT &&
          block.role != ParameterRole::UPDATE_OUTPUT_WEIGHT &&
          block.role != ParameterRole::BACKFLOW_UP_WEIGHT &&
          block.role != ParameterRole::BACKFLOW_DOWN_WEIGHT)
        throw std::invalid_argument("Device affine role is not a supported real open layer");
      const auto& tensor = plan.parameter(block.role, block.attention_block, block.head_component);
      if (tensor.shape != std::vector<std::size_t>{block.input_width, block.output_width})
        throw std::invalid_argument("Device affine dimensions do not match the canonical tensor");
      for (std::size_t earlier = 0; earlier < index; ++earlier)
        if (blocks_[earlier].role == block.role &&
            blocks_[earlier].attention_block == block.attention_block &&
            blocks_[earlier].head_component == block.head_component)
          throw std::invalid_argument("Device affine registry contains a duplicate tensor");
      const bool biased = block.role == ParameterRole::UPDATE_HIDDEN_WEIGHT ||
                          block.role == ParameterRole::UPDATE_OUTPUT_WEIGHT;
      if (block.bias_augmented != biased)
        throw std::invalid_argument("Device affine bias registration is incompatible");
      const auto rows = block.role == ParameterRole::BACKFLOW_UP_WEIGHT ? shape.spin_up_electrons :
                        block.role == ParameterRole::BACKFLOW_DOWN_WEIGHT ? shape.spin_down_electrons :
                        shape.electrons();
      if (!rows)
        throw std::invalid_argument("Device affine block has no observation rows");
      const auto activation = mul(block.input_width, block.input_width);
      const auto sensitivity = mul(block.output_width, block.output_width);
      layout_.push_back({offset, offset + activation, offset + activation + sensitivity, rows});
      offset = add(offset, add(add(activation, sensitivity), block.bias_augmented ? block.input_width : 0));
    }
    elements_ = offset;
    accepted_ = std::make_unique<DeviceBuffer<double>>(elements_, "AffineStatistics::accepted", __FILE__, __LINE__);
    pending_ = std::make_unique<DeviceBuffer<double>>(elements_, "AffineStatistics::pending", __FILE__, __LINE__);
    finite_layout_ = {elements_, 4096, statusCount(elements_), 0, 1};
    status_.reserveNamed(finite_layout_.status_capacity, "AffineStatistics::status", __FILE__, __LINE__);
    host_status_.resize(finite_layout_.status_capacity);
    host_packed_.resize(elements_);
    observed_.resize(blocks_.size());
    activation_addresses_.resize(blocks_.size());
    if (deviceWorkspaceBytes() > device_budget || hostWorkspaceBytes() > host_budget)
      throw std::length_error("Actual device affine numeric capacities exceed prepared budgets");
    zero(*accepted_);
  }

  /// Initial admission matches the trainer's explicit beginSample(1) convention.
  void beginSample(std::size_t parameter_version, double sample_weight = 1.0)
  {
    if (active_ || prepared_)
      throw std::logic_error("Device affine sample transaction is already active");
    model_->validateCurrentDevice();
    if (parameter_version != model_->version() || sample_weight != 1.0)
      throw std::invalid_argument("Device affine statistics require current parameters and unit sample weight");
    if (samples_ == std::numeric_limits<std::size_t>::max() || generation_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Device affine sample count overflow");
    for (const auto& layout : layout_)
      (void)mul(samples_ + 1, layout.rows);
    std::fill(observed_.begin(), observed_.end(), 0);
    std::fill(activation_addresses_.begin(), activation_addresses_.end(), nullptr);
    zero(*pending_);
    ++generation_;
    active_ = true;
  }

  /// Frozen layers absent from the prepared registry have no factor destination.
  void observe(ParameterRole role, std::size_t layer, const double* activations,
      const double* sensitivities, std::size_t rows, std::size_t input_width,
      std::size_t output_width) override
  {
    observeChannel(role, layer, HeadTensorComponent::REAL, 0, activations, sensitivities,
        rows, input_width, output_width);
  }

  /// Pair two objectives of the SAME real tensor without doubling activations/counts.
  void observeChannel(ParameterRole role, std::size_t layer, HeadTensorComponent component,
      std::size_t objective, const double* activations, const double* sensitivities,
      std::size_t rows, std::size_t input_width, std::size_t output_width)
  {
    if (!active_)
      throw std::logic_error("Device affine observation requires an active sample");
    std::size_t ordinal = 0;
    for (; ordinal < blocks_.size(); ++ordinal)
      if (blocks_[ordinal].role == role && blocks_[ordinal].attention_block == layer &&
          blocks_[ordinal].head_component == component)
        break;
    if (ordinal == blocks_.size())
      return;
    const auto& block = blocks_[ordinal];
    const auto& layout = layout_[ordinal];
    if (objective > (paired_ ? 1u : 0u) || observed_[ordinal] != (objective ? 1u : 0u) ||
        !activations || !sensitivities || rows != layout.rows ||
        (objective && activation_addresses_[ordinal] != activations) ||
        input_width != block.input_width || output_width != block.output_width)
      throw std::invalid_argument("Device affine observation extent or transaction mismatch");
    const DenseForwardLayout activation{rows, input_width, input_width,
        input_width, input_width, input_width};
    const DenseForwardLayout sensitivity{rows, output_width, output_width,
        output_width, output_width, output_width};
    if (!objective)
    {
      activation_addresses_[ordinal] = activations;
      accumulateDenseWeightAdjoint(blas_, activation, activations, activations,
          pending_->data() + layout.activation);
    }
    accumulateDenseWeightAdjoint(blas_, sensitivity, sensitivities, sensitivities,
        pending_->data() + layout.sensitivity);
    if (block.bias_augmented && !objective)
      checkRuntime(launchDenseBiasAdjoint(queue_.getNative(), activation, activations,
          pending_->data() + layout.input_sum), "affine activation bias sums");
    observed_[ordinal] |= static_cast<unsigned char>(1u << objective);
  }

  /// Validate DEVICE pending sums before committing HOST fallback state.
  SampleCommit prepareSampleCommit()
  {
    const unsigned char expected = paired_ ? 3 : 1;
    if (!active_ || !std::all_of(observed_.begin(), observed_.end(),
        [expected](unsigned char value) { return value == expected; }))
    {
      active_ = false;
      throw std::logic_error("Device affine sample lacks a registered observation");
    }
    active_ = false;
    admitPending();
    if (gaussian_)
    {
      for (std::size_t block = 0; block < blocks_.size(); ++block)
        checkRuntime(launchDerivativeScale(queue_.getNative(),
            {mul(blocks_[block].output_width, blocks_[block].output_width), 0, 1, layout_[block].sensitivity, 1},
            2.0, pending_->data()), "Gaussian affine sensitivity scale");
      admitPending();
    }
    checkRuntime(launchDerivativeAxpy(queue_.getNative(), {elements_, 0, 1, 0, 1},
        1.0, accepted_->data(), pending_->data()), "transactional affine factor sum");
    admitPending();
    prepared_ = true;
    return SampleCommit(*this, generation_);
  }

  /// Standalone numerical callers need no intervening HOST fallback transaction.
  void finishSample()
  {
    auto prepared = prepareSampleCommit();
    prepared.commit();
  }

  void abortSample() noexcept
  {
    active_ = false;
    prepared_ = false;
  }
  std::size_t sampleCount() const noexcept { return samples_; }

  std::size_t parameterVersion() const noexcept { return model_->version(); }

  /// Borrow only the exact immutable image used to prepare these factor banks.
  void requireModel(const PreparedDeviceModel* image) const
  {
    if (image != model_.get())
      throw std::invalid_argument("Device affine statistics belong to a different image");
    model_->validateCurrentDevice();
  }
  const PreparedDeviceModel* modelIdentity() const noexcept { return model_.get(); }

  /// Count the two raw numeric banks and their finite-scan status owner.
  std::size_t deviceWorkspaceBytes() const
  {
    return add(checkedDeviceBytes<double>(add(accepted_->capacity(), pending_->capacity())),
        checkedDeviceBytes<DerivativeFiniteStatus>(status_.capacity()));
  }

  std::size_t hostWorkspaceBytes() const
  {
    return add(checkedDeviceBytes<double>(host_packed_.capacity()),
        checkedDeviceBytes<DerivativeFiniteStatus>(host_status_.capacity()));
  }

  /// Download factor-sized sums once after collection, into caller-sized arrays.
  void download(std::vector<DeviceAffineFactorSums>& destination)
  {
    if (active_ || prepared_ || !samples_ || destination.size() != blocks_.size())
      throw std::logic_error("Device affine factors have no complete matching destination");
    for (std::size_t block = 0; block < blocks_.size(); ++block)
    {
      const auto input = blocks_[block].input_width + blocks_[block].bias_augmented;
      if (destination[block].activation_outer_sum.size() != mul(input, input) ||
          destination[block].sensitivity_outer_sum.size() != mul(blocks_[block].output_width, blocks_[block].output_width))
        throw std::invalid_argument("Device affine download arrays must be prepared to registry dimensions");
    }
    accepted_->download(host_packed_.data(), elements_);
    for (std::size_t ordinal = 0; ordinal < blocks_.size(); ++ordinal)
    {
      const auto& block = blocks_[ordinal];
      const auto& layout = layout_[ordinal];
      auto& target = destination[ordinal];
      const auto augmented = block.input_width + block.bias_augmented;
      affine_statistics_detail::copySymmetricGram(host_packed_.data() + layout.activation,
          block.input_width, target.activation_outer_sum.data(), augmented);
      if (block.bias_augmented)
      {
        for (std::size_t input = 0; input < block.input_width; ++input)
          target.activation_outer_sum[input * augmented + block.input_width] =
              target.activation_outer_sum[block.input_width * augmented + input] = host_packed_[layout.input_sum + input];
        target.activation_outer_sum.back() = static_cast<double>(mul(samples_, layout.rows));
      }
      affine_statistics_detail::copySymmetricGram(host_packed_.data() + layout.sensitivity,
          block.output_width, target.sensitivity_outer_sum.data(), block.output_width);
      target.row_count = mul(samples_, layout.rows);
      target.sample_weight_sum = static_cast<double>(samples_);
      target.row_weight_sum = static_cast<double>(target.row_count);
    }
  }

private:
  struct BlockLayout { std::size_t activation, sensitivity, input_sum, rows; };
  static std::size_t mul(std::size_t left, std::size_t right)
  { return spatial_detail::checkedProduct(left, right, "Device affine statistics extent overflow"); }
  static std::size_t add(std::size_t left, std::size_t right)
  { return spatial_detail::checkedAdd(left, right, "Device affine statistics extent overflow"); }
  static std::size_t statusCount(std::size_t elements)
  { return elements / 4096 + (elements % 4096 != 0); }
  static std::size_t requiredElements(const std::vector<DeviceAffineFactorDescriptor>& blocks)
  {
    std::size_t elements = 0;
    for (const auto& block : blocks)
    {
      if (!block.input_width || !block.output_width)
        throw std::invalid_argument("Device affine statistics require positive block dimensions");
      elements = add(elements, add(add(mul(block.input_width, block.input_width),
          mul(block.output_width, block.output_width)), block.bias_augmented ? block.input_width : 0));
    }
    return elements;
  }
  void zero(DeviceBuffer<double>& buffer)
  { checkRuntime(launchDerivativeZero(queue_.getNative(), {elements_, 0, 1, 0, 1}, buffer.data()), "affine statistics zero"); }
  /// Raw scale/Axpy callers require finite inputs as well as final admission.
  void admitPending()
  {
    checkRuntime(launchDerivativeFiniteScan(queue_.getNative(), finite_layout_, pending_->data(),
        status_.data()), "affine factor finite scan");
    queue_.sync();
    status_.download(host_status_.data(), host_status_.size());
    for (const auto& status : host_status_)
      if (status.nonfinite_count)
        throw std::runtime_error("Device affine sample or accumulated factors are nonfinite");
  }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::vector<DeviceAffineFactorDescriptor> blocks_;
  std::vector<BlockLayout> layout_;
  bool gaussian_, paired_ = false, active_ = false, prepared_ = false;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  std::size_t elements_ = 0, samples_ = 0, generation_ = 0;
  std::unique_ptr<DeviceBuffer<double>> accepted_, pending_;
  DeviceBuffer<DerivativeFiniteStatus> status_;
  DerivativeFiniteScanLayout finite_layout_;
  std::vector<DerivativeFiniteStatus> host_status_;
  std::vector<double> host_packed_;
  std::vector<unsigned char> observed_;
  std::vector<const double*> activation_addresses_;
};

} // namespace qmcplusplus::psiformer::device
#endif
