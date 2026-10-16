//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceCoordinateGradient.h
 * @brief Borrowed-value-tape real-core all-coordinate log-amplitude gradient.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_H

#include "PsiFormerDeviceNetworkReverse.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerCoordinateGradientLeaf.h"

#include <limits>

namespace qmcplusplus::psiformer::device
{

/// Complete independent electron rows using the already admitted raw input VJP.
Error launchCoordinateGradientLeaves(Stream stream, const OpenOrbitalJetLayout& orbital,
    const OpenFeatureJetLayout& raw, const double* positions, const double* nuclei,
    const double* raw_bar, const double* envelope_bar, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, double same_alpha, double opposite_alpha,
    double* gradient, OpenSpatialStatus* status, const periodic_geometry::Cell* cell = nullptr);

/** Borrowed completed packed [configuration,electron,xyz] device gradients.
 * The real-core derivative excludes any fixed outer-twist phase. The forward
 * source must stay alive; its model identity and generation tag this result.
 */
struct CoordinateGradientView
{
  const double* data;
  std::size_t configuration_count, electron_count;
  const PreparedDeviceModel* model_image;
  std::size_t model_version, evaluation_generation;
};

/// Separate fixed-capacity reverse device and host payload requirements.
struct CoordinateGradientCost
{
  std::size_t device_workspace_bytes = 0;
  std::size_t host_workspace_bytes = 0;
};

/** Exact fixed-capacity no-P reverse payload, excluding its borrowed forward.
 * Count owned numeric/status buffers only, not model images, vendor storage,
 * allocator overhead, caller coordinates or caller gradient copies.
 */
inline CoordinateGradientCost estimateCoordinateGradientCost(const PsiFormerExecutionPlan& plan,
                                                             std::size_t capacity)
{
  if (capacity == 0)
    throw std::invalid_argument("Coordinate gradient preparation requires positive capacity");
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Coordinate gradient cost extent overflow");
  };
  const auto add = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedAdd(a, b, "Coordinate gradient cost storage overflow");
  };
  const auto chunks = [](std::size_t n) { return n / 4096 + (n % 4096 != 0); };
  const auto& s = plan.modelShape();
  const auto ne = s.electrons(), rows = mul(capacity, ne);
  const auto raw = mul(rows, plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0]);
  const auto coordinates = mul(rows, 3);
  std::size_t doubles = 0;
  for (const auto count : {mul(11, mul(rows, s.feature_dimension)),
      mul(mul(rows, s.attention_heads), ne), raw, mul(4, mul(mul(rows, s.determinants), ne)),
      coordinates, capacity})
    doubles = add(doubles, count);
  const auto finite = checkedDeviceBytes<DerivativeFiniteStatus>(add(chunks(raw), chunks(coordinates)));
  const auto status = add(checkedDeviceBytes<device_determinant::DerivativeStatus>(capacity),
      checkedDeviceBytes<OpenSpatialStatus>(rows));
  CoordinateGradientCost cost;
  cost.device_workspace_bytes = add(checkedDeviceBytes<double>(doubles), add(finite, status));
  cost.host_workspace_bytes = add(add(checkedDeviceBytes<double>(capacity),
      checkedDeviceBytes<device_determinant::FactorizationMetadata>(mul(capacity, s.determinants))),
      add(finite, status));
  return cost;
}

/** Own only reverse storage; borrow one completed retained value forward.
 * The caller must keep the source object alive through every result access until
 * the next reverse/rebind or owner destruction: this pointer is not shared ownership.
 * Publication checks its generation again, preventing reuse with changed inputs.
 * No parameter-response vector, coordinate tangent planes or Laplacian exists.
 */
class PreparedDeviceCoordinateGradient
{
public:
  /// Admit this reverse owner's device payload before constructing any buffers.
  /// The retained VALUE graph and host payload remain separately reported costs.
  static std::unique_ptr<PreparedDeviceCoordinateGradient> prepare(
      std::shared_ptr<const PreparedDeviceModel> model, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, std::size_t device_budget_bytes)
  {
    if (!model || device_budget_bytes == 0 || device_budget_bytes == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("Coordinate gradient requires a model and positive finite device byte budget");
    model->validateCurrentDevice();
    if (model->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL)
      throw std::invalid_argument("Coordinate gradient requires real amplitude metadata");
    const auto cost = estimateCoordinateGradientCost(model->plan(), capacity);
    if (cost.device_workspace_bytes > device_budget_bytes)
      throw std::length_error("Coordinate gradient reverse owner exceeds device byte budget");
    auto owner = std::make_unique<PreparedDeviceCoordinateGradient>(std::move(model), capacity, queue, blas);
    if (owner->deviceWorkspaceBytes() != cost.device_workspace_bytes ||
        owner->hostWorkspaceBytes() != cost.host_workspace_bytes)
      throw std::logic_error("Coordinate gradient storage disagrees with pre-allocation accounting");
    return owner;
  }

  /// Prepare fixed capacities independently of the caller's forward allocation.
  PreparedDeviceCoordinateGradient(std::shared_ptr<const PreparedDeviceModel> model, std::size_t capacity,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
      : model_(std::move(model)), capacity_(capacity), queue_(queue), blas_(blas),
        network_(model_, capacity, queue, blas, NetworkReverseOutput::RAW_FEATURE_ADJOINT_ONLY)
  {
    const auto& s = model_->plan().modelShape();
    const auto rows = multiply(capacity_, s.electrons());
    const auto matrices = multiply(multiply(rows, s.determinants), s.electrons());
    for (auto* buffer : {&seeds_, &up_, &down_, &envelope_})
      buffer->reserve(matrices);
    feature_.reserve(multiply(rows, s.feature_dimension));
    gradient_.reserve(multiply(rows, 3));
    weights_.reserve(capacity_);
    unit_weights_.resize(capacity_, 1.0);
    weights_.upload(unit_weights_.data(), capacity_);
    seed_status_.reserve(capacity_);
    row_status_.reserve(rows);
    host_seed_status_.resize(capacity_);
    host_row_status_.resize(rows);
    host_factors_.resize(multiply(capacity_, s.determinants));
    const auto chunks = gradient_.capacity() / 4096 + (gradient_.capacity() % 4096 != 0);
    finite_status_.reserve(chunks);
    host_finite_status_.resize(chunks);
  }

  /// Reverse unit-weight log|Psi| for each independent configuration in the tape.
  CoordinateGradientView reverse(const PreparedDeviceInference& forward)
  {
    ready_ = false;
    model_->validateCurrentDevice();
    const auto count = forward.valueTapeBatchCount();
    const auto& s = model_->plan().modelShape();
    const auto ne = s.electrons();
    if (forward.modelImage().get() != model_.get() || count == 0 || count > capacity_)
      throw std::invalid_argument("Coordinate gradient model or live batch is incompatible");
    const auto generation = forward.evaluationGeneration();
    const auto head = forward.valueHeadTape();

    // Match existing exact score admission: singular channels require a cofactor
    // route and cannot be silently omitted, even when their normalized seed is zero.
    forward.downloadValueFactorizations(host_factors_);
    for (std::size_t channel = 0; channel < count * s.determinants; ++channel)
      if (!host_factors_[channel].inverse_available)
        throw std::runtime_error("Coordinate gradient requires regular determinant channels");
    const auto stream = queue_.getNative();
    checkRuntime(launchDeterminantMatrixReverseSeeds(stream, head.factorizations, head.combinations,
        head.normalized_weights, head.inverses, count, s.determinants, ne, seeds_.data(), seed_status_.data()),
        "coordinate determinant seeds");
    queue_.sync();
    seed_status_.download(host_seed_status_.data(), count);
    for (std::size_t configuration = 0; configuration < count; ++configuration)
      if (host_seed_status_[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
        throw std::runtime_error("Coordinate log gradient is undefined at this determinant combination");

    // Existing product packing preserves finite zero factors without division.
    const OrbitalProductReverseLayout product{count, s.determinants, ne, ne, ne * ne, s.determinants * ne * ne};
    checkRuntime(launchWeightedOrbitalProductReverse(stream, product, s.spin_up_electrons,
        weights_.data(), head.backflow, head.envelope, seeds_.data(), up_.data(), down_.data(), envelope_.data()),
        "coordinate head product reverse");
    checkRuntime(launchDerivativeZero(stream, {count * ne * s.feature_dimension, 0, 1, 0, 1}, feature_.data()),
        "coordinate feature adjoint zero");
    const DenseForwardLayout dense{count * ne, s.feature_dimension, s.determinants * ne,
        s.feature_dimension, s.determinants * ne, s.determinants * ne};
    for (const auto& spin : {std::pair{ParameterRole::BACKFLOW_UP_WEIGHT, &up_},
                            std::pair{ParameterRole::BACKFLOW_DOWN_WEIGHT, &down_}})
      accumulateDenseInputAdjoint(blas_, dense, spin.second->data(),
          parameter(spin.first), feature_.data());
    const auto raw = network_.reverseRawFeatures(forward, {}, feature_.data());
    if (raw.model_version != model_->version() || raw.evaluation_generation != generation)
      throw std::logic_error("Coordinate raw cotangent provenance changed during reverse");

    const auto orbital = makeOpenOrbitalJetLayout(count, s.determinants, ne, s.nuclei,
        s.feature_dimension, s.spin_up_electrons, SpatialJetMode::VALUE_ONLY);
    const auto input = model_->periodicCell()
        ? makePeriodicFeatureJetLayout(count, ne, s.nuclei, s.spin_up_electrons, SpatialJetMode::VALUE_ONLY)
        : makeOpenFeatureJetLayout(count, ne, s.nuclei, s.spin_up_electrons, SpatialJetMode::VALUE_ONLY);
    if (raw.layout.width != input.feature_width || raw.layout.configuration_stride != ne * input.feature_width)
      throw std::logic_error("Coordinate raw feature layout differs from the model geometry policy");
    checkRuntime(launchCoordinateGradientLeaves(stream, orbital, input, head.positions, model_->nuclei(),
        raw.data, envelope_.data(), parameter(ParameterRole::ENVELOPE_PI_UP), parameter(ParameterRole::ENVELOPE_PI_DOWN),
        parameter(ParameterRole::ENVELOPE_ZETA_UP), parameter(ParameterRole::ENVELOPE_ZETA_DOWN),
        model_->sameAlpha(), model_->oppositeAlpha(), gradient_.data(), row_status_.data(), model_->periodicCell()),
        "coordinate direct and feature leaves");
    const auto elements = count * ne * 3;
    const DerivativeFiniteScanLayout scan{elements, 4096, elements / 4096 + (elements % 4096 != 0), 0, 1};
    checkRuntime(launchDerivativeFiniteScan(stream, scan, gradient_.data(), finite_status_.data()),
        "coordinate gradient finite scan");
    queue_.sync();
    row_status_.download(host_row_status_.data(), count * ne);
    finite_status_.download(host_finite_status_.data(), scan.status_capacity);
    for (std::size_t row = 0; row < count * ne; ++row)
      if (host_row_status_[row] != OpenSpatialStatus::REGULAR)
        throw std::runtime_error("Coordinate gradient has a nondifferentiable or nonfinite geometry row");
    for (std::size_t chunk = 0; chunk < scan.status_capacity; ++chunk)
      if (host_finite_status_[chunk].nonfinite_count != 0)
        throw std::runtime_error("Coordinate gradient is nonfinite");
    forward_ = &forward;
    count_ = count;
    generation_ = generation;
    ready_ = true;
    return view();
  }

  /// Recheck the borrowed generation before publishing or downloading a result.
  CoordinateGradientView view() const
  {
    if (!ready_ || !forward_)
      throw std::logic_error("Coordinate gradient has no completed borrowed source");
    model_->validateCurrentDevice();
    if (forward_->modelImage().get() != model_.get() ||
        forward_->evaluationGeneration() != generation_ || forward_->valueTapeBatchCount() != count_)
      throw std::logic_error("Coordinate gradient has no completed current source generation");
    return {gradient_.data(), count_, model_->plan().modelShape().electrons(), model_.get(), model_->version(), generation_};
  }

  /// Download the completed live gradient without exposing unused capacity tails.
  void downloadGradient(std::vector<double>& destination) const
  {
    const auto result = view();
    if (destination.size() != result.configuration_count * result.electron_count * 3)
      throw std::invalid_argument("Coordinate gradient destination extent mismatch");
    gradient_.download(destination.data(), destination.size());
  }

  /// Rebind compatible immutable weights and revoke the previous borrowed result.
  void rebindModel(std::shared_ptr<const PreparedDeviceModel> successor)
  {
    ready_ = false;
    model_->validateSuccessor(successor);
    network_.rebindModel(successor);
    model_ = std::move(successor);
    forward_ = nullptr;
  }

  /// Count only this reverse owner's device allocations, excluding source/model.
  std::size_t deviceWorkspaceBytes() const
  {
    auto bytes = network_.deviceWorkspaceBytes();
    for (const auto* buffer : {&seeds_, &up_, &down_, &envelope_, &feature_, &gradient_, &weights_})
      bytes = spatial_detail::checkedAdd(bytes, checkedDeviceBytes<double>(buffer->capacity()), "Coordinate storage overflow");
    for (const auto allocation : {checkedDeviceBytes<device_determinant::DerivativeStatus>(seed_status_.capacity()),
             checkedDeviceBytes<OpenSpatialStatus>(row_status_.capacity()),
             checkedDeviceBytes<DerivativeFiniteStatus>(finite_status_.capacity())})
      bytes = spatial_detail::checkedAdd(bytes, allocation, "Coordinate storage overflow");
    return bytes;
  }

  /// Count retained host admission/status storage, excluding caller coordinates.
  std::size_t hostWorkspaceBytes() const
  {
    auto bytes = network_.hostWorkspaceBytes();
    for (const auto allocation : {checkedDeviceBytes<double>(unit_weights_.capacity()),
             checkedDeviceBytes<device_determinant::FactorizationMetadata>(host_factors_.capacity()),
             checkedDeviceBytes<device_determinant::DerivativeStatus>(host_seed_status_.capacity()),
             checkedDeviceBytes<OpenSpatialStatus>(host_row_status_.capacity()),
             checkedDeviceBytes<DerivativeFiniteStatus>(host_finite_status_.capacity())})
      bytes = spatial_detail::checkedAdd(bytes, allocation, "Coordinate host storage overflow");
    return bytes;
  }

  /// Report the live borrowed source separately; this owner never allocates it.
  std::size_t borrowedForwardDeviceBytes() const
  {
    (void)view();
    return forward_->deviceWorkspaceBytes();
  }

  /// Keep borrowed forward host admission storage separate from this owner too.
  std::size_t borrowedForwardHostBytes() const
  {
    (void)view();
    return forward_->hostWorkspaceBytes();
  }

private:
  /// Check preparation extents before reserving numerical buffers.
  static std::size_t multiply(std::size_t a, std::size_t b)
  { return spatial_detail::checkedProduct(a, b, "Coordinate gradient extent overflow"); }

  /// Bind immutable canonical head weights without allocating their adjoints.
  const double* parameter(ParameterRole role) const
  { return model_->parameters() + model_->plan().parameter(role).begin; }

  std::shared_ptr<const PreparedDeviceModel> model_;
  std::size_t capacity_, count_ = 0, generation_ = 0;
  compute::Queue<PlatformKind::CUDA>& queue_;
  AcceleratorBlasHandle& blas_;
  PreparedDeviceNetworkReverse network_;
  DeviceBuffer<double> seeds_, up_, down_, envelope_, feature_, gradient_, weights_;
  DeviceBuffer<device_determinant::DerivativeStatus> seed_status_;
  DeviceBuffer<OpenSpatialStatus> row_status_;
  DeviceBuffer<DerivativeFiniteStatus> finite_status_;
  std::vector<double> unit_weights_;
  std::vector<device_determinant::FactorizationMetadata> host_factors_;
  std::vector<device_determinant::DerivativeStatus> host_seed_status_;
  std::vector<OpenSpatialStatus> host_row_status_;
  std::vector<DerivativeFiniteStatus> host_finite_status_;
  const PreparedDeviceInference* forward_ = nullptr;
  bool ready_ = false;
};

} // namespace qmcplusplus::psiformer::device

#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_COORDINATE_GRADIENT_H
