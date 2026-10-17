//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexOrbital.h
 * @brief Prepared CPU complex projections of real trunk and real envelope jets.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_ORBITAL_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_ORBITAL_H

#include "PsiFormerComplexDeterminant.h"
#include "PsiFormerSpatialLayout.h"

namespace qmcplusplus::psiformer
{
/// Matrix structure is independent of the real trunk and the orbital phase policy.
enum class ComplexOrbitalStructure { FULL, SPIN_BLOCKED };

/// Only reciprocal inner phases are admitted; the common twist belongs to the adapter.
enum class ComplexOrbitalPhase { NONE, RECIPROCAL_ONLY };

/** Shape and inner momentum metadata copied into immutable prepared ownership.
 * Momenta are Cartesian G vectors in [spin,determinant,orbital] order, with spin-up
 * entries first. FULL uses N columns for each nonempty spin; SPIN_BLOCKED uses Ns.
 * RECIPROCAL_ONLY requires exactly the corresponding number of finite G vectors.
 * The caller must establish reciprocal-lattice membership and gauge identity before
 * creating this numerical descriptor; this helper has no lattice or outer twist.
 */
struct ComplexOrbitalHeadDescriptor
{
  std::size_t electrons = 0;
  std::size_t spin_up = 0;
  std::size_t determinants = 0;
  std::size_t feature_width = 0;
  ComplexOrbitalStructure structure = ComplexOrbitalStructure::FULL;
  ComplexOrbitalPhase phase = ComplexOrbitalPhase::NONE;
  std::vector<std::array<double, 3>> reciprocal_momenta;
};

/** Two real affine projections for one row-spin sector.
 * Weights use [feature,determinant,orbital]; optional biases use [determinant,orbital].
 * Both weight arrays have weight_count entries. Each supplied bias has bias_count
 * entries. Empty spin sectors need no pointers. Parameters and directions are real.
 */
struct ComplexOrbitalSpinParameters
{
  const double* real_weights = nullptr;
  const double* imaginary_weights = nullptr;
  const double* real_bias = nullptr;
  const double* imaginary_bias = nullptr;
  std::size_t weight_count = 0;
  std::size_t bias_count = 0;
};

/// Separate real parameter views share the same real trunk by physical row.
struct ComplexOrbitalHeadParameters
{
  ComplexOrbitalSpinParameters up, down;
};

/** Borrow compact separate real planes without repacking the neural feature jets.
 * Values are [configuration,element], gradients [configuration,lane,element],
 * and Laplacians [configuration,target,element], with the prepared layout's lanes.
 */
struct ComplexOrbitalFeatureJetView
{
  const double* values = nullptr;
  const double* gradients = nullptr;
  const double* laplacians = nullptr;
  std::size_t value_count = 0, gradient_count = 0, laplacian_count = 0;
};

/** Prepared bounded head algebra for A=E*(h Wr+br+i(h Wi+bi))*exp(iG.r_row).
 *
 * Features contain packed [physical row,feature] elements. Real envelopes and
 * complex output share channel-major matrix elements: FULL [D,N,N], or
 * SPIN_BLOCKED [D,(Nu*Nu followed by Nd*Nd)]. Down-block local row j maps to
 * physical row Nu+j. Spatial planes always differentiate physical target electrons,
 * including all-electron backflow dependence in either block. Existing real geometry
 * and envelope kernels supply those real jets; this class duplicates neither.
 *
 * All storage is prepared at construction, O(planes*D*matrix_elements). This is a
 * CPU numerical helper, without public model, optimizer or accelerator admission.
 */
class ComplexOrbitalHeadWorkspace
{
  using C = determinant::Complex;
  using W = std::complex<long double>;

public:
  /// Copy descriptor and layouts, validate all extents, then reserve hot-call scratch.
  ComplexOrbitalHeadWorkspace(ComplexOrbitalHeadDescriptor descriptor,
      SpatialJetLayout feature_layout, SpatialJetLayout envelope_layout)
      : descriptor_(std::move(descriptor)), features_(feature_layout), envelopes_(envelope_layout),
        output_(makeOutputLayout()), projected_(spatialJetSpanElements(output_)),
        result_(projected_.size())
  {}

  /// Workspaces own mutable staged jets exclusively.
  ComplexOrbitalHeadWorkspace(const ComplexOrbitalHeadWorkspace&) = delete;

  /// Copy assignment cannot share prepared scratch.
  ComplexOrbitalHeadWorkspace& operator=(const ComplexOrbitalHeadWorkspace&) = delete;

  /// Transfer immutable metadata and mutable buffers together.
  ComplexOrbitalHeadWorkspace(ComplexOrbitalHeadWorkspace&&) = default;

  /// Const metadata prevents move assignment; move construction remains available.
  ComplexOrbitalHeadWorkspace& operator=(ComplexOrbitalHeadWorkspace&&) = delete;

  /// Return the immutable numerical descriptor, including copied reciprocal momenta.
  const ComplexOrbitalHeadDescriptor& descriptor() const noexcept { return descriptor_; }

  /// Return the compact output planes suitable for complex determinant value/VGL.
  const SpatialJetLayout& layout() const noexcept { return output_; }

  /** Project and compose one prepared batch, with checked input spans.
   * Positions are packed [configuration,physical electron,Cartesian dimension].
   * ACTIVE uses the explicit physical active_electron, while FULL_VGL honors the
   * supplied target interval. Failed evaluation invalidates all published jets.
   */
  void evaluate(const double* features, std::size_t feature_count,
      const double* envelopes, std::size_t envelope_count,
      const double* positions, std::size_t position_count,
      const ComplexOrbitalHeadParameters& parameters, std::size_t active_electron = 0)
  {
    evaluateImpl(features, feature_count, nullptr, envelopes, envelope_count,
                 positions, position_count, parameters, active_electron);
  }

  /// Consume separate real trunk buffers directly; no additional feature storage is owned.
  void evaluate(const ComplexOrbitalFeatureJetView& features,
      const double* envelopes, std::size_t envelope_count,
      const double* positions, std::size_t position_count,
      const ComplexOrbitalHeadParameters& parameters, std::size_t active_electron = 0)
  {
    evaluateImpl(nullptr, 0, &features, envelopes, envelope_count,
                 positions, position_count, parameters, active_electron);
  }

private:
  /// Validate either feature representation, then share the exact projection/product loops.
  void evaluateImpl(const double* features, std::size_t feature_count,
      const ComplexOrbitalFeatureJetView* split,
      const double* envelopes, std::size_t envelope_count,
      const double* positions, std::size_t position_count,
      const ComplexOrbitalHeadParameters& parameters, std::size_t active_electron)
  {
    valid_ = false;
    if (split)
    {
      const std::size_t count = multiply(features_.configuration_count, features_.element_count);
      validateInput(split->values, split->value_count, count);
      validateInput(split->gradients, split->gradient_count, multiply(count, features_.gradient_lanes));
      validateInput(split->laplacians, split->laplacian_count, multiply(count, features_.laplacian_lanes));
    }
    else
      validateInput(features, feature_count, spatialJetSpanElements(features_));
    validateInput(envelopes, envelope_count, spatialJetSpanElements(envelopes_));
    validateInput(positions, position_count, multiply(multiply(features_.configuration_count,
        descriptor_.electrons), 3));
    if (active_electron >= descriptor_.electrons)
      throw std::invalid_argument("complex orbital active electron exceeds physical count");
    validateParameters(parameters.up, 0);
    validateParameters(parameters.down, 1);
    for (std::size_t configuration = 0; configuration < output_.configuration_count; ++configuration)
      for (std::size_t d = 0; d < descriptor_.determinants; ++d)
        for (std::size_t row = 0; row < descriptor_.electrons; ++row)
        {
          const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
          const auto& p = spin ? parameters.down : parameters.up;
          for (std::size_t orbital = 0; orbital < columns(spin); ++orbital)
          {
            const std::size_t element = matrixElement(d, row, orbital);
            project(features, split, p, configuration, d, row, orbital, element);
            compose(envelopes, positions, configuration, d, row, orbital, element, active_electron);
          }
        }
    valid_ = true;
  }

public:
  /** Real head-parameter JVP with feature and envelope jets held fixed.
   * Linearity permits the same projection using delta-W/delta-bias. This does not
   * claim envelope-parameter response or a real-trunk reverse implementation.
   */
  void evaluateDirection(const double* features, std::size_t feature_count,
      const double* envelopes, std::size_t envelope_count,
      const double* positions, std::size_t position_count,
      const ComplexOrbitalHeadParameters& direction, std::size_t active_electron = 0)
  {
    evaluate(features, feature_count, envelopes, envelope_count, positions, position_count,
             direction, active_electron);
  }

  /// Return completed jets; a failed evaluation cannot expose a preceding generation.
  const C* data() const
  {
    if (!valid_)
      throw std::logic_error("complex orbital head jets unavailable");
    return result_.data();
  }

  /// Allocation identity for alias admission only; this is not published numeric data.
  const void* outputStorageAddress() const noexcept { return result_.data(); }

  /// Borrow true paired affine jets before envelope/phase multiplication, even at E=0.
  const C* affineProjectionData() const
  {
    if (!valid_)
      throw std::logic_error("complex orbital affine projection unavailable");
    return projected_.data();
  }

  /// Count explicitly owned numeric buffers and copied reciprocal momentum storage.
  static std::size_t storageRequirement(const SpatialJetLayout& output, std::size_t momentum_count = 0)
  {
    return spatial_detail::checkedAdd(multiply(multiply(2, spatialJetSpanElements(output)), sizeof(C)),
        multiply(momentum_count, sizeof(std::array<double, 3>)), "complex head storage bytes overflow");
  }

  /// Count live vector capacities rather than shared parameter storage.
  std::size_t storageBytes() const
  {
    std::size_t bytes = multiply(projected_.capacity(), sizeof(C));
    bytes = spatial_detail::checkedAdd(bytes, multiply(result_.capacity(), sizeof(C)),
                                       "complex head storage bytes overflow");
    return spatial_detail::checkedAdd(bytes,
        multiply(descriptor_.reciprocal_momenta.capacity(), sizeof(std::array<double, 3>)),
        "complex head storage bytes overflow");
  }

  /// Fingerprint prepared allocation identities for warmed no-growth checks.
  std::size_t storageFingerprint() const noexcept
  {
    std::size_t hash = 1469598103934665603ULL;
    determinant::detail::mixStorage(hash, descriptor_.reciprocal_momenta);
    determinant::detail::mixStorage(hash, projected_);
    determinant::detail::mixStorage(hash, result_);
    return hash;
  }

private:
  /// Check products before using them as tensor or parameter extents.
  static std::size_t multiply(std::size_t a, std::size_t b)
  {
    return spatial_detail::checkedProduct(a, b, "complex orbital extent overflow");
  }

  /// Count physical rows in one spin sector, including a possible empty block.
  std::size_t rows(std::size_t spin) const noexcept
  {
    return spin ? descriptor_.electrons - descriptor_.spin_up : descriptor_.spin_up;
  }

  /// Select full-matrix columns or the spin block's column extent.
  std::size_t columns(std::size_t spin) const noexcept
  {
    return descriptor_.structure == ComplexOrbitalStructure::FULL ? descriptor_.electrons : rows(spin);
  }

  /// Count channel/orbital parameter and momentum entries in one nonempty spin.
  std::size_t spinProjectionCount(std::size_t spin) const
  {
    return rows(spin) ? multiply(descriptor_.determinants, columns(spin)) : 0;
  }

  /// Validate descriptor/layout compatibility without changing legacy real layouts.
  SpatialJetLayout makeOutputLayout() const
  {
    validateSpatialJetLayout(features_);
    validateSpatialJetLayout(envelopes_);
    if (!descriptor_.electrons || descriptor_.spin_up > descriptor_.electrons ||
        !descriptor_.determinants || !descriptor_.feature_width ||
        (descriptor_.structure != ComplexOrbitalStructure::FULL &&
         descriptor_.structure != ComplexOrbitalStructure::SPIN_BLOCKED) ||
        (descriptor_.phase != ComplexOrbitalPhase::NONE &&
         descriptor_.phase != ComplexOrbitalPhase::RECIPROCAL_ONLY) ||
        descriptor_.electrons != features_.electron_count ||
        !haveMatchingSpatialPlanes(features_, envelopes_) ||
        features_.element_count != multiply(descriptor_.electrons, descriptor_.feature_width))
      throw std::invalid_argument("complex orbital descriptor and real jet layouts disagree");
    const std::size_t matrix_elements = descriptor_.structure == ComplexOrbitalStructure::FULL
        ? multiply(descriptor_.electrons, descriptor_.electrons)
        : spatial_detail::checkedAdd(multiply(rows(0), rows(0)), multiply(rows(1), rows(1)),
                                    "complex orbital block extent overflow");
    if (envelopes_.element_count != multiply(descriptor_.determinants, matrix_elements))
      throw std::invalid_argument("complex orbital envelope block layout disagrees");
    const std::size_t momentum_count = descriptor_.phase == ComplexOrbitalPhase::NONE ? 0 :
        spatial_detail::checkedAdd(spinProjectionCount(0), spinProjectionCount(1),
                                  "complex orbital momentum extent overflow");
    if (descriptor_.reciprocal_momenta.size() != momentum_count)
      throw std::invalid_argument("complex orbital reciprocal momentum table has wrong extent");
    for (const auto& momentum : descriptor_.reciprocal_momenta)
      for (double component : momentum)
        if (!determinant::isFiniteReal(component))
          throw std::invalid_argument("complex orbital reciprocal momentum is nonfinite");
    SpatialJetLayout result = envelopes_;
    result.plane_stride = result.element_count;
    result.configuration_stride = multiply(result.plane_count, result.element_count);
    validateSpatialJetLayout(result);
    return result;
  }

  /// Validate supplied real input storage before entering unchecked indexing loops.
  static void validateInput(const double* input, std::size_t supplied, std::size_t required)
  {
    if (supplied < required || (required && !input))
      throw std::invalid_argument("complex orbital input span is too small");
    for (std::size_t i = 0; i < required; ++i)
      if (!determinant::isFiniteReal(input[i]))
        throw std::invalid_argument("complex orbital real input is nonfinite");
  }

  /// Validate both real heads, allowing independent optional biases and empty sectors.
  void validateParameters(const ComplexOrbitalSpinParameters& parameters, std::size_t spin) const
  {
    const std::size_t count = spinProjectionCount(spin);
    if (!count)
      return;
    validateInput(parameters.real_weights, parameters.weight_count, multiply(descriptor_.feature_width, count));
    validateInput(parameters.imaginary_weights, parameters.weight_count, multiply(descriptor_.feature_width, count));
    if (parameters.real_bias)
      validateInput(parameters.real_bias, parameters.bias_count, count);
    if (parameters.imaginary_bias)
      validateInput(parameters.imaginary_bias, parameters.bias_count, count);
  }

  /// Map a physical row and orbital to full or channel-concatenated spin-block storage.
  std::size_t matrixElement(std::size_t d, std::size_t row, std::size_t orbital) const noexcept
  {
    const std::size_t stride = output_.element_count / descriptor_.determinants;
    if (descriptor_.structure == ComplexOrbitalStructure::FULL)
      return d * stride + row * descriptor_.electrons + orbital;
    if (row < descriptor_.spin_up)
      return d * stride + row * descriptor_.spin_up + orbital;
    return d * stride + descriptor_.spin_up * descriptor_.spin_up +
        (row - descriptor_.spin_up) * rows(1) + orbital;
  }

  /// Form the paired affine projection on every supplied real spatial plane.
  void project(const double* input, const ComplexOrbitalFeatureJetView* split,
      const ComplexOrbitalSpinParameters& p,
      std::size_t configuration, std::size_t d, std::size_t row,
      std::size_t orbital, std::size_t element)
  {
    const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
    const std::size_t count = spinProjectionCount(spin), index = d * columns(spin) + orbital;
    for (std::size_t plane = 0; plane < output_.plane_count; ++plane)
    {
      W value{};
      if (plane == 0)
        value = W{p.real_bias ? p.real_bias[index] : 0, p.imaginary_bias ? p.imaginary_bias[index] : 0};
      for (std::size_t feature = 0; feature < descriptor_.feature_width; ++feature)
      {
        const std::size_t parameter = feature * count + index;
        value += static_cast<long double>(featureValue(input, split, configuration,
            plane, row * descriptor_.feature_width + feature)) *
            W{p.real_weights[parameter], p.imaginary_weights[parameter]};
      }
      projected_[output_.uncheckedPlaneOffset(configuration, plane, element)] = narrow(value);
    }
  }

  /// Read one prepared spatial plane from packed or compact split real storage.
  double featureValue(const double* packed, const ComplexOrbitalFeatureJetView* split,
      std::size_t configuration, std::size_t plane, std::size_t element) const
  {
    if (!split)
      return packed[features_.uncheckedPlaneOffset(configuration, plane, element)];
    if (plane == 0)
      return split->values[configuration * features_.element_count + element];
    if (plane <= features_.gradient_lanes)
      return split->gradients[(configuration * features_.gradient_lanes + plane - 1) *
          features_.element_count + element];
    return split->laplacians[(configuration * features_.laplacian_lanes +
        plane - 1 - features_.gradient_lanes) * features_.element_count + element];
  }

  /// Return one stored reciprocal G; no common twist is accepted or added here.
  std::array<double, 3> momentum(std::size_t d, std::size_t row, std::size_t orbital) const
  {
    if (descriptor_.phase == ComplexOrbitalPhase::NONE)
      return {};
    const std::size_t spin = row < descriptor_.spin_up ? 0 : 1;
    return descriptor_.reciprocal_momenta[(spin ? spinProjectionCount(0) : 0) + d * columns(spin) + orbital];
  }

  /// Compose real-envelope and reciprocal-phase product jets, including both cross terms.
  void compose(const double* envelope, const double* positions, std::size_t configuration,
      std::size_t d, std::size_t row, std::size_t orbital, std::size_t element,
      std::size_t active_electron)
  {
    const auto g = momentum(d, row, orbital);
    long double argument = 0;
    for (std::size_t dim = 0; dim < 3; ++dim)
      argument += static_cast<long double>(g[dim]) *
          positions[(configuration * descriptor_.electrons + row) * 3 + dim];
    const W phase{std::cos(argument), std::sin(argument)};
    const W b = projected_[output_.uncheckedValueOffset(configuration, element)];
    const long double e = envelope[envelopes_.uncheckedValueOffset(configuration, element)];
    const W value = e * b;
    result_[output_.uncheckedValueOffset(configuration, element)] = narrow(phase * value);
    for (std::size_t lane = 0; lane < output_.gradient_lanes; ++lane)
    {
      const std::size_t target = output_.mode == SpatialJetMode::ACTIVE ? active_electron :
          output_.physicalTargetElectron(lane / 3);
      const W first = e * W(projected_[output_.uncheckedGradientOffset(configuration, lane, element)]) +
          static_cast<long double>(envelope[envelopes_.uncheckedGradientOffset(configuration, lane, element)]) * b;
      const W phase_first = target == row ? W{0, g[lane % 3]} : W{};
      result_[output_.uncheckedGradientOffset(configuration, lane, element)] =
          narrow(phase * (first + phase_first * value));
    }
    for (std::size_t target = 0; target < output_.laplacian_lanes; ++target)
    {
      W lap = e * W(projected_[output_.uncheckedLaplacianOffset(configuration, target, element)]) +
          static_cast<long double>(envelope[envelopes_.uncheckedLaplacianOffset(configuration, target, element)]) * b;
      const bool row_target = output_.physicalTargetElectron(target) == row;
      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        const std::size_t lane = 3 * target + dim;
        const W db = projected_[output_.uncheckedGradientOffset(configuration, lane, element)];
        const long double de = envelope[envelopes_.uncheckedGradientOffset(configuration, lane, element)];
        lap += 2.0L * de * db;
        if (row_target)
          lap += W{0, 2 * g[dim]} * (e * db + de * b) -
              static_cast<long double>(g[dim]) * g[dim] * value;
      }
      result_[output_.uncheckedLaplacianOffset(configuration, target, element)] = narrow(phase * lap);
    }
  }

  /// Reject nonfinite or unrepresentable complex outputs before publishing the batch.
  static C narrow(W value)
  {
    return {determinant::detail::narrowFinite(value.real(), "complex orbital real output overflow"),
            determinant::detail::narrowFinite(value.imag(), "complex orbital imaginary output overflow")};
  }

  const ComplexOrbitalHeadDescriptor descriptor_;
  const SpatialJetLayout features_, envelopes_, output_;
  std::vector<C> projected_, result_;
  bool valid_ = false;
};
} // namespace qmcplusplus::psiformer
#endif
