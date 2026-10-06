//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerGeometry.cpp
 * @brief Implementation of fixed-size molecular PsiFormer geometry tables.
 */

#include "PsiFormerGeometry.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pf
{
namespace
{

/// Reject pair-table sizes whose multiplication would overflow size_t.
std::size_t checkedProduct(std::size_t left, std::size_t right)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error("PsiFormer geometry pair count overflows size_t");
  return left * right;
}

/// Reject byte-accounting sums whose addition would overflow size_t.
std::size_t checkedSum(std::size_t left, std::size_t right)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error("PsiFormer geometry storage bytes overflow size_t");
  return left + right;
}

/// Convert a vector capacity to bytes with checked arithmetic.
template<class T>
std::size_t checkedCapacityBytes(const std::vector<T>& values)
{
  return checkedProduct(values.capacity(), sizeof(T));
}

/// Return n*(n-1)/2 without overflowing the intermediate product.
std::size_t checkedUniquePairCount(std::size_t particle_count)
{
  if (particle_count < 2)
    return 0;

  std::size_t left  = particle_count;
  std::size_t right = particle_count - 1;
  if ((left & 1U) == 0)
    left /= 2;
  else
    right /= 2;
  return checkedProduct(left, right);
}

/// Test IEEE-754 finiteness without fast-math-sensitive classification builtins.
bool isFiniteGeometryValue(GeometryReal value)
{
  static_assert(sizeof(GeometryReal) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/// Verify that a position can safely enter distance and radial arithmetic.
void validatePosition(const GeometryPosition& position)
{
  for (GeometryReal component : position)
    if (!isFiniteGeometryValue(component))
      throw std::invalid_argument("PsiFormer geometry position contains a non-finite component");
}

/// Validate every derived pair quantity without changing an accepted table.
void preflightPair(const GeometryPosition& displacement)
{
  validatePosition(displacement);
  const GeometryReal distance =
      std::hypot(displacement[0], displacement[1], displacement[2]);
  if (!isFiniteGeometryValue(distance))
    throw std::invalid_argument(
        "PsiFormer geometry pair distance is non-finite");
  (void)evaluateSoftenedRadialFactors(distance);
}

/// Evaluate a polynomial with coefficients ordered from highest to lowest degree.
template<std::size_t N>
GeometryReal evaluatePolynomial(const std::array<GeometryReal, N>& coefficients, GeometryReal argument)
{
  GeometryReal value = coefficients[0];
  for (std::size_t coefficient = 1; coefficient < N; ++coefficient)
    value = value * argument + coefficients[coefficient];
  return value;
}

} // namespace

bool supportsGeometryBoundary(GeometryBoundaryKind kind) noexcept
{
  return kind == GeometryBoundaryKind::OPEN;
}

GeometryPositionView::GeometryPositionView(std::array<const GeometryReal*, 3> components,
                                           std::array<std::size_t, 3> strides,
                                           std::size_t particle_count)
    : components_(components), strides_(strides), particle_count_(particle_count)
{
  if (particle_count_ == 0)
    return;

  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    if (components_[dimension] == nullptr || strides_[dimension] == 0)
      throw std::invalid_argument("PsiFormer geometry view has null storage or a zero stride");
}

GeometryPositionView GeometryPositionView::interleaved(const GeometryReal* values,
                                                       std::size_t particle_count,
                                                       std::size_t particle_stride)
{
  if (particle_count != 0 && (values == nullptr || particle_stride < 3))
    throw std::invalid_argument("PsiFormer interleaved geometry view requires xyz storage and stride >= 3");

  if (particle_count == 0)
    return {};

  return GeometryPositionView({values, values + 1, values + 2},
                              {particle_stride, particle_stride, particle_stride}, particle_count);
}

GeometryPositionView GeometryPositionView::components(const GeometryReal* x,
                                                      const GeometryReal* y,
                                                      const GeometryReal* z,
                                                      std::size_t particle_count,
                                                      std::size_t component_stride)
{
  return GeometryPositionView({x, y, z}, {component_stride, component_stride, component_stride}, particle_count);
}

GeometryReal GeometryPositionView::operator()(std::size_t particle, std::size_t dimension) const
{
  if (particle >= particle_count_ || dimension >= 3)
    throw std::out_of_range("PsiFormer geometry coordinate index is out of range");
  return components_[dimension][particle * strides_[dimension]];
}

GeometryPosition GeometryPositionView::position(std::size_t particle) const
{
  return {(*this)(particle, 0), (*this)(particle, 1), (*this)(particle, 2)};
}

SoftenedRadialFactors evaluateSoftenedRadialFactors(GeometryReal radius)
{
  if (radius < 0 || !isFiniteGeometryValue(radius))
    throw std::invalid_argument("PsiFormer softened radial factor requires a finite nonnegative radius");

  SoftenedRadialFactors factors;
  factors.log1p_radius = std::log1p(radius);
  const GeometryReal inverse_one_plus_radius = 1 / (1 + radius);
  factors.log1p_first                         = inverse_one_plus_radius;
  factors.log1p_second                        = -inverse_one_plus_radius * inverse_one_plus_radius;

  // Direct quotient derivative formulas lose most significant digits near the
  // origin.  Taylor polynomials through r^10 (and their analytic derivatives)
  // retain the finite limits needed by the embedding feature.
  constexpr GeometryReal small_radius = 1.0e-2;
  if (radius < small_radius)
  {
    constexpr std::array<GeometryReal, 11> value_coefficients{
        1.0 / 11.0, -1.0 / 10.0, 1.0 / 9.0, -1.0 / 8.0, 1.0 / 7.0, -1.0 / 6.0,
        1.0 / 5.0,  -1.0 / 4.0,  1.0 / 3.0, -1.0 / 2.0, 1.0};
    constexpr std::array<GeometryReal, 10> first_coefficients{
        10.0 / 11.0, -9.0 / 10.0, 8.0 / 9.0, -7.0 / 8.0, 6.0 / 7.0,
        -5.0 / 6.0,  4.0 / 5.0,   -3.0 / 4.0, 2.0 / 3.0, -1.0 / 2.0};
    constexpr std::array<GeometryReal, 9> second_coefficients{
        90.0 / 11.0, -36.0 / 5.0, 56.0 / 9.0, -21.0 / 4.0, 30.0 / 7.0,
        -10.0 / 3.0, 12.0 / 5.0, -3.0 / 2.0, 2.0 / 3.0};

    factors.log1p_over_radius        = evaluatePolynomial(value_coefficients, radius);
    factors.log1p_over_radius_first  = evaluatePolynomial(first_coefficients, radius);
    factors.log1p_over_radius_second = evaluatePolynomial(second_coefficients, radius);
    return factors;
  }

  factors.log1p_over_radius = factors.log1p_radius / radius;
  const GeometryReal quotient_numerator = radius * inverse_one_plus_radius - factors.log1p_radius;
  factors.log1p_over_radius_first       = quotient_numerator / (radius * radius);
  factors.log1p_over_radius_second =
      -1 / (radius * (1 + radius) * (1 + radius)) - 2 * quotient_numerator / (radius * radius * radius);
  return factors;
}

GeometryPairTable::GeometryPairTable(std::size_t pair_count)
    : displacements_(pair_count),
      distances_(pair_count),
      inverse_distances_(pair_count),
      softened_radial_factors_(pair_count)
{}

std::size_t GeometryPairTable::storageFingerprint() const noexcept
{
  std::size_t hash = 1469598103934665603ULL;
  auto mix = [&hash](const auto& buffer) {
    hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
    hash *= 1099511628211ULL;
    hash ^= buffer.capacity();
    hash *= 1099511628211ULL;
  };
  mix(displacements_);
  mix(distances_);
  mix(inverse_distances_);
  mix(softened_radial_factors_);
  return hash;
}

std::size_t GeometryPairTable::storageBytes() const
{
  std::size_t bytes = 0;
  for (const std::size_t contribution : {
           checkedCapacityBytes(displacements_), checkedCapacityBytes(distances_),
           checkedCapacityBytes(inverse_distances_),
           checkedCapacityBytes(softened_radial_factors_)})
    bytes = checkedSum(bytes, contribution);
  return bytes;
}

void GeometryPairTable::updatePair(std::size_t pair_index, const GeometryPosition& displacement)
{
  displacements_[pair_index] = displacement;
  const GeometryReal distance = std::hypot(displacement[0], displacement[1], displacement[2]);
  distances_[pair_index]      = distance;
  inverse_distances_[pair_index] =
      distance == 0 ? std::numeric_limits<GeometryReal>::infinity() : 1 / distance;
  softened_radial_factors_[pair_index] = evaluateSoftenedRadialFactors(distance);
}

PsiFormerGeometryCache::PsiFormerGeometryCache(std::size_t electron_count,
                                               GeometryPositionView nuclei,
                                               GeometryBoundary boundary)
    : boundary_(boundary),
      nuclei_(nuclei.size()),
      electrons_(electron_count),
      electron_nucleus_pairs_(checkedProduct(electron_count, nuclei.size())),
      electron_electron_pairs_(checkedUniquePairCount(electron_count)),
      electron_pairs_(checkedUniquePairCount(electron_count)),
      incidence_offsets_(electron_count + 1),
      incidences_(checkedProduct(2, checkedUniquePairCount(electron_count)))
{
  if (!supportsGeometryBoundary(boundary_.kind))
    throw std::invalid_argument("PsiFormer geometry supports only open boundary conditions");
  if (electron_count == 0)
    throw std::invalid_argument("PsiFormer geometry requires at least one electron");
  if (nuclei.size() == 0)
    throw std::invalid_argument("Molecular PsiFormer geometry requires at least one fixed nucleus");

  for (std::size_t nucleus = 0; nucleus < nuclei_.size(); ++nucleus)
  {
    nuclei_[nucleus] = nuclei.position(nucleus);
    validatePosition(nuclei_[nucleus]);
  }

  initializeElectronPairs();
}

std::size_t PsiFormerGeometryCache::storageFingerprint() const noexcept
{
  std::size_t hash = 1469598103934665603ULL;
  auto mix = [&hash](const auto& buffer) {
    hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
    hash *= 1099511628211ULL;
    hash ^= buffer.capacity();
    hash *= 1099511628211ULL;
  };
  mix(nuclei_);
  mix(electrons_);
  mix(electron_pairs_);
  mix(incidence_offsets_);
  mix(incidences_);
  hash ^= electron_nucleus_pairs_.storageFingerprint();
  hash *= 1099511628211ULL;
  hash ^= electron_electron_pairs_.storageFingerprint();
  hash *= 1099511628211ULL;
  return hash;
}

std::size_t PsiFormerGeometryCache::storageBytes() const
{
  std::size_t bytes = 0;
  for (const std::size_t contribution : {
           checkedCapacityBytes(nuclei_), checkedCapacityBytes(electrons_),
           checkedCapacityBytes(electron_pairs_),
           checkedCapacityBytes(incidence_offsets_),
           checkedCapacityBytes(incidences_),
           electron_nucleus_pairs_.storageBytes(),
           electron_electron_pairs_.storageBytes()})
    bytes = checkedSum(bytes, contribution);
  return bytes;
}

std::size_t PsiFormerGeometryCache::electronNucleusPairIndex(std::size_t electron, std::size_t nucleus) const
{
  if (electron >= electronCount() || nucleus >= nucleusCount())
    throw std::out_of_range("PsiFormer electron-nucleus pair index is out of range");
  return electron * nucleusCount() + nucleus;
}

void PsiFormerGeometryCache::update(GeometryPositionView electrons)
{
  if (electrons.size() != electronCount())
    throw std::invalid_argument("PsiFormer geometry update has the wrong electron count");

  // Validate the complete input before mutating the accepted geometry cache.
  for (std::size_t electron = 0; electron < electronCount(); ++electron)
    validatePosition(electrons.position(electron));

  // Preflight all subtraction, norm, and radial-factor arithmetic before
  // changing accepted electron or pair-table state.
  for (std::size_t electron = 0; electron < electronCount(); ++electron)
    for (std::size_t nucleus = 0; nucleus < nucleusCount(); ++nucleus)
      preflightPair(displacement(electrons.position(electron), nuclei_[nucleus]));
  for (const ElectronPair& pair : electron_pairs_)
    preflightPair(displacement(electrons.position(pair.first),
                               electrons.position(pair.second)));

  for (std::size_t electron = 0; electron < electronCount(); ++electron)
    electrons_[electron] = electrons.position(electron);

  for (std::size_t electron = 0; electron < electronCount(); ++electron)
    for (std::size_t nucleus = 0; nucleus < nucleusCount(); ++nucleus)
      electron_nucleus_pairs_.updatePair(electronNucleusPairIndex(electron, nucleus),
                                         displacement(electrons_[electron], nuclei_[nucleus]));

  for (std::size_t pair_index = 0; pair_index < electron_pairs_.size(); ++pair_index)
  {
    const ElectronPair& pair = electron_pairs_[pair_index];
    electron_electron_pairs_.updatePair(pair_index, displacement(electrons_[pair.first], electrons_[pair.second]));
  }

  valid_ = true;
  ++generation_;
}

void PsiFormerGeometryCache::updateElectron(std::size_t electron, const GeometryPosition& position)
{
  if (!valid_)
    throw std::logic_error("PsiFormer single-electron geometry update requires an initialized cache");
  if (electron >= electronCount())
    throw std::out_of_range("PsiFormer geometry electron index is out of range");
  validatePosition(position);

  // Validate every affected derived quantity while the accepted cache is
  // untouched.  The publication pass below repeats only proven-safe arithmetic.
  for (std::size_t nucleus = 0; nucleus < nucleusCount(); ++nucleus)
    preflightPair(displacement(position, nuclei_[nucleus]));
  for (std::size_t incidence_index = incidence_offsets_[electron];
       incidence_index < incidence_offsets_[electron + 1]; ++incidence_index)
  {
    const ElectronPair& pair =
        electron_pairs_[incidences_[incidence_index].pair_index];
    const GeometryPosition& first =
        pair.first == electron ? position : electrons_[pair.first];
    const GeometryPosition& second =
        pair.second == electron ? position : electrons_[pair.second];
    preflightPair(displacement(first, second));
  }

  electrons_[electron] = position;
  for (std::size_t nucleus = 0; nucleus < nucleusCount(); ++nucleus)
    electron_nucleus_pairs_.updatePair(electronNucleusPairIndex(electron, nucleus),
                                       displacement(electrons_[electron], nuclei_[nucleus]));

  for (std::size_t incidence_index = incidence_offsets_[electron];
       incidence_index < incidence_offsets_[electron + 1]; ++incidence_index)
  {
    const std::size_t pair_index = incidences_[incidence_index].pair_index;
    const ElectronPair& pair     = electron_pairs_[pair_index];
    electron_electron_pairs_.updatePair(pair_index, displacement(electrons_[pair.first], electrons_[pair.second]));
  }

  ++generation_;
}

GeometryPosition PsiFormerGeometryCache::displacement(const GeometryPosition& target,
                                                      const GeometryPosition& source) const
{
  if (boundary_.kind != GeometryBoundaryKind::OPEN)
    throw std::logic_error("Unsupported PsiFormer boundary policy reached displacement evaluation");
  return {target[0] - source[0], target[1] - source[1], target[2] - source[2]};
}

void PsiFormerGeometryCache::initializeElectronPairs()
{
  std::size_t pair_index = 0;
  for (std::size_t first = 0; first < electronCount(); ++first)
    for (std::size_t second = first + 1; second < electronCount(); ++second)
      electron_pairs_[pair_index++] = {first, second};

  // Each electron is incident on exactly Ne-1 unique pairs.  CSR storage keeps
  // proposal-update traversal contiguous and avoids per-electron vector objects.
  for (std::size_t electron = 0; electron <= electronCount(); ++electron)
    incidence_offsets_[electron] = electron * (electronCount() - 1);
  std::vector<std::size_t> next_incidence(incidence_offsets_.begin(), incidence_offsets_.end() - 1);

  for (pair_index = 0; pair_index < electron_pairs_.size(); ++pair_index)
  {
    const ElectronPair pair = electron_pairs_[pair_index];
    incidences_[next_incidence[pair.first]++]  = {pair_index, pair.second, std::int8_t{1}};
    incidences_[next_incidence[pair.second]++] = {pair_index, pair.first, std::int8_t{-1}};
  }
}

} // namespace pf
