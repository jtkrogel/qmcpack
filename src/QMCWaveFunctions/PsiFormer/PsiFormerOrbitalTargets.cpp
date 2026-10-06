//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerOrbitalTargets.cpp
 * @brief Construction and SPO evaluation of immutable PsiFormer orbital targets.
 */

#include "PsiFormerOrbitalTargets.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/Fermion/MultiDiracDeterminant.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace qmcplusplus::psiformer
{
namespace
{

/// Fold one unsigned integer into the stable FNV-1a target fingerprint.
void hashUnsigned(std::uint64_t& hash, std::uint64_t value) noexcept
{
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
  {
    hash ^= (value >> (8 * byte)) & 0xffU;
    hash *= 1099511628211ULL;
  }
}

/// Test IEEE-754 finiteness without relying on fast-math-sensitive predicates.
bool isFiniteDouble(double value) noexcept
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/// Fold an exact byte string into the stable target fingerprint.
void hashString(std::uint64_t& hash, const std::string& value) noexcept
{
  hashUnsigned(hash, value.size());
  for (unsigned char character : value)
  {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
}

/// Reject duplicate occupations and return the largest referenced orbital.
std::size_t validateOccupations(const std::vector<std::size_t>& occupations,
                                std::size_t expected,
                                const char* spin)
{
  if (occupations.size() != expected)
    throw std::invalid_argument(std::string("PsiFormer ") + spin +
                                " target occupation count does not match its electron count");
  std::vector<std::size_t> ordered = occupations;
  std::sort(ordered.begin(), ordered.end());
  if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
    throw std::invalid_argument(std::string("PsiFormer ") + spin +
                                " target contains a duplicate occupied orbital");
  return ordered.empty() ? 0 : ordered.back();
}

/// Convert a QMCPACK scalar to a finite real target value or reject it.
template<class Scalar>
double checkedRealValue(const Scalar& value, const char* description)
{
  const double real_value = std::real(value);
  const double imaginary_value = std::imag(value);
  if (!isFiniteDouble(real_value) || !isFiniteDouble(imaginary_value) || imaginary_value != 0.0)
    throw std::invalid_argument(std::string("PsiFormer ") + description +
                                " must be finite and real");
  return real_value;
}

} // namespace

OrbitalTargetDescriptor::OrbitalTargetDescriptor(OrbitalTargetMode mode,
                                                 std::string source_identity,
                                                 std::size_t spin_up_electrons,
                                                 std::size_t spin_down_electrons,
                                                 std::vector<OrbitalTargetTerm> terms)
    : mode_(mode),
      source_identity_(std::move(source_identity)),
      spin_up_electrons_(spin_up_electrons),
      spin_down_electrons_(spin_down_electrons),
      terms_(std::move(terms)),
      fingerprint_(1469598103934665603ULL)
{
  if (source_identity_.empty())
    throw std::invalid_argument("PsiFormer orbital target requires a non-empty source identity");
  if (electronCount() == 0 || terms_.empty())
    throw std::invalid_argument("PsiFormer orbital target requires electrons and determinant terms");

  // These explicit convention tags make restarts reject future changes to the
  // full-matrix layout, raw coefficient policy, root/sign embedding, or the
  // separately normalized alpha/beta orbital-MSE definition.
  hashUnsigned(fingerprint_, formatVersion());
  hashUnsigned(fingerprint_, 1); // full spin-block N_e x N_e matrices
  hashUnsigned(fingerprint_, 1); // |c|^(1/N_e), sign in first global column
  hashUnsigned(fingerprint_, 1); // raw coefficients, no selected-space renormalization
  hashUnsigned(fingerprint_, 1); // separate alpha/beta row-sector means
  hashUnsigned(fingerprint_, static_cast<std::uint64_t>(mode_));
  hashString(fingerprint_, source_identity_);
  hashUnsigned(fingerprint_, spin_up_electrons_);
  hashUnsigned(fingerprint_, spin_down_electrons_);
  hashUnsigned(fingerprint_, terms_.size());
  for (const OrbitalTargetTerm& term : terms_)
  {
    if (!isFiniteDouble(term.coefficient))
      throw std::invalid_argument("PsiFormer orbital target coefficient is not finite");
    validateOccupations(term.occupations_up, spin_up_electrons_, "alpha");
    validateOccupations(term.occupations_down, spin_down_electrons_, "beta");
    hashUnsigned(fingerprint_, term.source_ordinal);
    std::uint64_t coefficient_bits;
    static_assert(sizeof(coefficient_bits) == sizeof(term.coefficient));
    std::memcpy(&coefficient_bits, &term.coefficient, sizeof(coefficient_bits));
    hashUnsigned(fingerprint_, coefficient_bits);
    hashUnsigned(fingerprint_, term.occupations_up.size());
    for (std::size_t orbital : term.occupations_up)
      hashUnsigned(fingerprint_, orbital);
    hashUnsigned(fingerprint_, term.occupations_down.size());
    for (std::size_t orbital : term.occupations_down)
      hashUnsigned(fingerprint_, orbital);
  }
}

OrbitalTargetDescriptor OrbitalTargetDescriptor::makeSingleReference(
    std::string source_identity,
    std::size_t spin_up_electrons,
    std::size_t spin_down_electrons,
    std::size_t determinant_count)
{
  if (determinant_count == 0)
    throw std::invalid_argument("PsiFormer single-reference target requires determinant channels");
  OrbitalTargetTerm reference;
  reference.occupations_up.resize(spin_up_electrons);
  reference.occupations_down.resize(spin_down_electrons);
  std::iota(reference.occupations_up.begin(), reference.occupations_up.end(), 0);
  std::iota(reference.occupations_down.begin(), reference.occupations_down.end(), 0);
  std::vector<OrbitalTargetTerm> terms(determinant_count, reference);
  return {OrbitalTargetMode::SINGLE_REFERENCE, std::move(source_identity),
          spin_up_electrons, spin_down_electrons, std::move(terms)};
}

OrbitalTargetDescriptor OrbitalTargetDescriptor::makeTruncatedMultideterminant(
    std::string source_identity,
    std::size_t spin_up_electrons,
    std::size_t spin_down_electrons,
    std::size_t determinant_count,
    const std::vector<double>& coefficients,
    const std::vector<std::vector<std::size_t>>& determinant_to_unique,
    const std::vector<std::vector<std::vector<std::size_t>>>& unique_occupations)
{
  if (determinant_count == 0 || coefficients.size() < determinant_count)
    throw std::invalid_argument("PsiFormer DETS target count exceeds the source expansion");
  const std::size_t spin_sectors = (spin_up_electrons == 0 ? 0 : 1) +
      (spin_down_electrons == 0 ? 0 : 1);
  if (determinant_to_unique.size() != unique_occupations.size() ||
      determinant_to_unique.size() != spin_sectors)
    throw std::invalid_argument("PsiFormer DETS target requires one or two consistent spin maps");
  for (std::size_t spin = 0; spin < determinant_to_unique.size(); ++spin)
    if (determinant_to_unique[spin].size() != coefficients.size())
      throw std::invalid_argument("PsiFormer DETS coefficient and occupation maps have different sizes");

  std::vector<std::size_t> order(coefficients.size());
  std::iota(order.begin(), order.end(), 0);
  for (double coefficient : coefficients)
    if (!isFiniteDouble(coefficient))
      throw std::invalid_argument("PsiFormer DETS coefficient is not finite");
  std::stable_sort(order.begin(), order.end(), [&coefficients](std::size_t left, std::size_t right) {
    return std::abs(coefficients[left]) > std::abs(coefficients[right]);
  });

  std::vector<OrbitalTargetTerm> terms;
  terms.reserve(determinant_count);
  for (std::size_t selected = 0; selected < determinant_count; ++selected)
  {
    const std::size_t ordinal = order[selected];
    OrbitalTargetTerm term;
    term.source_ordinal = ordinal;
    term.coefficient = coefficients[ordinal];
    std::size_t source_spin = 0;
    if (spin_up_electrons != 0)
    {
      const std::size_t up_unique = determinant_to_unique[source_spin][ordinal];
      if (up_unique >= unique_occupations[source_spin].size())
        throw std::invalid_argument("PsiFormer alpha DETS occupation map is out of range");
      term.occupations_up = unique_occupations[source_spin][up_unique];
      ++source_spin;
    }
    if (spin_down_electrons != 0)
    {
      const std::size_t down_unique = determinant_to_unique[source_spin][ordinal];
      if (down_unique >= unique_occupations[source_spin].size())
        throw std::invalid_argument("PsiFormer beta DETS occupation map is out of range");
      term.occupations_down = unique_occupations[source_spin][down_unique];
    }
    terms.push_back(std::move(term));
  }
  return {OrbitalTargetMode::TRUNCATED_MULTIDETERMINANT, std::move(source_identity),
          spin_up_electrons, spin_down_electrons, std::move(terms)};
}

OrbitalTargetEvaluator::OrbitalTargetEvaluator(
    OrbitalTargetDescriptor descriptor,
    std::vector<std::unique_ptr<SPOSet>>&& source_orbitals)
    : descriptor_(std::move(descriptor)), source_orbitals_(std::move(source_orbitals))
{
#if defined(QMC_COMPLEX)
  throw std::invalid_argument("PsiFormer orbital pretraining does not support complex SPO builds");
#endif
  const std::size_t spin_sectors = (descriptor_.spinUpElectrons() == 0 ? 0 : 1) +
      (descriptor_.spinDownElectrons() == 0 ? 0 : 1);
  if (source_orbitals_.size() != spin_sectors)
    throw std::invalid_argument("PsiFormer orbital target received the wrong number of SPO sets");

  orbital_scratch_.resize(spin_sectors);
  for (std::size_t spin = 0; spin < spin_sectors; ++spin)
  {
    if (!source_orbitals_[spin])
      throw std::invalid_argument("PsiFormer orbital target received a null SPO set");
    orbital_scratch_[spin].resize(source_orbitals_[spin]->getOrbitalSetSize());
  }
  for (const OrbitalTargetTerm& term : descriptor_.terms())
  {
    const std::size_t down_source = descriptor_.spinUpElectrons() == 0 ? 0 : 1;
    if (!term.occupations_up.empty() &&
        validateOccupations(term.occupations_up, descriptor_.spinUpElectrons(), "alpha") >=
            static_cast<std::size_t>(source_orbitals_[0]->getOrbitalSetSize()))
      throw std::invalid_argument("PsiFormer alpha target occupation exceeds its SPO set");
    if (!term.occupations_down.empty() &&
        validateOccupations(term.occupations_down, descriptor_.spinDownElectrons(), "beta") >=
            static_cast<std::size_t>(source_orbitals_[down_source]->getOrbitalSetSize()))
      throw std::invalid_argument("PsiFormer beta target occupation exceeds its SPO set");
  }

  coefficient_roots_.reserve(descriptor_.terms().size());
  const double inverse_electrons = 1.0 / static_cast<double>(descriptor_.electronCount());
  for (const OrbitalTargetTerm& term : descriptor_.terms())
    coefficient_roots_.push_back(std::pow(std::abs(term.coefficient), inverse_electrons));
  target_matrices_.resize(descriptor_.terms().size() * descriptor_.electronCount() *
                          descriptor_.electronCount());
}

OrbitalTargetEvaluator OrbitalTargetEvaluator::fromSingleReference(
    const SlaterDet& source,
    std::string source_identity,
    std::size_t determinant_count)
{
#if defined(QMC_COMPLEX)
  throw std::invalid_argument("PsiFormer orbital pretraining does not support complex SPO builds");
#else
  if (source.getNumDets() == 0 || source.getNumDets() > 2)
    throw std::invalid_argument("PsiFormer single-reference target requires one or two spin determinants");
  std::vector<std::unique_ptr<SPOSet>> orbitals;
  std::size_t up = 0;
  std::size_t down = 0;
  std::size_t expected_first = 0;
  for (int spin = 0; spin < source.getNumDets(); ++spin)
  {
    const auto& determinant = source.getDet(spin);
    if (determinant.getPhi().getClassName() == "SpinorSet")
      throw std::invalid_argument("PsiFormer orbital pretraining does not support spinor determinants");
    const int first = determinant.getFirstIndex();
    const int last  = determinant.getLastIndex();
    if (first < 0 || last <= first || static_cast<std::size_t>(first) != expected_first)
      throw std::invalid_argument(
          "PsiFormer single-reference determinant particle ranges are not contiguous spin blocks");
    const std::size_t count = static_cast<std::size_t>(last - first);
    if (spin == 0)
      up = count;
    else
      down = count;
    expected_first += count;
    orbitals.push_back(determinant.getPhi().makeClone());
  }
  return {OrbitalTargetDescriptor::makeSingleReference(std::move(source_identity), up, down,
                                                       determinant_count),
          std::move(orbitals)};
#endif
}

OrbitalTargetEvaluator OrbitalTargetEvaluator::fromMultideterminant(
    const MultiSlaterDetTableMethod& source,
    std::string source_identity,
    std::size_t determinant_count)
{
  if (source.hasCSFExpansion())
    throw std::invalid_argument("PsiFormer orbital pretraining does not support CSF target selection");
#if defined(QMC_COMPLEX)
  throw std::invalid_argument("PsiFormer orbital pretraining does not support complex DETS coefficients");
#else
  if (source.getDetSize() == 0 || source.getDetSize() > 2)
    throw std::invalid_argument("PsiFormer DETS target requires one or two spin determinant sets");

  std::vector<double> coefficients;
  coefficients.reserve(source.get_C().size());
  for (const auto& coefficient : source.get_C())
    coefficients.push_back(checkedRealValue(coefficient, "DETS coefficient"));

  std::vector<std::vector<std::vector<std::size_t>>> configurations(source.getDetSize());
  std::vector<std::unique_ptr<SPOSet>> orbitals;
  std::size_t up = 0;
  std::size_t down = 0;
  std::size_t expected_first = 0;
  for (int spin = 0; spin < source.getDetSize(); ++spin)
  {
    const MultiDiracDeterminant& determinant = source.getDet(spin);
    if (determinant.isSpinor())
      throw std::invalid_argument("PsiFormer orbital pretraining does not support spinor determinants");
    if (determinant.getFirstIndex() < 0 || determinant.getNumPtcls() <= 0 ||
        static_cast<std::size_t>(determinant.getFirstIndex()) != expected_first)
      throw std::invalid_argument(
          "PsiFormer DETS determinant particle ranges are not contiguous spin blocks");
    if (spin == 0)
      up = determinant.getNumPtcls();
    else
      down = determinant.getNumPtcls();
    expected_first += static_cast<std::size_t>(determinant.getNumPtcls());
    configurations[spin].reserve(determinant.getNumDets());
    for (int index = 0; index < determinant.getNumDets(); ++index)
      configurations[spin].push_back(determinant.getConfiguration(index).occup);
    orbitals.push_back(determinant.clonePhi());
  }
  OrbitalTargetDescriptor descriptor = OrbitalTargetDescriptor::makeTruncatedMultideterminant(
      std::move(source_identity), up, down, determinant_count, coefficients,
      source.get_C2node(), configurations);
  return {std::move(descriptor), std::move(orbitals)};
#endif
}

const std::vector<double>& OrbitalTargetEvaluator::evaluate(const ParticleSet& electrons)
{
  if (static_cast<std::size_t>(electrons.getTotalNum()) != descriptor_.electronCount())
    throw std::invalid_argument("PsiFormer orbital target electron count does not match the sample");
  std::fill(target_matrices_.begin(), target_matrices_.end(), 0.0);
  const std::size_t electron_count = descriptor_.electronCount();
  const std::size_t matrix_elements = electron_count * electron_count;
  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    const bool up = electron < descriptor_.spinUpElectrons();
    const std::size_t spin = up || descriptor_.spinUpElectrons() == 0 ? 0 : 1;
    source_orbitals_[spin]->evaluateValue(electrons, static_cast<int>(electron), orbital_scratch_[spin]);
    const std::size_t column_offset = up ? 0 : descriptor_.spinUpElectrons();
    for (std::size_t determinant = 0; determinant < descriptor_.terms().size(); ++determinant)
    {
      const OrbitalTargetTerm& term = descriptor_.terms()[determinant];
      const auto& occupations = up ? term.occupations_up : term.occupations_down;
      const double root = coefficient_roots_[determinant];
      for (std::size_t local_column = 0; local_column < occupations.size(); ++local_column)
      {
        const std::size_t global_column = column_offset + local_column;
        const double sign = global_column == 0 && term.coefficient < 0.0 ? -1.0 : 1.0;
        const double value = checkedRealValue(orbital_scratch_[spin][occupations[local_column]],
                                              "SPO target value");
        target_matrices_[determinant * matrix_elements + electron * electron_count + global_column] =
            sign * root * value;
      }
    }
  }
  return target_matrices_;
}

std::size_t OrbitalTargetEvaluator::retainedScalarBytes() const noexcept
{
  std::size_t bytes = (coefficient_roots_.capacity() + target_matrices_.capacity()) * sizeof(double);
  for (const auto& scratch : orbital_scratch_)
    bytes += scratch.capacity() * sizeof(SPOSet::ValueType);
  return bytes;
}

std::size_t OrbitalTargetEvaluator::storageFingerprint() const noexcept
{
  std::size_t hash = 1469598103934665603ULL;
  auto mix = [&hash](const auto& buffer) {
    hash ^= reinterpret_cast<std::uintptr_t>(buffer.data());
    hash *= 1099511628211ULL;
    hash ^= buffer.capacity();
    hash *= 1099511628211ULL;
  };
  mix(coefficient_roots_);
  mix(target_matrices_);
  for (const auto& scratch : orbital_scratch_)
    mix(scratch);
  return hash;
}

} // namespace qmcplusplus::psiformer
