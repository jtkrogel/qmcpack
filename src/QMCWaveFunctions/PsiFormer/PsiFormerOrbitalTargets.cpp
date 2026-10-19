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
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/ElectronGas/FreeOrbital.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"

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
bool isFiniteDouble(double value) noexcept;
}
#ifdef QMC_COMPLEX
/// Freeze the actual occupied orbitals and fixed periodic-part gauge, not parsed names.
ComplexOrbitalTargetEvaluator ComplexOrbitalTargetEvaluator::fromSingleReference(const SlaterDet& source,
    const ParticleSet& electrons, std::array<double, 3> reduced, std::size_t channels, std::size_t maximum_numeric_bytes)
{
  const auto n = static_cast<std::size_t>(electrons.getTotalNum());
  if (!channels || !n || n > std::numeric_limits<std::size_t>::max() / n ||
      channels > std::numeric_limits<std::size_t>::max() / sizeof(std::complex<double>) / (n * n))
    throw std::invalid_argument("Complex orbital target extent is zero or overflows");
  if (!maximum_numeric_bytes || maximum_numeric_bytes == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Complex orbital target requires a positive finite numeric cap");
  if (electrons.isSpinor() || electrons.groups() != 2 || !electrons.groupsize(0) || !electrons.groupsize(1) ||
      electrons.getLattice().SuperCellEnum != SUPERCELL_BULK || source.getNumDets() != 2)
    throw std::invalid_argument("Complex orbital target requires two-spin bulk single Slater");
  std::array<double, 9> lattice;
  ParticleSet::PosType reduced_vector;
  const bool localized = dynamic_cast<const LCAOrbitalSet*>(&source.getDet(0).getPhi()) != nullptr;
  std::string identity = localized ? "complex-lcao-fixed-gauge-periodic-part-v1;" :
                                    "complex-free-orbital-fixed-gauge-periodic-part-v1;";
  auto append = [&](double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
      throw std::invalid_argument("Complex orbital target metadata is nonfinite");
    identity += std::to_string(bits) + ",";
  };
  for (int d = 0; d < 3; ++d)
  {
    if (electrons.getTwist()[d] != reduced[d])
      throw std::invalid_argument("Complex orbital target twist mismatch");
    append(reduced[d]);
    reduced_vector[d] = reduced[d];
    for (int e = 0; e < 3; ++e)
      append(lattice[3 * d + e] = electrons.getLattice().R(d, e));
  }
  const auto twist = electrons.getLattice().k_cart(reduced_vector);
  std::array<double, 3> cartesian{twist[0], twist[1], twist[2]};
  for (double value : cartesian)
    append(value);
  std::array<std::unique_ptr<SPOSet>, 2> orbitals;
  std::array<const SPOSet*, 2> sources;
  std::array<const ParticleSet*, 2> source_ions{};
  std::size_t detached_coefficient_elements = 0;
  std::size_t numeric_elements = channels * n * n;
  int first = 0;
  for (int spin = 0; spin < 2; ++spin)
  {
    const auto& determinant = source.getDet(spin);
    const auto* free = dynamic_cast<const FreeOrbital*>(&determinant.getPhi());
    const auto* lcao = dynamic_cast<const LCAOrbitalSet*>(&determinant.getPhi());
    const auto count = electrons.groupsize(spin);
    if (determinant.getFirstIndex() != first || determinant.getLastIndex() != first + count ||
        determinant.getPhi().getOrbitalSetSize() < count || (localized ? !lcao : !free))
      throw std::invalid_argument("Complex orbital target requires matching occupied periodic spin blocks");
    if (localized)
    {
      if (lcao->isOMPoffload() || lcao->isIdentity() || !lcao->C || !lcao->fileImageIdentity() ||
          !lcao->periodicIdentity() || !lcao->periodicIdentity()->source_ions)
        throw std::invalid_argument("Complex Gaussian target requires CPU coefficients and periodic reader identity");
      const auto& periodic = *lcao->periodicIdentity();
      source_ions[spin] = periodic.source_ions;
      for (int d = 0; d < 3; ++d)
      {
        // The reader uses Cartesian Bloch momentum; the student supplies reduced simulation twist.
        if (!isFiniteDouble(periodic.cartesian_twist[d]) || periodic.images[d] <= 0 ||
            std::abs(periodic.cartesian_twist[d] - cartesian[d]) >
                1e-10 * std::max(1.0, std::abs(cartesian[d])))
          throw std::invalid_argument("Complex Gaussian target requires matching periodic Cartesian twist and images");
        append(periodic.images[d]);
        for (int c = 0; c < 3; ++c)
        {
          const double actual = periodic.lattice[d][c];
          if (!isFiniteDouble(actual) || std::abs(actual - lattice[3 * d + c]) >
              1e-10 * std::max(1.0, std::abs(lattice[3 * d + c])))
            throw std::invalid_argument("Complex Gaussian target basis lattice differs from electron cell");
          append(actual);
        }
      }
      const auto& ions = *source_ions[spin];
      if (ions.R.size() != static_cast<std::size_t>(ions.getTotalNum()) || ions.GroupID.size() != ions.R.size())
        throw std::invalid_argument("Complex Gaussian target ionic metadata is incomplete");
      for (std::size_t ion = 0; ion < ions.R.size(); ++ion)
      {
        if (ions.GroupID[ion] < 0 || ions.GroupID[ion] >= ions.getSpeciesSet().getTotalNum())
          throw std::invalid_argument("Complex Gaussian target ionic species is invalid");
        identity += ions.getSpeciesSet().getSpeciesName(ions.GroupID[ion]) + ";";
        for (int d = 0; d < 3; ++d)
          append(ions.R[ion][d]);
      }
      // Exact reader image and accepted coefficients preserve column order and gauge.
      identity += lcao->fileImageIdentity()->image_hex + ";";
      for (std::size_t coefficient = 0; coefficient < lcao->C->size(); ++coefficient)
      {
        append(std::real(lcao->C->data()[coefficient]));
        append(std::imag(lcao->C->data()[coefficient]));
      }
      if (lcao->C->size() > std::numeric_limits<std::size_t>::max() - detached_coefficient_elements)
        throw std::length_error("Complex Gaussian target coefficient extent overflow");
      detached_coefficient_elements += lcao->C->size();
    }
    else
    {
      if (free->kVectors().size() < static_cast<std::size_t>(count))
        throw std::invalid_argument("Complex orbital target lacks occupied plane-wave momenta");
      for (int column = 0; column < count; ++column)
      {
        const auto& k = free->kVectors()[column];
        for (int d = 0; d < 3; ++d)
        {
          append(k[d]);
          double reciprocal = 0;
          for (int e = 0; e < 3; ++e)
            reciprocal += (k[e] - twist[e]) * lattice[3 * d + e] / (2 * std::acos(-1.0));
          if (!isFiniteDouble(reciprocal) || std::abs(reciprocal - std::round(reciprocal)) > 1e-10)
            throw std::invalid_argument("Complex orbital target momentum/twist is not reciprocal");
        }
      }
    }
    const auto scratch = static_cast<std::size_t>(determinant.getPhi().getOrbitalSetSize());
    if (scratch > std::numeric_limits<std::size_t>::max() - numeric_elements)
      throw std::length_error("Complex orbital target numeric extent overflow");
    numeric_elements += scratch;
    sources[spin] = &determinant.getPhi();
    first += count;
  }
  // Admit target plus per-spin scratch before cloning or allocating either numeric buffer.
  if (detached_coefficient_elements > std::numeric_limits<std::size_t>::max() - numeric_elements ||
      numeric_elements + detached_coefficient_elements > maximum_numeric_bytes / sizeof(std::complex<double>))
    throw std::length_error("Complex orbital target numeric cap exceeded before allocation");
  for (int spin = 0; spin < 2; ++spin)
  {
    orbitals[spin] = sources[spin]->makeClone();
    if (localized)
    {
      auto& frozen = static_cast<LCAOrbitalSet&>(*orbitals[spin]);
      frozen.C = std::make_shared<LCAOrbitalSet::OffloadValueMatrix>(*frozen.C);
    }
  }
  return {OrbitalTargetDescriptor::makeSingleReference(identity, electrons.groupsize(0),
      electrons.groupsize(1), channels), std::move(orbitals), lattice, reduced, cartesian, source_ions,
      detached_coefficient_elements * sizeof(std::complex<double>)};
}

/// Allocate only the frozen target and per-spin value scratch; SPO clone storage is separate.
ComplexOrbitalTargetEvaluator::ComplexOrbitalTargetEvaluator(OrbitalTargetDescriptor descriptor,
    std::array<std::unique_ptr<SPOSet>, 2>&& orbitals, std::array<double, 9> lattice,
    std::array<double, 3> reduced, std::array<double, 3> cartesian,
    std::array<const ParticleSet*, 2> source_ions, std::size_t detached_coefficient_bytes)
    : descriptor_(std::move(descriptor)), orbitals_(std::move(orbitals)), lattice_(lattice),
      reduced_twist_(reduced), cartesian_twist_(cartesian),
      target_(descriptor_.terms().size() * descriptor_.electronCount() * descriptor_.electronCount()),
      source_ions_(source_ions), detached_coefficient_bytes_(detached_coefficient_bytes)
{
  for (int spin = 0; spin < 2; ++spin)
  {
    scratch_[spin].resize(orbitals_[spin]->getOrbitalSetSize());
    if (source_ions_[spin])
      for (std::size_t ion = 0; ion < source_ions_[spin]->R.size(); ++ion)
      {
        const auto& r = source_ions_[spin]->R[ion];
        source_positions_[spin].push_back({r[0], r[1], r[2]});
        source_species_[spin].push_back(source_ions_[spin]->GroupID[ion]);
      }
  }
}

/// Validate all sample metadata/coordinates before filling one reusable target matrix.
const std::vector<std::complex<double>>& ComplexOrbitalTargetEvaluator::evaluate(const ParticleSet& electrons)
{
  // Localized basis clones refer to fixed physical ions; reject later ionic moves.
  for (int spin = 0; spin < 2; ++spin)
    if (source_ions_[spin])
    {
      const auto& ions = *source_ions_[spin];
      if (ions.R.size() != source_positions_[spin].size() || ions.GroupID.size() != source_species_[spin].size())
        throw std::invalid_argument("Complex Gaussian target ionic count changed");
      for (std::size_t ion = 0; ion < ions.R.size(); ++ion)
      {
        if (ions.GroupID[ion] != source_species_[spin][ion])
          throw std::invalid_argument("Complex Gaussian target ionic species changed");
        for (int d = 0; d < 3; ++d)
          if (ions.R[ion][d] != source_positions_[spin][ion][d])
            throw std::invalid_argument("Complex Gaussian target requires fixed ionic positions");
      }
    }
  const auto n = descriptor_.electronCount(), up = descriptor_.spinUpElectrons();
  if (electrons.isSpinor() || electrons.getActivePtcl() >= 0 || electrons.getTotalNum() != n ||
      electrons.groups() != 2 || electrons.groupsize(0) != up || electrons.groupsize(1) != n - up ||
      electrons.getLattice().SuperCellEnum != SUPERCELL_BULK || electrons.R.size() != n || electrons.GroupID.size() != n)
    throw std::invalid_argument("Complex orbital target sample/spin mismatch");
  for (std::size_t row = 0; row < n; ++row)
    for (int d = 0; d < 3; ++d)
      if (electrons.GroupID[row] != (row < up ? 0 : 1) || !isFiniteDouble(electrons.R[row][d]))
        throw std::invalid_argument("Complex orbital target sample coordinates/spin order invalid");
  for (int d = 0; d < 3; ++d)
  {
    if (electrons.getTwist()[d] != reduced_twist_[d])
      throw std::invalid_argument("Complex orbital target sample twist changed");
    for (int e = 0; e < 3; ++e)
      if (electrons.getLattice().R(d, e) != lattice_[3 * d + e])
        throw std::invalid_argument("Complex orbital target lattice changed");
  }
  std::fill(target_.begin(), target_.end(), std::complex<double>{});
  for (std::size_t row = 0; row < n; ++row)
  {
    const int spin = row < up ? 0 : 1;
    double phase = 0;
    for (int d = 0; d < 3; ++d)
      phase -= cartesian_twist_[d] * electrons.R[row][d];
    orbitals_[spin]->evaluateValue(electrons, row, scratch_[spin]);
    // Remove the common row twist exactly once; reciprocal orbital phases remain.
    const std::complex<double> detwist{std::cos(phase), std::sin(phase)};
    for (std::size_t column = 0; column < (spin ? n - up : up); ++column)
      for (std::size_t channel = 0; channel < descriptor_.terms().size(); ++channel)
        target_[(channel * n + row) * n + column + (spin ? up : 0)] = detwist * scratch_[spin][column];
  }
  for (auto value : target_)
    if (!isFiniteDouble(value.real()) || !isFiniteDouble(value.imag()))
      throw std::overflow_error("Complex orbital target value is nonfinite");
  return target_;
}

/// Excludes opaque SPO clone storage and descriptor metadata, as documented by the API.
std::size_t ComplexOrbitalTargetEvaluator::retainedScalarBytes() const noexcept
{
  return (target_.capacity() + scratch_[0].capacity() + scratch_[1].capacity()) * sizeof(std::complex<double>) +
      detached_coefficient_bytes_;
}
#endif

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

/// Convert QMCPACK's complex-log convention to a real signed-log amplitude.
RealSignedLogAmplitude checkedRealLogAmplitude(const WaveFunctionComponent::LogValue& value,
                                               const char* description)
{
  const double log_abs = std::real(value);
  const double angle   = std::imag(value);
  if (!isFiniteDouble(log_abs) && log_abs != -std::numeric_limits<double>::infinity())
    throw std::domain_error(std::string("PsiFormer ") + description + " log magnitude is invalid");
  if (!isFiniteDouble(angle) || std::abs(std::sin(angle)) > 1.0e-12)
    throw std::invalid_argument(std::string("PsiFormer ") + description +
                                " is not a real signed wavefunction");
  if (log_abs == -std::numeric_limits<double>::infinity())
    return {};
  return {std::cos(angle) < 0.0 ? -1.0 : 1.0, log_abs};
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

DeterminantOverlapTargetDescriptor::DeterminantOverlapTargetDescriptor(
    DeterminantOverlapTargetMode mode,
    std::string source_identity,
    std::size_t channel_count,
    std::vector<std::size_t> source_ordinals,
    std::vector<double> selected_coefficients)
    : mode_(mode),
      source_identity_(std::move(source_identity)),
      channel_count_(channel_count),
      source_ordinals_(std::move(source_ordinals)),
      selected_coefficients_(std::move(selected_coefficients)),
      fingerprint_(1469598103934665603ULL)
{
  if (source_identity_.empty() || channel_count_ == 0)
    throw std::invalid_argument("PsiFormer overlap target requires a source identity and channels");
  if (mode_ == DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER)
  {
    if (!source_ordinals_.empty() || !selected_coefficients_.empty())
      throw std::invalid_argument("General PsiFormer overlap target cannot select CI terms");
  }
  else if (source_ordinals_.size() != channel_count_ ||
           selected_coefficients_.size() != channel_count_)
    throw std::invalid_argument("Paired PsiFormer overlap target has the wrong CI term count");

  hashUnsigned(fingerprint_, 1); // overlap-target serialization convention
  hashUnsigned(fingerprint_, static_cast<std::uint64_t>(mode_));
  hashString(fingerprint_, source_identity_);
  hashUnsigned(fingerprint_, channel_count_);
  for (std::size_t channel = 0; channel < source_ordinals_.size(); ++channel)
  {
    if (!isFiniteDouble(selected_coefficients_[channel]) || selected_coefficients_[channel] == 0.0)
      throw std::invalid_argument("Paired PsiFormer overlap target coefficient must be finite and nonzero");
    hashUnsigned(fingerprint_, source_ordinals_[channel]);
    std::uint64_t coefficient_bits;
    std::memcpy(&coefficient_bits, &selected_coefficients_[channel], sizeof(coefficient_bits));
    hashUnsigned(fingerprint_, coefficient_bits);
  }
}

DeterminantOverlapTargetDescriptor
DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(std::string source_identity,
                                                             std::size_t determinant_count)
{
  return {DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER,
          std::move(source_identity), determinant_count, {}, {}};
}

DeterminantOverlapTargetDescriptor
DeterminantOverlapTargetDescriptor::makeTopCoefficientMultideterminant(
    const MultiSlaterDetTableMethod& source,
    std::string source_identity,
    std::size_t determinant_count)
{
#if defined(QMC_COMPLEX)
  throw std::invalid_argument("PsiFormer determinant-overlap targets are real-only");
#else
  if (source.hasCSFExpansion())
    throw std::invalid_argument("PsiFormer determinant-overlap target requires explicit DETS");
  std::vector<std::size_t> order(source.get_C().size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&source](std::size_t left, std::size_t right) {
    return std::abs(source.get_C()[left]) > std::abs(source.get_C()[right]);
  });

  std::vector<std::size_t> ordinals;
  std::vector<double> coefficients;
  ordinals.reserve(determinant_count);
  coefficients.reserve(determinant_count);
  for (std::size_t ordinal : order)
  {
    const double coefficient = checkedRealValue(source.get_C()[ordinal], "overlap DETS coefficient");
    if (coefficient == 0.0)
      continue;
    ordinals.push_back(ordinal);
    coefficients.push_back(coefficient);
    if (ordinals.size() == determinant_count)
      break;
  }
  if (ordinals.size() != determinant_count)
    throw std::invalid_argument("PsiFormer overlap target requests more channels than nonzero DETS terms");
  return {DeterminantOverlapTargetMode::TOP_COEFFICIENT_MULTIDETERMINANT,
          std::move(source_identity), determinant_count, std::move(ordinals),
          std::move(coefficients)};
#endif
}

DeterminantOverlapTargetEvaluator::DeterminantOverlapTargetEvaluator(
    DeterminantOverlapTargetDescriptor descriptor)
    : descriptor_(std::move(descriptor))
{
#if defined(QMC_COMPLEX)
  throw std::invalid_argument("PsiFormer determinant-overlap targets are real-only");
#endif
}

DeterminantOverlapTargetEvaluator
DeterminantOverlapTargetEvaluator::fromFrozenTeacher(
    const TrialWaveFunction& teacher,
    DeterminantOverlapTargetMode mode,
    std::string source_identity,
    std::size_t channel_count)
{
  if (mode == DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER)
    return DeterminantOverlapTargetEvaluator(
        DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(
            std::move(source_identity), channel_count));

  const MultiSlaterDetTableMethod* multideterminant = nullptr;
  std::size_t fermionic_components = 0;
  for (const auto& component : teacher.getOrbitals())
  {
    fermionic_components += component->isFermionic() ? 1 : 0;
    if (const auto* candidate =
            dynamic_cast<const MultiSlaterDetTableMethod*>(component.get()))
    {
      if (multideterminant)
        throw std::invalid_argument(
            "Paired PsiFormer overlap teacher has multiple multideterminant components");
      multideterminant = candidate;
    }
  }
  if (!multideterminant || fermionic_components != 1)
    throw std::invalid_argument(
        "Paired PsiFormer overlap teacher requires one explicit multideterminant fermion component");
  return DeterminantOverlapTargetEvaluator(
      DeterminantOverlapTargetDescriptor::makeTopCoefficientMultideterminant(
          *multideterminant, std::move(source_identity), channel_count));
}

void DeterminantOverlapTargetEvaluator::evaluate(TrialWaveFunction& frozen_teacher,
                                                 ParticleSet& electrons,
                                                 RealSignedLogAmplitude& teacher,
                                                 RealSignedLogAmplitude* targets,
                                                 std::size_t target_count) const
{
  if (!targets || target_count != descriptor_.channelCount())
    throw std::invalid_argument("PsiFormer overlap target output has the wrong channel count");

  frozen_teacher.evaluateLog(electrons);
  const WaveFunctionComponent::LogValue full_log(frozen_teacher.getLogPsi(), frozen_teacher.getPhase());
  teacher = checkedRealLogAmplitude(full_log, "overlap teacher");
  if (descriptor_.mode() == DeterminantOverlapTargetMode::GENERAL_FROZEN_TEACHER)
  {
    std::fill_n(targets, target_count, teacher);
    return;
  }

  const MultiSlaterDetTableMethod* msd = nullptr;
  std::size_t fermionic_components = 0;
  for (const auto& component : frozen_teacher.getOrbitals())
  {
    fermionic_components += component->isFermionic() ? 1 : 0;
    if (const auto* candidate = dynamic_cast<const MultiSlaterDetTableMethod*>(component.get()))
    {
      if (msd)
        throw std::invalid_argument(
            "Paired PsiFormer overlap target requires exactly one multideterminant component");
      msd = candidate;
    }
  }
  if (!msd)
    throw std::invalid_argument(
        "Paired PsiFormer overlap target requires exactly one multideterminant component");
  if (fermionic_components != 1)
    throw std::invalid_argument("Paired PsiFormer overlap target cannot disambiguate multiple fermionic components");

  // Keep every common component by replacing only the full MSD logarithm with
  // the selected coefficient-weighted determinant term.
  const auto common_log = full_log - msd->get_log_value();
  for (std::size_t channel = 0; channel < target_count; ++channel)
  {
    const auto selected_log = msd->getIndividualTermLogValue(descriptor_.sourceOrdinals()[channel]);
    targets[channel] = checkedRealLogAmplitude(common_log + selected_log,
                                               "paired multideterminant target");
  }
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
