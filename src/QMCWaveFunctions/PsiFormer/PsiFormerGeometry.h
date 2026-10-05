//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerGeometry.h
 * @brief Allocation-free geometry cache for molecular PsiFormer evaluations.
 *
 * Geometry remains real even when a future wavefunction value type is complex.  Keeping
 * this cache independent of the wavefunction scalar type lets a later periodic/complex
 * evaluator reuse the same pair tables while a separate phase policy handles twists.
 * Only open boundaries are implemented here.  The boundary descriptor deliberately
 * reserves the cell and periodic-axis data needed by a future minimum-image policy;
 * requesting that unsupported policy fails immediately rather than silently evaluating
 * open-boundary displacements.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_GEOMETRY_H
#define QMCPLUSPLUS_PSIFORMER_GEOMETRY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pf
{

/// Real scalar used by molecular geometry independently of wavefunction value type.
using GeometryReal = double;

/// Three-dimensional Cartesian coordinate or displacement.
using GeometryPosition = std::array<GeometryReal, 3>;

/** Identify the displacement convention requested by an evaluator.
 *
 * PERIODIC is an interface reservation, not a currently supported calculation mode.
 */
enum class GeometryBoundaryKind
{
  OPEN,
  PERIODIC
};

/** Describe the real-space cell without embedding boundary arithmetic in consumers.
 *
 * Cell vectors and periodic axes are ignored for OPEN and retained only so a future
 * periodic displacement policy can be introduced without changing geometry clients.
 */
struct GeometryBoundary
{
  GeometryBoundaryKind kind = GeometryBoundaryKind::OPEN;
  std::array<GeometryPosition, 3> lattice_vectors{};
  std::array<bool, 3> periodic_axes{};
};

/// Report whether this implementation can evaluate the requested boundary policy.
bool supportsGeometryBoundary(GeometryBoundaryKind kind) noexcept;

/** Non-owning Cartesian coordinate view supporting interleaved and component storage.
 *
 * The view is intentionally independent of ParticleSet and Tensor.  Adapters can expose
 * either AoS or SoA storage without copying positions into the geometry layer.
 */
class GeometryPositionView
{
public:
  /// Construct an empty coordinate view.
  GeometryPositionView() = default;

  /// Build a view over xyzxyz... storage with a configurable particle stride.
  static GeometryPositionView interleaved(const GeometryReal* values,
                                          std::size_t particle_count,
                                          std::size_t particle_stride = 3);

  /// Build a view over three separately stored Cartesian component arrays.
  static GeometryPositionView components(const GeometryReal* x,
                                         const GeometryReal* y,
                                         const GeometryReal* z,
                                         std::size_t particle_count,
                                         std::size_t component_stride = 1);

  /// Return the number of particles represented by the view.
  std::size_t size() const noexcept { return particle_count_; }

  /// Read one Cartesian component, checking particle and dimension indices.
  GeometryReal operator()(std::size_t particle, std::size_t dimension) const;

  /// Materialize one position from the non-owning storage.
  GeometryPosition position(std::size_t particle) const;

private:
  /// Construct a validated view from component base pointers and strides.
  GeometryPositionView(std::array<const GeometryReal*, 3> components,
                       std::array<std::size_t, 3> strides,
                       std::size_t particle_count);

  std::array<const GeometryReal*, 3> components_{};
  std::array<std::size_t, 3> strides_{};
  std::size_t particle_count_ = 0;
};

/** Stable scalar factors for the PsiFormer feature f(r) = log(1+r)/r.
 *
 * The scalar radial derivatives are finite at r=0 as one-sided radial limits.
 * Cartesian cusp derivatives still require a displacement direction and are therefore
 * intentionally left to the consuming derivative kernel.
 */
struct SoftenedRadialFactors
{
  GeometryReal log1p_radius                 = 0;
  GeometryReal log1p_over_radius            = 1;
  GeometryReal log1p_first                  = 1;
  GeometryReal log1p_second                 = -1;
  GeometryReal log1p_over_radius_first      = -0.5;
  GeometryReal log1p_over_radius_second     = 2.0 / 3.0;
};

/// Evaluate softened radial values and first two scalar derivatives stably near zero.
SoftenedRadialFactors evaluateSoftenedRadialFactors(GeometryReal radius);

/** Store structure-of-arrays data for one homogeneous family of particle pairs.
 *
 * Storage is sized at construction.  Subsequent geometry refreshes overwrite entries
 * without changing vector sizes or capacities.
 */
class GeometryPairTable
{
public:
  /// Construct storage for a fixed number of pairs.
  explicit GeometryPairTable(std::size_t pair_count = 0);

  /// Return the fixed number of cached pairs.
  std::size_t size() const noexcept { return distances_.size(); }

  /// Return all pair displacements in pair-major order.
  const std::vector<GeometryPosition>& displacements() const noexcept { return displacements_; }

  /// Return all pair distances in pair-major order.
  const std::vector<GeometryReal>& distances() const noexcept { return distances_; }

  /// Return all inverse pair distances in pair-major order.
  const std::vector<GeometryReal>& inverseDistances() const noexcept { return inverse_distances_; }

  /// Return all PsiFormer softened radial factors in pair-major order.
  const std::vector<SoftenedRadialFactors>& softenedRadialFactors() const noexcept
  {
    return softened_radial_factors_;
  }

  /// Hash backing addresses and capacities for warmed-workspace stability checks.
  std::size_t storageFingerprint() const noexcept;

  /// Return bytes reserved by all pair-table vectors.
  std::size_t storageBytes() const noexcept;

private:
  friend class PsiFormerGeometryCache;

  /// Replace one pair entry from its already boundary-adjusted displacement.
  void updatePair(std::size_t pair_index, const GeometryPosition& displacement);

  std::vector<GeometryPosition> displacements_;
  std::vector<GeometryReal> distances_;
  std::vector<GeometryReal> inverse_distances_;
  std::vector<SoftenedRadialFactors> softened_radial_factors_;
};

/// Identify the two electrons represented by one unique electron-electron pair.
struct ElectronPair
{
  std::size_t first;
  std::size_t second;
};

/** Map one electron to a unique pair and the pair displacement orientation.
 *
 * The cached pair displacement is R[first]-R[second].  displacement_sign is +1
 * for the first electron and -1 for the second, allowing derivative kernels to apply
 * the correct incidence sign without searching the pair list.
 */
struct ElectronPairIncidence
{
  std::size_t pair_index;
  std::size_t other_electron;
  std::int8_t displacement_sign;
};

/** Cache fixed-nucleus electron-nucleus and unique electron-electron geometry.
 *
 * Full and single-electron refreshes perform no allocations.  Nuclei are copied once
 * at construction and cannot be moved, matching the fixed-ion scope of the current
 * PsiFormer implementation.
 */
class PsiFormerGeometryCache
{
public:
  /// Construct fixed-size pair tables and immutable open-boundary nuclear geometry.
  PsiFormerGeometryCache(std::size_t electron_count,
                         GeometryPositionView nuclei,
                         GeometryBoundary boundary = {});

  /// Report whether the cache contains at least one complete electron configuration.
  bool valid() const noexcept { return valid_; }

  /// Return a monotonically increasing identifier for successfully refreshed geometry.
  std::size_t generation() const noexcept { return generation_; }

  /// Return the number of electrons represented by this fixed-size cache.
  std::size_t electronCount() const noexcept { return electrons_.size(); }

  /// Return the number of fixed nuclei represented by this cache.
  std::size_t nucleusCount() const noexcept { return nuclei_.size(); }

  /// Return the boundary descriptor selected at construction.
  const GeometryBoundary& boundary() const noexcept { return boundary_; }

  /// Return the immutable nuclear positions copied at construction.
  const std::vector<GeometryPosition>& nuclei() const noexcept { return nuclei_; }

  /// Return the most recently cached electron positions.
  const std::vector<GeometryPosition>& electrons() const noexcept { return electrons_; }

  /// Return electron-nucleus pairs ordered as electron*nucleus_count+nucleus.
  const GeometryPairTable& electronNucleusPairs() const noexcept { return electron_nucleus_pairs_; }

  /// Return unique electron-electron pairs in lexicographic (first,second) order.
  const GeometryPairTable& electronElectronPairs() const noexcept { return electron_electron_pairs_; }

  /// Return the electron indices associated with each unique pair table entry.
  const std::vector<ElectronPair>& electronPairs() const noexcept { return electron_pairs_; }

  /// Return CSR offsets delimiting each electron's incident-pair entries.
  const std::vector<std::size_t>& incidenceOffsets() const noexcept { return incidence_offsets_; }

  /// Return all electron-pair incidence entries grouped by electron.
  const std::vector<ElectronPairIncidence>& incidences() const noexcept { return incidences_; }

  /// Hash every geometry backing allocation for warmed-workspace stability checks.
  std::size_t storageFingerprint() const noexcept;

  /// Return bytes reserved by all geometry-owned vectors and pair tables.
  std::size_t storageBytes() const noexcept;

  /// Return the regular electron-nucleus flat index for one particle pair.
  std::size_t electronNucleusPairIndex(std::size_t electron, std::size_t nucleus) const;

  /// Refresh every electron-nucleus and unique electron-electron pair in place.
  void update(GeometryPositionView electrons);

  /// Refresh pair entries incident on one accepted electron move in place.
  void updateElectron(std::size_t electron, const GeometryPosition& position);

private:
  /// Form an open-boundary target-minus-source displacement.
  GeometryPosition displacement(const GeometryPosition& target, const GeometryPosition& source) const;

  /// Populate immutable pair identities and grouped incidence metadata once.
  void initializeElectronPairs();

  GeometryBoundary boundary_;
  std::vector<GeometryPosition> nuclei_;
  std::vector<GeometryPosition> electrons_;
  GeometryPairTable electron_nucleus_pairs_;
  GeometryPairTable electron_electron_pairs_;
  std::vector<ElectronPair> electron_pairs_;
  std::vector<std::size_t> incidence_offsets_;
  std::vector<ElectronPairIncidence> incidences_;
  bool valid_             = false;
  std::size_t generation_ = 0;
};

} // namespace pf

#endif // QMCPLUSPLUS_PSIFORMER_GEOMETRY_H
