//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Lattice/CrystalLattice.h"
#include "Particle/DistanceTable.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace qmcplusplus
{
namespace
{
static_assert(!std::is_constructible_v<VirtualParticleBatch::PositionView,
                                       std::vector<VirtualParticleBatch::PosType>&&>);
static_assert(!std::is_constructible_v<VirtualParticleBatch::PositionView,
                                       const std::vector<VirtualParticleBatch::PosType>&&>);

SimulationCell makeVirtualBatchOpenCell()
{
  Lattice lattice;
  lattice.BoxBConds          = false;
  lattice.R                  = ParticleSet::Tensor_t(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0);
  lattice.explicitly_defined = true;
  lattice.reset();
  return SimulationCell(lattice);
}

template<typename T>
bool sameObjectRepresentation(const T& left, const T& right)
{
  return std::memcmp(&left, &right, sizeof(T)) == 0;
}
} // namespace

TEST_CASE("VirtualParticleBatch validates CSR shape slices and fingerprints", "[particle]")
{
  using Batch   = VirtualParticleBatch;
  using Segment = Batch::Segment;
  const std::vector<std::size_t> offsets{0, 2, 3, 5};
  const std::vector<Segment> segments{{0, 1}, {0, 0, true, 1}, {2, 2, true, 0}};
  const std::vector<Batch::PosType> positions{{0.25, -0.0, 0.75},
                                               {0.50, 0.75, 1.00},
                                               {1.25, 1.50, 1.75},
                                               {2.00, 2.25, 2.50},
                                               {2.75, 3.00, 3.25}};
  const Batch batch(3, offsets, segments, positions);
  const Batch equivalent(3, offsets, segments, positions);

  CHECK(batch.walkerCount() == 3);
  CHECK(batch.segmentCount() == 3);
  CHECK(batch.size() == positions.size());
  CHECK_FALSE(batch.empty());
  CHECK(batch.segmentOffsets().size() == offsets.size());
  CHECK(batch.segmentOffsets().at(2) == 3);
  CHECK(batch.segments().at(1).walkerId() == 0);
  CHECK(batch.absolutePositions().at(4) == positions[4]);
  REQUIRE_THROWS_AS(batch.segmentOffsets().at(offsets.size()), std::out_of_range);

  const Batch::Slice first = batch.slice(0);
  CHECK(first.size() == 2);
  CHECK_FALSE(first.empty());
  CHECK(first.flatOffset() == 0);
  CHECK(first.walkerId() == 0);
  CHECK(first.electronId() == 1);
  CHECK_FALSE(first.isOnSphere());
  CHECK(first.sourceCenterId() == Batch::NO_SOURCE);
  CHECK(first.absolutePosition(1) == positions[1]);
  CHECK(first.positions().data() == positions.data());
  CHECK(first.positions().size() == 2);

  const Batch::Slice last = batch.slice(2);
  CHECK(last.flatOffset() == 3);
  CHECK(last.walkerId() == 2);
  CHECK(last.electronId() == 2);
  CHECK(last.isOnSphere());
  CHECK(last.sourceCenterId() == 0);
  CHECK(last.positions().data() == positions.data() + 3);
  REQUIRE_THROWS_AS(batch.slice(3), std::out_of_range);
  REQUIRE_THROWS_AS(first.absolutePosition(2), std::out_of_range);

  CHECK_NOTHROW(batch.validateOutputExtent(positions.size()));
  REQUIRE_THROWS_AS(batch.validateOutputExtent(positions.size() - 1), std::invalid_argument);
  CHECK(batch.fingerprint() == equivalent.fingerprint());

  auto changed_positions = positions;
  changed_positions[0][0] = std::nextafter(changed_positions[0][0], Batch::RealType(1));
  CHECK(Batch(3, offsets, segments, changed_positions).fingerprint() != batch.fingerprint());

  const std::vector<std::size_t> changed_offsets{0, 1, 3, 5};
  CHECK(Batch(3, changed_offsets, segments, positions).fingerprint() != batch.fingerprint());

  const std::vector<Segment> changed_context{{0, 1, true, 1}, {0, 0, true, 1}, {2, 2, true, 0}};
  CHECK(Batch(3, offsets, changed_context, positions).fingerprint() != batch.fingerprint());
  const std::vector<Segment> changed_walker{{1, 1}, {0, 0, true, 1}, {2, 2, true, 0}};
  CHECK(Batch(3, offsets, changed_walker, positions).fingerprint() != batch.fingerprint());
  const std::vector<Segment> changed_electron{{0, 2}, {0, 0, true, 1}, {2, 2, true, 0}};
  CHECK(Batch(3, offsets, changed_electron, positions).fingerprint() != batch.fingerprint());
  const std::vector<Segment> changed_source{{0, 1}, {0, 0, true, 2}, {2, 2, true, 0}};
  CHECK(Batch(3, offsets, changed_source, positions).fingerprint() != batch.fingerprint());
  CHECK(Batch(4, offsets, segments, positions).fingerprint() != batch.fingerprint());

  auto changed_zero = positions;
  changed_zero[0][1] = 0.0;
  CHECK(Batch(3, offsets, segments, changed_zero).fingerprint() != batch.fingerprint());
}

TEST_CASE("VirtualParticleBatch rejects malformed intrinsic data before access", "[particle]")
{
  using Batch   = VirtualParticleBatch;
  using Segment = Batch::Segment;
  const std::vector<Batch::PosType> one_position{{0.25, 0.50, 0.75}};
  const std::vector<Batch::PosType> two_positions{{0.25, 0.50, 0.75}, {1.00, 1.25, 1.50}};
  const std::vector<Segment> one_segment{{0, 0}};

  const std::vector<std::size_t> no_offsets;
  REQUIRE_THROWS_AS(Batch(1, no_offsets, one_segment, one_position), std::invalid_argument);
  const std::vector<std::size_t> too_many_offsets{0, 1, 1};
  REQUIRE_THROWS_AS(Batch(1, too_many_offsets, one_segment, one_position), std::invalid_argument);
  const std::vector<std::size_t> nonzero_first{1, 1};
  REQUIRE_THROWS_AS(Batch(1, nonzero_first, one_segment, one_position), std::invalid_argument);
  const std::vector<std::size_t> wrong_final{0, 0};
  REQUIRE_THROWS_AS(Batch(1, wrong_final, one_segment, one_position), std::invalid_argument);

  const std::vector<Segment> two_segments{{0, 0}, {0, 1}};
  const std::vector<std::size_t> decreasing{0, 2, 1};
  REQUIRE_THROWS_AS(Batch(1, decreasing, two_segments, one_position), std::invalid_argument);
  const std::vector<std::size_t> out_of_range{0, 1, 3};
  REQUIRE_THROWS_AS(Batch(1, out_of_range, two_segments, two_positions), std::invalid_argument);
  const std::vector<std::size_t> empty_segment{0, 0, 2};
  REQUIRE_THROWS_AS(Batch(1, empty_segment, two_segments, two_positions), std::invalid_argument);

  const std::vector<std::size_t> one_offset_range{0, 1};
  const std::vector<Segment> negative_walker{{-1, 0}};
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, negative_walker, one_position), std::invalid_argument);
  const std::vector<Segment> large_walker{{1, 0}};
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, large_walker, one_position), std::invalid_argument);
  const std::vector<Segment> negative_electron{{0, -1}};
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, negative_electron, one_position), std::invalid_argument);
  const std::vector<Segment> missing_source{{0, 0, true, Batch::NO_SOURCE}};
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, missing_source, one_position), std::invalid_argument);
  const std::vector<Segment> unexpected_source{{0, 0, false, 0}};
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, unexpected_source, one_position), std::invalid_argument);

  auto nonfinite = one_position;
  nonfinite[0][0] = std::numeric_limits<Batch::RealType>::infinity();
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, one_segment, nonfinite), std::invalid_argument);
  nonfinite[0][0] = std::numeric_limits<Batch::RealType>::quiet_NaN();
  REQUIRE_THROWS_AS(Batch(1, one_offset_range, one_segment, nonfinite), std::invalid_argument);

  REQUIRE_THROWS_AS(Batch::PositionView(nullptr, 1), std::invalid_argument);
  const Segment segment(0, 0);
  const Batch::SegmentView overflowing_segments(&segment, std::numeric_limits<std::size_t>::max());
  const Batch::OffsetView offsets_view(one_offset_range);
  const Batch::PositionView positions_view(one_position);
  REQUIRE_THROWS_AS(Batch(1, offsets_view, overflowing_segments, positions_view), std::overflow_error);
}

TEST_CASE("VirtualParticleBatch validates unique walker and particle ranges", "[particle]")
{
  using Batch   = VirtualParticleBatch;
  using Segment = Batch::Segment;
  const SimulationCell cell = makeVirtualBatchOpenCell();
  ParticleSet p0(cell);
  p0.create({3});
  ParticleSet p1(p0);
  ParticleSet p2(p0);
  RefVectorWithLeader<ParticleSet> p_list(p0, {p0, p1, p2});

  const std::vector<std::size_t> offsets{0, 1, 2};
  const std::vector<Segment> segments{{0, 2, true, 1}, {2, 1}};
  const std::vector<Batch::PosType> positions{{0.25, 0.50, 0.75}, {1.00, 1.25, 1.50}};
  const Batch batch(3, offsets, segments, positions);
  CHECK_NOTHROW(batch.validateFor(p_list));
  CHECK_NOTHROW(batch.validateFor(p_list, 2));
  REQUIRE_THROWS_AS(batch.validateFor(p_list, 1), std::invalid_argument);
  RefVectorWithLeader<const ParticleSet> const_p_list(p0, {p0, p1, p2});
  CHECK_NOTHROW(batch.validateFor(const_p_list, 2));

  const std::vector<std::size_t> empty_offsets{0};
  const std::vector<Segment> no_segments;
  const std::vector<Batch::PosType> no_positions;
  const Batch empty_walkers(3, empty_offsets, no_segments, no_positions);
  CHECK(empty_walkers.empty());
  CHECK_NOTHROW(empty_walkers.validateFor(p_list, 0));

  RefVectorWithLeader<ParticleSet> empty_list(p0);
  const Batch zero_walkers(0, empty_offsets, no_segments, no_positions);
  CHECK(zero_walkers.walkerCount() == 0);
  CHECK(zero_walkers.segmentCount() == 0);
  CHECK(zero_walkers.fingerprint() != empty_walkers.fingerprint());
  CHECK_NOTHROW(zero_walkers.validateFor(empty_list));
  REQUIRE_THROWS_AS(zero_walkers.validateFor(p_list), std::invalid_argument);

  RefVectorWithLeader<ParticleSet> duplicate_list(p0, {p0, p1, p0});
  REQUIRE_THROWS_AS(batch.validateFor(duplicate_list, 2), std::invalid_argument);

  const std::vector<Segment> bad_electron{{0, 3, true, 0}};
  const std::vector<std::size_t> bad_electron_offsets{0, 1};
  const std::vector<Batch::PosType> bad_electron_positions{{0.25, 0.50, 0.75}};
  const Batch out_of_range_electron(3, bad_electron_offsets, bad_electron, bad_electron_positions);
  REQUIRE_THROWS_AS(out_of_range_electron.validateFor(p_list, 2), std::invalid_argument);
}

TEST_CASE("VirtualParticleSet installs exact absolute positions atomically after validation", "[particle]")
{
  using Batch = VirtualParticleBatch;
  const SimulationCell cell = makeVirtualBatchOpenCell();
  ParticleSet ions(cell);
  ions.setName("ion");
  ions.create({2});
  ions.R[0] = {0.0, 0.0, 0.0};
  ions.R[1] = {2.0, 2.0, 2.0};
  ions.update();

  ParticleSet electrons(cell);
  electrons.setName("electron");
  electrons.create({2});
  electrons.setSpinor(true);
  electrons.R[0]  = {0.25, 0.50, 0.75};
  electrons.R[1]  = {1.25, 1.50, 1.75};
  electrons.spins = {0.125, -0.375};
  electrons.addTable(ions);
  electrons.addTable(electrons);
  electrons.update();

  ParticleSet nonspin_electrons(cell);
  nonspin_electrons.setName("nonspin_electron");
  nonspin_electrons.create({2});
  nonspin_electrons.R[0] = electrons.R[0];
  nonspin_electrons.R[1] = electrons.R[1];
  nonspin_electrons.addTable(ions);
  nonspin_electrons.addTable(nonspin_electrons);
  nonspin_electrons.update();

  VirtualParticleSet virtual_particles(electrons);
  const std::vector<std::size_t> offsets{0, 2};
  const std::vector<Batch::Segment> segments{{0, 1, true, 1}};
  const Batch::PosType exact0{std::nextafter(Batch::RealType(0.1), Batch::RealType(1.0)), -0.0,
                              std::nextafter(Batch::RealType(2.0), Batch::RealType(3.0))};
  const Batch::PosType exact1{std::nextafter(Batch::RealType(-0.25), Batch::RealType(-1.0)), 1.0 / 3.0,
                              std::nextafter(Batch::RealType(4.0), Batch::RealType(3.0))};
  const std::vector<Batch::PosType> positions{exact0, exact1};
  const Batch batch(1, offsets, segments, positions);

  RefVectorWithLeader<ParticleSet> p_list(electrons, {electrons});
  CHECK_NOTHROW(batch.validateFor(p_list, ions.getTotalNum()));
  const Batch::Slice job = batch.slice(0);
  virtual_particles.makeMovesAbsolute(electrons, job.electronId(), job.positions(), job.isOnSphere(),
                                      job.sourceCenterId());

  REQUIRE(virtual_particles.getTotalNum() == positions.size());
  CHECK(virtual_particles.refPtcl == 1);
  CHECK(virtual_particles.isOnSphere());
  CHECK(virtual_particles.refSourcePtcl == 1);
  CHECK(std::addressof(virtual_particles.getRefPS()) == std::addressof(electrons));
  for (std::size_t ivp = 0; ivp < positions.size(); ++ivp)
  {
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      CHECK(sameObjectRepresentation(virtual_particles.R[ivp][idim], positions[ivp][idim]));
      CHECK(sameObjectRepresentation(virtual_particles.getCoordinates().getAllParticlePos()[ivp][idim],
                                     positions[ivp][idim]));
    }
    CHECK(virtual_particles.spins[ivp] == electrons.spins[1]);
    for (std::size_t ion = 0; ion < ions.getTotalNum(); ++ion)
    {
      const Batch::PosType displacement = positions[ivp] - ions.R[ion];
      const Batch::RealType expected_distance = std::sqrt(dot(displacement, displacement));
      CHECK(virtual_particles.getDistTableAB(0).getDistances()[ivp][ion] == Approx(expected_distance));
    }
  }

  const std::vector<Batch::PosType> saved_positions(virtual_particles.R.begin(), virtual_particles.R.end());
  const std::vector<Batch::RealType> saved_spins(virtual_particles.spins.begin(), virtual_particles.spins.end());
  std::vector<Batch::RealType> saved_ion_distances;
  for (std::size_t ivp = 0; ivp < positions.size(); ++ivp)
    for (std::size_t ion = 0; ion < ions.getTotalNum(); ++ion)
      saved_ion_distances.push_back(virtual_particles.getDistTableAB(0).getDistances()[ivp][ion]);
  const auto check_unchanged = [&]() {
    REQUIRE(virtual_particles.getTotalNum() == saved_positions.size());
    CHECK(virtual_particles.refPtcl == 1);
    CHECK(virtual_particles.isOnSphere());
    CHECK(virtual_particles.refSourcePtcl == 1);
    CHECK(std::addressof(virtual_particles.getRefPS()) == std::addressof(electrons));
    for (std::size_t ivp = 0; ivp < saved_positions.size(); ++ivp)
    {
      CHECK(virtual_particles.R[ivp] == saved_positions[ivp]);
      CHECK(virtual_particles.spins[ivp] == saved_spins[ivp]);
      for (std::size_t ion = 0; ion < ions.getTotalNum(); ++ion)
        CHECK(virtual_particles.getDistTableAB(0).getDistances()[ivp][ion] ==
              saved_ion_distances[ivp * ions.getTotalNum() + ion]);
    }
  };

  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, -1, positions, true, 1),
                    std::invalid_argument);
  check_unchanged();
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, 2, positions, true, 1),
                    std::invalid_argument);
  check_unchanged();
  const std::vector<Batch::PosType> no_positions;
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, 1, no_positions, true, 1),
                    std::invalid_argument);
  check_unchanged();
  auto nonfinite_positions = positions;
  nonfinite_positions[0][2] = std::numeric_limits<Batch::RealType>::infinity();
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, 1, nonfinite_positions, true, 1),
                    std::invalid_argument);
  check_unchanged();
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, 1, positions, true, Batch::NO_SOURCE),
                    std::invalid_argument);
  check_unchanged();
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(electrons, 1, positions, false, 0),
                    std::invalid_argument);
  check_unchanged();
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(nonspin_electrons, 1, positions, true, 1),
                    std::invalid_argument);
  check_unchanged();
  const std::vector<Batch::PosType> self_reference_positions{{0.5, 0.75, 1.0}};
  REQUIRE_THROWS_AS(virtual_particles.makeMovesAbsolute(virtual_particles, 1, self_reference_positions),
                    std::invalid_argument);
  check_unchanged();

  const std::vector<Batch::PosType> off_sphere_positions{{0.625, 0.875, 1.125}};
  CHECK_NOTHROW(virtual_particles.makeMovesAbsolute(electrons, 0, off_sphere_positions));
  REQUIRE(virtual_particles.getTotalNum() == 1);
  CHECK(virtual_particles.refPtcl == 0);
  CHECK_FALSE(virtual_particles.isOnSphere());
  CHECK(virtual_particles.refSourcePtcl == Batch::NO_SOURCE);
  CHECK(virtual_particles.R[0] == off_sphere_positions[0]);
  CHECK(virtual_particles.spins[0] == electrons.spins[0]);
  for (std::size_t ion = 0; ion < ions.getTotalNum(); ++ion)
  {
    const Batch::PosType displacement = off_sphere_positions[0] - ions.R[ion];
    const Batch::RealType expected_distance = std::sqrt(dot(displacement, displacement));
    CHECK(virtual_particles.getDistTableAB(0).getDistances()[0][ion] == Approx(expected_distance));
  }

  VirtualParticleSet nonspin_virtual_particles(nonspin_electrons);
  const std::vector<Batch::PosType> nonspin_initial_positions{{0.75, 1.0, 1.25}};
  nonspin_virtual_particles.makeMovesAbsolute(nonspin_electrons, 0, nonspin_initial_positions);
  REQUIRE(nonspin_virtual_particles.getTotalNum() == 1);
  REQUIRE_THROWS_AS(nonspin_virtual_particles.makeMovesAbsolute(electrons, 1, positions, true, 1),
                    std::invalid_argument);
  CHECK(nonspin_virtual_particles.getTotalNum() == 1);
  CHECK(nonspin_virtual_particles.R[0] == nonspin_initial_positions[0]);
  CHECK(nonspin_virtual_particles.refPtcl == 0);
  CHECK_FALSE(nonspin_virtual_particles.isOnSphere());
  CHECK(nonspin_virtual_particles.refSourcePtcl == Batch::NO_SOURCE);
}

} // namespace qmcplusplus
