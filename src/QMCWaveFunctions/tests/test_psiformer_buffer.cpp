//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_buffer.cpp
 * @brief Public lifecycle tests for persistent PsiFormer accepted-state buffers.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "psiformer_test_utils.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <hdf5.h>
#include <limits>
#include <memory>
#include <vector>

namespace qmcplusplus
{
namespace
{
using namespace testing::psiformer;
using Value = QMCTraits::ValueType;
using Buffer = PsiFormerWF::WFBufferType;

/// Construct the four-electron runtime configuration used by generated LiH models.
ParticleSet makeBufferElectrons(const SimulationCell& simulation_cell)
{
  const Geometry geometry = makeGeometry("lih");
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({2, 2});
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Capture one component's accepted logarithm and component-only VGL contribution.
struct AcceptedSnapshot
{
  PsiFormerWF::LogValue log;
  ParticleSet::ParticleGradient gradient;
  ParticleSet::ParticleLaplacian laplacian;
};

/// Evaluate a component from scratch with zeroed caller accumulators.
AcceptedSnapshot evaluateAccepted(PsiFormerWF& component, ParticleSet& electrons)
{
  electrons.G = Value(0);
  electrons.L = Value(0);
  AcceptedSnapshot snapshot{component.evaluateLog(electrons, electrons.G, electrons.L),
                            electrons.G,
                            electrons.L};
  return snapshot;
}

/// Compare real or complex scalar values at direct-evaluator tolerance.
void checkValue(const Value& actual, const Value& expected, double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

/// Compare two real-log/phase amplitudes at direct-evaluator tolerance.
void checkLog(const PsiFormerWF::LogValue& actual, const PsiFormerWF::LogValue& expected)
{
  CHECK(std::real(actual) == Catch::Approx(std::real(expected)).epsilon(3.0e-9).margin(3.0e-9));
  CHECK(std::imag(actual) == Catch::Approx(std::imag(expected)).epsilon(3.0e-9).margin(3.0e-9));
}

/// Check that one cached component contribution was added to supplied sentinels.
void checkAccumulatedSpatial(const ParticleSet& electrons,
                             const AcceptedSnapshot& component,
                             const Value& gradient_sentinel,
                             const Value& laplacian_sentinel)
{
  REQUIRE(electrons.G.size() == component.gradient.size());
  REQUIRE(electrons.L.size() == component.laplacian.size());
  for (std::size_t electron = 0; electron < electrons.G.size(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      checkValue(electrons.G[electron][dimension],
                 gradient_sentinel + component.gradient[electron][dimension]);
    checkValue(electrons.L[electron], laplacian_sentinel + component.laplacian[electron]);
  }
}

/// Allocate the exact bulk/scalar layout registered by one component.
Buffer makeRegisteredBuffer(PsiFormerWF& component,
                            ParticleSet& electrons,
                            std::size_t& bulk_cursor,
                            std::size_t& scalar_cursor)
{
  Buffer buffer;
  component.registerData(electrons, buffer);
  bulk_cursor   = buffer.current();
  scalar_cursor = buffer.current_scalar();
  buffer.allocate();
  return buffer;
}

/// Change one stored model weight after an existing component has loaded the file.
void shiftFirstParameter(const std::filesystem::path& parameter_file)
{
  const hid_t file = H5Fopen(parameter_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const hid_t dataset = H5Dopen2(file, "/values", H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  const hid_t file_space = H5Dget_space(dataset);
  REQUIRE(file_space >= 0);
  const hsize_t start = 0;
  const hsize_t count = 1;
  REQUIRE(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, &start, nullptr, &count, nullptr) >= 0);
  const hid_t memory_space = H5Screate_simple(1, &count, nullptr);
  REQUIRE(memory_space >= 0);
  double value;
  REQUIRE(H5Dread(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, H5P_DEFAULT, &value) >= 0);
  value += 1.0e-3;
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, H5P_DEFAULT, &value) >= 0);
  H5Sclose(memory_space);
  H5Sclose(file_space);
  H5Dclose(dataset);
  H5Fclose(file);
}

} // namespace

TEST_CASE("PsiFormer walker buffer round-trip, branch, and move lifecycle",
          "[wavefunction][psiformer][buffer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeBufferElectrons(simulation_cell);
  PsiFormerWF component("pf_buffer", files.parameters.string(), files.configuration.string());

  std::size_t bulk_cursor;
  std::size_t scalar_cursor;
  Buffer buffer = makeRegisteredBuffer(component, electrons, bulk_cursor, scalar_cursor);
  CHECK(bulk_cursor > 0);
  CHECK(scalar_cursor == 17);

  // The real driver copies a newly allocated, zero-filled record before its first
  // full evaluation.  The invalid sentinel must be consumed without throwing.
  buffer.rewind();
  component.copyFromBuffer(electrons, buffer);
  CHECK(buffer.current() == bulk_cursor);
  CHECK(buffer.current_scalar() == scalar_cursor);

  electrons.G = Value(0);
  electrons.L = Value(0);
  buffer.rewind();
  const auto initial_log = component.updateBuffer(electrons, buffer, true);
  const AcceptedSnapshot initial{initial_log, electrons.G, electrons.L};
  CHECK(buffer.current() == bulk_cursor);
  CHECK(buffer.current_scalar() == scalar_cursor);
  Buffer branch_buffer = buffer;

  // A clone restores only its component contribution and adds it to existing
  // TrialWaveFunction accumulators rather than replacing them.
  ParticleSet branch_particles = makeBufferElectrons(simulation_cell);
  std::unique_ptr<WaveFunctionComponent> branch_base = component.makeClone(branch_particles);
  auto& branch_component = dynamic_cast<PsiFormerWF&>(*branch_base);
  branch_buffer.rewind();
  branch_component.copyFromBuffer(branch_particles, branch_buffer);
  CHECK(branch_buffer.current() == bulk_cursor);
  CHECK(branch_buffer.current_scalar() == scalar_cursor);
  const Value gradient_sentinel(0.125);
  const Value laplacian_sentinel(-0.375);
  branch_particles.G = gradient_sentinel;
  branch_particles.L = laplacian_sentinel;
  branch_buffer.rewind();
  checkLog(branch_component.updateBuffer(branch_particles, branch_buffer, false), initial.log);
  checkAccumulatedSpatial(branch_particles, initial, gradient_sentinel, laplacian_sentinel);

  // Rejecting a delayed proposal leaves the full accepted products live.
  constexpr int moved_electron = 1;
  const ParticleSet::SingleParticlePos rejected_displacement{0.025, -0.018, 0.011};
  branch_particles.makeMove(moved_electron, rejected_displacement);
  branch_component.ratio(branch_particles, moved_electron);
  branch_component.restore(moved_electron);
  branch_particles.rejectMove(moved_electron);
  branch_particles.G = Value(0);
  branch_particles.L = Value(0);
  branch_buffer.rewind();
  checkLog(branch_component.updateBuffer(branch_particles, branch_buffer, false), initial.log);
  checkAccumulatedSpatial(branch_particles, initial, Value(0), Value(0));

  // Accepting commits the proposal value immediately but invalidates spatial
  // products until updateBuffer refreshes the moved configuration.
  const ParticleSet::SingleParticlePos accepted_displacement{0.041, -0.027, 0.019};
  electrons.makeMove(moved_electron, accepted_displacement);
  const Value accepted_ratio = component.ratio(electrons, moved_electron);
  component.acceptMove(electrons, moved_electron, true);
  electrons.acceptMove(moved_electron);
  const PsiFormerWF::LogValue committed_log = component.get_log_value();
  CHECK(std::real(committed_log) ==
        Catch::Approx(std::real(initial.log) + std::log(std::abs(std::real(accepted_ratio))))
            .epsilon(3.0e-9)
            .margin(3.0e-9));

  PsiFormerWF moved_oracle("pf_buffer_moved", files.parameters.string(), files.configuration.string());
  const AcceptedSnapshot moved_expected = evaluateAccepted(moved_oracle, electrons);
  electrons.G = Value(0);
  electrons.L = Value(0);
  buffer.rewind();
  checkLog(component.updateBuffer(electrons, buffer, false), moved_expected.log);
  checkAccumulatedSpatial(electrons, moved_expected, Value(0), Value(0));

  // A branched copy retains the old accepted configuration even after the source
  // walker moves and rewrites its own buffer.
  ParticleSet restored_particles = makeBufferElectrons(simulation_cell);
  std::unique_ptr<WaveFunctionComponent> restored_base = component.makeClone(restored_particles);
  auto& restored_component = dynamic_cast<PsiFormerWF&>(*restored_base);
  branch_buffer.rewind();
  restored_component.copyFromBuffer(restored_particles, branch_buffer);
  restored_particles.G = Value(0);
  restored_particles.L = Value(0);
  branch_buffer.rewind();
  checkLog(restored_component.updateBuffer(restored_particles, branch_buffer, false), initial.log);
  checkAccumulatedSpatial(restored_particles, initial, Value(0), Value(0));

  // from_scratch forces a refresh while preserving the same public result.
  restored_particles.G = Value(0);
  restored_particles.L = Value(0);
  branch_buffer.rewind();
  checkLog(restored_component.updateBuffer(restored_particles, branch_buffer, true), initial.log);
  checkAccumulatedSpatial(restored_particles, initial, Value(0), Value(0));

  // Copying an old branch record onto different coordinates consumes but
  // invalidates it; a particle ratio cannot then use the stale denominator.
  std::unique_ptr<WaveFunctionComponent> stale_base = component.makeClone(electrons);
  auto& stale_component = dynamic_cast<PsiFormerWF&>(*stale_base);
  branch_buffer.rewind();
  stale_component.copyFromBuffer(electrons, branch_buffer);
  electrons.makeMove(0, ParticleSet::SingleParticlePos{0.009, -0.004, 0.006});
  CHECK_THROWS_AS(stale_component.ratio(electrons, 0), std::logic_error);
  electrons.rejectMove(0);
}

TEST_CASE("PsiFormer walker buffer rejects stale parameters, corruption, and foreign models",
          "[wavefunction][psiformer][buffer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeBufferElectrons(simulation_cell);
  PsiFormerWF component("pf_buffer_versioned",
                        files.parameters.string(),
                        files.configuration.string(),
                        true,
                        {0});

  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);

  std::size_t bulk_cursor;
  std::size_t scalar_cursor;
  Buffer buffer = makeRegisteredBuffer(component, electrons, bulk_cursor, scalar_cursor);
  electrons.G = Value(0);
  electrons.L = Value(0);
  buffer.rewind();
  component.updateBuffer(electrons, buffer, true);
  Buffer old_parameter_buffer = buffer;

  // Every integer is represented numerically as two exact uint32 limbs, never
  // by bit-casting arbitrary payloads into floating-point slots.
  REQUIRE(buffer.Scalar_ptr != nullptr);
  for (std::size_t scalar = 0; scalar < 14; ++scalar)
  {
    const double limb = buffer.Scalar_ptr[scalar];
    CHECK(limb >= 0.0);
    CHECK(limb <= static_cast<double>(std::numeric_limits<std::uint32_t>::max()));
    CHECK(limb == static_cast<double>(static_cast<std::uint32_t>(limb)));
  }

  active[0] = std::real(active[0]) + 2.5e-4;
  component.resetParametersExclusive(active);
  old_parameter_buffer.rewind();
  component.copyFromBuffer(electrons, old_parameter_buffer);
  CHECK(old_parameter_buffer.current() == bulk_cursor);
  CHECK(old_parameter_buffer.current_scalar() == scalar_cursor);
  electrons.makeMove(0, ParticleSet::SingleParticlePos{0.012, -0.007, 0.004});
  CHECK_THROWS_AS(component.ratio(electrons, 0), std::logic_error);
  electrons.rejectMove(0);

  // A normal update refreshes the stale record under the new parameter version.
  electrons.G = Value(0);
  electrons.L = Value(0);
  old_parameter_buffer.rewind();
  const auto refreshed_log = component.updateBuffer(electrons, old_parameter_buffer, false);
  ParticleSet oracle_particles = makeBufferElectrons(simulation_cell);
  std::unique_ptr<WaveFunctionComponent> oracle_base = component.makeClone(oracle_particles);
  auto& oracle_component = dynamic_cast<PsiFormerWF&>(*oracle_base);
  const AcceptedSnapshot refreshed_expected = evaluateAccepted(oracle_component, oracle_particles);
  checkLog(refreshed_log, refreshed_expected.log);
  checkAccumulatedSpatial(electrons, refreshed_expected, Value(0), Value(0));

  // A value-only record cannot masquerade as the full G/L state required by
  // updateBuffer, even when every other identity field agrees.
  Buffer insufficient_buffer = old_parameter_buffer;
  REQUIRE(insufficient_buffer.Scalar_ptr != nullptr);
  insufficient_buffer.Scalar_ptr[4] = 1.0;
  insufficient_buffer.Scalar_ptr[5] = 0.0;
  insufficient_buffer.rewind();
  component.copyFromBuffer(electrons, insufficient_buffer);
  electrons.makeMove(0, ParticleSet::SingleParticlePos{-0.008, 0.006, 0.003});
  CHECK_THROWS_AS(component.ratio(electrons, 0), std::logic_error);
  electrons.rejectMove(0);

  // A fractional integer limb is rejected before any cache data can become live.
  old_parameter_buffer.rewind();
  component.copyFromBuffer(electrons, old_parameter_buffer);

  // Only the completely zero-filled registration record is an initialization
  // sentinel; partially written metadata must be reported as corruption.
  std::size_t sentinel_bulk_cursor;
  std::size_t sentinel_scalar_cursor;
  Buffer malformed_sentinel =
      makeRegisteredBuffer(component, electrons, sentinel_bulk_cursor, sentinel_scalar_cursor);
  REQUIRE(malformed_sentinel.Scalar_ptr != nullptr);
  malformed_sentinel.Scalar_ptr[2] = 1.0;
  malformed_sentinel.rewind();
  CHECK_THROWS_WITH(component.copyFromBuffer(electrons, malformed_sentinel),
                    "PsiFormer walker buffer has a malformed zero sentinel");

  old_parameter_buffer.rewind();
  component.copyFromBuffer(electrons, old_parameter_buffer);
  Buffer corrupt_buffer = old_parameter_buffer;
  REQUIRE(corrupt_buffer.Scalar_ptr != nullptr);
  corrupt_buffer.Scalar_ptr[0] += 0.5;
  corrupt_buffer.rewind();
  CHECK_THROWS_AS(component.copyFromBuffer(electrons, corrupt_buffer), std::runtime_error);
  electrons.makeMove(0, ParticleSet::SingleParticlePos{0.004, -0.003, 0.007});
  CHECK_THROWS_AS(component.ratio(electrons, 0), std::logic_error);
  electrons.rejectMove(0);

  // The same layout and geometry with different initial weights is a distinct
  // persistent model, even when its local parameter-version counter is equal.
  shiftFirstParameter(files.parameters);
  PsiFormerWF foreign("pf_buffer_foreign", files.parameters.string(), files.configuration.string());
  ParticleSet foreign_particles = makeBufferElectrons(simulation_cell);
  evaluateAccepted(foreign, foreign_particles);
  old_parameter_buffer.rewind();
  CHECK_THROWS_WITH(foreign.copyFromBuffer(foreign_particles, old_parameter_buffer),
                    "PsiFormer walker buffer belongs to a different physical model");
  foreign_particles.makeMove(0, ParticleSet::SingleParticlePos{0.006, 0.002, -0.005});
  CHECK_THROWS_AS(foreign.ratio(foreign_particles, 0), std::logic_error);
  foreign_particles.rejectMove(0);
}

} // namespace qmcplusplus
