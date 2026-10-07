//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_precision_execution.cpp
 * @brief CPU oracles for mixed-precision storage, conversion, and publication.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionExecution.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Simulate the one-allocation preparation boundary used by a device resource owner.
class RecordingPrecisionArena
{
public:
  /// Allocate one canonical arena before execution is sealed.
  void prepare(PsiFormerDeviceArenaLayout arena)
  {
    if (sealed_)
      throw std::logic_error("cannot allocate after sealing a precision arena");
    arena_ = std::move(arena);
    ++allocation_count_;
  }

  /// Seal the resource owner against further allocations.
  void seal() noexcept { sealed_ = true; }

  /// Record conversion only when every required prepared slice exists.
  void convert(const PsiFormerParameterConversionTile& tile)
  {
    if (!sealed_ ||
        arena_.find(PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING) == nullptr ||
        arena_.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1) == nullptr ||
        arena_.find(PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS) == nullptr)
      throw std::logic_error("precision conversion lacks a sealed arena");
    if (tile.count != tile.launch.item_count)
      throw std::invalid_argument("precision conversion tile is inconsistent");
    converted_items_ += tile.count;
  }

  std::size_t allocationCount() const noexcept { return allocation_count_; }
  std::size_t convertedItems() const noexcept { return converted_items_; }

private:
  PsiFormerDeviceArenaLayout arena_;
  std::size_t allocation_count_ = 0;
  std::size_t converted_items_  = 0;
  bool sealed_ = false;
};

/// Inspect a binary32 representation without fast-math zero/subnormal assumptions.
std::uint32_t floatBits(float value) noexcept
{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

} // namespace

TEST_CASE("PsiFormer precision execution accounts two slot pairs exactly",
          "[wavefunction][psiformer][precision_execution]")
{
  const PsiFormerPrecisionExecutionPlan plan = makePsiFormerPrecisionExecutionPlan(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerBackendMathMode::FP32_STRICT,
      /*parameter_count=*/10, /*cast_tile_parameters=*/4, /*block_size=*/3,
      /*device_layout_fingerprint=*/77, /*source_version=*/5,
      /*full_precision_retry_available=*/true);

  CHECK(plan.master_slot_bytes == 80);
  CHECK(plan.compute_slot_bytes == 40);
  CHECK(plan.conversion_workspace_bytes == 16);
  CHECK(plan.diagnostic_bytes == sizeof(PsiFormerParameterConversionDiagnostics));
  CHECK(plan.arena.slices.size() == 6);
  REQUIRE(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_PARAMETERS));
  REQUIRE(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING));
  REQUIRE(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0));
  REQUIRE(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1));
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_PARAMETERS)->offset == 0);
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_PARAMETERS_STAGING)->offset == 256);
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_0)->offset == 512);
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::MODEL_COMPUTE_PARAMETERS_1)->offset == 768);
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::PRECISION_CONVERSION_WORKSPACE)->offset == 1024);
  CHECK(plan.arena.find(PsiFormerDeviceArenaRegion::NUMERICAL_DIAGNOSTICS)->offset == 1040);
  CHECK(plan.arena.total_bytes == 1040 + sizeof(PsiFormerParameterConversionDiagnostics));

  REQUIRE(plan.conversion.tiles.size() == 3);
  CHECK(plan.conversion.tiles[0].count == 4);
  CHECK(plan.conversion.tiles[1].source_begin == 4);
  CHECK(plan.conversion.tiles[2].source_begin == 8);
  CHECK(plan.conversion.tiles[2].count == 2);
  CHECK(plan.fingerprint != 0);

  RecordingPrecisionArena recording;
  recording.prepare(plan.arena);
  recording.seal();
  for (const PsiFormerParameterConversionTile& tile : plan.conversion.tiles)
    recording.convert(tile);
  CHECK(recording.allocationCount() == 1);
  CHECK(recording.convertedItems() == 10);
  CHECK_THROWS_AS(recording.prepare(plan.arena), std::logic_error);
}

TEST_CASE("PsiFormer precision execution validates policy and extent boundaries",
          "[wavefunction][psiformer][precision_execution]")
{
  CHECK_THROWS_AS(makePsiFormerPrecisionExecutionPlan(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                      PsiFormerBackendMathMode::FP64_STRICT, 4, 2, 32, 1, 1, true),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionExecutionPlan(
                      PsiFormerPrecisionPolicy::FP64_REFERENCE,
                      PsiFormerBackendMathMode::FP64_STRICT, 4, 1, 32, 1, 1, true),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionExecutionPlan(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                      PsiFormerBackendMathMode::FP32_STRICT, 4, 5, 32, 1, 1, true),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerPrecisionExecutionPlan(
                      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
                      PsiFormerBackendMathMode::FP32_STRICT,
                      std::numeric_limits<std::size_t>::max(), 1, 32, 1, 1, true),
                  std::overflow_error);

  const PsiFormerPrecisionExecutionPlan empty = makePsiFormerPrecisionExecutionPlan(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerBackendMathMode::FP32_STRICT, 0, 0, 32, 9, 0, true);
  CHECK(empty.conversion.tiles.empty());
  CHECK(empty.master_slot_bytes == 0);
  CHECK(empty.compute_slot_bytes == 0);
  CHECK(empty.fingerprint != 0);

  const PsiFormerPrecisionExecutionPlan changed_version = makePsiFormerPrecisionExecutionPlan(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
      PsiFormerBackendMathMode::FP32_STRICT, 0, 0, 32, 9, 1, true);
  CHECK(changed_version.fingerprint != empty.fingerprint);
}

TEST_CASE("PsiFormer FP64 to FP32 reference conversion classifies edge values",
          "[wavefunction][psiformer][precision_execution]")
{
  const double fp32_max = static_cast<double>(std::numeric_limits<float>::max());
  const double fp32_min = static_cast<double>(std::numeric_limits<float>::min());
  const std::vector<double> source{
      0.0,
      -0.0,
      17.0,
      1.0 + std::ldexp(1.0, -24),
      fp32_min,
      static_cast<double>(std::numeric_limits<float>::denorm_min()),
      fp32_min / 2.0,
      std::numeric_limits<double>::denorm_min(),
      fp32_max,
      std::nextafter(fp32_max, std::numeric_limits<double>::infinity()),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  std::vector<float> destination(source.size(), 123.0F);
  PsiFormerParameterConversionDiagnostics diagnostics;
  convertPsiFormerFp64ToFp32Reference(source.data(), source.size(), destination.data(),
                                      diagnostics);

  CHECK(destination[0] == 0.0F);
  CHECK(floatBits(destination[0]) == UINT32_C(0x00000000));
  CHECK(destination[1] == 0.0F);
  CHECK(floatBits(destination[1]) == UINT32_C(0x80000000));
  CHECK(destination[2] == 17.0F);
  CHECK(destination[3] == 1.0F); // round-to-nearest, ties-to-even on IEEE hosts
  CHECK(destination[4] == std::numeric_limits<float>::min());
  CHECK(floatBits(destination[5]) == UINT32_C(0x00000001));
  CHECK(floatBits(destination[6]) == UINT32_C(0x00400000));
  CHECK(floatBits(destination[7]) == UINT32_C(0x00000000));
  CHECK(destination[8] == std::numeric_limits<float>::max());
  CHECK(destination[9] == std::numeric_limits<float>::max());
  CHECK(destination[10] == 0.0F);
  CHECK(destination[11] == 0.0F);
  CHECK(destination[12] == 0.0F);
  CHECK(diagnostics.nonfinite_input_count == 3);
  CHECK(diagnostics.overflow_count == 1);
  CHECK(diagnostics.subnormal_output_count == 2);
  CHECK(diagnostics.underflow_to_zero_count == 1);
  CHECK(diagnostics.maximum_absolute_error > 0.0);
  CHECK(diagnostics.maximum_relative_error > 0.0);
  CHECK(diagnostics.hasPublicationHazard());

  CHECK_NOTHROW(convertPsiFormerFp64ToFp32Reference(nullptr, 0, nullptr, diagnostics));
  CHECK_THROWS_AS(convertPsiFormerFp64ToFp32Reference(nullptr, 1, destination.data(), diagnostics),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer mixed publication is failure atomic and reuses inactive slots",
          "[wavefunction][psiformer][precision_execution]")
{
  PsiFormerMixedPublicationState publication(
      PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE, 4, 40, 0);
  publication.begin(5, 50);
  REQUIRE(publication.pendingSlot());
  CHECK(*publication.pendingSlot() == 1);
  CHECK_THROWS_AS(publication.markMasterCopied(6), std::logic_error);
  publication.markMasterCopied(5);
  CHECK_THROWS_AS(publication.publish(5), std::logic_error);
  publication.cancel(5);
  CHECK(publication.activeVersion() == 4);
  CHECK(publication.activeSlot() == 0);
  CHECK_FALSE(publication.publicationPending());

  publication.begin(5, 50);
  publication.markMasterCopied(5);
  publication.markComputeConversionComplete(5);
  PsiFormerParameterConversionDiagnostics hazard;
  hazard.overflow_count = 1;
  CHECK_FALSE(publication.acceptDiagnostics(5, hazard));
  CHECK(publication.activeVersion() == 4);
  CHECK(publication.activeSlot() == 0);

  publication.begin(5, 50);
  publication.markMasterCopied(5);
  publication.markComputeConversionComplete(5);
  PsiFormerParameterConversionDiagnostics benign;
  benign.subnormal_output_count = 2;
  CHECK(publication.acceptDiagnostics(5, benign));
  publication.publish(5);
  CHECK(publication.isReady(5, 50));
  CHECK(publication.activeSlot() == 1);

  publication.begin(6, 60);
  REQUIRE(publication.pendingSlot());
  CHECK(*publication.pendingSlot() == 0);
  publication.markMasterCopied(6);
  publication.markComputeConversionComplete(6);
  CHECK(publication.acceptDiagnostics(6, {}));
  publication.publish(6);
  CHECK(publication.activeSlot() == 0);
  CHECK(publication.isReady(6, 60));
  CHECK_THROWS_AS(publication.begin(6, 61), std::invalid_argument);
  CHECK_THROWS_AS(publication.publish(6), std::logic_error);
}

TEST_CASE("PsiFormer FP64 publication bypasses only the conversion milestone",
          "[wavefunction][psiformer][precision_execution]")
{
  PsiFormerMixedPublicationState publication(
      PsiFormerPrecisionPolicy::FP64_REFERENCE, 1, 10, 1);
  publication.begin(2, 20);
  publication.markMasterCopied(2);
  CHECK_THROWS_AS(publication.markComputeConversionComplete(2), std::logic_error);
  CHECK(publication.acceptDiagnostics(2, {}));
  publication.publish(2);
  CHECK(publication.activeSlot() == 0);
  CHECK(publication.isReady(2, 20));

  PsiFormerMixedPublicationState maximum(
      PsiFormerPrecisionPolicy::FP64_REFERENCE,
      std::numeric_limits<std::size_t>::max(), 1, 0);
  CHECK_THROWS_AS(maximum.begin(0, 2), std::invalid_argument);
}

} // namespace qmcplusplus::psiformer
