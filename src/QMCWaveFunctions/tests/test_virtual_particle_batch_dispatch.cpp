//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Lattice/CrystalLattice.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qmcplusplus
{
namespace
{
using Batch     = VirtualParticleBatch;
using RealType  = QMCTraits::RealType;
using ValueType = QMCTraits::ValueType;

SimulationCell makeVirtualDispatchOpenCell()
{
  Lattice lattice;
  lattice.BoxBConds          = false;
  lattice.R                  = ParticleSet::Tensor_t(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0);
  lattice.explicitly_defined = true;
  lattice.reset();
  return SimulationCell(lattice);
}

class FlattenedVirtualRatioComponent : public ConstantOrbital
{
public:
  FlattenedVirtualRatioComponent(RealType scale,
                                 bool fermionic,
                                 const void* stamp_source = nullptr,
                                 std::uint64_t stamp_version = 0)
      : scale_(scale),
        fermionic_(fermionic),
        stamp_source_(stamp_source),
        stamp_version_(stamp_version)
  {}

  std::string getClassName() const override { return "FlattenedVirtualRatioComponent"; }
  bool isFermionic() const override { return fermionic_; }

  void evaluateRatios(const VirtualParticleSet& virtual_particles, std::vector<ValueType>& ratios) override
  {
    ++scalar_call_count_;
    if (throw_in_scalar_)
      throw std::runtime_error("deliberate flattened virtual-ratio failure");
    if (resize_in_scalar_)
    {
      ratios.push_back(ValueType(-17));
      return;
    }
    if (ratios.size() != virtual_particles.getTotalNum())
      throw std::invalid_argument("test component received a mismatched scalar ratio extent");

    for (std::size_t virtual_index = 0; virtual_index < ratios.size(); ++virtual_index)
      ratios[virtual_index] = valueFor(scale_, virtual_particles.refPtcl, virtual_particles.R[virtual_index],
                                      virtual_particles.isOnSphere(), virtual_particles.refSourcePtcl);
  }

  EvaluationStamp mw_evaluateVirtualRatios(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      std::vector<ValueType>& ratios) const override
  {
    ++flattened_call_count_;
    WaveFunctionComponent::mw_evaluateVirtualRatios(wfc_list, p_list, vp_scratch_list, batch, ratios);
    if (resize_after_batch_)
      ratios.push_back(ValueType(-23));
    return stamp_source_ == nullptr ? EvaluationStamp{} : EvaluationStamp::versioned(stamp_source_, stamp_version_);
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override
  {
    return std::make_unique<FlattenedVirtualRatioComponent>(scale_, fermionic_, stamp_source_, stamp_version_);
  }

  static ValueType valueFor(RealType scale,
                            int electron,
                            const Batch::PosType& position,
                            bool on_sphere,
                            int source_center)
  {
    RealType value = scale + RealType(0.25) * (electron + 1) + position[0] + RealType(2) * position[1] +
        RealType(3) * position[2];
    if (on_sphere)
      value += RealType(0.125) * (source_center + 1);
    return ValueType(value);
  }

  void setStamp(const void* source, std::uint64_t version) noexcept
  {
    stamp_source_  = source;
    stamp_version_ = version;
  }
  void setThrowInScalar(bool value) noexcept { throw_in_scalar_ = value; }
  void setResizeInScalar(bool value) noexcept { resize_in_scalar_ = value; }
  void setResizeAfterBatch(bool value) noexcept { resize_after_batch_ = value; }
  int scalarCallCount() const noexcept { return scalar_call_count_; }
  int flattenedCallCount() const noexcept { return flattened_call_count_; }

private:
  RealType scale_;
  bool fermionic_;
  const void* stamp_source_;
  std::uint64_t stamp_version_;
  bool throw_in_scalar_            = false;
  bool resize_in_scalar_           = false;
  bool resize_after_batch_         = false;
  int scalar_call_count_           = 0;
  mutable int flattened_call_count_ = 0;
};

class VirtualRatioDispatchFixture
{
public:
  VirtualRatioDispatchFixture()
      : cell_(makeVirtualDispatchOpenCell()),
        p0_(cell_),
        p1_(cell_),
        wf0_(runtime_options_, "virtual_ratio_0", false),
        wf1_(runtime_options_, "virtual_ratio_1", false)
  {
    p0_.setName("electron_0");
    p0_.create({2});
    p0_.R[0] = {0.0, 0.1, 0.2};
    p0_.R[1] = {0.3, 0.4, 0.5};
    p0_.update();

    p1_.setName("electron_1");
    p1_.create({2});
    p1_.R[0] = {0.6, 0.7, 0.8};
    p1_.R[1] = {0.9, 1.0, 1.1};
    p1_.update();

    vp0_ = std::make_unique<VirtualParticleSet>(p0_);
    vp1_ = std::make_unique<VirtualParticleSet>(p1_);

    offsets_ = {0, 2, 3, 5};
    segments_ = {{0, 0}, {1, 1, true, 4}, {0, 1}};
    positions_ = {{0.10, 0.20, 0.30},
                  {0.40, 0.50, 0.60},
                  {0.70, 0.80, 0.90},
                  {1.00, 1.10, 1.20},
                  {1.30, 1.40, 1.50}};
    batch_ = std::make_unique<Batch>(2, offsets_, segments_, positions_);

    addComponents(wf0_, 0);
    addComponents(wf1_, 1);
  }

  RefVectorWithLeader<TrialWaveFunction> wavefunctions()
  { return RefVectorWithLeader<TrialWaveFunction>(wf0_, {wf0_, wf1_}); }

  RefVectorWithLeader<ParticleSet> particles()
  { return RefVectorWithLeader<ParticleSet>(p0_, {p0_, p1_}); }

  RefVectorWithLeader<VirtualParticleSet> scratches()
  { return RefVectorWithLeader<VirtualParticleSet>(*vp0_, {*vp0_, *vp1_}); }

  void evaluate(std::vector<ValueType>& ratios,
                std::vector<TrialWaveFunction::EvaluationStamp>& stamps,
                TrialWaveFunction::ComputeType compute_type)
  {
    auto wf_list      = wavefunctions();
    auto p_list       = particles();
    auto scratch_list = scratches();
    TrialWaveFunction::mw_evaluateVirtualRatios(wf_list, p_list, scratch_list, *batch_, ratios, stamps,
                                                 compute_type);
  }

  std::vector<ValueType> expected(std::initializer_list<std::size_t> selected_components) const
  {
    const std::array<RealType, 3> scales{2.0, 3.0, 4.0};
    std::vector<ValueType> result(batch_->size(), ValueType(1));
    for (std::size_t segment_index = 0; segment_index < batch_->segmentCount(); ++segment_index)
    {
      const Batch::Slice slice = batch_->slice(segment_index);
      for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
        for (std::size_t component : selected_components)
          result[slice.flatOffset() + local_index] *= FlattenedVirtualRatioComponent::valueFor(
              scales[component], slice.electronId(), slice.absolutePosition(local_index), slice.isOnSphere(),
              slice.sourceCenterId());
    }
    return result;
  }

  int totalScalarCalls() const noexcept
  {
    int calls = 0;
    for (const auto& walker_components : components_)
      for (const FlattenedVirtualRatioComponent* component : walker_components)
        calls += component->scalarCallCount();
    return calls;
  }

  SimulationCell cell_;
  ParticleSet p0_;
  ParticleSet p1_;
  std::unique_ptr<VirtualParticleSet> vp0_;
  std::unique_ptr<VirtualParticleSet> vp1_;
  RuntimeOptions runtime_options_;
  TrialWaveFunction wf0_;
  TrialWaveFunction wf1_;
  int stamp_source_a_ = 0;
  int stamp_source_b_ = 0;
  int sentinel_source_ = 0;
  std::vector<std::size_t> offsets_;
  std::vector<Batch::Segment> segments_;
  std::vector<Batch::PosType> positions_;
  std::unique_ptr<Batch> batch_;
  std::array<std::array<FlattenedVirtualRatioComponent*, 3>, 2> components_{};

private:
  void addComponents(TrialWaveFunction& wavefunction, std::size_t walker)
  {
    const std::array<RealType, 3> scales{2.0, 3.0, 4.0};
    const std::array<bool, 3> fermionic{true, false, false};
    const std::array<const void*, 3> sources{&stamp_source_a_, nullptr, &stamp_source_b_};
    const std::array<std::uint64_t, 3> versions{7, 0, 11};
    for (std::size_t component = 0; component < components_[walker].size(); ++component)
    {
      auto value = std::make_unique<FlattenedVirtualRatioComponent>(scales[component], fermionic[component],
                                                                    sources[component], versions[component]);
      components_[walker][component] = value.get();
      wavefunction.addComponent(std::move(value));
    }
  }
};

void checkValues(const std::vector<ValueType>& actual, const std::vector<ValueType>& expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    CHECK(actual[index] == ValueApprox(expected[index]));
}
} // namespace

TEST_CASE("Flattened virtual ratios form selected complete products", "[wavefunction][virtual_batch]")
{
  VirtualRatioDispatchFixture fixture;
  using Stamp = TrialWaveFunction::EvaluationStamp;

  SECTION("all components and process-local stamps")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    std::vector<Stamp> stamps{Stamp::versioned(&fixture.sentinel_source_, 99)};
    fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL);

    checkValues(ratios, fixture.expected({0, 1, 2}));
    REQUIRE(stamps.size() == 2);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_b_, 11));
    for (std::size_t component = 0; component < 3; ++component)
    {
      CHECK(fixture.components_[0][component]->scalarCallCount() == 2);
      CHECK(fixture.components_[1][component]->scalarCallCount() == 1);
      CHECK(fixture.components_[0][component]->flattenedCallCount() == 1);
      CHECK(fixture.components_[1][component]->flattenedCallCount() == 0);
    }

    CHECK(fixture.vp0_->refPtcl == 1);
    CHECK_FALSE(fixture.vp0_->isOnSphere());
    CHECK(fixture.vp0_->R[0] == fixture.positions_[3]);
    CHECK(fixture.vp0_->R[1] == fixture.positions_[4]);
    CHECK(fixture.vp1_->refPtcl == 1);
    CHECK(fixture.vp1_->isOnSphere());
    CHECK(fixture.vp1_->refSourcePtcl == 4);
    CHECK(fixture.vp1_->R[0] == fixture.positions_[2]);
  }

  SECTION("fermionic only")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    std::vector<Stamp> stamps;
    fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::FERMIONIC);

    checkValues(ratios, fixture.expected({0}));
    REQUIRE(stamps.size() == 1);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(fixture.components_[0][0]->scalarCallCount() == 2);
    CHECK(fixture.components_[1][0]->scalarCallCount() == 1);
    CHECK(fixture.components_[0][1]->scalarCallCount() == 0);
    CHECK(fixture.components_[0][2]->scalarCallCount() == 0);
  }

  SECTION("nonfermionic only")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    std::vector<Stamp> stamps;
    fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::NONFERMIONIC);

    checkValues(ratios, fixture.expected({1, 2}));
    REQUIRE(stamps.size() == 1);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_b_, 11));
    CHECK(fixture.components_[0][0]->scalarCallCount() == 0);
    CHECK(fixture.components_[0][1]->scalarCallCount() == 2);
    CHECK(fixture.components_[0][2]->scalarCallCount() == 2);
  }

  SECTION("zero selected components form the identity product")
  {
    TrialWaveFunction empty0(fixture.runtime_options_, "empty_0", false);
    TrialWaveFunction empty1(fixture.runtime_options_, "empty_1", false);
    RefVectorWithLeader<TrialWaveFunction> empty_wavefunctions(empty0, {empty0, empty1});
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    std::vector<Stamp> stamps{Stamp::versioned(&fixture.sentinel_source_, 99)};

    TrialWaveFunction::mw_evaluateVirtualRatios(empty_wavefunctions, p_list, scratch_list, *fixture.batch_, ratios,
                                                 stamps);
    CHECK(ratios == std::vector<ValueType>(fixture.batch_->size(), ValueType(1)));
    CHECK(stamps.empty());
  }
}

TEST_CASE("Flattened virtual ratios publish products and stamps atomically", "[wavefunction][virtual_batch]")
{
  VirtualRatioDispatchFixture fixture;
  using Stamp = TrialWaveFunction::EvaluationStamp;
  const std::vector<ValueType> ratio_sentinel(fixture.batch_->size(), ValueType(-31));
  const std::vector<Stamp> stamp_sentinel{Stamp::versioned(&fixture.sentinel_source_, 101)};

  SECTION("later scalar component throws")
  {
    fixture.components_[1][2]->setThrowInScalar(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL), std::runtime_error);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.components_[0][0]->scalarCallCount() == 2);
    CHECK(fixture.components_[0][1]->scalarCallCount() == 2);
    CHECK(fixture.components_[0][2]->scalarCallCount() == 1);
  }

  SECTION("component violates flat output extent")
  {
    fixture.components_[0][2]->setResizeAfterBatch(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL), std::runtime_error);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
  }

  SECTION("component violates serialized segment extent")
  {
    fixture.components_[1][2]->setResizeInScalar(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL), std::runtime_error);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
  }

  SECTION("one shared identity reports conflicting versions")
  {
    fixture.components_[0][2]->setStamp(&fixture.stamp_source_a_, 8);
    fixture.components_[1][2]->setStamp(&fixture.stamp_source_a_, 8);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL), std::runtime_error);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
  }

  SECTION("duplicate identity at one version remains component ordered")
  {
    fixture.components_[0][2]->setStamp(&fixture.stamp_source_a_, 7);
    fixture.components_[1][2]->setStamp(&fixture.stamp_source_a_, 7);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL);
    checkValues(ratios, fixture.expected({0, 1, 2}));
    REQUIRE(stamps.size() == 2);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_a_, 7));
  }
}

TEST_CASE("Flattened virtual-ratio malformed inputs preserve caller outputs", "[wavefunction][virtual_batch]")
{
  VirtualRatioDispatchFixture fixture;
  using Stamp = TrialWaveFunction::EvaluationStamp;
  const std::vector<ValueType> ratio_sentinel(fixture.batch_->size(), ValueType(-37));
  const std::vector<Stamp> stamp_sentinel{Stamp::versioned(&fixture.sentinel_source_, 103)};

  SECTION("wrong flat output extent")
  {
    std::vector<ValueType> ratios(fixture.batch_->size() - 1, ValueType(-37));
    const std::vector<ValueType> original_ratios = ratios;
    std::vector<Stamp> stamps                    = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, TrialWaveFunction::ComputeType::ALL),
                      std::invalid_argument);
    CHECK(ratios == original_ratios);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("duplicate mutable scratch")
  {
    auto wf_list = fixture.wavefunctions();
    auto p_list  = fixture.particles();
    RefVectorWithLeader<VirtualParticleSet> duplicate_scratch(*fixture.vp0_, {*fixture.vp0_, *fixture.vp0_});
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          wf_list, p_list, duplicate_scratch, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
    CHECK(fixture.vp0_->getTotalNum() == 0);
    CHECK(fixture.vp1_->getTotalNum() == 0);
  }

  SECTION("duplicate reference walker")
  {
    auto wf_list = fixture.wavefunctions();
    RefVectorWithLeader<ParticleSet> duplicate_reference(fixture.p0_, {fixture.p0_, fixture.p0_});
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          wf_list, duplicate_reference, scratch_list, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
    CHECK(fixture.vp0_->getTotalNum() == 0);
    CHECK(fixture.vp1_->getTotalNum() == 0);
  }

  SECTION("scratch aliases a reference walker")
  {
    const std::vector<Batch::PosType> initial_positions{{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}};
    fixture.vp0_->makeMovesAbsolute(fixture.p0_, 0, initial_positions);
    const std::vector<Batch::PosType> saved_positions(fixture.vp0_->R.begin(), fixture.vp0_->R.end());

    auto wf_list = fixture.wavefunctions();
    RefVectorWithLeader<ParticleSet> aliased_reference(*fixture.vp0_, {*fixture.vp0_, fixture.p1_});
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          wf_list, aliased_reference, scratch_list, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
    CHECK(std::vector<Batch::PosType>(fixture.vp0_->R.begin(), fixture.vp0_->R.end()) == saved_positions);
  }

  SECTION("duplicate wavefunction clone")
  {
    RefVectorWithLeader<TrialWaveFunction> duplicate_wf(fixture.wf0_, {fixture.wf0_, fixture.wf0_});
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          duplicate_wf, p_list, scratch_list, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("clone component count differs")
  {
    TrialWaveFunction incomplete(fixture.runtime_options_, "incomplete", false);
    incomplete.addComponent(std::make_unique<FlattenedVirtualRatioComponent>(2.0, true));
    incomplete.addComponent(std::make_unique<FlattenedVirtualRatioComponent>(3.0, false));
    RefVectorWithLeader<TrialWaveFunction> malformed_wf(fixture.wf0_, {fixture.wf0_, incomplete});
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          malformed_wf, p_list, scratch_list, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("clone component filtering topology differs")
  {
    TrialWaveFunction malformed(fixture.runtime_options_, "malformed", false);
    malformed.addComponent(std::make_unique<FlattenedVirtualRatioComponent>(2.0, false));
    malformed.addComponent(std::make_unique<FlattenedVirtualRatioComponent>(3.0, false));
    malformed.addComponent(std::make_unique<FlattenedVirtualRatioComponent>(4.0, false));
    RefVectorWithLeader<TrialWaveFunction> malformed_wf(fixture.wf0_, {fixture.wf0_, malformed});
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualRatios(
                          malformed_wf, p_list, scratch_list, *fixture.batch_, ratios, stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("invalid compute selection")
  {
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<Stamp> stamps     = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluate(ratios, stamps, static_cast<TrialWaveFunction::ComputeType>(99)),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }
}

TEST_CASE("Evaluation stamps are opaque process-local equality tokens", "[wavefunction][virtual_batch]")
{
  using Stamp = TrialWaveFunction::EvaluationStamp;
  int source_a = 0;
  int source_b = 0;
  const Stamp unversioned;
  const Stamp a7 = Stamp::versioned(&source_a, 7);

  CHECK_FALSE(unversioned.isVersioned());
  CHECK(a7.isVersioned());
  CHECK(a7 == Stamp::versioned(&source_a, 7));
  CHECK(a7 != Stamp::versioned(&source_a, 8));
  CHECK(a7 != Stamp::versioned(&source_b, 7));
  REQUIRE_THROWS_AS(Stamp::versioned(nullptr, 7), std::invalid_argument);
}
} // namespace qmcplusplus
