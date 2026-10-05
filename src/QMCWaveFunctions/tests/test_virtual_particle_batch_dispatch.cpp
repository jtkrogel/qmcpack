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

  void evaluateDerivRatios(const VirtualParticleSet& virtual_particles,
                           const OptVariables& optvars,
                           std::vector<ValueType>& ratios,
                           Matrix<ValueType>& derivative_ratios) override
  {
    ++derivative_scalar_call_count_;
    if (throw_in_derivative_)
      throw std::runtime_error("deliberate flattened weighted-derivative failure");
    if (ratios.size() != virtual_particles.getTotalNum() || derivative_ratios.rows() != ratios.size())
      throw std::invalid_argument("test component received a mismatched scalar derivative extent");

    for (std::size_t virtual_index = 0; virtual_index < ratios.size(); ++virtual_index)
    {
      ratios[virtual_index] = valueFor(scale_, virtual_particles.refPtcl, virtual_particles.R[virtual_index],
                                       virtual_particles.isOnSphere(), virtual_particles.refSourcePtcl);
      for (std::size_t local_index = 0; local_index < optvars.size(); ++local_index)
      {
        const int global_index = optvars.where(local_index);
        if (global_index >= 0)
        {
          if (static_cast<std::size_t>(global_index) >= derivative_ratios.cols())
            throw std::out_of_range("test component received a short mapped derivative destination");
          derivative_ratios(virtual_index, global_index) =
              derivativeFor(scale_, virtual_particles.refPtcl, virtual_particles.R[virtual_index], global_index);
        }
      }
    }
  }

  EvaluationStamp mw_evaluateVirtualDerivRatiosWeighted(
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
      const RefVectorWithLeader<ParticleSet>& p_list,
      const RefVectorWithLeader<VirtualParticleSet>& vp_scratch_list,
      const VirtualParticleBatch& batch,
      const OptVariables& optvars,
      const std::vector<ValueType>& total_weights,
      const std::vector<ParameterDerivativeView>& weighted_derivatives) const override
  {
    ++flattened_weighted_call_count_;
    WaveFunctionComponent::mw_evaluateVirtualDerivRatiosWeighted(
        wfc_list, p_list, vp_scratch_list, batch, optvars, total_weights, weighted_derivatives);
    if (throw_after_weighted_)
      throw std::runtime_error("deliberate post-reduction weighted-derivative failure");

    const void* source = weighted_stamp_overridden_ ? weighted_stamp_source_ : stamp_source_;
    const std::uint64_t version = weighted_stamp_overridden_ ? weighted_stamp_version_ : stamp_version_;
    return source == nullptr ? EvaluationStamp{} : EvaluationStamp::versioned(source, version);
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

  static ValueType derivativeFor(RealType scale,
                                 int electron,
                                 const Batch::PosType& position,
                                 std::size_t parameter)
  {
    const RealType coordinate = position[parameter % OHMMS_DIM];
    return ValueType(RealType(0.01) * scale * (parameter + 1) + RealType(0.02) * (electron + 1) +
                     RealType(0.03) * coordinate);
  }

  void setStamp(const void* source, std::uint64_t version) noexcept
  {
    stamp_source_  = source;
    stamp_version_ = version;
  }
  void setThrowInScalar(bool value) noexcept { throw_in_scalar_ = value; }
  void setResizeInScalar(bool value) noexcept { resize_in_scalar_ = value; }
  void setResizeAfterBatch(bool value) noexcept { resize_after_batch_ = value; }
  void setThrowInDerivative(bool value) noexcept { throw_in_derivative_ = value; }
  void setThrowAfterWeighted(bool value) noexcept { throw_after_weighted_ = value; }
  void setWeightedStamp(const void* source, std::uint64_t version) noexcept
  {
    weighted_stamp_source_     = source;
    weighted_stamp_version_    = version;
    weighted_stamp_overridden_ = true;
  }
  int scalarCallCount() const noexcept { return scalar_call_count_; }
  int flattenedCallCount() const noexcept { return flattened_call_count_; }
  int derivativeScalarCallCount() const noexcept { return derivative_scalar_call_count_; }
  int flattenedWeightedCallCount() const noexcept { return flattened_weighted_call_count_; }

private:
  RealType scale_;
  bool fermionic_;
  const void* stamp_source_;
  std::uint64_t stamp_version_;
  bool throw_in_scalar_                            = false;
  bool resize_in_scalar_                           = false;
  bool resize_after_batch_                         = false;
  int scalar_call_count_                           = 0;
  mutable int flattened_call_count_                = 0;
  bool throw_in_derivative_                        = false;
  bool throw_after_weighted_                       = false;
  bool weighted_stamp_overridden_                  = false;
  const void* weighted_stamp_source_                = nullptr;
  std::uint64_t weighted_stamp_version_             = 0;
  int derivative_scalar_call_count_                 = 0;
  mutable int flattened_weighted_call_count_        = 0;
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

  void evaluateWeighted(const OptVariables& optvars,
                        const std::vector<ValueType>& bare_weights,
                        std::vector<ValueType>& ratios,
                        std::vector<std::vector<ValueType>>& derivatives,
                        std::vector<TrialWaveFunction::EvaluationStamp>& stamps,
                        TrialWaveFunction::ComputeType compute_type)
  {
    auto wf_list      = wavefunctions();
    auto p_list       = particles();
    auto scratch_list = scratches();
    std::vector<TrialWaveFunction::ParameterDerivativeView> derivative_views;
    derivative_views.reserve(derivatives.size());
    for (std::vector<ValueType>& row : derivatives)
      derivative_views.push_back({row.data(), row.size()});
    TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
        wf_list, p_list, scratch_list, *batch_, optvars, bare_weights, ratios, derivative_views, stamps,
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

  std::vector<std::vector<ValueType>> expectedWeighted(
      const OptVariables& optvars,
      const std::vector<ValueType>& bare_weights,
      std::initializer_list<std::size_t> selected_components,
      const std::vector<std::vector<ValueType>>& initial) const
  {
    std::vector<std::vector<ValueType>> result = initial;
    const std::array<RealType, 3> scales{2.0, 3.0, 4.0};
    const std::vector<ValueType> complete_ratios = expected(selected_components);
    for (std::size_t segment_index = 0; segment_index < batch_->segmentCount(); ++segment_index)
    {
      const Batch::Slice slice = batch_->slice(segment_index);
      for (std::size_t local_index = 0; local_index < slice.size(); ++local_index)
      {
        const std::size_t virtual_index = slice.flatOffset() + local_index;
        const ValueType total_weight    = bare_weights[virtual_index] * complete_ratios[virtual_index];
        for (std::size_t component : selected_components)
          for (std::size_t parameter = 0; parameter < optvars.size(); ++parameter)
          {
            const int global_index = optvars.where(parameter);
            if (global_index >= 0)
              result[slice.walkerId()][global_index] += total_weight * FlattenedVirtualRatioComponent::derivativeFor(
                  scales[component], slice.electronId(), slice.absolutePosition(local_index), global_index);
          }
      }
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

  int totalDerivativeScalarCalls() const noexcept
  {
    int calls = 0;
    for (const auto& walker_components : components_)
      for (const FlattenedVirtualRatioComponent* component : walker_components)
        calls += component->derivativeScalarCallCount();
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

void checkRows(const std::vector<std::vector<ValueType>>& actual,
               const std::vector<std::vector<ValueType>>& expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t walker = 0; walker < actual.size(); ++walker)
    checkValues(actual[walker], expected[walker]);
}

OptVariables makeDenseOptVariables()
{
  OptVariables optvars;
  optvars.insert("dispatch_p0", 0.0);
  optvars.insert("dispatch_p1", 0.0);
  optvars.insert("dispatch_p2", 0.0);
  optvars.resetIndex();
  return optvars;
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

TEST_CASE("Flattened weighted virtual derivatives use complete selected products",
          "[wavefunction][virtual_batch][weighted]")
{
  VirtualRatioDispatchFixture fixture;
  const OptVariables optvars = makeDenseOptVariables();
  const std::vector<ValueType> bare_weights{ValueType(0.25), ValueType(-0.40), ValueType(0.15),
                                             ValueType(0.30), ValueType(-0.20)};
  using Stamp = TrialWaveFunction::EvaluationStamp;

  SECTION("ragged all-component reduction preserves nonzero destinations")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    const std::vector<std::vector<ValueType>> initial{{ValueType(1.0), ValueType(-2.0), ValueType(3.0)},
                                                       {ValueType(-4.0), ValueType(5.0), ValueType(-6.0)}};
    std::vector<std::vector<ValueType>> derivatives = initial;
    std::vector<Stamp> stamps{Stamp::versioned(&fixture.sentinel_source_, 107)};

    fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::ALL);

    checkValues(ratios, fixture.expected({0, 1, 2}));
    checkRows(derivatives, fixture.expectedWeighted(optvars, bare_weights, {0, 1, 2}, initial));
    REQUIRE(stamps.size() == 2);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_b_, 11));
    CHECK(fixture.totalDerivativeScalarCalls() == 9);
    for (std::size_t component = 0; component < 3; ++component)
      CHECK(fixture.components_[0][component]->flattenedWeightedCallCount() == 1);
  }

  SECTION("fermionic filtering is shared by value and weighted phases")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    const std::vector<std::vector<ValueType>> initial(2, std::vector<ValueType>(3, ValueType(0.75)));
    std::vector<std::vector<ValueType>> derivatives = initial;
    std::vector<Stamp> stamps;

    fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::FERMIONIC);

    checkValues(ratios, fixture.expected({0}));
    checkRows(derivatives, fixture.expectedWeighted(optvars, bare_weights, {0}, initial));
    REQUIRE(stamps.size() == 1);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(fixture.totalDerivativeScalarCalls() == 3);
    CHECK(fixture.components_[0][0]->flattenedWeightedCallCount() == 1);
    CHECK(fixture.components_[0][1]->flattenedWeightedCallCount() == 0);
    CHECK(fixture.components_[0][2]->flattenedWeightedCallCount() == 0);
  }

  SECTION("nonfermionic filtering includes both selected components")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    const std::vector<std::vector<ValueType>> initial(2, std::vector<ValueType>(3, ValueType(-0.5)));
    std::vector<std::vector<ValueType>> derivatives = initial;
    std::vector<Stamp> stamps;

    fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::NONFERMIONIC);

    checkValues(ratios, fixture.expected({1, 2}));
    checkRows(derivatives, fixture.expectedWeighted(optvars, bare_weights, {1, 2}, initial));
    REQUIRE(stamps.size() == 1);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_b_, 11));
    CHECK(fixture.totalDerivativeScalarCalls() == 6);
    CHECK(fixture.components_[0][0]->flattenedWeightedCallCount() == 0);
    CHECK(fixture.components_[0][1]->flattenedWeightedCallCount() == 1);
    CHECK(fixture.components_[0][2]->flattenedWeightedCallCount() == 1);
  }

  SECTION("zero active parameters still publish ratios and stamps")
  {
    const OptVariables no_parameters;
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    std::vector<std::vector<ValueType>> derivatives(2);
    std::vector<Stamp> stamps;

    fixture.evaluateWeighted(no_parameters, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::ALL);

    checkValues(ratios, fixture.expected({0, 1, 2}));
    CHECK(derivatives[0].empty());
    CHECK(derivatives[1].empty());
    REQUIRE(stamps.size() == 2);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_b_, 11));
  }

  SECTION("oversized derivative rows preserve trailing entries")
  {
    std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
    const std::vector<std::vector<ValueType>> initial(2, std::vector<ValueType>(5, ValueType(13)));
    std::vector<std::vector<ValueType>> derivatives = initial;
    std::vector<Stamp> stamps;

    fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::ALL);

    checkValues(ratios, fixture.expected({0, 1, 2}));
    checkRows(derivatives, fixture.expectedWeighted(optvars, bare_weights, {0, 1, 2}, initial));
    for (const std::vector<ValueType>& row : derivatives)
    {
      CHECK(row[3] == ValueApprox(ValueType(13)));
      CHECK(row[4] == ValueApprox(ValueType(13)));
    }
  }
}

TEST_CASE("Flattened weighted virtual outputs commit atomically", "[wavefunction][virtual_batch][weighted]")
{
  VirtualRatioDispatchFixture fixture;
  const OptVariables optvars = makeDenseOptVariables();
  const std::vector<ValueType> bare_weights{ValueType(0.25), ValueType(-0.40), ValueType(0.15),
                                             ValueType(0.30), ValueType(-0.20)};
  const std::vector<ValueType> ratio_sentinel(fixture.batch_->size(), ValueType(-41));
  const std::vector<std::vector<ValueType>> derivative_sentinel{
      {ValueType(1), ValueType(2), ValueType(3)}, {ValueType(4), ValueType(5), ValueType(6)}};
  using Stamp = TrialWaveFunction::EvaluationStamp;
  const std::vector<Stamp> stamp_sentinel{Stamp::versioned(&fixture.sentinel_source_, 109)};

  const auto require_unchanged = [&](const std::vector<ValueType>& ratios,
                                     const std::vector<std::vector<ValueType>>& derivatives,
                                     const std::vector<Stamp>& stamps) {
    CHECK(ratios == ratio_sentinel);
    CHECK(derivatives == derivative_sentinel);
    CHECK(stamps == stamp_sentinel);
  };

  SECTION("late value exception")
  {
    fixture.components_[1][2]->setThrowInScalar(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::runtime_error);
    require_unchanged(ratios, derivatives, stamps);
    CHECK(fixture.totalDerivativeScalarCalls() == 0);
  }

  SECTION("late compatibility derivative exception")
  {
    fixture.components_[1][2]->setThrowInDerivative(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::runtime_error);
    require_unchanged(ratios, derivatives, stamps);
    CHECK(fixture.totalDerivativeScalarCalls() > 0);
  }

  SECTION("component throws after reducing its complete crowd")
  {
    fixture.components_[0][2]->setThrowAfterWeighted(true);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::runtime_error);
    require_unchanged(ratios, derivatives, stamps);
  }

  SECTION("value and weighted phases report different versions")
  {
    fixture.components_[0][2]->setWeightedStamp(&fixture.stamp_source_b_, 12);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::runtime_error);
    require_unchanged(ratios, derivatives, stamps);
  }

  SECTION("one shared value identity reports conflicting versions")
  {
    fixture.components_[0][2]->setStamp(&fixture.stamp_source_a_, 8);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::runtime_error);
    require_unchanged(ratios, derivatives, stamps);
    CHECK(fixture.totalDerivativeScalarCalls() == 0);
  }

  SECTION("duplicate identity at one version remains component ordered")
  {
    fixture.components_[0][2]->setStamp(&fixture.stamp_source_a_, 7);
    fixture.components_[0][2]->setWeightedStamp(&fixture.stamp_source_a_, 7);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives = derivative_sentinel;
    std::vector<Stamp> stamps = stamp_sentinel;
    fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                             TrialWaveFunction::ComputeType::ALL);
    checkValues(ratios, fixture.expected({0, 1, 2}));
    checkRows(derivatives,
              fixture.expectedWeighted(optvars, bare_weights, {0, 1, 2}, derivative_sentinel));
    REQUIRE(stamps.size() == 2);
    CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
    CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_a_, 7));
  }
}

TEST_CASE("Flattened weighted virtual inputs are validated before evaluation",
          "[wavefunction][virtual_batch][weighted]")
{
  VirtualRatioDispatchFixture fixture;
  const OptVariables optvars = makeDenseOptVariables();
  const std::vector<ValueType> bare_weights{ValueType(0.25), ValueType(-0.40), ValueType(0.15),
                                             ValueType(0.30), ValueType(-0.20)};
  const std::vector<ValueType> ratio_sentinel(fixture.batch_->size(), ValueType(-43));
  using Stamp = TrialWaveFunction::EvaluationStamp;
  const std::vector<Stamp> stamp_sentinel{Stamp::versioned(&fixture.sentinel_source_, 113)};

  SECTION("wrong bare-weight extent")
  {
    std::vector<ValueType> short_weights(bare_weights.begin(), bare_weights.end() - 1);
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives(2, std::vector<ValueType>(3, ValueType(7)));
    const auto original_derivatives = derivatives;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, short_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(derivatives == original_derivatives);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("inconsistent derivative row widths")
  {
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives{{ValueType(7), ValueType(7), ValueType(7)},
                                                     {ValueType(7), ValueType(7), ValueType(7), ValueType(7)}};
    const auto original_derivatives = derivatives;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               TrialWaveFunction::ComputeType::ALL),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(derivatives == original_derivatives);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("overlapping derivative rows")
  {
    auto wf_list      = fixture.wavefunctions();
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<ValueType> shared_derivatives(6, ValueType(7));
    const auto original_derivatives = shared_derivatives;
    std::vector<TrialWaveFunction::ParameterDerivativeView> views{
        {shared_derivatives.data(), 3}, {shared_derivatives.data() + 2, 3}};
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
                          wf_list, p_list, scratch_list, *fixture.batch_, optvars, bare_weights, ratios, views,
                          stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(shared_derivatives == original_derivatives);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("derivative destination aliases the replaceable ratio storage")
  {
    auto wf_list      = fixture.wavefunctions();
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<ValueType> second_row(3, ValueType(7));
    std::vector<TrialWaveFunction::ParameterDerivativeView> views{
        {ratios.data(), 3}, {second_row.data(), second_row.size()}};
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
                          wf_list, p_list, scratch_list, *fixture.batch_, optvars, bare_weights, ratios, views,
                          stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(second_row == std::vector<ValueType>(3, ValueType(7)));
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("derivative destination aliases the bare-weight input")
  {
    auto wf_list      = fixture.wavefunctions();
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> aliased_weights = bare_weights;
    std::vector<ValueType> ratios          = ratio_sentinel;
    std::vector<ValueType> second_row(3, ValueType(7));
    std::vector<TrialWaveFunction::ParameterDerivativeView> views{
        {aliased_weights.data(), 3}, {second_row.data(), second_row.size()}};
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
                          wf_list, p_list, scratch_list, *fixture.batch_, optvars, aliased_weights, ratios, views,
                          stamps),
                      std::invalid_argument);
    CHECK(aliased_weights == bare_weights);
    CHECK(ratios == ratio_sentinel);
    CHECK(second_row == std::vector<ValueType>(3, ValueType(7)));
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("nonempty derivative row cannot have a null destination")
  {
    auto wf_list      = fixture.wavefunctions();
    auto p_list       = fixture.particles();
    auto scratch_list = fixture.scratches();
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<ValueType> second_row(3, ValueType(7));
    std::vector<TrialWaveFunction::ParameterDerivativeView> views{
        {nullptr, 3}, {second_row.data(), second_row.size()}};
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
                          wf_list, p_list, scratch_list, *fixture.batch_, optvars, bare_weights, ratios, views,
                          stamps),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(second_row == std::vector<ValueType>(3, ValueType(7)));
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }

  SECTION("invalid compute selection")
  {
    std::vector<ValueType> ratios = ratio_sentinel;
    std::vector<std::vector<ValueType>> derivatives(2, std::vector<ValueType>(3, ValueType(7)));
    const auto original_derivatives = derivatives;
    std::vector<Stamp> stamps = stamp_sentinel;
    REQUIRE_THROWS_AS(fixture.evaluateWeighted(optvars, bare_weights, ratios, derivatives, stamps,
                                               static_cast<TrialWaveFunction::ComputeType>(99)),
                      std::invalid_argument);
    CHECK(ratios == ratio_sentinel);
    CHECK(derivatives == original_derivatives);
    CHECK(stamps == stamp_sentinel);
    CHECK(fixture.totalScalarCalls() == 0);
  }
}

TEST_CASE("Flattened weighted virtual rows honor sparse global parameter indices",
          "[wavefunction][virtual_batch][weighted]")
{
  VirtualRatioDispatchFixture fixture;
  OptVariables global;
  global.insert("padding_0", 0.0);
  global.insert("dispatch_p0", 0.0);
  global.insert("padding_1", 0.0);
  global.insert("padding_2", 0.0);
  global.insert("dispatch_p2", 0.0);
  global.resetIndex();

  OptVariables sparse;
  sparse.insert("dispatch_p0", 0.0);
  sparse.insert("dispatch_p2", 0.0);
  sparse.getIndex(global);
  REQUIRE(sparse.size_of_active() == 2);
  REQUIRE(sparse.where(0) == 1);
  REQUIRE(sparse.where(1) == 4);

  const std::vector<ValueType> bare_weights{ValueType(0.25), ValueType(-0.40), ValueType(0.15),
                                             ValueType(0.30), ValueType(-0.20)};
  std::vector<ValueType> ratios(fixture.batch_->size(), ValueType(-1));
  std::vector<std::vector<ValueType>> short_derivatives(2, std::vector<ValueType>(2, ValueType(9)));
  std::vector<TrialWaveFunction::EvaluationStamp> stamps;
  REQUIRE_THROWS_AS(fixture.evaluateWeighted(sparse, bare_weights, ratios, short_derivatives, stamps,
                                             TrialWaveFunction::ComputeType::ALL),
                    std::invalid_argument);
  CHECK(fixture.totalScalarCalls() == 0);

  const std::vector<std::vector<ValueType>> initial(2, std::vector<ValueType>(5, ValueType(9)));
  std::vector<std::vector<ValueType>> derivatives = initial;
  fixture.evaluateWeighted(sparse, bare_weights, ratios, derivatives, stamps,
                           TrialWaveFunction::ComputeType::ALL);
  checkValues(ratios, fixture.expected({0, 1, 2}));
  checkRows(derivatives, fixture.expectedWeighted(sparse, bare_weights, {0, 1, 2}, initial));
  for (std::size_t walker = 0; walker < derivatives.size(); ++walker)
  {
    CHECK(derivatives[walker][0] == ValueApprox(ValueType(9)));
    CHECK(derivatives[walker][2] == ValueApprox(ValueType(9)));
    CHECK(derivatives[walker][3] == ValueApprox(ValueType(9)));
  }
}

TEST_CASE("Flattened weighted virtual dispatch accepts an empty crowd",
          "[wavefunction][virtual_batch][weighted]")
{
  VirtualRatioDispatchFixture fixture;
  const std::vector<std::size_t> offsets{0};
  const std::vector<Batch::Segment> segments;
  const std::vector<Batch::PosType> positions;
  const Batch empty_batch(0, offsets, segments, positions);
  RefVectorWithLeader<TrialWaveFunction> wf_list(fixture.wf0_);
  RefVectorWithLeader<ParticleSet> p_list(fixture.p0_);
  RefVectorWithLeader<VirtualParticleSet> scratch_list(*fixture.vp0_);
  const OptVariables optvars = makeDenseOptVariables();
  const std::vector<ValueType> bare_weights;
  std::vector<ValueType> ratios;
  const std::vector<TrialWaveFunction::ParameterDerivativeView> derivative_views;
  using Stamp = TrialWaveFunction::EvaluationStamp;
  std::vector<Stamp> stamps{Stamp::versioned(&fixture.sentinel_source_, 127)};

  TrialWaveFunction::mw_evaluateVirtualDerivRatiosWeighted(
      wf_list, p_list, scratch_list, empty_batch, optvars, bare_weights, ratios, derivative_views, stamps);

  CHECK(ratios.empty());
  REQUIRE(stamps.size() == 2);
  CHECK(stamps[0] == Stamp::versioned(&fixture.stamp_source_a_, 7));
  CHECK(stamps[1] == Stamp::versioned(&fixture.stamp_source_b_, 11));
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
