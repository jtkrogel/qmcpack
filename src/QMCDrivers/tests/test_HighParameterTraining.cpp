//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_HighParameterTraining.cpp
 * @brief Unit tests for bounded derivative products and atomic training iterations.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/NonLocalEnergyAccumulator.h"
#include "QMCDrivers/WFTrain/WFTrainBatched.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/tests/psiformer_test_utils.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/CoulombPBCAA.h"
#include "QMCHamiltonians/CoulombPBCAB.h"
#include "QMCHamiltonians/HarmonicExternalPotential.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCHamiltonians/ECPotentialBuilder.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Utilities/FakeRandom.h"
#include "LongRange/LRCoulombSingleton.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using ComplexRows = std::vector<std::vector<DerivativeValue>>;

/// Small versioned provider used to verify coordinator transaction semantics.
class ToyProvider final : public StructuredParameterProvider
{
public:
  explicit ToyProvider(std::vector<std::string>* events = nullptr)
      : schema_("toy/model",
                {{"weight", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "weights"},
                 {"frozen", {1}, 2, 1, ParameterScalarDomain::REAL64, false, "fixed"}}),
        values_{1.0, -2.0, 7.0},
        events_(events)
  {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override
  {
    return {schema_.fingerprint(), version_, values_};
  }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate,
                                std::size_t expected_version) override
  {
    if (reject_next_publish_)
    {
      reject_next_publish_ = false;
      throw std::runtime_error("stale toy version");
    }
    if (expected_version != version_)
      throw std::runtime_error("stale toy version");
    if (candidate.schema_fingerprint != schema_.fingerprint() ||
        candidate.values.size() != values_.size())
      throw std::invalid_argument("invalid toy candidate");
    for (const double value : candidate.values)
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite toy candidate");
    if (events_)
      events_->push_back("publish");
    values_ = candidate.values;
    return ++version_;
  }

  void rejectNextPublish() noexcept { reject_next_publish_ = true; }

private:
  StructuredParameterSchema schema_;
  std::size_t version_ = 0;
  std::vector<double> values_;
  std::vector<std::string>* events_ = nullptr;
  bool reject_next_publish_ = false;
};

/** Tiny dense derivative oracle confined to test code.
 *
 * Production APIs see only bounded parameter chunks. The explicit sample-by-parameter
 * rows here make the expected VJP algebra independent and easy to inspect.
 */
class ToyStreamingOperator : public StreamingDerivativeOperator
{
public:
  ToyStreamingOperator(const StructuredParameterSchema& schema,
                       std::size_t parameter_version,
                       ComplexRows scores,
                       ComplexRows local_energy_derivatives,
                       bool fail_after_first_chunk = false,
                       const EnergyGradientAccumulator* merge_source = nullptr,
                       std::uint32_t local_energy_term_mask =
                           localEnergyTermBit(LocalEnergyTerm::KINETIC),
                       std::size_t batch_ordinal = 4,
                       std::size_t sample_offset = 9)
      : schema_(schema),
        parameter_version_(parameter_version),
        scores_(std::move(scores)),
        local_energy_derivatives_(std::move(local_energy_derivatives)),
        plan_(schema, parameter_version, 1),
        fail_after_first_chunk_(fail_after_first_chunk),
        merge_source_(merge_source),
        local_energy_term_mask_(local_energy_term_mask),
        batch_ordinal_(batch_ordinal),
        sample_offset_(sample_offset)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) |
        derivativeProductBit(DerivativeProduct::LOCAL_ENERGY_VJP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) |
        derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain    = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain        = ReductionDomain::CROWD_LOCAL;
    result.execution_domain        = DerivativeExecutionDomain::HOST;
    result.local_energy_term_mask     = local_energy_term_mask_;
    result.maximum_vjp_channels          = 3;
    result.maximum_parameter_chunk_size = 1;
    result.maximum_sample_tile_size      = scores_.size();
    result.block_streaming                = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  std::size_t parameterVersion() const noexcept override { return parameter_version_; }

  std::size_t batchOrdinal() const noexcept override { return batch_ordinal_; }

  std::size_t sampleOffset() const noexcept override { return sample_offset_; }

  std::size_t sampleCount() const noexcept override { return scores_.size(); }

  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

  /// Synthetic storage evidence exercises aggregate shared-owner accounting.
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override { return storage_; }

  void setSharedStorage(std::size_t owned)
  {
    storage_.shared_workspace_identity = 7;
    storage_.shared_workspace_bytes = 100;
    storage_.retained_numeric_bytes = owned + 100;
  }

protected:
  StreamingDerivativeStorageDiagnostics storage_;
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
                    DerivativeAdjoint adjoint,
                    ParameterReductionSink& sink) const override
  {
    std::size_t emitted_chunks = 0;
    for (const ParameterChunkDescriptor& descriptor : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
      {
        const ComplexRows& jacobian =
            channels[channel].product == DerivativeProduct::SCORE_VJP
            ? scores_
            : local_energy_derivatives_;
        std::vector<DerivativeValue> contraction(descriptor.count);
        for (std::size_t parameter = 0; parameter < descriptor.count; ++parameter)
          for (std::size_t sample = 0; sample < scores_.size(); ++sample)
          {
            DerivativeValue derivative =
                jacobian[sample][descriptor.parameter_offset + parameter];
            if (adjoint == DerivativeAdjoint::HERMITIAN)
              derivative = std::conj(derivative);
            contraction[parameter] +=
                derivative * channels[channel].coefficients.values[sample];
          }

        sink.add(channel, {descriptor, {contraction.data(), contraction.size()}});
        if (merge_source_ && emitted_chunks == 0)
          dynamic_cast<EnergyGradientAccumulator&>(sink).merge(*merge_source_);
        if (fail_after_first_chunk_ && emitted_chunks == 0)
          throw std::runtime_error("synthetic streaming failure");
        ++emitted_chunks;
      }
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView&,
                        SampleProductSink&) const override
  {
    throw std::logic_error("Toy energy-gradient operator does not implement score JVP");
  }

private:
  const StructuredParameterSchema& schema_;
  std::size_t parameter_version_;
  ComplexRows scores_;
  ComplexRows local_energy_derivatives_;
  ParameterChunkPlan plan_;
  bool fail_after_first_chunk_;
  const EnergyGradientAccumulator* merge_source_;
  std::uint32_t local_energy_term_mask_;
  std::size_t batch_ordinal_;
  std::size_t sample_offset_;
};

/// Return the shared deterministic score rows for energy-objective tests.
ComplexRows makeScores()
{
  return {{1.0, 0.0, 3.0}, {2.0, -1.0, 3.0}, {-1.0, 4.0, 3.0}};
}

/// Return the shared deterministic local-energy response rows.
ComplexRows makeLocalEnergyDerivatives()
{
  return {{0.1, 0.2, 0.0}, {0.3, -0.1, 0.0}, {-0.2, 0.4, 0.0}};
}

/// Apply only the derivative half of an objective transaction for lifecycle tests.
void applyEnergyVJPsOnly(const StreamingDerivativeOperator& derivative_operator,
                         DerivativeArrayView<const DerivativeReal> weights,
                         DerivativeArrayView<const DerivativeValue> local_energies,
                         EnergyGradientAccumulator& accumulator)
{
  std::vector<DerivativeValue> weighted_score(weights.size());
  std::vector<DerivativeValue> energy_weighted_score(weights.size());
  for (std::size_t sample = 0; sample < weights.size(); ++sample)
  {
    weighted_score[sample] = weights[sample];
    energy_weighted_score[sample] = weights[sample] * local_energies[sample];
  }

  const StructuredParameterSchema& schema = derivative_operator.parameterSchema();
  const auto coefficients = [&](const std::vector<DerivativeValue>& values) {
    return CoefficientView{schema.providerId(), schema.fingerprint(),
                           derivative_operator.parameterVersion(),
                           derivative_operator.batchOrdinal(),
                           derivative_operator.sampleOffset(),
                           {values.data(), values.size()}};
  };
  const std::array<VJPCoefficientChannel, 3> channels{{
      {EnergyGradientAccumulator::WEIGHTED_SCORE_CHANNEL, DerivativeProduct::SCORE_VJP,
       coefficients(weighted_score), 0},
      {EnergyGradientAccumulator::ENERGY_WEIGHTED_SCORE_CHANNEL,
       DerivativeProduct::SCORE_VJP, coefficients(energy_weighted_score), 0},
      {EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
       DerivativeProduct::LOCAL_ENERGY_VJP, coefficients(weighted_score),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  derivative_operator.applyVJPs({channels.data(), channels.size()},
                                DerivativeAdjoint::TRANSPOSE, accumulator);
}

/// Produce one objective through the checked three-channel streaming boundary.
class ReferenceProducer final : public GradientProducer
{
public:
  explicit ReferenceProducer(std::vector<std::string>* events = nullptr,
                             bool offer_local_energy_vjp = true)
      : events_(events), offer_local_energy_vjp_(offer_local_energy_vjp)
  {}

  TrainingCapabilities capabilities() const noexcept override
  {
    TrainingCapabilities result{TrainingCapability::REAL_PARAMETERS,
                                TrainingCapability::SCORE_VJP};
    if (offer_local_energy_vjp_)
      result.add(TrainingCapability::LOCAL_ENERGY_VJP);
    return result;
  }

  void accumulate(const StructuredParameterSnapshot& parameters,
                  EnergyGradientAccumulator& accumulator) override
  {
    ++calls;
    if (events_)
      events_->push_back("produce");
    ToyStreamingOperator derivative_operator(*schema_, parameters.version, makeScores(),
                                               makeLocalEnergyDerivatives(), fail_next_);
    fail_next_ = false;
    const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
    const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
  }

  void bindSchema(const StructuredParameterSchema& schema) noexcept { schema_ = &schema; }

  void failNext() noexcept { fail_next_ = true; }

  std::size_t calls = 0;

private:
  const StructuredParameterSchema* schema_ = nullptr;
  std::vector<std::string>* events_ = nullptr;
  bool offer_local_energy_vjp_ = true;
  bool fail_next_ = false;
};

/// Deterministic testing update used before production optimizers are introduced.
class ScaleAndSubtract final : public TrainingUpdateRule
{
public:
  ScaleAndSubtract(double scale, std::vector<std::string>* events = nullptr)
      : scale_(scale), events_(events)
  {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      ParameterGradientView objective) override
  {
    if (events_)
      events_->push_back("propose");
    StructuredParameterSnapshot result = parameters;
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (block.trainable)
        for (std::size_t index = block.offset; index < block.offset + block.count; ++index)
          result.values[index] -= scale_ * objective.gradient[index];
    return result;
  }

private:
  double scale_;
  std::vector<std::string>* events_;
};

/// Adversarial updater used to verify schema-owned frozen parameters are immutable.
class ModifyFrozenParameter final : public TrainingUpdateRule
{
public:
  explicit ModifyFrozenParameter(std::vector<std::string>* events = nullptr) : events_(events) {}

  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
                                      const StructuredParameterSnapshot& parameters,
                                      ParameterGradientView) override
  {
    if (events_)
      events_->push_back("propose");
    StructuredParameterSnapshot result = parameters;
    for (const ParameterBlockDescriptor& block : schema.blocks())
      if (!block.trainable && block.count != 0)
      {
        result.values[block.offset] += 1.0;
        return result;
      }
    throw std::logic_error("Toy schema has no frozen parameter");
  }

private:
  std::vector<std::string>* events_;
};

/// Record the sampler-cache refresh barrier without introducing sampler dependencies.
class CountingObserver final : public ParameterUpdateObserver
{
public:
  explicit CountingObserver(std::vector<std::string>* events = nullptr,
                            const TrainingIterationState* state = nullptr)
      : events_(events), state_(state)
  {}

  void parametersPublished(std::size_t new_version) noexcept override
  {
    if (events_)
      events_->push_back("observe");
    observed_committed_state =
        !state_ || (state_->completed_iterations != 0 &&
                    state_->parameter_version == new_version &&
                    !state_->schema_fingerprint.empty());
    ++calls;
    version = new_version;
  }

  std::size_t calls   = 0;
  std::size_t version = 0;
  bool observed_committed_state = false;

private:
  std::vector<std::string>* events_;
  const TrainingIterationState* state_;
};

} // namespace

TEST_CASE("Real OPEN PP identity requires complete native bare-ion source coverage",
          "[drivers][wftrain][open_ecp_physics]")
{
  namespace ps = psiformer;
  namespace fixture = testing::psiformer;
  const SimulationCell cell;
  ParticleSet electrons(cell);
  ParticleSet ions(cell);
  ParticleSet auxiliary(cell);
  electrons.setName("e");
  electrons.create({1, 1});
  auto& es = electrons.getSpeciesSet();
  es.addSpecies("u");
  es.addSpecies("d");
  const int mass = es.addAttribute("mass");
  const int charge = es.addAttribute("charge");
  es(mass, 0) = es(mass, 1) = 1;
  es(charge, 0) = es(charge, 1) = -1;
  electrons.R[0] = {.3, .4, .5};
  electrons.R[1] = {.7, .8, 2.1};
  electrons.resetGroups();
  ions.setName("ion0");
  ions.create({1, 1});
  auto& is = ions.getSpeciesSet();
  is.addSpecies("Li");
  is.addSpecies("H");
  const int ion_charge = is.addAttribute("charge");
  is(ion_charge, 0) = is(ion_charge, 1) = 1;
  const int atomic_number = is.addAttribute("atomicnumber");
  is(atomic_number, 0) = 3;
  is(atomic_number, 1) = 1;
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {0, 0, 3.01392421};
  ions.resetGroups();
  ions.update();
  auxiliary.setName("bare_h");
  auxiliary.create({1});
  auto& as = auxiliary.getSpeciesSet();
  as.addSpecies("H");
  const int auxiliary_charge = as.addAttribute("charge");
  as(auxiliary_charge, 0) = 1;
  auxiliary.R[0] = ions.R[1];
  auxiliary.resetGroups();
  auxiliary.update();
  electrons.addTable(ions);
  electrons.update();
  ps::InitializedPsiFormerParameters parameters;
  parameters.model_shape = {1, 1, 2, 1, 4, 1, 1};
  parameters.profile = "unit_open_physics";
  parameters.seed = 37;
  auto leaves = fixture::makeLayout(2, 2, false, 1, 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256)
        extent = 4;
      else if (extent == 32)
        extent = 2;
  parameters.values = fixture::makeParameters("h2", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto end = cursor + fixture::product(leaf.shape);
    parameters.layouts.push_back({leaf.module, leaf.name, leaf.shape, cursor, end});
    cursor = end;
  }
  PsiFormerWF component("open_physics", parameters, electrons, ions, false);
  component.validateSystem(electrons, ions, "pseudopotential");
  ParticleSet other(electrons);
  other.setName("other_e");
  other.update();
  // Count any accidental quadrature draws during metadata-only admission.
  struct CountingRandom : FakeRandom<QMCTraits::FullPrecRealType>
  {
    std::size_t draws = 0;
    QMCTraits::FullPrecRealType operator()() override
    {
      ++draws;
      return .371;
    }
  } random;
  const auto pp_directory = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      "tests/molecules/LiH_pp";
  // Construct the actual native PP and Coulomb partition, never a mock energy term.
  const auto makeHamiltonian = [&](bool all_pp, bool include_bare, ParticleSet* pp_target = nullptr) {
    auto result = std::make_unique<QMCHamiltonian>();
    result->addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
    result->addOperator(std::make_unique<CoulombPotential>(electrons, true, false), "ee");
    result->addOperator(std::make_unique<CoulombPotential>(ions, false, false), "ii");
    Libxml2Document document;
    const std::string pseudo = "<pairpot type='pseudo' format='xml' algorithm='batched' DLA='no'>"
        "<pseudo elementType='Li' href='" + (pp_directory / "Li.BFD.xml").string() + "'/>" +
        (all_pp ? "<pseudo elementType='H' href='" + (pp_directory / "H.BFD.xml").string() + "'/>" : "") + "</pairpot>";
    REQUIRE(document.parseFromString(pseudo));
    ECPotentialBuilder builder(*result, ions, pp_target ? *pp_target : electrons, OHMMS::Controller);
    REQUIRE(builder.put(document.getRoot()));
    for (int term = 0; term < result->size(); ++term)
      if (auto* nonlocal = dynamic_cast<NonLocalECPotential*>(result->getComponent(term)))
        nonlocal->setRandomGenerator(&random);
    if (include_bare)
      result->addOperator(std::make_unique<CoulombPotential>(auxiliary, electrons, true), "bare");
    return result;
  };
  auto mixed = makeHamiltonian(false, true);
  const auto identity = realECPPhysicalIdentity(component, electrons, *mixed);
  CHECK(identity.mixed_open_sources);
  CHECK_FALSE(identity.fingerprint.empty());
  CHECK(realECPPhysicalIdentity(component, electrons, *mixed).fingerprint == identity.fingerprint);
  es(mass, 0) = 1 + std::numeric_limits<double>::epsilon();
  electrons.resetGroups();
  CHECK(realECPPhysicalIdentity(component, electrons, *mixed).fingerprint != identity.fingerprint);
  es(mass, 0) = 1;
  electrons.resetGroups();
  auto all_pp = makeHamiltonian(true, false);
  CHECK_FALSE(realECPPhysicalIdentity(component, electrons, *all_pp).mixed_open_sources);
  CHECK(realECPPhysicalIdentity(component, electrons, *all_pp).fingerprint != identity.fingerprint);
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *makeHamiltonian(false, false)));
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *makeHamiltonian(true, true)));
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *makeHamiltonian(false, true, &other)));
  auto foreign_bare = makeHamiltonian(false, false);
  foreign_bare->addOperator(std::make_unique<CoulombPotential>(auxiliary, other, true), "foreign_bare");
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *foreign_bare));
  auxiliary.R[0][0] += .01;
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  auxiliary.R[0] = ions.R[1];
  auxiliary.R[0][0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  auxiliary.R[0] = ions.R[1];
  as(auxiliary_charge, 0) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  as(auxiliary_charge, 0) = 2;
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  as(auxiliary_charge, 0) = 1;
  auxiliary.R[0] = ions.R[0];
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  auxiliary.R[0] = ions.R[1];
  CHECK(realECPPhysicalIdentity(component, electrons, *mixed).fingerprint == identity.fingerprint);
  mixed->addOperator(std::make_unique<CoulombPotential>(auxiliary, electrons, true), "duplicate_bare");
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *mixed));
  all_pp->addOperator(std::make_unique<CoulombPotential>(ions, false, false), "duplicate_ii");
  CHECK_THROWS(realECPPhysicalIdentity(component, electrons, *all_pp));
  CHECK(random.draws == 0);
}

TEST_CASE("Selected real pair physics identity binds actual native Ewald sources and targets",
          "[drivers][wftrain][pair_physics]")
{
  namespace ps = psiformer;
  namespace fixture = testing::psiformer;
  // Isolate the handler cache/policy so this fixture cannot relabel other tests' Ewald physics.
  struct RestoreHandler
  {
    decltype(LRCoulombSingleton::CoulombHandler) saved = std::move(LRCoulombSingleton::CoulombHandler);
    const LRCoulombSingleton::lr_type saved_type = LRCoulombSingleton::this_lr_type;
    ~RestoreHandler()
    {
      LRCoulombSingleton::CoulombHandler = std::move(saved);
      LRCoulombSingleton::this_lr_type = saved_type;
    }
  } restore;
  LRCoulombSingleton::this_lr_type = LRCoulombSingleton::EWALD;
  // Construct neutral physical particles and their structure factors in the full simulation cell.
  Lattice lattice;
  lattice.R = {8, 0, 0, 0, 6, 0, 0, 0, 6};
  lattice.BoxBConds = {true, true, true};
  lattice.LR_dim_cutoff = 15;
  lattice.reset();
  const SimulationCell simulation_cell(lattice);
  ParticleSet electrons(simulation_cell), ions(simulation_cell);
  electrons.setName("e");
  electrons.create({1, 1});
  auto& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass"), charge = species.addAttribute("charge");
  species(mass, 0) = species(mass, 1) = 1;
  species(charge, 0) = species(charge, 1) = -1;
  electrons.resetGroups();
  electrons.R[0] = {.3, .4, .5};
  electrons.R[1] = {2.1, 1.5, 2.3};
  ions.setName("ion0");
  ions.create({2});
  ions.getSpeciesSet().addSpecies("H");
  ions.getSpeciesSet()(ions.getSpeciesSet().addAttribute("charge"), 0) = 1;
  ions.resetGroups();
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {4, 0, 0};
  electrons.createSK();
  ions.createSK();
  electrons.update();
  ions.update();
  // Import a tiny complete descriptor; the production Gaussian initializer keeps its fixed architecture.
  ps::InitializedPsiFormerParameters parameters;
  parameters.model_shape = {1, 1, 1, 1, 4, 1, 1};
  parameters.profile = ps::PSIFORMER_PERIODIC_PAIR_V1;
  parameters.seed = 17;
  auto leaves = fixture::makeLayout(2, 1, false, 1, 7);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256) extent = 4;
      else if (extent == 32) extent = 2;
  parameters.values = fixture::makeParameters("h2", leaves);
  std::size_t cursor = 0;
  for (const auto& leaf : leaves)
  {
    const auto end = cursor + fixture::product(leaf.shape);
    parameters.layouts.push_back({leaf.module, leaf.name, leaf.shape, cursor, end});
    cursor = end;
  }
  for (bool bias : {false, true})
  {
    const auto begin = parameters.values.size();
    parameters.values.resize(begin + (bias ? 1 : 8), .02);
    parameters.layouts.push_back({"periodic/electron_gnn_layer/pair_attention", bias ? "b" : "w",
        bias ? std::vector<std::size_t>{1} : std::vector<std::size_t>{8, 1}, begin, parameters.values.size()});
  }
  // The ion features use one primitive motif while physical Coulomb operators use both replicas.
  auto& environment = parameters.environment;
  environment.boundary = ps::BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  environment.periodic_axes = {true, true, true};
  for (int a = 0; a < 3; ++a)
    for (int d = 0; d < 3; ++d) environment.lattice_vectors[a][d] = lattice.R(a, d);
  auto profile = std::make_shared<ps::PeriodicProfileMetadata>();
  profile->cells.simulation = profile->cells.primitive =
      {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  profile->cells.primitive.lattice_vectors[0][0] = 4;
  profile->cells.tiling[0][0] = 2;
  profile->primitive_motif = {{0, 0, 0}};
  profile->motif_species = {0};
  profile->occupied_columns = {{{}, 0, 0, 1}, {{}, 0, 1, 1}};
  environment.periodic_profile = profile;
  PsiFormerWF component("pair_physics", parameters, electrons, ions, true, {}, true);
  component.validateSystem(electrons, ions, "all_electron");
  ParticleSet other(electrons);
  other.setName("other_e");
  other.update();
  // Assemble actual typed native T+ee+ei+ii operators, including their constructed Ewald definitions.
  const auto makeHamiltonian = [&](ParticleSet& kinetic_target, ParticleSet& aa_target, ParticleSet& ab_target) {
    auto result = std::make_unique<QMCHamiltonian>();
    result->addOperator(std::make_unique<BareKineticEnergy>(kinetic_target), "Kinetic");
    result->addOperator(std::make_unique<CoulombPBCAA>(aa_target, true, false, false), "ee");
    result->addOperator(std::make_unique<CoulombPBCAB>(ions, ab_target), "ei");
    result->addOperator(std::make_unique<CoulombPBCAA>(ions, false, false, false), "ii");
    return result;
  };
  auto physical = makeHamiltonian(electrons, electrons, electrons);
  const auto identity = realPairCoulombIdentity(component, electrons, *physical);
  CHECK_FALSE(identity.empty());
  // Reject misbound targets and changed physics while ignoring later singleton policy changes.
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *makeHamiltonian(other, electrons, electrons)));
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *makeHamiltonian(electrons, other, electrons)));
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *makeHamiltonian(electrons, electrons, other)));
  LRCoulombSingleton::this_lr_type = LRCoulombSingleton::ESLER;
  CHECK(realPairCoulombIdentity(component, electrons, *physical) == identity);
  species(charge, 0) = -.5;
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *physical));
  species(charge, 0) = -1;
  QMCHamiltonian missing;
  missing.addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, missing));
  ions.R[1][0] += .01;
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *physical));
  ions.R[1][0] -= .01;
  CHECK(realPairCoulombIdentity(component, electrons, *physical) == identity);
  physical->addOperator(std::make_unique<HarmonicExternalPotential>(electrons), "extra");
  CHECK_THROWS(realPairCoulombIdentity(component, electrons, *physical));
  // Direct constructors use the same CPU profile gate and physical PP identity as XML.
  // Li effective charge one supplies an actual nonlocal PP channel, unlike the H local-only file.
  ions.getSpeciesSet().speciesName[0] = "Li";
  const int atomic_number = ions.getSpeciesSet().addAttribute("atomicnumber");
  ions.getSpeciesSet()(atomic_number, 0) = 3;
  PsiFormerWF pp_component("pair_ecp_physics", parameters, electrons, ions, true, {}, true);
  pp_component.validateSystem(electrons, ions, "pseudopotential");
  REQUIRE_FALSE(pp_component.usesAcceleratorParameterProducts());
  REQUIRE(pp_component.matchesRealDevicePairTarget(electrons));
  const auto pp_directory = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      "tests/molecules/LiH_pp";
  const auto makePPHamiltonian = [&]() {
    auto result = std::make_unique<QMCHamiltonian>();
    result->addOperator(std::make_unique<BareKineticEnergy>(electrons), "Kinetic");
    result->addOperator(std::make_unique<CoulombPBCAA>(electrons, true, false, false), "ee");
    result->addOperator(std::make_unique<CoulombPBCAA>(ions, false, false, false), "ii");
    Libxml2Document document;
    REQUIRE(document.parseFromString("<pairpot type='pseudo' format='xml' algorithm='batched' DLA='no'>"
        "<pseudo elementType='Li' href='" + (pp_directory / "Li.BFD.xml").string() + "'/></pairpot>"));
    ECPotentialBuilder builder(*result, ions, electrons, OHMMS::Controller);
    REQUIRE(builder.put(document.getRoot()));
    return result;
  };
  auto pp = makePPHamiltonian();
  const auto pp_identity = realECPPhysicalIdentity(pp_component, electrons, *pp).fingerprint;
  CHECK(pp_identity.find("real-cpu-pair-score-ecp-v1/") == 0);
  const auto original_provenance = pp->ecpModelProvenance();
  REQUIRE(original_provenance);
  auto altered_provenance = std::make_shared<QMCHamiltonian::ECPModelProvenance>(*original_provenance);
  altered_provenance->scientific_definition += "/different-pp-content";
  pp->setECPModelProvenance(altered_provenance);
  CHECK(realECPPhysicalIdentity(pp_component, electrons, *pp).fingerprint != pp_identity);
  altered_provenance->ordinary_locality_vp = false;
  CHECK_THROWS(realECPPhysicalIdentity(pp_component, electrons, *pp));
  pp->setECPModelProvenance(nullptr);
  CHECK_THROWS(realECPPhysicalIdentity(pp_component, electrons, *pp));
  pp->setECPModelProvenance(original_provenance);
  CHECK(realECPPhysicalIdentity(pp_component, electrons, *pp).fingerprint == pp_identity);
  auto wrong_parameters = parameters;
  auto wrong_profile = std::make_shared<ps::PeriodicProfileMetadata>(*profile);
  wrong_profile->feature_transform = ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
  wrong_parameters.environment.periodic_profile = wrong_profile;
  PsiFormerWF wrong_component("wrong_pair_ecp", wrong_parameters, electrons, ions, true, {}, true);
  wrong_component.validateSystem(electrons, ions, "pseudopotential");
  CHECK_THROWS(realECPPhysicalIdentity(wrong_component, electrons, *pp));
  electrons.setTwist({.125, 0, 0});
  CHECK_THROWS(realECPPhysicalIdentity(pp_component, electrons, *pp));
  electrons.setTwist({0, 0, 0});
  CHECK(realECPPhysicalIdentity(pp_component, electrons, *pp).fingerprint == pp_identity);
  ions.getSpeciesSet().speciesName[0] = "H";
  ions.getSpeciesSet()(atomic_number, 0) = 1;
#ifdef QMC_COMPLEX
  // The paired descriptor keeps the same neutral physical source and adds independent imaginary heads.
  auto paired = parameters;
  paired.environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
  paired.environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
  const auto real_layouts = paired.layouts;
  for (const auto& leaf : real_layouts)
    if (leaf.module.find("Backflow") != std::string::npos)
    {
      const auto begin = paired.values.size();
      const auto count = leaf.end - leaf.begin;
      paired.values.resize(begin + count, .03);
      paired.layouts.push_back({leaf.module, "w_imag", leaf.shape, begin, begin + count});
    }
  PsiFormerComplexWF paired_component("paired_pair_physics", paired, electrons, ions, 16777216);
  REQUIRE(paired_component.matchesPairedDevicePairTarget(electrons));
  CHECK_FALSE(paired_component.matchesLegacyTorusAllElectronTarget(electrons));
  auto paired_physical = makeHamiltonian(electrons, electrons, electrons);
  const auto paired_identity = pairedPairCoulombIdentity(paired_component, electrons, *paired_physical);
  CHECK_FALSE(paired_identity.empty());
  CHECK(paired_identity != identity);
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *makeHamiltonian(other, electrons, electrons)));
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *makeHamiltonian(electrons, other, electrons)));
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *makeHamiltonian(electrons, electrons, other)));
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, other, *paired_physical));
  species(charge, 0) = -.5;
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *paired_physical));
  species(charge, 0) = -1;
  ions.R[1][0] += .01;
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *paired_physical));
  ions.R[1][0] -= .01;
  CHECK(pairedPairCoulombIdentity(paired_component, electrons, *paired_physical) == paired_identity);
  paired_physical->addOperator(std::make_unique<HarmonicExternalPotential>(electrons), "extra");
  CHECK_THROWS(pairedPairCoulombIdentity(paired_component, electrons, *paired_physical));
#endif
}

TEST_CASE("Energy gradient streaming matches both finite-sample estimators",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  for (const auto [estimator, expected_0, expected_1] : {
           std::tuple{EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN, -1.25, 3.2},
           std::tuple{EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY, -1.375, 3.1}})
  {
    ToyStreamingOperator derivative_operator(provider.parameterSchema(), parameters.version,
                                               makeScores(), makeLocalEnergyDerivatives());
    EnergyGradientAccumulator accumulator(provider.parameterSchema(), parameters.version,
                                           estimator);
    accumulateEnergyGradientBatch(
        derivative_operator, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
        accumulator);
    CHECK(accumulator.retainedBytes() == 3 * 3 * sizeof(DerivativeValue));
    accumulator.completeSingleParticipantReduction();

    const EnergyGradientResult result = accumulator.finalize();
    CHECK(result.sample_count == 3);
    CHECK(result.weight_sum == Catch::Approx(4.0));
    CHECK(result.mean_energy.real() == Catch::Approx(0.5));
    CHECK(result.energy_variance == Catch::Approx(1.125));
    CHECK(result.local_energy_term_mask == localEnergyTermBit(LocalEnergyTerm::KINETIC));
    REQUIRE(result.gradient.size() == 3);
    CHECK(result.gradient[0] == Catch::Approx(expected_0));
    CHECK(result.gradient[1] == Catch::Approx(expected_1));
    CHECK(result.gradient[2] == Catch::Approx(0.0));
  }
}

TEST_CASE("Complex score covariance includes phase response without local-energy derivatives",
          "[drivers][training][complex_covariance]")
{
  ToyProvider provider;
  const ComplexRows scores{{{0, 1}, {1, -2}, {0, 3}},
                           {{0, -2}, {-1, 1}, {0, 3}},
                           {{0, 4}, {2, 3}, {0, 3}}};
  const std::vector<DerivativeReal> weights{1, 2, 3};
  const std::vector<DerivativeValue> energies{{-1, 2}, {3, -1}, {2, 4}};
  // Any accidental request for a local-energy response contaminates the stream.
  const DerivativeValue poison{std::numeric_limits<double>::quiet_NaN(), 0};
  const ComplexRows unused_responses(3, std::vector<DerivativeValue>(3, poison));
  DerivativeValue mean_energy{};
  for (std::size_t sample = 0; sample < weights.size(); ++sample)
    mean_energy += weights[sample] * energies[sample] / 6.0;
  std::array<double, 3> expected{};
  for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
    for (std::size_t sample = 0; sample < weights.size(); ++sample)
      expected[parameter] += 2.0 * weights[sample] *
          std::real(std::conj(scores[sample][parameter]) * (energies[sample] - mean_energy)) / 6.0;
  REQUIRE(std::abs(expected[0]) > 1);
  CHECK(expected[2] == Catch::Approx(0).margin(1e-14));
  for (auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    ToyStreamingOperator derivative(provider.parameterSchema(), 0, scores, unused_responses);
    EnergyGradientAccumulator accumulator(provider.parameterSchema(), 0,
                                           EnergyGradientEstimator::SCORE_COVARIANCE, adjoint);
    accumulateEnergyGradientBatch(derivative, {weights.data(), weights.size()},
        {energies.data(), energies.size()}, 0, accumulator, adjoint);
    const std::array<DerivativeValue, 3> extra{};
    CHECK_THROWS(accumulator.addLocalEnergyVJP({extra.data(), extra.size()},
                                               localEnergyTermBit(LocalEnergyTerm::KINETIC)));
    accumulator.completeSingleParticipantReduction();
    const auto result = accumulator.finalize();
    CHECK(result.local_energy_term_mask == 0);
    CHECK(result.mean_energy.real() == Catch::Approx(mean_energy.real()));
    CHECK(result.mean_energy.imag() == Catch::Approx(mean_energy.imag()));
    for (std::size_t parameter = 0; parameter < expected.size(); ++parameter)
      CHECK(result.gradient[parameter] == Catch::Approx(expected[parameter]).margin(1e-14));
  }
}

TEST_CASE("Energy-gradient batches reject nonempty null input views before sink work",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  ToyStreamingOperator derivative_operator(provider.parameterSchema(), parameters.version,
                                             makeScores(), makeLocalEnergyDerivatives());
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  EnergyGradientAccumulator invalid_weights(provider.parameterSchema(), parameters.version);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          derivative_operator, {nullptr, weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          invalid_weights),
      Catch::Matchers::ContainsSubstring("require valid storage"));
  CHECK(invalid_weights.state() == DerivativeSinkState::IDLE);

  EnergyGradientAccumulator invalid_energies(provider.parameterSchema(), parameters.version);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          derivative_operator, {weights.data(), weights.size()}, {nullptr, energies.size()},
          localEnergyTermBit(LocalEnergyTerm::KINETIC), invalid_energies),
      Catch::Matchers::ContainsSubstring("require valid storage"));
  CHECK(invalid_energies.state() == DerivativeSinkState::IDLE);
}

TEST_CASE("Poisoned energy-gradient stream requires reset before retry",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
  EnergyGradientAccumulator accumulator(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator failing_operator(provider.parameterSchema(), parameters.version,
                                         makeScores(), makeLocalEnergyDerivatives(), true);

  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          failing_operator, {weights.data(), weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          accumulator),
      "synthetic streaming failure");
  CHECK(accumulator.state() == DerivativeSinkState::POISONED);
  CHECK_FALSE(accumulator.hasCompleteContribution());
  CHECK(accumulator.localEnergyTermMask() == 0);
  CHECK_THROWS(accumulator.finalize());

  accumulator.reset();
  CHECK(accumulator.localEnergyTermMask() == 0);
  ToyStreamingOperator retry_operator(provider.parameterSchema(), parameters.version,
                                       makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      retry_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), accumulator);
  CHECK(accumulator.localEnergyTermMask() == localEnergyTermBit(LocalEnergyTerm::KINETIC));
  accumulator.completeSingleParticipantReduction();
  CHECK(accumulator.finalize().gradient[0] == Catch::Approx(-1.25));
}

TEST_CASE("Complete crowd-local energy-gradient contributions merge deterministically",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const ComplexRows scores = makeScores();
  const ComplexRows local_energy_derivatives = makeLocalEnergyDerivatives();

  EnergyGradientAccumulator first(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator first_operator(provider.parameterSchema(), parameters.version,
                                       {scores[0]}, {local_energy_derivatives[0]});
  const std::vector<DerivativeReal> first_weights{1.0};
  const std::vector<DerivativeValue> first_energies{-1.0};
  accumulateEnergyGradientBatch(
      first_operator, {first_weights.data(), first_weights.size()},
      {first_energies.data(), first_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), first);

  EnergyGradientAccumulator second(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator second_operator(
      provider.parameterSchema(), parameters.version, {scores[1], scores[2]},
      {local_energy_derivatives[1], local_energy_derivatives[2]});
  const std::vector<DerivativeReal> second_weights{2.0, 1.0};
  const std::vector<DerivativeValue> second_energies{0.5, 2.0};
  accumulateEnergyGradientBatch(
      second_operator, {second_weights.data(), second_weights.size()},
      {second_energies.data(), second_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), second);

  // The first merge exercises an empty destination; the second exercises a
  // destination that already owns one complete compatible contribution.
  EnergyGradientAccumulator merged(provider.parameterSchema(), parameters.version);
  merged.merge(first);
  merged.merge(second);
  merged.completeSingleParticipantReduction();
  const EnergyGradientResult merged_result = merged.finalize();

  EnergyGradientAccumulator reference(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator reference_operator(provider.parameterSchema(), parameters.version,
                                           scores, local_energy_derivatives);
  const std::vector<DerivativeReal> reference_weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> reference_energies{-1.0, 0.5, 2.0};
  accumulateEnergyGradientBatch(
      reference_operator, {reference_weights.data(), reference_weights.size()},
      {reference_energies.data(), reference_energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), reference);
  reference.completeSingleParticipantReduction();
  const EnergyGradientResult reference_result = reference.finalize();

  CHECK(merged_result.sample_count == 3);
  CHECK(merged_result.sample_count == reference_result.sample_count);
  CHECK(merged_result.weight_sum == Catch::Approx(4.0));
  CHECK(merged_result.weight_sum == Catch::Approx(reference_result.weight_sum));
  CHECK(merged_result.mean_energy.real() == Catch::Approx(reference_result.mean_energy.real()));
  CHECK(merged_result.energy_variance == Catch::Approx(reference_result.energy_variance));
  CHECK(merged_result.local_energy_term_mask ==
        localEnergyTermBit(LocalEnergyTerm::KINETIC));
  CHECK(merged_result.local_energy_term_mask == reference_result.local_energy_term_mask);
  REQUIRE(merged_result.gradient.size() == reference_result.gradient.size());
  for (std::size_t parameter = 0; parameter < merged_result.gradient.size(); ++parameter)
    CHECK(merged_result.gradient[parameter] ==
          Catch::Approx(reference_result.gradient[parameter]));
}

TEST_CASE("Prepared live batches compose one bounded rank-local objective",
          "[drivers][training][prepared]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const ComplexRows scores = makeScores();
  const ComplexRows responses = makeLocalEnergyDerivatives();

  std::vector<PreparedEnergyGradientBatch> batches;
  PreparedEnergyGradientBatch first;
  first.derivative_operator = std::make_unique<ToyStreamingOperator>(
      provider.parameterSchema(), parameters.version, ComplexRows{scores[0]},
      ComplexRows{responses[0]}, false, nullptr,
      localEnergyTermBit(LocalEnergyTerm::KINETIC), 0, 0);
  first.weights = {1.0};
  dynamic_cast<ToyStreamingOperator&>(*first.derivative_operator).setSharedStorage(20);
  first.local_energies = {-1.0};
  first.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  batches.push_back(std::move(first));

  PreparedEnergyGradientBatch second;
  second.derivative_operator = std::make_unique<ToyStreamingOperator>(
      provider.parameterSchema(), parameters.version,
      ComplexRows{scores[1], scores[2]},
      ComplexRows{responses[1], responses[2]}, false, nullptr,
      localEnergyTermBit(LocalEnergyTerm::KINETIC), 1, 1);
  second.weights = {2.0, 1.0};
  dynamic_cast<ToyStreamingOperator&>(*second.derivative_operator).setSharedStorage(30);
  second.local_energies = {0.5, 2.0};
  second.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  batches.push_back(std::move(second));

  PreparedEnergyGradientProducer producer(std::move(batches));
  CHECK(producer.capabilities().contains(TrainingCapability::SCORE_VJP));
  CHECK(producer.capabilities().contains(TrainingCapability::LOCAL_ENERGY_VJP));
  CHECK_FALSE(producer.capabilities().contains(TrainingCapability::SCORE_JVP));
  const PreparedEnergyGradientStorageDiagnostics storage =
      producer.storageDiagnostics();
  CHECK(storage.batch_count == 2);
  CHECK(storage.sample_count == 3);
  CHECK(storage.derivative_owner_bytes == 150);
  CHECK(storage.sample_scalar_bytes ==
        3 * (sizeof(DerivativeReal) + sizeof(DerivativeValue)));

  EnergyGradientAccumulator accumulator(provider.parameterSchema(),
                                         parameters.version);
  producer.accumulate(parameters, accumulator);
  accumulator.completeSingleParticipantReduction();
  const EnergyGradientResult result = accumulator.finalize();
  CHECK(result.sample_count == 3);
  CHECK(result.weight_sum == Catch::Approx(4.0));
  CHECK(result.mean_energy.real() == Catch::Approx(0.5));
  REQUIRE(result.gradient.size() == 3);
  CHECK(result.gradient[0] == Catch::Approx(-1.25));
  CHECK(result.gradient[1] == Catch::Approx(3.2));
  CHECK(result.gradient[2] == Catch::Approx(0.0));

  EnergyGradientAccumulator reused(provider.parameterSchema(), parameters.version);
  CHECK_THROWS_WITH(producer.accumulate(parameters, reused),
                    "Prepared energy-gradient producer can be consumed only once");
}

TEST_CASE("Prepared ECP response composes additively without term overlap",
          "[drivers][training][prepared][ecp]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
  const std::vector<DerivativeValue> coefficient_values{1.0, 2.0, 1.0};
  const std::uint32_t kinetic_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const std::uint32_t ecp_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);

  ToyStreamingOperator ecp_operator(
      provider.parameterSchema(), parameters.version, makeScores(),
      makeLocalEnergyDerivatives(), false, nullptr, ecp_mask, 0, 0);
  const CoefficientView coefficients{
      provider.parameterSchema().providerId(), provider.parameterSchema().fingerprint(),
      parameters.version, 0, 0,
      {coefficient_values.data(), coefficient_values.size()}};
  const VJPCoefficientChannel ecp_channel{
      EnergyGradientAccumulator::WEIGHTED_LOCAL_ENERGY_CHANNEL,
      DerivativeProduct::LOCAL_ENERGY_VJP, coefficients, ecp_mask};
  PreparedLocalEnergyVJPCapture capture(
      provider.parameterSchema(), parameters.version, ecp_mask);
  const std::size_t capture_fingerprint = capture.storageFingerprint();
  ecp_operator.applyVJPs({&ecp_channel, 1}, DerivativeAdjoint::TRANSPOSE, capture);
  CHECK(capture.storageFingerprint() == capture_fingerprint);
  CHECK(capture.state() == DerivativeSinkState::COMPLETE);

  PreparedEnergyGradientBatch batch;
  batch.derivative_operator = std::make_unique<ToyStreamingOperator>(
      provider.parameterSchema(), parameters.version, makeScores(),
      makeLocalEnergyDerivatives(), false, nullptr, kinetic_mask, 0, 0);
  batch.weights = weights;
  batch.local_energies = energies;
  batch.local_energy_term_mask = kinetic_mask;
  batch.additive_local_energy_vjp = capture.release();
  std::vector<PreparedEnergyGradientBatch> prepared_batches;
  prepared_batches.push_back(std::move(batch));
  PreparedEnergyGradientProducer producer(std::move(prepared_batches));

  EnergyGradientAccumulator accumulator(provider.parameterSchema(), parameters.version);
  producer.accumulate(parameters, accumulator);
  CHECK(accumulator.localEnergyTermMask() == (kinetic_mask | ecp_mask));
  accumulator.completeSingleParticipantReduction();
  const EnergyGradientResult result = accumulator.finalize();
  REQUIRE(result.gradient.size() == 3);
  CHECK(result.gradient[0] == Catch::Approx(-1.0));
  CHECK(result.gradient[1] == Catch::Approx(3.4));
  CHECK(result.gradient[2] == Catch::Approx(0.0));
  CHECK(result.local_energy_term_mask == (kinetic_mask | ecp_mask));

  // Both overlapping coverage and non-finite input fail before changing the
  // already complete primary contribution.
  EnergyGradientAccumulator primary(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator primary_operator(provider.parameterSchema(), parameters.version,
                                         makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      primary_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}, kinetic_mask, primary);
  const std::vector<DerivativeValue> finite_additive{{0.5, 0.0}, {0.4, 0.0}, {0.0, 0.0}};
  CHECK_THROWS_AS(primary.addLocalEnergyVJP(
                      {finite_additive.data(), finite_additive.size()}, kinetic_mask),
                  std::invalid_argument);
  std::vector<DerivativeValue> nonfinite_additive = finite_additive;
  nonfinite_additive[1] = {std::numeric_limits<double>::infinity(), 0.0};
  CHECK_THROWS_AS(primary.addLocalEnergyVJP(
                      {nonfinite_additive.data(), nonfinite_additive.size()}, ecp_mask),
                  std::invalid_argument);
  CHECK(primary.localEnergyTermMask() == kinetic_mask);
  primary.completeSingleParticipantReduction();
  const EnergyGradientResult primary_result = primary.finalize();
  CHECK(primary_result.gradient[0] == Catch::Approx(-1.25));
  CHECK(primary_result.gradient[1] == Catch::Approx(3.2));
}

TEST_CASE("Energy-gradient merge rejects invalid receiver lifecycle states",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};

  EnergyGradientAccumulator source(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator source_operator(provider.parameterSchema(), parameters.version,
                                        makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      source_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), source);

  // A checked derivative stream without its scalar moments is half complete.
  // After rejection, supplying those moments must recover the original result,
  // which proves merge did not mutate the receiver.
  EnergyGradientAccumulator half_complete(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator half_operator(provider.parameterSchema(), parameters.version,
                                      makeScores(), makeLocalEnergyDerivatives());
  applyEnergyVJPsOnly(half_operator, {weights.data(), weights.size()},
                      {energies.data(), energies.size()}, half_complete);
  CHECK_THROWS_WITH(half_complete.merge(source),
                    "Cannot merge into a half-complete energy-gradient accumulator");
  half_complete.addScalarSums(3, 4.0, DerivativeValue{2.0, 0.0}, 5.5);
  half_complete.completeSingleParticipantReduction();
  CHECK(half_complete.finalize().gradient[0] == Catch::Approx(-1.25));

  EnergyGradientAccumulator finalized(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator finalized_operator(provider.parameterSchema(), parameters.version,
                                           makeScores(), makeLocalEnergyDerivatives());
  accumulateEnergyGradientBatch(
      finalized_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      localEnergyTermBit(LocalEnergyTerm::KINETIC), finalized);
  finalized.completeSingleParticipantReduction();
  finalized.finalize();
  CHECK_THROWS_WITH(finalized.merge(source),
                    "Cannot merge into a finalized energy-gradient accumulator");

  // Force merge to execute while the checked sink transaction is active. The
  // operator wrapper poisons the receiver after propagating the lifecycle error.
  EnergyGradientAccumulator active(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator active_operator(provider.parameterSchema(), parameters.version,
                                        makeScores(), makeLocalEnergyDerivatives(), false,
                                        &source);
  CHECK_THROWS_WITH(
      accumulateEnergyGradientBatch(
          active_operator, {weights.data(), weights.size()},
          {energies.data(), energies.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC),
          active),
      "Cannot merge into an active energy-gradient accumulator");
  CHECK(active.state() == DerivativeSinkState::POISONED);
  CHECK_THROWS_WITH(active.merge(source),
                    "Cannot merge into a poisoned energy-gradient accumulator");
}

TEST_CASE("Energy-gradient merge requires identical Hamiltonian-term coverage",
          "[drivers][training]")
{
  ToyProvider provider;
  const StructuredParameterSnapshot parameters = provider.snapshotParameters();
  const std::vector<DerivativeReal> weights{1.0, 2.0, 1.0};
  const std::vector<DerivativeValue> energies{-1.0, 0.5, 2.0};
  const std::uint32_t kinetic_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const std::uint32_t kinetic_ecp_mask =
      kinetic_mask | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);

  EnergyGradientAccumulator kinetic(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator kinetic_operator(provider.parameterSchema(), parameters.version,
                                         makeScores(), makeLocalEnergyDerivatives(), false,
                                         nullptr, kinetic_mask);
  accumulateEnergyGradientBatch(
      kinetic_operator, {weights.data(), weights.size()}, {energies.data(), energies.size()},
      kinetic_mask, kinetic);

  EnergyGradientAccumulator kinetic_ecp(provider.parameterSchema(), parameters.version);
  ToyStreamingOperator kinetic_ecp_operator(
      provider.parameterSchema(), parameters.version, makeScores(),
      makeLocalEnergyDerivatives(), false, nullptr, kinetic_ecp_mask);
  accumulateEnergyGradientBatch(
      kinetic_ecp_operator, {weights.data(), weights.size()},
      {energies.data(), energies.size()}, kinetic_ecp_mask, kinetic_ecp);

  EnergyGradientAccumulator merged(provider.parameterSchema(), parameters.version);
  merged.merge(kinetic);
  CHECK_THROWS_WITH(
      merged.merge(kinetic_ecp),
      Catch::Matchers::ContainsSubstring("different local-energy term coverage"));

  // The failed second merge leaves the original kinetic-only contribution usable.
  merged.completeSingleParticipantReduction();
  const EnergyGradientResult result = merged.finalize();
  CHECK(result.sample_count == 3);
  CHECK(result.weight_sum == Catch::Approx(4.0));
  CHECK(result.gradient[0] == Catch::Approx(-1.25));
  CHECK(result.local_energy_term_mask == kinetic_mask);
}

TEST_CASE("Training capability preflight lists every missing operation",
          "[drivers][training]")
{
  const TrainingCapabilities offered{TrainingCapability::REAL_PARAMETERS};
  const TrainingCapabilities required{TrainingCapability::SCORE_VJP,
                                      TrainingCapability::NONLOCAL_ECP};
  CHECK_THROWS_WITH(requireTrainingCapabilities(offered, required, "toy producer"),
                    Catch::Matchers::ContainsSubstring("score_vjp") &&
                        Catch::Matchers::ContainsSubstring("nonlocal_ecp"));
}

TEST_CASE("High-parameter iteration publishes then refreshes sampler state",
          "[drivers][training]")
{
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1, &events);
  TrainingIterationState state;
  CountingObserver observer(&events, &state);
  HighParameterTraining training({});

  const TrainingIterationResult result =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(result.completed_iteration == 1);
  CHECK(result.parameter_version == 1);
  CHECK(result.objective.parameter_version == 0);
  CHECK(observer.calls == 1);
  CHECK(observer.version == 1);
  CHECK(observer.observed_committed_state);
  CHECK(events == std::vector<std::string>{"produce", "propose", "publish", "observe"});

  const StructuredParameterSnapshot after = provider.snapshotParameters();
  CHECK(after.values[0] == Catch::Approx(1.125));
  CHECK(after.values[1] == Catch::Approx(-2.32));
  CHECK(after.values[2] == Catch::Approx(7.0));
}

TEST_CASE("Streaming and stale publication failures are atomic and retryable",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  CountingObserver observer;
  TrainingIterationState state;
  HighParameterTraining training({});
  const StructuredParameterSnapshot initial = provider.snapshotParameters();

  producer.failNext();
  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "synthetic streaming failure");
  CHECK(provider.snapshotParameters().values == initial.values);
  CHECK(state.completed_iterations == 0);
  CHECK(observer.calls == 0);

  provider.rejectNextPublish();
  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "stale toy version");
  CHECK(provider.snapshotParameters().values == initial.values);
  CHECK(provider.snapshotParameters().version == initial.version);
  CHECK(state.completed_iterations == 0);
  CHECK(observer.calls == 0);

  const TrainingIterationResult retry =
      training.runIteration(provider, producer, updater, state, &observer);
  CHECK(retry.completed_iteration == 1);
  CHECK(observer.calls == 1);
}

TEST_CASE("Stateful optimizer publication failure preserves exact retry state",
          "[drivers][training][optimizer]")
{
  FirstOrderOptimizerOptions options;
  options.method         = FirstOrderMethod::ADAM;
  options.learning_rates = {{"weights", 0.05}};
  options.adam_beta1     = 0.5;
  options.adam_beta2     = 0.75;
  options.epsilon        = 1.0e-7;

  ToyProvider retry_provider;
  ReferenceProducer retry_producer;
  retry_producer.bindSchema(retry_provider.parameterSchema());
  FirstOrderOptimizer retry_optimizer(retry_provider.parameterSchema(), options);
  TrainingIterationState retry_state;
  HighParameterTraining training({});
  const StructuredParameterSnapshot initial = retry_provider.snapshotParameters();

  retry_provider.rejectNextPublish();
  CHECK_THROWS_WITH(
      training.runIteration(retry_provider, retry_producer, retry_optimizer, retry_state),
      "stale toy version");
  CHECK(retry_provider.snapshotParameters().values == initial.values);
  CHECK(retry_optimizer.acceptedUpdateCount() == 0);
  CHECK_FALSE(retry_optimizer.hasLiveProposal());
  CHECK(std::all_of(retry_optimizer.firstMoment().begin(),
                    retry_optimizer.firstMoment().end(),
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(retry_optimizer.secondMoment().begin(),
                    retry_optimizer.secondMoment().end(),
                    [](double value) { return value == 0.0; }));

  training.runIteration(retry_provider, retry_producer, retry_optimizer, retry_state);

  ToyProvider reference_provider;
  ReferenceProducer reference_producer;
  reference_producer.bindSchema(reference_provider.parameterSchema());
  FirstOrderOptimizer reference_optimizer(reference_provider.parameterSchema(), options);
  TrainingIterationState reference_state;
  training.runIteration(reference_provider, reference_producer, reference_optimizer,
                        reference_state);

  CHECK(retry_provider.snapshotParameters().values ==
        reference_provider.snapshotParameters().values);
  CHECK(retry_optimizer.firstMoment() == reference_optimizer.firstMoment());
  CHECK(retry_optimizer.secondMoment() == reference_optimizer.secondMoment());
  CHECK(retry_optimizer.acceptedUpdateCount() == 1);
  CHECK(retry_state.completed_iterations == 1);
}

TEST_CASE("Training iteration-count overflow fails before producer and publication",
          "[drivers][training]")
{
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1, &events);
  CountingObserver observer(&events);
  const StructuredParameterSnapshot before = provider.snapshotParameters();
  TrainingIterationState state{std::numeric_limits<std::size_t>::max(), before.version,
                               provider.parameterSchema().fingerprint()};
  HighParameterTraining training({});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    "High-parameter training iteration count overflow");
  CHECK(producer.calls == 0);
  CHECK(events.empty());
  CHECK(provider.snapshotParameters().version == before.version);
  CHECK(provider.snapshotParameters().values == before.values);
  CHECK(state.completed_iterations == std::numeric_limits<std::size_t>::max());
  CHECK(observer.calls == 0);
}

TEST_CASE("Updater cannot modify a frozen parameter block",
          "[drivers][training]")
{
  std::vector<std::string> events;
  ToyProvider provider(&events);
  ReferenceProducer producer(&events);
  producer.bindSchema(provider.parameterSchema());
  ModifyFrozenParameter updater(&events);
  TrainingIterationState state;
  CountingObserver observer(&events, &state);
  HighParameterTraining training({});
  const StructuredParameterSnapshot before = provider.snapshotParameters();

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state, &observer),
                    Catch::Matchers::ContainsSubstring("modified frozen parameter block"));
  const StructuredParameterSnapshot after = provider.snapshotParameters();
  CHECK(after.version == before.version);
  CHECK(after.values == before.values);
  CHECK(state.completed_iterations == 0);
  CHECK(state.schema_fingerprint.empty());
  CHECK(observer.calls == 0);
  CHECK(events == std::vector<std::string>{"produce", "propose"});
}

TEST_CASE("Invalid distributed reduction policy fails before producer work",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer;
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  HighParameterTraining training(
      {}, EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
      DistributedParameterReduction{DistributedReductionPolicy{0}});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state),
                    Catch::Matchers::ContainsSubstring("invalid reduction policy"));
  CHECK(producer.calls == 0);
}

TEST_CASE("Missing streaming capability fails before producer work",
          "[drivers][training]")
{
  ToyProvider provider;
  ReferenceProducer producer(nullptr, false);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  HighParameterTraining training({});

  CHECK_THROWS_WITH(training.runIteration(provider, producer, updater, state),
                    Catch::Matchers::ContainsSubstring("local_energy_vjp"));
  CHECK(producer.calls == 0);
}

TEST_CASE("Score training requires no local-energy product unless explicitly requested",
          "[drivers][training][score_covariance]")
{
  ToyProvider provider;
  ReferenceProducer producer(nullptr, false);
  producer.bindSchema(provider.parameterSchema());
  ScaleAndSubtract updater(0.1);
  TrainingIterationState state;
  CHECK_FALSE(producer.capabilities().contains(TrainingCapability::LOCAL_ENERGY_VJP));
  for (const auto estimator : {EnergyGradientEstimator::SYMMETRIZED_HAMILTONIAN,
                               EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY})
  {
    HighParameterTraining response({}, estimator);
    CHECK_THROWS_WITH(response.runIteration(provider, producer, updater, state),
        Catch::Matchers::ContainsSubstring("local_energy_vjp"));
  }
  HighParameterTraining explicit_requirement({TrainingCapability::LOCAL_ENERGY_VJP},
                                            EnergyGradientEstimator::SCORE_COVARIANCE);
  CHECK_THROWS_WITH(explicit_requirement.runIteration(provider, producer, updater, state),
      Catch::Matchers::ContainsSubstring("local_energy_vjp"));
  CHECK(producer.calls == 0);
  CHECK(provider.snapshotParameters().version == 0);
  HighParameterTraining score({}, EnergyGradientEstimator::SCORE_COVARIANCE);
  const auto result = score.runIteration(provider, producer, updater, state);
  CHECK(result.parameter_version == 1);
  CHECK(result.objective.estimator == EnergyGradientEstimator::SCORE_COVARIANCE);
  CHECK(result.objective.local_energy_term_mask == 0);
  CHECK(producer.calls == 1);
}

TEST_CASE("Prepared score-only ownership does not imply a local-energy response",
          "[drivers][training][prepared][score_covariance]")
{
  ToyProvider provider;
  const auto parameters = provider.snapshotParameters();
  class ScoreOnlyOperator final : public ToyStreamingOperator
  {
  public:
    using ToyStreamingOperator::ToyStreamingOperator;
    StreamingDerivativeCapabilities capabilities() const noexcept override
    {
      auto result = ToyStreamingOperator::capabilities();
      result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP);
      result.local_energy_term_mask = 0;
      return result;
    }
  };
  const auto make_batches = [&] {
    std::vector<PreparedEnergyGradientBatch> batches;
    PreparedEnergyGradientBatch batch;
    batch.derivative_operator = std::make_unique<ScoreOnlyOperator>(
        provider.parameterSchema(), parameters.version, makeScores(), ComplexRows{});
    batch.weights = {1, 2, 1};
    batch.local_energies = {-1, .5, 2};
    batches.push_back(std::move(batch));
    return batches;
  };
  CHECK_THROWS(PreparedEnergyGradientProducer(make_batches()));
  auto additive = make_batches();
  additive.front().additive_local_energy_vjp.emplace();
  CHECK_THROWS(PreparedEnergyGradientProducer(std::move(additive), std::nullopt, true));
  PreparedEnergyGradientProducer producer(make_batches(), std::nullopt, true);
  CHECK(producer.capabilities().contains(TrainingCapability::SCORE_VJP));
  CHECK_FALSE(producer.capabilities().contains(TrainingCapability::LOCAL_ENERGY_VJP));
  EnergyGradientAccumulator full(provider.parameterSchema(), parameters.version);
  CHECK_THROWS(producer.accumulate(parameters, full));
  CHECK(full.state() == DerivativeSinkState::IDLE);
  EnergyGradientAccumulator score(provider.parameterSchema(), parameters.version,
                                  EnergyGradientEstimator::SCORE_COVARIANCE);
  CHECK_NOTHROW(producer.accumulate(parameters, score));
  score.completeSingleParticipantReduction();
  const auto result = score.finalize();
  CHECK(result.sample_count == 3);
  CHECK(result.local_energy_term_mask == 0);
  CHECK(result.weight_sum == 4);
}

TEST_CASE("Accepted nonlocal tiles reduce complete complex energies without derivatives",
          "[drivers][training][nonlocal_energy]")
{
  ToyProvider provider;
  const auto& schema = provider.parameterSchema();
  const auto cap = NonLocalEnergyAccumulator::requiredBytes(2);
  CHECK(cap == 2 * sizeof(DerivativeValue));
  CHECK_THROWS(NonLocalEnergyAccumulator::requiredBytes(0));
  CHECK_THROWS(NonLocalEnergyAccumulator::requiredBytes(std::numeric_limits<std::size_t>::max()));
  CHECK_THROWS(NonLocalEnergyAccumulator(schema, 3, 7, 11, 2, 2, cap - 1));
  std::vector<DerivativeValue> ones(2, {1, 0});
  VJPCoefficientChannel channel{"energy", DerivativeProduct::LOCAL_ENERGY_VJP,
      {schema.providerId(), schema.fingerprint(), 3, 7, 11, {ones.data(), ones.size()}},
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  NonLocalECPDerivativeContext context{schema.providerId(), schema.fingerprint(), 3, 7, 11, 2, 5, 17};
  class NoDerivativeSink final : public ParameterReductionSink
  {
    /// Any parameter output is a violation of this score-only energy transaction.
    void consume(std::size_t, const ParameterChunkConstView&) override
    { throw std::logic_error("Unexpected parameter output from energy accumulation"); }
  } sink;
  bool imaginary = false;
  SECTION("Exact real limit") {}
#ifdef QMC_COMPLEX
  SECTION("Nonzero imaginary energy") { imaginary = true; }
#endif
  NonLocalEnergyAccumulator accumulator(schema, 3, 7, 11, 2, 2, cap);
  const auto storage = accumulator.storageDiagnostics();
  CHECK(storage.retained_numeric_bytes == cap);
  CHECK(storage.parameter_scratch_bytes == 0);
  accumulator.preflight(context, {&channel, 1}, sink);
  CHECK_THROWS(accumulator.energies());
  accumulator.begin(context, {&channel, 1}, sink);
  using Value = QMCTraits::ValueType;
  std::vector<Value> weights{.2, -.4, .7, -1.2, .25};
  std::vector<Value> ratios{1.1, -.8, .9, 1.3, -.2};
#ifdef QMC_COMPLEX
  if (imaginary)
    for (std::size_t q = 0; q < ratios.size(); ++q) ratios[q] += Value{0, .15 * (q + 1)};
#endif
  std::array<DerivativeValue, 2> expected{};
  const std::array<std::size_t, 5> sample{0, 0, 1, 0, 1};
  for (std::size_t q = 0; q < weights.size(); ++q)
    expected[sample[q]] += DerivativeValue(weights[q]) * DerivativeValue(ratios[q]);
  std::vector<std::size_t> offsets_a{0, 2, 3}, offsets_b{0, 1, 2};
  std::vector<VirtualParticleBatch::Segment> segments_a{{0, 0}, {1, 1}}, segments_b{{0, 1}, {1, 0}};
  std::vector<ParticleSet::PosType> positions_a(3), positions_b(2);
  VirtualParticleBatch a(2, {offsets_a.data(), offsets_a.size()}, {segments_a.data(), segments_a.size()},
                         {positions_a.data(), positions_a.size()});
  VirtualParticleBatch b(2, {offsets_b.data(), offsets_b.size()}, {segments_b.data(), segments_b.size()},
                         {positions_b.data(), positions_b.size()});
  accumulator.consume({a, {weights.data(), 3}, {ratios.data(), 3}, {}, 0, 17});
  accumulator.consume({b, {weights.data() + 3, 2}, {ratios.data() + 3, 2}, {}, 1, 17});
  CHECK_THROWS(accumulator.energies());
  accumulator.end();
  for (std::size_t sample_index = 0; sample_index < expected.size(); ++sample_index)
  {
    CHECK(accumulator.energies()[sample_index].real() == Catch::Approx(expected[sample_index].real()));
    CHECK(accumulator.energies()[sample_index].imag() == Catch::Approx(expected[sample_index].imag()));
  }
  CHECK((std::abs(expected[0].imag()) + std::abs(expected[1].imag()) > .01) == imaginary);
  CHECK(accumulator.gridFingerprint() == 17);
  CHECK(sink.state() == DerivativeSinkState::IDLE);
  CHECK(accumulator.storageDiagnostics().storage_fingerprint == storage.storage_fingerprint);
  CHECK(accumulator.storageDiagnostics().retained_numeric_bytes == cap);
  CHECK_THROWS(accumulator.releaseJVPReplay());
  CHECK_THROWS(accumulator.begin(context, {&channel, 1}, sink));

  // Feed the same complete captured energies into a half-covariance objective.
  // Poisoned dEL rows make an accidental local-energy derivative request fail.
  const std::vector<double> sample_weights{1, 2};
  const std::array<DerivativeValue, 2> local{DerivativeValue{.3, 0}, DerivativeValue{-.8, 0}};
  std::vector<DerivativeValue> total{local[0] + accumulator.energies()[0], local[1] + accumulator.energies()[1]};
  const ComplexRows scores{{{1.2, .2}, {-.7, .1}, {}}, {{-.4, .6}, {1.1, -.3}, {}}};
  const DerivativeValue poison{std::numeric_limits<double>::quiet_NaN(), 0};
  const ComplexRows unused_responses(2, std::vector<DerivativeValue>(3, poison));
  ToyStreamingOperator derivative(schema, 3, scores, unused_responses, false, nullptr,
      localEnergyTermBit(LocalEnergyTerm::KINETIC), 7, 11);
  EnergyGradientAccumulator covariance(schema, 3, EnergyGradientEstimator::HALF_SCORE_COVARIANCE);
  accumulateEnergyGradientBatch(derivative, {sample_weights.data(), sample_weights.size()},
      {total.data(), total.size()}, localEnergyTermBit(LocalEnergyTerm::KINETIC), covariance);
  covariance.completeSingleParticipantReduction();
  const auto result = covariance.finalize();
  const auto mean = (total[0] + 2.0 * total[1]) / 3.0;
  CHECK(result.local_energy_term_mask == 0);
  CHECK(result.mean_energy.real() == Catch::Approx(mean.real()));
  CHECK(result.mean_energy.imag() == Catch::Approx(mean.imag()));
  for (std::size_t parameter = 0; parameter < 3; ++parameter)
  {
    double expected_gradient = 0;
    for (std::size_t sample_index = 0; sample_index < 2; ++sample_index)
      expected_gradient += sample_weights[sample_index] *
          std::real(std::conj(scores[sample_index][parameter]) * (total[sample_index] - mean)) / 3.0;
    CHECK(result.gradient[parameter] == Catch::Approx(expected_gradient).margin(1e-14));
  }
}

TEST_CASE("Nonlocal energy transactions reject incomplete or mismatched native tiles",
          "[drivers][training][nonlocal_energy]")
{
  ToyProvider provider;
  const auto& schema = provider.parameterSchema();
  NonLocalEnergyAccumulator accumulator(schema, 3, 7, 11, 1, 2, sizeof(DerivativeValue));
  std::vector<DerivativeValue> ones(1, {1, 0});
  VJPCoefficientChannel channel{"energy", DerivativeProduct::LOCAL_ENERGY_VJP,
      {schema.providerId(), schema.fingerprint(), 3, 7, 11, {ones.data(), 1}},
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  NonLocalECPDerivativeContext context{schema.providerId(), schema.fingerprint(), 3, 7, 11, 1, 1, 17};
  class NoDerivativeSink final : public ParameterReductionSink
  {
    /// Energy-only capture leaves the checked parameter sink idle.
    void consume(std::size_t, const ParameterChunkConstView&) override
    { throw std::logic_error("Unexpected parameter output from energy accumulation"); }
  } sink;
  SECTION("Stale provider version fails before grid activation")
  {
    ++context.parameter_version;
    CHECK_THROWS(accumulator.preflight(context, {&channel, 1}, sink));
    CHECK(sink.state() == DerivativeSinkState::IDLE);
  }
  SECTION("Unsupported localization fails before grid activation")
  {
    context.localization = NonLocalECPLocalization::DLA;
    CHECK_THROWS(accumulator.preflight(context, {&channel, 1}, sink));
  }
  SECTION("Missing accepted grid is rejected")
  {
    context.grid_fingerprint = 0;
    CHECK_THROWS(accumulator.begin(context, {&channel, 1}, sink));
  }
  SECTION("Incomplete traversal cannot publish sums")
  {
    accumulator.begin(context, {&channel, 1}, sink);
    CHECK_THROWS(accumulator.end());
    CHECK_THROWS(accumulator.energies());
  }
  SECTION("Nonfinite coefficient is rejected before arithmetic")
  {
    ones[0] = {std::numeric_limits<double>::infinity(), 0};
    CHECK_THROWS(accumulator.preflight(context, {&channel, 1}, sink));
  }
  SECTION("Wrong tile identity poisons partial sums")
  {
    std::vector<std::size_t> offsets{0, 1};
    std::vector<VirtualParticleBatch::Segment> segments{{0, 0}};
    std::vector<ParticleSet::PosType> positions(1);
    VirtualParticleBatch batch(1, {offsets.data(), 2}, {segments.data(), 1}, {positions.data(), 1});
    const QMCTraits::ValueType value{1};
    accumulator.begin(context, {&channel, 1}, sink);
    CHECK_THROWS(accumulator.consume({batch, {&value, 1}, {&value, 1}, {}, 0, 18}));
    CHECK_THROWS(accumulator.energies());
    CHECK_THROWS(accumulator.end());
  }
  SECTION("Nonfinite complete ratios poison the accepted traversal")
  {
    std::vector<std::size_t> offsets{0, 1};
    std::vector<VirtualParticleBatch::Segment> segments{{0, 0}};
    std::vector<ParticleSet::PosType> positions(1);
    VirtualParticleBatch batch(1, {offsets.data(), 2}, {segments.data(), 1}, {positions.data(), 1});
    const QMCTraits::ValueType weight{1}, ratio{std::numeric_limits<double>::infinity()};
    accumulator.begin(context, {&channel, 1}, sink);
    CHECK_THROWS(accumulator.consume({batch, {&weight, 1}, {&ratio, 1}, {}, 0, 17}));
    CHECK_THROWS(accumulator.energies());
  }
  SECTION("A mismatched coefficient batch fails before grid activation")
  {
    ++channel.coefficients.batch_ordinal;
    CHECK_THROWS(accumulator.preflight(context, {&channel, 1}, sink));
  }
  SECTION("Traversal overflow fails before consuming an extra tile")
  {
    std::vector<std::size_t> offsets{0, 1};
    std::vector<VirtualParticleBatch::Segment> segments{{0, 0}};
    std::vector<ParticleSet::PosType> positions(1);
    VirtualParticleBatch batch(1, {offsets.data(), 2}, {segments.data(), 1}, {positions.data(), 1});
    const QMCTraits::ValueType value{1};
    accumulator.begin(context, {&channel, 1}, sink);
    accumulator.consume({batch, {&value, 1}, {&value, 1}, {}, 0, 17});
    CHECK_THROWS(accumulator.consume({batch, {&value, 1}, {&value, 1}, {}, 1, 17}));
    CHECK_THROWS(accumulator.energies());
  }
}

} // namespace qmcplusplus::wftrain
