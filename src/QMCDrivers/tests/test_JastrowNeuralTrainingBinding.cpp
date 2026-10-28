//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCDrivers/WFTrain/JastrowNeuralTrainingBinding.h"
#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/NonLocalEnergyAccumulator.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "ResourceCollection.h"
#include "Utilities/FakeRandom.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "io/hdf/hdf_archive.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerResidualPair.h"
#include "QMCWaveFunctions/Jastrow/J1OrbitalSoA.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "SimulationCell.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>

namespace qmcplusplus::wftrain
{
namespace
{
/// Fixed seeded periodic model; every reference, spline and geometry is inline.
struct ProductFixture
{
  static Lattice makeLattice()
  {
    Lattice result;
    result.R = {8.,0.,0.,0.,8.,0.,0.,0.,8.};
    result.BoxBConds = {true,true,true};
    result.reset();
    return result;
  }

  explicit ProductFixture(bool frozen = false)
      : cell(makeLattice()), electrons(cell), ions(cell), wavefunction(runtime, "student")
  {
    electrons.setName("e");
    electrons.create({2,2});
    electrons.R[0] = {.2,.3,.4}; electrons.R[1] = {-.4,.6,-.3};
    electrons.R[2] = {1.,-1.,.4}; electrons.R[3] = {-.3,-.5,1.};
    auto& species = electrons.getSpeciesSet();
    const int up = species.addSpecies("u"), down = species.addSpecies("d");
    const int mass = species.addAttribute("mass"), charge = species.addAttribute("charge");
    species(mass,up) = species(mass,down) = 1;
    species(charge,up) = species(charge,down) = -1;
    electrons.resetGroups();
    ions.setName("ion0");
    ions.create({2});
    ions.R[0] = {0.,0.,0.}; ions.R[1] = {2.,0.,0.};
    auto& ion_species = ions.getSpeciesSet();
    const int carbon = ion_species.addSpecies("C");
    ion_species(ion_species.addAttribute("charge"),carbon) = 4;
    ions.resetGroups();

    psiformer::ExecutionEnvironment environment;
    environment.boundary = psiformer::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = psiformer::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.periodic_axes = {true,true,true};
    environment.lattice_vectors = {{{8,0,0},{0,8,0},{0,0,8}}};
    environment.residual_pair = std::make_shared<const psiformer::ResidualPairGeometry>(
        psiformer::ResidualPairMetadata{1,1,1,3}, environment.lattice_vectors);
    auto initialized = psiformer::initializePsiFormerParameters({2,2,2,16,256,4,4}, 31,
        psiformer::PSIFORMER_PERIODIC_JASTROW_V1, environment);
    auto neural = std::make_unique<PsiFormerWF>("neural", std::move(initialized), electrons, ions);
    neural->validateSystem(electrons, ions, "pseudopotential");
    wavefunction.addComponent(std::move(neural));

    const auto spline = [&](const std::string& id) {
      auto result = std::make_unique<BsplineFunctor<QMCTraits::RealType>>(id);
      result->cutoff_radius = 3;
      result->resize(3);
      result->notOpt = frozen;
      for (int i = 0; i < 3; ++i)
      {
        result->Parameters[i] = .04*(i+1);
        result->myVars.insert(id+"_"+std::to_string(i), result->Parameters[i], !frozen, optimize::LOGLINEAR_P);
      }
      result->reset();
      return result;
    };
    using Spline = BsplineFunctor<QMCTraits::RealType>;
    auto j1 = std::make_unique<J1OrbitalSoA<Spline>>("J1", ions, electrons, false);
    j1->addFunc(0, spline("eC"));
    wavefunction.addComponent(std::move(j1));
    auto j2 = std::make_unique<TwoBodyJastrow<Spline>>("J2", electrons, false);
    j2->addFunc(0,0,spline("uu"));
    j2->addFunc(0,1,spline("ud"));
    wavefunction.addComponent(std::move(j2));
    const auto optimize = frozen ? "no" : "yes";
    REQUIRE(definition.parseFromString(std::string("<wavefunction name='student' target='e'>")+
        "<psiformer name='neural' initialization='psiformer_periodic_jastrow_v1' neural_jastrow='zero_residual_pair_v1' "
        "neural_jastrow_cutoff='3' source='ion0' system='pseudopotential'/>"+
        "<jastrow name='J1' type='One-Body' function='bspline' source='ion0'><correlation elementType='C' size='3' rcut='3' cusp='0'>"+
        "<coefficients id='eC' type='Array' optimize='"+optimize+"'>.04 .08 .12</coefficients></correlation></jastrow>"+
        "<jastrow name='J2' type='Two-Body' function='bspline'><correlation speciesA='u' speciesB='u' size='3' rcut='3' cusp='0'>"+
        "<coefficients id='uu' type='Array' optimize='"+optimize+"'>.04 .08 .12</coefficients></correlation>"+
        "<correlation speciesA='u' speciesB='d' size='3' rcut='3' cusp='0'><coefficients id='ud' type='Array' optimize='"+
        optimize+"'>.04 .08 .12</coefficients></correlation></jastrow></wavefunction>"));
    wavefunction.storeXMLNode(definition.getRoot());
    ions.update();
    electrons.update();
  }

  SimulationCell cell;
  ParticleSet electrons, ions;
  const RuntimeOptions runtime;
  TrialWaveFunction wavefunction;
  Libxml2Document definition;
};

/// Capture only the final conventional block's single complete-configuration row.
class CaptureProductObserver final : public AffineScoreObserver
{
public:
  explicit CaptureProductObserver(std::size_t ordinal) : ordinal_(ordinal) {}
  void observe(const AffineScoreObservation& observation) override
  {
    if (observation.affine_block != ordinal_) return;
    CHECK(observation.row_count == 1);
    CHECK(observation.sensitivities[0] == 1);
    ++calls;
  }
  std::size_t calls = 0;
private:
  std::size_t ordinal_;
};
} // namespace

TEST_CASE("Product finite guard inspects stored IEEE values", "[wftrain][jastrow_neural_product][product_finite_guard]")
{
  // Runtime storage covers finite-math policies without constructing infinity in a scalar ABI.
  volatile std::uint64_t stored[] = {UINT64_C(0), UINT64_C(0x8000000000000000),
      UINT64_C(0x7fefffffffffffff), UINT64_C(1), UINT64_C(0x7ff0000000000000),
      UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8000000000001)};
  for (std::size_t i = 0; i < sizeof(stored)/sizeof(stored[0]); ++i)
  {
    const std::uint64_t bits = stored[i];
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    CHECK(psiformer::residualPairFiniteReference(value) == (i < 4));
  }
}

TEST_CASE("Joint Jastrow product scores and kinetic cross terms match finite differences",
          "[wftrain][jastrow_neural_product]")
{
  ProductFixture fixture;
  auto& p = fixture.electrons;
  auto& wf = fixture.wavefunction;
  JastrowNeuralTrainingBinding binding(wf,p,*OHMMS::Controller,false);
  REQUIRE(binding.jastrowParameterCount() == 9);
  auto initial = binding.snapshotParameters();
  std::size_t residual = 0;
  for (const auto& block : binding.parameterSchema().blocks())
    if (block.update_group == "neural_pair_residual") { residual = block.offset; break; }
  initial.values[residual] = .2;
  binding.publishParameters(initial,initial.version);
  auto snapshot = binding.snapshotParameters();
  const auto eta = snapshot.values.size()-binding.jastrowParameterCount();
  std::vector<double> scores;
  CaptureProductObserver observer(binding.affineParameterBlocks().size()-1);
  binding.observeAffineScores(p,observer,scores);
  CHECK(observer.calls == 1);
  const double h = 2e-5;
  for (auto index : {residual,eta,eta+3,eta+6})
  {
    auto upper = snapshot, lower = snapshot;
    upper.values[index] += h; lower.values[index] -= h;
    ParticleSet plus_p(p), minus_p(p);
    plus_p.update(); minus_p.update();
    auto plus = binding.makeCandidate(plus_p,upper), minus = binding.makeCandidate(minus_p,lower);
    plus->evaluateLog(plus_p); minus->evaluateLog(minus_p);
    CHECK(scores[index] == Catch::Approx((plus->getLogPsi()-minus->getLogPsi())/(2*h)).epsilon(2e-5).margin(2e-5));
  }
  CHECK((binding.snapshotParameters().values == snapshot.values));
  p.update(); wf.evaluateLog(p);
  const double base = wf.getLogPsi();
  const auto gradient = p.G;
  const auto laplacian = p.L;
  double fd_kinetic = 0, physical_kinetic = 0;
  for (int e = 0; e < p.getTotalNum(); ++e)
  {
    double fd_lap = 0, g2 = 0;
    for (int d = 0; d < 3; ++d)
    {
      const double position = p.R[e][d];
      p.R[e][d] = position+h; p.update(); wf.evaluateLog(p); const double plus = wf.getLogPsi();
      p.R[e][d] = position-h; p.update(); wf.evaluateLog(p); const double minus = wf.getLogPsi();
      p.R[e][d] = position;
      CHECK(std::real(gradient[e][d]) == Catch::Approx((plus-minus)/(2*h)).epsilon(2e-4).margin(2e-4));
      fd_lap += (plus+minus-2*base)/(h*h);
      g2 += std::real(gradient[e][d]*gradient[e][d]);
    }
    fd_kinetic -= .5*(fd_lap+g2);
    physical_kinetic -= .5*(std::real(laplacian[e])+g2);
  }
  CHECK(physical_kinetic == Catch::Approx(fd_kinetic).epsilon(3e-3).margin(3e-3));
  p.update(); wf.evaluateLog(p);
  VirtualParticleSet virtuals(p);
  const std::vector<ParticleSet::PosType> moves{{.03,-.02,.01},{-.02,.01,.04}};
  virtuals.makeMoves(p,0,moves);
  std::vector<QMCTraits::ValueType> ratios(moves.size());
  wf.evaluateRatios(virtuals,ratios);
  for (std::size_t i = 0; i < moves.size(); ++i)
  {
    ParticleSet moved(p);
    moved.R[0] += moves[i]; moved.update();
    auto trial = wf.makeClone(moved); trial->evaluateLog(moved);
    const double signed_ratio = std::cos(trial->getPhase()-wf.getPhase())*std::exp(trial->getLogPsi()-wf.getLogPsi());
    CHECK(std::real(ratios[i]) == Catch::Approx(signed_ratio).epsilon(3e-9).margin(3e-9));
  }
}

TEST_CASE("Joint product publication is atomic and preserves teacher Jastrows",
          "[wftrain][jastrow_neural_product]")
{
  ProductFixture fixture;
  ParticleSet teacher_p(fixture.electrons), live_p(fixture.electrons);
  TrialWaveFunction teacher(fixture.runtime,"teacher");
  for (std::size_t i = 1; i < fixture.wavefunction.getOrbitals().size(); ++i)
    teacher.addComponent(fixture.wavefunction.getOrbitals()[i]->makeClone(teacher_p));
  teacher_p.update(); teacher.evaluateLog(teacher_p); const auto teacher_log = teacher.getLogPsi();
  JastrowNeuralTrainingBinding binding(fixture.wavefunction,fixture.electrons,*OHMMS::Controller,false);
  auto live = fixture.wavefunction.makeClone(live_p);
  binding.attachLive(*live,live_p);
  const auto initial = binding.snapshotParameters();
  auto invalid = initial;
  for (std::size_t p = invalid.values.size()-9; p < invalid.values.size(); ++p) invalid.values[p] = 1e308;
  CHECK_THROWS(binding.publishParameters(invalid,initial.version));
  CHECK((binding.snapshotParameters().values == initial.values));
  CHECK(binding.snapshotParameters().version == initial.version);
  for (int update = 0; update < 3; ++update)
  {
    auto candidate = binding.snapshotParameters();
    candidate.values[0] += .01;
    candidate.values.back() += .001;
    CHECK(binding.publishParameters(candidate,candidate.version) == initial.version+update+1);
    fixture.electrons.update(); live_p.update();
    fixture.wavefunction.evaluateLog(fixture.electrons); live->evaluateLog(live_p);
    CHECK(live->getLogPsi() == fixture.wavefunction.getLogPsi());
    teacher_p.update(); teacher.evaluateLog(teacher_p);
    CHECK(teacher.getLogPsi() == teacher_log);
  }
  const auto path = std::filesystem::temp_directory_path()/
      ("product_"+std::to_string(OHMMS::Controller->rank())+"_"+
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".h5");
  binding.exportModel(path);
  CHECK((binding.readModel(path).values == binding.snapshotParameters().values));
  ParticleSet relocated_p(fixture.electrons);
  auto relocated = fixture.wavefunction.makeClone(relocated_p);
  relocated->storeXMLNode(fixture.wavefunction.getNode());
  for (auto* child = relocated->getNode()->children; child; child = child->next)
    if (child->type == XML_ELEMENT_NODE && xmlStrEqual(child->name,BAD_CAST "psiformer"))
      xmlSetProp(child,BAD_CAST "parameters",BAD_CAST "/relocated/native.parameters.h5");
  JastrowNeuralTrainingBinding relocated_binding(*relocated,relocated_p,*OHMMS::Controller,false);
  CHECK(relocated_binding.modelIdentity() == binding.modelIdentity());
  CHECK((relocated_binding.readModel(path).values == binding.snapshotParameters().values));
  {
    hdf_archive changed;
    REQUIRE(changed.open(path,H5F_ACC_RDWR));
    changed.push("psiformer_model",false); changed.push("residual_pair",false);
    changed.write(2.5,"cutoff");
  }
  CHECK_THROWS(binding.readModel(path));
  std::filesystem::remove(path);
}

TEST_CASE("Product pretraining and coefficient flags define explicit masks",
          "[wftrain][jastrow_neural_product]")
{
  for (const bool frozen : {false,true})
  {
    ProductFixture fixture(frozen);
    JastrowNeuralTrainingBinding binding(fixture.wavefunction,fixture.electrons,*OHMMS::Controller,true);
    CHECK(binding.jastrowParameterCount() == (frozen ? 0 : 9));
    std::size_t residuals = 0;
    for (const auto& block : binding.parameterSchema().blocks())
      if (block.update_group == "neural_pair_residual" || block.update_group == "traditional_jastrow")
      {
        CHECK_FALSE(block.trainable);
        if (block.update_group == "neural_pair_residual") ++residuals;
      }
    CHECK(residuals == 2);
    const auto before = binding.snapshotParameters();
    auto changed = before;
    for (const auto& block : binding.parameterSchema().blocks())
      if (block.update_group == "neural_pair_residual") { changed.values[block.offset] = .01; break; }
    CHECK_THROWS(binding.publishParameters(changed,before.version));
    CHECK((binding.snapshotParameters().values == before.values));
  }
}

TEST_CASE("Complete-configuration Jastrow KFAC block matches its dense metric",
          "[wftrain][jastrow_neural_product]")
{
  const StructuredParameterSchema schema("small_jastrow",{{"eta",{2,1},0,2,ParameterScalarDomain::REAL64,true,"jastrow"}});
  for (const auto convention : {KFACCurvatureConvention::NATIVE_SCORE,KFACCurvatureConvention::REGISTERED_GAUSSIAN})
  {
    KFACFactorAccumulator factors({schema,{{"complete_configuration",0,NO_KFAC_BIAS_BLOCK,2,1}},{}},0,convention);
    const double sensitivity = 1;
    const std::array<std::array<double,2>,3> scores{{{1.,2.},{-2.,.5},{.3,-1.}}};
    for (std::size_t sample = OHMMS::Controller->rank(); sample < scores.size(); sample += OHMMS::Controller->size())
    {
      const auto& score = scores[sample];
      factors.beginSample(1);
      factors.addObservation({0,{score.data(),2},{&sensitivity,1},1});
      factors.endSample();
    }
    reduceKFACFactorStatistics(factors,OHMMS::Controller,2);
    REQUIRE(factors.sampleCount() == scores.size());
    const auto& factor = factors.factors()[0];
    CHECK(factor.row_count == scores.size());
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
      {
        double expected = 0;
        for (const auto& score : scores) expected += score[i]*score[j];
        CHECK(factor.activation_outer_sum[2*i+j] == Catch::Approx(expected));
      }
    CHECK(factor.sensitivity_outer_sum[0] == (convention == KFACCurvatureConvention::REGISTERED_GAUSSIAN ? 6 : 3));
  }
}

namespace
{
/// The scalar capture never emits parameter responses during this value-ratio oracle.
class IdleProductECPSink final : public ParameterReductionSink
{
  void consume(std::size_t, const ParameterChunkConstView&) override
  { throw std::logic_error("Value-only BFD capture emitted parameter derivatives"); }
};

/// Compare actual accepted BFD quadrature ratios with independent full-product evaluations.
class ProductBFDRatioOracle final : public NonLocalECPDerivativeConsumer
{
public:
  ProductBFDRatioOracle(const StructuredParameterSchema& schema, std::size_t version,
      TrialWaveFunction& wavefunction, ParticleSet& particles)
      : energy(schema,version,0,0,1,particles.getTotalNum(),sizeof(DerivativeValue)),
        reference(particles), private_particles(particles), private_wavefunction(wavefunction.makeClone(private_particles)),
        reference_log(wavefunction.getLogPsi()), reference_phase(wavefunction.getPhase())
  {}

  void preflight(const NonLocalECPDerivativeContext& context, DerivativeArrayView<const VJPCoefficientChannel> channels,
      const ParameterReductionSink& sink) const override { energy.preflight(context,channels,sink); }
  void begin(const NonLocalECPDerivativeContext& context, DerivativeArrayView<const VJPCoefficientChannel> channels,
      ParameterReductionSink& sink) override { energy.begin(context,channels,sink); }
  void consume(const NonLocalECPDerivativeTile& tile) override
  {
    energy.consume(tile);
    for (std::size_t segment = 0; segment < tile.virtual_particles.segmentCount(); ++segment)
    {
      const auto slice = tile.virtual_particles.slice(segment);
      REQUIRE(slice.walkerId() == 0);
      for (std::size_t point = 0; point < slice.size(); ++point)
      {
        private_particles.R = reference.R;
        private_particles.R[slice.electronId()] = slice.absolutePosition(point);
        private_particles.update();
        private_wavefunction->evaluateLog(private_particles);
        const double expected = std::cos(private_wavefunction->getPhase()-reference_phase)*
            std::exp(private_wavefunction->getLogPsi()-reference_log);
        const auto index = slice.flatOffset()+point;
        CHECK(std::real(tile.complete_ratios[index]) == Catch::Approx(expected).epsilon(2e-8).margin(2e-8));
        oracle_energy += std::real(tile.bare_weights[index])*expected;
        ++points;
      }
    }
  }
  void end() override { energy.end(); }
  void abort() noexcept override { energy.abort(); }
  std::unique_ptr<NonLocalECPJVPReplay> releaseJVPReplay() override { return energy.releaseJVPReplay(); }
  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override { return energy.storageDiagnostics(); }

  NonLocalEnergyAccumulator energy;
  const ParticleSet& reference;
  ParticleSet private_particles;
  std::unique_ptr<TrialWaveFunction> private_wavefunction;
  double reference_log, reference_phase, oracle_energy = 0;
  std::size_t points = 0;
};
} // namespace

TEST_CASE("Physical BFD accepted-grid ratios include the entire Jastrow product",
          "[wftrain][jastrow_neural_product]")
{
  ProductFixture fixture;
  auto& p = fixture.electrons;
  auto& wf = fixture.wavefunction;
  JastrowNeuralTrainingBinding binding(wf,p,*OHMMS::Controller,false);
  auto parameters = binding.snapshotParameters();
  for (const auto& block : binding.parameterSchema().blocks())
    if (block.update_group == "neural_pair_residual") parameters.values[block.offset] = .15;
  binding.publishParameters(parameters,parameters.version);
  ECPComponentBuilder builder("product_bfd_oracle",OHMMS::Controller);
  const auto source = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  REQUIRE(builder.read_pp_file((source/"tests/solids/diamondC_1x1x1_pp/C.BFD.xml").string()));
  REQUIRE(builder.pp_nonloc != nullptr);
  auto nonlocal = std::make_unique<NonLocalECPotential>(fixture.ions,p,false,true);
  nonlocal->addComponent(0,std::move(builder.pp_nonloc));
  QMCHamiltonian hamiltonian("product_bfd");
  hamiltonian.addOperator(std::move(nonlocal),"NonLocalECP");
  hamiltonian.addObservables(p);
  FakeRandom<QMCTraits::FullPrecRealType> random;
  hamiltonian.setRandomGenerator(&random);
  p.update(); wf.evaluateLog(p);
  ProductBFDRatioOracle oracle(binding.parameterSchema(),binding.parameterVersion(),wf,p);
  IdleProductECPSink sink;
  const DerivativeValue one{1,0};
  const CoefficientView coefficients{binding.parameterSchema().providerId(),binding.parameterSchema().fingerprint(),
      binding.parameterVersion(),0,0,{&one,1}};
  const VJPCoefficientChannel channel{"nonlocal_value",DerivativeProduct::LOCAL_ENERGY_VJP,coefficients,
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  RefVectorWithLeader<ParticleSet> particles(p,{p});
  RefVectorWithLeader<TrialWaveFunction> wavefunctions(wf,{wf});
  RefVectorWithLeader<QMCHamiltonian> hamiltonians(hamiltonian,{hamiltonian});
  ResourceCollection particle_resources("product_bfd_particles"), wavefunction_resources("product_bfd_wavefunctions"),
      hamiltonian_resources("product_bfd_hamiltonians");
  p.createResource(particle_resources); wf.createResource(wavefunction_resources); hamiltonian.createResource(hamiltonian_resources);
  ResourceCollectionTeamLock<ParticleSet> particle_lock(particle_resources,particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wavefunction_lock(wavefunction_resources,wavefunctions);
  ResourceCollectionTeamLock<QMCHamiltonian> hamiltonian_lock(hamiltonian_resources,hamiltonians);
  const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonians,wavefunctions,particles,oracle,{&channel,1},sink);
  CHECK(oracle.points > 0);
  CHECK(oracle.oracle_energy == Catch::Approx(std::real(oracle.energy.energies()[0])).epsilon(2e-8).margin(2e-8));
  CHECK(result.local_energies[0] == Catch::Approx(oracle.oracle_energy).epsilon(2e-8).margin(2e-8));
}

} // namespace qmcplusplus::wftrain
