//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2024 QMCPACK developers.
//
// File developed by: Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include <MinimalParticlePool.h>
#include <MinimalHamiltonianPool.h>
#include <MinimalWaveFunctionPool.h>
#include "Utilities/ResourceCollection.h"
#include "Utilities/ProjectData.h"

#include "QMCDrivers/WalkerLogCollector.h"


namespace qmcplusplus
{

TEST_CASE("WalkerLogCollector::collect", "[estimators]")
{
  app_log() << "\n\n=======================================================\n";
  app_log() << "test WalkerLogCollector::collect\n";

  ProjectData test_project("test", ProjectData::DriverVersion::BATCH);
  Communicate* comm = OHMMS::Controller;


  auto particle_pool = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunction_pool =
      MinimalWaveFunctionPool::make_diamondC_1x1x1(test_project.getRuntimeOptions(), comm, particle_pool);
  auto& pset = *(particle_pool.getParticleSet("e"));
  // This is where the pset properties "properies" gain the different hamiltonian operator values.
  auto hamiltonian_pool = MinimalHamiltonianPool::make_hamWithEE(comm, particle_pool, wavefunction_pool);

  TrialWaveFunction& twf(wavefunction_pool.getWaveFunction().value());
  QMCHamiltonian& ham(hamiltonian_pool.getHamiltonian().value());

  // setup data structures for multiple walkers

  UPtrVector<QMCHamiltonian> hams;
  UPtrVector<TrialWaveFunction> twfs;
  std::vector<ParticleSet> psets;

  int num_walkers = 4;

  for (int iw = 0; iw < num_walkers; ++iw)
  {
    psets.emplace_back(pset);
    psets.back().randomizeFromSource(*particle_pool.getParticleSet("ion"));
    twfs.emplace_back(twf.makeClone(psets.back()));
    hams.emplace_back(ham.makeClone(psets.back(), *twfs.back()));
  }

  using MCPWalker = Walker<QMCTraits, PtclOnLatticeTraits>;

  std::vector<MCPWalker> walkers(num_walkers, MCPWalker(pset.getTotalNum()));

  for (auto& walker : walkers)
  {
    walker.R          = pset.R;
    walker.spins      = pset.spins;
    walker.Properties = pset.Properties;
    walker.registerData();
    walker.DataSet.allocate();
  }

  WalkerLogState state{true, 1, false};
  WalkerLogCollector wlc(state);

  auto& bsi = wlc.walker_property_int_buffer;
  auto& bsr = wlc.walker_property_real_buffer;
  auto& bpr = wlc.walker_particle_real_buffer;

  CHECK(bsi.nrows() == 0);
  CHECK(bsr.nrows() == 0);
  CHECK(bpr.nrows() == 0);
  CHECK(bsi.ncols() == 0);
  CHECK(bsr.ncols() == 0);
  CHECK(bpr.ncols() == 0);

#ifndef QMC_COMPLEX
  const size_t npcols = 56;
#else
  const size_t npcols = 88;
#endif

  int step = 0;
  wlc.collect(walkers[0], psets[0], *(twfs[0]), *(hams[0]), step);

  CHECK(bsi.nrows() == 1);
  CHECK(bsr.nrows() == 1);
  CHECK(bpr.nrows() == 1);
  CHECK(bsi.ncols() == 4);
  CHECK(bsr.ncols() == 13);
  CHECK(bpr.ncols() == npcols);

  for (size_t iw = 1; iw < walkers.size(); ++iw)
    wlc.collect(walkers[iw], psets[iw], *(twfs[iw]), *(hams[iw]), step);

  CHECK(bsi.nrows() == 4);
  CHECK(bsr.nrows() == 4);
  CHECK(bpr.nrows() == 4);
  CHECK(bsi.ncols() == 4);
  CHECK(bsr.ncols() == 13);
  CHECK(bpr.ncols() == npcols);

  for (step = 1; step < 3; ++step)
    for (size_t iw = 0; iw < walkers.size(); ++iw)
      wlc.collect(walkers[iw], psets[iw], *(twfs[iw]), *(hams[iw]), step);

  CHECK(bsi.nrows() == 12);
  CHECK(bsr.nrows() == 12);
  CHECK(bpr.nrows() == 12);
  CHECK(bsi.ncols() == 4);
  CHECK(bsr.ncols() == 13);
  CHECK(bpr.ncols() == npcols);

  app_log() << "\nend test WalkerLogCollector::collect\n";
  app_log() << "=======================================================\n";
}


} // namespace qmcplusplus


namespace qmcplusplus
{
TEST_CASE("Walker logs retain signed MFR scalar values and layout", "[drivers][mfr_walkerlog]")
{
  const SimulationCell cell;
  ParticleSet particles(cell);
  particles.create({2});
  particles.R[0]={.1,.2,.3}; particles.R[1]={.4,.5,.6};
  particles.G=QMCTraits::GradType(.1);
  particles.L=0.0;
  const std::vector<std::string> names={"MFRPotential","MFRCoreAction","MFRCoreActionImag","TotalEnergyMF",
      "IonIonMF","KineticJastrowResidual","MFRMetaU","MFRMetaDifferential","MFRMetaDifferentialImag"};
  std::vector<int> indices;
  for (const auto& name:names) indices.push_back(particles.PropertyList.add(name));
  ProjectData project("mfr_log",ProjectData::DriverVersion::BATCH);
  TrialWaveFunction psi(project.getRuntimeOptions());
  QMCHamiltonian ham;
  Walker<QMCTraits,PtclOnLatticeTraits> walker(2);
  walker.R=particles.R;
  walker.resizeProperty(1,particles.PropertyList.size());
  for (int i=0;i<particles.PropertyList.size();++i) walker.Properties(0,i)=0.0;
  for (size_t i=0;i<indices.size();++i) walker.Properties(0,indices[i])=-(i+1.0)*.25;
  WalkerLogState state{true,1,false};
  WalkerLogCollector collector(state);
  collector.collect(walker,particles,psi,ham,0);
  for (const int index:indices) walker.Properties(0,index)+=.5;
  collector.collect(walker,particles,psi,ham,1);
  const std::string filename="mfr_walker_log_scalars_unit.h5";
  hdf_archive file;
  REQUIRE(file.create(filename));
  collector.walker_property_real_buffer.registerHDFData(file);
  collector.walker_property_real_buffer.writeHDF(file);
  file.close();
  REQUIRE(file.open(filename,H5F_ACC_RDONLY));
  Matrix<double> data;
  REQUIRE_NOTHROW(file.read(data,"walker_property_real/data"));
  REQUIRE(data.rows()==2);
  for (size_t i=0;i<indices.size();++i)
  {
    size_t first=0,last=0;
    const std::string path="walker_property_real/data_layout/"+names[i];
    REQUIRE_NOTHROW(file.read(first,path+"/index_start"));
    REQUIRE_NOTHROW(file.read(last,path+"/index_end"));
    REQUIRE(last==first+1);
    CHECK(data(0,first)==Approx(-(i+1.0)*.25));
    CHECK(data(1,first)==Approx(-(i+1.0)*.25+.5));
  }
  file.close();
  std::remove(filename.c_str());
}
} // namespace qmcplusplus
