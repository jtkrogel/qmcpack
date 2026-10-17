//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@inte.com, Intel Corp.
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "SplineSetReader.h"
#include "OneSplineOrbData.hpp"
#include "Utilities/FairDivide.h"
#include <Timer.h>
#if defined(QMC_COMPLEX)
#include "SplineC2C.h"
#else
#include "SplineR2R.h"
#include "spline2/MultiBsplineOffloadMapper.hpp"
#include "SplineC2R.h"
#endif
#include "Message/CommOperators.h"
#include "spline2/SplineUtils.h"
#include "spline2/MultiBspline.hpp"
#include <cstring>
#include <iomanip>
#include <type_traits>
#if defined(HAVE_MPI)
#include "spline2/MultiBsplineMPIShared.hpp"
#endif


namespace qmcplusplus
{
template<typename ST>
SplineSetReader<ST>::SplineSetReader(EinsplineSetBuilder* e, bool use_duplex_splines)
    : BsplineReader(e, use_duplex_splines)
{}

template<typename ST>
std::unique_ptr<SPOSet> SplineSetReader<ST>::create_spline_set(const std::string& my_name,
                                                               int spin,
                                                               const std::pair<int, int>& distributed_and_shared_ranks,
                                                               const BandInfoGroup& bandgroup)
{
  if (use_offload)
    app_summary() << "    Running OpenMP offload code path." << std::endl;
  else
    app_summary() << "    Running on CPU." << std::endl;

  auto [distributed_ranks, shared_ranks] = distributed_and_shared_ranks;

  // Only the bounded real Gamma CPU table is captured; legacy readers keep
  // their existing evaluation/cache behavior when provenance is unsupported.
  bool capture = false;
#ifndef QMC_COMPLEX
  capture = std::is_same_v<ST, double> && !use_offload && !use_duplex_splines_ &&
      distributed_ranks == 1 && shared_ranks == 1 && spin == 0 && mybuilder->NumTwists == 1;
  for (int i = 0; i < 3; ++i) capture = capture && mybuilder->TargetPtcl.getLattice().BoxBConds[i];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      capture = capture && mybuilder->TileMatrix(i, j) == (i == j ? 1 : 0);
  for (const auto& band : bandgroup.myBands)
    capture = capture && mybuilder->primcell_kpoints[band.TwistIndex] == TinyVector<double, 3>(0);
#endif
  int eligible = capture;
  myComm->allreduce(eligible);
  capture = eligible == myComm->size();
  int image_size = capture && myComm->rank() == 0 && mybuilder->source_image
      ? mybuilder->source_image->size() : 0;
  myComm->bcast(image_size);
  int image_ready = image_size > 0;
  std::shared_ptr<std::string> received_image;
  if (image_ready)
  {
    try
    {
      if (myComm->rank() != 0) received_image = std::make_shared<std::string>(image_size, '0');
    }
    catch (...) { image_ready = 0; }
    myComm->allreduce(image_ready);
    if (image_ready == myComm->size())
    {
      // MPI reads but does not modify the root buffer; preserve its immutable
      // ownership without copying the image to satisfy the mutable MPI API.
      auto* data = myComm->rank() == 0 ? const_cast<char*>(mybuilder->source_image->data()) : received_image->data();
      myComm->bcast(data, image_size);
      if (received_image) mybuilder->source_image = std::move(received_image);
    }
    else capture = false;
  }
  else capture = false;

  if (use_offload)
  {
    if (distributed_ranks > 1)
      app_warning() << "Offload implemenation doesn't support distributing the memory of spline "
                       "coefficients. Overriding distributed_ranks to 1."
                    << std::endl;
    distributed_ranks = 1;
  }

  auto dist_comm_ptr = std::make_unique<Communicate>(*myComm, myComm->size() / (distributed_ranks * shared_ranks));

  app_log() << "  Using " << (use_duplex_splines_ ? "complex" : "real") << " einspline table." << std::endl;
  if (distributed_ranks > 1)
    app_log() << "  Distributed across " << distributed_ranks << " MPI ranks." << std::endl;
  if (shared_ranks > 1)
    app_log() << "  Shared across " << shared_ranks << " MPI ranks." << std::endl;

  const TinyVector<int, 3> half_g = use_duplex_splines_
      ? TinyVector<int, 3>(0, 0, 0)
      : computeHalfG(mybuilder->TargetPtcl.getLattice().BoxBConds, mybuilder->primcell_kpoints,
                     bandgroup.myBands[0].TwistIndex);

  Ugrid xyz_grid[3];
  typename bspline_traits<ST, 3>::BCType xyz_bc[3];
  set_grid(mybuilder->MeshSize, half_g, xyz_grid, xyz_bc);

  const int N              = bandgroup.getNumDistinctOrbitals();
  const size_t num_splines = getAlignedSize<ST>(use_duplex_splines_ ? N * 2 : N);
  std::unique_ptr<MultiBsplineBase<ST>> multi_splines_ptr;
#if defined(HAVE_MPI)
  if (distributed_ranks * shared_ranks > 1)
    multi_splines_ptr = std::make_unique<MultiBsplineMPIShared<ST>>(xyz_grid, xyz_bc, num_splines,
                                                                    std::move(dist_comm_ptr), distributed_ranks);
  else
#endif
    multi_splines_ptr = std::make_unique<MultiBspline<ST>>(xyz_grid, xyz_bc, num_splines);

  auto& multi_splines(*multi_splines_ptr);
  app_log() << "MEMORY " << multi_splines.sizeInByte() / (1 << 20) << " MB allocated "
            << "for the coefficients in 3D spline orbital representation" << std::endl;

  std::unique_ptr<MultiBsplineOffloadMapperBase<ST>> offload_mapper;
  if (use_offload)
    offload_mapper = std::make_unique<MultiBsplineOffloadMapper<ST>>(*multi_splines_ptr);

  std::unique_ptr<BsplineSet> bspline;
#if defined(QMC_COMPLEX)
  bspline = std::make_unique<SplineC2C<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                            std::move(multi_splines_ptr), std::move(offload_mapper));
#else
  if (use_duplex_splines_)
    bspline = std::make_unique<SplineC2R<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                              std::move(multi_splines_ptr), std::move(offload_mapper));
  else
    bspline = std::make_unique<SplineR2R<ST>>(my_name, bandgroup.getNumSPOs(), mybuilder->PrimCell,
                                              std::move(multi_splines_ptr), std::move(offload_mapper));
#endif

  app_log() << "  ClassName = " << bspline->getClassName() << std::endl;

  bspline->resizeStorage(N);

  bspline->HalfG = half_g;

  if (use_duplex_splines_)
  {
    //baseclass handles twists
    check_twists(*bspline, bandgroup);
  }

  bool foundspline = lookforSplineDataDumpFile(bandgroup, bspline->getKeyword(), sizeof(ST));
  if (foundspline && myComm->rank() == 0)
  {
    Timer now;
    hdf_archive h5f(myComm);
    const auto splinefile = getSplineDumpFileName(bandgroup);
    h5f.open(splinefile, H5F_ACC_RDONLY);
    foundspline = SplineUtils<ST>::read(multi_splines, h5f);
    if (foundspline)
      app_log() << "  Successfully restored 3D B-spline coefficients from " << splinefile << ". The reading time is "
                << now.elapsed() << " sec." << std::endl;
  }

  /* create a sub communicator. spline table is shared across MPI ranks with identical subcomm rank id.
   *  myComm->size() == 12 ; shared_ranks = 2; distributed_ranks = 3;
   *          | group 0 | group 1 |
   *          |  0 1 2  |  3 4 5  |
   *          |  6 7 8  | 9 10 11 |
   */
  Communicate shared_comm(*myComm, shared_ranks, distributed_ranks);
  Communicate dist_comm(shared_comm, shared_comm.size() / distributed_ranks);

  if (!foundspline)
  {
    if (shared_comm.getGroupID() == 0)
    {
      multi_splines.flush_zero(dist_comm.rank());
      Timer now;
      initialize_spline_pio_gather(spin, bandgroup, half_g, bspline->BandIndexMap, multi_splines, dist_comm);
      app_log() << "  SplineSetReader initialize_spline_pio " << now.elapsed() << " sec" << std::endl;
    }

    if (saveSplineCoefs && myComm->rank() == 0)
    {
      Timer now;
      const std::string splinefile(getSplineDumpFileName(bandgroup));
      hdf_archive h5f;
      h5f.create(splinefile);
      std::string classname = bspline->getClassName();
      h5f.write(classname, "class_name");
      int sizeD = sizeof(ST);
      h5f.write(sizeD, "sizeof");
      SplineUtils<ST>::write(multi_splines, h5f);
      h5f.close();
      app_log() << "  Stored spline coefficients in " << splinefile << " for potential reuse. The writing time is "
                << now.elapsed() << " sec." << std::endl;
    }
  }

  {
    myComm->barrier();
    Timer now;
    if (shared_comm.getGroupID() == 0)
      SplineUtils<ST>::bcast(multi_splines, dist_comm.rank(), dist_comm.getInterGroupComm());
    myComm->barrier();
    app_log() << "  Time to bcast the table = " << now.elapsed() << std::endl;
  }

  // Bound cold used-lane hash work, not ordinary table allocation or padding.
  int table_ready = capture && multi_splines.sizeInByte() / multi_splines.num_splines_padded() * N <= 16 * 1024 * 1024;
  myComm->allreduce(table_ready);
  if (table_ready == myComm->size())
  {
    int ready = mybuilder->source_image_consistent;
    if (myComm->rank() == 0)
      ready = ready && EinsplineSetBuilder::captureSourceImage(mybuilder->H5File) == *mybuilder->source_image;
    std::shared_ptr<BsplineSourceIdentity> identity;
    std::uint64_t table_hash = 14695981039346656037ULL;
    try
    {
      // Fingerprint only used lanes, never alignment padding. The physical
      // identity follows the realized bank, including any restored cache.
      const auto& table = multi_splines.getBlock(0);
      const auto mix = [&](unsigned char byte) { table_hash = (table_hash ^ byte) * 1099511628211ULL; };
      std::ostringstream metadata;
      metadata << "real-gamma-spline-v1:" << sizeof(ST) << ':' << N << ':' << spin << ':' << std::hexfloat;
      for (const auto& grid : {table.x_grid, table.y_grid, table.z_grid})
        metadata << grid.start << ':' << grid.end << ':' << grid.num << ':';
      for (const auto& bc : {table.xBC, table.yBC, table.zBC})
        metadata << bc.lCode << ':' << bc.rCode << ':' << bc.lVal << ':' << bc.rVal << ':';
      for (const auto& band : bandgroup.myBands) metadata << band.TwistIndex << ':' << band.BandIndex << ':';
      for (int orbital = 0; orbital < N; ++orbital) metadata << bspline->BandIndexMap[orbital] << ':';
      for (unsigned char byte : metadata.str()) mix(byte);
      std::size_t used_bytes = 0;
      for (int x = 0; x < table.x_grid.num + 3; ++x)
        for (int y = 0; y < table.y_grid.num + 3; ++y)
          for (int z = 0; z < table.z_grid.num + 3; ++z)
            for (int orbital = 0; orbital < N; ++orbital)
            {
              const ST value = table.coefs[x * table.x_stride + y * table.y_stride + z * table.z_stride + orbital];
              std::uint64_t bits = 0;
              std::memcpy(&bits, &value, sizeof(value));
              // The capture path is FP64 only; inspect exponent bits so this
              // rejection also holds under the project's fast-math flags.
              if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
                throw std::runtime_error("Nonfinite used B-spline coefficient");
              for (std::size_t byte = 0; byte < sizeof(value); ++byte) mix((bits >> (8 * byte)) & 255);
              used_bytes += sizeof(value);
            }
      std::ostringstream fingerprint;
      fingerprint << metadata.str() << std::hex << table_hash;
      identity = std::make_shared<BsplineSourceIdentity>(BsplineSourceIdentity{
          mybuilder->H5FileName.string(),
          fingerprint.str(), mybuilder->source_image, used_bytes});
    }
    catch (...) { ready = 0; }
    myComm->allreduce(ready);
    if (ready == myComm->size())
    {
      auto root_hash = table_hash;
      myComm->bcast(root_hash);
      int same_table = root_hash == table_hash;
      myComm->allreduce(same_table);
      if (same_table == myComm->size()) bspline->source_identity_ = std::move(identity);
    }
  }
  return bspline;
}

template<typename ST>
void SplineSetReader<ST>::initialize_spline_pio_gather(const int spin,
                                                       const BandInfoGroup& bandgroup,
                                                       const TinyVector<int, 3>& half_g,
                                                       const aligned_vector<int>& band_index_map,
                                                       MultiBsplineBase<ST>& multi_splines,
                                                       Communicate& dist_comm) const
{
  const auto& block_offsets = multi_splines.getBlockOffsets();
  const size_t iblock       = dist_comm.rank();
  auto& comm                = dist_comm.getInterGroupComm();
  //distribute bands over processor groups
  const int orb_block_start = use_duplex_splines_ ? block_offsets[iblock] / 2 : block_offsets[iblock];
  const int orb_block_end   = use_duplex_splines_ ? block_offsets[iblock + 1] / 2 : block_offsets[iblock + 1];
  const int Nbands          = std::min(orb_block_end, bandgroup.getNumDistinctOrbitals()) - orb_block_start;
  const int Nprocs          = comm.size();
  const int Nbandgroups     = Nbands > 0 ? std::min(Nbands, Nprocs) : 1;

  Communicate band_group_comm(comm, Nbandgroups);
  std::vector<int> band_groups(Nbandgroups + 1, 0);
  FairDivideLow(Nbands, Nbandgroups, band_groups);
  const int iorb_first = orb_block_start + band_groups[band_group_comm.getGroupID()];
  const int iorb_last  = orb_block_start + band_groups[band_group_comm.getGroupID() + 1];

  app_log() << "Start transforming plane waves to 3D B-Splines." << std::endl;
  OneSplineOrbData oneband(mybuilder->MeshSize, half_g, use_duplex_splines_);
  hdf_archive h5f(&band_group_comm, false);
  Vector<std::complex<double>> cG(mybuilder->Gvecs[0].size());
  const std::vector<BandInfo>& cur_bands = bandgroup.myBands;
  if (band_group_comm.isGroupLeader()) h5f.open(mybuilder->H5FileName, H5F_ACC_RDONLY);
  int source_matches = 1;
  if (mybuilder->source_image && band_group_comm.isGroupLeader())
    source_matches = EinsplineSetBuilder::captureSourceImage(h5f) == *mybuilder->source_image;
  // Best-effort capture never throws into the existing FFT/gather schedule.
  // Agree within the actual reader subgroup before beginning its work.
  band_group_comm.bcast(source_matches);
  mybuilder->source_image_consistent = mybuilder->source_image_consistent && source_matches;
  if (band_group_comm.isGroupLeader())
  {
    auto& group_leader_comm = band_group_comm.getInterGroupComm();
    for (int iorb = iorb_first; iorb < iorb_last; iorb++)
    {
      const auto& cur_band = cur_bands[band_index_map[iorb]];
      const int ti         = cur_band.TwistIndex;
      readOneOrbitalCoefs(psi_g_path(ti, spin, cur_band.BandIndex), h5f, cG);
      oneband.fft_spline(cG, mybuilder->Gvecs[0], mybuilder->primcell_kpoints[ti], rotate);
      if (use_duplex_splines_)
      {
        multi_splines.setOneSpline(oneband.get_spline_r(), iorb * 2);
        multi_splines.setOneSpline(oneband.get_spline_i(), iorb * 2 + 1);
      }
      else
        multi_splines.setOneSpline(oneband.get_spline_r(), iorb);
    }

    if (mybuilder->source_image)
      mybuilder->source_image_consistent = mybuilder->source_image_consistent &&
          EinsplineSetBuilder::captureSourceImage(h5f) == *mybuilder->source_image;
    {
      group_leader_comm.barrier();
      Timer now;
      if (use_duplex_splines_)
      {
        std::vector<int> offset(band_groups.size());
        for (int i = 0; i < offset.size(); i++)
          offset[i] = band_groups[i] * 2;
        SplineUtils<ST>::gatherv(multi_splines, iblock, offset, group_leader_comm);
      }
      else
        SplineUtils<ST>::gatherv(multi_splines, iblock, band_groups, group_leader_comm);
      app_log() << "  Time to gather the table = " << now.elapsed() << std::endl;
    }
  }
}

template class SplineSetReader<float>;
#if !defined(QMC_MIXED_PRECISION)
template class SplineSetReader<double>;
#endif
} // namespace qmcplusplus
