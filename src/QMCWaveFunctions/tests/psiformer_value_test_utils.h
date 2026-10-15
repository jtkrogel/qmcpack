//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file psiformer_value_test_utils.h
 * @brief Shared tiny real/paired model fixtures for CPU value composition tests.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_VALUE_TEST_UTILS_H
#define QMCPLUSPLUS_PSIFORMER_VALUE_TEST_UTILS_H
#include "psiformer_test_utils.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"
#ifndef QMCPLUSPLUS_PSIFORMER_NATIVE_H
#error "Include PsiFormerNative.h before psiformer_value_test_utils.h"
#endif

namespace qmcplusplus::testing::psiformer::value
{
namespace ps = qmcplusplus::psiformer;
namespace fixture = qmcplusplus::testing::psiformer;

/// Generate a tiny existing-architecture model without external files or new storage.
inline pf::PsiFormer tinyModel(bool periodic = false, bool paired = false, std::size_t width = 8)
{
  const auto geometry = fixture::makeGeometry("lih");
  auto leaves = fixture::makeLayout(4, 2, true, 2, periodic ? 7 : 4);
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
    {
      if (extent == 256)
        extent = width;
      else if (extent == 64)
        extent = 8;
    }
  if (paired)
  {
    std::vector<fixture::Leaf> imaginary;
    for (const auto& leaf : leaves)
      if (leaf.module.find("Backflow") != std::string::npos)
        imaginary.push_back({leaf.module, "w_imag", leaf.shape});
    leaves.insert(leaves.end(), imaginary.begin(), imaginary.end());
  }
  std::vector<pf::Layout> layouts;
  std::size_t begin = 0;
  for (const auto& leaf : leaves)
  {
    const std::size_t size = fixture::product(leaf.shape);
    layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, begin + size});
    begin += size;
  }
  pf::Parameters parameters(fixture::makeParameters("lih", leaves), std::move(layouts));
  pf::ConfigData configuration(pf::Tensor({1,4,3}, geometry.electrons),
      pf::Tensor({2,3}, geometry.nuclei), pf::Tensor({2}, geometry.charges), 2, 2, 2);
  return pf::PsiFormer(std::move(parameters), std::move(configuration), 2, width, 2, 2);
}

/// Build the explicit periodic/paired metadata without changing legacy defaults.
inline ps::ExecutionEnvironment environment(bool periodic, bool paired)
{
  ps::ExecutionEnvironment result;
  if (periodic)
  {
    result.boundary = ps::BoundaryCondition::PERIODIC;
    result.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    result.periodic_axes = {true,true,true};
    result.lattice_vectors = {{{8,0,0},{0,8,0},{0,0,8}}};
  }
  if (paired)
  {
    result.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    result.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
  }
  return result;
}

/// Validate the existing canonical real store through the unchanged plan boundary.
inline ps::PsiFormerExecutionPlan plan(const pf::PsiFormer& model, bool periodic = false, bool paired = false)
{
  return ps::PsiFormerExecutionPlan::fromParameters(model.p,
      {model.cfg.nup, model.cfg.ndown, model.cfg.nuclei.shape[0], model.ndet,
       model.dim, model.heads, model.blocks}, environment(periodic, paired));
}
} // namespace qmcplusplus::testing::psiformer::value
#endif
