//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOccupiedHead.h"

TEST_CASE("Occupied full head preserves integer column identities in a skew simulation cell",
          "[wavefunction][psiformer][occupied_head]")
{
  namespace ps = qmcplusplus::psiformer;
  ps::ExecutionEnvironment environment;
  environment.boundary = ps::BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
  environment.periodic_axes = {true, true, true};
  environment.lattice_vectors = {{{4, 0, 0}, {.7, 3, 0}, {-.4, .9, 5}}};
  auto metadata = std::make_shared<ps::PeriodicProfileMetadata>();
  metadata->cells.simulation = metadata->cells.primitive =
      {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
  metadata->cells.primitive.lattice_vectors[0][0] = 2;
  metadata->cells.tiling[0][0] = 2;
  metadata->primitive_motif = {{.1, .2, .3}};
  metadata->motif_species = {0};
  metadata->occupied_columns = {{{1, -2, 0}, 4, 0, 1}, {{-1, 0, 2}, 4, 0, 1}, {{1, -2, 0}, 8, 1, 1}};
  environment.periodic_profile = metadata;
  const ps::ModelShape shape{2, 1, 1, 16, 256, 4, 4};
  const auto parameters = ps::initializePsiFormerParameters(shape, 19, ps::PSIFORMER_PERIODIC_PAIR_V1, environment);
  const ps::PsiFormerExecutionPlan plan(shape, parameters.layouts, parameters.environment);
  const auto descriptor = ps::occupiedFullHeadDescriptor(plan);
  REQUIRE(descriptor.phase == ps::ComplexOrbitalPhase::RECIPROCAL_ONLY);
  REQUIRE(descriptor.reciprocal_momenta.size() == 2 * shape.determinants * shape.electrons());
  const double two_pi = 2 * std::acos(-1.0);
  for (std::size_t spin = 0; spin < 2; ++spin)
    for (std::size_t channel = 0; channel < shape.determinants; ++channel)
      for (std::size_t column = 0; column < shape.electrons(); ++column)
      {
        const auto& momentum = descriptor.reciprocal_momenta[(spin * shape.determinants + channel) *
                                                            shape.electrons() + column];
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
          double dot = 0;
          for (std::size_t d = 0; d < 3; ++d)
            dot += environment.lattice_vectors[axis][d] * momentum[d];
          CHECK(dot == Catch::Approx(two_pi * metadata->occupied_columns[column].reciprocal[axis]).margin(1e-12));
        }
      }
  for (auto& column : metadata->occupied_columns)
    column.reciprocal = {};
  // The prepared plan remains immutable even when its original metadata owner changes.
  CHECK(ps::occupiedFullHeadDescriptor(plan).reciprocal_momenta == descriptor.reciprocal_momenta);
  metadata->occupied_columns[1].band = 5;
  const ps::PsiFormerExecutionPlan gamma(shape, parameters.layouts, environment);
  const auto gamma_descriptor = ps::occupiedFullHeadDescriptor(gamma);
  CHECK(gamma_descriptor.phase == ps::ComplexOrbitalPhase::NONE);
  CHECK(gamma_descriptor.reciprocal_momenta.empty());
}

TEST_CASE("Occupied phase pullback is the ordinary transpose of complete spatial products",
          "[wavefunction][psiformer][occupied_head]")
{
  namespace ps = qmcplusplus::psiformer;
  using C = ps::determinant::Complex;
  for (const auto mode : {ps::SpatialJetMode::VALUE_ONLY, ps::SpatialJetMode::ACTIVE,
                         ps::SpatialJetMode::FULL_VGL})
  {
    ps::ComplexOrbitalHeadDescriptor descriptor{2, 1, 1, 2, ps::ComplexOrbitalStructure::FULL,
        ps::ComplexOrbitalPhase::RECIPROCAL_ONLY,
        {{.7, -.4, .2}, {-.3, .8, -.5}, {.7, -.4, .2}, {-.3, .8, -.5}}};
    const auto layout = ps::makeSpatialJetLayout(1, 4, 2, mode);
    ps::ComplexOrbitalHeadWorkspace phased(descriptor, layout, layout);
    descriptor.phase = ps::ComplexOrbitalPhase::NONE;
    descriptor.reciprocal_momenta.clear();
    ps::ComplexOrbitalHeadWorkspace periodic(descriptor, layout, layout);
    std::vector<double> features(ps::spatialJetSpanElements(layout)), envelopes(features.size());
    std::vector<C> seed(features.size());
    for (std::size_t i = 0; i < features.size(); ++i)
    {
      features[i] = .3 * std::sin(.7 * (i + 1));
      envelopes[i] = .8 + .2 * std::cos(.4 * (i + 1));
      seed[i] = C{.1 * std::sin(i + .3), .2 * std::cos(.3 * i + .8)};
    }
    const std::vector<double> positions{.2, -.3, .8, .7, .5, -.4};
    const std::vector<double> real{.3, .2, -.4, .7}, imaginary{.1, -.5, .2, .6};
    const ps::ComplexOrbitalHeadParameters parameters{
        {real.data(), imaginary.data(), nullptr, nullptr, 4, 0},
        {imaginary.data(), real.data(), nullptr, nullptr, 4, 0}};
    phased.evaluate(features.data(), features.size(), envelopes.data(), envelopes.size(),
                    positions.data(), positions.size(), parameters, 1);
    periodic.evaluate(features.data(), features.size(), envelopes.data(), envelopes.size(),
                      positions.data(), positions.size(), parameters, 1);
    C forward{};
    for (std::size_t i = 0; i < seed.size(); ++i)
      forward += seed[i] * phased.data()[i];
    auto pulled = seed;
    ps::pullBackOccupiedPhase(phased.descriptor(), layout, positions.data(), positions.size(),
                              pulled.data(), pulled.size(), 1);
    C reverse{};
    for (std::size_t i = 0; i < pulled.size(); ++i)
      reverse += pulled[i] * periodic.data()[i];
    CHECK(std::abs(forward - reverse) < 2e-13 * (1 + std::abs(forward)));
    CHECK_THROWS_AS(ps::pullBackOccupiedPhase(phased.descriptor(), layout, positions.data(), 1,
        pulled.data(), pulled.size(), 1), std::invalid_argument);
  }
}
