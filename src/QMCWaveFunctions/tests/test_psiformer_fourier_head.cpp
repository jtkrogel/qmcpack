//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "QMCWaveFunctions/PsiFormer/PsiFormerFourierHead.h"

namespace
{
namespace ps = qmcplusplus::psiformer;
using C = ps::FourierFactorJet::Complex;
pf::GeometryBoundary skewCell()
{
  return {pf::GeometryBoundaryKind::PERIODIC, {{{4, 0, 0}, {.7, 3, 0}, {-.4, .9, 5}}}, {true, true, true}};
}
C contract(const ps::FourierFactorJet& a, const ps::FourierFactorJet& b)
{
  C value = a.value * b.value + a.laplacian * b.laplacian;
  for (std::size_t d = 0; d < 3; ++d)
    value += a.gradient[d] * b.gradient[d];
  return value;
}
}

TEST_CASE("Prepared Fourier modes retain simulation duality and full physical origin",
          "[psiformer][fourier_head]")
{
  const std::array<pf::GeometryPosition, 4> ions{{{.1, .2, .3}, {.8, .7, .6},
      {2.1, .2, .3}, {2.8, .7, .6}}};
  const auto source = pf::GeometryPositionView::interleaved(ions.front().data(), ions.size());
  const auto origin = ps::physicalIonFourierOrigin(source);
  CHECK(std::abs(origin[0] - 1.45) < 1e-14);
  CHECK(std::abs(origin[1] - .45) < 1e-14);
  CHECK(std::abs(origin[2] - .45) < 1e-14);
  CHECK_NOTHROW(ps::validateFourierOrigin(origin, source));
  const auto motif_origin = ps::physicalIonFourierOrigin(
      pf::GeometryPositionView::interleaved(ions.front().data(), 2));
  CHECK_THROWS_AS(ps::validateFourierOrigin(motif_origin, source), std::invalid_argument);
  const std::vector<std::array<int, 3>> integers{{0, 0, 0}, {1, -2, 1}, {-1, 0, 2}};
  const auto cell = skewCell();
  const ps::FourierHeadModes modes(cell, integers, origin);
  REQUIRE(modes.momenta().size() == integers.size());
  CHECK(modes.storageBytes() >= 3 * integers.size() * sizeof(double));
  const double two_pi = 2 * std::acos(-1.0);
  for (std::size_t mode = 0; mode < integers.size(); ++mode)
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
      double dot = 0;
      for (std::size_t d = 0; d < 3; ++d)
        dot += modes.momenta()[mode][d] * cell.lattice_vectors[axis][d];
      CHECK(std::abs(dot - two_pi * integers[mode][axis]) < 1e-12);
    }
  CHECK_THROWS_AS(ps::FourierHeadModes(cell, {}, origin), std::invalid_argument);
  CHECK_THROWS_AS(ps::FourierHeadModes(cell, {{1, 0, 0}}, origin), std::invalid_argument);
  CHECK_THROWS_AS(ps::FourierHeadModes(cell, {{0, 0, 0}, {0, 0, 0}}, origin), std::invalid_argument);
  CHECK_THROWS_AS(ps::FourierHeadModes(cell, std::vector<std::array<int, 3>>(65), origin), std::invalid_argument);
  auto nonperiodic = cell;
  nonperiodic.periodic_axes[1] = false;
  CHECK_THROWS_AS(ps::FourierHeadModes(nonperiodic, integers, origin), std::invalid_argument);
  volatile double nonfinite = std::numeric_limits<double>::infinity();
  auto invalid_origin = origin;
  invalid_origin[1] = nonfinite;
  CHECK_THROWS_AS(ps::FourierHeadModes(cell, integers, invalid_origin), std::invalid_argument);
  CHECK_THROWS_AS(ps::validateFourierOrigin(invalid_origin, source), std::invalid_argument);
  auto invalid_ions = ions;
  invalid_ions[2][0] = nonfinite;
  CHECK_THROWS_AS(ps::physicalIonFourierOrigin(pf::GeometryPositionView::interleaved(
      invalid_ions.front().data(), invalid_ions.size())), std::invalid_argument);
}

TEST_CASE("Fourier full-column real coefficient responses use exact spin and mode strides",
          "[psiformer][fourier_head]")
{
  const std::array<pf::GeometryPosition, 4> positions{{{.2, .4, .8}, {1.3, -.2, .5},
      {-.4, 1.1, .7}, {.8, .9, -.3}}};
  const ps::FourierHeadModes modes(skewCell(), {{0, 0, 0}, {1, -1, 0}, {0, 1, 2}}, {.1, -.2, .3});
  constexpr std::size_t channels = 8;
  const std::size_t count = channels * modes.momenta().size();
  std::array<std::vector<double>, 2> coefficients{std::vector<double>(count), std::vector<double>(count)};
  for (auto& sector : coefficients)
    std::fill_n(sector.begin(), channels, 1.0);
  for (std::size_t row = 0; row < positions.size(); ++row)
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      const auto initial = modes.evaluate(positions[row], coefficients[row / 2].data(), count, channels, channel);
      CHECK(std::abs(initial.value - C(1)) < 1e-14);
      CHECK(std::abs(initial.laplacian) < 1e-14);
    }
  for (std::size_t spin = 0; spin < 2; ++spin)
    for (std::size_t index = 0; index < count; ++index)
      coefficients[spin][index] = .1 * std::sin(.7 * (index + 1 + 3 * spin));
  const ps::FourierFactorJet base{C(.4, -.2), {C(.3, .7), C(-.5, .1), C(.6, -.8)}, C(-.9, .2)};
  const ps::FourierFactorJet seed{C(.2, .6), {C(-.3, .5), C(.4, -.1), C(.8, .7)}, C(.1, -.4)};
  auto objective = [&]() {
    C result{};
    for (std::size_t row = 0; row < positions.size(); ++row)
      for (std::size_t channel = 0; channel < channels; ++channel)
        result += contract(ps::multiplyFourierJets(base,
            modes.evaluate(positions[row], coefficients[row / 2].data(), count, channels, channel)), seed);
    return result;
  };
  const double step = 1e-5;
  for (std::size_t spin = 0; spin < 2; ++spin)
    for (std::size_t index = 0; index < count; ++index)
    {
      const std::size_t mode = index / channels;
      C expected{};
      for (std::size_t row = 2 * spin; row < 2 * spin + 2; ++row)
        expected += contract(ps::multiplyFourierJets(base,
            ps::fourierModeJet(positions[row], modes.origin(), modes.momenta()[mode])), seed);
      coefficients[spin][index] += step;
      const C plus = objective();
      coefficients[spin][index] -= 2 * step;
      const C minus = objective();
      coefficients[spin][index] += step;
      CHECK(std::abs(expected - (plus - minus) / (2 * step)) < 1e-8);
    }
  CHECK_THROWS_AS(modes.evaluate(positions[0], coefficients[0].data(), count - 1, channels, 0), std::invalid_argument);
  CHECK_THROWS_AS(modes.evaluate(positions[0], coefficients[0].data(), count, channels, channels), std::invalid_argument);
}

TEST_CASE("Fourier full-head in-place jets and transpose agree across physical target planes",
          "[psiformer][fourier_head]")
{
  const ps::FourierHeadModes modes(skewCell(), {{0, 0, 0}, {1, -1, 2}}, {.1, -.2, .3});
  constexpr std::size_t n = 3, determinants = 2, channels = n * determinants;
  const std::array<double, 18> positions{.1, -.2, .3, .4, .6, .9, -.3, .8, .5,
      .7, .2, -.1, .9, -.5, .4, .2, .8, -.6};
  std::array<std::vector<double>, 2> coefficients{std::vector<double>(2 * channels), std::vector<double>(2 * channels)};
  for (std::size_t spin = 0; spin < 2; ++spin)
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      coefficients[spin][channel] = spin ? .8 : 1;
      coefficients[spin][channels + channel] = spin ? .3 : -1;
    }
  for (auto mode : {ps::SpatialJetMode::VALUE_ONLY, ps::SpatialJetMode::ACTIVE, ps::SpatialJetMode::FULL_VGL})
  {
    const auto layout = ps::makeSpatialJetLayout(2, determinants * n * n, n, mode);
    const auto count = ps::spatialJetSpanElements(layout);
    std::vector<C> base(count), seed(count);
    for (std::size_t index = 0; index < count; ++index)
    {
      base[index] = C(std::sin(.37 * (index + 1)), std::cos(.23 * (index + 3)));
      seed[index] = C(std::cos(.31 * (index + 2)), std::sin(.41 * (index + 1)));
    }
    auto product = base, pullback = seed;
    const std::array<const double*, 2> views{coefficients[0].data(), coefficients[1].data()};
    const std::array<std::size_t, 2> sizes{coefficients[0].size(), coefficients[1].size()};
    ps::transformFourierHeadJets(modes, layout, determinants, 1, positions.data(), positions.size(),
        views, sizes, product.data(), product.size(), false, 2);
    ps::transformFourierHeadJets(modes, layout, determinants, 1, positions.data(), positions.size(),
        views, sizes, pullback.data(), pullback.size(), true, 2);
    C forward{}, reverse{};
    for (std::size_t index = 0; index < count; ++index)
    {
      forward += product[index] * seed[index];
      reverse += base[index] * pullback[index];
    }
    CHECK(std::abs(forward - reverse) < 1e-9);
    std::array<std::vector<C>, 2> responses{std::vector<C>(2 * channels), std::vector<C>(2 * channels)};
    ps::accumulateFourierCoefficientResponses(modes, layout, determinants, 1, positions.data(), positions.size(),
        base.data(), seed.data(), base.size(), {responses[0].data(), responses[1].data()}, sizes, 2);
    auto objective = [&]() {
      auto perturbed = base;
      ps::transformFourierHeadJets(modes, layout, determinants, 1, positions.data(), positions.size(),
          views, sizes, perturbed.data(), perturbed.size(), false, 2);
      C result{};
      for (std::size_t index = 0; index < count; ++index)
        result += perturbed[index] * seed[index];
      return result;
    };
    const double step = 1e-5;
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t parameter = 0; parameter < coefficients[spin].size(); ++parameter)
      {
        coefficients[spin][parameter] += step;
        const C plus = objective();
        coefficients[spin][parameter] -= 2 * step;
        const C minus = objective();
        coefficients[spin][parameter] += step;
        CHECK(std::abs(responses[spin][parameter] - (plus - minus) / (2 * step)) < 2e-7);
      }
    // The first physical row sits exactly at a factor node; its value is zero.
    for (std::size_t column = 0; column < n; ++column)
      CHECK(std::abs(product[layout.uncheckedValueOffset(0, column)]) < 1e-14);
    CHECK_THROWS_AS(ps::transformFourierHeadJets(modes, layout, determinants, 1,
        positions.data(), positions.size() - 1, views, sizes, product.data(), product.size(), false), std::invalid_argument);
  }
}

TEST_CASE("Fourier full-head spatial jets differentiate correlated affine head inputs",
          "[psiformer][fourier_head]")
{
  const ps::FourierHeadModes modes(skewCell(), {{0, 0, 0}, {1, -1, 1}}, {.1, -.2, .3});
  const std::array<std::vector<double>, 2> coefficients{{{.8, .9, -.3, .2}, {.7, 1.1, .4, -.1}}};
  const std::array<const double*, 2> views{coefficients[0].data(), coefficients[1].data()};
  const std::array<std::size_t, 2> sizes{4, 4};
  const std::array<double, 6> positions{.3, -.4, .7, .8, .2, -.5};
  const std::array<C, 6> rates{C(.1, .3), C(-.2, .4), C(.15, -.3), C(-.1, .2), C(.2, -.1), C(.3, .4)};
  auto evaluate = [&](const auto& coordinates, ps::SpatialJetMode mode) {
    const auto layout = ps::makeSpatialJetLayout(1, 4, 2, mode);
    std::vector<C> jets(ps::spatialJetSpanElements(layout));
    C argument{};
    for (std::size_t lane = 0; lane < 6; ++lane)
      argument += rates[lane] * coordinates[lane];
    for (std::size_t element = 0; element < 4; ++element)
    {
      const C value = C(.4 + .1 * element, -.2 + .07 * element) * std::exp(argument);
      jets[layout.uncheckedValueOffset(0, element)] = value;
      for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
        jets[layout.uncheckedGradientOffset(0, lane, element)] = rates[lane] * value;
      for (std::size_t electron = 0; electron < layout.laplacian_lanes; ++electron)
      {
        C laplacian{};
        for (std::size_t d = 0; d < 3; ++d)
          laplacian += rates[electron * 3 + d] * rates[electron * 3 + d] * value;
        jets[layout.uncheckedLaplacianOffset(0, electron, element)] = laplacian;
      }
    }
    ps::transformFourierHeadJets(modes, layout, 1, 1, coordinates.data(), coordinates.size(),
        views, sizes, jets.data(), jets.size(), false);
    return jets;
  };
  const auto layout = ps::makeSpatialJetLayout(1, 4, 2, ps::SpatialJetMode::FULL_VGL);
  const auto jets = evaluate(positions, ps::SpatialJetMode::FULL_VGL);
  const double step = 1e-4;
  std::array<C, 8> laplacians{};
  for (std::size_t lane = 0; lane < 6; ++lane)
  {
    auto plus = positions, minus = positions;
    plus[lane] += step;
    minus[lane] -= step;
    const auto vp = evaluate(plus, ps::SpatialJetMode::VALUE_ONLY);
    const auto vm = evaluate(minus, ps::SpatialJetMode::VALUE_ONLY);
    for (std::size_t element = 0; element < 4; ++element)
    {
      CHECK(std::abs(jets[layout.uncheckedGradientOffset(0, lane, element)] -
          (vp[element] - vm[element]) / (2 * step)) < 2e-8);
      laplacians[(lane / 3) * 4 + element] +=
          (vp[element] - 2.0 * jets[element] + vm[element]) / (step * step);
    }
  }
  for (std::size_t electron = 0; electron < 2; ++electron)
    for (std::size_t element = 0; element < 4; ++element)
      CHECK(std::abs(jets[layout.uncheckedLaplacianOffset(0, electron, element)] -
          laplacians[electron * 4 + element]) < 2e-7);
}
