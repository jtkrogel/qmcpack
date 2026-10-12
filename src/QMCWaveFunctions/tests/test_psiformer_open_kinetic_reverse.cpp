//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_open_kinetic_reverse.cpp
 * @brief Independent CPU/long-double oracle for the open lifted reverse leaves.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenKineticReverse.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace pf = qmcplusplus::psiformer;
namespace reverse = qmcplusplus::psiformer::open_kinetic_reverse;

namespace
{

constexpr double canary = -817.25;

void checkClose(double actual, long double expected, double tolerance = 3.0e-10)
{
  CHECK(actual == Catch::Approx(static_cast<double>(expected))
                      .epsilon(tolerance).margin(tolerance));
}

template<class Function>
long double centralDifference(Function&& function, long double x,
                              long double step = 2.0e-6L)
{
  return (function(x + step) - function(x - step)) / (2.0L * step);
}

pf::OpenOrbitalKineticReverseLayout makeOrbitalLayout()
{
  pf::OpenOrbitalKineticReverseLayout layout{};
  layout.forward = pf::makeOpenOrbitalJetLayout(
      1, 1, 2, 1, 2, 1, pf::SpatialJetMode::FULL_VGL,
      3, 3, 7, 3, 5, 2, 5, 4, 9, 4, 7, 0, 7, 0);
  layout.response_count = 80;
  layout.backflow_up_response_offset = 7;
  layout.backflow_down_response_offset = 22;
  layout.envelope_pi_up_response_offset = 40;
  layout.envelope_pi_down_response_offset = 50;
  layout.envelope_zeta_up_response_offset = 60;
  layout.envelope_zeta_down_response_offset = 70;
  pf::validateOpenOrbitalKineticReverseLayout(layout);
  return layout;
}

void fillFeatures(const pf::OpenOrbitalKineticReverseLayout& layout,
                  std::vector<double>& features)
{
  for (std::size_t plane = 0; plane < layout.forward.features.plane_count; ++plane)
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t feature = 0; feature < 2; ++feature)
      {
        const std::size_t element = row * layout.forward.feature_row_stride + feature;
        features[layout.forward.features.uncheckedPlaneOffset(0, plane, element)] =
            0.07 * static_cast<double>(1 + feature + 3 * row + 5 * plane) - 0.4;
      }
}

void buildRetainedJets(const pf::OpenOrbitalKineticReverseLayout& layout,
                       const std::vector<double>& features,
                       const std::vector<double>& positions,
                       const std::vector<double>& nuclei,
                       const std::vector<double>& backflow_up,
                       const std::vector<double>& backflow_down,
                       const std::vector<double>& pi_up,
                       const std::vector<double>& pi_down,
                       const std::vector<double>& zeta_up,
                       const std::vector<double>& zeta_down,
                       std::vector<double>& backflow,
                       std::vector<double>& envelope)
{
  const auto& source = layout.forward.features;
  const auto& target = layout.forward.orbitals;
  for (std::size_t row = 0; row < 2; ++row)
  {
    const bool up = row == 0;
    const auto& weights = up ? backflow_up : backflow_down;
    const auto& pi = up ? pi_up : pi_down;
    const auto& zeta = up ? zeta_up : zeta_down;
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t element = pf::open_orbital::outputElement(
          layout.forward, 0, row, orbital);
      for (std::size_t plane = 0; plane < source.plane_count; ++plane)
      {
        long double sum = 0.0L;
        for (std::size_t feature = 0; feature < 2; ++feature)
          sum += static_cast<long double>(features[source.uncheckedPlaneOffset(
                     0, plane, row * layout.forward.feature_row_stride + feature)]) *
              weights[pf::open_orbital::weightOffset(
                  layout.forward, feature, 0, orbital)];
        backflow[target.uncheckedPlaneOffset(0, plane, element)] =
            static_cast<double>(sum);
        envelope[target.uncheckedPlaneOffset(0, plane, element)] = 0.0;
      }
      const std::size_t parameter = pf::open_orbital::envelopeOffset(
          layout.forward, 0, orbital, 0);
      std::array<double, 3> displacement{};
      double radius2 = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        displacement[dimension] = positions[row * 4 + dimension] - nuclei[dimension];
        radius2 += displacement[dimension] * displacement[dimension];
      }
      const double radius = std::sqrt(radius2);
      const double a = std::abs(zeta[parameter]);
      const double decay = std::exp(-a * radius);
      const double weighted = pi[parameter] * decay;
      envelope[target.uncheckedValueOffset(0, element)] = weighted;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        envelope[target.uncheckedGradientOffset(0, 3 * row + dimension, element)] =
            -a * weighted * displacement[dimension] / radius;
      envelope[target.uncheckedLaplacianOffset(0, row, element)] =
          weighted * (a * a - 2.0 * a / radius);
    }
  }
}

long double orbitalObjective(
    const pf::OpenOrbitalKineticReverseLayout& layout,
    const std::vector<double>& features,
    const std::vector<double>& positions,
    const std::vector<double>& nuclei,
    const std::vector<long double>& backflow_up,
    const std::vector<long double>& backflow_down,
    const std::vector<long double>& pi_up,
    const std::vector<long double>& pi_down,
    const std::vector<long double>& zeta_up,
    const std::vector<long double>& zeta_down,
    const std::vector<double>& orbital_adjoint)
{
  const auto& source = layout.forward.features;
  const auto& target = layout.forward.orbitals;
  long double objective = 0.0L;
  for (std::size_t row = 0; row < 2; ++row)
  {
    const bool up = row == 0;
    const auto& weights = up ? backflow_up : backflow_down;
    const auto& pi = up ? pi_up : pi_down;
    const auto& zeta = up ? zeta_up : zeta_down;
    std::array<long double, 3> displacement{};
    long double radius2 = 0.0L;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      displacement[dimension] = positions[row * 4 + dimension] - nuclei[dimension];
      radius2 += displacement[dimension] * displacement[dimension];
    }
    const long double radius = std::sqrt(radius2);
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t element = pf::open_orbital::outputElement(
          layout.forward, 0, row, orbital);
      std::vector<long double> b(source.plane_count, 0.0L);
      for (std::size_t plane = 0; plane < source.plane_count; ++plane)
        for (std::size_t feature = 0; feature < 2; ++feature)
          b[plane] += features[source.uncheckedPlaneOffset(
                          0, plane,
                          row * layout.forward.feature_row_stride + feature)] *
              weights[pf::open_orbital::weightOffset(
                  layout.forward, feature, 0, orbital)];
      const std::size_t parameter = pf::open_orbital::envelopeOffset(
          layout.forward, 0, orbital, 0);
      const long double a = std::abs(zeta[parameter]);
      const long double decay = std::exp(-a * radius);
      const long double weighted = pi[parameter] * decay;
      std::vector<long double> e(target.plane_count, 0.0L);
      e[0] = weighted;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        e[1 + 3 * row + dimension] =
            -a * weighted * displacement[dimension] / radius;
      e[1 + target.gradient_lanes + row] =
          weighted * (a * a - 2.0L * a / radius);

      objective += orbital_adjoint[target.uncheckedValueOffset(0, element)] * b[0] * e[0];
      for (std::size_t lane = 0; lane < target.gradient_lanes; ++lane)
        objective += orbital_adjoint[target.uncheckedGradientOffset(0, lane, element)] *
            (b[1 + lane] * e[0] + b[0] * e[1 + lane]);
      for (std::size_t electron = 0; electron < target.laplacian_lanes; ++electron)
      {
        long double gradient_dot = 0.0L;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_dot += b[1 + 3 * electron + dimension] *
              e[1 + 3 * electron + dimension];
        objective += orbital_adjoint[target.uncheckedLaplacianOffset(
                         0, electron, element)] *
            (b[1 + target.gradient_lanes + electron] * e[0] +
             2.0L * gradient_dot +
             b[0] * e[1 + target.gradient_lanes + electron]);
      }
    }
  }
  return objective;
}

std::vector<long double> asLongDouble(const std::vector<double>& values)
{
  return {values.begin(), values.end()};
}

} // namespace

TEST_CASE("PsiFormer open orbital lifted reverse matches independent long-double FD",
          "[psiformer][device][kinetic][open]")
{
  const auto layout = makeOrbitalLayout();
  const std::size_t feature_storage = pf::spatialJetSpanElements(layout.forward.features);
  const std::size_t orbital_storage = pf::spatialJetSpanElements(layout.forward.orbitals);
  std::vector<double> features(feature_storage, canary);
  fillFeatures(layout, features);
  const std::vector<double> positions{0.7, -0.2, 0.5, 91.0,
                                      -0.4, 0.8, 1.1, 92.0, 93.0};
  const std::vector<double> nuclei{0.1, 0.2, -0.3, 94.0};
  std::vector<double> backflow_up(pf::openOrbitalBackflowSpan(layout.forward), canary);
  std::vector<double> backflow_down(backflow_up.size(), canary);
  for (std::size_t feature = 0; feature < 2; ++feature)
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t parameter = pf::open_orbital::weightOffset(
          layout.forward, feature, 0, orbital);
      backflow_up[parameter] = 0.13 * (1 + feature + 2 * orbital) - 0.2;
      backflow_down[parameter] = -0.09 * (1 + 2 * feature + orbital) + 0.15;
    }
  std::vector<double> pi_up(pf::openOrbitalEnvelopeSpan(layout.forward), canary);
  std::vector<double> pi_down(pi_up.size(), canary);
  std::vector<double> zeta_up(pi_up.size(), canary);
  std::vector<double> zeta_down(pi_up.size(), canary);
  for (std::size_t orbital = 0; orbital < 2; ++orbital)
  {
    const std::size_t parameter = pf::open_orbital::envelopeOffset(
        layout.forward, 0, orbital, 0);
    pi_up[parameter] = 0.8 - 0.3 * orbital;
    pi_down[parameter] = -0.6 + 0.2 * orbital;
    zeta_up[parameter] = -0.7 - 0.1 * orbital;
    zeta_down[parameter] = orbital == 0 ? 0.0 : 0.45;
  }

  std::vector<double> backflow(orbital_storage, canary);
  std::vector<double> envelope(orbital_storage, canary);
  buildRetainedJets(layout, features, positions, nuclei, backflow_up, backflow_down,
                    pi_up, pi_down, zeta_up, zeta_down, backflow, envelope);
  std::vector<double> orbital_adjoint(orbital_storage, canary);
  for (std::size_t plane = 0; plane < layout.forward.orbitals.plane_count; ++plane)
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t orbital = 0; orbital < 2; ++orbital)
      {
        const std::size_t element = pf::open_orbital::outputElement(
            layout.forward, 0, row, orbital);
        orbital_adjoint[layout.forward.orbitals.uncheckedPlaneOffset(
            0, plane, element)] =
            0.03 * static_cast<double>(1 + orbital + 2 * row + 3 * plane) - 0.2;
      }

  std::vector<double> backflow_adjoint(orbital_storage, canary);
  std::vector<double> envelope_adjoint(orbital_storage, canary);
  std::vector<double> feature_adjoint(feature_storage, canary);
  std::vector<double> response(layout.response_count, canary);
  std::vector<bool> response_logical(layout.response_count, false);
  const std::array<std::size_t, 2> backflow_offsets{
      layout.backflow_up_response_offset, layout.backflow_down_response_offset};
  const std::array<std::size_t, 4> envelope_offsets{
      layout.envelope_pi_up_response_offset, layout.envelope_pi_down_response_offset,
      layout.envelope_zeta_up_response_offset, layout.envelope_zeta_down_response_offset};
  for (std::size_t offset : backflow_offsets)
    for (std::size_t feature = 0; feature < 2; ++feature)
      for (std::size_t orbital = 0; orbital < 2; ++orbital)
      {
        const std::size_t index = offset + pf::open_orbital::weightOffset(
            layout.forward, feature, 0, orbital);
        response[index] = 0.25;
        response_logical[index] = true;
      }
  for (std::size_t offset : envelope_offsets)
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t index = offset + pf::open_orbital::envelopeOffset(
          layout.forward, 0, orbital, 0);
      response[index] = 0.25;
      response_logical[index] = true;
    }
  pf::OpenKineticReverseStatus status = pf::OpenKineticReverseStatus::COALESCENCE;
  CHECK(reverse::reverseOpenOrbitalKineticSerial(
            layout, features.data(), positions.data(), nuclei.data(),
            backflow_up.data(), backflow_down.data(), pi_up.data(), pi_down.data(),
            zeta_up.data(), zeta_down.data(), backflow.data(), envelope.data(),
            orbital_adjoint.data(), backflow_adjoint.data(), envelope_adjoint.data(),
            feature_adjoint.data(), response.data(), &status) ==
        pf::OpenKineticReverseStatus::REGULAR);
  CHECK(status == pf::OpenKineticReverseStatus::REGULAR);

  // Product reverse is checked directly, including the 2*Lbar*gradient cross term.
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t element = pf::open_orbital::outputElement(
          layout.forward, 0, row, orbital);
      const auto& jets = layout.forward.orbitals;
      long double expected_b_value =
          orbital_adjoint[jets.uncheckedValueOffset(0, element)] *
          envelope[jets.uncheckedValueOffset(0, element)];
      long double expected_e_value =
          orbital_adjoint[jets.uncheckedValueOffset(0, element)] *
          backflow[jets.uncheckedValueOffset(0, element)];
      for (std::size_t lane = 0; lane < jets.gradient_lanes; ++lane)
      {
        expected_b_value += orbital_adjoint[jets.uncheckedGradientOffset(0, lane, element)] *
            envelope[jets.uncheckedGradientOffset(0, lane, element)];
        expected_e_value += orbital_adjoint[jets.uncheckedGradientOffset(0, lane, element)] *
            backflow[jets.uncheckedGradientOffset(0, lane, element)];
        const std::size_t electron = lane / 3;
        checkClose(backflow_adjoint[jets.uncheckedGradientOffset(0, lane, element)],
                   orbital_adjoint[jets.uncheckedGradientOffset(0, lane, element)] *
                           envelope[jets.uncheckedValueOffset(0, element)] +
                       2.0L * orbital_adjoint[jets.uncheckedLaplacianOffset(
                                  0, electron, element)] *
                           envelope[jets.uncheckedGradientOffset(0, lane, element)]);
      }
      for (std::size_t electron = 0; electron < jets.laplacian_lanes; ++electron)
      {
        expected_b_value += orbital_adjoint[jets.uncheckedLaplacianOffset(
                                0, electron, element)] *
            envelope[jets.uncheckedLaplacianOffset(0, electron, element)];
        expected_e_value += orbital_adjoint[jets.uncheckedLaplacianOffset(
                                0, electron, element)] *
            backflow[jets.uncheckedLaplacianOffset(0, electron, element)];
      }
      checkClose(backflow_adjoint[jets.uncheckedValueOffset(0, element)], expected_b_value);
      checkClose(envelope_adjoint[jets.uncheckedValueOffset(0, element)], expected_e_value);
    }

  // Treat every retained B/E jet plane as an independent variable and finite
  // difference the complete lifted product objective for one matrix element.
  {
    const auto& jets = layout.forward.orbitals;
    const std::size_t element = pf::open_orbital::outputElement(layout.forward, 0, 1, 0);
    std::vector<long double> b(jets.plane_count);
    std::vector<long double> e(jets.plane_count);
    std::vector<long double> a_bar(jets.plane_count);
    for (std::size_t plane = 0; plane < jets.plane_count; ++plane)
    {
      const std::size_t offset = jets.uncheckedPlaneOffset(0, plane, element);
      b[plane] = backflow[offset];
      e[plane] = envelope[offset];
      a_bar[plane] = orbital_adjoint[offset];
    }
    const auto product_objective = [&](const std::vector<long double>& left,
                                       const std::vector<long double>& right) {
      long double result = a_bar[0] * left[0] * right[0];
      for (std::size_t lane = 0; lane < jets.gradient_lanes; ++lane)
        result += a_bar[1 + lane] *
            (left[1 + lane] * right[0] + left[0] * right[1 + lane]);
      for (std::size_t electron = 0; electron < jets.laplacian_lanes; ++electron)
      {
        long double gradient_dot = 0.0L;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
          gradient_dot += left[1 + 3 * electron + dimension] *
              right[1 + 3 * electron + dimension];
        result += a_bar[1 + jets.gradient_lanes + electron] *
            (left[1 + jets.gradient_lanes + electron] * right[0] +
             2.0L * gradient_dot +
             left[0] * right[1 + jets.gradient_lanes + electron]);
      }
      return result;
    };
    for (std::size_t plane = 0; plane < jets.plane_count; ++plane)
    {
      const long double expected_b = centralDifference([&](long double value) {
        auto varied = b;
        varied[plane] = value;
        return product_objective(varied, e);
      }, b[plane]);
      const long double expected_e = centralDifference([&](long double value) {
        auto varied = e;
        varied[plane] = value;
        return product_objective(b, varied);
      }, e[plane]);
      const std::size_t offset = jets.uncheckedPlaneOffset(0, plane, element);
      checkClose(backflow_adjoint[offset], expected_b);
      checkClose(envelope_adjoint[offset], expected_e);
    }
  }

  auto bf_up_ld = asLongDouble(backflow_up);
  auto bf_down_ld = asLongDouble(backflow_down);
  auto pi_up_ld = asLongDouble(pi_up);
  auto pi_down_ld = asLongDouble(pi_down);
  auto zeta_up_ld = asLongDouble(zeta_up);
  auto zeta_down_ld = asLongDouble(zeta_down);
  const auto objective = [&](const auto& bu, const auto& bd, const auto& pu,
                             const auto& pd, const auto& zu, const auto& zd) {
    return orbitalObjective(layout, features, positions, nuclei, bu, bd, pu, pd,
                            zu, zd, orbital_adjoint);
  };
  for (std::size_t feature = 0; feature < 2; ++feature)
    for (std::size_t orbital = 0; orbital < 2; ++orbital)
    {
      const std::size_t parameter = pf::open_orbital::weightOffset(
          layout.forward, feature, 0, orbital);
      const long double up_expected = centralDifference([&](long double value) {
        auto varied = bf_up_ld;
        varied[parameter] = value;
        return objective(varied, bf_down_ld, pi_up_ld, pi_down_ld, zeta_up_ld,
                         zeta_down_ld);
      }, bf_up_ld[parameter]);
      const long double down_expected = centralDifference([&](long double value) {
        auto varied = bf_down_ld;
        varied[parameter] = value;
        return objective(bf_up_ld, varied, pi_up_ld, pi_down_ld, zeta_up_ld,
                         zeta_down_ld);
      }, bf_down_ld[parameter]);
      checkClose(response[layout.backflow_up_response_offset + parameter] - 0.25,
                 up_expected);
      checkClose(response[layout.backflow_down_response_offset + parameter] - 0.25,
                 down_expected);
    }
  for (std::size_t orbital = 0; orbital < 2; ++orbital)
  {
    const std::size_t parameter = pf::open_orbital::envelopeOffset(
        layout.forward, 0, orbital, 0);
    const long double pi_up_expected = centralDifference([&](long double value) {
      auto varied = pi_up_ld;
      varied[parameter] = value;
      return objective(bf_up_ld, bf_down_ld, varied, pi_down_ld, zeta_up_ld,
                       zeta_down_ld);
    }, pi_up_ld[parameter]);
    const long double pi_down_expected = centralDifference([&](long double value) {
      auto varied = pi_down_ld;
      varied[parameter] = value;
      return objective(bf_up_ld, bf_down_ld, pi_up_ld, varied, zeta_up_ld,
                       zeta_down_ld);
    }, pi_down_ld[parameter]);
    const long double zeta_up_expected = centralDifference([&](long double value) {
      auto varied = zeta_up_ld;
      varied[parameter] = value;
      return objective(bf_up_ld, bf_down_ld, pi_up_ld, pi_down_ld, varied,
                       zeta_down_ld);
    }, zeta_up_ld[parameter]);
    checkClose(response[layout.envelope_pi_up_response_offset + parameter] - 0.25,
               pi_up_expected);
    checkClose(response[layout.envelope_pi_down_response_offset + parameter] - 0.25,
               pi_down_expected);
    checkClose(response[layout.envelope_zeta_up_response_offset + parameter] - 0.25,
               zeta_up_expected);
    if (zeta_down_ld[parameter] == 0.0L)
      CHECK(response[layout.envelope_zeta_down_response_offset + parameter] == 0.25);
    else
    {
      const long double expected = centralDifference([&](long double value) {
        auto varied = zeta_down_ld;
        varied[parameter] = value;
        return objective(bf_up_ld, bf_down_ld, pi_up_ld, pi_down_ld, zeta_up_ld,
                         varied);
      }, zeta_down_ld[parameter]);
      checkClose(response[layout.envelope_zeta_down_response_offset + parameter] - 0.25,
                 expected);
    }
  }

  for (std::size_t index = 0; index < response.size(); ++index)
    if (!response_logical[index])
      CHECK(response[index] == canary);
  for (std::size_t plane = 0; plane < layout.forward.features.plane_count; ++plane)
    CHECK(feature_adjoint[layout.forward.features.uncheckedPlaneOffset(0, plane, 2)] ==
          canary);
  for (std::size_t plane = 0; plane < layout.forward.orbitals.plane_count; ++plane)
    CHECK(backflow_adjoint[
              layout.forward.orbitals.uncheckedPlaneOffset(0, plane, 2)] == canary);
}

TEST_CASE("PsiFormer open cusp lifted alpha reverse matches long-double FD",
          "[psiformer][device][kinetic][open]")
{
  pf::OpenCuspKineticReverseLayout layout{};
  layout.forward = pf::makeOpenCuspJetLayout(
      1, 3, 2, pf::SpatialJetMode::FULL_VGL, 4, 13, 3, 0);
  layout.response_count = 14;
  layout.same_response_offset = 5;
  layout.opposite_response_offset = 10;
  layout.has_same_alpha = true;
  pf::validateOpenCuspKineticReverseLayout(layout);
  const std::vector<double> positions{0.2, -0.4, 0.7, 91.0,
                                      1.1, 0.3, -0.2, 92.0,
                                      -0.6, 0.8, 1.4, 93.0, 94.0};
  std::vector<double> cusp_adjoint(pf::spatialJetSpanElements(layout.forward.output), canary);
  for (std::size_t plane = 0; plane < layout.forward.output.plane_count; ++plane)
    cusp_adjoint[layout.forward.output.uncheckedPlaneOffset(0, plane, 0)] =
        0.11 * static_cast<double>(plane + 1) - 0.35;
  std::vector<double> response(layout.response_count, canary);
  response[layout.same_response_offset] = -0.2;
  response[layout.opposite_response_offset] = -0.2;
  pf::OpenKineticReverseStatus status{};
  constexpr double same_alpha = 0.9;
  constexpr double opposite_alpha = 1.3;
  CHECK(reverse::reverseOpenCuspKineticSerial(
            layout, positions.data(), same_alpha, opposite_alpha,
            cusp_adjoint.data(), response.data(), &status) ==
        pf::OpenKineticReverseStatus::REGULAR);

  const auto objective = [&](long double same, long double opposite) {
    std::vector<long double> jet(layout.forward.output.plane_count, 0.0L);
    for (std::size_t first = 0; first < 3; ++first)
      for (std::size_t second = first + 1; second < 3; ++second)
      {
        const bool same_spin = (first < 2) == (second < 2);
        const long double alpha = same_spin ? same : opposite;
        const long double factor = same_spin ? 0.25L : 0.5L;
        std::array<long double, 3> displacement{};
        long double radius2 = 0.0L;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          displacement[dimension] = positions[first * 4 + dimension] -
              positions[second * 4 + dimension];
          radius2 += displacement[dimension] * displacement[dimension];
        }
        const long double radius = std::sqrt(radius2);
        const long double denominator = alpha + radius;
        const long double radial_first = factor * alpha * alpha /
            (denominator * denominator);
        const long double radial_second = -2.0L * factor * alpha * alpha /
            (denominator * denominator * denominator);
        jet[0] -= factor * alpha * alpha / denominator;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const long double component =
              radial_first * displacement[dimension] / radius;
          jet[1 + 3 * first + dimension] += component;
          jet[1 + 3 * second + dimension] -= component;
        }
        const long double laplacian = radial_second + 2.0L * radial_first / radius;
        jet[1 + layout.forward.output.gradient_lanes + first] += laplacian;
        jet[1 + layout.forward.output.gradient_lanes + second] += laplacian;
      }
    long double result = 0.0L;
    for (std::size_t plane = 0; plane < jet.size(); ++plane)
      result += jet[plane] * cusp_adjoint[
          layout.forward.output.uncheckedPlaneOffset(0, plane, 0)];
    return result;
  };
  checkClose(response[layout.same_response_offset] + 0.2,
             centralDifference([&](long double value) {
               return objective(value, opposite_alpha);
             }, same_alpha));
  checkClose(response[layout.opposite_response_offset] + 0.2,
             centralDifference([&](long double value) {
               return objective(same_alpha, value);
             }, opposite_alpha));
  for (std::size_t index = 0; index < response.size(); ++index)
    if (index != layout.same_response_offset &&
        index != layout.opposite_response_offset)
      CHECK(response[index] == canary);
}

TEST_CASE("PsiFormer open reverse validates regions and fails closed without padding writes",
          "[psiformer][device][kinetic][open]")
{
  pf::OpenOrbitalKineticReverseLayout zero{};
  pf::validateOpenOrbitalKineticReverseLayout(zero);
  CHECK(reverse::reverseOpenOrbitalKineticSerial(
            zero, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr) == pf::OpenKineticReverseStatus::REGULAR);

  auto layout = makeOrbitalLayout();
  auto overlapping = layout;
  overlapping.backflow_down_response_offset = overlapping.backflow_up_response_offset + 1;
  CHECK_THROWS_AS(pf::validateOpenOrbitalKineticReverseLayout(overlapping),
                  std::invalid_argument);
  auto out_of_bounds = layout;
  out_of_bounds.envelope_zeta_down_response_offset = out_of_bounds.response_count - 1;
  CHECK_THROWS_AS(pf::validateOpenOrbitalKineticReverseLayout(out_of_bounds),
                  std::out_of_range);
  auto active = layout;
  active.forward.features.mode = pf::SpatialJetMode::ACTIVE;
  CHECK_THROWS_AS(pf::validateOpenOrbitalKineticReverseLayout(active),
                  std::invalid_argument);

  const std::size_t feature_storage = pf::spatialJetSpanElements(layout.forward.features);
  const std::size_t orbital_storage = pf::spatialJetSpanElements(layout.forward.orbitals);
  std::vector<double> features(feature_storage, 0.1);
  std::vector<double> positions{0.1, 0.2, 0.3, canary,
                                0.4, 0.5, 0.6, canary, canary};
  std::vector<double> nuclei{0.1, 0.2, 0.3, canary};
  std::vector<double> backflow_parameter(pf::openOrbitalBackflowSpan(layout.forward), 0.2);
  std::vector<double> envelope_parameter(pf::openOrbitalEnvelopeSpan(layout.forward), 0.4);
  std::vector<double> retained(orbital_storage, 0.3);
  std::vector<double> orbital_adjoint(orbital_storage, 0.5);
  std::vector<double> backflow_adjoint(orbital_storage, canary);
  std::vector<double> envelope_adjoint(orbital_storage, canary);
  std::vector<double> feature_adjoint(feature_storage, canary);
  std::vector<double> response(layout.response_count, canary);
  pf::OpenKineticReverseStatus status{};
  CHECK(reverse::reverseOpenOrbitalKineticSerial(
            layout, features.data(), positions.data(), nuclei.data(),
            backflow_parameter.data(), backflow_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(), retained.data(),
            retained.data(), orbital_adjoint.data(), backflow_adjoint.data(),
            envelope_adjoint.data(), feature_adjoint.data(), response.data(), &status) ==
        pf::OpenKineticReverseStatus::COALESCENCE);
  CHECK(status == pf::OpenKineticReverseStatus::COALESCENCE);
  for (std::size_t plane = 0; plane < layout.forward.orbitals.plane_count; ++plane)
  {
    CHECK(backflow_adjoint[
              layout.forward.orbitals.uncheckedPlaneOffset(0, plane, 0)] == 0.0);
    CHECK(backflow_adjoint[
              layout.forward.orbitals.uncheckedPlaneOffset(0, plane, 2)] == canary);
  }
  for (std::size_t plane = 0; plane < layout.forward.features.plane_count; ++plane)
  {
    for (std::size_t row = 0; row < layout.forward.features.electron_count; ++row)
      for (std::size_t feature = 0; feature < layout.forward.feature_width; ++feature)
        CHECK(feature_adjoint[layout.forward.features.uncheckedPlaneOffset(
                  0, plane, row * layout.forward.feature_row_stride + feature)] == 0.0);
    CHECK(feature_adjoint[layout.forward.features.uncheckedPlaneOffset(0, plane, 2)] ==
          canary);
  }
  CHECK(response[layout.backflow_up_response_offset] == 0.0);
  CHECK(response[layout.backflow_up_response_offset + 2] == canary);
  CHECK(response[0] == canary);

  // A late non-finite parameter failure also rolls back every logical output,
  // while retaining response and jet padding canaries.
  positions[0] = 0.8;
  envelope_parameter[0] = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t index = 0; index < response.size(); ++index)
    if (index != layout.backflow_up_response_offset + 2)
      response[index] = canary;
  response[layout.backflow_up_response_offset] = 1.0;
  CHECK(reverse::reverseOpenOrbitalKineticSerial(
            layout, features.data(), positions.data(), nuclei.data(),
            backflow_parameter.data(), backflow_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(), retained.data(),
            retained.data(), orbital_adjoint.data(), backflow_adjoint.data(),
            envelope_adjoint.data(), feature_adjoint.data(), response.data(), &status) ==
        pf::OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT);
  CHECK(status == pf::OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT);
  CHECK(response[layout.backflow_up_response_offset] == 0.0);
  CHECK(response[layout.backflow_up_response_offset + 2] == canary);

  // A finite product can overflow an envelope Laplacian adjoint in a lane that
  // does not belong to the orbital's row electron.  That lane is not consumed by
  // the envelope-parameter reverse, so the product leaf must diagnose it directly.
  std::fill(envelope_parameter.begin(), envelope_parameter.end(), 0.4);
  std::vector<double> overflow_backflow(orbital_storage, 0.0);
  std::vector<double> overflow_envelope(orbital_storage, 0.0);
  std::vector<double> overflow_upstream(orbital_storage, 0.0);
  const std::size_t overflow_element =
      pf::open_orbital::outputElement(layout.forward, 0, 0, 0);
  overflow_backflow[layout.forward.orbitals.uncheckedValueOffset(
      0, overflow_element)] = 2.0;
  overflow_upstream[layout.forward.orbitals.uncheckedLaplacianOffset(
      0, 1, overflow_element)] = std::numeric_limits<double>::max();
  std::fill(backflow_adjoint.begin(), backflow_adjoint.end(), canary);
  std::fill(envelope_adjoint.begin(), envelope_adjoint.end(), canary);
  std::fill(feature_adjoint.begin(), feature_adjoint.end(), canary);
  std::fill(response.begin(), response.end(), canary);
  response[layout.backflow_up_response_offset] = 1.0;
  CHECK(reverse::reverseOpenOrbitalKineticSerial(
            layout, features.data(), positions.data(), nuclei.data(),
            backflow_parameter.data(), backflow_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(),
            envelope_parameter.data(), envelope_parameter.data(),
            overflow_backflow.data(), overflow_envelope.data(),
            overflow_upstream.data(), backflow_adjoint.data(),
            envelope_adjoint.data(), feature_adjoint.data(), response.data(),
            &status) == pf::OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT);
  CHECK(status == pf::OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT);
  CHECK(envelope_adjoint[layout.forward.orbitals.uncheckedLaplacianOffset(
            0, 1, overflow_element)] == 0.0);
  CHECK(envelope_adjoint[layout.forward.orbitals.uncheckedLaplacianOffset(
            0, 1, 2)] == canary);
  CHECK(response[layout.backflow_up_response_offset] == 0.0);
  CHECK(response[layout.backflow_up_response_offset + 2] == canary);

  pf::OpenCuspKineticReverseLayout cusp{};
  cusp.forward = pf::makeOpenCuspJetLayout(
      1, 2, 1, pf::SpatialJetMode::FULL_VGL, 4, 9, 3, 0);
  cusp.response_count = 8;
  cusp.same_response_offset = 2;
  cusp.opposite_response_offset = 6;
  cusp.has_same_alpha = true;
  pf::validateOpenCuspKineticReverseLayout(cusp);
  std::vector<double> cusp_adjoint(pf::spatialJetSpanElements(cusp.forward.output), 0.4);
  std::vector<double> cusp_response(cusp.response_count, canary);
  cusp_response[2] = 1.0;
  cusp_response[6] = 1.0;
  const std::vector<double> coalesced_positions{0.2, 0.3, 0.4, canary,
                                                 0.2, 0.3, 0.4, canary, canary};
  CHECK(reverse::reverseOpenCuspKineticSerial(
            cusp, coalesced_positions.data(), 0.7, 1.1, cusp_adjoint.data(),
            cusp_response.data(), &status) == pf::OpenKineticReverseStatus::COALESCENCE);
  CHECK(cusp_response[2] == 0.0);
  CHECK(cusp_response[6] == 0.0);
  CHECK(cusp_response[3] == canary);
  cusp_response[2] = 1.0;
  cusp_response[6] = 1.0;
  CHECK(reverse::reverseOpenCuspKineticSerial(
            cusp, coalesced_positions.data(), 0.7,
            std::numeric_limits<double>::infinity(), cusp_adjoint.data(),
            cusp_response.data(), &status) ==
        pf::OpenKineticReverseStatus::NONFINITE_INPUT_OR_RESULT);
  CHECK(cusp_response[2] == 0.0);
  CHECK(cusp_response[6] == 0.0);
}

TEST_CASE("PsiFormer open reverse rejects every output alias class before launch",
          "[psiformer][device][kinetic][open]")
{
  const auto layout = makeOrbitalLayout();
  std::vector<double> features(pf::spatialJetSpanElements(layout.forward.features));
  std::vector<double> positions(layout.forward.position_configuration_stride);
  std::vector<double> nuclei(layout.forward.nucleus_row_stride);
  std::vector<double> backflow_parameter(pf::openOrbitalBackflowSpan(layout.forward));
  std::vector<double> envelope_parameter(pf::openOrbitalEnvelopeSpan(layout.forward));
  std::vector<double> retained(pf::spatialJetSpanElements(layout.forward.orbitals));
  std::vector<double> backflow_adjoint(retained.size());
  std::vector<double> envelope_adjoint(retained.size());
  std::vector<double> feature_adjoint(features.size());
  std::vector<double> response(layout.response_count);
  pf::OpenKineticReverseStatus status{};

  const auto launch_orbital = [&](double* backflow_output, double* envelope_output,
                                  double* feature_output, double* response_output,
                                  pf::OpenKineticReverseStatus* status_output) {
    pf::validateOpenOrbitalKineticReverseNoAlias(
        features.data(), positions.data(), nuclei.data(),
        backflow_parameter.data(), backflow_parameter.data(),
        envelope_parameter.data(), envelope_parameter.data(),
        envelope_parameter.data(), envelope_parameter.data(), retained.data(),
        retained.data(), retained.data(), backflow_output, envelope_output,
        feature_output, response_output, status_output);
  };

  CHECK_NOTHROW(launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                               feature_adjoint.data(), response.data(), &status));

  // Cover each output kind against a read buffer.
  CHECK_THROWS_AS(launch_orbital(features.data(), envelope_adjoint.data(),
                                 feature_adjoint.data(), response.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(launch_orbital(backflow_adjoint.data(), positions.data(),
                                 feature_adjoint.data(), response.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                                 backflow_parameter.data(), response.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                                 feature_adjoint.data(), envelope_parameter.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                     feature_adjoint.data(), response.data(),
                     reinterpret_cast<pf::OpenKineticReverseStatus*>(retained.data())),
      std::invalid_argument);

  // Cover output/output aliases, including the differently typed status output.
  CHECK_THROWS_AS(launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                                 backflow_adjoint.data(), response.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      launch_orbital(backflow_adjoint.data(), envelope_adjoint.data(),
                     feature_adjoint.data(), response.data(),
                     reinterpret_cast<pf::OpenKineticReverseStatus*>(response.data())),
      std::invalid_argument);

  pf::OpenCuspKineticReverseLayout cusp{};
  cusp.forward = pf::makeOpenCuspJetLayout(
      1, 2, 1, pf::SpatialJetMode::FULL_VGL, 4, 9, 3, 0);
  cusp.response_count = 8;
  cusp.same_response_offset = 2;
  cusp.opposite_response_offset = 6;
  cusp.has_same_alpha = true;
  pf::validateOpenCuspKineticReverseLayout(cusp);
  std::vector<double> cusp_positions(cusp.forward.position_configuration_stride);
  std::vector<double> cusp_adjoint(pf::spatialJetSpanElements(cusp.forward.output));
  std::vector<double> cusp_response(cusp.response_count);

  CHECK_NOTHROW(pf::validateOpenCuspKineticReverseNoAlias(
      cusp_positions.data(), cusp_adjoint.data(), cusp_response.data(), &status));

  CHECK_THROWS_AS(pf::validateOpenCuspKineticReverseNoAlias(
                      cusp_positions.data(), cusp_adjoint.data(),
                      cusp_positions.data(), &status),
                  std::invalid_argument);
  CHECK_THROWS_AS(pf::validateOpenCuspKineticReverseNoAlias(
                      cusp_positions.data(), cusp_adjoint.data(), cusp_response.data(),
                      reinterpret_cast<pf::OpenKineticReverseStatus*>(
                          cusp_adjoint.data())),
                  std::invalid_argument);
  CHECK_THROWS_AS(pf::validateOpenCuspKineticReverseNoAlias(
                      cusp_positions.data(), cusp_adjoint.data(), cusp_response.data(),
                      reinterpret_cast<pf::OpenKineticReverseStatus*>(
                          cusp_response.data())),
                  std::invalid_argument);
}

// Keep this file-boundary oracle independent of accelerator initialization so it
// remains runnable in CUDA/HIP builds on driverless hosts.
int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
