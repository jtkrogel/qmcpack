//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDenseKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalProjection.h"

#include <cmath>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

// Execute the public row-major slice through the same CPU BLAS ABI as the trunk.
void projectSlice(const OpenOrbitalProjectionSlice& slice, const double* features,
    const double* up, const double* down, double* target)
{
  const auto& d = slice.dense;
  const auto as_int = [](std::size_t value) {
    return dense::checkedBlasDimension(value, "Orbital test exceeds BLAS ABI");
  };
  BLAS::gemm('N', 'N', as_int(d.output_width), as_int(d.rows), as_int(d.input_width), 1.0,
      (slice.spin_up ? up : down) + slice.weight_offset, as_int(d.weight_row_stride),
      features + slice.source_offset, as_int(d.source_row_stride), 0.0,
      target + slice.target_offset, as_int(d.target_row_stride));
}

} // namespace

TEST_CASE("PsiFormer orbital GEMM preserves padded spin and plane products",
    "[wavefunction][psiformer][orbital_projection]")
{
  SpatialJetMode mode = SpatialJetMode::FULL_VGL;
  SECTION("full VGL") {}
  SECTION("active gradient") { mode = SpatialJetMode::ACTIVE; }
  SECTION("value") { mode = SpatialJetMode::VALUE_ONLY; }

  // Asymmetric spins, two channels, and distinct padding at every tensor level
  // expose determinant/row transpositions and accidental padding contractions.
  auto layout = makeOpenOrbitalJetLayout(2, 2, 3, 1, 5, 1, mode,
      7, 5, 17, 4, 11, 2, 7, 3, 11, 3, 23, 401, 37, 601);
  constexpr double sentinel = 12345.0;
  std::vector<double> features(2 * layout.features.configuration_stride, sentinel);
  std::vector<double> up(openOrbitalBackflowSpan(layout), sentinel), down(up.size(), sentinel);
  std::vector<double> pi(openOrbitalEnvelopeSpan(layout), 0.7), zeta(pi.size(), -0.8);
  std::vector<double> reference(2 * layout.orbitals.configuration_stride, sentinel);
  auto product = reference, projected = reference, backflow = reference, envelope = reference;
  std::vector<bool> touched(reference.size(), false);
  const double positions[]{0.2, 0.3, 0.4, 0.6, -0.3, 0.4, -0.4, 0.3, 0.8, 0, 0,
                           0.5, -0.2, 0.4, 0.3, 0.7, -0.6, 0.4, -0.8, 0.3, 0, 0};
  const double nuclei[]{0.1, -0.1, 0.2};

  // A finite zero envelope exercises factor separation without division.
  pi[open_orbital::envelopeOffset(layout, 0, 0, 0)] = 0.0;
  for (std::size_t f = 0; f < 5; ++f)
    for (std::size_t d = 0; d < 2; ++d)
      for (std::size_t o = 0; o < 3; ++o)
      {
        const auto offset = open_orbital::weightOffset(layout, f, d, o);
        up[offset] = std::sin(0.2 + 0.4 * f - 0.3 * d + 0.6 * o);
        down[offset] = std::cos(0.7 - 0.5 * f + 0.8 * d - 0.2 * o);
      }
  for (std::size_t c = 0; c < 2; ++c)
    for (std::size_t p = 0; p < layout.features.plane_count; ++p)
      for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t f = 0; f < 5; ++f)
          features[layout.features.uncheckedPlaneOffset(c, p, r * 7 + f)] =
              std::sin(0.1 + c + 0.13 * p - 0.3 * r + 0.07 * f);

  std::size_t slices = 0;
  forEachOpenOrbitalProjection(layout, [&](const OpenOrbitalProjectionSlice& slice) {
    ++slices;
    projectSlice(slice, features.data(), up.data(), down.data(), projected.data());
  });
  CHECK(slices == 2 * 2 * 2 * layout.features.plane_count);

  for (std::size_t c = 0; c < 2; ++c)
    for (std::size_t d = 0; d < 2; ++d)
      for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t o = 0; o < 3; ++o)
        {
          const auto element = open_orbital::outputElement(layout, d, r, o);
          REQUIRE(open_orbital::buildElement(layout, features.data(), positions, nuclei,
              up.data(), down.data(), pi.data(), pi.data(), zeta.data(), zeta.data(),
              c, 2 - c, d, r, o, reference.data(), backflow.data(), envelope.data()) ==
              OpenSpatialStatus::REGULAR);
          REQUIRE(open_orbital::buildElement(layout, nullptr, positions, nuclei,
              nullptr, nullptr, pi.data(), pi.data(), zeta.data(), zeta.data(),
              c, 2 - c, d, r, o, product.data(), projected.data(), nullptr, nullptr,
              projected.data()) == OpenSpatialStatus::REGULAR);
          for (std::size_t p = 0; p < layout.orbitals.plane_count; ++p)
          {
            const auto offset = layout.orbitals.uncheckedPlaneOffset(c, p, element);
            touched[offset] = true;
            CHECK(projected[offset] == Approx(backflow[offset]).margin(2e-13));
            CHECK(product[offset] == Approx(reference[offset]).margin(2e-13));
          }
        }

  // Both the affine BLAS pass and product pass must leave every padding slot alone.
  for (std::size_t i = 0; i < touched.size(); ++i)
    if (!touched[i])
    {
      CHECK(projected[i] == sentinel);
      CHECK(product[i] == sentinel);
    }
}

TEST_CASE("PsiFormer orbital projection omits empty spin groups",
    "[wavefunction][psiformer][orbital_projection]")
{
  auto layout = makeOpenOrbitalJetLayout(1, 2, 3, 1, 5, 0, SpatialJetMode::VALUE_ONLY);
  std::size_t slices = 0;
  forEachOpenOrbitalProjection(layout, [&](const OpenOrbitalProjectionSlice& slice) {
    ++slices;
    CHECK_FALSE(slice.spin_up);
    CHECK(slice.dense.rows == 3);
  });
  CHECK(slices == 2);
}

} // namespace qmcplusplus::psiformer

// Match the neighboring standalone algebra tests without application linkage.
int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
