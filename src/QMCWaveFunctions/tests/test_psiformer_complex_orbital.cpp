//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_orbital.cpp
 * @brief Generated real-trunk/complex-head/determinant CPU value and VGL checks.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexOrbital.h"
#include <iostream>
#include <string>

namespace pf = qmcplusplus::psiformer;
namespace det = pf::determinant;
using C = std::complex<double>;
int checks = 0, failures = 0;

/// Record a standalone assertion without linking the full wavefunction library.
void check(bool value, const std::string& message)
{
  ++checks;
  if (!value)
  {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

/// Compare analytic and finite-difference complex quantities at fixture tolerances.
bool near(C a, C b, double tolerance = 2e-8)
{
  return std::abs(a - b) < tolerance * (1 + std::abs(b));
}

/// Require failure for a malformed request or unavailable generation.
template<class F>
void throws(F function, const std::string& message)
{
  bool caught = false;
  try
  {
    function();
  }
  catch (const std::exception&)
  {
    caught = true;
  }
  check(caught, message);
}

/// Materialize the tiny generated wavefunction values used in finite differences.
C value(const det::ComplexLogAmplitude& amplitude)
{
  return amplitude.isZero() ? C{} : amplitude.phase * std::exp(amplitude.log_abs);
}

/// Real analytic scalar jets for test-only generated periodic features/envelopes.
struct ScalarJet
{
  double value = 0;
  std::array<double, 9> gradient{};
  std::array<double, 3> laplacian{};
};

/// Generate amplitude+sin(linear phase) jets without using production head algebra.
ScalarJet sinusoid(const std::vector<double>& positions, std::size_t configuration,
    const std::array<double, 9>& coefficients, double offset, double amplitude)
{
  double argument = 0;
  for (std::size_t i = 0; i < 9; ++i)
    argument += positions[configuration * 9 + i] * coefficients[i];
  ScalarJet result;
  result.value = offset + amplitude * std::sin(argument);
  for (std::size_t i = 0; i < 9; ++i)
  {
    result.gradient[i] = amplitude * std::cos(argument) * coefficients[i];
    result.laplacian[i / 3] -= amplitude * std::sin(argument) * coefficients[i] * coefficients[i];
  }
  return result;
}

/// Store physical scalar jets through either ACTIVE or compact target-range layouts.
void store(const pf::SpatialJetLayout& layout, std::vector<double>& storage,
    std::size_t configuration, std::size_t element, const ScalarJet& jet,
    std::size_t active)
{
  storage[layout.uncheckedValueOffset(configuration, element)] = jet.value;
  for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
  {
    const std::size_t target = layout.mode == pf::SpatialJetMode::ACTIVE ? active :
        layout.physicalTargetElectron(lane / 3);
    storage[layout.uncheckedGradientOffset(configuration, lane, element)] = jet.gradient[3 * target + lane % 3];
  }
  for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
    storage[layout.uncheckedLaplacianOffset(configuration, target, element)] =
        jet.laplacian[layout.physicalTargetElectron(target)];
}

/// Tiny shared real-trunk fixture: all physical electrons influence each feature row.
struct Fixture
{
  pf::ComplexOrbitalHeadDescriptor descriptor;
  pf::SpatialJetLayout feature_layout, envelope_layout;
  std::vector<double> positions, features, envelopes;
  std::array<std::vector<double>, 2> wr, wi, br, bi;
  std::size_t active = 2;

  /// Construct full or spin-blocked matrices, batched and compactly target-tiled.
  Fixture(bool blocked, pf::SpatialJetMode mode, std::size_t begin = 0, std::size_t count = 3,
      bool inner_phase = true)
  {
    descriptor.electrons = 3;
    descriptor.spin_up = 2;
    descriptor.determinants = 2;
    descriptor.feature_width = 3;
    descriptor.structure = blocked ? pf::ComplexOrbitalStructure::SPIN_BLOCKED : pf::ComplexOrbitalStructure::FULL;
    descriptor.phase = inner_phase ? pf::ComplexOrbitalPhase::RECIPROCAL_ONLY : pf::ComplexOrbitalPhase::NONE;
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const std::size_t columns = blocked ? (spin ? 1 : 2) : 3, bias_count = 2 * columns;
      wr[spin].resize(3 * bias_count);
      wi[spin].resize(3 * bias_count);
      br[spin].resize(bias_count);
      bi[spin].resize(bias_count);
      for (std::size_t i = 0; i < 3 * bias_count; ++i)
      {
        wr[spin][i] = std::sin(.73 * (i + 1 + spin));
        wi[spin][i] = .4 * std::cos(.41 * (i + 2 + spin));
      }
      for (std::size_t i = 0; i < bias_count; ++i)
      {
        br[spin][i] = .2 * std::cos(.31 * (i + 1));
        bi[spin][i] = .1 * std::sin(.29 * (i + 1));
        if (inner_phase)
          descriptor.reciprocal_momenta.push_back({.7 * (static_cast<double>(i % columns) - 1),
              .7 * static_cast<double>(spin), -.7 * static_cast<double>(i / columns)});
      }
    }
    const std::size_t orbital_elements = blocked ? 10 : 18;
    feature_layout = mode == pf::SpatialJetMode::FULL_VGL ?
        pf::makeSpatialTargetJetLayout(2, 9, 3, begin, count, 11) :
        pf::makeSpatialJetLayout(2, 9, 3, mode, 11);
    envelope_layout = mode == pf::SpatialJetMode::FULL_VGL ?
        pf::makeSpatialTargetJetLayout(2, orbital_elements, 3, begin, count, orbital_elements + 2) :
        pf::makeSpatialJetLayout(2, orbital_elements, 3, mode, orbital_elements + 2);
    positions.resize(18);
    for (std::size_t i = 0; i < positions.size(); ++i)
      positions[i] = .17 + .19 * i;
    features.resize(pf::spatialJetSpanElements(feature_layout));
    envelopes.resize(pf::spatialJetSpanElements(envelope_layout));
    refresh();
  }

  /// Expose paired real arrays with exact per-spin parameter spans.
  pf::ComplexOrbitalHeadParameters parameters() const
  {
    pf::ComplexOrbitalHeadParameters result;
    result.up = {wr[0].data(), wi[0].data(), br[0].data(), bi[0].data(), wr[0].size(), br[0].size()};
    result.down = {wr[1].data(), wi[1].data(), br[1].data(), bi[1].data(), wr[1].size(), br[1].size()};
    return result;
  }

  /// Recompute real jets from physical positions; envelope jets remain row-local.
  void refresh()
  {
    for (std::size_t configuration = 0; configuration < 2; ++configuration)
    {
      for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t f = 0; f < 3; ++f)
        {
          std::array<double, 9> coefficients{};
          for (std::size_t i = 0; i < 9; ++i)
            coefficients[i] = .07 * (1 + ((i + 2 * f + row * f) % 5));
          coefficients[3 * row + f] += .8;
          store(feature_layout, features, configuration, row * 3 + f,
                sinusoid(positions, configuration, coefficients, .3 + .1 * f, 1), active);
        }
      const bool blocked = descriptor.structure == pf::ComplexOrbitalStructure::SPIN_BLOCKED;
      const std::size_t stride = envelope_layout.element_count / 2;
      for (std::size_t d = 0; d < 2; ++d)
        for (std::size_t row = 0; row < 3; ++row)
          for (std::size_t orbital = 0; orbital < (blocked ? (row < 2 ? 2 : 1) : 3); ++orbital)
          {
            const std::size_t element = d * stride + (blocked ? (row < 2 ? row * 2 + orbital : 4) : row * 3 + orbital);
            std::array<double, 9> coefficients{};
            coefficients[3 * row] = .7 * (orbital + 1);
            coefficients[3 * row + 1] = -.7 * (d + 1);
            store(envelope_layout, envelopes, configuration, element,
                  sinusoid(positions, configuration, coefficients, 1.2, .2), active);
          }
    }
  }

  /// Execute the production helper on the fixture's current real jets and parameters.
  void evaluate(pf::ComplexOrbitalHeadWorkspace& head) const
  {
    head.evaluate(features.data(), features.size(), envelopes.data(), envelopes.size(),
                  positions.data(), positions.size(), parameters(), active);
  }
};

/// Independently expand a tiny complex matrix determinant by permutations.
C permutationDeterminant(const C* matrix, std::size_t n)
{
  std::vector<std::size_t> permutation(n);
  for (std::size_t i = 0; i < n; ++i)
    permutation[i] = i;
  C result{};
  do
  {
    C term{1};
    std::size_t inversions = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      term *= matrix[i * n + permutation[i]];
      for (std::size_t j = 0; j < i; ++j)
        inversions += permutation[j] > permutation[i];
    }
    result += inversions % 2 ? -term : term;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return result;
}

/// Evaluate physical amplitudes through direct scalar projections and permutation sums.
C referenceValue(const Fixture& f, std::size_t configuration)
{
  const bool blocked = f.descriptor.structure == pf::ComplexOrbitalStructure::SPIN_BLOCKED;
  const std::size_t stride = f.envelope_layout.element_count / 2;
  std::vector<C> matrices(f.envelope_layout.element_count);
  for (std::size_t d = 0; d < 2; ++d)
    for (std::size_t row = 0; row < 3; ++row)
    {
      const std::size_t spin = row < 2 ? 0 : 1, columns = blocked ? (spin ? 1 : 2) : 3;
      for (std::size_t orbital = 0; orbital < columns; ++orbital)
      {
        const std::size_t index = d * columns + orbital;
        C projection{f.br[spin][index], f.bi[spin][index]};
        for (std::size_t feature = 0; feature < 3; ++feature)
          projection += C{f.wr[spin][feature * 2 * columns + index], f.wi[spin][feature * 2 * columns + index]} *
              f.features[f.feature_layout.uncheckedValueOffset(configuration, row * 3 + feature)];
        double phase = 0;
        if (f.descriptor.phase == pf::ComplexOrbitalPhase::RECIPROCAL_ONLY)
        {
          const auto& momentum = f.descriptor.reciprocal_momenta[(spin ? (blocked ? 4 : 6) : 0) + index];
          for (std::size_t dim = 0; dim < 3; ++dim)
            phase += momentum[dim] * f.positions[configuration * 9 + row * 3 + dim];
        }
        const std::size_t element = d * stride + (blocked ? (spin ? 4 : row * 2 + orbital) : row * 3 + orbital);
        matrices[element] = projection * f.envelopes[f.envelope_layout.uncheckedValueOffset(configuration, element)] *
            std::exp(C{0, phase});
      }
    }
  C result{};
  for (std::size_t d = 0; d < 2; ++d)
    result += blocked ? permutationDeterminant(matrices.data() + d * stride, 2) * matrices[d * stride + 4] :
        permutationDeterminant(matrices.data() + d * stride, 3);
  return result;
}

/// Compare HEAD->determinant spatial results to independent amplitude differences.
void spatialFixture(bool blocked, pf::SpatialJetMode mode, std::size_t begin = 0, std::size_t count = 3)
{
  Fixture f(blocked, mode, begin, count);
  pf::ComplexOrbitalHeadWorkspace head(f.descriptor, f.feature_layout, f.envelope_layout);
  f.evaluate(head);
  const auto fingerprint = head.storageFingerprint();
  const auto layout = head.layout();
  det::ComplexDeterminantWorkspace determinant(2, blocked ? 2 : 3, blocked ? 1 : 0,
      layout.gradient_lanes, layout.laplacian_lanes);
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    f.evaluate(head);
    const C* matrices = head.data() + configuration * layout.configuration_stride;
    C gradient[9]{}, log[3]{}, ratio[3]{};
    const auto result = mode == pf::SpatialJetMode::VALUE_ONLY ? determinant.evaluate(matrices) :
        determinant.evaluateSpatial(matrices, matrices + layout.plane_stride, layout.gradient_lanes,
            matrices + (1 + layout.gradient_lanes) * layout.plane_stride, layout.laplacian_lanes,
            gradient, log, ratio);
    const C baseline = referenceValue(f, configuration);
    check(near(value(result.amplitude), baseline, 3e-12), "independent head projection and channel value");
    std::array<C, 3> finite_laplacian{}, squared_gradient{};
    for (std::size_t lane = 0; lane < layout.gradient_lanes; ++lane)
    {
      const std::size_t target = mode == pf::SpatialJetMode::ACTIVE ? f.active : layout.physicalTargetElectron(lane / 3);
      const std::size_t coordinate = configuration * 9 + 3 * target + lane % 3;
      const double original = f.positions[coordinate], step = 2e-5;
      f.positions[coordinate] = original + step;
      f.refresh();
      const C plus = referenceValue(f, configuration);
      f.positions[coordinate] = original - step;
      f.refresh();
      const C minus = referenceValue(f, configuration);
      f.positions[coordinate] = original;
      f.refresh();
      const C expected = (plus - minus) / (2 * step * baseline);
      check(near(gradient[lane], expected, 2e-7), "physical target log gradient, reciprocal cross terms");
      finite_laplacian[lane / 3] += (plus + minus - 2.0 * baseline) / (step * step * baseline);
      squared_gradient[lane / 3] += gradient[lane] * gradient[lane];
    }
    for (std::size_t target = 0; target < layout.laplacian_lanes; ++target)
    {
      check(near(ratio[target], finite_laplacian[target], 2e-4), "head-to-determinant Laplacian ratio finite difference");
      check(near(log[target], finite_laplacian[target] - squared_gradient[target], 2e-4),
            "head-to-determinant logarithmic Laplacian");
    }
  }
  check(head.storageFingerprint() == fingerprint, "head warmed allocation identity unchanged");
}

/// Check real head-parameter directions with fixed envelopes and shared real features.
void parameterDirections()
{
  Fixture f(true, pf::SpatialJetMode::FULL_VGL, 1, 2);
  Fixture direction = f;
  for (std::size_t spin = 0; spin < 2; ++spin)
  {
    for (std::size_t i = 0; i < direction.wr[spin].size(); ++i)
    {
      direction.wr[spin][i] = .13 * std::cos(i + 1.0);
      direction.wi[spin][i] = -.17 * std::sin(i + 1.0);
    }
    for (std::size_t i = 0; i < direction.br[spin].size(); ++i)
    {
      direction.br[spin][i] = .11;
      direction.bi[spin][i] = -.09;
    }
  }
  pf::ComplexOrbitalHeadWorkspace head(f.descriptor, f.feature_layout, f.envelope_layout);
  f.evaluate(head);
  const auto layout = head.layout();
  det::ComplexDeterminantWorkspace determinant(2, 2, 1);
  determinant.evaluate(head.data());
  std::vector<C> seed(layout.element_count);
  determinant.fillLogMatrixAdjoints(seed.data());
  const C baseline = referenceValue(f, 0);
  head.evaluateDirection(f.features.data(), f.features.size(), f.envelopes.data(), f.envelopes.size(),
      f.positions.data(), f.positions.size(), direction.parameters(), f.active);
  const std::vector<C> delta(head.data(), head.data() + pf::spatialJetSpanElements(layout));
  C response{};
  for (std::size_t element = 0; element < layout.element_count; ++element)
    response += seed[element] * delta[element];
  const double step = 1e-5;
  Fixture plus = f, minus = f;
  for (std::size_t spin = 0; spin < 2; ++spin)
    for (auto pair : {std::array<std::vector<double>*, 3>{&plus.wr[spin], &minus.wr[spin], &direction.wr[spin]},
                     {&plus.wi[spin], &minus.wi[spin], &direction.wi[spin]},
                     {&plus.br[spin], &minus.br[spin], &direction.br[spin]},
                     {&plus.bi[spin], &minus.bi[spin], &direction.bi[spin]}})
      for (std::size_t i = 0; i < pair[0]->size(); ++i)
      {
        (*pair[0])[i] += step * (*pair[2])[i];
        (*pair[1])[i] -= step * (*pair[2])[i];
      }
  check(near(response, (referenceValue(plus, 0) - referenceValue(minus, 0)) / (2 * step * baseline)),
        "real parameter complex log-score direction, no conjugation");
  plus.evaluate(head);
  const std::vector<C> plus_jets(head.data(), head.data() + delta.size());
  minus.evaluate(head);
  bool all_planes = true;
  for (std::size_t i = 0; i < delta.size(); ++i)
    all_planes = all_planes && near(delta[i], (plus_jets[i] - head.data()[i]) / (2 * step));
  check(all_planes, "paired real head JVP matches every complex spatial plane");
}

/// Cover zero factors, real specialization, descriptor copies and failed generations.
void edgeCases()
{
  Fixture f(false, pf::SpatialJetMode::FULL_VGL, 2, 1, false);
  for (auto& weights : f.wi)
    std::fill(weights.begin(), weights.end(), 0);
  for (auto& bias : f.bi)
    std::fill(bias.begin(), bias.end(), 0);
  pf::ComplexOrbitalHeadWorkspace head(f.descriptor, f.feature_layout, f.envelope_layout);
  f.evaluate(head);
  bool real = true;
  for (std::size_t i = 0; i < pf::spatialJetSpanElements(head.layout()); ++i)
    real = real && head.data()[i].imag() == 0;
  check(real, "zero imaginary head and no inner phase remain exactly real");
  std::vector<double> matrices(head.layout().element_count);
  for (std::size_t i = 0; i < matrices.size(); ++i)
    matrices[i] = head.data()[i].real();
  det::RealOpenDeterminantWorkspace real_determinant(2, 3);
  check(near(det::realValue(real_determinant.evaluate(matrices.data()).amplitude), referenceValue(f, 0)),
        "real head specialization agrees with legacy determinant");
  std::fill(f.envelopes.begin(), f.envelopes.end(), 0);
  f.evaluate(head);
  bool zero = true;
  for (std::size_t i = 0; i < pf::spatialJetSpanElements(head.layout()); ++i)
    zero = zero && head.data()[i] == C{};
  check(zero, "zero envelope jets require no division");
  f.refresh();
  for (std::size_t spin = 0; spin < 2; ++spin)
  {
    std::fill(f.wr[spin].begin(), f.wr[spin].end(), 0);
    std::fill(f.br[spin].begin(), f.br[spin].end(), 0);
  }
  f.evaluate(head);
  zero = true;
  for (std::size_t i = 0; i < pf::spatialJetSpanElements(head.layout()); ++i)
    zero = zero && head.data()[i] == C{};
  check(zero, "zero projected backflow requires no division");
  throws([&] { head.evaluate(f.features.data(), 1, f.envelopes.data(), f.envelopes.size(),
      f.positions.data(), f.positions.size(), f.parameters()); }, "short feature span rejected");
  throws([&] { head.data(); }, "failed request invalidates preceding head generation");

  Fixture phase(true, pf::SpatialJetMode::ACTIVE);
  auto descriptor = phase.descriptor;
  pf::ComplexOrbitalHeadWorkspace immutable(descriptor, phase.feature_layout, phase.envelope_layout);
  descriptor.reciprocal_momenta[0][0] = 123;
  check(immutable.descriptor().reciprocal_momenta[0][0] != 123, "prepared momenta are immutable copies");
  descriptor.reciprocal_momenta.pop_back();
  throws([&] { pf::ComplexOrbitalHeadWorkspace bad(descriptor, phase.feature_layout, phase.envelope_layout); },
         "momentum extent mismatch rejected before evaluation");
  descriptor = phase.descriptor;
  descriptor.reciprocal_momenta[0][0] = std::numeric_limits<double>::quiet_NaN();
  throws([&] { pf::ComplexOrbitalHeadWorkspace bad(descriptor, phase.feature_layout, phase.envelope_layout); },
         "nonfinite momentum rejected before evaluation");
}

/// Exercise bounded CPU math only; no public XML or GPU capability is asserted.
int main()
{
  spatialFixture(false, pf::SpatialJetMode::FULL_VGL);
  spatialFixture(true, pf::SpatialJetMode::FULL_VGL, 1, 2);
  spatialFixture(true, pf::SpatialJetMode::FULL_VGL, 2, 1);
  spatialFixture(true, pf::SpatialJetMode::ACTIVE);
  spatialFixture(false, pf::SpatialJetMode::VALUE_ONLY);
  parameterDirections();
  edgeCases();
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
