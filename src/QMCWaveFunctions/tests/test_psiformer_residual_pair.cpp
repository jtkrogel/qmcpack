#include "QMCWaveFunctions/PsiFormer/PsiFormerResidualPair.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace qmcplusplus::psiformer;
namespace {
int checks = 0;
void check(bool condition) { ++checks; if (!condition) throw std::runtime_error("Residual pair check " + std::to_string(checks)); }
bool near(double a, double b, double tol = 2e-7) { return std::abs(a-b) <= tol * (1 + std::abs(a) + std::abs(b)); }
}
int main()
{
  for (double r : {0., 1e-12, .1, .9, 1.999999, 2., 3.})
  {
    auto j = residualPairJet(r, 0, 1, 2, .5);
    check(j.value == 0 && j.radial_first == 0 && j.radial_second == 0 && j.radial_first_over_r == 0);
    check(r >= 2 || j.score != 0);
  }
  for (double lambda : {-.8, 1e-10, .7})
    for (double r : {.1, .9, 1.8})
    {
      const double h = 1e-5;
      auto j = residualPairJet(r, lambda, 1.3, 2, .25);
      auto left = residualPairJet(r-h, lambda, 1.3, 2, .25);
      auto right = residualPairJet(r+h, lambda, 1.3, 2, .25);
      check(near(j.radial_first, (right.value-left.value)/(2*h)));
      check(near(j.radial_second, (right.radial_first-left.radial_first)/(2*h)));
      check(near(j.radial_first_over_r, j.radial_first/r));
      auto lower = residualPairJet(r, lambda-h, 1.3, 2, .25);
      auto upper = residualPairJet(r, lambda+h, 1.3, 2, .25);
      check(near(j.score, (upper.value-lower.value)/(2*h)));
      const double a = 1.3*std::exp(lambda);
      const double u = 1-r*r/4;
      check(near(j.value, u*u*u*(-.25*a*a/(a+r)+.25*1.3*1.3/(1.3+r)), 1e-13));
    }
  for (double lambda : {-.8, .7})
  {
    auto zero = residualPairJet(0, lambda, 1, 2, .5);
    check(zero.radial_first == 0);
    check(near(zero.radial_second, zero.radial_first_over_r));
    auto edge = residualPairJet(2-1e-7, lambda, 1, 2, .5);
    check(std::abs(edge.value) < 1e-18 && std::abs(edge.radial_first) < 1e-12 && std::abs(edge.radial_second) < 1e-6);
  }
  const ResidualPairGeometry::Lattice lattice{{{4,0,0},{1.2,4,0},{.6,.4,4}}};
  ResidualPairGeometry geometry({1,1,1,1.7}, lattice);
  const ResidualPairGeometry::Position first{.2,.3,.4}, second{-.4,.1,-.1};
  const auto d = geometry.displacement(first, second);
  const auto j = geometry.evaluate(d, .4, false);
  for (const auto& vector : lattice)
  {
    auto image = first;
    for (int axis = 0; axis < 3; ++axis) image[axis] += vector[axis];
    auto shifted = geometry.displacement(image, second);
    for (int axis = 0; axis < 3; ++axis) check(near(shifted[axis], d[axis], 1e-14));
    check(near(geometry.evaluate(shifted, .4, false).value, j.value, 1e-14));
  }
  bool bad_cutoff = false;
  try { ResidualPairGeometry invalid({1,1,1,2.1}, lattice); }
  catch (const std::invalid_argument&) { bad_cutoff = true; }
  check(bad_cutoff);
  bool invalid_provenance = false;
  try { ResidualPairGeometry invalid({1,1,1,1.7,"invalid"}, lattice); }
  catch (const std::invalid_argument&) { invalid_provenance = true; }
  check(invalid_provenance);
  for (double invalid : {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
  {
    bool rejected_nonfinite = false;
    try { residualPairJet(.1,invalid,1,2,.5); }
    catch (const std::invalid_argument&) { rejected_nonfinite = true; }
    check(rejected_nonfinite);
  }
  bool rejected = false;
  try { residualPairJet(.1, 1000, 1, 2, .5); } catch (const std::overflow_error&) { rejected = true; }
  check(rejected);
  std::cout << checks << " residual pair checks passed\n";
}
