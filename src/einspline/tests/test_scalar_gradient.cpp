//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "einspline/bspline.h"
#include <array>
#include <cmath>
#include <memory>
#include <vector>

TEST_CASE("scalar cubic gradient on unequal nonunit grids", "[einspline][mfr_meta]")
{
  Ugrid gx{-0.7, 2.3, 5}, gy{0.2, 4.2, 7}, gz{-2.1, 0.9, 6};
  BCtype_d bc{NATURAL, NATURAL, 0.0, 0.0};
  std::vector<double> data(gx.num * gy.num * gz.num);
  for (int i = 0; i < gx.num; ++i)
    for (int j = 0; j < gy.num; ++j)
      for (int k = 0; k < gz.num; ++k)
      {
        const double x = gx.start + i * (gx.end - gx.start) / (gx.num - 1);
        const double y = gy.start + j * (gy.end - gy.start) / (gy.num - 1);
        const double z = gz.start + k * (gz.end - gz.start) / (gz.num - 1);
        data[(i * gy.num + j) * gz.num + k] = 1.2 + 0.7*x - 0.3*y + 0.9*z + 0.4*x*y*z;
      }
  auto spline = std::unique_ptr<UBspline_3d_d, void (*)(void*)>(
      create_UBspline_3d_d(gx, gy, gz, bc, bc, bc, data.data()), destroy_Bspline);
  REQUIRE(spline);
  const double x=0.31, y=2.73, z=-0.43;
  double value, gradient[3];
  eval_UBspline_3d_d_vg(spline.get(), x, y, z, &value, gradient);
  CHECK(value == Approx(1.2+0.7*x-0.3*y+0.9*z+0.4*x*y*z).margin(1e-12));
  CHECK(gradient[0] == Approx(0.7+0.4*y*z).margin(1e-12));
  CHECK(gradient[1] == Approx(-0.3+0.4*x*z).margin(1e-12));
  CHECK(gradient[2] == Approx(0.9+0.4*x*y).margin(1e-12));
}

TEST_CASE("periodic scalar gradient is derivative of the value interpolant", "[einspline][mfr_meta]")
{
  constexpr int n=48;
  constexpr double pi=3.14159265358979323846;
  Ugrid grid{0.0, 1.0, n};
  BCtype_d bc{PERIODIC, PERIODIC, 0.0, 0.0};
  std::vector<double> data(n*n*n);
  for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j)
      for (int k=0; k<n; ++k)
        data[(i*n+j)*n+k] = 0.2 + 0.3*std::cos(2*pi*i/n) + 0.7*std::sin(2*pi*(j+k)/n);
  auto spline=std::unique_ptr<UBspline_3d_d, void (*)(void*)>(
      create_UBspline_3d_d(grid,grid,grid,bc,bc,bc,data.data()), destroy_Bspline);
  REQUIRE(spline);
  for (const std::array<double,3>& point : {std::array<double,3>{0.137,0.294,0.761},
                                           std::array<double,3>{0.0,0.0,0.0},
                                           std::array<double,3>{1.0-1e-12,1.0-1e-12,1.0-1e-12}})
  {
    double value, reference, gradient[3];
    eval_UBspline_3d_d_vg(spline.get(),point[0],point[1],point[2],&value,gradient);
    eval_UBspline_3d_d(spline.get(),point[0],point[1],point[2],&reference);
    CHECK(value == Approx(reference).margin(2e-14));
    const double analytic[3]={-0.6*pi*std::sin(2*pi*point[0]),
                              1.4*pi*std::cos(2*pi*(point[1]+point[2])),
                              1.4*pi*std::cos(2*pi*(point[1]+point[2]))};
    for (int axis=0;axis<3;++axis)
    {
      auto plus=point, minus=point;
      constexpr double step=1e-6;
      plus[axis]+=step; minus[axis]-=step;
      plus[axis]-=std::floor(plus[axis]); minus[axis]-=std::floor(minus[axis]);
      double vp,vm;
      eval_UBspline_3d_d(spline.get(),plus[0],plus[1],plus[2],&vp);
      eval_UBspline_3d_d(spline.get(),minus[0],minus[1],minus[2],&vm);
      CHECK(gradient[axis] == Approx((vp-vm)/(2*step)).margin(2e-8));
      CHECK(gradient[axis] == Approx(analytic[axis]).margin(2e-4));
    }
  }
}
