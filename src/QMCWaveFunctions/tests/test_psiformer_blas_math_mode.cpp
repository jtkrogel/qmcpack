//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_blas_math_mode.cpp
 * @brief CPU tests for PsiFormer backend BLAS mode mapping and scoped restoration.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerBlasMathMode.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <stdexcept>
#include <string>
#include <vector>

using namespace qmcplusplus::psiformer;

namespace
{

/// Small native-mode stand-in used to validate transitions without a GPU runtime.
enum class MockMode
{
  AMBIENT,
  STRICT,
  REDUCED
};

/** Record native get/set operations and optionally fail one numbered update. */
class RecordingMathModeController
{
public:
  using mode_type = MockMode;

  explicit RecordingMathModeController(MockMode initial, int failing_set = 0)
      : current_(initial), failing_set_(failing_set)
  {}

  mode_type currentMode()
  {
    ++query_count_;
    return current_;
  }

  void setMode(mode_type mode)
  {
    ++set_count_;
    requested_.push_back(mode);
    if (set_count_ == failing_set_)
      throw std::runtime_error("mock native set failure");
    current_ = mode;
  }

  mode_type current() const noexcept { return current_; }
  int queryCount() const noexcept { return query_count_; }
  int setCount() const noexcept { return set_count_; }
  const std::vector<mode_type>& requested() const noexcept { return requested_; }

private:
  mode_type current_;
  int failing_set_ = 0;
  int query_count_ = 0;
  int set_count_   = 0;
  std::vector<mode_type> requested_;
};

} // namespace

TEST_CASE("PsiFormer BLAS math modes map to enforceable CUDA and HIP behavior",
          "[psiformer][device][blas_math_mode]")
{
  const PsiFormerBlasMathModePlan cuda_fp64 = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::FP64_STRICT, PsiFormerAcceleratorBackend::CUDA);
  const PsiFormerBlasMathModePlan cuda_fp32 = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::FP32_STRICT, PsiFormerAcceleratorBackend::CUDA);
  const PsiFormerBlasMathModePlan cuda_tf32 = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::CUDA_TF32, PsiFormerAcceleratorBackend::CUDA);
  CHECK(cuda_fp64.native == PsiFormerNativeBlasMathMode::CUDA_PEDANTIC);
  CHECK(cuda_fp32.native == PsiFormerNativeBlasMathMode::CUDA_PEDANTIC);
  CHECK_FALSE(cuda_fp32.reduced_multiply);
  CHECK(cuda_tf32.native == PsiFormerNativeBlasMathMode::CUDA_TF32);
  CHECK(cuda_tf32.reduced_multiply);

  const PsiFormerBlasMathModePlan hip_fp64 = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::FP64_STRICT, PsiFormerAcceleratorBackend::HIP);
  const PsiFormerBlasMathModePlan hip_fp32 = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::FP32_STRICT, PsiFormerAcceleratorBackend::HIP);
  CHECK(hip_fp64.native == PsiFormerNativeBlasMathMode::HIP_DEFAULT_STRICT);
  CHECK(hip_fp32.native == PsiFormerNativeBlasMathMode::HIP_DEFAULT_STRICT);
  CHECK_FALSE(hip_fp32.reduced_multiply);
  CHECK_THROWS_WITH(makePsiFormerBlasMathModePlan(
                        PsiFormerBackendMathMode::CUDA_TF32,
                        PsiFormerAcceleratorBackend::HIP),
                    Catch::Matchers::ContainsSubstring("unavailable on HIP"));

  const PsiFormerBlasMathModePlan host = makePsiFormerBlasMathModePlan(
      PsiFormerBackendMathMode::FP64_STRICT, PsiFormerAcceleratorBackend::CPU);
  CHECK(host.native == PsiFormerNativeBlasMathMode::HOST_DEFAULT);
  CHECK_THROWS_AS(makePsiFormerBlasMathModePlan(
                      PsiFormerBackendMathMode::FP32_STRICT,
                      PsiFormerAcceleratorBackend::CPU),
                  std::invalid_argument);
  CHECK_THROWS_AS(makePsiFormerBlasMathModePlan(
                      PsiFormerBackendMathMode::FP64_STRICT,
                      PsiFormerAcceleratorBackend::SYCL),
                  std::invalid_argument);
  CHECK(std::string(psiFormerNativeBlasMathModeName(cuda_fp32.native)) ==
        "cuda_pedantic");
}

TEST_CASE("PsiFormer scoped BLAS mode restores ambient state on success",
          "[psiformer][device][blas_math_mode]")
{
  RecordingMathModeController controller(MockMode::AMBIENT);
  bool operation_called = false;
  executeWithPsiFormerBlasMathMode(controller, MockMode::STRICT, [&] {
    operation_called = true;
    CHECK(controller.current() == MockMode::STRICT);
  });
  CHECK(operation_called);
  CHECK(controller.current() == MockMode::AMBIENT);
  CHECK(controller.queryCount() == 1);
  CHECK(controller.setCount() == 2);
  REQUIRE(controller.requested().size() == 2);
  CHECK(controller.requested()[0] == MockMode::STRICT);
  CHECK(controller.requested()[1] == MockMode::AMBIENT);

  RecordingMathModeController already_strict(MockMode::STRICT);
  executeWithPsiFormerBlasMathMode(already_strict, MockMode::STRICT, [] {});
  CHECK(already_strict.queryCount() == 1);
  CHECK(already_strict.setCount() == 0);
}

TEST_CASE("PsiFormer scoped BLAS mode restores state after operation failure",
          "[psiformer][device][blas_math_mode]")
{
  RecordingMathModeController controller(MockMode::AMBIENT);
  CHECK_THROWS_WITH(
      executeWithPsiFormerBlasMathMode(controller, MockMode::REDUCED, [] {
        throw std::domain_error("mock GEMM failure");
      }),
      Catch::Matchers::ContainsSubstring("mock GEMM failure"));
  CHECK(controller.current() == MockMode::AMBIENT);
  CHECK(controller.setCount() == 2);
}

TEST_CASE("PsiFormer scoped BLAS mode surfaces set and restoration failures",
          "[psiformer][device][blas_math_mode]")
{
  RecordingMathModeController set_failure(MockMode::AMBIENT, 1);
  bool operation_called = false;
  CHECK_THROWS_WITH(
      executeWithPsiFormerBlasMathMode(set_failure, MockMode::STRICT,
                                       [&] { operation_called = true; }),
      Catch::Matchers::ContainsSubstring("mock native set failure"));
  CHECK_FALSE(operation_called);
  CHECK(set_failure.current() == MockMode::AMBIENT);

  RecordingMathModeController restore_failure(MockMode::AMBIENT, 2);
  CHECK_THROWS_WITH(
      executeWithPsiFormerBlasMathMode(restore_failure, MockMode::STRICT, [] {}),
      Catch::Matchers::ContainsSubstring("mock native set failure"));
  CHECK(restore_failure.current() == MockMode::STRICT);

  RecordingMathModeController double_failure(MockMode::AMBIENT, 2);
  CHECK_THROWS_WITH(
      executeWithPsiFormerBlasMathMode(double_failure, MockMode::STRICT, [] {
        throw std::domain_error("mock GEMM failure");
      }),
      Catch::Matchers::ContainsSubstring("restoration failed"));
  CHECK(double_failure.current() == MockMode::STRICT);
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
