//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_accelerator_planning.cpp
 * @brief CPU-only tests for PsiFormer accelerator layout and publication contracts.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerCapabilityReport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <numeric>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{

/// Append one canonical test tensor and derive its contiguous interval.
void appendLayout(std::vector<ParameterLayoutInput>& layouts,
                  std::string module,
                  std::string name,
                  std::vector<std::size_t> shape)
{
  const std::size_t begin = layouts.empty() ? 0 : layouts.back().end;
  const std::size_t size = std::accumulate(shape.begin(), shape.end(),
                                           std::size_t{1}, std::multiplies<>());
  layouts.push_back({std::move(module), std::move(name), std::move(shape), begin, begin + size});
}

/// Construct a compact complete 1+1-electron model for pure planning tests.
PsiFormerExecutionPlan makePlan()
{
  const ModelShape model{/*spin_up_electrons=*/1,
                         /*spin_down_electrons=*/1,
                         /*nuclei=*/1,
                         /*determinants=*/2,
                         /*feature_dimension=*/8,
                         /*attention_heads=*/2,
                         /*attention_blocks=*/1};
  const std::size_t electrons = model.electrons();
  const std::string prefix = "neural_network_wave_function/~/";
  std::vector<ParameterLayoutInput> layouts;
  appendLayout(layouts, prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
  appendLayout(layouts, prefix + "exponential_envelopes", "pi_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "pi_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "zetas_down",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "exponential_envelopes", "zetas_up",
               {model.determinants * electrons, model.nuclei});
  appendLayout(layouts, prefix + "omni_net/~/Backflow/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(layouts, prefix + "omni_net/~/Backflow_1/~/mlp/linear_0", "w",
               {model.feature_dimension, model.determinants * electrons});
  appendLayout(layouts, prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w",
               {4 * model.nuclei + 1, model.feature_dimension});

  const std::string block = prefix +
      "omni_net/~/electron_gnn/~/electron_gnn_layer/~/node_attention_electron_update_feature/";
  appendLayout(layouts, block + "mlp/linear_0", "b", {model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_0", "w", {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_1", "b", {model.feature_dimension});
  appendLayout(layouts, block + "mlp/linear_1", "w", {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/key", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/linear", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/query", "w",
               {model.feature_dimension, model.feature_dimension});
  appendLayout(layouts, block + "multi_head_attention/value", "w",
               {model.feature_dimension, model.feature_dimension});
  return PsiFormerExecutionPlan(model, std::move(layouts));
}

} // namespace

TEST_CASE("PsiFormer capability failures preserve strict admission categories",
          "[wavefunction][psiformer][accelerator][capability]")
{
  const auto category = [](const std::string& request, const PsiFormerAcceleratorReadiness& readiness) {
    try { resolvePsiFormerAcceleratorRequest(request, readiness); }
    catch (const std::exception& error) { return capabilityFailure(error); }
    return CapabilityFailure::UNCLASSIFIED;
  };
  PsiFormerAcceleratorReadiness readiness;
  CHECK(category("typo", readiness) == CapabilityFailure::INVALID_INPUT);
  CHECK(category("sycl", readiness) == CapabilityFailure::UNSUPPORTED_OPERATION);
  CHECK(category("cuda", readiness) == CapabilityFailure::UNSUPPORTED_OPERATION);
  readiness.cuda = {true, false, allPsiFormerAcceleratorOperations(), PsiFormerRuntimeValidation::PASSED};
  CHECK(category("cuda", readiness) == CapabilityFailure::UNQUALIFIED_HARDWARE);
  readiness.cuda.usable_device = true;
  readiness.cuda.runtime_validation = PsiFormerRuntimeValidation::NOT_RUN;
  CHECK(category("cuda", readiness) == CapabilityFailure::UNQUALIFIED_HARDWARE);
  readiness.cuda.runtime_validation = PsiFormerRuntimeValidation::FAILED;
  CHECK(category("cuda", readiness) == CapabilityFailure::UNQUALIFIED_HARDWARE);
  readiness.cuda.operation_coverage = 0;
  CHECK(category("cuda", readiness) == CapabilityFailure::UNSUPPORTED_OPERATION);
  const auto automatic = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(automatic.backend == PsiFormerAcceleratorBackend::CPU);
  CHECK(automatic.fallback_count == 1);
}

TEST_CASE("Real two-cell device metadata matching never implies backend readiness",
          "[wavefunction][psiformer][accelerator][pair_metadata]")
{
  ExecutionEnvironment environment;
  CHECK_FALSE(matchesRealDevicePairMetadata(environment));
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  environment.periodic_axes = {true, true, true};
  auto profile = std::make_shared<PeriodicProfileMetadata>();
  profile->occupied_columns = {{{}, 0, 0, 1}, {{}, 0, 1, 1}};
  environment.periodic_profile = profile;
  REQUIRE(matchesRealDevicePairMetadata(environment));
  SECTION("raw features") { profile->feature_transform = PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1; }
  SECTION("polynomial features") { profile->feature_transform = PeriodicFeatureTransform::POLYNOMIAL_V1; }
  SECTION("wrong pair arity") { profile->pair_feature_dimension = 4; }
  SECTION("no cusp") { profile->electron_electron_cusp = false; }
  SECTION("Fourier envelopes") { profile->envelope_policy = PeriodicEnvelopePolicy::FOURIER; }
  SECTION("Fourier modes") { profile->fourier_modes = {{0, 0, 0}}; }
  SECTION("occupied phase") { profile->occupied_columns[0].reciprocal = {1, 0, 0}; }
  SECTION("paired head") { environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX; }
  SECTION("complex amplitude") { environment.amplitude_scalar_domain = ScalarDomain::COMPLEX; }
  SECTION("complex parameters") { environment.parameter_scalar_domain = ScalarDomain::COMPLEX; }
  SECTION("complex compute") { environment.compute_scalar_domain = ScalarDomain::COMPLEX; }
  SECTION("moving ions") { environment.fixed_nuclei = false; }
  SECTION("partial periodicity") { environment.periodic_axes[2] = false; }
  SECTION("open boundary") { environment.boundary = BoundaryCondition::OPEN; }
  SECTION("legacy torus") { environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1; }
  SECTION("unknown version") { profile->version = 2; }
  CHECK_FALSE(matchesRealDevicePairMetadata(environment));
}

TEST_CASE("Paired two-cell numerical metadata excludes unimplemented phases and envelopes",
          "[wavefunction][psiformer][accelerator][paired_pair_metadata]")
{
  ExecutionEnvironment environment;
  CHECK_FALSE(matchesPairedDevicePairMetadata(environment));
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
  environment.periodic_axes = {true, true, true};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  auto profile = std::make_shared<PeriodicProfileMetadata>();
  profile->occupied_columns = {{{}, 0, 0, 1}, {{}, 0, 1, 1}};
  environment.periodic_profile = profile;
  REQUIRE(matchesPairedDevicePairMetadata(environment));
  CHECK_FALSE(matchesRealDevicePairMetadata(environment));
  SECTION("raw features") { profile->feature_transform = PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1; }
  SECTION("polynomial features") { profile->feature_transform = PeriodicFeatureTransform::POLYNOMIAL_V1; }
  SECTION("wrong pair arity") { profile->pair_feature_dimension = 4; }
  SECTION("no cusp") { profile->electron_electron_cusp = false; }
  SECTION("Fourier envelopes") { profile->envelope_policy = PeriodicEnvelopePolicy::FOURIER; }
  SECTION("Fourier modes") { profile->fourier_modes = {{0, 0, 0}}; }
  SECTION("occupied phase") { profile->occupied_columns[0].reciprocal = {1, 0, 0}; }
  SECTION("real head") { environment.head_representation = HeadRepresentation::REAL; }
  SECTION("real amplitude") { environment.amplitude_scalar_domain = ScalarDomain::REAL; }
  SECTION("complex parameters") { environment.parameter_scalar_domain = ScalarDomain::COMPLEX; }
  SECTION("complex compute") { environment.compute_scalar_domain = ScalarDomain::COMPLEX; }
  SECTION("moving ions") { environment.fixed_nuclei = false; }
  SECTION("partial periodicity") { environment.periodic_axes[2] = false; }
  SECTION("open boundary") { environment.boundary = BoundaryCondition::OPEN; }
  SECTION("legacy torus") { environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1; }
  SECTION("unknown version") { profile->version = 2; }
  CHECK_FALSE(matchesPairedDevicePairMetadata(environment));
}

TEST_CASE("PsiFormer accelerator backend selection fails closed", "[wavefunction][psiformer][accelerator]")
{
  const PsiFormerCompiledAcceleratorSupport none;
  CHECK(selectPsiFormerAcceleratorBackend("", none) == PsiFormerAcceleratorBackend::CPU);
  CHECK(selectPsiFormerAcceleratorBackend("AUTO", none) == PsiFormerAcceleratorBackend::CPU);
  CHECK_THROWS_WITH(selectPsiFormerAcceleratorBackend("yes", none),
                    Catch::Matchers::ContainsSubstring("requires a compiled accelerator"));
  CHECK_THROWS_WITH(selectPsiFormerAcceleratorBackend("cuda", none),
                    Catch::Matchers::ContainsSubstring("not compiled"));
  CHECK_THROWS_AS(selectPsiFormerAcceleratorBackend("mystery", none), std::invalid_argument);

  PsiFormerCompiledAcceleratorSupport support;
  support.openmp_target = true;
  support.sycl          = true;
  CHECK(selectPsiFormerAcceleratorBackend("yes", support) == PsiFormerAcceleratorBackend::SYCL);
  CHECK(selectPsiFormerAcceleratorBackend("omptarget", support) ==
        PsiFormerAcceleratorBackend::OPENMP_TARGET);
  support.cuda = true;
  CHECK(selectPsiFormerAcceleratorBackend("auto", support) == PsiFormerAcceleratorBackend::CUDA);
}

TEST_CASE("HIP readiness keeps untested runtime and missing training operations closed",
          "[wavefunction][psiformer][accelerator][hip_readiness]")
{
  const auto inference = psiFormerAcceleratorInferenceOperations();
  PsiFormerAcceleratorReadiness readiness;
  readiness.hip = {true, true, inference, PsiFormerRuntimeValidation::NOT_RUN};
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("hip", readiness, inference), std::runtime_error);
  readiness.hip.runtime_validation = PsiFormerRuntimeValidation::PASSED;
  const auto resolved = resolvePsiFormerAcceleratorRequest("hip", readiness, inference);
  CHECK(resolved.backend == PsiFormerAcceleratorBackend::HIP);
  CHECK(resolved.validated_operations == inference);
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("hip", readiness, allPsiFormerAcceleratorOperations()),
                  std::runtime_error);
  readiness.hip.runtime_validation = PsiFormerRuntimeValidation::FAILED;
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("hip", readiness, inference), std::runtime_error);
  readiness.hip.runtime_validation = PsiFormerRuntimeValidation::PASSED;
  readiness.hip.operation_coverage = allPsiFormerAcceleratorOperations();
  CHECK(resolvePsiFormerAcceleratorRequest("hip", readiness, allPsiFormerAcceleratorOperations()).backend ==
        PsiFormerAcceleratorBackend::HIP);
}

TEST_CASE("PsiFormer production accelerator resolution applies every readiness gate",
          "[wavefunction][psiformer][accelerator]")
{
  PsiFormerAcceleratorReadiness readiness;

  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("auto", readiness, 0),
                  std::invalid_argument);
  readiness.cuda.operation_coverage = UINT32_C(1) << 31;
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("auto", readiness),
                  std::invalid_argument);
  readiness.cuda.operation_coverage = 0;

  PsiFormerAcceleratorReadiness selective_validation;
  selective_validation.cuda = {true, true, allPsiFormerAcceleratorOperations(),
                               PsiFormerRuntimeValidation::PASSED};
  selective_validation.hip.operation_coverage = UINT32_C(1) << 31;
  CHECK(resolvePsiFormerAcceleratorRequest("auto", selective_validation).backend ==
        PsiFormerAcceleratorBackend::CUDA);
  CHECK(resolvePsiFormerAcceleratorRequest("cuda", selective_validation).backend ==
        PsiFormerAcceleratorBackend::CUDA);
  selective_validation.cuda.operation_coverage = UINT32_C(1) << 31;
  selective_validation.hip = {true, true, allPsiFormerAcceleratorOperations(),
                              PsiFormerRuntimeValidation::PASSED};
  CHECK(resolvePsiFormerAcceleratorRequest("hip", selective_validation).backend ==
        PsiFormerAcceleratorBackend::HIP);
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("auto", selective_validation),
                  std::invalid_argument);
  CHECK_THROWS_AS(resolvePsiFormerAcceleratorRequest("mystery", selective_validation),
                  std::invalid_argument);
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest("sycl", selective_validation),
                    Catch::Matchers::ContainsSubstring("no production executor"));

  // Explicit CPU requests do not inspect runtime accelerator state and are not fallbacks.
  for (const std::string request : {"", "no", "CPU"})
  {
    const PsiFormerAcceleratorResolution result =
        resolvePsiFormerAcceleratorRequest(request, readiness);
    CHECK(result.backend == PsiFormerAcceleratorBackend::CPU);
    CHECK(result.reason == PsiFormerAcceleratorResolutionReason::CPU_REQUESTED);
    CHECK(result.readiness_records_consulted == 0);
    CHECK(result.fallback_count == 0);
  }

  PsiFormerAcceleratorResolution result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.backend == PsiFormerAcceleratorBackend::CPU);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::NOT_COMPILED);
  CHECK(result.readiness_records_consulted == 2);
  CHECK(result.fallback_count == 1);

  readiness.cuda.compiled = true;
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::NO_USABLE_DEVICE);

  readiness.cuda.usable_device = true;
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::INCOMPLETE_OPERATION_COVERAGE);

  readiness.cuda.operation_coverage = allPsiFormerAcceleratorOperations();
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::RUNTIME_VALIDATION_NOT_RUN);

  readiness.cuda.runtime_validation = PsiFormerRuntimeValidation::FAILED;
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::RUNTIME_VALIDATION_FAILED);

  readiness.cuda.runtime_validation = PsiFormerRuntimeValidation::PASSED;
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.backend == PsiFormerAcceleratorBackend::CUDA);
  CHECK(result.reason == PsiFormerAcceleratorResolutionReason::ACCELERATOR_READY);
  CHECK(result.readiness_records_consulted == 1);
  CHECK(result.fallback_count == 0);

  // Auto may pass an unusable preferred family and select a fully ready HIP path.
  readiness.cuda.usable_device = false;
  readiness.hip = {true, true, allPsiFormerAcceleratorOperations(),
                   PsiFormerRuntimeValidation::PASSED};
  result = resolvePsiFormerAcceleratorRequest("auto", readiness);
  CHECK(result.backend == PsiFormerAcceleratorBackend::HIP);
  CHECK(result.readiness_records_consulted == 2);

  // Strict yes selects the preferred compiled family and never silently falls back.
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest("yes", readiness),
                    Catch::Matchers::ContainsSubstring("no usable runtime device"));
  CHECK(resolvePsiFormerAcceleratorRequest("hip", readiness).backend ==
        PsiFormerAcceleratorBackend::HIP);
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest("cuda", readiness),
                    Catch::Matchers::ContainsSubstring("no usable runtime device"));
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest("sycl", readiness),
                    Catch::Matchers::ContainsSubstring("no production executor"));
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest("omptarget", readiness),
                    Catch::Matchers::ContainsSubstring("no production executor"));

  PsiFormerAcceleratorReadiness hip_only;
  hip_only.hip = {true, true, allPsiFormerAcceleratorOperations(),
                  PsiFormerRuntimeValidation::PASSED};
  const PsiFormerAcceleratorResolution yes_hip =
      resolvePsiFormerAcceleratorRequest("yes", hip_only);
  CHECK(yes_hip.backend == PsiFormerAcceleratorBackend::HIP);
  CHECK(yes_hip.readiness_records_consulted == 2);
  CHECK(resolvePsiFormerAcceleratorRequest("hip", hip_only).readiness_records_consulted == 1);

  PsiFormerAcceleratorReadiness cuda_only;
  cuda_only.cuda = {true, true, allPsiFormerAcceleratorOperations(),
                    PsiFormerRuntimeValidation::PASSED};
  CHECK(resolvePsiFormerAcceleratorRequest("yes", cuda_only).readiness_records_consulted == 1);
  CHECK(resolvePsiFormerAcceleratorRequest("cuda", cuda_only).readiness_records_consulted == 1);

  // A focused route may request a strict subset, but linked leaves still grant no bits.
  readiness.cuda.usable_device = true;
  readiness.cuda.operation_coverage = acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES);
  readiness.cuda.runtime_validation = PsiFormerRuntimeValidation::PASSED;
  CHECK(resolvePsiFormerAcceleratorRequest(
            "cuda", readiness,
            acceleratorOperationBit(PsiFormerAcceleratorOperation::VALUES)).backend ==
        PsiFormerAcceleratorBackend::CUDA);
  CHECK_THROWS_WITH(resolvePsiFormerAcceleratorRequest(
                        "cuda", readiness,
                        acceleratorOperationBit(PsiFormerAcceleratorOperation::VGL)),
                    Catch::Matchers::ContainsSubstring("incomplete end-to-end operation coverage"));
}

TEST_CASE("PsiFormer device layout is pointer-free and deterministic",
          "[wavefunction][psiformer][accelerator]")
{
  const PsiFormerExecutionPlan plan = makePlan();
  const PsiFormerDeviceLayout first = makePsiFormerDeviceLayout(plan);
  const PsiFormerDeviceLayout second = makePsiFormerDeviceLayout(plan);

  REQUIRE(first.tensors.size() == plan.parameterTensors().size());
  CHECK(first.parameter_count == plan.parameterCount());
  CHECK(first.fingerprint != 0);
  CHECK(first.fingerprint == second.fingerprint);
  CHECK(first.tensors == second.tensors);
  CHECK(first.tensors.front().rank == 0);
  CHECK(first.tensors.front().size() == 1);
  CHECK(first.tensors.back().rank == 2);
  CHECK(first.tensors.back().end == first.parameter_count);

  for (std::size_t tensor = 0; tensor < first.tensors.size(); ++tensor)
  {
    CHECK(first.tensors[tensor].begin == plan.parameterTensors()[tensor].begin);
    CHECK(first.tensors[tensor].end == plan.parameterTensors()[tensor].end);
    CHECK(first.tensors[tensor].role == plan.parameterTensors()[tensor].role);
  }
}

TEST_CASE("PsiFormer device publication is versioned and failure atomic",
          "[wavefunction][psiformer][accelerator]")
{
  PsiFormerDevicePublicationState publication(4);
  CHECK(publication.activeVersion() == 4);
  CHECK(publication.isReady(4));
  CHECK_FALSE(publication.publicationPending());
  CHECK_THROWS_AS(publication.beginPublication(4), std::invalid_argument);

  publication.beginPublication(7);
  CHECK(publication.activeVersion() == 4);
  REQUIRE(publication.pendingVersion());
  CHECK(*publication.pendingVersion() == 7);
  CHECK_THROWS_AS(publication.beginPublication(8), std::logic_error);
  CHECK_THROWS_AS(publication.completePublication(6), std::logic_error);

  publication.cancelPublication(7);
  CHECK(publication.activeVersion() == 4);
  CHECK_FALSE(publication.publicationPending());

  publication.beginPublication(8);
  publication.completePublication(8);
  CHECK(publication.activeVersion() == 8);
  CHECK(publication.isReady(8));
  CHECK_FALSE(publication.isReady(4));
  CHECK_FALSE(publication.publicationPending());
}

} // namespace qmcplusplus::psiformer
