//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerInitialization.h
 * @brief Reproducible, dependency-light construction of fresh PsiFormer parameters.
 *
 * The initializer produces the same tensor layout and the same per-role parameter
 * distributions as DeepQMC's current ``ansatz=psiformer`` configuration.  It does
 * not attempt bitwise reproduction of JAX's PRNG splitting or Haiku initializers.
 * The profile name is versioned so distribution or layout changes require an
 * explicit new contract rather than silently changing restarted calculations.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_INITIALIZATION_H
#define QMCPLUSPLUS_PSIFORMER_INITIALIZATION_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Stable name of the first internally constructible DeepQMC-compatible profile.
inline constexpr char DEEPQMC_PSIFORMER_V1[] = "deepqmc_psiformer_v1";

/// Stable name of the integer generator and Gaussian transform used by v1.
inline constexpr char PSIFORMER_INITIALIZATION_RNG_V1[] = "splitmix64_box_muller_v1";

/// Identify the probability law used to initialize one parameter tensor.
enum class InitializationLaw
{
  CONSTANT,
  NORMAL,
  TRUNCATED_NORMAL
};

/** Summarize the requested and observed statistics of one initialized tensor. */
struct TensorInitializationDiagnostic
{
  std::string module;
  std::string name;
  ParameterRole role;
  std::size_t attention_block = NO_ATTENTION_BLOCK;
  InitializationLaw law       = InitializationLaw::CONSTANT;
  std::size_t element_count   = 0;
  double expected_mean        = 0;
  double distribution_standard_deviation = 0;
  double expected_standard_deviation     = 0;
  double observed_mean                   = 0;
  double observed_standard_deviation     = 0;
  double observed_minimum                = 0;
  double observed_maximum                = 0;
};

/** Aggregate diagnostics for one complete model initialization. */
struct InitializationDiagnostics
{
  std::string random_generator;
  std::size_t tensor_count             = 0;
  std::size_t parameter_count          = 0;
  std::size_t random_parameter_count   = 0;
  std::size_t constant_parameter_count = 0;
  std::vector<TensorInitializationDiagnostic> tensors;
};

/**
 * Own a fresh flat parameter vector and its canonical DeepQMC tensor metadata.
 *
 * This neutral representation deliberately does not depend on the HDF5-backed
 * native evaluator.  A runtime adapter can materialize its graph leaves directly,
 * while the existing execution plan can validate ``layouts`` without conversion.
 */
struct InitializedPsiFormerParameters
{
  ModelShape model_shape;
  std::string profile;
  std::uint64_t seed = 0;
  std::vector<ParameterLayoutInput> layouts;
  std::vector<double> values;
  InitializationDiagnostics diagnostics;
};

/**
 * Construct a complete fresh PsiFormer parameter set.
 *
 * ``deepqmc_psiformer_v1`` fixes the production architecture to 16 full
 * determinants, width 256, four attention heads, and four attention blocks.
 * Electron spin populations and the number of nuclei remain system dependent.
 */
InitializedPsiFormerParameters initializePsiFormerParameters(
    const ModelShape& model_shape,
    std::uint64_t seed,
    const std::string& profile = DEEPQMC_PSIFORMER_V1);

/// Return a stable display name for an initialization law.
const char* initializationLawName(InitializationLaw law);

} // namespace qmcplusplus::psiformer

#endif
