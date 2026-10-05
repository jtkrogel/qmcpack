//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerExecutionPlan.h
 * @brief Immutable, pointer-free metadata for executing one imported PsiFormer model.
 *
 * The plan translates DeepQMC's string-keyed flat-parameter layout into typed
 * descriptors once, outside hot evaluation paths.  It deliberately owns only
 * immutable metadata: parameter values remain in the versioned model store and
 * are supplied to kernels as non-owning views.
 *
 * Boundary and scalar-domain choices are represented now so future periodic
 * and genuinely complex implementations do not require another evaluator API.
 * This implementation accepts only open-boundary, real-scalar, fixed-nucleus
 * models; unsupported choices fail during plan construction.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_EXECUTION_PLAN_H
#define QMCPLUSPLUS_PSIFORMER_EXECUTION_PLAN_H

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer
{

/// Sentinel used when a parameter role is not associated with an attention block.
inline constexpr std::size_t NO_ATTENTION_BLOCK = std::numeric_limits<std::size_t>::max();

/// Identify each learned tensor used by the fixed DeepQMC PsiFormer architecture.
enum class ParameterRole
{
  ELECTRON_EMBEDDING_WEIGHT,
  ATTENTION_QUERY_WEIGHT,
  ATTENTION_KEY_WEIGHT,
  ATTENTION_VALUE_WEIGHT,
  ATTENTION_OUTPUT_WEIGHT,
  UPDATE_HIDDEN_WEIGHT,
  UPDATE_HIDDEN_BIAS,
  UPDATE_OUTPUT_WEIGHT,
  UPDATE_OUTPUT_BIAS,
  BACKFLOW_UP_WEIGHT,
  BACKFLOW_DOWN_WEIGHT,
  ENVELOPE_PI_UP,
  ENVELOPE_PI_DOWN,
  ENVELOPE_ZETA_UP,
  ENVELOPE_ZETA_DOWN,
  CUSP_SAME_ALPHA,
  CUSP_OPPOSITE_ALPHA,
  COUNT
};

inline constexpr std::size_t PARAMETER_ROLE_COUNT = static_cast<std::size_t>(ParameterRole::COUNT);

/// Select the displacement policy expected by geometry-producing kernels.
enum class BoundaryCondition
{
  OPEN,
  PERIODIC
};

/// Select the scalar algebra expected by forward and adjoint kernels.
enum class ScalarDomain
{
  REAL,
  COMPLEX
};

/// Record whether reverse products use a transpose or a conjugate transpose.
enum class AdjointConvention
{
  TRANSPOSE,
  HERMITIAN
};

/** Describe the physical and architectural dimensions of one imported model. */
struct ModelShape
{
  std::size_t spin_up_electrons   = 0;
  std::size_t spin_down_electrons = 0;
  std::size_t nuclei              = 0;
  std::size_t determinants        = 0;
  std::size_t feature_dimension   = 0;
  std::size_t attention_heads     = 0;
  std::size_t attention_blocks    = 0;

  /// Return the total number of electrons represented by the model.
  std::size_t electrons() const { return spin_up_electrons + spin_down_electrons; }
};

/** Describe the execution semantics requested for one model. */
struct ExecutionEnvironment
{
  BoundaryCondition boundary = BoundaryCondition::OPEN;
  ScalarDomain parameter_scalar_domain = ScalarDomain::REAL;
  ScalarDomain compute_scalar_domain   = ScalarDomain::REAL;
  ScalarDomain amplitude_scalar_domain = ScalarDomain::REAL;
  bool fixed_nuclei                     = true;
};

/** Advertise implemented and reserved evaluator capabilities. */
struct ModelCapabilities
{
  bool open_boundary       = true;
  bool periodic_boundary   = false;
  bool real_scalars        = true;
  bool complex_scalars     = false;
  bool fixed_nuclei        = true;
  bool moving_nuclei       = false;
};

/** Neutral, owning copy of one imported layout record used while building a plan. */
struct ParameterLayoutInput
{
  std::string module;
  std::string name;
  std::vector<std::size_t> shape;
  std::size_t begin = 0;
  std::size_t end   = 0;
};

/** Typed description of one parameter tensor in canonical flat-vector order. */
struct ParameterTensorDescriptor
{
  ParameterRole role;
  std::size_t attention_block = NO_ATTENTION_BLOCK;
  std::string module;
  std::string name;
  std::vector<std::size_t> shape;
  std::size_t begin = 0;
  std::size_t end   = 0;

  /// Return the number of scalar values in this tensor's flat interval.
  std::size_t size() const { return end - begin; }
};

/**
 * Immutable execution metadata shared by all walkers using the same model.
 *
 * The plan copies names, shapes, and flat offsets, but neither parameter values
 * nor pointers to the source `pf::Parameters` object.  It therefore remains
 * valid across parameter updates and can be shared safely by component clones.
 */
class PsiFormerExecutionPlan
{
public:
  /// Validate neutral layout records and construct a typed execution plan.
  PsiFormerExecutionPlan(ModelShape model_shape,
                         std::vector<ParameterLayoutInput> parameter_layouts,
                         ExecutionEnvironment environment = {});

  /**
   * Construct directly from an object exposing the existing `layouts` member.
   *
   * This template is intentionally dependency-free: it can be instantiated for
   * `pf::Parameters` in PsiFormerWF.cpp after PsiFormerNative.h is included,
   * without including that definition-heavy native header from a second TU.
   */
  template<class ParameterStore>
  static PsiFormerExecutionPlan fromParameters(const ParameterStore& parameters,
                                               ModelShape model_shape,
                                               ExecutionEnvironment environment = {})
  {
    std::vector<ParameterLayoutInput> inputs;
    inputs.reserve(parameters.layouts.size());
    for (const auto& layout : parameters.layouts)
      inputs.push_back({layout.module, layout.name, layout.shape, layout.begin, layout.end});
    return PsiFormerExecutionPlan(std::move(model_shape), std::move(inputs), environment);
  }

  /// Return the validated architectural dimensions.
  const ModelShape& modelShape() const { return model_shape_; }

  /// Return the selected boundary and scalar execution semantics.
  const ExecutionEnvironment& environment() const { return environment_; }

  /// Return the capabilities of this implementation revision.
  static ModelCapabilities capabilities();

  /// Return the reverse-product convention implied by the scalar domain.
  AdjointConvention adjointConvention() const;

  /// Return the number of scalar parameters in canonical flat order.
  std::size_t parameterCount() const { return parameter_count_; }

  /// Return every typed tensor descriptor in canonical flat order.
  const std::vector<ParameterTensorDescriptor>& parameterTensors() const { return parameter_tensors_; }

  /// Resolve one unique typed descriptor, optionally within an attention block.
  const ParameterTensorDescriptor& parameter(ParameterRole role,
                                             std::size_t attention_block = NO_ATTENTION_BLOCK) const;

private:
  ModelShape model_shape_;
  ExecutionEnvironment environment_;
  std::vector<ParameterTensorDescriptor> parameter_tensors_;
  std::vector<std::size_t> descriptor_lookup_;
  std::size_t parameter_count_ = 0;
};

/// Return a stable diagnostic name for a typed parameter role.
const char* parameterRoleName(ParameterRole role);

} // namespace qmcplusplus::psiformer

#endif
