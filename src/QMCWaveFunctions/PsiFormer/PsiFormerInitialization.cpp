//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerInitialization.cpp
 * @brief Versioned fresh-parameter construction for the molecular PsiFormer.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmcplusplus::psiformer
{
namespace
{

/// Actual standard deviation of a standard normal conditioned on |x| <= 2.
inline constexpr double TRUNCATED_NORMAL_STANDARD_DEVIATION = 0.87962566103423978;

/// DeepQMC's first fixed molecular PsiFormer architecture profile.
inline constexpr std::size_t PROFILE_DETERMINANTS      = 16;
inline constexpr std::size_t PROFILE_FEATURE_DIMENSION = 256;
inline constexpr std::size_t PROFILE_ATTENTION_HEADS   = 4;
inline constexpr std::size_t PROFILE_ATTENTION_BLOCKS  = 4;

/// Describe a tensor and the law used to populate its row-major interval.
struct TensorSpecification
{
  std::string module;
  std::string name;
  std::vector<std::size_t> shape;
  ParameterRole role;
  std::size_t attention_block;
  InitializationLaw law;
  double mean;
  double distribution_standard_deviation;
  double expected_standard_deviation;
};

/**
 * Small deterministic 64-bit generator used only for parameter initialization.
 *
 * SplitMix64 has a fully specified integer transition.  Gaussian conversion is
 * also implemented locally instead of relying on the implementation-defined
 * sequence of ``std::normal_distribution``.  The final transcendental results
 * can still differ in their last bits between platform math libraries.
 */
class SplitMix64NormalGenerator
{
public:
  /// Start one independent tensor stream from a mixed 64-bit seed.
  explicit SplitMix64NormalGenerator(std::uint64_t seed) : state_(seed) {}

  /// Draw one standard normal variate with a deterministic Box--Muller transform.
  double normal()
  {
    if (has_spare_)
    {
      has_spare_ = false;
      return spare_;
    }

    constexpr double two_pi = 6.283185307179586476925286766559;
    const double radius     = std::sqrt(-2.0 * std::log(uniformOpen()));
    const double angle      = two_pi * uniformOpen();
    spare_                  = radius * std::sin(angle);
    has_spare_              = true;
    return radius * std::cos(angle);
  }

  /// Draw a standard normal conditioned on the closed interval [-2, 2].
  double truncatedNormal()
  {
    double value;
    do
      value = normal();
    while (value < -2.0 || value > 2.0);
    return value;
  }

private:
  /// Advance the exact SplitMix64 transition and return its mixed output.
  std::uint64_t next()
  {
    std::uint64_t value = (state_ += 0x9E3779B97F4A7C15ULL);
    value               = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value               = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
  }

  /// Convert the high 53 random bits to a value strictly between zero and one.
  double uniformOpen()
  {
    constexpr double inverse_two_to_53 = 1.0 / 9007199254740992.0;
    return (static_cast<double>(next() >> 11) + 0.5) * inverse_two_to_53;
  }

  std::uint64_t state_ = 0;
  bool has_spare_      = false;
  double spare_        = 0;
};

/// Multiply two extents while reporting architecture-size overflow explicitly.
std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* description)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string("PsiFormer initialization overflow in ") + description);
  return left * right;
}

/// Add two extents while reporting architecture-size overflow explicitly.
std::size_t checkedAdd(std::size_t left, std::size_t right, const char* description)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string("PsiFormer initialization overflow in ") + description);
  return left + right;
}

/// Return the checked scalar product of one tensor shape; scalar shapes have size one.
std::size_t checkedShapeProduct(const std::vector<std::size_t>& shape)
{
  std::size_t result = 1;
  for (std::size_t extent : shape)
    result = checkedMultiply(result, extent, "tensor shape");
  return result;
}

/// Mix one byte into an FNV-1a stream used to derive independent tensor seeds.
void mixByte(std::uint64_t& hash, std::uint8_t byte)
{
  hash ^= byte;
  hash *= 1099511628211ULL;
}

/// Mix one integer into a stable little-endian FNV-1a stream.
void mixInteger(std::uint64_t& hash, std::uint64_t value)
{
  for (int byte = 0; byte < 8; ++byte)
    mixByte(hash, static_cast<std::uint8_t>(value >> (8 * byte)));
}

/// Mix one length-delimited string into a stable FNV-1a stream.
void mixString(std::uint64_t& hash, const std::string& value)
{
  mixInteger(hash, value.size());
  for (unsigned char character : value)
    mixByte(hash, character);
}

/// Derive a refactor-resistant random stream from the profile, seed, and tensor identity.
std::uint64_t tensorSeed(const std::string& profile,
                         std::uint64_t seed,
                         const TensorSpecification& tensor)
{
  std::uint64_t hash = 14695981039346656037ULL;
  mixString(hash, profile);
  mixInteger(hash, seed);
  mixString(hash, tensor.module);
  mixString(hash, tensor.name);
  mixInteger(hash, static_cast<std::uint64_t>(tensor.role));
  mixInteger(hash, tensor.attention_block == NO_ATTENTION_BLOCK ? std::numeric_limits<std::uint64_t>::max()
                                                                 : tensor.attention_block);
  mixInteger(hash, tensor.shape.size());
  for (std::size_t extent : tensor.shape)
    mixInteger(hash, extent);
  return hash;
}

/// Enforce the architecture fixed by the versioned v1 profile before allocating.
void validateProfileShape(const ModelShape& model)
{
  if (model.spin_up_electrons == 0 || model.spin_down_electrons == 0)
    throw std::invalid_argument("deepqmc_psiformer_v1 requires positive spin-up and spin-down populations");
  if (model.nuclei == 0)
    throw std::invalid_argument("deepqmc_psiformer_v1 requires at least one nucleus");
  if (model.determinants != PROFILE_DETERMINANTS || model.feature_dimension != PROFILE_FEATURE_DIMENSION ||
      model.attention_heads != PROFILE_ATTENTION_HEADS || model.attention_blocks != PROFILE_ATTENTION_BLOCKS)
    throw std::invalid_argument(
        "deepqmc_psiformer_v1 requires determinants=16, feature_dimension=256, attention_heads=4, "
        "attention_blocks=4");

  checkedAdd(model.spin_up_electrons, model.spin_down_electrons, "electron count");
  checkedAdd(checkedMultiply(7, model.nuclei, "maximum embedding input width"), 1,
             "maximum embedding input width");
}

/// Append one tensor specification before canonical lexical ordering is applied.
void appendSpecification(std::vector<TensorSpecification>& tensors,
                         std::string module,
                         std::string name,
                         std::vector<std::size_t> shape,
                         ParameterRole role,
                         std::size_t attention_block,
                         InitializationLaw law,
                         double mean,
                         double distribution_standard_deviation,
                         double expected_standard_deviation)
{
  tensors.push_back({std::move(module), std::move(name), std::move(shape), role, attention_block, law, mean,
                     distribution_standard_deviation, expected_standard_deviation});
}

/// Build every DeepQMC-compatible leaf and put it in portable export order.
std::vector<TensorSpecification> makeTensorSpecifications(
    const ModelShape& model,
    const ExecutionEnvironment& environment)
{
  const std::size_t electrons = checkedAdd(model.spin_up_electrons, model.spin_down_electrons, "electron count");
  const std::size_t channels = checkedMultiply(model.determinants, electrons, "determinant-orbital channels");
  const std::size_t channels_per_nucleus =
      environment.boundary == BoundaryCondition::PERIODIC ? 7 : 4;
  const std::size_t input_width =
      checkedAdd(checkedMultiply(channels_per_nucleus, model.nuclei, "embedding input width"), 1,
                 "embedding input width");
  const double feature_sigma = 1.0 / std::sqrt(static_cast<double>(model.feature_dimension));
  const double embedding_sigma = 1.0 / std::sqrt(static_cast<double>(input_width));
  const std::string prefix = "neural_network_wave_function/~/";

  std::vector<TensorSpecification> tensors;
  const bool has_same_spin_pair = model.spin_up_electrons > 1 || model.spin_down_electrons > 1;
  tensors.reserve(8 + static_cast<std::size_t>(has_same_spin_pair) + 8 * model.attention_blocks);

  appendSpecification(tensors, prefix + "electronic_cusp_asymptotic", "anti_alpha", {},
                      ParameterRole::CUSP_OPPOSITE_ALPHA, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  // Haiku materializes this trainable scalar only when the system contains a
  // same-spin electron pair.  Preserving that conditional leaf is necessary
  // for canonical round trips with two-electron pseudopotential calculations.
  if (has_same_spin_pair)
    appendSpecification(tensors, prefix + "electronic_cusp_asymptotic", "same_alpha", {},
                        ParameterRole::CUSP_SAME_ALPHA, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  appendSpecification(tensors, prefix + "exponential_envelopes", "pi_down", {channels, model.nuclei},
                      ParameterRole::ENVELOPE_PI_DOWN, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  appendSpecification(tensors, prefix + "exponential_envelopes", "pi_up", {channels, model.nuclei},
                      ParameterRole::ENVELOPE_PI_UP, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  appendSpecification(tensors, prefix + "exponential_envelopes", "zetas_down", {channels, model.nuclei},
                      ParameterRole::ENVELOPE_ZETA_DOWN, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  appendSpecification(tensors, prefix + "exponential_envelopes", "zetas_up", {channels, model.nuclei},
                      ParameterRole::ENVELOPE_ZETA_UP, NO_ATTENTION_BLOCK, InitializationLaw::CONSTANT, 1, 0, 0);
  appendSpecification(tensors, prefix + "omni_net/~/Backflow/~/mlp/linear_0", "w",
                      {model.feature_dimension, channels}, ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK,
                      InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
  appendSpecification(tensors, prefix + "omni_net/~/Backflow_1/~/mlp/linear_0", "w",
                      {model.feature_dimension, channels}, ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK,
                      InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
  appendSpecification(tensors, prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w",
                      {input_width, model.feature_dimension}, ParameterRole::ELECTRON_EMBEDDING_WEIGHT,
                      NO_ATTENTION_BLOCK, InitializationLaw::TRUNCATED_NORMAL, 0, embedding_sigma,
                      TRUNCATED_NORMAL_STANDARD_DEVIATION * embedding_sigma);

  for (std::size_t block = 0; block < model.attention_blocks; ++block)
  {
    const std::string layer = block == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(block);
    const std::string base = prefix + "omni_net/~/electron_gnn/~/" + layer +
        "/~/node_attention_electron_update_feature/";
    appendSpecification(tensors, base + "mlp/linear_0", "b", {model.feature_dimension},
                        ParameterRole::UPDATE_HIDDEN_BIAS, block, InitializationLaw::NORMAL, 0, feature_sigma,
                        feature_sigma);
    appendSpecification(tensors, base + "mlp/linear_0", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::UPDATE_HIDDEN_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
    appendSpecification(tensors, base + "mlp/linear_1", "b", {model.feature_dimension},
                        ParameterRole::UPDATE_OUTPUT_BIAS, block, InitializationLaw::NORMAL, 0, feature_sigma,
                        feature_sigma);
    appendSpecification(tensors, base + "mlp/linear_1", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::UPDATE_OUTPUT_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
    appendSpecification(tensors, base + "multi_head_attention/key", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::ATTENTION_KEY_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
    appendSpecification(tensors, base + "multi_head_attention/linear", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::ATTENTION_OUTPUT_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
    appendSpecification(tensors, base + "multi_head_attention/query", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::ATTENTION_QUERY_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
    appendSpecification(tensors, base + "multi_head_attention/value", "w",
                        {model.feature_dimension, model.feature_dimension}, ParameterRole::ATTENTION_VALUE_WEIGHT,
                        block, InitializationLaw::NORMAL, 0, feature_sigma, feature_sigma);
  }

  std::sort(tensors.begin(), tensors.end(), [](const TensorSpecification& left, const TensorSpecification& right) {
    return left.module < right.module || (left.module == right.module && left.name < right.name);
  });
  return tensors;
}

/// Compute stable population statistics for one newly filled flat interval.
TensorInitializationDiagnostic makeDiagnostic(const TensorSpecification& tensor,
                                               const std::vector<double>& values,
                                               std::size_t begin,
                                               std::size_t end)
{
  long double mean = 0;
  long double sum_squared_deviation = 0;
  std::size_t count = 0;
  double minimum = values[begin];
  double maximum = values[begin];
  for (std::size_t index = begin; index < end; ++index)
  {
    const double value = values[index];
    minimum            = std::min(minimum, value);
    maximum            = std::max(maximum, value);
    ++count;
    const long double delta = static_cast<long double>(value) - mean;
    mean += delta / static_cast<long double>(count);
    sum_squared_deviation += delta * (static_cast<long double>(value) - mean);
  }
  const double standard_deviation =
      std::sqrt(static_cast<double>(sum_squared_deviation / static_cast<long double>(count)));

  return {tensor.module,
          tensor.name,
          tensor.role,
          tensor.attention_block,
          tensor.law,
          count,
          tensor.mean,
          tensor.distribution_standard_deviation,
          tensor.expected_standard_deviation,
          static_cast<double>(mean),
          standard_deviation,
          minimum,
          maximum};
}

} // namespace

// Construct a complete neutral parameter store from the versioned profile.
InitializedPsiFormerParameters initializePsiFormerParameters(const ModelShape& model_shape,
                                                             std::uint64_t seed,
                                                             const std::string& profile,
                                                             ExecutionEnvironment environment)
{
  if (profile != DEEPQMC_PSIFORMER_V1)
    throw std::invalid_argument("Unknown PsiFormer initialization profile: " + profile);
  validateProfileShape(model_shape);

  const std::vector<TensorSpecification> specifications =
      makeTensorSpecifications(model_shape, environment);
  InitializedPsiFormerParameters initialized;
  initialized.model_shape = model_shape;
  initialized.profile     = profile;
  initialized.seed        = seed;
  initialized.diagnostics.random_generator = PSIFORMER_INITIALIZATION_RNG_V1;
  initialized.layouts.reserve(specifications.size());
  initialized.diagnostics.tensors.reserve(specifications.size());

  std::size_t parameter_count = 0;
  for (const TensorSpecification& tensor : specifications)
    parameter_count = checkedAdd(parameter_count, checkedShapeProduct(tensor.shape), "flat parameter count");
  initialized.values.reserve(parameter_count);

  for (const TensorSpecification& tensor : specifications)
  {
    const std::size_t begin = initialized.values.size();
    const std::size_t size  = checkedShapeProduct(tensor.shape);
    SplitMix64NormalGenerator random(tensorSeed(profile, seed, tensor));
    for (std::size_t element = 0; element < size; ++element)
    {
      double value = tensor.mean;
      if (tensor.law == InitializationLaw::NORMAL)
        value += tensor.distribution_standard_deviation * random.normal();
      else if (tensor.law == InitializationLaw::TRUNCATED_NORMAL)
        value += tensor.distribution_standard_deviation * random.truncatedNormal();
      initialized.values.push_back(value);
    }
    const std::size_t end = initialized.values.size();
    initialized.layouts.push_back({tensor.module, tensor.name, tensor.shape, begin, end});
    initialized.diagnostics.tensors.push_back(makeDiagnostic(tensor, initialized.values, begin, end));
    if (tensor.law == InitializationLaw::CONSTANT)
      initialized.diagnostics.constant_parameter_count += size;
    else
      initialized.diagnostics.random_parameter_count += size;
  }

  initialized.diagnostics.tensor_count    = initialized.layouts.size();
  initialized.diagnostics.parameter_count = initialized.values.size();

  // Reuse the production plan validator as a final assertion that generated
  // names, shapes, intervals, and architecture completeness remain compatible.
  const PsiFormerExecutionPlan plan(model_shape, initialized.layouts, environment);
  if (plan.parameterCount() != initialized.values.size())
    throw std::logic_error("Generated PsiFormer layout and flat parameter vector disagree");
  return initialized;
}

// Provide readable law names for input reports and restart diagnostics.
const char* initializationLawName(InitializationLaw law)
{
  switch (law)
  {
  case InitializationLaw::CONSTANT:
    return "constant";
  case InitializationLaw::NORMAL:
    return "normal";
  case InitializationLaw::TRUNCATED_NORMAL:
    return "truncated_normal";
  }
  return "unknown";
}

} // namespace qmcplusplus::psiformer
