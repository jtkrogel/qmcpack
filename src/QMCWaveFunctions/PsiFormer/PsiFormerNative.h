//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerNative.h
 * @brief Native implementation of the DeepQMC PsiFormer architecture.
 *
 * This implementation evaluates a fixed, HDF5-exported PsiFormer model without
 * JAX or another external automatic-differentiation framework. It contains a
 * small, purpose-built differentiation engine: spatial derivatives are
 * propagated forward as first- and diagonal-second-derivative coordinate jets,
 * while parameter derivatives are accumulated in reverse through an internal
 * computation graph. Local-energy parameter derivatives reverse through the
 * complete coordinate-jet program, producing the required mixed parameter and
 * spatial derivatives. Each primitive operation supplies explicitly coded
 * analytic propagation rules or vector-Jacobian products (VJPs).
 *
 * Consequently, derivatives are assembled automatically by composing these
 * hand-written primitive rules; they are not finite-difference estimates, but
 * neither are they fully expanded, hand-derived formulas for each top-level
 * PsiFormer observable. Parameter derivatives support the standalone
 * validation driver and QMCPACK selected-parameter optimization.
 *
 * Define PSIFORMER_LIBRARY before including this file to omit the standalone
 * comparison driver. The QMCPACK WaveFunctionComponent does this in
 * PsiFormerWF.cpp.
 */

//////////////////////////////////////////////////////////////////////////////////////
// INCLUSION RESTRICTION
// This implementation header defines non-inline functions and must be included by
// exactly one translation unit in each linked target. Define PSIFORMER_LIBRARY before
// inclusion when embedding it in QMCPACK or a test to suppress the standalone main().
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PSIFORMER_NATIVE_H
#define QMCPLUSPLUS_PSIFORMER_NATIVE_H

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pf
{

// Tensor storage and NumPy-style broadcasting utilities.
//
// Tensors use row-major storage. Keeping these operations local makes the
// imported model independent of a third-party tensor or
// automatic-differentiation library.
using Shape = std::vector<size_t>;

/// Return the number of scalar elements represented by a tensor shape.
size_t product(const Shape& shape)
{
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
}

/** Lightweight, owning, row-major tensor used throughout the native evaluator.
 */
struct Tensor
{
  Shape shape;
  std::vector<double> x;

  /// Construct an empty sentinel tensor.
  Tensor() = default;

  /// Allocate a shaped tensor and initialize every element to one value.
  explicit Tensor(Shape tensor_shape, double initial_value = 0)
      : shape(std::move(tensor_shape)), x(product(shape), initial_value)
  {}

  /// Construct a shaped tensor from an existing row-major value vector.
  Tensor(Shape tensor_shape, std::vector<double> values) : shape(std::move(tensor_shape)), x(std::move(values))
  {
    if (x.size() != product(shape))
      throw std::runtime_error("Tensor size mismatch");
  }

  /// Report whether this is the empty sentinel rather than a scalar tensor.
  bool empty() const { return shape.empty() && x.empty(); }

  /// Return the number of stored scalar elements.
  size_t size() const { return x.size(); }
};

/// Compute row-major strides for each tensor axis.
Shape strides(const Shape& shape)
{
  Shape row_major_strides(shape.size(), 1);
  for (int axis = int(shape.size()) - 2; axis >= 0; --axis)
    row_major_strides[axis] = row_major_strides[axis + 1] * shape[axis + 1];
  return row_major_strides;
}

/// Convert a row-major flat offset into a multidimensional index.
Shape unravel(size_t flat_index, const Shape& shape)
{
  Shape multi_index(shape.size());
  const Shape row_major_strides = strides(shape);
  for (size_t axis = 0; axis < shape.size(); ++axis)
  {
    multi_index[axis] = flat_index / row_major_strides[axis];
    flat_index %= row_major_strides[axis];
  }
  return multi_index;
}

/// Convert a multidimensional index into its row-major flat offset.
size_t ravel(const Shape& multi_index, const Shape& shape)
{
  const Shape row_major_strides = strides(shape);
  size_t flat_index             = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis)
    flat_index += multi_index[axis] * row_major_strides[axis];
  return flat_index;
}

/// Determine the NumPy-style broadcast result shape of two tensors.
Shape broadcast_shape(const Shape& left_shape, const Shape& right_shape)
{
  const size_t output_rank = std::max(left_shape.size(), right_shape.size());
  Shape output_shape(output_rank, 1);

  // Align trailing axes, following NumPy broadcasting rules. Each aligned pair
  // must match or contain a singleton dimension.
  for (size_t output_axis = 0; output_axis < output_rank; ++output_axis)
  {
    const size_t left_extent =
        output_axis < output_rank - left_shape.size() ? 1 : left_shape[output_axis - (output_rank - left_shape.size())];
    const size_t right_extent = output_axis < output_rank - right_shape.size()
        ? 1
        : right_shape[output_axis - (output_rank - right_shape.size())];
    if (left_extent != right_extent && left_extent != 1 && right_extent != 1)
      throw std::runtime_error("incompatible broadcast");
    output_shape[output_axis] = std::max(left_extent, right_extent);
  }
  return output_shape;
}

/// Map one broadcast-output index to the corresponding input flat offset.
size_t broadcast_offset(const Shape& output_index, const Shape& input_shape)
{
  const size_t leading_output_axes = output_index.size() - input_shape.size();
  Shape input_index(input_shape.size());
  for (size_t input_axis = 0; input_axis < input_shape.size(); ++input_axis)
    input_index[input_axis] = input_shape[input_axis] == 1 ? 0 : output_index[leading_output_axes + input_axis];
  return ravel(input_index, input_shape);
}

/// Sum a broadcast gradient back into the original input shape.
Tensor unbroadcast(const Tensor& broadcast_gradient, const Shape& target_shape)
{
  // Several broadcast output elements may map to one singleton input element;
  // sum all of those contributions into the original input shape.
  Tensor target_gradient(target_shape);
  for (size_t output_flat_index = 0; output_flat_index < broadcast_gradient.size(); ++output_flat_index)
  {
    const Shape output_index = unravel(output_flat_index, broadcast_gradient.shape);
    target_gradient.x[broadcast_offset(output_index, target_shape)] += broadcast_gradient.x[output_flat_index];
  }
  return target_gradient;
}

/// Apply a broadcast-aware binary scalar operation to two plain tensors.
Tensor elem_binary(const Tensor& left, const Tensor& right, const std::function<double(double, double)>& operation)
{
  const Shape output_shape = broadcast_shape(left.shape, right.shape);
  Tensor output(output_shape);
  for (size_t output_flat_index = 0; output_flat_index < output.size(); ++output_flat_index)
  {
    const Shape output_index    = unravel(output_flat_index, output_shape);
    const double left_value     = left.x[broadcast_offset(output_index, left.shape)];
    const double right_value    = right.x[broadcast_offset(output_index, right.shape)];
    output.x[output_flat_index] = operation(left_value, right_value);
  }
  return output;
}

/// Apply a scalar operation independently to every element of a plain tensor.
Tensor elem_unary(const Tensor& input, const std::function<double(double)>& operation)
{
  Tensor output(input.shape);
  for (size_t element = 0; element < input.size(); ++element)
    output.x[element] = operation(input.x[element]);
  return output;
}

// Differentiation graph and coordinate jets.
//
// A coordinate-dependent node stores d1[c,...] = d(value)/dR_c and the diagonal
// d2[c,...] = d^2(value)/dR_c^2. Mixed second derivatives are not needed for
// the electronic Laplacian. Each parent edge stores a VJP for reverse parameter
// derivatives.
// Forward declaration used by graph edge shared pointers.
struct Node;
using NodePtr = std::shared_ptr<Node>;

/** Adjoint of a coordinate jet, with sensitivities to its value and spatial
 * first- and diagonal-second-derivative components. */
struct JetAdjoint
{
  Tensor value;
  Tensor d1;
  Tensor d2;
};

/// Connect a graph node to one parent and define its value and coordinate-jet VJPs.
struct Edge
{
  NodePtr parent;
  std::function<Tensor(const Tensor&)> vjp;
  std::function<JetAdjoint(const JetAdjoint&)> jet_vjp;
};

/** A tensor value together with spatial jets and reverse-mode graph edges. */
struct Node
{
  Tensor value;
  Tensor d1; // First Cartesian-coordinate derivatives: [coordinate, value...].
  Tensor d2; // Diagonal second derivatives with the same jet shape.
  std::vector<Edge> parents;
  std::string parameter_key;
};

/// Construct a graph node from its value, coordinate jets, and parent edges.
NodePtr node(Tensor value,
             Tensor first_coordinate_derivative  = {},
             Tensor second_coordinate_derivative = {},
             std::vector<Edge> parent_edges      = {})
{
  auto result     = std::make_shared<Node>();
  result->value   = std::move(value);
  result->d1      = std::move(first_coordinate_derivative);
  result->d2      = std::move(second_coordinate_derivative);
  result->parents = std::move(parent_edges);
  return result;
}

/// Wrap a coordinate- and parameter-independent tensor in a graph node.
NodePtr constant(Tensor value) { return node(std::move(value)); }

/// Construct a scalar constant node.
NodePtr scalar(double value) { return constant(Tensor({}, std::vector<double>{value})); }

/** Create the coordinate leaf and seed its first derivative with the identity
 * matrix. */
NodePtr coordinates(const Tensor& value)
{
  // Flattened Cartesian coordinates form the leading jet axis. Seeding an
  // identity matrix makes dR_i/dR_j = delta_ij; all second derivatives vanish.
  Tensor first_coordinate_derivative(Shape{value.size()});
  first_coordinate_derivative.shape.insert(first_coordinate_derivative.shape.end(), value.shape.begin(),
                                           value.shape.end());
  first_coordinate_derivative.x.assign(value.size() * value.size(), 0);
  for (size_t coordinate = 0; coordinate < value.size(); ++coordinate)
    first_coordinate_derivative.x[coordinate * value.size() + coordinate] = 1;
  Tensor second_coordinate_derivative(first_coordinate_derivative.shape);
  return node(value, std::move(first_coordinate_derivative), std::move(second_coordinate_derivative));
}

/// Broadcast every coordinate slice of a derivative jet to an output value shape.
Tensor broadcast_jet(const Tensor& jet, const Shape& source_shape, const Shape& output_shape, size_t coordinate_count)
{
  Shape output_jet_shape{coordinate_count};
  output_jet_shape.insert(output_jet_shape.end(), output_shape.begin(), output_shape.end());
  Tensor output_jet(output_jet_shape);
  if (jet.empty())
    return output_jet;

  const size_t source_size = product(source_shape);
  const size_t output_size = product(output_shape);
  for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
    for (size_t output_flat_index = 0; output_flat_index < output_size; ++output_flat_index)
    {
      const Shape output_index = unravel(output_flat_index, output_shape);
      output_jet.x[coordinate * output_size + output_flat_index] =
          jet.x[coordinate * source_size + broadcast_offset(output_index, source_shape)];
    }
  return output_jet;
}

// Primitive differentiable operations. Each operation propagates the coordinate
// jets and records the VJP needed to send a parameter adjoint back to its
// parents.
/// Apply an elementwise scalar function while constructing spatial jets and its VJP.
NodePtr unary(const NodePtr& input,
              const std::function<double(double)>& function,
              const std::function<double(double)>& function_first_derivative,
              const std::function<double(double)>& function_second_derivative,
              const std::function<double(double)>& function_third_derivative)
{
  // Cache f and its first three derivatives. Third derivatives enter only when
  // reverse-differentiating the diagonal second-coordinate jet.
  Tensor value                   = elem_unary(input->value, function);
  const Tensor first_derivative  = elem_unary(input->value, function_first_derivative);
  const Tensor second_derivative = elem_unary(input->value, function_second_derivative);
  const Tensor third_derivative  = elem_unary(input->value, function_third_derivative);
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;

  if (!input->d1.empty())
  {
    first_coordinate_derivative   = Tensor(input->d1.shape);
    second_coordinate_derivative  = Tensor(input->d2.shape);
    const size_t value_size       = input->value.size();
    const size_t coordinate_count = input->d1.size() / value_size;
    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t element = 0; element < value_size; ++element)
      {
        const size_t jet_index                         = coordinate * value_size + element;
        const double input_first_coordinate_derivative = input->d1.x[jet_index];
        first_coordinate_derivative.x[jet_index]  = first_derivative.x[element] * input_first_coordinate_derivative;
        second_coordinate_derivative.x[jet_index] = first_derivative.x[element] * input->d2.x[jet_index] +
            second_derivative.x[element] * input_first_coordinate_derivative * input_first_coordinate_derivative;
      }
  }

  // The ordinary value VJP supplies d log|psi| / d theta.
  const Shape input_shape = input->value.shape;
  auto input_vjp          = [first_derivative, input_shape](const Tensor& upstream_gradient) {
    Tensor scaled_gradient = elem_binary(upstream_gradient, first_derivative,
                                         [](double gradient, double derivative) { return gradient * derivative; });
    return unbroadcast(scaled_gradient, input_shape);
  };

  // The lifted VJP differentiates all three outputs (value, d1, and d2)
  // with respect to all three input jet components.
  const Tensor input_first_coordinate_derivative  = input->d1;
  const Tensor input_second_coordinate_derivative = input->d2;
  auto input_jet_vjp                              = [first_derivative, second_derivative, third_derivative, input_shape,
                                                     input_first_coordinate_derivative,
                                                     input_second_coordinate_derivative](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor(input_shape);
    for (size_t element = 0; element < result.value.size(); ++element)
      result.value.x[element] = upstream.value.x[element] * first_derivative.x[element];

    if (!input_first_coordinate_derivative.empty())
    {
      result.d1                     = Tensor(input_first_coordinate_derivative.shape);
      result.d2                     = Tensor(input_second_coordinate_derivative.shape);
      const size_t value_size       = result.value.size();
      const size_t coordinate_count = input_first_coordinate_derivative.size() / value_size;
      for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
        for (size_t element = 0; element < value_size; ++element)
        {
          const size_t jet_index                          = coordinate * value_size + element;
          const double input_first_derivative_element     = input_first_coordinate_derivative.x[jet_index];
          const double input_second_derivative_element    = input_second_coordinate_derivative.x[jet_index];
          const double function_first_derivative_element  = first_derivative.x[element];
          const double function_second_derivative_element = second_derivative.x[element];
          const double function_third_derivative_element  = third_derivative.x[element];
          result.value.x[element] +=
              upstream.d1.x[jet_index] * function_second_derivative_element * input_first_derivative_element +
              upstream.d2.x[jet_index] *
                  (function_second_derivative_element * input_second_derivative_element +
                   function_third_derivative_element * input_first_derivative_element * input_first_derivative_element);
          result.d1.x[jet_index] = upstream.d1.x[jet_index] * function_first_derivative_element +
              2 * upstream.d2.x[jet_index] * function_second_derivative_element * input_first_derivative_element;
          result.d2.x[jet_index] = upstream.d2.x[jet_index] * function_first_derivative_element;
        }
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Add two broadcast-compatible graph nodes and propagate both derivative forms.
NodePtr add(const NodePtr& left, const NodePtr& right)
{
  Tensor value = elem_binary(left->value, right->value,
                             [](double left_value, double right_value) { return left_value + right_value; });
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;

  // Addition applies the same broadcast-and-add operation independently to
  // the value and both spatial jets.
  size_t coordinate_count = 0;
  if (!left->d1.empty())
    coordinate_count = left->d1.shape[0];
  else if (!right->d1.empty())
    coordinate_count = right->d1.shape[0];

  if (coordinate_count)
  {
    const Tensor left_first_coordinate_derivative =
        broadcast_jet(left->d1, left->value.shape, value.shape, coordinate_count);
    const Tensor right_first_coordinate_derivative =
        broadcast_jet(right->d1, right->value.shape, value.shape, coordinate_count);
    const Tensor left_second_coordinate_derivative =
        broadcast_jet(left->d2, left->value.shape, value.shape, coordinate_count);
    const Tensor right_second_coordinate_derivative =
        broadcast_jet(right->d2, right->value.shape, value.shape, coordinate_count);
    first_coordinate_derivative  = elem_binary(left_first_coordinate_derivative, right_first_coordinate_derivative,
                                               [](double a, double b) { return a + b; });
    second_coordinate_derivative = elem_binary(left_second_coordinate_derivative, right_second_coordinate_derivative,
                                               [](double a, double b) { return a + b; });
  }

  // Reverse broadcasting by reducing each output adjoint to its parent's
  // original value or jet shape.
  const Shape left_shape  = left->value.shape;
  const Shape right_shape = right->value.shape;
  auto left_vjp           = [left_shape](const Tensor& upstream) { return unbroadcast(upstream, left_shape); };
  auto right_vjp          = [right_shape](const Tensor& upstream) { return unbroadcast(upstream, right_shape); };

  const Shape left_jet_shape  = left->d1.shape;
  const Shape right_jet_shape = right->d1.shape;
  auto left_jet_vjp           = [left_shape, left_jet_shape](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = unbroadcast(upstream.value, left_shape);
    if (!left_jet_shape.empty())
    {
      result.d1 = unbroadcast(upstream.d1, left_jet_shape);
      result.d2 = unbroadcast(upstream.d2, left_jet_shape);
    }
    return result;
  };
  auto right_jet_vjp = [right_shape, right_jet_shape](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = unbroadcast(upstream.value, right_shape);
    if (!right_jet_shape.empty())
    {
      result.d1 = unbroadcast(upstream.d1, right_jet_shape);
      result.d2 = unbroadcast(upstream.d2, right_jet_shape);
    }
    return result;
  };

  std::vector<Edge> parent_edges{{left, std::move(left_vjp), std::move(left_jet_vjp)},
                                 {right, std::move(right_vjp), std::move(right_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Negate a graph node elementwise.
NodePtr neg(const NodePtr& input)
{
  return unary(
      input, [](double value) { return -value; }, [](double) { return -1.; }, [](double) { return 0.; },
      [](double) { return 0.; });
}

/// Multiply two broadcast-compatible graph nodes using first- and second-order product rules.
NodePtr mul(const NodePtr& left, const NodePtr& right)
{
  Tensor value = elem_binary(left->value, right->value,
                             [](double left_value, double right_value) { return left_value * right_value; });
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  // Broadcast both operands once and apply the first- and second-order
  // product rules in the common output shape.
  size_t coordinate_count = !left->d1.empty() ? left->d1.shape[0] : (!right->d1.empty() ? right->d1.shape[0] : 0);
  Tensor left_first_coordinate_derivative;
  Tensor right_first_coordinate_derivative;
  Tensor left_second_coordinate_derivative;
  Tensor right_second_coordinate_derivative;
  if (coordinate_count)
  {
    left_first_coordinate_derivative   = broadcast_jet(left->d1, left->value.shape, value.shape, coordinate_count);
    right_first_coordinate_derivative  = broadcast_jet(right->d1, right->value.shape, value.shape, coordinate_count);
    left_second_coordinate_derivative  = broadcast_jet(left->d2, left->value.shape, value.shape, coordinate_count);
    right_second_coordinate_derivative = broadcast_jet(right->d2, right->value.shape, value.shape, coordinate_count);
    Shape jet_shape{coordinate_count};
    jet_shape.insert(jet_shape.end(), value.shape.begin(), value.shape.end());
    first_coordinate_derivative  = Tensor(jet_shape);
    second_coordinate_derivative = Tensor(jet_shape);
    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t element = 0; element < value.size(); ++element)
      {
        const Shape output_index                 = unravel(element, value.shape);
        const double left_value_element          = left->value.x[broadcast_offset(output_index, left->value.shape)];
        const double right_value_element         = right->value.x[broadcast_offset(output_index, right->value.shape)];
        const size_t jet_index                   = coordinate * value.size() + element;
        first_coordinate_derivative.x[jet_index] = left_first_coordinate_derivative.x[jet_index] * right_value_element +
            left_value_element * right_first_coordinate_derivative.x[jet_index];
        second_coordinate_derivative.x[jet_index] =
            left_second_coordinate_derivative.x[jet_index] * right_value_element +
            2 * left_first_coordinate_derivative.x[jet_index] * right_first_coordinate_derivative.x[jet_index] +
            left_value_element * right_second_coordinate_derivative.x[jet_index];
      }
  }

  // Retain the inexpensive value-only VJPs for log-wavefunction gradients.
  const Tensor right_value = right->value;
  const Shape left_shape   = left->value.shape;
  auto left_vjp            = [right_value, left_shape](const Tensor& upstream) {
    return unbroadcast(elem_binary(upstream, right_value, [](double a, double b) { return a * b; }), left_shape);
  };
  const Tensor left_value = left->value;
  const Shape right_shape = right->value.shape;
  auto right_vjp          = [left_value, right_shape](const Tensor& upstream) {
    return unbroadcast(elem_binary(upstream, left_value, [](double a, double b) { return a * b; }), right_shape);
  };

  // For one operand, downstream jet adjoints contribute to its value through
  // the other operand's d1/d2, and to its jets through the usual product rule.
  const Shape output_shape    = value.shape;
  const Shape left_jet_shape  = left->d1.shape;
  const Shape right_jet_shape = right->d1.shape;
  auto make_operand_jet_vjp =
      [output_shape, coordinate_count](const Tensor& other_value, const Tensor& other_first_coordinate_derivative,
                                       const Tensor& other_second_coordinate_derivative, const Shape& operand_shape,
                                       const Shape& operand_jet_shape) {
        return [output_shape, coordinate_count, other_value, other_first_coordinate_derivative,
                other_second_coordinate_derivative, operand_shape, operand_jet_shape](const JetAdjoint& upstream) {
          Tensor value_gradient(output_shape);
          Tensor first_coordinate_gradient;
          Tensor second_coordinate_gradient;
          if (coordinate_count)
          {
            Shape jet_shape{coordinate_count};
            jet_shape.insert(jet_shape.end(), output_shape.begin(), output_shape.end());
            first_coordinate_gradient  = Tensor(jet_shape);
            second_coordinate_gradient = Tensor(jet_shape);
          }
          for (size_t element = 0; element < value_gradient.size(); ++element)
          {
            const Shape output_index         = unravel(element, output_shape);
            const double other_value_element = other_value.x[broadcast_offset(output_index, other_value.shape)];
            value_gradient.x[element]        = upstream.value.x[element] * other_value_element;
            for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
            {
              const size_t jet_index = coordinate * value_gradient.size() + element;
              value_gradient.x[element] += upstream.d1.x[jet_index] * other_first_coordinate_derivative.x[jet_index] +
                  upstream.d2.x[jet_index] * other_second_coordinate_derivative.x[jet_index];
              first_coordinate_gradient.x[jet_index] = upstream.d1.x[jet_index] * other_value_element +
                  2 * upstream.d2.x[jet_index] * other_first_coordinate_derivative.x[jet_index];
              second_coordinate_gradient.x[jet_index] = upstream.d2.x[jet_index] * other_value_element;
            }
          }
          JetAdjoint result;
          result.value = unbroadcast(value_gradient, operand_shape);
          if (!operand_jet_shape.empty())
          {
            result.d1 = unbroadcast(first_coordinate_gradient, operand_jet_shape);
            result.d2 = unbroadcast(second_coordinate_gradient, operand_jet_shape);
          }
          return result;
        };
      };
  auto left_jet_vjp  = make_operand_jet_vjp(right_value, right_first_coordinate_derivative,
                                            right_second_coordinate_derivative, left_shape, left_jet_shape);
  auto right_jet_vjp = make_operand_jet_vjp(left_value, left_first_coordinate_derivative,
                                            left_second_coordinate_derivative, right_shape, right_jet_shape);

  std::vector<Edge> parent_edges{{left, std::move(left_vjp), std::move(left_jet_vjp)},
                                 {right, std::move(right_vjp), std::move(right_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Compute an elementwise reciprocal node and its analytic derivatives.
NodePtr recip(const NodePtr& input)
{
  return unary(
      input, [](double value) { return 1 / value; }, [](double value) { return -1 / (value * value); },
      [](double value) { return 2 / (value * value * value); },
      [](double value) { return -6 / (value * value * value * value); });
}

/// Divide two graph nodes by composing multiplication with a reciprocal.
NodePtr divide(const NodePtr& numerator, const NodePtr& denominator) { return mul(numerator, recip(denominator)); }

/// Compute an elementwise exponential node.
NodePtr exp_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::exp(value); }, [](double value) { return std::exp(value); },
      [](double value) { return std::exp(value); }, [](double value) { return std::exp(value); });
}

/// Compute an elementwise natural-logarithm node.
NodePtr log_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::log(value); }, [](double value) { return 1 / value; },
      [](double value) { return -1 / (value * value); }, [](double value) { return 2 / (value * value * value); });
}

/// Compute log(1+x) elementwise with stable standard-library evaluation.
NodePtr log1p_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::log1p(value); }, [](double value) { return 1 / (1 + value); },
      [](double value) { return -1 / ((1 + value) * (1 + value)); },
      [](double value) { return 2 / ((1 + value) * (1 + value) * (1 + value)); });
}

/// Compute an elementwise square-root node.
NodePtr sqrt_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::sqrt(value); }, [](double value) { return .5 / std::sqrt(value); },
      [](double value) { return -.25 / std::pow(value, 1.5); },
      [](double value) { return .375 / std::pow(value, 2.5); });
}

/// Compute an elementwise hyperbolic-tangent node.
NodePtr tanh_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::tanh(value); },
      [](double value) {
        const double tanh_value = std::tanh(value);
        return 1 - tanh_value * tanh_value;
      },
      [](double value) {
        const double tanh_value = std::tanh(value);
        return -2 * tanh_value * (1 - tanh_value * tanh_value);
      },
      [](double value) {
        const double tanh_value = std::tanh(value);
        return -2 * (1 - tanh_value * tanh_value) * (1 - 3 * tanh_value * tanh_value);
      });
}

/// Compute elementwise absolute values, using zero derivative at the nondifferentiable origin.
NodePtr abs_node(const NodePtr& input)
{
  return unary(
      input, [](double value) { return std::abs(value); },
      [](double value) { return value > 0 ? 1. : (value < 0 ? -1. : 0.); }, [](double) { return 0.; },
      [](double) { return 0.; });
}

// Tensor transformations and reductions.
/// Change tensor value axes without changing storage order or element count.
NodePtr reshape(const NodePtr& input, Shape output_shape)
{
  if (product(output_shape) != input->value.size())
    throw std::runtime_error("reshape changes the number of tensor elements");

  Tensor value(output_shape, input->value.x);
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;

  // The coordinate axis is stored before all value axes and is not reshaped.
  if (!input->d1.empty())
  {
    Shape jet_shape{input->d1.shape[0]};
    jet_shape.insert(jet_shape.end(), output_shape.begin(), output_shape.end());
    first_coordinate_derivative  = Tensor(jet_shape, input->d1.x);
    second_coordinate_derivative = Tensor(jet_shape, input->d2.x);
  }

  const Shape input_shape     = input->value.shape;
  const Shape input_jet_shape = input->d1.shape;
  auto input_vjp = [input_shape](const Tensor& upstream_gradient) { return Tensor(input_shape, upstream_gradient.x); };
  auto input_jet_vjp = [input_shape, input_jet_shape](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor(input_shape, upstream.value.x);
    if (!input_jet_shape.empty())
    {
      result.d1 = Tensor(input_jet_shape, upstream.d1.x);
      result.d2 = Tensor(input_jet_shape, upstream.d2.x);
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Permute the axes of a plain row-major tensor.
Tensor transpose_t(const Tensor& input, const std::vector<size_t>& axes)
{
  Shape output_shape;
  for (size_t input_axis : axes)
    output_shape.push_back(input.shape[input_axis]);

  Tensor output(output_shape);
  for (size_t output_flat_index = 0; output_flat_index < output.size(); ++output_flat_index)
  {
    const Shape output_index = unravel(output_flat_index, output_shape);
    Shape input_index(input.shape.size());
    for (size_t output_axis = 0; output_axis < axes.size(); ++output_axis)
      input_index[axes[output_axis]] = output_index[output_axis];
    output.x[output_flat_index] = input.x[ravel(input_index, input.shape)];
  }
  return output;
}

/// Permute value axes and apply the corresponding permutation to jets and adjoints.
NodePtr transpose(const NodePtr& input, const std::vector<size_t>& axes)
{
  Tensor value = transpose_t(input->value, axes);
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;

  if (!input->d1.empty())
  {
    // Coordinate jets have a leading coordinate axis. Preserve it and shift all
    // value-axis indices by one when applying the requested transpose.
    std::vector<size_t> jet_axes{0};
    for (size_t value_axis : axes)
      jet_axes.push_back(value_axis + 1);
    first_coordinate_derivative  = transpose_t(input->d1, jet_axes);
    second_coordinate_derivative = transpose_t(input->d2, jet_axes);
  }

  // The VJP of a transpose applies the inverse permutation.
  std::vector<size_t> inverse_axes(axes.size());
  for (size_t output_axis = 0; output_axis < axes.size(); ++output_axis)
    inverse_axes[axes[output_axis]] = output_axis;
  auto input_vjp = [inverse_axes](const Tensor& upstream_gradient) {
    return transpose_t(upstream_gradient, inverse_axes);
  };
  const bool has_input_jets = !input->d1.empty();
  auto input_jet_vjp        = [inverse_axes, has_input_jets](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = transpose_t(upstream.value, inverse_axes);
    if (has_input_jets)
    {
      std::vector<size_t> inverse_jet_axes{0};
      for (size_t axis : inverse_axes)
        inverse_jet_axes.push_back(axis + 1);
      result.d1 = transpose_t(upstream.d1, inverse_jet_axes);
      result.d2 = transpose_t(upstream.d2, inverse_jet_axes);
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Sum a plain tensor over the requested axes.
Tensor reduce_sum_t(const Tensor& input, const std::vector<size_t>& axes)
{
  std::vector<bool> is_reduced_axis(input.shape.size(), false);
  for (size_t axis : axes)
    is_reduced_axis[axis] = true;

  Shape output_shape;
  for (size_t axis = 0; axis < input.shape.size(); ++axis)
    if (!is_reduced_axis[axis])
      output_shape.push_back(input.shape[axis]);

  Tensor output(output_shape);
  for (size_t input_flat_index = 0; input_flat_index < input.size(); ++input_flat_index)
  {
    const Shape input_index = unravel(input_flat_index, input.shape);
    Shape output_index;
    for (size_t axis = 0; axis < input_index.size(); ++axis)
      if (!is_reduced_axis[axis])
        output_index.push_back(input_index[axis]);
    output.x[ravel(output_index, output_shape)] += input.x[input_flat_index];
  }
  return output;
}

/// Reduce a graph node over selected axes and broadcast its VJP back to the input.
NodePtr sum_axes(const NodePtr& input, std::vector<size_t> axes)
{
  Tensor value = reduce_sum_t(input->value, axes);
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;

  if (!input->d1.empty())
  {
    // Account for the leading coordinate axis in the jet tensors.
    std::vector<size_t> jet_axes;
    for (size_t value_axis : axes)
      jet_axes.push_back(value_axis + 1);
    first_coordinate_derivative  = reduce_sum_t(input->d1, jet_axes);
    second_coordinate_derivative = reduce_sum_t(input->d2, jet_axes);
  }

  // The VJP broadcasts each reduced output element back over every input
  // element that contributed to it.
  const Shape input_shape = input->value.shape;
  auto input_vjp          = [axes, input_shape](const Tensor& upstream_gradient) {
    Tensor input_gradient(input_shape);
    std::vector<bool> is_reduced_axis(input_shape.size(), false);
    for (size_t axis : axes)
      is_reduced_axis[axis] = true;

    for (size_t input_flat_index = 0; input_flat_index < input_gradient.size(); ++input_flat_index)
    {
      const Shape input_index = unravel(input_flat_index, input_shape);
      Shape output_index;
      for (size_t axis = 0; axis < input_index.size(); ++axis)
        if (!is_reduced_axis[axis])
          output_index.push_back(input_index[axis]);
      input_gradient.x[input_flat_index] = upstream_gradient.x[ravel(output_index, upstream_gradient.shape)];
    }
    return input_gradient;
  };

  const Shape input_jet_shape = input->d1.shape;
  auto input_jet_vjp          = [axes, input_shape, input_jet_shape](const JetAdjoint& upstream) {
    auto expand_sum = [](const Tensor& gradient, const Shape& target_shape, const std::vector<size_t>& reduced_axes) {
      Tensor result(target_shape);
      std::vector<bool> is_reduced_axis(target_shape.size(), false);
      for (size_t axis : reduced_axes)
        is_reduced_axis[axis] = true;
      for (size_t flat_index = 0; flat_index < result.size(); ++flat_index)
      {
        const Shape target_index = unravel(flat_index, target_shape);
        Shape source_index;
        for (size_t axis = 0; axis < target_index.size(); ++axis)
          if (!is_reduced_axis[axis])
            source_index.push_back(target_index[axis]);
        result.x[flat_index] = gradient.x[ravel(source_index, gradient.shape)];
      }
      return result;
    };
    JetAdjoint result;
    result.value = expand_sum(upstream.value, input_shape, axes);
    if (!input_jet_shape.empty())
    {
      std::vector<size_t> jet_axes;
      for (size_t axis : axes)
        jet_axes.push_back(axis + 1);
      result.d1 = expand_sum(upstream.d1, input_jet_shape, jet_axes);
      result.d2 = expand_sum(upstream.d2, input_jet_shape, jet_axes);
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Reduce every value axis of a graph node to a scalar.
NodePtr sum_all(const NodePtr& input)
{
  std::vector<size_t> axes(input->value.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  return sum_axes(input, axes);
}

/// Extract a half-open interval along the leading value axis.
NodePtr slice0(const NodePtr& input, size_t begin, size_t end)
{
  Shape output_shape             = input->value.shape;
  const size_t trailing_elements = product(Shape(output_shape.begin() + 1, output_shape.end()));
  output_shape[0]                = end - begin;

  Tensor value(output_shape);
  std::copy(input->value.x.begin() + begin * trailing_elements, input->value.x.begin() + end * trailing_elements,
            value.x.begin());

  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  if (!input->d1.empty())
  {
    Shape jet_shape              = input->d1.shape;
    jet_shape[1]                 = end - begin;
    first_coordinate_derivative  = Tensor(jet_shape);
    second_coordinate_derivative = Tensor(jet_shape);

    for (size_t coordinate = 0; coordinate < jet_shape[0]; ++coordinate)
    {
      const size_t input_value_size  = input->value.size();
      const size_t output_value_size = value.size();
      std::copy(input->d1.x.begin() + coordinate * input_value_size + begin * trailing_elements,
                input->d1.x.begin() + coordinate * input_value_size + end * trailing_elements,
                first_coordinate_derivative.x.begin() + coordinate * output_value_size);
      std::copy(input->d2.x.begin() + coordinate * input_value_size + begin * trailing_elements,
                input->d2.x.begin() + coordinate * input_value_size + end * trailing_elements,
                second_coordinate_derivative.x.begin() + coordinate * output_value_size);
    }
  }

  // Scatter the sliced upstream gradient back into its original first-axis
  // interval; elements outside the slice receive zero.
  const Shape input_shape = input->value.shape;
  auto input_vjp          = [begin, input_shape](const Tensor& upstream_gradient) {
    Tensor input_gradient(input_shape);
    const size_t trailing_elements = product(Shape(input_shape.begin() + 1, input_shape.end()));
    std::copy(upstream_gradient.x.begin(), upstream_gradient.x.end(),
              input_gradient.x.begin() + begin * trailing_elements);
    return input_gradient;
  };

  const Shape input_jet_shape = input->d1.shape;
  auto input_jet_vjp          = [begin, input_shape, input_jet_shape](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value                   = Tensor(input_shape);
    const size_t trailing_elements = product(Shape(input_shape.begin() + 1, input_shape.end()));
    std::copy(upstream.value.x.begin(), upstream.value.x.end(), result.value.x.begin() + begin * trailing_elements);
    if (!input_jet_shape.empty())
    {
      result.d1                      = Tensor(input_jet_shape);
      result.d2                      = Tensor(input_jet_shape);
      const size_t input_value_size  = product(input_shape);
      const size_t output_value_size = upstream.value.size();
      for (size_t coordinate = 0; coordinate < input_jet_shape[0]; ++coordinate)
      {
        std::copy(upstream.d1.x.begin() + coordinate * output_value_size,
                  upstream.d1.x.begin() + (coordinate + 1) * output_value_size,
                  result.d1.x.begin() + coordinate * input_value_size + begin * trailing_elements);
        std::copy(upstream.d2.x.begin() + coordinate * output_value_size,
                  upstream.d2.x.begin() + (coordinate + 1) * output_value_size,
                  result.d2.x.begin() + coordinate * input_value_size + begin * trailing_elements);
      }
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Concatenate graph nodes along one value axis and construct interval-extraction VJPs.
NodePtr concat(const std::vector<NodePtr>& inputs, size_t axis)
{
  if (inputs.empty())
    throw std::runtime_error("cannot concatenate an empty node list");

  // Promote scalar nodes to one-element vectors so they can share the normal
  // axis-zero concatenation path.
  if (inputs[0]->value.shape.empty())
  {
    std::vector<NodePtr> promoted_inputs;
    for (const NodePtr& input : inputs)
      promoted_inputs.push_back(reshape(input, {1}));
    return concat(promoted_inputs, 0);
  }

  Shape output_shape      = inputs[0]->value.shape;
  size_t output_axis_size = 0;
  for (const NodePtr& input : inputs)
    output_axis_size += input->value.shape[axis];
  output_shape[axis] = output_axis_size;

  size_t outer_block_count = 1;
  size_t inner_block_size  = 1;
  for (size_t dimension = 0; dimension < axis; ++dimension)
    outer_block_count *= output_shape[dimension];
  for (size_t dimension = axis + 1; dimension < output_shape.size(); ++dimension)
    inner_block_size *= output_shape[dimension];

  // Copy contiguous blocks from every input into the output axis interval.
  Tensor value(output_shape);
  for (size_t outer = 0; outer < outer_block_count; ++outer)
  {
    size_t output_element_offset = 0;
    for (const NodePtr& input : inputs)
    {
      const size_t input_block_size = input->value.shape[axis] * inner_block_size;
      std::copy(input->value.x.begin() + outer * input_block_size,
                input->value.x.begin() + (outer + 1) * input_block_size,
                value.x.begin() + outer * output_axis_size * inner_block_size + output_element_offset);
      output_element_offset += input_block_size;
    }
  }

  // Concatenate coordinate jets with the same block layout as their values.
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  size_t coordinate_count = 0;
  for (const NodePtr& input : inputs)
    if (!input->d1.empty())
    {
      coordinate_count = input->d1.shape[0];
      break;
    }

  if (coordinate_count)
  {
    Shape jet_shape{coordinate_count};
    jet_shape.insert(jet_shape.end(), output_shape.begin(), output_shape.end());
    first_coordinate_derivative  = Tensor(jet_shape);
    second_coordinate_derivative = Tensor(jet_shape);

    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t outer = 0; outer < outer_block_count; ++outer)
      {
        size_t output_element_offset = 0;
        for (const NodePtr& input : inputs)
        {
          const size_t input_block_size = input->value.shape[axis] * inner_block_size;
          const size_t input_value_size = input->value.size();
          if (!input->d1.empty())
          {
            const size_t input_begin = coordinate * input_value_size + outer * input_block_size;
            const size_t output_begin =
                coordinate * value.size() + outer * output_axis_size * inner_block_size + output_element_offset;
            std::copy(input->d1.x.begin() + input_begin, input->d1.x.begin() + input_begin + input_block_size,
                      first_coordinate_derivative.x.begin() + output_begin);
            std::copy(input->d2.x.begin() + input_begin, input->d2.x.begin() + input_begin + input_block_size,
                      second_coordinate_derivative.x.begin() + output_begin);
          }
          output_element_offset += input_block_size;
        }
      }
  }

  // Each parent VJP extracts the axis interval contributed by that parent.
  std::vector<Edge> parent_edges;
  size_t axis_offset = 0;
  for (const NodePtr& input : inputs)
  {
    const size_t input_axis_size  = input->value.shape[axis];
    const size_t input_axis_begin = axis_offset;
    const Shape input_shape       = input->value.shape;
    axis_offset += input_axis_size;

    auto input_vjp = [input_axis_begin, input_axis_size, axis, input_shape](const Tensor& upstream_gradient) {
      Tensor input_gradient(input_shape);
      size_t outer_block_count      = 1;
      size_t inner_block_size       = 1;
      const size_t output_axis_size = upstream_gradient.shape[axis];
      for (size_t dimension = 0; dimension < axis; ++dimension)
        outer_block_count *= upstream_gradient.shape[dimension];
      for (size_t dimension = axis + 1; dimension < upstream_gradient.shape.size(); ++dimension)
        inner_block_size *= upstream_gradient.shape[dimension];

      const size_t input_block_size = input_axis_size * inner_block_size;
      for (size_t outer = 0; outer < outer_block_count; ++outer)
      {
        const size_t output_begin = outer * output_axis_size * inner_block_size + input_axis_begin * inner_block_size;
        std::copy(upstream_gradient.x.begin() + output_begin,
                  upstream_gradient.x.begin() + output_begin + input_block_size,
                  input_gradient.x.begin() + outer * input_block_size);
      }
      return input_gradient;
    };

    const Shape input_jet_shape = input->d1.shape;
    auto input_jet_vjp          = [input_axis_begin, input_axis_size, axis, input_shape,
                                   input_jet_shape](const JetAdjoint& upstream) {
      auto extract_interval = [input_axis_begin, input_axis_size](const Tensor& gradient, size_t gradient_axis,
                                                                  const Shape& target_shape) {
        Tensor result(target_shape);
        size_t outer_block_count      = 1;
        size_t inner_block_size       = 1;
        const size_t output_axis_size = gradient.shape[gradient_axis];
        for (size_t dimension = 0; dimension < gradient_axis; ++dimension)
          outer_block_count *= gradient.shape[dimension];
        for (size_t dimension = gradient_axis + 1; dimension < gradient.shape.size(); ++dimension)
          inner_block_size *= gradient.shape[dimension];
        const size_t input_block_size = input_axis_size * inner_block_size;
        for (size_t outer = 0; outer < outer_block_count; ++outer)
        {
          const size_t output_begin = outer * output_axis_size * inner_block_size + input_axis_begin * inner_block_size;
          std::copy(gradient.x.begin() + output_begin, gradient.x.begin() + output_begin + input_block_size,
                    result.x.begin() + outer * input_block_size);
        }
        return result;
      };
      JetAdjoint result;
      result.value = extract_interval(upstream.value, axis, input_shape);
      if (!input_jet_shape.empty())
      {
        result.d1 = extract_interval(upstream.d1, axis + 1, input_jet_shape);
        result.d2 = extract_interval(upstream.d2, axis + 1, input_jet_shape);
      }
      return result;
    };

    parent_edges.push_back({input, std::move(input_vjp), std::move(input_jet_vjp)});
  }

  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

// Dense and self-attention kernels used by the PsiFormer feature layers.
/// Apply a shared dense layer to the final input axis, with an optional broadcast bias.
NodePtr linear(const NodePtr& input, const NodePtr& weight, const NodePtr& bias = nullptr)
{
  const size_t input_width  = input->value.shape.back();
  const size_t output_width = weight->value.shape[1];
  const size_t row_count    = input->value.size() / input_width;
  Shape output_shape        = input->value.shape;
  output_shape.back()       = output_width;

  // Reuse the dense contraction for the value and every coordinate-jet row.
  auto apply_weight = [input_width, output_width](const Tensor& source, const Tensor& weights, Shape target_shape) {
    Tensor target(std::move(target_shape));
    const size_t rows = source.size() / input_width;
    for (size_t row = 0; row < rows; ++row)
      for (size_t output_column = 0; output_column < output_width; ++output_column)
        for (size_t input_column = 0; input_column < input_width; ++input_column)
          target.x[row * output_width + output_column] +=
              source.x[row * input_width + input_column] * weights.x[input_column * output_width + output_column];
    return target;
  };

  Tensor value = apply_weight(input->value, weight->value, output_shape);
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  if (!input->d1.empty())
  {
    Shape output_jet_shape       = input->d1.shape;
    output_jet_shape.back()      = output_width;
    first_coordinate_derivative  = apply_weight(input->d1, weight->value, output_jet_shape);
    second_coordinate_derivative = apply_weight(input->d2, weight->value, output_jet_shape);
  }

  // Reverse the dense contraction into input values and input jets.
  const Tensor weight_value   = weight->value;
  const Shape input_shape     = input->value.shape;
  const Shape input_jet_shape = input->d1.shape;
  auto apply_weight_transpose = [input_width, output_width](const Tensor& source, const Tensor& weights,
                                                            Shape target_shape) {
    Tensor target(std::move(target_shape));
    const size_t rows = source.size() / output_width;
    for (size_t row = 0; row < rows; ++row)
      for (size_t input_column = 0; input_column < input_width; ++input_column)
        for (size_t output_column = 0; output_column < output_width; ++output_column)
          target.x[row * input_width + input_column] +=
              source.x[row * output_width + output_column] * weights.x[input_column * output_width + output_column];
    return target;
  };
  auto input_vjp = [weight_value, input_shape, apply_weight_transpose](const Tensor& upstream) {
    return apply_weight_transpose(upstream, weight_value, input_shape);
  };
  auto input_jet_vjp = [weight_value, input_shape, input_jet_shape,
                        apply_weight_transpose](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = apply_weight_transpose(upstream.value, weight_value, input_shape);
    if (!input_jet_shape.empty())
    {
      result.d1 = apply_weight_transpose(upstream.d1, weight_value, input_jet_shape);
      result.d2 = apply_weight_transpose(upstream.d2, weight_value, input_jet_shape);
    }
    return result;
  };

  // A shared weight receives contractions from value, d1, and d2 rows; this
  // is where mixed spatial-parameter derivatives reach the parameter leaf.
  const Tensor input_value        = input->value;
  const Tensor input_d1           = input->d1;
  const Tensor input_d2           = input->d2;
  auto accumulate_weight_gradient = [input_width, output_width](Tensor& target, const Tensor& source,
                                                                const Tensor& upstream) {
    const size_t rows = source.size() / input_width;
    for (size_t input_column = 0; input_column < input_width; ++input_column)
      for (size_t output_column = 0; output_column < output_width; ++output_column)
        for (size_t row = 0; row < rows; ++row)
          target.x[input_column * output_width + output_column] +=
              source.x[row * input_width + input_column] * upstream.x[row * output_width + output_column];
  };
  auto weight_vjp = [input_value, input_width, output_width, accumulate_weight_gradient](const Tensor& upstream) {
    Tensor result({input_width, output_width});
    accumulate_weight_gradient(result, input_value, upstream);
    return result;
  };
  auto weight_jet_vjp = [input_value, input_d1, input_d2, input_width, output_width,
                         accumulate_weight_gradient](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor({input_width, output_width});
    accumulate_weight_gradient(result.value, input_value, upstream.value);
    if (!input_d1.empty())
    {
      accumulate_weight_gradient(result.value, input_d1, upstream.d1);
      accumulate_weight_gradient(result.value, input_d2, upstream.d2);
    }
    return result;
  };

  std::vector<Edge> parent_edges{{input, std::move(input_vjp), std::move(input_jet_vjp)},
                                 {weight, std::move(weight_vjp), std::move(weight_jet_vjp)}};
  NodePtr output = node(std::move(value), std::move(first_coordinate_derivative),
                        std::move(second_coordinate_derivative), std::move(parent_edges));
  return bias ? add(output, bias) : output;
}

/// Form per-head query-key dot products for every ordered electron pair.
NodePtr attention_logits(const NodePtr& query, const NodePtr& key)
{
  const size_t electron_count = query->value.shape[0];
  const size_t head_count     = query->value.shape[1];
  const size_t head_width     = query->value.shape[2];

  auto feature_index = [=](size_t electron, size_t head, size_t feature) {
    return (electron * head_count + head) * head_width + feature;
  };

  auto logit_index = [=](size_t head, size_t query_electron, size_t key_electron) {
    return (head * electron_count + query_electron) * electron_count + key_electron;
  };

  // For each head, construct Q K^T over all ordered electron pairs.
  Tensor value({head_count, electron_count, electron_count});
  for (size_t head = 0; head < head_count; ++head)
    for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
      for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
        for (size_t feature = 0; feature < head_width; ++feature)
          value.x[logit_index(head, query_electron, key_electron)] +=
              query->value.x[feature_index(query_electron, head, feature)] *
              key->value.x[feature_index(key_electron, head, feature)];

  // Apply the product rule to the query/key coordinate jets.
  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  if (!query->d1.empty())
  {
    const size_t coordinate_count = query->d1.shape[0];
    first_coordinate_derivative   = Tensor({coordinate_count, head_count, electron_count, electron_count});
    second_coordinate_derivative  = Tensor({coordinate_count, head_count, electron_count, electron_count});

    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
          for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
            for (size_t feature = 0; feature < head_width; ++feature)
            {
              const size_t output_index = coordinate * value.size() + logit_index(head, query_electron, key_electron);
              const size_t query_value_index = feature_index(query_electron, head, feature);
              const size_t key_value_index   = feature_index(key_electron, head, feature);
              const size_t query_jet_index   = coordinate * query->value.size() + query_value_index;
              const size_t key_jet_index     = coordinate * key->value.size() + key_value_index;

              first_coordinate_derivative.x[output_index] +=
                  query->d1.x[query_jet_index] * key->value.x[key_value_index] +
                  query->value.x[query_value_index] * key->d1.x[key_jet_index];
              second_coordinate_derivative.x[output_index] +=
                  query->d2.x[query_jet_index] * key->value.x[key_value_index] +
                  2 * query->d1.x[query_jet_index] * key->d1.x[key_jet_index] +
                  query->value.x[query_value_index] * key->d2.x[key_jet_index];
            }
  }

  const Tensor key_value = key->value;
  auto query_vjp         = [key_value, electron_count, head_count, head_width, feature_index,
                            logit_index](const Tensor& upstream_gradient) {
    Tensor query_gradient({electron_count, head_count, head_width});
    for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
            query_gradient.x[feature_index(query_electron, head, feature)] +=
                upstream_gradient.x[logit_index(head, query_electron, key_electron)] *
                key_value.x[feature_index(key_electron, head, feature)];
    return query_gradient;
  };

  const Tensor query_value = query->value;
  auto key_vjp             = [query_value, electron_count, head_count, head_width, feature_index,
                              logit_index](const Tensor& upstream_gradient) {
    Tensor key_gradient({electron_count, head_count, head_width});
    for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
            key_gradient.x[feature_index(key_electron, head, feature)] +=
                upstream_gradient.x[logit_index(head, query_electron, key_electron)] *
                query_value.x[feature_index(query_electron, head, feature)];
    return key_gradient;
  };

  // Reverse the lifted bilinear Q K^T contraction. Value adjoints collect
  // contributions from all jet orders, while d1/d2 adjoints retain their
  // coordinate-leading shapes.
  const Tensor query_first_coordinate_derivative  = query->d1;
  const Tensor query_second_coordinate_derivative = query->d2;
  const Tensor key_first_coordinate_derivative    = key->d1;
  const Tensor key_second_coordinate_derivative   = key->d2;
  const size_t coordinate_count =
      query_first_coordinate_derivative.empty() ? 0 : query_first_coordinate_derivative.shape[0];
  const size_t output_size  = value.size();
  const size_t feature_size = query->value.size();

  auto query_jet_vjp = [key_value, key_first_coordinate_derivative, key_second_coordinate_derivative, coordinate_count,
                        output_size, feature_size, electron_count, head_count, head_width, feature_index,
                        logit_index](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor({electron_count, head_count, head_width});
    if (coordinate_count)
    {
      result.d1 = Tensor({coordinate_count, electron_count, head_count, head_width});
      result.d2 = Tensor({coordinate_count, electron_count, head_count, head_width});
    }
    for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
          {
            const size_t query_value_index = feature_index(query_electron, head, feature);
            const size_t key_value_index   = feature_index(key_electron, head, feature);
            const size_t logit_value_index = logit_index(head, query_electron, key_electron);
            result.value.x[query_value_index] += upstream.value.x[logit_value_index] * key_value.x[key_value_index];
            for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
            {
              const size_t query_jet_index = coordinate * feature_size + query_value_index;
              const size_t key_jet_index   = coordinate * feature_size + key_value_index;
              const size_t logit_jet_index = coordinate * output_size + logit_value_index;
              result.value.x[query_value_index] +=
                  upstream.d1.x[logit_jet_index] * key_first_coordinate_derivative.x[key_jet_index] +
                  upstream.d2.x[logit_jet_index] * key_second_coordinate_derivative.x[key_jet_index];
              result.d1.x[query_jet_index] += upstream.d1.x[logit_jet_index] * key_value.x[key_value_index] +
                  2 * upstream.d2.x[logit_jet_index] * key_first_coordinate_derivative.x[key_jet_index];
              result.d2.x[query_jet_index] += upstream.d2.x[logit_jet_index] * key_value.x[key_value_index];
            }
          }
    return result;
  };

  auto key_jet_vjp = [query_value, query_first_coordinate_derivative, query_second_coordinate_derivative,
                      coordinate_count, output_size, feature_size, electron_count, head_count, head_width,
                      feature_index, logit_index](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor({electron_count, head_count, head_width});
    if (coordinate_count)
    {
      result.d1 = Tensor({coordinate_count, electron_count, head_count, head_width});
      result.d2 = Tensor({coordinate_count, electron_count, head_count, head_width});
    }
    for (size_t key_electron = 0; key_electron < electron_count; ++key_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t query_electron = 0; query_electron < electron_count; ++query_electron)
          {
            const size_t key_value_index   = feature_index(key_electron, head, feature);
            const size_t query_value_index = feature_index(query_electron, head, feature);
            const size_t logit_value_index = logit_index(head, query_electron, key_electron);
            result.value.x[key_value_index] += upstream.value.x[logit_value_index] * query_value.x[query_value_index];
            for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
            {
              const size_t key_jet_index   = coordinate * feature_size + key_value_index;
              const size_t query_jet_index = coordinate * feature_size + query_value_index;
              const size_t logit_jet_index = coordinate * output_size + logit_value_index;
              result.value.x[key_value_index] +=
                  upstream.d1.x[logit_jet_index] * query_first_coordinate_derivative.x[query_jet_index] +
                  upstream.d2.x[logit_jet_index] * query_second_coordinate_derivative.x[query_jet_index];
              result.d1.x[key_jet_index] += upstream.d1.x[logit_jet_index] * query_value.x[query_value_index] +
                  2 * upstream.d2.x[logit_jet_index] * query_first_coordinate_derivative.x[query_jet_index];
              result.d2.x[key_jet_index] += upstream.d2.x[logit_jet_index] * query_value.x[query_value_index];
            }
          }
    return result;
  };

  std::vector<Edge> parent_edges{{query, std::move(query_vjp), std::move(query_jet_vjp)},
                                 {key, std::move(key_vjp), std::move(key_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Aggregate value features over source electrons using per-head attention weights.
NodePtr attention_context(const NodePtr& attention_weight, const NodePtr& feature_value)
{
  const size_t head_count     = attention_weight->value.shape[0];
  const size_t electron_count = attention_weight->value.shape[1];
  const size_t head_width     = feature_value->value.shape[2];

  auto weight_index = [=](size_t head, size_t output_electron, size_t source_electron) {
    return (head * electron_count + output_electron) * electron_count + source_electron;
  };

  auto feature_index = [=](size_t electron, size_t head, size_t feature) {
    return (electron * head_count + head) * head_width + feature;
  };

  // Weighted aggregation over source electrons for every output electron/head.
  Tensor value({electron_count, head_count, head_width});
  for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
    for (size_t head = 0; head < head_count; ++head)
      for (size_t feature = 0; feature < head_width; ++feature)
        for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
          value.x[feature_index(output_electron, head, feature)] +=
              attention_weight->value.x[weight_index(head, output_electron, source_electron)] *
              feature_value->value.x[feature_index(source_electron, head, feature)];

  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  if (!attention_weight->d1.empty())
  {
    const size_t coordinate_count = attention_weight->d1.shape[0];
    first_coordinate_derivative   = Tensor({coordinate_count, electron_count, head_count, head_width});
    second_coordinate_derivative  = Tensor({coordinate_count, electron_count, head_count, head_width});

    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
        for (size_t head = 0; head < head_count; ++head)
          for (size_t feature = 0; feature < head_width; ++feature)
            for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
            {
              const size_t output_index = coordinate * value.size() + feature_index(output_electron, head, feature);
              const size_t weight_value_index  = weight_index(head, output_electron, source_electron);
              const size_t feature_value_index = feature_index(source_electron, head, feature);
              const size_t weight_jet_index    = coordinate * attention_weight->value.size() + weight_value_index;
              const size_t feature_jet_index   = coordinate * feature_value->value.size() + feature_value_index;

              first_coordinate_derivative.x[output_index] +=
                  attention_weight->d1.x[weight_jet_index] * feature_value->value.x[feature_value_index] +
                  attention_weight->value.x[weight_value_index] * feature_value->d1.x[feature_jet_index];
              second_coordinate_derivative.x[output_index] +=
                  attention_weight->d2.x[weight_jet_index] * feature_value->value.x[feature_value_index] +
                  2 * attention_weight->d1.x[weight_jet_index] * feature_value->d1.x[feature_jet_index] +
                  attention_weight->value.x[weight_value_index] * feature_value->d2.x[feature_jet_index];
            }
  }

  const Tensor feature_value_tensor = feature_value->value;
  auto attention_weight_vjp         = [feature_value_tensor, electron_count, head_count, head_width, weight_index,
                                       feature_index](const Tensor& upstream_gradient) {
    Tensor weight_gradient({head_count, electron_count, electron_count});
    for (size_t head = 0; head < head_count; ++head)
      for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
        for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
          for (size_t feature = 0; feature < head_width; ++feature)
            weight_gradient.x[weight_index(head, output_electron, source_electron)] +=
                upstream_gradient.x[feature_index(output_electron, head, feature)] *
                feature_value_tensor.x[feature_index(source_electron, head, feature)];
    return weight_gradient;
  };

  const Tensor attention_weight_value = attention_weight->value;
  auto feature_value_vjp              = [attention_weight_value, electron_count, head_count, head_width, weight_index,
                                         feature_index](const Tensor& upstream_gradient) {
    Tensor feature_gradient({electron_count, head_count, head_width});
    for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
            feature_gradient.x[feature_index(source_electron, head, feature)] +=
                attention_weight_value.x[weight_index(head, output_electron, source_electron)] *
                upstream_gradient.x[feature_index(output_electron, head, feature)];
    return feature_gradient;
  };

  // Apply the same lifted bilinear reverse rule to attention-weighted feature
  // aggregation over source electrons.
  const Tensor attention_weight_first_coordinate_derivative  = attention_weight->d1;
  const Tensor attention_weight_second_coordinate_derivative = attention_weight->d2;
  const Tensor feature_first_coordinate_derivative           = feature_value->d1;
  const Tensor feature_second_coordinate_derivative          = feature_value->d2;
  const size_t coordinate_count =
      attention_weight_first_coordinate_derivative.empty() ? 0 : attention_weight_first_coordinate_derivative.shape[0];
  const size_t output_size  = value.size();
  const size_t weight_size  = attention_weight->value.size();
  const size_t feature_size = feature_value->value.size();

  auto attention_weight_jet_vjp = [feature_value_tensor, feature_first_coordinate_derivative,
                                   feature_second_coordinate_derivative, coordinate_count, output_size, weight_size,
                                   feature_size, electron_count, head_count, head_width, weight_index,
                                   feature_index](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor({head_count, electron_count, electron_count});
    if (coordinate_count)
    {
      result.d1 = Tensor({coordinate_count, head_count, electron_count, electron_count});
      result.d2 = Tensor({coordinate_count, head_count, electron_count, electron_count});
    }
    for (size_t head = 0; head < head_count; ++head)
      for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
        for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
          for (size_t feature = 0; feature < head_width; ++feature)
          {
            const size_t weight_value_index   = weight_index(head, output_electron, source_electron);
            const size_t source_feature_index = feature_index(source_electron, head, feature);
            const size_t output_feature_index = feature_index(output_electron, head, feature);
            result.value.x[weight_value_index] +=
                upstream.value.x[output_feature_index] * feature_value_tensor.x[source_feature_index];
            for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
            {
              const size_t weight_jet_index  = coordinate * weight_size + weight_value_index;
              const size_t feature_jet_index = coordinate * feature_size + source_feature_index;
              const size_t output_jet_index  = coordinate * output_size + output_feature_index;
              result.value.x[weight_value_index] +=
                  upstream.d1.x[output_jet_index] * feature_first_coordinate_derivative.x[feature_jet_index] +
                  upstream.d2.x[output_jet_index] * feature_second_coordinate_derivative.x[feature_jet_index];
              result.d1.x[weight_jet_index] +=
                  upstream.d1.x[output_jet_index] * feature_value_tensor.x[source_feature_index] +
                  2 * upstream.d2.x[output_jet_index] * feature_first_coordinate_derivative.x[feature_jet_index];
              result.d2.x[weight_jet_index] +=
                  upstream.d2.x[output_jet_index] * feature_value_tensor.x[source_feature_index];
            }
          }
    return result;
  };

  auto feature_value_jet_vjp = [attention_weight_value, attention_weight_first_coordinate_derivative,
                                attention_weight_second_coordinate_derivative, coordinate_count, output_size,
                                weight_size, feature_size, electron_count, head_count, head_width, weight_index,
                                feature_index](const JetAdjoint& upstream) {
    JetAdjoint result;
    result.value = Tensor({electron_count, head_count, head_width});
    if (coordinate_count)
    {
      result.d1 = Tensor({coordinate_count, electron_count, head_count, head_width});
      result.d2 = Tensor({coordinate_count, electron_count, head_count, head_width});
    }
    for (size_t source_electron = 0; source_electron < electron_count; ++source_electron)
      for (size_t head = 0; head < head_count; ++head)
        for (size_t feature = 0; feature < head_width; ++feature)
          for (size_t output_electron = 0; output_electron < electron_count; ++output_electron)
          {
            const size_t source_feature_index = feature_index(source_electron, head, feature);
            const size_t weight_value_index   = weight_index(head, output_electron, source_electron);
            const size_t output_feature_index = feature_index(output_electron, head, feature);
            result.value.x[source_feature_index] +=
                upstream.value.x[output_feature_index] * attention_weight_value.x[weight_value_index];
            for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
            {
              const size_t feature_jet_index = coordinate * feature_size + source_feature_index;
              const size_t weight_jet_index  = coordinate * weight_size + weight_value_index;
              const size_t output_jet_index  = coordinate * output_size + output_feature_index;
              result.value.x[source_feature_index] +=
                  upstream.d1.x[output_jet_index] * attention_weight_first_coordinate_derivative.x[weight_jet_index] +
                  upstream.d2.x[output_jet_index] * attention_weight_second_coordinate_derivative.x[weight_jet_index];
              result.d1.x[feature_jet_index] +=
                  upstream.d1.x[output_jet_index] * attention_weight_value.x[weight_value_index] +
                  2 * upstream.d2.x[output_jet_index] *
                      attention_weight_first_coordinate_derivative.x[weight_jet_index];
              result.d2.x[feature_jet_index] +=
                  upstream.d2.x[output_jet_index] * attention_weight_value.x[weight_value_index];
            }
          }
    return result;
  };

  std::vector<Edge> parent_edges{{attention_weight, std::move(attention_weight_vjp),
                                  std::move(attention_weight_jet_vjp)},
                                 {feature_value, std::move(feature_value_vjp), std::move(feature_value_jet_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

/// Normalize each final-axis row with a numerically stable softmax.
NodePtr softmax(const NodePtr& input)
{
  const size_t row_width = input->value.shape.back();
  const size_t row_count = input->value.size() / row_width;

  // Treat the row maximum as a locally constant shift. Softmax is invariant to
  // a common row shift, so its omitted derivative cancels exactly while the
  // subtraction retains numerical stability.
  Tensor negative_row_maximum(input->value.shape);
  for (size_t row = 0; row < row_count; ++row)
  {
    const auto row_begin     = input->value.x.begin() + row * row_width;
    const double row_maximum = *std::max_element(row_begin, row_begin + row_width);
    std::fill(negative_row_maximum.x.begin() + row * row_width, negative_row_maximum.x.begin() + (row + 1) * row_width,
              -row_maximum);
  }

  NodePtr exponentials      = exp_node(add(input, constant(std::move(negative_row_maximum))));
  NodePtr normalization     = sum_axes(exponentials, {input->value.shape.size() - 1});
  Shape normalization_shape = normalization->value.shape;
  normalization_shape.push_back(1);
  return divide(exponentials, reshape(normalization, std::move(normalization_shape)));
}

/// Compute a square matrix determinant and inverse together by pivoted elimination.
std::pair<double, std::vector<double>> det_inv(const double* matrix, size_t matrix_size)
{
  // Gauss-Jordan elimination with partial pivoting produces the determinant
  // and inverse together. Matrices are small electron-count orbital matrices.
  std::vector<double> work(matrix, matrix + matrix_size * matrix_size);
  std::vector<double> inverse(matrix_size * matrix_size);
  for (size_t row = 0; row < matrix_size; ++row)
    inverse[row * matrix_size + row] = 1;

  double determinant   = 1;
  int permutation_sign = 1;
  for (size_t column = 0; column < matrix_size; ++column)
  {
    // Select the largest available pivot in this column for stability.
    size_t pivot_row = column;
    for (size_t row = column + 1; row < matrix_size; ++row)
      if (std::abs(work[row * matrix_size + column]) > std::abs(work[pivot_row * matrix_size + column]))
        pivot_row = row;

    if (std::abs(work[pivot_row * matrix_size + column]) < 1e-14)
      throw std::runtime_error("singular determinant");

    if (pivot_row != column)
    {
      for (size_t entry = 0; entry < matrix_size; ++entry)
      {
        std::swap(work[column * matrix_size + entry], work[pivot_row * matrix_size + entry]);
        std::swap(inverse[column * matrix_size + entry], inverse[pivot_row * matrix_size + entry]);
      }
      permutation_sign = -permutation_sign;
    }

    // Normalize the pivot row, tracking the unnormalized pivot in det(A).
    const double pivot = work[column * matrix_size + column];
    determinant *= pivot;
    for (size_t entry = 0; entry < matrix_size; ++entry)
    {
      work[column * matrix_size + entry] /= pivot;
      inverse[column * matrix_size + entry] /= pivot;
    }

    // Eliminate this column from all non-pivot rows in both work and inverse.
    for (size_t row = 0; row < matrix_size; ++row)
      if (row != column)
      {
        const double elimination_factor = work[row * matrix_size + column];
        for (size_t entry = 0; entry < matrix_size; ++entry)
        {
          work[row * matrix_size + entry] -= elimination_factor * work[column * matrix_size + entry];
          inverse[row * matrix_size + entry] -= elimination_factor * inverse[column * matrix_size + entry];
        }
      }
  }

  return {determinant * permutation_sign, std::move(inverse)};
}

/// Return tr(left*right) for two row-major square matrices.
double trace_product(const double* left, const double* right, size_t matrix_size)
{
  double trace = 0;
  for (size_t row = 0; row < matrix_size; ++row)
    for (size_t column = 0; column < matrix_size; ++column)
      trace += left[row * matrix_size + column] * right[column * matrix_size + row];
  return trace;
}

/** Evaluate determinant values and spatial jets with the optimized matrix kernel.
 * This path is used when parameter derivatives are not requested. */
NodePtr determinants(const NodePtr& matrices)
{
  const size_t matrix_count             = matrices->value.shape[0];
  const size_t matrix_size              = matrices->value.shape[1];
  const size_t element_count_per_matrix = matrix_size * matrix_size;

  // Cache each inverse because it is reused by both coordinate-derivative
  // orders and by the ordinary value VJP.
  Tensor value({matrix_count});
  std::vector<std::vector<double>> inverses(matrix_count);
  for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
  {
    auto determinant_and_inverse =
        det_inv(matrices->value.x.data() + matrix_index * element_count_per_matrix, matrix_size);
    value.x[matrix_index]  = determinant_and_inverse.first;
    inverses[matrix_index] = std::move(determinant_and_inverse.second);
  }

  Tensor first_coordinate_derivative;
  Tensor second_coordinate_derivative;
  if (!matrices->d1.empty())
  {
    const size_t coordinate_count = matrices->d1.shape[0];
    first_coordinate_derivative   = Tensor({coordinate_count, matrix_count});
    second_coordinate_derivative  = Tensor({coordinate_count, matrix_count});
    std::vector<double> inverse_times_first_derivative(element_count_per_matrix);

    for (size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
      for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
      {
        const double* matrix_d1 =
            matrices->d1.x.data() + coordinate * matrices->value.size() + matrix_index * element_count_per_matrix;
        const double* matrix_d2 =
            matrices->d2.x.data() + coordinate * matrices->value.size() + matrix_index * element_count_per_matrix;

        // d(det A) = det(A) tr(A^-1 dA). The diagonal second derivative adds
        // tr(A^-1 d2A) and subtracts tr((A^-1 dA)^2).
        const double first_trace  = trace_product(inverses[matrix_index].data(), matrix_d1, matrix_size);
        const double second_trace = trace_product(inverses[matrix_index].data(), matrix_d2, matrix_size);
        for (size_t row = 0; row < matrix_size; ++row)
          for (size_t column = 0; column < matrix_size; ++column)
          {
            double& product_element = inverse_times_first_derivative[row * matrix_size + column];
            product_element         = 0;
            for (size_t inner = 0; inner < matrix_size; ++inner)
              product_element +=
                  inverses[matrix_index][row * matrix_size + inner] * matrix_d1[inner * matrix_size + column];
          }
        const double squared_trace =
            trace_product(inverse_times_first_derivative.data(), inverse_times_first_derivative.data(), matrix_size);

        const size_t output_index                   = coordinate * matrix_count + matrix_index;
        first_coordinate_derivative.x[output_index] = value.x[matrix_index] * first_trace;
        second_coordinate_derivative.x[output_index] =
            value.x[matrix_index] * (first_trace * first_trace + second_trace - squared_trace);
      }
  }

  // d(det A)/dA = det(A) A^-T for each matrix in the determinant batch.
  auto matrices_vjp = [value, inverses, matrix_count, matrix_size,
                       element_count_per_matrix](const Tensor& upstream_gradient) {
    Tensor matrix_gradient({matrix_count, matrix_size, matrix_size});
    for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
      for (size_t row = 0; row < matrix_size; ++row)
        for (size_t column = 0; column < matrix_size; ++column)
          matrix_gradient.x[matrix_index * element_count_per_matrix + row * matrix_size + column] =
              upstream_gradient.x[matrix_index] * value.x[matrix_index] *
              inverses[matrix_index][column * matrix_size + row];
    return matrix_gradient;
  };

  std::vector<Edge> parent_edges{{matrices, std::move(matrices_vjp)}};
  return node(std::move(value), std::move(first_coordinate_derivative), std::move(second_coordinate_derivative),
              std::move(parent_edges));
}

// Differentiable determinant composition used by the mixed-derivative reverse pass.
NodePtr differentiable_determinants(const NodePtr& matrices)
{
  const size_t matrix_count    = matrices->value.shape[0];
  const size_t matrix_size     = matrices->value.shape[1];
  const size_t matrix_elements = matrix_size * matrix_size;
  NodePtr flattened            = reshape(matrices, {matrix_count * matrix_elements});
  std::vector<NodePtr> determinant_values;
  determinant_values.reserve(matrix_count);

  for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
  {
    std::vector<NodePtr> work;
    work.reserve(matrix_elements);
    for (size_t element = 0; element < matrix_elements; ++element)
    {
      const size_t flat_index = matrix_index * matrix_elements + element;
      work.push_back(reshape(slice0(flattened, flat_index, flat_index + 1), {}));
    }

    // Pivot choices are discrete and locally constant. The subsequent scalar
    // elimination is composed entirely from jet-aware differentiable primitives.
    NodePtr determinant  = scalar(1);
    int permutation_sign = 1;
    for (size_t column = 0; column < matrix_size; ++column)
    {
      size_t pivot_row = column;
      for (size_t row = column + 1; row < matrix_size; ++row)
        if (std::abs(work[row * matrix_size + column]->value.x[0]) >
            std::abs(work[pivot_row * matrix_size + column]->value.x[0]))
          pivot_row = row;

      if (std::abs(work[pivot_row * matrix_size + column]->value.x[0]) < 1e-14)
        throw std::runtime_error("singular determinant");

      if (pivot_row != column)
      {
        for (size_t entry = 0; entry < matrix_size; ++entry)
          std::swap(work[column * matrix_size + entry], work[pivot_row * matrix_size + entry]);
        permutation_sign = -permutation_sign;
      }

      NodePtr pivot = work[column * matrix_size + column];
      determinant   = mul(determinant, pivot);
      for (size_t row = column + 1; row < matrix_size; ++row)
      {
        NodePtr elimination_factor = divide(work[row * matrix_size + column], pivot);
        for (size_t entry = column + 1; entry < matrix_size; ++entry)
          work[row * matrix_size + entry] =
              add(work[row * matrix_size + entry], neg(mul(elimination_factor, work[column * matrix_size + entry])));
      }
    }

    determinant_values.push_back(permutation_sign < 0 ? neg(determinant) : determinant);
  }

  return concat(determinant_values, 0);
}

/** Reverse-accumulate adjoints from a scalar output through the recorded VJP
 * graph. */
std::unordered_map<const Node*, Tensor> backward(const NodePtr& root)
{
  // Build a topological ordering once so every child contribution is available
  // before its parents are visited in reverse order.
  std::vector<NodePtr> topological_order;
  std::unordered_map<const Node*, bool> visited;
  std::function<void(const NodePtr&)> visit = [&](const NodePtr& current) {
    if (visited[current.get()])
      return;
    visited[current.get()] = true;
    for (const Edge& edge : current->parents)
      visit(edge.parent);
    topological_order.push_back(current);
  };
  visit(root);

  // Seed the scalar output with an all-ones adjoint and accumulate every VJP
  // contribution into the corresponding parent node.
  std::unordered_map<const Node*, Tensor> adjoints;
  adjoints[root.get()] = Tensor(root->value.shape, 1);
  for (auto node_it = topological_order.rbegin(); node_it != topological_order.rend(); ++node_it)
  {
    const auto current_adjoint = adjoints.find(node_it->get());
    if (current_adjoint == adjoints.end())
      continue;

    for (const Edge& edge : (*node_it)->parents)
    {
      Tensor contribution = edge.vjp(current_adjoint->second);
      auto parent_adjoint = adjoints.find(edge.parent.get());
      if (parent_adjoint == adjoints.end())
        adjoints[edge.parent.get()] = std::move(contribution);
      else
        for (size_t element = 0; element < contribution.size(); ++element)
          parent_adjoint->second.x[element] += contribution.x[element];
    }
  }
  return adjoints;
}

/** Reverse-accumulate through the lifted coordinate-jet program. This computes
 * mixed parameter-coordinate derivatives without finite differences. */
std::unordered_map<const Node*, JetAdjoint> backward_coordinate_jets(const NodePtr& root, JetAdjoint root_adjoint)
{
  // Use the same graph topology as the value reverse pass, but carry an
  // adjoint for each of the three coordinate-jet components.
  std::vector<NodePtr> topological_order;
  std::unordered_map<const Node*, bool> visited;
  std::function<void(const NodePtr&)> visit = [&](const NodePtr& current) {
    if (visited[current.get()])
      return;
    visited[current.get()] = true;
    for (const Edge& edge : current->parents)
      visit(edge.parent);
    topological_order.push_back(current);
  };
  visit(root);

  auto accumulate_tensor = [](Tensor& target, const Tensor& contribution) {
    if (contribution.empty())
      return;
    if (target.empty())
      target = contribution;
    else
      for (size_t element = 0; element < contribution.size(); ++element)
        target.x[element] += contribution.x[element];
  };

  std::unordered_map<const Node*, JetAdjoint> adjoints;
  adjoints[root.get()] = std::move(root_adjoint);
  for (auto node_it = topological_order.rbegin(); node_it != topological_order.rend(); ++node_it)
  {
    const auto current_adjoint = adjoints.find(node_it->get());
    if (current_adjoint == adjoints.end())
      continue;

    for (const Edge& edge : (*node_it)->parents)
    {
      if (!edge.jet_vjp)
        throw std::runtime_error("missing coordinate-jet VJP");
      JetAdjoint contribution    = edge.jet_vjp(current_adjoint->second);
      JetAdjoint& parent_adjoint = adjoints[edge.parent.get()];
      accumulate_tensor(parent_adjoint.value, contribution.value);
      accumulate_tensor(parent_adjoint.d1, contribution.d1);
      accumulate_tensor(parent_adjoint.d2, contribution.d2);
    }
  }
  return adjoints;
}

// HDF5 import helpers for the flattened DeepQMC parameter export and
// configuration data.
/// Read a floating-point HDF5 dataset and optionally return its shape.
std::vector<double> read_double(hid_t file, const std::string& path, Shape* shape = nullptr)
{
  const hid_t dataset   = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  const hid_t dataspace = H5Dget_space(dataset);
  const int rank        = H5Sget_simple_extent_ndims(dataspace);

  std::vector<hsize_t> dimensions(rank);
  H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr);
  const Shape dataset_shape(dimensions.begin(), dimensions.end());

  std::vector<double> values(product(dataset_shape));
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);

  if (shape)
    *shape = dataset_shape;
  return values;
}

/// Read a signed 64-bit integer HDF5 dataset and optionally return its shape.
std::vector<int64_t> read_i64(hid_t file, const std::string& path, Shape* shape = nullptr)
{
  const hid_t dataset   = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  const hid_t dataspace = H5Dget_space(dataset);
  const int rank        = H5Sget_simple_extent_ndims(dataspace);

  std::vector<hsize_t> dimensions(rank);
  H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr);
  const Shape dataset_shape(dimensions.begin(), dimensions.end());

  std::vector<int64_t> values(product(dataset_shape));
  H5Dread(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);

  if (shape)
    *shape = dataset_shape;
  return values;
}

/// Read and reclaim a variable-length string dataset from HDF5.
std::vector<std::string> read_strings(hid_t file, const std::string& path)
{
  const hid_t dataset   = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  const hid_t dataspace = H5Dget_space(dataset);
  hsize_t string_count;
  H5Sget_simple_extent_dims(dataspace, &string_count, nullptr);

  const hid_t string_type = H5Dget_type(dataset);
  std::vector<char*> raw_strings(string_count);
  H5Dread(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_strings.data());

  std::vector<std::string> strings;
  strings.reserve(string_count);
  for (const char* raw_string : raw_strings)
    strings.emplace_back(raw_string);

  // Variable-length HDF5 strings are library-owned allocations and must be
  // reclaimed with the matching datatype and dataspace.
  H5Dvlen_reclaim(string_type, dataspace, H5P_DEFAULT, raw_strings.data());
  H5Tclose(string_type);
  H5Sclose(dataspace);
  H5Dclose(dataset);
  return strings;
}

/// Read one signed 64-bit integer attribute from an HDF5 object.
int64_t read_attr_i64(hid_t file, const std::string& name)
{
  const hid_t attribute = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  int64_t value;
  H5Aread(attribute, H5T_NATIVE_LLONG, &value);
  H5Aclose(attribute);
  return value;
}

/// Test IEEE-754 finiteness without relying on fast-math-sensitive classification builtins.
bool is_finite_parameter_value(double value)
{
  static_assert(sizeof(double) == sizeof(uint64_t));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/// Describe one named parameter tensor within the flattened export vector.
struct Layout
{
  std::string module;
  std::string name;
  Shape shape;
  size_t begin;
  size_t end;
};

/** Owns the flat exported parameters and exposes named graph leaves for model
 * assembly. */
struct Parameters
{
  std::vector<double> values;
  std::vector<Layout> layouts;
  std::map<std::pair<std::string, std::string>, NodePtr> nodes;
  size_t parameter_version = 0;

  /// Load flattened parameters and materialize their named graph leaves.
  explicit Parameters(const std::string& path)
  {
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read the single flat parameter array and its parallel layout metadata.
    values                                 = read_double(file, "/values");
    const std::vector<std::string> modules = read_strings(file, "/layout/modules");
    const std::vector<std::string> names   = read_strings(file, "/layout/names");
    const std::vector<int64_t> ranks       = read_i64(file, "/layout/ranks");
    Shape shape_table_shape;
    const std::vector<int64_t> shape_table = read_i64(file, "/layout/shapes", &shape_table_shape);
    const std::vector<int64_t> offsets     = read_i64(file, "/layout/offsets");

    // Materialize each flattened interval as a named parameter leaf. These
    // leaves become parents in the reverse-mode graph assembled at evaluation.
    for (size_t parameter_index = 0; parameter_index < modules.size(); ++parameter_index)
    {
      Shape parameter_shape;
      for (int axis = 0; axis < ranks[parameter_index]; ++axis)
        parameter_shape.push_back(shape_table[parameter_index * shape_table_shape[1] + axis]);

      Layout layout{modules[parameter_index], names[parameter_index], parameter_shape, size_t(offsets[parameter_index]),
                    size_t(offsets[parameter_index + 1])};
      std::vector<double> parameter_values(values.begin() + layout.begin, values.begin() + layout.end);
      NodePtr parameter_node        = node(Tensor(parameter_shape, std::move(parameter_values)));
      parameter_node->parameter_key = modules[parameter_index] + "/" + names[parameter_index];

      layouts.push_back(layout);
      nodes[{modules[parameter_index], names[parameter_index]}] = parameter_node;
    }
    H5Fclose(file);
  }

  /// Return the number of scalar parameters in canonical DeepQMC export order.
  size_t size() const { return values.size(); }

  /// Expose the current canonical flat values without allowing unsynchronized mutation.
  const std::vector<double>& flat_values() const { return values; }

  /// Return the version incremented after each successful parameter mutation.
  size_t version() const { return parameter_version; }

  /// Return a stable fingerprint of the immutable exported tensor layout.
  std::string layout_fingerprint() const
  {
    // FNV-1a is used only as a deterministic compatibility fingerprint, not
    // for adversarial input. Length prefixes keep adjacent strings and shapes
    // unambiguous.
    uint64_t hash = 14695981039346656037ULL;
    auto mix_byte = [&hash](uint8_t byte) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    };
    auto mix_integer = [&mix_byte](uint64_t value) {
      for (int byte = 0; byte < 8; ++byte)
        mix_byte(static_cast<uint8_t>(value >> (8 * byte)));
    };
    auto mix_string = [&mix_byte, &mix_integer](const std::string& value) {
      mix_integer(value.size());
      for (unsigned char character : value)
        mix_byte(character);
    };

    mix_integer(layouts.size());
    mix_integer(values.size());
    for (const Layout& layout : layouts)
    {
      mix_string(layout.module);
      mix_string(layout.name);
      mix_integer(layout.shape.size());
      for (size_t extent : layout.shape)
        mix_integer(extent);
      mix_integer(layout.begin);
      mix_integer(layout.end);
    }

    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setfill('0') << std::setw(16) << hash;
    return fingerprint.str();
  }

  /// Resolve the parameter tensor containing one canonical flat index.
  const Layout& layout_for_flat_index(size_t flat_index) const
  {
    if (flat_index >= values.size())
      throw std::out_of_range("PsiFormer flat parameter index out of range");

    const auto layout = std::find_if(layouts.begin(), layouts.end(), [flat_index](const Layout& candidate) {
      return candidate.begin <= flat_index && flat_index < candidate.end;
    });
    if (layout == layouts.end())
      throw std::runtime_error("PsiFormer parameter layout does not cover flat index " +
                               std::to_string(flat_index));
    return *layout;
  }

  /// Atomically replace every flat value and synchronize all graph parameter leaves.
  void set_flat_values(const std::vector<double>& new_values)
  {
    if (new_values.size() != values.size())
      throw std::invalid_argument("PsiFormer flat parameter vector has the wrong size");
    if (std::any_of(new_values.begin(), new_values.end(),
                    [](double value) { return !is_finite_parameter_value(value); }))
      throw std::invalid_argument("PsiFormer flat parameter vector contains a non-finite value");

    values = new_values;
    for (const Layout& layout : layouts)
    {
      NodePtr parameter_node = nodes.at({layout.module, layout.name});
      std::copy(values.begin() + layout.begin, values.begin() + layout.end, parameter_node->value.x.begin());
    }
    ++parameter_version;
  }

  /// Atomically replace a selected set of flat values and synchronize their graph leaves.
  void set_flat_values(const std::vector<size_t>& flat_indices, const std::vector<double>& new_values)
  {
    if (flat_indices.size() != new_values.size())
      throw std::invalid_argument("PsiFormer selected parameter indices and values differ in size");

    std::vector<size_t> sorted_indices = flat_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    if (std::adjacent_find(sorted_indices.begin(), sorted_indices.end()) != sorted_indices.end())
      throw std::invalid_argument("PsiFormer selected parameter indices contain a duplicate");

    for (size_t parameter = 0; parameter < flat_indices.size(); ++parameter)
    {
      layout_for_flat_index(flat_indices[parameter]);
      if (!is_finite_parameter_value(new_values[parameter]))
        throw std::invalid_argument("PsiFormer selected parameter value is not finite");
    }

    for (size_t parameter = 0; parameter < flat_indices.size(); ++parameter)
    {
      const size_t flat_index = flat_indices[parameter];
      const Layout& layout    = layout_for_flat_index(flat_index);
      values[flat_index]      = new_values[parameter];
      nodes.at({layout.module, layout.name})->value.x[flat_index - layout.begin] = new_values[parameter];
    }
    if (!flat_indices.empty())
      ++parameter_version;
  }

  /// Replace one flat value through the same synchronized mutation path.
  void set_flat_value(size_t flat_index, double new_value)
  {
    set_flat_values(std::vector<size_t>{flat_index}, std::vector<double>{new_value});
  }

  /// Export current values and immutable tensor layout in the DeepQMC HDF5 format.
  void write(const std::string& path) const
  {
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
      throw std::runtime_error("Unable to create PsiFormer parameter file " + path);

    auto write_numeric = [file](const std::string& dataset_path, hid_t type, const std::vector<hsize_t>& shape,
                                const void* data) {
      const hid_t dataspace = H5Screate_simple(shape.size(), shape.data(), nullptr);
      const hid_t dataset = H5Dcreate2(file, dataset_path.c_str(), type, dataspace, H5P_DEFAULT, H5P_DEFAULT,
                                       H5P_DEFAULT);
      if (dataset < 0 || H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
        throw std::runtime_error("Unable to write PsiFormer dataset " + dataset_path);
      H5Dclose(dataset);
      H5Sclose(dataspace);
    };

    write_numeric("/values", H5T_NATIVE_DOUBLE, {values.size()}, values.data());
    H5Gclose(H5Gcreate2(file, "/layout", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    std::vector<std::string> modules, names;
    std::vector<int64_t> ranks;
    std::vector<int64_t> offsets{0};
    size_t maximum_rank = 1;
    for (const Layout& layout : layouts)
      maximum_rank = std::max(maximum_rank, layout.shape.size());
    std::vector<int64_t> shapes(layouts.size() * maximum_rank, 1);
    for (size_t parameter = 0; parameter < layouts.size(); ++parameter)
    {
      const Layout& layout = layouts[parameter];
      modules.push_back(layout.module);
      names.push_back(layout.name);
      ranks.push_back(layout.shape.size());
      offsets.push_back(layout.end);
      for (size_t axis = 0; axis < layout.shape.size(); ++axis)
        shapes[parameter * maximum_rank + axis] = layout.shape[axis];
    }

    auto write_strings = [file](const std::string& dataset_path, const std::vector<std::string>& strings) {
      const hsize_t count = strings.size();
      const hid_t space   = H5Screate_simple(1, &count, nullptr);
      const hid_t type    = H5Tcopy(H5T_C_S1);
      H5Tset_size(type, H5T_VARIABLE);
      const hid_t dataset =
          H5Dcreate2(file, dataset_path.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      std::vector<const char*> pointers;
      pointers.reserve(strings.size());
      for (const std::string& string : strings)
        pointers.push_back(string.c_str());
      if (dataset < 0 || H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()) < 0)
        throw std::runtime_error("Unable to write PsiFormer dataset " + dataset_path);
      H5Dclose(dataset);
      H5Tclose(type);
      H5Sclose(space);
    };

    write_strings("/layout/modules", modules);
    write_strings("/layout/names", names);
    write_numeric("/layout/ranks", H5T_NATIVE_LLONG, {ranks.size()}, ranks.data());
    write_numeric("/layout/shapes", H5T_NATIVE_LLONG, {layouts.size(), maximum_rank}, shapes.data());
    write_numeric("/layout/offsets", H5T_NATIVE_LLONG, {offsets.size()}, offsets.data());
    H5Fclose(file);
  }

  /// Resolve exactly one exported parameter by Haiku module suffix and leaf name.
  NodePtr find(const std::string& suffix, const std::string& name)
  {
    // DeepQMC/Haiku module paths contain generated prefixes. A unique suffix
    // plus the parameter name identifies the corresponding weight or bias.
    NodePtr match;
    int match_count = 0;
    for (auto& [key, parameter_node] : nodes)
      if (key.second == name && key.first.size() >= suffix.size() &&
          key.first.compare(key.first.size() - suffix.size(), suffix.size(), suffix) == 0)
      {
        match = parameter_node;
        ++match_count;
      }

    if (match_count != 1)
      throw std::runtime_error("parameter lookup " + suffix + "/" + name + " count=" + std::to_string(match_count));
    return match;
  }

  /// Pack named value adjoints back into exported flat-vector order.
  std::vector<double> flat_gradient(const std::unordered_map<const Node*, Tensor>& adjoints)
  {
    std::vector<double> flat_gradient(values.size());
    for (const Layout& layout : layouts)
    {
      const NodePtr parameter_node = nodes[{layout.module, layout.name}];
      const auto adjoint           = adjoints.find(parameter_node.get());
      if (adjoint != adjoints.end())
        std::copy(adjoint->second.x.begin(), adjoint->second.x.end(), flat_gradient.begin() + layout.begin);
    }
    return flat_gradient;
  }

  /// Pack the value component of mixed coordinate-jet adjoints in export order.
  std::vector<double> flat_gradient(const std::unordered_map<const Node*, JetAdjoint>& adjoints)
  {
    std::vector<double> flat_gradient(values.size());
    for (const Layout& layout : layouts)
    {
      const NodePtr parameter_node = nodes[{layout.module, layout.name}];
      const auto adjoint           = adjoints.find(parameter_node.get());
      if (adjoint != adjoints.end())
        std::copy(adjoint->second.value.x.begin(), adjoint->second.value.x.end(), flat_gradient.begin() + layout.begin);
    }
    return flat_gradient;
  }
};

/** System metadata and reference configurations read from the configuration
 * HDF5 file. */
struct ConfigData
{
  Tensor electrons;
  Tensor nuclei;
  Tensor charges;
  size_t nup;
  size_t ndown;
  size_t nconfig;

  /// Load electron configurations, nuclei, charges, and spin populations.
  explicit ConfigData(const std::string& path)
  {
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Preserve complete tensor shapes while reading the physical system and
    // the batch of electron configurations used by the validation driver.
    electrons.x = read_double(file, "/electron_positions", &electrons.shape);
    nuclei.x    = read_double(file, "/nuclear_positions", &nuclei.shape);
    charges.x   = read_double(file, "/nuclear_charges", &charges.shape);
    nup         = read_attr_i64(file, "n_up");
    ndown       = read_attr_i64(file, "n_down");
    nconfig     = electrons.shape[0];

    H5Fclose(file);
  }

  /// Extract one electron configuration as an [electron, Cartesian] tensor.
  Tensor configuration(size_t configuration_index) const
  {
    const size_t configuration_size = (nup + ndown) * 3;
    const auto begin                = electrons.x.begin() + configuration_index * configuration_size;
    const auto end                  = begin + configuration_size;
    return Tensor({nup + ndown, 3}, std::vector<double>(begin, end));
  }
};

/** Quantities produced by one all-electron PsiFormer evaluation. */
struct Result
{
  double sign;
  double logabs;
  double value;
  double local_energy;
  std::vector<double> gradient;
  std::vector<double> lap_log;
  std::vector<double> lap_ratio;
  std::vector<double> potential;
  std::vector<double> param_gradient;
  std::vector<double> local_energy_param_gradient;
};

/// Select the parameter reverse products required by one native evaluation.
enum class ParameterDerivativeRequest
{
  NONE,
  LOG_ONLY,
  LOG_AND_KINETIC
};

/** Describe optional parameter derivatives and the total wavefunction gradient
 * needed by QMCPACK's component kinetic-energy derivative. */
struct EvaluationRequest
{
  ParameterDerivativeRequest parameter_derivatives = ParameterDerivativeRequest::NONE;
  const std::vector<double>* total_log_gradient     = nullptr;
};

/** Builds and evaluates the four-block PsiFormer wavefunction from imported
 * parameters. */
struct PsiFormer
{
  // Short member names are retained for compatibility with the native
  // validation test, while their roles are documented here.
  Parameters p;   // Exported model parameters.
  ConfigData cfg; // Physical system and stored validation configurations.
  size_t ne;      // Total electron count.
  size_t ndet  = 16;
  size_t dim   = 256;
  size_t heads = 4;

  /// Load one fixed exported PsiFormer model and its physical-system metadata.
  PsiFormer(const std::string& parameter_path, const std::string& configuration_path)
      : p(parameter_path), cfg(configuration_path), ne(cfg.nup + cfg.ndown)
  {}

  /// Build learned electron features from electron-nucleus geometry and spin labels.
  NodePtr embedding(const NodePtr& positions)
  {
    // Construct four features for every electron-nucleus pair:
    //   log(1+r), dx*log(1+r)/r, dy*log(1+r)/r, dz*log(1+r)/r.
    // The softened radial form keeps the feature scale controlled while
    // retaining directional information.
    std::vector<NodePtr> electron_nucleus_features;
    for (size_t electron = 0; electron < ne; ++electron)
      for (size_t nucleus = 0; nucleus < cfg.nuclei.shape[0]; ++nucleus)
      {
        std::vector<NodePtr> displacement_components;
        for (size_t dimension = 0; dimension < 3; ++dimension)
        {
          const double nucleus_coordinate = cfg.nuclei.x[nucleus * 3 + dimension];
          displacement_components.push_back(
              add(slice_scalar(positions, electron, dimension), scalar(-nucleus_coordinate)));
        }

        NodePtr displacement   = concat(displacement_components, 0);
        NodePtr distance       = sqrt_node(sum_all(mul(displacement, displacement)));
        NodePtr radial_feature = log1p_node(distance);
        electron_nucleus_features.push_back(radial_feature);
        for (const NodePtr& displacement_component : displacement_components)
          electron_nucleus_features.push_back(mul(displacement_component, divide(radial_feature, distance)));
      }

    // Regroup the flat pair-feature list into one row per electron and append
    // +1/-1 as the spin label expected by the exported DeepQMC model.
    const size_t pair_feature_width = cfg.nuclei.shape[0] * 4;
    std::vector<NodePtr> electron_feature_rows;
    for (size_t electron = 0; electron < ne; ++electron)
    {
      const auto feature_begin = electron_nucleus_features.begin() + electron * pair_feature_width;
      std::vector<NodePtr> electron_features(feature_begin, feature_begin + pair_feature_width);
      electron_features.push_back(scalar(electron < cfg.nup ? 1 : -1));
      electron_feature_rows.push_back(reshape(concat(electron_features, 0), {1, pair_feature_width + 1}));
    }

    NodePtr raw_features     = concat(electron_feature_rows, 0);
    NodePtr embedding_weight = p.find("electron_embedding/linear", "w");
    return linear(raw_features, embedding_weight);
  }

  /// Select one scalar from a rank-two Cartesian tensor while preserving the graph.
  NodePtr slice_scalar(const NodePtr& tensor, size_t row, size_t column)
  {
    // slice0 operates on the leading axis, so flatten the selected Cartesian
    // row before selecting its requested scalar component.
    NodePtr selected_row  = slice0(tensor, row, row + 1);
    NodePtr flattened_row = reshape(selected_row, {3});
    return reshape(slice0(flattened_row, column, column + 1), {});
  }

  /// Apply one self-attention block and its residual two-layer MLP update.
  NodePtr attention_block(const NodePtr& input_features, int layer)
  {
    // Exported Haiku paths number the first block implicitly and suffix later
    // blocks with their zero-based layer index.
    const std::string layer_module  = layer == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(layer);
    const std::string update_module = layer_module + "/~/node_attention_electron_update_feature";
    const std::string attention_module = update_module + "/multi_head_attention";

    // Project electron features into query, key, and value heads.
    const Shape headed_feature_shape{ne, heads, dim / heads};
    NodePtr query = reshape(linear(input_features, p.find(attention_module + "/query", "w")), headed_feature_shape);
    NodePtr key   = reshape(linear(input_features, p.find(attention_module + "/key", "w")), headed_feature_shape);
    NodePtr value = reshape(linear(input_features, p.find(attention_module + "/value", "w")), headed_feature_shape);

    // Scaled dot-product attention mixes information across all electrons.
    NodePtr unscaled_logits   = attention_logits(query, key);
    NodePtr scaled_logits     = mul(unscaled_logits, scalar(1 / std::sqrt(double(dim / heads))));
    NodePtr attention_weights = softmax(scaled_logits);
    NodePtr attended_heads    = attention_context(attention_weights, value);
    NodePtr attended_features = reshape(attended_heads, {ne, dim});

    // Project concatenated heads and add the attention residual connection.
    NodePtr projected_attention = linear(attended_features, p.find(attention_module + "/linear", "w"));
    NodePtr attention_residual  = add(input_features, projected_attention);

    // A two-layer tanh MLP supplies the second residual update.
    NodePtr hidden_features = tanh_node(linear(attention_residual, p.find(update_module + "/mlp/linear_0", "w"),
                                               p.find(update_module + "/mlp/linear_0", "b")));
    NodePtr feature_update  = tanh_node(linear(hidden_features, p.find(update_module + "/mlp/linear_1", "w"),
                                               p.find(update_module + "/mlp/linear_1", "b")));
    return add(attention_residual, feature_update);
  }

  /// Project one spin block into determinant-specific orbital values.
  NodePtr backflow(const NodePtr& electron_features, bool spin_up)
  {
    // Select one spin block, project every electron feature into all
    // determinant/orbital channels, and arrange it as [det, spin electron,
    // orbital] for determinant assembly.
    const size_t spin_electron_count   = spin_up ? cfg.nup : cfg.ndown;
    const size_t spin_begin            = spin_up ? 0 : cfg.nup;
    const size_t spin_end              = spin_up ? cfg.nup : ne;
    const std::string parameter_module = spin_up ? "Backflow/~/mlp/linear_0" : "Backflow_1/~/mlp/linear_0";

    NodePtr spin_features           = slice0(electron_features, spin_begin, spin_end);
    NodePtr projected_orbitals      = linear(spin_features, p.find(parameter_module, "w"));
    NodePtr electron_major_orbitals = reshape(projected_orbitals, {spin_electron_count, ndet, ne});
    return transpose(electron_major_orbitals, {1, 0, 2});
  }

  /// Construct learned atom-centred exponential envelopes for one spin block.
  NodePtr envelope(const NodePtr& positions, bool spin_up)
  {
    // Each orbital is multiplied by a learned sum of atom-centred exponential
    // decays. Separate parameter arrays are exported for the two spin blocks.
    const size_t spin_begin          = spin_up ? 0 : cfg.nup;
    const size_t spin_electron_count = spin_up ? cfg.nup : cfg.ndown;
    NodePtr pi                       = p.find("exponential_envelopes", spin_up ? "pi_up" : "pi_down");
    NodePtr zeta                     = p.find("exponential_envelopes", spin_up ? "zetas_up" : "zetas_down");
    NodePtr flattened_pi             = reshape(pi, {pi->value.size()});
    NodePtr flattened_zeta           = reshape(zeta, {zeta->value.size()});

    std::vector<NodePtr> envelope_elements;
    for (size_t spin_electron = 0; spin_electron < spin_electron_count; ++spin_electron)
      for (size_t determinant = 0; determinant < ndet; ++determinant)
        for (size_t orbital = 0; orbital < ne; ++orbital)
        {
          NodePtr atom_sum = scalar(0);
          for (size_t nucleus = 0; nucleus < cfg.nuclei.shape[0]; ++nucleus)
          {
            // Compute the electron-nucleus distance for this spin electron.
            std::vector<NodePtr> displacement_components;
            for (size_t dimension = 0; dimension < 3; ++dimension)
            {
              const double nucleus_coordinate = cfg.nuclei.x[nucleus * 3 + dimension];
              displacement_components.push_back(
                  add(slice_scalar(positions, spin_begin + spin_electron, dimension), scalar(-nucleus_coordinate)));
            }
            NodePtr displacement = concat(displacement_components, 0);
            NodePtr distance     = sqrt_node(sum_all(mul(displacement, displacement)));

            // Parameter layout is [determinant, orbital, nucleus].
            const size_t parameter_index =
                determinant * ne * cfg.nuclei.shape[0] + orbital * cfg.nuclei.shape[0] + nucleus;
            NodePtr zeta_element = reshape(slice0(flattened_zeta, parameter_index, parameter_index + 1), {});
            NodePtr pi_element   = reshape(slice0(flattened_pi, parameter_index, parameter_index + 1), {});

            NodePtr exponential_decay = exp_node(neg(abs_node(mul(zeta_element, distance))));
            atom_sum                  = add(atom_sum, mul(pi_element, exponential_decay));
          }
          envelope_elements.push_back(atom_sum);
        }

    NodePtr electron_major_envelope = reshape(concat(envelope_elements, 0), {spin_electron_count, ndet, ne});
    return transpose(electron_major_envelope, {1, 0, 2});
  }

  /// Construct the analytic same-spin and opposite-spin electron cusp correction.
  NodePtr cusp(const NodePtr& positions)
  {
    // Same-spin and opposite-spin electron pairs use the physical 1/4 and 1/2
    // cusp factors, respectively, with learned asymptotic length scales.
    NodePtr same_spin_alpha     = p.find("electronic_cusp_asymptotic", "same_alpha");
    NodePtr opposite_spin_alpha = p.find("electronic_cusp_asymptotic", "anti_alpha");
    NodePtr cusp_value          = scalar(0);

    auto add_pair_cusp = [&](size_t first_electron, size_t second_electron, const NodePtr& alpha, double cusp_factor) {
      std::vector<NodePtr> displacement_components;
      for (size_t dimension = 0; dimension < 3; ++dimension)
        displacement_components.push_back(add(slice_scalar(positions, first_electron, dimension),
                                              neg(slice_scalar(positions, second_electron, dimension))));
      NodePtr displacement = concat(displacement_components, 0);
      NodePtr distance     = sqrt_node(sum_all(mul(displacement, displacement)));

      // -factor * alpha^2 / (alpha + r) approaches the desired linear cusp at
      // coalescence while saturating at large separation.
      NodePtr numerator   = mul(scalar(cusp_factor), mul(alpha, alpha));
      NodePtr denominator = add(alpha, distance);
      cusp_value          = add(cusp_value, neg(divide(numerator, denominator)));
    };

    // Accumulate up-up and down-down pairs without double counting.
    for (size_t spin_block = 0; spin_block < 2; ++spin_block)
    {
      const size_t spin_begin = spin_block ? cfg.nup : 0;
      const size_t spin_end   = spin_block ? ne : cfg.nup;
      for (size_t first_electron = spin_begin; first_electron < spin_end; ++first_electron)
        for (size_t second_electron = first_electron + 1; second_electron < spin_end; ++second_electron)
          add_pair_cusp(first_electron, second_electron, same_spin_alpha, .25);
    }

    // Accumulate every up-down pair.
    for (size_t up_electron = 0; up_electron < cfg.nup; ++up_electron)
      for (size_t down_electron = cfg.nup; down_electron < ne; ++down_electron)
        add_pair_cusp(up_electron, down_electron, opposite_spin_alpha, .5);
    return cusp_value;
  }

  /// Evaluate observables and exactly the parameter reverse products requested by the caller.
  Result evaluate(const Tensor& electron_positions, const EvaluationRequest& request)
  {
    const bool with_parameter_gradient = request.parameter_derivatives != ParameterDerivativeRequest::NONE;

    // Feature layers: electron-nucleus embedding followed by four attention
    // blocks operating on all electrons.
    NodePtr positions         = coordinates(electron_positions);
    NodePtr electron_features = embedding(positions);
    for (int layer = 0; layer < 4; ++layer)
      electron_features = attention_block(electron_features, layer);

    // Remaining wavefunction layers: create spin-resolved orbital matrices by
    // multiplying backflow outputs by their exponential envelopes.
    NodePtr up_orbitals      = mul(envelope(positions, true), backflow(electron_features, true));
    NodePtr down_orbitals    = mul(envelope(positions, false), backflow(electron_features, false));
    NodePtr orbital_matrices = concat({up_orbitals, down_orbitals}, 1);

    // Sum determinant channels, then add the analytic cusp in log space.
    NodePtr determinant_channels =
        with_parameter_gradient ? differentiable_determinants(orbital_matrices) : determinants(orbital_matrices);
    NodePtr determinant_sum  = sum_all(determinant_channels);
    NodePtr log_wavefunction = add(log_node(abs_node(determinant_sum)), cusp(positions));

    Result result;
    result.sign     = determinant_sum->value.x[0] > 0 ? 1 : -1;
    result.logabs   = log_wavefunction->value.x[0];
    result.value    = result.sign * std::exp(result.logabs);
    result.gradient = log_wavefunction->d1.x;

    // Convert Cartesian diagonal second derivatives of log|psi| into the
    // per-electron logarithmic Laplacian and (nabla^2 psi)/psi.
    result.lap_log.resize(ne);
    result.lap_ratio.resize(ne);
    for (size_t electron = 0; electron < ne; ++electron)
    {
      double squared_gradient_norm = 0;
      for (size_t dimension = 0; dimension < 3; ++dimension)
      {
        const size_t coordinate = electron * 3 + dimension;
        result.lap_log[electron] += log_wavefunction->d2.x[coordinate];
        squared_gradient_norm += result.gradient[coordinate] * result.gradient[coordinate];
      }
      result.lap_ratio[electron] = result.lap_log[electron] + squared_gradient_norm;
    }

    // Accumulate electron-electron, electron-nucleus, and nucleus-nucleus
    // straight-Coulomb terms separately for validation and local energy.
    auto distance = [](const double* first, const double* second) {
      double squared_distance = 0;
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        const double displacement = first[dimension] - second[dimension];
        squared_distance += displacement * displacement;
      }
      return std::sqrt(squared_distance);
    };

    double electron_electron_potential = 0;
    for (size_t first_electron = 0; first_electron < ne; ++first_electron)
      for (size_t second_electron = first_electron + 1; second_electron < ne; ++second_electron)
        electron_electron_potential +=
            1 / distance(&electron_positions.x[first_electron * 3], &electron_positions.x[second_electron * 3]);

    double electron_nucleus_potential = 0;
    for (size_t electron = 0; electron < ne; ++electron)
      for (size_t nucleus = 0; nucleus < cfg.nuclei.shape[0]; ++nucleus)
        electron_nucleus_potential -=
            cfg.charges.x[nucleus] / distance(&electron_positions.x[electron * 3], &cfg.nuclei.x[nucleus * 3]);

    double nucleus_nucleus_potential = 0;
    for (size_t first_nucleus = 0; first_nucleus < cfg.nuclei.shape[0]; ++first_nucleus)
      for (size_t second_nucleus = first_nucleus + 1; second_nucleus < cfg.nuclei.shape[0]; ++second_nucleus)
        nucleus_nucleus_potential += cfg.charges.x[first_nucleus] * cfg.charges.x[second_nucleus] /
            distance(&cfg.nuclei.x[first_nucleus * 3], &cfg.nuclei.x[second_nucleus * 3]);

    result.potential = {electron_electron_potential, electron_nucleus_potential, nucleus_nucleus_potential};

    // Local energy is kinetic energy plus the three Coulomb contributions.
    const double laplacian_ratio = std::accumulate(result.lap_ratio.begin(), result.lap_ratio.end(), 0.);
    result.local_energy =
        -.5 * laplacian_ratio + electron_electron_potential + electron_nucleus_potential + nucleus_nucleus_potential;

    // Reverse the ordinary value graph only when the caller needs score
    // derivatives. SR-style callers intentionally avoid the more expensive
    // mixed coordinate-jet reverse below.
    if (with_parameter_gradient)
      result.param_gradient = p.flat_gradient(backward(log_wavefunction));

    // QMCPACK composes wavefunction factors, so the first-derivative seed must
    // use the total trial-wavefunction gradient rather than necessarily this
    // component's gradient. For standalone evaluation the component gradient
    // remains the default and reproduces dE_L/dtheta for the full PsiFormer.
    if (request.parameter_derivatives == ParameterDerivativeRequest::LOG_AND_KINETIC)
    {
      const std::vector<double>& total_gradient =
          request.total_log_gradient ? *request.total_log_gradient : log_wavefunction->d1.x;
      if (total_gradient.size() != log_wavefunction->d1.size())
        throw std::invalid_argument("PsiFormer total log-gradient seed has the wrong size");

      JetAdjoint local_energy_seed;
      local_energy_seed.value = Tensor(log_wavefunction->value.shape);
      local_energy_seed.d1    = Tensor(log_wavefunction->d1.shape);
      local_energy_seed.d2    = Tensor(log_wavefunction->d2.shape, -0.5);
      for (size_t coordinate = 0; coordinate < log_wavefunction->d1.size(); ++coordinate)
        local_energy_seed.d1.x[coordinate] = -total_gradient[coordinate];
      result.local_energy_param_gradient =
          p.flat_gradient(backward_coordinate_jets(log_wavefunction, std::move(local_energy_seed)));
    }
    return result;
  }

  /// Preserve the original standalone boolean API while routing through explicit requests.
  Result evaluate(const Tensor& electron_positions, bool with_parameter_gradient = true)
  {
    const ParameterDerivativeRequest derivative_request = with_parameter_gradient
        ? ParameterDerivativeRequest::LOG_AND_KINETIC
        : ParameterDerivativeRequest::NONE;
    return evaluate(electron_positions, EvaluationRequest{derivative_request, nullptr});
  }
};

// Standalone validation support. This code is excluded when included by
// QMCPACK.
/// Read one named validation observable from the reference HDF5 file.
std::vector<double> read_reference(hid_t reference_file, const std::string& field_name, Shape* shape = nullptr)
{ return read_double(reference_file, "/" + field_name, shape); }

/// Write one computed validation observable to the output HDF5 file.
void write_dataset(hid_t output_file,
                   const std::string& field_name,
                   const Shape& shape,
                   const std::vector<double>& values)
{
  std::vector<hsize_t> dimensions(shape.begin(), shape.end());
  const hid_t dataspace = H5Screate_simple(dimensions.size(), dimensions.data(), nullptr);
  const hid_t dataset =
      H5Dcreate2(output_file, field_name.c_str(), H5T_IEEE_F64LE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

/// Run standalone evaluation, reference comparison, and HDF5 output generation.
int run(const std::string& parameter_path,
        const std::string& configuration_path,
        const std::string& reference_path,
        const std::string& output_path)
{
  PsiFormer model(parameter_path, configuration_path);

  // Evaluate every stored configuration once and retain all high-level
  // observables for comparison and output.
  std::vector<Result> results;
  for (size_t configuration_index = 0; configuration_index < model.cfg.nconfig; ++configuration_index)
  {
    Result result = model.evaluate(model.cfg.configuration(configuration_index));
    std::cout << "config " << configuration_index + 1 << " sign=" << result.sign << " logabs=" << std::setprecision(12)
              << result.logabs << " E_L=" << result.local_energy << "\n";
    results.push_back(std::move(result));
  }

  const hid_t reference_file = H5Fopen(reference_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t output_file    = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Describe each observable uniformly by its dataset name, per-configuration
  // trailing shape, and extraction function.
  struct ValidationField
  {
    std::string name;
    Shape trailing_shape;
    std::function<std::vector<double>(const Result&)> extract;
  };

  std::vector<ValidationField> fields =
      {{"sign", {}, [](const Result& result) { return std::vector<double>{result.sign}; }},
       {"logabs", {}, [](const Result& result) { return std::vector<double>{result.logabs}; }},
       {"value", {}, [](const Result& result) { return std::vector<double>{result.value}; }},
       {"gradient_logabs", {model.ne, 3}, [](const Result& result) { return result.gradient; }},
       {"laplacian_logabs_electron", {model.ne}, [](const Result& result) { return result.lap_log; }},
       {"laplacian_ratio_electron", {model.ne}, [](const Result& result) { return result.lap_ratio; }},
       {"potential_ee_en_nn", {3}, [](const Result& result) { return result.potential; }},
       {"local_energy", {}, [](const Result& result) { return std::vector<double>{result.local_energy}; }},
       {"parameter_gradient_logabs",
        {model.p.values.size()},
        [](const Result& result) { return result.param_gradient; }},
       {"parameter_gradient_local_energy", {model.p.values.size()}, [](const Result& result) {
          return result.local_energy_param_gradient;
        }}};

  bool all_fields_pass = true;
  for (const ValidationField& field : fields)
  {
    // Flatten this field across configurations using the same leading batch
    // dimension as the reference and output datasets.
    std::vector<double> computed_values;
    for (const Result& result : results)
    {
      const std::vector<double> configuration_values = field.extract(result);
      computed_values.insert(computed_values.end(), configuration_values.begin(), configuration_values.end());
    }

    if (H5Lexists(reference_file, field.name.c_str(), H5P_DEFAULT) > 0)
    {
      const std::vector<double> reference_values = read_reference(reference_file, field.name);
      double maximum_absolute_error              = 0;
      double maximum_relative_error              = 0;
      for (size_t element = 0; element < computed_values.size(); ++element)
      {
        const double absolute_error = std::abs(computed_values[element] - reference_values[element]);
        const double relative_error = absolute_error / (1e-12 + std::abs(reference_values[element]));
        maximum_absolute_error      = std::max(maximum_absolute_error, absolute_error);
        maximum_relative_error      = std::max(maximum_relative_error, relative_error);
      }

      constexpr double tolerance = 2e-9;
      const bool field_passes    = maximum_absolute_error <= tolerance || maximum_relative_error <= tolerance;
      all_fields_pass &= field_passes;
      std::cout << std::setw(32) << field.name << " max_abs=" << std::scientific << maximum_absolute_error
                << " max_rel=" << maximum_relative_error << " " << (field_passes ? "PASS" : "FAIL") << "\n";
    }
    else
      std::cout << std::setw(32) << field.name << " NOT PRESENT IN REFERENCE\n";

    Shape output_shape{results.size()};
    output_shape.insert(output_shape.end(), field.trailing_shape.begin(), field.trailing_shape.end());
    write_dataset(output_file, field.name, output_shape, computed_values);
  }

  H5Fclose(output_file);
  H5Fclose(reference_file);
  return all_fields_pass ? 0 : 2;
}

} // namespace pf

#ifndef PSIFORMER_LIBRARY
/// Parse standalone-driver paths and report evaluation failures to the shell.
int main(int argc, char** argv)
{
  if (argc != 5)
  {
    std::cerr << "usage: psiformer_cpp PARAMETERS.h5 CONFIGS.h5 REFERENCE.h5 "
                 "OUTPUT.h5\n";
    return 1;
  }
  try
  {
    return pf::run(argv[1], argv[2], argv[3], argv[4]);
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
#endif // PSIFORMER_LIBRARY

#endif // QMCPLUSPLUS_PSIFORMER_NATIVE_H
