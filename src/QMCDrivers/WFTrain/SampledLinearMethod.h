//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_SAMPLED_LINEAR_METHOD_H
#define QMCPLUSPLUS_SAMPLED_LINEAR_METHOD_H
#include "MatrixFreeLinearMethod.h"
#include "MinimumStepStochasticReconfiguration.h"
#include <memory>

namespace qmcplusplus::wftrain
{
/** Stream augmented D=[1,centered score] and L=[E_L,dE_L+E_L*centered score].
 * Global actions are D^T q D, D^T q L, or its symmetric part. Only samples,
 * derivative owners, bounded products and O(P) scratch are retained.
 */
class SampledLinearMethodOperator final : public AugmentedLinearMethodOperator
{
public:
  /// Bind unchanged sampled batches and add a tangent-only identity stabilization.
  SampledLinearMethodOperator(const ScoreCovarianceFactor& factor,
      DerivativeArrayView<const ScoreCovarianceBatch> batches, std::uint32_t term_mask,
      bool overlap, bool symmetrize, double tangent_shift);

  /// Report owned numeric scratch for the driver memory ceiling.
  std::size_t retainedBytes() const noexcept;

protected:
  /// Apply the sampled augmented action without constructing derivative matrices.
  void evaluate(DerivativeArrayView<const DerivativeValue> direction,
                DerivativeArrayView<DerivativeValue> result) const override;

private:
  const ScoreCovarianceFactor& factor_;
  std::vector<ScoreCovarianceBatch> batches_;
  std::uint32_t term_mask_;
  bool overlap_, symmetrize_;
  double shift_;
  std::unique_ptr<ProjectedInverseIterationFactor> h_factor_;
  std::vector<double> sqrt_q_, energies_;
  mutable std::vector<DerivativeValue> d_, l_, adjoint_, coefficients_, local_gradient_;
};

/// Nonlinear xi damping or exact linear-state retraction with a frozen reference coefficient.
enum class LinearMethodNormalization
{
  NONLINEAR,
  FIXED_REFERENCE_LINEAR
};

/** Stateless live LM update with explicit coordinate normalization, never publication. */
class LinearMethodUpdateRule final : public TrainingUpdateRule
{
public:
  /// Bind a converged eigensolve to its immutable sampled metric and step controls.
  /// Linear batch views are nonowning: producers, weights and energies must outlive the rule.
  LinearMethodUpdateRule(const ScoreCovarianceFactor& factor,
                        MatrixFreeLinearMethodResult result, double learning_rate,
                        double normalization_xi,
                        LinearMethodNormalization normalization = LinearMethodNormalization::NONLINEAR,
                        DerivativeArrayView<const ScoreCovarianceBatch> linear_batches = {});
  StructuredParameterSnapshot propose(const StructuredParameterSchema& schema,
      const StructuredParameterSnapshot& parameters, ParameterGradientView objective) override;

private:
  const ScoreCovarianceFactor& factor_;
  MatrixFreeLinearMethodResult result_;
  double learning_rate_, xi_;
  LinearMethodNormalization normalization_;
  std::vector<ScoreCovarianceBatch> linear_batches_;
};
}
#endif

