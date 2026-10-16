//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "QMCDrivers/WFTrain/ProposalSelection.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"
#include "OhmmsData/Libxml2Doc.h"
#include <cmath>

namespace qmcplusplus::wftrain
{
namespace
{
/// Publish-counting provider for checking that rejection leaves the model untouched.
class SelectionProvider final : public StructuredParameterProvider
{
public:
  StructuredParameterSchema schema{"selection", {
      {"weight", {2}, 0, 2, ParameterScalarDomain::REAL64, true, "weights"},
      {"bias", {1}, 2, 1, ParameterScalarDomain::REAL64, true, "bias"},
      {"fixed", {1}, 3, 1, ParameterScalarDomain::REAL64, false, "fixed"}}};
  StructuredParameterSnapshot parameters{schema.fingerprint(), 0, {2, 4, 6, 8}};
  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema; }
  StructuredParameterSnapshot snapshotParameters() const override { return parameters; }
  std::size_t publishParameters(const StructuredParameterSnapshot& candidate, std::size_t version) override
  {
    REQUIRE(version == parameters.version);
    parameters = candidate;
    return ++parameters.version;
  }
};

/// A deterministic direction and recurrence counters independent of the selector.
class SelectionOptimizer final : public TrainingUpdateRule
{
public:
  int accepted = 0;
  int rejected = 0;
  StructuredParameterSnapshot propose(const StructuredParameterSchema&,
                                      const StructuredParameterSnapshot& old,
                                      ParameterGradientView) override
  {
    auto result = old;
    result.values[0] += 4;
    result.values[1] -= 2;
    result.values[2] += 1;
    return result;
  }
  void proposalAccepted(const StructuredParameterSchema&, const StructuredParameterSnapshot&,
                        ParameterGradientView) noexcept override { ++accepted; }
  void proposalRejected() noexcept override { ++rejected; }
};
}

TEST_CASE("Proposal scaling and decay preserve frozen and nonselected tensors", "[drivers][training][selection]")
{
  SelectionProvider provider;
  SelectionOptimizer optimizer;
  const std::vector<double> gradient(4, 0.0);
  const ParameterGradientView objective{provider.schema.fingerprint(), 0, ReductionDomain::GLOBAL,
                                        {gradient.data(), gradient.size()}};
  ProposalSelectionControl control;
  control.update_scale = 0.5;
  control.weight_decay = 0.1;
  control.decay_blocks = {"weight", "fixed"};
  SelectedUpdateRule selection(optimizer, control);
  const auto old = provider.parameters;
  CHECK(completeParameterUpdate(provider, old, objective, selection, {}) == 1);
  CHECK(provider.parameters.values[0] == Catch::Approx(3.8));
  CHECK(provider.parameters.values[1] == Catch::Approx(2.6));
  CHECK(provider.parameters.values[2] == Catch::Approx(6.5));
  CHECK(provider.parameters.values[3] == 8.0);
  CHECK(optimizer.accepted == 1);
  CHECK(optimizer.rejected == 0);
}

TEST_CASE("Line-search rejection preserves version parameters and recurrence", "[drivers][training][selection]")
{
  SelectionProvider provider;
  SelectionOptimizer optimizer;
  const std::vector<double> gradient(4, 0.0);
  const ParameterGradientView objective{provider.schema.fingerprint(), 0, ReductionDomain::GLOBAL,
                                        {gradient.data(), gradient.size()}};
  SelectedUpdateRule selection(optimizer, {}, [](const auto&, const auto&) { return 0.0; });
  const auto old = provider.parameters;
  CHECK(completeParameterUpdate(provider, old, objective, selection, {}) == 0);
  CHECK(provider.parameters.values == old.values);
  CHECK(optimizer.accepted == 0);
  CHECK(optimizer.rejected == 1);
}

TEST_CASE("Proposal expansion respects final displacement bound", "[drivers][training][selection]")
{
  SelectionProvider provider;
  SelectionOptimizer optimizer;
  const std::vector<double> gradient(4, 0.0);
  const ParameterGradientView objective{provider.schema.fingerprint(), 0, ReductionDomain::GLOBAL,
                                        {gradient.data(), gradient.size()}};
  ProposalSelectionControl control;
  control.maximum_update_norm = 1.0;
  SelectedUpdateRule selection(optimizer, control, [](const auto&, const auto&) { return 2.0; });
  selection.propose(provider.schema, provider.parameters, objective);
  CHECK(selection.displacementNorm() == Catch::Approx(1.0));
}

TEST_CASE("Host Adam controls match explicit scaled shrunk and bounded displacement",
          "[drivers][training][selection][host_controls]")
{
  SelectionProvider provider;
  FirstOrderOptimizerOptions options;
  options.learning_rates = {{"weights", 0.2}, {"bias", 0.1}};
  options.adam_beta1 = 0.8;
  options.adam_beta2 = 0.9;
  options.epsilon = 1e-4;
  FirstOrderOptimizer optimizer(provider.schema, options);
  const std::vector<double> gradient{0.3, -0.4, 0.7, 0.0};
  const ParameterGradientView objective{provider.schema.fingerprint(), 0, ReductionDomain::GLOBAL,
                                        {gradient.data(), gradient.size()}};
  const auto old = provider.parameters;
  ProposalSelectionControl control;
  control.update_scale = 0.25;
  control.weight_decay = 0.1;
  control.decay_blocks = {"weight"};
  control.maximum_update_norm = 0.12;

  // First Adam step has bias-corrected m=g and v=g^2. This oracle does not
  // obtain its raw direction or ordering from SelectedUpdateRule.
  const std::vector<double> rates{0.2, 0.2, 0.1, 0.0};
  std::vector<double> delta(4, 0.0);
  double combined_norm = 0.0;
  for (std::size_t i = 0; i < 3; ++i)
  {
    delta[i] = control.update_scale * (-rates[i] * gradient[i] / (std::abs(gradient[i]) + options.epsilon));
    if (i < 2)
      delta[i] -= control.weight_decay * old.values[i];
    combined_norm = std::hypot(combined_norm, delta[i]);
  }
  REQUIRE(combined_norm > control.maximum_update_norm);
  SelectedUpdateRule selection(optimizer, control);
  CHECK(completeParameterUpdate(provider, old, objective, selection, {}) == 1);
  double applied_norm = 0.0;
  for (std::size_t i = 0; i < old.values.size(); ++i)
  {
    const double displacement = delta[i] * control.maximum_update_norm / combined_norm;
    CHECK(provider.parameters.values[i] == Catch::Approx(old.values[i] + displacement));
    applied_norm = std::hypot(applied_norm, provider.parameters.values[i] - old.values[i]);
    CHECK(optimizer.firstMoment()[i] == Catch::Approx((1 - options.adam_beta1) * gradient[i]));
    CHECK(optimizer.secondMoment()[i] == Catch::Approx((1 - options.adam_beta2) * gradient[i] * gradient[i]));
  }
  CHECK(applied_norm == Catch::Approx(control.maximum_update_norm));
  CHECK(provider.parameters.values[3] == old.values[3]);
  CHECK(optimizer.acceptedUpdateCount() == 1);
  CHECK_FALSE(optimizer.hasLiveProposal());
}

TEST_CASE("Host zero-scale off-selection rejects without committing Adam moments",
          "[drivers][training][selection][host_controls]")
{
  SelectionProvider provider;
  FirstOrderOptimizerOptions options;
  options.learning_rates = {{"weights", 0.2}, {"bias", 0.1}};
  FirstOrderOptimizer optimizer(provider.schema, options);
  const std::vector<double> gradient{0.3, -0.4, 0.7, 0.0};
  const ParameterGradientView objective{provider.schema.fingerprint(), 0, ReductionDomain::GLOBAL,
                                        {gradient.data(), gradient.size()}};
  const auto old = provider.parameters;
  const auto old_first = optimizer.firstMoment();
  const auto old_second = optimizer.secondMoment();
  ProposalSelectionControl control;
  control.update_scale = 0.0;
  // Match the driver's line_search=off zero-scale/no-decay rejection policy.
  SelectedUpdateRule zero_step(optimizer, control, [](const auto&, const auto&) { return 0.0; });
  CHECK(completeParameterUpdate(provider, old, objective, zero_step, {}) == 0);
  CHECK(provider.parameters.values == old.values);
  CHECK(provider.parameters.version == old.version);
  CHECK(optimizer.firstMoment() == old_first);
  CHECK(optimizer.secondMoment() == old_second);
  CHECK(optimizer.acceptedUpdateCount() == 0);
  CHECK_FALSE(optimizer.hasLiveProposal());

  // With shrinkage requested, scale zero suppresses only the optimizer step.
  control.weight_decay = 0.1;
  control.decay_blocks = {"weight"};
  SelectedUpdateRule shrink_only(optimizer, control, [](const auto&, const auto&) { return 1.0; });
  CHECK(completeParameterUpdate(provider, old, objective, shrink_only, {}) == 1);
  CHECK(provider.parameters.values[0] == Catch::Approx(0.9 * old.values[0]));
  CHECK(provider.parameters.values[1] == Catch::Approx(0.9 * old.values[1]));
  CHECK(provider.parameters.values[2] == old.values[2]);
  CHECK(provider.parameters.values[3] == old.values[3]);
  CHECK(optimizer.acceptedUpdateCount() == 1);
  CHECK_FALSE(optimizer.hasLiveProposal());
}

TEST_CASE("Line-search parameters parse strictly with unchanged numerical defaults", "[drivers][training][selection]")
{
  auto parse = [](const std::string& parameters) {
    Libxml2Document document;
    REQUIRE(document.parseFromString("<qmc><training>" + parameters + "</training></qmc>"));
    WFTrainDriverInput input;
    input.readXML(document.getRoot());
    return input;
  };
  const auto defaults = parse("");
  CHECK(defaults.getLineSearch() == WFTrainDriverInput::LineSearch::OFF);
  CHECK(defaults.getUpdateScale() == 1.0);
  CHECK(defaults.getWeightDecay() == 0.0);
  const auto input = parse(R"(
    <parameter name="line_search">endpoints</parameter>
    <parameter name="update_scale">0.25</parameter>
    <parameter name="weight_decay">0.001</parameter>
    <parameter name="weight_decay_scope">nn_weights</parameter>
    <parameter name="line_search_warmup_steps">3</parameter>
    <parameter name="line_search_sampling_steps">4</parameter>
  )");
  CHECK(input.getLineSearch() == WFTrainDriverInput::LineSearch::ENDPOINTS);
  CHECK(input.getUpdateScale() == 0.25);
  CHECK(input.getWeightDecay() == 0.001);
  CHECK(input.getLineSearchWarmupSteps() == 3);
  CHECK(input.getLineSearchSamplingSteps() == 4);
  for (const auto& invalid : {
           R"(<parameter name="line_search">unknown</parameter>)",
           R"(<parameter name="update_scale">-1</parameter>)",
           R"(<parameter name="weight_decay">1</parameter>)",
           R"(<parameter name="weight_decay_scope">all</parameter>)",
           R"(<parameter name="line_search_minimum_ess_fraction">0</parameter>)",
           R"(<parameter name="line_search_maximum_weight">2</parameter>)",
           R"(<parameter name="line_search_sampling_steps">0</parameter>)"})
    CHECK_THROWS(parse(invalid));
}
}
