//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "QMCDrivers/WFTrain/KFACOptimizer.h"
#include "QMCDrivers/WFTrain/ProposalSelection.h"
#include "io/hdf/hdf_archive.h"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>

namespace qmcplusplus::wftrain
{
namespace
{
/// Minimal scalar provider with a controllable publication failure.
class KFACToyProvider final : public StructuredParameterProvider
{
public:
  KFACToyProvider() : schema_("kfac/toy", {{"generic", {1}, 0, 1, ParameterScalarDomain::REAL64, true, "all"}}) {}

  const StructuredParameterSchema& parameterSchema() const noexcept override { return schema_; }

  StructuredParameterSnapshot snapshotParameters() const override { return {schema_.fingerprint(), version_, {value_}}; }

  std::size_t publishParameters(const StructuredParameterSnapshot& candidate, std::size_t version) override
  {
    if (reject_) { reject_ = false; throw std::runtime_error("Synthetic KFAC publication failure"); }
    if (version != version_ || candidate.version != version_ || candidate.values.size() != 1)
      throw std::runtime_error("Invalid KFAC toy publication");
    value_ = candidate.values[0];
    return ++version_;
  }

  void rejectNext() { reject_ = true; }

private:
  StructuredParameterSchema schema_;
  double value_ = 1.0;
  std::size_t version_ = 0;
  bool reject_ = false;
};

/// Isolate fallback recurrence math from the independently qualified affine solver.
KFACBlockRegistry registry(const StructuredParameterSchema& schema) { return {schema, {}, {0}, {0}}; }

/// Generate two exact score samples and exercise the serial reduction boundary.
KFACFactorAccumulator statistics(const StructuredParameterSchema& schema, std::size_t version, double scale,
    KFACCurvatureConvention convention = KFACCurvatureConvention::NATIVE_SCORE)
{
  KFACFactorAccumulator result(registry(schema), version, convention);
  for (double score : {scale, 2.0 * scale})
  {
    result.beginSample(1.0);
    result.addFallbackScores({&score, 1});
    result.endSample();
  }
  reduceKFACFactorStatistics(result);
  return result;
}

/// Drive the actual publication transaction, including optimizer accept/reject hooks.
void update(KFACOptimizer& optimizer, KFACToyProvider& provider, double scale, double gradient = 1.0)
{
  const auto parameters = provider.snapshotParameters();
  auto batch = statistics(provider.parameterSchema(), parameters.version, scale);
  optimizer.prepareStatistics(batch);
  const ParameterGradientView objective{provider.parameterSchema().fingerprint(), parameters.version,
      ReductionDomain::GLOBAL, {&gradient, 1}};
  completeParameterUpdate(provider, parameters, objective, optimizer, {});
}
}

TEST_CASE("Persistent KFAC commits normalized factors only after successful publication",
          "[wftrain][kfac_state]")
{
  KFACToyProvider provider;
  KFACOptimizerControl control;
  control.learning_rate = 0.1;
  control.ema_decay = 0.5;
  control.inverse_period = 2;
  control.inverse = {0.1, 0.1, KFACDampingPolicy::TRACE_BALANCED};
  KFACOptimizer optimizer(registry(provider.parameterSchema()), control);
  const auto original = optimizer.checkpointMetadata().state_fingerprint;
  provider.rejectNext();
  CHECK_THROWS(update(optimizer, provider, 1.0));
  CHECK(optimizer.step() == 0);
  CHECK(optimizer.dataSeen() == 0);
  CHECK(optimizer.checkpointMetadata().state_fingerprint == original);
  CHECK(provider.snapshotParameters().values[0] == 1.0);
  update(optimizer, provider, 1.0);
  CHECK(optimizer.step() == 1);
  CHECK(optimizer.dataSeen() == 2);
  CHECK(optimizer.inverseStep() == 0);
  // Native mean(score^2)=2.5. Fresh normalized EMA is the first batch itself.
  CHECK(provider.snapshotParameters().values[0] == Catch::Approx(1.0 - 0.1 / 2.6));
  update(optimizer, provider, 2.0);
  // EMA=(.5*2.5+10)/1.5=7.5; fallback always refreshes despite stale affine cadence.
  CHECK(provider.snapshotParameters().values[0] == Catch::Approx(1.0 - 0.1 / 2.6 - 0.1 / 7.6));
  CHECK(optimizer.inverseStep() == 0);
  update(optimizer, provider, 3.0);
  CHECK(optimizer.inverseStep() == 2);
  CHECK(optimizer.dataSeen() == 6);
  auto wrong_convention = statistics(provider.parameterSchema(), provider.snapshotParameters().version, 1.0,
                                     KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  CHECK_THROWS(optimizer.prepareStatistics(wrong_convention));
}

TEST_CASE("Complex KFAC factors and decorated recurrence survive failed publication",
          "[wftrain][kfac_state][complex_kfac_transaction]")
{
  KFACToyProvider provider;
  const auto original = provider.snapshotParameters();
  KFACOptimizerControl control;
  control.learning_rate = 0.1;
  control.convention = KFACCurvatureConvention::REGISTERED_GAUSSIAN;
  control.inverse = {0.1, 0.1, KFACDampingPolicy::TRACE_BALANCED};
  KFACOptimizer optimizer(registry(provider.parameterSchema()), control);
  KFACFactorAccumulator factors(registry(provider.parameterSchema()), 0, control.convention);
  const std::array<double, 2> real{1, 2}, imaginary{2, -1};
  for (std::size_t sample = 0; sample < real.size(); ++sample)
  {
    factors.beginSample(1);
    factors.addFallbackScores({real.data() + sample, 1}, {imaginary.data() + sample, 1});
    factors.endSample();
  }
  reduceKFACFactorStatistics(factors);
  const auto original_state = optimizer.checkpointMetadata().state_fingerprint;
  const auto original_diagonal = factors.fallbackDiagonalSum()[0];
  const double gradient = 2;
  const ParameterGradientView objective{provider.parameterSchema().fingerprint(), 0,
                                        ReductionDomain::GLOBAL, {&gradient, 1}};
  optimizer.prepareStatistics(factors);
  SelectedUpdateRule decorated(optimizer, {});
  provider.rejectNext();
  CHECK_THROWS(completeParameterUpdate(provider, original, objective, decorated, {}));
  CHECK(provider.snapshotParameters().values == original.values);
  CHECK(provider.snapshotParameters().version == original.version);
  CHECK(optimizer.step() == 0);
  CHECK(optimizer.dataSeen() == 0);
  CHECK(optimizer.checkpointMetadata().state_fingerprint == original_state);
  CHECK(factors.fallbackDiagonalSum()[0] == original_diagonal);
  optimizer.prepareStatistics(factors);
  completeParameterUpdate(provider, original, objective, decorated, {});
  CHECK(optimizer.step() == 1);
  CHECK(optimizer.dataSeen() == 2);
  CHECK(provider.snapshotParameters().values[0] == Catch::Approx(1 - 0.1 * 2 / 10.1));
}

TEST_CASE("Persistent KFAC continuous and split checkpoints preserve the next update exactly",
          "[wftrain][kfac_state][checkpoint]")
{
  KFACToyProvider continuous, split;
  KFACOptimizerControl control;
  control.learning_rate = 0.05;
  control.learning_rate_decay = 10000.0;
  control.ema_decay = 0.95;
  control.inverse_period = 3;
  KFACOptimizer whole(registry(continuous.parameterSchema()), control);
  KFACOptimizer first(registry(split.parameterSchema()), control);
  for (int i = 1; i <= 4; ++i) update(whole, continuous, i);
  for (int i = 1; i <= 2; ++i) update(first, split, i);
  const auto path = std::filesystem::temp_directory_path() /
      ("kfac_checkpoint_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  struct Cleanup
  {
    std::filesystem::path path;

    ~Cleanup() { std::error_code ignored; std::filesystem::remove(path, ignored); }
  } cleanup{path};
  TrainingIterationState coordinator{2, split.snapshotParameters().version, split.parameterSchema().fingerprint()};
  TrainingStageState stage{"energy", 0, 2, "toy-stateful-kfac"};
  TrainingCheckpoint::saveAtomic(path, split, coordinator, stage, &first);

  KFACToyProvider resumed;
  KFACOptimizer restored(registry(resumed.parameterSchema()), control);
  TrainingIterationState restored_coordinator{0, 0, resumed.parameterSchema().fingerprint()};
  TrainingStageState restored_stage{"energy", 0, 0, "toy-stateful-kfac"};
  TrainingCheckpoint::restore(path, resumed, restored_coordinator, restored_stage, &restored);
  CHECK(restored.step() == 2);
  CHECK(restored.dataSeen() == 4);
  CHECK(restored.inverseStep() == 0);
  for (int i = 3; i <= 4; ++i) update(restored, resumed, i);
  CHECK(resumed.snapshotParameters().values == continuous.snapshotParameters().values);
  CHECK(restored.checkpointMetadata().state_fingerprint == whole.checkpointMetadata().state_fingerprint);

  auto incompatible_control = control;
  incompatible_control.ema_decay = 0.5;
  KFACToyProvider incompatible_provider;
  KFACOptimizer incompatible(registry(incompatible_provider.parameterSchema()), incompatible_control);
  TrainingIterationState bad_coordinator{0, 0, incompatible_provider.parameterSchema().fingerprint()};
  TrainingStageState bad_stage{"energy", 0, 0, "toy-stateful-kfac"};
  CHECK_THROWS(TrainingCheckpoint::restore(path, incompatible_provider, bad_coordinator, bad_stage, &incompatible));
  CHECK(incompatible_provider.snapshotParameters().values[0] == 1.0);
  CHECK(incompatible.step() == 0);
  CHECK_THROWS(TrainingCheckpoint::restore(path, incompatible_provider, bad_coordinator, bad_stage));
}

TEST_CASE("Persistent KFAC schedule precision and constraint retain scalar dtype semantics",
          "[wftrain][kfac_state]")
{
  for (bool fp32 : {false, true})
  {
    KFACToyProvider provider;
    KFACOptimizerControl control;
    control.learning_rate = 0.05;
    control.learning_rate_decay = 10000.0;
    control.schedule_float32 = fp32;
    control.norm_constraint = 1e-5;
    // score=[s,2s], mean square=1 for s=sqrt(.4); damping1 gives g.P(g)=.5.
    control.inverse = {1.0, 1.0};
    KFACOptimizer optimizer(registry(provider.parameterSchema()), control);
    CHECK(optimizer.effectiveLearningRate() == (fp32 ? static_cast<double>(0.05F) : 0.05));
    for (int i = 0; i < 100; ++i) update(optimizer, provider, std::sqrt(0.4), 0.0);
    CHECK(optimizer.effectiveLearningRate() == (fp32 ? 0.049504950642585754 : 0.05 / 1.01));
    update(optimizer, provider, std::sqrt(0.4));
    const double expected = fp32 ? 0.00122537009883672 : (0.05 / 1.01) * (0.05 / 1.01) * 0.5;
    CHECK(optimizer.constraintMetric() == Catch::Approx(expected).epsilon(1e-12));
    CHECK(optimizer.constraintScale() == Catch::Approx(std::sqrt(1e-5 / expected)).epsilon(1e-12));
    CHECK(optimizer.constraintScale() < 1.0);
  }
}

TEST_CASE("Pre-inverse rate scaling uses full precision parameter-space constraint arithmetic",
          "[wftrain][kfac_state][kfac_update_order]")
{
  for (const double bound : {0.0, 1e-5})
  {
    KFACToyProvider provider;
    KFACOptimizerControl control;
    control.learning_rate = .05;
    control.learning_rate_decay = 2000;
    control.schedule_float32 = true;
    control.inverse = {1.0, 1.0};
    control.norm_constraint = bound;
    control.update_order = KFACUpdateOrder::PRE_INVERSE_RATE;
    KFACOptimizer optimizer(registry(provider.parameterSchema()), control);
    const double rate = static_cast<double>(.05F);
    update(optimizer, provider, std::sqrt(.4));
    const double metric = rate * (rate / 2);
    CHECK(optimizer.constraintMetric() == Catch::Approx(metric).epsilon(1e-14));
    volatile float rounded = .05F * .05F;
    CHECK(optimizer.constraintMetric() != static_cast<double>(rounded) / 2);
    const double scale = bound ? std::sqrt(bound / metric) : 1;
    CHECK(optimizer.constraintScale() == Catch::Approx(scale).epsilon(1e-14));
    CHECK(provider.snapshotParameters().values[0] == Catch::Approx(1 - rate * scale / 2).epsilon(1e-14));
    CHECK(optimizer.effectiveLearningRate() == static_cast<double>(.049975015223026276F));
  }
}
}
