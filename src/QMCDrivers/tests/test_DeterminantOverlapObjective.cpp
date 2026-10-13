//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_DeterminantOverlapObjective.cpp
 * @brief Deterministic serial/MPI application tests for two-pass overlap training.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Message/Communicate.h"
#include "Message/CommOperators.h"
#include "Particle/ParticleSet.h"
#include "QMCDrivers/WFTrain/DeterminantOverlapObjective.h"
#include "QMCWaveFunctions/ConstantOrbital.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "SimulationCell.h"
#include "Utilities/RuntimeOptions.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
namespace
{

using Catch::Approx;

/** Deterministic channel model whose log amplitudes are affine in two parameters. */
class AffineChannelBatch final : public psiformer::DeterminantChannelBatch
{
public:
  explicit AffineChannelBatch(std::vector<std::size_t> sample_ids)
      : sample_ids_(std::move(sample_ids))
  {}

  const std::string& providerId() const noexcept override { return provider_id_; }
  const std::string& schemaFingerprint() const noexcept override { return schema_; }
  std::size_t parameterVersion() const noexcept override { return 7; }
  std::size_t sampleCount() const noexcept override { return sample_ids_.size(); }
  std::size_t channelCount() const noexcept override { return 2; }
  std::size_t parameterCount() const noexcept override { return 2; }

  void evaluateAmplitudes(std::size_t sample,
                          psiformer::RealSignedLogAmplitude* output,
                          std::size_t output_count) const override
  {
    REQUIRE(sample < sample_ids_.size());
    REQUIRE(output_count == 2);
    const std::size_t id = sample_ids_[sample];
    for (std::size_t channel = 0; channel < 2; ++channel)
    {
      const auto score = scoreRow(id, channel);
      output[channel] = {1.0, baseLogAmplitude(id, channel) +
                         score[0] * parameters_[0] + score[1] * parameters_[1]};
    }
    ++amplitude_calls_;
  }

  void evaluateWeightedScore(std::size_t sample,
                             const double* channel_weights,
                             std::size_t channel_count,
                             double* output,
                             std::size_t output_count) const override
  {
    REQUIRE(sample < sample_ids_.size());
    REQUIRE(channel_count == 2);
    REQUIRE(output_count == 2);
    output[0] = 0.0;
    output[1] = 0.0;
    for (std::size_t channel = 0; channel < 2; ++channel)
    {
      const auto score = scoreRow(sample_ids_[sample], channel);
      output[0] += channel_weights[channel] * score[0];
      output[1] += channel_weights[channel] * score[1];
    }
    ++reverse_calls_;
  }

  std::size_t retainedNumericBytes() const noexcept override
  {
    return sample_ids_.capacity() * sizeof(std::size_t) + 2 * sizeof(double);
  }

  std::size_t amplitudeCalls() const noexcept { return amplitude_calls_; }
  std::size_t reverseCalls() const noexcept { return reverse_calls_; }

  static double baseLogAmplitude(std::size_t sample, std::size_t channel)
  {
    return 0.08 * static_cast<double>(sample + 1) *
        (channel == 0 ? 1.0 : -0.6);
  }

  static std::array<double, 2> scoreRow(std::size_t sample, std::size_t channel)
  {
    const double x = static_cast<double>(sample + 1);
    return channel == 0 ? std::array<double, 2>{0.03 * x, -0.02}
                        : std::array<double, 2>{-0.01, 0.025 * x};
  }

private:
  std::vector<std::size_t> sample_ids_;
  const std::array<double, 2> parameters_{0.4, -0.3};
  mutable std::size_t amplitude_calls_ = 0;
  mutable std::size_t reverse_calls_ = 0;
  const std::string provider_id_ = "overlap/test";
  const std::string schema_ = "overlap-test-schema";
};

/** Return an uneven deterministic partition with one zero-sample rank. */
std::vector<std::size_t> localSampleIds(int rank, int ranks)
{
  if (ranks > 1 && rank == 1)
    return {};
  std::vector<std::size_t> ids;
  std::size_t offset = 0;
  for (int prior = 0; prior < rank; ++prior)
    if (!(ranks > 1 && prior == 1))
      offset += static_cast<std::size_t>(prior + 1);
  ids.resize(static_cast<std::size_t>(rank + 1));
  for (std::size_t sample = 0; sample < ids.size(); ++sample)
    ids[sample] = offset + sample;
  return ids;
}

/** Return the total sample count produced by the deterministic partition. */
std::size_t globalSampleCount(int ranks)
{
  std::size_t count = 0;
  for (int rank = 0; rank < ranks; ++rank)
    count += localSampleIds(rank, ranks).size();
  return count;
}

/** Construct constant frozen teachers and their one-electron configurations. */
void makeTeachers(std::size_t count,
                  std::vector<std::unique_ptr<ParticleSet>>& particles,
                  std::vector<std::unique_ptr<TrialWaveFunction>>& teachers)
{
  static const SimulationCell cell;
  RuntimeOptions runtime_options;
  for (std::size_t sample = 0; sample < count; ++sample)
  {
    auto electrons = std::make_unique<ParticleSet>(cell);
    electrons->create({1});
    electrons->R[0] = {0.1 * static_cast<double>(sample + 1), 0.0, 0.0};
    auto teacher = std::make_unique<TrialWaveFunction>(runtime_options, "teacher");
    teacher->addComponent(std::make_unique<ConstantOrbital>());
    particles.push_back(std::move(electrons));
    teachers.push_back(std::move(teacher));
  }
}

/** Compute the dense deterministic oracle directly from the public moment algebra. */
std::pair<double, std::array<double, 2>> makeOracle(std::size_t sample_count)
{
  psiformer::DeterminantOverlapAccumulator accumulator({0.5, 0.5});
  std::array<psiformer::RealSignedLogAmplitude, 2> target{{{1.0, 0.0}, {1.0, 0.0}}};
  std::array<psiformer::RealSignedLogAmplitude, 2> student;
  constexpr std::array<double, 2> parameters{0.4, -0.3};
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    for (std::size_t channel = 0; channel < 2; ++channel)
    {
      const auto score = AffineChannelBatch::scoreRow(sample, channel);
      student[channel] = {1.0, AffineChannelBatch::baseLogAmplitude(sample, channel) +
                          score[0] * parameters[0] + score[1] * parameters[1]};
    }
    accumulator.addSample({1.0, 0.0}, target.data(), student.data(), 2);
  }
  const double loss = accumulator.finalize();
  std::array<double, 2> gradient{};
  std::array<double, 2> weights{};
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    for (std::size_t channel = 0; channel < 2; ++channel)
    {
      const auto score = AffineChannelBatch::scoreRow(sample, channel);
      student[channel] = {1.0, AffineChannelBatch::baseLogAmplitude(sample, channel) +
                          score[0] * parameters[0] + score[1] * parameters[1]};
    }
    accumulator.channelScoreWeights({1.0, 0.0}, target.data(), student.data(), 2,
                                    weights.data());
    for (std::size_t channel = 0; channel < 2; ++channel)
    {
      const auto score = AffineChannelBatch::scoreRow(sample, channel);
      gradient[0] += weights[channel] * score[0];
      gradient[1] += weights[channel] * score[1];
    }
  }
  return {loss, gradient};
}

} // namespace

TEST_CASE("Determinant overlap applies two bounded passes on uneven MPI ranks",
          "[drivers][wf_train][overlap][mpi]")
{
#if defined(QMC_COMPLEX)
  SUCCEED("The determinant-overlap production objective is intentionally real-only");
#else
  Communicate& communicator = *OHMMS::Controller;
  const std::vector<std::size_t> sample_ids =
      localSampleIds(communicator.rank(), communicator.size());
  AffineChannelBatch student(sample_ids);
  std::vector<std::unique_ptr<ParticleSet>> particle_owners;
  std::vector<std::unique_ptr<TrialWaveFunction>> teacher_owners;
  makeTeachers(sample_ids.size(), particle_owners, teacher_owners);

  std::vector<DeterminantOverlapBatchView> batches;
  if (!sample_ids.empty())
  {
    DeterminantOverlapBatchView batch;
    batch.student = &student;
    for (std::size_t sample = 0; sample < sample_ids.size(); ++sample)
    {
      batch.teachers.emplace_back(*teacher_owners[sample]);
      batch.electrons.emplace_back(*particle_owners[sample]);
    }
    batches.push_back(std::move(batch));
  }

  const auto descriptor =
      psiformer::DeterminantOverlapTargetDescriptor::makeGeneralFrozenTeacher(
          "constant-teacher-v1", 2);
  DeterminantOverlapObjective objective(
      psiformer::DeterminantOverlapTargetEvaluator(descriptor), {0.5, 0.5});
  const DeterminantOverlapObjectiveResult result = objective.evaluate(communicator, batches);
  const auto oracle = makeOracle(globalSampleCount(communicator.size()));

  CHECK(result.loss == Approx(oracle.first).margin(2.0e-13));
  REQUIRE(result.parameter_gradient.size() == 2);
  CHECK(result.parameter_gradient[0] == Approx(oracle.second[0]).margin(2.0e-13));
  CHECK(result.parameter_gradient[1] == Approx(oracle.second[1]).margin(2.0e-13));
  CHECK(result.sample_count == globalSampleCount(communicator.size()));
  CHECK(result.parameter_version == 7);
  CHECK(result.target_fingerprint == descriptor.fingerprint());
  CHECK(result.target_provenance == "constant-teacher-v1");
  CHECK(result.channels.size() == 2);
  CHECK(student.amplitudeCalls() == 2 * sample_ids.size());
  CHECK(student.reverseCalls() == sample_ids.size());
  const auto loss_only = objective.evaluate(communicator, batches, false);
  CHECK(loss_only.loss == Approx(result.loss).margin(2.0e-13));
  CHECK(loss_only.parameter_gradient.empty());
  CHECK(loss_only.sample_loss_influences.size() == sample_ids.size());
  CHECK(student.reverseCalls() == sample_ids.size());
  double influence_sum = 0.0;
  for (double influence : loss_only.sample_loss_influences)
    influence_sum += influence;
  communicator.allreduce(influence_sum);
  CHECK(influence_sum == Approx(0.0).margin(2.0e-13));
#endif
}

} // namespace qmcplusplus::wftrain
