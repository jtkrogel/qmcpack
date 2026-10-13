//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include "DeterminantOverlapObjective.h"

#include "Message/CommOperators.h"
#include "Message/Communicate.h"
#include "Message/UniformCommunicateError.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace qmcplusplus::wftrain
{
namespace
{

void extendFingerprint(std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
  const auto* input = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < bytes; ++index)
  {
    hash ^= input[index];
    hash *= UINT64_C(1099511628211);
  }
}

std::uint64_t makeObjectiveFingerprint(
    const psiformer::DeterminantOverlapTargetDescriptor& target,
    const std::vector<double>& weights,
    const psiformer::DeterminantOverlapPolicy& policy) noexcept
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const std::uint64_t format_version = 1;
  const std::uint64_t target_fingerprint = target.fingerprint();
  const std::uint64_t channel_count = weights.size();
  extendFingerprint(hash, &format_version, sizeof(format_version));
  extendFingerprint(hash, &target_fingerprint, sizeof(target_fingerprint));
  extendFingerprint(hash, &channel_count, sizeof(channel_count));
  if (!weights.empty())
    extendFingerprint(hash, weights.data(), weights.size() * sizeof(double));
  extendFingerprint(hash, &policy.minimum_fidelity, sizeof(policy.minimum_fidelity));
  extendFingerprint(hash, &policy.minimum_effective_samples,
                    sizeof(policy.minimum_effective_samples));
  return hash;
}

/** Reduce signed/log sums with a global exponent shift and bounded O(M) storage. */
void reduceSignedLogVector(Communicate& communicator,
                           std::vector<psiformer::RealSignedLogAmplitude>& values)
{
  if (communicator.size() == 1 || values.empty())
    return;
#ifdef HAVE_MPI
  int invalid_extent =
      values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
  communicator.allreduce(invalid_extent);
  if (invalid_extent != 0)
    throw std::overflow_error("Determinant overlap MPI moment count exceeds INT_MAX");

  std::vector<double> shifts(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    shifts[index] = values[index].isZero()
        ? -std::numeric_limits<double>::infinity()
        : values[index].log_abs;
  const int count = static_cast<int>(values.size());
  if (MPI_Allreduce(MPI_IN_PLACE, shifts.data(), count, MPI_DOUBLE, MPI_MAX,
                    communicator.getMPI()) != MPI_SUCCESS)
    throw UniformCommunicateError("Determinant overlap MPI maximum reduction failed");

  std::vector<double> scaled(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index)
    if (!values[index].isZero())
      scaled[index] = values[index].phase *
          std::exp(values[index].log_abs - shifts[index]);
  if (MPI_Allreduce(MPI_IN_PLACE, scaled.data(), count, MPI_DOUBLE, MPI_SUM,
                    communicator.getMPI()) != MPI_SUCCESS)
    throw UniformCommunicateError("Determinant overlap MPI sum reduction failed");

  for (std::size_t index = 0; index < values.size(); ++index)
  {
    if (scaled[index] == 0.0)
      values[index] = {};
    else
      values[index] = {scaled[index] < 0.0 ? -1.0 : 1.0,
                       shifts[index] + std::log(std::abs(scaled[index]))};
  }
#else
  throw UniformCommunicateError(
      "Determinant overlap received a multi-rank communicator in a non-MPI build");
#endif
}

/** Reduce every moment family and the global sample count in place. */
void reduceMoments(Communicate& communicator,
                   psiformer::DeterminantOverlapMomentSnapshot& moments)
{
  reduceSignedLogVector(communicator, moments.cross);
  reduceSignedLogVector(communicator, moments.target_square);
  reduceSignedLogVector(communicator, moments.student_square);
  reduceSignedLogVector(communicator, moments.target_absolute);
  reduceSignedLogVector(communicator, moments.student_absolute);
  communicator.allreduce(moments.samples);
}

/** Check batch identities and return their common parameter and channel extents. */
std::pair<std::size_t, std::size_t> validateBatches(
    const std::vector<DeterminantOverlapBatchView>& batches,
    const psiformer::DeterminantOverlapTargetDescriptor& target,
    std::size_t& parameter_version,
    std::string& provider_id,
    std::string& schema_fingerprint)
{
  std::size_t local_samples = 0;
  std::size_t parameter_count = 0;
  for (const DeterminantOverlapBatchView& batch : batches)
  {
    if (!batch.student || batch.student->sampleCount() != batch.teachers.size() ||
        batch.student->sampleCount() != batch.electrons.size())
      throw std::invalid_argument("Determinant overlap batch has inconsistent sample ownership");
    if (batch.student->channelCount() != target.channelCount())
      throw std::invalid_argument("Determinant overlap student/target channel counts differ");
    if (parameter_count == 0)
    {
      parameter_count    = batch.student->parameterCount();
      parameter_version  = batch.student->parameterVersion();
      provider_id        = batch.student->providerId();
      schema_fingerprint = batch.student->schemaFingerprint();
    }
    else if (batch.student->parameterCount() != parameter_count ||
             batch.student->parameterVersion() != parameter_version ||
             batch.student->providerId() != provider_id ||
             batch.student->schemaFingerprint() != schema_fingerprint)
      throw std::invalid_argument("Determinant overlap student batches have inconsistent parameter identity");
    if (local_samples > std::numeric_limits<std::size_t>::max() - batch.student->sampleCount())
      throw std::overflow_error("Determinant overlap local sample count overflowed");
    local_samples += batch.student->sampleCount();
  }
  return {local_samples, parameter_count};
}

/** Synchronize a local exception before entering the next objective collective. */
void synchronizeFailure(Communicate& communicator,
                        const std::exception_ptr& local_failure,
                        const char* stage)
{
  int failed = local_failure ? 1 : 0;
  communicator.allreduce(failed);
  if (failed == 0)
    return;
  if (local_failure)
    std::rethrow_exception(local_failure);
  throw UniformCommunicateError(std::string("Another rank failed during determinant overlap ") + stage);
}

} // namespace

DeterminantOverlapObjective::DeterminantOverlapObjective(
    psiformer::DeterminantOverlapTargetEvaluator target,
    std::vector<double> objective_weights,
    psiformer::DeterminantOverlapPolicy policy)
    : target_(std::move(target)),
      objective_weights_(std::move(objective_weights)),
      policy_(policy)
{
  if (objective_weights_.size() != target_.descriptor().channelCount())
    throw std::invalid_argument("Determinant overlap objective weight count is incorrect");
  // Reuse the accumulator's strict policy and weight validation at construction.
  [[maybe_unused]] psiformer::DeterminantOverlapAccumulator validator(objective_weights_, policy_);
  configuration_fingerprint_ =
      makeObjectiveFingerprint(target_.descriptor(), objective_weights_, policy_);
}

DeterminantOverlapObjectiveResult DeterminantOverlapObjective::evaluate(
    Communicate& communicator,
    const std::vector<DeterminantOverlapBatchView>& batches) const
{
#ifdef HAVE_MPI
  if (communicator.size() > 1)
  {
    const std::array<unsigned long long, 2> local_identity{
        static_cast<unsigned long long>(target_.descriptor().fingerprint()),
        static_cast<unsigned long long>(configuration_fingerprint_)};
    auto minimum_identity = local_identity;
    auto maximum_identity = local_identity;
    constexpr int identity_fields = 2;
    if (MPI_Allreduce(MPI_IN_PLACE, minimum_identity.data(), identity_fields,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator.getMPI()) != MPI_SUCCESS ||
        MPI_Allreduce(MPI_IN_PLACE, maximum_identity.data(), identity_fields,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator.getMPI()) != MPI_SUCCESS)
      throw UniformCommunicateError("Determinant overlap objective-identity reduction failed");
    if (minimum_identity != maximum_identity)
      throw UniformCommunicateError(
          "Determinant overlap target or numerical policy differs between ranks");
  }
#endif

  std::size_t parameter_version = 0;
  std::string provider_id;
  std::string schema_fingerprint;
  std::size_t local_samples = 0;
  std::size_t parameter_count = 0;
  std::exception_ptr validation_failure;
  try
  {
    const auto extents = validateBatches(batches, target_.descriptor(), parameter_version,
                                         provider_id, schema_fingerprint);
    local_samples   = extents.first;
    parameter_count = extents.second;
  }
  catch (...)
  {
    validation_failure = std::current_exception();
  }
  synchronizeFailure(communicator, validation_failure, "batch validation");

  // Let zero-walker ranks allocate the same canonical P-vector.  Exact strings,
  // not hashes, are broadcast from the first populated rank and compared before
  // any subsequent rank can diverge into a different collective sequence.
#ifdef HAVE_MPI
  if (communicator.size() > 1)
  {
    int identity_root = local_samples == 0 ? communicator.size() : communicator.rank();
    if (MPI_Allreduce(MPI_IN_PLACE, &identity_root, 1, MPI_INT, MPI_MIN,
                      communicator.getMPI()) != MPI_SUCCESS)
      throw UniformCommunicateError("Determinant overlap identity-root reduction failed");
    if (identity_root < communicator.size())
    {
      unsigned long long common_parameter_count =
          static_cast<unsigned long long>(parameter_count);
      unsigned long long common_parameter_version =
          static_cast<unsigned long long>(parameter_version);
      if (MPI_Bcast(&common_parameter_count, 1, MPI_UNSIGNED_LONG_LONG, identity_root,
                    communicator.getMPI()) != MPI_SUCCESS ||
          MPI_Bcast(&common_parameter_version, 1, MPI_UNSIGNED_LONG_LONG, identity_root,
                    communicator.getMPI()) != MPI_SUCCESS)
        throw UniformCommunicateError("Determinant overlap extent broadcast failed");

      auto broadcast_string = [&](std::string& value) {
        unsigned long long length = static_cast<unsigned long long>(value.size());
        if (MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, identity_root,
                      communicator.getMPI()) != MPI_SUCCESS)
          throw UniformCommunicateError("Determinant overlap string-length broadcast failed");
        int invalid_length = length > static_cast<unsigned long long>(
                                          std::numeric_limits<int>::max())
            ? 1
            : 0;
        if (MPI_Allreduce(MPI_IN_PLACE, &invalid_length, 1, MPI_INT, MPI_MAX,
                          communicator.getMPI()) != MPI_SUCCESS)
          throw UniformCommunicateError(
              "Determinant overlap string-length validation failed");
        if (invalid_length != 0)
          throw UniformCommunicateError("Determinant overlap identity string exceeds INT_MAX");
        value.resize(static_cast<std::size_t>(length));
        if (length != 0 &&
            MPI_Bcast(value.data(), static_cast<int>(length), MPI_CHAR, identity_root,
                      communicator.getMPI()) != MPI_SUCCESS)
          throw UniformCommunicateError("Determinant overlap string broadcast failed");
      };

      std::string common_provider_id = provider_id;
      std::string common_schema_fingerprint = schema_fingerprint;
      broadcast_string(common_provider_id);
      broadcast_string(common_schema_fingerprint);

      int identity_mismatch = local_samples != 0 &&
              (parameter_count != common_parameter_count ||
               parameter_version != common_parameter_version ||
               provider_id != common_provider_id ||
               schema_fingerprint != common_schema_fingerprint)
          ? 1
          : 0;
      communicator.allreduce(identity_mismatch);
      if (identity_mismatch != 0)
        throw UniformCommunicateError(
            "Determinant overlap parameter identity differs between ranks");
      parameter_count   = static_cast<std::size_t>(common_parameter_count);
      parameter_version = static_cast<std::size_t>(common_parameter_version);
      provider_id       = std::move(common_provider_id);
      schema_fingerprint = std::move(common_schema_fingerprint);
    }
  }
#endif

  // Empty ranks are valid; global identity is supplied by a nonempty rank and checked.
  std::size_t global_samples = local_samples;
  communicator.allreduce(global_samples);
  if (global_samples == 0)
    throw UniformCommunicateError("Determinant overlap requires at least one global sample");

  // The first pass retains only five moment vectors and M-sized evaluation scratch.
  psiformer::DeterminantOverlapAccumulator accumulator(objective_weights_, policy_);
  std::vector<psiformer::RealSignedLogAmplitude> targets(target_.descriptor().channelCount());
  std::vector<psiformer::RealSignedLogAmplitude> students(target_.descriptor().channelCount());
  std::exception_ptr first_pass_failure;
  try
  {
    for (const DeterminantOverlapBatchView& batch : batches)
      for (std::size_t sample = 0; sample < batch.student->sampleCount(); ++sample)
      {
        psiformer::RealSignedLogAmplitude teacher;
        target_.evaluate(batch.teachers[sample], batch.electrons[sample], teacher,
                         targets.data(), targets.size());
        batch.student->evaluateAmplitudes(sample, students.data(), students.size());
        accumulator.addSample(teacher, targets.data(), students.data(), students.size());
      }
  }
  catch (...)
  {
    first_pass_failure = std::current_exception();
  }
  synchronizeFailure(communicator, first_pass_failure, "first pass");

  auto moments = accumulator.snapshotMoments();
  std::exception_ptr moment_failure;
  try
  {
    const std::size_t channels = target_.descriptor().channelCount();
    if (moments.cross.size() != channels || moments.target_square.size() != channels ||
        moments.student_square.size() != channels ||
        moments.target_absolute.size() != channels ||
        moments.student_absolute.size() != channels || moments.samples != local_samples)
      throw std::logic_error("Determinant overlap moment snapshot has an invalid shape");
    const auto validate_family = [](const auto& family) {
      for (const auto amplitude : family)
        psiformer::overlap_detail::validateAmplitude(amplitude, "moment");
    };
    validate_family(moments.cross);
    validate_family(moments.target_square);
    validate_family(moments.student_square);
    validate_family(moments.target_absolute);
    validate_family(moments.student_absolute);
  }
  catch (...)
  {
    moment_failure = std::current_exception();
  }
  synchronizeFailure(communicator, moment_failure, "moment preflight");
  reduceMoments(communicator, moments);
  accumulator.replaceMoments(moments);
  double loss = 0.0;
  std::exception_ptr finalization_failure;
  try
  {
    loss = accumulator.finalize();
  }
  catch (...)
  {
    finalization_failure = std::current_exception();
  }
  synchronizeFailure(communicator, finalization_failure, "moment finalization");

  // Recompute amplitudes and immediately contract M seeds through one P-vector reverse.
  std::vector<double> channel_weights(target_.descriptor().channelCount());
  std::vector<double> sample_gradient(parameter_count);
  std::vector<double> gradient(parameter_count, 0.0);
  std::size_t student_storage = 0;
  std::exception_ptr second_pass_failure;
  try
  {
    for (const DeterminantOverlapBatchView& batch : batches)
    {
      if (student_storage > std::numeric_limits<std::size_t>::max() -
              batch.student->retainedNumericBytes())
        throw std::overflow_error("Determinant overlap student storage count overflowed");
      student_storage += batch.student->retainedNumericBytes();
      for (std::size_t sample = 0; sample < batch.student->sampleCount(); ++sample)
      {
        psiformer::RealSignedLogAmplitude teacher;
        target_.evaluate(batch.teachers[sample], batch.electrons[sample], teacher,
                         targets.data(), targets.size());
        batch.student->evaluateAmplitudes(sample, students.data(), students.size());
        accumulator.channelScoreWeights(teacher, targets.data(), students.data(),
                                        students.size(), channel_weights.data());
        batch.student->evaluateWeightedScore(sample, channel_weights.data(),
                                             channel_weights.size(), sample_gradient.data(),
                                             sample_gradient.size());
        for (std::size_t parameter = 0; parameter < parameter_count; ++parameter)
          gradient[parameter] += sample_gradient[parameter];
      }
    }
  }
  catch (...)
  {
    second_pass_failure = std::current_exception();
  }
  synchronizeFailure(communicator, second_pass_failure, "second pass");

  std::exception_ptr finite_failure;
  try
  {
    for (double value : gradient)
      if (!psiformer::overlap_detail::isFinite(value))
        throw std::overflow_error("Determinant overlap local gradient is non-finite");
  }
  catch (...)
  {
    finite_failure = std::current_exception();
  }
  synchronizeFailure(communicator, finite_failure, "gradient validation");

  if (!gradient.empty())
    communicator.allreduce_in_place(gradient.data(), gradient.size());
  finite_failure = nullptr;
  try
  {
    for (double value : gradient)
      if (!psiformer::overlap_detail::isFinite(value))
        throw std::overflow_error("Determinant overlap global gradient is non-finite");
  }
  catch (...)
  {
    finite_failure = std::current_exception();
  }
  synchronizeFailure(communicator, finite_failure, "global gradient validation");

  DeterminantOverlapObjectiveResult result;
  result.loss                   = loss;
  result.sample_count           = moments.samples;
  result.parameter_version      = parameter_version;
  result.target_fingerprint     = target_.descriptor().fingerprint();
  result.target_provenance      = target_.descriptor().sourceIdentity();
  result.channels               = accumulator.diagnostics();
  result.parameter_gradient     = std::move(gradient);
  const auto checked_add = [](std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
      throw std::overflow_error("Determinant overlap retained-byte count overflowed");
    return left + right;
  };
  const auto checked_multiply = [](std::size_t count, std::size_t element_size) {
    if (element_size != 0 && count > std::numeric_limits<std::size_t>::max() / element_size)
      throw std::overflow_error("Determinant overlap retained-byte count overflowed");
    return count * element_size;
  };
  const std::size_t amplitude_capacity = checked_add(targets.capacity(), students.capacity());
  const std::size_t gradient_capacity = checked_add(
      checked_add(channel_weights.capacity(), sample_gradient.capacity()),
      result.parameter_gradient.capacity());
  result.retained_numeric_bytes = checked_add(
      checked_add(accumulator.retainedScalarBytes(),
                  checked_multiply(amplitude_capacity,
                                   sizeof(psiformer::RealSignedLogAmplitude))),
      checked_add(checked_multiply(gradient_capacity, sizeof(double)), student_storage));
  return result;
}

std::vector<double> makeEqualOverlapWeights(std::size_t channel_count)
{
  if (channel_count == 0)
    throw std::invalid_argument("Determinant overlap requires at least one channel");
  return std::vector<double>(channel_count, 1.0 / static_cast<double>(channel_count));
}

std::vector<double> makeCoefficientSquaredOverlapWeights(
    const psiformer::DeterminantOverlapTargetDescriptor& descriptor)
{
  if (descriptor.mode() !=
          psiformer::DeterminantOverlapTargetMode::TOP_COEFFICIENT_MULTIDETERMINANT ||
      descriptor.selectedCoefficients().size() != descriptor.channelCount())
    throw std::invalid_argument("Coefficient-squared overlap weights require paired CI targets");
  std::vector<double> weights(descriptor.channelCount());
  double maximum_coefficient = 0.0;
  for (const double coefficient : descriptor.selectedCoefficients())
  {
    if (!psiformer::overlap_detail::isFinite(coefficient))
      throw std::invalid_argument("Coefficient-squared overlap coefficient is non-finite");
    maximum_coefficient = std::max(maximum_coefficient, std::abs(coefficient));
  }
  if (!(maximum_coefficient > 0.0))
    throw std::invalid_argument("Coefficient-squared overlap coefficients are all zero");
  double normalization = 0.0;
  for (std::size_t channel = 0; channel < weights.size(); ++channel)
  {
    const double scaled = descriptor.selectedCoefficients()[channel] / maximum_coefficient;
    weights[channel] = scaled * scaled;
    normalization += weights[channel];
  }
  if (!psiformer::overlap_detail::isFinite(normalization) || normalization <= 0.0)
    throw std::invalid_argument("Coefficient-squared overlap normalization is invalid");
  for (double& weight : weights)
    weight /= normalization;
  return weights;
}

} // namespace qmcplusplus::wftrain
