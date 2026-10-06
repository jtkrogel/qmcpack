//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file benchmark_psiformer_optimization.cpp
 * @brief Short fixed-population smoke benchmark for high-parameter PsiFormer updates.
 *
 * This developer benchmark intentionally does not sample a training trajectory.  It
 * reuses a small exported walker population for a few conservative updates to expose
 * numerical failures, energy blow-ups, and the wall time of the production-neutral
 * Adam and matrix-free SR paths.  Its results are not convergence measurements.
 */

#include "Particle/ParticleSet.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"

#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace
{
using Clock = std::chrono::steady_clock;
using wftrain::DerivativeArrayView;
using wftrain::DerivativeReal;
using wftrain::DerivativeValue;

/// Own one HDF5 identifier and close it with the matching HDF5 function.
class H5Handle
{
public:
  using Closer = herr_t (*)(hid_t);

  H5Handle(hid_t id, Closer closer) : id_(id), closer_(closer)
  {
    if (id_ < 0)
      throw std::runtime_error("Failed to open required HDF5 object");
  }

  ~H5Handle()
  {
    if (id_ >= 0)
      closer_(id_);
  }

  H5Handle(const H5Handle&)            = delete;
  H5Handle& operator=(const H5Handle&) = delete;

  hid_t get() const noexcept { return id_; }

private:
  hid_t id_;
  Closer closer_;
};

/// Portable fixed-population geometry read from the existing PsiFormer export.
struct PopulationInput
{
  std::size_t sample_count   = 0;
  std::size_t electron_count = 0;
  std::size_t nucleus_count  = 0;
  int spin_up                = 0;
  int spin_down              = 0;
  std::vector<double> electrons;
  std::vector<double> nuclei;
  std::vector<double> charges;
};

/// Read a scalar integer attribute used to define the spin partition.
int readIntAttribute(hid_t file, const char* name)
{
  H5Handle attribute(H5Aopen(file, name, H5P_DEFAULT), H5Aclose);
  std::int64_t value = 0;
  if (H5Aread(attribute.get(), H5T_NATIVE_LLONG, &value) < 0 || value < 0 ||
      value > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("Invalid HDF5 attribute: ") + name);
  return static_cast<int>(value);
}

/// Read one floating-point dataset and return both its shape and flat values.
std::pair<std::vector<hsize_t>, std::vector<double>> readDoubleDataset(hid_t file,
                                                                       const char* name)
{
  H5Handle dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
  H5Handle space(H5Dget_space(dataset.get()), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank <= 0)
    throw std::runtime_error(std::string("Invalid HDF5 dataset rank: ") + name);
  std::vector<hsize_t> shape(static_cast<std::size_t>(rank));
  if (H5Sget_simple_extent_dims(space.get(), shape.data(), nullptr) < 0)
    throw std::runtime_error(std::string("Failed to read HDF5 shape: ") + name);
  std::size_t count = 1;
  for (const hsize_t extent : shape)
  {
    if (extent == 0 || extent > std::numeric_limits<std::size_t>::max() / count)
      throw std::runtime_error(std::string("Invalid HDF5 dataset extent: ") + name);
    count *= static_cast<std::size_t>(extent);
  }
  std::vector<double> values(count);
  if (H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              values.data()) < 0)
    throw std::runtime_error(std::string("Failed to read HDF5 dataset: ") + name);
  return {std::move(shape), std::move(values)};
}

/// Load and validate the open-boundary electron and nuclear coordinates.
PopulationInput readPopulationInput(const std::string& path)
{
  H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  auto [electron_shape, electrons] =
      readDoubleDataset(file.get(), "/electron_positions");
  auto [nucleus_shape, nuclei] =
      readDoubleDataset(file.get(), "/nuclear_positions");
  auto [charge_shape, charges] =
      readDoubleDataset(file.get(), "/nuclear_charges");
  if (electron_shape.size() != 3 || electron_shape[2] != 3 ||
      nucleus_shape.size() != 2 || nucleus_shape[1] != 3 ||
      charge_shape.size() != 1 || charge_shape[0] != nucleus_shape[0])
    throw std::runtime_error("PsiFormer optimization input has incompatible shapes");

  PopulationInput input;
  input.sample_count   = static_cast<std::size_t>(electron_shape[0]);
  input.electron_count = static_cast<std::size_t>(electron_shape[1]);
  input.nucleus_count  = static_cast<std::size_t>(nucleus_shape[0]);
  input.spin_up        = readIntAttribute(file.get(), "n_up");
  input.spin_down      = readIntAttribute(file.get(), "n_down");
  input.electrons      = std::move(electrons);
  input.nuclei         = std::move(nuclei);
  input.charges        = std::move(charges);
  if (input.sample_count == 0 || input.nucleus_count == 0 ||
      input.spin_up + input.spin_down != static_cast<int>(input.electron_count))
    throw std::runtime_error("PsiFormer optimization input has invalid population metadata");
  return input;
}

/// Return the Euclidean distance between two packed Cartesian positions.
double distance(const double* left, const double* right)
{
  double squared = 0.0;
  for (int dimension = 0; dimension < 3; ++dimension)
  {
    const double delta = left[dimension] - right[dimension];
    squared += delta * delta;
  }
  const double result = std::sqrt(squared);
  if (!(result > 0.0) || !std::isfinite(result))
    throw std::runtime_error("Coincident or non-finite particles in benchmark population");
  return result;
}

/// Compute the straight all-electron Coulomb potential for one configuration.
double coulombPotential(const PopulationInput& input, std::size_t sample)
{
  const double* electrons = input.electrons.data() + sample * input.electron_count * 3;
  double potential        = 0.0;
  for (std::size_t first = 0; first < input.electron_count; ++first)
    for (std::size_t second = first + 1; second < input.electron_count; ++second)
      potential += 1.0 / distance(electrons + 3 * first, electrons + 3 * second);
  for (std::size_t electron = 0; electron < input.electron_count; ++electron)
    for (std::size_t nucleus = 0; nucleus < input.nucleus_count; ++nucleus)
      potential -= input.charges[nucleus] /
          distance(electrons + 3 * electron, input.nuclei.data() + 3 * nucleus);
  for (std::size_t first = 0; first < input.nucleus_count; ++first)
    for (std::size_t second = first + 1; second < input.nucleus_count; ++second)
      potential += input.charges[first] * input.charges[second] /
          distance(input.nuclei.data() + 3 * first, input.nuclei.data() + 3 * second);
  return potential;
}

/// Build one electron ParticleSet with initialized unit masses and copied positions.
std::unique_ptr<ParticleSet> makeParticleSet(const SimulationCell& cell,
                                             const PopulationInput& input,
                                             std::size_t sample)
{
  auto particles = std::make_unique<ParticleSet>(cell);
  particles->setName("e");
  particles->create({input.spin_up, input.spin_down});
  SpeciesSet& species = particles->getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  const int charge = species.addAttribute("charge");
  species(charge, 0) = -1.0;
  species(charge, 1) = -1.0;
  particles->resetGroups();
  const double* positions =
      input.electrons.data() + sample * input.electron_count * 3;
  for (std::size_t electron = 0; electron < input.electron_count; ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      particles->R[electron][dimension] = positions[3 * electron + dimension];
  particles->update();
  return particles;
}

/// Scalar population statistics evaluated at one immutable parameter version.
struct EnergyStatistics
{
  double mean     = 0.0;
  double variance = 0.0;
};

/** Own one shared PsiFormer clone family and its fixed walker population.
 *
 * The family is rebuilt independently for each optimizer so both methods start
 * from the exact same portable 50k parameters.
 */
class FixedPopulation
{
public:
  FixedPopulation(const std::string& parameter_path,
                  const std::string& configuration_path,
                  PopulationInput input)
      : input_(std::move(input)), potentials_(input_.sample_count), cell_()
  {
    components_.push_back(std::make_unique<PsiFormerWF>(
        "pf_optimization_benchmark", parameter_path, configuration_path));
    for (std::size_t sample = 1; sample < input_.sample_count; ++sample)
      components_.push_back(std::make_unique<PsiFormerWF>(*components_.front()));
    for (std::size_t sample = 0; sample < input_.sample_count; ++sample)
    {
      particles_.push_back(makeParticleSet(cell_, input_, sample));
      potentials_[sample] = coulombPotential(input_, sample);
    }
  }

  PsiFormerWF& leader() noexcept { return *components_.front(); }

  const wftrain::StructuredParameterSchema& schema() const noexcept
  {
    return components_.front()->parameterSchema();
  }

  /// Refresh component drifts and return total local energies for all samples.
  std::vector<DerivativeValue> evaluateLocalEnergies()
  {
    std::vector<DerivativeValue> energies(input_.sample_count);
    for (std::size_t sample = 0; sample < input_.sample_count; ++sample)
    {
      ParticleSet& particles = *particles_[sample];
      particles.G = QMCTraits::GradType{};
      particles.L = QMCTraits::ValueType{};
      components_[sample]->evaluateLog(particles, particles.G, particles.L);
      double laplacian_ratio = 0.0;
      for (std::size_t electron = 0; electron < input_.electron_count; ++electron)
      {
        laplacian_ratio += std::real(particles.L[electron]);
        for (int dimension = 0; dimension < 3; ++dimension)
          laplacian_ratio += std::norm(std::complex<double>(particles.G[electron][dimension]));
      }
      energies[sample] = {-0.5 * laplacian_ratio + potentials_[sample], 0.0};
      if (!std::isfinite(energies[sample].real()))
        throw std::runtime_error("PsiFormer benchmark produced a non-finite local energy");
    }
    return energies;
  }

  /// Construct a version-bound O(P)+O(B) derivative operator from refreshed samples.
  std::unique_ptr<wftrain::StreamingDerivativeOperator> makeDerivativeOperator(
      std::size_t batch_ordinal,
      std::size_t maximum_chunk_size)
  {
    RefVectorWithLeader<WaveFunctionComponent> component_refs(*components_.front());
    RefVectorWithLeader<ParticleSet> particle_refs(*particles_.front());
    for (std::size_t sample = 0; sample < input_.sample_count; ++sample)
    {
      component_refs.push_back(*components_[sample]);
      particle_refs.push_back(*particles_[sample]);
    }
    return components_.front()->makeStreamingDerivativeOperator(
        component_refs, particle_refs, batch_ordinal, 0, maximum_chunk_size);
  }

  EnergyStatistics energyStatistics()
  {
    const std::vector<DerivativeValue> energies = evaluateLocalEnergies();
    const double mean = std::accumulate(
                            energies.begin(), energies.end(), 0.0,
                            [](double sum, DerivativeValue value) {
                              return sum + value.real();
                            }) /
        static_cast<double>(energies.size());
    double variance = 0.0;
    for (const DerivativeValue value : energies)
    {
      const double delta = value.real() - mean;
      variance += delta * delta;
    }
    variance /= static_cast<double>(energies.size());
    return {mean, variance};
  }

  std::size_t sampleCount() const noexcept { return input_.sample_count; }

private:
  PopulationInput input_;
  std::vector<double> potentials_;
  SimulationCell cell_;
  std::vector<std::unique_ptr<PsiFormerWF>> components_;
  std::vector<std::unique_ptr<ParticleSet>> particles_;
};

/** Adapt one prepared PsiFormer operator and fixed population to the coordinator.
 *
 * The derivative operator is prepared explicitly at each version so SR can reuse
 * the identical owner for its covariance action during the same iteration.
 */
class FixedPopulationGradientProducer final : public wftrain::GradientProducer
{
public:
  explicit FixedPopulationGradientProducer(FixedPopulation& population)
      : population_(population), weights_(population.sampleCount(), 1.0)
  {}

  wftrain::TrainingCapabilities capabilities() const noexcept override
  {
    return {wftrain::TrainingCapability::REAL_PARAMETERS,
            wftrain::TrainingCapability::SCORE_VJP,
            wftrain::TrainingCapability::SCORE_JVP,
            wftrain::TrainingCapability::LOCAL_ENERGY_VJP};
  }

  /// Refresh local observables and bind a new derivative owner to the live version.
  void prepare(std::size_t batch_ordinal)
  {
    energies_ = population_.evaluateLocalEnergies();
    derivative_operator_ =
        population_.makeDerivativeOperator(batch_ordinal, maximum_chunk_size_);
  }

  void accumulate(const wftrain::StructuredParameterSnapshot& parameters,
                  wftrain::EnergyGradientAccumulator& accumulator) override
  {
    if (!derivative_operator_ ||
        derivative_operator_->parameterVersion() != parameters.version)
      throw std::logic_error("PsiFormer benchmark producer was not prepared for this version");
    wftrain::accumulateEnergyGradientBatch(
        *derivative_operator_, {weights_.data(), weights_.size()},
        {energies_.data(), energies_.size()},
        wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::KINETIC), accumulator);
  }

  const wftrain::StreamingDerivativeOperator& derivativeOperator() const
  {
    if (!derivative_operator_)
      throw std::logic_error("PsiFormer benchmark derivative operator is not prepared");
    return *derivative_operator_;
  }

  DerivativeArrayView<const DerivativeReal> weights() const noexcept
  {
    return {weights_.data(), weights_.size()};
  }

  std::size_t retainedDerivativeBytes() const
  {
    return derivativeOperator().storageDiagnostics().retained_numeric_bytes;
  }

private:
  static constexpr std::size_t maximum_chunk_size_ = 65536;
  FixedPopulation& population_;
  std::vector<DerivativeReal> weights_;
  std::vector<DerivativeValue> energies_;
  std::unique_ptr<wftrain::StreamingDerivativeOperator> derivative_operator_;
};

/// One accepted update and its objective/solver/resource diagnostics.
struct IterationRecord
{
  std::size_t iteration             = 0;
  std::size_t parameter_version     = 0;
  double seconds                    = 0.0;
  double mean_energy                = 0.0;
  double energy_variance            = 0.0;
  double gradient_norm              = 0.0;
  double update_norm                = 0.0;
  std::size_t derivative_bytes      = 0;
  std::size_t krylov_iterations     = 0;
  std::size_t operator_applications = 0;
  double krylov_relative_residual   = 0.0;
  double damping                     = 0.0;
  double applied_scale               = 1.0;
};

/// Aggregate records for one optimizer beginning from an independent model reload.
struct MethodResult
{
  std::string name;
  std::size_t parameter_count = 0;
  std::vector<IterationRecord> iterations;
  EnergyStatistics final_energy;
  double total_seconds = 0.0;
  double maximum_energy_increase = 0.0;
  bool stable = false;
};

/// Compute a Euclidean norm without retaining another parameter-sized vector.
double realNorm(const std::vector<double>& values)
{
  long double norm_squared = 0.0;
  for (const double value : values)
    norm_squared += static_cast<long double>(value) * value;
  return std::sqrt(static_cast<double>(norm_squared));
}

/// Compute the norm of a published parameter difference.
double updateNorm(const wftrain::StructuredParameterSnapshot& before,
                  const wftrain::StructuredParameterSnapshot& after)
{
  if (before.values.size() != after.values.size())
    throw std::logic_error("PsiFormer benchmark snapshots have different extents");
  long double norm_squared = 0.0;
  for (std::size_t index = 0; index < before.values.size(); ++index)
  {
    const long double delta = after.values[index] - before.values[index];
    norm_squared += delta * delta;
  }
  return std::sqrt(static_cast<double>(norm_squared));
}

/// Complete the common final energy and loose no-blow-up assessment.
void finalizeMethod(MethodResult& result, FixedPopulation& population)
{
  result.final_energy = population.energyStatistics();
  const double initial_energy = result.iterations.front().mean_energy;
  result.maximum_energy_increase =
      std::max(0.0, result.final_energy.mean - initial_energy);
  for (const IterationRecord& record : result.iterations)
    result.maximum_energy_increase = std::max(
        result.maximum_energy_increase, record.mean_energy - initial_energy);
  result.stable = std::isfinite(result.final_energy.mean) &&
      std::isfinite(result.final_energy.variance) &&
      result.maximum_energy_increase <= 1.0;
}

/// Run a few persistent-state Adam updates through the atomic coordinator.
MethodResult runAdam(const std::string& parameter_path,
                     const std::string& configuration_path,
                     const PopulationInput& input,
                     std::size_t iteration_count)
{
  FixedPopulation population(parameter_path, configuration_path, input);
  FixedPopulationGradientProducer producer(population);
  wftrain::FirstOrderOptimizerOptions options;
  options.method = wftrain::FirstOrderMethod::ADAM;
  options.learning_rates = {{"neural_network", 1.0e-6}};
  wftrain::FirstOrderOptimizer optimizer(population.schema(), options);
  wftrain::HighParameterTraining training({});
  wftrain::TrainingIterationState state;
  MethodResult result;
  result.name            = "adam";
  result.parameter_count = population.schema().parameterCount();

  for (std::size_t iteration = 0; iteration < iteration_count; ++iteration)
  {
    const auto start = Clock::now();
    producer.prepare(iteration);
    const wftrain::StructuredParameterSnapshot before =
        population.leader().snapshotParameters();
    wftrain::TrainingIterationResult training_result =
        training.runIteration(population.leader(), producer, optimizer, state);
    const wftrain::StructuredParameterSnapshot after =
        population.leader().snapshotParameters();
    if (after.version != before.version + 1 ||
        training_result.parameter_version != after.version ||
        state.completed_iterations != iteration + 1)
      throw std::runtime_error("Adam benchmark observed an invalid committed version");
    IterationRecord record;
    record.iteration         = iteration + 1;
    record.parameter_version = training_result.parameter_version;
    record.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    record.mean_energy       = training_result.objective.mean_energy.real();
    record.energy_variance   = training_result.objective.energy_variance;
    record.gradient_norm     = realNorm(training_result.objective.gradient);
    record.update_norm       = updateNorm(before, after);
    record.derivative_bytes  = producer.retainedDerivativeBytes();
    result.total_seconds += record.seconds;
    result.iterations.push_back(record);
  }
  finalizeMethod(result, population);
  return result;
}

/// Run a few version-local damped matrix-free SR updates through the coordinator.
MethodResult runSR(const std::string& parameter_path,
                   const std::string& configuration_path,
                   const PopulationInput& input,
                   std::size_t iteration_count)
{
  FixedPopulation population(parameter_path, configuration_path, input);
  FixedPopulationGradientProducer producer(population);
  wftrain::HighParameterTraining training({});
  wftrain::TrainingIterationState state;
  MethodResult result;
  result.name            = "matrix_free_sr";
  result.parameter_count = population.schema().parameterCount();

  for (std::size_t iteration = 0; iteration < iteration_count; ++iteration)
  {
    const auto start = Clock::now();
    producer.prepare(iteration);
    const wftrain::StructuredParameterSnapshot before =
        population.leader().snapshotParameters();
    const wftrain::StochasticReconfigurationBatch batch{
        &producer.derivativeOperator(), producer.weights()};
    const DerivativeArrayView<const wftrain::StochasticReconfigurationBatch> batches{
        &batch, 1};
    wftrain::StochasticReconfigurationOperator covariance(
        population.schema(), before.version, batches);
    wftrain::IdentityPreconditioner preconditioner(population.schema(), before.version);
    wftrain::StochasticReconfigurationUpdateControl control;
    control.learning_rate                = 1.0e-2;
    control.initial_damping              = 1.0e-2;
    control.maximum_damping_attempts     = 2;
    control.maximum_update_norm          = 1.0e-3;
    control.krylov.relative_tolerance    = 1.0e-4;
    control.krylov.maximum_iterations    = 16;
    control.krylov.maximum_operator_applications = 17;
    control.krylov.maximum_seconds       = 120.0;
    wftrain::StochasticReconfigurationUpdateRule update_rule(
        covariance, preconditioner, control);
    wftrain::TrainingIterationResult training_result =
        training.runIteration(population.leader(), producer, update_rule, state);
    const wftrain::StructuredParameterSnapshot after =
        population.leader().snapshotParameters();
    if (after.version != before.version + 1 ||
        training_result.parameter_version != after.version ||
        state.completed_iterations != iteration + 1)
      throw std::runtime_error("SR benchmark observed an invalid committed version");

    IterationRecord record;
    record.iteration         = iteration + 1;
    record.parameter_version = training_result.parameter_version;
    record.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    record.mean_energy       = training_result.objective.mean_energy.real();
    record.energy_variance   = training_result.objective.energy_variance;
    record.gradient_norm     = realNorm(training_result.objective.gradient);
    record.update_norm       = updateNorm(before, after);
    record.derivative_bytes  = producer.retainedDerivativeBytes();
    const auto& diagnostics = update_rule.lastDiagnostics();
    if (!diagnostics || !diagnostics->solve.converged)
      throw std::runtime_error("Matrix-free SR did not publish a converged update");
    record.krylov_iterations        = diagnostics->solve.iterations;
    record.operator_applications    = diagnostics->solve.operator_applications;
    record.krylov_relative_residual = diagnostics->solve.relative_residual_norm;
    record.damping                  = diagnostics->damping;
    record.applied_scale            = diagnostics->applied_scale;
    result.total_seconds += record.seconds;
    result.iterations.push_back(record);
  }
  finalizeMethod(result, population);
  return result;
}

/// Escape arbitrary byte strings without emitting raw JSON control characters.
std::string jsonString(const std::string& value)
{
  static constexpr char hex_digits[] = "0123456789abcdef";
  std::string result = "\"";
  for (const unsigned char character : value)
  {
    switch (character)
    {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character < 0x20)
      {
        result += "\\u00";
        result += hex_digits[character >> 4];
        result += hex_digits[character & 0x0f];
      }
      else
        result += static_cast<char>(character);
    }
  }
  return result + '"';
}

/// Reject values that the JSON number grammar cannot represent.
void requireFiniteResult(const MethodResult& result)
{
  const auto require_finite = [&](double value, const char* field) {
    if (!std::isfinite(value))
      throw std::runtime_error(result.name + " produced a non-finite " + field);
  };
  require_finite(result.total_seconds, "total time");
  require_finite(result.final_energy.mean, "final mean energy");
  require_finite(result.final_energy.variance, "final energy variance");
  require_finite(result.maximum_energy_increase, "maximum energy increase");
  for (const IterationRecord& record : result.iterations)
  {
    require_finite(record.seconds, "iteration time");
    require_finite(record.mean_energy, "iteration mean energy");
    require_finite(record.energy_variance, "iteration energy variance");
    require_finite(record.gradient_norm, "gradient norm");
    require_finite(record.update_norm, "update norm");
    require_finite(record.krylov_relative_residual, "Krylov relative residual");
    require_finite(record.damping, "damping");
    require_finite(record.applied_scale, "applied scale");
  }
}

/// Emit one optimizer result without external JSON dependencies.
void writeMethod(std::ostream& output, const MethodResult& result)
{
  output << "{\"name\":" << jsonString(result.name)
         << ",\"total_seconds\":" << result.total_seconds
         << ",\"final_mean_energy\":" << result.final_energy.mean
         << ",\"final_energy_variance\":" << result.final_energy.variance
         << ",\"maximum_energy_increase\":" << result.maximum_energy_increase
         << ",\"stable\":" << (result.stable ? "true" : "false")
         << ",\"iterations\":[";
  for (std::size_t index = 0; index < result.iterations.size(); ++index)
  {
    if (index)
      output << ',';
    const IterationRecord& record = result.iterations[index];
    output << "{\"iteration\":" << record.iteration
           << ",\"parameter_version\":" << record.parameter_version
           << ",\"seconds\":" << record.seconds
           << ",\"mean_energy\":" << record.mean_energy
           << ",\"energy_variance\":" << record.energy_variance
           << ",\"gradient_norm\":" << record.gradient_norm
           << ",\"update_norm\":" << record.update_norm
           << ",\"derivative_retained_bytes\":" << record.derivative_bytes
           << ",\"krylov_iterations\":" << record.krylov_iterations
           << ",\"operator_applications\":" << record.operator_applications
           << ",\"krylov_relative_residual\":" << record.krylov_relative_residual
           << ",\"damping\":" << record.damping
           << ",\"applied_scale\":" << record.applied_scale << '}';
  }
  output << "]}";
}

} // namespace
} // namespace qmcplusplus

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5)
  {
    std::cerr << "usage: benchmark_psiformer_optimization PARAMETERS CONFIGURATIONS "
                 "OUTPUT_JSON [ITERATIONS]\n";
    return 1;
  }

  try
  {
    const std::string parameter_path     = argv[1];
    const std::string configuration_path = argv[2];
    const std::string output_path        = argv[3];
    const std::size_t iteration_count =
        argc == 5 ? static_cast<std::size_t>(std::stoul(argv[4])) : 3;
    if (iteration_count == 0 || iteration_count > 5)
      throw std::invalid_argument("Iteration count must be between one and five");

    const qmcplusplus::PopulationInput input =
        qmcplusplus::readPopulationInput(configuration_path);
    const qmcplusplus::MethodResult adam = qmcplusplus::runAdam(
        parameter_path, configuration_path, input, iteration_count);
    const qmcplusplus::MethodResult sr = qmcplusplus::runSR(
        parameter_path, configuration_path, input, iteration_count);
    if (adam.parameter_count != sr.parameter_count)
      throw std::logic_error("Independent optimizer models exposed different schemas");
    qmcplusplus::requireFiniteResult(adam);
    qmcplusplus::requireFiniteResult(sr);

    std::ofstream output(output_path);
    if (!output)
      throw std::runtime_error("Could not open benchmark JSON output");
    output << std::setprecision(17)
           << "{\n  \"schema\":\"qmcpack.psiformer.optimization_smoke.v1\",\n"
           << "  \"completed\":true,\n"
           << "  \"parameter_file\":" << qmcplusplus::jsonString(parameter_path) << ",\n"
           << "  \"configuration_file\":" << qmcplusplus::jsonString(configuration_path) << ",\n"
           << "  \"parameter_count\":" << adam.parameter_count
           << ",\n  \"sample_count\":" << input.sample_count
           << ",\n  \"updates_per_method\":" << iteration_count
           << ",\n  \"population_reuse\":\"fixed checkpoint walkers; no MCMC resampling\",\n"
           << "  \"stability_threshold_hartree\":1.0,\n"
           << "  \"methods\":[\n    ";
    qmcplusplus::writeMethod(output, adam);
    output << ",\n    ";
    qmcplusplus::writeMethod(output, sr);
    output << "\n  ],\n  \"all_stable\":"
           << (adam.stable && sr.stable ? "true" : "false") << "\n}\n";
    output.flush();
    if (!output)
      throw std::runtime_error("Could not write complete benchmark JSON output");

    std::cout << std::setprecision(10)
              << "Adam final E=" << adam.final_energy.mean
              << " max rise=" << adam.maximum_energy_increase
              << " seconds=" << adam.total_seconds << '\n'
              << "SR final E=" << sr.final_energy.mean
              << " max rise=" << sr.maximum_energy_increase
              << " seconds=" << sr.total_seconds << '\n'
              << "Wrote " << output_path << '\n';
    return adam.stable && sr.stable ? 0 : 2;
  }
  catch (const std::exception& error)
  {
    // Preserve a compact machine-readable failure record even when preflight,
    // numerical, version, or solver validation prevents a complete comparison.
    if (argc >= 4)
    {
      std::ofstream output(argv[3]);
      if (output)
      {
        output << "{\n  \"schema\":\"qmcpack.psiformer.optimization_smoke.v1\",\n"
               << "  \"completed\":false,\n  \"error\":"
               << qmcplusplus::jsonString(error.what()) << "\n}\n";
        output.flush();
      }
    }
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
