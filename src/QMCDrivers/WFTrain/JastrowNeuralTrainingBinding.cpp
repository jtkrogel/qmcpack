//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "JastrowNeuralTrainingBinding.h"
#include "HighParameterTraining.h"
#include "TrainingCheckpoint.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"
#include "io/hdf/hdf_archive.h"
#include "Message/CommOperators.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerResidualPair.h"
#include "QMCWaveFunctions/Jastrow/J1OrbitalSoA.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "OhmmsData/Libxml2Doc.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>
#include <functional>
#include <iomanip>
#include <map>

namespace qmcplusplus::wftrain
{
/// One scratch bank is shared by all retained batches, never one P-row per sample.
struct JastrowNeuralScoreWorkspace
{
  explicit JastrowNeuralScoreWorkspace(std::size_t count) : row(count), channels(3, std::vector<DerivativeValue>(count)) {}
  std::mutex mutex;
  std::vector<double> row;
  std::vector<std::vector<DerivativeValue>> channels;
};

/// One Jastrow-only replay bank reuses distance tables and component derivative caches.
struct JastrowNeuralReplayLane
{
  JastrowNeuralReplayLane(TrialWaveFunction& prototype, const ParticleSet& source,
      const std::vector<std::size_t>& indices, const optimize::VariableSet& registration)
      : particles(std::make_unique<ParticleSet>(source)), row(registration.size_of_active())
  {
    particles->update();
    for (auto index : indices)
    {
      components.push_back(prototype.getOrbitals()[index]->makeClone(*particles));
      components.back()->checkOutVariables(registration);
    }
  }

  /// Count explicit retained arrays; component-owned modest derivative caches are bounded separately.
  std::size_t numericBytes() const
  {
    return particles->R.capacity()*sizeof(particles->R[0]) + particles->G.capacity()*sizeof(particles->G[0]) +
        particles->L.capacity()*sizeof(particles->L[0]) + row.size()*sizeof(QMCTraits::ValueType);
  }
  std::unique_ptr<ParticleSet> particles;
  std::vector<std::unique_ptr<WaveFunctionComponent>> components;
  Vector<QMCTraits::ValueType> row;
};

namespace
{
using Spline = BsplineFunctor<QMCTraits::RealType>;

/// Require one qualified native factor and stable ordinary component ordering.
PsiFormerWF& findNeural(TrialWaveFunction& wavefunction)
{
  PsiFormerWF* result = nullptr;
  for (const auto& component : wavefunction.getOrbitals())
    if (auto* neural = dynamic_cast<PsiFormerWF*>(component.get()))
    {
      if (result) throw std::invalid_argument("Jastrow product has multiple neural factors");
      result = neural;
    }
  if (!result || !result->isJastrowAugmented() || result->usesAcceleratorParameterProducts())
    throw std::invalid_argument("Jastrow product requires its CPU residual neural profile");
  return *result;
}

/// Gather only modest B-spline objects; neural tensors never enter VariableSet.
UniqueOptObjRefs jastrowObjects(TrialWaveFunction& wavefunction, const std::vector<std::size_t>& indices)
{
  UniqueOptObjRefs objects;
  for (auto index : indices) wavefunction.getOrbitals()[index]->extractOptimizableObjectRefs(objects);
  for (OptimizableObject& object : objects)
    if (!dynamic_cast<Spline*>(&object))
      throw std::invalid_argument("Jastrow product admits only conventional B-spline coefficient objects");
  return objects;
}

/// Preserve declaration, component identities and coefficient ordering in model identity.
std::string serializedDefinition(TrialWaveFunction& wavefunction)
{
  if (!wavefunction.getNode()) throw std::invalid_argument("Jastrow product needs its complete XML model definition");
  xmlBufferPtr buffer = xmlBufferCreate();
  if (!buffer) throw std::bad_alloc();
  const auto result = xmlNodeDump(buffer, wavefunction.getNode()->doc, wavefunction.getNode(), 0, 0);
  std::string definition;
  if (result >= 0) definition.assign(reinterpret_cast<const char*>(buffer->content), buffer->use);
  xmlBufferFree(buffer);
  if (result < 0) throw std::runtime_error("Cannot capture Jastrow product XML definition");
  return definition;
}

/// Canonicalize scientific Jastrow declarations independently of paths and XML formatting.
std::string canonicalJastrowDefinition(TrialWaveFunction& wavefunction)
{
  std::ostringstream result;
  result << std::setprecision(17);
  std::function<void(xmlNodePtr)> visit = [&](xmlNodePtr node) {
    if (node->type != XML_ELEMENT_NODE) return;
    const std::string name(reinterpret_cast<const char*>(node->name));
    result << '<' << name;
    std::map<std::string, std::string> attributes;
    for (auto* attribute = node->properties; attribute; attribute = attribute->next)
    {
      const std::string key(reinterpret_cast<const char*>(attribute->name));
      if (key == "print" || key == "optimize") continue;
      xmlChar* content = xmlNodeListGetString(node->doc, attribute->children, 1);
      if (!content) throw std::runtime_error("Missing Jastrow attribute content");
      attributes.emplace(key, reinterpret_cast<const char*>(content));
      xmlFree(content);
    }
    for (const auto& [key, value] : attributes) result << ' ' << key << '=' << value.size() << ':' << value;
    result << '>';
    if (name == "coefficients")
    {
      xmlChar* content = xmlNodeGetContent(node);
      if (!content) throw std::runtime_error("Missing Jastrow coefficient content");
      std::istringstream values(reinterpret_cast<const char*>(content));
      xmlFree(content);
      double value;
      while (values >> value) result << value << ' ';
      if (!values.eof()) throw std::invalid_argument("Non-numeric Jastrow coefficient definition");
    }
    else for (auto* child = node->children; child; child = child->next) visit(child);
    result << "</" << name << '>';
  };
  for (auto* child = wavefunction.getNode()->children; child; child = child->next)
    if (child->type == XML_ELEMENT_NODE && xmlStrEqual(child->name, BAD_CAST "jastrow")) visit(child);
  return result.str();
}

/// Discard affine rows when only a score contraction is requested.
class NullAffineObserver final : public AffineScoreObserver
{
  void observe(const AffineScoreObservation&) override {}
};

/// Stream complete-product log scores with bounded O(P) shared contractions.
class ProductScoreOperator final : public StreamingDerivativeOperator
{
public:
  ProductScoreOperator(const JastrowNeuralTrainingBinding& binding, const RefVector<ParticleSet>& particles,
      std::shared_ptr<JastrowNeuralScoreWorkspace> workspace, std::size_t version,
      std::size_t batch, std::size_t offset, std::size_t chunk)
      : binding_(binding), particles_(particles), workspace_(std::move(workspace)), version_(version),
        batch_(batch), offset_(offset), plan_(binding.parameterSchema(), version, chunk)
  {}

  StreamingDerivativeCapabilities capabilities() const noexcept override
  {
    StreamingDerivativeCapabilities result;
    result.product_mask = derivativeProductBit(DerivativeProduct::SCORE_VJP) | derivativeProductBit(DerivativeProduct::SCORE_JVP);
    result.adjoint_mask = derivativeAdjointBit(DerivativeAdjoint::TRANSPOSE) | derivativeAdjointBit(DerivativeAdjoint::HERMITIAN);
    result.parameter_scalar_domain = ParameterScalarDomain::REAL64;
    result.result_scalar_domain = ParameterScalarDomain::COMPLEX128;
    result.reduction_domain = ReductionDomain::CROWD_LOCAL;
    result.execution_domain = DerivativeExecutionDomain::HOST;
    result.maximum_vjp_channels = 3;
    result.maximum_parameter_chunk_size = plan_.maximumChunkSize();
    result.maximum_sample_tile_size = 1;
    result.fixed_parameter_scratch_vectors = 4;
    result.block_streaming = true;
    return result;
  }

  const StructuredParameterSchema& parameterSchema() const noexcept override { return binding_.parameterSchema(); }
  std::size_t parameterVersion() const noexcept override { return version_; }
  std::size_t batchOrdinal() const noexcept override { return batch_; }
  std::size_t sampleOffset() const noexcept override { return offset_; }
  std::size_t sampleCount() const noexcept override { return particles_.size(); }
  const ParameterChunkPlan& parameterChunkPlan() const noexcept override { return plan_; }

  StreamingDerivativeStorageDiagnostics storageDiagnostics() const override
  {
    StreamingDerivativeStorageDiagnostics result;
    result.shared_workspace_identity = reinterpret_cast<std::uintptr_t>(workspace_.get());
    result.shared_workspace_bytes = workspace_->row.capacity()*sizeof(double);
    for (const auto& channel : workspace_->channels) result.shared_workspace_bytes += channel.capacity()*sizeof(DerivativeValue);
    result.shared_workspace_bytes += binding_.replayNumericBytes();
    result.retained_numeric_bytes = result.shared_workspace_bytes;
    return result;
  }

protected:
  void evaluateVJPs(DerivativeArrayView<const VJPCoefficientChannel> channels,
      DerivativeAdjoint, ParameterReductionSink& sink) const override
  {
    std::lock_guard lock(workspace_->mutex);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
      std::fill(workspace_->channels[channel].begin(), workspace_->channels[channel].end(), DerivativeValue{});
    NullAffineObserver observer;
    for (std::size_t sample = 0; sample < sampleCount(); ++sample)
    {
      requireVersion();
      binding_.observeAffineScores(particles_[sample], observer, workspace_->row);
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        for (std::size_t p = 0; p < workspace_->row.size(); ++p)
          workspace_->channels[channel][p] += workspace_->row[p]*channels[channel].coefficients.values[sample];
    }
    for (const auto& chunk : plan_.chunks())
      for (std::size_t channel = 0; channel < channels.size(); ++channel)
        sink.add(channel, {chunk, {workspace_->channels[channel].data()+chunk.parameter_offset, chunk.count}});
  }

  void evaluateScoreJVP(const StructuredParameterVectorConstView& direction, SampleProductSink& sink) const override
  {
    std::lock_guard lock(workspace_->mutex);
    NullAffineObserver observer;
    for (std::size_t sample = 0; sample < sampleCount(); ++sample)
    {
      requireVersion();
      binding_.observeAffineScores(particles_[sample], observer, workspace_->row);
      DerivativeValue product{};
      for (std::size_t p = 0; p < workspace_->row.size(); ++p) product += workspace_->row[p]*direction.values()[p];
      const SampleProductTileDescriptor descriptor{parameterSchema().providerId(), parameterSchema().fingerprint(),
          version_, batch_, offset_+sample, 1, sample};
      sink.add({descriptor, {&product, 1}});
    }
  }

private:
  /// A retained operator must never consume a later optimizer publication.
  void requireVersion() const
  {
    if (binding_.parameterVersion() != version_)
      throw std::runtime_error("Stale Jastrow product score replay");
  }
  const JastrowNeuralTrainingBinding& binding_;
  RefVector<ParticleSet> particles_;
  std::shared_ptr<JastrowNeuralScoreWorkspace> workspace_;
  std::size_t version_, batch_, offset_;
  ParameterChunkPlan plan_;
};
} // namespace

JastrowNeuralTrainingBinding::JastrowNeuralTrainingBinding(TrialWaveFunction& prototype, ParticleSet& particles,
    Communicate& communicator, bool pretraining)
    : prototype_(prototype), particles_(particles), communicator_(communicator), pretraining_(pretraining)
{
  auto& factor = findNeural(prototype_);
  const auto& components = prototype_.getOrbitals();
  for (std::size_t index = 0; index < components.size(); ++index)
    if (components[index].get() != &factor)
    {
      if (!dynamic_cast<J1OrbitalSoA<Spline>*>(components[index].get()) &&
          !dynamic_cast<TwoBodyJastrow<Spline>*>(components[index].get()))
        throw std::invalid_argument("Jastrow product supports one neural factor with ordinary B-spline J1/J2 only");
      jastrow_indices_.push_back(index);
    }
  if (jastrow_indices_.empty()) throw std::invalid_argument("Jastrow product requires traditional J1/J2 components");

  auto objects = jastrowObjects(prototype_, jastrow_indices_);
  for (OptimizableObject& object : objects)
  {
    auto& spline = dynamic_cast<Spline&>(object);
    jastrow_total_count_ += spline.Parameters.size();
    spline.isolateOptimizationState();
    object.checkInVariablesExclusive(registration_);
  }
  registration_.resetIndex();
  jastrow_count_ = registration_.size_of_active();
  if (jastrow_count_ > MAXIMUM_JASTROW_PARAMETERS || jastrow_total_count_ > MAXIMUM_JASTROW_PARAMETERS)
    throw std::length_error("Jastrow product exceeds its 256-coefficient admission");
  active_locations_.resize(jastrow_count_);
  for (std::size_t location = 0; location < registration_.size(); ++location)
    if (registration_.where(location) >= 0) active_locations_[registration_.where(location)] = location;
  for (auto index : jastrow_indices_) prototype_.getOrbitals()[index]->checkOutVariables(registration_);

  auto blocks = factor.parameterSchema().blocks();
  neural_count_ = factor.parameterSchema().parameterCount();
  neural_score_.resize(neural_count_);
  for (auto& block : blocks)
    if (block.id.find("/residual_pair_v1/") != std::string::npos)
    {
      block.trainable = !pretraining_;
      block.update_group = "neural_pair_residual";
    }
  std::ostringstream mapping;
  for (auto index : jastrow_indices_)
    mapping << index << ':' << components[index]->getName() << ':' << components[index]->getClassName() << ';';
  for (std::size_t location = 0; location < registration_.size(); ++location)
    mapping << registration_.name(location) << ':' << registration_.where(location) << ';';
  coefficient_mapping_ = mapping.str();
  if (jastrow_count_)
    blocks.push_back({"traditional_jastrow/"+coefficient_mapping_, {jastrow_count_, 1}, neural_count_,
        jastrow_count_, ParameterScalarDomain::REAL64, !pretraining_, "traditional_jastrow"});
  schema_ = std::make_shared<const StructuredParameterSchema>("jastrow_product/"+factor.WaveFunctionComponent::getName(), std::move(blocks));
  version_ = factor.snapshotParameters().version;
  score_workspace_ = std::make_shared<JastrowNeuralScoreWorkspace>(schema_->parameterCount());
  definition_ = serializedDefinition(prototype_);
  immutable_jastrow_definition_ = canonicalJastrowDefinition(prototype_);
  replay_lane_ = std::make_unique<JastrowNeuralReplayLane>(prototype_, particles_, jastrow_indices_, registration_);
  if (pretraining_)
  {
    const auto initial = factor.snapshotParameters();
    for (const auto& block : schema_->blocks())
      if (block.update_group == "neural_pair_residual")
        for (std::size_t p = block.offset; p < block.offset+block.count; ++p)
          if (initial.values[p] != 0) throw std::invalid_argument("Orbital pretraining requires exactly zero residual log lengths");
  }
  attachLive(prototype_, particles_);
}

JastrowNeuralTrainingBinding::~JastrowNeuralTrainingBinding() = default;

PsiFormerWF& JastrowNeuralTrainingBinding::neural() const { return findNeural(prototype_); }

std::size_t JastrowNeuralTrainingBinding::replayNumericBytes() const
{
  std::lock_guard lock(mutex_);
  const auto n = static_cast<std::size_t>(particles_.getTotalNum());
  // Reserve J1/J2 gradient/Laplacian coefficient caches and transient basis rows.
  const auto jastrow_cache_bound = (n+1)*jastrow_total_count_*8*sizeof(double);
  return replay_lane_->numericBytes()+neural_score_.capacity()*sizeof(double)+jastrow_cache_bound;
}

std::size_t JastrowNeuralTrainingBinding::parameterVersion() const
{
  std::lock_guard lock(mutex_);
  return version_;
}

StructuredParameterSnapshot JastrowNeuralTrainingBinding::snapshotParameters() const
{
  std::lock_guard lock(mutex_);
  auto snapshot = neural().snapshotParameters();
  if (snapshot.version != version_ || snapshot.values.size() != neural_count_)
    throw std::logic_error("Joint neural/Jastrow version mismatch");
  snapshot.schema_fingerprint = schema_->fingerprint();
  snapshot.values.resize(schema_->parameterCount());
  for (std::size_t location = 0; location < registration_.size(); ++location)
    if (registration_.where(location) >= 0)
      snapshot.values[neural_count_+registration_.where(location)] = registration_[location];
  return snapshot;
}

void JastrowNeuralTrainingBinding::attachLive(TrialWaveFunction& wavefunction, ParticleSet& particles, QMCHamiltonian* hamiltonian)
{
  findNeural(wavefunction);
  for (auto& live : live_)
    if (live.wavefunction == &wavefunction)
    {
      if (live.particles != &particles) throw std::invalid_argument("Joint live particle binding changed");
      if (hamiltonian) live.hamiltonian = hamiltonian;
      return;
    }
  live_.push_back({&wavefunction, &particles, hamiltonian});
}

void JastrowNeuralTrainingBinding::validateCandidate(const StructuredParameterSnapshot& candidate,
    std::size_t expected_version) const
{
  if (candidate.schema_fingerprint != schema_->fingerprint() || candidate.values.size() != schema_->parameterCount() ||
      candidate.version != version_ || expected_version != version_ || version_ == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Invalid or stale joint Jastrow product candidate");
  for (const double& value : candidate.values)
    if (!psiformer::residualPairFiniteReference(value)) throw std::invalid_argument("Nonfinite joint Jastrow product candidate");
}

void JastrowNeuralTrainingBinding::resetJastrows(TrialWaveFunction& wavefunction,
    const StructuredParameterSnapshot& candidate) const
{
  auto active = registration_;
  for (std::size_t location = 0; location < active.size(); ++location)
    if (active.where(location) >= 0) active[location] = candidate.values[neural_count_+active.where(location)];
  auto objects = jastrowObjects(wavefunction, jastrow_indices_);
  for (OptimizableObject& object : objects) dynamic_cast<Spline&>(object).isolateOptimizationState();
  for (auto index : jastrow_indices_) wavefunction.getOrbitals()[index]->checkOutVariables(active);
  for (OptimizableObject& object : objects) object.resetParametersExclusive(active);
}

std::unique_ptr<TrialWaveFunction> JastrowNeuralTrainingBinding::makeCandidate(ParticleSet& particles,
    const StructuredParameterSnapshot& candidate) const
{
  validateCandidate(candidate, version_);
  auto result = prototype_.makeClone(particles);
  auto& factor = findNeural(*result);
  factor.isolateParameterState();
  auto native = factor.snapshotParameters();
  std::copy_n(candidate.values.begin(), neural_count_, native.values.begin());
  factor.publishParameters(native, native.version);
  resetJastrows(*result, candidate);
  return result;
}

std::size_t JastrowNeuralTrainingBinding::publishParameters(const StructuredParameterSnapshot& candidate,
    std::size_t expected_version)
{
  // The driver is the exclusive publisher. Hold the binding lock through native
  // preparation/readiness/commit; score replay acquires this lock before native locks.
  std::lock_guard lock(mutex_);
  std::vector<std::unique_ptr<TrialWaveFunction>> prepared;
  std::unique_ptr<JastrowNeuralReplayLane> prepared_replay;
  std::unique_ptr<PsiFormerWF::PreparedCpuParameterPublication> native_publication;
  optimize::VariableSet replacement_registration;
  std::exception_ptr failure;
  try
  {
    replacement_registration = registration_;
    validateCandidate(candidate, expected_version);
    const auto old = neural().snapshotParameters();
    for (const auto& block : schema_->blocks())
      if (!block.trainable)
        for (std::size_t p = block.offset; p < block.offset+block.count; ++p)
        {
          const double original = p < neural_count_ ? old.values[p] : registration_[active_locations_[p-neural_count_]];
          if (candidate.values[p] != original) throw std::invalid_argument("Frozen product coordinate changed");
        }
    prepared.reserve(live_.size());

    for (const auto& live : live_)
    {
      auto replacement = live.wavefunction->makeClone(*live.particles);
      auto& factor = findNeural(*replacement);
      if (prepared.empty())
      {
        factor.isolateParameterState();
        auto native = factor.snapshotParameters();
        std::copy_n(candidate.values.begin(), neural_count_, native.values.begin());
        factor.publishParameters(native, expected_version);
      }
      else factor.adoptPreparedParameterState(findNeural(*prepared.front()));
      resetJastrows(*replacement, candidate);
      if (live.particles != &particles_)
      {
        ParticleSet validation_particles(*live.particles);
        validation_particles.update();
        auto validation = replacement->makeClone(validation_particles);
        validation->evaluateLog(validation_particles);
        if (!psiformer::residualPairFiniteReference(validation->getLogPsi())) throw std::runtime_error("Nonfinite product candidate value");
        if (live.hamiltonian)
        {
          auto hamiltonian = live.hamiltonian->makeClone(validation_particles, *validation);
          if (!psiformer::residualPairFiniteReference(hamiltonian->evaluateDeterministic(*validation, validation_particles)))
            throw std::runtime_error("Nonfinite product candidate local energy");
        }
      }
      prepared.push_back(std::move(replacement));
    }
    for (std::size_t location = 0; location < replacement_registration.size(); ++location)
      if (replacement_registration.where(location) >= 0)
        replacement_registration[location] = candidate.values[neural_count_+replacement_registration.where(location)];
    prepared_replay = std::make_unique<JastrowNeuralReplayLane>(*prepared.front(), particles_, jastrow_indices_, replacement_registration);
    auto native = old;
    std::copy_n(candidate.values.begin(), neural_count_, native.values.begin());
    native_publication = neural().prepareCpuParameterPublication(native, expected_version);
  }
  catch (...) { failure = std::current_exception(); }
  int failed = failure ? 1 : 0;
  communicator_.allreduce(failed);
  if (failed)
  {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("Another rank rejected the joint product publication");
  }

  // Preparation above owns every allocation/reset. The following state transition
  // swaps coefficient banks and shares one native parameter image without throwing.
  for (std::size_t i = 0; i < live_.size(); ++i)
  {
    live_[i].wavefunction->activatePreparedJastrows(*prepared[i], jastrow_indices_);
  }
  registration_ = std::move(replacement_registration);
  replay_lane_.swap(prepared_replay);
  version_ = native_publication->commit();
  return version_;
}

std::vector<std::string> JastrowNeuralTrainingBinding::neuralWeightBlocks() const { return neural().neuralWeightBlocks(); }

std::vector<AffineParameterBlock> JastrowNeuralTrainingBinding::affineParameterBlocks() const
{
  auto result = neural().affineParameterBlocks();
  if (jastrow_count_ && !pretraining_)
    result.push_back({"traditional_jastrow_complete_configuration", schema_->blocks().size()-1,
        NO_AFFINE_BIAS_BLOCK, jastrow_count_, 1});
  return result;
}

std::vector<std::size_t> JastrowNeuralTrainingBinding::genericKFACParameterBlocks() const
{
  return pretraining_ ? std::vector<std::size_t>{} : neural().genericKFACParameterBlocks();
}

void JastrowNeuralTrainingBinding::observeAffineScores(const ParticleSet& particles,
    AffineScoreObserver& observer, std::vector<double>& score) const
{
  std::lock_guard lock(mutex_);
  score.resize(schema_->parameterCount());
  neural().observeAffineScores(particles, observer, neural_score_);
  std::copy(neural_score_.begin(), neural_score_.end(), score.begin());
  if (!jastrow_count_) return;
  replay_lane_->particles->R = particles.R;
  replay_lane_->particles->update();
  replay_lane_->row = 0;
  for (const auto& component : replay_lane_->components)
    component->evaluateDerivativesWF(*replay_lane_->particles, registration_, replay_lane_->row);
  for (std::size_t p = 0; p < jastrow_count_; ++p) score[neural_count_+p] = std::real(replay_lane_->row[p]);
  if (!pretraining_)
  {
    const double sensitivity = 1;
    // One activation is the complete configuration's aggregated coefficient score.
    // The existing Gaussian/native-score policy owns curvature scaling/centering.
    observer.observe({neural().affineParameterBlocks().size(), score.data()+neural_count_, &sensitivity, 1});
  }
}

std::unique_ptr<StreamingDerivativeOperator> JastrowNeuralTrainingBinding::makeScoreOperator(
    const RefVector<ParticleSet>& particles, std::size_t batch, std::size_t offset, std::size_t chunk) const
{
  return std::make_unique<ProductScoreOperator>(*this, particles, score_workspace_, version_, batch, offset, chunk);
}

std::string JastrowNeuralTrainingBinding::modelIdentity() const
{
  return "jastrow_product_v1:"+neural().trainingConfigurationFingerprint()+":"+coefficient_mapping_+":"+immutable_jastrow_definition_;
}

void JastrowNeuralTrainingBinding::exportModel(const std::filesystem::path& path,
    const std::string& associated_model_filename) const
{
  const auto snapshot = snapshotParameters();
  neural().exportModel(path.string());
  Libxml2Document document;
  if (!document.parseFromString(definition_)) throw std::runtime_error("Cannot reconstruct product association XML");
  std::map<std::string, double> selected;
  for (auto location : active_locations_) selected.emplace(registration_.name(location), registration_[location]);
  std::function<void(xmlNodePtr)> update = [&](xmlNodePtr node) {
    if (node->type != XML_ELEMENT_NODE) return;
    if (xmlStrEqual(node->name, BAD_CAST "psiformer"))
    {
      for (const auto* attribute : {"initialization", "initialization_seed", "neural_jastrow", "neural_jastrow_cutoff",
           "neural_jastrow_same_reference", "neural_jastrow_opposite_reference", "export_parameters", "configuration"})
        xmlUnsetProp(node, BAD_CAST attribute);
      const auto filename = associated_model_filename.empty() ? path.filename().string() : associated_model_filename;
      xmlSetProp(node, BAD_CAST "parameters", BAD_CAST filename.c_str());
    }
    else if (xmlStrEqual(node->name, BAD_CAST "coefficients"))
    {
      xmlChar* id_text = xmlGetProp(node, BAD_CAST "id");
      xmlChar* content = xmlNodeGetContent(node);
      if (!id_text || !content)
      {
        if (id_text) xmlFree(id_text);
        if (content) xmlFree(content);
        throw std::runtime_error("Incomplete product coefficient XML");
      }
      const std::string id(reinterpret_cast<const char*>(id_text));
      std::istringstream values(reinterpret_cast<const char*>(content));
      xmlFree(id_text);
      xmlFree(content);
      std::ostringstream replacement;
      replacement << std::setprecision(17);
      double value;
      std::size_t index = 0;
      while (values >> value)
      {
        const auto found = selected.find(id+"_"+std::to_string(index++));
        replacement << (found == selected.end() ? value : found->second) << ' ';
      }
      if (!values.eof()) throw std::runtime_error("Invalid product coefficient XML");
      xmlNodeSetContent(node, BAD_CAST replacement.str().c_str());
    }
    for (auto* child = node->children; child; child = child->next) update(child);
  };
  update(document.getRoot());
  xmlBufferPtr buffer = xmlBufferCreate();
  if (!buffer) throw std::bad_alloc();
  const int result = xmlNodeDump(buffer, document.getRoot()->doc, document.getRoot(), 0, 1);
  const std::string association(reinterpret_cast<const char*>(buffer->content), buffer->use);
  xmlBufferFree(buffer);
  if (result < 0) throw std::runtime_error("Cannot serialize current-J product association");

  hdf_archive output;
  if (!output.open(path, H5F_ACC_RDWR)) throw std::runtime_error("Cannot append product model association");
  output.push("jastrow_neural_product");
  output.write(std::vector<int>{1,0,0}, "format_version");
  output.write(snapshot.schema_fingerprint, "schema_fingerprint");
  output.write(static_cast<std::uint64_t>(snapshot.version), "parameter_version");
  output.write(snapshot.values, "joint_values");
  output.write(TrainingCheckpoint::valuesFingerprint(snapshot.values), "values_fingerprint");
  output.write(modelIdentity(), "model_identity");
  output.write(coefficient_mapping_, "coefficient_mapping");
  output.write(association, "complete_wavefunction_xml");
  output.write(std::string("jastrow-neural-product-complete-v1"), "completion_cookie");
}

StructuredParameterSnapshot JastrowNeuralTrainingBinding::readModel(const std::filesystem::path& path) const
{
  const auto native = psiformer::readPsiFormerModelFile(path.string());
  if (!neural().matchesJastrowCorrelation(native.parameters.environment))
    throw std::runtime_error("Imported product correlation/cell differs from this binding");
  const auto& neural_blocks = neural().parameterSchema().blocks();
  if (native.parameters.layouts.size() != neural_blocks.size())
    throw std::runtime_error("Imported product neural tensor count differs");
  for (std::size_t i = 0; i < neural_blocks.size(); ++i)
  {
    const auto& tensor = native.parameters.layouts[i];
    const auto& block = neural_blocks[i];
    if (tensor.module+"/"+tensor.name != block.id || tensor.shape != block.shape ||
        tensor.begin != block.offset || tensor.end != block.offset+block.count)
      throw std::runtime_error("Imported product neural tensor layout differs");
  }
  hdf_archive input;
  if (!input.open(path, H5F_ACC_RDONLY) || !input.is_group("jastrow_neural_product"))
    throw std::runtime_error("Product model association is missing");
  input.push("jastrow_neural_product", false);
  std::vector<int> format;
  std::string identity, mapping, fingerprint, cookie, association;
  std::uint64_t version;
  StructuredParameterSnapshot result;
  input.read(format, "format_version");
  input.read(result.schema_fingerprint, "schema_fingerprint");
  input.read(version, "parameter_version");
  input.read(result.values, "joint_values");
  input.read(fingerprint, "values_fingerprint");
  input.read(identity, "model_identity");
  input.read(mapping, "coefficient_mapping");
  input.read(association, "complete_wavefunction_xml");
  input.read(cookie, "completion_cookie");
  if (format != std::vector<int>{1,0,0} || cookie != "jastrow-neural-product-complete-v1" ||
      result.schema_fingerprint != schema_->fingerprint() || result.values.size() != schema_->parameterCount() ||
      version > std::numeric_limits<std::size_t>::max() || mapping != coefficient_mapping_ || identity != modelIdentity() ||
      TrainingCheckpoint::valuesFingerprint(result.values) != fingerprint || association.empty() ||
      native.parameters.values.size() != neural_count_ ||
      !std::equal(native.parameters.values.begin(), native.parameters.values.end(), result.values.begin()))
    throw std::runtime_error("Product model association/schema/physical identity mismatch");
  result.version = version;
  return result;
}

double JastrowNeuralTrainingBinding::evaluateOrbitalMSE(const ParticleSet& particles,
    const std::vector<double>& target, std::vector<double>& gradient, std::size_t expected_version) const
{
  std::lock_guard lock(mutex_);
  if (!pretraining_ || expected_version != version_ || gradient.size() != schema_->parameterCount())
    throw std::invalid_argument("Invalid joint orbital-pretraining request");
  const double loss = neural().evaluateOrbitalMSE(particles, target, neural_score_, expected_version);
  std::fill(gradient.begin(), gradient.end(), 0);
  std::copy(neural_score_.begin(), neural_score_.end(), gradient.begin());
  return loss;
}
} // namespace qmcplusplus::wftrain
