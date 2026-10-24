//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_GEOMETRY_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_GEOMETRY_VALIDATION_H
#include "PsiFormerDeviceComplexScoreValidation.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <cstring>

namespace qmcplusplus::psiformer::device
{
/// Construct explicit immutable old/new policies without changing the cold environment oracle.
struct ComplexScoreGeometryValidationAccess
{
  /// Keep explicit primal-policy oracles independent of the process-cached selectors.
  static std::unique_ptr<PreparedDeviceComplexScore> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, bool geometry,
      bool cooperative_lu = false)
  {
    const auto cost = estimateComplexScoreCost(image->plan());
    auto owner = std::unique_ptr<PreparedDeviceComplexScore>(
        new PreparedDeviceComplexScore(std::move(image), queue, blas, cost, false, true, geometry, cooperative_lu));
    if (owner->forward_->trunk_->usesCooperativePeriodicValueGeometry() != geometry ||
        owner->forward_->score_cooperative_lu_ != cooperative_lu)
      throw std::runtime_error("Retained score owner did not dispatch its explicit primal policies");
    return owner;
  }

  /// Snapshot every primal tape byte consumed by reverse, plus forward raw/cusp values.
  static std::vector<double> tape(const PreparedDeviceComplexScore& owner)
  {
    const auto& graph = *owner.forward_->trunk_;
    std::vector<double> result;
    const auto append = [&](const DeviceBuffer<double>& buffer) {
      const auto offset = result.size();
      result.resize(offset + buffer.capacity());
      if (buffer.capacity())
        buffer.download(result.data() + offset, buffer.capacity());
    };
    append(graph.raw_);
    append(graph.cusp_jets_);
    for (const auto& block : graph.value_tape_)
      for (const auto* buffer : {&block->input, &block->query, &block->key, &block->value,
          &block->probabilities, &block->context, &block->residual, &block->hidden, &block->projected})
        append(*buffer);
    return result;
  }

  /// Keep exact primal failure status available without publishing a partial score.
  static std::vector<OpenSpatialStatus> status(const PreparedDeviceComplexScore& owner)
  {
    const auto& buffer = owner.forward_->trunk_->open_status_;
    std::vector<OpenSpatialStatus> result(buffer.capacity());
    buffer.download(result.data(), result.size());
    return result;
  }
};

/// Compare retained objectives, publication and rebind under geometry or cooperative-LU policies.
inline void validateComplexScorePeriodicGeometry(bool qualify_cooperative_lu = false)
{
  const auto descriptor = makeTargetCostPlan(64, true, true);
  const auto initialized = initializePsiFormerParameters(descriptor.modelShape(), 1729,
      PSIFORMER_COMPLEX_PERIODIC_V1, descriptor.environment());
  const PsiFormerExecutionPlan plan(descriptor.modelShape(), initialized.layouts, descriptor.environment());
  std::vector<double> positions(3 * 64), nuclei(3 * plan.modelShape().nuclei);
  // Fixed rational three-dimensional positions; no parameter/seed/conditioning search.
  for (std::size_t i = 0; i < 64; ++i)
    for (std::size_t d = 0; d < 3; ++d)
      positions[3 * i + d] = 8.0 * double((i * (d == 0 ? 17 : d == 1 ? 29 : 43) + 7 * d + 1) % 67) / 67;
  for (std::size_t i = 0; i < nuclei.size(); ++i)
    nuclei[i] = 8.0 * double((i * 19 + 3) % 41) / 41;
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  auto image = PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
  const auto cost = estimateComplexScoreCost(plan);
  auto& metrics = deviceMemoryMetrics(0);
  const auto before = metrics.snapshot();
  bool denied = false;
  try { (void)PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes - 1); }
  catch (const std::length_error&) { denied = true; }
  if (!denied || metrics.snapshot().allocation_calls != before.allocation_calls)
    throw std::runtime_error("Retained geometry minus-one allocated");
  auto selected = PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes);
  if (selected->usesPeriodicGeometry() != configuredComplexScorePeriodicGeometry() ||
      (qualify_cooperative_lu && (selected->usesCooperativeLu() != configuredComplexScoreCooperativeLu() ||
          !selected->usesPeriodicGeometry() || !selected->usesInverseColumns())) ||
      metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Retained geometry selector or exact unchanged cost differs");
  selected.reset();
  auto scalar = ComplexScoreGeometryValidationAccess::prepare(image, queue, blas, qualify_cooperative_lu);
  auto cooperative_owner = ComplexScoreGeometryValidationAccess::prepare(image, queue, blas, true, qualify_cooperative_lu);
  if ((qualify_cooperative_lu ? scalar->usesCooperativeLu() || !cooperative_owner->usesCooperativeLu() :
          scalar->usesPeriodicGeometry() || !cooperative_owner->usesPeriodicGeometry()))
    throw std::runtime_error("Explicit retained score primal oracles not distinct");
  std::vector<cuDoubleComplex> reference(plan.parameterCount()), actual(plan.parameterCount());
  ComplexValueResult expected_amplitude{}, actual_amplitude{};
  const auto compare = [&] {
    scalar->evaluate(positions);
    cooperative_owner->evaluate(positions);
    scalar->download(scalar->resultView(), reference, expected_amplitude);
    cooperative_owner->download(cooperative_owner->resultView(), actual, actual_amplitude);
    if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(cuDoubleComplex)) ||
        std::memcmp(&expected_amplitude, &actual_amplitude, sizeof(ComplexValueResult)))
      throw std::runtime_error("Retained geometry complete score/amplitude not bitwise exact");
    const auto raw = ComplexScoreGeometryValidationAccess::tape(*scalar);
    const auto cooperative_tape = ComplexScoreGeometryValidationAccess::tape(*cooperative_owner);
    if (raw.size() != cooperative_tape.size() ||
        std::memcmp(raw.data(), cooperative_tape.data(), raw.size() * sizeof(double)))
      throw std::runtime_error("Retained geometry raw/cusp/tape bits differ");
    for (const auto value : actual)
      if (!device_math::isFiniteBinary64(value.x) || !device_math::isFiniteBinary64(value.y))
        throw std::runtime_error("Retained geometry nonfinite complete score");
  };
  compare();
  const auto stale = cooperative_owner->resultView();
  const auto warm = metrics.snapshot();
  compare();
  if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Retained geometry warm evaluation allocated/freed");
  auto changed = initialized.values;
  changed[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += 0.03125;
  changed[plan.parameter(ParameterRole::CUSP_SAME_ALPHA).begin] += 0.0625;
  changed[plan.parameter(ParameterRole::CUSP_OPPOSITE_ALPHA).begin] += 0.03125;
  auto successor = image->makeUpdatedImage(changed, 1);
  const auto rebind = metrics.snapshot();
  scalar->rebindModel(successor);
  cooperative_owner->rebindModel(successor);
  if (metrics.snapshot().allocation_calls != rebind.allocation_calls)
    throw std::runtime_error("Retained geometry image rebind allocated");
  denied = false;
  try { cooperative_owner->download(stale, actual, actual_amplitude); }
  catch (const std::invalid_argument&) { denied = true; }
  if (!denied)
    throw std::runtime_error("Retained geometry rebind published stale response");
  compare();
  for (auto* owner : {scalar.get(), cooperative_owner.get()})
  {
    ComplexScoreValidationAccess::failSecond(*owner, positions);
    denied = false;
    try { (void)owner->resultView(); }
    catch (const std::logic_error&) { denied = true; }
    if (!denied)
      throw std::runtime_error("Retained geometry second-objective failure published partial score");
  }
  compare();
  const auto saved_positions = positions;
  positions[0] = std::numeric_limits<double>::quiet_NaN();
  for (auto* owner : {scalar.get(), cooperative_owner.get()})
  {
    denied = false;
    try { owner->evaluate(positions); }
    catch (const std::runtime_error&) { denied = true; }
    if (!denied)
      throw std::runtime_error("Retained geometry nonfinite input admitted");
    denied = false;
    try { (void)owner->resultView(); }
    catch (const std::logic_error&) { denied = true; }
    if (!denied)
      throw std::runtime_error("Retained geometry failure published score");
  }
  const auto scalar_status = ComplexScoreGeometryValidationAccess::status(*scalar);
  const auto cooperative_status = ComplexScoreGeometryValidationAccess::status(*cooperative_owner);
  if (scalar_status.size() != cooperative_status.size() ||
      std::memcmp(scalar_status.data(), cooperative_status.data(),
          scalar_status.size() * sizeof(OpenSpatialStatus)))
    throw std::runtime_error("Retained geometry failed status differs");
  positions = saved_positions;
  compare();
  const auto raw_cost = estimateComplexOrbitalVJPCost(plan);
  auto raw = PreparedDeviceComplexScore::prepareOrbitalVJP(successor, queue, blas, raw_cost.device_bytes);
  if (raw->usesPeriodicGeometry() || raw->usesCooperativeLu())
    throw std::runtime_error("Raw-orbital VJP admitted geometry-only selection");
  for (const auto n : {63, 64, 300, 301})
    if (complexScorePeriodicGeometryEligible(makeTargetCostPlan(n, true, true)) != (n >= 64 && n <= 300))
      throw std::runtime_error("Retained geometry electron range differs from qualification");
  if (complexScorePeriodicGeometryEligible(makeTargetCostPlan(64, false)) ||
      complexScorePeriodicGeometryEligible(makeTargetCostPlan(64, true)))
    throw std::runtime_error("Open/real retained geometry selection widened");
  const auto layouts = [&](bool extra_determinant) {
    std::vector<ParameterLayoutInput> result;
    std::size_t offset = 0;
    for (const auto& tensor : plan.parameterTensors())
    {
      auto dimensions = tensor.shape;
      if (extra_determinant)
      {
        if (tensor.role == ParameterRole::ENVELOPE_PI_UP || tensor.role == ParameterRole::ENVELOPE_PI_DOWN ||
            tensor.role == ParameterRole::ENVELOPE_ZETA_UP || tensor.role == ParameterRole::ENVELOPE_ZETA_DOWN)
          dimensions[0] = 17 * 64;
        if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
          dimensions[1] = 17 * 64;
      }
      std::size_t extent = 1;
      for (auto dimension : dimensions)
        extent *= dimension;
      result.push_back({tensor.module, tensor.name, dimensions, offset, offset + extent});
      offset += extent;
    }
    return result;
  };
  auto d17 = plan.modelShape();
  d17.determinants = 17;
  if (complexScorePeriodicGeometryEligible(PsiFormerExecutionPlan(d17, layouts(true), plan.environment())))
    throw std::runtime_error("Retained geometry D17 selection widened");
  for (bool partial_axes : {false, true})
  {
    auto environment = plan.environment();
    if (partial_axes)
      environment.periodic_axes[2] = false;
    else
      environment.fixed_nuclei = false;
    // Current plan admission rejects these schemas; future representable plans must still fall back.
    try
    {
      const PsiFormerExecutionPlan excluded(plan.modelShape(), layouts(false), environment);
      if (complexScorePeriodicGeometryEligible(excluded))
        throw std::runtime_error("Retained geometry partial-axes/moving-nuclei selection widened");
    }
    catch (const std::invalid_argument&) {}
  }
  std::cout << (qualify_cooperative_lu ? "PASS retained_score_cooperative_lu " : "PASS retained_score_periodic_geometry ")
            << "raw_tape_allP_amplitude_bits=exact objectives=2 changed_alpha_embedding_image=1"
            << " stale_rejected=1 second_failure_hidden=1 warm_allocations=0 minus_one=1 raw_excluded=1"
            << " extra_device_bytes=0 selected="
            << (qualify_cooperative_lu ? configuredComplexScoreCooperativeLu() : configuredComplexScorePeriodicGeometry()) << '\n';
}
} // namespace qmcplusplus::psiformer::device
#endif
