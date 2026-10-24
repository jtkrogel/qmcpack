//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_INVERSE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_SCORE_INVERSE_VALIDATION_H
#include "PsiFormerDeviceComplexScoreValidation.h"
#include "PsiFormerDeviceTargetCostValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <cstring>

namespace qmcplusplus::psiformer::device
{
/// Construct explicit immutable old/new policies without changing the cold environment oracle.
struct ComplexScoreInverseValidationAccess
{
  /// Keep explicit scalar/column arithmetic oracles independent of the process-cached selector.
  static std::unique_ptr<PreparedDeviceComplexScore> prepare(std::shared_ptr<const PreparedDeviceModel> image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas, bool columns)
  {
    const auto cost = estimateComplexScoreCost(image->plan());
    return std::unique_ptr<PreparedDeviceComplexScore>(
        new PreparedDeviceComplexScore(std::move(image), queue, blas, cost, false, columns));
  }
};

/// Compare complete retained objectives, publication and same-family rebind under both inverse policies.
inline void validateComplexScoreInverseColumns()
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
    throw std::runtime_error("Retained inverse columns minus-one allocated");
  auto selected = PreparedDeviceComplexScore::prepare(image, queue, blas, cost.device_bytes);
  if (selected->usesInverseColumns() != configuredComplexScoreInverseColumns() ||
      metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes)
    throw std::runtime_error("Retained inverse selector or exact unchanged cost differs");
  selected.reset();
  auto scalar = ComplexScoreInverseValidationAccess::prepare(image, queue, blas, false);
  auto columns = ComplexScoreInverseValidationAccess::prepare(image, queue, blas, true);
  if (scalar->usesInverseColumns() || !columns->usesInverseColumns())
    throw std::runtime_error("Explicit retained inverse oracles not distinct");
  std::vector<cuDoubleComplex> reference(plan.parameterCount()), actual(plan.parameterCount());
  ComplexValueResult expected_amplitude{}, actual_amplitude{};
  const auto compare = [&] {
    scalar->evaluate(positions);
    columns->evaluate(positions);
    scalar->download(scalar->resultView(), reference, expected_amplitude);
    columns->download(columns->resultView(), actual, actual_amplitude);
    if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(cuDoubleComplex)) ||
        std::memcmp(&expected_amplitude, &actual_amplitude, sizeof(ComplexValueResult)))
      throw std::runtime_error("Retained inverse complete score/amplitude not bitwise exact");
    for (const auto value : actual)
      if (!device_math::isFiniteBinary64(value.x) || !device_math::isFiniteBinary64(value.y))
        throw std::runtime_error("Retained inverse nonfinite complete score");
  };
  compare();
  const auto stale = columns->resultView();
  const auto warm = metrics.snapshot();
  compare();
  if (metrics.snapshot().allocation_calls != warm.allocation_calls || metrics.snapshot().free_calls != warm.free_calls)
    throw std::runtime_error("Retained inverse warm evaluation allocated/freed");
  auto changed = initialized.values;
  changed[plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).begin] += 0.03125;
  changed[plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG).begin] += 0.0625;
  auto successor = image->makeUpdatedImage(changed, 1);
  const auto rebind = metrics.snapshot();
  scalar->rebindModel(successor);
  columns->rebindModel(successor);
  if (metrics.snapshot().allocation_calls != rebind.allocation_calls)
    throw std::runtime_error("Retained inverse image rebind allocated");
  denied = false;
  try { columns->download(stale, actual, actual_amplitude); }
  catch (const std::invalid_argument&) { denied = true; }
  if (!denied)
    throw std::runtime_error("Retained inverse rebind published stale response");
  compare();
  for (auto* owner : {scalar.get(), columns.get()})
  {
    ComplexScoreValidationAccess::failSecond(*owner, positions);
    denied = false;
    try { (void)owner->resultView(); }
    catch (const std::logic_error&) { denied = true; }
    if (!denied)
      throw std::runtime_error("Retained inverse second-objective failure published partial score");
  }
  compare();
  const auto raw_cost = estimateComplexOrbitalVJPCost(plan);
  auto raw = PreparedDeviceComplexScore::prepareOrbitalVJP(successor, queue, blas, raw_cost.device_bytes);
  if (raw->usesInverseColumns())
    throw std::runtime_error("Raw-orbital VJP admitted inverse-only selection");
  for (const auto n : {63, 64, 300, 301})
    if (complexScoreInverseColumnsEligible(makeTargetCostPlan(n, true, true)) != (n >= 64 && n <= 300))
      throw std::runtime_error("Retained inverse electron range differs from qualification");
  if (complexScoreInverseColumnsEligible(makeTargetCostPlan(64, false)) ||
      complexScoreInverseColumnsEligible(makeTargetCostPlan(64, true)))
    throw std::runtime_error("Open/real retained inverse selection widened");
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
  if (complexScoreInverseColumnsEligible(PsiFormerExecutionPlan(d17, layouts(true), plan.environment())))
    throw std::runtime_error("Retained inverse D17 selection widened");
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
      if (complexScoreInverseColumnsEligible(excluded))
        throw std::runtime_error("Retained inverse partial-axes/moving-nuclei selection widened");
    }
    catch (const std::invalid_argument&) {}
  }
  std::cout << "PASS retained_score_inverse_columns allP_amplitude_bits=exact objectives=2 changed_image=1"
            << " stale_rejected=1 second_failure_hidden=1 warm_allocations=0 minus_one=1 raw_excluded=1"
            << " extra_device_bytes=0 selected=" << configuredComplexScoreInverseColumns() << '\n';
}
} // namespace qmcplusplus::psiformer::device
#endif
