//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_CANONICAL_HEAD_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_CANONICAL_HEAD_VALIDATION_H
#include "PsiFormerDeviceMixedTargetValidation.h"
#include "PsiFormerDeviceComplexScore.h"

namespace qmcplusplus::psiformer::device
{
/// Exercise canonical paired physical owners on the unchanged existing localized Ne4 fixture.
inline void validateComplexCanonicalHeads()
{
  using namespace mixed_target_validation;
  const ModelShape shape{2, 2, 4, 2, 8, 1, 1};
  ExecutionEnvironment environment;
  environment.boundary = BoundaryCondition::PERIODIC;
  environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
  environment.periodic_axes = {true, true, true};
  environment.lattice_vectors = {{{8., 0., 0.}, {0., 8., 0.}, {0., 0., 8.}}};
  environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
  environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  const auto parameters = mixed_periodic_validation::parameters(environment, shape);
  const auto plan = PsiFormerExecutionPlan::fromParameters(parameters, shape, environment);
  const std::vector<double> nuclei{0,0,0, 2,0,0, 0,2,0, 0,0,2};
  const std::vector<double> first{.1,.2,.3, 2.1,.2,.3, .1,2.2,.3, .1,.2,2.3};
  const auto image = PreparedDeviceModel::preparePairedComplexValue(plan, parameters.flat_values(), nuclei);
  compute::Queue<PlatformKind::CUDA> queue;
  AcceleratorBlasHandle blas(queue);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "canonical paired validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "canonical paired validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  for (const auto& request : {std::pair<std::size_t,std::size_t>{0,1}, {1,1}, {3,2}})
  {
    const auto k = request.first, batch = request.second;
    const std::size_t concurrent = k == 3 ? 4 : 0;
    auto positions = first;
    if (batch == 2) { positions.insert(positions.end(), first.begin(), first.end()); positions[12] += .031; }
    auto preparation = complexPreparation(plan, batch, k, concurrent);
    if (k)
    {
      preparation.scalar_work_budget = spatial_detail::checkedAdd(preparation.scalar_work_budget,
          canonicalHeadScalarWork(plan, batch), "Canonical paired validation work overflow");
      preparation.device_workspace_budget_bytes = std::size_t(2) * 1024 * 1024 * 1024;
      preparation.device_workspace_budget_bytes = PreparedDeviceMixedComplexSpatial::estimatePrivateDeviceBytes(
          plan, batch, concurrent, preparation, true);
    }
    const auto bytes = PreparedDeviceMixedComplexSpatial::estimateDeviceBytes(plan, batch, concurrent, preparation, true);
    const auto before = metrics.snapshot();
    bool rejected = false;
    try { PreparedDeviceMixedComplexSpatial short_owner(image, batch, queue, blas, bytes - 1, true,
        concurrent, {}, preparation, true); }
    catch (const std::length_error&) { rejected = true; }
    require(rejected && metrics.snapshot().allocation_calls == before.allocation_calls,
        "Canonical paired minus-one cap allocated before rejection");
    auto reference = k ? PreparedDeviceComplexSpatial::prepare(image, queue, blas, preparation, batch, concurrent)
                       : PreparedDeviceComplexSpatial::prepare(image, batch, queue, blas,
                           estimateComplexSpatialCost(plan, batch, concurrent).device_bytes, concurrent);
    reference->evaluate(positions);
    const auto expected = complexFields(*reference, batch, shape.electrons());
    const auto previous_ticket = reference->resultView();
    auto foreign_epoch = PreparedDeviceCanonicalComplexHead::prepare(image, batch, queue, blas,
        PreparedDeviceCanonicalComplexHead::required(plan, batch).device_bytes);
    auto stale_head = foreign_epoch->evaluate(positions);
    (void)foreign_epoch->evaluate(positions);
    rejected = false;
    try { reference->evaluateCanonicalHead(stale_head, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Paired spatial consumer admitted a stale canonical head");
    rejected = false;
    try { (void)reference->resultView(); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Failed paired head preflight retained the prior physical result");
    auto rejected_copy = expected;
    rejected = false;
    try { reference->download(previous_ticket, rejected_copy.amplitude, rejected_copy.gradient,
        rejected_copy.lap_log, rejected_copy.lap_ratio); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Failed paired head preflight allowed an old physical-ticket download");
    PreparedDeviceMixedComplexSpatial owner(image, batch, queue, blas, bytes, false, concurrent, {}, preparation, true);
    owner.evaluateCanonicalValue(positions);
    std::vector<ComplexValueResult> scalar(batch);
    owner.downloadCanonicalValue(scalar);
    owner.evaluate(positions, true);
    require(!owner.usedFp64Retry() && owner.statistics().projections > 0,
        "Canonical paired physical owner did not complete genuine mixed jets");
    const auto actual = complexFields(owner, batch, shape.electrons());
    complexAccuracy(actual, expected, batch, shape.electrons());
    for (std::size_t i = 0; i < batch; ++i)
      require(std::memcmp(&scalar[i].log_abs, &actual.amplitude[i].log_abs, sizeof(double)) == 0 &&
          std::memcmp(&scalar[i].phase, &actual.amplitude[i].phase, sizeof(cuDoubleComplex)) == 0,
          "Canonical paired scalar and FULL requests represent different primals");
    rejected = false;
    try { owner.downloadCanonicalValue(scalar); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "New paired FULL request retained stale scalar authority");
    const auto warm = metrics.snapshot();
    owner.evaluate(positions, true);
    require(identical(actual, complexFields(owner, batch, shape.electrons())), "Warm canonical paired fields changed bits");
    const auto after = metrics.snapshot();
    require(warm.allocation_calls == after.allocation_calls && warm.free_calls == after.free_calls &&
        warm.live_bytes == after.live_bytes, "Warm canonical paired request changed tracked DeviceBuffer storage");
    std::cout << "PASS canonical paired physical K=" << k << " B=" << batch
              << " actual_mixed_jets=1 scalar_FULL_primal_bits=exact warm_DeviceBuffer_delta=0\n";
  }
  {
    const auto cost = PreparedDeviceCanonicalComplexHead::required(plan, 1);
    auto producer = PreparedDeviceCanonicalComplexHead::prepare(image, 1, queue, blas, cost.device_bytes);
    auto old = producer->evaluate(first);
    old.validate(image, image->version(), &queue, 1, shape.electrons(), shape.determinants);
    auto changed = first; changed[0] += .01;
    (void)producer->evaluate(changed);
    bool rejected = false;
    try { old.validate(image, image->version(), &queue, 1, shape.electrons(), shape.determinants); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Canonical paired producer accepted a prior coordinate generation");
    auto current = producer->view();
    producer->evaluateScalar(first);
    rejected = false;
    try { current.validate(image, image->version(), &queue, 1, shape.electrons(), shape.determinants); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Scalar paired proposal retained the previous guarded head authority");
    rejected = false;
    try { (void)producer->view(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Scalar paired proposal minted unguarded inverse authority");
    auto next = parameters.flat_values(); next[0] += .001;
    auto successor = image->makeUpdatedImage(next, image->version() + 1);
    current = producer->evaluate(first);
    rejected = false;
    try { producer->validateModelRebind(successor); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Canonical paired producer replaced storage with a live coefficient borrow");
    old = {}; current = {};
    producer->rebindModel(successor);
    const auto fresh = producer->evaluate(first);
    fresh.validate(successor, successor->version(), &queue, 1, shape.electrons(), shape.determinants);
    rejected = false;
    try { fresh.validate(image, image->version(), &queue, 1, shape.electrons(), shape.determinants); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Canonical paired producer accepted a stale immutable model image");
    std::cout << "PASS canonical paired provenance stale_R=reject live_rebind=reject successor=valid foreign_image=reject\n";
  }
  {
    const auto before = metrics.snapshot();
    const auto cost = PreparedDeviceCanonicalComplexHead::required(plan, 1);
    ComplexCanonicalHeadView retired;
    {
      auto producer = PreparedDeviceCanonicalComplexHead::prepare(image, 1, queue, blas, cost.device_bytes);
      retired = producer->evaluate(first);
    }
    require(metrics.snapshot().live_bytes == before.live_bytes + cost.device_bytes,
        "Retired paired producer freed storage retained by its issued identity");
    bool rejected = false;
    try { retired.validate(image, image->version(), &queue, 1, shape.electrons(), shape.determinants); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Retired paired producer left usable inverse authority");
    retired = {};
    const auto after = metrics.snapshot();
    require(after.live_bytes == before.live_bytes && after.failed_frees == before.failed_frees,
        "Last paired canonical identity failed to retire the exact device owner");
    std::cout << "PASS canonical paired retirement stale_view=reject retained_storage=exact last_borrow_freed=exact\n";
  }
  {
    auto node_values = parameters.flat_values();
    for (const auto& tensor : plan.parameterTensors())
      if (tensor.role == ParameterRole::BACKFLOW_UP_WEIGHT || tensor.role == ParameterRole::BACKFLOW_DOWN_WEIGHT)
        std::fill_n(node_values.begin() + tensor.begin, tensor.size(), 0.0);
    auto node_image = image->makeUpdatedImage(node_values, image->version() + 1);
    auto node = PreparedDeviceCanonicalComplexHead::prepare(node_image, 1, queue, blas,
        PreparedDeviceCanonicalComplexHead::required(plan, 1).device_bytes);
    node->evaluateScalar(first);
    std::vector<ComplexValueResult> scalar(1);
    node->downloadScalar(scalar);
    require(scalar[0].status == 1 && scalar[0].singular_channels == shape.determinants &&
        cuCreal(scalar[0].phase) == 0 && cuCimag(scalar[0].phase) == 0,
        "Nodal paired scalar request incorrectly required a regular inverse");
    bool rejected = false;
    try { (void)node->evaluate(first); } catch (const MixedPrecisionRetryRequired&) { rejected = true; }
    require(rejected, "Nodal paired derivative request minted unavailable inverse authority");
    std::cout << "PASS canonical paired nodal_scalar=valid guarded_head=typed_reject\n";
  }
  {
    auto cache = std::make_shared<PreparedDeviceMixedDenseParameters>(image, queue,
        MixedTrunkHeadPolicy::PAIRED_COMPLEX_FP64_HEAD, false);
    const auto cost = estimateMixedComplexScoreCost(plan);
    auto score = PreparedDeviceComplexScore::prepareMixed(cache, queue, blas, cost.device_bytes);
    auto reference = PreparedDeviceComplexScore::prepare(image, queue, blas, estimateComplexScoreCost(plan).device_bytes);
    score->evaluate(first); reference->evaluate(first);
    std::vector<cuDoubleComplex> actual(plan.parameterCount()), expected(plan.parameterCount());
    ComplexValueResult amplitude, reference_amplitude;
    score->download(score->resultView(), actual, amplitude);
    reference->download(reference->resultView(), expected, reference_amplitude);
    double difference = 0, norm = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
      const auto a = std::complex<double>(cuCreal(actual[i]), cuCimag(actual[i]));
      const auto e = std::complex<double>(cuCreal(expected[i]), cuCimag(expected[i]));
      require(device_math::isFiniteBinary64(a.real()) && device_math::isFiniteBinary64(a.imag()) &&
          std::abs(a-e) <= 5e-4 + 1e-3*std::abs(e), "Canonical paired score component exceeded its frozen bound");
      difference += std::norm(a-e); norm += std::norm(e);
    }
    require(std::sqrt(difference) <= 1e-3 * std::sqrt(norm), "Canonical paired score exceeded frozen relative L2");
    const auto warm = metrics.snapshot();
    score->evaluate(first);
    const auto after = metrics.snapshot();
    require(warm.allocation_calls == after.allocation_calls && warm.free_calls == after.free_calls &&
        warm.live_bytes == after.live_bytes, "Canonical paired score allocated after warming");
    std::cout << "PASS canonical paired score all_components=bounded relative_L2=bounded warm_DeviceBuffer_delta=0\n";
  }
}
} // namespace qmcplusplus::psiformer::device
#endif
