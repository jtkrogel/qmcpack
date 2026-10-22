//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceTargetCostValidation.h
 * @brief Metadata-only large-shape target cost and per-owner admission checks.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_COST_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_COST_VALIDATION_H

#include "PsiFormerDeviceExecutionPreparation.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer::device
{

/** Generate the existing real canonical layout without creating parameter values.
 * Names/shapes match the small generated integration fixture; only this vector
 * of roughly forty tensor descriptors is allocated at the large physical shape.
 */
inline PsiFormerExecutionPlan makeTargetCostPlan(std::size_t electrons, bool periodic, bool paired = false)
{
  const ModelShape shape{electrons / 2, electrons - electrons / 2, electrons / 5, 16, 256, 4, 4};
  ExecutionEnvironment environment;
  if (periodic)
  {
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.periodic_axes = {true, true, true};
    environment.lattice_vectors = {{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}};
  }
  if (paired)
  {
    environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
  }
  std::vector<ParameterLayoutInput> layouts;
  const auto append = [&](std::string module, std::string name, std::vector<std::size_t> dimensions) {
    std::size_t extent = 1;
    for (auto dimension : dimensions)
      extent = spatial_detail::checkedProduct(extent, dimension, "Target cost fixture tensor overflow");
    const auto begin = layouts.empty() ? 0 : layouts.back().end;
    const auto end = spatial_detail::checkedAdd(begin, extent, "Target cost fixture offset overflow");
    layouts.push_back({std::move(module), std::move(name), std::move(dimensions), begin, end});
  };
  const std::string prefix = "neural_network_wave_function/~/";
  append(prefix + "electronic_cusp_asymptotic", "anti_alpha", {});
  append(prefix + "electronic_cusp_asymptotic", "same_alpha", {});
  const auto channels = spatial_detail::checkedProduct(shape.determinants, electrons, "Target cost head overflow");
  for (const auto* name : {"pi_down", "pi_up", "zetas_down", "zetas_up"})
    append(prefix + "exponential_envelopes", name, {channels, shape.nuclei});
  for (const auto* backflow : {"Backflow", "Backflow_1"})
  {
    append(prefix + "omni_net/~/" + backflow + "/~/mlp/linear_0", "w", {shape.feature_dimension, channels});
    if (paired)
      append(prefix + "omni_net/~/" + backflow + "/~/mlp/linear_0", "w_imag", {shape.feature_dimension, channels});
  }
  append(prefix + "omni_net/~/electron_gnn/~/electron_embedding/linear", "w",
      {(periodic ? 7 : 4) * shape.nuclei + 1, shape.feature_dimension});
  for (std::size_t block = 0; block < shape.attention_blocks; ++block)
  {
    const auto layer = block == 0 ? std::string("electron_gnn_layer") : "electron_gnn_layer_" + std::to_string(block);
    const auto base = prefix + "omni_net/~/electron_gnn/~/" + layer + "/~/node_attention_electron_update_feature/";
    for (const auto* linear : {"linear_0", "linear_1"})
    {
      append(base + "mlp/" + linear, "b", {shape.feature_dimension});
      append(base + "mlp/" + linear, "w", {shape.feature_dimension, shape.feature_dimension});
    }
    for (const auto* projection : {"key", "linear", "query", "value"})
      append(base + "multi_head_attention/" + projection, "w", {shape.feature_dimension, shape.feature_dimension});
  }
  return PsiFormerExecutionPlan(shape, std::move(layouts), environment);
}

/// Check feasibility through the production descriptor seam without an owner.
inline bool admitTargetCost(const PsiFormerExecutionPlan& plan, std::size_t targets,
                             std::size_t bytes, bool kinetic, std::size_t slots = 0)
{
  const auto work = std::numeric_limits<std::size_t>::max();
  DeviceExecutionPreparation preparation;
  if (kinetic)
    preparation.kinetic = {targets, bytes, work, slots};
  else
    preparation.vgl = {targets, bytes, work};
  try
  {
    validateDeviceExecutionPreparation(plan, preparation);
    return true;
  }
  catch (const std::length_error&)
  {
    return false;
  }
}

/// Exercise bounded work/memory selection and actual-B admission without numeric owners.
inline void validateTargetSelectionCosts()
{
  const auto before = deviceMemoryMetrics(0).snapshot(); // Pure host counters, no runtime query.
  const auto invalid = [](auto&& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Target selection accepted invalid metadata");
  };
  const auto infeasible = [](auto&& operation) {
    try { operation(); }
    catch (const std::length_error&) { return; }
    throw std::runtime_error("Target selection accepted infeasible metadata");
  };
  if (parseDeviceTargetSelection(nullptr) != std::make_pair(false, false) ||
      parseDeviceTargetSelection("off") != std::make_pair(false, false) ||
      parseDeviceTargetSelection("both") != std::make_pair(true, true) ||
      parseDeviceTargetSelection("vgl") != std::make_pair(true, false) ||
      parseDeviceTargetSelection("kinetic") != std::make_pair(false, true))
    throw std::runtime_error("Target selection policy parsing changed");
  invalid([&] { (void)parseDeviceTargetSelection(""); });
  invalid([&] { (void)parseDeviceTargetSelection("fastest"); });
  std::vector<std::size_t> visited;
  const auto synthetic = [&](std::size_t k) {
    visited.push_back(k);
    return TargetVglCost{k, k, 100 - k, 1};
  };
  if (selectDeviceTargetCapacity(7, 99, 99, synthetic) != 7 || visited != std::vector<std::size_t>{1, 2, 4, 7})
    throw std::runtime_error("Target selection candidate filtering or deterministic rank changed");
  visited.clear();
  if (selectDeviceTargetCapacity(4, 99, 99, synthetic) != 4 || visited != std::vector<std::size_t>{1, 2, 4})
    throw std::runtime_error("Target selection duplicated the full target interval");
  if (selectDeviceTargetCapacity(7, 100, 10, [](std::size_t k) {
        return TargetVglCost{k == 2 ? std::size_t(1) : std::size_t(2), k == 2 ? std::size_t(99) : 0, 5, 1};
      }) != 2 ||
      selectDeviceTargetCapacity(7, 100, 10, [](std::size_t k) {
        return TargetVglCost{10, k == 4 ? std::size_t(1) : std::size_t(2), 5, 1};
      }) != 4 ||
      selectDeviceTargetCapacity(7, 100, 10, [](std::size_t) {
        return TargetVglCost{10, 2, 5, 1};
      }) != 1)
    throw std::runtime_error("Target selection work tie did not rank by device bytes, host bytes, then K");
  if (selectDeviceTargetCapacity(7, 99, 99, [](std::size_t k) {
        if (k == 2)
          throw std::length_error("Synthetic single-candidate extent overflow");
        return TargetVglCost{10, 0, 100 - k, 1};
      }) != 7)
    throw std::runtime_error("Target selection failed to skip one overflowing candidate while another fits");
  for (const auto limit : {std::size_t(0), std::numeric_limits<std::size_t>::max()})
  {
    invalid([&] { (void)selectDeviceTargetCapacity(4, limit, 99, synthetic); });
    invalid([&] { (void)selectDeviceTargetCapacity(4, 99, limit, synthetic); });
  }
  infeasible([&] { (void)selectDeviceTargetCapacity(4, 99, 99, [](std::size_t) -> TargetVglCost {
    throw std::length_error("Synthetic checked extent overflow");
  }); });
  invalid([&] { (void)selectDeviceTargetCapacity(4, 99, 99, [](std::size_t) -> TargetVglCost {
    throw std::invalid_argument("Synthetic structural mismatch");
  }); });
  for (bool periodic : {false, true})
    for (const std::size_t n : {std::size_t(100), std::size_t(200), std::size_t(300)})
    {
      const auto plan = makeTargetCostPlan(n, periodic);
      const auto vgl = estimateTargetVglCost(plan, 1, 32);
      const auto kinetic = estimateTargetKineticCost(plan, 32);
      DeviceExecutionPreparation request;
      request.select_vgl_targets = request.select_kinetic_targets = true;
      request.vgl = {0, vgl.device_workspace_bytes, estimateTargetVglCost(plan, 1, 1).charged_scalar_work_units};
      request.kinetic = {0, kinetic.device_workspace_bytes, estimateTargetKineticCost(plan, 1).charged_scalar_work_units, 0};
      const auto selected = resolveDeviceTargetSelection(plan, request);
      const auto repeat = resolveDeviceTargetSelection(plan, selected);
      if (selected.vgl.max_targets != 32 || selected.kinetic.max_targets != 32 ||
          repeat.vgl.max_targets != 32 || repeat.kinetic.max_targets != 32)
        throw std::runtime_error("Target selection missed a feasible moderate tile or was nondeterministic");
      validateDeviceExecutionPreparation(plan, selected);
      const auto full_vgl = estimateTargetVglCost(plan, 1, n);
      const auto full_kinetic = estimateTargetKineticCost(plan, n);
      request.vgl = {0, full_vgl.device_workspace_bytes, full_vgl.charged_scalar_work_units};
      request.kinetic = {0, full_kinetic.device_workspace_bytes, full_kinetic.charged_scalar_work_units, 0};
      const auto complete = resolveDeviceTargetSelection(plan, request);
      if (complete.vgl.max_targets != n || complete.kinetic.max_targets != n)
        throw std::runtime_error("Target selection rejected exact full-interval budgets");
      --request.vgl.scalar_work_budget; // No K has less work than the complete interval.
      infeasible([&] { (void)resolveDeviceTargetSelection(plan, request); });
      request.select_kinetic_targets = false;
      request.kinetic = {};
      const auto minimum = estimateTargetVglCost(plan, 1, 1);
      request.vgl = {0, minimum.device_workspace_bytes, minimum.charged_scalar_work_units};
      const auto one = resolveDeviceTargetSelection(plan, request);
      if (one.vgl.max_targets != 1)
        throw std::runtime_error("Target selection failed exact minimum-owner admission");
      infeasible([&] { (void)resolveDeviceTargetSelection(plan, one, 2); });
      --request.vgl.device_workspace_budget_bytes;
      infeasible([&] { (void)resolveDeviceTargetSelection(plan, request); });
      request.select_vgl_targets = false;
      request.vgl = {3, 1, 1};
      if (resolveDeviceTargetSelection(plan, request, 0).vgl.max_targets != 3)
        throw std::runtime_error("Unset selector changed an explicit noncandidate K");
    }
  // The optional cusp cache is charged before graph reserves, including its status owner.
  for (const bool periodic : {false, true})
  {
    const auto cusp_plan = makeTargetCostPlan(100, periodic);
    TargetVglPreparation reference{1, 0, 0, 0, false};
    auto cached = reference;
    cached.reuse_cusp_state = true;
    const auto original = estimatePreparedTargetVglCost(cusp_plan, 4, reference);
    const auto candidate = estimatePreparedTargetVglCost(cusp_plan, 4, cached);
    const auto extra = checkedDeviceBytes<double>(4 * 401) + checkedDeviceBytes<OpenSpatialStatus>(4);
    if (candidate.device_workspace_bytes != original.device_workspace_bytes + extra ||
        candidate.host_workspace_bytes != original.host_workspace_bytes ||
        candidate.charged_scalar_work_units != original.charged_scalar_work_units)
      throw std::runtime_error("Cusp state cache preflight charge disagrees with exact owner storage");
    cached.max_targets = reference.max_targets = 100;
    if (estimatePreparedTargetVglCost(cusp_plan, 4, cached).device_workspace_bytes !=
        estimatePreparedTargetVglCost(cusp_plan, 4, reference).device_workspace_bytes)
      throw std::runtime_error("Full-target VGL charged an unused cusp cache");
  }
  // The same whole-owner budget selects K2 at B1 but only K1 at actual B2.
  const auto plan = makeTargetCostPlan(100, false);
  DeviceExecutionPreparation parallel;
  parallel.select_vgl_targets = true;
  parallel.vgl = {1, 0, 0, 1024 * 1024};
  const auto minimum_b2 = estimatePreparedTargetVglCost(plan, 2, parallel.vgl);
  parallel.vgl = {0, minimum_b2.device_workspace_bytes, minimum_b2.charged_scalar_work_units,
                  parallel.vgl.parallel_trace_workspace_budget_bytes};
  const auto b1 = resolveDeviceTargetSelection(plan, parallel, 1);
  const auto b2 = resolveDeviceTargetSelection(plan, parallel, 2);
  if (b1.vgl.max_targets != 2 || b2.vgl.max_targets != 1 ||
      resolveDeviceTargetSelection(plan, b2, 2).vgl.max_targets != 1)
    throw std::runtime_error("Parallel selection leaked its B1 capacity or was nondeterministic");
  validateDeviceExecutionPreparation(plan, b1, 1);
  validateDeviceExecutionPreparation(plan, b2, 2);
  infeasible([&] { validateDeviceExecutionPreparation(plan, b1, 2); });
  auto denied = parallel;
  --denied.vgl.device_workspace_budget_bytes;
  infeasible([&] { (void)resolveDeviceTargetSelection(plan, denied, 2); });
  denied = parallel;
  denied.select_kinetic_targets = true;
  invalid([&] { (void)resolveDeviceTargetSelection(plan, denied); });
  invalid([&] { validateDeviceExecutionPreparation(plan, denied); });
  denied.select_vgl_targets = false;
  denied.vgl.max_targets = 1;
  invalid([&] { (void)resolveDeviceTargetSelection(plan, denied); });
  invalid([&] { validateDeviceExecutionPreparation(plan, denied); });
  denied = parallel;
  denied.active_inference_crowds = 2;
  invalid([&] { (void)resolveDeviceTargetSelection(plan, denied); });
  invalid([&] { validateDeviceExecutionPreparation(plan, denied); });
  // Geometry is consumed by the orbital jets, not by the bounded trace leaf.
  // Periodic real admission must still charge every slot before owner creation.
  const auto periodic_plan = makeTargetCostPlan(100, true);
  auto periodic = parallel;
  periodic.vgl = {1, 0, 0, 1024 * 1024};
  const auto periodic_cost = estimatePreparedTargetVglCost(periodic_plan, 2, periodic.vgl);
  const auto periodic_base = estimateTargetVglCost(periodic_plan, 2, 1);
  const auto open_base = estimateTargetVglCost(plan, 2, 1);
  if (periodic_cost.device_workspace_bytes - periodic_base.device_workspace_bytes !=
      minimum_b2.device_workspace_bytes - open_base.device_workspace_bytes ||
      periodic_cost.charged_scalar_work_units != minimum_b2.charged_scalar_work_units)
    throw std::runtime_error("Periodic parallel traces changed geometry-independent slot or work charging");
  periodic.vgl.device_workspace_budget_bytes = periodic_cost.device_workspace_bytes;
  periodic.vgl.scalar_work_budget = periodic_cost.charged_scalar_work_units;
  periodic.select_vgl_targets = false;
  validateDeviceExecutionPreparation(periodic_plan, periodic, 2);
  --periodic.vgl.device_workspace_budget_bytes;
  infeasible([&] { validateDeviceExecutionPreparation(periodic_plan, periodic, 2); });
  invalid([&] { (void)resolveDeviceTargetSelection(makeTargetCostPlan(100, true, true), parallel); });
  invalid([&] { (void)resolveDeviceTargetSelection(makeTargetCostPlan(100, false, true), parallel); });
  std::cout << "# PASS parallel_composite_selection exact_B2_budget B1K2_B2K1 capacity_denial profiles\n";
  const auto after = deviceMemoryMetrics(0).snapshot();
  if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
      after.live_bytes != before.live_bytes || after.upload_calls != before.upload_calls ||
      after.download_calls != before.download_calls)
    throw std::runtime_error("Metadata selection touched a numeric device owner");
  std::cout << "# PASS deterministic_work_memory_v1 metadata_selection; Ne=100,200,300; "
               "capacity_trap=guarded; no_numeric_owners\n";
}

/** Emit CSV predictions and exercise per-owner byte boundaries with no GPU calls.
 * SIZE_MAX work limits isolate memory feasibility; quartic scalar-reference work
 * is still reported and is not a production runtime/throughput qualification.
 */
inline void validateTargetCosts()
{
  const auto before = deviceMemoryMetrics(0).snapshot(); // Pure host counters; no device query.
  validateTargetSelectionCosts();
  constexpr std::size_t gib = std::size_t{1024} * 1024 * 1024;
  const auto budget40 = spatial_detail::checkedProduct(40, gib, "40GiB cost budget overflow");
  const auto budget80 = spatial_detail::checkedProduct(80, gib, "80GiB cost budget overflow");
  std::cout << "# qualification=METADATA_ONLY; B=1; real_FP64; no_parameter_values_or_device_graphs\n"
            << "# budgets=40GiB,80GiB; GiB=1073741824; work_limit=SIZE_MAX; admission_is_memory_only\n"
            << "# work_units=scalar_reference_contraction_sites_not_FLOPs_or_time\n"
            << "# owner_bytes=separate_VGL_or_retained_kinetic_buffers; model_columns=ONE_separate_immutable_image\n"
            << "# excluded=additional_images,crowds,score_JVP,solver_channels,ECP,teacher,vendor_reserve,allocator_overhead\n"
            << "# no_aggregate_process_fit_or_runtime_or_throughput_claim; K=Ne_is_full_target_interval\n"
            << "boundary,Ne,nuclei,D,F,H,L,K,P,tiles,vgl_device_bytes,vgl_host_bytes,kinetic_device_bytes,kinetic_host_bytes,"
               "one_parameter_image_bytes,shared_nuclei_bytes,vgl_work,kinetic_work,vgl_fit40,vgl_fit80,kinetic_fit40,kinetic_fit80\n";
  for (bool periodic : {false, true})
    for (std::size_t ne : {std::size_t(100), std::size_t(200), std::size_t(300)})
    {
      const auto plan = makeTargetCostPlan(ne, periodic);
      std::size_t previous_vgl = 0, previous_kinetic = 0;
      for (auto targets : {std::size_t(1), std::size_t(2), std::size_t(4), std::size_t(8), ne})
      {
        const auto vgl = estimateTargetVglCost(plan, 1, targets);
        const auto kinetic = estimateTargetKineticCost(plan, targets);
        // Optional products replace exactly three matrices, never a second graph.
        for (std::size_t slots : {std::size_t(1), std::size_t(2), std::size_t(16)})
        {
          const auto products = planLiftedDeterminantProducts(ne, slots, std::numeric_limits<std::size_t>::max());
          const auto gemm = estimateTargetKineticCost(plan, targets, slots);
          const auto expected = spatial_detail::checkedAdd(
              kinetic.device_workspace_bytes - checkedDeviceBytes<double>(3 * ne * ne),
              spatial_detail::checkedAdd(products.device_workspace_bytes, checkedDeviceBytes<unsigned int>(1),
                  "Parallel orbital fault storage overflow"), "Lifted target cost fixture overflow");
          if (gemm.device_workspace_bytes != expected || gemm.host_workspace_bytes != kinetic.host_workspace_bytes ||
              gemm.charged_scalar_work_units != kinetic.charged_scalar_work_units ||
              !admitTargetCost(plan, targets, expected, true, slots) ||
              admitTargetCost(plan, targets, expected - 1, true, slots))
            throw std::runtime_error("Lifted target cost replacement or exact-byte admission failed");
        }
        if (vgl.device_workspace_bytes <= previous_vgl || kinetic.device_workspace_bytes <= previous_kinetic ||
            kinetic.tile_count != ne / targets + (ne % targets != 0))
          throw std::runtime_error("Target cost capacity monotonicity or tile coverage failed");
        previous_vgl = vgl.device_workspace_bytes;
        previous_kinetic = kinetic.device_workspace_bytes;
        for (bool reverse : {false, true})
        {
          const auto exact = reverse ? kinetic.device_workspace_bytes : vgl.device_workspace_bytes;
          if (!admitTargetCost(plan, targets, exact, reverse) || admitTargetCost(plan, targets, exact - 1, reverse))
            throw std::runtime_error("Target cost exact-byte admission boundary failed");
        }
        std::cout << (periodic ? "periodic_gamma" : "open") << ',' << ne << ',' << plan.modelShape().nuclei
                  << ",16,256,4,4," << targets << ',' << plan.parameterCount() << ',' << kinetic.tile_count
                  << ',' << vgl.device_workspace_bytes << ',' << vgl.host_workspace_bytes
                  << ',' << kinetic.device_workspace_bytes << ',' << kinetic.host_workspace_bytes
                  << ',' << checkedDeviceBytes<double>(plan.parameterCount())
                  << ',' << checkedDeviceBytes<double>(3 * plan.modelShape().nuclei)
                  << ',' << vgl.charged_scalar_work_units << ',' << kinetic.charged_scalar_work_units
                  << ',' << admitTargetCost(plan, targets, budget40, false)
                  << ',' << admitTargetCost(plan, targets, budget80, false)
                  << ',' << admitTargetCost(plan, targets, budget40, true)
                  << ',' << admitTargetCost(plan, targets, budget80, true) << '\n';
      }
      bool invalid_range = false, overflow = false;
      try { (void)estimateTargetKineticCost(plan, ne + 1); }
      catch (const std::invalid_argument&) { invalid_range = true; }
      try { (void)estimateTargetVglCost(plan, std::numeric_limits<std::size_t>::max(), 1); }
      catch (const std::length_error&) { overflow = true; }
      if (!invalid_range || !overflow)
        throw std::runtime_error("Target cost invalid-range or checked-overflow guard failed");
      bool invalid_slots = false, missing_targets = false;
      try { (void)estimateTargetKineticCost(plan, 1, 17); }
      catch (const std::invalid_argument&) { invalid_slots = true; }
      DeviceExecutionPreparation invalid;
      invalid.kinetic.lifted_channel_slots = 1;
      try { validateDeviceExecutionPreparation(plan, invalid); }
      catch (const std::invalid_argument&) { missing_targets = true; }
      if (!invalid_slots || !missing_targets)
        throw std::runtime_error("Lifted target channel capacity/admission guard failed");
    }
  const auto after = deviceMemoryMetrics(0).snapshot();
  if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
      after.live_bytes != before.live_bytes || after.upload_calls != before.upload_calls ||
      after.download_calls != before.download_calls)
    throw std::runtime_error("Metadata-only target cost check touched a numeric device owner");
  std::cout << "# PASS metadata_only_target_costs rows=30; observed_device_allocations=0; uploads=0; downloads=0\n";
}

} // namespace qmcplusplus::psiformer::device
#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_COST_VALIDATION_H
