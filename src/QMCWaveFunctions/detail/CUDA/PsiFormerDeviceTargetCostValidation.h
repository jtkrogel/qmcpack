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
inline PsiFormerExecutionPlan makeTargetCostPlan(std::size_t electrons, bool periodic)
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
    append(prefix + "omni_net/~/" + backflow + "/~/mlp/linear_0", "w", {shape.feature_dimension, channels});
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

/** Emit CSV predictions and exercise per-owner byte boundaries with no GPU calls.
 * SIZE_MAX work limits isolate memory feasibility; quartic scalar-reference work
 * is still reported and is not a production runtime/throughput qualification.
 */
inline void validateTargetCosts()
{
  const auto before = deviceMemoryMetrics(0).snapshot(); // Pure host counters; no device query.
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
