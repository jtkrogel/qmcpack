//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceTargetKineticValidation.h
 * @brief Small retained-target checks shared by existing open and periodic fixtures.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_VALIDATION_H

#include "PsiFormerDeviceTargetKinetic.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"

#include <iostream>
#include <limits>

namespace qmcplusplus::psiformer::device
{

/// All reference/full owners here are deliberately confined to tiny test scopes.
inline void validateTargetKinetic(pf::PsiFormer& model, const PsiFormerExecutionPlan& plan,
    const std::shared_ptr<const PreparedDeviceModel>& image,
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto ne = plan.modelShape().electrons(), parameters = plan.parameterCount();
  const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * ne);
  std::vector<double> masses(ne), drift(3 * ne), imaginary_drift(3 * ne);
  for (std::size_t electron = 0; electron < ne; ++electron)
    masses[electron] = 0.8 + 0.17 * electron;
  for (std::size_t lane = 0; lane < 3 * ne; ++lane)
  {
    drift[lane] = 0.03 * (static_cast<int>(lane % 7) - 3);
    imaginary_drift[lane] = -0.07 + 0.011 * lane;
  }
  double max_difference = 0.0;
  const auto compare = [&](const std::vector<double>& actual, const std::vector<double>& expected,
                           double weight, const char* operation) {
    for (std::size_t parameter = 0; parameter < parameters; ++parameter)
    {
      max_difference = std::max(max_difference, std::abs(actual[parameter] - weight * expected[parameter]));
      if (!device_math::isFiniteBinary64(actual[parameter]) ||
          std::abs(actual[parameter] - weight * expected[parameter]) >
              2e-7 * (1 + std::abs(weight * expected[parameter])))
        throw std::runtime_error(std::string("Target kinetic ") + operation + " parameter mismatch");
    }
  };
  pf::DirectKineticExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), ne));
  const auto own = cpu.evaluate(*workspace);
  const std::vector<double> own_reference(own.kinetic_parameter_response.begin(), own.kinetic_parameter_response.end());
  const auto real = cpu.evaluate(*workspace, drift.data(), drift.size(), masses.data(), masses.size());
  const std::vector<double> real_reference(real.kinetic_parameter_response.begin(), real.kinetic_parameter_response.end());
  const auto imaginary = cpu.evaluateGradientParameterResponse(*workspace, imaginary_drift.data(), imaginary_drift.size(),
      masses.data(), masses.size());
  const std::vector<double> imaginary_reference(imaginary.begin(), imaginary.end());

  PreparedDeviceInference full(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true);
  PreparedDeviceKineticReverse full_reverse(image, 1, queue, blas);
  full.evaluate(positions);
  full_reverse.evaluate(full, {1.0}, masses, drift);
  std::vector<double> full_reference(parameters), response(parameters);
  full_reverse.downloadResponse(full_reference);
  compare(full_reference, real_reference, 1.0, "independent CPU/full reference");

  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "target kinetic metrics device");
#else
  checkRuntime(cudaGetDevice(&device), "target kinetic metrics device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  for (std::size_t slots : {std::size_t(0), std::size_t(1), std::size_t(2)})
  {
    for (std::size_t targets : {std::size_t(1), std::min(std::size_t(2), ne), std::min(std::size_t(3), ne), ne})
    {
      const auto cost = estimateTargetKineticCost(plan, targets, slots);
      const TargetKineticPreparation preparation{targets, cost.device_workspace_bytes, cost.charged_scalar_work_units, slots};
      for (bool device_limit : {true, false})
      {
        auto rejected = preparation;
        if (device_limit)
          --rejected.device_workspace_budget_bytes;
        else
          --rejected.scalar_work_budget;
        const auto before = metrics.snapshot();
        bool failed = false;
        try { PreparedDeviceTargetKinetic impossible(image, rejected, queue, blas); }
        catch (const std::length_error&) { failed = true; }
        const auto after = metrics.snapshot();
        if (!failed || after.allocation_calls != before.allocation_calls || after.live_bytes != before.live_bytes)
          throw std::runtime_error("Target kinetic budget rejection allocated a workspace");
      }
      const auto before_prepare = metrics.snapshot();
      PreparedDeviceTargetKinetic target(image, preparation, queue, blas);
      const auto after_prepare = metrics.snapshot();
      if (target.deviceWorkspaceBytes() != cost.device_workspace_bytes ||
          target.hostWorkspaceBytes() != cost.host_workspace_bytes ||
          after_prepare.live_bytes - before_prepare.live_bytes != cost.device_workspace_bytes)
        throw std::runtime_error("Target kinetic estimated and owned persistent storage differ");
      std::cout << "Target kinetic capacity: K=" << targets
                << " lifted_channel_slots=" << slots
                << " predicted_device_bytes=" << cost.device_workspace_bytes
                << " actual_device_bytes=" << target.deviceWorkspaceBytes()
                << " predicted_host_bytes=" << cost.host_workspace_bytes
                << " actual_host_bytes=" << target.hostWorkspaceBytes()
                << " reference_plus_target_owned_device_bytes="
                << full.deviceWorkspaceBytes() + full_reverse.deviceWorkspaceBytes() + target.deviceWorkspaceBytes()
                << '\n';
      for (double weight : {0.7, -0.25, 0.0})
      {
        target.evaluate(positions, {weight}, masses, drift);
        target.downloadResponse(response);
        compare(response, real_reference, weight, "complete real drift/masses");
        compare(response, full_reference, weight, "full GPU parity");
        target.evaluateGradientResponse(positions, {weight}, masses, imaginary_drift);
        target.downloadResponse(response);
        compare(response, imaginary_reference, weight, "complete imaginary drift/masses");
      }
      target.evaluate(positions, {1.0});
      target.downloadResponse(response);
      compare(response, own_reference, 1.0, "unit masses/component drift");

      const auto generation = target.evaluationGeneration();
      const auto before = metrics.snapshot();
      target.evaluate(positions, {0.7}, masses, drift);
      const auto after = metrics.snapshot();
      if (target.evaluationGeneration() != generation + 1 || target.admittedTargetTiles() != cost.tile_count ||
          after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
          after.live_bytes != before.live_bytes || after.upload_calls - before.upload_calls != 4)
        throw std::runtime_error("Target kinetic warmed request changed capacity, coverage or single-upload contract");

      auto bad_positions = positions;
      std::copy_n(model.cfg.nuclei.x.data(), 3, bad_positions.data());
      bool failed = false, unavailable = false;
      try { target.evaluate(bad_positions, {1.0}); }
      catch (const std::runtime_error&) { failed = true; }
      try { target.downloadResponse(response); }
      catch (const std::logic_error&) { unavailable = true; }
      if (!failed || !unavailable)
        throw std::runtime_error("Target kinetic failure exposed an old/partial response");
      target.evaluate(positions, {1.0}, masses, drift);
      target.downloadResponse(response);
      compare(response, real_reference, 1.0, "failure recovery");

      const auto successor = image->makeUpdatedImage(model.p.flat_values(), image->version() + 1);
      target.rebindModel(successor);
      unavailable = false;
      try { target.downloadResponse(response); }
      catch (const std::logic_error&) { unavailable = true; }
      if (!unavailable)
        throw std::runtime_error("Target kinetic rebind exposed an old response");
      target.evaluate(positions, {1.0}, masses, drift);
      target.downloadResponse(response);
      compare(response, real_reference, 1.0, "successor image");
      bool stale = false;
      try { target.rebindModel(image); }
      catch (const std::invalid_argument&) { stale = true; }
      if (!stale)
        throw std::runtime_error("Target kinetic accepted an older model version");
    }
  }

  if (ne > 1)
  {
    for (std::size_t slots : {std::size_t(0), std::size_t(1), std::size_t(2)})
    {
      const auto cost = estimateTargetKineticCost(plan, 1, slots);
      PreparedDeviceTargetKinetic target(image,
          {1, cost.device_workspace_bytes, cost.charged_scalar_work_units, slots}, queue, blas);
      // Only the final target receives a finite near-maximum drift. Earlier zero
      // gradient-only contractions succeed; lifted reverse must reject overflow
      // before that final tile can be admitted or any partial P becomes visible.
      std::vector<double> overflow_drift(3 * ne, 0.0);
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        overflow_drift[3 * (ne - 1) + dimension] = 0.5 * std::numeric_limits<double>::max();
      bool late_reverse_failed = false, partial_hidden = false;
      try { target.evaluateGradientResponse(positions, {1.0}, {}, overflow_drift); }
      catch (const std::runtime_error& error)
      {
        const std::string message(error.what());
        late_reverse_failed = target.admittedTargetTiles() == ne - 1 &&
            (message.find("Kinetic determinant response") != std::string::npos ||
             message.find("Kinetic orbital/cusp response") != std::string::npos);
      }
      try { target.downloadResponse(response); }
      catch (const std::logic_error&) { partial_hidden = true; }
      if (!late_reverse_failed || !partial_hidden)
        throw std::runtime_error("Target kinetic did not hide a genuine late reverse failure");
      target.evaluate(positions, {1.0}, masses, drift);
      target.downloadResponse(response);
      compare(response, real_reference, 1.0, "late reverse failure recovery");

      // A zero-weight singular channel still needs an inverse in lifted reverse,
      // including when the entire sample's contraction weight is zero.
      auto singular_parameters = model.p.flat_values();
      for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
      {
        const auto& tensor = plan.parameter(role);
        for (std::size_t feature = 0; feature < tensor.shape[0]; ++feature)
          std::fill_n(singular_parameters.begin() + tensor.begin + feature * tensor.shape[1], ne, 0.0);
      }
      const auto singular = image->makeUpdatedImage(singular_parameters, image->version() + 1);
      target.rebindModel(singular);
      bool inverse_rejected = false;
      try { target.evaluate(positions, {0.0}); }
      catch (const std::runtime_error&) { inverse_rejected = true; }
      if (!inverse_rejected)
        throw std::runtime_error("Target kinetic admitted a singular channel at zero sample weight");
      for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
      {
        const auto& tensor = plan.parameter(role);
        std::fill(singular_parameters.begin() + tensor.begin, singular_parameters.begin() + tensor.end, 0.0);
      }
      const auto node = singular->makeUpdatedImage(singular_parameters, singular->version() + 1);
      target.rebindModel(node);
      bool node_rejected = false;
      try { target.evaluate(positions, {0.0}); }
      catch (const std::runtime_error&) { node_rejected = true; }
      if (!node_rejected)
        throw std::runtime_error("Target kinetic admitted an exact total node at zero sample weight");
      const auto recovered = node->makeUpdatedImage(model.p.flat_values(), node->version() + 1);
      target.rebindModel(recovered);
      target.evaluate(positions, {1.0}, masses, drift);
      target.downloadResponse(response);
      compare(response, real_reference, 1.0, "singular-channel failure recovery");
    }
  }

  std::cout << "PASS retained target kinetic: scalar/GEMM C1/C2, K1/2/3/Ne, all-P CPU/full parity, signed/zero weights, "
               "physical real/imaginary drifts and masses, one upload/generation, zero warm allocations, "
               "exact combined bytes, pre-allocation budgets, late reverse failure/recovery, "
               "all-channel inverse, total-node and rebind guards; max_abs_difference=" << max_difference << '\n';
}

} // namespace qmcplusplus::psiformer::device
#endif // QMCPLUSPLUS_PSIFORMER_DEVICE_TARGET_KINETIC_VALIDATION_H
