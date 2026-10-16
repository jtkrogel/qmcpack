//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_TRUNK_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_KINETIC_TRUNK_VALIDATION_H
#include "PsiFormerDeviceKineticTrunk.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerKineticExecutor.h"

namespace qmcplusplus::psiformer::device
{
/// Focused friend adapter; private tickets and mutable seeds never enter production APIs.
struct KineticTrunkValidationAccess
{
  /// Generate one signed global objective partitioned across physical target planes.
  static void seeds(const SpatialJetLayout& layout, std::vector<double>& features, std::vector<double>& cusp)
  {
    const auto n = layout.electron_count, begin = layout.target_electron_begin;
    const auto k = layout.laplacian_lanes, u = layout.element_count;
    features.resize(layout.plane_count * u);
    cusp.resize(layout.plane_count);
    for (std::size_t plane = 0; plane < layout.plane_count; ++plane)
    {
      const auto global = !plane ? 0 : (plane <= 3 * k ? plane + 3 * begin : plane - 3 * k + 3 * n + begin);
      const double weight = plane ? 1 : double(k) / n;
      cusp[plane] = weight * 0.031 * std::cos(double(global + 1));
      for (std::size_t i = 0; i < u; ++i)
        features[plane * u + i] = weight * 0.013 * std::sin(double(global * u + i + 1));
    }
  }

  /// Compare independent CPU FULL roots with the same real graph at each admitted K.
  static void run(pf::PsiFormer& model, const std::shared_ptr<const PreparedDeviceModel>& image,
      compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
  {
    const auto& plan = image->plan();
    const auto& shape = plan.modelShape();
    const auto n = shape.electrons(), f = shape.feature_dimension, p = plan.parameterCount(), u = n * f;
    const auto layout = makeSpatialDenseJetLayout(1, n, f, f, n, SpatialJetMode::FULL_VGL).source;
    const std::vector<double> positions(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * n);
    std::vector<double> full_features, full_cusp;
    seeds(layout, full_features, full_cusp);
    pf::DirectKineticExecutor cpu(model, plan, pf::DirectKineticPreparation::REAL_TRUNK_ONLY);
    const auto cpu_bytes = pf::kineticTrunkWorkspaceStorageRequirement({n, shape.nuclei, shape.determinants, f,
        shape.attention_heads, plan.parameter(ParameterRole::ELECTRON_EMBEDDING_WEIGHT).shape[0], shape.attention_blocks, p});
    auto workspace = cpu.makeTrunkWorkspace(cpu_bytes);
    workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), n));
    const auto tape = cpu.evaluateTrunk(*workspace);
    pf::DirectTraceJetBuffer feature_seed(u, 3 * n, n), cusp_seed(1, 3 * n, n);
    std::copy_n(full_features.data(), u, feature_seed.value.data());
    std::copy_n(full_features.data() + u, 3 * n * u, feature_seed.gradient.data());
    std::copy_n(full_features.data() + (1 + 3 * n) * u, n * u, feature_seed.laplacian.data());
    cusp_seed.value[0] = full_cusp[0];
    std::copy_n(full_cusp.data() + 1, 3 * n, cusp_seed.gradient.data());
    std::copy_n(full_cusp.data() + 1 + 3 * n, n, cusp_seed.laplacian.data());
    const auto cpu_result = cpu.reverseTrunk(*workspace, tape, feature_seed, cusp_seed);
    std::vector<double> expected(cpu_result.begin(), cpu_result.end()), actual(p);
    const auto head_parameter = plan.parameter(ParameterRole::BACKFLOW_UP_WEIGHT).begin;
    expected[head_parameter] += 0.37; // A preseeded head response must survive the trunk tail.

    std::vector<double> ordinary_features(full_features.size());
    if (plan.environment().amplitude_scalar_domain == ScalarDomain::REAL)
    {
      PreparedDeviceInference ordinary(image, 1, queue, blas, SpatialJetMode::FULL_VGL, false, true);
      ordinary.evaluate(positions);
      ordinary.a_.download(ordinary_features.data(), ordinary_features.size());
    }
    else
    {
      std::copy(tape.features->value.begin(), tape.features->value.end(), ordinary_features.begin());
      std::copy(tape.features->gradient.begin(), tape.features->gradient.end(), ordinary_features.begin() + u);
      std::copy(tape.features->laplacian.begin(), tape.features->laplacian.end(), ordinary_features.begin() + (1 + 3 * n) * u);
    }
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "kinetic trunk metric device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "kinetic trunk metric device");
#endif
    auto& metrics = deviceMemoryMetrics(ordinal);
    double max_error = 0, max_prefix = 0;
    std::size_t checks = 0;
    for (auto k : {std::size_t(1), std::min(std::size_t(3), n), n})
    {
      const auto cost = estimateKineticTrunkCost(plan, k);
      auto before = metrics.snapshot();
      bool failed = false;
      try { (void)PreparedDeviceKineticTrunk::prepare(image, k, cost.device_bytes - 1, queue, blas); }
      catch (const std::length_error&) { failed = true; }
      if (!failed || metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Kinetic trunk budget rejection allocated");
      auto gpu = PreparedDeviceKineticTrunk::prepare(image, k, cost.device_bytes, queue, blas);
      if (metrics.snapshot().live_bytes - before.live_bytes != cost.device_bytes ||
          gpu->forward_.orbital_jets_.capacity() || gpu->forward_.factors_.capacity() ||
          gpu->forward_.normalized_.capacity() || gpu->reverse_.factor_.capacity() || gpu->reverse_.orbital_.capacity())
        throw std::runtime_error("Head-free kinetic trunk bytes or omitted buffers differ");
      std::vector<double> feature_host, cusp_host, prefix_host;
      feature_host.reserve(full_features.size()); cusp_host.reserve(full_cusp.size()); prefix_host.reserve(full_features.size());
      PreparedDeviceInference::KineticTrunkTicket saved{};
      bool inject = false;
      double objective_scale = 1;
      const auto produce = [&](const auto& ticket, double* feature, double* cusp, double* response) {
        ticket.owner->validateKineticTrunk(ticket);
        auto forged = ticket;
        ++forged.cusp.laplacian_lanes;
        bool forged_rejected = false;
        try { gpu->reverse_.accumulateTrunkSeeds(forged); }
        catch (const std::logic_error&) { forged_rejected = true; }
        if (!forged_rejected)
          throw std::runtime_error("Forged cusp Laplacian lanes admitted a trunk reverse");
        if (feature != gpu->reverse_.feature_.data() || cusp != gpu->reverse_.root_adjoint_.data() ||
            response != gpu->reverse_.response_.data())
          throw std::runtime_error("Private kinetic seed destination differs");
        if (ticket.ordinal && saved.ordinal != ticket.ordinal)
        {
          bool stale = false;
          try { ticket.owner->validateKineticTrunk(saved); }
          catch (const std::logic_error&) { stale = true; }
          if (!stale) throw std::runtime_error("Previous physical tile ticket survived mutation");
        }
        saved = ticket;
        seeds(ticket.features, feature_host, cusp_host);
        for (auto& value : feature_host) value *= objective_scale;
        for (auto& value : cusp_host) value *= objective_scale;
        prefix_host.resize(feature_host.size());
        gpu->forward_.a_.download(prefix_host.data(), prefix_host.size());
        const auto begin = ticket.features.target_electron_begin, count = ticket.features.laplacian_lanes;
        for (std::size_t plane = 0; plane < ticket.features.plane_count; ++plane)
        {
          const auto global = !plane ? 0 : (plane <= 3 * count ? plane + 3 * begin : plane - 3 * count + 3 * n + begin);
          for (std::size_t i = 0; i < u; ++i)
            max_prefix = std::max(max_prefix, std::abs(prefix_host[plane * u + i] - ordinary_features[global * u + i]));
        }
        if (inject && begin + count == n)
          feature_host[0] = std::numeric_limits<double>::quiet_NaN();
        gpu->reverse_.feature_.upload(feature_host.data(), feature_host.size());
        gpu->reverse_.root_adjoint_.upload(cusp_host.data(), cusp_host.size());
        if (!ticket.ordinal)
        {
          const double head = 0.37 * objective_scale;
#ifdef QMC_CUDA2HIP
          checkRuntime(hipMemcpy(response + head_parameter, &head, sizeof(double), hipMemcpyHostToDevice), "trunk head sentinel");
#else
          checkRuntime(cudaMemcpy(response + head_parameter, &head, sizeof(double), cudaMemcpyHostToDevice), "trunk head sentinel");
#endif
        }
      };
      const auto evaluate = [&] {
        gpu->evaluate(positions, produce);
        gpu->reverse_.downloadResponse(actual);
        for (std::size_t i = 0; i < p; ++i)
        {
          max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
          if (!device_math::isFiniteBinary64(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-11 * (1 + std::abs(expected[i])))
            throw std::runtime_error("Arbitrary kinetic trunk response differs from CPU FULL oracle");
        }
        ++checks;
      };
      evaluate();
      before = metrics.snapshot();
      evaluate();
      if (metrics.snapshot().allocation_calls != before.allocation_calls || metrics.snapshot().free_calls != before.free_calls)
        throw std::runtime_error("Warmed kinetic trunk allocated/freed DEVICE storage");
      std::vector<double> first_objective(p);
      before = metrics.snapshot();
      gpu->withTargets(positions, [&](const auto& ticket) {
        const auto generation = gpu->forward_.evaluationGeneration();
        for (std::size_t component = 0; component < 2; ++component)
        {
          objective_scale = component ? -0.7 : 1;
          gpu->reverseSeeded(ticket, produce, true);
          gpu->finishResponse();
          if (gpu->response() != gpu->reverse_.response_.data() || generation != gpu->forward_.evaluationGeneration())
            throw std::runtime_error("Two trunk objectives did not share the same actual forward ticket");
          gpu->reverse_.downloadResponse(actual);
          if (!component) first_objective = actual;
          else
            for (std::size_t i = 0; i < p; ++i)
              if (std::abs(actual[i] + 0.7 * first_objective[i]) > 2e-11 * (1 + std::abs(first_objective[i])))
                throw std::runtime_error("Second signed objective differs on the same kinetic K tape");
        }
      });
      objective_scale = 1;
      if (metrics.snapshot().allocation_calls != before.allocation_calls || metrics.snapshot().free_calls != before.free_calls)
        throw std::runtime_error("Two objectives on one K tape allocated/freed DEVICE storage");
      failed = false;
      try { gpu->reverse_.accumulateTrunkSeeds(saved); }
      catch (const std::logic_error&) { failed = true; }
      if (!failed) throw std::runtime_error("Completed kinetic trunk ticket survived");
      inject = true;
      failed = false;
      try { gpu->evaluate(positions, produce); }
      catch (const std::runtime_error&) { failed = true; }
      if (!failed) throw std::runtime_error("Late nonfinite kinetic trunk seed admitted");
      failed = false;
      try { gpu->reverse_.downloadResponse(actual); }
      catch (const std::logic_error&) { failed = true; }
      if (!failed) throw std::runtime_error("Failed kinetic trunk published a partial P");
      inject = false;
      evaluate();
      const auto successor = image->makeUpdatedImage(model.p.flat_values(), image->version() + 1);
      before = metrics.snapshot();
      gpu->rebindModel(successor);
      if (metrics.snapshot().allocation_calls != before.allocation_calls)
        throw std::runtime_error("Kinetic trunk successor allocated");
      failed = false;
      try { gpu->reverse_.accumulateTrunkSeeds(saved); }
      catch (const std::logic_error&) { failed = true; }
      if (!failed) throw std::runtime_error("Old-image kinetic trunk ticket survived rebind");
      evaluate();
      std::cout << "Kinetic trunk K=" << k << " exact_device_bytes=" << cost.device_bytes
          << " host_bytes=" << cost.host_bytes << " canonical_bytes=" << 8 * p << '\n';
    }
    if (max_prefix > 2e-11)
      throw std::runtime_error("Head-free kinetic prefix differs from ordinary real FULL graph");
    std::cout << "PASS private kinetic trunk: checks=" << checks << " max_CPU_response_error=" << max_error
        << " max_real_FULL_prefix_error=" << max_prefix
        << " signed_feature_cusp=1 head_P_preserved=1 two_objectives_same_ticket=1 warm_DEVICE_allocations=0 scope=device_wide\n";
  }
};
} // namespace qmcplusplus::psiformer::device
#endif
