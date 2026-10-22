//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SHARED_ROW_MEANS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SHARED_ROW_MEANS_VALIDATION_H

#include "PsiFormerDeviceDerivativeKernels.h"
#include "PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDerivativeMath.h"
#include "Platforms/QueueAliases.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Compare defined finite output bits, including signed zero, and IEEE failure classes.
inline void sharedRowMeanExact(double actual, double expected, std::size_t offset)
{
  const bool finite           = device_math::isFiniteBinary64(actual);
  const bool reference_finite = device_math::isFiniteBinary64(expected);
  if (finite != reference_finite || (finite && std::memcmp(&actual, &expected, sizeof(double))) ||
      (!finite &&
       (std::isnan(actual) != std::isnan(expected) ||
        (std::isinf(actual) && std::signbit(actual) != std::signbit(expected)))))
    throw std::runtime_error("Shared row means changed canonical GPU output at " + std::to_string(offset));
}

/// Validate borrowed BHN means without changing the old GPU bodies or CPU bound.
inline void validateSharedSoftmaxRowMeans(compute::Queue<PlatformKind::CUDA>& queue)
{
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "shared means device");
#else
  checkRuntime(cudaGetDevice(&device), "shared means device");
#endif
  const auto parser_before = deviceMemoryMetrics(device).snapshot();
  // Pure policy parsing must reject malformed values without device allocation.
  const char* selector          = "PSIFORMER_GPU_SOFTMAX_KEY_SHARED_MEANS";
  const char* prior             = std::getenv(selector);
  const bool had_prior          = prior != nullptr;
  const std::string prior_value = prior ? prior : "";
  // Restore the caller's frozen-selector environment after all exit paths.
  struct Restore
  {
    const char* name;
    bool present;
    std::string value;
    ~Restore()
    {
      if (present)
        ::setenv(name, value.c_str(), 1);
      else
        ::unsetenv(name);
    }
  } restore{selector, had_prior, prior_value};
  ::unsetenv(selector);
  if (configuredSoftmaxKeySharedMeans())
    throw std::runtime_error("Shared means default changed");
  ::setenv(selector, "1", 1);
  if (!configuredSoftmaxKeySharedMeans())
    throw std::runtime_error("Shared means opt-in not parsed");
  ::setenv(selector, "0", 1);
  if (configuredSoftmaxKeySharedMeans())
    throw std::runtime_error("Shared means opt-out not parsed");
  for (const char* invalid : {"", "true", "2", "1 "})
  {
    ::setenv(selector, invalid, 1);
    bool rejected = false;
    try
    {
      (void)configuredSoftmaxKeySharedMeans();
    }
    catch (const std::invalid_argument&)
    {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Malformed shared means policy admitted");
  }
  ::setenv(selector, "0", 1);
  const auto parser_after = deviceMemoryMetrics(device).snapshot();
  if (parser_before.allocation_calls != parser_after.allocation_calls ||
      parser_before.free_calls != parser_after.free_calls || parser_before.live_bytes != parser_after.live_bytes)
    throw std::runtime_error("Shared means policy parsing allocated an owner");
  // Sparse shapes cover the cutoff, headed/feature tails and padded batches.
  struct Shape
  {
    std::size_t n, h, f, b;
  };
  const Shape shapes[]{{63, 1, 3, 1}, {64, 2, 17, 2}, {65, 3, 1, 2}, {100, 2, 7, 1}, {200, 1, 17, 1}, {300, 1, 1, 1}};
  constexpr double sentinel = -937.25, output_sentinel = 29.5;
  std::size_t checks = 0, positively_selected = 0;
  for (const auto shape : shapes)
  {
    const auto attention = makeAttentionForwardLayout(shape.n, shape.h, shape.f, shape.h * shape.f + 3, shape.n + 2,
                                                      shape.n * (shape.n + 2) + 5);
    const auto layout =
        makeBatchedAttentionForwardLayout(shape.b, attention, shape.n * attention.feature_row_stride + 7,
                                          shape.h * attention.attention_head_stride + 11);
    const auto features      = shape.b * layout.feature_configuration_stride + 5;
    const auto probabilities = shape.b * layout.attention_configuration_stride + 5;
    const auto means         = shape.b * shape.h * shape.n;
    const auto nan           = std::numeric_limits<double>::quiet_NaN();
    DeviceBuffer<double> q(features), k(features), p(probabilities), a(probabilities);
    DeviceBuffer<double> old_q(features), old_k(features), new_q(features), new_k(features);
    DeviceBuffer<double> scratch(means + 14), alias_old(features), alias_new(features);
    DeviceBuffer<DerivativeFiniteStatus> old_status(2), new_status(2);
    std::vector<double> query(features, nan), key(features, nan), prob(probabilities, nan), adj(probabilities, nan);
    std::vector<double> expected_q(features, output_sentinel), expected_k(features, output_sentinel),
        expected_mean(means);
    const std::vector<double> initial(features, output_sentinel), initial_scratch(means + 14, sentinel);
    for (std::size_t b = 0; b < shape.b; ++b)
      for (std::size_t row = 0; row < shape.n; ++row)
        for (std::size_t h = 0; h < shape.h; ++h)
        {
          for (std::size_t f = 0; f < shape.f; ++f)
          {
            const auto i = b * layout.feature_configuration_stride + attention.featureOffset(row, h, f);
            query[i]     = .013 * (int((i * 7 + 3) % 23) - 11);
            key[i]       = .017 * (int((i * 3 + 1) % 19) - 9);
          }
          double normalization = 0;
          for (std::size_t col = 0; col < shape.n; ++col)
            normalization += 1 + (col + row + h) % 7;
          for (std::size_t col = 0; col < shape.n; ++col)
          {
            const auto i = b * layout.attention_configuration_stride + attention.attentionOffset(h, row, col);
            prob[i]      = double(1 + (col + row + h) % 7) / normalization;
            adj[i]       = .019 * (int((i * 5 + 2) % 29) - 14);
          }
        }
    const auto original_adj = adj, original_prob = prob;
    // Concentrate signed zero, cancellation and IEEE hazards on the boundary shape.
    const auto variants = shape.n == 64 ? 6U : 1U;
    for (unsigned int variant = 0; variant < variants; ++variant)
    {
      adj            = original_adj;
      prob           = original_prob;
      const auto bad = (shape.b - 1) * layout.attention_configuration_stride +
          attention.attentionOffset(shape.h - 1, shape.n - 1, shape.n - 2);
      if (variant == 1)
        for (double& value : adj)
          if (device_math::isFiniteBinary64(value))
            value = std::signbit(value) ? -0.0 : 0.0;
      if (variant == 2)
        for (std::size_t b = 0; b < shape.b; ++b)
          for (std::size_t h = 0; h < shape.h; ++h)
            for (std::size_t row = 0; row < shape.n; ++row)
              for (std::size_t col = 0; col < shape.n; ++col)
                adj[b * layout.attention_configuration_stride + attention.attentionOffset(h, row, col)] = col % 3 == 0
                    ? 1e6
                    : col % 3 == 1 ? -1e6
                                   : .00013;
      if (variant == 3)
        adj[bad] = nan;
      if (variant == 4)
        adj[bad] = std::numeric_limits<double>::infinity();
      if (variant == 5)
      {
        prob[bad] = 1e308;
        adj[bad]  = 1e308;
      }
      q.upload(query.data(), features);
      k.upload(key.data(), features);
      p.upload(prob.data(), probabilities);
      a.upload(adj.data(), probabilities);
      old_q.upload(initial.data(), features);
      old_k.upload(initial.data(), features);
      new_q.upload(initial.data(), features);
      new_k.upload(initial.data(), features);
      scratch.upload(initial_scratch.data(), initial_scratch.size());
      checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, q.data(), k.data(), p.data(), a.data(),
                                                old_q.data(), old_k.data()),
                   "canonical softmax Q/K reverse");
      checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, q.data(), k.data(), p.data(), a.data(),
                                                new_q.data(), new_k.data(), scratch.data() + 5, means),
                   "borrowed means Q/K reverse");
      DerivativeFiniteScanLayout scan{features, features, 1, 0, 1};
      checkRuntime(launchDerivativeFiniteScan(queue.getNative(), scan, old_q.data(), old_status.data()),
                   "old Q admission");
      checkRuntime(launchDerivativeFiniteScan(queue.getNative(), scan, old_k.data(), old_status.data() + 1),
                   "old K admission");
      checkRuntime(launchDerivativeFiniteScan(queue.getNative(), scan, new_q.data(), new_status.data()),
                   "new Q admission");
      checkRuntime(launchDerivativeFiniteScan(queue.getNative(), scan, new_k.data(), new_status.data() + 1),
                   "new K admission");
      queue.sync();
      std::vector<double> oq(features), ok(features), nq(features), nk(features), work(means + 14);
      old_q.download(oq.data(), features);
      old_k.download(ok.data(), features);
      new_q.download(nq.data(), features);
      new_k.download(nk.data(), features);
      scratch.download(work.data(), work.size());
      DerivativeFiniteStatus os[2], ns[2];
      old_status.download(os, 2);
      new_status.download(ns, 2);
      for (std::size_t i = 0; i < features; ++i)
      {
        sharedRowMeanExact(nq[i], oq[i], i);
        sharedRowMeanExact(nk[i], ok[i], i);
      }
      for (unsigned int i = 0; i < 2; ++i)
        if (os[i].first_nonfinite != ns[i].first_nonfinite || os[i].nonfinite_count != ns[i].nonfinite_count)
          throw std::runtime_error("Shared row means changed finite-scan admission");
      for (std::size_t i = 0; i < work.size(); ++i)
        if ((i < 5 || i >= 5 + means || shape.n < 64) && work[i] != sentinel)
          throw std::runtime_error("Shared row means changed a borrowed prefix/tail or fallback workspace");
      // Independent CPU assembly reuses the existing analytic scalar primitives.
      if (variant < 3)
      {
        std::fill(expected_q.begin(), expected_q.end(), output_sentinel);
        std::fill(expected_k.begin(), expected_k.end(), output_sentinel);
        const auto scale = 1 / std::sqrt(double(shape.f));
        for (std::size_t b = 0; b < shape.b; ++b)
          for (std::size_t h = 0; h < shape.h; ++h)
            for (std::size_t row = 0; row < shape.n; ++row)
            {
              double mean = 0;
              for (std::size_t col = 0; col < shape.n; ++col)
              {
                const auto j = b * layout.attention_configuration_stride + attention.attentionOffset(h, row, col);
                mean += device_derivative::softmaxRowReductionContribution(prob[j], adj[j]);
              }
              expected_mean[(b * shape.h + h) * shape.n + row] = mean;
              for (std::size_t col = 0; col < shape.n; ++col)
              {
                const auto j   = b * layout.attention_configuration_stride + attention.attentionOffset(h, row, col);
                const auto bar = device_derivative::softmaxLogitAdjoint(prob[j], adj[j], mean);
                for (std::size_t f = 0; f < shape.f; ++f)
                {
                  const auto qi = b * layout.feature_configuration_stride + attention.featureOffset(row, h, f);
                  const auto ki = b * layout.feature_configuration_stride + attention.featureOffset(col, h, f);
                  if (col == 0)
                    expected_q[qi] = 0;
                  if (row == 0)
                    expected_k[ki] = 0;
                  expected_q[qi] += device_derivative::scaledQkAdjointContribution(bar, key[ki], scale);
                  expected_k[ki] += device_derivative::scaledQkAdjointContribution(bar, query[qi], scale);
                }
              }
            }
        const auto near = [](double x, double y) {
          if (!device_math::isFiniteBinary64(x) || !device_math::isFiniteBinary64(y) ||
              std::abs(x - y) > 3e-11 * (1 + std::abs(y)))
            throw std::runtime_error("Shared row means exceeds existing Runtime CPU analytic bound");
        };
        for (std::size_t i = 0; i < features; ++i)
        {
          near(nq[i], expected_q[i]);
          near(nk[i], expected_k[i]);
        }
        if (shape.n >= 64)
        {
          for (std::size_t i = 0; i < means; ++i)
          {
            near(work[5 + i], expected_mean[i]);
            if (work[5 + i] == sentinel)
              throw std::runtime_error("Shared means path was not selected");
          }
          ++positively_selected;
        }
      }
      ++checks;
    }
    // Unsafe/insufficient borrowed storage must take the old route before writes.
    if (shape.n == 64)
      for (unsigned int fallback = 0; fallback < 4; ++fallback)
      {
        q.upload(query.data(), features);
        p.upload(original_prob.data(), probabilities);
        a.upload(original_adj.data(), probabilities);
        old_q.upload(initial.data(), features);
        old_k.upload(initial.data(), features);
        new_q.upload(initial.data(), features);
        new_k.upload(initial.data(), features);
        scratch.upload(initial_scratch.data(), initial_scratch.size());
        double* borrowed    = fallback == 0 ? nullptr
               : fallback == 2              ? p.data() + 1
               : fallback == 3              ? new_q.data()
                                            : scratch.data() + 5;
        const auto capacity = fallback == 1 ? means - 1 : means;
        checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, q.data(), k.data(), p.data(), a.data(),
                                                  old_q.data(), old_k.data()),
                     "fallback oracle");
        checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, q.data(), k.data(), p.data(), a.data(),
                                                  new_q.data(), new_k.data(), borrowed, capacity),
                     "fallback borrowed means");
        queue.sync();
        std::vector<double> oq(features), ok(features), nq(features), nk(features), work(means + 14),
            after_prob(probabilities);
        old_q.download(oq.data(), features);
        old_k.download(ok.data(), features);
        new_q.download(nq.data(), features);
        new_k.download(nk.data(), features);
        scratch.download(work.data(), work.size());
        p.download(after_prob.data(), probabilities);
        for (std::size_t i = 0; i < features; ++i)
        {
          sharedRowMeanExact(nq[i], oq[i], i);
          sharedRowMeanExact(nk[i], ok[i], i);
        }
        if (work != initial_scratch ||
            std::memcmp(after_prob.data(), original_prob.data(), probabilities * sizeof(double)))
          throw std::runtime_error("Fallback modified borrowed scratch/input before legacy launch");
        ++checks;
      }
    // A smaller live batch may borrow a larger prepared arena without touching its tail.
    if (shape.b > 1)
    {
      auto live                = layout;
      live.configuration_count = 1;
      const auto live_means    = shape.h * shape.n;
      q.upload(query.data(), features);
      p.upload(original_prob.data(), probabilities);
      a.upload(original_adj.data(), probabilities);
      old_q.upload(initial.data(), features);
      old_k.upload(initial.data(), features);
      new_q.upload(initial.data(), features);
      new_k.upload(initial.data(), features);
      scratch.upload(initial_scratch.data(), initial_scratch.size());
      checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), live, q.data(), k.data(), p.data(), a.data(),
                                                old_q.data(), old_k.data()),
                   "live batch oracle");
      checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), live, q.data(), k.data(), p.data(), a.data(),
                                                new_q.data(), new_k.data(), scratch.data() + 5, means),
                   "live batch borrowed prefix");
      queue.sync();
      std::vector<double> oq(features), ok(features), nq(features), nk(features), work(means + 14);
      old_q.download(oq.data(), features);
      old_k.download(ok.data(), features);
      new_q.download(nq.data(), features);
      new_k.download(nk.data(), features);
      scratch.download(work.data(), work.size());
      for (std::size_t i = 0; i < features; ++i)
      {
        sharedRowMeanExact(nq[i], oq[i], i);
        sharedRowMeanExact(nk[i], ok[i], i);
      }
      for (std::size_t i = 0; i < work.size(); ++i)
        if ((i < 5 || i >= 5 + live_means) && work[i] != sentinel)
          throw std::runtime_error("Live batch overwrote borrowed capacity tail");
      for (std::size_t i = 0; i < live_means; ++i)
        if (!device_math::isFiniteBinary64(work[5 + i]) || work[5 + i] == sentinel)
          throw std::runtime_error("Live batch borrowed path fell back");
      ++checks;
      ++positively_selected;
    }
    // Safe legacy query->query-adjoint alias must force fallback before precompute.
    alias_old.upload(query.data(), features);
    alias_new.upload(query.data(), features);
    p.upload(original_prob.data(), probabilities);
    a.upload(original_adj.data(), probabilities);
    old_k.upload(initial.data(), features);
    new_k.upload(initial.data(), features);
    scratch.upload(initial_scratch.data(), initial_scratch.size());
    checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, alias_old.data(), k.data(), p.data(), a.data(),
                                              alias_old.data(), old_k.data()),
                 "legacy query alias");
    checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, alias_new.data(), k.data(), p.data(), a.data(),
                                              alias_new.data(), new_k.data(), scratch.data() + 5, means),
                 "borrowed query alias fallback");
    queue.sync();
    std::vector<double> aq(features), bq(features), ak(features), bk(features), work(means + 14);
    alias_old.download(aq.data(), features);
    alias_new.download(bq.data(), features);
    old_k.download(ak.data(), features);
    new_k.download(bk.data(), features);
    scratch.download(work.data(), work.size());
    for (std::size_t i = 0; i < features; ++i)
    {
      sharedRowMeanExact(bq[i], aq[i], i);
      sharedRowMeanExact(bk[i], ak[i], i);
    }
    if (work != initial_scratch)
      throw std::runtime_error("Aliased output admitted borrowed means");
    ++checks;
    q.upload(query.data(), features);
    p.upload(original_prob.data(), probabilities);
    a.upload(original_adj.data(), probabilities);
    if (shape.n == 64)
    {
      // The legacy feature grid itself is unlaunchable: both APIs reject before work.
      auto excessive = layout;
      excessive.configuration_count =
          (std::size_t(std::numeric_limits<unsigned int>::max()) + 1) * 128 / (shape.n * shape.h * shape.f) + 1;
      for (bool borrowed : {false, true})
      {
        bool rejected = false;
        try
        {
          (void)launchSoftmaxScaledQkReverse(queue.getNative(), excessive, q.data(), k.data(), p.data(), a.data(),
                                             new_q.data(), new_k.data(), borrowed ? scratch.data() + 5 : nullptr,
                                             borrowed ? means : 0);
        }
        catch (const std::length_error&)
        {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("Unlaunchable legacy grid admitted");
        ++checks;
      }
    }
    const auto before = deviceMemoryMetrics(device).snapshot();
    checkRuntime(launchSoftmaxScaledQkReverse(queue.getNative(), layout, q.data(), k.data(), p.data(), a.data(),
                                              new_q.data(), new_k.data(), scratch.data() + 5, means),
                 "warmed borrowed means");
    queue.sync();
    const auto after = deviceMemoryMetrics(device).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
        before.live_bytes != after.live_bytes)
      throw std::runtime_error("Shared row means allocated a hidden owner");
  }
  if (!positively_selected)
    throw std::runtime_error("Shared row means fixture never selected new kernels");
  std::cout << "PASS shared softmax row means cases=" << checks << " positive_paths=" << positively_selected
            << " finite_GPU_bits/status/padding CPU_bound=3e-11 fallback_alias_capacity recovery warm_zero_owners\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
