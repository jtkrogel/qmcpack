//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_SOFTMAX_COLUMN_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_SOFTMAX_COLUMN_VALIDATION_H

#include "PsiFormerDeviceSpatialReverse.h"
#include "PsiFormerDeviceResources.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Retain the unchanged CPU row helper and original GPU row kernel as independent oracles.
inline void validateSoftmaxColumnReverse(compute::Queue<PlatformKind::CUDA>& queue)
{
  const std::pair<std::size_t, std::size_t> cases[] = {
      {1, 1}, {7, 7}, {17, 3}, {100, 1}, {101, 3}, {200, 1}, {203, 3}, {300, 1}, {301, 3}};
  for (const auto& [width, targets] : cases)
  {
    const auto layout = makeSoftmaxJetRowLayout(1, 2, width, targets, SpatialJetMode::FULL_VGL,
        width + 3, 2 * (width + 7) + 5, 0, 1, width + 7);
    const auto span = spatialJetSpanElements(layout.jets);
    std::vector<double> logits(span, 17.5), seed(span, -19.5), initial(span, 23.5);
    for (std::size_t row = 0; row < layout.row_count; ++row)
      for (std::size_t plane = 0; plane < layout.jets.plane_count; ++plane)
        for (std::size_t column = 0; column < width; ++column)
        {
          const auto index = row * layout.row_group_stride + plane * layout.jets.plane_stride + column;
          logits[index] = (static_cast<int>((index * 7) % 23) - 11) / 128.0;
          seed[index] = (static_cast<int>((index * 11) % 29) - 14) / 64.0;
          initial[index] = (static_cast<int>(index % 7) - 3) / 32.0;
        }
    auto probabilities = logits;
    stableSoftmaxJetRows(layout, probabilities.data());
    auto expected = initial;
    spatial_jet_reverse::softmaxJetRowsReverseHost(layout, logits.data(), probabilities.data(),
        seed.data(), expected.data());
    DeviceBuffer<double> d_logits, d_probabilities, d_seed, d_old, d_columns;
    for (auto* buffer : {&d_logits, &d_probabilities, &d_seed, &d_old, &d_columns})
      buffer->reserve(span);
    d_logits.upload(logits.data(), span);
    d_probabilities.upload(probabilities.data(), span);
    d_seed.upload(seed.data(), span);
    d_old.upload(initial.data(), span);
    d_columns.upload(initial.data(), span);
    checkRuntime(launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
        d_seed.data(), d_old.data(), false), "softmax original-row oracle");
    checkRuntime(launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
        d_seed.data(), d_columns.data(), true), "softmax independent-column candidate");
    queue.sync();
    std::vector<double> original(span), columns(span);
    d_old.download(original.data(), span);
    d_columns.download(columns.data(), span);
    if (std::memcmp(original.data(), columns.data(), span * sizeof(double)) != 0)
      throw std::runtime_error("Softmax column reverse differs bitwise from unchanged GPU row oracle");
    for (std::size_t index = 0; index < span; ++index)
      if (!device_math::isFiniteBinary64(columns[index]) ||
          std::abs(columns[index] - expected[index]) > 2e-12 * (1 + std::abs(expected[index])))
        throw std::runtime_error("Softmax column reverse differs from unchanged CPU row oracle/padding");

    // Reuse fixed buffers and the original nonzero destination: no cold owner in warmed work.
    d_columns.upload(initial.data(), span);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "softmax validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "softmax validation device");
#endif
    const auto before = deviceMemoryMetrics(ordinal).snapshot();
    checkRuntime(launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
        d_seed.data(), d_columns.data(), true), "softmax warmed column reverse");
    queue.sync();
    const auto after = deviceMemoryMetrics(ordinal).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls)
      throw std::runtime_error("Warmed softmax column reverse allocated");

    for (bool parallel : {false, true})
    {
      bool alias_rejected = false;
      try
      {
        (void)launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
            d_seed.data(), d_logits.data(), parallel);
      }
      catch (const std::invalid_argument&)
      {
        alias_rejected = true;
      }
      if (!alias_rejected) throw std::runtime_error("Softmax reverse accepted an aliased destination");
    }

    if (width == 7)
    {
      // IEEE failures remain visible to the existing caller's finite admission.
      const double invalids[] = {std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1e308};
      for (const auto invalid : invalids)
      {
        auto invalid_seed = seed;
        auto invalid_logits = logits;
        invalid_seed[0] = invalid;
        if (invalid == 1e308)
          invalid_logits[layout.jets.plane_stride] = 1e308;
        d_seed.upload(invalid_seed.data(), span);
        d_logits.upload(invalid_logits.data(), span);
        d_old.upload(initial.data(), span);
        d_columns.upload(initial.data(), span);
        checkRuntime(launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
            d_seed.data(), d_old.data(), false), "softmax original IEEE failure");
        checkRuntime(launchSoftmaxJetReverse(queue.getNative(), layout, d_logits.data(), d_probabilities.data(),
            d_seed.data(), d_columns.data(), true), "softmax column IEEE failure");
        queue.sync();
        d_old.download(original.data(), span);
        d_columns.download(columns.data(), span);
        bool nonfinite = false;
        const auto ieee_class = [](double value) {
          std::uint64_t bits;
          std::memcpy(&bits, &value, sizeof(bits));
          if ((bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000))
            return 3; // NaN payload/sign are not a physical diagnostic.
          return (bits >> 63) ? 2 : 1; // Negative/positive infinity.
        };
        for (std::size_t index = 0; index < span; ++index)
        {
          const bool old_finite = device_math::isFiniteBinary64(original[index]);
          const bool new_finite = device_math::isFiniteBinary64(columns[index]);
          if (old_finite != new_finite || (old_finite && original[index] != columns[index]) ||
              (!old_finite && ieee_class(original[index]) != ieee_class(columns[index])))
            throw std::runtime_error("Softmax column reverse changed IEEE failure visibility");
          nonfinite = nonfinite || !new_finite;
        }
        if (!nonfinite) throw std::runtime_error("Softmax reverse masked a nonfinite/overflow seed");
      }
    }
  }
  for (bool parallel : {false, true})
  {
    checkRuntime(launchSoftmaxJetReverse(queue.getNative(), {}, nullptr, nullptr, nullptr, nullptr, parallel),
        "softmax empty layout");
    bool overflow_rejected = false;
    try
    {
      auto invalid = makeSoftmaxJetRowLayout(1, 2, 2, 1, SpatialJetMode::FULL_VGL);
      invalid.row_count = std::numeric_limits<std::size_t>::max();
      (void)launchSoftmaxJetReverse(queue.getNative(), invalid, nullptr, nullptr, nullptr, nullptr, parallel);
    }
    catch (const std::overflow_error&)
    {
      overflow_rejected = true;
    }
    catch (const std::invalid_argument&)
    {
      overflow_rejected = true;
    }
    if (!overflow_rejected) throw std::runtime_error("Softmax reverse accepted overflowing metadata");
  }
  std::cout << "PASS softmax column reverse: independent unchanged CPU/GPU row oracles, exact GPU outputs, K1/full/tails/padding, additive seeds, warm zero allocation, alias/overflow guards, IEEE failures\n";
}
}
#endif
