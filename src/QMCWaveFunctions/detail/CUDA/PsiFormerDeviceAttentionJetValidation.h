//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ATTENTION_JET_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ATTENTION_JET_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetKernels.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Compare both launch maps with the native host algebra, including padded tile tails.
inline void validateAttentionJetTiles(compute::Queue<PlatformKind::CUDA>& queue)
{
  const char* configured = std::getenv("PSIFORMER_GPU_ATTENTION_JET_TILED");
  struct RestoreOption
  {
    bool existed;
    std::string value;
    ~RestoreOption()
    {
      if (existed)
        ::setenv("PSIFORMER_GPU_ATTENTION_JET_TILED", value.c_str(), 1);
      else
        ::unsetenv("PSIFORMER_GPU_ATTENTION_JET_TILED");
    }
  } restore{configured != nullptr, configured ? configured : ""};
  const auto compare = [](const std::vector<double>& actual, const std::vector<double>& expected,
                          const char* field) {
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (!device_math::isFiniteBinary64(actual[i]) ||
          std::abs(actual[i] - expected[i]) > 2e-12 * (1 + std::abs(expected[i])))
        throw std::runtime_error(std::string(field) + " mismatch at offset " + std::to_string(i));
  };

  std::size_t cases = 0;
  for (auto mode : {SpatialJetMode::VALUE_ONLY, SpatialJetMode::ACTIVE, SpatialJetMode::FULL_VGL})
    for (std::size_t rows : {1, 7, 8, 9, 17})
      for (std::size_t width : {1, 7, 8, 9, 17})
      {
        constexpr std::size_t heads = 2, configurations = 2;
        const std::size_t feature_row = heads * width + 3;
        const std::size_t attention_row = rows + 2;
        const std::size_t attention_head = rows * attention_row + 5;
        const std::size_t feature_plane = rows * feature_row + 7;
        const std::size_t attention_plane = heads * attention_head + 11;
        const auto preliminary = makeSpatialAttentionJetLayout(configurations, rows, heads, width, 3,
            mode, feature_row, attention_row, attention_head, feature_plane, attention_plane);
        const std::size_t planes = preliminary.features.plane_count;
        const auto layout = makeSpatialAttentionJetLayout(configurations, rows, heads, width, 3,
            mode, feature_row, attention_row, attention_head, feature_plane, attention_plane,
            planes * feature_plane + 13, planes * attention_plane + 17);
        const std::size_t nf = spatialJetSpanElements(layout.features);
        const std::size_t na = spatialJetSpanElements(layout.attention);
        std::vector<double> query(nf), key(nf), value(nf);
        for (std::size_t i = 0; i < nf; ++i)
        {
          query[i] = 0.12 * std::sin(0.13 * static_cast<double>(i + 1));
          key[i] = 0.17 * std::cos(0.17 * static_cast<double>(i + 3));
          value[i] = 0.21 * std::sin(0.11 * static_cast<double>(i + 7));
        }
        std::vector<double> raw(na, -7.25), expected_context(nf, -7.25);
        spatial_jet::attentionLogitJetsHost(layout, query.data(), key.data(), raw.data());
        auto expected = raw;
        stableSoftmaxJetRows(makeAttentionSoftmaxJetRowLayout(layout), expected.data());
        spatial_jet::attentionContextJetsHost(layout, expected.data(), value.data(), expected_context.data());
        DeviceBuffer<double> dq(nf), dk(nf), dv(nf), attention(na), retained(na), context(nf);
        DeviceBuffer<device_math::JetMathStatus> status(configurations * heads * rows);
        dq.upload(query.data(), nf);
        dk.upload(key.data(), nf);
        dv.upload(value.data(), nf);
        for (const char* option : {"0", "1"})
        {
          if (::setenv("PSIFORMER_GPU_ATTENTION_JET_TILED", option, 1) != 0)
            throw std::runtime_error("Cannot configure attention tile validation");
          std::vector<double> actual(na, -7.25), actual_raw(na, -7.25), actual_context(nf, -7.25);
          attention.upload(actual.data(), na);
          retained.upload(actual_raw.data(), na);
          context.upload(actual_context.data(), nf);
          checkRuntime(launchAttentionJetWeights(queue.getNative(), layout, dq.data(), dk.data(),
              attention.data(), status.data(), retained.data()), "attention tile weights");
          checkRuntime(launchAttentionContextJets(queue.getNative(), layout, attention.data(), dv.data(),
              context.data()), "attention tile context");
          queue.sync();
          attention.download(actual.data(), na);
          retained.download(actual_raw.data(), na);
          context.download(actual_context.data(), nf);
          std::vector<device_math::JetMathStatus> host_status(configurations * heads * rows);
          status.download(host_status.data(), host_status.size());
          for (auto row_status : host_status)
            if (row_status != device_math::JetMathStatus::REGULAR)
              throw std::runtime_error("Attention tile validation returned a nonregular row");
          compare(actual_raw, raw, "attention raw logit");
          compare(actual, expected, "attention probability");
          compare(actual_context, expected_context, "attention context");
          ++cases;
        }
      }
  std::cout << "PASS " << cases << " native attention cases: raw logits, softmax, context, padding\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
