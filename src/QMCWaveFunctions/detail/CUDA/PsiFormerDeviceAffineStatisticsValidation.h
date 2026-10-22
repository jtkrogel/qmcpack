//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_AFFINE_STATISTICS_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_AFFINE_STATISTICS_VALIDATION_H

#include "PsiFormerDeviceAffineStatistics.h"
#include "PsiFormerDeviceScoreReverse.h"

namespace qmcplusplus::psiformer::device
{

/// Exercise N100 attention/head dimensions without requiring a large fixture.
inline void validateDeviceAffineGramSymmetry(AcceleratorBlasHandle& blas)
{
  for (const std::size_t width : {std::size_t{256}, std::size_t{1600}})
  {
    constexpr std::size_t rows = 50;
    std::vector<double> values(rows * width), packed(width * width, 0.0), symmetric(packed.size());
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < width; ++column)
        values[row * width + column] = std::sin(0.173 * (row + 1) * (column + 3)) *
            (column % 3 == 0 ? 100.0 : 0.01);
    DeviceBuffer<double> source(values.size()), gram(packed.size());
    source.upload(values.data(), values.size());
    gram.upload(packed.data(), packed.size());
    accumulateDenseWeightAdjoint(blas, {rows, width, width, width, width, width},
        source.data(), source.data(), gram.data());
    gram.download(packed.data(), packed.size());
    affine_statistics_detail::copySymmetricGram(packed.data(), width, symmetric.data(), width);
    for (std::size_t row = 0; row < width; ++row)
      for (std::size_t column = 0; column <= row; ++column)
      {
        const double actual = symmetric[row * width + column];
        if (!device_math::isFiniteBinary64(actual) ||
            actual != symmetric[column * width + row] || actual != packed[row * width + column])
          throw std::runtime_error("Device affine Gram publication is not exactly symmetric");
        if (column == 0 || row == column)
        {
          double expected = 0.0;
          for (std::size_t k = 0; k < rows; ++k)
            expected += values[k * width + row] * values[k * width + column];
          if (std::abs(actual - expected) > 3e-7 * (1 + std::abs(expected)))
            throw std::runtime_error("Device affine symmetric Gram differs from HOST rows");
        }
      }
  }
  std::cout << "PASS affine_Gram_exact_symmetry widths_256_1600 HOST_rows\n";
}

/// Compare DEVICE Gram factors against independently looped HOST affine rows.
inline void validateDeviceAffineStatistics(pf::PsiFormer& model,
    const std::shared_ptr<const PreparedDeviceModel>& image,
    compute::Queue<PlatformKind::CUDA>& queue, AcceleratorBlasHandle& blas)
{
  const auto& plan = image->plan();
  if (plan.environment().periodic_profile)
    return;
  validateDeviceAffineGramSymmetry(blas);
  std::vector<DeviceAffineFactorDescriptor> blocks;
  const auto append = [&](ParameterRole role, std::size_t layer, bool bias = false) {
    const auto& tensor = plan.parameter(role, layer);
    blocks.push_back({role, layer, tensor.shape.at(0), tensor.shape.at(1), bias});
  };
  append(ParameterRole::ELECTRON_EMBEDDING_WEIGHT, NO_ATTENTION_BLOCK);
  for (std::size_t layer = 0; layer < plan.modelShape().attention_blocks; ++layer)
  {
    for (auto role : {ParameterRole::ATTENTION_QUERY_WEIGHT, ParameterRole::ATTENTION_KEY_WEIGHT,
        ParameterRole::ATTENTION_VALUE_WEIGHT, ParameterRole::ATTENTION_OUTPUT_WEIGHT})
      append(role, layer);
    append(ParameterRole::UPDATE_HIDDEN_WEIGHT, layer, true);
    append(ParameterRole::UPDATE_OUTPUT_WEIGHT, layer, true);
  }
  if (plan.modelShape().spin_up_electrons)
    append(ParameterRole::BACKFLOW_UP_WEIGHT, NO_ATTENTION_BLOCK);
  if (plan.modelShape().spin_down_electrons)
    append(ParameterRole::BACKFLOW_DOWN_WEIGHT, NO_ATTENTION_BLOCK);

  const auto prepare = [&] {
    std::vector<DeviceAffineFactorSums> factors(blocks.size());
    for (std::size_t block = 0; block < blocks.size(); ++block)
    {
      const auto width = blocks[block].input_width + blocks[block].bias_augmented;
      factors[block].activation_outer_sum.resize(width * width);
      factors[block].sensitivity_outer_sum.resize(blocks[block].output_width * blocks[block].output_width);
    }
    return factors;
  };
  struct HostRows final : pf::DirectAffineObservationSink
  {
    const std::vector<DeviceAffineFactorDescriptor>& blocks;
    std::vector<DeviceAffineFactorSums>& factors;
    double curvature;
    HostRows(const std::vector<DeviceAffineFactorDescriptor>& b, std::vector<DeviceAffineFactorSums>& f, bool gaussian)
        : blocks(b), factors(f), curvature(gaussian ? 2.0 : 1.0) {}
    void observe(const pf::DirectAffineObservation& observation) override
    {
      const auto found = std::find_if(blocks.begin(), blocks.end(), [&](const auto& block) {
        return block.role == observation.role && block.attention_block == observation.attention_block;
      });
      if (found == blocks.end())
        throw std::runtime_error("HOST affine oracle emitted an unregistered block");
      const auto ordinal = std::size_t(found - blocks.begin());
      auto& factor = factors[ordinal];
      const auto activation = found->input_width + found->bias_augmented;
      for (std::size_t row = 0; row < observation.row_count; ++row)
      {
        for (std::size_t left = 0; left < activation; ++left)
          for (std::size_t right = 0; right < activation; ++right)
          {
            const double a = left == found->input_width ? 1 :
                observation.activations[row * found->input_width + left];
            const double b = right == found->input_width ? 1 :
                observation.activations[row * found->input_width + right];
            factor.activation_outer_sum[left * activation + right] += a * b;
          }
        for (std::size_t left = 0; left < found->output_width; ++left)
          for (std::size_t right = 0; right < found->output_width; ++right)
            factor.sensitivity_outer_sum[left * found->output_width + right] += curvature *
                observation.sensitivities[row * found->output_width + left] *
                observation.sensitivities[row * found->output_width + right];
      }
      ++factor.sample_weight_sum;
      factor.row_weight_sum += observation.row_count;
      factor.row_count += observation.row_count;
    }
  };
  const auto compare = [](double actual, double expected) {
    if (!device_math::isFiniteBinary64(actual) || !device_math::isFiniteBinary64(expected) ||
        std::abs(actual - expected) > 3e-7 * (1 + std::abs(expected)))
      throw std::runtime_error("Device affine statistics differ from HOST row oracle");
  };

  pf::DirectScoreExecutor cpu(model, plan);
  auto workspace = cpu.makeWorkspace();
  PreparedDeviceInference forward(image, 1, queue, blas, SpatialJetMode::VALUE_ONLY, true);
  PreparedDeviceScoreReverse reverse(image, 1, queue, blas);
  const std::vector<double> original(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 3 * model.ne);
  for (bool gaussian : {false, true})
  {
    auto expected = prepare(), actual = prepare();
    PreparedDeviceAffineStatistics statistics(image, blocks, gaussian,
        PreparedDeviceAffineStatistics::requiredDeviceBytes(blocks),
        PreparedDeviceAffineStatistics::requiredHostBytes(blocks), queue, blas);
    HostRows host(blocks, expected, gaussian);
    for (std::size_t sample = 0; sample < 2; ++sample)
    {
      auto positions = original;
      positions[0] += sample * 0.017;
      workspace->setPositions(pf::GeometryPositionView::interleaved(positions.data(), model.ne));
      const auto reference = cpu.evaluateWithAffineObservations(*workspace, host);
      forward.evaluate(positions);
      statistics.beginSample(image->version());
      reverse.observeAffine(forward, statistics);
      std::vector<double> score(plan.parameterCount());
      reverse.downloadResponse(score);
      for (std::size_t p = 0; p < score.size(); ++p)
        compare(score[p], reference.parameter_score[p]);
      statistics.finishSample();
    }
    bool rejected_weight = false, rejected_version = false, rejected_incomplete = false;
    try { statistics.beginSample(image->version(), 0.5); }
    catch (const std::invalid_argument&) { rejected_weight = true; }
    try { statistics.beginSample(image->version() + 1); }
    catch (const std::invalid_argument&) { rejected_version = true; }
    statistics.beginSample(image->version());
    try { statistics.finishSample(); }
    catch (const std::logic_error&) { rejected_incomplete = true; }
    if (!rejected_weight || !rejected_version || !rejected_incomplete || statistics.sampleCount() != 2)
      throw std::runtime_error("Device affine guards modified accepted sample statistics");

    // Inject a bad activation into the observer only, then verify bank rollback.
    struct BadRows final : DeviceAffineObservationSink
    {
      PreparedDeviceAffineStatistics& factors;
      const double* bad;
      bool first = true, sensitivity;
      BadRows(PreparedDeviceAffineStatistics& f, const double* b, bool g = false) : factors(f), bad(b), sensitivity(g) {}
      void observe(ParameterRole role, std::size_t layer, const double* a, const double* g,
          std::size_t rows, std::size_t input, std::size_t output) override
      {
        factors.observe(role, layer, first && !sensitivity ? bad : a,
            first && sensitivity ? bad : g, rows, input, output);
        first = false;
      }
    };
    std::size_t bad_elements = 0;
    for (const auto& block : blocks)
      bad_elements = std::max(bad_elements, model.ne * std::max(block.input_width, block.output_width));
    std::vector<double> bad_host(bad_elements, std::numeric_limits<double>::quiet_NaN());
    DeviceBuffer<double> bad(bad_elements);
    bad.upload(bad_host.data(), bad_host.size());
    BadRows bad_rows(statistics, bad.data());
    statistics.beginSample(image->version());
    reverse.observeAffine(forward, bad_rows);
    bool rejected_nonfinite = false;
    try { statistics.finishSample(); }
    catch (const std::runtime_error&) { rejected_nonfinite = true; }
    if (!rejected_nonfinite || statistics.sampleCount() != 2)
      throw std::runtime_error("Nonfinite device affine sample changed accepted bank");
    std::fill(bad_host.begin(), bad_host.end(), std::numeric_limits<double>::max());
    bad.upload(bad_host.data(), bad_host.size());
    BadRows overflow_rows(statistics, bad.data(), true);
    statistics.beginSample(image->version());
    reverse.observeAffine(forward, overflow_rows);
    bool rejected_overflow = false;
    try { statistics.finishSample(); }
    catch (const std::runtime_error&) { rejected_overflow = true; }
    if (!rejected_overflow || statistics.sampleCount() != 2)
      throw std::runtime_error("Finite sensitivity Gram overflow changed accepted bank");
    // Simulate a HOST fallback rejection after DEVICE finite admission. The
    // uncommitted receipt must discard the sample without changing the bank.
    statistics.beginSample(image->version());
    reverse.observeAffine(forward, statistics);
    {
      auto prepared = statistics.prepareSampleCommit();
      if (statistics.sampleCount() != 2)
        throw std::runtime_error("Device affine prepare prematurely published a sample");
      bool pending_download_rejected = false;
      try { statistics.download(actual); }
      catch (const std::logic_error&) { pending_download_rejected = true; }
      if (!pending_download_rejected)
        throw std::runtime_error("Prepared device affine factors escaped before HOST commit");
    }
    statistics.download(actual);
    for (std::size_t block = 0; block < blocks.size(); ++block)
    {
      if (actual[block].row_count != expected[block].row_count)
        throw std::runtime_error("Device affine spin-sector row count differs");
      compare(actual[block].sample_weight_sum, expected[block].sample_weight_sum);
      compare(actual[block].row_weight_sum, expected[block].row_weight_sum);
      for (std::size_t i = 0; i < actual[block].activation_outer_sum.size(); ++i)
        compare(actual[block].activation_outer_sum[i], expected[block].activation_outer_sum[i]);
      for (std::size_t i = 0; i < actual[block].sensitivity_outer_sum.size(); ++i)
        compare(actual[block].sensitivity_outer_sum[i], expected[block].sensitivity_outer_sum[i]);
    }
    statistics.beginSample(image->version());
    reverse.observeAffine(forward, statistics);
    statistics.finishSample();
    if (statistics.sampleCount() != 3)
      throw std::runtime_error("Device affine statistics failed valid recovery");
  }
  std::cout << "PASS real_open_affine_statistics HOST_rows bias spin_counts Gaussian sample_rollback recovery\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
