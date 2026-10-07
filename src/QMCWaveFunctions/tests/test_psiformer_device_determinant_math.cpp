//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_determinant_math.cpp
 * @brief Independent CPU oracle for the shared CUDA/HIP determinant scalar core.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDeterminantMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace determinant = qmcplusplus::psiformer::device_determinant;

namespace
{

struct FactorizationStorage
{
  explicit FactorizationStorage(std::size_t matrix_size)
      : lu(matrix_size * matrix_size), inverse(matrix_size * matrix_size),
        permutation(matrix_size), solve(matrix_size)
  {}

  std::vector<double> lu;
  std::vector<double> inverse;
  std::vector<std::size_t> permutation;
  std::vector<double> solve;
};

struct CombinationStorage
{
  explicit CombinationStorage(std::size_t channels)
      : term_phase(channels), term_log_abs(channels), scaled_terms(channels), weights(channels)
  {}

  std::vector<double> term_phase;
  std::vector<double> term_log_abs;
  std::vector<double> scaled_terms;
  std::vector<double> weights;
};

determinant::FactorizationMetadata factor(const std::vector<double>& matrix,
                                          std::size_t matrix_size,
                                          FactorizationStorage& storage,
                                          bool prepare_inverse = true)
{
  return determinant::factorizeScaledReal(
      matrix.data(), matrix_size, storage.lu.data(), storage.permutation.data(),
      prepare_inverse, storage.inverse.data(), storage.solve.data());
}

determinant::FactorizationMetadata signedLogChannel(double phase,
                                                    double log_abs,
                                                    bool inverse_available = true)
{
  determinant::FactorizationMetadata metadata;
  metadata.phase             = phase;
  metadata.log_abs           = log_abs;
  metadata.status            = determinant::FactorizationStatus::REGULAR;
  metadata.inverse_available = inverse_available;
  return metadata;
}

determinant::CombinationMetadata combine(
    const std::vector<determinant::FactorizationMetadata>& channels,
    const double* coefficients,
    CombinationStorage& storage)
{
  return determinant::combineChannelsReal(
      channels.data(), coefficients, channels.size(), storage.term_phase.data(),
      storage.term_log_abs.data(), storage.scaled_terms.data(), storage.weights.data());
}

long double referenceDeterminant(const double* matrix, std::size_t matrix_size)
{
  std::vector<std::size_t> permutation(matrix_size);
  std::iota(permutation.begin(), permutation.end(), 0);
  long double determinant_value = 0;
  do
  {
    std::size_t inversions = 0;
    long double product    = 1;
    for (std::size_t row = 0; row < matrix_size; ++row)
    {
      product *= static_cast<long double>(matrix[row * matrix_size + permutation[row]]);
      for (std::size_t previous = 0; previous < row; ++previous)
        inversions += permutation[previous] > permutation[row] ? 1 : 0;
    }
    determinant_value += (inversions & 1U) ? -product : product;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return determinant_value;
}

void checkMetadata(const determinant::FactorizationMetadata& metadata,
                   const std::vector<double>& matrix,
                   std::size_t matrix_size,
                   double tolerance = 5.0e-13)
{
  const long double expected = referenceDeterminant(matrix.data(), matrix_size);
  REQUIRE(expected != 0);
  CHECK(metadata.status == determinant::FactorizationStatus::REGULAR);
  CHECK(metadata.phase == (expected < 0 ? -1.0 : 1.0));
  CHECK(metadata.log_abs == Catch::Approx(
      static_cast<double>(std::log(std::abs(expected)))).epsilon(tolerance).margin(tolerance));
}

void checkInverseIdentity(const std::vector<double>& matrix,
                          const std::vector<double>& inverse,
                          std::size_t matrix_size,
                          double tolerance = 2.0e-11)
{
  for (std::size_t row = 0; row < matrix_size; ++row)
    for (std::size_t column = 0; column < matrix_size; ++column)
    {
      long double product = 0;
      for (std::size_t inner = 0; inner < matrix_size; ++inner)
        product += static_cast<long double>(matrix[row * matrix_size + inner]) *
            inverse[inner * matrix_size + column];
      CHECK(static_cast<double>(product) ==
            Catch::Approx(row == column ? 1.0 : 0.0).epsilon(tolerance).margin(tolerance));
    }
}

} // namespace

TEST_CASE("PsiFormer device LU matches independent small determinant oracles",
          "[psiformer][device][determinant]")
{
  const std::vector<std::vector<double>> matrices{
      {-3.5},
      {0.0, 2.0, 1.0, 3.0},
      {4.0, 1.0, -2.0, 0.5,
       1.5, 3.0, 0.25, -1.0,
       -2.0, 0.5, 5.0, 1.25,
       0.75, -1.5, 2.0, 4.5}};
  const std::array<std::size_t, 3> sizes{1, 2, 4};
  for (std::size_t test = 0; test < matrices.size(); ++test)
  {
    FactorizationStorage storage(sizes[test]);
    const auto metadata = factor(matrices[test], sizes[test], storage);
    checkMetadata(metadata, matrices[test], sizes[test]);
    CHECK(metadata.inverse_available);
    checkInverseIdentity(matrices[test], storage.inverse, sizes[test]);
  }

  FactorizationStorage value_only_storage(1);
  const auto value_only = factor(matrices[0], 1, value_only_storage, false);
  CHECK(value_only.status == determinant::FactorizationStatus::REGULAR);
  CHECK_FALSE(value_only.inverse_available);
}

TEST_CASE("PsiFormer device LU pivoting is deterministic and singularity is exact",
          "[psiformer][device][determinant]")
{
  const std::vector<double> tied{2.0, 1.0, -2.0, 3.0};
  FactorizationStorage tied_storage(2);
  const auto tied_metadata = factor(tied, 2, tied_storage);
  checkMetadata(tied_metadata, tied, 2);
  CHECK(tied_storage.permutation == std::vector<std::size_t>{0, 1});

  const std::vector<double> swapped{0.0, 2.0, 1.0, 3.0};
  FactorizationStorage swapped_storage(2);
  const auto swapped_metadata = factor(swapped, 2, swapped_storage);
  CHECK(swapped_storage.permutation == std::vector<std::size_t>{1, 0});
  CHECK(swapped_metadata.phase == -1.0);

  const std::vector<double> singular{1.0, 2.0, 2.0, 4.0};
  FactorizationStorage singular_storage(2);
  std::fill(singular_storage.inverse.begin(), singular_storage.inverse.end(), 17.0);
  const auto singular_metadata = factor(singular, 2, singular_storage);
  CHECK(singular_metadata.status == determinant::FactorizationStatus::SINGULAR);
  CHECK(singular_metadata.phase == 0.0);
  CHECK(singular_metadata.log_abs == -std::numeric_limits<double>::infinity());
  CHECK_FALSE(singular_metadata.inverse_available);
  CHECK(std::all_of(singular_storage.inverse.begin(), singular_storage.inverse.end(),
                    [](double value) { return value == 0.0; }));

  const std::vector<double> nonfinite{1.0, 0.0, 0.0,
                                      std::numeric_limits<double>::quiet_NaN()};
  FactorizationStorage nonfinite_storage(2);
  const auto nonfinite_metadata = factor(nonfinite, 2, nonfinite_storage);
  CHECK(nonfinite_metadata.status == determinant::FactorizationStatus::NONFINITE_INPUT);
  CHECK_FALSE(nonfinite_metadata.inverse_available);
}

TEST_CASE("PsiFormer device LU remains scaled across extreme magnitudes",
          "[psiformer][device][determinant]")
{
  for (double scale : {1.0e-200, 1.0e200})
  {
    const std::vector<double> matrix{2.0 * scale, 0.25 * scale,
                                     -0.5 * scale, 1.5 * scale};
    FactorizationStorage storage(2);
    const auto metadata = factor(matrix, 2, storage);
    checkMetadata(metadata, matrix, 2, 2.0e-12);
    CHECK(metadata.matrix_scale == 2.0 * scale);
    CHECK(metadata.minimum_scaled_pivot > 0.0);
    checkInverseIdentity(matrix, storage.inverse, 2, 5.0e-12);
  }

  const std::vector<double> near_singular{1.0, 1.0, 1.0, 1.0 + 0x1p-40};
  FactorizationStorage storage(2);
  const auto metadata = factor(near_singular, 2, storage);
  CHECK(metadata.status == determinant::FactorizationStatus::REGULAR);
  CHECK(metadata.minimum_scaled_pivot > 0.0);
  checkInverseIdentity(near_singular, storage.inverse, 2, 2.0e-4);
}

TEST_CASE("PsiFormer determinant batch indexing keeps configurations isolated",
          "[psiformer][device][determinant]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t channels       = 2;
  constexpr std::size_t matrix_size    = 2;
  constexpr std::size_t matrix_elements = matrix_size * matrix_size;
  const std::vector<double> matrices{
      1.0, 0.0, 0.0, 2.0,
      0.0, 1.0, 3.0, 0.0,
      -2.0, 0.0, 0.0, 4.0,
      2.0, 1.0, 1.0, 2.0};
  std::vector<double> lu(matrices.size());
  std::vector<double> inverse(matrices.size());
  std::vector<std::size_t> permutation(configurations * channels * matrix_size);
  std::vector<double> solve(configurations * channels * matrix_size);
  std::vector<determinant::FactorizationMetadata> metadata(configurations * channels);

  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      const std::size_t matrix = determinant::matrixIndex(configuration, channel, channels);
      const std::size_t offset = determinant::matrixOffset(matrix, matrix_size);
      metadata[matrix] = determinant::factorizeScaledReal(
          matrices.data() + offset, matrix_size, lu.data() + offset,
          permutation.data() + matrix * matrix_size, true,
          inverse.data() + offset, solve.data() + matrix * matrix_size);
      const long double expected = referenceDeterminant(matrices.data() + offset, matrix_size);
      CHECK(metadata[matrix].phase == (expected < 0 ? -1.0 : 1.0));
      CHECK(metadata[matrix].log_abs == Catch::Approx(
          static_cast<double>(std::log(std::abs(expected)))).epsilon(5.0e-13));
      std::vector<double> one_matrix(matrices.begin() + offset,
                                     matrices.begin() + offset + matrix_elements);
      std::vector<double> one_inverse(inverse.begin() + offset,
                                      inverse.begin() + offset + matrix_elements);
      checkInverseIdentity(one_matrix, one_inverse, matrix_size);
    }

  CHECK(determinant::matrixIndex(1, 0, channels) == 2);
  CHECK(determinant::matrixOffset(3, matrix_size) == 12);
}

TEST_CASE("PsiFormer determinant channels preserve signed-log and singular semantics",
          "[psiformer][device][determinant]")
{
  std::vector<determinant::FactorizationMetadata> channels{
      signedLogChannel(1.0, std::log(2.0)),
      signedLogChannel(-1.0, std::log(3.0)),
      signedLogChannel(1.0, std::log(5.0)),
      determinant::FactorizationMetadata{}};
  std::array<double, 4> coefficients{1.0, -2.0, 0.0, 0.0};
  CombinationStorage storage(channels.size());
  auto result = combine(channels, coefficients.data(), storage);

  CHECK(result.status == determinant::CombinationStatus::REGULAR);
  CHECK(result.phase == 1.0);
  CHECK(result.log_abs == Catch::Approx(std::log(8.0)));
  CHECK(result.singular_channels == 1);
  CHECK(result.all_required_inverses_available);
  CHECK(result.normalized_weights_available);
  CHECK(storage.term_phase == std::vector<double>{1.0, 1.0, 0.0, 0.0});
  CHECK(storage.weights[0] == Catch::Approx(0.25));
  CHECK(storage.weights[1] == Catch::Approx(0.75));
  CHECK(storage.weights[2] == 0.0);
  CHECK(storage.weights[3] == 0.0);

  coefficients[3] = 1.0;
  result = combine(channels, coefficients.data(), storage);
  CHECK(result.status == determinant::CombinationStatus::REGULAR);
  CHECK_FALSE(result.all_required_inverses_available);
  CHECK(storage.scaled_terms[3] == 0.0);

  coefficients[0] = std::numeric_limits<double>::infinity();
  result = combine(channels, coefficients.data(), storage);
  CHECK(result.status == determinant::CombinationStatus::NONFINITE_INPUT);
  CHECK_FALSE(result.normalized_weights_available);
}

TEST_CASE("PsiFormer determinant channel cancellation is explicit and deterministic",
          "[psiformer][device][determinant]")
{
  const auto unit = signedLogChannel(1.0, 0.0);
  {
    const std::vector<determinant::FactorizationMetadata> channels{unit, unit};
    const std::array<double, 2> coefficients{1.0, -1.0};
    CombinationStorage storage(channels.size());
    const auto result = combine(channels, coefficients.data(), storage);
    CHECK(result.status == determinant::CombinationStatus::NODE);
    CHECK(result.phase == 0.0);
    CHECK(result.log_abs == -std::numeric_limits<double>::infinity());
    CHECK(result.sum_abs_scaled == 2.0);
    CHECK(result.abs_sum_scaled == 0.0);
    CHECK(result.underflowed_nonzero_terms == 0);
    CHECK(result.severe_cancellation);
    CHECK_FALSE(result.normalized_weights_available);
    CHECK(storage.weights == std::vector<double>{0.0, 0.0});
  }

  {
    constexpr double residual = 0x1p-40;
    const std::vector<determinant::FactorizationMetadata> channels{
        unit, signedLogChannel(1.0, std::log1p(-residual))};
    const std::array<double, 2> coefficients{1.0, -1.0};
    CombinationStorage storage(channels.size());
    const auto result = combine(channels, coefficients.data(), storage);
    CHECK(result.status == determinant::CombinationStatus::REGULAR);
    CHECK(result.phase == 1.0);
    CHECK(result.log_abs == Catch::Approx(std::log(residual)).epsilon(2.0e-4));
    CHECK_FALSE(result.severe_cancellation);
    CHECK(result.normalized_weights_available);
    CHECK(storage.weights[0] == Catch::Approx(1.0 / residual).epsilon(2.0e-4));
    CHECK(storage.weights[1] == Catch::Approx(-(1.0 / residual - 1.0)).epsilon(2.0e-4));
  }

  {
    const std::vector<determinant::FactorizationMetadata> channels{
        unit, unit, signedLogChannel(1.0, -800.0)};
    const std::array<double, 3> coefficients{1.0, -1.0, 1.0};
    CombinationStorage storage(channels.size());
    const auto result = combine(channels, coefficients.data(), storage);
    CHECK(result.status == determinant::CombinationStatus::NODE);
    CHECK(result.underflowed_nonzero_terms == 1);
    CHECK(result.sum_abs_scaled == 2.0);
    CHECK(result.abs_sum_scaled == 0.0);
    CHECK(result.severe_cancellation);
    CHECK(storage.scaled_terms[2] == 0.0);
  }
}

TEST_CASE("PsiFormer determinant channel batches remain configuration isolated",
          "[psiformer][device][determinant]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t channels_per_configuration = 2;
  const std::vector<determinant::FactorizationMetadata> channels{
      signedLogChannel(1.0, 0.0), signedLogChannel(1.0, std::log(2.0)),
      signedLogChannel(-1.0, std::log(4.0)), signedLogChannel(1.0, 0.0)};
  CombinationStorage storage(channels.size());
  std::array<determinant::CombinationMetadata, configurations> results;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const std::size_t offset = configuration * channels_per_configuration;
    results[configuration] = determinant::combineChannelsReal(
        channels.data() + offset, nullptr, channels_per_configuration,
        storage.term_phase.data() + offset, storage.term_log_abs.data() + offset,
        storage.scaled_terms.data() + offset, storage.weights.data() + offset);
  }

  CHECK(results[0].phase == 1.0);
  CHECK(results[0].log_abs == Catch::Approx(std::log(3.0)));
  CHECK(storage.weights[0] == Catch::Approx(1.0 / 3.0));
  CHECK(storage.weights[1] == Catch::Approx(2.0 / 3.0));
  CHECK(results[1].phase == -1.0);
  CHECK(results[1].log_abs == Catch::Approx(std::log(3.0)));
  CHECK(storage.weights[2] == Catch::Approx(4.0 / 3.0));
  CHECK(storage.weights[3] == Catch::Approx(-1.0 / 3.0));
}

TEST_CASE("PsiFormer determinant reverse and spatial traces match finite differences",
          "[psiformer][device][determinant]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t channels       = 2;
  constexpr std::size_t matrix_size    = 2;
  constexpr std::size_t matrix_elements = matrix_size * matrix_size;
  constexpr std::size_t gradient_lanes = 3;
  constexpr std::size_t electrons      = 1;
  constexpr std::size_t channel_elements = channels * matrix_elements;
  const std::array<double, channels> coefficients{1.0, -0.35};
  const std::vector<double> matrices{
      1.4, 0.2, -0.1, 1.1,
      0.8, -0.3, 0.4, 1.2,
      1.1, 0.25, 0.15, 0.95,
      1.3, -0.2, 0.1, 0.85};
  const std::array<double, 8> gradient_pattern{
      0.07, -0.03, 0.02, 0.05, -0.04, 0.06, 0.01, -0.02};
  const std::array<double, 8> laplacian_pattern{
      0.03, 0.01, -0.02, 0.04, -0.01, 0.02, 0.05, -0.03};
  std::vector<double> gradients(configurations * gradient_lanes * channel_elements);
  std::vector<double> laplacians(configurations * electrons * channel_elements);
  for (std::size_t element = 0; element < gradients.size(); ++element)
    gradients[element] = gradient_pattern[element % gradient_pattern.size()] *
        (1.0 + 0.1 * static_cast<double>(element / gradient_pattern.size()));
  for (std::size_t element = 0; element < laplacians.size(); ++element)
    laplacians[element] = laplacian_pattern[element % laplacian_pattern.size()] *
        (1.0 + 0.2 * static_cast<double>(element / laplacian_pattern.size()));

  std::vector<double> lu(matrices.size());
  std::vector<double> inverses(matrices.size());
  std::vector<std::size_t> permutations(configurations * channels * matrix_size);
  std::vector<double> solve(configurations * channels * matrix_size);
  std::vector<determinant::FactorizationMetadata> factorization(configurations * channels);
  std::vector<double> term_phase(configurations * channels);
  std::vector<double> term_log_abs(configurations * channels);
  std::vector<double> scaled_terms(configurations * channels);
  std::vector<double> weights(configurations * channels);
  std::array<determinant::CombinationMetadata, configurations> combination;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      const std::size_t matrix = configuration * channels + channel;
      const std::size_t offset = matrix * matrix_elements;
      factorization[matrix] = determinant::factorizeScaledReal(
          matrices.data() + offset, matrix_size, lu.data() + offset,
          permutations.data() + matrix * matrix_size, true,
          inverses.data() + offset, solve.data() + matrix * matrix_size);
    }
    const std::size_t channel_offset = configuration * channels;
    combination[configuration] = determinant::combineChannelsReal(
        factorization.data() + channel_offset, coefficients.data(), channels,
        term_phase.data() + channel_offset, term_log_abs.data() + channel_offset,
        scaled_terms.data() + channel_offset, weights.data() + channel_offset);
  }

  const auto reference_wave = [&](const std::vector<double>& batch,
                                  std::size_t configuration) {
    long double value = 0.0L;
    const std::size_t base = configuration * channel_elements;
    for (std::size_t channel = 0; channel < channels; ++channel)
      value += static_cast<long double>(coefficients[channel]) *
          referenceDeterminant(batch.data() + base + channel * matrix_elements, matrix_size);
    return value;
  };

  std::vector<double> reverse_seeds(matrices.size());
  std::array<determinant::DerivativeStatus, configurations> reverse_status;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const std::size_t channel_offset = configuration * channels;
    const std::size_t matrix_offset  = configuration * channel_elements;
    reverse_status[configuration] = determinant::fillMatrixReverseSeeds(
        factorization.data() + channel_offset, combination[configuration],
        weights.data() + channel_offset, inverses.data() + matrix_offset,
        channels, matrix_size, reverse_seeds.data() + matrix_offset);
    CHECK(reverse_status[configuration] == determinant::DerivativeStatus::AVAILABLE);
  }

  constexpr double reverse_step = 1.0e-6;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t element = 0; element < channel_elements; ++element)
    {
      std::vector<double> plus = matrices;
      std::vector<double> minus = matrices;
      const std::size_t offset = configuration * channel_elements + element;
      plus[offset] += reverse_step;
      minus[offset] -= reverse_step;
      const double finite_difference = static_cast<double>(
          (std::log(std::abs(reference_wave(plus, configuration))) -
           std::log(std::abs(reference_wave(minus, configuration)))) /
          (2.0L * reverse_step));
      CHECK(reverse_seeds[offset] == Catch::Approx(finite_difference).epsilon(2.0e-8).margin(2.0e-8));
    }

  std::vector<double> matrix_product_scratch(configurations * matrix_elements);
  std::vector<double> output_gradient(configurations * gradient_lanes);
  std::vector<double> output_lap_ratio(configurations * electrons);
  std::vector<double> output_lap_log(configurations * electrons);
  std::array<determinant::DerivativeStatus, configurations> spatial_status;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const std::size_t channel_offset = configuration * channels;
    const std::size_t matrix_offset  = configuration * channel_elements;
    spatial_status[configuration] = determinant::combineSpatialTraces(
        factorization.data() + channel_offset, combination[configuration],
        weights.data() + channel_offset, inverses.data() + matrix_offset,
        gradients.data() + configuration * gradient_lanes * channel_elements,
        laplacians.data() + configuration * electrons * channel_elements,
        channels, matrix_size, gradient_lanes, electrons,
        matrix_product_scratch.data() + configuration * matrix_elements,
        output_gradient.data() + configuration * gradient_lanes,
        output_lap_ratio.data() + configuration * electrons,
        output_lap_log.data() + configuration * electrons);
    CHECK(spatial_status[configuration] == determinant::DerivativeStatus::AVAILABLE);
  }

  constexpr double spatial_step = 2.0e-4;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const long double wave = reference_wave(matrices, configuration);
    long double finite_difference_lap_ratio = 0.0L;
    long double squared_gradient            = 0.0L;
    for (std::size_t lane = 0; lane < gradient_lanes; ++lane)
    {
      std::vector<double> plus = matrices;
      std::vector<double> minus = matrices;
      const std::size_t matrix_offset = configuration * channel_elements;
      const std::size_t gradient_offset =
          (configuration * gradient_lanes + lane) * channel_elements;
      const std::size_t laplacian_offset = configuration * channel_elements;
      for (std::size_t element = 0; element < channel_elements; ++element)
      {
        const double first = gradients[gradient_offset + element];
        const double diagonal_second = laplacians[laplacian_offset + element] / 3.0;
        plus[matrix_offset + element] += spatial_step * first +
            0.5 * spatial_step * spatial_step * diagonal_second;
        minus[matrix_offset + element] += -spatial_step * first +
            0.5 * spatial_step * spatial_step * diagonal_second;
      }
      const long double plus_wave  = reference_wave(plus, configuration);
      const long double minus_wave = reference_wave(minus, configuration);
      const long double finite_difference_gradient =
          (plus_wave - minus_wave) / (2.0L * spatial_step * wave);
      finite_difference_lap_ratio +=
          (plus_wave - 2.0L * wave + minus_wave) /
          (spatial_step * spatial_step * wave);
      squared_gradient += finite_difference_gradient * finite_difference_gradient;
      CHECK(output_gradient[configuration * gradient_lanes + lane] ==
            Catch::Approx(static_cast<double>(finite_difference_gradient))
                .epsilon(2.0e-8).margin(2.0e-8));
    }
    CHECK(output_lap_ratio[configuration] ==
          Catch::Approx(static_cast<double>(finite_difference_lap_ratio))
              .epsilon(2.0e-6).margin(2.0e-7));
    CHECK(output_lap_log[configuration] ==
          Catch::Approx(static_cast<double>(finite_difference_lap_ratio - squared_gradient))
              .epsilon(2.0e-6).margin(2.0e-7));
  }

  std::array<double, channel_elements> unavailable_output;
  unavailable_output.fill(7.0);
  const auto node_status = determinant::fillMatrixReverseSeeds(
      factorization.data(), determinant::CombinationMetadata{}, weights.data(),
      inverses.data(), channels, matrix_size, unavailable_output.data());
  CHECK(node_status == determinant::DerivativeStatus::NODE);
  CHECK(std::all_of(unavailable_output.begin(), unavailable_output.end(),
                    [](double value) { return value == 0.0; }));

  auto unavailable_combination = combination[0];
  unavailable_combination.all_required_inverses_available = false;
  const auto unavailable_status = determinant::fillMatrixReverseSeeds(
      factorization.data(), unavailable_combination, weights.data(), inverses.data(),
      channels, matrix_size, unavailable_output.data());
  CHECK(unavailable_status == determinant::DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE);
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
