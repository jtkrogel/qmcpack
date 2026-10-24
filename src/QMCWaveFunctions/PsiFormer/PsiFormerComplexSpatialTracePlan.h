//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerComplexSpatialTracePlan.h
 * @brief Pointer-free checked scratch inventory for concurrent forward traces.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_SPATIAL_TRACE_PLAN_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_SPATIAL_TRACE_PLAN_H
#include "PsiFormerSpatialLayout.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::psiformer::device
{
/// Explicit forward-only scratch; model/head/output storage remains with the owner.
struct ComplexSpatialTraceWorkspacePlan
{
  std::size_t target_capacity = 0, product_elements = 0, channel_elements = 0, device_bytes = 0;
};

/// Check immutable concurrency and every byte/grid extent without preparing a device.
inline ComplexSpatialTraceWorkspacePlan planComplexSpatialTraceWorkspace(std::size_t batch,
    std::size_t determinants, std::size_t n, std::size_t targets, std::size_t concurrent_targets)
{
  if (!concurrent_targets)
    return {};
  if (concurrent_targets > 8 || !batch || !determinants || !n || !targets || targets > n)
    throw std::invalid_argument("Complex spatial trace concurrency requires valid extents and capacity1..8");
  const auto mul = [](auto a, auto b) {
    return spatial_detail::checkedProduct(a, b, "Complex concurrent spatial trace extent overflow");
  };
  const auto capacity = std::min(targets, concurrent_targets);
  const auto cells = mul(mul(batch, capacity), determinants);
  if (cells > std::size_t(std::numeric_limits<int>::max()) ||
      batch - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::length_error("Complex concurrent spatial trace grid exceeds launch/status admission");
  const auto products = mul(cells, mul(n, n)), channels = mul(cells, std::size_t(4));
  const auto elements = spatial_detail::checkedAdd(products, channels,
      "Complex concurrent spatial trace storage overflow");
  return {capacity, products, channels, mul(elements, std::size_t(16))};
}
} // namespace qmcplusplus::psiformer::device
#endif
