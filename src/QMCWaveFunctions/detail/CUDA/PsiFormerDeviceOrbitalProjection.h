//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceOrbitalProjection.h
 * @brief Fixed-capacity device pointer banks for independent determinant GEMMs.
 * One ordered pointer-write launch binds current addresses/layout per component;
 * no neural packing, HOST pointer transfer, allocation or fence occurs in use.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_PROJECTION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ORBITAL_PROJECTION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceSpatial.h"

namespace qmcplusplus::psiformer::device
{
/// Exact DEVICE bytes for three pointer arrays; HOST numeric storage is zero.
inline std::size_t orbitalProjectionPointerBytes(std::size_t batch, std::size_t planes,
                                                std::size_t determinants, std::size_t spins)
{
  const auto mul = [](auto a, auto b) {
    return spatial_detail::checkedProduct(a, b, "Orbital projection pointer extent overflow");
  };
  if (!batch || !planes || !determinants || !spins || spins > 2 ||
      determinants > std::size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Orbital projection pointer bank requires admitted positive extents");
  const auto entries = mul(mul(mul(batch, planes), determinants), spins);
  if (entries > std::size_t(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("Orbital projection pointer grid exceeds its launch ABI");
  return mul(entries, sizeof(const double*) * 2 + sizeof(double*));
}

/// Bind all live layout/address triples on the same queue as their consumers.
Error launchOrbitalProjectionPointers(Stream stream, const OpenOrbitalJetLayout& layout,
    const double* features, const double* up, const double* down, double* output,
    const double** weights, const double** source, double** destinations);

/// Execute one independent D group through the existing vendor BLAS wrapper.
void projectOrbitalDeterminantsBatched(AcceleratorBlasHandle& handle, const DenseForwardLayout& layout,
    std::size_t determinants, const double* const* weights, const double* const* source,
    double* const* output);

/// Reject a caller that would race pointer refresh against a different BLAS queue.
void validateOrbitalProjectionStream(const AcceleratorBlasHandle& handle, Stream stream);

/** An exclusive owner/queue orders pointer refresh before grouped consumers.
 * Refresh occurs once per complete component projection, never per group.
 * No addresses persist semantically between stages, so noexcept model adoption
 * cannot leave stale weights or target-tail offsets in the next operation.
 */
class PreparedDeviceOrbitalProjection
{
public:
  explicit PreparedDeviceOrbitalProjection(const OpenOrbitalJetLayout& layout)
      : capacity_(layout), entries_(entryCount(layout)),
        weights_(entries_, "OrbitalProjection::weights", __FILE__, __LINE__),
        source_(entries_, "OrbitalProjection::source", __FILE__, __LINE__),
        output_(entries_, "OrbitalProjection::output", __FILE__, __LINE__)
  {
  }

  /// Validate the full live layout before writing any pointer or numerical output.
  void project(AcceleratorBlasHandle& handle, Stream stream, const OpenOrbitalJetLayout& layout,
      const double* features, const double* up, const double* down, double* output) const
  {
    (void)entryCount(layout);
    validateOrbitalProjectionStream(handle, stream);
    if (!layout.features.configuration_count ||
        layout.features.configuration_count > capacity_.features.configuration_count ||
        layout.features.plane_count > capacity_.features.plane_count ||
        layout.features.electron_count != capacity_.features.electron_count ||
        layout.determinant_count != capacity_.determinant_count ||
        layout.spin_up_count != capacity_.spin_up_count || layout.feature_width != capacity_.feature_width ||
        layout.feature_row_stride != capacity_.feature_row_stride ||
        layout.orbital_row_stride != capacity_.orbital_row_stride ||
        layout.orbital_determinant_stride != capacity_.orbital_determinant_stride ||
        layout.backflow_feature_stride != capacity_.backflow_feature_stride ||
        layout.backflow_determinant_stride != capacity_.backflow_determinant_stride ||
        spatialJetSpanElements(layout.features) > spatialJetSpanElements(capacity_.features) ||
        spatialJetSpanElements(layout.orbitals) > spatialJetSpanElements(capacity_.orbitals) ||
        !features || !output || output == features || output == up || output == down ||
        (layout.spin_up_count && !up) ||
        (layout.spin_up_count != layout.features.electron_count && !down))
      throw std::invalid_argument("Orbital projection exceeds its prepared pointer/layout capacity");
    checkRuntime(launchOrbitalProjectionPointers(stream, layout, features, up, down, output,
        weights_.data(), source_.data(), output_.data()), "orbital projection current pointers");
    std::size_t bank_spin = 0;
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto rows = spin ? layout.features.electron_count - layout.spin_up_count : layout.spin_up_count;
      if (!rows)
        continue;
      const DenseForwardLayout dense{rows, layout.feature_width, layout.features.electron_count,
          layout.feature_row_stride, layout.backflow_feature_stride, layout.orbital_row_stride};
      for (std::size_t b = 0; b < layout.features.configuration_count; ++b)
        for (std::size_t p = 0; p < layout.features.plane_count; ++p)
        {
          const auto offset = ((bank_spin * layout.features.configuration_count + b) *
              layout.features.plane_count + p) * layout.determinant_count;
          projectOrbitalDeterminantsBatched(handle, dense, layout.determinant_count,
              weights_.data() + offset, source_.data() + offset, output_.data() + offset);
        }
      ++bank_spin;
    }
  }

  /// Count actual DEVICE capacities; no HOST numeric pointer bank exists.
  std::size_t deviceBytes() const noexcept
  {
    return entries_ * (sizeof(const double*) * 2 + sizeof(double*));
  }

private:
  /// Complete layout and byte validation precedes every pointer allocation.
  static std::size_t entryCount(const OpenOrbitalJetLayout& layout)
  {
    validateOpenOrbitalJetLayout(layout);
    for (std::size_t spin = 0; spin < 2; ++spin)
    {
      const auto rows = spin ? layout.features.electron_count - layout.spin_up_count : layout.spin_up_count;
      if (rows)
        validateDenseForwardLayout({rows, layout.feature_width, layout.features.electron_count,
            layout.feature_row_stride, layout.backflow_feature_stride, layout.orbital_row_stride});
    }
    return orbitalProjectionPointerBytes(layout.features.configuration_count,
        layout.features.plane_count, layout.determinant_count,
        (layout.spin_up_count != 0) + (layout.spin_up_count != layout.features.electron_count)) /
        (sizeof(const double*) * 2 + sizeof(double*));
  }

  const OpenOrbitalJetLayout capacity_;
  const std::size_t entries_;
  mutable DeviceBuffer<const double*> weights_, source_;
  mutable DeviceBuffer<double*> output_;
};
} // namespace qmcplusplus::psiformer::device
#endif
