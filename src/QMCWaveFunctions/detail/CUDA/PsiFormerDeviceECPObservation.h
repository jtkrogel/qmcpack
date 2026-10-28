//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_ECP_OBSERVATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_ECP_OBSERVATION_H

#include "PsiFormerDeviceInference.h"
#include "PsiFormerDeviceOpenKineticReverse.h"
#include <cstring>
#include <limits>
#include <string>

namespace qmcplusplus::psiformer::device
{
/** Own immutable physical finals, not a borrow of a reusable neural FULL tape.
 * Native ECP evaluation still consumes HOST coordinates/G/L. Only its scalar
 * energies and actual grid identity are attached after the complete traversal.
 */
class PreparedDeviceECPObservation
{
public:
  struct Cost { std::size_t device_bytes, host_bytes; };

  static Cost estimate(std::size_t samples, std::size_t electrons)
  {
    if (!samples || !electrons) throw std::invalid_argument("ECP observation requires nonempty physical extents");
    const auto mul = [](std::size_t a, std::size_t b) {
      return spatial_detail::checkedProduct(a, b, "ECP observation extent overflow"); };
    const auto add = [](std::size_t a, std::size_t b) {
      return spatial_detail::checkedAdd(a, b, "ECP observation extent overflow"); };
    return {checkedDeviceBytes<double>(add(mul(samples, add(mul(7, electrons), 4)), electrons)),
            checkedDeviceBytes<double>(add(mul(samples, add(mul(3, electrons), 4)), electrons))};
  }

  PreparedDeviceECPObservation(std::shared_ptr<const PreparedDeviceModel> image,
      std::size_t samples, const std::vector<double>& inverse_masses,
      std::shared_ptr<const std::size_t> source, std::string physical_source, std::size_t device_cap, std::size_t host_cap)
      : image_(std::move(image)), source_(std::move(source)), samples_(samples),
        electrons_(image_ ? image_->plan().modelShape().electrons() : 0),
        layout_(makeSpatialJetLayout(samples, 1, electrons_, SpatialJetMode::FULL_VGL)), physical_source_(std::move(physical_source))
  {
    if (!image_ || !source_ || physical_source_.empty() || image_->periodicCell() ||
        image_->plan().environment().amplitude_scalar_domain != ScalarDomain::REAL ||
        inverse_masses.size() != electrons_ || !device_cap || !host_cap ||
        device_cap == std::numeric_limits<std::size_t>::max() || host_cap == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("ECP observation requires admitted real OPEN image/source/masses and finite caps");
    image_->validateCurrentDevice();
    const auto cost = estimate(samples_, electrons_);
    if (cost.device_bytes > device_cap || cost.host_bytes > host_cap)
      throw std::length_error("ECP observation numeric owner exceeds its explicit cap");
    for (const double& mass : inverse_masses)
      if (!finite(mass) || mass <= 0) throw std::invalid_argument("ECP observation mass is invalid");
    coordinates_ = std::make_shared<coordinate_detail::SourceState>(image_->metadataFamily(), electrons_, samples_);
    roots_.reserve(spatialJetSpanElements(layout_));
    masses_.reserve(electrons_); masses_.upload(inverse_masses.data(), inverse_masses.size());
    kinetic_.reserve(samples_); energy_.reserve(samples_); status_.reserve(samples_);
    host_positions_.reserve(3 * samples_ * electrons_);
    host_masses_ = inverse_masses;
    host_kinetic_.resize(samples_); host_energy_.resize(samples_); host_status_.resize(samples_);
    generations_.reserve(samples_);
    if (hostBytes() > host_cap || deviceBytes() > device_cap)
      throw std::length_error("ECP observation actual numeric capacities exceed admission");
  }

  ~PreparedDeviceECPObservation()
  {
    std::lock_guard lock(coordinates_->mutex);
    coordinates_->ready = false;
  }

  /// Copy each complete microtile while its producer queue/tape is exclusively leased.
  void append(const PreparedDeviceInference::SpatialFinalView& tape,
      compute::Queue<PlatformKind::CUDA>& queue, const std::vector<double>& positions,
      std::size_t count, std::size_t coordinate_generation)
  {
    image_->validateCurrentDevice();
    if (captured_ || tape.image != image_.get() || !tape.evaluation_generation || !count || count > samples_ - filled_ || !coordinate_generation ||
        (!generations_.empty() && coordinate_generation <= generations_.back()) ||
        positions.size() != 3 * count * electrons_ || tape.root_layout.mode != SpatialJetMode::FULL_VGL ||
        tape.root_layout.configuration_count != count || tape.root_layout.element_count != 1 ||
        tape.root_layout.electron_count != electrons_ || tape.root_layout.target_electron_begin != 0 ||
        tape.root_layout.laplacian_lanes != electrons_ || tape.root_layout.plane_stride != 1 ||
        tape.root_layout.configuration_stride != layout_.configuration_stride || !tape.root || !tape.positions)
      throw std::invalid_argument("ECP observation microtile is incomplete, stale or has a foreign FULL layout");
    for (const double& x : positions)
      if (!finite(x)) throw std::invalid_argument("ECP observation coordinate proof is nonfinite");
    copy(queue, coordinates_->positions.data() + 3 * filled_ * electrons_, tape.positions, positions.size());
    copy(queue, roots_.data() + filled_ * layout_.configuration_stride, tape.root,
        count * layout_.configuration_stride);
    queue.sync();
    host_positions_.insert(host_positions_.end(), positions.begin(), positions.end());
    generations_.push_back(coordinate_generation);
    filled_ += count;
  }

  void finishCapture(std::size_t version)
  {
    if (captured_ || filled_ != samples_ || version != image_->version())
      throw std::invalid_argument("ECP observation cannot finish incomplete or stale capture");
    coordinates_->generation = generations_.back();
    coordinates_->live = samples_;
    coordinates_->ready = true;
    captured_ = true;
  }

  /// Seal once, only after native physical ECP has completed its accepted grid.
  void seal(compute::Queue<PlatformKind::CUDA>& queue, std::size_t version, std::uint64_t grid,
      const std::string& physical_source, const std::vector<double>& positions,
      const std::vector<double>& energies, const std::vector<double>& native_kinetic)
  {
    image_->validateCurrentDevice();
    if (!captured_ || sealed_ || version != image_->version() || !grid || physical_source != physical_source_ ||
        positions != host_positions_ || energies.size() != samples_ || native_kinetic.size() != samples_)
      throw std::invalid_argument("ECP observation cannot seal a stale/foreign coordinate, grid or source receipt");
    for (std::size_t i = 0; i < samples_; ++i)
      if (!finite(energies[i]) || !finite(native_kinetic[i]))
        throw std::invalid_argument("ECP observation native energy is nonfinite");
    checkRuntime(launchResidentFinalKinetic(queue.getNative(), layout_, roots_.data(), masses_.data(),
        kinetic_.data(), status_.data()), "ECP immutable final kinetic observation");
    queue.sync();
    kinetic_.download(host_kinetic_.data(), samples_); status_.download(host_status_.data(), samples_);
    for (std::size_t i = 0; i < samples_; ++i)
      if (host_status_[i] != 0 || !finite(host_kinetic_[i]) ||
          std::abs(host_kinetic_[i] - native_kinetic[i]) >
              2048 * std::numeric_limits<double>::epsilon() * (1 + std::abs(native_kinetic[i])))
        throw std::runtime_error("ECP immutable device kinetic differs from its native physical evaluation");
    host_energy_ = energies; energy_.upload(host_energy_.data(), samples_);
    grid_ = grid;
    sealed_ = true;
  }

  void validate(std::size_t version, std::uint64_t grid, const std::string& physical_source,
      std::size_t coordinate_generation) const
  {
    if (!sealed_ || version != image_->version() || grid != grid_ || physical_source != physical_source_ ||
        coordinate_generation != coordinates_->generation)
      throw std::invalid_argument("ECP observation receipt is stale or belongs to another source/grid/generation");
    image_->validateCurrentDevice();
  }

  PackedDeviceCoordinatesView view(std::size_t begin, std::size_t count, std::size_t version) const
  {
    if (!sealed_ || version != image_->version() || !count || begin > samples_ || count > samples_ - begin)
      throw std::invalid_argument("ECP observation coordinate view is unsealed, stale or out of range");
    return PackedDeviceCoordinatesView(coordinates_, begin, count);
  }

  std::size_t sourceVersion() const noexcept { return image_->version(); }
  std::size_t coordinateGeneration() const noexcept { return coordinates_->generation; }
  std::uint64_t gridFingerprint() const noexcept { return grid_; }
  const std::string& physicalSource() const noexcept { return physical_source_; }
  std::size_t sampleCount() const noexcept { return samples_; }
  const std::vector<double>& energies() const { if (!sealed_) throw std::logic_error("Unsealed ECP energy"); return host_energy_; }
  const std::vector<double>& kinetic() const { if (!sealed_) throw std::logic_error("Unsealed ECP kinetic"); return host_kinetic_; }
  std::size_t coordinateBytes() const { return checkedDeviceBytes<double>(coordinates_->positions.capacity()); }
  std::size_t deviceBytes() const
  { return checkedDeviceBytes<double>(coordinates_->positions.capacity() + roots_.capacity() + masses_.capacity() +
      kinetic_.capacity() + energy_.capacity()) + checkedDeviceBytes<std::uint64_t>(status_.capacity()); }
  std::size_t hostBytes() const
  { return sizeof(double) * (host_positions_.capacity() + host_masses_.capacity() + host_kinetic_.capacity() +
      host_energy_.capacity()) + sizeof(std::uint64_t) * host_status_.capacity() + sizeof(std::size_t) * generations_.capacity(); }

private:
  friend class PreparedDeviceBatchBackend;
  static bool finite(const double& value)
  { std::uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000); }
  static void copy(compute::Queue<PlatformKind::CUDA>& queue, double* dst, const double* src, std::size_t n)
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(dst, src, checkedDeviceBytes<double>(n), hipMemcpyDeviceToDevice, queue.getNative()), "ECP final D2D copy");
#else
    checkRuntime(cudaMemcpyAsync(dst, src, checkedDeviceBytes<double>(n), cudaMemcpyDeviceToDevice, queue.getNative()), "ECP final D2D copy");
#endif
    int device = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device), "ECP final copy device");
#else
    checkRuntime(cudaGetDevice(&device), "ECP final copy device");
#endif
    deviceMemoryMetrics(device).copied(checkedDeviceBytes<double>(n));
  }
  std::shared_ptr<const PreparedDeviceModel> image_;
  std::shared_ptr<const std::size_t> source_;
  const std::size_t samples_, electrons_;
  const SpatialJetLayout layout_;
  std::shared_ptr<coordinate_detail::SourceState> coordinates_;
  DeviceBuffer<double> roots_, masses_, kinetic_, energy_;
  DeviceBuffer<std::uint64_t> status_;
  std::vector<double> host_positions_, host_masses_, host_kinetic_, host_energy_;
  std::vector<std::uint64_t> host_status_;
  std::vector<std::size_t> generations_;
  std::string physical_source_;
  std::uint64_t grid_ = 0;
  std::size_t filled_ = 0;
  bool captured_ = false, sealed_ = false;
};
} // namespace qmcplusplus::psiformer::device
#endif
