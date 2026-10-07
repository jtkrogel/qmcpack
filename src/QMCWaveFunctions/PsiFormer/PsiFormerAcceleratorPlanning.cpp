//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerAcceleratorPlanning.cpp
 * @brief Pure accelerator planning and publication validation for PsiFormer.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAcceleratorPlanning.h"

#include <config.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace qmcplusplus::psiformer
{
namespace
{

/// Extend a stable FNV-1a fingerprint with one trivially represented value.
template<class T>
void extendFingerprint(std::uint64_t& fingerprint, const T& value) noexcept
{
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t byte = 0; byte < sizeof(T); ++byte)
  {
    fingerprint ^= bytes[byte];
    fingerprint *= UINT64_C(1099511628211);
  }
}

/// Normalize one short XML/configuration token without locale-dependent rules.
std::string normalizedToken(std::string_view input)
{
  std::string result(input);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

/// Reject an unavailable strict request with a stable diagnostic.
[[noreturn]] void throwUnavailable(PsiFormerAcceleratorBackend backend)
{
  throw std::runtime_error(std::string("PsiFormer accelerator backend '") +
                           psiFormerAcceleratorBackendName(backend) +
                           "' is not compiled into this executable");
}

} // namespace

bool PsiFormerCompiledAcceleratorSupport::supports(PsiFormerAcceleratorBackend backend) const noexcept
{
  switch (backend)
  {
  case PsiFormerAcceleratorBackend::CPU:
    return true;
  case PsiFormerAcceleratorBackend::OPENMP_TARGET:
    return openmp_target;
  case PsiFormerAcceleratorBackend::CUDA:
    return cuda;
  case PsiFormerAcceleratorBackend::HIP:
    return hip;
  case PsiFormerAcceleratorBackend::SYCL:
    return sycl;
  }
  return false;
}

void PsiFormerDevicePublicationState::beginPublication(std::size_t source_version)
{
  if (pending_version_)
    throw std::logic_error("PsiFormer device parameter publication is already pending");
  if (source_version <= active_version_)
    throw std::invalid_argument("PsiFormer device publication requires a strictly newer model version");
  pending_version_ = source_version;
}

void PsiFormerDevicePublicationState::completePublication(std::size_t source_version)
{
  if (!pending_version_ || *pending_version_ != source_version)
    throw std::logic_error("PsiFormer device publication completion does not match the pending version");
  active_version_ = source_version;
  pending_version_.reset();
}

void PsiFormerDevicePublicationState::cancelPublication(std::size_t source_version)
{
  if (!pending_version_ || *pending_version_ != source_version)
    throw std::logic_error("PsiFormer device publication cancellation does not match the pending version");
  pending_version_.reset();
}

PsiFormerCompiledAcceleratorSupport compiledPsiFormerAcceleratorSupport() noexcept
{
  PsiFormerCompiledAcceleratorSupport support;
#if defined(ENABLE_OFFLOAD)
  support.openmp_target = true;
#endif
#if defined(ENABLE_CUDA) && defined(QMC_CUDA2HIP)
  support.hip = true;
#elif defined(ENABLE_CUDA)
  support.cuda = true;
#endif
#if defined(ENABLE_SYCL)
  support.sycl = true;
#endif
  return support;
}

const char* psiFormerAcceleratorBackendName(PsiFormerAcceleratorBackend backend) noexcept
{
  switch (backend)
  {
  case PsiFormerAcceleratorBackend::CPU:
    return "cpu";
  case PsiFormerAcceleratorBackend::OPENMP_TARGET:
    return "omptarget";
  case PsiFormerAcceleratorBackend::CUDA:
    return "cuda";
  case PsiFormerAcceleratorBackend::HIP:
    return "hip";
  case PsiFormerAcceleratorBackend::SYCL:
    return "sycl";
  }
  return "unknown";
}

PsiFormerAcceleratorBackend selectPsiFormerAcceleratorBackend(
    std::string_view request,
    PsiFormerCompiledAcceleratorSupport support)
{
  const std::string value = normalizedToken(request);
  if (value.empty() || value == "no" || value == "cpu")
    return PsiFormerAcceleratorBackend::CPU;

  const auto preferred_backend = [&support]() {
    if (support.cuda)
      return PsiFormerAcceleratorBackend::CUDA;
    if (support.hip)
      return PsiFormerAcceleratorBackend::HIP;
    if (support.sycl)
      return PsiFormerAcceleratorBackend::SYCL;
    if (support.openmp_target)
      return PsiFormerAcceleratorBackend::OPENMP_TARGET;
    return PsiFormerAcceleratorBackend::CPU;
  };

  if (value == "auto")
    return preferred_backend();
  if (value == "yes")
  {
    const PsiFormerAcceleratorBackend selected = preferred_backend();
    if (selected == PsiFormerAcceleratorBackend::CPU)
      throw std::runtime_error("PsiFormer gpu=yes requires a compiled accelerator backend");
    return selected;
  }

  PsiFormerAcceleratorBackend selected;
  if (value == "omptarget")
    selected = PsiFormerAcceleratorBackend::OPENMP_TARGET;
  else if (value == "cuda")
    selected = PsiFormerAcceleratorBackend::CUDA;
  else if (value == "hip")
    selected = PsiFormerAcceleratorBackend::HIP;
  else if (value == "sycl")
    selected = PsiFormerAcceleratorBackend::SYCL;
  else
    throw std::invalid_argument("PsiFormer gpu must be no, cpu, auto, yes, omptarget, cuda, hip, or sycl");

  if (!support.supports(selected))
    throwUnavailable(selected);
  return selected;
}

PsiFormerDeviceLayout makePsiFormerDeviceLayout(const PsiFormerExecutionPlan& plan)
{
  PsiFormerDeviceLayout layout;
  layout.model_shape             = plan.modelShape();
  layout.boundary                = plan.environment().boundary;
  layout.geometry_feature_policy = plan.environment().geometry_feature_policy;
  layout.parameter_scalar_domain = plan.environment().parameter_scalar_domain;
  layout.compute_scalar_domain   = plan.environment().compute_scalar_domain;
  layout.amplitude_scalar_domain = plan.environment().amplitude_scalar_domain;
  layout.parameter_count         = plan.parameterCount();
  layout.tensors.reserve(plan.parameterTensors().size());

  std::size_t expected_begin = 0;
  for (const ParameterTensorDescriptor& host : plan.parameterTensors())
  {
    if (host.shape.size() > 2)
      throw std::invalid_argument("PsiFormer accelerator layout supports only scalar, vector, and matrix tensors");
    if (host.begin != expected_begin || host.end < host.begin)
      throw std::logic_error("PsiFormer execution plan contains a noncanonical tensor interval");

    PsiFormerDeviceTensorDescriptor device;
    device.role            = host.role;
    device.attention_block = host.attention_block;
    device.rank            = static_cast<std::uint8_t>(host.shape.size());
    device.begin           = host.begin;
    device.end             = host.end;
    for (std::size_t axis = 0; axis < host.shape.size(); ++axis)
      device.extents[axis] = host.shape[axis];
    layout.tensors.push_back(device);
    expected_begin = host.end;
  }
  if (expected_begin != layout.parameter_count)
    throw std::logic_error("PsiFormer accelerator layout does not cover the canonical parameter vector");

  std::uint64_t fingerprint = UINT64_C(14695981039346656037);
  for (const std::size_t value : {layout.model_shape.spin_up_electrons,
                                  layout.model_shape.spin_down_electrons,
                                  layout.model_shape.nuclei,
                                  layout.model_shape.determinants,
                                  layout.model_shape.feature_dimension,
                                  layout.model_shape.attention_heads,
                                  layout.model_shape.attention_blocks,
                                  layout.parameter_count})
    extendFingerprint(fingerprint, value);
  extendFingerprint(fingerprint, layout.boundary);
  extendFingerprint(fingerprint, layout.geometry_feature_policy);
  extendFingerprint(fingerprint, layout.parameter_scalar_domain);
  extendFingerprint(fingerprint, layout.compute_scalar_domain);
  extendFingerprint(fingerprint, layout.amplitude_scalar_domain);
  for (const PsiFormerDeviceTensorDescriptor& tensor : layout.tensors)
  {
    extendFingerprint(fingerprint, tensor.role);
    extendFingerprint(fingerprint, tensor.attention_block);
    extendFingerprint(fingerprint, tensor.rank);
    extendFingerprint(fingerprint, tensor.extents[0]);
    extendFingerprint(fingerprint, tensor.extents[1]);
    extendFingerprint(fingerprint, tensor.begin);
    extendFingerprint(fingerprint, tensor.end);
  }
  layout.fingerprint = fingerprint;
  return layout;
}

} // namespace qmcplusplus::psiformer
