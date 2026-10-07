//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceLinkCheck.cu
 * @brief Driverless developer target proving that PsiFormer wrappers device-link.
 */

#include "PsiFormerDeviceKernels.h"

int main()
{
  using namespace qmcplusplus::psiformer::device;
  // Zero work performs no runtime call but keeps the wrapper translation unit in the
  // executable, forcing CUDA nvlink or the HIP device-link step to inspect it.
  return launchResidualJets(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, 0, nullptr, nullptr, nullptr) == success
      ? 0
      : 1;
}
