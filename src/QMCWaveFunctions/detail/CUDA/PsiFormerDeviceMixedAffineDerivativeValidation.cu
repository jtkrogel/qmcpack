//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** Focused hardware oracle; compile/link is permitted on driverless hosts.
 * It is deliberately not an offline CTest: ordinary execution requires a GPU.
 */
#define PSIFORMER_LIBRARY
#include "PsiFormerDeviceMixedAffineDerivativeValidation.h"

/// Execute the generated oracle only when an actual GPU is visible.
int main()
{
  using namespace qmcplusplus::psiformer::device;
  try
  {
    int devices = 0;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDeviceCount(&devices), "mixed affine visible devices");
    const char* backend = "HIP";
#else
    checkRuntime(cudaGetDeviceCount(&devices), "mixed affine visible devices");
    const char* backend = "CUDA";
#endif
    if (devices == 0)
      throw std::runtime_error("Mixed affine numerical oracle requires a visible GPU");
    std::cout << "Actual-device mixed affine validation backend=" << backend
              << " visible_devices=" << devices << '\n';
    validateMixedAffineDerivatives();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL mixed affine validation: " << error.what() << '\n';
    return 1;
  }
}
