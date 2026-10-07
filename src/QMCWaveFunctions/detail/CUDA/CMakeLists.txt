#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2022 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

set(WFSSRCS_CUDA cuBLAS_LU.cu delayed_update_helper.cu matrix_update_helper.cu)

set(PSIFORMER_DEVICE_SRCS
    PsiFormerDeviceKernels.cu
    PsiFormerDeviceDense.cu
    PsiFormerDeviceDeterminantKernels.cu
    PsiFormerDevicePrecision.cu
    PsiFormerDeviceSpatial.cu)

add_library(qmcwfs_cuda ${WFSSRCS_CUDA})
if(QMC_CUDA2HIP)
  set_source_files_properties(${WFSSRCS_CUDA} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(qmcwfs_cuda PRIVATE platform_LA)

# The PsiFormer algebra lives in one host/device header and one device translation
# unit.  CMake changes only the source language, so CUDA and HIP cannot drift into
# separate mathematical implementations.
add_library(psiformer_device STATIC ${PSIFORMER_DEVICE_SRCS})
if(QMC_CUDA2HIP)
  set_source_files_properties(${PSIFORMER_DEVICE_SRCS} PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_include_directories(psiformer_device PUBLIC
                           "${qmcpack_SOURCE_DIR}/src"
                           "${qmcpack_BINARY_DIR}/src")
target_link_libraries(psiformer_device PUBLIC platform_LA)

# Explicit developer target: it is not a runtime CTest and may be built safely on
# driverless development hosts to force the backend device-link step.
add_executable(psiformer_device_link_check EXCLUDE_FROM_ALL PsiFormerDeviceLinkCheck.cu)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceLinkCheck.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_link_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_link_check PRIVATE psiformer_device)
