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
    PsiFormerDeviceDerivativeKernels.cu
    PsiFormerDeviceDeterminantKernels.cu
    PsiFormerDevicePrecision.cu
    PsiFormerDeviceSpatial.cu
    PsiFormerDeviceSpatialReverse.cu
    PsiFormerDeviceOpenKineticReverse.cu)

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

# Opt-in actual-device checks, deliberately not registered with offline CTest.
# --host-check is safe on driverless hosts; ordinary execution requires hardware.
add_executable(psiformer_device_runtime_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceRuntimeValidation.cu)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceRuntimeValidation.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_runtime_validation PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_runtime_validation PRIVATE psiformer_device)

# Whole-model validation imports portable HDF5 data but remains independent of
# the production QMC application and its intentionally disabled device policy.
add_executable(psiformer_device_model_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceModelValidation.cu
               ../../PsiFormer/PsiFormerAcceleratorExecutor.cpp
               ../../PsiFormer/PsiFormerAcceleratorPlanning.cpp
               ../../PsiFormer/PsiFormerExecutionPlan.cpp
               ../../PsiFormer/PsiFormerGeometry.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceModelValidation.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_model_validation PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_model_validation PRIVATE psiformer_device IO::HDF5 platform_LA)

# Opt-in warm timing/copy evidence; deliberately not a hardware-independent CTest.
add_executable(psiformer_device_benchmark EXCLUDE_FROM_ALL
               PsiFormerDeviceBenchmark.cu
               ../../PsiFormer/PsiFormerExecutionPlan.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceBenchmark.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_benchmark PRIVATE psiformer_device IO::HDF5 platform_LA)

# Correctness oracles must retain IEEE NaN/infinity semantics even when the
# application's Release flags use fast-math. Library kernel policy is unchanged.
foreach(validation_target psiformer_device_runtime_validation psiformer_device_model_validation psiformer_device_benchmark)
  target_compile_options(${validation_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math>)
endforeach()
