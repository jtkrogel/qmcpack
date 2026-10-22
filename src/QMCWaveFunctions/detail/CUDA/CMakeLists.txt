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
    PsiFormerDeviceCoordinateGradient.cu
    PsiFormerDeviceMetropolisKernels.cu
    PsiFormerDeviceAdamKernels.cu
    PsiFormerDeviceComplexValueKernels.cu
    PsiFormerDeviceComplexSpatialKernels.cu
    PsiFormerDeviceComplexScoreKernels.cu
    PsiFormerDeviceComplexKineticKernels.cu
    PsiFormerDeviceDeterminantKernels.cu
    PsiFormerDeviceLiftedDeterminantProducts.cu
    PsiFormerDeviceSpatialTraces.cu
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
# Compensation and injected proposal parity require rounded products/additions,
# not reassociation or fused coordinate updates.
# Keep this leaf strict even when a parent Release policy enables fast math.
set_property(SOURCE PsiFormerDeviceSpatialTraces.cu PsiFormerDeviceDeterminantKernels.cu
    PsiFormerDeviceLiftedDeterminantProducts.cu
    PsiFormerDeviceMetropolisKernels.cu PsiFormerDeviceAdamKernels.cu
    PsiFormerDeviceComplexValueKernels.cu PsiFormerDeviceComplexSpatialKernels.cu
    PsiFormerDeviceComplexScoreKernels.cu PsiFormerDeviceComplexKineticKernels.cu APPEND PROPERTY COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math>)
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
add_executable(psiformer_device_link_check EXCLUDE_FROM_ALL PsiFormerDeviceLinkCheck.cu
               ../../PsiFormer/PsiFormerExecutionPlan.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceLinkCheck.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_link_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_link_check PRIVATE psiformer_device)

# Opt-in actual-device checks, deliberately not registered with offline CTest.
# --host-check is safe on driverless hosts; ordinary execution requires hardware.
add_executable(psiformer_device_runtime_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceRuntimeValidation.cu
               # Dense mixed-affine helpers also reference canonical plan intervals.
               ../../PsiFormer/PsiFormerExecutionPlan.cpp)
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
               ../../PsiFormer/PsiFormerInitialization.cpp
               ../../../QMCDrivers/GreenFunctionModifiers/DriftModifierUNR.cpp
               ../../../Utilities/StdRandom.cpp
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
target_link_libraries(psiformer_device_model_validation PRIVATE psiformer_device IO::HDF5 platform_LA qmcio)

# Focused native crowd concurrency validation with generated parameters only.
add_executable(psiformer_device_concurrency_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceConcurrencyValidation.cpp
               ../../../QMCDrivers/GreenFunctionModifiers/DriftModifierUNR.cpp
               ../../../Utilities/StdRandom.cpp
               ../../PsiFormer/PsiFormerAcceleratorExecutor.cpp
               ../../PsiFormer/PsiFormerAcceleratorPlanning.cpp
               ../../PsiFormer/PsiFormerExecutionPlan.cpp
               ../../PsiFormer/PsiFormerInitialization.cpp
               ../../PsiFormer/PsiFormerGeometry.cpp)
target_link_libraries(psiformer_device_concurrency_validation PRIVATE psiformer_device IO::HDF5 platform_LA qmcio)
target_include_directories(psiformer_device_concurrency_validation PRIVATE
                           "${qmcpack_SOURCE_DIR}/src/Containers"
                           "${qmcpack_SOURCE_DIR}/src/QMCDrivers"
                           "${qmcpack_SOURCE_DIR}/src/Particle")
target_include_directories(psiformer_device_model_validation PRIVATE
                           "${qmcpack_SOURCE_DIR}/src/Containers"
                           "${qmcpack_SOURCE_DIR}/src/QMCDrivers"
                           "${qmcpack_SOURCE_DIR}/src/Particle")
if(NOT QMC_CUDA2HIP AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # Native RNG headers transitively include mpi3's annotated host/device lambda.
  target_compile_options(psiformer_device_model_validation PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

# Self-contained generated periodic fixture: no trained checkpoint data required.
add_executable(psiformer_device_periodic_validation EXCLUDE_FROM_ALL
               PsiFormerDevicePeriodicValidation.cu
               ../../PsiFormer/PsiFormerInitialization.cpp
               ../../PsiFormer/PsiFormerExecutionPlan.cpp
               ../../PsiFormer/PsiFormerGeometry.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDevicePeriodicValidation.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_periodic_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_periodic_validation PRIVATE psiformer_device IO::HDF5 platform_LA)

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

# Standalone bounded trace algebra/performance check, independent of full VGL.
add_executable(psiformer_device_spatial_trace_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceSpatialTraceValidation.cu
               # The linked CUDA library also owns canonical-plan dense helpers.
               ../../PsiFormer/PsiFormerExecutionPlan.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceSpatialTraceValidation.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_spatial_trace_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_spatial_trace_validation PRIVATE psiformer_device Boost::boost)

# One bounded numerical owner, independent of the complete kinetic provider.
add_executable(psiformer_device_lifted_products_validation EXCLUDE_FROM_ALL
               PsiFormerDeviceLiftedDeterminantProductValidation.cu
               ../../PsiFormer/PsiFormerExecutionPlan.cpp)
if(QMC_CUDA2HIP)
  set_source_files_properties(PsiFormerDeviceLiftedDeterminantProductValidation.cu PROPERTIES LANGUAGE HIP)
else()
  set_target_properties(psiformer_device_lifted_products_validation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(psiformer_device_lifted_products_validation PRIVATE psiformer_device)

# Plan metadata validation now checks the primitive/simulation-cell mapping.
# Keep standalone consumers linked to that host implementation as well.
foreach(validation_target psiformer_device_link_check psiformer_device_runtime_validation
    psiformer_device_benchmark psiformer_device_spatial_trace_validation
    psiformer_device_lifted_products_validation)
  target_sources(${validation_target} PRIVATE ../../PsiFormer/PsiFormerGeometry.cpp)
endforeach()

# Correctness oracles must retain IEEE NaN/infinity semantics even when the
# application's Release flags use fast-math. Library kernel policy is unchanged.
foreach(validation_target psiformer_device_runtime_validation psiformer_device_model_validation psiformer_device_periodic_validation psiformer_device_benchmark psiformer_device_spatial_trace_validation psiformer_device_lifted_products_validation psiformer_device_concurrency_validation)
  target_compile_options(${validation_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math>)
endforeach()
