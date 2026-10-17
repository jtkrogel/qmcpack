# Run the same native-initialized XML with CPU and strict CUDA evaluation.
# No trained parameter, orbital or externally initialized walker files are used.
if(ENABLE_CUDA AND NOT QMC_CUDA2HIP AND NOT QMC_COMPLEX)
  add_test(NAME deterministic-LiH-ae-psiformer-cuda-xml
           COMMAND ${QMC_GPU_TEST_LAUNCHER} ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compare_xml.py
                   --binary $<TARGET_FILE:qmcpack>
                   --input ${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                   --output ${CMAKE_CURRENT_BINARY_DIR}/comparison)
  SET_TEST_GPU_RESOURCES(deterministic-LiH-ae-psiformer-cuda-xml)
  set_tests_properties(deterministic-LiH-ae-psiformer-cuda-xml
                       PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                                  TIMEOUT 300 LABELS "deterministic;psiformer;gpu")
  add_test(NAME deterministic-LiH-ae-psiformer-mixed-cuda-xml
           COMMAND ${QMC_GPU_TEST_LAUNCHER} ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compare_xml.py
                   --binary $<TARGET_FILE:qmcpack> --mixed
                   --input ${CMAKE_CURRENT_SOURCE_DIR}/input_mixed.xml
                   --output ${CMAKE_CURRENT_BINARY_DIR}/comparison_mixed)
  SET_TEST_GPU_RESOURCES(deterministic-LiH-ae-psiformer-mixed-cuda-xml)
  set_tests_properties(deterministic-LiH-ae-psiformer-mixed-cuda-xml PROPERTIES
                       ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                       TIMEOUT 300 LABELS "deterministic;psiformer;gpu;precision")
endif()

# Optional generated numerical fixture: no external trained/scientific asset is bundled.
if(ENABLE_CUDA AND NOT QMC_CUDA2HIP AND NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND Python3_EXECUTABLE
   AND DEFINED ENV{PSIFORMER_TEST_RESIDENT_MODEL_EXPORT} AND EXISTS "$ENV{PSIFORMER_TEST_RESIDENT_MODEL_EXPORT}")
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE DEVICE_SR_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(DEVICE_SR_PYTHON_MODULES EQUAL 0)
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${qmcpack_SOURCE_DIR}
                    OUTPUT_VARIABLE DEVICE_SR_SOURCE_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT DEVICE_SR_SOURCE_COMMIT)
      set(DEVICE_SR_SOURCE_COMMIT unavailable)
    endif()
    add_test(NAME deterministic-LiH-ae-psiformer-device-sr-restart
             COMMAND ${QMC_GPU_TEST_LAUNCHER} ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/check_resident_device_sr_native.py
                     --binary $<TARGET_FILE:qmcpack> --source ${qmcpack_SOURCE_DIR}
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                     --model "$ENV{PSIFORMER_TEST_RESIDENT_MODEL_EXPORT}"
                     --output-parent ${CMAKE_CURRENT_BINARY_DIR}/device_sr_native
                     --source-commit ${DEVICE_SR_SOURCE_COMMIT} --full-workflow)
    SET_TEST_GPU_RESOURCES(deterministic-LiH-ae-psiformer-device-sr-restart)
    set_tests_properties(deterministic-LiH-ae-psiformer-device-sr-restart PROPERTIES
                         TIMEOUT 200 PROCESSORS 1 LABELS "deterministic;psiformer;gpu;sr;restart")
  endif()
endif()

# Gamma periodic runs in real or complex builds; a complex build additionally
# checks the fixed outer Bloch phase through the identical production XML path.
if(ENABLE_CUDA AND NOT QMC_CUDA2HIP)
  set(periodic_variants gamma)
  if(QMC_COMPLEX)
    list(APPEND periodic_variants twist)
  endif()
  foreach(variant IN LISTS periodic_variants)
    set(twist_args)
    if(variant STREQUAL "twist")
      set(twist_args --twist "0.17 -0.11 0.07")
    endif()
    set(test_name deterministic-LiH-periodic-${variant}-psiformer-cuda-xml)
    add_test(NAME ${test_name}
             COMMAND ${QMC_GPU_TEST_LAUNCHER} ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/compare_xml.py
                     --binary $<TARGET_FILE:qmcpack>
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/input_periodic.xml
                     --output ${CMAKE_CURRENT_BINARY_DIR}/comparison_${variant} ${twist_args})
    SET_TEST_GPU_RESOURCES(${test_name})
    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                         TIMEOUT 300 LABELS "deterministic;psiformer;gpu;periodic")
  endforeach()
endif()
