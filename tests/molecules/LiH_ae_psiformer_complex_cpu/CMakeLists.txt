# CPU-only profile qualification: no CUDA/HIP launcher, runtime device or trained assets.
if(QMC_COMPLEX AND NOT QMC_MIXED_PRECISION)
  foreach(variant real_gamma complex_gamma complex_twist ambiguous_h other_wavefunction_h)
    set(test_name deterministic-LiH-psiformer-cpu-${variant}-xml)
    set(rejection_args)
    if(variant STREQUAL "ambiguous_h" OR variant STREQUAL "other_wavefunction_h")
      set(rejection_args --expect-selected-system-rejection)
    endif()
    add_test(NAME ${test_name}
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cpu_xml.py
                     --binary $<TARGET_FILE:qmcpack>
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.xml
                     --output ${CMAKE_CURRENT_BINARY_DIR}/${variant} ${rejection_args})
    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                         TIMEOUT 300 LABELS "deterministic;psiformer;cpu;periodic;complex")
  endforeach()
endif()
