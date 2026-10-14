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
endif()
