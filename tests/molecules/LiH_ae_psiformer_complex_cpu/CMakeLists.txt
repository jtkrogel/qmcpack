# CPU checks need no device; explicit paired GPU target checks use the existing resource launcher.
add_test(NAME deterministic-psiformer-independent-twist-workflow-unit
         COMMAND ${Python3_EXECUTABLE}
                 ${PROJECT_SOURCE_DIR}/tests/scripts/test_psiformer_twist_workflow.py)
set_tests_properties(deterministic-psiformer-independent-twist-workflow-unit PROPERTIES
                     TIMEOUT 30 LABELS "deterministic;psiformer;cpu;workflow")
if(QMC_COMPLEX AND NOT QMC_MIXED_PRECISION)
  foreach(mode estimators dmc)
    set(test_name deterministic-LiH-psiformer-cpu-modern-${mode}-restart)
    add_test(NAME ${test_name}
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_modern_integration.py
                     --binary $<TARGET_FILE:qmcpack>
                     --fixture ${CMAKE_CURRENT_SOURCE_DIR}/complex_twist.xml
                     --mode ${mode} --restart
                     --output ${CMAKE_CURRENT_BINARY_DIR}/modern_${mode}_restart)
    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1"
                         PROCESSORS 2 TIMEOUT 300
                         LABELS "deterministic;psiformer;cpu;periodic;complex;modern;restart")
  endforeach()
  add_test(NAME deterministic-LiH-psiformer-cpu-modern-dmc-tmove-rejection
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_modern_integration.py
                   --binary $<TARGET_FILE:qmcpack>
                   --fixture ${CMAKE_CURRENT_SOURCE_DIR}/complex_twist.xml
                   --mode dmc --expect-tmove-rejection
                   --output ${CMAKE_CURRENT_BINARY_DIR}/modern_dmc_tmove_rejection)
  set_tests_properties(deterministic-LiH-psiformer-cpu-modern-dmc-tmove-rejection PROPERTIES
                       ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                       TIMEOUT 120 LABELS "deterministic;psiformer;cpu;periodic;complex;modern")
  if(ENABLE_CUDA AND NOT QMC_CUDA2HIP)
    foreach(variant complex_gamma complex_twist)
      set(test_name deterministic-LiH-paired-${variant}-gpu-target-xml)
      add_test(NAME ${test_name}
               COMMAND ${QMC_GPU_TEST_LAUNCHER} ${Python3_EXECUTABLE}
                       ${CMAKE_CURRENT_SOURCE_DIR}/check_gpu_target_xml.py
                       --binary $<TARGET_FILE:qmcpack> --input ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.xml
                       --output ${CMAKE_CURRENT_BINARY_DIR}/gpu_target_${variant})
      SET_TEST_GPU_RESOURCES(${test_name})
      set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                           TIMEOUT 300 LABELS "deterministic;psiformer;gpu;periodic;complex")
    endforeach()
  endif()
  add_test(NAME deterministic-LiH-psiformer-cpu-twist-workflow-restart
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_twist_workflow_restart.py
                   --binary $<TARGET_FILE:qmcpack>
                   --output ${CMAKE_CURRENT_BINARY_DIR}/twist_workflow_restart)
  set_tests_properties(deterministic-LiH-psiformer-cpu-twist-workflow-restart PROPERTIES
                       ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                       TIMEOUT 180 LABELS "deterministic;psiformer;cpu;periodic;complex;workflow;wftrain")
  add_test(NAME deterministic-LiH-psiformer-cpu-independent-twists
           COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/utils/psiformer_twist_workflow.py
                   ${CMAKE_CURRENT_SOURCE_DIR}/independent_twists.json
                   --binary $<TARGET_FILE:qmcpack>
                   --output ${CMAKE_CURRENT_BINARY_DIR}/independent_twists)
  set_tests_properties(deterministic-LiH-psiformer-cpu-independent-twists PROPERTIES
                       ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                       TIMEOUT 300 LABELS "deterministic;psiformer;cpu;periodic;complex;workflow")
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
  foreach(optimizer adam sgd sr)
    set(variant complex_gamma)
    if(NOT optimizer STREQUAL "adam")
      set(variant complex_twist)
    endif()
    set(test_name deterministic-LiH-psiformer-cpu-${optimizer}-training)
    add_test(NAME ${test_name}
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cpu_training.py
                     --binary $<TARGET_FILE:qmcpack> --optimizer ${optimizer}
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.xml
                     --output ${CMAKE_CURRENT_BINARY_DIR}/training_${optimizer})
    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                         TIMEOUT 1200 LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
  endforeach()
endif()
