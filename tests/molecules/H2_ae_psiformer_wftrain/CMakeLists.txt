# The fixture is external-data-free: both walkers and PsiFormer parameters are
# initialized by QMCPACK.  One block exercises sampling, a committed Adam update,
# and the mandatory post-publication cache refresh.  The iteration diagnostic is
# emitted only after refresh succeeds, so its presence is the application-level
# completion criterion.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION)
  if(Python3_EXECUTABLE)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                    RESULT_VARIABLE PSIFORMER_INFERENCE_PYTHON_MODULES
                    OUTPUT_QUIET ERROR_QUIET)
    if(PSIFORMER_INFERENCE_PYTHON_MODULES EQUAL 0)
      add_test(NAME short-H2-ae-psiformer-wftrain-interval-inference
               COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_interval_inference.py"
                       $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_checkpoint.xml")
      set_tests_properties(short-H2-ae-psiformer-wftrain-interval-inference
                           PROPERTIES TIMEOUT 180 PROCESSORS 1)
    endif()
  endif()
  # Native neural parameters and inline conventional SPOs require no external data.
  set(ORBITAL_TEST short-H2-ae-psiformer-orbital-mse-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(${ORBITAL_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
              1 1 TEST_ADDED wftrain_orbital_mse.xml)
  if(TEST_ADDED)
    set_property(TEST ${ORBITAL_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain iteration 1 version 2 samples 2 orbital_loss")
  endif()

  foreach(POOLED_CASE optimizers search)
    set(TEST_ADDED FALSE)
    set(POOLED_TEST short-H2-psiformer-wftrain-pooled-${POOLED_CASE}-r1-t1)
    run_qmc_app(${POOLED_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
                1 1 TEST_ADDED wftrain_pooled_${POOLED_CASE}.xml)
    if(TEST_ADDED)
      if(POOLED_CASE STREQUAL "optimizers")
        set_property(TEST ${POOLED_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                     "samples 8 mean_energy.*optimizer kfac")
      else()
        set_property(TEST ${POOLED_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                     "WFTrain iteration 1.*samples 8 mean_energy")
      endif()
    endif()
  endforeach()
  # Independent iteration and sample counts: 2 walkers/rank * 2 blocks * 2 steps.
  foreach(POOLED_RANKS 1 2)
    set(POOLED_TEST short-H2-psiformer-wftrain-pooled-r${POOLED_RANKS}-t1)
    set(TEST_ADDED FALSE)
    run_qmc_app(${POOLED_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
                ${POOLED_RANKS} 1 TEST_ADDED wftrain_pooled.xml)
    if(TEST_ADDED)
      math(EXPR POOLED_SAMPLES "8 * ${POOLED_RANKS}")
      set_property(TEST ${POOLED_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                   "WFTrain iteration 1 version 2 samples ${POOLED_SAMPLES} mean_energy")
    endif()
  endforeach()
  foreach(SEARCH_CASE reference endpoints overlap_search rejected)
    foreach(SEARCH_RANKS 1 2)
      set(SEARCH_TEST short-H2-psiformer-search-${SEARCH_CASE}-r${SEARCH_RANKS}-t1)
      set(TEST_ADDED FALSE)
      run_qmc_app(${SEARCH_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
                  ${SEARCH_RANKS} 1 TEST_ADDED wftrain_${SEARCH_CASE}.xml)
      if(TEST_ADDED)
        if(SEARCH_CASE STREQUAL "rejected")
          set_property(TEST ${SEARCH_TEST} PROPERTY PASS_REGULAR_EXPRESSION "WFTrain iteration 1 version 0")
        else()
          set_property(TEST ${SEARCH_TEST} PROPERTY PASS_REGULAR_EXPRESSION "selected_alpha")
        endif()
      endif()
    endforeach()
  endforeach()
  set(TEST_NAME short-H2-ae-psiformer-wftrain-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 0 version 1 samples 1 mean_energy")
  endif()

  set(OVERLAP_TEST_NAME short-H2-ae-psiformer-wftrain-overlap-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${OVERLAP_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_overlap.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${OVERLAP_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 0 version 1 samples 2 overlap_loss")
  endif()

  set(PII_TEST_NAME short-H2-ae-psiformer-wftrain-pii-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${PII_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_pii.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${PII_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 0 version 2 samples 2 mean_energy.*optimizer minpii.*krylov_operator_applications [1-9]")
  endif()

  set(OPTIMIZERS_TEST_NAME short-H2-ae-psiformer-wftrain-optimizers-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${OPTIMIZERS_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_optimizers.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${OPTIMIZERS_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 0 version 3 samples 2 mean_energy.*optimizer spring.*krylov_operator_applications [1-9]")
  endif()

  set(FIRST_ORDER_TEST_NAME short-H2-ae-psiformer-wftrain-first-order-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${FIRST_ORDER_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_first_order.xml)
  if(TEST_ADDED)
    set_property(TEST ${FIRST_ORDER_TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain iteration 0 version 5 samples 2 mean_energy")
  endif()

  foreach(MATRIX_OPTIMIZER lm alm kfac)
    set(MATRIX_OPTIMIZER_TEST_NAME short-H2-ae-psiformer-wftrain-${MATRIX_OPTIMIZER}-r1-t1)
    set(TEST_ADDED FALSE)
    run_qmc_app(
      ${MATRIX_OPTIMIZER_TEST_NAME}
      "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
      1
      1
      TEST_ADDED
      wftrain_${MATRIX_OPTIMIZER}.xml)
    if(TEST_ADDED)
      set_property(TEST ${MATRIX_OPTIMIZER_TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                   "WFTrain iteration 0 version 1 samples 2 mean_energy.*optimizer ${MATRIX_OPTIMIZER}")
    endif()
  endforeach()

  set(AMSGRAD_CHECKPOINT_TEST_NAME short-H2-ae-psiformer-wftrain-amsgrad-checkpoint-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${AMSGRAD_CHECKPOINT_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_amsgrad_checkpoint.xml)
  if(TEST_ADDED)
    set_property(TEST ${AMSGRAD_CHECKPOINT_TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain checkpointed 1 completed iteration\\(s\\)")
  endif()

  set(IGNORED_ESTIMATORS_TEST short-H2-ae-psiformer-wftrain-ignored-estimators-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(${IGNORED_ESTIMATORS_TEST}
              "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
              1 1 TEST_ADDED wftrain_ignored_estimators.xml)
  if(TEST_ADDED)
    set_property(TEST ${IGNORED_ESTIMATORS_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain ignores supplemental global/local estimator requests")
    find_program(PSIFORMER_H5DIFF h5diff)
    if(PSIFORMER_H5DIFF AND Python3_EXECUTABLE)
      add_test(NAME short-H2-ae-psiformer-wftrain-ignored-estimators-equivalence
               COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_ignored_estimators.py"
                 "${CMAKE_CURRENT_BINARY_DIR}/${AMSGRAD_CHECKPOINT_TEST_NAME}/h2_psiformer_amsgrad_checkpoint.s000"
                 "${CMAKE_CURRENT_BINARY_DIR}/${IGNORED_ESTIMATORS_TEST}/h2_psiformer_ignored_estimators.s000"
                 --h5diff ${PSIFORMER_H5DIFF})
      set_property(TEST short-H2-ae-psiformer-wftrain-ignored-estimators-equivalence PROPERTY DEPENDS
                   "${AMSGRAD_CHECKPOINT_TEST_NAME};${IGNORED_ESTIMATORS_TEST}")
    endif()
  endif()

  set(AMSGRAD_RESTART_TEST_NAME short-H2-ae-psiformer-wftrain-amsgrad-restart-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${AMSGRAD_RESTART_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_amsgrad_restart.xml)
  if(TEST_ADDED)
    maybe_symlink(
      "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain/wftrain_amsgrad_restart.xml"
      "${CMAKE_CURRENT_BINARY_DIR}/${AMSGRAD_CHECKPOINT_TEST_NAME}/wftrain_amsgrad_restart.xml")
    set_property(TEST ${AMSGRAD_RESTART_TEST_NAME} PROPERTY DEPENDS ${AMSGRAD_CHECKPOINT_TEST_NAME})
    set_property(
      TEST ${AMSGRAD_RESTART_TEST_NAME}
      PROPERTY WORKING_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/${AMSGRAD_CHECKPOINT_TEST_NAME}")
    set_property(TEST ${AMSGRAD_RESTART_TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain iteration 1 version 2")
  endif()

  set(LIH_AE_TEST_NAME short-LiH-ae-psiformer-wftrain-adam-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${LIH_AE_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_lih_ae_adam.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${LIH_AE_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 0 version 1 samples 1 mean_energy [-+]?[0-9]")
  endif()

  # Exercise the production manifest/root/shard route in two processes launched
  # sequentially by CTest. The second process reconstructs its native population,
  # restores the exact post-update Markov state, and performs only update two.
  set(CHECKPOINT_TEST_NAME short-H2-ae-psiformer-wftrain-checkpoint-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${CHECKPOINT_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_checkpoint.xml)
  if(TEST_ADDED)
    set_property(
      TEST ${CHECKPOINT_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain checkpointed 1 completed iteration\\(s\\)")
  endif()

  set(RESTART_TEST_NAME short-H2-ae-psiformer-wftrain-restart-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${RESTART_TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_restart.xml)
  if(TEST_ADDED)
    maybe_symlink(
      "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain/wftrain_restart.xml"
      "${CMAKE_CURRENT_BINARY_DIR}/${CHECKPOINT_TEST_NAME}/wftrain_restart.xml")
    set_property(TEST ${RESTART_TEST_NAME} PROPERTY DEPENDS ${CHECKPOINT_TEST_NAME})
    set_property(
      TEST ${RESTART_TEST_NAME}
      PROPERTY WORKING_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/${CHECKPOINT_TEST_NAME}")
    set_property(
      TEST ${RESTART_TEST_NAME}
      PROPERTY PASS_REGULAR_EXPRESSION
               "WFTrain iteration 1 version 2")
  endif()

  if(HAVE_MPI AND TEST_MAX_PROCS GREATER_EQUAL 2)
    set(TEST_NAME short-H2-ae-psiformer-wftrain-r2-t1)
    set(TEST_ADDED FALSE)
    run_qmc_app(
      ${TEST_NAME}
      "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
      2
      1
      TEST_ADDED
      wftrain.xml)
    if(TEST_ADDED)
      set_property(
        TEST ${TEST_NAME}
        PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain iteration 0 version 1 samples 2 mean_energy")
    endif()
  endif()
endif()

if(QMC_COMPLEX AND NOT QMC_MIXED_PRECISION AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE COMPLEX_ORBITAL_MSE_PYTHON_MODULES
                  OUTPUT_QUIET ERROR_QUIET)
  if(COMPLEX_ORBITAL_MSE_PYTHON_MODULES EQUAL 0)
    add_test(NAME deterministic-H2-psiformer-complex-cpu-kfac-restart
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_complex_kfac.py"
                     --qmcpack $<TARGET_FILE:qmcpack>)
    set_tests_properties(deterministic-H2-psiformer-complex-cpu-kfac-restart PROPERTIES
                         TIMEOUT 240 PROCESSORS 1
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
                         LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
    add_test(NAME deterministic-H2-psiformer-complex-cpu-kfac-clipped-restart
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_complex_kfac.py"
                     --qmcpack $<TARGET_FILE:qmcpack> --clipped)
    set_tests_properties(deterministic-H2-psiformer-complex-cpu-kfac-clipped-restart PROPERTIES
                         TIMEOUT 240 PROCESSORS 1
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
                         LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
    set(PRESET_ORDINAL 0)
    foreach(PRESET_LABEL IN ITEMS deepqmc deepsolid jaqmc)
      math(EXPR PRESET_ORDINAL "${PRESET_ORDINAL}+1")
      set(PRESET_TEST deterministic-H2-psiformer-complex-cpu-kfac-preset-${PRESET_ORDINAL}-restart)
      add_test(NAME ${PRESET_TEST}
               COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_complex_kfac.py"
                       --qmcpack $<TARGET_FILE:qmcpack> --defaults ${PRESET_LABEL})
      set_tests_properties(${PRESET_TEST} PROPERTIES TIMEOUT 300 PROCESSORS 1
                           ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
                           LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
    endforeach()
    add_test(NAME deterministic-H2-psiformer-complex-cpu-orbital-mse-restart
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_complex_orbital_mse.py"
                     --qmcpack $<TARGET_FILE:qmcpack>)
    set_tests_properties(deterministic-H2-psiformer-complex-cpu-orbital-mse-restart PROPERTIES
                         TIMEOUT 150 PROCESSORS 1
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
                         LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
    add_test(NAME deterministic-H2-psiformer-complex-cpu-reference-restart
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_complex_orbital_mse.py"
                     --qmcpack $<TARGET_FILE:qmcpack> --energy-reference-restart)
    set_tests_properties(deterministic-H2-psiformer-complex-cpu-reference-restart PROPERTIES
                         TIMEOUT 150 PROCESSORS 1
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
                         LABELS "deterministic;psiformer;cpu;periodic;complex;wftrain")
  endif()
endif()

get_property(CURRENT_TESTS DIRECTORY PROPERTY TESTS)
add_labels_for_tests("${CURRENT_TESTS}")
