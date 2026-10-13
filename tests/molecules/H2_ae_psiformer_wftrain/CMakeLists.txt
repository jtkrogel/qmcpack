# The fixture is external-data-free: both walkers and PsiFormer parameters are
# initialized by QMCPACK.  One block exercises sampling, a committed Adam update,
# and the mandatory post-publication cache refresh.  The iteration diagnostic is
# emitted only after refresh succeeds, so its presence is the application-level
# completion criterion.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION)
  foreach(SEARCH_CASE reference endpoints overlap_search rejected)
    foreach(SEARCH_RANKS 1 2)
      set(SEARCH_TEST short-H2-psiformer-search-${SEARCH_CASE}-r${SEARCH_RANKS}-t1)
      set(TEST_ADDED FALSE)
      run_qmc_app(${SEARCH_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_psiformer_wftrain"
                  ${SEARCH_RANKS} 1 TEST_ADDED wftrain_${SEARCH_CASE}.xml)
      if(TEST_ADDED)
        if(SEARCH_CASE STREQUAL "rejected")
          set_property(TEST ${SEARCH_TEST} PROPERTY PASS_REGULAR_EXPRESSION "WFTrain iteration 2 version 0")
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
               "WFTrain iteration 1 version 1 samples 1 mean_energy")
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
               "WFTrain iteration 1 version 1 samples 2 overlap_loss")
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
               "WFTrain iteration 1 version 2 samples 2 mean_energy.*optimizer minpii.*krylov_operator_applications [1-9]")
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
               "WFTrain iteration 1 version 3 samples 2 mean_energy.*optimizer spring.*krylov_operator_applications [1-9]")
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
                 "WFTrain iteration 1 version 5 samples 2 mean_energy")
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
                   "WFTrain iteration 1 version 1 samples 2 mean_energy.*optimizer ${MATRIX_OPTIMIZER}")
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
                 "WFTrain checkpointed 1 completed update\\(s\\)")
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
                 "WFTrain iteration 2 version 2")
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
               "WFTrain iteration 1 version 1 samples 1 mean_energy [-+]?[0-9]")
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
               "WFTrain checkpointed 1 completed update\\(s\\)")
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
               "WFTrain iteration 2 version 2")
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
                 "WFTrain iteration 1 version 1 samples 2 mean_energy")
    endif()
  endif()
endif()

get_property(CURRENT_TESTS DIRECTORY PROPERTY TESTS)
add_labels_for_tests("${CURRENT_TESTS}")
