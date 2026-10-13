# The fixture is external-data-free: both walkers and PsiFormer parameters are
# initialized by QMCPACK.  One block exercises sampling, a committed Adam update,
# and the mandatory post-publication cache refresh.  The iteration diagnostic is
# emitted only after refresh succeeds, so its presence is the application-level
# completion criterion.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION)
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
               "WFTrain iteration 1 version 1 samples 1 overlap_loss")
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
