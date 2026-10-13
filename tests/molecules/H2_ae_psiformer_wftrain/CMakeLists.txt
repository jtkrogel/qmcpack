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
