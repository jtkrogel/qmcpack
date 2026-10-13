# These fixtures initialize both walkers and PsiFormer parameters internally.
# Only the small repository BFD pseudopotential tables are linked into each test
# directory; no orbital or trained-parameter state is required.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION)
  set(SEARCH_TEST short-LiH-pp-psiformer-search-reference-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(${SEARCH_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain"
              1 1 TEST_ADDED wftrain_reference.xml)
  if(TEST_ADDED)
    foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
      maybe_symlink("${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}"
                    "${CMAKE_CURRENT_BINARY_DIR}/${SEARCH_TEST}/${PP_FILE}")
    endforeach()
    set_property(TEST ${SEARCH_TEST} PROPERTY PASS_REGULAR_EXPRESSION "selected_alpha")
  endif()
  foreach(INPUT_NAME IN ITEMS wftrain_adam.xml wftrain_pii.xml)
    if(INPUT_NAME STREQUAL "wftrain_adam.xml")
      set(TEST_NAME short-LiH-pp-psiformer-wftrain-adam-r1-t1)
      set(EXPECTED_PATTERN
          "WFTrain iteration 1 version 1 samples 1 mean_energy.*optimizer adam")
    else()
      set(TEST_NAME short-LiH-pp-psiformer-wftrain-pii-r1-t1)
      set(EXPECTED_PATTERN
          "WFTrain iteration 1 version 2 samples 2 mean_energy.*optimizer minpii.*krylov_operator_applications [1-9]")
    endif()

    set(TEST_ADDED FALSE)
    run_qmc_app(
      ${TEST_NAME}
      "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain"
      1
      1
      TEST_ADDED
      ${INPUT_NAME})
    if(TEST_ADDED)
      foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
        maybe_symlink(
          "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}"
          "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}/${PP_FILE}")
      endforeach()
      set_property(TEST ${TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                                             "${EXPECTED_PATTERN}")
    endif()
  endforeach()

  set(TEST_NAME short-LiH-pp-psiformer-wftrain-lm-alm-kfac-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(
    ${TEST_NAME}
    "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain"
    1
    1
    TEST_ADDED
    wftrain_lm_alm_kfac.xml)
  if(TEST_ADDED)
    foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
      maybe_symlink(
        "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}/${PP_FILE}")
    endforeach()
    set_property(TEST ${TEST_NAME} PROPERTY PASS_REGULAR_EXPRESSION
                                           "WFTrain iteration 1 version 3 samples 2 mean_energy.*optimizer kfac")
  endif()
endif()
