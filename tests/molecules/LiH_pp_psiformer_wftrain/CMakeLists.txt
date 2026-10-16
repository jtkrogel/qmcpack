# These fixtures initialize both walkers and PsiFormer parameters internally.
# Only the small repository BFD pseudopotential tables are linked into each test
# directory; no orbital or trained-parameter state is required.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION)
  foreach(ECP_CASE continuous checkpoint restart)
    set(TEST_ADDED FALSE)
    set(ECP_TEST short-LiH-pp-psiformer-ecp-${ECP_CASE}-r1-t1)
    run_qmc_app(${ECP_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain"
                1 1 TEST_ADDED wftrain_ecp_${ECP_CASE}.xml)
    if(TEST_ADDED)
      foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
        maybe_symlink("${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}"
                      "${CMAKE_CURRENT_BINARY_DIR}/${ECP_TEST}/${PP_FILE}")
      endforeach()
      set_property(TEST ${ECP_TEST} PROPERTY PASS_REGULAR_EXPRESSION "WFTrain checkpointed")
      if(ECP_CASE STREQUAL "restart")
        set(FIRST_TEST short-LiH-pp-psiformer-ecp-checkpoint-r1-t1)
        maybe_symlink("${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain/wftrain_ecp_restart.xml"
                      "${CMAKE_CURRENT_BINARY_DIR}/${FIRST_TEST}/wftrain_ecp_restart.xml")
        set_property(TEST ${ECP_TEST} PROPERTY DEPENDS ${FIRST_TEST})
        set_property(TEST ${ECP_TEST} PROPERTY WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${FIRST_TEST}")
      endif()
    endif()
  endforeach()
  find_program(PSIFORMER_H5DIFF h5diff)
  find_program(PSIFORMER_H5DUMP h5dump)
  if(TEST_ADDED AND PSIFORMER_H5DIFF AND PSIFORMER_H5DUMP AND Python3_EXECUTABLE)
    add_test(NAME short-LiH-pp-psiformer-ecp-restart-equivalence
             COMMAND ${Python3_EXECUTABLE}
               "${CMAKE_CURRENT_SOURCE_DIR}/compare_ecp_restart.py"
               "${CMAKE_CURRENT_BINARY_DIR}/short-LiH-pp-psiformer-ecp-continuous-r1-t1/lih_pp_ecp_continuous.s000.last_checkpoint.h5"
               "${CMAKE_CURRENT_BINARY_DIR}/short-LiH-pp-psiformer-ecp-checkpoint-r1-t1/lih_pp_ecp_restart.s000.last_checkpoint.h5"
               --h5diff ${PSIFORMER_H5DIFF} --h5dump ${PSIFORMER_H5DUMP})
    set_property(TEST short-LiH-pp-psiformer-ecp-restart-equivalence PROPERTY DEPENDS
                 "short-LiH-pp-psiformer-ecp-continuous-r1-t1;short-LiH-pp-psiformer-ecp-restart-r1-t1")
  endif()
  if(Python3_EXECUTABLE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RSS_TEST short-LiH-pp-psiformer-rss-guard)
    set(RSS_DIR "${CMAKE_CURRENT_BINARY_DIR}/${RSS_TEST}")
    file(MAKE_DIRECTORY "${RSS_DIR}")
    maybe_symlink("${CMAKE_CURRENT_SOURCE_DIR}/wftrain_rss_guard.xml" "${RSS_DIR}/wftrain_rss_guard.xml")
    foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
      maybe_symlink("${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}" "${RSS_DIR}/${PP_FILE}")
    endforeach()
    add_test(NAME ${RSS_TEST} COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/check_rss_guard.py" $<TARGET_FILE:qmcpack>)
    set_tests_properties(${RSS_TEST} PROPERTIES WORKING_DIRECTORY "${RSS_DIR}"
                         ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    add_test(NAME short-H2-psiformer-training-history
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_training_history.py"
             "${qmcpack_BINARY_DIR}/tests/molecules/H2_ae_psiformer_wftrain/short-H2-psiformer-wftrain-pooled-r1-t1/h2_psiformer_pooled")
    set_property(TEST short-H2-psiformer-training-history PROPERTY DEPENDS
                 short-H2-psiformer-wftrain-pooled-r1-t1)
  endif()
  set(POOLED_TEST short-LiH-pp-psiformer-wftrain-pooled-r1-t1)
  set(TEST_ADDED FALSE)
  run_qmc_app(${POOLED_TEST} "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp_psiformer_wftrain"
              1 1 TEST_ADDED wftrain_pooled.xml)
  if(TEST_ADDED)
    foreach(PP_FILE IN ITEMS Li.BFD.xml H.BFD.xml)
      maybe_symlink("${qmcpack_SOURCE_DIR}/tests/molecules/LiH_pp/${PP_FILE}"
                    "${CMAKE_CURRENT_BINARY_DIR}/${POOLED_TEST}/${PP_FILE}")
    endforeach()
    set_property(TEST ${POOLED_TEST} PROPERTY PASS_REGULAR_EXPRESSION
                 "WFTrain iteration 0 version 1 samples 4 mean_energy.*optimizer adam")
  endif()
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
          "WFTrain iteration 0 version 1 samples 1 mean_energy.*optimizer adam")
    else()
      set(TEST_NAME short-LiH-pp-psiformer-wftrain-pii-r1-t1)
      set(EXPECTED_PATTERN
          "WFTrain iteration 0 version 2 samples 2 mean_energy.*optimizer minpii.*krylov_operator_applications [1-9]")
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
                                           "WFTrain iteration 0 version 3 samples 2 mean_energy.*optimizer kfac")
  endif()
endif()
