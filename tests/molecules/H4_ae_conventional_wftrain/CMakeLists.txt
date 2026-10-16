if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE CONVENTIONAL_MDJ_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(CONVENTIONAL_MDJ_PYTHON_MODULES EQUAL 0)
    set(MDJ_CHECKER "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py")
    add_test(NAME short-H4-ae-conventional-wftrain-mdj
             COMMAND ${Python3_EXECUTABLE} "${MDJ_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj.xml" --companion-optimizer sr --restart)
    add_test(NAME short-H4-ae-conventional-wftrain-csf
             COMMAND ${Python3_EXECUTABLE} "${MDJ_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj.xml" --ci-representation csf --profile-only)
    set_tests_properties(short-H4-ae-conventional-wftrain-mdj short-H4-ae-conventional-wftrain-csf
                         PROPERTIES TIMEOUT 300 PROCESSORS 1)
  endif()
endif()
