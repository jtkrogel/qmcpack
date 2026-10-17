# CPU-only, external-data-free conventional SJ first-order workflow.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE CONVENTIONAL_SJ_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(CONVENTIONAL_SJ_PYTHON_MODULES EQUAL 0)
    add_test(NAME short-H2-ae-conventional-wftrain-sj
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_sj.xml"
                     --restart --preflight)
    set_tests_properties(short-H2-ae-conventional-wftrain-sj PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-H2-ae-conventional-wftrain-j3
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_sj_j3.xml"
                     --companion-optimizer sr --restart)
    set_tests_properties(short-H2-ae-conventional-wftrain-j3 PROPERTIES TIMEOUT 300 PROCESSORS 1)
    # OneShiftOnly is the built-in LAPACK route; no optional LMY engine required.
    if(NOT ENABLE_OFFLOAD)
      add_test(NAME short-H2-ae-conventional-wftrain-reference-search
               COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_sj_training.py"
                       $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_sj.xml" --sj-reference-line-search)
      set_tests_properties(short-H2-ae-conventional-wftrain-reference-search PROPERTIES TIMEOUT 300 PROCESSORS 1)
      add_test(NAME short-H2-ae-conventional-optimizers-old-new
               COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_sj_training.py"
                       $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_compare.xml" --old-new)
      set_tests_properties(short-H2-ae-conventional-optimizers-old-new PROPERTIES TIMEOUT 300 PROCESSORS 1)
    endif()
  endif()
endif()
