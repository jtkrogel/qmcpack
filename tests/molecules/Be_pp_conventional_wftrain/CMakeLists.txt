# Qualified fixed-Slater real CPU ordinary-locality VP pseudopotential workflow.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND NOT ENABLE_OFFLOAD AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE CONVENTIONAL_ECP_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(CONVENTIONAL_ECP_PYTHON_MODULES EQUAL 0)
    add_test(NAME short-Be-pp-conventional-wftrain-ecp
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_ecp.xml"
                     --companion-optimizer sr --restart --preflight)
    set_tests_properties(short-Be-pp-conventional-wftrain-ecp PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-mdj
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj_ecp.xml" --restart --preflight)
    set_tests_properties(short-Be-pp-conventional-wftrain-mdj PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-csf
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj_ecp.xml"
                     --ci-representation csf --restart --preflight)
    set_tests_properties(short-Be-pp-conventional-wftrain-csf PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-mdj-pii
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj_ecp.xml" --mdj-pii)
    set_tests_properties(short-Be-pp-conventional-wftrain-mdj-pii PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-ci-lm
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_mdj_ecp.xml" --ci-lm)
    set_tests_properties(short-Be-pp-conventional-wftrain-ci-lm PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-j3
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_j3_ecp.xml" --restart --preflight)
    set_tests_properties(short-Be-pp-conventional-wftrain-j3 PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-wftrain-orbital
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_orbital_ecp.xml" --orbital-ecp)
    set_tests_properties(short-Be-pp-conventional-wftrain-orbital PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-Be-pp-conventional-model-import
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_ecp.xml" --model-import)
    set_tests_properties(short-Be-pp-conventional-model-import PROPERTIES TIMEOUT 300 PROCESSORS 1)
  endif()
endif()
