# Qualified fixed-CI, distinct-owner real CPU ordinary-locality orbital workflow.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND NOT ENABLE_OFFLOAD AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE MD_ORBITAL_ECP_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(MD_ORBITAL_ECP_PYTHON_MODULES EQUAL 0)
    add_test(NAME short-C-pp-conventional-wftrain-md-orbital
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_md_orbital_ecp.xml"
                     --md-orbital-ecp)
    set_tests_properties(short-C-pp-conventional-wftrain-md-orbital PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-C-pp-conventional-wftrain-joint-ci-orbital
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_md_orbital_ecp.xml"
                     --joint-ci-orbital-ecp)
    set_tests_properties(short-C-pp-conventional-wftrain-joint-ci-orbital PROPERTIES TIMEOUT 300 PROCESSORS 1)
    add_test(NAME short-C-pp-conventional-wftrain-unequal-orbital
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_md_orbital_ecp.xml"
                     --unequal-spin-orbital-ecp)
    set_tests_properties(short-C-pp-conventional-wftrain-unequal-orbital PROPERTIES TIMEOUT 300 PROCESSORS 1)
  endif()
endif()
