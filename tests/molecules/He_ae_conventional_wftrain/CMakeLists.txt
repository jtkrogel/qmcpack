# Qualified real CPU single-Slater orbital-only and joint-Jastrow workflows.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND NOT ENABLE_OFFLOAD AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE CONVENTIONAL_ORBITAL_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(CONVENTIONAL_ORBITAL_PYTHON_MODULES EQUAL 0)
    set(ORBITAL_CHECKER "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py")
    add_test(NAME short-He-ae-conventional-wftrain-orbital
             COMMAND ${Python3_EXECUTABLE} "${ORBITAL_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_orbital.xml"
                     --companion-optimizer sr --restart --preflight)
    add_test(NAME short-He-ae-conventional-wftrain-orbital-jastrow
             COMMAND ${Python3_EXECUTABLE} "${ORBITAL_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_orbital.xml"
                     --orbital-jastrow --companion-optimizer sr --restart)
    add_test(NAME short-He-ae-conventional-model-import
             COMMAND ${Python3_EXECUTABLE} "${ORBITAL_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_orbital.xml" --model-import)
    add_test(NAME short-He-ae-conventional-wftrain-md-orbital
             COMMAND ${Python3_EXECUTABLE} "${ORBITAL_CHECKER}" $<TARGET_FILE:qmcpack>
                     "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_md_orbital.xml" --md-orbital)
    set_tests_properties(short-He-ae-conventional-wftrain-orbital
                         short-He-ae-conventional-wftrain-orbital-jastrow
                         short-He-ae-conventional-model-import
                         short-He-ae-conventional-wftrain-md-orbital
                         PROPERTIES TIMEOUT 300 PROCESSORS 1)
  endif()
endif()
