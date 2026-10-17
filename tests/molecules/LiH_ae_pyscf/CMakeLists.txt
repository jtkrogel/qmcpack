if(NOT QMC_COMPLEX)
  #
  # LiH molecule gaussian tests, all electron using PySCF
  # Also check results for different number of mpi tasks and threads keeping total constant
  # Energy from PySCF: E=-7.9873236457148
  #
  #   "kinetic" "7.991344  0.000065") # kinetic energy
  #   "totenergy" "-7.9873125  0.0000086 ") # total energy
  #   "eeenergy" "3.488827 0.000010") # e-e energy
  #   "samples" "80117760000 0.0") # samples

  list(APPEND LIH_SHORT "kinetic" "7.991344 0.01") # kinetic energy
  list(APPEND LIH_SHORT "totenergy" "-7.9873125 0.002") # total energy
  list(APPEND LIH_SHORT "eeenergy" "3.488827 0.001") # e-e energy
  list(APPEND LIH_SHORT "samples" "9600000 0.0") # samples
  qmc_run_and_check(
    short-LiH_ae_pyscf-vmc_hf_noj
    "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_ae_pyscf"
    vmc_short_noj
    vmc_short_noj.in.xml
    1
    16
    TRUE
    0
    LIH_SHORT # VMC
  )

  list(APPEND LIH_LONG "kinetic" "7.991344 0.0022") # kinetic energy
  list(APPEND LIH_LONG "totenergy" "-7.9873125 0.00024") # total energy
  list(APPEND LIH_LONG "eeenergy" "3.488827 0.00029") # e-e energy
  list(APPEND LIH_LONG "samples" "96000000 0.0") # samples
  qmc_run_and_check(
    long-LiH_ae_pyscf-vmc_hf_noj
    "${qmcpack_SOURCE_DIR}/tests/molecules/LiH_ae_pyscf"
    vmc_long_noj
    vmc_long_noj.in.xml
    1
    16
    TRUE
    0
    LIH_LONG # VMC
  )

else()
  message(VERBOSE
          "Skipping LiH_ae tests because gaussian basis sets are not supported by complex build (QMC_COMPLEX=1)")
endif()

# Qualified small external LCAO profile: real CPU only, no backend matrix.
if(NOT QMC_COMPLEX AND NOT MIXED_PRECISION AND NOT ENABLE_OFFLOAD AND Python3_EXECUTABLE)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                  RESULT_VARIABLE CONVENTIONAL_LCAO_PYTHON_MODULES OUTPUT_QUIET ERROR_QUIET)
  if(CONVENTIONAL_LCAO_PYTHON_MODULES EQUAL 0)
    add_test(NAME short-LiH-ae-conventional-external-lcao
             COMMAND ${Python3_EXECUTABLE}
                     "${qmcpack_SOURCE_DIR}/tests/molecules/H2_ae_conventional_wftrain/check_sj_training.py"
                     $<TARGET_FILE:qmcpack> "${CMAKE_CURRENT_SOURCE_DIR}/wftrain_external_lcao.xml" --external-lcao)
    set_tests_properties(short-LiH-ae-conventional-external-lcao PROPERTIES TIMEOUT 300 PROCESSORS 1)
  endif()
endif()

get_property(CURRENT_TESTS DIRECTORY PROPERTY TESTS)
add_labels_for_tests("${CURRENT_TESTS}")
