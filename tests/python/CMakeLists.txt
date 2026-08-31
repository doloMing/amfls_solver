if(TARGET amfls_torch)
    add_test(
        NAME test_torch_binding
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/test_torch_binding.py"
                "$<TARGET_FILE:amfls_torch>"
                "${PROJECT_SOURCE_DIR}")

    add_test(
        NAME test_lsrn_differential
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/test_lsrn_differential.py"
                "$<TARGET_FILE:amfls_torch>")
    set_tests_properties(
        test_lsrn_differential PROPERTIES
        ENVIRONMENT
            "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
endif()
