cmake_minimum_required(VERSION 3.24)

project(amfls_solver VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AMFLS_BUILD_TORCH "Build the PyTorch CPU binding" OFF)
option(AMFLS_BUILD_EXAMPLES "Build C++ examples" ON)

set(BLA_SIZEOF_INTEGER 4)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    set(AMFLS_BLAS_TARGET BLAS::BLAS)
else()
    find_file(AMFLS_BLAS_LIBRARY
        NAMES libopenblas.so.0 libblas.so.3
        PATHS /usr/lib /usr/lib64 /usr/local/lib
        PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
        REQUIRED)
    add_library(amfls_blas UNKNOWN IMPORTED)
    set_target_properties(amfls_blas PROPERTIES
        IMPORTED_LOCATION "${AMFLS_BLAS_LIBRARY}")
    set(AMFLS_BLAS_TARGET amfls_blas)
endif()

find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    set(AMFLS_LAPACK_TARGET LAPACK::LAPACK)
else()
    find_file(AMFLS_LAPACK_LIBRARY
        NAMES liblapack.so.3
        PATHS /usr/lib /usr/lib64 /usr/local/lib
        PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
        REQUIRED)
    add_library(amfls_lapack UNKNOWN IMPORTED)
    set_target_properties(amfls_lapack PROPERTIES
        IMPORTED_LOCATION "${AMFLS_LAPACK_LIBRARY}")
    set(AMFLS_LAPACK_TARGET amfls_lapack)
endif()
find_package(OpenMP QUIET)

set(AMFLS_CORE_SOURCES
    src/algorithms/amfls/amfls.cpp
    src/algorithms/aplicur/aplicur.cpp
    src/algorithms/fixed_rbgk/fixed_rbgk.cpp
    src/algorithms/lsmb/lsmb.cpp
    src/algorithms/lsmr/lsmr.cpp
    src/algorithms/lsrn/lsrn.cpp
    src/algorithms/lsqr/lsqr.cpp
    src/algorithms/randomized_block_cg/randomized_block_cg.cpp
    src/algorithms/sparse_embedding_lsqr/sparse_embedding_lsqr.cpp
    src/algorithms/spir_fossils/spir_fossils.cpp
    src/algorithms/mathematics/krylov/candidate_validation.cpp
    src/algorithms/mathematics/krylov/krylov_workspace.cpp
    src/algorithms/mathematics/linalg/blas_lapack.cpp
    src/algorithms/mathematics/linalg/matrix.cpp
    src/algorithms/mathematics/linalg/svd.cpp
    src/algorithms/mathematics/operators/dense_operator.cpp
    src/algorithms/mathematics/operators/row_major_dense_operator.cpp
    src/algorithms/mathematics/operators/sparse_csr_operator.cpp
    src/algorithms/mathematics/random/gaussian.cpp)

add_library(amfls_core STATIC ${AMFLS_CORE_SOURCES})
set_target_properties(amfls_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(amfls_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(amfls_core PUBLIC
    ${AMFLS_LAPACK_TARGET} ${AMFLS_BLAS_TARGET})
if(OpenMP_CXX_FOUND)
    target_link_libraries(amfls_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(amfls_core PRIVATE -Wall -Wextra -Wpedantic)

if(AMFLS_BUILD_TORCH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])"
        OUTPUT_VARIABLE TORCH_LIBRARY_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import torch; print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE TORCH_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    set(AMFLS_TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" CACHE INTERNAL
        "Torch include directories used by AMFLS")
    set(AMFLS_TORCH_LIBRARY_DIR "${TORCH_LIBRARY_DIR}" CACHE INTERNAL
        "Torch library directory used by AMFLS")
    set(AMFLS_TORCH_CXX11_ABI "${TORCH_CXX11_ABI}" CACHE INTERNAL
        "Torch C++ ABI used by AMFLS")

    find_library(TORCH_CPU_LIBRARY NAMES torch_cpu
        PATHS "${TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(TORCH_C10_LIBRARY NAMES c10
        PATHS "${TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)

    add_library(amfls_torch MODULE bindings/torch/torch_binding.cpp)
    target_include_directories(amfls_torch PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(amfls_torch PRIVATE
        _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
    target_link_libraries(amfls_torch PRIVATE
        amfls_core "${TORCH_CPU_LIBRARY}" "${TORCH_C10_LIBRARY}")
    set_target_properties(amfls_torch PROPERTIES
        PREFIX "" BUILD_RPATH "${TORCH_LIBRARY_DIR}")
endif()

if(AMFLS_BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_subdirectory(tests/cpp)
    add_subdirectory(tests/python)
endif()
