#cmake_minimum_required(VERSION 3.18)
#project(GA_solver LANGUAGES CXX CUDA)



# # Set C++ and CUDA standards first
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# # MSVC runtime settings
# if(MSVC)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
#     set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/MD")
#     set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/MDd")
# endif()

# # Find packages
# find_package(CUDAToolkit REQUIRED)
# find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG REQUIRED)

# # Source files
# set(SRC_FILES
#     src/cuda/genome.cu
#     src/cuda/population.cu
#     src/cuda/cost_matrix.cu
#     src/cuda/fitness_evaluator.cu
#     src/cuda/parent_selection.cu
#     src/cuda/crossover.cu
#     src/cuda/mutation.cu
# )

# # Core library
# add_library(ga_solver_lib SHARED ${SRC_FILES})
# target_include_directories(ga_solver_lib
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
# )
# target_link_libraries(ga_solver_lib PRIVATE CUDA::cudart)

# # Python module
# pybind11_add_module(gasolver src/python/python_bindings.cpp)
# target_include_directories(gasolver
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
# )
# target_link_libraries(gasolver PRIVATE ga_solver_lib)

# # Python module properties
# set_target_properties(gasolver PROPERTIES 
#     OUTPUT_NAME "_core"
#     PREFIX ""
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gasolver"
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gasolver"
# )

# # Copy Python package files
# configure_file(
#     "${CMAKE_SOURCE_DIR}/src/python/__init__.py"
#     "${CMAKE_BINARY_DIR}/gasolver/__init__.py"
#     COPYONLY
# )
# configure_file(
#     "${CMAKE_SOURCE_DIR}/src/python/gasolver_api.py"
#     "${CMAKE_BINARY_DIR}/gasolver/gasolver_api.py"
#     COPYONLY
# )

# # Main executable
# add_executable(ga_solver src/cuda/ga_solver_main.cu)
# target_link_libraries(ga_solver PRIVATE ga_solver_lib)

# # Tests (keeping your existing test setup)
# enable_testing()

# # Add tests for various components
# add_executable(test_genome tests/test_genome.cu)
# target_link_libraries(test_genome PRIVATE ga_solver_lib)
# add_test(NAME GenomeTest COMMAND test_genome)

# add_executable(test_population tests/test_population.cu)
# target_link_libraries(test_population PRIVATE ga_solver_lib)
# add_test(NAME PopulationTest COMMAND test_population)

# add_executable(test_cost_matrix tests/test_cost_matrix.cu)
# target_link_libraries(test_cost_matrix PRIVATE ga_solver_lib)
# add_test(NAME CostMatrix COMMAND test_cost_matrix)

# add_executable(test_fitness_evaluator tests/test_fitness_evaluator.cu)
# target_link_libraries(test_fitness_evaluator PRIVATE ga_solver_lib)
# add_test(NAME FitnessEvaluatorTest COMMAND test_fitness_evaluator)

# add_executable(test_parent_selection tests/test_parent_selection.cu)
# target_link_libraries(test_parent_selection PRIVATE ga_solver_lib)
# add_test(NAME ParentSelectionTest COMMAND test_parent_selection)

# add_executable(test_crossover tests/test_crossover.cu)
# target_link_libraries(test_crossover PRIVATE ga_solver_lib)
# add_test(NAME CrossoverTest COMMAND test_crossover)

# add_executable(test_mutation tests/test_mutation.cu)
# target_link_libraries(test_mutation PRIVATE ga_solver_lib)
# add_test(NAME MutationTest COMMAND test_mutation)


#################################################################################################
#ROUND 2 (Issues with like the right .pyd file being generated and used across all the files)
#################################################################################################

# cmake_minimum_required(VERSION 3.18)
# project(GA_solver LANGUAGES CXX CUDA)

# # Set runtime flags for MSVC early
# if(MSVC)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
#     set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/MD")
#     set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/MDd")
# endif()

# # Set the C++ and CUDA standards
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# enable_language(CUDA)

# # Set build type and append debug flags for CUDA
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# # Set CUDA architectures
# set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# # Copy data files
# configure_file(${CMAKE_SOURCE_DIR}/data/dsj1000.tsp ${CMAKE_BINARY_DIR}/data/dsj1000.tsp COPYONLY)

# # Define source files for the shared library
# set(SRC_FILES
#     src/cuda/genome.cu
#     src/cuda/population.cu
#     src/cuda/cost_matrix.cu
#     src/cuda/fitness_evaluator.cu
#     src/cuda/parent_selection.cu
#     src/cuda/crossover.cu
#     src/cuda/mutation.cu
# )

# # Find required packages
# find_package(CUDAToolkit REQUIRED)
# find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

# # Get pybind11's CMake directory from Python and find pybind11
# execute_process(
#     COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
#     OUTPUT_VARIABLE pybind11_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# find_package(pybind11 CONFIG REQUIRED)

# message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# # Include directories for all targets
# include_directories(
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/src
#     ${Python_INCLUDE_DIRS}
#     ${pybind11_INCLUDE_DIRS}
# )

# # Create the shared library with common CUDA sources
# add_library(ga_solver_lib ${SRC_FILES})
# target_include_directories(ga_solver_lib PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(ga_solver_lib PRIVATE ${CMAKE_CUDA_LIBRARIES})

# # Create the Python module.
# # Instead of compiling the CUDA sources again, we link against the shared library.
# pybind11_add_module(gasolver 
#     gasolver/python_bindings.cpp
# )

# # This snippet ensures that for *each* config, we place outputs in the same folder:
# foreach(config_var IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
#     string(TOUPPER ${config_var} config_upper)
#     set_target_properties(gasolver PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
#         LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
#         ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
#     )
# endforeach()

# target_include_directories(gasolver PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/src
#     ${pybind11_INCLUDE_DIRS}
# )
# target_link_libraries(gasolver PRIVATE ga_solver_lib ${CMAKE_CUDA_LIBRARIES})

# # Get the actual Python package directory from setup.py
# execute_process(
#     COMMAND "${Python_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix=''))"
#     OUTPUT_VARIABLE PYTHON_PACKAGE_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# set_target_properties(gasolver PROPERTIES 
#     # OUTPUT_NAME "_core"
#     # PREFIX ""
#     # SUFFIX ".pyd"
#     # SUFFIX ".cp312-win_amd64.pyd"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver"
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver"
#     # Put it directly in the gasolver package directory
#     # LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gasolver"
#     # RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gasolver"
#     MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
# )

# # Also add this for the main library to fix LIBCMT conflicts
# set_target_properties(ga_solver_lib PROPERTIES
#     MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
# )

# # Create Python package structure
# file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver")
# configure_file(
#     "${CMAKE_SOURCE_DIR}/gasolver/__init__.py"
#     "${CMAKE_BINARY_DIR}/lib/gasolver/__init__.py"
#     COPYONLY
# )
# configure_file(
#     "${CMAKE_SOURCE_DIR}/gasolver/gasolver_api.py"
#     "${CMAKE_BINARY_DIR}/lib/gasolver/gasolver_api.py"
#     COPYONLY
# )

# # Create the main executable
# add_executable(ga_solver src/cuda/ga_solver_main.cu)
# target_link_libraries(ga_solver PUBLIC ga_solver_lib)

# # Enable testing
# enable_testing()

# # Add tests for various components
# add_executable(test_genome tests/test_genome.cu)
# target_link_libraries(test_genome PRIVATE ga_solver_lib)
# add_test(NAME GenomeTest COMMAND test_genome)

# add_executable(test_population tests/test_population.cu)
# target_link_libraries(test_population PRIVATE ga_solver_lib)
# add_test(NAME PopulationTest COMMAND test_population)

# add_executable(test_cost_matrix tests/test_cost_matrix.cu)
# target_link_libraries(test_cost_matrix PRIVATE ga_solver_lib)
# add_test(NAME CostMatrix COMMAND test_cost_matrix)

# add_executable(test_fitness_evaluator tests/test_fitness_evaluator.cu)
# target_link_libraries(test_fitness_evaluator PRIVATE ga_solver_lib)
# add_test(NAME FitnessEvaluatorTest COMMAND test_fitness_evaluator)

# add_executable(test_parent_selection tests/test_parent_selection.cu)
# target_link_libraries(test_parent_selection PRIVATE ga_solver_lib)
# add_test(NAME ParentSelectionTest COMMAND test_parent_selection)

# add_executable(test_crossover tests/test_crossover.cu)
# target_link_libraries(test_crossover PRIVATE ga_solver_lib)
# add_test(NAME CrossoverTest COMMAND test_crossover)

# add_executable(test_mutation tests/test_mutation.cu)
# target_link_libraries(test_mutation PRIVATE ga_solver_lib)
# add_test(NAME MutationTest COMMAND test_mutation)


#################################################################################################
#ROUND 3
#################################################################################################
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")

set(ENV{CC} "/usr/bin/x86_64-pc-linux-gnu-gcc-12")
set(ENV{CXX} "/usr/bin/x86_64-pc-linux-gnu-g++-12")

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")

cmake_minimum_required(VERSION 3.18)
project(GA_solver LANGUAGES CXX CUDA)

if(UNIX)
    # Linux-specific settings
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
endif()

# Platform-specific compiler settings
# Ensure CMake uses GCC 12 if available
if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND bash -c "which x86_64-pc-linux-gnu-gcc-12 || which gcc-12 || which gcc-11 || which gcc-10"
        OUTPUT_VARIABLE GCC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "which x86_64-pc-linux-gnu-g++-12 || which g++-12 || which g++-11 || which g++-10"
        OUTPUT_VARIABLE GXX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(GCC_PATH AND GXX_PATH)
        message(STATUS "Using GCC: ${GCC_PATH}, GXX: ${GXX_PATH}")
        set(CMAKE_C_COMPILER ${GCC_PATH} CACHE FILEPATH "GCC 12 Compiler" FORCE)
        set(CMAKE_CXX_COMPILER ${GXX_PATH} CACHE FILEPATH "G++ 12 Compiler" FORCE)

        # Ensure NVCC uses GCC for compilation
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --compiler-bindir=${GXX_PATH}")
        # Set environment variables to ensure NVCC uses the correct compiler
#        set(ENV{CC} ${GCC_PATH})
#        set(ENV{CXX} ${GXX_PATH})
    else()
        message(FATAL_ERROR "No compatible GCC version (≤12) found. Install GCC 12 manually.")
    endif()
elseif(WIN32)
    # Windows-specific settings
    if(MSVC)
        message(STATUS "Configuring CUDA for Windows with MSVC")
        # Ensure consistent runtime libraries
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=/MD")
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/MDd")
    else()
        message(FATAL_ERROR "Windows builds require MSVC for CUDA compatibility.")
    endif()
endif()

# Set consistent standards across platforms
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

# Set build type and append debug flags for CUDA
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Copy data files
configure_file(${CMAKE_SOURCE_DIR}/data/dsj1000.tsp ${CMAKE_BINARY_DIR}/data/dsj1000.tsp COPYONLY)

# Define source files for the shared library
set(SRC_FILES
    src/cuda/genome.cu
    src/cuda/population.cu
    src/cuda/cost_matrix.cu
    src/cuda/fitness_evaluator.cu
    src/cuda/parent_selection.cu
    src/cuda/crossover.cu
    src/cuda/mutation.cu
)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)

# Get pybind11's CMake directory from Python and find pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# Include directories for all targets
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Create the shared library with common CUDA sources
if(UNIX)
    add_library(ga_solver_lib SHARED ${SRC_FILES})
    target_compile_options(ga_solver_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
else()
#    add_library(ga_solver_lib SHARED ${SRC_FILES})
#    if(MSVC)
#        set_property(TARGET ga_solver_lib PROPERTY
#            MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
#    endif()
    add_library(ga_solver_lib ${SRC_FILES})
    target_include_directories(ga_solver_lib PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
endif()

# Common settings remain the same
target_include_directories(ga_solver_lib PRIVATE 
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(ga_solver_lib PRIVATE ${CMAKE_CUDA_LIBRARIES})

# Create the Python module.
# Instead of compiling the CUDA sources again, we link against the shared library.
if(NOT TARGET _core)
    pybind11_add_module(_core 
        gasolver/python_bindings.cpp
    )
endif()

# This snippet ensures that for *each* config, we place outputs in the same folder:
foreach(config_var IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${config_var} config_upper)
    set_target_properties(_core PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
        ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/gasolver"
    )
endforeach()

target_include_directories(_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${pybind11_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE ga_solver_lib ${CMAKE_CUDA_LIBRARIES})

# Get the actual Python package directory from setup.py
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix=''))"
    OUTPUT_VARIABLE PYTHON_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set_target_properties(_core PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver"
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
)

# Also add this for the main library to fix LIBCMT conflicts
set_target_properties(ga_solver_lib PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
)

# Create Python package structure
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gasolver")
configure_file(
    "${CMAKE_SOURCE_DIR}/gasolver/__init__.py"
    "${CMAKE_BINARY_DIR}/lib/gasolver/__init__.py"
    COPYONLY
)
configure_file(
    "${CMAKE_SOURCE_DIR}/gasolver/gasolver_api.py"
    "${CMAKE_BINARY_DIR}/lib/gasolver/gasolver_api.py"
    COPYONLY
)

# Create the main executable
add_executable(ga_solver src/cuda/ga_solver_main.cu)
target_link_libraries(ga_solver PUBLIC ga_solver_lib)

# Enable testing
enable_testing()

# Add tests for various components
add_executable(test_genome tests/test_genome.cu)
target_link_libraries(test_genome PRIVATE ga_solver_lib)
add_test(NAME GenomeTest COMMAND test_genome)

add_executable(test_population tests/test_population.cu)
target_link_libraries(test_population PRIVATE ga_solver_lib)
add_test(NAME PopulationTest COMMAND test_population)

add_executable(test_cost_matrix tests/test_cost_matrix.cu)
target_link_libraries(test_cost_matrix PRIVATE ga_solver_lib)
add_test(NAME CostMatrix COMMAND test_cost_matrix)

add_executable(test_fitness_evaluator tests/test_fitness_evaluator.cu)
target_link_libraries(test_fitness_evaluator PRIVATE ga_solver_lib)
add_test(NAME FitnessEvaluatorTest COMMAND test_fitness_evaluator)

add_executable(test_parent_selection tests/test_parent_selection.cu)
target_link_libraries(test_parent_selection PRIVATE ga_solver_lib)
add_test(NAME ParentSelectionTest COMMAND test_parent_selection)

add_executable(test_crossover tests/test_crossover.cu)
target_link_libraries(test_crossover PRIVATE ga_solver_lib)
add_test(NAME CrossoverTest COMMAND test_crossover)

add_executable(test_mutation tests/test_mutation.cu)
target_link_libraries(test_mutation PRIVATE ga_solver_lib)
add_test(NAME MutationTest COMMAND test_mutation)

