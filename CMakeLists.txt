cmake_minimum_required(VERSION 3.18)

# Project name and languages
project(GA_Solver LANGUAGES CXX CUDA)

# Specify C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

# Add executable
add_executable(ga_solver ${SOURCE_FILES})

# Link CUDA libraries
target_link_libraries(ga_solver ${CUDA_LIBRARIES})

# Enable warnings for development
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ga_solver PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

# Add source files for GPU kernels
set(GPU_KERNELS ${PROJECT_SOURCE_DIR}/src/gpu/genetic_kernels.cu)

# Add test executable
file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/*.cpp)
add_executable(tests ${TEST_SOURCES} ${GPU_KERNELS})

# Link with Google Test and CUDA
target_link_libraries(tests gtest gtest_main pthread ${CUDA_LIBRARIES})

# Add test command
add_test(NAME MutationTests COMMAND tests)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
