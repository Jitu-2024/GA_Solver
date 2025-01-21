# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GA_solver LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Project name and version
project(GA_SOLVER LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA
enable_language(CUDA)

# Enable debug flags for host and device code
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Define data directories
set(DATA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/data")
set(DATA_BINARY_DIR "${CMAKE_BINARY_DIR}/data")

# Create binary data directory
file(MAKE_DIRECTORY ${DATA_BINARY_DIR})

# Find all CSV files in the data directory
file(GLOB DATA_FILES "${DATA_SOURCE_DIR}/*.csv")

# Copy each data file to the binary directory
foreach(DATA_FILE ${DATA_FILES})
    get_filename_component(FILENAME ${DATA_FILE} NAME)
    configure_file(${DATA_FILE} ${DATA_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

# Define DATA_DIR for the source code
add_compile_definitions(DATA_DIR="${DATA_BINARY_DIR}")

add_compile_definitions(DATA_DIR="${CMAKE_BINARY_DIR}/data")

add_compile_definitions(LOGS_DIR="${CMAKE_SOURCE_DIR}/logs")

# Add source files
set(SRC_FILES
    src/genome.cu
    src/population.cu
    src/cost_matrix.cu
    src/fitness_evaluator.cu
    src/ga_solver_main.cu
    src/parent_selection.cu
    src/crossover.cu
    src/mutation.cu
    # Add other source files as needed
)

# Include header files
include_directories(include)

# Create a library for shared logic
add_library(ga_solver_lib ${SRC_FILES})
target_include_directories(ga_solver_lib PRIVATE include)
target_link_libraries(ga_solver_lib PRIVATE ${CMAKE_CUDA_LIBRARIES})

# Add executable for the main application
add_executable(ga_solver src/ga_solver_main.cu)
target_link_libraries(ga_solver PUBLIC ga_solver_lib)

# Enable testing
enable_testing()

# Add test for Genome
add_executable(test_genome tests/test_genome.cu)
target_link_libraries(test_genome PRIVATE ga_solver_lib)
add_test(NAME GenomeTest COMMAND test_genome)


# Add test for Population
add_executable(test_population tests/test_population.cu)
target_link_libraries(test_population PRIVATE ga_solver_lib)
add_test(NAME PopulationTest COMMAND test_population)

# Add test for Cost_Matrix
add_executable(test_cost_matrix tests/test_cost_matrix.cu)
target_link_libraries(test_cost_matrix PRIVATE ga_solver_lib)
add_test(NAME CostMatrix COMMAND test_cost_matrix)

# Add test for FitnessEvaluator
add_executable(test_fitness_evaluator tests/test_fitness_evaluator.cu)
target_link_libraries(test_fitness_evaluator PRIVATE ga_solver_lib)
add_test(NAME FitnessEvaluatorTest COMMAND test_fitness_evaluator)

# Add test for Parent Selection
add_executable(test_parent_selection tests/test_parent_selection.cu)
target_link_libraries(test_parent_selection PRIVATE ga_solver_lib)
add_test(NAME ParentSelectionTest COMMAND test_parent_selection)

# Add test for Crossover
add_executable(test_crossover tests/test_crossover.cu)
target_link_libraries(test_crossover PRIVATE ga_solver_lib)
add_test(NAME CrossoverTest COMMAND test_crossover)

# Add test for Mutation
add_executable(test_mutation tests/test_mutation.cu)
target_link_libraries(test_mutation PRIVATE ga_solver_lib)
add_test(NAME MutationTest COMMAND test_mutation)
