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

# Create logs, plots, and reports directories if they don't exist
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/logs")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/plots")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/reports")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/experiments")

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
    src/parent_selection.cu
    src/crossover.cu
    src/mutation.cu
)

# Include header files
include_directories(include)

# Create a library for shared logic
add_library(ga_solver_lib ${SRC_FILES})
target_include_directories(ga_solver_lib PRIVATE include)
target_link_libraries(ga_solver_lib PRIVATE ${CMAKE_CUDA_LIBRARIES})

# Add executable for the main application (original)
add_executable(ga_solver src/ga_solver_main.cu)
target_link_libraries(ga_solver PUBLIC ga_solver_lib)

# Add executable for the enhanced application with logging
add_executable(ga_solver_enhanced src/enhanced_ga_solver_main.cu)
target_link_libraries(ga_solver_enhanced PUBLIC ga_solver_lib)

# Add a custom target to run quick experiments
add_custom_target(quick_experiment
    COMMAND python3 ${CMAKE_SOURCE_DIR}/quick_experiment_runner.py --quick
    DEPENDS ga_solver_enhanced
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running quick experiments with enhanced GA solver"
)

# Add a custom target to run full experiments
add_custom_target(full_experiment
    COMMAND python3 ${CMAKE_SOURCE_DIR}/quick_experiment_runner.py
    DEPENDS ga_solver_enhanced
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running full experiments with enhanced GA solver"
)

# Add a custom target to clean experiment data
add_custom_target(clean_experiments
    COMMAND ${CMAKE_COMMAND} -E remove_directory logs
    COMMAND ${CMAKE_COMMAND} -E remove_directory plots
    COMMAND ${CMAKE_COMMAND} -E remove_directory reports
    COMMAND ${CMAKE_COMMAND} -E remove_directory experiments
    COMMAND ${CMAKE_COMMAND} -E make_directory logs
    COMMAND ${CMAKE_COMMAND} -E make_directory plots
    COMMAND ${CMAKE_COMMAND} -E make_directory reports
    COMMAND ${CMAKE_COMMAND} -E make_directory experiments
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning experiment data directories"
)

# Copy Python scripts to build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/quick_experiment_runner.py
    ${CMAKE_BINARY_DIR}/quick_experiment_runner.py
    COPYONLY
)

# Enable testing
enable_testing()

# Add test for quick experiment
add_test(NAME QuickExperimentTest 
         COMMAND python3 quick_experiment_runner.py --quick --datasets gr17 --runs 2
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Data directory: ${DATA_BINARY_DIR}")
message(STATUS "Logs directory: ${CMAKE_SOURCE_DIR}/logs")

# Add install targets
install(TARGETS ga_solver ga_solver_enhanced
        RUNTIME DESTINATION bin)

install(FILES quick_experiment_runner.py
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(DIRECTORY ${DATA_SOURCE_DIR}/
        DESTINATION share/ga_solver/data
        FILES_MATCHING PATTERN "*.csv")