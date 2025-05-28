CXX = nvcc

# Add all your .cu source files here
SRCS = src/cuda/ga_solver_main.cu \
       src/cuda/population.cu \
       src/cuda/genome.cu \
       src/cuda/cost_matrix.cu \
       src/cuda/fitness_evaluator.cu \
       src/cuda/parent_selection.cu \
       src/cuda/crossover.cu \
       src/cuda/mutation.cu

# Executable name
TARGET = ga_solver_executable

# Compiler flags
# IMPORTANT: Replace sm_XX with the compute capability of your target GPU
# e.g., sm_70, sm_75, sm_80, sm_86. You can list multiple if needed.
# Use -std=c++11 or higher if your code requires it (e.g., c++14, c++17)
CXXFLAGS = -std=c++11 -arch=sm_75 # <-- MODIFY sm_XX HERE

# Linker flags (if any)
LDFLAGS = 

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling CUDA GA Solver..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)
	@echo "Build finished: $(TARGET)"

# Target for running tests (assuming test executables are built separately or tests are integrated)
# This is a placeholder; you'll need to define how your tests are compiled and run.
# Example: If each test_X.cu compiles to test_X_executable
# TESTS_SRCS = $(wildcard tests/test_*.cu)
# TEST_TARGETS = $(patsubst tests/test_%.cu, tests/test_%_executable, $(TESTS_SRCS))
# run_tests: $(TEST_TARGETS)
# 	@echo "Running tests..."
# 	for test_exec in $(TEST_TARGETS); do ./$$test_exec; done

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) # Removes the main executable
	# Add commands to remove test executables if you have them, e.g.:
	# rm -f tests/test_*_executable 

.PHONY: all clean run_tests 