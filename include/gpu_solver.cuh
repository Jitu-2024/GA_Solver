// gpu_solver.cuh: Main GPU-resident GA solver interface

#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include "gpu_population.cuh"
#include "gpu_memory_manager.cuh"
#include <vector>
#include <string>

// Solver parameters
struct GPUSolverParams {
    uint32_t populationSize = 500;
    uint32_t maxGenerations = 10000;
    uint32_t maxStagnationGenerations = 1500;
    float baseMutationRate = 0.3f;
    uint32_t tournamentSize = 5;
    uint32_t loggingInterval = 100;  // Log every N generations
    bool verbose = true;
    uint64_t rngSeed = 12345;
    bool useCostAware = false;  // Use cost-aware EAX assembly selection
};

// Solver result
struct GPUSolverResult {
    std::vector<uint16_t> bestCitySequence;
    std::vector<uint16_t> bestJobSequence;
    std::vector<uint16_t> bestPickupSequence;
    float bestFitness;
    uint32_t generationsRun;
    uint32_t solutionGeneration;  // Generation where best was found
    double elapsedTimeSeconds;
};

// Main solver function
GPUSolverResult runGPUSolver(
    const std::vector<std::vector<float>>& travelCosts,
    const std::vector<std::vector<float>>& jobCosts,
    int mode,
    const GPUSolverParams& params = GPUSolverParams()
);

// Fitness evaluation kernel (GPU-resident version)
__global__ void evaluateFitnessGPUKernel(
    const uint16_t* __restrict__ citySequences,
    const uint16_t* __restrict__ jobSequences,
    const uint16_t* __restrict__ pickupSequences,  // nullptr if mode != 1
    const float* __restrict__ travelCosts,
    const float* __restrict__ jobCosts,
    float* __restrict__ fitness,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t numGenomes,
    uint32_t stride,
    int mode
);

// Tournament selection kernel
__global__ void tournamentSelectionKernel(
    const float* __restrict__ fitness,
    uint32_t* __restrict__ parent1Indices,
    uint32_t* __restrict__ parent2Indices,
    curandState* __restrict__ rngStates,
    uint32_t populationSize,
    uint32_t numPairs,
    uint32_t tournamentSize
);

// Survivor selection: merge population and offspring, keep best
cudaError_t performSurvivorSelection(
    GPUPopulation& population,
    uint32_t numOffspring
);

#endif // GPU_SOLVER_CUH
