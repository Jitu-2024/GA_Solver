// gpu_population.cuh: GPU-resident population data structures for TSPJ solver
// Uses Structure of Arrays (SoA) layout for coalesced memory access

#ifndef GPU_POPULATION_CUH
#define GPU_POPULATION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// Configuration - can be overridden at compile time
#ifndef GPU_MAX_CITIES
#define GPU_MAX_CITIES 2048
#endif

#ifndef GPU_MAX_POPULATION
#define GPU_MAX_POPULATION 2048
#endif

// EAX configuration
#define GPU_MAX_CYCLES 128
#define GPU_MAX_CYCLE_LENGTH 512
#define GPU_NEIGHBORS_PER_CITY 4  // 2 from each parent in union graph

// GPU-resident population using Structure of Arrays (SoA)
// Layout: sequence[position * popSize + genomeIdx] for coalesced access
struct GPUPopulation {
    // Main population sequences
    uint16_t* d_citySequences;      // [numCities * popSize] - tour order
    uint16_t* d_jobSequences;       // [numJobs * popSize] - job assignments
    uint16_t* d_pickupSequences;    // [numCities * popSize] or nullptr (mode 1 only)
    float* d_fitness;               // [popSize] - makespan values

    // Offspring buffers (pre-allocated, reusable each generation)
    uint16_t* d_offspringCity;      // [numCities * maxOffspring]
    uint16_t* d_offspringJob;       // [numJobs * maxOffspring]
    uint16_t* d_offspringPickup;    // [numCities * maxOffspring] or nullptr
    float* d_offspringFitness;      // [maxOffspring]

    // Selection and sorting scratch space
    uint32_t* d_parent1Indices;     // [maxOffspring] - selected parent1 indices
    uint32_t* d_parent2Indices;     // [maxOffspring] - selected parent2 indices
    uint32_t* d_sortIndices;        // [popSize + maxOffspring] - for sorting

    // Population metadata
    uint32_t popSize;               // Current population size
    uint32_t maxPopSize;            // Maximum population capacity
    uint32_t numCities;             // Number of cities (excluding depot)
    uint32_t numJobs;               // Number of job types
    uint32_t maxOffspring;          // Maximum offspring per generation
    uint32_t currentOffspring;      // Current offspring count
    int mode;                       // 0 = no pickup, 1 = with pickup
};

// Problem data - stays constant on GPU after initialization
struct GPUProblemData {
    float* d_travelCosts;           // [numCities * numCities] - travel time matrix
    float* d_jobCosts;              // [(numCities-1) * numJobs] - job time at each city

    uint32_t numCities;             // Total cities including depot (city 0)
    uint32_t numJobs;               // Number of job types
};

// EAX workspace - adjacency list representation
struct CityAdjacency {
    uint16_t neighbors[GPU_NEIGHBORS_PER_CITY];  // [p1_prev, p1_next, p2_prev, p2_next]
    // Bits 0-1: parent mask for each neighbor (0=parent1, 1=parent2)
};

// Cycle data for EAX
struct EAXCycleData {
    uint16_t* d_cycleBuffer;        // [GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * numPairs]
    uint16_t* d_cycleLengths;       // [GPU_MAX_CYCLES * numPairs]
    uint16_t* d_numCycles;          // [numPairs]
    uint8_t* d_cycleParentFlags;    // [GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * numPairs]
};

// Complete EAX workspace
struct EAXWorkspace {
    CityAdjacency* d_adjacencyLists;  // [numCities * numPairs]
    uint8_t* d_visited;               // [numCities * numPairs]
    EAXCycleData cycles;
    uint32_t* d_bestAssembly;         // [numPairs] - bitmask of selected cycles
    float* d_assemblyCosts;           // [numPairs]

    // Gathered parent tours for crossover
    uint16_t* d_parent1Gathered;      // [numCities * maxPairs] - gathered parent1 tours
    uint16_t* d_parent2Gathered;      // [numCities * maxPairs] - gathered parent2 tours

    // Gathered parent job sequences for crossover
    uint16_t* d_parent1Jobs;          // [numJobs * maxPairs] - gathered parent1 jobs
    uint16_t* d_parent2Jobs;          // [numJobs * maxPairs] - gathered parent2 jobs

    uint32_t maxPairs;                // Maximum crossover pairs
    uint32_t numCities;
    uint32_t numJobs;
};

// RNG state for GPU kernels
struct GPURngState {
    curandState* d_states;          // [maxPopSize] - one per potential genome
    uint32_t numStates;
};

// Helper functions - host side
#ifdef __cplusplus
extern "C" {
#endif

// Population management
cudaError_t allocateGPUPopulation(GPUPopulation* pop, uint32_t maxPopSize,
                                   uint32_t numCities, uint32_t numJobs, int mode);
cudaError_t freeGPUPopulation(GPUPopulation* pop);

// Problem data management
cudaError_t allocateGPUProblemData(GPUProblemData* data, uint32_t numCities, uint32_t numJobs);
cudaError_t freeGPUProblemData(GPUProblemData* data);
cudaError_t loadProblemData(GPUProblemData* data,
                            const float* travelCosts, const float* jobCosts,
                            uint32_t numCities, uint32_t numJobs);

// EAX workspace management
cudaError_t allocateEAXWorkspace(EAXWorkspace* workspace, uint32_t numCities, uint32_t maxPairs);
cudaError_t freeEAXWorkspace(EAXWorkspace* workspace);

// RNG management
cudaError_t allocateGPURngState(GPURngState* rng, uint32_t numStates, uint64_t seed);
cudaError_t freeGPURngState(GPURngState* rng);

#ifdef __cplusplus
}
#endif

// Inline device helper for SoA indexing
__device__ __host__ inline uint32_t soaIndex(uint32_t position, uint32_t genomeIdx, uint32_t popSize) {
    return position * popSize + genomeIdx;
}

// Inline device helper for getting city at position in genome
__device__ inline uint16_t getCity(const uint16_t* citySequences,
                                    uint32_t position, uint32_t genomeIdx, uint32_t popSize) {
    return citySequences[soaIndex(position, genomeIdx, popSize)];
}

// Inline device helper for setting city at position in genome
__device__ inline void setCity(uint16_t* citySequences,
                                uint32_t position, uint32_t genomeIdx, uint32_t popSize, uint16_t city) {
    citySequences[soaIndex(position, genomeIdx, popSize)] = city;
}

#endif // GPU_POPULATION_CUH
