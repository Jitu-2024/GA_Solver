// mutation_gpu.cuh: GPU-based mutation operators for TSPJ solver

#ifndef MUTATION_GPU_CUH
#define MUTATION_GPU_CUH

#include "gpu_population.cuh"

// Apply mutations to offspring population
cudaError_t performGPUMutations(
    GPUPopulation& population,
    GPUProblemData& problemData,
    GPURngState& rngState,
    uint32_t numGenomes,
    float mutationRate,
    uint32_t stagnationCount,
    bool isOffspring
);

// 2-opt mutation kernel with escape mechanism for city/pickup sequences
// Reverses a segment of the tour, preferring larger moves
__global__ void twoOptMutationKernel(
    uint16_t* __restrict__ sequences,        // [numCities * numGenomes]
    const float* __restrict__ travelCosts,   // [numCities * numCities]
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride,
    uint32_t stagnationCount                 // For escape probability calculation
);

// Double-bridge mutation for escaping local optima
// Swaps 4 segments of the tour
__global__ void doubleBridgeMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride
);

// Swap mutation for job sequences
// Swaps two random positions
__global__ void swapMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
);

// Scramble mutation - shuffles a random segment
__global__ void scrambleMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
);

// Inversion mutation - reverses a random segment
__global__ void inversionMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
);

// 3-opt style mutation - more disruptive than 2-opt
// Performs one of 4 possible segment operations
__global__ void threeOptMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride
);

// Helper: Reverse a segment of sequence in-place
__device__ void reverseSegment(uint16_t* sequence, uint32_t start, uint32_t end,
                                uint32_t genomeIdx, uint32_t stride);

#endif // MUTATION_GPU_CUH
