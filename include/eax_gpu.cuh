// eax_gpu.cuh: Scalable Edge Assembly Crossover using O(n) adjacency lists
// Replaces the O(n^2) adjacency matrix approach for large TSP instances

#ifndef EAX_GPU_CUH
#define EAX_GPU_CUH

#include "gpu_population.cuh"

// Perform EAX crossover on GPU-resident population
// Reads parent tours from population at parent indices
// Writes offspring to offspring buffers
// useCostAware: true = use actual travel costs, false = use uniform costs
cudaError_t performGPUEAXCrossover(
    GPUPopulation& population,
    GPUProblemData& problemData,
    EAXWorkspace& workspace,
    uint32_t numPairs,
    bool useCostAware = true
);

// Individual kernel declarations

// Kernel 1: Build adjacency lists from parent tours
// Each city records its 4 neighbors (2 from each parent)
// O(n) memory instead of O(n^2)
__global__ void buildAdjacencyListsKernel(
    const uint16_t* __restrict__ parent1Tours,   // [numCities * numPairs] SoA
    const uint16_t* __restrict__ parent2Tours,   // [numCities * numPairs] SoA
    CityAdjacency* __restrict__ adjacencyLists,  // [numCities * numPairs]
    uint32_t numCities,
    uint32_t numPairs,
    uint32_t parent1Stride,  // Stride for parent1 access (popSize for main pop)
    uint32_t parent2Stride   // Stride for parent2 access
);

// Kernel 2: Find alternating AB-cycles
// Uses O(1) neighbor lookup from adjacency lists
__global__ void findAlternatingCyclesKernel(
    const CityAdjacency* __restrict__ adjacencyLists,
    uint8_t* __restrict__ visited,
    uint16_t* __restrict__ cycleBuffer,      // [GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * numPairs]
    uint8_t* __restrict__ cycleParentFlags,  // [GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * numPairs]
    uint16_t* __restrict__ cycleLengths,     // [GPU_MAX_CYCLES * numPairs]
    uint16_t* __restrict__ numCycles,        // [numPairs]
    uint32_t numCities,
    uint32_t numPairs
);

// Kernel 3: Evaluate assemblies and select best
// For k cycles, evaluates up to 2^k combinations (limited to 2^16)
// Uses greedy fallback for k > 16
// useCostAware: true = use actual travel costs, false = use uniform costs
__global__ void evaluateAssembliesKernel(
    const uint16_t* __restrict__ cycleBuffer,
    const uint8_t* __restrict__ cycleParentFlags,
    const uint16_t* __restrict__ cycleLengths,
    const uint16_t* __restrict__ numCycles,
    const uint16_t* __restrict__ parent1Tours,
    const uint16_t* __restrict__ parent2Tours,
    const float* __restrict__ travelCosts,
    uint32_t* __restrict__ bestAssembly,
    float* __restrict__ assemblyCosts,
    uint32_t numCities,
    uint32_t numPairs,
    uint32_t parent1Stride,
    uint32_t parent2Stride,
    bool useCostAware
);

// Kernel 4: Construct offspring tour from selected assembly
// Applies selected cycles to parent1 tour to create offspring
__global__ void constructOffspringTourKernel(
    const CityAdjacency* __restrict__ adjacencyLists,
    const uint16_t* __restrict__ cycleBuffer,
    const uint8_t* __restrict__ cycleParentFlags,
    const uint16_t* __restrict__ cycleLengths,
    const uint16_t* __restrict__ numCycles,
    const uint32_t* __restrict__ bestAssembly,
    const uint16_t* __restrict__ parent1Tours,
    const uint16_t* __restrict__ parent2Tours,
    uint16_t* __restrict__ offspringTours,
    uint32_t numCities,
    uint32_t numPairs,
    uint32_t parent1Stride,
    uint32_t parent2Stride,
    uint32_t offspringStride
);

// Helper: Gather parent tours based on selection indices
__global__ void gatherParentToursKernel(
    const uint16_t* __restrict__ populationTours,  // [numCities * popSize]
    const uint32_t* __restrict__ parentIndices,    // [numPairs]
    uint16_t* __restrict__ gatheredTours,          // [numCities * numPairs]
    uint32_t numCities,
    uint32_t popSize,
    uint32_t numPairs
);

// Device helper functions

// Get neighbor from adjacency list
// neighborType: 0=p1_prev, 1=p1_next, 2=p2_prev, 3=p2_next
__device__ inline uint16_t getNeighbor(const CityAdjacency& adj, int neighborType) {
    return adj.neighbors[neighborType];
}

// Check if city has an edge to target from specified parent
__device__ inline bool hasEdgeToCity(const CityAdjacency& adj, uint16_t target, int parentId) {
    int offset = parentId * 2;  // 0 for parent1, 2 for parent2
    return (adj.neighbors[offset] == target) || (adj.neighbors[offset + 1] == target);
}

// Get the "other" neighbor from the same parent
// If we came from neighbor[offset], return neighbor[offset+1], and vice versa
__device__ inline uint16_t getOtherNeighbor(const CityAdjacency& adj, uint16_t cameFrom, int parentId) {
    int offset = parentId * 2;
    if (adj.neighbors[offset] == cameFrom) {
        return adj.neighbors[offset + 1];
    } else {
        return adj.neighbors[offset];
    }
}

#endif // EAX_GPU_CUH
