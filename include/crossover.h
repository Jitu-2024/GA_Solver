// crossover.h: Header file for GPU-based Hybrid Crossover (EAX + Uniform) in TSPJ

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "genome.h"
#include <vector>

// Maximum problem size constraints for GPU memory allocation
#define MAX_CITIES 256
#define MAX_CYCLES 64
#define MAX_CYCLE_LENGTH 64

// Edge structure for EAX
struct Edge {
    uint16_t from, to;
    __device__ __host__ Edge() : from(0), to(0) {}
    __device__ __host__ Edge(uint16_t f, uint16_t t) : from(f), to(t) {}
    __device__ __host__ bool operator==(const Edge& other) const {
        return (from == other.from && to == other.to) || 
               (from == other.to && to == other.from);
    }
};

// Cycle structure for EAX
struct Cycle {
    uint16_t length;
    uint16_t cities[MAX_CYCLE_LENGTH];
    bool usesParent1[MAX_CYCLE_LENGTH]; // Which parent each edge comes from
    
    __device__ __host__ Cycle() : length(0) {}
};

// =============================================================================
// MAIN CROSSOVER INTERFACE FUNCTIONS
// =============================================================================

/**
 * Main crossover function - uses EAX for routing + uniform for job assignment
 * @param parent1 First parent genome
 * @param parent2 Second parent genome
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Child genome created by hybrid crossover
 */
Genome performCrossover(const Genome& parent1, const Genome& parent2, int mode);

/**
 * Batch processing version for multiple parent pairs
 * Uses EAX for routing chromosomes and uniform crossover for job assignment
 * @param parents1 Vector of first parents
 * @param parents2 Vector of second parents
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Vector of child genomes
 */
std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                            const std::vector<Genome>& parents2, 
                                            int mode);

/**
 * Cost-aware version of hybrid crossover
 * @param parent1 First parent genome
 * @param parent2 Second parent genome
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Child genome
 */
Genome performCostAwareEAXCrossover(const Genome& parent1, const Genome& parent2, int mode);

/**
 * Batch cost-aware version
 * @param parents1 Vector of first parents
 * @param parents2 Vector of second parents
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Vector of child genomes
 */
std::vector<Genome> performBatchCostAwareEAXCrossover(const std::vector<Genome>& parents1,
                                                      const std::vector<Genome>& parents2, 
                                                      int mode);

// =============================================================================
// EAX IMPLEMENTATION FUNCTIONS (for routing chromosomes)
// =============================================================================

/**
 * Internal EAX implementation for single pair
 * @param parent1 First parent genome
 * @param parent2 Second parent genome
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Child genome with routing chromosomes crossed via EAX
 */
Genome performEAXCrossover(const Genome& parent1, const Genome& parent2, int mode);

// =============================================================================
// JOB ASSIGNMENT CROSSOVER FUNCTIONS
// =============================================================================

/**
 * GPU kernel for uniform crossover of job sequences
 * Each job position has 50% chance to inherit from each parent
 * @param parent1Jobs First parent's job sequence
 * @param parent2Jobs Second parent's job sequence
 * @param childJobs Output child's job sequence
 * @param numPairs Number of parent pairs being processed
 * @param jobLength Length of job sequences
 * @param seed Random seed for CUDA random number generation
 */
__global__ void uniformJobCrossoverKernel(const size_t* parent1Jobs, const size_t* parent2Jobs,
                                         size_t* childJobs, uint32_t numPairs, uint16_t jobLength,
                                         unsigned long seed);

// =============================================================================
// EAX GPU KERNELS (internal use)
// =============================================================================

/**
 * Extract edges from parent tours in parallel
 * @param tour1 First parent's tour
 * @param tour2 Second parent's tour
 * @param edges1 Output edges from first parent
 * @param edges2 Output edges from second parent
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void extractEdgesKernel(const size_t* tour1, const size_t* tour2,
                                   Edge* edges1, Edge* edges2, 
                                   uint16_t tourLength, uint32_t numPairs);

/**
 * Build union graph with adjacency matrix from extracted edges
 * @param edges1 Edges from first parent
 * @param edges2 Edges from second parent
 * @param adjacencyMatrix Output union graph adjacency matrix
 * @param degrees Output vertex degrees in union graph
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void buildUnionGraphKernel(const Edge* edges1, const Edge* edges2,
                                      uint8_t* adjacencyMatrix, uint16_t* degrees,
                                      uint16_t tourLength, uint32_t numPairs);

/**
 * Find alternating cycles in the union graph using parallel DFS
 * @param adjacencyMatrix Union graph adjacency matrix
 * @param degrees Vertex degrees
 * @param cycles Output detected cycles
 * @param numCycles Output number of cycles found
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void findAlternatingCyclesKernel(const uint8_t* adjacencyMatrix,
                                            const uint16_t* degrees,
                                            Cycle* cycles, uint16_t* numCycles,
                                            uint16_t tourLength, uint32_t numPairs);

/**
 * Evaluate all possible cycle assemblies and find the best one
 * @param cycles Available cycles
 * @param numCycles Number of cycles for each pair
 * @param costMatrix Travel cost matrix
 * @param assemblyCosts Output costs for best assemblies
 * @param bestAssemblies Output best assembly selections
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void evaluateAssembliesKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const float* costMatrix, float* assemblyCosts,
                                         uint32_t* bestAssemblies, uint16_t tourLength,
                                         uint32_t numPairs);

/**
 * Construct offspring tours from best cycle assemblies
 * @param cycles Available cycles
 * @param numCycles Number of cycles for each pair
 * @param bestAssemblies Best assembly selections
 * @param parent1Tours First parent tours
 * @param parent2Tours Second parent tours
 * @param childTours Output child tours
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void constructOffspringKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const uint32_t* bestAssemblies,
                                         const size_t* parent1Tours, const size_t* parent2Tours,
                                         size_t* childTours, uint16_t tourLength,
                                         uint32_t numPairs);

#endif // CROSSOVER_H