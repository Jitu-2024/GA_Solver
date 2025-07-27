// mutation.h: Header file for Fixed GPU-parallelized mutations in TSPJ

#ifndef MUTATION_H
#define MUTATION_H

#include "genome.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Maximum cities constraint for GPU memory allocation
#ifndef MAX_CITIES
#define MAX_CITIES 256
#endif

/**
 * Initialize the global cost matrix for GPU-based 2-opt mutations
 * @param travelTimes The travel time/cost matrix
 */
void initializeMutationCostMatrix(const std::vector<std::vector<float>>& travelTimes);

/**
 * Cleanup the global cost matrix
 */
void cleanupMutationCostMatrix();

/**
 * Perform mutation on a single genome (wrapper for batch mutation)
 * - Uses simplified GPU 2-opt mutation for routing sequences
 * - Uses GPU swap mutation for job assignment sequence
 * - Stagnation-aware escape mechanisms
 * 
 * @param genome The genome to mutate
 * @param mutationRate Base probability of applying mutation
 * @param mode 0=no pickup, 1=with pickup sequence
 * @param stagnationCount Number of generations without improvement
 */
void performMutation(Genome& genome, float mutationRate, int mode, size_t stagnationCount = 0);

/**
 * Fixed GPU-parallelized batch mutation (simplified and reliable)
 * Features:
 * - One thread per genome (no shared memory races)
 * - Simple 2-opt mutation for routing chromosomes
 * - Multi-swap mutation for job assignment
 * - Comprehensive error checking and debug output
 * 
 * @param genomes Vector of genomes to mutate (modified in-place)
 * @param mutationRate Base probability of applying mutation
 * @param mode 0=no pickup, 1=with pickup sequence
 * @param travelTimes Travel time matrix (not used in fixed version)
 * @param stagnationCount Number of generations without improvement
 */
void performBatchMutation(std::vector<Genome>& genomes, float mutationRate, int mode,
                         const std::vector<std::vector<float>>& travelTimes,
                         size_t stagnationCount = 0);

// Simplified GPU kernels (internal use)
__global__ void setupRandomStates(curandState* states, unsigned long seed, size_t n);

__global__ void simple2OptMutationKernel(size_t* sequences, size_t numGenomes, 
                                         size_t sequenceLength, float mutationRate,
                                         size_t stagnationCount, curandState* states);

__global__ void simpleSwapMutationKernel(size_t* sequences, size_t numGenomes,
                                         size_t sequenceLength, float mutationRate,
                                         size_t stagnationCount, curandState* states);

__global__ void scrambleMutationKernel(size_t* sequences, size_t numGenomes,
                                      size_t sequenceLength, float mutationRate,
                                      curandState* states);

// Simplified device utility functions
__device__ void apply2OptMove(size_t* tour, size_t tourLength, size_t i, size_t j);

#endif // MUTATION_H