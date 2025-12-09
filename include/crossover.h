// crossover.h: Header file for OX Crossover + Uniform Job Crossover for TSPJ

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "genome.h"
#include <vector>
#include <random>

// Maximum problem size constraints
#ifndef MAX_CITIES
#define MAX_CITIES 1200
#endif

// =============================================================================
// MAIN CROSSOVER INTERFACE FUNCTIONS
// =============================================================================

/**
 * Main crossover function - uses OX for routing + uniform for job assignment
 * @param parent1 First parent genome
 * @param parent2 Second parent genome
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Child genome created by hybrid crossover
 */
Genome performCrossover(const Genome& parent1, const Genome& parent2, int mode);

/**
 * Batch processing version for multiple parent pairs (CPU)
 * @param parents1 Vector of first parents
 * @param parents2 Vector of second parents
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Vector of child genomes
 */
std::vector<Genome> performBatchCrossover(const std::vector<Genome>& parents1,
                                          const std::vector<Genome>& parents2,
                                          int mode);

/**
 * Batch processing version using GPU acceleration
 * @param parents1 Vector of first parents
 * @param parents2 Vector of second parents
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Vector of child genomes
 */
std::vector<Genome> performBatchGPUCrossover(const std::vector<Genome>& parents1,
                                              const std::vector<Genome>& parents2,
                                              int mode);

// =============================================================================
// LEGACY INTERFACE FUNCTIONS (for compatibility with existing code)
// =============================================================================

// These functions now use OX crossover internally instead of broken EAX
Genome performEAXCrossover(const Genome& parent1, const Genome& parent2, int mode);

std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                             const std::vector<Genome>& parents2,
                                             int mode);

Genome performCostAwareEAXCrossover(const Genome& parent1, const Genome& parent2, int mode);

std::vector<Genome> performBatchCostAwareEAXCrossover(const std::vector<Genome>& parents1,
                                                       const std::vector<Genome>& parents2,
                                                       int mode);

// =============================================================================
// CPU CROSSOVER IMPLEMENTATIONS
// =============================================================================

/**
 * Order Crossover (OX) for city/tour sequences
 * Preserves tour validity by ensuring each city appears exactly once
 */
void orderCrossover(const std::vector<size_t>& parent1, const std::vector<size_t>& parent2,
                    std::vector<size_t>& child, std::mt19937& rng);

/**
 * Uniform crossover for job sequences
 * Each position has 50% chance to inherit from each parent
 */
void uniformCrossover(const std::vector<size_t>& parent1, const std::vector<size_t>& parent2,
                      std::vector<size_t>& child, std::mt19937& rng);

// =============================================================================
// GPU KERNELS
// =============================================================================

/**
 * GPU kernel for OX crossover on tour sequences
 */
__global__ void oxCrossoverKernel(const size_t* parent1Tours, const size_t* parent2Tours,
                                  size_t* childTours, uint32_t numPairs, uint16_t tourLength,
                                  unsigned long seed);

/**
 * GPU kernel for uniform crossover on job sequences
 */
__global__ void uniformJobCrossoverKernel(const size_t* parent1Jobs, const size_t* parent2Jobs,
                                          size_t* childJobs, uint32_t numPairs, uint16_t jobLength,
                                          unsigned long seed);

#endif // CROSSOVER_H
