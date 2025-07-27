// eax_cost_integration.h: Integration of cost matrix with EAX crossover (Updated for Hybrid)

#ifndef EAX_COST_INTEGRATION_H
#define EAX_COST_INTEGRATION_H

#include "genome.h"
#include "crossover.h"  // Include crossover.h to get Cycle definition and main interface
#include <vector>

// =============================================================================
// COST MATRIX MANAGEMENT
// =============================================================================

/**
 * Initialize EAX with proper cost matrices for cost-aware evaluation
 * Should be called once at the start of the GA if using cost-aware mode
 * @param travelTimes Travel time/cost matrix between cities
 * @param jobTimes Job execution time matrix for each city-job combination
 */
void initializeEAXCostMatrices(const std::vector<std::vector<float>>& travelTimes,
                               const std::vector<std::vector<float>>& jobTimes);

/**
 * Cleanup cost matrices - should be called at end of GA or on error
 */
void cleanupEAXCostMatrices();

// =============================================================================
// COST-AWARE CROSSOVER FUNCTIONS
// =============================================================================

// Note: The main cost-aware functions are now declared in crossover.h
// These are kept here for backward compatibility and internal implementation

/**
 * Internal cost-aware EAX implementation for single pair
 * Uses actual travel and job costs for assembly evaluation
 * @param parent1 First parent genome
 * @param parent2 Second parent genome
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Child genome
 */
Genome performCostAwareEAXCrossoverInternal(const Genome& parent1, const Genome& parent2, int mode);

/**
 * Internal batch cost-aware EAX implementation
 * @param parents1 Vector of first parents
 * @param parents2 Vector of second parents
 * @param mode 0=no pickup, 1=sequential, 2=interleaved
 * @return Vector of child genomes
 */
std::vector<Genome> performBatchCostAwareEAXCrossoverInternal(const std::vector<Genome>& parents1,
                                                              const std::vector<Genome>& parents2, 
                                                              int mode);

// =============================================================================
// COST-AWARE GPU KERNELS (internal use)
// =============================================================================

/**
 * Calculate total tour cost using actual travel cost matrix
 * @param tour Tour sequence
 * @param travelCosts Flattened travel cost matrix
 * @param tourLength Length of tour
 * @param totalCost Output total cost
 */
__global__ void calculateTourCostKernel(const size_t* tour, const float* travelCosts, 
                                        size_t tourLength, float* totalCost);

/**
 * Enhanced assembly evaluation kernel using actual travel and job costs
 * Replaces simple cost estimation with real cost calculation
 * @param cycles Available cycles from EAX
 * @param numCycles Number of cycles for each pair
 * @param travelCosts Travel cost matrix (flattened)
 * @param jobCosts Job execution cost matrix (flattened)
 * @param assemblyCosts Output costs for best assemblies
 * @param bestAssemblies Output best assembly selections
 * @param tourLength Length of tours
 * @param numPairs Number of parent pairs
 */
__global__ void evaluateAssembliesWithCostsKernel(const Cycle* cycles, const uint16_t* numCycles,
                                                   const float* travelCosts, const float* jobCosts,
                                                   float* assemblyCosts, uint32_t* bestAssemblies,
                                                   uint16_t tourLength, uint32_t numPairs);

/**
 * Job-cost aware uniform crossover kernel
 * Could incorporate job-city compatibility costs in the future
 * @param parent1Jobs First parent's job sequence
 * @param parent2Jobs Second parent's job sequence
 * @param childJobs Output child's job sequence
 * @param jobCosts Job execution costs (for future cost-aware job selection)
 * @param numPairs Number of parent pairs
 * @param jobLength Length of job sequences
 * @param seed Random seed
 */
__global__ void costAwareUniformJobCrossoverKernel(const size_t* parent1Jobs, const size_t* parent2Jobs,
                                                   size_t* childJobs, const float* jobCosts,
                                                   uint32_t numPairs, uint16_t jobLength,
                                                   unsigned long seed);

#endif // EAX_COST_INTEGRATION_H