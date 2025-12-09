// eax_cost_integration.h: Cost matrix management for GA operations

#ifndef EAX_COST_INTEGRATION_H
#define EAX_COST_INTEGRATION_H

#include "genome.h"
#include <vector>

// =============================================================================
// COST MATRIX MANAGEMENT
// =============================================================================

/**
 * Initialize cost matrices on GPU for cost-aware operations
 * Should be called once at the start of the GA
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
// COST-AWARE CROSSOVER FUNCTIONS (Internal implementations)
// =============================================================================

/**
 * Internal cost-aware crossover implementation for single pair
 * Note: Now delegates to OX crossover since EAX was broken
 */
Genome performCostAwareEAXCrossoverInternal(const Genome& parent1, const Genome& parent2, int mode);

/**
 * Internal batch cost-aware crossover implementation
 * Note: Now delegates to batch OX crossover since EAX was broken
 */
std::vector<Genome> performBatchCostAwareEAXCrossoverInternal(const std::vector<Genome>& parents1,
                                                              const std::vector<Genome>& parents2,
                                                              int mode);

// =============================================================================
// COST MATRIX ACCESS (for other kernels)
// =============================================================================

/**
 * Get pointer to device travel costs matrix
 */
const float* getDeviceTravelCosts();

/**
 * Get pointer to device job costs matrix
 */
const float* getDeviceJobCosts();

/**
 * Get number of cities in cost matrices
 */
size_t getNumCities();

/**
 * Get number of jobs in cost matrices
 */
size_t getNumJobs();

#endif // EAX_COST_INTEGRATION_H
