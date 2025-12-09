// eax_cost_integration.cu: Cost matrix management for crossover and fitness evaluation
// Note: EAX was broken and replaced with OX crossover - this file now just manages cost matrices

#include "eax_cost_integration.h"
#include "crossover.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

// Global device cost matrices (managed internally)
static thrust::device_vector<float>* g_d_travelCosts = nullptr;
static thrust::device_vector<float>* g_d_jobCosts = nullptr;
static size_t g_numCities = 0;
static size_t g_numJobs = 0;

void initializeEAXCostMatrices(const std::vector<std::vector<float>>& travelTimes,
                               const std::vector<std::vector<float>>& jobTimes) {
    // Clean up existing matrices
    cleanupEAXCostMatrices();

    g_numCities = travelTimes.size();
    g_numJobs = jobTimes[0].size();

    // Flatten travel times matrix
    std::vector<float> flatTravelTimes(g_numCities * g_numCities);
    for (size_t i = 0; i < g_numCities; i++) {
        for (size_t j = 0; j < g_numCities; j++) {
            flatTravelTimes[i * g_numCities + j] = travelTimes[i][j];
        }
    }

    // Flatten job times matrix
    std::vector<float> flatJobTimes(g_numCities * g_numJobs);
    for (size_t i = 0; i < g_numCities; i++) {
        for (size_t j = 0; j < g_numJobs; j++) {
            flatJobTimes[i * g_numJobs + j] = jobTimes[i][j];
        }
    }

    // Allocate and copy to device
    g_d_travelCosts = new thrust::device_vector<float>(flatTravelTimes);
    g_d_jobCosts = new thrust::device_vector<float>(flatJobTimes);

    std::cout << "Cost matrices initialized: "
              << g_numCities << " cities, " << g_numJobs << " jobs" << std::endl;
}

void cleanupEAXCostMatrices() {
    delete g_d_travelCosts;
    delete g_d_jobCosts;
    g_d_travelCosts = nullptr;
    g_d_jobCosts = nullptr;
}

// Cost-aware crossover internal implementations
// These now just delegate to the OX-based crossover since EAX was broken

Genome performCostAwareEAXCrossoverInternal(const Genome& parent1, const Genome& parent2, int mode) {
    // Use the working OX crossover
    return performCrossover(parent1, parent2, mode);
}

std::vector<Genome> performBatchCostAwareEAXCrossoverInternal(const std::vector<Genome>& parents1,
                                                              const std::vector<Genome>& parents2,
                                                              int mode) {
    // Use the working batch OX crossover
    return performBatchGPUCrossover(parents1, parents2, mode);
}

// Get raw pointers to cost matrices (for other kernels if needed)
const float* getDeviceTravelCosts() {
    return g_d_travelCosts ? thrust::raw_pointer_cast(g_d_travelCosts->data()) : nullptr;
}

const float* getDeviceJobCosts() {
    return g_d_jobCosts ? thrust::raw_pointer_cast(g_d_jobCosts->data()) : nullptr;
}

size_t getNumCities() {
    return g_numCities;
}

size_t getNumJobs() {
    return g_numJobs;
}
