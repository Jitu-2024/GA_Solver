#include "crossover.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n";  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// GPU Kernel for Order Crossover
__global__ void orderCrossoverKernel(const int* parents, int* offspring, const int* crossoverPoints, int numParents, int numCities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParents / 2) { // Each pair of parents produces two offspring
        int parent1Idx = idx * 2;
        int parent2Idx = parent1Idx + 1;

        const int* parent1 = &parents[parent1Idx * numCities];
        const int* parent2 = &parents[parent2Idx * numCities];
        int* child1 = &offspring[parent1Idx * numCities];
        int* child2 = &offspring[parent2Idx * numCities];

        int start = crossoverPoints[idx * 2];
        int end = crossoverPoints[idx * 2 + 1];

        // Initialize offspring with -1
        for (int i = 0; i < numCities; ++i) {
            child1[i] = -1;
            child2[i] = -1;
        }

        // Copy subsequence from parent1 to child1 and parent2 to child2
        for (int i = start; i <= end; ++i) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }

        // Fill remaining cities for child1
        int child1Idx = (end + 1) % numCities;
        for (int i = 0; i < numCities; ++i) {
            int city = parent2[(end + 1 + i) % numCities];
            bool isDuplicate = false;
            for (int j = start; j <= end; ++j) {
                if (child1[j] == city) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) {
                child1[child1Idx] = city;
                child1Idx = (child1Idx + 1) % numCities;
            }
        }

        // Fill remaining cities for child2
        int child2Idx = (end + 1) % numCities;
        for (int i = 0; i < numCities; ++i) {
            int city = parent1[(end + 1 + i) % numCities];
            bool isDuplicate = false;
            for (int j = start; j <= end; ++j) {
                if (child2[j] == city) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) {
                child2[child2Idx] = city;
                child2Idx = (child2Idx + 1) % numCities;
            }
        }
    }
}

// Crossover Constructor
Crossover::Crossover(int numCities)
    : numCities(numCities), d_parents(nullptr), d_offspring(nullptr), d_crossoverPoints(nullptr) {}

// Crossover Destructor
Crossover::~Crossover() {
    cudaFree(d_parents);
    cudaFree(d_offspring);
    cudaFree(d_crossoverPoints);
}

// Perform Order Crossover
void Crossover::orderCrossover(const std::vector<std::vector<int>>& parents,
                               std::vector<std::vector<int>>& offspring,
                               const std::vector<std::pair<int, int>>& crossoverPoints) {
    int numParents = parents.size();
    if (numParents % 2 != 0) {
        throw std::invalid_argument("Number of parents must be even.");
    }

    size_t parentSize = numParents * numCities;
    size_t crossoverPointsSize = crossoverPoints.size() * 2;

    // Flatten parent chromosomes
    std::vector<int> flattenedParents(parentSize);
    for (int i = 0; i < numParents; ++i) {
        std::copy(parents[i].begin(), parents[i].end(), &flattenedParents[i * numCities]);
    }

    // Flatten crossover points
    std::vector<int> flattenedCrossoverPoints(crossoverPointsSize);
    for (size_t i = 0; i < crossoverPoints.size(); ++i) {
        flattenedCrossoverPoints[i * 2] = crossoverPoints[i].first;
        flattenedCrossoverPoints[i * 2 + 1] = crossoverPoints[i].second;
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_parents, parentSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_offspring, parentSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_crossoverPoints, crossoverPointsSize * sizeof(int)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_parents, flattenedParents.data(), parentSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_crossoverPoints, flattenedCrossoverPoints.data(), crossoverPointsSize * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (numParents / 2 + threadsPerBlock - 1) / threadsPerBlock;
    orderCrossoverKernel<<<blocks, threadsPerBlock>>>(d_parents, d_offspring, d_crossoverPoints, numParents, numCities);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy offspring back to host
    std::vector<int> flattenedOffspring(parentSize);
    CUDA_CHECK(cudaMemcpy(flattenedOffspring.data(), d_offspring, parentSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Reshape offspring into 2D chromosomes
    offspring.clear();
    for (int i = 0; i < numParents; ++i) {
        offspring.emplace_back(flattenedOffspring.begin() + i * numCities,
                               flattenedOffspring.begin() + (i + 1) * numCities);
    }

    // Free GPU memory
    cudaFree(d_parents);
    cudaFree(d_offspring);
    cudaFree(d_crossoverPoints);
}
