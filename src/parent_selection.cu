#include <cuda_runtime.h>
#include "parent_selection.h"
#include "genome.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand_kernel.h>
#include <algorithm>
#include <stdexcept>

__global__ void setupRandStates(curandState* states, unsigned int seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void performTournament(
    const float* fitness,
    int* selectedIndices,
    curandState* randStates,
    int populationSize,
    int tournamentSize,
    int numParents
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParents) {
        int bestIndex = -1;
        float bestFitness = 1e9f;
        
        for (int i = 0; i < tournamentSize; ++i) {
            int competitor = curand(&randStates[idx]) % populationSize;
            if (fitness[competitor] < bestFitness) { // Select competitor with lower fitness
                bestFitness = fitness[competitor];
                bestIndex = competitor;
            }
        }
        selectedIndices[idx] = bestIndex;
    }
}

ParentSelection::ParentSelection(const std::vector<Genome>& population)
    : populationSize(population.size()) {
    if (populationSize == 0) {
        throw std::invalid_argument("Population cannot be empty");
    }

    // Allocate memory for device fitness array
    cudaMalloc(&d_fitness, populationSize * sizeof(float));

    // Copy fitness values to device
    std::vector<float> fitness;
    for (const auto& genome : population) {
        fitness.push_back(genome.getFitness());
    }
    cudaMemcpy(d_fitness, fitness.data(), populationSize * sizeof(float), cudaMemcpyHostToDevice);
}

ParentSelection::~ParentSelection() {
    cudaFree(d_fitness);
}

void ParentSelection::tournamentSelection(
    const std::vector<Genome>& population,
    std::vector<Genome>& parents,
    int numParents,
    int tournamentSize
) {
    if (numParents <= 0 || tournamentSize <= 0) {
        throw std::invalid_argument("Number of parents and tournament size must be positive");
    }

    curandState* d_randStates;
    int* d_selectedIndices;

    // Allocate device memory for random states and selected indices
    cudaMalloc(&d_randStates, numParents * sizeof(curandState));
    cudaMalloc(&d_selectedIndices, numParents * sizeof(int));

    // Setup random states
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParents + threadsPerBlock - 1) / threadsPerBlock;
    setupRandStates<<<blocksPerGrid, threadsPerBlock>>>(d_randStates, time(NULL), numParents);

    // Perform tournament selection on GPU
    performTournament<<<blocksPerGrid, threadsPerBlock>>>(
        d_fitness,
        d_selectedIndices,
        d_randStates,
        populationSize,
        tournamentSize,
        numParents
    );

    // Copy selected indices back to host
    std::vector<int> selectedIndices(numParents);
    cudaMemcpy(selectedIndices.data(), d_selectedIndices, numParents * sizeof(int), cudaMemcpyDeviceToHost);

    // Populate parents vector
    parents.clear();
    for (int idx : selectedIndices) {
        parents.push_back(population[idx]);
    }

    // Free device memory
    cudaFree(d_randStates);
    cudaFree(d_selectedIndices);
}
