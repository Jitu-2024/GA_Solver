#include "mutation.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <stdexcept>

// CUDA kernel for performing swap mutation
__global__ void swapMutationKernel(int* d_population, int numGenomes, int numCities, float mutationRate, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numGenomes) {
        curandState localState = states[idx];
        float randomValue = curand_uniform(&localState);

        // Perform mutation based on mutationRate
        if (randomValue < mutationRate) {
            // Randomly select two indices in the chromosome to swap
            int city1 = curand(&localState) % numCities;
            int city2 = curand(&localState) % numCities;

            // Ensure city1 and city2 are different
            while (city1 == city2) {
                city2 = curand(&localState) % numCities;
            }

            // Swap cities in the chromosome
            int temp = d_population[idx * numCities + city1];
            d_population[idx * numCities + city1] = d_population[idx * numCities + city2];
            d_population[idx * numCities + city2] = temp;
        }

        states[idx] = localState; // Save back the state
    }
}

// Initialize CUDA random states
__global__ void initializeRandomStates(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Constructor
Mutation::Mutation(int numCities, float mutationRate)
    : numCities(numCities), mutationRate(mutationRate), d_population(nullptr), d_mutatedPopulation(nullptr) {}

// Destructor
Mutation::~Mutation() {
    if (d_population) cudaFree(d_population);
}

// Apply mutation using GPU
void Mutation::applyMutation(std::vector<std::vector<int>>& population) {
    int numGenomes = population.size();

    // Flatten the population into a 1D array for GPU processing
    std::vector<int> flatPopulation(numGenomes * numCities);
    for (int i = 0; i < numGenomes; ++i) {
        std::copy(population[i].begin(), population[i].end(), flatPopulation.begin() + i * numCities);
    }

    // Allocate memory on GPU
    cudaMalloc(&d_population, flatPopulation.size() * sizeof(int));
    cudaMemcpy(d_population, flatPopulation.data(), flatPopulation.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate random states for each genome
    curandState* d_states;
    cudaMalloc(&d_states, numGenomes * sizeof(curandState));
    initializeRandomStates<<<(numGenomes + 255) / 256, 256>>>(d_states, time(nullptr), numGenomes);

    // Launch kernel to perform swap mutation
    swapMutationKernel<<<(numGenomes + 255) / 256, 256>>>(d_population, numGenomes, numCities, mutationRate, d_states);

    // Copy back mutated population to host
    cudaMemcpy(flatPopulation.data(), d_population, flatPopulation.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Reshape flattened population back to 2D vector
    for (int i = 0; i < numGenomes; ++i) {
        std::copy(flatPopulation.begin() + i * numCities,
                  flatPopulation.begin() + (i + 1) * numCities,
                  population[i].begin());
    }

    // Free GPU memory
    cudaFree(d_population);
    cudaFree(d_states);
}
