#include "fitness_evaluator.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel for calculating fitness of genomes using the cost matrix
__global__ void evaluateFitnessKernel(
    const int* chromosomes, // Chromosome data for all genomes
    const float* costMatrix, // Cost matrix (flattened)
    float* fitnessValues, // Output fitness values
    int numCities, // Number of cities in the problem
    int populationSize // Number of genomes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        float fitness = 0.0f;
        const int* chromosome = chromosomes + idx * numCities;

        for (int i = 0; i < numCities - 1; ++i) {
            int from = chromosome[i];
            int to = chromosome[i + 1];
            fitness += costMatrix[from * numCities + to];
        }
        // Add the cost of returning to the starting city
        fitness += costMatrix[chromosome[numCities - 1] * numCities + chromosome[0]];

        fitnessValues[idx] = fitness; // Store the result
    }
}

// Evaluate the fitness of a population
void FitnessEvaluator::evaluatePopulation(
    std::vector<Genome>& genomes, 
    const float* deviceCostMatrix, 
    int numCities
) {
    int populationSize = genomes.size();
    size_t chromosomeSize = numCities * sizeof(int);
    size_t populationChromosomeSize = populationSize * chromosomeSize;
    size_t fitnessSize = populationSize * sizeof(float);

    // Allocate memory on device
    int* d_chromosomes;
    float* d_fitnessValues;

    cudaMalloc(&d_chromosomes, populationChromosomeSize);
    cudaMalloc(&d_fitnessValues, fitnessSize);

    // Prepare chromosome data
    std::vector<int> chromosomes(populationSize * numCities);
    for (int i = 0; i < populationSize; ++i) {
        const auto& chrom = genomes[i].getChromosome();
        std::copy(chrom.begin(), chrom.end(), chromosomes.begin() + i * numCities);
    }

    // Copy data to device
    cudaMemcpy(d_chromosomes, chromosomes.data(), populationChromosomeSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (populationSize + threadsPerBlock - 1) / threadsPerBlock;
    evaluateFitnessKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_chromosomes, 
        deviceCostMatrix, 
        d_fitnessValues, 
        numCities, 
        populationSize
    );

    // Copy results back to host
    std::vector<float> fitnessValues(populationSize);
    cudaMemcpy(fitnessValues.data(), d_fitnessValues, fitnessSize, cudaMemcpyDeviceToHost);

    // Update genomes with fitness values
    for (int i = 0; i < populationSize; ++i) {
        genomes[i].setFitness(fitnessValues[i]);
    }

    // Free device memory
    cudaFree(d_chromosomes);
    cudaFree(d_fitnessValues);
}


