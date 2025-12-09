// fitness_evaluator.cu: Fitness evaluation for TSPJ using flattened arrays

#include "fitness_evaluator.h"
#include "genome.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

// Maximum cities for fitness evaluation local arrays
#ifndef MAX_CITIES
#define MAX_CITIES 1200
#endif

// Kernel to evaluate fitness for a flattened population
__global__ void evaluateFitnessKernel(const size_t* flatArray, float* fitnessArray, size_t numGenomes,
                                       size_t chromosomeLength, const float* travelTimes, size_t numCities,
                                       const float* jobTimes, size_t numJobs, int mode) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (genomeIdx < numGenomes) {
        size_t startIdxCity = genomeIdx * chromosomeLength;
        size_t startIdxJob = numGenomes * chromosomeLength + genomeIdx * chromosomeLength;
        size_t startIdxPickup = numGenomes * chromosomeLength * 2 + genomeIdx * chromosomeLength;

        float maxCompletionTime = 0.0f;
        float currentTime = 0.0f;
        size_t prevCity = 0; // Start at depot

        // Array to store job completion times for each city
        // Use MAX_CITIES to support large problems
        float jobCompletionTimes[MAX_CITIES];
        // Initialize to 0 (only need to init up to numCities)
        for (size_t k = 0; k < numCities && k < MAX_CITIES; k++) {
            jobCompletionTimes[k] = 0.0f;
        }

        // Job starting phase
        for (size_t i = 0; i < chromosomeLength; ++i) {
            size_t city = flatArray[startIdxCity + i];
            size_t job = flatArray[startIdxJob + i];

            // Add travel time to the next city
            currentTime += travelTimes[prevCity * numCities + city];

            if (city > 0 && job > 0) { // Exclude depot and ensure valid job index
                // Compute job completion time and update maxCompletionTime
                // jobTimes layout: (city-1) * numJobs + (job-1) where city and job are 1-indexed
                size_t jobIdx = (city - 1) * numJobs + (job - 1);
                float jobCompletionTime = currentTime + jobTimes[jobIdx];
                jobCompletionTimes[city] = jobCompletionTime; // Record job completion time
                maxCompletionTime = fmaxf(maxCompletionTime, jobCompletionTime);
            }

            prevCity = city;
        }

        // Pickup phase (only if mode == 1)
        if (mode == 1) {
            for (size_t i = 0; i < chromosomeLength; ++i) {
                size_t pickupCity = flatArray[startIdxPickup + i];

                // Add travel time to the pickup city
                float travelTime = travelTimes[prevCity * numCities + pickupCity];
                currentTime += travelTime;

                // Check if we need to wait for job completion at the pickup city
                if (pickupCity > 0 && jobCompletionTimes[pickupCity] > currentTime) {
                    currentTime = jobCompletionTimes[pickupCity]; // Wait for job to finish
                }

                prevCity = pickupCity;
            }
        }

        // Add travel time back to the depot
        currentTime += travelTimes[prevCity * numCities + 0];

        // Final fitness is the max of maxCompletionTime and currentTime
        fitnessArray[genomeIdx] = fmaxf(maxCompletionTime, currentTime);
    }
}


// CPU implementation of fitness evaluation for debugging
void evaluateFitnessCPU(const std::vector<size_t>& flatArray, std::vector<float>& fitnessArray, size_t numGenomes,
                        size_t chromosomeLength, const std::vector<float>& travelTimes, size_t numCities,
                        const std::vector<float>& jobTimes, size_t numJobs, int mode) {
    for (size_t genomeIdx = 0; genomeIdx < numGenomes; ++genomeIdx) {
        size_t startIdxCity = genomeIdx * chromosomeLength;
        size_t startIdxJob = numGenomes * chromosomeLength + genomeIdx * chromosomeLength;
        size_t startIdxPickup = numGenomes * chromosomeLength * 2 + genomeIdx * chromosomeLength;

        float maxCompletionTime = 0.0f;
        float currentTime = 0.0f;
        size_t prevCity = 0; // Start at depot

        // Array to store job completion times for each city
        std::vector<float> jobCompletionTimes(numCities, 0.0f);

        // Job starting phase
        for (size_t i = 0; i < chromosomeLength; ++i) {
            size_t city = flatArray[startIdxCity + i];
            size_t job = flatArray[startIdxJob + i];

            // Add travel time to the next city
            currentTime += travelTimes[prevCity * numCities + city];

            if (city > 0 && job > 0) { // Exclude depot and ensure valid job index
                // Compute job completion time
                size_t jobIdx = (city - 1) * numJobs + (job - 1);
                float jobCompletionTime = currentTime + jobTimes[jobIdx];
                jobCompletionTimes[city] = jobCompletionTime; // Record job completion time
                maxCompletionTime = std::fmax(maxCompletionTime, jobCompletionTime);
            }

            prevCity = city;
        }

        // Pickup phase (only if mode == 1)
        if (mode == 1) {
            for (size_t i = 0; i < chromosomeLength; ++i) {
                size_t pickupCity = flatArray[startIdxPickup + i];
                float travelTime = travelTimes[prevCity * numCities + pickupCity];
                currentTime += travelTime;

                // Check if we need to wait for job completion at the pickup city
                if (pickupCity > 0 && jobCompletionTimes[pickupCity] > currentTime) {
                    currentTime = jobCompletionTimes[pickupCity]; // Wait for job to finish
                }
                std::cout << "Genome " << genomeIdx << ", Pickup City: " << pickupCity
                      << ", Travel Time: " << travelTime
                      << ", Current Time after Pickup: " << currentTime << std::endl;

                prevCity = pickupCity;
            }
        }

        // Add travel time back to the depot
        float travelTimeToDepot = travelTimes[prevCity * numCities + 0];
        currentTime += travelTimeToDepot;

        // Final fitness is the max of maxCompletionTime and currentTime
        fitnessArray[genomeIdx] = std::fmax(maxCompletionTime, currentTime);
    }
}

// Host function to flatten the population
void flattenPopulation(const std::vector<Genome>& population, std::vector<size_t>& flatArray,
                       std::vector<float>& fitnessArray, int mode) { // Added mode
    size_t numGenomes = population.size();
    size_t chromosomeLength = population[0].citySequence.size();

    // Adjust flatArray size based on mode
    size_t sequenceMultiplier = (mode == 1) ? 3 : 2; // Include pickup sequence only if mode == 1
    flatArray.resize(numGenomes * chromosomeLength * sequenceMultiplier); 
    fitnessArray.resize(numGenomes);

    for (size_t i = 0; i < numGenomes; ++i) {
        for (size_t j = 0; j < chromosomeLength; ++j) {
            flatArray[i * chromosomeLength + j] = population[i].citySequence[j];
            flatArray[numGenomes * chromosomeLength + i * chromosomeLength + j] = population[i].jobSequence[j];
            if (mode == 1) { // Include pickup sequence only for mode 1
                flatArray[numGenomes * chromosomeLength * 2 + i * chromosomeLength + j] = population[i].pickupSequence[j];
            }
        }
        fitnessArray[i] = population[i].fitness;
    }
}

// Host function to unflatten fitness data
void unflattenFitness(const std::vector<float>& fitnessArray, std::vector<Genome>& population) {
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].fitness = fitnessArray[i];
    }
}

// Host function to evaluate fitness for a population
void evaluatePopulationFitness(std::vector<Genome>& population,
                               const std::vector<std::vector<float>>& travelTimes,
                               const std::vector<std::vector<float>>& jobTimes, int mode) {
    size_t numGenomes = population.size();
    size_t chromosomeLength = population[0].citySequence.size();
    size_t numCities = travelTimes.size();
    // numJobs is the actual number of jobs (column 0 is placeholder with value 0)
    size_t numJobs = jobTimes[0].size() - 1;

    // Flatten travelTimes and jobTimes
    std::vector<float> flatTravelTimes(numCities * numCities);
    std::vector<float> flatJobTimes((numCities - 1) * numJobs); // Exclude depot for job times

    for (size_t i = 0; i < numCities; ++i) {
        for (size_t j = 0; j < numCities; ++j) {
            flatTravelTimes[i * numCities + j] = travelTimes[i][j];
        }
        if (i > 0) { // Skip depot row for job times
            for (size_t k = 0; k < numJobs; ++k) {
                // jobTimes[i][k+1] because column 0 is placeholder (always 0)
                // flatJobTimes index: (city-1) * numJobs + k maps to job (k+1)
                flatJobTimes[(i - 1) * numJobs + k] = jobTimes[i][k + 1];
            }
        }
    }

    // Flatten population
    std::vector<size_t> flatArray;
    std::vector<float> fitnessArray;
    flattenPopulation(population, flatArray, fitnessArray, mode); // Pass mode to flattenPopulation

    // Allocate device memory
    size_t* d_flatArray;
    float* d_fitnessArray;
    float* d_travelTimes;
    float* d_jobTimes;

    cudaMalloc(&d_flatArray, sizeof(size_t) * flatArray.size());
    cudaMalloc(&d_fitnessArray, sizeof(float) * fitnessArray.size());
    cudaMalloc(&d_travelTimes, sizeof(float) * flatTravelTimes.size());
    cudaMalloc(&d_jobTimes, sizeof(float) * flatJobTimes.size());

    // Copy data to device
    cudaMemcpy(d_flatArray, flatArray.data(), sizeof(size_t) * flatArray.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fitnessArray, fitnessArray.data(), sizeof(float) * fitnessArray.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_travelTimes, flatTravelTimes.data(), sizeof(float) * flatTravelTimes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jobTimes, flatJobTimes.data(), sizeof(float) * flatJobTimes.size(), cudaMemcpyHostToDevice);

    // evaluateFitnessCPU(flatArray, fitnessArray, numGenomes, chromosomeLength, flatTravelTimes, numCities, flatJobTimes, numJobs, mode);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;
    evaluateFitnessKernel<<<blocksPerGrid, threadsPerBlock>>>(d_flatArray, d_fitnessArray, numGenomes, chromosomeLength,
                                                              d_travelTimes, numCities, d_jobTimes, numJobs, mode);

    // Copy fitness results back to host
    cudaMemcpy(fitnessArray.data(), d_fitnessArray, sizeof(float) * fitnessArray.size(), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flatArray);
    cudaFree(d_fitnessArray);
    cudaFree(d_travelTimes);
    cudaFree(d_jobTimes);

    // Update population fitness
    unflattenFitness(fitnessArray, population);
}
