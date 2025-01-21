// fitness_evaluator.cu: Fitness evaluation for TSPJ using flattened arrays

#include "fitness_evaluator.h"
#include "genome.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

__global__ void computeCombinedSequenceKernel(const size_t* cityArray, const size_t* pickupOffsets,
                                              size_t* combinedArray, size_t numGenomes, size_t chromosomeLength) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (genomeIdx < numGenomes) {
        size_t startIdxCity = genomeIdx * chromosomeLength;
        size_t startIdxCombined = genomeIdx * 2 * chromosomeLength;

        // Initialize combined sequence with zeros
        for (size_t i = 0; i < 2 * chromosomeLength; ++i) {
            combinedArray[startIdxCombined + i] = 0;
        }

        // Place dropoffs and pickups
        for (size_t i = 0; i < chromosomeLength; ++i) {
            size_t dropoffCity = cityArray[startIdxCity + i];
            size_t pickupCity = dropoffCity + chromosomeLength;

            // 1. Dropoff Placement
            size_t dropoffIdx = i;
            while (dropoffIdx < 2 * chromosomeLength && combinedArray[startIdxCombined + dropoffIdx] != 0) {
                dropoffIdx++; // Move to the next available space
            }
            if (dropoffIdx < 2 * chromosomeLength) {
                combinedArray[startIdxCombined + dropoffIdx] = dropoffCity;
            } else {
                printf("Error: No space for dropoff city %lu in genome %d\n", dropoffCity, genomeIdx);
            }

            // 2. Pickup Placement
            size_t intendedPickupIdx = i + pickupOffsets[startIdxCity + i] + 1;
            size_t pickupIdx = intendedPickupIdx;

            // Find the next available space for the pickup
            while (pickupIdx < 2 * chromosomeLength && combinedArray[startIdxCombined + pickupIdx] != 0) {
                pickupIdx++;
            }

            if (pickupIdx < 2 * chromosomeLength) {
                combinedArray[startIdxCombined + pickupIdx] = pickupCity;
            } else {
                printf("Error: No space for pickup city %lu in genome %d\n", pickupCity, genomeIdx);
            }
        }
    }
}



// Kernel to evaluate fitness for a flattened population
__global__ void evaluateFitnessKernel(const size_t* flatArray, float* fitnessArray, size_t numGenomes,
                                      size_t chromosomeLength, const float* travelTimes, size_t numCities,
                                      const float* jobTimes, size_t numJobs, int mode) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (genomeIdx < numGenomes) {
        // Determine starting indices based on mode
        size_t startIdxCombined = (mode == 2) ? genomeIdx * (2 * chromosomeLength + chromosomeLength) : 0;
        size_t startIdxCity = (mode != 2) ? genomeIdx * chromosomeLength : 0;
        size_t startIdxJob = (mode == 2) ? (genomeIdx * (2 * chromosomeLength + chromosomeLength) + 2 * chromosomeLength)
                                         : (numGenomes * chromosomeLength + genomeIdx * chromosomeLength);

        float maxCompletionTime = 0.0f;
        float currentTime = 0.0f;
        size_t prevCity = 0; // Start at depot

        // Array to store job completion times for each city
        float jobCompletionTimes[256] = {0.0f};

        if (mode == 0 || mode == 1) {
            // Modes 0 and 1: Process dropoffs
            for (size_t i = 0; i < chromosomeLength; ++i) {
                size_t city = flatArray[startIdxCity + i];
                size_t job = flatArray[startIdxJob + i];

                if (job == 0 || city == 0 || job > numJobs) {
                    // Invalid job or city, skip this entry
                    continue;
                }

                // Add travel time to the next city
                float travelTime = travelTimes[prevCity * numCities + city];
                currentTime += travelTime;

                // Compute job completion time
                float jobCompletionTime = currentTime + jobTimes[(city - 1) * numJobs + (job - 1)];
                jobCompletionTimes[city] = jobCompletionTime;
                maxCompletionTime = fmaxf(maxCompletionTime, jobCompletionTime);

                prevCity = city; // Update previous city
            }

            // Mode 1: Process pickups
            if (mode == 1) {
                for (size_t i = 0; i < chromosomeLength; ++i) {
                    size_t pickupCity = flatArray[startIdxCity + i];

                    // Add travel time to the pickup city
                    float travelTime = travelTimes[prevCity * numCities + pickupCity];
                    currentTime += travelTime;

                    // Wait for job completion if needed
                    if (pickupCity > 0 && jobCompletionTimes[pickupCity] > currentTime) {
                        currentTime = jobCompletionTimes[pickupCity];
                    }

                    prevCity = pickupCity; // Update previous city
                }
            }
        } else if (mode == 2) {
            // Mode 2: Process combined sequence and job sequence
            size_t dropoffCounter = 0; // Counter for assigning jobs to dropoff cities

            for (size_t i = 0; i < 2 * chromosomeLength; ++i) {
                size_t city = flatArray[startIdxCombined + i];

                // Add travel time to the city (dropoff or pickup)
                size_t effectiveCity = (city < numCities ? city : city - numCities + 1);
                float travelTime = travelTimes[prevCity * numCities + effectiveCity];
                currentTime += travelTime;

                if (city < numCities) {
                    // Dropoff phase: Start the job
                    if (dropoffCounter < chromosomeLength) {
                        size_t job = flatArray[startIdxJob + dropoffCounter];
                        dropoffCounter++;

                        if (job > 0 && job <= numJobs) {
                            float jobCompletionTime = currentTime + jobTimes[(city - 1) * numJobs + (job - 1)];
                            jobCompletionTimes[effectiveCity] = jobCompletionTime;
                            maxCompletionTime = fmaxf(maxCompletionTime, jobCompletionTime);
                        }
                    }
                } else {
                    // Pickup phase: Wait for the corresponding dropoff job to complete
                    size_t dropoffCity = city - numCities + 1;
                    if (jobCompletionTimes[dropoffCity] > currentTime) {
                        currentTime = jobCompletionTimes[dropoffCity];
                    }
                }

                prevCity = effectiveCity; // Update previous city
            }
        }

        // Add travel time back to the depot
        float travelTimeToDepot = travelTimes[prevCity * numCities + 0];
        currentTime += travelTimeToDepot;

        // Final fitness is the max of maxCompletionTime and currentTime
        fitnessArray[genomeIdx] = fmaxf(maxCompletionTime, currentTime);
    }
}


// CPU implementation of fitness evaluation for debugging
void evaluateFitnessCPU(const std::vector<size_t>& flatArray, std::vector<float>& fitnessArray, size_t numGenomes,
                        size_t chromosomeLength, const std::vector<float>& travelTimes, size_t numCities,
                        const std::vector<float>& jobTimes, size_t numJobs, int mode) {
    for (size_t genomeIdx = 0; genomeIdx < numGenomes; ++genomeIdx) {
        // Determine starting indices based on mode
        size_t startIdxCombined = (mode == 2) ? genomeIdx * (2 * chromosomeLength + chromosomeLength) : 0;
        size_t startIdxCity = (mode != 2) ? genomeIdx * chromosomeLength : 0;
        size_t startIdxJob = (mode == 2) ? (genomeIdx * (2 * chromosomeLength + chromosomeLength) + 2 * chromosomeLength) : (numGenomes * chromosomeLength + genomeIdx * chromosomeLength);

        float maxCompletionTime = 0.0f;
        float currentTime = 0.0f;
        size_t prevCity = 0; // Start at depot

        // Array to store job completion times for each city
        std::vector<float> jobCompletionTimes(numCities, 0.0f);

        if (mode == 0 || mode == 1) {
            // Modes 0 and 1: Process dropoffs using city and job sequences
            for (size_t i = 0; i < chromosomeLength; ++i) {
                size_t city = flatArray[startIdxCity + i];
                size_t job = flatArray[startIdxJob + i];

                // Validate job
                if (job == 0 || city == 0 || job > numJobs) {
                    std::cerr << "Error: Invalid job " << job << " or city " << city << " for genome " << genomeIdx << std::endl;
                    continue;
                }

                // Add travel time to the next city
                float travelTime = travelTimes[prevCity * numCities + city];
                currentTime += travelTime;

                // Compute job completion time and update maxCompletionTime
                float jobCompletionTime = currentTime + jobTimes[(city - 1) * numJobs + (job - 1)];
                jobCompletionTimes[city] = jobCompletionTime;
                maxCompletionTime = std::fmax(maxCompletionTime, jobCompletionTime);

                // Debugging output
                std::cout << "[Genome " << genomeIdx << "] Dropoff: City = " << city << ", Job = " << job
                          << ", Travel Time = " << travelTime
                          << ", Current Time = " << currentTime
                          << ", Job Completion Time = " << jobCompletionTime
                          << ", Max Completion Time = " << maxCompletionTime << std::endl;

                prevCity = city; // Update previous city
            }

            // Process pickups only for mode 1
            if (mode == 1) {
                for (size_t i = 0; i < chromosomeLength; ++i) {
                    size_t pickupCity = flatArray[startIdxCity + i];

                    // Add travel time to the pickup city
                    float travelTime = travelTimes[prevCity * numCities + pickupCity];
                    currentTime += travelTime;

                    // Wait for job completion if needed
                    if (pickupCity > 0 && jobCompletionTimes[pickupCity] > currentTime) {
                        currentTime = jobCompletionTimes[pickupCity];
                    }

                    // Debugging output
                    std::cout << "[Genome " << genomeIdx << "] Pickup: City = " << pickupCity
                              << ", Travel Time = " << travelTime
                              << ", Current Time after Pickup = " << currentTime << std::endl;

                    prevCity = pickupCity; // Update previous city
                }
            }
        } else if (mode == 2) {
            // Mode 2: Process combined sequence and job sequence
            size_t dropoffCounter = 0; // Counter for assigning jobs to dropoff cities
            std::cout << "Genome " << genomeIdx << " - Combined Sequence: ";
            for (size_t j = 0; j < 2 * chromosomeLength; ++j) {
                std::cout << flatArray[genomeIdx * (2 * chromosomeLength + chromosomeLength) + j] << " ";
            }
            std::cout << "\nGenome " << genomeIdx << " - Job Sequence: ";
            for (size_t j = 0; j < chromosomeLength; ++j) {
            std::cout << flatArray[genomeIdx * (2 * chromosomeLength + chromosomeLength) + 2 * chromosomeLength + j] << " ";
        }
            std::cout << "\n";

            for (size_t i = 0; i < 2 * chromosomeLength; ++i) {

                size_t city = flatArray[startIdxCombined + i];

                // Add travel time to the city (dropoff or pickup)
                float travelTime = travelTimes[prevCity * numCities + (city < numCities ? city : city - numCities + 1)];
                currentTime += travelTime;

                if (city < numCities) {
                    // Dropoff phase: Start the job
                    if (dropoffCounter >= chromosomeLength) {
                        std::cerr << "Error: Dropoff counter exceeds chromosome length for genome " << genomeIdx << std::endl;
                        continue;
                    }

                    size_t job = flatArray[startIdxJob + dropoffCounter];
                    dropoffCounter++;

                    if (job > 0 && job <= numJobs) {
                        float jobCompletionTime = currentTime + jobTimes[(city - 1) * numJobs + (job - 1)];
                        jobCompletionTimes[city] = jobCompletionTime; // Record job completion time
                        maxCompletionTime = std::fmax(maxCompletionTime, jobCompletionTime);

                        // Debugging output
                        std::cout << "[Genome " << genomeIdx << "] Dropoff: City = " << city << ", Job = " << job
                                  << ", Travel Time = " << travelTime
                                  << ", Current Time = " << currentTime
                                  << ", Job Completion Time = " << jobCompletionTime
                                  << ", Max Completion Time = " << maxCompletionTime << std::endl;
                    } else {
                        std::cerr << "Error: Invalid job " << job << " for genome " << genomeIdx << std::endl;
                    }
                } else {
                    // Pickup phase: Wait for the corresponding dropoff job to complete
                    size_t dropoffCity = city - numCities + 1;
                    if (jobCompletionTimes[dropoffCity] > currentTime) {
                        currentTime = jobCompletionTimes[dropoffCity]; // Synchronize with dropoff completion
                    }

                    // Debugging output
                    std::cout << "[Genome " << genomeIdx << "] Pickup: City = " << dropoffCity
                              << ", Travel Time = " << travelTime
                              << ", Current Time after Pickup = " << currentTime << std::endl;
                }

                prevCity = (city < numCities) ? city : city - numCities + 1; // Update previous city
            }
        }

        // Add travel time back to the depot
        float travelTimeToDepot = travelTimes[prevCity * numCities + 0];
        currentTime += travelTimeToDepot;

        // Final fitness is the max of maxCompletionTime and currentTime
        fitnessArray[genomeIdx] = std::fmax(maxCompletionTime, currentTime);

        // Debugging output for final fitness
        std::cout << "[Genome " << genomeIdx << "] Final Fitness: Max Completion Time = " << maxCompletionTime
                  << ", Current Time = " << currentTime
                  << ", Fitness = " << fitnessArray[genomeIdx] << std::endl;
    }
}



void flattenPopulationGPU(const std::vector<Genome>& population, std::vector<size_t>& flatArray,
                          size_t numGenomes, size_t chromosomeLength) {
    size_t* d_cityArray;
    size_t* d_pickupOffsets;
    size_t* d_combinedArray;

    // Allocate device memory
    cudaMalloc(&d_cityArray, sizeof(size_t) * numGenomes * chromosomeLength);
    cudaMalloc(&d_pickupOffsets, sizeof(size_t) * numGenomes * chromosomeLength);
    cudaMalloc(&d_combinedArray, sizeof(size_t) * numGenomes * 2 * chromosomeLength);

    // Prepare host arrays for citySequence and pickupOffsets
    std::vector<size_t> cityArray(numGenomes * chromosomeLength);
    std::vector<size_t> pickupOffsets(numGenomes * chromosomeLength);

    for (size_t i = 0; i < numGenomes; ++i) {
        for (size_t j = 0; j < chromosomeLength; ++j) {
            cityArray[i * chromosomeLength + j] = population[i].citySequence[j];
            pickupOffsets[i * chromosomeLength + j] = population[i].pickupOffset[j];
        }
    }

    // Copy data to the device
    cudaMemcpy(d_cityArray, cityArray.data(), sizeof(size_t) * numGenomes * chromosomeLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pickupOffsets, pickupOffsets.data(), sizeof(size_t) * numGenomes * chromosomeLength, cudaMemcpyHostToDevice);
    cudaMemset(d_combinedArray, 0, sizeof(size_t) * numGenomes * 2 * chromosomeLength);

    // Launch kernel to compute combined sequence
    int threadsPerBlock = 256;
    int blocksPerGrid = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;
    computeCombinedSequenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_cityArray, d_pickupOffsets,
                                                                      d_combinedArray, numGenomes, chromosomeLength);

    // Copy the combined sequence back to the host
    std::vector<size_t> combinedArray(numGenomes * 2 * chromosomeLength);
    cudaMemcpy(combinedArray.data(), d_combinedArray, sizeof(size_t) * numGenomes * 2 * chromosomeLength, cudaMemcpyDeviceToHost);

    // Flatten jobSequence and combinedSequence into flatArray
    flatArray.resize(numGenomes * (2 * chromosomeLength + chromosomeLength));
    for (size_t i = 0; i < numGenomes; ++i) {
        // Copy combined sequence
        for (size_t j = 0; j < 2 * chromosomeLength; ++j) {
            flatArray[i * (2 * chromosomeLength + chromosomeLength) + j] = combinedArray[i * 2 * chromosomeLength + j];
        }
        // Copy job sequence
        for (size_t j = 0; j < chromosomeLength; ++j) {
            flatArray[i * (2 * chromosomeLength + chromosomeLength) + 2 * chromosomeLength + j] = population[i].jobSequence[j];
        }
    }

    // Debugging: Print all genomes
    // for (size_t i = 0; i < numGenomes; ++i) {
    //     std::cout << "Genome " << i << " - Combined Sequence: ";
    //     for (size_t j = 0; j < 2 * chromosomeLength; ++j) {
    //         std::cout << flatArray[i * (2 * chromosomeLength + chromosomeLength) + j] << " ";
    //     }
    //     std::cout << "\nGenome " << i << " - Job Sequence: ";
    //     for (size_t j = 0; j < chromosomeLength; ++j) {
    //         std::cout << flatArray[i * (2 * chromosomeLength + chromosomeLength) + 2 * chromosomeLength + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // Free device memory
    cudaFree(d_cityArray);
    cudaFree(d_pickupOffsets);
    cudaFree(d_combinedArray);
}


// Host function to flatten the population
void flattenPopulation(const std::vector<Genome>& population, std::vector<size_t>& flatArray, int mode) {
    size_t numGenomes = population.size();
    size_t chromosomeLength = population[0].citySequence.size();

    // Adjust flatArray size based on mode
    size_t sequenceMultiplier = (mode == 1) ? 3 : 2; // Include pickup sequence only if mode == 1
    flatArray.resize(numGenomes * chromosomeLength * sequenceMultiplier); 

    for (size_t i = 0; i < numGenomes; ++i) {
        for (size_t j = 0; j < chromosomeLength; ++j) {
            flatArray[i * chromosomeLength + j] = population[i].citySequence[j];
            flatArray[numGenomes * chromosomeLength + i * chromosomeLength + j] = population[i].jobSequence[j];
            if (mode == 1) { // Include pickup sequence only for mode 1
                flatArray[numGenomes * chromosomeLength * 2 + i * chromosomeLength + j] = population[i].pickupSequence[j];
            }
        }
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
    size_t numJobs = jobTimes[0].size();

    // Flatten travelTimes and jobTimes
    std::vector<float> flatTravelTimes(numCities * numCities);
    std::vector<float> flatJobTimes((numCities - 1) * numJobs); // Exclude depot for job times

    for (size_t i = 0; i < numCities; ++i) {
        for (size_t j = 0; j < numCities; ++j) {
            flatTravelTimes[i * numCities + j] = travelTimes[i][j];
        }
        if (i > 0) { // Skip depot row for job times
            for (size_t k = 0; k < numJobs; ++k) {
                flatJobTimes[(i - 1) * (numJobs) + (k - 1)] = jobTimes[i][k];;
            }
        }
    }

    // Flatten population
    std::vector<size_t> flatArray;
    std::vector<float> fitnessArray;
    std::vector<float> fitnessArrayCPU(numGenomes);
    fitnessArray.resize(numGenomes);
    if (mode == 2) {
        // Use GPU-based flattening for mode 2
        flattenPopulationGPU(population, flatArray, numGenomes, chromosomeLength);
    } else {
        // Use flattening logic for modes 0 and 1
        flattenPopulation(population, flatArray, mode);
    }

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

    // evaluateFitnessCPU(flatArray, fitnessArrayCPU, numGenomes, chromosomeLength, flatTravelTimes, numCities, flatJobTimes, numJobs, mode);

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
