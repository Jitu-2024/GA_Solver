// gpu_solver.cu: Main GPU-resident GA solver implementation

#include "gpu_solver.cuh"
#include "eax_gpu.cuh"
#include "job_assignment_gpu.cuh"
#include "mutation_gpu.cuh"
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

// Fitness evaluation kernel
__global__ void evaluateFitnessGPUKernel(
    const uint16_t* __restrict__ citySequences,
    const uint16_t* __restrict__ jobSequences,
    const uint16_t* __restrict__ pickupSequences,
    const float* __restrict__ travelCosts,
    const float* __restrict__ jobCosts,
    float* __restrict__ fitness,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t numGenomes,
    uint32_t stride,
    int mode
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    float currentTime = 0.0f;
    float maxCompletionTime = 0.0f;
    uint16_t prevCity = 0;  // Start at depot

    // Local storage for job completion times
    float jobCompletionTimes[GPU_MAX_CITIES];

    // Job delivery phase
    for (uint32_t pos = 0; pos < numCities; ++pos) {
        uint16_t city = citySequences[pos * stride + genomeIdx];
        uint16_t job = jobSequences[pos * stride + genomeIdx];

        // Travel to city
        currentTime += travelCosts[prevCity * (numCities + 1) + city];

        // Calculate job completion time
        if (city > 0 && job > 0) {
            float jobTime = jobCosts[(city - 1) * numJobs + (job - 1)];
            float completion = currentTime + jobTime;
            jobCompletionTimes[city] = completion;
            maxCompletionTime = fmaxf(maxCompletionTime, completion);
        }

        prevCity = city;
    }

    // Pickup phase (mode 1 only)
    if (mode == 1 && pickupSequences != nullptr) {
        for (uint32_t pos = 0; pos < numCities; ++pos) {
            uint16_t pickupCity = pickupSequences[pos * stride + genomeIdx];

            // Travel to pickup city
            currentTime += travelCosts[prevCity * (numCities + 1) + pickupCity];

            // Wait for job completion if necessary
            if (pickupCity > 0 && jobCompletionTimes[pickupCity] > currentTime) {
                currentTime = jobCompletionTimes[pickupCity];
            }

            prevCity = pickupCity;
        }
    }

    // Return to depot
    currentTime += travelCosts[prevCity * (numCities + 1) + 0];

    // Makespan is max of job completion and return time
    fitness[genomeIdx] = fmaxf(maxCompletionTime, currentTime);
}

// Tournament selection kernel
__global__ void tournamentSelectionKernel(
    const float* __restrict__ fitness,
    uint32_t* __restrict__ parent1Indices,
    uint32_t* __restrict__ parent2Indices,
    curandState* __restrict__ rngStates,
    uint32_t populationSize,
    uint32_t numPairs,
    uint32_t tournamentSize
) {
    uint32_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    curandState localState = rngStates[pairIdx];

    // Select parent1
    float bestFitness1 = 1e30f;
    uint32_t bestIdx1 = 0;
    for (uint32_t t = 0; t < tournamentSize; ++t) {
        uint32_t idx = curand(&localState) % populationSize;
        if (fitness[idx] < bestFitness1) {
            bestFitness1 = fitness[idx];
            bestIdx1 = idx;
        }
    }
    parent1Indices[pairIdx] = bestIdx1;

    // Select parent2 (different from parent1)
    float bestFitness2 = 1e30f;
    uint32_t bestIdx2 = 0;
    for (uint32_t t = 0; t < tournamentSize; ++t) {
        uint32_t idx = curand(&localState) % populationSize;
        if (idx != bestIdx1 && fitness[idx] < bestFitness2) {
            bestFitness2 = fitness[idx];
            bestIdx2 = idx;
        }
    }
    parent2Indices[pairIdx] = bestIdx2;

    rngStates[pairIdx] = localState;
}

// Helper to evaluate fitness of population or offspring
cudaError_t evaluateFitness(GPUPopulation& pop, GPUProblemData& problemData, bool isOffspring) {
    uint32_t numGenomes = isOffspring ? pop.currentOffspring : pop.popSize;
    uint32_t stride = isOffspring ? pop.currentOffspring : pop.popSize;

    uint16_t* citySeq = isOffspring ? pop.d_offspringCity : pop.d_citySequences;
    uint16_t* jobSeq = isOffspring ? pop.d_offspringJob : pop.d_jobSequences;
    uint16_t* pickupSeq = isOffspring ? pop.d_offspringPickup : pop.d_pickupSequences;
    float* fitness = isOffspring ? pop.d_offspringFitness : pop.d_fitness;

    int threadsPerBlock = 256;
    int numBlocks = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;

    evaluateFitnessGPUKernel<<<numBlocks, threadsPerBlock>>>(
        citySeq, jobSeq, pickupSeq,
        problemData.d_travelCosts, problemData.d_jobCosts,
        fitness,
        pop.numCities, pop.numJobs, numGenomes, stride, pop.mode
    );

    return cudaDeviceSynchronize();
}

// Kernel to copy offspring to population at specific positions
__global__ void copyOffspringToPopulationKernel(
    const uint16_t* __restrict__ offspringCity,
    const uint16_t* __restrict__ offspringJob,
    const float* __restrict__ offspringFitness,
    uint16_t* __restrict__ popCity,
    uint16_t* __restrict__ popJob,
    float* __restrict__ popFitness,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t popSize,
    uint32_t numOffspring,
    uint32_t targetStartIdx
) {
    uint32_t offspringIdx = blockIdx.x;
    uint32_t posIdx = threadIdx.x;

    if (offspringIdx >= numOffspring) return;

    uint32_t targetIdx = targetStartIdx + offspringIdx;
    if (targetIdx >= popSize) return;

    // Copy city sequence
    if (posIdx < numCities) {
        popCity[posIdx * popSize + targetIdx] = offspringCity[posIdx * numOffspring + offspringIdx];
    }

    // Copy job sequence (use the first numJobs threads)
    if (posIdx < numJobs) {
        popJob[posIdx * popSize + targetIdx] = offspringJob[posIdx * numOffspring + offspringIdx];
    }

    // Thread 0 copies fitness
    if (posIdx == 0) {
        popFitness[targetIdx] = offspringFitness[offspringIdx];
    }
}

// Kernel to reorder population based on sorted indices (city and job only, not fitness)
__global__ void reorderPopulationKernel(
    const uint16_t* __restrict__ srcCity,
    const uint16_t* __restrict__ srcJob,
    const uint32_t* __restrict__ sortedIndices,
    uint16_t* __restrict__ dstCity,
    uint16_t* __restrict__ dstJob,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t popSize
) {
    uint32_t newIdx = blockIdx.x;
    uint32_t posIdx = threadIdx.x;

    if (newIdx >= popSize) return;

    uint32_t oldIdx = sortedIndices[newIdx];

    // Copy city sequence
    if (posIdx < numCities) {
        dstCity[posIdx * popSize + newIdx] = srcCity[posIdx * popSize + oldIdx];
    }

    // Copy job sequence
    if (posIdx < numJobs) {
        dstJob[posIdx * popSize + newIdx] = srcJob[posIdx * popSize + oldIdx];
    }
    // Note: fitness is NOT copied here because it's already sorted in-place
}

// Survivor selection with proper sorting using thrust
cudaError_t performSurvivorSelection(GPUPopulation& population, uint32_t numOffspring) {
    if (numOffspring == 0) return cudaSuccess;

    uint32_t popSize = population.popSize;
    uint32_t numCities = population.numCities;
    uint32_t numJobs = population.numJobs;

    // Step 1: Copy offspring to the end of population (replace worst individuals)
    uint32_t numToReplace = min(numOffspring, popSize);
    uint32_t targetStartIdx = popSize - numToReplace;

    int threadsPerBlock = min(256, (int)max(numCities, numJobs));
    dim3 gridDim(numToReplace);
    dim3 blockDim(threadsPerBlock);

    copyOffspringToPopulationKernel<<<gridDim, blockDim>>>(
        population.d_offspringCity,
        population.d_offspringJob,
        population.d_offspringFitness,
        population.d_citySequences,
        population.d_jobSequences,
        population.d_fitness,
        numCities,
        numJobs,
        popSize,
        numToReplace,
        targetStartIdx
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;

    // Step 2: Sort population by fitness using thrust
    // Create device vectors wrapping raw pointers
    thrust::device_ptr<float> d_fitness_ptr(population.d_fitness);
    thrust::device_ptr<uint32_t> d_indices_ptr(population.d_sortIndices);

    // Initialize indices to [0, 1, 2, ..., popSize-1]
    thrust::sequence(d_indices_ptr, d_indices_ptr + popSize);

    // Sort indices by fitness (ascending order - best fitness first)
    thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + popSize, d_indices_ptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;

    // After sorting, fitness array is already sorted in-place
    // We need to reorder city and job sequences according to sorted indices

    // But wait - the indices now point to the OLD positions, but fitness is already sorted
    // This is tricky - we actually need a different approach

    // Let's use a simpler approach: create a copy of fitness, sort with indices, then gather

    // Actually, thrust::sort_by_key sorts keys in-place AND rearranges values accordingly
    // So d_fitness is now sorted and d_sortIndices contains the original indices
    // We need to gather city/job sequences using these original indices

    // Step 3: Reorder city and job sequences based on sorted indices
    // Use offspring buffers as temporary storage
    reorderPopulationKernel<<<popSize, threadsPerBlock>>>(
        population.d_citySequences,
        population.d_jobSequences,
        population.d_sortIndices,  // Contains original indices
        population.d_offspringCity,  // Temp destination
        population.d_offspringJob,
        numCities,
        numJobs,
        popSize
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;

    // Step 4: Copy reordered sequences back to main population
    cudaMemcpy(population.d_citySequences, population.d_offspringCity,
               numCities * popSize * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(population.d_jobSequences, population.d_offspringJob,
               numJobs * popSize * sizeof(uint16_t), cudaMemcpyDeviceToDevice);

    return cudaDeviceSynchronize();
}

// Main solver function
GPUSolverResult runGPUSolver(
    const std::vector<std::vector<float>>& travelCosts,
    const std::vector<std::vector<float>>& jobCosts,
    int mode,
    const GPUSolverParams& params
) {
    GPUSolverResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t numCities = travelCosts.size() - 1;  // Exclude depot
    uint32_t numJobs = jobCosts[1].size() - 1;    // Exclude dummy job 0

    if (params.verbose) {
        std::cout << "GPU Solver: " << numCities << " cities, " << numJobs << " jobs, mode " << mode << std::endl;
    }

    // Initialize GPU memory manager
    GPUMemoryManager memManager;
    cudaError_t err = memManager.initialize(params.populationSize, numCities, numJobs, mode, params.rngSeed);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize GPU memory: " << cudaGetErrorString(err) << std::endl;
        return result;
    }

    // Load problem data
    err = memManager.loadProblemData(travelCosts, jobCosts);
    if (err != cudaSuccess) {
        std::cerr << "Failed to load problem data: " << cudaGetErrorString(err) << std::endl;
        return result;
    }

    // Initialize population
    err = memManager.initializePopulation();
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize population: " << cudaGetErrorString(err) << std::endl;
        return result;
    }

    GPUPopulation& population = memManager.getPopulation();
    GPUProblemData& problemData = memManager.getProblemData();
    EAXWorkspace& eaxWorkspace = memManager.getEAXWorkspace();
    GPURngState& rngState = memManager.getRngState();

    // Evaluate initial population
    err = evaluateFitness(population, problemData, false);
    if (err != cudaSuccess) {
        std::cerr << "Failed to evaluate initial fitness: " << cudaGetErrorString(err) << std::endl;
        return result;
    }

    // Get initial best fitness and store the best genome
    float bestFitness = 1e30f;
    uint32_t bestIdx = 0;
    std::vector<float> fitnessHost(params.populationSize);
    cudaMemcpy(fitnessHost.data(), population.d_fitness, params.populationSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < params.populationSize; ++i) {
        if (fitnessHost[i] < bestFitness) {
            bestFitness = fitnessHost[i];
            bestIdx = i;
        }
    }
    // Store the initial best genome
    memManager.storeBestEverGenome(bestIdx, bestFitness);

    uint32_t stagnationCount = 0;
    uint32_t solutionGeneration = 0;
    uint32_t generationsRun = 0;
    float previousBest = bestFitness;

    if (params.verbose) {
        std::cout << "Initial best fitness: " << bestFitness << std::endl;
        memManager.printMemoryUsage();
    }

    // Main GA loop
    for (uint32_t gen = 0; gen < params.maxGenerations; ++gen) {
        ++generationsRun;

        // Adaptive mutation rate
        float mutationRate = params.baseMutationRate;
        if (stagnationCount > 50) {
            float factor = std::min(3.0f, 1.0f + (stagnationCount - 50) / 200.0f);
            mutationRate = std::min(0.9f, params.baseMutationRate * factor);
        }

        // Number of offspring to generate
        uint32_t numOffspring = params.populationSize / 2;
        population.currentOffspring = numOffspring;

        // Tournament selection
        int threadsPerBlock = 256;
        int numBlocks = (numOffspring + threadsPerBlock - 1) / threadsPerBlock;

        tournamentSelectionKernel<<<numBlocks, threadsPerBlock>>>(
            population.d_fitness,
            population.d_parent1Indices,
            population.d_parent2Indices,
            rngState.d_states,
            params.populationSize,
            numOffspring,
            params.tournamentSize
        );
        cudaDeviceSynchronize();

        // EAX crossover for city sequences AND job sequences
        // (constructOffspringTourKernel inherits jobs from parents based on cycle assembly)
        // useCostAware controls whether assembly selection uses actual travel costs or uniform costs
        err = performGPUEAXCrossover(population, problemData, eaxWorkspace, numOffspring, params.useCostAware);
        if (err != cudaSuccess) {
            std::cerr << "EAX crossover failed at generation " << gen << std::endl;
            break;
        }

        // Note: Job assignment is now done via EAX crossover (matching legacy behavior)
        // Jobs are inherited from the parent that contributed each city's edge
        // Greedy assignment is no longer used here

        // Mutation
        err = performGPUMutations(population, problemData, rngState, numOffspring,
                                   mutationRate, stagnationCount, true);
        if (err != cudaSuccess) {
            std::cerr << "Mutation failed at generation " << gen << std::endl;
            break;
        }

        // Evaluate offspring fitness
        err = evaluateFitness(population, problemData, true);
        if (err != cudaSuccess) {
            std::cerr << "Offspring fitness evaluation failed at generation " << gen << std::endl;
            break;
        }

        // Survivor selection
        err = performSurvivorSelection(population, numOffspring);
        if (err != cudaSuccess) {
            std::cerr << "Survivor selection failed at generation " << gen << std::endl;
            break;
        }

        // Re-evaluate population after replacement
        err = evaluateFitness(population, problemData, false);
        if (err != cudaSuccess) break;

        // Check for improvement - find minimum fitness and its index
        cudaMemcpy(fitnessHost.data(), population.d_fitness, params.populationSize * sizeof(float), cudaMemcpyDeviceToHost);
        float currentBest = 1e30f;
        uint32_t bestIdx = 0;
        for (uint32_t i = 0; i < params.populationSize; ++i) {
            if (fitnessHost[i] < currentBest) {
                currentBest = fitnessHost[i];
                bestIdx = i;
            }
        }

        if (currentBest < bestFitness) {
            bestFitness = currentBest;
            stagnationCount = 0;
            solutionGeneration = gen;
            // Store the best genome when found (elitism)
            memManager.storeBestEverGenome(bestIdx, currentBest);
        } else {
            stagnationCount++;
        }

        // Logging
        if (params.verbose && (gen % params.loggingInterval == 0 || gen == params.maxGenerations - 1)) {
            std::cout << "Gen " << gen << ": Best = " << bestFitness
                      << ", Stagnation = " << stagnationCount
                      << ", MutRate = " << mutationRate << std::endl;
        }

        // Early stopping
        if (stagnationCount >= params.maxStagnationGenerations) {
            if (params.verbose) {
                std::cout << "Early stopping at generation " << gen << " due to stagnation" << std::endl;
            }
            break;
        }

        previousBest = bestFitness;
    }

    // Get the stored best-ever genome (guaranteed to match bestFitness)
    float storedFitness;
    if (memManager.getBestEverGenome(result.bestCitySequence, result.bestJobSequence,
                                      result.bestPickupSequence, storedFitness)) {
        result.bestFitness = storedFitness;
    } else {
        // Fallback to index 0 if no best ever stored (shouldn't happen)
        memManager.copyBestGenome(result.bestCitySequence, result.bestJobSequence,
                                   result.bestPickupSequence, result.bestFitness);
        result.bestFitness = bestFitness;
    }
    result.solutionGeneration = solutionGeneration;
    result.generationsRun = generationsRun;

    auto endTime = std::chrono::high_resolution_clock::now();
    result.elapsedTimeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    if (params.verbose) {
        std::cout << "Solver completed in " << result.elapsedTimeSeconds << " seconds" << std::endl;
        std::cout << "Generations run: " << result.generationsRun << std::endl;
        std::cout << "Best fitness: " << result.bestFitness << " found at generation " << result.solutionGeneration << std::endl;
    }

    return result;
}
