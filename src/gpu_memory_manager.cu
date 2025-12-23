// gpu_memory_manager.cu: Implementation of GPU memory management for TSPJ solver

#include "gpu_memory_manager.cuh"
#include <iostream>
#include <cstring>

// RNG initialization kernel
__global__ void initRngStatesKernel(curandState* states, uint32_t numStates, uint64_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Fisher-Yates shuffle on GPU
__device__ void shuffleSequence(uint16_t* sequence, uint32_t length, curandState* state) {
    for (uint32_t i = length - 1; i > 0; --i) {
        uint32_t j = curand(state) % (i + 1);
        uint16_t temp = sequence[i];
        sequence[i] = sequence[j];
        sequence[j] = temp;
    }
}

// Population initialization kernel
__global__ void initPopulationKernel(
    uint16_t* citySequences,
    uint16_t* jobSequences,
    uint16_t* pickupSequences,
    curandState* rngStates,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t popSize,
    int mode
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= popSize) return;

    curandState localState = rngStates[genomeIdx];

    // Temporary arrays in local memory for shuffling
    // Note: For large numCities, this may need to use global memory
    uint16_t cityTemp[GPU_MAX_CITIES];
    uint16_t jobTemp[GPU_MAX_CITIES];

    // Initialize city sequence [1, numCities] (city 0 is depot, not in tour)
    for (uint32_t i = 0; i < numCities; ++i) {
        cityTemp[i] = i + 1;  // Cities 1 to numCities
    }

    // Shuffle city sequence
    shuffleSequence(cityTemp, numCities, &localState);

    // Copy to SoA format
    for (uint32_t i = 0; i < numCities; ++i) {
        citySequences[i * popSize + genomeIdx] = cityTemp[i];
    }

    // Initialize job sequence [1, numJobs]
    for (uint32_t i = 0; i < numJobs; ++i) {
        jobTemp[i] = i + 1;  // Jobs 1 to numJobs
    }

    // Shuffle job sequence
    shuffleSequence(jobTemp, numJobs, &localState);

    // Copy to SoA format
    for (uint32_t i = 0; i < numJobs; ++i) {
        jobSequences[i * popSize + genomeIdx] = jobTemp[i];
    }

    // Initialize pickup sequence if mode == 1
    if (mode == 1 && pickupSequences != nullptr) {
        // Pickup sequence is also a permutation of cities
        for (uint32_t i = 0; i < numCities; ++i) {
            cityTemp[i] = i + 1;
        }
        shuffleSequence(cityTemp, numCities, &localState);
        for (uint32_t i = 0; i < numCities; ++i) {
            pickupSequences[i * popSize + genomeIdx] = cityTemp[i];
        }
    }

    // Save RNG state back
    rngStates[genomeIdx] = localState;
}

// Constructor
GPUMemoryManager::GPUMemoryManager()
    : d_sortTempStorage_(nullptr)
    , sortTempStorageBytes_(0)
    , totalAllocatedBytes_(0)
    , initialized_(false)
    , bestEverFitness_(1e30f)
    , hasBestEver_(false)
{
    memset(&population_, 0, sizeof(population_));
    memset(&problemData_, 0, sizeof(problemData_));
    memset(&eaxWorkspace_, 0, sizeof(eaxWorkspace_));
    memset(&rngState_, 0, sizeof(rngState_));
}

// Destructor
GPUMemoryManager::~GPUMemoryManager() {
    cleanup();
}

cudaError_t GPUMemoryManager::initialize(uint32_t populationSize, uint32_t numCities,
                                          uint32_t numJobs, int mode, uint64_t rngSeed) {
    if (initialized_) {
        cleanup();
    }

    cudaError_t err;

    // Allocate population
    err = allocatePopulation(populationSize, numCities, numJobs, mode);
    if (err != cudaSuccess) return err;

    // Allocate problem data
    err = allocateProblemData(numCities + 1, numJobs);  // +1 for depot
    if (err != cudaSuccess) return err;

    // Allocate EAX workspace (max pairs = popSize/2)
    err = allocateEAXWorkspace(numCities, numJobs, populationSize / 2);
    if (err != cudaSuccess) return err;

    // Allocate RNG states
    err = allocateRngStates(populationSize * 2, rngSeed);  // *2 for offspring
    if (err != cudaSuccess) return err;

    // Allocate sorting temporary storage
    err = allocateSortingStorage(populationSize * 2);
    if (err != cudaSuccess) return err;

    initialized_ = true;
    return cudaSuccess;
}

void GPUMemoryManager::cleanup() {
    if (!initialized_) return;

    // Free population
    if (population_.d_citySequences) cudaFree(population_.d_citySequences);
    if (population_.d_jobSequences) cudaFree(population_.d_jobSequences);
    if (population_.d_pickupSequences) cudaFree(population_.d_pickupSequences);
    if (population_.d_fitness) cudaFree(population_.d_fitness);
    if (population_.d_offspringCity) cudaFree(population_.d_offspringCity);
    if (population_.d_offspringJob) cudaFree(population_.d_offspringJob);
    if (population_.d_offspringPickup) cudaFree(population_.d_offspringPickup);
    if (population_.d_offspringFitness) cudaFree(population_.d_offspringFitness);
    if (population_.d_parent1Indices) cudaFree(population_.d_parent1Indices);
    if (population_.d_parent2Indices) cudaFree(population_.d_parent2Indices);
    if (population_.d_sortIndices) cudaFree(population_.d_sortIndices);

    // Free problem data
    if (problemData_.d_travelCosts) cudaFree(problemData_.d_travelCosts);
    if (problemData_.d_jobCosts) cudaFree(problemData_.d_jobCosts);

    // Free EAX workspace
    if (eaxWorkspace_.d_adjacencyLists) cudaFree(eaxWorkspace_.d_adjacencyLists);
    if (eaxWorkspace_.d_visited) cudaFree(eaxWorkspace_.d_visited);
    if (eaxWorkspace_.cycles.d_cycleBuffer) cudaFree(eaxWorkspace_.cycles.d_cycleBuffer);
    if (eaxWorkspace_.cycles.d_cycleLengths) cudaFree(eaxWorkspace_.cycles.d_cycleLengths);
    if (eaxWorkspace_.cycles.d_numCycles) cudaFree(eaxWorkspace_.cycles.d_numCycles);
    if (eaxWorkspace_.cycles.d_cycleParentFlags) cudaFree(eaxWorkspace_.cycles.d_cycleParentFlags);
    if (eaxWorkspace_.d_bestAssembly) cudaFree(eaxWorkspace_.d_bestAssembly);
    if (eaxWorkspace_.d_assemblyCosts) cudaFree(eaxWorkspace_.d_assemblyCosts);
    if (eaxWorkspace_.d_parent1Gathered) cudaFree(eaxWorkspace_.d_parent1Gathered);
    if (eaxWorkspace_.d_parent2Gathered) cudaFree(eaxWorkspace_.d_parent2Gathered);
    if (eaxWorkspace_.d_parent1Jobs) cudaFree(eaxWorkspace_.d_parent1Jobs);
    if (eaxWorkspace_.d_parent2Jobs) cudaFree(eaxWorkspace_.d_parent2Jobs);

    // Free RNG states
    if (rngState_.d_states) cudaFree(rngState_.d_states);

    // Free sorting storage
    if (d_sortTempStorage_) cudaFree(d_sortTempStorage_);

    memset(&population_, 0, sizeof(population_));
    memset(&problemData_, 0, sizeof(problemData_));
    memset(&eaxWorkspace_, 0, sizeof(eaxWorkspace_));
    memset(&rngState_, 0, sizeof(rngState_));
    d_sortTempStorage_ = nullptr;
    sortTempStorageBytes_ = 0;
    totalAllocatedBytes_ = 0;
    initialized_ = false;

    // Reset best-ever tracking
    bestEverCitySeq_.clear();
    bestEverJobSeq_.clear();
    bestEverPickupSeq_.clear();
    bestEverFitness_ = 1e30f;
    hasBestEver_ = false;
}

cudaError_t GPUMemoryManager::allocatePopulation(uint32_t popSize, uint32_t numCities,
                                                   uint32_t numJobs, int mode) {
    cudaError_t err;
    uint32_t maxOffspring = popSize;  // Can generate up to popSize offspring

    // City sequences
    size_t citySeqBytes = numCities * popSize * sizeof(uint16_t);
    err = cudaMalloc(&population_.d_citySequences, citySeqBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += citySeqBytes;

    // Job sequences
    size_t jobSeqBytes = numJobs * popSize * sizeof(uint16_t);
    err = cudaMalloc(&population_.d_jobSequences, jobSeqBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += jobSeqBytes;

    // Pickup sequences (only for mode 1)
    if (mode == 1) {
        err = cudaMalloc(&population_.d_pickupSequences, citySeqBytes);
        if (err != cudaSuccess) return err;
        totalAllocatedBytes_ += citySeqBytes;
    } else {
        population_.d_pickupSequences = nullptr;
    }

    // Fitness
    size_t fitnessBytes = popSize * sizeof(float);
    err = cudaMalloc(&population_.d_fitness, fitnessBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += fitnessBytes;

    // Offspring buffers
    size_t offspringCityBytes = numCities * maxOffspring * sizeof(uint16_t);
    err = cudaMalloc(&population_.d_offspringCity, offspringCityBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += offspringCityBytes;

    size_t offspringJobBytes = numJobs * maxOffspring * sizeof(uint16_t);
    err = cudaMalloc(&population_.d_offspringJob, offspringJobBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += offspringJobBytes;

    if (mode == 1) {
        err = cudaMalloc(&population_.d_offspringPickup, offspringCityBytes);
        if (err != cudaSuccess) return err;
        totalAllocatedBytes_ += offspringCityBytes;
    } else {
        population_.d_offspringPickup = nullptr;
    }

    size_t offspringFitnessBytes = maxOffspring * sizeof(float);
    err = cudaMalloc(&population_.d_offspringFitness, offspringFitnessBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += offspringFitnessBytes;

    // Selection indices
    size_t indicesBytes = maxOffspring * sizeof(uint32_t);
    err = cudaMalloc(&population_.d_parent1Indices, indicesBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += indicesBytes;

    err = cudaMalloc(&population_.d_parent2Indices, indicesBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += indicesBytes;

    // Sort indices
    size_t sortIndicesBytes = (popSize + maxOffspring) * sizeof(uint32_t);
    err = cudaMalloc(&population_.d_sortIndices, sortIndicesBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += sortIndicesBytes;

    // Set metadata
    population_.popSize = popSize;
    population_.maxPopSize = popSize;
    population_.numCities = numCities;
    population_.numJobs = numJobs;
    population_.maxOffspring = maxOffspring;
    population_.currentOffspring = 0;
    population_.mode = mode;

    return cudaSuccess;
}

cudaError_t GPUMemoryManager::allocateProblemData(uint32_t numCities, uint32_t numJobs) {
    cudaError_t err;

    // Travel costs: [numCities x numCities]
    size_t travelBytes = numCities * numCities * sizeof(float);
    err = cudaMalloc(&problemData_.d_travelCosts, travelBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += travelBytes;

    // Job costs: [(numCities-1) x numJobs] - excluding depot
    size_t jobBytes = (numCities - 1) * numJobs * sizeof(float);
    err = cudaMalloc(&problemData_.d_jobCosts, jobBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += jobBytes;

    problemData_.numCities = numCities;
    problemData_.numJobs = numJobs;

    return cudaSuccess;
}

cudaError_t GPUMemoryManager::allocateEAXWorkspace(uint32_t numCities, uint32_t numJobs, uint32_t maxPairs) {
    cudaError_t err;

    // Adjacency lists: 4 neighbors per city per pair - cities are 1-indexed
    size_t adjBytes = (numCities + 1) * maxPairs * sizeof(CityAdjacency);
    err = cudaMalloc(&eaxWorkspace_.d_adjacencyLists, adjBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += adjBytes;

    // Visited flags - cities are 1-indexed, so need (numCities + 1) elements
    size_t visitedBytes = (numCities + 1) * maxPairs * sizeof(uint8_t);
    err = cudaMalloc(&eaxWorkspace_.d_visited, visitedBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += visitedBytes;

    // Cycle buffer
    size_t cycleBufferBytes = GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * maxPairs * sizeof(uint16_t);
    err = cudaMalloc(&eaxWorkspace_.cycles.d_cycleBuffer, cycleBufferBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += cycleBufferBytes;

    // Cycle lengths
    size_t cycleLengthsBytes = GPU_MAX_CYCLES * maxPairs * sizeof(uint16_t);
    err = cudaMalloc(&eaxWorkspace_.cycles.d_cycleLengths, cycleLengthsBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += cycleLengthsBytes;

    // Number of cycles per pair
    size_t numCyclesBytes = maxPairs * sizeof(uint16_t);
    err = cudaMalloc(&eaxWorkspace_.cycles.d_numCycles, numCyclesBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += numCyclesBytes;

    // Cycle parent flags
    size_t cycleParentBytes = GPU_MAX_CYCLES * GPU_MAX_CYCLE_LENGTH * maxPairs * sizeof(uint8_t);
    err = cudaMalloc(&eaxWorkspace_.cycles.d_cycleParentFlags, cycleParentBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += cycleParentBytes;

    // Best assembly (bitmask of selected cycles)
    size_t assemblyBytes = maxPairs * sizeof(uint32_t);
    err = cudaMalloc(&eaxWorkspace_.d_bestAssembly, assemblyBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += assemblyBytes;

    // Assembly costs
    size_t assemblyCostBytes = maxPairs * sizeof(float);
    err = cudaMalloc(&eaxWorkspace_.d_assemblyCosts, assemblyCostBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += assemblyCostBytes;

    // Parent gathering buffers for EAX (city tours)
    size_t parentBytes = numCities * maxPairs * sizeof(uint16_t);
    err = cudaMalloc(&eaxWorkspace_.d_parent1Gathered, parentBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += parentBytes;

    err = cudaMalloc(&eaxWorkspace_.d_parent2Gathered, parentBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += parentBytes;

    // Parent gathering buffers for EAX (job sequences)
    size_t jobBytes = numJobs * maxPairs * sizeof(uint16_t);
    err = cudaMalloc(&eaxWorkspace_.d_parent1Jobs, jobBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += jobBytes;

    err = cudaMalloc(&eaxWorkspace_.d_parent2Jobs, jobBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += jobBytes;

    eaxWorkspace_.maxPairs = maxPairs;
    eaxWorkspace_.numCities = numCities;
    eaxWorkspace_.numJobs = numJobs;

    return cudaSuccess;
}

cudaError_t GPUMemoryManager::allocateRngStates(uint32_t numStates, uint64_t seed) {
    cudaError_t err;

    size_t rngBytes = numStates * sizeof(curandState);
    err = cudaMalloc(&rngState_.d_states, rngBytes);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += rngBytes;

    rngState_.numStates = numStates;

    // Initialize RNG states
    int threadsPerBlock = 256;
    int numBlocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
    initRngStatesKernel<<<numBlocks, threadsPerBlock>>>(rngState_.d_states, numStates, seed);

    return cudaDeviceSynchronize();
}

cudaError_t GPUMemoryManager::allocateSortingStorage(uint32_t maxElements) {
    // For now, allocate a reasonable amount
    // In a full implementation, we'd query CUB for exact requirements
    sortTempStorageBytes_ = maxElements * sizeof(float) * 4;
    cudaError_t err = cudaMalloc(&d_sortTempStorage_, sortTempStorageBytes_);
    if (err != cudaSuccess) return err;
    totalAllocatedBytes_ += sortTempStorageBytes_;
    return cudaSuccess;
}

cudaError_t GPUMemoryManager::loadProblemData(const std::vector<std::vector<float>>& travelCosts,
                                               const std::vector<std::vector<float>>& jobCosts) {
    uint32_t numCities = problemData_.numCities;
    uint32_t numJobs = problemData_.numJobs;

    // Flatten and copy travel costs
    std::vector<float> flatTravel(numCities * numCities);
    for (uint32_t i = 0; i < numCities; ++i) {
        for (uint32_t j = 0; j < numCities; ++j) {
            flatTravel[i * numCities + j] = travelCosts[i][j];
        }
    }
    cudaError_t err = cudaMemcpy(problemData_.d_travelCosts, flatTravel.data(),
                                  numCities * numCities * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // Flatten and copy job costs (excluding depot row)
    std::vector<float> flatJobs((numCities - 1) * numJobs);
    for (uint32_t i = 1; i < numCities; ++i) {  // Skip depot (city 0)
        for (uint32_t j = 0; j < numJobs; ++j) {
            flatJobs[(i - 1) * numJobs + j] = jobCosts[i][j + 1];  // Jobs are 1-indexed
        }
    }

    err = cudaMemcpy(problemData_.d_jobCosts, flatJobs.data(),
                      (numCities - 1) * numJobs * sizeof(float), cudaMemcpyHostToDevice);

    return err;
}

cudaError_t GPUMemoryManager::initializePopulation() {
    int threadsPerBlock = 256;
    int numBlocks = (population_.popSize + threadsPerBlock - 1) / threadsPerBlock;

    initPopulationKernel<<<numBlocks, threadsPerBlock>>>(
        population_.d_citySequences,
        population_.d_jobSequences,
        population_.d_pickupSequences,
        rngState_.d_states,
        population_.numCities,
        population_.numJobs,
        population_.popSize,
        population_.mode
    );

    return cudaDeviceSynchronize();
}

cudaError_t GPUMemoryManager::copyBestGenome(std::vector<uint16_t>& citySequence,
                                              std::vector<uint16_t>& jobSequence,
                                              std::vector<uint16_t>& pickupSequence,
                                              float& fitness) {
    // Assume population is sorted by fitness (best at index 0)
    return copyGenomeToHost(0, citySequence, jobSequence, pickupSequence);
}

cudaError_t GPUMemoryManager::copyFitnessToHost(std::vector<float>& fitness) {
    fitness.resize(population_.popSize);
    return cudaMemcpy(fitness.data(), population_.d_fitness,
                      population_.popSize * sizeof(float), cudaMemcpyDeviceToHost);
}

cudaError_t GPUMemoryManager::copyGenomeToHost(uint32_t genomeIdx,
                                                 std::vector<uint16_t>& citySequence,
                                                 std::vector<uint16_t>& jobSequence,
                                                 std::vector<uint16_t>& pickupSequence) {
    uint32_t numCities = population_.numCities;
    uint32_t numJobs = population_.numJobs;
    uint32_t popSize = population_.popSize;

    citySequence.resize(numCities);
    jobSequence.resize(numJobs);

    // Copy city sequence (need to gather from SoA format)
    std::vector<uint16_t> tempCity(numCities * popSize);
    cudaError_t err = cudaMemcpy(tempCity.data(), population_.d_citySequences,
                                  numCities * popSize * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    for (uint32_t i = 0; i < numCities; ++i) {
        citySequence[i] = tempCity[i * popSize + genomeIdx];
    }

    // Copy job sequence
    std::vector<uint16_t> tempJob(numJobs * popSize);
    err = cudaMemcpy(tempJob.data(), population_.d_jobSequences,
                      numJobs * popSize * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    for (uint32_t i = 0; i < numJobs; ++i) {
        jobSequence[i] = tempJob[i * popSize + genomeIdx];
    }

    // Copy pickup sequence if applicable
    if (population_.mode == 1 && population_.d_pickupSequences != nullptr) {
        pickupSequence.resize(numCities);
        std::vector<uint16_t> tempPickup(numCities * popSize);
        err = cudaMemcpy(tempPickup.data(), population_.d_pickupSequences,
                          numCities * popSize * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return err;

        for (uint32_t i = 0; i < numCities; ++i) {
            pickupSequence[i] = tempPickup[i * popSize + genomeIdx];
        }
    } else {
        pickupSequence.clear();
    }

    return cudaSuccess;
}

void GPUMemoryManager::printMemoryUsage() const {
    std::cout << "GPU Memory Usage:" << std::endl;
    std::cout << "  Total allocated: " << (totalAllocatedBytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Population size: " << population_.popSize << std::endl;
    std::cout << "  Num cities: " << population_.numCities << std::endl;
    std::cout << "  Num jobs: " << population_.numJobs << std::endl;
    std::cout << "  Mode: " << population_.mode << std::endl;
}

cudaError_t GPUMemoryManager::storeBestEverGenome(uint32_t genomeIdx, float fitness) {
    // Copy the genome from GPU to host and store it
    std::vector<uint16_t> citySeq, jobSeq, pickupSeq;
    cudaError_t err = copyGenomeToHost(genomeIdx, citySeq, jobSeq, pickupSeq);
    if (err != cudaSuccess) return err;

    // Store as best-ever
    bestEverCitySeq_ = std::move(citySeq);
    bestEverJobSeq_ = std::move(jobSeq);
    bestEverPickupSeq_ = std::move(pickupSeq);
    bestEverFitness_ = fitness;
    hasBestEver_ = true;

    return cudaSuccess;
}

bool GPUMemoryManager::getBestEverGenome(std::vector<uint16_t>& citySequence,
                                          std::vector<uint16_t>& jobSequence,
                                          std::vector<uint16_t>& pickupSequence,
                                          float& fitness) const {
    if (!hasBestEver_) {
        return false;
    }

    citySequence = bestEverCitySeq_;
    jobSequence = bestEverJobSeq_;
    pickupSequence = bestEverPickupSeq_;
    fitness = bestEverFitness_;

    return true;
}
