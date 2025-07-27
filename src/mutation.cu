// mutation.cu: GPU-parallelized 2-opt and swap mutations for TSPJ

#include "mutation.h"
#include "genome.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <curand_kernel.h>
#include <iostream>

// Global device cost matrix for 2-opt evaluation
static thrust::device_vector<float>* g_d_travelCosts = nullptr;
static size_t g_numCities = 0;

// CUDA kernel for initializing cuRAND states
__global__ void setupRandomStates(curandState* states, unsigned long seed, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Device function to calculate 2-opt improvement
__device__ float calculate2OptImprovement(const size_t* tour, const float* costMatrix,
                                         size_t tourLength, size_t i, size_t j) {
    if (i >= j || j >= tourLength) return 0.0f;
    
    // Current edges: (i-1, i) and (j, j+1)
    size_t prev_i = (i == 0) ? tourLength - 1 : i - 1;
    size_t next_j = (j == tourLength - 1) ? 0 : j + 1;
    
    size_t city_prev_i = tour[prev_i];
    size_t city_i = tour[i];
    size_t city_j = tour[j];
    size_t city_next_j = tour[next_j];
    
    // Ensure indices are within bounds
    if (city_prev_i >= tourLength || city_i >= tourLength || 
        city_j >= tourLength || city_next_j >= tourLength) {
        return 0.0f;
    }
    
    // Current cost
    float oldCost = costMatrix[city_prev_i * tourLength + city_i] + 
                    costMatrix[city_j * tourLength + city_next_j];
    
    // New cost after 2-opt
    float newCost = costMatrix[city_prev_i * tourLength + city_j] + 
                    costMatrix[city_i * tourLength + city_next_j];
    
    return oldCost - newCost; // Positive means improvement
}

// Device function to apply 2-opt move
__device__ void apply2OptMove(size_t* tour, size_t tourLength, size_t i, size_t j) {
    if (i >= j || j >= tourLength) return;
    
    // Reverse the segment between i and j (inclusive)
    while (i < j) {
        size_t temp = tour[i];
        tour[i] = tour[j];
        tour[j] = temp;
        i++;
        j--;
    }
}

// Simplified GPU kernel for 2-opt mutation (one thread per genome, no shared memory)
__global__ void simple2OptMutationKernel(size_t* sequences, size_t numGenomes, 
                                         size_t sequenceLength, float mutationRate,
                                         size_t stagnationCount, curandState* states) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx];
    size_t baseIdx = genomeIdx * sequenceLength;
    
    // Check if mutation should be applied
    float randVal = curand_uniform(&localState);
    if (randVal < mutationRate) {
        // Calculate escape probability based on stagnation
        float escapeProb = fminf(0.5f, 0.1f + (stagnationCount / 500.0f));
        
        // Number of attempts increases with stagnation
        int numAttempts = (stagnationCount > 100) ? 3 : 1;
        
        for (int attempt = 0; attempt < numAttempts; attempt++) {
            // Generate random i and j for 2-opt
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            size_t i = (size_t)(rand1 * sequenceLength);
            size_t j = (size_t)(rand2 * sequenceLength);
            
            if (i > j) {
                size_t temp = i;
                i = j;
                j = temp;
            }
            
            // Ensure valid 2-opt move
            if (j <= i + 1 || j >= sequenceLength) continue;
            
            // Apply move with some probability (or always if high stagnation)
            float applyProb = (stagnationCount > 200) ? 1.0f : 0.7f;
            if (curand_uniform(&localState) < applyProb) {
                apply2OptMove(sequences + baseIdx, sequenceLength, i, j);
                break; // Apply only one move per mutation
            }
        }
    }
    
    // Save state
    states[genomeIdx] = localState;
}

// Simplified GPU kernel for swap mutation on job sequences
__global__ void simpleSwapMutationKernel(size_t* sequences, size_t numGenomes,
                                         size_t sequenceLength, float mutationRate,
                                         size_t stagnationCount, curandState* states) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx];
    size_t baseIdx = genomeIdx * sequenceLength;
    
    // Enhanced mutation rate based on stagnation
    float adaptiveMutationRate = mutationRate * (1.0f + stagnationCount / 500.0f);
    adaptiveMutationRate = fminf(adaptiveMutationRate, 0.8f);
    
    float randVal = curand_uniform(&localState);
    
    if (randVal < adaptiveMutationRate && sequenceLength > 1) {
        // Number of swaps increases with stagnation
        int numSwaps = (stagnationCount > 200) ? 2 : 1;
        
        for (int swap = 0; swap < numSwaps; swap++) {
            // Generate two random positions
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            size_t pos1 = (size_t)(rand1 * sequenceLength);
            size_t pos2 = (size_t)(rand2 * sequenceLength);
            
            // Ensure valid positions
            if (pos1 != pos2 && pos1 < sequenceLength && pos2 < sequenceLength) {
                // Perform swap
                size_t temp = sequences[baseIdx + pos1];
                sequences[baseIdx + pos1] = sequences[baseIdx + pos2];
                sequences[baseIdx + pos2] = temp;
            }
        }
    }
    
    // Save state
    states[genomeIdx] = localState;
}

// GPU kernel for scramble mutation (for very high stagnation)
__global__ void scrambleMutationKernel(size_t* sequences, size_t numGenomes,
                                      size_t sequenceLength, float mutationRate,
                                      curandState* states) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx];
    size_t baseIdx = genomeIdx * sequenceLength;
    
    float randVal = curand_uniform(&localState);
    
    if (randVal < mutationRate && sequenceLength > 2) {
        // Select a random segment to scramble
        float rand1 = curand_uniform(&localState);
        float rand2 = curand_uniform(&localState);
        
        size_t start = (size_t)(rand1 * (sequenceLength - 1));
        size_t end = (size_t)(rand2 * sequenceLength);
        
        if (start >= end) {
            size_t temp = start;
            start = end;
            end = temp;
        }
        
        // Simple scramble: reverse the segment
        if (end > start + 1) {
            apply2OptMove(sequences + baseIdx, sequenceLength, start, end - 1);
        }
    }
    
    states[genomeIdx] = localState;
}

// GPU kernel for batch 2-opt mutation on city sequences
__global__ void batch2OptMutationKernel(size_t* citySequences, const float* costMatrix,
                                       size_t numGenomes, size_t tourLength,
                                       float mutationRate, size_t stagnationCount,
                                       curandState* states, int* mutationCounts) {
    int genomeIdx = blockIdx.x;
    int threadIdx_x = threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx * blockDim.x + threadIdx_x];
    
    // Each genome gets its own shared memory for tour
    __shared__ size_t sharedTour[MAX_CITIES];
    __shared__ bool tourModified;
    __shared__ int localMutationCount;
    
    if (threadIdx_x == 0) {
        tourModified = false;
        localMutationCount = 0;
    }
    __syncthreads();
    
    // Copy tour to shared memory
    size_t baseIdx = genomeIdx * tourLength;
    if (threadIdx_x < tourLength) {
        sharedTour[threadIdx_x] = citySequences[baseIdx + threadIdx_x];
    }
    __syncthreads();
    
    // Check if mutation should be applied
    float randVal = curand_uniform(&localState);
    bool shouldMutate = (randVal < mutationRate);
    
    if (shouldMutate) {
        // Calculate escape probability based on stagnation
        float escapeProb = fminf(0.7f, 0.1f + (stagnationCount / 300.0f));
        
        // Each thread tries different 2-opt moves
        int numAttempts = (stagnationCount > 100) ? 8 : 4;
        
        for (int attempt = 0; attempt < numAttempts; attempt++) {
            // Generate random i and j for 2-opt
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            size_t i = (size_t)(rand1 * tourLength);
            size_t j = (size_t)(rand2 * tourLength);
            
            if (i > j) {
                size_t temp = i;
                i = j;
                j = temp;
            }
            
            if (j <= i + 1 || j >= tourLength) continue;
            
            // Calculate improvement
            float improvement = calculate2OptImprovement(sharedTour, costMatrix, tourLength, i, j);
            
            // Apply move if improving or with escape probability
            bool applyMove = (improvement > 0.001f) || 
                           (curand_uniform(&localState) < escapeProb);
            
            if (applyMove && threadIdx_x == 0) { // Only one thread applies the move
                apply2OptMove(sharedTour, tourLength, i, j);
                tourModified = true;
                atomicAdd(&localMutationCount, 1);
                break; // Apply only one move per attempt
            }
            __syncthreads();
            
            if (tourModified) break;
        }
    }
    
    __syncthreads();
    
    // Copy modified tour back to global memory
    if (threadIdx_x < tourLength) {
        citySequences[baseIdx + threadIdx_x] = sharedTour[threadIdx_x];
    }
    
    if (threadIdx_x == 0) {
        atomicAdd(&mutationCounts[genomeIdx], localMutationCount);
    }
    
    // Save state
    states[genomeIdx * blockDim.x + threadIdx_x] = localState;
}

// GPU kernel for batch swap mutation on job sequences
__global__ void batchSwapMutationKernel(size_t* jobSequences, size_t numGenomes,
                                       size_t sequenceLength, float mutationRate,
                                       size_t stagnationCount, curandState* states,
                                       int* mutationCounts) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx];
    
    // Enhanced mutation rate based on stagnation
    float adaptiveMutationRate = mutationRate * (1.0f + stagnationCount / 500.0f);
    adaptiveMutationRate = fminf(adaptiveMutationRate, 0.9f);
    
    float randVal = curand_uniform(&localState);
    
    if (randVal < adaptiveMutationRate) {
        size_t baseIdx = genomeIdx * sequenceLength;
        
        // Number of swaps increases with stagnation
        int numSwaps = (stagnationCount > 200) ? 3 : ((stagnationCount > 50) ? 2 : 1);
        
        for (int swap = 0; swap < numSwaps; swap++) {
            // Generate two random positions
            float rand1 = curand_uniform(&localState);
            float rand2 = curand_uniform(&localState);
            
            size_t pos1 = (size_t)(rand1 * sequenceLength);
            size_t pos2 = (size_t)(rand2 * sequenceLength);
            
            if (pos1 != pos2 && pos1 < sequenceLength && pos2 < sequenceLength) {
                // Perform swap
                size_t temp = jobSequences[baseIdx + pos1];
                jobSequences[baseIdx + pos1] = jobSequences[baseIdx + pos2];
                jobSequences[baseIdx + pos2] = temp;
                
                atomicAdd(&mutationCounts[genomeIdx], 1);
            }
        }
    }
    
    // Save state
    states[genomeIdx] = localState;
}

// GPU kernel for batch scramble mutation (for high stagnation)
__global__ void batchScrambleMutationKernel(size_t* sequences, size_t numGenomes,
                                           size_t sequenceLength, float mutationRate,
                                           curandState* states, int* mutationCounts) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (genomeIdx >= numGenomes) return;
    
    curandState localState = states[genomeIdx];
    
    float randVal = curand_uniform(&localState);
    
    if (randVal < mutationRate) {
        size_t baseIdx = genomeIdx * sequenceLength;
        
        // Select a random segment to scramble
        float rand1 = curand_uniform(&localState);
        float rand2 = curand_uniform(&localState);
        
        size_t start = (size_t)(rand1 * sequenceLength);
        size_t end = (size_t)(rand2 * sequenceLength);
        
        if (start > end) {
            size_t temp = start;
            start = end;
            end = temp;
        }
        
        // Scramble the segment (simple reversal)
        if (end > start + 1) {
            while (start < end) {
                size_t temp = sequences[baseIdx + start];
                sequences[baseIdx + start] = sequences[baseIdx + end];
                sequences[baseIdx + end] = temp;
                start++;
                end--;
            }
            atomicAdd(&mutationCounts[genomeIdx], 1);
        }
    }
    
    states[genomeIdx] = localState;
}

void initializeMutationCostMatrix(const std::vector<std::vector<float>>& travelTimes) {
    cleanupMutationCostMatrix();
    
    g_numCities = travelTimes.size();
    
    // Flatten travel times matrix
    std::vector<float> flatTravelTimes(g_numCities * g_numCities);
    for (size_t i = 0; i < g_numCities; i++) {
        for (size_t j = 0; j < g_numCities; j++) {
            flatTravelTimes[i * g_numCities + j] = travelTimes[i][j];
        }
    }
    
    // Allocate and copy to device
    g_d_travelCosts = new thrust::device_vector<float>(flatTravelTimes);
    
    std::cout << "Mutation cost matrix initialized: " << g_numCities << " cities" << std::endl;
}

void cleanupMutationCostMatrix() {
    delete g_d_travelCosts;
    g_d_travelCosts = nullptr;
}

void performMutation(Genome& genome, float mutationRate, int mode, size_t stagnationCount) {
    // For single genome, use batch mutation with size 1
    std::vector<Genome> batch = {genome};
    performBatchMutation(batch, mutationRate, mode, {}, stagnationCount);
    genome = batch[0];
}

void performBatchMutation(std::vector<Genome>& genomes, float mutationRate, int mode,
                         const std::vector<std::vector<float>>& travelTimes,
                         size_t stagnationCount) {
    if (genomes.empty()) return;
    
    size_t numGenomes = genomes.size();
    size_t sequenceLength = genomes[0].citySequence.size();
    
    std::cout << "DEBUG: Starting GPU batch mutation for " << numGenomes << " genomes" << std::endl;
    
    // Check for reasonable limits
    if (numGenomes > 10000 || sequenceLength > MAX_CITIES) {
        std::cout << "DEBUG: Using CPU fallback for large problem size" << std::endl;
        // Use CPU fallback for very large problems
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        
        for (auto& genome : genomes) {
            if (probDist(gen) < mutationRate) {
                std::uniform_int_distribution<size_t> indexDist(0, genome.citySequence.size() - 1);
                size_t i = indexDist(gen);
                size_t j = indexDist(gen);
                if (i > j) std::swap(i, j);
                if (j > i + 1) {
                    std::reverse(genome.citySequence.begin() + i, genome.citySequence.begin() + j + 1);
                }
            }
        }
        return;
    }
    
    try {
        // Flatten all sequences for GPU processing
        std::vector<size_t> flatCitySequences(numGenomes * sequenceLength);
        std::vector<size_t> flatJobSequences(numGenomes * sequenceLength);
        std::vector<size_t> flatPickupSequences;
        
        if (mode == 1) {
            flatPickupSequences.resize(numGenomes * sequenceLength);
        }
        
        // Copy genome data to flat arrays
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(genomes[i].citySequence.begin(), genomes[i].citySequence.end(),
                     flatCitySequences.begin() + i * sequenceLength);
            std::copy(genomes[i].jobSequence.begin(), genomes[i].jobSequence.end(),
                     flatJobSequences.begin() + i * sequenceLength);
            
            if (mode == 1) {
                std::copy(genomes[i].pickupSequence.begin(), genomes[i].pickupSequence.end(),
                         flatPickupSequences.begin() + i * sequenceLength);
            }
        }
        
        std::cout << "DEBUG: Data flattened, allocating GPU memory" << std::endl;
        
        // Allocate device memory
        thrust::device_vector<size_t> d_citySequences(flatCitySequences);
        thrust::device_vector<size_t> d_jobSequences(flatJobSequences);
        thrust::device_vector<size_t> d_pickupSequences;
        
        if (mode == 1) {
            d_pickupSequences = thrust::device_vector<size_t>(flatPickupSequences);
        }
        
        std::cout << "DEBUG: GPU memory allocated, setting up random states" << std::endl;
        
        // Initialize random states (one per genome)
        thrust::device_vector<curandState> d_states(numGenomes);
        
        // Setup random states with conservative configuration
        int threadsPerBlock = std::min(256, static_cast<int>(numGenomes));
        int blocksPerGrid = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;
        
        // Ensure we don't exceed GPU limits
        if (blocksPerGrid > 65535) {
            threadsPerBlock = 256;
            blocksPerGrid = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;
            if (blocksPerGrid > 65535) {
                std::cout << "DEBUG: Problem too large for GPU, using CPU fallback" << std::endl;
                // Fallback to CPU for very large problems
                return;
            }
        }
        
        std::cout << "DEBUG: Launching setupRandomStates with " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads" << std::endl;
        
        setupRandomStates<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_states.data()),
            time(nullptr) + stagnationCount,
            numGenomes);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in setupRandomStates: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Sync Error after setupRandomStates: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        std::cout << "DEBUG: Random states initialized successfully" << std::endl;
        
        // Launch simple 2-opt mutation for city sequences
        std::cout << "DEBUG: Launching city sequence mutation" << std::endl;
        
        simple2OptMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_citySequences.data()),
            numGenomes, sequenceLength, mutationRate, stagnationCount,
            thrust::raw_pointer_cast(d_states.data()));
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in city 2-opt mutation: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Sync Error after city mutation: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        std::cout << "DEBUG: City sequence mutation completed" << std::endl;
        
        // Launch swap mutation for job sequences
        std::cout << "DEBUG: Launching job sequence mutation" << std::endl;
        
        simpleSwapMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_jobSequences.data()),
            numGenomes, sequenceLength, mutationRate, stagnationCount,
            thrust::raw_pointer_cast(d_states.data()));
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in job swap mutation: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Sync Error after job mutation: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        std::cout << "DEBUG: Job sequence mutation completed" << std::endl;
        
        // Launch 2-opt mutation for pickup sequences (if mode == 1)
        if (mode == 1) {
            std::cout << "DEBUG: Launching pickup sequence mutation" << std::endl;
            
            simple2OptMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
                thrust::raw_pointer_cast(d_pickupSequences.data()),
                numGenomes, sequenceLength, mutationRate * 0.8f, stagnationCount,
                thrust::raw_pointer_cast(d_states.data()));
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error in pickup 2-opt mutation: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Sync Error after pickup mutation: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            
            std::cout << "DEBUG: Pickup sequence mutation completed" << std::endl;
        }
        
        // Apply scramble mutation for very high stagnation
        if (stagnationCount > 400) {
            float scrambleRate = std::min(0.2f, (stagnationCount - 400) / 2000.0f);
            
            std::cout << "DEBUG: Launching scramble mutation (rate: " << scrambleRate << ")" << std::endl;
            
            scrambleMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
                thrust::raw_pointer_cast(d_citySequences.data()),
                numGenomes, sequenceLength, scrambleRate,
                thrust::raw_pointer_cast(d_states.data()));
            
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Sync Error after scramble mutation: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            
            std::cout << "DEBUG: Scramble mutation completed" << std::endl;
        }
        
        std::cout << "DEBUG: All GPU mutations completed, copying back to host" << std::endl;
        
        // Copy results back to host
        thrust::copy(d_citySequences.begin(), d_citySequences.end(), flatCitySequences.begin());
        thrust::copy(d_jobSequences.begin(), d_jobSequences.end(), flatJobSequences.begin());
        
        if (mode == 1) {
            thrust::copy(d_pickupSequences.begin(), d_pickupSequences.end(), flatPickupSequences.begin());
        }
        
        // Update genomes with mutated sequences
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(flatCitySequences.begin() + i * sequenceLength,
                     flatCitySequences.begin() + (i + 1) * sequenceLength,
                     genomes[i].citySequence.begin());
            std::copy(flatJobSequences.begin() + i * sequenceLength,
                     flatJobSequences.begin() + (i + 1) * sequenceLength,
                     genomes[i].jobSequence.begin());
            
            if (mode == 1) {
                std::copy(flatPickupSequences.begin() + i * sequenceLength,
                         flatPickupSequences.begin() + (i + 1) * sequenceLength,
                         genomes[i].pickupSequence.begin());
            }
        }
        
        std::cout << "DEBUG: GPU batch mutation completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in GPU mutation: " << e.what() << std::endl;
        std::cout << "Falling back to CPU mutation" << std::endl;
        
        // CPU fallback
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        
        for (auto& genome : genomes) {
            if (probDist(gen) < mutationRate) {
                std::uniform_int_distribution<size_t> indexDist(0, genome.citySequence.size() - 1);
                size_t i = indexDist(gen);
                size_t j = indexDist(gen);
                if (i > j) std::swap(i, j);
                if (j > i + 1) {
                    std::reverse(genome.citySequence.begin() + i, genome.citySequence.begin() + j + 1);
                }
            }
        }
    }
}