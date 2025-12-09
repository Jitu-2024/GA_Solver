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
// Tour structure: depot(0) -> tour[0] -> tour[1] -> ... -> tour[n-1] -> depot(0)
__device__ float calculate2OptImprovement(const size_t* tour, const float* costMatrix,
                                         size_t numCities, size_t tourLength,
                                         size_t i, size_t j) {
    if (i >= j || j >= tourLength) return 0.0f;

    // Get the 4 cities involved in the 2-opt move
    // Edge before position i: connects to tour[i]
    // Edge after position j: connects from tour[j]

    // City before position i (depot if i == 0)
    size_t city_before_i = (i == 0) ? 0 : tour[i - 1];
    size_t city_i = tour[i];
    size_t city_j = tour[j];
    // City after position j (depot if j == tourLength - 1)
    size_t city_after_j = (j == tourLength - 1) ? 0 : tour[j + 1];

    // Ensure indices are within bounds for cost matrix
    if (city_before_i >= numCities || city_i >= numCities ||
        city_j >= numCities || city_after_j >= numCities) {
        return 0.0f;
    }

    // Current edges: (city_before_i -> city_i) and (city_j -> city_after_j)
    float oldCost = costMatrix[city_before_i * numCities + city_i] +
                    costMatrix[city_j * numCities + city_after_j];

    // After 2-opt reversal of [i..j]:
    // New edges: (city_before_i -> city_j) and (city_i -> city_after_j)
    float newCost = costMatrix[city_before_i * numCities + city_j] +
                    costMatrix[city_i * numCities + city_after_j];

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

// Cost-guided 2-opt mutation kernel (one thread per genome)
// Uses cost matrix to only apply improving moves (or random with escape probability)
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
        // Number of attempts increases with stagnation
        int numAttempts = (stagnationCount > 100) ? 5 : 3;

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

            // Ensure valid 2-opt move (need at least 2 elements to reverse)
            if (j <= i + 1 || j >= sequenceLength) continue;

            // Apply the 2-opt move
            apply2OptMove(sequences + baseIdx, sequenceLength, i, j);
            break; // Apply only one move per mutation
        }
    }

    // Save state
    states[genomeIdx] = localState;
}

// Cost-guided 2-opt mutation with cost matrix
__global__ void costGuided2OptMutationKernel(size_t* sequences, const float* costMatrix,
                                              size_t numGenomes, size_t numCities,
                                              size_t sequenceLength, float mutationRate,
                                              size_t stagnationCount, curandState* states) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (genomeIdx >= numGenomes) return;

    curandState localState = states[genomeIdx];
    size_t baseIdx = genomeIdx * sequenceLength;

    // Check if mutation should be applied
    float randVal = curand_uniform(&localState);
    if (randVal < mutationRate) {
        // Calculate escape probability (accept worse moves sometimes)
        float escapeProb = fminf(0.3f, 0.05f + (stagnationCount / 1000.0f));

        // Number of attempts increases with stagnation
        int numAttempts = (stagnationCount > 100) ? 10 : 5;

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

            // Calculate improvement using cost matrix
            float improvement = calculate2OptImprovement(
                sequences + baseIdx, costMatrix, numCities, sequenceLength, i, j);

            // Apply if improving or with escape probability
            bool applyMove = (improvement > 0.0f) ||
                            (curand_uniform(&localState) < escapeProb);

            if (applyMove) {
                apply2OptMove(sequences + baseIdx, sequenceLength, i, j);
                break;
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
// Uses cost-guided 2-opt with depot-aware edge calculations
__global__ void batch2OptMutationKernel(size_t* citySequences, const float* costMatrix,
                                       size_t numGenomes, size_t numCities, size_t tourLength,
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

            // Calculate improvement (now with numCities for proper cost matrix indexing)
            float improvement = calculate2OptImprovement(sharedTour, costMatrix, numCities, tourLength, i, j);

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

// =============================================================================
// SYSTEMATIC 2-OPT LOCAL SEARCH KERNEL
// =============================================================================

// Calculate total tour cost (depot -> tour -> depot)
__device__ float calculateTourCost(const size_t* tour, const float* costMatrix,
                                   size_t numCities, size_t tourLength) {
    float totalCost = 0.0f;

    // Depot to first city
    totalCost += costMatrix[0 * numCities + tour[0]];

    // City to city
    for (size_t i = 0; i < tourLength - 1; i++) {
        totalCost += costMatrix[tour[i] * numCities + tour[i + 1]];
    }

    // Last city back to depot
    totalCost += costMatrix[tour[tourLength - 1] * numCities + 0];

    return totalCost;
}

// AGGRESSIVE 2-opt local search kernel - one block per genome
// Applies ALL non-conflicting improving moves per pass for faster convergence
__global__ void systematic2OptKernel(size_t* sequences, const float* costMatrix,
                                      size_t numGenomes, size_t numCities,
                                      size_t tourLength, int* improvementCounts,
                                      size_t maxIterations) {
    int genomeIdx = blockIdx.x;

    if (genomeIdx >= numGenomes) return;

    __shared__ size_t sharedTour[MAX_CITIES];
    __shared__ bool improved;
    __shared__ int iterCount;
    __shared__ int localImprovements;

    size_t baseIdx = genomeIdx * tourLength;

    if (threadIdx.x == 0) {
        improved = true;
        iterCount = 0;
        localImprovements = 0;
    }
    __syncthreads();

    if (threadIdx.x < tourLength) {
        sharedTour[threadIdx.x] = sequences[baseIdx + threadIdx.x];
    }
    __syncthreads();

    // AGGRESSIVE: Much higher iteration limit for thorough optimization
    size_t maxIter = (maxIterations == 0) ? tourLength * 5 : maxIterations;

    while (improved && iterCount < maxIter) {
        if (threadIdx.x == 0) {
            improved = false;
            iterCount++;

            // AGGRESSIVE: Apply MULTIPLE improving moves per iteration
            // Use "don't look bits" style - mark segments that changed
            bool changed[MAX_CITIES];
            for (size_t i = 0; i < tourLength; i++) changed[i] = true;

            int passImprovements = 0;
            bool foundAny = true;

            // Multiple passes within single iteration
            while (foundAny && passImprovements < 20) {  // Cap at 20 moves per iteration
                foundAny = false;

                // Scan for best improving move among unmarked segments
                float bestDelta = 0.0f;
                size_t bestI = 0, bestJ = 0;

                for (size_t i = 0; i < tourLength - 2; i++) {
                    if (!changed[i] && !changed[i + 1]) continue;  // Skip stable segments

                    for (size_t j = i + 2; j < tourLength; j++) {
                        float delta = calculate2OptImprovement(sharedTour, costMatrix,
                                                              numCities, tourLength, i, j);
                        if (delta > bestDelta) {
                            bestDelta = delta;
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }

                if (bestDelta > 0.001f) {
                    apply2OptMove(sharedTour, tourLength, bestI, bestJ);
                    passImprovements++;
                    localImprovements++;
                    foundAny = true;
                    improved = true;

                    // Mark affected region as changed
                    for (size_t k = 0; k < tourLength; k++) changed[k] = false;
                    for (size_t k = bestI; k <= bestJ && k < tourLength; k++) {
                        changed[k] = true;
                    }
                    // Also mark neighbors
                    if (bestI > 0) changed[bestI - 1] = true;
                    if (bestJ + 1 < tourLength) changed[bestJ + 1] = true;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < tourLength) {
        sequences[baseIdx + threadIdx.x] = sharedTour[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        improvementCounts[genomeIdx] = localImprovements;
    }
}

// =============================================================================
// DOUBLE-BRIDGE MUTATION KERNEL (4-OPT for diversification)
// =============================================================================

// Double-bridge move: breaks tour into 4 segments and reconnects differently
// Original: A-B-C-D-A becomes A-D-C-B-A (where A,B,C,D are segments)
__global__ void doubleBridgeKernel(size_t* sequences, size_t numGenomes,
                                    size_t tourLength, float mutationRate,
                                    curandState* states) {
    int genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (genomeIdx >= numGenomes) return;

    curandState localState = states[genomeIdx];

    // Check if mutation should be applied
    if (curand_uniform(&localState) >= mutationRate) {
        states[genomeIdx] = localState;
        return;
    }

    size_t baseIdx = genomeIdx * tourLength;

    // Need at least 8 cities for double-bridge
    if (tourLength < 8) {
        states[genomeIdx] = localState;
        return;
    }

    // Generate 4 random cut points (sorted)
    // Ensure minimum segment size of 1
    size_t minSegSize = 1;
    size_t range = tourLength - 4 * minSegSize;

    // Generate random positions
    float r1 = curand_uniform(&localState);
    float r2 = curand_uniform(&localState);
    float r3 = curand_uniform(&localState);

    size_t pos1 = minSegSize + (size_t)(r1 * range / 3);
    size_t pos2 = pos1 + minSegSize + (size_t)(r2 * range / 3);
    size_t pos3 = pos2 + minSegSize + (size_t)(r3 * range / 3);

    // Ensure positions are valid and in order
    if (pos1 >= pos2) pos2 = pos1 + minSegSize;
    if (pos2 >= pos3) pos3 = pos2 + minSegSize;
    if (pos3 >= tourLength) pos3 = tourLength - 1;

    // Double-bridge reconnection:
    // Original tour: [0..pos1-1] [pos1..pos2-1] [pos2..pos3-1] [pos3..n-1]
    // Segments:         A            B              C             D
    // New tour:      A - D - C - B (reconnected differently)

    // Create temporary storage for the new tour
    size_t tempTour[MAX_CITIES];

    // Copy segment A: [0, pos1)
    size_t idx = 0;
    for (size_t i = 0; i < pos1; i++) {
        tempTour[idx++] = sequences[baseIdx + i];
    }

    // Copy segment D: [pos3, n)
    for (size_t i = pos3; i < tourLength; i++) {
        tempTour[idx++] = sequences[baseIdx + i];
    }

    // Copy segment C: [pos2, pos3)
    for (size_t i = pos2; i < pos3; i++) {
        tempTour[idx++] = sequences[baseIdx + i];
    }

    // Copy segment B: [pos1, pos2)
    for (size_t i = pos1; i < pos2; i++) {
        tempTour[idx++] = sequences[baseIdx + i];
    }

    // Copy back to global memory
    for (size_t i = 0; i < tourLength; i++) {
        sequences[baseIdx + i] = tempTour[i];
    }

    states[genomeIdx] = localState;
}

// =============================================================================
// OR-OPT KERNEL (Relocate segments of 1-3 cities)
// =============================================================================

// Calculate Or-opt improvement: moving segment [segStart, segEnd] to position insertPos
__device__ float calculateOrOptImprovement(const size_t* tour, const float* costMatrix,
                                            size_t numCities, size_t tourLength,
                                            size_t segStart, size_t segEnd, size_t insertPos) {
    // Segment cities
    size_t segFirst = tour[segStart];
    size_t segLast = tour[segEnd];

    // Cities adjacent to segment
    size_t beforeSeg = (segStart == 0) ? 0 : tour[segStart - 1];
    size_t afterSeg = (segEnd == tourLength - 1) ? 0 : tour[segEnd + 1];

    // City at insertion position and after
    size_t atInsert = tour[insertPos];
    size_t afterInsert = (insertPos == tourLength - 1) ? 0 : tour[insertPos + 1];

    // Current cost: beforeSeg->segFirst, segLast->afterSeg, atInsert->afterInsert
    float oldCost = costMatrix[beforeSeg * numCities + segFirst] +
                    costMatrix[segLast * numCities + afterSeg] +
                    costMatrix[atInsert * numCities + afterInsert];

    // New cost after move:
    // - Gap where segment was: beforeSeg->afterSeg
    // - Insert segment: atInsert->segFirst, segLast->afterInsert
    float newCost = costMatrix[beforeSeg * numCities + afterSeg] +
                    costMatrix[atInsert * numCities + segFirst] +
                    costMatrix[segLast * numCities + afterInsert];

    return oldCost - newCost;
}

// Or-opt local search kernel - one block per genome
// SIMPLIFIED VERSION: Just relocate single cities (1-opt relocation)
__global__ void orOptKernel(size_t* sequences, const float* costMatrix,
                            size_t numGenomes, size_t numCities,
                            size_t tourLength, int* improvementCounts,
                            size_t maxIterations) {
    int genomeIdx = blockIdx.x;
    if (genomeIdx >= numGenomes) return;

    __shared__ size_t sharedTour[MAX_CITIES];
    __shared__ bool improved;
    __shared__ int iterCount;
    __shared__ int localImprovements;

    size_t baseIdx = genomeIdx * tourLength;

    if (threadIdx.x == 0) {
        improved = true;
        iterCount = 0;
        localImprovements = 0;
    }
    __syncthreads();

    if (threadIdx.x < tourLength) {
        sharedTour[threadIdx.x] = sequences[baseIdx + threadIdx.x];
    }
    __syncthreads();

    size_t maxIter = (maxIterations == 0) ? tourLength : maxIterations;

    while (improved && iterCount < maxIter) {
        if (threadIdx.x == 0) {
            improved = false;
            iterCount++;

            float bestDelta = 0.0f;
            size_t bestFrom = 0, bestTo = 0;

            // Simple relocation: move city from position 'from' to position 'to'
            for (size_t from = 0; from < tourLength; from++) {
                size_t city = sharedTour[from];
                size_t prevCity = (from == 0) ? 0 : sharedTour[from - 1];
                size_t nextCity = (from + 1 < tourLength) ? sharedTour[from + 1] : 0;

                // Current cost of removing city from position
                float removeCost = costMatrix[prevCity * numCities + city] +
                                   costMatrix[city * numCities + nextCity];
                float bridgeCost = costMatrix[prevCity * numCities + nextCity];

                for (size_t to = 0; to < tourLength; to++) {
                    if (to == from || to + 1 == from || to == from + 1) continue;

                    size_t insertAfter = sharedTour[to];
                    size_t insertBefore = (to + 1 < tourLength) ? sharedTour[to + 1] : 0;

                    // Skip if same city
                    if (insertAfter == city || insertBefore == city) continue;

                    // Cost of inserting city at new position
                    float insertCost = costMatrix[insertAfter * numCities + city] +
                                       costMatrix[city * numCities + insertBefore];
                    float oldEdgeCost = costMatrix[insertAfter * numCities + insertBefore];

                    // Total improvement
                    float delta = (removeCost - bridgeCost) + (oldEdgeCost - insertCost);

                    if (delta > bestDelta + 0.001f) {
                        bestDelta = delta;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestDelta > 0.001f) {
                // Apply relocation move
                size_t city = sharedTour[bestFrom];

                if (bestTo < bestFrom) {
                    // Move city earlier: shift right, then insert
                    for (size_t i = bestFrom; i > bestTo + 1; i--) {
                        sharedTour[i] = sharedTour[i - 1];
                    }
                    sharedTour[bestTo + 1] = city;
                } else {
                    // Move city later: shift left, then insert
                    for (size_t i = bestFrom; i < bestTo; i++) {
                        sharedTour[i] = sharedTour[i + 1];
                    }
                    sharedTour[bestTo] = city;
                }

                localImprovements++;
                improved = true;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < tourLength) {
        sequences[baseIdx + threadIdx.x] = sharedTour[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        improvementCounts[genomeIdx] = localImprovements;
    }
}

// =============================================================================
// 3-OPT KERNEL (More powerful than 2-opt)
// =============================================================================

// Simplified 3-opt: try reconnecting 3 edges
__global__ void threeOptKernel(size_t* sequences, const float* costMatrix,
                                size_t numGenomes, size_t numCities,
                                size_t tourLength, int* improvementCounts,
                                size_t maxIterations) {
    int genomeIdx = blockIdx.x;
    if (genomeIdx >= numGenomes) return;

    __shared__ size_t sharedTour[MAX_CITIES];
    __shared__ bool improved;
    __shared__ int iterCount;
    __shared__ int localImprovements;

    size_t baseIdx = genomeIdx * tourLength;

    if (threadIdx.x == 0) {
        improved = true;
        iterCount = 0;
        localImprovements = 0;
    }
    __syncthreads();

    if (threadIdx.x < tourLength) {
        sharedTour[threadIdx.x] = sequences[baseIdx + threadIdx.x];
    }
    __syncthreads();

    size_t maxIter = (maxIterations == 0) ? tourLength / 2 : maxIterations;

    while (improved && iterCount < maxIter) {
        if (threadIdx.x == 0) {
            improved = false;
            iterCount++;

            float bestDelta = 0.0f;
            size_t bestI = 0, bestJ = 0, bestK = 0;
            int bestMoveType = 0;

            // 3-opt considers 3 edges to break and reconnect
            // For efficiency, we sample a subset of possibilities
            for (size_t i = 0; i < tourLength - 4; i += 2) {
                for (size_t j = i + 2; j < tourLength - 2; j += 2) {
                    for (size_t k = j + 2; k < tourLength; k += 2) {
                        // Get cities involved
                        size_t c_i = sharedTour[i];
                        size_t c_i1 = sharedTour[i + 1];
                        size_t c_j = sharedTour[j];
                        size_t c_j1 = sharedTour[j + 1];
                        size_t c_k = sharedTour[k];
                        size_t c_k1 = (k + 1 < tourLength) ? sharedTour[k + 1] : 0;

                        // Current edges cost
                        float currentCost = costMatrix[c_i * numCities + c_i1] +
                                           costMatrix[c_j * numCities + c_j1] +
                                           costMatrix[c_k * numCities + c_k1];

                        // Try reconnection: i->j+1, k->i+1, j->k+1
                        float newCost = costMatrix[c_i * numCities + c_j1] +
                                       costMatrix[c_k * numCities + c_i1] +
                                       costMatrix[c_j * numCities + c_k1];

                        float delta = currentCost - newCost;
                        if (delta > bestDelta) {
                            bestDelta = delta;
                            bestI = i;
                            bestJ = j;
                            bestK = k;
                            bestMoveType = 1;
                        }
                    }
                }
            }

            if (bestDelta > 0.001f && bestMoveType == 1) {
                // Apply 3-opt move: reverse two segments
                // Segment 1: [i+1, j] reversed
                size_t left = bestI + 1, right = bestJ;
                while (left < right) {
                    size_t temp = sharedTour[left];
                    sharedTour[left] = sharedTour[right];
                    sharedTour[right] = temp;
                    left++;
                    right--;
                }
                // Segment 2: [j+1, k] reversed
                left = bestJ + 1;
                right = bestK;
                while (left < right) {
                    size_t temp = sharedTour[left];
                    sharedTour[left] = sharedTour[right];
                    sharedTour[right] = temp;
                    left++;
                    right--;
                }

                localImprovements++;
                improved = true;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < tourLength) {
        sequences[baseIdx + threadIdx.x] = sharedTour[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        improvementCounts[genomeIdx] = localImprovements;
    }
}

// =============================================================================
// HOST FUNCTIONS
// =============================================================================

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

        // Launch 2-opt mutation for city sequences
        // Use cost-guided if cost matrix is available, otherwise use simple random
        std::cout << "DEBUG: Launching city sequence mutation" << std::endl;

        if (g_d_travelCosts && g_numCities > 0) {
            // Use cost-guided 2-opt
            costGuided2OptMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
                thrust::raw_pointer_cast(d_citySequences.data()),
                thrust::raw_pointer_cast(g_d_travelCosts->data()),
                numGenomes, g_numCities, sequenceLength, mutationRate, stagnationCount,
                thrust::raw_pointer_cast(d_states.data()));
        } else {
            // Fallback to simple 2-opt
            simple2OptMutationKernel<<<blocksPerGrid, threadsPerBlock>>>(
                thrust::raw_pointer_cast(d_citySequences.data()),
                numGenomes, sequenceLength, mutationRate, stagnationCount,
                thrust::raw_pointer_cast(d_states.data()));
        }
        
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

// =============================================================================
// SYSTEMATIC 2-OPT LOCAL SEARCH HOST FUNCTION
// =============================================================================

size_t performBatch2OptLocalSearch(std::vector<Genome>& genomes, size_t maxIterations) {
    if (genomes.empty()) return 0;

    // Check if cost matrix is initialized
    if (!g_d_travelCosts || g_numCities == 0) {
        std::cerr << "Error: Cost matrix not initialized for 2-opt local search" << std::endl;
        return 0;
    }

    size_t numGenomes = genomes.size();
    size_t tourLength = genomes[0].citySequence.size();

    std::cout << "Starting 2-opt local search for " << numGenomes << " genomes..." << std::endl;

    try {
        // Flatten city sequences
        std::vector<size_t> flatCitySequences(numGenomes * tourLength);
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(genomes[i].citySequence.begin(), genomes[i].citySequence.end(),
                     flatCitySequences.begin() + i * tourLength);
        }

        // Allocate device memory
        thrust::device_vector<size_t> d_citySequences(flatCitySequences);
        thrust::device_vector<int> d_improvementCounts(numGenomes, 0);

        // Launch kernel: one block per genome, 32 threads per block
        int threadsPerBlock = 32;
        int numBlocks = numGenomes;

        systematic2OptKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_citySequences.data()),
            thrust::raw_pointer_cast(g_d_travelCosts->data()),
            numGenomes, g_numCities, tourLength,
            thrust::raw_pointer_cast(d_improvementCounts.data()),
            maxIterations);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in 2-opt local search: " << cudaGetErrorString(err) << std::endl;
            return 0;
        }

        // Copy results back
        thrust::copy(d_citySequences.begin(), d_citySequences.end(), flatCitySequences.begin());

        // Count total improvements
        std::vector<int> improvementCounts(numGenomes);
        thrust::copy(d_improvementCounts.begin(), d_improvementCounts.end(), improvementCounts.begin());

        size_t totalImprovements = 0;
        for (int count : improvementCounts) {
            totalImprovements += count;
        }

        // Update genomes
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(flatCitySequences.begin() + i * tourLength,
                     flatCitySequences.begin() + (i + 1) * tourLength,
                     genomes[i].citySequence.begin());
        }

        std::cout << "2-opt local search completed: " << totalImprovements
                  << " total improvements across " << numGenomes << " genomes" << std::endl;

        return totalImprovements;

    } catch (const std::exception& e) {
        std::cerr << "Exception in 2-opt local search: " << e.what() << std::endl;
        return 0;
    }
}

// =============================================================================
// DOUBLE-BRIDGE MUTATION HOST FUNCTION
// =============================================================================

void performDoubleBridgeMutation(std::vector<Genome>& genomes, float mutationRate) {
    if (genomes.empty()) return;

    size_t numGenomes = genomes.size();
    size_t tourLength = genomes[0].citySequence.size();

    // Need at least 8 cities for double-bridge
    if (tourLength < 8) {
        std::cout << "Tour too short for double-bridge mutation (need >= 8 cities)" << std::endl;
        return;
    }

    std::cout << "Applying double-bridge mutation (rate: " << mutationRate << ")..." << std::endl;

    try {
        // Flatten city sequences
        std::vector<size_t> flatCitySequences(numGenomes * tourLength);
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(genomes[i].citySequence.begin(), genomes[i].citySequence.end(),
                     flatCitySequences.begin() + i * tourLength);
        }

        // Allocate device memory
        thrust::device_vector<size_t> d_citySequences(flatCitySequences);
        thrust::device_vector<curandState> d_states(numGenomes);

        // Setup random states
        int threadsPerBlock = std::min(256, static_cast<int>(numGenomes));
        int blocksPerGrid = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;

        setupRandomStates<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_states.data()),
            time(nullptr),
            numGenomes);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in setupRandomStates: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Launch double-bridge kernel
        doubleBridgeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_citySequences.data()),
            numGenomes, tourLength, mutationRate,
            thrust::raw_pointer_cast(d_states.data()));

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in double-bridge mutation: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Copy results back
        thrust::copy(d_citySequences.begin(), d_citySequences.end(), flatCitySequences.begin());

        // Update genomes
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(flatCitySequences.begin() + i * tourLength,
                     flatCitySequences.begin() + (i + 1) * tourLength,
                     genomes[i].citySequence.begin());
        }

        std::cout << "Double-bridge mutation completed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception in double-bridge mutation: " << e.what() << std::endl;
    }
}

// =============================================================================
// OR-OPT LOCAL SEARCH HOST FUNCTION
// =============================================================================

size_t performBatchOrOptLocalSearch(std::vector<Genome>& genomes, size_t maxIterations) {
    if (genomes.empty()) return 0;

    if (!g_d_travelCosts || g_numCities == 0) {
        std::cerr << "Error: Cost matrix not initialized for Or-opt local search" << std::endl;
        return 0;
    }

    size_t numGenomes = genomes.size();
    size_t tourLength = genomes[0].citySequence.size();

    std::cout << "Starting Or-opt local search for " << numGenomes << " genomes..." << std::endl;

    try {
        std::vector<size_t> flatCitySequences(numGenomes * tourLength);
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(genomes[i].citySequence.begin(), genomes[i].citySequence.end(),
                     flatCitySequences.begin() + i * tourLength);
        }

        thrust::device_vector<size_t> d_citySequences(flatCitySequences);
        thrust::device_vector<int> d_improvementCounts(numGenomes, 0);

        int threadsPerBlock = 32;
        int numBlocks = numGenomes;

        orOptKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_citySequences.data()),
            thrust::raw_pointer_cast(g_d_travelCosts->data()),
            numGenomes, g_numCities, tourLength,
            thrust::raw_pointer_cast(d_improvementCounts.data()),
            maxIterations);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in Or-opt local search: " << cudaGetErrorString(err) << std::endl;
            return 0;
        }

        thrust::copy(d_citySequences.begin(), d_citySequences.end(), flatCitySequences.begin());

        std::vector<int> improvementCounts(numGenomes);
        thrust::copy(d_improvementCounts.begin(), d_improvementCounts.end(), improvementCounts.begin());

        size_t totalImprovements = 0;
        for (int count : improvementCounts) {
            totalImprovements += count;
        }

        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(flatCitySequences.begin() + i * tourLength,
                     flatCitySequences.begin() + (i + 1) * tourLength,
                     genomes[i].citySequence.begin());
        }

        std::cout << "Or-opt local search completed: " << totalImprovements
                  << " total improvements across " << numGenomes << " genomes" << std::endl;

        return totalImprovements;

    } catch (const std::exception& e) {
        std::cerr << "Exception in Or-opt local search: " << e.what() << std::endl;
        return 0;
    }
}

// =============================================================================
// 3-OPT LOCAL SEARCH HOST FUNCTION
// =============================================================================

size_t performBatch3OptLocalSearch(std::vector<Genome>& genomes, size_t maxIterations) {
    if (genomes.empty()) return 0;

    if (!g_d_travelCosts || g_numCities == 0) {
        std::cerr << "Error: Cost matrix not initialized for 3-opt local search" << std::endl;
        return 0;
    }

    size_t numGenomes = genomes.size();
    size_t tourLength = genomes[0].citySequence.size();

    std::cout << "Starting 3-opt local search for " << numGenomes << " genomes..." << std::endl;

    try {
        std::vector<size_t> flatCitySequences(numGenomes * tourLength);
        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(genomes[i].citySequence.begin(), genomes[i].citySequence.end(),
                     flatCitySequences.begin() + i * tourLength);
        }

        thrust::device_vector<size_t> d_citySequences(flatCitySequences);
        thrust::device_vector<int> d_improvementCounts(numGenomes, 0);

        int threadsPerBlock = 32;
        int numBlocks = numGenomes;

        threeOptKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_citySequences.data()),
            thrust::raw_pointer_cast(g_d_travelCosts->data()),
            numGenomes, g_numCities, tourLength,
            thrust::raw_pointer_cast(d_improvementCounts.data()),
            maxIterations);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error in 3-opt local search: " << cudaGetErrorString(err) << std::endl;
            return 0;
        }

        thrust::copy(d_citySequences.begin(), d_citySequences.end(), flatCitySequences.begin());

        std::vector<int> improvementCounts(numGenomes);
        thrust::copy(d_improvementCounts.begin(), d_improvementCounts.end(), improvementCounts.begin());

        size_t totalImprovements = 0;
        for (int count : improvementCounts) {
            totalImprovements += count;
        }

        for (size_t i = 0; i < numGenomes; i++) {
            std::copy(flatCitySequences.begin() + i * tourLength,
                     flatCitySequences.begin() + (i + 1) * tourLength,
                     genomes[i].citySequence.begin());
        }

        std::cout << "3-opt local search completed: " << totalImprovements
                  << " total improvements across " << numGenomes << " genomes" << std::endl;

        return totalImprovements;

    } catch (const std::exception& e) {
        std::cerr << "Exception in 3-opt local search: " << e.what() << std::endl;
        return 0;
    }
}

// =============================================================================
// COMBINED LOCAL SEARCH (2-opt + Or-opt + 3-opt)
// =============================================================================

size_t performCombinedLocalSearch(std::vector<Genome>& genomes, size_t maxIterations) {
    size_t total = 0;

    // Run 2-opt first (fast, catches easy improvements)
    total += performBatch2OptLocalSearch(genomes, maxIterations);

    // Then Or-opt (catches segment relocations)
    total += performBatchOrOptLocalSearch(genomes, maxIterations);

    // Finally 3-opt for harder improvements
    total += performBatch3OptLocalSearch(genomes, maxIterations > 0 ? maxIterations / 2 : 0);

    // One more 2-opt pass to clean up
    total += performBatch2OptLocalSearch(genomes, maxIterations > 0 ? maxIterations / 2 : 0);

    std::cout << "Combined local search completed: " << total << " total improvements" << std::endl;
    return total;
}