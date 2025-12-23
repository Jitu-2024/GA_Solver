// mutation_gpu.cu: GPU-based mutation operators implementation

#include "mutation_gpu.cuh"
#include <cstdio>
#include <cfloat>

// Device helper: Reverse a segment of sequence
__device__ void reverseSegment(uint16_t* sequence, uint32_t start, uint32_t end,
                                uint32_t genomeIdx, uint32_t stride) {
    while (start < end) {
        uint16_t temp = sequence[start * stride + genomeIdx];
        sequence[start * stride + genomeIdx] = sequence[end * stride + genomeIdx];
        sequence[end * stride + genomeIdx] = temp;
        start++;
        end--;
    }
}

// 2-opt mutation kernel with escape mechanism (matching legacy behavior)
__global__ void twoOptMutationKernel(
    uint16_t* __restrict__ sequences,
    const float* __restrict__ travelCosts,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride,
    uint32_t stagnationCount  // Added for escape mechanism
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Calculate escape probability based on stagnation (like legacy)
    float escapeProb = fminf(0.5f, 0.05f + stagnationCount / 500.0f);

    bool moveApplied = false;
    uint32_t bestPos1 = 0, bestPos2 = 0;

    // Try to find a good move (5 attempts like legacy)
    for (int attempt = 0; attempt < 5 && !moveApplied; attempt++) {
        uint32_t pos1 = curand(&localState) % numCities;
        uint32_t pos2 = curand(&localState) % numCities;

        if (pos1 > pos2) {
            uint32_t temp = pos1;
            pos1 = pos2;
            pos2 = temp;
        }

        // Skip if segment too small
        if (pos2 <= pos1 + 1) continue;

        // Calculate move size as fraction of tour (like legacy)
        float moveSize = (float)(pos2 - pos1) / (float)numCities;

        // Prefer larger moves (>30% of tour) like legacy
        float applyProb = (moveSize > 0.3f) ? 0.8f : 0.3f;

        // Apply based on move quality OR escape probability
        float rand1 = curand_uniform(&localState);
        float rand2 = curand_uniform(&localState);

        if (rand1 < applyProb || rand2 < escapeProb) {
            bestPos1 = pos1;
            bestPos2 = pos2;
            moveApplied = true;
        }
    }

    // Force a random move if stagnation > 200 and nothing applied (like legacy)
    if (!moveApplied && stagnationCount > 200) {
        bestPos1 = curand(&localState) % numCities;
        bestPos2 = curand(&localState) % numCities;
        if (bestPos1 > bestPos2) {
            uint32_t temp = bestPos1;
            bestPos1 = bestPos2;
            bestPos2 = temp;
        }
        if (bestPos2 > bestPos1 + 1) {
            moveApplied = true;
        }
    }

    // Apply the move if one was selected
    if (moveApplied) {
        reverseSegment(sequences, bestPos1, bestPos2, genomeIdx, stride);
    }

    rngStates[genomeIdx] = localState;
}

// Double-bridge mutation kernel
__global__ void doubleBridgeMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Need at least 8 cities for double-bridge
    if (numCities < 8) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Select 4 random cut points
    uint32_t segmentSize = numCities / 4;
    uint32_t p1 = 1 + curand(&localState) % segmentSize;
    uint32_t p2 = p1 + 1 + curand(&localState) % segmentSize;
    uint32_t p3 = p2 + 1 + curand(&localState) % segmentSize;

    // Ensure valid ordering
    if (p1 >= p2) p2 = p1 + 1;
    if (p2 >= p3) p3 = p2 + 1;
    if (p3 >= numCities) p3 = numCities - 1;

    // Copy to local memory for rearrangement
    uint16_t temp[GPU_MAX_CITIES];
    for (uint32_t i = 0; i < numCities; ++i) {
        temp[i] = sequences[i * stride + genomeIdx];
    }

    // Double-bridge move: reconnect segments
    // Original: A-B-C-D
    // After:    A-C-B-D
    uint32_t idx = 0;

    // Segment A: [0, p1)
    for (uint32_t i = 0; i < p1; ++i) {
        sequences[idx * stride + genomeIdx] = temp[i];
        idx++;
    }

    // Segment C: [p2, p3)
    for (uint32_t i = p2; i < p3; ++i) {
        sequences[idx * stride + genomeIdx] = temp[i];
        idx++;
    }

    // Segment B: [p1, p2)
    for (uint32_t i = p1; i < p2; ++i) {
        sequences[idx * stride + genomeIdx] = temp[i];
        idx++;
    }

    // Segment D: [p3, numCities)
    for (uint32_t i = p3; i < numCities; ++i) {
        sequences[idx * stride + genomeIdx] = temp[i];
        idx++;
    }

    rngStates[genomeIdx] = localState;
}

// Swap mutation kernel
__global__ void swapMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Select two random positions
    uint32_t pos1 = curand(&localState) % seqLength;
    uint32_t pos2 = curand(&localState) % seqLength;

    if (pos1 != pos2) {
        uint16_t temp = sequences[pos1 * stride + genomeIdx];
        sequences[pos1 * stride + genomeIdx] = sequences[pos2 * stride + genomeIdx];
        sequences[pos2 * stride + genomeIdx] = temp;
    }

    rngStates[genomeIdx] = localState;
}

// Scramble mutation kernel
__global__ void scrambleMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Select segment to scramble
    uint32_t start = curand(&localState) % seqLength;
    uint32_t length = 2 + curand(&localState) % min(10U, seqLength - start);
    uint32_t end = min(start + length, seqLength);

    // Fisher-Yates shuffle of the segment
    for (uint32_t i = end - 1; i > start; --i) {
        uint32_t j = start + curand(&localState) % (i - start + 1);
        uint16_t temp = sequences[i * stride + genomeIdx];
        sequences[i * stride + genomeIdx] = sequences[j * stride + genomeIdx];
        sequences[j * stride + genomeIdx] = temp;
    }

    rngStates[genomeIdx] = localState;
}

// Inversion mutation kernel
__global__ void inversionMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t seqLength,
    uint32_t numGenomes,
    uint32_t stride
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Select segment to reverse
    uint32_t start = curand(&localState) % seqLength;
    uint32_t end = curand(&localState) % seqLength;

    if (start > end) {
        uint32_t temp = start;
        start = end;
        end = temp;
    }

    reverseSegment(sequences, start, end, genomeIdx, stride);

    rngStates[genomeIdx] = localState;
}

// 3-opt style mutation kernel - more disruptive than 2-opt
// Performs one of 4 possible segment operations
__global__ void threeOptMutationKernel(
    uint16_t* __restrict__ sequences,
    curandState* __restrict__ rngStates,
    float mutationRate,
    uint32_t numCities,
    uint32_t numGenomes,
    uint32_t stride
) {
    uint32_t genomeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIdx >= numGenomes) return;

    curandState localState = rngStates[genomeIdx];

    // Check if mutation should be applied
    float r = curand_uniform(&localState);
    if (r >= mutationRate) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Need at least 6 cities for 3-opt
    if (numCities < 6) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Select 3 random cut points
    uint32_t p1 = curand(&localState) % numCities;
    uint32_t p2 = curand(&localState) % numCities;
    uint32_t p3 = curand(&localState) % numCities;

    // Sort the points
    if (p1 > p2) { uint32_t t = p1; p1 = p2; p2 = t; }
    if (p2 > p3) { uint32_t t = p2; p2 = p3; p3 = t; }
    if (p1 > p2) { uint32_t t = p1; p1 = p2; p2 = t; }

    // Ensure distinct points
    if (p1 == p2 || p2 == p3) {
        rngStates[genomeIdx] = localState;
        return;
    }

    // Randomly choose a 3-opt move type
    uint32_t moveType = curand(&localState) % 4;

    switch(moveType) {
        case 0: // Reverse first segment [p1, p2)
            reverseSegment(sequences, p1, p2 - 1, genomeIdx, stride);
            break;
        case 1: // Reverse second segment [p2, p3)
            reverseSegment(sequences, p2, p3 - 1, genomeIdx, stride);
            break;
        case 2: // Reverse both segments
            reverseSegment(sequences, p1, p2 - 1, genomeIdx, stride);
            reverseSegment(sequences, p2, p3 - 1, genomeIdx, stride);
            break;
        case 3: // Rotate segments (swap segments)
            {
                // Copy to temp buffer
                uint16_t temp[GPU_MAX_CITIES];
                for (uint32_t i = 0; i < numCities; ++i) {
                    temp[i] = sequences[i * stride + genomeIdx];
                }

                // Rebuild: [0,p1) + [p2,p3) + [p1,p2) + [p3,end)
                uint32_t idx = 0;

                // First segment [0, p1)
                for (uint32_t i = 0; i < p1; ++i) {
                    sequences[idx * stride + genomeIdx] = temp[i];
                    idx++;
                }

                // Moved segment [p2, p3)
                for (uint32_t i = p2; i < p3; ++i) {
                    sequences[idx * stride + genomeIdx] = temp[i];
                    idx++;
                }

                // Middle segment [p1, p2)
                for (uint32_t i = p1; i < p2; ++i) {
                    sequences[idx * stride + genomeIdx] = temp[i];
                    idx++;
                }

                // Last segment [p3, end)
                for (uint32_t i = p3; i < numCities; ++i) {
                    sequences[idx * stride + genomeIdx] = temp[i];
                    idx++;
                }
            }
            break;
    }

    rngStates[genomeIdx] = localState;
}

// Main mutation function - matches legacy stagnation-aware behavior
cudaError_t performGPUMutations(
    GPUPopulation& population,
    GPUProblemData& problemData,
    GPURngState& rngState,
    uint32_t numGenomes,
    float mutationRate,
    uint32_t stagnationCount,
    bool isOffspring
) {
    if (numGenomes == 0) return cudaSuccess;

    uint32_t numCities = population.numCities;
    uint32_t numJobs = population.numJobs;

    // Select buffers
    uint16_t* citySeq = isOffspring ? population.d_offspringCity : population.d_citySequences;
    uint16_t* jobSeq = isOffspring ? population.d_offspringJob : population.d_jobSequences;
    uint16_t* pickupSeq = isOffspring ? population.d_offspringPickup : population.d_pickupSequences;
    uint32_t stride = isOffspring ? numGenomes : population.popSize;

    // Kernel configuration
    int threadsPerBlock = 256;
    int numBlocks = (numGenomes + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t err;

    // Mutation operates on full sequences
    uint32_t mutationLength = numCities;
    uint32_t jobMutationLength = numJobs;

    // ========================================================================
    // STEP 2: Stagnation-aware mutation strategy (matching legacy behavior)
    // ========================================================================
    //
    // Legacy thresholds:
    // - stagnation < 100: standard 2-opt
    // - stagnation 100-300: 70% 2-opt, 30% 3-opt
    // - stagnation > 300: 40% 3-opt, 30% double-bridge, 30% multiple 2-opts
    //
    // Legacy also has:
    // - escapeProb = min(0.5, 0.05 + stagnation/500.0)
    // - Job mutation rate = mutationRate * (1 + stagnation/500)

    // Calculate adaptive rates like legacy
    float escapeProb = fminf(0.5f, 0.05f + stagnationCount / 500.0f);
    float effectiveMutationRate = mutationRate + escapeProb * 0.3f;  // Boost with escape
    float jobMutationRate = mutationRate * (1.0f + stagnationCount / 500.0f);
    jobMutationRate = fminf(jobMutationRate, 0.8f);  // Cap at 0.8

    if (stagnationCount < 100) {
        // Low stagnation: standard 2-opt
        twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            problemData.d_travelCosts,
            rngState.d_states,
            effectiveMutationRate,
            mutationLength,
            numGenomes,
            stride,
            stagnationCount
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

    } else if (stagnationCount < 300) {
        // Medium stagnation: mix of 2-opt (70%) and 3-opt (30%)
        // Apply 2-opt with 70% chance (0.7 * rate)
        twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            problemData.d_travelCosts,
            rngState.d_states,
            effectiveMutationRate * 0.7f,
            mutationLength,
            numGenomes,
            stride,
            stagnationCount
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

        // Apply 3-opt with 30% chance (0.3 * rate)
        threeOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            rngState.d_states,
            effectiveMutationRate * 0.3f,
            mutationLength,
            numGenomes,
            stride
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

    } else {
        // High stagnation: 40% 3-opt, 30% double-bridge, 30% multiple 2-opts
        float boostedRate = fminf(effectiveMutationRate * 1.5f, 0.9f);

        // Apply 3-opt with 40% chance
        threeOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            rngState.d_states,
            boostedRate * 0.4f,
            mutationLength,
            numGenomes,
            stride
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

        // Apply double-bridge with 30% chance
        doubleBridgeMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            rngState.d_states,
            boostedRate * 0.3f,
            mutationLength,
            numGenomes,
            stride
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

        // Apply multiple 2-opts (two passes) with 30% chance
        twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            problemData.d_travelCosts,
            rngState.d_states,
            boostedRate * 0.3f,
            mutationLength,
            numGenomes,
            stride,
            stagnationCount
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

        // Second 2-opt pass (like legacy's multiple 2-opts)
        twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
            citySeq,
            problemData.d_travelCosts,
            rngState.d_states,
            boostedRate * 0.15f,  // Half rate for second pass
            mutationLength,
            numGenomes,
            stride,
            stagnationCount
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
    }

    // Apply swap mutation to job sequences with scaled rate
    // Multiple swaps for high stagnation (like legacy)
    int numSwaps = (stagnationCount > 200) ? 2 : 1;
    for (int s = 0; s < numSwaps; s++) {
        swapMutationKernel<<<numBlocks, threadsPerBlock>>>(
            jobSeq,
            rngState.d_states,
            jobMutationRate,
            jobMutationLength,
            numGenomes,
            stride
        );
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
    }

    // Apply mutation to pickup sequences if mode == 1
    if (population.mode == 1 && pickupSeq != nullptr) {
        if (stagnationCount < 100) {
            twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
                pickupSeq,
                problemData.d_travelCosts,
                rngState.d_states,
                effectiveMutationRate,
                mutationLength,
                numGenomes,
                stride,
                stagnationCount
            );
        } else if (stagnationCount < 300) {
            twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
                pickupSeq,
                problemData.d_travelCosts,
                rngState.d_states,
                effectiveMutationRate,
                mutationLength,
                numGenomes,
                stride,
                stagnationCount
            );
        } else {
            // High stagnation: 50% 3-opt, 50% 2-opt for pickups
            threeOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
                pickupSeq,
                rngState.d_states,
                effectiveMutationRate * 0.5f,
                mutationLength,
                numGenomes,
                stride
            );
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) return err;

            twoOptMutationKernel<<<numBlocks, threadsPerBlock>>>(
                pickupSeq,
                problemData.d_travelCosts,
                rngState.d_states,
                effectiveMutationRate * 0.5f,
                mutationLength,
                numGenomes,
                stride,
                stagnationCount
            );
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}
