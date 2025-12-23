// eax_gpu.cu: Scalable Edge Assembly Crossover implementation
// Uses O(n) adjacency lists instead of O(n^2) adjacency matrix

#include "eax_gpu.cuh"
#include <cstdio>

// Kernel 1: Build adjacency lists from parent tours
// Each thread handles one position in the tour
__global__ void buildAdjacencyListsKernel(
    const uint16_t* __restrict__ parent1Tours,
    const uint16_t* __restrict__ parent2Tours,
    CityAdjacency* __restrict__ adjacencyLists,
    uint32_t numCities,
    uint32_t numPairs,
    uint32_t parent1Stride,
    uint32_t parent2Stride
) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t posIdx = threadIdx.x;

    if (pairIdx >= numPairs || posIdx >= numCities) return;

    // Calculate indices for prev and next positions (circular tour)
    uint32_t prevPos = (posIdx + numCities - 1) % numCities;
    uint32_t nextPos = (posIdx + 1) % numCities;

    // Get cities at current, prev, next positions for parent1
    uint16_t city1 = parent1Tours[posIdx * parent1Stride + pairIdx];
    uint16_t prev1 = parent1Tours[prevPos * parent1Stride + pairIdx];
    uint16_t next1 = parent1Tours[nextPos * parent1Stride + pairIdx];

    // Get cities at current, prev, next positions for parent2
    uint16_t city2 = parent2Tours[posIdx * parent2Stride + pairIdx];
    uint16_t prev2 = parent2Tours[prevPos * parent2Stride + pairIdx];
    uint16_t next2 = parent2Tours[nextPos * parent2Stride + pairIdx];

    // Store neighbors for city from parent1 (city1's neighbors in parent1 tour)
    // Each city appears once in tour, so each thread writes to a unique city slot
    CityAdjacency& adj1 = adjacencyLists[city1 * numPairs + pairIdx];
    adj1.neighbors[0] = prev1;  // Parent1 predecessor
    adj1.neighbors[1] = next1;  // Parent1 successor

    // Store neighbors for city from parent2
    CityAdjacency& adj2 = adjacencyLists[city2 * numPairs + pairIdx];
    adj2.neighbors[2] = prev2;  // Parent2 predecessor
    adj2.neighbors[3] = next2;  // Parent2 successor
}

// Kernel 2: Find alternating AB-cycles
// One block per parent pair, sequential cycle finding within block
__global__ void findAlternatingCyclesKernel(
    const CityAdjacency* __restrict__ adjacencyLists,
    uint8_t* __restrict__ visited,
    uint16_t* __restrict__ cycleBuffer,
    uint8_t* __restrict__ cycleParentFlags,
    uint16_t* __restrict__ cycleLengths,
    uint16_t* __restrict__ numCycles,
    uint32_t numCities,
    uint32_t numPairs
) {
    uint32_t pairIdx = blockIdx.x;
    if (pairIdx >= numPairs) return;

    // Only thread 0 finds cycles (sequential within pair, parallel across pairs)
    if (threadIdx.x != 0) return;

    // Clear visited flags for this pair
    for (uint32_t i = 0; i < numCities; ++i) {
        visited[i * numPairs + pairIdx] = 0;
    }

    uint16_t cycleCount = 0;
    uint32_t cycleBaseOffset = pairIdx;  // Base offset for this pair's cycle data

    // Try to find cycles starting from each unvisited city
    for (uint16_t startCity = 1; startCity <= numCities && cycleCount < GPU_MAX_CYCLES; ++startCity) {
        if (visited[startCity * numPairs + pairIdx]) continue;

        // Try to find an alternating cycle starting from this city
        uint16_t currentCity = startCity;
        uint8_t currentParent = 0;  // Start with parent1 edge
        uint16_t cycleLen = 0;
        bool validCycle = true;

        // Temporary storage for this cycle
        uint16_t tempCycle[GPU_MAX_CYCLE_LENGTH];
        uint8_t tempParents[GPU_MAX_CYCLE_LENGTH];

        // Mark start city as visited
        visited[startCity * numPairs + pairIdx] = 1;

        do {
            if (cycleLen >= GPU_MAX_CYCLE_LENGTH) {
                validCycle = false;
                break;
            }

            // Record current city in cycle
            tempCycle[cycleLen] = currentCity;
            tempParents[cycleLen] = currentParent;

            // Get adjacency for current city
            const CityAdjacency& adj = adjacencyLists[currentCity * numPairs + pairIdx];

            // Find next city using the OTHER parent's edge
            uint8_t nextParent = 1 - currentParent;
            int neighborOffset = nextParent * 2;  // 0 or 2

            // Get both neighbors from next parent
            uint16_t neighbor1 = adj.neighbors[neighborOffset];
            uint16_t neighbor2 = adj.neighbors[neighborOffset + 1];

            // Choose an unvisited neighbor (or startCity to complete cycle)
            uint16_t nextCity = 0;
            if (neighbor1 == startCity && cycleLen >= 3) {
                nextCity = neighbor1;  // Can complete cycle
            } else if (neighbor2 == startCity && cycleLen >= 3) {
                nextCity = neighbor2;  // Can complete cycle
            } else if (!visited[neighbor1 * numPairs + pairIdx] && neighbor1 != 0) {
                nextCity = neighbor1;
                visited[neighbor1 * numPairs + pairIdx] = 1;
            } else if (!visited[neighbor2 * numPairs + pairIdx] && neighbor2 != 0) {
                nextCity = neighbor2;
                visited[neighbor2 * numPairs + pairIdx] = 1;
            } else {
                // No valid next city - cycle broken
                validCycle = false;
                break;
            }

            cycleLen++;
            currentCity = nextCity;
            currentParent = nextParent;

        } while (currentCity != startCity);

        // Store valid cycle (minimum length 3 like legacy)
        if (validCycle && currentCity == startCity && cycleLen >= 3) {
            // Copy cycle to global memory
            uint32_t cycleOffset = cycleCount * GPU_MAX_CYCLE_LENGTH * numPairs + cycleBaseOffset;
            for (uint16_t i = 0; i < cycleLen; ++i) {
                cycleBuffer[i * numPairs + cycleOffset] = tempCycle[i];
                cycleParentFlags[i * numPairs + cycleOffset] = tempParents[i];
            }
            cycleLengths[cycleCount * numPairs + pairIdx] = cycleLen;
            cycleCount++;
        }
    }

    numCycles[pairIdx] = cycleCount;
}

// Kernel 3: Evaluate assemblies and select best
// Each thread evaluates one assembly combination
// useCostAware=true: use actual travel costs to prefer assemblies with shorter edges
// useCostAware=false: use uniform costs (pure structural preference)
__global__ void evaluateAssembliesKernel(
    const uint16_t* __restrict__ cycleBuffer,
    const uint8_t* __restrict__ cycleParentFlags,
    const uint16_t* __restrict__ cycleLengths,
    const uint16_t* __restrict__ numCycles,
    const uint16_t* __restrict__ parent1Tours,
    const uint16_t* __restrict__ parent2Tours,
    const float* __restrict__ travelCosts,
    uint32_t* __restrict__ bestAssembly,
    float* __restrict__ assemblyCosts,
    uint32_t numCities,
    uint32_t numPairs,
    uint32_t parent1Stride,
    uint32_t parent2Stride,
    bool useCostAware
) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t assemblyIdx = threadIdx.x;

    if (pairIdx >= numPairs) return;

    uint16_t cycleCount = numCycles[pairIdx];

    // Limit assemblies to prevent exponential blowup
    uint32_t maxAssemblies = min(256U, 1U << min(cycleCount, (uint16_t)16));

    __shared__ float sharedCosts[256];
    __shared__ uint32_t sharedAssemblies[256];

    float cost = 1e30f;  // Large value for invalid assemblies

    if (assemblyIdx < maxAssemblies && cycleCount > 0) {
        // Evaluate this assembly using ACTUAL travel costs
        // Assembly is represented as bitmask: bit i = 1 means include cycle i
        // Including a cycle means using parent2's edges for cities in that cycle
        // Not including means keeping parent1's edges
        //
        // We evaluate the cost contribution of edges that will be ADDED to the tour
        // by including each cycle (parent2 edges replace parent1 edges)
        cost = 0.0f;
        bool validAssembly = true;

        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount && validAssembly; ++cycleIdx) {
            bool includeCycle = (assemblyIdx >> cycleIdx) & 1;
            uint16_t cycleLen = cycleLengths[cycleIdx * numPairs + pairIdx];
            uint32_t cycleOffset = cycleIdx * GPU_MAX_CYCLE_LENGTH * numPairs + pairIdx;

            // For each edge in the cycle, compute the cost difference
            // between using parent2's edge vs parent1's edge
            for (uint16_t i = 0; i < cycleLen; ++i) {
                uint16_t city = cycleBuffer[i * numPairs + cycleOffset];
                uint16_t nextCity = cycleBuffer[((i + 1) % cycleLen) * numPairs + cycleOffset];
                uint8_t parentFlag = cycleParentFlags[i * numPairs + cycleOffset];

                if (city > 0 && city <= numCities && nextCity > 0 && nextCity <= numCities) {
                    // Get edge cost: actual travel cost if cost-aware, uniform 1.0 otherwise
                    float edgeCost = useCostAware ? travelCosts[city * (numCities + 1) + nextCity] : 1.0f;

                    if (includeCycle) {
                        // If including cycle: add cost for parent2 edges (parentFlag == 1)
                        // and subtract cost for parent1 edges we're removing (parentFlag == 0)
                        if (parentFlag == 1) {
                            cost += edgeCost;  // Adding this parent2 edge
                        } else {
                            cost -= edgeCost;  // Removing this parent1 edge
                        }
                    } else {
                        // If NOT including cycle: add cost for parent1 edges (parentFlag == 0)
                        // and subtract cost for parent2 edges we're not using
                        if (parentFlag == 0) {
                            cost += edgeCost;  // Keeping this parent1 edge
                        } else {
                            cost -= edgeCost;  // Not using this parent2 edge
                        }
                    }
                } else {
                    validAssembly = false;
                }
            }
        }

        // Small tie-breaker to prefer simpler assemblies
        cost += assemblyIdx * 0.0001f;

        if (!validAssembly) cost = 1e30f;
    }

    sharedCosts[threadIdx.x] = cost;
    sharedAssemblies[threadIdx.x] = assemblyIdx;
    __syncthreads();

    // Parallel reduction to find minimum cost
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
            if (sharedCosts[threadIdx.x + stride] < sharedCosts[threadIdx.x]) {
                sharedCosts[threadIdx.x] = sharedCosts[threadIdx.x + stride];
                sharedAssemblies[threadIdx.x] = sharedAssemblies[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bestAssembly[pairIdx] = sharedAssemblies[0];
        assemblyCosts[pairIdx] = sharedCosts[0];
    }
}

// Kernel 4: Construct offspring tour and job sequence from selected assembly
__global__ void constructOffspringTourKernel(
    const CityAdjacency* __restrict__ adjacencyLists,
    const uint16_t* __restrict__ cycleBuffer,
    const uint8_t* __restrict__ cycleParentFlags,
    const uint16_t* __restrict__ cycleLengths,
    const uint16_t* __restrict__ numCycles,
    const uint32_t* __restrict__ bestAssembly,
    const uint16_t* __restrict__ parent1Tours,
    const uint16_t* __restrict__ parent2Tours,
    const uint16_t* __restrict__ parent1Jobs,
    const uint16_t* __restrict__ parent2Jobs,
    uint16_t* __restrict__ offspringTours,
    uint16_t* __restrict__ offspringJobs,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t numPairs,
    uint32_t parent1Stride,
    uint32_t parent2Stride,
    uint32_t offspringStride
) {
    uint32_t pairIdx = blockIdx.x;
    if (pairIdx >= numPairs) return;

    // Only thread 0 constructs offspring (sequential per pair)
    if (threadIdx.x != 0) return;

    uint32_t assembly = bestAssembly[pairIdx];
    uint16_t cycleCount = numCycles[pairIdx];

    // Create a mapping: for each city, which parent's edge to use
    // 0 = use parent1's edge, 1 = use parent2's edge
    // Default: use parent1
    uint8_t useParent2[GPU_MAX_CITIES];
    for (uint32_t i = 0; i <= numCities; ++i) {
        useParent2[i] = 0;
    }

    // Build inverse mapping: city -> position in each parent
    uint16_t cityToPos1[GPU_MAX_CITIES];
    uint16_t cityToPos2[GPU_MAX_CITIES];
    for (uint32_t pos = 0; pos < numCities; ++pos) {
        uint16_t city1 = parent1Tours[pos * parent1Stride + pairIdx];
        uint16_t city2 = parent2Tours[pos * parent2Stride + pairIdx];
        if (city1 <= numCities) cityToPos1[city1] = pos;
        if (city2 <= numCities) cityToPos2[city2] = pos;
    }

    // Mark cities in selected cycles to use parent2 edges
    for (uint16_t cycleIdx = 0; cycleIdx < cycleCount; ++cycleIdx) {
        bool includeCycle = (assembly >> cycleIdx) & 1;
        if (includeCycle) {
            uint16_t cycleLen = cycleLengths[cycleIdx * numPairs + pairIdx];
            uint32_t cycleOffset = cycleIdx * GPU_MAX_CYCLE_LENGTH * numPairs + pairIdx;

            for (uint16_t i = 0; i < cycleLen; ++i) {
                uint16_t city = cycleBuffer[i * numPairs + cycleOffset];
                uint8_t parentFlag = cycleParentFlags[i * numPairs + cycleOffset];

                // If this edge in the cycle comes from parent2, mark city
                if (parentFlag == 1 && city <= numCities) {
                    useParent2[city] = 1;
                }
            }
        }
    }

    // Build offspring tour by following edges
    // Start from city in position 0 of parent1
    uint16_t startCity = parent1Tours[0 * parent1Stride + pairIdx];
    uint16_t currentCity = startCity;
    uint16_t tourLen = 0;

    uint8_t visitedTour[GPU_MAX_CITIES];
    for (uint32_t i = 0; i <= numCities; ++i) {
        visitedTour[i] = 0;
    }

    do {
        // Add current city to offspring
        offspringTours[tourLen * offspringStride + pairIdx] = currentCity;

        // Assign job: use same parent that contributed the city's edge
        // Look up the job at this city's position in the appropriate parent
        if (useParent2[currentCity]) {
            uint16_t pos2 = cityToPos2[currentCity];
            offspringJobs[tourLen * offspringStride + pairIdx] = parent2Jobs[pos2 * parent2Stride + pairIdx];
        } else {
            uint16_t pos1 = cityToPos1[currentCity];
            offspringJobs[tourLen * offspringStride + pairIdx] = parent1Jobs[pos1 * parent1Stride + pairIdx];
        }

        visitedTour[currentCity] = 1;
        tourLen++;

        if (tourLen >= numCities) break;

        // Get next city based on parent selection
        const CityAdjacency& adj = adjacencyLists[currentCity * numPairs + pairIdx];

        uint16_t nextCity = 0;
        if (useParent2[currentCity]) {
            // Use parent2's successor edge
            uint16_t n1 = adj.neighbors[2];  // parent2 prev
            uint16_t n2 = adj.neighbors[3];  // parent2 next

            // Choose unvisited neighbor
            if (!visitedTour[n2] && n2 != 0) {
                nextCity = n2;
            } else if (!visitedTour[n1] && n1 != 0) {
                nextCity = n1;
            }
        } else {
            // Use parent1's successor edge
            uint16_t n1 = adj.neighbors[0];  // parent1 prev
            uint16_t n2 = adj.neighbors[1];  // parent1 next

            // Choose unvisited neighbor
            if (!visitedTour[n2] && n2 != 0) {
                nextCity = n2;
            } else if (!visitedTour[n1] && n1 != 0) {
                nextCity = n1;
            }
        }

        // Fallback: find any unvisited city if construction fails
        if (nextCity == 0) {
            for (uint16_t c = 1; c <= numCities; ++c) {
                if (!visitedTour[c]) {
                    nextCity = c;
                    break;
                }
            }
        }

        if (nextCity == 0) break;  // Should not happen
        currentCity = nextCity;

    } while (tourLen < numCities);

    // Fix job duplicates that may have been created by EAX inheritance
    // Track which jobs are used and at which position
    uint8_t jobUsed[GPU_MAX_CITIES];
    uint16_t jobPosition[GPU_MAX_CITIES];  // Which position has this job
    for (uint32_t i = 0; i <= numJobs; ++i) {
        jobUsed[i] = 0;
        jobPosition[i] = 0xFFFF;  // Invalid position
    }

    // First pass: detect duplicates and mark first occurrence
    for (uint16_t pos = 0; pos < tourLen; ++pos) {
        uint16_t job = offspringJobs[pos * offspringStride + pairIdx];
        if (job > 0 && job <= numJobs) {
            if (jobUsed[job]) {
                // Duplicate! Mark this position for reassignment
                offspringJobs[pos * offspringStride + pairIdx] = 0;  // Clear for now
            } else {
                jobUsed[job] = 1;
                jobPosition[job] = pos;
            }
        }
    }

    // Second pass: assign unused jobs to positions that need them
    uint16_t nextUnusedJob = 1;
    for (uint16_t pos = 0; pos < tourLen; ++pos) {
        uint16_t job = offspringJobs[pos * offspringStride + pairIdx];
        if (job == 0) {
            // Find next unused job
            while (nextUnusedJob <= numJobs && jobUsed[nextUnusedJob]) {
                nextUnusedJob++;
            }
            if (nextUnusedJob <= numJobs) {
                offspringJobs[pos * offspringStride + pairIdx] = nextUnusedJob;
                jobUsed[nextUnusedJob] = 1;
                nextUnusedJob++;
            }
        }
    }

    // If tour is incomplete, fill remaining positions
    if (tourLen < numCities) {
        for (uint16_t c = 1; c <= numCities; ++c) {
            if (!visitedTour[c]) {
                offspringTours[tourLen * offspringStride + pairIdx] = c;
                // Find an unused job
                while (nextUnusedJob <= numJobs && jobUsed[nextUnusedJob]) {
                    nextUnusedJob++;
                }
                if (nextUnusedJob <= numJobs) {
                    offspringJobs[tourLen * offspringStride + pairIdx] = nextUnusedJob;
                    jobUsed[nextUnusedJob] = 1;
                    nextUnusedJob++;
                }
                tourLen++;
                if (tourLen >= numCities) break;
            }
        }
    }
}

// Helper kernel: Gather parent tours based on selection indices
__global__ void gatherParentToursKernel(
    const uint16_t* __restrict__ populationTours,
    const uint32_t* __restrict__ parentIndices,
    uint16_t* __restrict__ gatheredTours,
    uint32_t numCities,
    uint32_t popSize,
    uint32_t numPairs
) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t posIdx = threadIdx.x;

    if (pairIdx >= numPairs || posIdx >= numCities) return;

    uint32_t parentIdx = parentIndices[pairIdx];

    // Copy from SoA population to gathered buffer
    uint16_t city = populationTours[posIdx * popSize + parentIdx];
    gatheredTours[posIdx * numPairs + pairIdx] = city;
}

// Main EAX function
cudaError_t performGPUEAXCrossover(
    GPUPopulation& population,
    GPUProblemData& problemData,
    EAXWorkspace& workspace,
    uint32_t numPairs,
    bool useCostAware
) {
    if (numPairs == 0) return cudaSuccess;

    uint32_t numCities = population.numCities;
    uint32_t popSize = population.popSize;

    // Kernel configuration
    int threadsPerBlock = min(256, (int)numCities);
    dim3 gridDim(numPairs);
    dim3 blockDim(threadsPerBlock);

    // Clear visited array - cities are 1-indexed, so need (numCities + 1) elements
    cudaMemset(workspace.d_visited, 0, (numCities + 1) * numPairs * sizeof(uint8_t));

    uint32_t numJobs = population.numJobs;

    // Step 1: Gather parent tours based on selection indices
    // Gather parent1 city tours to dedicated workspace buffer
    gatherParentToursKernel<<<gridDim, blockDim>>>(
        population.d_citySequences,
        population.d_parent1Indices,
        workspace.d_parent1Gathered,
        numCities,
        popSize,
        numPairs
    );

    // Gather parent2 city tours to dedicated workspace buffer
    gatherParentToursKernel<<<gridDim, blockDim>>>(
        population.d_citySequences,
        population.d_parent2Indices,
        workspace.d_parent2Gathered,
        numCities,
        popSize,
        numPairs
    );

    // Gather parent1 job sequences
    dim3 jobGridDim(numPairs);
    dim3 jobBlockDim(min(256, (int)numJobs));
    gatherParentToursKernel<<<jobGridDim, jobBlockDim>>>(
        population.d_jobSequences,
        population.d_parent1Indices,
        workspace.d_parent1Jobs,
        numJobs,
        popSize,
        numPairs
    );

    // Gather parent2 job sequences
    gatherParentToursKernel<<<jobGridDim, jobBlockDim>>>(
        population.d_jobSequences,
        population.d_parent2Indices,
        workspace.d_parent2Jobs,
        numJobs,
        popSize,
        numPairs
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EAX Error: Parent gathering failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Step 2: Build adjacency lists from both parents
    buildAdjacencyListsKernel<<<gridDim, blockDim>>>(
        workspace.d_parent1Gathered,
        workspace.d_parent2Gathered,
        workspace.d_adjacencyLists,
        numCities,
        numPairs,
        numPairs,  // stride for gathered parents
        numPairs
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EAX Error: Adjacency list building failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Step 3: Find alternating cycles
    findAlternatingCyclesKernel<<<numPairs, 1>>>(
        workspace.d_adjacencyLists,
        workspace.d_visited,
        workspace.cycles.d_cycleBuffer,
        workspace.cycles.d_cycleParentFlags,
        workspace.cycles.d_cycleLengths,
        workspace.cycles.d_numCycles,
        numCities,
        numPairs
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EAX Error: Cycle finding failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Step 4: Evaluate assemblies
    dim3 evalBlockDim(256);
    evaluateAssembliesKernel<<<numPairs, evalBlockDim>>>(
        workspace.cycles.d_cycleBuffer,
        workspace.cycles.d_cycleParentFlags,
        workspace.cycles.d_cycleLengths,
        workspace.cycles.d_numCycles,
        workspace.d_parent1Gathered,
        workspace.d_parent2Gathered,
        problemData.d_travelCosts,
        workspace.d_bestAssembly,
        workspace.d_assemblyCosts,
        numCities,
        numPairs,
        numPairs,
        numPairs,
        useCostAware
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EAX Error: Assembly evaluation failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Step 5: Construct offspring tours and job sequences
    constructOffspringTourKernel<<<numPairs, 1>>>(
        workspace.d_adjacencyLists,
        workspace.cycles.d_cycleBuffer,
        workspace.cycles.d_cycleParentFlags,
        workspace.cycles.d_cycleLengths,
        workspace.cycles.d_numCycles,
        workspace.d_bestAssembly,
        workspace.d_parent1Gathered,
        workspace.d_parent2Gathered,
        workspace.d_parent1Jobs,
        workspace.d_parent2Jobs,
        population.d_offspringCity,  // output offspring cities
        population.d_offspringJob,   // output offspring jobs
        numCities,
        numJobs,
        numPairs,
        numPairs,  // parent1Stride
        numPairs,  // parent2Stride
        numPairs   // offspringStride
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("EAX Error: Offspring construction failed: %s\n", cudaGetErrorString(err));
    }
    return err;
}
