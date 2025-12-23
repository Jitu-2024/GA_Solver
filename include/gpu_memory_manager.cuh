// gpu_memory_manager.cuh: Centralized GPU memory management for TSPJ solver
// Handles allocation, deallocation, and data transfer for all GPU resources

#ifndef GPU_MEMORY_MANAGER_CUH
#define GPU_MEMORY_MANAGER_CUH

#include "gpu_population.cuh"
#include <vector>

class GPUMemoryManager {
public:
    GPUMemoryManager();
    ~GPUMemoryManager();

    // Initialize all GPU resources
    // Call once at program start
    cudaError_t initialize(uint32_t populationSize, uint32_t numCities,
                           uint32_t numJobs, int mode, uint64_t rngSeed = 12345);

    // Cleanup all GPU resources
    void cleanup();

    // Load problem data (travel costs, job costs) to GPU
    // Should be called after initialize()
    cudaError_t loadProblemData(const std::vector<std::vector<float>>& travelCosts,
                                 const std::vector<std::vector<float>>& jobCosts);

    // Initialize population with random permutations on GPU
    cudaError_t initializePopulation();

    // Copy best genome from GPU to host
    cudaError_t copyBestGenome(std::vector<uint16_t>& citySequence,
                                std::vector<uint16_t>& jobSequence,
                                std::vector<uint16_t>& pickupSequence,
                                float& fitness);

    // Store the best genome ever found (call when new best is discovered)
    cudaError_t storeBestEverGenome(uint32_t genomeIdx, float fitness);

    // Get the stored best-ever genome
    bool getBestEverGenome(std::vector<uint16_t>& citySequence,
                           std::vector<uint16_t>& jobSequence,
                           std::vector<uint16_t>& pickupSequence,
                           float& fitness) const;

    // Copy all fitness values to host (for logging)
    cudaError_t copyFitnessToHost(std::vector<float>& fitness);

    // Copy specific genome to host
    cudaError_t copyGenomeToHost(uint32_t genomeIdx,
                                  std::vector<uint16_t>& citySequence,
                                  std::vector<uint16_t>& jobSequence,
                                  std::vector<uint16_t>& pickupSequence);

    // Accessors for GPU data structures
    GPUPopulation& getPopulation() { return population_; }
    const GPUPopulation& getPopulation() const { return population_; }

    GPUProblemData& getProblemData() { return problemData_; }
    const GPUProblemData& getProblemData() const { return problemData_; }

    EAXWorkspace& getEAXWorkspace() { return eaxWorkspace_; }
    const EAXWorkspace& getEAXWorkspace() const { return eaxWorkspace_; }

    GPURngState& getRngState() { return rngState_; }
    const GPURngState& getRngState() const { return rngState_; }

    // Get memory usage statistics
    size_t getTotalAllocatedBytes() const { return totalAllocatedBytes_; }
    void printMemoryUsage() const;

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Get configuration
    uint32_t getPopulationSize() const { return population_.popSize; }
    uint32_t getNumCities() const { return population_.numCities; }
    uint32_t getNumJobs() const { return population_.numJobs; }
    int getMode() const { return population_.mode; }

private:
    GPUPopulation population_;
    GPUProblemData problemData_;
    EAXWorkspace eaxWorkspace_;
    GPURngState rngState_;

    // Sorting temporary storage (for CUB)
    void* d_sortTempStorage_;
    size_t sortTempStorageBytes_;

    // Tracking
    size_t totalAllocatedBytes_;
    bool initialized_;

    // Best-ever genome storage (host-side elitism)
    std::vector<uint16_t> bestEverCitySeq_;
    std::vector<uint16_t> bestEverJobSeq_;
    std::vector<uint16_t> bestEverPickupSeq_;
    float bestEverFitness_;
    bool hasBestEver_;

    // Helper functions
    cudaError_t allocatePopulation(uint32_t popSize, uint32_t numCities,
                                    uint32_t numJobs, int mode);
    cudaError_t allocateProblemData(uint32_t numCities, uint32_t numJobs);
    cudaError_t allocateEAXWorkspace(uint32_t numCities, uint32_t numJobs, uint32_t maxPairs);
    cudaError_t allocateRngStates(uint32_t numStates, uint64_t seed);
    cudaError_t allocateSortingStorage(uint32_t maxElements);
};

// Kernel declarations for initialization
__global__ void initRngStatesKernel(curandState* states, uint32_t numStates, uint64_t seed);

__global__ void initPopulationKernel(
    uint16_t* citySequences,
    uint16_t* jobSequences,
    uint16_t* pickupSequences,  // nullptr if mode != 1
    curandState* rngStates,
    uint32_t numCities,
    uint32_t numJobs,
    uint32_t popSize,
    int mode
);

// Fisher-Yates shuffle helper
__device__ void shuffleSequence(uint16_t* sequence, uint32_t length, curandState* state);

#endif // GPU_MEMORY_MANAGER_CUH
