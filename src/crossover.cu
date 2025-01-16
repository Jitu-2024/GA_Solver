// crossover.cu: Order crossover for TSPJ problem with optional pickup sequence

#include "crossover.h"
#include "genome.h"
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/random.h>

// Kernel for order crossover on GPU
__global__ void orderCrossoverKernel(const size_t* parent1, const size_t* parent2,
                                     size_t* child, size_t chromosomeLength,
                                     unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) { // Single thread selects crossover points
        thrust::default_random_engine rng(seed);
        thrust::uniform_int_distribution<size_t> dist(0, chromosomeLength - 1);

        size_t start = dist(rng);
        size_t end = dist(rng);

        if (start > end) {
            size_t temp = start;
            start = end;
            end = temp;
        }

        // Copy segment from parent1 to child
        for (size_t i = start; i <= end; ++i) {
            child[i] = parent1[i];
        }

        // Fill the rest from parent2, preserving order
        size_t childIdx = (end + 1) % chromosomeLength;
        for (size_t i = 0; i < chromosomeLength; ++i) {
            size_t candidate = parent2[(end + 1 + i) % chromosomeLength];

            // Check if candidate is already in the child
            bool exists = false;
            for (size_t j = start; j <= end; ++j) {
                if (child[j] == candidate) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                child[childIdx] = candidate;
                childIdx = (childIdx + 1) % chromosomeLength;
            }
        }
    }
}

// Host function for order crossover
std::pair<Genome, Genome> performCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    size_t chromosomeLength = parent1.citySequence.size();

    // Prepare device memory
    thrust::device_vector<size_t> d_parent1City(parent1.citySequence);
    thrust::device_vector<size_t> d_parent2City(parent2.citySequence);
    thrust::device_vector<size_t> d_childCity1(chromosomeLength);
    thrust::device_vector<size_t> d_childCity2(chromosomeLength);

    thrust::device_vector<size_t> d_parent1Job(parent1.jobSequence);
    thrust::device_vector<size_t> d_parent2Job(parent2.jobSequence);
    thrust::device_vector<size_t> d_childJob1(chromosomeLength);
    thrust::device_vector<size_t> d_childJob2(chromosomeLength);

    thrust::device_vector<size_t> d_parent1Pickup, d_parent2Pickup, d_childPickup1, d_childPickup2;
    if (mode == 1) { // Only initialize pickup sequences for TSP-J with Pickup
        d_parent1Pickup = thrust::device_vector<size_t>(parent1.pickupSequence);
        d_parent2Pickup = thrust::device_vector<size_t>(parent2.pickupSequence);
        d_childPickup1 = thrust::device_vector<size_t>(chromosomeLength);
        d_childPickup2 = thrust::device_vector<size_t>(chromosomeLength);
    }

    // Launch kernels for cities and jobs
    int threadsPerBlock = 1; // Single thread selects crossover points
    int blocksPerGrid = 1;
    unsigned long seed = time(nullptr);

    orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_parent1City.data()),
        thrust::raw_pointer_cast(d_parent2City.data()),
        thrust::raw_pointer_cast(d_childCity1.data()),
        chromosomeLength, seed);

    orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_parent2City.data()),
        thrust::raw_pointer_cast(d_parent1City.data()),
        thrust::raw_pointer_cast(d_childCity2.data()),
        chromosomeLength, seed);

    orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_parent1Job.data()),
        thrust::raw_pointer_cast(d_parent2Job.data()),
        thrust::raw_pointer_cast(d_childJob1.data()),
        chromosomeLength, seed);

    orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_parent2Job.data()),
        thrust::raw_pointer_cast(d_parent1Job.data()),
        thrust::raw_pointer_cast(d_childJob2.data()),
        chromosomeLength, seed);

    // Launch kernels for pickups (only if mode == 1)
    if (mode == 1) {
        orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_parent1Pickup.data()),
            thrust::raw_pointer_cast(d_parent2Pickup.data()),
            thrust::raw_pointer_cast(d_childPickup1.data()),
            chromosomeLength, seed);

        orderCrossoverKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_parent2Pickup.data()),
            thrust::raw_pointer_cast(d_parent1Pickup.data()),
            thrust::raw_pointer_cast(d_childPickup2.data()),
            chromosomeLength, seed);
    }

    // Copy results back to host
    Genome child1(chromosomeLength, chromosomeLength, mode);
    Genome child2(chromosomeLength, chromosomeLength, mode);

    thrust::copy(d_childCity1.begin(), d_childCity1.end(), child1.citySequence.begin());
    thrust::copy(d_childCity2.begin(), d_childCity2.end(), child2.citySequence.begin());

    thrust::copy(d_childJob1.begin(), d_childJob1.end(), child1.jobSequence.begin());
    thrust::copy(d_childJob2.begin(), d_childJob2.end(), child2.jobSequence.begin());

    if (mode == 1) { // Only copy pickup sequences for TSP-J with Pickup
        thrust::copy(d_childPickup1.begin(), d_childPickup1.end(), child1.pickupSequence.begin());
        thrust::copy(d_childPickup2.begin(), d_childPickup2.end(), child2.pickupSequence.begin());
    }

    return {child1, child2};
}
