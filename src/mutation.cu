// mutation.cu: Mutation for TSPJ problem

#include "mutation.h"
#include "genome.h"
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// Kernel for mutation using swap on GPU
__global__ void mutationKernel(size_t* sequence, size_t chromosomeLength, float mutationRate, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < chromosomeLength) {
        thrust::default_random_engine rng(seed + idx);
        thrust::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        thrust::uniform_int_distribution<size_t> indexDist(0, chromosomeLength - 1);

        if (probDist(rng) < mutationRate) {
            size_t start = indexDist(rng);
            size_t end = indexDist(rng);

            if (start > end) {
                size_t temp = start;
                start = end;
                end = temp;
            }

            // Reverse the subsequence
            while (start < end) {
                size_t temp = sequence[start];
                sequence[start] = sequence[end];
                sequence[end] = temp;
                start++;
                end--;
            }
        }
    }
}

// Host function for mutation
void performMutation(Genome& genome, float mutationRate) {
    size_t chromosomeLength = genome.citySequence.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> indexDist(0, chromosomeLength - 1);

    // Mutate city sequence
    if (probDist(gen) < mutationRate) {
        size_t start = indexDist(gen);
        size_t end = indexDist(gen);

        if (start > end) std::swap(start, end);

        // Reverse subsequence in city sequence
        while (start < end) {
            std::swap(genome.citySequence[start], genome.citySequence[end]);
            start++;
            end--;
        }
    }

    // Mutate job sequence
    if (probDist(gen) < mutationRate) {
        size_t start = indexDist(gen);
        size_t end = indexDist(gen);

        if (start > end) std::swap(start, end);

        // Reverse subsequence in job sequence
        while (start < end) {
            std::swap(genome.jobSequence[start], genome.jobSequence[end]);
            start++;
            end--;
        }
    }

    // Mutate pickup sequence
    if (probDist(gen) < mutationRate) {
        size_t start = indexDist(gen);
        size_t end = indexDist(gen);

        if (start > end) std::swap(start, end);

        // Reverse subsequence in pickup sequence
        while (start < end) {
            std::swap(genome.pickupSequence[start], genome.pickupSequence[end]);
            start++;
            end--;
        }
    }

    // Debugging output for mutation
    std::cout << "Genome after mutation:\n";
    genome.print();
}
