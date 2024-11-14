#include <curand_kernel.h>
#include "genetic_kernels.cuh"

// CUDA kernel for mutating genomes
__global__ void mutation_kernel(uint8_t* population, int total_bits, float mutation_rate, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    if (idx < total_bits) {
        // Generate a random number
        float rand_val = curand_uniform(&states[idx]);

        // Apply mutation based on mutation rate
        if (rand_val < mutation_rate) {
            population[idx / 8] ^= (1 << (idx % 8)); // Flip the bit
        }
    }
}

// Kernel for initializing random states
__global__ void init_random_states(curandState* states, unsigned long seed, int total_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_threads) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

void gpu_mutate(uint8_t* population, int total_bits, float mutation_rate) {
    // Number of threads and blocks
    int threads_per_block = 256;
    int blocks = (total_bits + threads_per_block - 1) / threads_per_block;

    // Allocate memory for random states on GPU
    curandState* d_states;
    cudaMalloc(&d_states, total_bits * sizeof(curandState));

    // Initialize random states
    init_random_states<<<blocks, threads_per_block>>>(d_states, 12345, total_bits);
    cudaDeviceSynchronize();

    // Launch mutation kernel
    mutation_kernel<<<blocks, threads_per_block>>>(population, total_bits, mutation_rate, d_states);
    cudaDeviceSynchronize();

    // Free random states memory
    cudaFree(d_states);
}