#include "genetic_kernels.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <bitset>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Helper to generate a random population
std::vector<uint8_t> generate_population(int num_genomes, int gene_length_bits) {
    std::vector<uint8_t> population((num_genomes * gene_length_bits + 7) / 8, 0);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 255);

    for (size_t i = 0; i < population.size(); ++i) {
        population[i] = distribution(generator);
    }
    return population;
}

// Test GPU mutation for correctness
TEST(MutationTest, MutationCorrectness) {
    int num_genomes = 10;
    int gene_length_bits = 8;
    int total_bits = num_genomes * gene_length_bits;
    float mutation_rate = 0.1;

    // Generate initial population
    std::vector<uint8_t> h_population = generate_population(num_genomes, gene_length_bits);

    // Copy population to GPU
    uint8_t* d_population;
    cudaMalloc(&d_population, h_population.size() * sizeof(uint8_t));
    cudaMemcpy(d_population, h_population.data(), h_population.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Perform mutation on GPU
    gpu_mutate(d_population, total_bits, mutation_rate);

    // Copy mutated population back to host
    std::vector<uint8_t> mutated_population(h_population.size());
    cudaMemcpy(mutated_population.data(), d_population, h_population.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Verify that some bits have been flipped
    int flipped_bits = 0;
    for (size_t i = 0; i < h_population.size(); ++i) {
        flipped_bits += __builtin_popcount(h_population[i] ^ mutated_population[i]);
    }
    EXPECT_GT(flipped_bits, 0); // At least some bits should be flipped
    EXPECT_LE(flipped_bits, total_bits * mutation_rate * 2); // Ensure mutation rate is reasonable

    // Free GPU memory
    cudaFree(d_population);
}


TEST(MutationTest, PerformanceBenchmarking) {
    int num_genomes = 10000;
    int gene_length_bits = 8;
    int total_bits = num_genomes * gene_length_bits;
    float mutation_rate = 0.1;

    // Generate initial population
    std::vector<uint8_t> h_population = generate_population(num_genomes, gene_length_bits);

    // Copy population to GPU
    uint8_t* d_population;
    cudaMalloc(&d_population, h_population.size() * sizeof(uint8_t));
    cudaMemcpy(d_population, h_population.data(), h_population.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Set up GPU timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer and perform mutation
    cudaEventRecord(start, 0);
    gpu_mutate(d_population, total_bits, mutation_rate);
    cudaEventRecord(stop, 0);

    // Synchronize and calculate elapsed time
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Print the timing result
    std::cout << "Time for mutation kernel: " << elapsed_time << " ms" << std::endl;

    // Free GPU memory
    cudaFree(d_population);
}


// Main function provided by Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}