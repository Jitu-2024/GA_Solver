#include "genetic_kernels.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <bitset>
#include <cuda_runtime.h>

// Helper function to generate a random population
std::vector<uint8_t> generate_population(int num_genomes, int gene_length_bits) {
    std::vector<uint8_t> population((num_genomes * gene_length_bits + 7) / 8, 0);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 255);

    for (size_t i = 0; i < population.size(); ++i) {
        population[i] = distribution(generator);
    }
    return population;
}

int main() {
    int num_genomes = 10;
    int gene_length_bits = 8; // Example: 8 bits per gene
    int total_bits = num_genomes * gene_length_bits;
    float mutation_rate = 0.1;

    // Generate initial population
    std::vector<uint8_t> h_population = generate_population(num_genomes, gene_length_bits);

    // Print initial population
    std::cout << "Initial Population:" << std::endl;
    for (auto gene : h_population) {
        std::cout << std::bitset<8>(static_cast<unsigned long>(gene)) << " ";

    }
    std::cout << std::endl;

    // Allocate memory on GPU
    uint8_t* d_population;
    cudaMalloc(&d_population, h_population.size() * sizeof(uint8_t));
    cudaMemcpy(d_population, h_population.data(), h_population.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Perform mutation on GPU
    gpu_mutate(d_population, total_bits, mutation_rate);

    // Copy mutated population back to host
    cudaMemcpy(h_population.data(), d_population, h_population.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Print mutated population
    std::cout << "Mutated Population:" << std::endl;
    for (auto gene : h_population) {
        std::cout << std::bitset<8>(static_cast<unsigned long>(gene)) << " ";

    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_population);

    return 0;
}
