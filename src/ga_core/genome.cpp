#include "genome.hpp"
#include <iostream>
#include <random>

// Constructor to initialize the genome with random genes
Genome::Genome(int num_genes) : genes(num_genes) {
    std::srand(std::time(0)); // Seed for random number generation
    for (auto& gene : genes) {
        for (int i = 0; i < gene_length; ++i) {
            gene[i] = std::rand() % 2; // Randomly assign 0 or 1 to each bit
        }
    }
}

// Mutation function: Flip random bits with the given mutation rate
void Genome::mutate(float mutation_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& gene : genes) {
        for (int i = 0; i < gene_length; ++i) {
            if (dis(gen) < mutation_rate) {
                gene.flip(i); // Flip the bit
            }
        }
    }
}

// Print the genome for debugging purposes
void Genome::print() const {
    for (const auto& gene : genes) {
        std::cout << gene << " ";
    }
    std::cout << std::endl;
}
