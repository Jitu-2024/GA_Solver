#include "population.hpp"
#include <iostream>

// Constructor to initialize the population with a given size and gene count
Population::Population(int population_size, int num_genes)
    : genomes(population_size, Genome(num_genes)) {}

// Apply mutation to all genomes in the population
void Population::mutate(float mutation_rate) {
    for (auto& genome : genomes) {
        genome.mutate(mutation_rate);
    }
}

// Print all genomes in the population
void Population::print() const {
    for (size_t i = 0; i < genomes.size(); ++i) {
        std::cout << "Genome " << i + 1 << ": ";
        genomes[i].print();
    }
}
