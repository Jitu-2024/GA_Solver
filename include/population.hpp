#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "genome.hpp"
#include <vector>

class Population {
public:
    Population(int population_size, int num_genes);

    // Apply mutation to the entire population
    void mutate(float mutation_rate);

    // Print all genomes in the population (for debugging)
    void print() const;

private:
    std::vector<Genome> genomes; // Vector of genomes in the population
};

#endif // POPULATION_HPP
