#ifndef MUTATION_H
#define MUTATION_H

#include <vector>
#include "genome.h"

class Mutation {
public:
    /**
     * Constructor
     * @param numCities Number of cities in each genome.
     * @param mutationRate Probability of mutation for each genome.
     */
    Mutation(int numCities, float mutationRate);

    /**
     * Destructor
     */
    ~Mutation();

    /**
     * Perform Swap Mutation on the GPU.
     * @param population A vector of genome chromosomes (std::vector<std::vector<int>>).
     */
    void applyMutation(std::vector<std::vector<int>>& population);

private:
    int numCities;           // Number of cities
    float mutationRate;      // Mutation probability
    int* d_population;       // GPU memory for population chromosomes
    int* d_mutatedPopulation; // GPU memory for mutated chromosomes
};

#endif // MUTATION_H
