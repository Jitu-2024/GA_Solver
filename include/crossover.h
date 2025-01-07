#ifndef CROSSOVER_H
#define CROSSOVER_H

#include <vector>
#include <utility>
#include "genome.h"

class Crossover {
public:
    /**
     * Constructor
     * @param numCities Number of cities in each genome.
     */
    Crossover(int numCities);

    /**
     * Destructor
     */
    ~Crossover();

    /**
     * Perform GPU-based Order Crossover (OX).
     * @param parents A vector of parent genomes.
     * @param offspring A vector where the offspring genomes will be stored.
     * @param crossoverPoints A vector of pairs indicating start and end crossover points for each offspring.
     */
    void orderCrossover(const std::vector<std::vector<int>>& parents,
                               std::vector<std::vector<int>>& offspring,
                               const std::vector<std::pair<int, int>>& crossoverPoints);

private:
    int numCities;        // Number of cities
    int* d_parents;       // GPU memory for parent chromosomes
    int* d_offspring;     // GPU memory for offspring chromosomes
    int* d_crossoverPoints; // GPU memory for crossover points
};

#endif // CROSSOVER_H
