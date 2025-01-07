#ifndef FITNESS_EVALUATOR_H
#define FITNESS_EVALUATOR_H

#include <vector>
#include "genome.h"

class FitnessEvaluator {
public:
    /**
     * Evaluate the fitness of a population using a cost matrix on the GPU.
     * @param population Vector of genomes to evaluate.
     * @param deviceCostMatrix Pointer to the GPU cost matrix.
     * @param numCities Number of cities in the TSP problem.
     */
    void evaluatePopulation(
        std::vector<Genome>& genomes, 
        const float* deviceCostMatrix, 
        int numCities
        );
};

#endif // FITNESS_EVALUATOR_H
