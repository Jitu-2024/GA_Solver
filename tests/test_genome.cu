#include <iostream>
#include "fitness_evaluator.h"
#include "population.h"
#include "cost_matrix.h"

int main() {
    const int numCities = 5;
    const int populationSize = 1; // Single genome test case

    // Mock coordinates for cost matrix
    std::vector<std::pair<float, float>> coordinates = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {0.5f, 0.5f}
    };

    // Create a population with a single genome
    Population population(populationSize, numCities);
    population.initialize();

    // Initialize cost matrix
    population.initializeCostMatrix(coordinates);

    // Create a FitnessEvaluator
    FitnessEvaluator evaluator;

    // Evaluate fitness of the single genome
    population.evaluateFitness(evaluator);

    // Retrieve the genome and print its fitness
    const auto& genome = population.getGenomes()[0];
    std::cout << "Chromosome: ";
    genome.printChromosome();
    std::cout << "Fitness: " << genome.getFitness() << std::endl;

    // Verify fitness
    if (genome.getFitness() > 0.0f) {
        std::cout << "Test Passed: Fitness calculated correctly." << std::endl;
        return 0; // Success
    } else {
        std::cerr << "Test Failed: Fitness not calculated correctly." << std::endl;
        return 1; // Failure
    }
}
