#include <iostream>
#include "fitness_evaluator.h"
#include "population.h"

int main() {
    const int populationSize = 10;
    const int numCities = 5;

    // Mock coordinates for cost matrix
    std::vector<std::pair<float, float>> coordinates = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {0.5f, 0.5f}
    };

    // Create a Population
    Population population(populationSize, numCities);
    population.initialize();

    // Initialize cost matrix
    population.initializeCostMatrix(coordinates);

    // Print initial population
    std::cout << "Initial Population:" << std::endl;
    population.printPopulation();

    // Create a FitnessEvaluator (cost matrix already part of Population)
    FitnessEvaluator evaluator;

    // Evaluate fitness
    population.evaluateFitness(evaluator);

    // Print evaluated population
    std::cout << "Population after fitness evaluation:" << std::endl;
    population.printPopulation();

    // Verify fitness values
    bool fitnessCalculated = true;
    for (const auto& genome : population.getGenomes()) {
        if (genome.getFitness() <= 0.0f) {
            std::cerr << "Fitness evaluation test failed! Fitness not calculated correctly." << std::endl;
            fitnessCalculated = false;
            break;
        }
    }

    if (fitnessCalculated) {
        std::cout << "Fitness evaluation test passed!" << std::endl;
    }

    return fitnessCalculated ? 0 : 1; // Return success or failure
}
