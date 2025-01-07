#include <iostream>
#include <cassert>
#include "population.h"
#include "fitness_evaluator.h"
#include "parent_selection.h"

void testParentSelection() {
    const int populationSize = 10;
    const int numCities = 5;
    const int numParents = 4;
    const int tournamentSize = 3;

    // Mock coordinates for cities
    std::vector<std::pair<float, float>> coordinates = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {0.5f, 0.5f}
    };

    // Create and initialize a Population
    Population population(populationSize, numCities);
    population.initialize();

    // Initialize the cost matrix
    population.initializeCostMatrix(coordinates);

    // Create a FitnessEvaluator
    FitnessEvaluator evaluator;

    // Evaluate fitness of the population
    population.evaluateFitness(evaluator);

    // Perform GPU-based parent selection
    std::vector<Genome> selectedParents;
    population.selectParentsTournamentGPU(selectedParents, numParents, tournamentSize);

    // Validate selected parents
    assert(selectedParents.size() == numParents);
    for (const Genome& parent : selectedParents) {
        bool found = false;
        for (const Genome& genome : population.getGenomes()) {
            if (parent.getChromosome() == genome.getChromosome()) {
                found = true;
                break;
            }
        }
        assert(found && "Selected parent must exist in the population.");
        assert(parent.getFitness() > 0.0f && "Parent fitness must be greater than zero.");
    }

    // Print results for debugging
    std::cout << "Parent Selection Test Passed!" << std::endl;
    for (const Genome& parent : selectedParents) {
        parent.printChromosome();
        std::cout << "Fitness: " << parent.getFitness() << std::endl;
    }
}

int main() {
    testParentSelection();
    return 0;
}
