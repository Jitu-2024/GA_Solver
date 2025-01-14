// ga_solver_main.cu: Main solver for TSPJ genetic algorithm

#include "genome.h"
#include "population.h"
#include "fitness_evaluator.h"
#include "parent_selection.h"
#include "crossover.h"
#include "mutation.h"
#include <iostream>
#include <vector>

int main() {
    // Example from TSP-J paper
    const std::vector<std::vector<float>> travelTimes = {
        {0, 5, 9, 12, 10, 6},
        {5, 0, 7, 9, 12, 10},
        {9, 7, 0, 5, 10, 12},
        {12, 9, 5, 0, 6, 10},
        {10, 12, 10, 6, 0, 7},
        {6, 10, 12, 10, 7, 0}};

    const std::vector<std::vector<float>> jobTimes = {
        {0, 0, 0, 0, 0},
        {20, 22, 32, 25, 33},
        {21, 20, 34, 23, 32},
        {20, 22, 30, 22, 34},
        {22, 24, 31, 22, 32},
        {21, 20, 32, 24, 34}};

    // Parameters
    size_t numCities = 5;
    size_t numJobs = 5;
    size_t populationSize = 50;
    size_t generations = 100;
    float mutationRate = 0.9f;
    size_t tournamentSize = 10;

    // Initialize population
    std::vector<Genome> population;
    initializePopulation(population, populationSize, numCities, numJobs);

    // Main GA loop
    for (size_t generation = 0; generation < generations; ++generation) {
        std::cout << "\nGeneration " << generation << ":\n";

        // Evaluate fitness
        evaluatePopulation(population, travelTimes, jobTimes);

        // Sort by fitness
        sortPopulationByFitness(population);

        // Select parents
        std::vector<Genome> parents = selectParents(population, populationSize / 2, tournamentSize);

        // Debug selected parents
        std::cout << "Selected Parents:\n";
        for (size_t i = 0; i < parents.size(); ++i) {
            std::cout << "Parent " << i << ":\n";
            parents[i].print();
        }

        // Generate offspring via crossover and mutation
        std::vector<Genome> offspring;
        for (size_t i = 0; i < parents.size(); i += 2) {
            Genome parent1 = parents[i];
            Genome parent2 = parents[(i + 1) % parents.size()];

            auto [child1, child2] = performCrossover(parent1, parent2);
            performMutation(child1, mutationRate);
            performMutation(child2, mutationRate);

            offspring.push_back(child1);
            offspring.push_back(child2);
        }

        // Replace worst genomes with offspring
        replaceWorst(population, offspring);

        // Output best fitness of current generation
        Genome bestGenome = getBestGenome(population);
        std::cout << "Best Fitness of Generation " << generation << ": " << bestGenome.fitness << "\n";
    }

    // Final result
    Genome bestGenome = getBestGenome(population);
    std::cout << "\nBest Solution:\n";
    bestGenome.print();

    return 0;
}