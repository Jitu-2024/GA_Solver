// ga_solver_main.cu: Main solver for TSPJ genetic algorithm

#include "genome.h"
#include "population.h"
#include "fitness_evaluator.h"
#include "parent_selection.h"
#include "crossover.h"
#include "mutation.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

// Function to parse a CSV file into a 2D vector
std::vector<std::vector<float>> parseCSV(const std::string &filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value)); // Convert string to float
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

int main() {

    // Parse the CSV files
    std::vector<std::vector<float>> travelTimes = parseCSV("data/gr17_TSPJ_TT.csv");
    std::vector<std::vector<float>> jobTimes = parseCSV("data/gr17_TSPJ_JT.csv");

    // Example from TSP-J paper
    // const std::vector<std::vector<float>> travelTimes = {
    //     {0, 5, 9, 12, 10, 6},
    //     {5, 0, 7, 9, 12, 10},
    //     {9, 7, 0, 5, 10, 12},
    //     {12, 9, 5, 0, 6, 10},
    //     {10, 12, 10, 6, 0, 7},
    //     {6, 10, 12, 10, 7, 0}};

    // const std::vector<std::vector<float>> jobTimes = {
    //     {0, 0, 0, 0, 0, 0},
    //     {0, 20, 22, 32, 25, 33},
    //     {0, 21, 20, 34, 23, 32},
    //     {0, 20, 22, 30, 22, 34},
    //     {0, 22, 24, 31, 22, 32},
    //     {0, 21, 20, 32, 24, 34}};

    // Parameters
    size_t numCities = travelTimes.size() - 1;
    size_t numJobs = jobTimes[0].size() - 1;
    size_t populationSize = 500;
    size_t generations = 1000;
    float mutationRate = 0.3f;
    size_t tournamentSize = 3;
    int mode = 1;  // 1 = pickup ; 0 = no pickup
    size_t diversityCount = static_cast<size_t>(populationSize * 0.3);  // 20% diversity

    // Initialize population
    std::vector<Genome> population;
    initializePopulation(population, populationSize, numCities, numJobs, mode);

    // Main GA loop
    for (size_t generation = 0; generation < generations; ++generation) {
        std::cout << "\nGeneration " << generation << ":\n";

        // Evaluate fitness
        evaluatePopulation(population, travelTimes, jobTimes, mode);

        // Sort by fitness
        sortPopulationByFitness(population);

        // Select parents
        std::vector<Genome> parents = selectParents(population, populationSize / 2, tournamentSize);

        // Generate offspring via crossover and mutation
        std::vector<Genome> offspring;
        for (size_t i = 0; i < parents.size(); i += 2) {
            Genome parent1 = parents[i];
            Genome parent2 = parents[(i + 1) % parents.size()];

            auto [child1, child2] = performCrossover(parent1, parent2, mode);
            performMutation(child1, mutationRate, mode);
            performMutation(child2, mutationRate, mode);

            offspring.push_back(child1);
            offspring.push_back(child2);
        }

        // Generate random genomes for diversity
        std::vector<Genome> diversityGenomes;
        initializePopulation(diversityGenomes, diversityCount, numCities, numJobs, mode);

        // Combine offspring and diversity genomes
        offspring.insert(offspring.end(), diversityGenomes.begin(), diversityGenomes.end());

        // Replace worst genomes with offspring and diversity
        replaceWorst(population, offspring);

        // Output best fitness of current generation
        Genome bestGenome = getBestGenome(population);
        std::cout << "Best Fitness of Generation " << generation << ": " << bestGenome.fitness << "\n";
    }

    // Final result
    Genome bestGenome = getBestGenome(population);
    std::cout << "\nBest Solution:\n";
    bestGenome.print(mode);

    return 0;
}