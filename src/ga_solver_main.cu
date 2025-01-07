#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono> // For timing
#include "population.h"
#include "fitness_evaluator.h"

std::vector<std::pair<int, int>> createRandomCrossoverPoints(int numParents, int numCities) {
    // Number of pairs to create
    int numPairs = numParents / 2;
    
    // Vector to store the pairs of crossover points
    std::vector<std::pair<int, int>> crossoverPoints;
    
    // Seed the random number generator
    std::srand(std::time(0));
    
    for (int i = 0; i < numPairs; ++i) {
        // Generate two random cutpoints in the range [1, numCities - 1]
        int cutPoint1 = std::rand() % (numCities - 1) + 1; // Random between 1 and numCities-1
        int cutPoint2 = std::rand() % (numCities - 1) + 1;
        
        // Ensure cutPoint1 <= cutPoint2
        if (cutPoint1 > cutPoint2) {
            std::swap(cutPoint1, cutPoint2);
        }
        
        // Add the pair to the vector
        crossoverPoints.emplace_back(cutPoint1, cutPoint2);
    }
    
    return crossoverPoints;
}

std::vector<std::pair<float, float>> loadTSPData(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<float, float>> coordinates;
    std::string line;

    // Skip the header lines
    while (std::getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) break;
    }

    // Read coordinates
    while (std::getline(file, line)) {
        if (line.find("EOF") != std::string::npos) break;

        std::istringstream iss(line);
        int id;
        float x, y;
        iss >> id >> x >> y;
        coordinates.emplace_back(x, y);
    }

    file.close();
    return coordinates;
}

int main() {
    const int populationSize = 1000;   // Number of genomes in the population
    const int numGenerations = 50000;  // Number of iterations
    const int tournamentSize = 10;    // Tournament size for parent selection
    const float mutationRate = 0.1f; // Mutation rate
    const float elitismRate = 0.05f;  // Percentage of elite individuals to preserve

    const std::string tspFileName = "data/dsj1000.tsp";

    // Load TSP data
    std::cout << "Loading TSP data from " << tspFileName << "..." << std::endl;
    auto coordinates = loadTSPData(tspFileName);
    int numCities = coordinates.size();
    std::cout << "Number of cities: " << numCities << std::endl;

    // Create and initialize population
    Population population(populationSize, numCities);
    population.initialize();

    // Initialize the cost matrix
    std::cout << "Initializing cost matrix..." << std::endl;
    population.initializeCostMatrix(coordinates);
    
    // Create fitness evaluator
    FitnessEvaluator evaluator;

    // Start timing the main loop
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run genetic algorithm
    for (int generation = 0; generation < numGenerations; ++generation) {
        // Evaluate fitness
        population.evaluateFitness(evaluator);

        // Select elite individuals
        int numElites = static_cast<int>(populationSize * elitismRate);
        std::vector<Genome> elites = population.getTopGenomes(numElites);

        // Calculate number of offspring needed
        int numOffspring = populationSize - numElites;

        // Adjust the number of parents dynamically
        int numParents = (numOffspring %2 == 0) ? numOffspring : numOffspring + 1; // Each pair produces 2 offspring

        // Select parents for crossover
        std::vector<Genome> parents;
        population.selectParentsTournamentGPU(parents, numParents, tournamentSize);

        // Perform crossover
        std::vector<Genome> offspring;
        std::vector<std::pair<int, int>> crossoverPoints = createRandomCrossoverPoints(numParents, numCities);
        population.performCrossoverGPU(parents, offspring, crossoverPoints);

        // Perform mutation on offspring
        population.performMutationGPU(offspring, mutationRate);

        // Combine elites and offspring to form new population
        offspring.insert(offspring.end(), elites.begin(), elites.end());

        // Replace old population with new one
        population.setGenomes(offspring);

        // Output the best genome of the generation
        const Genome& bestGenome = population.getFittest();
        std::cout << "Generation " << generation << ": Best Fitness = " << bestGenome.getFitness() << std::endl;
    }

    // Stop timing the main loop
    auto endTime = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double avgTimePerGen = static_cast<double>(duration.count()) / numGenerations;

    // Output statistics
    std::cout << "\nStatistics:\n";
    std::cout << "Total time taken: " << duration.count() << " ms\n";
    std::cout << "Average time per generation: " << avgTimePerGen << " ms\n";

    // Output the final result
    const Genome& bestGenome = population.getFittest();
    std::cout << "Final Best Fitness = " << bestGenome.getFitness() << std::endl;
    std::cout << "Best Route:" << std::endl;
    bestGenome.printChromosome();

    return 0;
}
