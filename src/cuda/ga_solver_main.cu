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
#include <string> // Required for std::string and std::stof, std::stoi

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

// Function to load initial genomes from a file
std::vector<std::vector<int>> loadInitialGenomes(const std::string& fileName, int numCities) {
    std::vector<std::vector<int>> initialGenomes;
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open initial genomes file " << fileName << ". Using random initialization." << std::endl;
        return initialGenomes; // Return empty vector, will trigger random initialization
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> chromosome;
        std::istringstream iss(line);
        int city;
        while (iss >> city) {
            chromosome.push_back(city);
        }
        if (!chromosome.empty()) {
            if (chromosome.size() != numCities) {
                 std::cerr << "Warning: Chromosome in " << fileName << " has incorrect length (" << chromosome.size() << " vs " << numCities << "). Skipping." << std::endl;
            } else {
                initialGenomes.push_back(chromosome);
            }
        }
    }
    file.close();
    std::cout << "Loaded " << initialGenomes.size() << " initial genomes from " << fileName << std::endl;
    return initialGenomes;
}

int main(int argc, char* argv[]) { // Add argc and argv for command-line arguments
    // Default values
    int populationSize = 1000;   // Number of genomes in the population
    int numGenerations = 5000;  // Number of iterations
    int tournamentSize = 10;    // Tournament size for parent selection
    float mutationRate = 0.1f; // Mutation rate
    float elitismRate = 0.05f;  // Percentage of elite individuals to preserve
    std::string tspFileName = "data/dsj1000.tsp";
    std::string initialGenomesFile = ""; // Empty means no file provided

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--populationSize" && i + 1 < argc) {
            populationSize = std::stoi(argv[++i]);
        } else if (arg == "--numGenerations" && i + 1 < argc) {
            numGenerations = std::stoi(argv[++i]);
        } else if (arg == "--tournamentSize" && i + 1 < argc) {
            tournamentSize = std::stoi(argv[++i]);
        } else if (arg == "--mutationRate" && i + 1 < argc) {
            mutationRate = std::stof(argv[++i]);
        } else if (arg == "--elitismRate" && i + 1 < argc) {
            elitismRate = std::stof(argv[++i]);
        } else if (arg == "--tspFileName" && i + 1 < argc) {
            tspFileName = argv[++i];
        } else if (arg == "--initialGenomesFile" && i + 1 < argc) {
            initialGenomesFile = argv[++i];
        } else {
            std::cerr << "Warning: Unknown or incomplete argument '" << arg << "'" << std::endl;
        }
    }

    // Output parameters being used
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Population Size: " << populationSize << std::endl;
    std::cout << "  Number of Generations: " << numGenerations << std::endl;
    std::cout << "  Tournament Size: " << tournamentSize << std::endl;
    std::cout << "  Mutation Rate: " << mutationRate << std::endl;
    std::cout << "  Elitism Rate: " << elitismRate << std::endl;
    std::cout << "  TSP File Name: " << tspFileName << std::endl;
    if (!initialGenomesFile.empty()) {
        std::cout << "  Initial Genomes File: " << initialGenomesFile << std::endl;
    }
    std::cout << std::endl;

    // Load TSP data
    std::cout << "Loading TSP data from " << tspFileName << "..." << std::endl;
    auto coordinates = loadTSPData(tspFileName);
    int numCities = coordinates.size();
    std::cout << "Number of cities: " << numCities << std::endl;

    // Create and initialize population
    Population population(populationSize, numCities);
    if (!initialGenomesFile.empty()) {
        std::vector<std::vector<int>> initialGenomes = loadInitialGenomes(initialGenomesFile, numCities);
        if (!initialGenomes.empty()) {
            population.initializeFromChromosomes(initialGenomes);
        } else {
            std::cout << "No valid initial genomes loaded or file not found. Using random initialization." << std::endl;
            population.initialize(); // Fallback to random
        }
    } else {
        population.initialize(); // Default random initialization
    }

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
