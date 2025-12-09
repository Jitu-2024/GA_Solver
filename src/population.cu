// population.cu: Manage population for TSPJ genetic algorithm

#include "population.h"
#include "genome.h"
#include "fitness_evaluator.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>

// Initialize a population with random genomes
void initializePopulation(std::vector<Genome>& population, size_t populationSize,
                          size_t numCities, size_t numJobs, int mode) {
    population.clear();
    population.reserve(populationSize);

    for (size_t i = 0; i < populationSize; ++i) {
        population.emplace_back(numCities, numJobs, mode);
    }
}

// Nearest-neighbor tour construction starting from a given city
std::vector<size_t> constructNearestNeighborTour(size_t startCity, size_t numCities,
                                                  const std::vector<std::vector<float>>& travelTimes) {
    std::vector<size_t> tour;
    tour.reserve(numCities);

    std::vector<bool> visited(numCities + 1, false);
    visited[0] = true;  // Depot is not part of tour

    size_t current = startCity;
    tour.push_back(current);
    visited[current] = true;

    // Build tour by always visiting nearest unvisited city
    for (size_t i = 1; i < numCities; i++) {
        float minDist = std::numeric_limits<float>::max();
        size_t nearest = 0;

        for (size_t j = 1; j <= numCities; j++) {
            if (!visited[j]) {
                float dist = travelTimes[current][j];
                if (dist < minDist) {
                    minDist = dist;
                    nearest = j;
                }
            }
        }

        if (nearest > 0) {
            tour.push_back(nearest);
            visited[nearest] = true;
            current = nearest;
        }
    }

    return tour;
}

// Initialize population with a mix of nearest-neighbor and random genomes
void initializePopulationWithNN(std::vector<Genome>& population, size_t populationSize,
                                 size_t numCities, size_t numJobs, int mode,
                                 const std::vector<std::vector<float>>& travelTimes,
                                 float nnPercent) {
    population.clear();
    population.reserve(populationSize);

    size_t nnCount = static_cast<size_t>(populationSize * (nnPercent / 100.0f));

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "Initializing population with " << nnCount << " nearest-neighbor and "
              << (populationSize - nnCount) << " random genomes" << std::endl;

    // Create nearest-neighbor initialized genomes
    for (size_t i = 0; i < nnCount; i++) {
        Genome genome(numCities, numJobs, mode);

        // Start from different cities to create diversity
        size_t startCity = 1 + (i % numCities);

        // Construct nearest-neighbor tour
        genome.citySequence = constructNearestNeighborTour(startCity, numCities, travelTimes);

        // Add small perturbation to create diversity (swap 2 random cities)
        if (i > 0 && genome.citySequence.size() > 3) {
            std::uniform_int_distribution<size_t> dist(0, genome.citySequence.size() - 1);
            size_t pos1 = dist(gen);
            size_t pos2 = dist(gen);
            if (pos1 != pos2) {
                std::swap(genome.citySequence[pos1], genome.citySequence[pos2]);
            }
        }

        population.push_back(std::move(genome));
    }

    // Create random genomes for the rest
    for (size_t i = nnCount; i < populationSize; i++) {
        population.emplace_back(numCities, numJobs, mode);
    }

    std::cout << "Population initialization completed" << std::endl;
}

// Evaluate fitness of the entire population using fitness_evaluator
void evaluatePopulation(std::vector<Genome>& population,
                        const std::vector<std::vector<float>>& travelTimes,
                        const std::vector<std::vector<float>>& jobTimes, int mode) {
    evaluatePopulationFitness(population, travelTimes, jobTimes, mode); // Pass mode to fitness evaluator
}

// Sort population by fitness (ascending order)
void sortPopulationByFitness(std::vector<Genome>& population) {
    std::sort(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
        return a.fitness < b.fitness;
    });
}

// Get the best genome in the population
Genome getBestGenome(const std::vector<Genome>& population) {
    return *std::min_element(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
        return a.fitness < b.fitness;
    });
}

// Replace the worst genomes with offspring
void replaceWorst(std::vector<Genome>& population, const std::vector<Genome>& offspring) {
    size_t numOffspring = offspring.size();
    size_t populationSize = population.size();

    // Replace the worst genomes
    for (size_t i = 0; i < numOffspring; ++i) {
        population[populationSize - numOffspring + i] = offspring[i];
    }

    // Re-sort population
    sortPopulationByFitness(population);
}

// Print the population for debugging
void printPopulation(const std::vector<Genome>& population, int mode) {
    std::cout << "Population:\n";
    for (size_t i = 0; i < population.size(); ++i) {
        std::cout << "Genome " << i << ":\n";
        population[i].print(mode); // Pass mode to Genome's print function
    }
}
