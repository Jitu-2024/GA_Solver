// population.h: Header file for managing population in TSPJ genetic algorithm

#ifndef POPULATION_H
#define POPULATION_H

#include "genome.h"
#include <vector>

// Initialize a population with random genomes
void initializePopulation(std::vector<Genome>& population, size_t populationSize,
                          size_t numCities, size_t numJobs, int mode);

/**
 * Initialize population with a mix of nearest-neighbor and random genomes
 * This provides better starting solutions for the TSP component
 *
 * @param population Output vector of genomes
 * @param populationSize Total population size
 * @param numCities Number of cities (excluding depot)
 * @param numJobs Number of jobs
 * @param mode 0=no pickup, 1=with pickup sequence
 * @param travelTimes Cost matrix for nearest-neighbor construction
 * @param nnPercent Percentage of population to initialize with nearest-neighbor (0-100)
 */
void initializePopulationWithNN(std::vector<Genome>& population, size_t populationSize,
                                 size_t numCities, size_t numJobs, int mode,
                                 const std::vector<std::vector<float>>& travelTimes,
                                 float nnPercent = 30.0f);

// Evaluate fitness of the entire population
void evaluatePopulation(std::vector<Genome>& population,
                        const std::vector<std::vector<float>>& travelTimes,
                        const std::vector<std::vector<float>>& jobTimes, int mode);

// Sort population by fitness (ascending order)
void sortPopulationByFitness(std::vector<Genome>& population);

// Get the best genome in the population
Genome getBestGenome(const std::vector<Genome>& population);

// Replace the worst genomes with offspring
void replaceWorst(std::vector<Genome>& population, const std::vector<Genome>& offspring);

// Print the population
void printPopulation(const std::vector<Genome>& population);

#endif // POPULATION_H
