#include "population.h"
#include "genome.h"
#include <iostream>
#include <algorithm>
#include <cassert>

// Constructor
Population::Population(int populationSize, int numCities)
    : populationSize(populationSize), numCities(numCities), costMatrix(numCities) {}

// Initialize population with random genomes
void Population::initialize() {
    genomes.clear();
    for (int i = 0; i < populationSize; ++i) {
        genomes.emplace_back(numCities); // Create a genome with random chromosome
    }
}

// Initialize the cost matrix
void Population::initializeCostMatrix(const std::vector<std::pair<float, float>>& coordinates) {\
    costMatrix.initialize(coordinates);
}

// Overload to initialize from a predefined cost matrix
void Population::initializeCostMatrix(const std::vector<std::vector<float>>& predefinedMatrix) {
    assert(predefinedMatrix.size() == numCities); // Ensure matrix dimensions match
    costMatrix.initialize(predefinedMatrix);     // Use a new method in CostMatrix
}


// Evaluate fitness of the entire population
void Population::evaluateFitness(FitnessEvaluator& evaluator) {
    const float* deviceCostMatrix = costMatrix.getDeviceCostMatrix();
    evaluator.evaluatePopulation(genomes, deviceCostMatrix, numCities);
}

// Get the fittest genome in the population
const Genome& Population::getFittest() const {
    return *std::max_element(genomes.begin(), genomes.end(),
                             [](const Genome& a, const Genome& b) {
                                 return a.getFitness() < b.getFitness(); // Higher fitness is better
                             });
}

// GPU-based tournament selection
void Population::selectParentsTournamentGPU(std::vector<Genome>& parents, int numParents, int tournamentSize) const {
    ParentSelection selector(genomes);
    selector.tournamentSelection(genomes, parents, numParents, tournamentSize);
}

// GPU-based crossover to generate offspring
void Population::performCrossoverGPU(const std::vector<Genome>& parents, 
                                     std::vector<Genome>& offspring, 
                                     const std::vector<std::pair<int, int>>& crossoverPoints) const {

    // std::cout << "Parent : " << std::endl;
    // for(const auto& parent : parents){
    //     parent.printChromosome();
    //     std::cout << std::endl;
    //     break;
    // }

    // Extract chromosomes from parent genomes
    std::vector<std::vector<int>> parentChromosomes;
    for (const auto& parent : parents) {
        parentChromosomes.push_back(parent.getChromosome());
    }

    // std::cout << "Parent Chromosomes : " << std::endl;
    // for(std::vector<int> parentChromosome : parentChromosomes){
    //     for (size_t i = 0; i < parentChromosome.size(); i++){
    //         std::cout << parentChromosome[i] << ", " ;
    //     }
    //     std::cout << std::endl;
    //     break;
    // }

    // Perform GPU-based Order Crossover using the Crossover class
    std::vector<std::vector<int>> offspringChromosomes;
    Crossover crossover(numCities);
    crossover.orderCrossover(parentChromosomes, offspringChromosomes, crossoverPoints);

    // Convert offspring chromosomes back to Genome objects
    offspring.clear();
    for (const auto& chromosome : offspringChromosomes) {
        offspring.emplace_back(chromosome); // Assumes Genome has a constructor for chromosomes
    }
}

// GPU-based mutation to introduce diversity
void Population::performMutationGPU(std::vector<Genome>& offspring, float mutationRate) const {
    // Flatten offspring chromosomes
    std::vector<std::vector<int>> offspringChromosomes;
    for (const auto& genome : offspring) {
        offspringChromosomes.push_back(genome.getChromosome());
    }

    // Perform GPU-based mutation
    Mutation mutation(numCities, mutationRate);
    mutation.applyMutation(offspringChromosomes);

    // Update offspring genomes with mutated chromosomes
    for (size_t i = 0; i < offspring.size(); ++i) {
        offspring[i] = Genome(offspringChromosomes[i]); // Replace with mutated chromosome
    }
}

std::vector<Genome> Population::getTopGenomes(int numElites) const {
    // Create a copy of the genomes to avoid modifying the original population
    std::vector<Genome> sortedGenomes = genomes;

    // Sort genomes by fitness in descending order (higher fitness first)
    std::sort(sortedGenomes.begin(), sortedGenomes.end(),
              [](const Genome& a, const Genome& b) {
                  return a.getFitness() < b.getFitness(); // Higher fitness is better
              });

    // Return the top 'numElites' genomes
    return std::vector<Genome>(sortedGenomes.begin(), sortedGenomes.begin() + numElites);
}

// Print the entire population (for debugging)
void Population::printPopulation() const {
    for (const auto& genome : genomes) {
        std::cout << "===============================================================================pop================================================================================" << std::endl;
        genome.printChromosome();
        std::cout << std::endl;
        break;
    }
}

// Getter for genomes
const std::vector<Genome>& Population::getGenomes() const {
    return genomes;
}

// Setter for genomes
void Population::setGenomes(const std::vector<Genome>& newGenomes) {
    genomes = newGenomes;
}
