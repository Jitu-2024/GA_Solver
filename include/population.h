#ifndef POPULATION_H
#define POPULATION_H

#include <vector>
#include <string>
#include "genome.h"
#include "cost_matrix.h"
#include "fitness_evaluator.h" // Include for FitnessEvaluator
#include "parent_selection.h"  // Include for ParentSelection (though methods are const, object is temp)
#include "crossover.h"         // Include for Crossover (though methods are const, object is temp)
#include "mutation.h"          // Include for Mutation (though methods are const, object is temp)


class Population {
public:
    // Constructor
    Population(int populationSize, int numCities);

    // Initialization methods
    void initialize();
    void initializeFromChromosomes(const std::vector<std::vector<int>>& initialChromosomes);
    void initializeCostMatrix(const std::vector<std::pair<float, float>>& coordinates);
    void initializeCostMatrix(const std::vector<std::vector<float>>& predefinedMatrix); // Overload

    // GA operations
    void evaluateFitness(FitnessEvaluator& evaluator);
    const Genome& getFittest() const; // Should return const ref if possible, or by value
    void selectParentsTournamentGPU(std::vector<Genome>& parents, int numParents, int tournamentSize) const;
    void performCrossoverGPU(const std::vector<Genome>& parents, 
                             std::vector<Genome>& offspring, 
                             const std::vector<std::pair<int, int>>& crossoverPoints) const;
    void performMutationGPU(std::vector<Genome>& offspring, float mutationRate) const;
    
    // Utility methods
    std::vector<Genome> getTopGenomes(int numElites) const;
    void printPopulation() const;

    // Getters and setters
    const std::vector<Genome>& getGenomes() const;
    void setGenomes(const std::vector<Genome>& newGenomes);
    int getPopulationSize() const { return populationSize; }
    int getNumCities() const { return numCities; }
    const CostMatrix& getCostMatrix() const { return costMatrix; }


private:
    int populationSize;
    int numCities;
    std::vector<Genome> genomes;
    CostMatrix costMatrix;
    // No need for FitnessEvaluator, ParentSelection, Crossover, Mutation as members
    // if they are passed into methods or created temporarily.
};

#endif // POPULATION_H 

// #ifndef POPULATION_H
// #define POPULATION_H

// #include <vector>
// #include "genome.h"
// #include "cost_matrix.h"
// #include "fitness_evaluator.h"
// #include "parent_selection.h"
// #include "crossover.h"
// #include "mutation.h"

// class Population {
// public:
//     // Constructor: Initialize with size and number of cities
//     Population(int populationSize, int numCities);

//     // Initialize the population with random genomes
//     void initialize();

//     // Initialize the cost matrix
//     void initializeCostMatrix(const std::vector<std::pair<float, float>>& coordinates);
//     void initializeCostMatrix(const std::vector<std::vector<float>>& predefinedMatrix); // overload

//     // Evaluate fitness of the entire population
//     void evaluateFitness(FitnessEvaluator& evaluator);

//     // Get the fittest genome
//     const Genome& getFittest() const;

//     // Select parents using tournament selection on the GPU
//     void selectParentsTournamentGPU(std::vector<Genome>& parents, int numParents, int tournamentSize) const;

//     // Perform Crossover for the parents
//     void performCrossoverGPU(const std::vector<Genome>& parents, 
//                              std::vector<Genome>& offspring, 
//                              const std::vector<std::pair<int, int>>& crossoverPoints) const;
    
//     // Perform Mutation on the offspring
//     void performMutationGPU(std::vector<Genome>& offspring, float mutationRate) const;

//     // Debug: Print all genomes
//     void printPopulation() const;

//     // Getters
//     const std::vector<Genome>& getGenomes() const;
//     std::vector<Genome> getTopGenomes(int numElites) const;

//     // Setters
//     void setGenomes(const std::vector<Genome>& newGenomes);

// private:
//     std::vector<Genome> genomes; // Vector of genomes
//     int populationSize;          // Size of the population
//     int numCities;               // Number of cities (length of genome)
//     CostMatrix costMatrix;       // Cost matrix for evaluating fitness
// };

// #endif // POPULATION_H
