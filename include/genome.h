#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include <iostream>

class Genome {
public:
    std::vector<int> chromosome; // Chromosome (city indices)

    // Constructor
    Genome(int numCities);

   // Constructor: Initialize with a given chromosome
    Genome(const std::vector<int>& chromosome);

    // Randomize the chromosome
    void shuffleChromosome();

    // Print chromosome (debugging)
    void printChromosome() const;

    // Getter and setter for fitness
    float getFitness() const;
    void setFitness(float fitnessValue);

    // Getter for chromosome
    std::vector<int> getChromosome() const;

    ~Genome() = default;

private:
    float fitness; // Fitness score
};

#endif // GENOME_H
