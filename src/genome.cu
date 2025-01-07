#include "genome.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

// Constructor

Genome::Genome(const std::vector<int>& chromosome)
    : chromosome(chromosome), fitness(0.0f) {}

Genome::Genome(int numCities) {
    chromosome.resize(numCities);
    std::iota(chromosome.begin(), chromosome.end(), 0); // 0 to numCities-1
    shuffleChromosome();
    fitness = 0.0f; // Initialize fitness to 0
}

void Genome::shuffleChromosome() {
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(chromosome.begin(), chromosome.end(), rng);
}

void Genome::printChromosome() const {
    for (int city : chromosome) {
        std::cout << city << " ";
    }
    std::cout << "| Fitness: " << fitness << std::endl;
}

float Genome::getFitness() const {
    return fitness;
}

void Genome::setFitness(float fitnessValue) {
    fitness = fitnessValue;
}

std::vector<int> Genome::getChromosome() const {
    return chromosome;
}

