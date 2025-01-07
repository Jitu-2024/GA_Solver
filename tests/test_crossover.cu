#include "crossover.h"
#include "genome.h"
#include <iostream>
#include <vector>
#include <cassert>

void testOrderCrossover() {
    int numCities = 8;

    // Example parent chromosomes as Genome objects
    std::vector<Genome> parents;
    parents.emplace_back(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    parents.emplace_back(std::vector<int>{7, 6, 5, 4, 3, 2, 1, 0});

    // Extract chromosomes from parents
    std::vector<std::vector<int>> parentChromosomes;
    for (const auto& parent : parents) {
        parentChromosomes.push_back(parent.getChromosome());
    }

    // Specify crossover points
    std::vector<std::pair<int, int>> crossoverPoints = {
        {2, 5}  // Start and end indices for crossover
    };

    // Create offspring container
    std::vector<std::vector<int>> offspringChromosomes;

    // Perform Order Crossover
    Crossover crossover(numCities);
    crossover.orderCrossover(parentChromosomes, offspringChromosomes, crossoverPoints);

    // Convert offspring chromosomes back to Genome objects
    std::vector<Genome> offspring;
    for (const auto& chromosome : offspringChromosomes) {
        offspring.emplace_back(chromosome);
    }

    // Check offspring correctness
    assert(offspring.size() == parents.size());

    // Offspring 1: Should preserve the segment [2, 3, 4, 5] from Parent 1
    // Fill remaining positions from Parent 2: [7, 6, 1, 0]
    std::vector<int> expectedOffspring1 = {7, 6, 2, 3, 4, 5, 1, 0};
    assert(offspring[0].getChromosome() == expectedOffspring1);

    // Offspring 2: Should preserve the segment [5, 4, 3, 2] from Parent 2
    // Fill remaining positions from Parent 1: [0, 1, 7, 6]
    std::vector<int> expectedOffspring2 = {0, 1, 5, 4, 3, 2, 7, 6};
    assert(offspring[1].getChromosome() == expectedOffspring2);

    // Print results for debugging
    std::cout << "Order Crossover Test Passed!" << std::endl;
}

int main() {
    testOrderCrossover();
    return 0;
}
