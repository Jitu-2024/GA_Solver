#include "mutation.h"
#include <iostream>
#include <cassert>
#include <algorithm>

void testMutation() {
    int numCities = 5;
    float mutationRate = 1.0f; // Force mutation for testing
    Mutation mutation(numCities, mutationRate);

    // Example population
    std::vector<std::vector<int>> population = {
        {0, 1, 2, 3, 4},
        {4, 3, 2, 1, 0}
    };

    // Apply mutation
    mutation.applyMutation(population);

    // Verify mutation results
    for (const auto& chromosome : population) {
        if (chromosome.size() != numCities) {
            std::cerr << "Error: Chromosome size mismatch!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<int> sorted = chromosome;
        std::sort(sorted.begin(), sorted.end());
        if (sorted != std::vector<int>{0, 1, 2, 3, 4}) {
            std::cerr << "Error: Invalid chromosome permutation detected!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Mutation Test Passed!" << std::endl;
}


int main() {
    testMutation();
    return 0;
}
