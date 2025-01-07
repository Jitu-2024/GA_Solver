#include <iostream>
#include "population.h"

int main() {
    const int populationSize = 10;
    const int numCities = 5;

    // Create a Population
    Population population(populationSize, numCities);

    // Initialize the population
    population.initialize();
    std::cout << "Initial Population:" << std::endl;
    population.printPopulation();

    // Verify population size
    if (population.getGenomes().size() == populationSize) {
        std::cout << "Population size test passed!" << std::endl;
    } else {
        std::cerr << "Population size test failed!" << std::endl;
        return 1;
    }

    return 0; // Test passed
}
