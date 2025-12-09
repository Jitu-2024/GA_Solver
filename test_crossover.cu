// test_crossover.cu: Simple test for OX+Uniform crossover
#include "crossover.h"
#include "genome.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

void printSequence(const std::string& name, const std::vector<size_t>& seq) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(seq.size(), (size_t)20); i++) {
        std::cout << seq[i];
        if (i < seq.size() - 1) std::cout << ",";
    }
    if (seq.size() > 20) std::cout << "...";
    std::cout << "] (len=" << seq.size() << ")" << std::endl;
}

bool validateTour(const std::vector<size_t>& tour) {
    std::vector<bool> seen(tour.size() + 1, false);
    for (size_t city : tour) {
        if (city < 1 || city > tour.size()) {
            std::cout << "Invalid city: " << city << std::endl;
            return false;
        }
        if (seen[city]) {
            std::cout << "Duplicate city: " << city << std::endl;
            return false;
        }
        seen[city] = true;
    }
    return true;
}

int main() {
    std::cout << "=== Simple Crossover Test ===" << std::endl;

    // Create two parent genomes with 10 cities for easy verification
    size_t numCities = 10;
    int mode = 0;  // no pickup

    Genome parent1(numCities, numCities, mode);
    Genome parent2(numCities, numCities, mode);

    // Set specific city sequences for testing
    // Parent1: 1,2,3,4,5,6,7,8,9,10
    // Parent2: 5,1,7,3,9,2,8,4,10,6
    parent1.citySequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    parent2.citySequence = {5, 1, 7, 3, 9, 2, 8, 4, 10, 6};

    // Jobs: simple sequences
    parent1.jobSequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    parent2.jobSequence = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    std::cout << "\n--- Parents ---" << std::endl;
    printSequence("P1 cities", parent1.citySequence);
    printSequence("P2 cities", parent2.citySequence);
    printSequence("P1 jobs", parent1.jobSequence);
    printSequence("P2 jobs", parent2.jobSequence);

    // Test single pair crossover
    std::vector<Genome> parents1 = {parent1};
    std::vector<Genome> parents2 = {parent2};

    std::cout << "\n--- Crossover (1 pair) ---" << std::endl;
    auto children = performSimpleBatchCrossover(parents1, parents2, mode);

    if (children.empty()) {
        std::cout << "ERROR: No children produced!" << std::endl;
        return 1;
    }

    printSequence("Child cities", children[0].citySequence);
    printSequence("Child jobs", children[0].jobSequence);

    // Validate child tour
    if (validateTour(children[0].citySequence)) {
        std::cout << "Child city tour is VALID" << std::endl;
    } else {
        std::cout << "Child city tour is INVALID!" << std::endl;
        return 1;
    }

    // Test with larger problem (422 cities like TSPJ_1M)
    std::cout << "\n--- Test with 422 cities ---" << std::endl;
    numCities = 422;

    Genome p1_large(numCities, numCities, mode);
    Genome p2_large(numCities, numCities, mode);

    // Create different permutations
    std::vector<size_t> seq1(numCities), seq2(numCities);
    for (size_t i = 0; i < numCities; i++) {
        seq1[i] = i + 1;
        seq2[i] = i + 1;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(seq1.begin(), seq1.end(), gen);
    std::shuffle(seq2.begin(), seq2.end(), gen);

    p1_large.citySequence = seq1;
    p2_large.citySequence = seq2;

    // Random job sequences
    std::shuffle(seq1.begin(), seq1.end(), gen);
    std::shuffle(seq2.begin(), seq2.end(), gen);
    p1_large.jobSequence = seq1;
    p2_large.jobSequence = seq2;

    printSequence("P1 cities", p1_large.citySequence);
    printSequence("P2 cities", p2_large.citySequence);

    parents1 = {p1_large};
    parents2 = {p2_large};

    auto children_large = performSimpleBatchCrossover(parents1, parents2, mode);

    if (children_large.empty()) {
        std::cout << "ERROR: No children produced for large problem!" << std::endl;
        return 1;
    }

    printSequence("Child cities", children_large[0].citySequence);
    printSequence("Child jobs", children_large[0].jobSequence);

    if (validateTour(children_large[0].citySequence)) {
        std::cout << "Large child city tour is VALID" << std::endl;
    } else {
        std::cout << "Large child city tour is INVALID!" << std::endl;
        return 1;
    }

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
