#include "genome.h"
#include <vector>
#include <algorithm>
#include <random>

// Constructor for Genome
Genome::Genome(size_t numCities, size_t numJobs, int mode)
    : citySequence(numCities), jobSequence(numJobs) {
    // Initialize city and job sequences
    std::iota(citySequence.begin(), citySequence.end(), 1); // Cities: 1 to numCities
    std::iota(jobSequence.begin(), jobSequence.end(), 1);   // Jobs: 1 to numJobs

    // Random device and engine
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle sequences
    std::shuffle(citySequence.begin(), citySequence.end(), g);
    std::shuffle(jobSequence.begin(), jobSequence.end(), g);

    if (mode == 1) {
        // Initialize pickup sequence for TSP-J with sequential pickups
        pickupSequence.resize(numCities);
        std::iota(pickupSequence.begin(), pickupSequence.end(), 1);
        std::shuffle(pickupSequence.begin(), pickupSequence.end(), g);
    }

    if (mode == 2) {
        // Initialize pickup offsets for interleaved pickups
        pickupOffset.resize(numCities);
        std::uniform_int_distribution<size_t> dist(0, numCities - 1);
        for (size_t i = 0; i < numCities; ++i) {
            pickupOffset[i] = dist(g); // Random offsets in [0, numCities - 1]
        }
    }

    fitness = std::numeric_limits<float>::max(); // Default high fitness
}

float Genome::getFitness() {
    return fitness;
}

// Print genome for debugging
void Genome::print(int mode) const {
    std::cout << "City Sequence: ";
    for (const auto& city : citySequence) {
        std::cout << city << " ";
    }
    std::cout << "\nJob Sequence: ";
    for (const auto& job : jobSequence) {
        std::cout << job << " ";
    }

    if (mode == 1) {
        std::cout << "\nPickup Sequence: ";
        for (const auto& pickup : pickupSequence) {
            std::cout << pickup << " ";
        }
    } else if (mode == 2) {
        std::cout << "\nPickup Offsets: ";
        for (const auto& offset : pickupOffset) {
            std::cout << offset << " ";
        }
    }

    std::cout << "\nFitness: " << fitness << "\n";
}

// Combined sequence for interleaved pickup
std::vector<size_t> Genome::constructCombinedSequence() const {
    size_t numCities = citySequence.size();
    std::vector<size_t> combinedSequence(numCities * 2, 0); // Combined sequence initialized with 0

    for (size_t i = 0; i < numCities; ++i) {
        size_t dropoffCity = citySequence[i];
        size_t pickupCity = dropoffCity + numCities;

        // 1. Dropoff Placement
        size_t dropoffIndex = i;
        while (combinedSequence[dropoffIndex] != 0) {
            dropoffIndex++; // Find the next available space for the dropoff
        }
        combinedSequence[dropoffIndex] = dropoffCity;

        // 2. Pickup Placement
        size_t intendedPickupIndex = i + pickupOffset[i] + 1;
        size_t pickupIndex = intendedPickupIndex;

        // Find the next available space for the pickup
        while (pickupIndex < combinedSequence.size() && combinedSequence[pickupIndex] != 0) {
            pickupIndex++;
        }

        // Handle conflicts by wrapping around if out of bounds
        if (pickupIndex >= combinedSequence.size()) {
            pickupIndex = 0; // Wrap around
            while (combinedSequence[pickupIndex] != 0) {
                pickupIndex++;
            }
        }

        // Place the pickup
        combinedSequence[pickupIndex] = pickupCity;
    }

    return combinedSequence;
}
