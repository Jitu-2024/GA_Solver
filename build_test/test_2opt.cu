// Quick test of 2-opt improvement
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// Simple tour cost calculation
float calculateTourCost(const std::vector<size_t>& tour, 
                        const std::vector<std::vector<float>>& travelTimes) {
    size_t n = tour.size();
    float cost = travelTimes[0][tour[0]];  // depot to first
    for (size_t i = 0; i < n - 1; i++) {
        cost += travelTimes[tour[i]][tour[i+1]];
    }
    cost += travelTimes[tour[n-1]][0];  // last to depot
    return cost;
}

// Apply 2-opt move
void apply2Opt(std::vector<size_t>& tour, size_t i, size_t j) {
    std::reverse(tour.begin() + i, tour.begin() + j + 1);
}

// Calculate delta for 2-opt
float calc2OptDelta(const std::vector<size_t>& tour,
                    const std::vector<std::vector<float>>& cost,
                    size_t i, size_t j) {
    size_t n = tour.size();
    
    // Get the cities
    size_t prev_i = (i == 0) ? 0 : tour[i-1];  // depot if i=0
    size_t city_i = tour[i];
    size_t city_j = tour[j];
    size_t next_j = (j == n-1) ? 0 : tour[j+1];  // depot if j=last
    
    float oldCost = cost[prev_i][city_i] + cost[city_j][next_j];
    float newCost = cost[prev_i][city_j] + cost[city_i][next_j];
    
    return oldCost - newCost;  // positive = improvement
}

int main() {
    // Load TSPJ_1M travel times
    std::ifstream file("data/TSPJ_1M_TSPJ_TT.csv");
    if (!file.is_open()) {
        std::cerr << "Cannot open file" << std::endl;
        return 1;
    }
    
    std::vector<std::vector<float>> travelTimes;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        float val;
        while (ss >> val) {
            row.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
        if (!row.empty()) travelTimes.push_back(row);
    }
    
    size_t numCities = travelTimes.size() - 1;  // exclude depot
    std::cout << "Loaded " << travelTimes.size() << " cities (including depot)" << std::endl;
    
    // Create random tour
    std::vector<size_t> tour(numCities);
    std::iota(tour.begin(), tour.end(), 1);  // 1 to numCities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(tour.begin(), tour.end(), gen);
    
    float initialCost = calculateTourCost(tour, travelTimes);
    std::cout << "Initial random tour cost: " << initialCost << std::endl;
    
    // Apply 2-opt improvements until no improvement found
    bool improved = true;
    int iterations = 0;
    while (improved) {
        improved = false;
        for (size_t i = 0; i < tour.size() - 1 && !improved; i++) {
            for (size_t j = i + 2; j < tour.size(); j++) {
                float delta = calc2OptDelta(tour, travelTimes, i, j);
                if (delta > 0.01f) {
                    apply2Opt(tour, i, j);
                    improved = true;
                    iterations++;
                    break;
                }
            }
        }
    }
    
    float finalCost = calculateTourCost(tour, travelTimes);
    std::cout << "After " << iterations << " 2-opt moves: " << finalCost << std::endl;
    std::cout << "Improvement: " << (initialCost - finalCost) << " (" 
              << (100.0f * (initialCost - finalCost) / initialCost) << "%)" << std::endl;
    
    return 0;
}
