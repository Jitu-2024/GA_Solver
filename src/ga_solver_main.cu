// ga_solver_main.cu: Main solver for TSPJ genetic algorithm

#include "genome.h"
#include "population.h"
#include "fitness_evaluator.h"
#include "parent_selection.h"
#include "crossover.h"
#include "mutation.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>

// Function to parse a CSV file into a 2D vector
std::vector<std::vector<float>> parseCSV(const std::string &filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value)); // Convert string to float
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to log results to a CSV file
void logResults(const std::string &logsFolder, const std::string &datasetName, float bestFitness,
                size_t solutionGeneration, size_t totalGenerations, double totalTime, double timePerGeneration) {
    // Construct the log file path
    std::string logFilePath = logsFolder + "/results_log.csv";

    // Open the log file
    std::ofstream logFile(logFilePath, std::ios::app);
    if (!logFile.is_open()) {
        throw std::runtime_error("Could not open " + logFilePath + " for writing");
    }

    // Write header if the file is empty
    static bool headerWritten = false;
    if (!headerWritten) {
        logFile << "Dataset Name,Best Fitness,Solution Generation,Total Generations,Total Time (s),Time Per Generation (s)\n";
        headerWritten = true;
    }

    // Write data
    logFile << datasetName << "," << bestFitness << "," << solutionGeneration << "," << totalGenerations << ","
            << totalTime << "," << timePerGeneration << "\n";

    logFile.close();
}

int main() {
    std::string logsFolder = "../logs";

    // Array of dataset names
    std::vector<std::string> datasets = {"gr17", "gr21", "gr24", "fri26", "bays29", "gr48", "eil51", "berlin52", "eil76", "eil101"};
    // std::vector<std::string> datasets = {"berlin52"};

    for (const auto& datasetName : datasets) {
        try {
            // Construct file paths
            std::string travelTimesFile = "data/" + datasetName + "_TSPJ_TT.csv";
            std::string jobTimesFile = "data/" + datasetName + "_TSPJ_JT.csv";

            // Parse the CSV files
            std::vector<std::vector<float>> travelTimes = parseCSV(travelTimesFile);
            std::vector<std::vector<float>> jobTimes = parseCSV(jobTimesFile);

            // Example from TSP-J paper
            // const std::vector<std::vector<float>> travelTimes = {
            //     {0, 5, 9, 12, 10, 6},
            //     {5, 0, 7, 9, 12, 10},
            //     {9, 7, 0, 5, 10, 12},
            //     {12, 9, 5, 0, 6, 10},
            //     {10, 12, 10, 6, 0, 7},
            //     {6, 10, 12, 10, 7, 0}};

            // const std::vector<std::vector<float>> jobTimes = {
            //     {0, 0, 0, 0, 0, 0},
            //     {0, 20, 22, 32, 25, 33},
            //     {0, 21, 20, 34, 23, 32},
            //     {0, 20, 22, 30, 22, 34},
            //     {0, 22, 24, 31, 22, 32},
            //     {0, 21, 20, 32, 24, 34}};

            // Parameters
            size_t numCities = travelTimes.size() - 1;
            size_t numJobs = jobTimes[0].size() - 1;
            size_t populationSize = 500;
            size_t generations = 10000;
            float mutationRate = 0.3f;
            size_t tournamentSize = 5;
            int mode = 2;  // 1 = pickup ; 0 = no pickup
            size_t diversityCount = static_cast<size_t>(populationSize * 0.2); // 20% diversity

            // Initialize population
            std::vector<Genome> population;
            initializePopulation(population, populationSize, numCities, numJobs, mode);
            // population[0].print(mode);
            // std::vector<size_t> combined = population[0].constructCombinedSequence();
            // std::cout << "Combined sequence : ";
            // for(size_t val : combined){
            //     std::cout << val << " ";
            // }
            // std::cout << std::endl;
            // return 0;

            auto startTime = std::chrono::high_resolution_clock::now();

            // Main GA loop
            float bestFitness = std::numeric_limits<float>::max();
            size_t solutionGeneration = 0;
            size_t stagnationCount = 0; // Counter for generations without fitness improvement
            const size_t maxStagnationGenerations = 1500; // Stopping criterion

            for (size_t generation = 0; generation < generations; ++generation) {
                std::cout << "=============================== GENERATION " << generation << " ===============================" << std::endl;
                auto generationStart = std::chrono::high_resolution_clock::now();

                // Evaluate fitness
                evaluatePopulation(population, travelTimes, jobTimes, mode);

                // Sort by fitness
                sortPopulationByFitness(population);

                // Select parents
                std::vector<Genome> parents = selectParents(population, populationSize / 8, tournamentSize);

                // Generate offspring via crossover and mutation
                std::vector<Genome> offspring;
                for (size_t i = 0; i < parents.size(); i += 2) {
                    Genome parent1 = parents[i];
                    Genome parent2 = parents[(i + 1) % parents.size()];

                    // Generate multiple children per parent pair
                    for (size_t j = 0; j < 3; ++j) {
                        auto [child1, child2] = performCrossover(parent1, parent2, mode);
                        performMutation(child1, mutationRate, mode);
                        performMutation(child2, mutationRate, mode);

                        offspring.push_back(child1);
                        offspring.push_back(child2);
                    }
                }

                // Generate random genomes for diversity
                std::vector<Genome> diversityGenomes;
                initializePopulation(diversityGenomes, diversityCount, numCities, numJobs, mode);

                // Combine offspring and diversity genomes
                offspring.insert(offspring.end(), diversityGenomes.begin(), diversityGenomes.end());

                // Replace worst genomes with offspring and diversity
                if (offspring.size() > population.size()) {
                    offspring.resize(population.size());
                }

                replaceWorst(population, offspring);

                // Output best fitness of current generation
                Genome bestGenome = getBestGenome(population);
                std::cout << "Best Fitness of Generation " << generation << ": " << bestGenome.fitness << "\n";

                if (bestGenome.fitness < bestFitness) {
                    bestFitness = bestGenome.fitness;
                    solutionGeneration = generation;
                    stagnationCount = 0; // Reset stagnation count
                } else {
                    stagnationCount++; // Increment stagnation count
                }

                if (stagnationCount >= maxStagnationGenerations) {
                    std::cout << "Stopping early due to no improvement in best fitness for " 
                            << maxStagnationGenerations << " generations.\n";
                    break;
                }

                auto generationEnd = std::chrono::high_resolution_clock::now();
                double generationTime = std::chrono::duration<double>(generationEnd - generationStart).count();
                std::cout << "Time for Generation " << generation << ": " << generationTime << " seconds\n";
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            double timePerGeneration = totalTime / generations;

            // Log results to CSV
            logResults(logsFolder, datasetName, bestFitness, solutionGeneration, generations, totalTime, timePerGeneration);
            std::cout << "Log written successfully for dataset " << datasetName << "\n";

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset " << datasetName << ": " << e.what() << std::endl;
        }
    }

    return 0;
}
