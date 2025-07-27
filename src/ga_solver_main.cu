// ga_solver_main.cu: Complete main solver for TSPJ genetic algorithm with hybrid crossover

#include "genome.h"
#include "population.h"
#include "fitness_evaluator.h"
#include "parent_selection.h"
#include "crossover.h"
#include "eax_cost_integration.h"
#include "mutation.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstring>

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
                size_t solutionGeneration, size_t totalGenerations, double totalTime, 
                double timePerGeneration, int mode, const std::string &extraInfo) {
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
        logFile << "Dataset Name,Best Fitness,Solution Generation,Total Generations,Total Time (s),Time Per Generation (s),Mode,Extra Info\n";
        headerWritten = true;
    }

    // Write data
    logFile << datasetName << "," << bestFitness << "," << solutionGeneration << "," << totalGenerations << ","
            << totalTime << "," << timePerGeneration << "," << mode << "," << extraInfo << "\n";

    logFile.close();
}

// Display help message
void displayHelp() {
    std::cout << "Usage: ga_solver [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help                Display this help message" << std::endl;
    std::cout << "  -d, --datasets            Comma-separated list of datasets to process (default: gr17)" << std::endl;
    std::cout << "  -p, --population-size     Population size (default: 500)" << std::endl;
    std::cout << "  -g, --generations         Maximum number of generations (default: 10000)" << std::endl;
    std::cout << "  -m, --mutation-rate       Mutation rate (default: 0.3)" << std::endl;
    std::cout << "  -t, --tournament-size     Tournament size for parent selection (default: 5)" << std::endl;
    std::cout << "  -o, --mode                Mode: 0=no pickup, 1=sequential, 2=interleaved (default: 2)" << std::endl;
    std::cout << "  -s, --stagnation-limit    Max generations without improvement before stopping (default: 1500)" << std::endl;
    std::cout << "  -v, --diversity-percent   Percentage of population to replace with random genomes (default: 20)" << std::endl;
    std::cout << "  -l, --logs-folder         Folder for logs (default: ../logs)" << std::endl;
    std::cout << "  -c, --use-cost-aware      Use cost-aware EAX crossover (default: false)" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameter values
    std::string datasetsArg = "gr17";
    std::string logsFolder = "../logs";
    size_t populationSize = 500;
    size_t generations = 10000;
    float mutationRate = 0.3f;
    size_t tournamentSize = 5;
    int mode = 2;
    size_t maxStagnationGenerations = 1500;
    float diversityPercent = 20.0f;
    bool useCostAware = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            displayHelp();
            return 0;
        } else if (arg == "-d" || arg == "--datasets") {
            if (i + 1 < argc) {
                datasetsArg = argv[++i];
            }
        } else if (arg == "-p" || arg == "--population-size") {
            if (i + 1 < argc) {
                populationSize = std::stoul(argv[++i]);
            }
        } else if (arg == "-g" || arg == "--generations") {
            if (i + 1 < argc) {
                generations = std::stoul(argv[++i]);
            }
        } else if (arg == "-m" || arg == "--mutation-rate") {
            if (i + 1 < argc) {
                mutationRate = std::stof(argv[++i]);
            }
        } else if (arg == "-t" || arg == "--tournament-size") {
            if (i + 1 < argc) {
                tournamentSize = std::stoul(argv[++i]);
            }
        } else if (arg == "-o" || arg == "--mode") {
            if (i + 1 < argc) {
                mode = std::stoi(argv[++i]);
            }
        } else if (arg == "-s" || arg == "--stagnation-limit") {
            if (i + 1 < argc) {
                maxStagnationGenerations = std::stoul(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--diversity-percent") {
            if (i + 1 < argc) {
                diversityPercent = std::stof(argv[++i]);
            }
        } else if (arg == "-l" || arg == "--logs-folder") {
            if (i + 1 < argc) {
                logsFolder = argv[++i];
            }
        } else if (arg == "-c" || arg == "--use-cost-aware") {
            useCostAware = true;
        }
    }

    // Parse datasets string into vector
    std::vector<std::string> datasets;
    std::stringstream ss(datasetsArg);
    std::string dataset;
    while (std::getline(ss, dataset, ',')) {
        datasets.push_back(dataset);
    }

    // Print configuration
    std::cout << "=============================== CONFIGURATION ===============================" << std::endl;
    std::cout << "Datasets: " << datasetsArg << std::endl;
    std::cout << "Population Size: " << populationSize << std::endl;
    std::cout << "Generations: " << generations << std::endl;
    std::cout << "Mutation Rate: " << mutationRate << std::endl;
    std::cout << "Tournament Size: " << tournamentSize << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Max Stagnation Generations: " << maxStagnationGenerations << std::endl;
    std::cout << "Diversity Percentage: " << diversityPercent << "%" << std::endl;
    std::cout << "Logs Folder: " << logsFolder << std::endl;
    std::cout << "Crossover Type: " << (useCostAware ? "Cost-Aware EAX+Uniform" : "EAX+Uniform") << std::endl;
    std::cout << "Mutation Type: 2-opt/Swap" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    // Calculate diversity count based on percentage
    size_t diversityCount = static_cast<size_t>(populationSize * (diversityPercent / 100.0));

    for (const auto& datasetName : datasets) {
        try {
            // Construct file paths
            std::string travelTimesFile = "data/" + datasetName + "_TSPJ_TT.csv";
            std::string jobTimesFile = "data/" + datasetName + "_TSPJ_JT.csv";

            // Parse the CSV files
            std::vector<std::vector<float>> travelTimes = parseCSV(travelTimesFile);
            std::vector<std::vector<float>> jobTimes = parseCSV(jobTimesFile);

            // Parameters
            size_t numCities = travelTimes.size() - 1;
            size_t numJobs = jobTimes[0].size() - 1;

            std::cout << "Dataset: " << datasetName << std::endl;
            std::cout << "Number of Cities: " << numCities << std::endl;
            std::cout << "Number of Jobs: " << numJobs << std::endl;

            // Initialize cost matrices for EAX if using cost-aware mode
            if (useCostAware) {
                std::cout << "Initializing cost-aware EAX..." << std::endl;
                initializeEAXCostMatrices(travelTimes, jobTimes);
            }
            
            // Initialize cost matrix for 2-opt mutation
            std::cout << "Initializing 2-opt mutation cost matrix..." << std::endl;
            initializeMutationCostMatrix(travelTimes);

            // Initialize population
            std::vector<Genome> population;
            initializePopulation(population, populationSize, numCities, numJobs, mode);

            auto startTime = std::chrono::high_resolution_clock::now();

            // Main GA loop with hybrid crossover (EAX+Uniform) and 2-opt mutation
            float bestFitness = std::numeric_limits<float>::max();
            size_t solutionGeneration = 0;
            size_t stagnationCount = 0;
            size_t lastImprovementGen = 0;
            float baseMutationRate = mutationRate;  // Store original mutation rate

            for (size_t generation = 0; generation < generations; ++generation) {
                std::cout << "=============================== GENERATION " << generation << " ===============================" << std::endl;
                auto generationStart = std::chrono::high_resolution_clock::now();

                // Evaluate fitness
                evaluatePopulation(population, travelTimes, jobTimes, mode);

                // Sort by fitness
                sortPopulationByFitness(population);

                // Adaptive mutation rate based on stagnation
                float adaptiveMutationRate = baseMutationRate;
                if (stagnationCount > 50) {
                    // Gradually increase mutation rate with stagnation
                    float stagnationFactor = std::min(3.0f, 1.0f + (stagnationCount - 50) / 200.0f);
                    adaptiveMutationRate = std::min(0.9f, baseMutationRate * stagnationFactor);
                    
                    if (stagnationCount % 100 == 0) {
                        std::cout << "Increased mutation rate to " << adaptiveMutationRate 
                                  << " due to " << stagnationCount << " generations of stagnation" << std::endl;
                    }
                }
                
                // Partial restart mechanism for severe stagnation
                if (stagnationCount > 300 && stagnationCount % 300 == 0) {
                    std::cout << "*** PARTIAL RESTART: Replacing bottom 40% of population due to long stagnation ***" << std::endl;
                    
                    size_t restartCount = static_cast<size_t>(populationSize * 0.4);
                    std::vector<Genome> newGenomes;
                    initializePopulation(newGenomes, restartCount, numCities, numJobs, mode);
                    
                    // Also apply heavy mutation to middle 20% to create variation
                    size_t middleStart = static_cast<size_t>(populationSize * 0.4);
                    size_t middleEnd = static_cast<size_t>(populationSize * 0.6);
                    
                    for (size_t i = middleStart; i < middleEnd; i++) {
                        // Apply multiple mutations to create more diversity
                        for (int m = 0; m < 3; m++) {
                            performMutation(population[i], 0.8f, mode, stagnationCount);  // High mutation rate
                        }
                    }
                    
                    // Replace worst individuals with new random ones
                    for (size_t i = 0; i < restartCount; i++) {
                        population[populationSize - 1 - i] = newGenomes[i];
                    }
                    
                    std::cout << "Restart complete: " << restartCount << " new individuals added, "
                              << (middleEnd - middleStart) << " heavily mutated" << std::endl;
                }

                // Select parents (increased number for crossover)
                std::vector<Genome> parents = selectParents(population, populationSize / 2, tournamentSize);
                
                // HYBRID CROSSOVER SECTION (EAX + Uniform)
                std::vector<Genome> offspring;
                
                // Method 1: Batch processing for maximum GPU utilization
                if (parents.size() >= 32) { // Use batch processing for larger parent sets
                    std::cout << "Using batch hybrid processing (EAX+Uniform) for " << parents.size() << " parents" << std::endl;
                    
                    // Prepare parent pairs for batch processing
                    std::vector<Genome> parents1, parents2;
                    for (size_t i = 0; i < parents.size(); i += 2) {
                        parents1.push_back(parents[i]);
                        parents2.push_back(parents[(i + 1) % parents.size()]);
                    }
                    
                    // Generate offspring using batch hybrid crossover
                    std::vector<Genome> batchOffspring;
                    if (useCostAware) {
                        batchOffspring = performBatchCostAwareEAXCrossover(parents1, parents2, mode);
                    } else {
                        batchOffspring = performBatchEAXCrossover(parents1, parents2, mode);
                    }
                    
                    offspring.insert(offspring.end(), batchOffspring.begin(), batchOffspring.end());
                    
                } else { // Method 2: Sequential processing for smaller parent sets
                    std::cout << "Using sequential hybrid processing (EAX+Uniform)" << std::endl;
                    
                    for (size_t i = 0; i < parents.size(); i += 2) {
                        Genome parent1 = parents[i];
                        Genome parent2 = parents[(i + 1) % parents.size()];

                        // Generate single high-quality offspring using hybrid crossover
                        Genome child;
                        if (useCostAware) {
                            child = performCostAwareEAXCrossover(parent1, parent2, mode);
                        } else {
                            child = performCrossover(parent1, parent2, mode); // Now uses EAX+Uniform
                        }
                        
                        offspring.push_back(child);
                        
                        // Generate additional offspring if needed by varying parent selection
                        if (offspring.size() < populationSize / 2) {
                            // Use different parent combinations for diversity
                            size_t altIdx = (i + parents.size() / 2) % parents.size();
                            Genome child2;
                            if (useCostAware) {
                                child2 = performCostAwareEAXCrossover(parent1, parents[altIdx], mode);
                            } else {
                                child2 = performCrossover(parent1, parents[altIdx], mode);
                            }
                            offspring.push_back(child2);
                        }
                    }
                }

                std::cout << "Generated " << offspring.size() << " offspring using hybrid crossover (EAX for routing, Uniform for jobs)" << std::endl;

                std::cout << "Generated " << offspring.size() << " offspring using hybrid crossover (EAX for routing, Uniform for jobs)" << std::endl;

                // DEBUG: Check offspring before mutation
                std::cout << "DEBUG: Starting mutation phase for " << offspring.size() << " offspring..." << std::endl;
                std::cout.flush();

                // Apply BATCH mutation to all offspring for efficiency
                if (offspring.size() > 0) {
                    std::cout << "DEBUG: Using batch GPU mutation for " << offspring.size() << " offspring" << std::endl;
                    std::cout.flush();
                    
                    // Store original sequences to check mutation effectiveness
                    std::vector<std::vector<size_t>> originalCitySeqs(offspring.size());
                    std::vector<std::vector<size_t>> originalJobSeqs(offspring.size());
                    std::vector<std::vector<size_t>> originalPickupSeqs(offspring.size());
                    
                    for (size_t i = 0; i < offspring.size(); i++) {
                        originalCitySeqs[i] = offspring[i].citySequence;
                        originalJobSeqs[i] = offspring[i].jobSequence;
                        if (mode == 1) {
                            originalPickupSeqs[i] = offspring[i].pickupSequence;
                        }
                    }
                    
                    // Apply batch mutation
                    performBatchMutation(offspring, adaptiveMutationRate, mode, travelTimes, stagnationCount);
                    
                    std::cout << "DEBUG: Batch mutation completed, checking changes..." << std::endl;
                    std::cout.flush();
                    
                    // Count actual mutations
                    int mutationCount = 0;
                    for (size_t i = 0; i < offspring.size(); i++) {
                        bool mutated = (offspring[i].citySequence != originalCitySeqs[i]) || 
                                      (offspring[i].jobSequence != originalJobSeqs[i]);
                        if (mode == 1) {
                            mutated = mutated || (offspring[i].pickupSequence != originalPickupSeqs[i]);
                        }
                        
                        if (mutated) {
                            mutationCount++;
                        }
                    }
                    
                    std::cout << "Applied " << mutationCount << " successful mutations to " 
                              << offspring.size() << " offspring ("
                              << (100.0 * mutationCount / offspring.size()) << "%)" << std::endl;
                } else {
                    std::cout << "DEBUG: No offspring to mutate" << std::endl;
                }

                std::cout << "DEBUG: Mutation phase completed successfully" << std::endl;
                std::cout.flush();

                // Generate random genomes for diversity
                std::cout << "DEBUG: Starting diversity generation..." << std::endl;
                std::cout.flush();
                
                std::vector<Genome> diversityGenomes;
                initializePopulation(diversityGenomes, diversityCount, numCities, numJobs, mode);
                
                std::cout << "DEBUG: Diversity population created, applying light mutations..." << std::endl;
                std::cout.flush();
                
                // Apply light mutation to some diversity genomes for better integration
                if (diversityCount > 0) {
                    std::vector<Genome> diversityToMutate;
                    for (size_t i = 0; i < diversityCount / 2; i++) {
                        diversityToMutate.push_back(diversityGenomes[i]);
                    }
                    
                    if (!diversityToMutate.empty()) {
                        performBatchMutation(diversityToMutate, 0.5f, mode, travelTimes, stagnationCount);
                        
                        // Copy mutated diversity genomes back
                        for (size_t i = 0; i < diversityToMutate.size(); i++) {
                            diversityGenomes[i] = diversityToMutate[i];
                        }
                    }
                }
                
                offspring.insert(offspring.end(), diversityGenomes.begin(), diversityGenomes.end());

                std::cout << "Added " << diversityCount << " diversity genomes" << std::endl;
                std::cout << "DEBUG: Diversity phase completed" << std::endl;
                std::cout.flush();

                // Ensure we don't exceed population size
                if (offspring.size() > population.size()) {
                    offspring.resize(population.size());
                }

                // Replace worst genomes with offspring
                replaceWorst(population, offspring);

                // Output best fitness of current generation
                Genome bestGenome = getBestGenome(population);
                std::cout << "Best Fitness of Generation " << generation << ": " << bestGenome.fitness << "\n";

                if (bestGenome.fitness < bestFitness) {
                    float improvement = bestFitness - bestGenome.fitness;
                    float improvementPercent = (bestFitness != std::numeric_limits<float>::max()) 
                                               ? (improvement / bestFitness) * 100.0f : 0.0f;
                    
                    bestFitness = bestGenome.fitness;
                    solutionGeneration = generation;
                    stagnationCount = 0; // Reset stagnation count
                    lastImprovementGen = generation;
                    
                    std::cout << "*** NEW BEST FITNESS: " << bestFitness;
                    if (improvementPercent > 0) {
                        std::cout << " (improvement: " << improvement 
                                  << ", " << improvementPercent << "%)";
                    }
                    std::cout << " ***" << std::endl;
                } else {
                    stagnationCount++; // Increment stagnation count
                }

                // Early stopping check
                if (stagnationCount >= maxStagnationGenerations) {
                    std::cout << "Stopping early due to no improvement in best fitness for " 
                            << maxStagnationGenerations << " generations.\n";
                    break;
                }

                auto generationEnd = std::chrono::high_resolution_clock::now();
                double generationTime = std::chrono::duration<double>(generationEnd - generationStart).count();
                std::cout << "Time for Generation " << generation << ": " << generationTime << " seconds\n";
                
                // Population diversity monitoring every 100 generations
                if (generation % 100 == 0 || stagnationCount == 100) {
                    float minFit = population.front().fitness;
                    float maxFit = population.back().fitness;
                    float avgFit = 0.0f;
                    for (const auto& genome : population) {
                        avgFit += genome.fitness;
                    }
                    avgFit /= population.size();
                    
                    std::cout << "=== Population Statistics ===" << std::endl;
                    std::cout << "Min Fitness: " << minFit << ", Avg: " << avgFit 
                              << ", Max: " << maxFit << ", Spread: " << (maxFit - minFit) 
                              << ", Stagnation: " << stagnationCount << " generations" << std::endl;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            double timePerGeneration = totalTime / (solutionGeneration + 1);

            // Create a string with the experiment parameters
            std::stringstream extraInfoSS;
            extraInfoSS << "pop=" << populationSize << ",mut=" << mutationRate 
                      << ",tour=" << tournamentSize << ",div=" << diversityPercent << "%"
                      << ",crossover=" << (useCostAware ? "CostEAX+Uniform" : "EAX+Uniform")
                      << ",mutation=2opt";
            std::string extraInfo = extraInfoSS.str();

            std::cout << "\n============================ BEST SOLUTION ============================" << std::endl;
            std::cout << "Best solution found:" << std::endl;
            Genome bestGenome = getBestGenome(population);
            bestGenome.print(mode);
            std::cout << "=========================================================================" << std::endl;

            // Log results to CSV
            logResults(logsFolder, datasetName, bestFitness, solutionGeneration, 
                       generations, totalTime, timePerGeneration, mode, extraInfo);
            std::cout << "Log written successfully for dataset " << datasetName << "\n";

            // Cleanup cost matrices if using cost-aware mode
            if (useCostAware) {
                cleanupEAXCostMatrices();
            }
            cleanupMutationCostMatrix();

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset " << datasetName << ": " << e.what() << std::endl;
            
            // Cleanup on error
            if (useCostAware) {
                cleanupEAXCostMatrices();
            }
            cleanupMutationCostMatrix();
        }
    }

    return 0;
}