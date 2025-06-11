// enhanced_ga_solver_main.cu: Enhanced main solver for TSPJ genetic algorithm with comprehensive logging

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
#include <string>
#include <cstring>
#include <iomanip>

// Structure to hold convergence data
struct ConvergenceData {
    std::vector<size_t> generations;
    std::vector<float> best_fitness;
    std::vector<float> avg_fitness;
    std::vector<double> time_stamps;
};

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

// Function to calculate average fitness of population
float calculateAverageFitness(const std::vector<Genome>& population) {
    float sum = 0.0f;
    for (const auto& genome : population) {
        sum += genome.fitness;
    }
    return sum / population.size();
}

// Function to log convergence data to CSV
void logConvergenceData(const std::string &filename, const ConvergenceData &convergenceData) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open convergence file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "Generation,Best_Fitness,Average_Fitness,Time_Elapsed" << std::endl;

    // Write data
    for (size_t i = 0; i < convergenceData.generations.size(); ++i) {
        file << convergenceData.generations[i] << ","
             << std::fixed << std::setprecision(6) << convergenceData.best_fitness[i] << ","
             << std::fixed << std::setprecision(6) << convergenceData.avg_fitness[i] << ","
             << std::fixed << std::setprecision(6) << convergenceData.time_stamps[i] << std::endl;
    }

    file.close();
}

// Function to log detailed results to a comprehensive CSV file
void logDetailedResults(const std::string &logsFolder, const std::string &datasetName, 
                       float bestFitness, size_t solutionGeneration, size_t totalGenerations, 
                       double totalTime, double timePerGeneration, int mode, 
                       const std::string &extraInfo, const Genome &bestGenome,
                       size_t populationSize, float mutationRate, size_t tournamentSize,
                       size_t maxStagnationGenerations, float diversityPercent, int seed = -1) {
    
    // Construct the detailed log file path
    std::string logFilePath = logsFolder + "/detailed_results.csv";

    // Check if file exists to determine if we need to write header
    bool fileExists = false;
    std::ifstream checkFile(logFilePath);
    if (checkFile.good()) {
        fileExists = true;
    }
    checkFile.close();

    // Open the log file in append mode
    std::ofstream logFile(logFilePath, std::ios::app);
    if (!logFile.is_open()) {
        throw std::runtime_error("Could not open " + logFilePath + " for writing");
    }

    // Write header if the file is new
    if (!fileExists) {
        logFile << "Timestamp,Dataset,Seed,Best_Fitness,Solution_Generation,Total_Generations,"
                << "Total_Time_s,Time_Per_Generation_s,Population_Size,Mutation_Rate,"
                << "Tournament_Size,Max_Stagnation_Generations,Diversity_Percent,Mode,"
                << "City_Sequence,Job_Sequence,Extra_Info" << std::endl;
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);

    // Write data
    logFile << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ","
            << datasetName << ","
            << seed << ","
            << std::fixed << std::setprecision(6) << bestFitness << ","
            << solutionGeneration << ","
            << totalGenerations << ","
            << std::fixed << std::setprecision(6) << totalTime << ","
            << std::fixed << std::setprecision(6) << timePerGeneration << ","
            << populationSize << ","
            << std::fixed << std::setprecision(3) << mutationRate << ","
            << tournamentSize << ","
            << maxStagnationGenerations << ","
            << std::fixed << std::setprecision(1) << diversityPercent << ","
            << mode << ",\"";

    // Log city sequence
    for (size_t i = 0; i < bestGenome.citySequence.size(); ++i) {
        if (i > 0) logFile << " ";
        logFile << bestGenome.citySequence[i];
    }
    logFile << "\",\"";

    // Log job sequence
    for (size_t i = 0; i < bestGenome.jobSequence.size(); ++i) {
        if (i > 0) logFile << " ";
        logFile << bestGenome.jobSequence[i];
    }
    logFile << "\"," << extraInfo << std::endl;

    logFile.close();
}

// Function to log summary statistics
void logSummaryStats(const std::string &logsFolder, const std::string &datasetName,
                    const ConvergenceData &convergenceData, float bestFitness,
                    double totalTime, size_t totalGenerations) {
    
    std::string summaryPath = logsFolder + "/summary_" + datasetName + ".txt";
    std::ofstream summaryFile(summaryPath);
    
    if (!summaryFile.is_open()) {
        std::cerr << "Could not open summary file: " << summaryPath << std::endl;
        return;
    }

    summaryFile << "=== GA SOLVER SUMMARY FOR " << datasetName << " ===" << std::endl;
    summaryFile << "Best Fitness: " << std::fixed << std::setprecision(6) << bestFitness << std::endl;
    summaryFile << "Total Time: " << std::fixed << std::setprecision(3) << totalTime << " seconds" << std::endl;
    summaryFile << "Total Generations: " << totalGenerations << std::endl;
    summaryFile << "Final Average Fitness: " << std::fixed << std::setprecision(6) 
                << convergenceData.avg_fitness.back() << std::endl;
    summaryFile << "Convergence Rate: " << std::fixed << std::setprecision(6)
                << (convergenceData.best_fitness[0] - bestFitness) / totalGenerations << " per generation" << std::endl;

    summaryFile.close();
}

// Display help message
void displayHelp() {
    std::cout << "Usage: ga_solver_enhanced [options]" << std::endl;
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
    std::cout << "  --seed                    Random seed for reproducibility (default: random)" << std::endl;
    std::cout << "  --num-runs                Number of runs per dataset (default: 1)" << std::endl;
    std::cout << "  --convergence-log         Enable convergence logging (default: enabled)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output Files:" << std::endl;
    std::cout << "  individual_results.csv    Individual run results" << std::endl;
    std::cout << "  aggregated_results.csv    Aggregated statistics per dataset" << std::endl;
    std::cout << "  convergence_*.csv         Generation-by-generation convergence data" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Single run on gr17" << std::endl;
    std::cout << "  ./ga_solver_enhanced --datasets gr17 --seed 42" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Multiple runs for statistical analysis" << std::endl;
    std::cout << "  ./ga_solver_enhanced --datasets gr17,gr21 --num-runs 5 --seed 42" << std::endl;
    std::cout << std::endl;
    std::cout << "  # All datasets with custom parameters" << std::endl;
    std::cout << "  ./ga_solver_enhanced --datasets bays29,berlin52,eil101,eil51,eil76,fri26,gr17,gr21,gr24,gr48 \\" << std::endl;
    std::cout << "                       --num-runs 5 --population-size 800 --mutation-rate 0.3" << std::endl;
}Tournament size for parent selection (default: 5)" << std::endl;
    std::cout << "  -o, --mode                Mode: 0=no pickup, 1=sequential, 2=interleaved (default: 2)" << std::endl;
    std::cout << "  -s, --stagnation-limit    Max generations without improvement before stopping (default: 1500)" << std::endl;
    std::cout << "  -v, --diversity-percent   Percentage of population to replace with random genomes (default: 20)" << std::endl;
    std::cout << "  -l, --logs-folder         Folder for logs (default: ../logs)" << std::endl;
    std::cout << "  --seed                    Random seed for reproducibility (default: random)" << std::endl;
    std::cout << "  --convergence-log         Enable convergence logging (default: enabled)" << std::endl;
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
    int seed = -1; // -1 means random seed
    bool enableConvergenceLogging = true;
    int numRuns = 1; // Number of runs per dataset for statistical analysis

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
        } else if (arg == "--seed") {
            if (i + 1 < argc) {
                seed = std::stoi(argv[++i]);
            }
        } else if (arg == "--num-runs") {
            if (i + 1 < argc) {
                numRuns = std::stoi(argv[++i]);
            }
        } else if (arg == "--convergence-log") {
            enableConvergenceLogging = true;
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
    std::cout << "Number of Runs per Dataset: " << numRuns << std::endl;
    std::cout << "Population Size: " << populationSize << std::endl;
    std::cout << "Generations: " << generations << std::endl;
    std::cout << "Mutation Rate: " << mutationRate << std::endl;
    std::cout << "Tournament Size: " << tournamentSize << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Max Stagnation Generations: " << maxStagnationGenerations << std::endl;
    std::cout << "Diversity Percentage: " << diversityPercent << "%" << std::endl;
    std::cout << "Logs Folder: " << logsFolder << std::endl;
    std::cout << "Base Seed: " << seed << std::endl;
    std::cout << "Convergence Logging: " << (enableConvergenceLogging ? "Enabled" : "Disabled") << std::endl;
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

            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "PROCESSING DATASET: " << datasetName << std::endl;
            std::cout << "Number of Cities: " << numCities << std::endl;
            std::cout << "Number of Jobs: " << numJobs << std::endl;
            std::cout << "Number of Runs: " << numRuns << std::endl;
            std::cout << std::string(80, '=') << std::endl;

            // Storage for multiple runs
            std::vector<float> allFitnessScores;
            std::vector<size_t> allSolutionGenerations;
            std::vector<double> allTotalTimes;
            std::vector<int> allSeeds;
            Genome bestOverallGenome;
            float bestOverallFitness = std::numeric_limits<float>::max();

            // Run multiple experiments for this dataset
            for (int runIndex = 0; runIndex < numRuns; ++runIndex) {
                std::cout << "\n" << std::string(60, '-') << std::endl;
                std::cout << "RUN " << (runIndex + 1) << "/" << numRuns << " FOR DATASET: " << datasetName << std::endl;
                std::cout << std::string(60, '-') << std::endl;

                // Set seed for this run
                int currentSeed = (seed == -1) ? 
                    static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + runIndex :
                    seed + runIndex;
                
                srand(currentSeed);
                cudaDeviceSynchronize();
                allSeeds.push_back(currentSeed);

                std::cout << "Current Seed: " << currentSeed << std::endl;

                // Initialize convergence tracking
                ConvergenceData convergenceData;
                
                // Initialize population
                std::vector<Genome> population;
                initializePopulation(population, populationSize, numCities, numJobs, mode);

                auto startTime = std::chrono::high_resolution_clock::now();

                // Main GA loop
                float bestFitness = std::numeric_limits<float>::max();
                size_t solutionGeneration = 0;
                size_t stagnationCount = 0;
                Genome bestGenome;

                for (size_t generation = 0; generation < generations; ++generation) {
                    auto generationStart = std::chrono::high_resolution_clock::now();

                    // Evaluate fitness
                    evaluatePopulation(population, travelTimes, jobTimes, mode);

                    // Sort by fitness
                    sortPopulationByFitness(population);

                    // Get current best and average fitness
                    Genome currentBest = getBestGenome(population);
                    float currentBestFitness = currentBest.fitness;
                    float avgFitness = calculateAverageFitness(population);

                    // Log convergence data
                    if (enableConvergenceLogging) {
                        auto currentTime = std::chrono::high_resolution_clock::now();
                        double elapsedTime = std::chrono::duration<double>(currentTime - startTime).count();
                        
                        convergenceData.generations.push_back(generation);
                        convergenceData.best_fitness.push_back(currentBestFitness);
                        convergenceData.avg_fitness.push_back(avgFitness);
                        convergenceData.time_stamps.push_back(elapsedTime);
                    }

                    // Update best solution if improved
                    if (currentBestFitness < bestFitness) {
                        bestFitness = currentBestFitness;
                        bestGenome = currentBest;
                        solutionGeneration = generation;
                        stagnationCount = 0;
                    } else {
                        stagnationCount++;
                    }

                    // Output progress every 100 generations or when new best is found
                    if (generation % 100 == 0 || stagnationCount == 0) {
                        std::cout << "Generation " << generation 
                                  << " | Best: " << currentBestFitness 
                                  << " | Avg: " << avgFitness 
                                  << " | Overall Best: " << bestFitness 
                                  << " (Gen " << solutionGeneration << ")"
                                  << " | Stagnation: " << stagnationCount << std::endl;
                    }

                    // Check for early stopping
                    if (stagnationCount >= maxStagnationGenerations) {
                        std::cout << "Early stopping at generation " << generation 
                                  << " due to " << maxStagnationGenerations << " generations without improvement." << std::endl;
                        break;
                    }

                    // Genetic operations
                    std::vector<Genome> parents = selectParents(population, populationSize / 8, tournamentSize);
                    
                    std::vector<Genome> offspring;
                    for (size_t i = 0; i < parents.size(); i += 2) {
                        Genome parent1 = parents[i];
                        Genome parent2 = parents[(i + 1) % parents.size()];

                        for (size_t j = 0; j < 3; ++j) {
                            auto [child1, child2] = performCrossover(parent1, parent2, mode);
                            performMutation(child1, mutationRate, mode);
                            performMutation(child2, mutationRate, mode);

                            offspring.push_back(child1);
                            offspring.push_back(child2);
                        }
                    }

                    // Add diversity
                    std::vector<Genome> diversityGenomes;
                    initializePopulation(diversityGenomes, diversityCount, numCities, numJobs, mode);
                    offspring.insert(offspring.end(), diversityGenomes.begin(), diversityGenomes.end());

                    if (offspring.size() > population.size()) {
                        offspring.resize(population.size());
                    }

                    replaceWorst(population, offspring);

                    auto generationEnd = std::chrono::high_resolution_clock::now();
                    double generationTime = std::chrono::duration<double>(generationEnd - generationStart).count();
                }

                auto endTime = std::chrono::high_resolution_clock::now();
                double totalTime = std::chrono::duration<double>(endTime - startTime).count();

                // Store results for this run
                allFitnessScores.push_back(bestFitness);
                allSolutionGenerations.push_back(solutionGeneration);
                allTotalTimes.push_back(totalTime);

                // Update overall best
                if (bestFitness < bestOverallFitness) {
                    bestOverallFitness = bestFitness;
                    bestOverallGenome = bestGenome;
                }

                // Log individual run result
                std::string runId = "R" + std::to_string(runIndex + 1);
                logIndividualResult(logsFolder, datasetName, bestFitness, solutionGeneration,
                                  convergenceData.generations.size(), totalTime, mode, bestGenome,
                                  populationSize, mutationRate, tournamentSize,
                                  maxStagnationGenerations, diversityPercent, currentSeed, runId);

                // Log convergence data for this run
                if (enableConvergenceLogging) {
                    std::string convergenceFile = logsFolder + "/convergence_" + datasetName + 
                                                "_run" + std::to_string(runIndex + 1) + 
                                                "_seed" + std::to_string(currentSeed) + ".csv";
                    logConvergenceData(convergenceFile, convergenceData);
                }

                std::cout << "Run " << (runIndex + 1) << " completed!" << std::endl;
                std::cout << "Best Fitness: " << bestFitness << std::endl;
                std::cout << "Solution Generation: " << solutionGeneration << std::endl;
                std::cout << "Total Time: " << totalTime << " seconds" << std::endl;
            }

            // Log aggregated results for this dataset
            logAggregatedResults(logsFolder, datasetName, allFitnessScores, allSolutionGenerations,
                               allTotalTimes, generations, mode, populationSize, mutationRate,
                               tournamentSize, maxStagnationGenerations, diversityPercent,
                               allSeeds, bestOverallGenome);

            // Print summary for this dataset
            float meanFitness = std::accumulate(allFitnessScores.begin(), allFitnessScores.end(), 0.0f) / allFitnessScores.size();
            float meanSolutionGen = std::accumulate(allSolutionGenerations.begin(), allSolutionGenerations.end(), 0.0) / allSolutionGenerations.size();
            double meanTime = std::accumulate(allTotalTimes.begin(), allTotalTimes.end(), 0.0) / allTotalTimes.size();

            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "SUMMARY FOR DATASET: " << datasetName << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "Best Fitness: " << bestOverallFitness << std::endl;
            std::cout << "Mean Fitness: " << meanFitness << std::endl;
            std::cout << "Mean Solution Generation: " << meanSolutionGen << std::endl;
            std::cout << "Mean Time: " << meanTime << " seconds" << std::endl;
            std::cout << "Seeds Used: ";
            for (int s : allSeeds) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "Best Solution:" << std::endl;
            bestOverallGenome.print(mode);
            std::cout << std::string(80, '=') << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset " << datasetName << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ALL EXPERIMENTS COMPLETED!" << std::endl;
    std::cout << "Individual results logged to: " << logsFolder << "/individual_results.csv" << std::endl;
    std::cout << "Aggregated results logged to: " << logsFolder << "/aggregated_results.csv" << std::endl;
    std::cout << "Convergence data logged to: " << logsFolder << "/convergence_*.csv" << std::endl;
    std::cout << std::string(80, '=') << std::endl; << datasetName << std::endl;
            std::cout << "Number of Cities: " << numCities << std::endl;
            std::cout << "Number of Jobs: " << numJobs << std::endl;

            // Initialize convergence tracking
            ConvergenceData convergenceData;
            
            // Initialize population
            std::vector<Genome> population;
            initializePopulation(population, populationSize, numCities, numJobs, mode);

            auto startTime = std::chrono::high_resolution_clock::now();

            // Main GA loop
            float bestFitness = std::numeric_limits<float>::max();
            size_t solutionGeneration = 0;
            size_t stagnationCount = 0; // Counter for generations without fitness improvement
            Genome bestGenome;

            for (size_t generation = 0; generation < generations; ++generation) {
                std::cout << "=============================== GENERATION " << generation << " ===============================" << std::endl;
                auto generationStart = std::chrono::high_resolution_clock::now();

                // Evaluate fitness
                evaluatePopulation(population, travelTimes, jobTimes, mode);

                // Sort by fitness
                sortPopulationByFitness(population);

                // Get current best and average fitness
                Genome currentBest = getBestGenome(population);
                float currentBestFitness = currentBest.fitness;
                float avgFitness = calculateAverageFitness(population);

                // Log convergence data
                if (enableConvergenceLogging) {
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    double elapsedTime = std::chrono::duration<double>(currentTime - startTime).count();
                    
                    convergenceData.generations.push_back(generation);
                    convergenceData.best_fitness.push_back(currentBestFitness);
                    convergenceData.avg_fitness.push_back(avgFitness);
                    convergenceData.time_stamps.push_back(elapsedTime);
                }

                // Update best solution if improved
                if (currentBestFitness < bestFitness) {
                    bestFitness = currentBestFitness;
                    bestGenome = currentBest;
                    solutionGeneration = generation;
                    stagnationCount = 0; // Reset stagnation count
                    std::cout << "*** NEW BEST SOLUTION FOUND ***" << std::endl;
                } else {
                    stagnationCount++; // Increment stagnation count
                }

                // Output current generation statistics
                std::cout << "Best Fitness of Generation " << generation << ": " << currentBestFitness << std::endl;
                std::cout << "Average Fitness: " << avgFitness << std::endl;
                std::cout << "Overall Best Fitness: " << bestFitness << " (Generation " << solutionGeneration << ")" << std::endl;
                std::cout << "Stagnation Count: " << stagnationCount << "/" << maxStagnationGenerations << std::endl;

                // Check for early stopping
                if (stagnationCount >= maxStagnationGenerations) {
                    std::cout << "Stopping early due to no improvement in best fitness for " 
                            << maxStagnationGenerations << " generations." << std::endl;
                    break;
                }

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

                auto generationEnd = std::chrono::high_resolution_clock::now();
                double generationTime = std::chrono::duration<double>(generationEnd - generationStart).count();
                std::cout << "Time for Generation " << generation << ": " << generationTime << " seconds" << std::endl;
                std::cout << std::endl;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            double timePerGeneration = totalTime / convergenceData.generations.size();

            // Create a string with the experiment parameters
            std::stringstream extraInfoSS;
            extraInfoSS << "pop=" << populationSize << ",mut=" << mutationRate 
                      << ",tour=" << tournamentSize << ",div=" << diversityPercent << "%,seed=" << seed;
            std::string extraInfo = extraInfoSS.str();

            // Print the best solution in the final population
            std::cout << "\n============================ BEST SOLUTION ============================" << std::endl;
            std::cout << "Best solution found:" << std::endl;
            bestGenome.print(mode);
            std::cout << "Solution found at generation: " << solutionGeneration << std::endl;
            std::cout << "Total time: " << totalTime << " seconds" << std::endl;
            std::cout << "Average time per generation: " << timePerGeneration << " seconds" << std::endl;
            std::cout << "=========================================================================" << std::endl;

            // Log detailed results
            logDetailedResults(logsFolder, datasetName, bestFitness, solutionGeneration, 
                             convergenceData.generations.size(), totalTime, timePerGeneration, 
                             mode, extraInfo, bestGenome, populationSize, mutationRate, 
                             tournamentSize, maxStagnationGenerations, diversityPercent, seed);

            // Log convergence data
            if (enableConvergenceLogging) {
                std::string convergenceFile = logsFolder + "/convergence_" + datasetName + "_seed" + std::to_string(seed) + ".csv";
                logConvergenceData(convergenceFile, convergenceData);
                std::cout << "Convergence data logged to: " << convergenceFile << std::endl;
            }

            // Log summary statistics
            logSummaryStats(logsFolder, datasetName, convergenceData, bestFitness, totalTime, convergenceData.generations.size());

            std::cout << "All logs written successfully for dataset " << datasetName << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset " << datasetName << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\n============================ EXPERIMENT COMPLETED ============================" << std::endl;
    std::cout << "All datasets processed successfully!" << std::endl;
    std::cout << "Logs saved to: " << logsFolder << std::endl;
    std::cout << "=============================================================================" << std::endl;

    return 0;
}