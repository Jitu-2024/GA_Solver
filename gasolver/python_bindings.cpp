#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/cost_matrix.h"
#include "../include/population.h"
#include "../include/fitness_evaluator.h"
#include <stdexcept>

namespace py = pybind11;

class PythonGASolverTSP {
public:
    struct SolverMetrics {
        float convergence_rate;
        float diversity_score;
        int generations_to_converge;
        std::vector<float> population_diversity_history;
    };

    PythonGASolverTSP(
        int populationSize = 1000,
        int numGenerations = 50000,
        int tournamentSize = 10,
        float mutationRate = 0.01f,
        float elitismRate = 0.05f
    ) {
        validateParameters(populationSize, numGenerations, tournamentSize, mutationRate, elitismRate);
        
        populationSize_ = populationSize;
        numGenerations_ = numGenerations;
        tournamentSize_ = tournamentSize;
        mutationRate_ = mutationRate;
        elitismRate_ = elitismRate;
        population_ = nullptr;
        current_generation_ = 0;
        best_fitness_ = std::numeric_limits<float>::max();
    }

    py::dict solve(const py::array_t<float>& coordinates_or_matrix, bool is_matrix = false) {
        try {
            if (is_matrix) {
                return solveFromMatrix(coordinates_or_matrix);
            } else {
                return solveFromCoordinates(coordinates_or_matrix);
            }
        } catch (const std::exception& e) {
            throw py::value_error(std::string("Solver error: ") + e.what());
        }
    }

    py::dict get_solver_state() const {
        if (!population_) {
            throw py::value_error("Solver has not been initialized yet");
        }

        py::dict state;
        state["current_generation"] = current_generation_;
        state["best_fitness"] = best_fitness_;
        state["population_diversity"] = calculateDiversity();
        state["convergence_score"] = calculateConvergenceScore();
        return state;
    }

    py::dict step(const py::array_t<float>& action) {
        validateAction(action);
        applyAction(action);
        runGeneration();
        return createStepResponse();
    }

    py::dict reset(const py::array_t<float>& coordinates_or_matrix, bool is_matrix = false) {
        current_generation_ = 0;
        best_fitness_ = std::numeric_limits<float>::max();
        initialize(coordinates_or_matrix);
        return get_solver_state();
    }

private:
    int populationSize_;
    int numGenerations_;
    int tournamentSize_;
    float mutationRate_;
    float elitismRate_;
    int current_generation_;
    float best_fitness_;
    std::unique_ptr<Population> population_;

    void validateParameters(int populationSize, int numGenerations, int tournamentSize, 
                          float mutationRate, float elitismRate) {
        if (populationSize <= 0) throw std::invalid_argument("Population size must be positive");
        if (numGenerations <= 0) throw std::invalid_argument("Number of generations must be positive");
        if (tournamentSize <= 0 || tournamentSize > populationSize) 
            throw std::invalid_argument("Invalid tournament size");
        if (mutationRate < 0.0f || mutationRate > 1.0f) 
            throw std::invalid_argument("Mutation rate must be between 0 and 1");
        if (elitismRate < 0.0f || elitismRate > 1.0f) 
            throw std::invalid_argument("Elitism rate must be between 0 and 1");
    }

    void validateAction(const py::array_t<float>& action) {
        auto buffer = action.request();
        if (buffer.ndim != 1) throw std::invalid_argument("Action must be 1-dimensional");
        if (buffer.shape[0] != 4) throw std::invalid_argument("Action must have 4 components");
    }

    void applyAction(const py::array_t<float>& action) {
        auto buffer = action.request();
        auto ptr = static_cast<float*>(buffer.ptr);
        
        // Scale actions to parameter ranges
        mutationRate_ = std::clamp(ptr[0], 0.0f, 1.0f);
        elitismRate_ = std::clamp(ptr[1], 0.0f, 1.0f);
        tournamentSize_ = static_cast<int>(std::clamp(ptr[2] * populationSize_, 2.0f, 
                                                     static_cast<float>(populationSize_)));
    }

    void runGeneration() {
        if (!population_) {
            throw std::runtime_error("Population not initialized");
        }

        FitnessEvaluator evaluator;
        population_->evaluateFitness(evaluator);

        // Select elite individuals
        int numElites = static_cast<int>(populationSize_ * elitismRate_);
        std::vector<Genome> elites = population_->getTopGenomes(numElites);

        // Calculate number of offspring needed
        int numOffspring = populationSize_ - numElites;
        int numParents = (numOffspring % 2 == 0) ? numOffspring : numOffspring + 1;

        // Select parents and perform crossover
        std::vector<Genome> parents;
        population_->selectParentsTournamentGPU(parents, numParents, tournamentSize_);
        
        std::vector<Genome> offspring;
        std::vector<std::pair<int, int>> crossoverPoints;
        population_->performCrossoverGPU(parents, offspring, crossoverPoints);

        // Perform mutation
        population_->performMutationGPU(offspring, mutationRate_);

        // Combine elites and offspring
        offspring.insert(offspring.end(), elites.begin(), elites.end());
        population_->setGenomes(offspring);

        current_generation_++;
        updateBestFitness();
    }

    py::dict createStepResponse() {
        py::dict response;
        response["state"] = get_solver_state();
        response["reward"] = calculateReward();
        response["done"] = current_generation_ >= numGenerations_;
        response["info"] = py::dict();
        return response;
    }

    py::dict solveFromMatrix(const py::array_t<float>& distance_matrix) {
        auto buffer = distance_matrix.request();
        if (buffer.ndim != 2) throw std::invalid_argument("Distance matrix must be 2-dimensional");
        if (buffer.shape[0] != buffer.shape[1]) 
            throw std::invalid_argument("Distance matrix must be square");

        population_ = std::make_unique<Population>(populationSize_, buffer.shape[0]);
        population_->initialize();
        population_->initializeCostMatrix(convertToVector2D(distance_matrix));

        return runOptimization();
    }

    py::dict solveFromCoordinates(const py::array_t<float>& coordinates) {
        auto buffer = coordinates.request();
        if (buffer.ndim != 2) throw std::invalid_argument("Coordinates must be 2-dimensional");
        if (buffer.shape[1] != 2) throw std::invalid_argument("Each coordinate must have 2 values (x,y)");

        population_ = std::make_unique<Population>(populationSize_, buffer.shape[0]);
        population_->initialize();
        population_->initializeCostMatrix(convertToCoordinates(coordinates));

        return runOptimization();
    }

    std::vector<std::vector<float>> convertToVector2D(const py::array_t<float>& array) {
        auto buffer = array.request();
        auto ptr = static_cast<float*>(buffer.ptr);
        std::vector<std::vector<float>> result(buffer.shape[0], 
                                             std::vector<float>(buffer.shape[1]));
        
        for (size_t i = 0; i < buffer.shape[0]; ++i) {
            for (size_t j = 0; j < buffer.shape[1]; ++j) {
                result[i][j] = ptr[i * buffer.shape[1] + j];
            }
        }
        return result;
    }

    std::vector<std::pair<float, float>> convertToCoordinates(const py::array_t<float>& array) {
        auto buffer = array.request();
        auto ptr = static_cast<float*>(buffer.ptr);
        std::vector<std::pair<float, float>> result;
        
        for (size_t i = 0; i < buffer.shape[0]; ++i) {
            result.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
        }
        return result;
    }

    py::dict runOptimization() {
        for (int i = 0; i < numGenerations_; ++i) {
            runGeneration();
        }

        py::dict result;
        result["best_fitness"] = best_fitness_;
        result["best_route"] = population_->getFittest().getChromosome();
        result["metrics"] = getSolverMetrics();
        return result;
    }

    void updateBestFitness() {
        float current_fitness = population_->getFittest().getFitness();
        if (current_fitness < best_fitness_) {
            best_fitness_ = current_fitness;
        }
    }

    float calculateDiversity() const {
        // Implement population diversity metric
        return 0.0f;
    }

    float calculateConvergenceScore() const {
        // Implement convergence metric
        return 0.0f;
    }

    float calculateReward() const {
        // Simple reward based on improvement in best fitness
        return -best_fitness_;  // Negative because we're minimizing
    }

    py::dict getSolverMetrics() const {
        py::dict metrics;
        metrics["convergence_rate"] = calculateConvergenceScore();
        metrics["diversity_score"] = calculateDiversity();
        metrics["generations_to_converge"] = current_generation_;
        return metrics;
    }

    void initialize(const py::array_t<float>& coordinates_or_matrix) {
        auto buffer = coordinates_or_matrix.request();
        if (buffer.ndim != 2) throw std::invalid_argument("Input must be 2-dimensional");
        
        population_ = std::make_unique<Population>(populationSize_, buffer.shape[0]);
        population_->initialize();  // Call Population's initialize method
        
        CostMatrix cost_matrix(buffer.shape[0]);
        if (buffer.shape[1] == 2) {
            // Handle coordinates
            std::vector<std::pair<float, float>> coords;
            float* ptr = static_cast<float*>(buffer.ptr);
            for (size_t i = 0; i < buffer.shape[0]; i++) {
                coords.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
            }
            cost_matrix.initialize(coords);
        } else {
            // Handle matrix
            std::vector<std::vector<float>> matrix(buffer.shape[0], 
                std::vector<float>(buffer.shape[1]));
            float* ptr = static_cast<float*>(buffer.ptr);
            for (size_t i = 0; i < buffer.shape[0]; i++) {
                for (size_t j = 0; j < buffer.shape[1]; j++) {
                    matrix[i][j] = ptr[i * buffer.shape[1] + j];
                }
            }
            cost_matrix.initialize(matrix);
        }
        population_->initializeCostMatrix(cost_matrix.getCostMatrix());
    }
};

PYBIND11_MODULE(_core, m) {
    //gasolver, m) {
    m.doc() = R"pbdoc(
        CUDA-accelerated Genetic Algorithm Solver for TSP
        ===============================================

        This module provides a high-performance implementation of a genetic algorithm
        for solving the Traveling Salesman Problem (TSP) using CUDA acceleration.

        Key Features:
            * CUDA-accelerated fitness evaluation and genetic operations
            * Support for both coordinate-based and distance matrix inputs
            * Reinforcement learning interface for parameter tuning
            * Comprehensive metrics and state tracking
            * Visualization utilities

        Example:
            >>> import gasolver
            >>> solver = gasolver.GASolver(population_size=1000)
            >>> result = solver.solve(coordinates)
            >>> print(f"Best fitness: {result['best_fitness']}")
    )pbdoc";

    py::class_<PythonGASolverTSP>(m, "GASolver")
        .def(py::init<int, int, int, float, float>(),
            py::arg("population_size") = 1000,
            py::arg("num_generations") = 50000,
            py::arg("tournament_size") = 10,
            py::arg("mutation_rate") = 0.1f,
            py::arg("elitism_rate") = 0.05f,
            R"pbdoc(
                Initialize the GA Solver.

                Parameters
                ----------
                population_size : int, optional (default=1000)
                    Size of the population in each generation
                num_generations : int, optional (default=50000)
                    Maximum number of generations to run
                tournament_size : int, optional (default=10)
                    Size of tournament for parent selection
                mutation_rate : float, optional (default=0.1)
                    Probability of mutation for each gene
                elitism_rate : float, optional (default=0.05)
                    Fraction of population preserved as elite individuals

                Raises
                ------
                ValueError
                    If any parameters are invalid
            )pbdoc")
        .def("solve", &PythonGASolverTSP::solve,
            py::arg("coordinates_or_matrix"),
            py::arg("is_matrix") = false,
            R"pbdoc(
                Solve the TSP instance.

                Parameters
                ----------
                coordinates_or_matrix : numpy.ndarray
                    Either a Nx2 array of coordinates or a NxN distance matrix
                is_matrix : bool, optional (default=False)
                    Whether the input is a distance matrix

                Returns
                -------
                dict
                    Dictionary containing:
                    - best_fitness: float
                    - best_route: list[int]
                    - fitness_history: list[float]
                    - metrics: dict
                        - convergence_rate: float
                        - diversity_score: float
                        - generations_to_converge: int
                        - population_diversity_history: list[float]

                Raises
                ------
                ValueError
                    If input data is invalid
            )pbdoc")
        .def("get_solver_state", &PythonGASolverTSP::get_solver_state,
            R"pbdoc(
                Get current solver state.

                Returns
                -------
                dict
                    Current solver state including metrics
            )pbdoc")
        .def("step", &PythonGASolverTSP::step,
            py::arg("action"),
            R"pbdoc(
                Perform one step for reinforcement learning.

                Parameters
                ----------
                action : numpy.ndarray
                    Action vector for modifying solver parameters

                Returns
                -------
                dict
                    Dictionary containing:
                    - state: Current solver state
                    - reward: float
                    - done: bool
                    - info: dict
            )pbdoc")
        .def("reset", &PythonGASolverTSP::reset,
            py::arg("coordinates_or_matrix"),
            py::arg("is_matrix") = false,
            R"pbdoc(
                Reset the solver for a new episode.

                Parameters
                ----------
                coordinates_or_matrix : numpy.ndarray
                    Problem instance data
                is_matrix : bool, optional (default=False)
                    Whether the input is a distance matrix

                Returns
                -------
                dict
                    Initial solver state
            )pbdoc");

    py::class_<CostMatrix>(m, "CostMatrix")
        .def(py::init<int>())
        .def("initialize_from_coordinates", [](CostMatrix& self, py::array_t<float> coords) {
            auto buf = coords.request();
            if (buf.ndim != 2) throw std::runtime_error("Coordinates must be 2-dimensional");
            if (buf.shape[1] != 2) throw std::runtime_error("Coordinates must be Nx2 array");
            
            std::vector<std::pair<float, float>> coordinates;
            float* ptr = static_cast<float*>(buf.ptr);
            
            for (size_t i = 0; i < buf.shape[0]; i++) {
                coordinates.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
            }
            self.initialize(coordinates);
        })
        .def("initialize_from_matrix", [](CostMatrix& self, py::array_t<float> matrix) {
            auto buf = matrix.request();
            if (buf.ndim != 2) throw std::runtime_error("Matrix must be 2-dimensional");
            if (buf.shape[0] != buf.shape[1]) throw std::runtime_error("Matrix must be square");
            
            std::vector<std::vector<float>> cost_matrix(buf.shape[0], 
                std::vector<float>(buf.shape[1]));
            float* ptr = static_cast<float*>(buf.ptr);
            
            for (size_t i = 0; i < buf.shape[0]; i++) {
                for (size_t j = 0; j < buf.shape[1]; j++) {
                    cost_matrix[i][j] = ptr[i * buf.shape[1] + j];
                }
            }
            self.initialize(cost_matrix);
        })
        .def("get_cost_matrix", &CostMatrix::getCostMatrix);
} 