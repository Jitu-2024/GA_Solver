from gasolver import GASolverAPI, GAParameters
import numpy as np

def main():
    # Initialize solver with more conservative parameters
    params = GAParameters(
        population_size=512,  # Changed to power of 2 for better memory alignment
        num_generations=1000,
        tournament_size=8,    # Reduced to be safer
        mutation_rate=0.1,
        elitism_rate=0.05
    )
    solver = GASolverAPI(parameters=params)

    # Generate random problem
    # If anything needs to be setup, it's purely the coordinates only?
    coords = np.random.rand(50, 2) * 100
    print(f"+++++++++++++++++++++++++++++++++++++++Coords: {coords.shape}")
    # Solve TSP
    result = solver.solve(coords)
    print(f"Best fitness: {result['best_fitness']}")
    print(f"Convergence rate: {result['metrics']['convergence_rate']}")

if __name__ == "__main__":
    main() 