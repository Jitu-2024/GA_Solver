import subprocess
import tempfile
import os
from typing import List, Tuple, Optional, Dict, Any

def run_gpu_ga_tsp_solver(
    tsp_data_file: Optional[str] = None,
    coordinates: Optional[List[Tuple[float, float]]] = None,
    predefined_cost_matrix: Optional[List[List[float]]] = None,
    population_size: int = 1000,
    num_generations: int = 50000,
    tournament_size: int = 10,
    mutation_rate: float = 0.1,
    elitism_rate: float = 0.05,
    initial_genomes_file: Optional[str] = None,
    cuda_solver_executable_path: str = "./ga_solver_main" # Path to the compiled CUDA solver
) -> Dict[str, Any]:
    """
    Executes a GPU-accelerated Genetic Algorithm to solve the Traveling Salesperson Problem (TSP).

    This function serves as a Python interface to a compiled CUDA-based GA solver (ga_solver_main).
    It allows users to specify TSP problem instances and configure GA parameters. The chatbot will
    use this function to process user requests, providing feedback on parameter choices and potential outcomes.

    Parameters:
    ----------
    tsp_data_file : Optional[str]
        Path to a TSP data file (e.g., in TSPLIB format like 'data/dsj1000.tsp'). If provided,
        this is the primary source for city data. The CUDA solver is expected to parse this.

    coordinates : Optional[List[Tuple[float, float]]]
        A list of (x, y) tuples defining city coordinates. If `tsp_data_file` is None, these
        coordinates will be used. The system will create a temporary TSPLIB-like file for the
        CUDA solver if this input is used.
        Example: `[(10.0, 20.5), (30.0, 40.0), ...]`

    predefined_cost_matrix : Optional[List[List[float]]]
        A user-supplied N x N matrix where `matrix[i][j]` is the cost from city `i` to city `j`.
        To be used if `tsp_data_file` and `coordinates` are None. The CUDA solver must be adapted
        or a specific command-line flag implemented to ingest this directly (CURRENTLY NOT SUPPORTED
        BY `ga_solver_main.cu`'s `loadTSPData` or command-line parsing for direct matrix input).
        Example: `[[0, 10, 15], [10, 0, 20], [15, 20, 0]]` for 3 cities.

    population_size : int, default=1000 (as per ga_solver_main.cu default)
        The number of candidate solutions (genomes) maintained in each generation.
        - Impact: Larger populations increase search space coverage, potentially leading to
          better solutions, but significantly increase computational cost per generation.
        - User Guidance: Chatbot should advise on trade-offs with `num_generations` and
          available computation time. Typical range: 50 - 2000.

    num_generations : int, default=50000 (as per ga_solver_main.cu default)
        The total number of iterations the GA will perform.
        - Impact: More generations allow for more thorough exploration and convergence towards
          an optimal solution, but directly increase total runtime.
        - User Guidance: Chatbot can estimate runtime based on this and `population_size`.
          If user has a time constraint (e.g., "solve in 2 minutes"), this parameter (and/or
          `population_size`) needs adjustment. The chatbot should highlight this.

    tournament_size : int, default=10 (as per ga_solver_main.cu default)
        In tournament selection, this many individuals are randomly picked from the population,
        and the fittest among them is chosen as a parent.
        - Impact: Higher values increase selection pressure (favors fitter individuals more
          strongly).
        - User Guidance: A common value. Extreme values might lead to premature convergence
          or too slow selection. Typical range: 2 - 20.

    mutation_rate : float, default=0.1 (as per ga_solver_main.cu default)
        The probability that an individual genome will undergo mutation (e.g., a swap of two
        cities in the tour) in a generation.
        - Impact: Introduces genetic diversity, helping to escape local optima. Too high can
          disrupt good solutions; too low can lead to stagnation.
        - User Guidance: Chatbot should explain its role in exploration vs. exploitation.
          Typical range: 0.001 - 0.2.

    elitism_rate : float, default=0.05 (as per ga_solver_main.cu default)
        The fraction of the best-performing individuals from the current generation that are
        guaranteed to be copied to the next generation without modification.
        - Impact: Ensures that the best solutions found so far are preserved.
        - User Guidance: Generally beneficial. Typical range: 0.01 - 0.1 (1% to 10%).

    initial_genomes_file : Optional[str]
        Path to a file specifying initial population members. Each line should be a space-separated
        list of 0-indexed city integers representing a tour.
        - Genome Representation: A genome is an ordered list (permutation) of city indices,
          e.g., `[0, 3, 1, 2]` for 4 cities, meaning visit city 0, then 3, then 1, then 2, and
          finally return to 0.
        - User Guidance: Useful if the user has prior knowledge or wants to start from specific
          solutions. If not provided or invalid, random initialization occurs.

    cuda_solver_executable_path : str, default="./ga_solver_main"
        The path to the compiled CUDA GA solver executable.

    Underlying CUDA Solver Details & Constraints:
    ---------------------------------------------
    - Genome (`genome.cu`): Represents a tour as a permutation of city indices. Fitness is typically the total tour length (lower is better).
    - Cost Matrix (`cost_matrix.cu`): Computed from Euclidean distances if coordinates are given. `ga_solver_main.cu` also loads a predefined cost matrix if the TSP file format specifies `EDGE_WEIGHT_TYPE` as `EXPLICIT` and provides an `EDGE_WEIGHT_SECTION`.
    - Fitness Evaluation (`fitness_evaluator.cu`): Calculates total tour length for each genome based on the cost matrix. This is the primary fitness function. If a user describes a TSP variant with a different objective (e.g., minimizing travel time with variable speeds, or multi-objective TSP), the chatbot should state the current solver only handles distance/cost minimization as per the cost matrix.
    - Crossover (`crossover.cu`): Implements Order Crossover (OX1).
        - Constraints: This specific type of crossover is used. If a user requests a different one (e.g., PMX, CX), the chatbot should inform them it's not supported by the current backend. It operates on pairs of parent chromosomes.
    - Mutation (`mutation.cu`): Implements swap mutation.
        - Constraints: Two cities in a tour are randomly swapped. Other mutation schemes are not currently implemented.
    - Parent Selection (`parent_selection.cu`): Uses tournament selection.

    Chatbot Interaction Notes:
    --------------------------
    The chatbot should:
    1.  Verify all necessary inputs are present (city data is critical).
    2.  Explain parameter impacts on runtime and solution quality. For example, if `population_size * num_generations` is very large, warn about potential long execution.
    3.  If `tsp_data_file` is provided, ensure it's accessible. If `coordinates` are discussed, the wrapper will handle creating a temporary file.
    4.  Confirm if the problem (as described by the user) can be solved by the current fitness function (total distance/cost) and crossover/mutation operators. Highlight limitations (e.g., `predefined_cost_matrix` as a direct Python arg needs solver adaptation or a TSPLIB-like file format that includes the matrix).
    5.  If parameters are missing, suggest sensible defaults or ask the user.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the execution status and results:
        {
            "status": "success" | "error" | "timeout",
            "message": "Descriptive message, e.g., error details.",
            "best_fitness": Optional[float], // The total cost of the best tour found.
            "best_route": Optional[List[int]], // The sequence of city indices in the best tour.
            "total_generations_run": Optional[int],
            "time_taken_ms": Optional[float],
            "avg_time_per_gen_ms": Optional[float],
            "solver_log": Optional[str] // Raw output from the CUDA executable.
        }
    """
    cmd = [cuda_solver_executable_path]
    temp_tsp_file = None

    if predefined_cost_matrix:
        # Current ga_solver_main.cu does not directly support passing a matrix via Python argument.
        # It expects a file. For this to work, one would need to either:
        # 1. Modify ga_solver_main.cu to take matrix data via CLI (complex).
        # 2. Create a temporary TSPLIB file with EDGE_WEIGHT_TYPE: EXPLICIT and an EDGE_WEIGHT_SECTION.
        # This is a complex formatting task for TSPLIB and is deferred here.
        # For now, we'll return an error if this is the primary data source without a file.
        return {
            "status": "error",
            "message": "Direct input of predefined_cost_matrix is not yet fully supported by the Python wrapper to ga_solver_main. Please provide data via tsp_data_file or coordinates, or format your matrix into a TSPLIB file with an explicit edge weight section.",
            "best_fitness": None, "best_route": None, "total_generations_run": None,
            "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": None
        }

    if coordinates:
        if tsp_data_file:
            print("Warning: Both tsp_data_file and coordinates provided. Using tsp_data_file.")
        else:
            # Create a temporary TSPLIB-like file from coordinates
            try:
                num_cities = len(coordinates)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".tsp") as tmpfile:
                    tmpfile.write(f"NAME: temp_tsp_coords\n")
                    tmpfile.write(f"TYPE: TSP\n")
                    tmpfile.write(f"COMMENT: Temporary file generated from coordinates\n")
                    tmpfile.write(f"DIMENSION: {num_cities}\n")
                    tmpfile.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n") # Assumes Euclidean distance for coordinates
                    tmpfile.write(f"NODE_COORD_SECTION\n")
                    for i, (x, y) in enumerate(coordinates):
                        tmpfile.write(f"{i + 1} {x} {y}\n")
                    tmpfile.write("EOF\n")
                    temp_tsp_file = tmpfile.name
                    tsp_data_file = temp_tsp_file
                    print(f"Generated temporary TSP file from coordinates: {tsp_data_file}")
            except Exception as e:
                return {
                    "status": "error", "message": f"Failed to create temporary TSP file from coordinates: {e}",
                    "best_fitness": None, "best_route": None, "total_generations_run": None,
                    "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": None
                }

    if not tsp_data_file:
        return {
            "status": "error",
            "message": "No TSP data source provided. Please specify tsp_data_file or coordinates.",
            "best_fitness": None, "best_route": None, "total_generations_run": None,
            "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": None
        }

    cmd.extend(["--tspFileName", tsp_data_file])
    cmd.extend(["--populationSize", str(population_size)])
    cmd.extend(["--numGenerations", str(num_generations)])
    cmd.extend(["--tournamentSize", str(tournament_size)])
    cmd.extend(["--mutationRate", str(mutation_rate)])
    cmd.extend(["--elitismRate", str(elitism_rate)])

    if initial_genomes_file:
        cmd.extend(["--initialGenomesFile", initial_genomes_file])

    solver_log_lines = []
    try:
        print(f"Executing GA solver with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Stream stdout and print it, also collecting it
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                print(line) 
                solver_log_lines.append(line)
            process.stdout.close()

        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()
            process.stderr.close()

        process.wait()
        solver_log = "\n".join(solver_log_lines)

        if process.returncode != 0:
            error_message = f"CUDA Solver Error (Return Code {process.returncode}):\n{stderr_output}\nStdout Log:\n{solver_log}"
            print(error_message)
            return {
                "status": "error", "message": error_message,
                "best_fitness": None, "best_route": None, "total_generations_run": None,
                "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": solver_log
            }

        # Parse output from solver_log
        # This is highly dependent on the exact format of your ga_solver_main's stdout
        best_fitness = None
        best_route_str = None
        total_time_ms = None
        avg_time_per_gen_ms = None

        for line in reversed(solver_log_lines):
            if "Final Best Fitness =" in line and best_fitness is None:
                try:
                    best_fitness = float(line.split("=")[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse best fitness from line: {line}")
            elif "Best Route:" in line:
                # Assuming the next line after "Best Route:" contains the route itself
                # This needs to be robust based on your actual output format
                try:
                    route_line_index = solver_log_lines.index(line) + 1
                    if route_line_index < len(solver_log_lines):
                        # Assuming route is space separated numbers, ending before a potential '| Fitness:' part
                        route_part = solver_log_lines[route_line_index].split("|")[0].strip()
                        best_route_str = [int(x) for x in route_part.split()]
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse best route from log: {e}")
            elif "Total time taken:" in line and total_time_ms is None:
                try:
                    total_time_ms = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                     print(f"Warning: Could not parse total time from line: {line}")
            elif "Average time per generation:" in line and avg_time_per_gen_ms is None:
                try:
                    avg_time_per_gen_ms = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse average time per gen from line: {line}")
            
            # Break if all main components found
            if best_fitness is not None and best_route_str is not None and total_time_ms is not None and avg_time_per_gen_ms is not None:
                break
        
        if best_fitness is None:
             print("Warning: Could not parse final best fitness from solver output.")

        return {
            "status": "success",
            "message": "GA run completed.",
            "best_fitness": best_fitness,
            "best_route": best_route_str,
            "total_generations_run": num_generations, # ga_solver_main runs for the specified numGenerations
            "time_taken_ms": total_time_ms,
            "avg_time_per_gen_ms": avg_time_per_gen_ms,
            "solver_log": solver_log
        }

    except FileNotFoundError:
        return {
            "status": "error", "message": f"CUDA solver executable not found at {cuda_solver_executable_path}",
            "best_fitness": None, "best_route": None, "total_generations_run": None,
            "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": None
        }
    except Exception as e:
        return {
            "status": "error", "message": f"An unexpected error occurred: {str(e)}",
            "best_fitness": None, "best_route": None, "total_generations_run": None,
            "time_taken_ms": None, "avg_time_per_gen_ms": None, "solver_log": "\n".join(solver_log_lines) # Include any partial log
        }
    finally:
        if temp_tsp_file and os.path.exists(temp_tsp_file):
            try:
                os.remove(temp_tsp_file)
                print(f"Removed temporary TSP file: {temp_tsp_file}")
            except Exception as e:
                print(f"Error removing temporary file {temp_tsp_file}: {e}")


if __name__ == '__main__':
    # Example Usage (assuming ga_solver_main is compiled and in the same directory or path):
    
    # 1. Test with a dummy TSP file (create a dummy 'data/dummy.tsp' for this to run)
    # Create a dummy TSP file for testing
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/dummy.tsp", "w") as f:
        f.write("NAME: dummy\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: 3\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("1 10 10\n")
        f.write("2 20 20\n")
        f.write("3 30 10\n")
        f.write("EOF\n")

    print("--- Test 1: Using a TSP file ---")
    results_file = run_gpu_ga_tsp_solver(
        tsp_data_file="data/dummy.tsp",
        population_size=50, # Smaller for quick test
        num_generations=100 # Smaller for quick test
    )
    print("Results (File):", results_file)
    print("\n")

    # 2. Test with coordinates
    print("--- Test 2: Using coordinates ---")
    coords = [(10.0, 10.0), (20.0, 20.0), (30.0, 10.0), (40.0, 20.0)]
    results_coords = run_gpu_ga_tsp_solver(
        coordinates=coords,
        population_size=60,
        num_generations=120
    )
    print("Results (Coords):", results_coords)
    print("\n")

    # 3. Test with initial genomes file (create a dummy 'data/initial_routes.txt')
    with open("data/initial_routes.txt", "w") as f:
        f.write("0 1 2\n") # For dummy.tsp (3 cities)
        f.write("2 1 0\n")
    
    print("--- Test 3: Using initial genomes file ---")
    results_initial_genomes = run_gpu_ga_tsp_solver(
        tsp_data_file="data/dummy.tsp",
        population_size=50,
        num_generations=100,
        initial_genomes_file="data/initial_routes.txt"
    )
    print("Results (Initial Genomes):", results_initial_genomes)
    print("\n")

    # 4. Test with missing data (should error)
    print("--- Test 4: Missing TSP data (expect error) ---")
    results_error = run_gpu_ga_tsp_solver(population_size=10, num_generations=10)
    print("Results (Error):", results_error)
    print("\n")

    # 5. Test with non-existent executable path (should error)
    print("--- Test 5: Non-existent solver path (expect error) ---")
    results_path_error = run_gpu_ga_tsp_solver(
        tsp_data_file="data/dummy.tsp",
        cuda_solver_executable_path="./non_existent_solver"
    )
    print("Results (Path Error):", results_path_error)

    # Clean up dummy files
    # os.remove("data/dummy.tsp")
    # os.remove("data/initial_routes.txt")
    # if os.path.exists("data") and not os.listdir("data"):
    #     os.rmdir("data")
