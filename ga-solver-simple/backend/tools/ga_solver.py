import subprocess
import tempfile
import os
from typing import List, Tuple, Optional, Dict, Any, Callable
# These imports are needed for the FunctionTool definition.
from google.adk.tools import FunctionTool
# With PYTHONPATH=/app, we can now use an absolute import from the app root.
from celery_worker import run_solver_task
# --- Shared State for Tasks ---
# This dictionary will be imported by other modules to track task status.
tasks: Dict[str, Dict[str, Any]] = {}

# --- Robust Path to Executable ---
# Assumes this script is in ga-solver-simple/backend/tools
# and the executable is in a 'build' directory in the project root.
try:
    # This constructs an absolute path to the project's root directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
except NameError:
    # Fallback for environments where __file__ is not defined
    PROJECT_ROOT = os.getcwd()
    BUILD_DIR = os.path.join(PROJECT_ROOT, "build")

def find_default_solver_path() -> Optional[str]:
    """
    Searches for the GA solver executable. It prioritizes an environment
    variable and then falls back to searching common build directories.
    Returns the path if found, otherwise None.
    """
    # 1. Prioritize Environment Variable for explicit path setting
    env_path = os.environ.get("GA_SOLVER_EXECUTABLE_PATH")
    if env_path:
        if os.path.exists(env_path):
            print(f"Found solver executable from environment variable 'GA_SOLVER_EXECUTABLE_PATH': {env_path}")
            return env_path
        else:
            print(f"Warning: Environment variable 'GA_SOLVER_EXECUTABLE_PATH' is set to '{env_path}', but the file does not exist.")
            # Continue to search default paths in case the env var is stale.

    # 2. Fallback to searching the build directory
    if not os.path.isdir(BUILD_DIR):
        # This is not a warning anymore, because if the env var failed, this is the last resort.
        print(f"Error: Build directory not found at '{BUILD_DIR}'. Please build the project or set the GA_SOLVER_EXECUTABLE_PATH environment variable.")
        return None # Build directory doesn't exist.

    # Common locations for executables in CMake/VS projects on Windows.
    # Order matters: prefer Release over Debug.
    search_paths = [
        os.path.join(BUILD_DIR, "Release", "ga_solver_executable.exe"),
        os.path.join(BUILD_DIR, "Debug", "ga_solver_executable.exe"),
        os.path.join(BUILD_DIR, "ga_solver_executable.exe"),
        os.path.join(BUILD_DIR, "ga_solver_executable"), # For non-windows builds
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found solver executable at: {path}")
            return path
            
    return None # Not found in any default location

# Get the default path once at startup
DEFAULT_SOLVER_PATH = find_default_solver_path()


def run_gpu_ga_solver(
    progress_callback: Optional[Callable[[str, Optional[float], Any], None]] = None,
    # The rest of the parameters
    tsp_data_file: Optional[str] = None,
    coordinates: Optional[List[Tuple[float, float]]] = None,
    predefined_cost_matrix: Optional[List[List[float]]] = None,
    population_size: int = 1000,
    num_generations: int = 50000,
    tournament_size: int = 10,
    mutation_rate: float = 0.1,
    elitism_rate: float = 0.05,
    initial_genomes_file: Optional[str] = None,
    cuda_solver_executable_path: Optional[str] = None
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

    cuda_solver_executable_path : Optional[str]
        The path to the compiled CUDA GA solver executable. If None, it defaults to
        searching within the project's 'build' directory.

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
    4.  Confirm if the problem (as described by the user) can be solved by the current fitness function (total distance/cost) and crossover/mutation/selection operators. Highlight limitations (e.g., `predefined_cost_matrix` as a direct Python arg needs solver adaptation or a TSPLIB-like file format that includes the matrix).
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
        
        This function is designed to be run in a background task.
        It executes the GA solver and continuously updates a shared dictionary
        (tasks) with the progress and final result.
    """
    # Use the user-provided path if available, otherwise use the found default path.
    solver_path = cuda_solver_executable_path if cuda_solver_executable_path is not None else DEFAULT_SOLVER_PATH

    def _update_status(status: str, progress: Optional[float] = None, result: Any = None):
        """Internal helper to call the progress callback if it exists."""
        if progress_callback:
            progress_callback(status, progress, result)

    # Verify the executable path was found or provided
    if not solver_path or not os.path.exists(solver_path):
        error_message = (f"CUDA solver executable not found at the specified path: {cuda_solver_executable_path}. "
                         if cuda_solver_executable_path else
                         f"Could not find CUDA solver executable in the default search locations within '{BUILD_DIR}'. "
                         "Please ensure the project is built successfully first.")
        _update_status("failed", result=error_message)
        return {"status": "error", "message": error_message}

    
    # if predefined_cost_matrix:
    #     # Current ga_solver_main.cu does not directly support passing a matrix via Python argument.
    #     # It expects a file. For this to work, one would need to either:
    #     # 1. Modify ga_solver_main.cu to take matrix data via CLI (complex).
    #     # 2. Create a temporary TSPLIB file with EDGE_WEIGHT_TYPE: EXPLICIT and an EDGE_WEIGHT_SECTION.
    #     # This is a complex formatting task for TSPLIB and is deferred here.
    #     # For now, we'll return an error if this is the primary data source without a file.
    #     _update_status("failed", result="Direct input of predefined_cost_matrix is not yet fully supported.")
    #     return {"status": "error", "message": "Direct input of predefined_cost_matrix is not yet fully supported."}

    # Create a temporary file for coordinates if needed
    temp_tsp_file = None
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
                    print(f"Generated temporary TSP file from coordinates: {temp_tsp_file}")
            except Exception as e:
                _update_status("failed", result=f"Failed to create temporary TSP file from coordinates: {e}")
                return {"status": "error", "message": f"Failed to create temporary TSP file from coordinates: {e}"}

    if not tsp_data_file:
        _update_status("failed", result="No TSP data source provided. Please specify tsp_data_file or coordinates.")
        return {"status": "error", "message": "No TSP data source provided. Please specify tsp_data_file or coordinates."}

    cmd = [
        solver_path,
        "--tspFileName", tsp_data_file,
        "--populationSize", str(population_size),
        "--numGenerations", str(num_generations),
        "--tournamentSize", str(tournament_size),
        "--mutationRate", str(mutation_rate),
        "--elitismRate", str(elitism_rate),
    ]
    if initial_genomes_file:
        cmd.extend(["--initialGenomesFile", initial_genomes_file])
    
    solver_log_lines = []
    try:
        _update_status("running", progress=0.0, result="Starting GA solver...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        # Stream stdout and update progress
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                print(f"Solver log: {line}")
                solver_log_lines.append(line)
                # Provide real-time progress update by parsing solver output
                if "Generation" in line and "/" in line and "Best Fitness" in line:
                    try:
                        parts = line.split()
                        gen_idx = parts.index("Generation")
                        current_gen = int(parts[gen_idx + 1])
                        # Assuming the format is 'Generation 5 / 50000 | ...'
                        total_gens = int(parts[gen_idx + 3])
                        progress_percent = (current_gen / total_gens) * 100
                        _update_status("running", progress=progress_percent, result=line)
                    except (ValueError, IndexError):
                        # If parsing fails, just update with the log line
                        _update_status("running", result=line)
                else:
                    # For other lines, just update the result/log
                    _update_status("running", result=line)

            process.stdout.close()

        process.wait()
        solver_log = "\n".join(solver_log_lines)

        if process.returncode != 0:
            error_message = f"CUDA Solver Error (Return Code {process.returncode}):\n{solver_log}"
            _update_status("failed", result=error_message)
            return {"status": "error", "message": error_message, "solver_log": solver_log}

        # Parse final output from solver_log
        best_fitness, best_route_str, total_time_ms, avg_time_per_gen_ms = None, None, None, None
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
        
        final_result = {
            "best_fitness": best_fitness,
            "best_route": best_route_str,
            "total_generations_run": num_generations, # ga_solver_main runs for the specified numGenerations
            "time_taken_ms": total_time_ms,
            "avg_time_per_gen_ms": avg_time_per_gen_ms,
            "solver_log": solver_log
        }
        _update_status("completed", progress=100.0, result=final_result)
        return {"status": "success", "message": "GA run completed successfully.", **final_result}

    except Exception as e:
        _update_status("failed", result=f"An unexpected error occurred: {str(e)}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
    finally:
        if temp_tsp_file and os.path.exists(temp_tsp_file):
            try:
                os.remove(temp_tsp_file)
                print(f"Removed temporary TSP file: {temp_tsp_file}")
            except Exception as e:
                print(f"Error removing temporary file {temp_tsp_file}: {e}")

# --- ADK Tool Definition ---



def create_ga_solver_tool() -> FunctionTool:
    """
    Creates a FunctionTool that correctly exposes the GA solver's parameters
    and documentation to the LLM, while dispatching the execution to a
    Celery worker.
    """

    def ga_gpu_solver_tool(
        tsp_data_file: Optional[str] = None,
        coordinates: Optional[List[Tuple[float, float]]] = None,
        predefined_cost_matrix: Optional[List[List[float]]] = None,
        population_size: int = 1000,
        num_generations: int = 50000,
        tournament_size: int = 10,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.05,
        initial_genomes_file: Optional[str] = None,
        cuda_solver_executable_path: Optional[str] = None
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
            coordinates will be used.
            Example: `[(10.0, 20.5), (30.0, 40.0), ...]`

        population_size : int, default=1000
            The number of candidate solutions (genomes) maintained in each generation.

        num_generations : int, default=50000
            The total number of iterations the GA will perform.
        
        ... and so on. The LLM will see the full docstring from the original function.
        """
        # --- 1. Parameter Validation ---
        # Before dispatching to Celery, validate the parameters to prevent bad tasks.
        if population_size <= 0:
            return {"error": "Validation failed: population_size must be a positive integer."}
        if num_generations <= 0:
            return {"error": "Validation failed: num_generations must be a positive integer."}
        if not (0.0 <= mutation_rate <= 1.0):
            return {"error": "Validation failed: mutation_rate must be between 0.0 and 1.0."}
        if not (0.0 <= elitism_rate <= 1.0):
            return {"error": "Validation failed: elitism_rate must be between 0.0 and 1.0."}
        if tsp_data_file is None and coordinates is None:
            return {"error": "Validation failed: No input data provided. Please specify either tsp_data_file or coordinates."}

        # --- 2. Dispatch to Celery ---
        # Create a clean dictionary of arguments to pass to the Celery task,
        # ensuring no stray local variables (like 'error' from validation) are included.
        task_args = {
            "tsp_data_file": tsp_data_file,
            "coordinates": coordinates,
            "predefined_cost_matrix": predefined_cost_matrix,
            "population_size": population_size,
            "num_generations": num_generations,
            "tournament_size": tournament_size,
            "mutation_rate": mutation_rate,
            "elitism_rate": elitism_rate,
            "initial_genomes_file": initial_genomes_file,
            "cuda_solver_executable_path": cuda_solver_executable_path,
        }
        
        task = run_solver_task.delay(**task_args)
        
        print(f"Dispatched solver task to Celery worker. Task ID: {task.id}")
        return {"task_id": task.id}

    # Explicitly copy the comprehensive docstring from the *original*
    # run_gpu_ga_solver function to our new ADK tool function.
    # This is what the LLM will see.
    ga_gpu_solver_tool.__doc__ = run_gpu_ga_solver.__doc__
    
    # We remove parameters from the tool's schema that the LLM shouldn't see.
    return FunctionTool(
        func=ga_gpu_solver_tool
        # The 'omit_parameters' argument is not supported.
        # The selective exposure of parameters is already handled by the fact
        # that `ga_gpu_solver_tool` has a different signature than the
        # full `run_gpu_ga_solver` function.
    )

# if __name__ == '__main__':
#     # Example Usage (assuming ga_solver_main is compiled and in the same directory or path):
    
#     # 1. Test with a dummy TSP file (create a dummy 'data/dummy.tsp' for this to run)
#     # Create a dummy TSP file for testing
#     # This assumes the script is run from ga-solver-simple/backend/
#     # So, 'data' directory needs to be relative to that or an absolute path.
#     # For simplicity, let's assume 'data' is one level up from 'backend', i.e., in 'ga-solver-simple/data'
    
#     example_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") #  ga-solver-simple/backend/ -> ga-solver-simple/
#     data_dir = os.path.join(example_base_dir, "data")
#     dummy_tsp_path = os.path.join(data_dir, "dummy.tsp")
#     initial_routes_path = os.path.join(data_dir, "initial_routes.txt")

#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     with open(dummy_tsp_path, "w") as f:
#         f.write("NAME: dummy\n")
#         f.write("TYPE: TSP\n")
#         f.write("DIMENSION: 3\n")
#         f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
#         f.write("NODE_COORD_SECTION\n")
#         f.write("1 10 10\n")
#         f.write("2 20 20\n")
#         f.write("3 30 10\n")
#         f.write("EOF\n")

#     print(f"--- Test 1: Using a TSP file ({dummy_tsp_path}) ---")
#     results_file = run_gpu_ga_tsp_solver(
#         tsp_data_file=dummy_tsp_path,
#         population_size=50, # Smaller for quick test
#         num_generations=100 # Smaller for quick test
#     )
#     print("Results (File):", results_file)
#     print("\n")

#     # 2. Test with coordinates
#     print("--- Test 2: Using coordinates ---")
#     coords = [(10.0, 10.0), (20.0, 20.0), (30.0, 10.0), (40.0, 20.0)]
#     results_coords = run_gpu_ga_tsp_solver(
#         coordinates=coords,
#         population_size=60,
#         num_generations=120
#     )
#     print("Results (Coords):", results_coords)
#     print("\n")

#     # 3. Test with initial genomes file (create a dummy 'data/initial_routes.txt')
#     with open(initial_routes_path, "w") as f:
#         f.write("0 1 2\n") # For dummy.tsp (3 cities)
#         f.write("2 1 0\n")
    
#     print(f"--- Test 3: Using initial genomes file ({initial_routes_path}) ---")
#     results_initial_genomes = run_gpu_ga_tsp_solver(
#         tsp_data_file=dummy_tsp_path,
#         population_size=50,
#         num_generations=100,
#         initial_genomes_file=initial_routes_path
#     )
#     print("Results (Initial Genomes):", results_initial_genomes)
#     print("\n")

#     # 4. Test with missing data (should error)
#     print("--- Test 4: Missing TSP data (expect error) ---")
#     results_error = run_gpu_ga_tsp_solver(population_size=10, num_generations=10)
#     print("Results (Error):", results_error)
#     print("\n")

#     # 5. Test with non-existent executable path (should error)
#     print("--- Test 5: Non-existent solver path (expect error) ---")
#     results_path_error = run_gpu_ga_tsp_solver(
#         tsp_data_file=dummy_tsp_path,
#         cuda_solver_executable_path="./non_existent_solver"
#     )
#     print("Results (Path Error):", results_path_error)

#     # Clean up dummy files
#     # os.remove(dummy_tsp_path)
#     # os.remove(initial_routes_path)
#     # if os.path.exists(data_dir) and not os.listdir(data_dir):
#     #     os.rmdir(data_dir)

