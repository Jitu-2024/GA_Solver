from google.adk.tools import FunctionTool
from typing import List, Tuple, Optional

# --- Tool for Human-in-the-Loop Approval ---
def request_human_approval(question: str) -> str:
    """
    Signals the need to pause and ask the user for input.

    Args:
        question: The question to present to the human for approval.
    """
    # This function's body is not executed by the runner; it's a signal.
    return "This response is a placeholder and should not be seen by the user."

human_approval_tool = FunctionTool(func=request_human_approval)


# --- Proxy Tool for the GPU GA TSP Solver ---
# This is a simplified version of the solver function, exposed to the LLM.
# It has a clean signature with only user-facing parameters.
def solve_tsp_with_ga(
    coordinates: List[List[float]],
    population_size: Optional[int] = None,
    num_generations: Optional[int] = None,
    tournament_size: Optional[int] = None,
    mutation_rate: Optional[float] = None,
    elitism_rate: Optional[float] = None,
) -> str:
    """
    Calls the GPU-accelerated Genetic Algorithm to solve a Traveling Salesperson Problem.

    Use this tool when you have the coordinates for the cities and are ready to run the solver.
    If optional parameters are not provided, sensible defaults will be used by the solver.

    Args:
        coordinates: A list of [x, y] lists defining city coordinates.
        population_size: (Optional) The number of candidate solutions in each generation.
        num_generations: (Optional) The total number of iterations the GA will perform.
        tournament_size: (Optional) The number of individuals selected for a parent selection tournament.
        mutation_rate: (Optional) The probability that a genome will undergo mutation.
        elitism_rate: (Optional) The fraction of the best individuals to carry over to the next generation.
    """
    # This body is a placeholder. The actual execution is handled in main.py,
    # which calls the real solver function with the necessary internal arguments.
    return "The TSP solver has been initiated."

# This is the tool the agent will see and use.
run_gpu_ga_tsp_solver_tool = FunctionTool(func=solve_tsp_with_ga) 