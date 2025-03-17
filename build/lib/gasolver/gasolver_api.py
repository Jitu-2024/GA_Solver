from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional
from ._core import GASolver  # This imports the compiled C++ module

@dataclass
class GAParameters:
    population_size: int = 1000
    num_generations: int = 50000
    tournament_size: int = 10
    mutation_rate: float = 0.1
    elitism_rate: float = 0.05

class GASolverAPI:
    """High-level Python API for the GA Solver."""
    
    def __init__(self, parameters: Optional[GAParameters] = None):
        self.parameters = parameters or GAParameters()
        self._solver = GASolver(
            self.parameters.population_size,
            self.parameters.num_generations,
            self.parameters.tournament_size,
            self.parameters.mutation_rate,
            self.parameters.elitism_rate
        )

    def solve(self, 
             data: Union[np.ndarray, pd.DataFrame, List], 
             is_matrix: bool = False) -> Dict:
        """
        Solve a TSP instance.
        
        Args:
            data: Input data (coordinates or distance matrix)
            is_matrix: Whether input is a distance matrix
            
        Returns:
            Dictionary containing solution and metrics
        """
        array_data = self._validate_input(data)
        return self._solver.solve(array_data, is_matrix)

    
    def _validate_input(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if 'x' in data.columns and 'y' in data.columns:
                return data[['x', 'y']].values
            return data.values
        return np.array(data, dtype=np.float32) 

