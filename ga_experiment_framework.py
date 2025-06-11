import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess
import csv
from pathlib import Path

class GAExperimentLogger:
    """Comprehensive logging system for GA experiments"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"ga_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path(f"experiments/{self.experiment_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result storage
        self.experiment_results = []
        self.convergence_data = {}
        
    def log_run(self, dataset: str, run_id: int, seed: int, hyperparams: Dict, 
                fitness_score: float, solution_generation: int, total_time: float,
                convergence_history: List[float], best_solution: List[int] = None):
        """Log a single GA run"""
        
        result = {
            'dataset': dataset,
            'run_id': run_id,
            'seed': seed,
            'fitness_score': fitness_score,
            'solution_generation': solution_generation,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'best_solution': best_solution,
            **hyperparams  # Unpack hyperparameters
        }
        
        self.experiment_results.append(result)
        
        # Store convergence data
        conv_key = f"{dataset}_run{run_id}_seed{seed}"
        self.convergence_data[conv_key] = {
            'generations': list(range(len(convergence_history))),
            'fitness': convergence_history,
            'dataset': dataset,
            'run_id': run_id,
            'seed': seed
        }
        
    def save_results(self):
        """Save all experimental results"""
        
        # Save main results to CSV
        df = pd.DataFrame(self.experiment_results)
        csv_path = self.results_dir / "experiment_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save convergence data to JSON
        conv_path = self.results_dir / "convergence_data.json"
        with open(conv_path, 'w') as f:
            json.dump(self.convergence_data, f, indent=2)
        
        # Save summary statistics
        self._generate_summary_stats(df)
        
        print(f"Results saved to: {self.results_dir}")
        
    def _generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics"""
        
        summary_stats = {}
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            summary_stats[dataset] = {
                'mean_fitness': float(dataset_df['fitness_score'].mean()),
                'std_fitness': float(dataset_df['fitness_score'].std()),
                'best_fitness': float(dataset_df['fitness_score'].min()),
                'worst_fitness': float(dataset_df['fitness_score'].max()),
                'mean_solution_generation': float(dataset_df['solution_generation'].mean()),
                'std_solution_generation': float(dataset_df['solution_generation'].std()),
                'mean_time': float(dataset_df['total_time'].mean()),
                'std_time': float(dataset_df['total_time'].std()),
                'success_rate': float((dataset_df['fitness_score'] == dataset_df['fitness_score'].min()).mean())
            }
        
        # Save summary statistics
        summary_path = self.results_dir / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def plot_convergence(self, datasets: List[str] = None, save_plots: bool = True):
        """Generate convergence plots"""
        
        if datasets is None:
            datasets = list(set(data['dataset'] for data in self.convergence_data.values()))
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        
        for dataset in datasets:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: All runs for this dataset
            dataset_data = [data for data in self.convergence_data.values() 
                          if data['dataset'] == dataset]
            
            for data in dataset_data:
                ax1.plot(data['generations'], data['fitness'], 
                        alpha=0.6, linewidth=1, 
                        label=f"Run {data['run_id']} (Seed {data['seed']})")
            
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Fitness')
            ax1.set_title(f'Convergence History - {dataset}')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Mean convergence with confidence intervals
            if len(dataset_data) > 1:
                max_gen = max(len(data['fitness']) for data in dataset_data)
                fitness_matrix = np.full((len(dataset_data), max_gen), np.nan)
                
                for i, data in enumerate(dataset_data):
                    fitness_matrix[i, :len(data['fitness'])] = data['fitness']
                
                mean_fitness = np.nanmean(fitness_matrix, axis=0)
                std_fitness = np.nanstd(fitness_matrix, axis=0)
                
                generations = range(max_gen)
                ax2.plot(generations, mean_fitness, 'b-', linewidth=2, label='Mean')
                ax2.fill_between(generations, 
                               mean_fitness - std_fitness,
                               mean_fitness + std_fitness,
                               alpha=0.3, label='±1 Std Dev')
                
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Best Fitness')
                ax2.set_title(f'Mean Convergence with Std Dev - {dataset}')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.results_dir / f"convergence_{dataset}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_comparison(self, save_plots: bool = True):
        """Generate comparison plots across datasets"""
        
        df = pd.DataFrame(self.experiment_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitness comparison
        sns.boxplot(data=df, x='dataset', y='fitness_score', ax=axes[0,0])
        axes[0,0].set_title('Fitness Score Distribution by Dataset')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Solution generation comparison
        sns.boxplot(data=df, x='dataset', y='solution_generation', ax=axes[0,1])
        axes[0,1].set_title('Solution Generation Distribution by Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Runtime comparison
        sns.boxplot(data=df, x='dataset', y='total_time', ax=axes[1,0])
        axes[1,0].set_title('Runtime Distribution by Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Correlation between fitness and time
        sns.scatterplot(data=df, x='total_time', y='fitness_score', 
                       hue='dataset', ax=axes[1,1])
        axes[1,1].set_title('Fitness vs Runtime')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / "dataset_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class GAExperimentRunner:
    """Main experiment runner for GA on TSP-J datasets"""
    
    def __init__(self, ga_executable_path: str = "./ga_solver"):
        self.ga_executable_path = ga_executable_path
        self.datasets = [
            "bays29", "berlin52", "eil101", "eil51", "eil76",
            "fri26", "gr17", "gr21", "gr24", "gr48"
        ]
        
        # Default hyperparameters
        self.default_hyperparams = {
            'population_size': 500,
            'generations': 10000,
            'mutation_rate': 0.3,
            'tournament_size': 5,
            'mode': 0,  # TSP-J without pickup
            'stagnation_limit': 1500,
            'diversity_percent': 20.0
        }
    
    def run_single_experiment(self, dataset: str, seed: int, hyperparams: Dict) -> Dict:
        """Run a single GA experiment"""
        
        # Prepare command
        cmd = [
            self.ga_executable_path,
            "--datasets", dataset,
            "--population-size", str(hyperparams['population_size']),
            "--generations", str(hyperparams['generations']),
            "--mutation-rate", str(hyperparams['mutation_rate']),
            "--tournament-size", str(hyperparams['tournament_size']),
            "--mode", str(hyperparams['mode']),
            "--stagnation-limit", str(hyperparams['stagnation_limit']),
            "--diversity-percent", str(hyperparams['diversity_percent'])
        ]
        
        # Set random seed in environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        
        start_time = time.time()
        
        try:
            # Run the GA
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=3600, env=env)  # 1 hour timeout
            
            if result.returncode != 0:
                print(f"Error running GA for {dataset}: {result.stderr}")
                return None
            
            total_time = time.time() - start_time
            
            # Parse the output to extract results
            output_lines = result.stdout.strip().split('\n')
            
            # Extract fitness score and solution generation from output
            fitness_score = None
            solution_generation = None
            convergence_history = []
            best_solution = None
            
            for line in output_lines:
                if "Best Fitness of Generation" in line:
                    # Extract fitness value
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            gen_fitness = float(parts[1].strip())
                            convergence_history.append(gen_fitness)
                        except ValueError:
                            continue
                
                elif "BEST SOLUTION" in line:
                    # Next few lines contain the solution
                    continue
                elif "Best solution found:" in line:
                    continue
                elif "City Sequence:" in line:
                    # Extract city sequence
                    try:
                        city_seq = [int(x) for x in line.split(":")[1].strip().split()]
                        best_solution = city_seq
                    except:
                        pass
                elif "Fitness:" in line:
                    # Extract final fitness
                    try:
                        fitness_score = float(line.split(":")[1].strip())
                    except:
                        pass
            
            # Determine solution generation (when best fitness was found)
            if convergence_history:
                best_fitness = min(convergence_history)
                solution_generation = convergence_history.index(best_fitness)
                if fitness_score is None:
                    fitness_score = best_fitness
            else:
                solution_generation = 0
                if fitness_score is None:
                    fitness_score = float('inf')
            
            return {
                'fitness_score': fitness_score,
                'solution_generation': solution_generation,
                'total_time': total_time,
                'convergence_history': convergence_history,
                'best_solution': best_solution,
                'success': True
            }
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for {dataset} with seed {seed}")
            return None
        except Exception as e:
            print(f"Exception running {dataset} with seed {seed}: {e}")
            return None
    
    def run_experiments(self, 
                       datasets: List[str] = None,
                       seeds: List[int] = None,
                       num_runs: int = 5,
                       hyperparam_configs: List[Dict] = None,
                       experiment_name: str = None):
        """Run complete experimental suite"""
        
        if datasets is None:
            datasets = self.datasets
        
        if seeds is None:
            seeds = [42, 123, 456, 789, 1024]  # Default seeds
        
        if hyperparam_configs is None:
            hyperparam_configs = [self.default_hyperparams]
        
        # Ensure we have the right number of seeds
        if len(seeds) < num_runs:
            seeds = list(range(42, 42 + num_runs))
        
        # Initialize logger
        logger = GAExperimentLogger(experiment_name)
        
        total_experiments = len(datasets) * len(hyperparam_configs) * num_runs
        current_experiment = 0
        
        print(f"Starting {total_experiments} experiments...")
        print(f"Datasets: {datasets}")
        print(f"Hyperparameter configurations: {len(hyperparam_configs)}")
        print(f"Runs per configuration: {num_runs}")
        
        for hyperparam_idx, hyperparams in enumerate(hyperparam_configs):
            print(f"\nHyperparameter Config {hyperparam_idx + 1}:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
            
            for dataset in datasets:
                print(f"\n  Processing dataset: {dataset}")
                
                for run_id in range(num_runs):
                    current_experiment += 1
                    seed = seeds[run_id]
                    
                    print(f"    Run {run_id + 1}/{num_runs} (Seed: {seed}) "
                          f"[{current_experiment}/{total_experiments}]")
                    
                    # Run single experiment
                    result = self.run_single_experiment(dataset, seed, hyperparams)
                    
                    if result and result['success']:
                        # Log the result
                        logger.log_run(
                            dataset=dataset,
                            run_id=run_id,
                            seed=seed,
                            hyperparams=hyperparams,
                            fitness_score=result['fitness_score'],
                            solution_generation=result['solution_generation'],
                            total_time=result['total_time'],
                            convergence_history=result['convergence_history'],
                            best_solution=result['best_solution']
                        )
                        
                        print(f"      ✓ Fitness: {result['fitness_score']:.2f}, "
                              f"Generation: {result['solution_generation']}, "
                              f"Time: {result['total_time']:.2f}s")
                    else:
                        print(f"      ✗ Failed")
        
        # Save results and generate plots
        print("\nSaving results and generating plots...")
        logger.save_results()
        logger.plot_convergence(datasets)
        logger.plot_comparison()
        
        return logger


# Example usage and configuration
def main():
    """Main experimental function"""
    
    # Define different hyperparameter configurations to test
    hyperparam_configs = [
        # Configuration 1: Default
        {
            'population_size': 500,
            'generations': 5000,
            'mutation_rate': 0.3,
            'tournament_size': 5,
            'mode': 0,
            'stagnation_limit': 1000,
            'diversity_percent': 20.0
        },
        # Configuration 2: Higher mutation
        {
            'population_size': 500,
            'generations': 5000,
            'mutation_rate': 0.5,
            'tournament_size': 5,
            'mode': 0,
            'stagnation_limit': 1000,
            'diversity_percent': 20.0
        },
        # Configuration 3: Larger population
        {
            'population_size': 800,
            'generations': 5000,
            'mutation_rate': 0.3,
            'tournament_size': 7,
            'mode': 0,
            'stagnation_limit': 1000,
            'diversity_percent': 15.0
        }
    ]
    
    # Define datasets to test (subset for faster testing)
    test_datasets = ["gr17", "gr21", "gr24", "fri26", "bays29"]
    
    # Create experiment runner
    runner = GAExperimentRunner(ga_executable_path="./ga_solver")
    
    # Run experiments
    logger = runner.run_experiments(
        datasets=test_datasets,
        seeds=[42, 123, 456, 789, 1024],
        num_runs=5,
        hyperparam_configs=hyperparam_configs,
        experiment_name="tspj_parameter_study"
    )
    
    print("Experiment completed!")


if __name__ == "__main__":
    main()