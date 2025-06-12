#!/usr/bin/env python3
"""
Fixed Hyperparameters TSP-J GA Experiment Script
================================================

This script runs experiments using the EXACT hyperparameters defined in the
configuration dictionary without any dataset-specific scaling.

MODIFICATION: Removed dataset-specific parameter scaling to use identical
hyperparameters across all datasets and seeds.
"""

import sys
import os
import argparse
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add the current directory to path to import the framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ga_experiment_framework import GAExperimentRunner, GAExperimentLogger
except ImportError:
    print("Error: ga_experiment_framework.py not found!")
    print("Please ensure ga_experiment_framework.py is in the same directory.")
    sys.exit(1)

class FixedHyperparametersExperimentSuite:
    """
    Experiment suite that uses EXACT hyperparameters without dataset scaling
    """
    
    def __init__(self, ga_executable="./build/ga_solver"):
        self.ga_executable = ga_executable
        
        # All 10 datasets
        self.all_datasets = [
            "bays29", "berlin52", "eil101", "eil51", "eil76",
            "fri26", "gr17", "gr21", "gr24", "gr48"
        ]
        
        # Multiple random seeds for statistical validity
        self.seeds = [42, 123, 456, 789, 1024, 2048, 4096]
        
        # Fixed hyperparameter configurations (NO SCALING APPLIED)
        self.hyperparameter_configs = self._define_hyperparameter_configs()
        
    def _define_hyperparameter_configs(self):
        """Define the 6 hyperparameter configurations - EXACT VALUES"""
        
        configs = {
            # Configuration 1: Balanced (default parameters)
            "balanced": {
                'population_size': 500,
                'generations': 3000,
                'mutation_rate': 0.3,
                'tournament_size': 5,
                'mode': 0,
                'stagnation_limit': 800,
                'diversity_percent': 20.0,
                'config_name': 'balanced'
            },
            
            # Configuration 2: Large Population
            "large_pop": {
                'population_size': 800,
                'generations': 3000,
                'mutation_rate': 0.3,
                'tournament_size': 5,
                'mode': 0,
                'stagnation_limit': 1000,
                'diversity_percent': 18.0,
                'config_name': 'large_pop'
            },
            
            # Configuration 3: Long Run (more generations)
            "long_run": {
                'population_size': 600,
                'generations': 5000,
                'mutation_rate': 0.25,
                'tournament_size': 6,
                'mode': 0,
                'stagnation_limit': 1200,
                'diversity_percent': 22.0,
                'config_name': 'long_run'
            },
            
            # Configuration 4: Aggressive (high mutation)
            "aggressive": {
                'population_size': 500,
                'generations': 3000,
                'mutation_rate': 0.5,
                'tournament_size': 7,
                'mode': 0,
                'stagnation_limit': 800,
                'diversity_percent': 25.0,
                'config_name': 'aggressive'
            },
            
            # Configuration 5: Conservative (low mutation)
            "conservative": {
                'population_size': 500,
                'generations': 3000,
                'mutation_rate': 0.2,
                'tournament_size': 3,
                'mode': 0,
                'stagnation_limit': 800,
                'diversity_percent': 15.0,
                'config_name': 'conservative'
            },
            
            # Configuration 6: High Diversity
            "high_diversity": {
                'population_size': 500,
                'generations': 3000,
                'mutation_rate': 0.4,
                'tournament_size': 4,
                'mode': 0,
                'stagnation_limit': 600,
                'diversity_percent': 35.0,
                'config_name': 'high_diversity'
            }
        }
        
        return configs
    
    def run_all_configs_all_datasets(self, selected_datasets=None, selected_seeds=None):
        """
        Run ALL 6 configurations on ALL datasets with ALL seeds
        No parameter scaling - uses exact hyperparameters
        """
        print("🎯 Running All Configurations on All Datasets (Fixed Hyperparameters)")
        print("=" * 70)
        
        # Use provided selections or defaults
        datasets = selected_datasets if selected_datasets else self.all_datasets
        seeds = selected_seeds if selected_seeds else self.seeds
        config_names = list(self.hyperparameter_configs.keys())
        
        return self._run_experiment_suite(
            datasets=datasets,
            config_names=config_names,
            seeds=seeds,
            experiment_name="fixed_hyperparams_all_configs"
        )
    
    def run_single_config_all_datasets(self, config_name, selected_datasets=None, selected_seeds=None):
        """
        Run a SINGLE configuration on ALL datasets with ALL seeds
        """
        if config_name not in self.hyperparameter_configs:
            raise ValueError(f"Configuration '{config_name}' not found. Available: {list(self.hyperparameter_configs.keys())}")
        
        print(f"🎯 Running '{config_name}' Configuration on All Datasets")
        print("=" * 70)
        
        # Use provided selections or defaults
        datasets = selected_datasets if selected_datasets else self.all_datasets
        seeds = selected_seeds if selected_seeds else self.seeds
        
        return self._run_experiment_suite(
            datasets=datasets,
            config_names=[config_name],
            seeds=seeds,
            experiment_name=f"fixed_hyperparams_{config_name}"
        )
    
    def run_quick_test(self):
        """Run a quick test with subset of datasets and configurations"""
        print("🚀 Running Quick Test (Fixed Hyperparameters)")
        print("=" * 50)
        
        # Use small subset for quick testing
        test_datasets = ["gr17", "gr21", "gr24"]
        test_config_names = ["balanced", "aggressive"]
        test_seeds = [42, 123, 456]
        
        return self._run_experiment_suite(
            datasets=test_datasets,
            config_names=test_config_names,
            seeds=test_seeds,
            experiment_name="quick_test_fixed_hyperparams"
        )
    
    def run_custom_experiment(self, datasets, config_names, seeds):
        """Run custom experiment with specified parameters"""
        print("🎨 Running Custom Experiment (Fixed Hyperparameters)")
        print("=" * 50)
        
        # Validate inputs
        invalid_configs = [c for c in config_names if c not in self.hyperparameter_configs]
        if invalid_configs:
            raise ValueError(f"Invalid configurations: {invalid_configs}. Available: {list(self.hyperparameter_configs.keys())}")
        
        return self._run_experiment_suite(
            datasets=datasets,
            config_names=config_names,
            seeds=seeds,
            experiment_name="custom_fixed_hyperparams"
        )
    
    def _run_experiment_suite(self, datasets, config_names, seeds, experiment_name):
        """Internal method to run experiments with FIXED hyperparameters"""
        
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_experiment_name = f"{experiment_name}_{timestamp}"
        
        # Calculate total runs
        total_runs = len(datasets) * len(config_names) * len(seeds)
        
        print(f"📋 Experiment Details:")
        print(f"   Name: {full_experiment_name}")
        print(f"   Datasets: {len(datasets)} ({', '.join(datasets)})")
        print(f"   Configurations: {len(config_names)} ({', '.join(config_names)})")
        print(f"   Seeds per config: {len(seeds)}")
        print(f"   Total runs: {total_runs}")
        print(f"   🔒 Using FIXED hyperparameters (no dataset scaling)")
        print()
        
        # Display configuration details
        print("📊 Configuration Details:")
        for config_name in config_names:
            config = self.hyperparameter_configs[config_name]
            print(f"   {config_name}: pop={config['population_size']}, "
                  f"gen={config['generations']}, mut={config['mutation_rate']}, "
                  f"tour={config['tournament_size']}, div={config['diversity_percent']}%")
        print()
        
        # Create experiment runner
        runner = GAExperimentRunner(ga_executable_path=self.ga_executable)
        
        try:
            logger = self._run_experiments_fixed_hyperparams(
                runner, datasets, config_names, seeds, full_experiment_name
            )
            
            elapsed_time = time.time() - start_time
            print(f"\n✅ Experiment Suite Completed!")
            print(f"   Total time: {elapsed_time/3600:.2f} hours")
            print(f"   Results saved in: experiments/{full_experiment_name}/")
            
            return logger
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_experiments_fixed_hyperparams(self, runner, datasets, config_names, seeds, experiment_name):
        """Run experiments with FIXED hyperparameters (no scaling)"""
        
        # Initialize logger
        logger = GAExperimentLogger(experiment_name)
        
        total_experiments = len(datasets) * len(config_names) * len(seeds)
        current_experiment = 0
        successful_runs = 0
        
        print(f"Starting {total_experiments} experiments with FIXED hyperparameters...")
        print("=" * 70)
        
        for config_name in config_names:
            print(f"\n🔧 Configuration: {config_name}")
            
            # Use EXACT configuration - NO SCALING
            config = self.hyperparameter_configs[config_name].copy()
            
            print(f"     Fixed params: pop={config['population_size']}, "
                  f"gen={config['generations']}, "
                  f"mut={config['mutation_rate']}, "
                  f"tour={config['tournament_size']}, "
                  f"div={config['diversity_percent']}%")
            
            for dataset in datasets:
                print(f"\n  📊 Dataset: {dataset}")
                
                for run_id, seed in enumerate(seeds):
                    current_experiment += 1
                    
                    print(f"    🎯 Run {run_id + 1}/{len(seeds)} (Seed: {seed}) "
                          f"[{current_experiment}/{total_experiments}]", end=" ")
                    
                    # Run single experiment with FIXED hyperparameters
                    result = runner.run_single_experiment(dataset, seed, config)
                    
                    if result and result['success']:
                        # Log the result
                        logger.log_run(
                            dataset=dataset,
                            run_id=run_id,
                            seed=seed,
                            hyperparams=config,
                            fitness_score=result['fitness_score'],
                            solution_generation=result['solution_generation'],
                            total_time=result['total_time'],
                            convergence_history=result['convergence_history'],
                            best_solution=result['best_solution']
                        )
                        
                        successful_runs += 1
                        print(f"✅ Fitness: {result['fitness_score']:.2f}, "
                              f"Gen: {result['solution_generation']}, "
                              f"Time: {result['total_time']:.1f}s")
                    else:
                        print("❌ Failed")
        
        print(f"\n📊 Experiment Summary:")
        print(f"   Successful runs: {successful_runs}/{total_experiments}")
        print(f"   Success rate: {successful_runs/total_experiments*100:.1f}%")
        
        # Save results and generate plots
        logger.save_results()
        
        try:
            logger.plot_convergence(datasets)
            logger.plot_comparison()
            print("📈 Plots generated successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")
        
        return logger


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Fixed Hyperparameters TSP-J GA Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Modes:
  all-configs    - All 6 configs on all datasets with all seeds (420 runs)
  single-config  - Single config on all datasets with all seeds (70 runs)
  quick          - Quick test: 3 datasets, 2 configs, 3 seeds (18 runs)
  custom         - Custom selection of datasets, configs, and seeds

Examples:
  # Run all configurations on all datasets
  python3 fixed_hyperparams_experiments.py --mode all-configs
  
  # Run only 'balanced' configuration on all datasets  
  python3 fixed_hyperparams_experiments.py --mode single-config --config balanced
  
  # Quick test
  python3 fixed_hyperparams_experiments.py --mode quick
  
  # Custom: specific datasets and configs
  python3 fixed_hyperparams_experiments.py --mode custom \\
    --datasets gr17,gr21,bays29 --configs balanced,aggressive --seeds 42,123,456
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['all-configs', 'single-config', 'quick', 'custom'],
                       default='quick',
                       help='Experiment mode to run')
    
    parser.add_argument('--config',
                       choices=['balanced', 'large_pop', 'long_run', 'aggressive', 'conservative', 'high_diversity'],
                       help='Configuration to use (required for single-config mode)')
    
    parser.add_argument('--datasets',
                       help='Comma-separated list of datasets (for custom mode)')
    
    parser.add_argument('--configs',
                       help='Comma-separated list of configurations (for custom mode)')
    
    parser.add_argument('--seeds',
                       help='Comma-separated list of seeds (for custom mode)')
    
    parser.add_argument('--ga-executable',
                       default='./build/ga_solver',
                       help='Path to GA executable')
    
    args = parser.parse_args()
    
    # Check if GA executable exists
    if not os.path.exists(args.ga_executable):
        print(f"❌ GA executable not found: {args.ga_executable}")
        print("Please build the GA solver first or specify correct path with --ga-executable")
        return 1
    
    # Create experiment suite
    suite = FixedHyperparametersExperimentSuite(ga_executable=args.ga_executable)
    
    # Run selected experiment mode
    print(f"🧬 TSP-J GA Fixed Hyperparameters Experiment Suite")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {args.mode}")
    print(f"⚡ GA Executable: {args.ga_executable}")
    print("=" * 60)
    
    logger = None
    
    if args.mode == 'all-configs':
        logger = suite.run_all_configs_all_datasets()
        
    elif args.mode == 'single-config':
        if not args.config:
            print("❌ --config is required for single-config mode")
            return 1
        logger = suite.run_single_config_all_datasets(args.config)
        
    elif args.mode == 'quick':
        logger = suite.run_quick_test()
        
    elif args.mode == 'custom':
        # Parse custom parameters
        datasets = args.datasets.split(',') if args.datasets else suite.all_datasets
        configs = args.configs.split(',') if args.configs else ['balanced']
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else suite.seeds
        
        print(f"Custom experiment: {len(datasets)} datasets, {len(configs)} configs, {len(seeds)} seeds")
        logger = suite.run_custom_experiment(datasets, configs, seeds)
    
    if logger:
        print(f"\n🎉 All experiments completed successfully!")
        print(f"📊 Check the experiments/ directory for detailed results and analysis.")
    else:
        print(f"\n❌ Experiments failed or were cancelled.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())