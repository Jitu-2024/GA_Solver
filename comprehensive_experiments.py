#!/usr/bin/env python3
"""
Simplified TSP-J GA Experiment Script
====================================

This script runs experiments with:
- 4 seeds for each configuration
- 4 identical hyperparameter sets for all datasets
- All 10 datasets
"""

import sys
import os
import argparse
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
import subprocess

class SimplifiedExperimentSuite:
    """
    Simplified experiment suite for TSP-J GA solver
    """
    
    def __init__(self, ga_executable="./ga_solver_enhanced"):
        self.ga_executable = ga_executable
        
        # All 10 datasets
        self.all_datasets = [
            "bays29", "berlin52", "eil101", "eil51", "eil76",
            "fri26", "gr17", "gr21", "gr24", "gr48"
        ]
        
        # Fixed 4 seeds for all experiments
        self.seeds = [42, 123, 456, 789]
        
        # 4 hyperparameter configurations (same for all datasets)
        self.hyperparameter_sets = self._define_hyperparameter_sets()
        
    def _define_hyperparameter_sets(self):
        """Define 4 hyperparameter configurations with large population sizes (1000-2000)"""
        
        configs = {
            # Configuration 1: Medium Large Population
            "config1_medium_large": {
                'population_size': 1000,
                'generations': 3000,
                'mutation_rate': 0.3,
                'tournament_size': 5,
                'stagnation_limit': 1000,
                'diversity_percent': 20.0,
                'mode': 0
            },
            
            # Configuration 2: Large Population
            "config2_large": {
                'population_size': 1500,
                'generations': 3000,
                'mutation_rate': 0.3,
                'tournament_size': 7,
                'stagnation_limit': 1200,
                'diversity_percent': 18.0,
                'mode': 0
            },
            
            # Configuration 3: Very Large Population + High Mutation
            "config3_very_large_high_mut": {
                'population_size': 2000,
                'generations': 2500,
                'mutation_rate': 0.4,
                'tournament_size': 8,
                'stagnation_limit': 1000,
                'diversity_percent': 25.0,
                'mode': 0
            },
            
            # Configuration 4: Large Population + Long Run
            "config4_large_long_run": {
                'population_size': 1200,
                'generations': 4000,
                'mutation_rate': 0.25,
                'tournament_size': 6,
                'stagnation_limit': 1500,
                'diversity_percent': 22.0,
                'mode': 0
            }
        }
        
        return configs
    
    def run_single_experiment(self, dataset, seed, hyperparams, timeout=3600):
        """Run a single GA experiment"""
        
        cmd = [
            self.ga_executable,
            "--datasets", dataset,
            "--population-size", str(hyperparams['population_size']),
            "--generations", str(hyperparams['generations']),
            "--mutation-rate", str(hyperparams['mutation_rate']),
            "--tournament-size", str(hyperparams['tournament_size']),
            "--mode", str(hyperparams['mode']),
            "--stagnation-limit", str(hyperparams['stagnation_limit']),
            "--diversity-percent", str(hyperparams['diversity_percent']),
            "--num-runs", "1",  # Single run per call
            "--seed", str(seed),
            "--logs-folder", "logs"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                return True, total_time, result.stdout
            else:
                return False, total_time, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, timeout, "Timeout"
        except Exception as e:
            return False, 0, str(e)
    
    def run_experiments(self, datasets=None, quick_test=False):
        """Run the complete experiment suite"""
        
        if datasets is None:
            datasets = self.all_datasets
            
        if quick_test:
            # Quick test: only 2 datasets, 2 configs, 2 seeds
            datasets = ["gr17", "gr21"]
            configs_to_run = ["config1_medium_large", "config2_large"]
            seeds_to_use = [42, 123]
        else:
            # Full test: all datasets, all configs, all seeds
            configs_to_run = list(self.hyperparameter_sets.keys())
            seeds_to_use = self.seeds
        
        total_experiments = len(datasets) * len(configs_to_run) * len(seeds_to_use)
        current_experiment = 0
        successful_runs = 0
        
        print(f"🧬 TSP-J GA Experiment Suite")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"📊 Datasets: {len(datasets)} ({', '.join(datasets)})")
        print(f"⚙️  Configurations: {len(configs_to_run)}")
        print(f"🎲 Seeds: {len(seeds_to_use)} ({seeds_to_use})")
        print(f"🎯 Total Experiments: {total_experiments}")
        print("=" * 60)
        
        # Show hyperparameter configurations
        print("📋 Hyperparameter Configurations:")
        for config_name in configs_to_run:
            config = self.hyperparameter_sets[config_name]
            print(f"  {config_name}:")
            print(f"    Population: {config['population_size']}")
            print(f"    Generations: {config['generations']}")
            print(f"    Mutation Rate: {config['mutation_rate']}")
            print(f"    Tournament Size: {config['tournament_size']}")
            print(f"    Stagnation Limit: {config['stagnation_limit']}")
            print(f"    Diversity %: {config['diversity_percent']}")
        print("=" * 60)
        
        start_time = time.time()
        
        for dataset in datasets:
            print(f"\n📊 Processing Dataset: {dataset}")
            print("-" * 40)
            
            for config_name in configs_to_run:
                config = self.hyperparameter_sets[config_name]
                print(f"  ⚙️  Configuration: {config_name}")
                
                for seed in seeds_to_use:
                    current_experiment += 1
                    
                    print(f"    🎲 Seed {seed} [{current_experiment}/{total_experiments}]", end=" ")
                    
                    success, runtime, output = self.run_single_experiment(
                        dataset, seed, config, timeout=3600
                    )
                    
                    if success:
                        successful_runs += 1
                        print(f"✅ ({runtime:.1f}s)")
                    else:
                        print(f"❌ Failed ({runtime:.1f}s)")
                        if "Timeout" not in output:
                            print(f"      Error: {output[:100]}...")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"✅ Successful runs: {successful_runs}/{total_experiments}")
        print(f"📈 Success rate: {successful_runs/total_experiments*100:.1f}%")
        print(f"⏱️  Total time: {elapsed_time/3600:.2f} hours")
        print(f"📁 Results saved in: logs/")
        print("=" * 60)
        
        return successful_runs, total_experiments
    
    def generate_experiment_summary(self):
        """Generate a summary of experiment parameters"""
        
        summary = {
            "experiment_info": {
                "total_datasets": len(self.all_datasets),
                "datasets": self.all_datasets,
                "seeds": self.seeds,
                "total_configs": len(self.hyperparameter_sets),
                "total_runs_per_dataset": len(self.hyperparameter_sets) * len(self.seeds),
                "total_experiments": len(self.all_datasets) * len(self.hyperparameter_sets) * len(self.seeds)
            },
            "hyperparameter_configurations": self.hyperparameter_sets
        }
        
        # Save to JSON file
        with open("logs/experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary


def load_and_analyze_results():
    """Load and analyze the experimental results"""
    
    try:
        # Try to load individual results first
        individual_file = "logs/individual_results.csv"
        aggregated_file = "logs/aggregated_results.csv"
        
        if os.path.exists(individual_file):
            df = pd.read_csv(individual_file)
            print(f"📊 Loaded {len(df)} individual results")
            
            # Basic statistics
            print("\n📈 Basic Statistics:")
            print(f"  Datasets tested: {df['Dataset'].nunique()}")
            print(f"  Unique seeds: {sorted(df['Seed'].unique())}")
            print(f"  Configuration variations: {df[['Population_Size', 'Mutation_Rate', 'Tournament_Size']].drop_duplicates().shape[0]}")
            
            # Best results per dataset
            print("\n🏆 Best Results per Dataset:")
            best_per_dataset = df.loc[df.groupby('Dataset')['Fitness_Score'].idxmin()]
            for _, row in best_per_dataset.iterrows():
                print(f"  {row['Dataset']}: {row['Fitness_Score']:.2f} "
                      f"(Config: pop={row['Population_Size']}, mut={row['Mutation_Rate']}, "
                      f"seed={row['Seed']})")
            
            # Average performance by configuration
            print("\n⚙️  Average Performance by Configuration:")
            config_perf = df.groupby(['Population_Size', 'Mutation_Rate', 'Tournament_Size'])['Fitness_Score'].agg(['mean', 'std', 'count']).round(2)
            print(config_perf)
            
        elif os.path.exists(aggregated_file):
            df = pd.read_csv(aggregated_file)
            print(f"📊 Loaded {len(df)} aggregated results")
            
            print("\n🏆 Best Results per Dataset:")
            for _, row in df.iterrows():
                print(f"  {row['Dataset']}: {row['Best_Fitness']:.2f} "
                      f"(Mean: {row['Mean_Fitness']:.2f} ± {row['Std_Fitness']:.2f})")
        else:
            print("❌ No results files found. Run experiments first.")
            
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Simplified TSP-J GA Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs experiments with:
- 4 seeds: [42, 123, 456, 789]
- 4 hyperparameter configurations (same for all datasets)
- All 10 datasets

Total experiments: 10 datasets × 4 configs × 4 seeds = 160 runs

Examples:
  python3 simplified_experiments.py --run              # Run all experiments
  python3 simplified_experiments.py --quick            # Quick test (8 runs)
  python3 simplified_experiments.py --analyze          # Analyze existing results
        """
    )
    
    parser.add_argument('--run', action='store_true',
                       help='Run the full experiment suite')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (2 datasets, 2 configs, 2 seeds)')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing results')
    
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to run (default: all)')
    
    parser.add_argument('--ga-executable',
                       default='./ga_solver_enhanced',
                       help='Path to GA executable')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    if args.analyze:
        load_and_analyze_results()
        return 0
    
    if not (args.run or args.quick):
        parser.print_help()
        return 0
    
    # Check if GA executable exists
    if not os.path.exists(args.ga_executable):
        print(f"❌ GA executable not found: {args.ga_executable}")
        print("Please build the GA solver first or specify correct path with --ga-executable")
        return 1
    
    # Create experiment suite
    suite = SimplifiedExperimentSuite(ga_executable=args.ga_executable)
    
    # Generate experiment summary
    summary = suite.generate_experiment_summary()
    print(f"📋 Experiment summary saved to: logs/experiment_summary.json")
    
    # Run experiments
    if args.quick:
        print("🚀 Running Quick Test...")
        successful, total = suite.run_experiments(quick_test=True)
    else:
        datasets_to_run = args.datasets if args.datasets else None
        print("🚀 Running Full Experiment Suite...")
        successful, total = suite.run_experiments(datasets=datasets_to_run)
    
    if successful > 0:
        print(f"\n📊 Analyzing results...")
        load_and_analyze_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())