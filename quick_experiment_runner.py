#!/usr/bin/env python3
"""
Quick experiment runner for TSP-J GA solver
This script runs multiple experiments and generates comprehensive analysis
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from pathlib import Path
import json
from datetime import datetime
import argparse

def run_ga_experiment(dataset, seed, hyperparams, ga_executable="./build/ga_solver", timeout=1800):
    """Run a single GA experiment with given parameters"""
    
    cmd = [
        ga_executable,
        "--datasets", dataset,
        "--population-size", str(hyperparams['population_size']),
        "--generations", str(hyperparams['generations']),
        "--mutation-rate", str(hyperparams['mutation_rate']),
        "--tournament-size", str(hyperparams['tournament_size']),
        "--mode", str(hyperparams['mode']),
        "--stagnation-limit", str(hyperparams['stagnation_limit']),
        "--diversity-percent", str(hyperparams['diversity_percent']),
        "--seed", str(seed),
        "--logs-folder", "logs"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Success in {total_time:.2f}s")
            return True, total_time, result.stdout
        else:
            print(f"✗ Failed: {result.stderr}")
            return False, total_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after {timeout}s")
        return False, timeout, "Timeout"
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, 0, str(e)

def load_convergence_data(dataset, seed, logs_folder="logs"):
    """Load convergence data from CSV file"""
    filename = f"{logs_folder}/convergence_{dataset}_seed{seed}.csv"
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Convergence file not found: {filename}")
        return None

def load_detailed_results(logs_folder="logs"):
    """Load detailed results from CSV file"""
    filename = f"{logs_folder}/detailed_results.csv"
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Detailed results file not found: {filename}")
        return None

def plot_convergence_comparison(datasets, seeds, logs_folder="logs", save_path="plots"):
    """Generate convergence comparison plots"""
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets[:6]):  # Limit to 6 datasets for layout
        ax = axes[idx]
        
        for seed_idx, seed in enumerate(seeds):
            conv_data = load_convergence_data(dataset, seed, logs_folder)
            if conv_data is not None:
                ax.plot(conv_data['Generation'], conv_data['Best_Fitness'], 
                       color=colors[seed_idx], alpha=0.7, linewidth=1.5,
                       label=f'Seed {seed}')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'Convergence - {dataset}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(datasets), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/convergence_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_analysis(logs_folder="logs", save_path="plots"):
    """Generate statistical analysis plots"""
    
    df = load_detailed_results(logs_folder)
    if df is None:
        print("No detailed results found for statistical analysis")
        return
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Fitness distribution by dataset
    sns.boxplot(data=df, x='Dataset', y='Best_Fitness', ax=axes[0,0])
    axes[0,0].set_title('Best Fitness Distribution by Dataset')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Solution generation distribution
    sns.boxplot(data=df, x='Dataset', y='Solution_Generation', ax=axes[0,1])
    axes[0,1].set_title('Solution Generation Distribution by Dataset')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Runtime distribution
    sns.boxplot(data=df, x='Dataset', y='Total_Time_s', ax=axes[1,0])
    axes[1,0].set_title('Runtime Distribution by Dataset')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Fitness vs Runtime scatter
    sns.scatterplot(data=df, x='Total_Time_s', y='Best_Fitness', 
                   hue='Dataset', ax=axes[1,1])
    axes[1,1].set_title('Best Fitness vs Runtime')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/statistical_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(logs_folder="logs", save_path="reports"):
    """Generate a comprehensive summary report"""
    
    df = load_detailed_results(logs_folder)
    if df is None:
        print("No detailed results found for summary report")
        return
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Generate summary statistics
    summary_stats = df.groupby('Dataset').agg({
        'Best_Fitness': ['mean', 'std', 'min', 'max'],
        'Solution_Generation': ['mean', 'std'],
        'Total_Time_s': ['mean', 'std'],
        'Seed': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    
    # Save to CSV
    summary_stats.to_csv(f"{save_path}/summary_statistics.csv")
    
    # Generate text report
    report_path = f"{save_path}/experiment_report.txt"
    with open(report_path, 'w') as f:
        f.write("TSP-J GA SOLVER EXPERIMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY STATISTICS BY DATASET:\n")
        f.write("-" * 30 + "\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")
        
        f.write("BEST SOLUTIONS BY DATASET:\n")
        f.write("-" * 30 + "\n")
        best_solutions = df.loc[df.groupby('Dataset')['Best_Fitness'].idxmin()]
        for _, row in best_solutions.iterrows():
            f.write(f"{row['Dataset']}: {row['Best_Fitness']:.4f} "
                   f"(Generation {row['Solution_Generation']}, "
                   f"Time: {row['Total_Time_s']:.2f}s, "
                   f"Seed: {row['Seed']})\n")
        
        f.write("\n\nPARAMETER ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        param_columns = ['Population_Size', 'Mutation_Rate', 'Tournament_Size', 'Diversity_Percent']
        for param in param_columns:
            if param in df.columns:
                f.write(f"{param}: {df[param].unique()}\n")
    
    print(f"Summary report saved to: {report_path}")
    print(f"Summary statistics saved to: {save_path}/summary_statistics.csv")

def main():
    parser = argparse.ArgumentParser(description='Run TSP-J GA experiments')
    parser.add_argument('--datasets', nargs='+', 
                       default=['gr17', 'gr21', 'gr24', 'fri26', 'bays29'],
                       help='Datasets to test')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456, 789, 1024],
                       help='Random seeds for experiments')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per dataset')
    parser.add_argument('--ga-executable', default='./build/ga_solver',
                       help='Path to GA executable')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Timeout per run in seconds')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Define hyperparameters
    if args.quick:
        hyperparams = {
            'population_size': 100,
            'generations': 500,
            'mutation_rate': 0.3,
            'tournament_size': 5,
            'mode': 0,
            'stagnation_limit': 100,
            'diversity_percent': 20.0
        }
        datasets = ['gr17', 'gr21']
        seeds = [42, 123]
    else:
        hyperparams = {
            'population_size': 500,
            'generations': 5000,
            'mutation_rate': 0.3,
            'tournament_size': 5,
            'mode': 0,
            'stagnation_limit': 1000,
            'diversity_percent': 20.0
        }
        datasets = args.datasets
        seeds = args.seeds[:args.runs]
    
    print("Starting TSP-J GA Experiments")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Hyperparameters: {hyperparams}")
    print("=" * 50)
    
    # Run experiments
    total_experiments = len(datasets) * len(seeds)
    current_experiment = 0
    successful_runs = 0
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        print("-" * 30)
        
        for seed in seeds:
            current_experiment += 1
            print(f"Run {current_experiment}/{total_experiments} - Dataset: {dataset}, Seed: {seed}")
            
            success, runtime, output = run_ga_experiment(
                dataset, seed, hyperparams, args.ga_executable, args.timeout
            )
            
            if success:
                successful_runs += 1
            
            # Small delay between runs
            time.sleep(1)
    
    print(f"\nExperiments completed!")
    print(f"Successful runs: {successful_runs}/{total_experiments}")
    print(f"Success rate: {successful_runs/total_experiments*100:.1f}%")
    
    # Generate analysis and plots
    print("\nGenerating analysis and plots...")
    
    try:
        plot_convergence_comparison(datasets, seeds)
        plot_statistical_analysis()
        generate_summary_report()
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error generating analysis: {e}")

if __name__ == "__main__":
    main()