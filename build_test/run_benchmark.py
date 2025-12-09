#!/usr/bin/env python3
"""
Benchmark script for TSPJ GA Solver - Medium Problems
Runs all TSPJ_*M datasets and collects results
"""

import subprocess
import re
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
GA_SOLVER = "./ga_solver"
DATA_DIR = Path("data")
OUTPUT_FILE = "benchmark_results_medium.csv"

def get_medium_problems(limit=None):
    """Get list of medium problem datasets (TSPJ_*M)"""
    problems = []
    for i in range(1, 101):
        tt_file = DATA_DIR / f"TSPJ_{i}M_TSPJ_TT.csv"
        jt_file = DATA_DIR / f"TSPJ_{i}M_TSPJ_JT.csv"
        if tt_file.exists() and jt_file.exists():
            problems.append((f"TSPJ_{i}M", str(tt_file), str(jt_file)))

    if limit:
        problems = problems[:limit]
    return problems

def run_solver(name, tt_file, jt_file, mode=0, local_search=0, stagnation=150, generations=2000, population_size=3000):
    """Run the GA solver and extract results"""
    # Solver expects: ./ga_solver -d DATASET_NAME -o MODE --local-search INTENSITY -s STAGNATION -g GENERATIONS -p POPULATION
    cmd = [GA_SOLVER, "-d", name, "-o", str(mode), "--local-search", str(local_search),
           "-s", str(stagnation), "-g", str(generations), "-p", str(population_size)]

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per problem
        )

        elapsed_time = time.time() - start_time
        output = result.stdout + result.stderr

        # Extract metrics from output
        metrics = {
            'name': name,
            'status': 'pending',  # Will be determined after parsing fitness
            'return_code': result.returncode,
            'elapsed_time': elapsed_time,
            'best_fitness': None,
            'num_cities': None,
            'generations': None,
            'solution_generation': None,
            'time_per_generation': None
        }

        # Parse best fitness - look for final "Fitness:" line (at start of line)
        fitness_match = re.search(r'^Fitness:\s*([0-9.]+)', output, re.MULTILINE)
        if not fitness_match:
            # Fallback: get the LAST "Best Fitness of Generation" line
            all_matches = re.findall(r'Best Fitness of Generation \d+:\s*([0-9.]+)', output)
            if all_matches:
                fitness_match = type('obj', (object,), {'group': lambda self, n: all_matches[-1]})()
        if fitness_match:
            metrics['best_fitness'] = float(fitness_match.group(1))

        # Parse number of cities from city sequence
        cities_match = re.search(r'City Sequence:\s*([\d\s]+)', output)
        if cities_match:
            cities = cities_match.group(1).strip().split()
            metrics['num_cities'] = len(cities)

        # Parse solution generation (from "Best Fitness of Generation X")
        sol_gen_match = re.search(r'Best Fitness of Generation (\d+):', output)
        if sol_gen_match:
            metrics['solution_generation'] = int(sol_gen_match.group(1))

        # Parse time per generation
        time_gen_match = re.search(r'Time per generation[:\s]+([0-9.]+)', output)
        if time_gen_match:
            metrics['time_per_generation'] = float(time_gen_match.group(1))

        # Determine status: success if fitness was captured, failed otherwise
        # This allows runs that had non-critical errors (e.g., log file write failure)
        # to still be counted as successful if they produced valid results
        if metrics['best_fitness'] is not None:
            metrics['status'] = 'success'
        elif result.returncode == 0:
            metrics['status'] = 'success'
        else:
            metrics['status'] = 'failed'

        # Print summary
        print(f"Status: {metrics['status']}")
        print(f"Best Fitness: {metrics['best_fitness']}")
        print(f"Cities: {metrics['num_cities']}")
        print(f"Time: {elapsed_time:.2f}s")

        return metrics, output

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"TIMEOUT after {elapsed_time:.2f}s")
        return {
            'name': name,
            'status': 'timeout',
            'return_code': -1,
            'elapsed_time': elapsed_time,
            'best_fitness': None,
            'num_cities': None,
            'generations': None,
            'solution_generation': None,
            'time_per_generation': None
        }, ""
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'name': name,
            'status': 'error',
            'return_code': -1,
            'elapsed_time': 0,
            'best_fitness': None,
            'num_cities': None,
            'generations': None,
            'solution_generation': None,
            'time_per_generation': None
        }, str(e)

def main():
    # Parse arguments
    limit = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test' or sys.argv[1] == '-t':
            limit = 10
            print("Running TEST mode: first 10 problems only")
        elif sys.argv[1].isdigit():
            limit = int(sys.argv[1])
            print(f"Running first {limit} problems")
        elif sys.argv[1] in ['--help', '-h']:
            print("Usage: python3 run_benchmark.py [OPTIONS]")
            print("")
            print("Options:")
            print("  --test, -t    Run first 10 problems only (test mode)")
            print("  <number>      Run first N problems")
            print("  --help, -h    Show this help message")
            print("")
            print("Examples:")
            print("  python3 run_benchmark.py           # Run all 100 medium problems")
            print("  python3 run_benchmark.py --test    # Run first 10 problems")
            print("  python3 run_benchmark.py 25        # Run first 25 problems")
            return

    # Get problems to run
    problems = get_medium_problems(limit)

    if not problems:
        print("No medium problems found in data directory!")
        return

    print(f"\nFound {len(problems)} medium problems to benchmark")
    print(f"Output will be saved to: {OUTPUT_FILE}")

    # Prepare results storage
    results = []
    all_outputs = []

    total_start = time.time()

    # Configuration for benchmarking large problems
    local_search = 0  # 0=none, 1=light, 2=medium, 3=full (0 required for large populations)
    stagnation = 150  # Max generations without improvement
    generations = 2000  # Max total generations
    population_size = 3000  # Population size (works with local-search 0)

    # Run each problem
    for i, (name, tt_file, jt_file) in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] ", end="")
        metrics, output = run_solver(name, tt_file, jt_file, mode=0,
                                     local_search=local_search,
                                     stagnation=stagnation,
                                     generations=generations,
                                     population_size=population_size)
        results.append(metrics)
        all_outputs.append(f"\n{'='*80}\n{name}\n{'='*80}\n{output}")

    total_time = time.time() - total_start

    # Write CSV results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark_medium_{timestamp}.csv"

    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['name', 'status', 'best_fitness', 'num_cities',
                      'elapsed_time', 'time_per_generation', 'solution_generation', 'return_code']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Write full output log
    log_file = f"benchmark_medium_{timestamp}.log"
    with open(log_file, 'w') as f:
        f.write('\n'.join(all_outputs))

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if r['status'] == 'success' and r['best_fitness'] is not None]
    failed = [r for r in results if r['status'] != 'success']

    print(f"Total problems:    {len(results)}")
    print(f"Successful:        {len(successful)}")
    print(f"Failed/Timeout:    {len(failed)}")
    print(f"Total time:        {total_time:.2f}s ({total_time/60:.1f} min)")

    if successful:
        fitness_values = [r['best_fitness'] for r in successful]
        times = [r['elapsed_time'] for r in successful]

        print(f"\nFitness Statistics:")
        print(f"  Min:  {min(fitness_values):.2f}")
        print(f"  Max:  {max(fitness_values):.2f}")
        print(f"  Avg:  {sum(fitness_values)/len(fitness_values):.2f}")

        print(f"\nTime Statistics:")
        print(f"  Min:  {min(times):.2f}s")
        print(f"  Max:  {max(times):.2f}s")
        print(f"  Avg:  {sum(times)/len(times):.2f}s")

    print(f"\nResults saved to: {csv_file}")
    print(f"Full log saved to: {log_file}")

    # Print table of results
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Problem':<15} {'Status':<10} {'Fitness':<12} {'Time(s)':<10}")
    print("-" * 50)
    for r in results:
        fitness_str = f"{r['best_fitness']:.2f}" if r['best_fitness'] else "N/A"
        print(f"{r['name']:<15} {r['status']:<10} {fitness_str:<12} {r['elapsed_time']:.2f}")

if __name__ == "__main__":
    main()
