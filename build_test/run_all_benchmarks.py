#!/usr/bin/env python3
"""
Benchmark script for TSPJ GA Solver - All Problems
Runs both Medium (TSPJ_*M) and Large (TSPJ_*L) datasets
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

def get_problems(problem_type="all", limit=None):
    """Get list of problem datasets

    Args:
        problem_type: "medium", "large", or "all"
        limit: Optional limit on number of problems per type
    """
    problems = []

    # Medium problems (TSPJ_*M)
    if problem_type in ["medium", "all"]:
        for i in range(1, 101):
            tt_file = DATA_DIR / f"TSPJ_{i}M_TSPJ_TT.csv"
            jt_file = DATA_DIR / f"TSPJ_{i}M_TSPJ_JT.csv"
            if tt_file.exists() and jt_file.exists():
                problems.append((f"TSPJ_{i}M", str(tt_file), str(jt_file), "medium"))

    # Large problems (TSPJ_*L)
    if problem_type in ["large", "all"]:
        for i in range(1, 101):
            tt_file = DATA_DIR / f"TSPJ_{i}L_TSPJ_TT.csv"
            jt_file = DATA_DIR / f"TSPJ_{i}L_TSPJ_JT.csv"
            if tt_file.exists() and jt_file.exists():
                problems.append((f"TSPJ_{i}L", str(tt_file), str(jt_file), "large"))

    if limit:
        problems = problems[:limit]
    return problems

def run_solver(name, tt_file, jt_file, mode=0, local_search=0, stagnation=150, generations=2000, population_size=3000, timeout=600):
    """Run the GA solver and extract results"""
    cmd = [GA_SOLVER, "-d", name, "-o", str(mode), "--local-search", str(local_search),
           "-s", str(stagnation), "-g", str(generations), "-p", str(population_size)]

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed_time = time.time() - start_time
        output = result.stdout + result.stderr

        # Extract metrics from output
        metrics = {
            'name': name,
            'status': 'pending',
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

        # Parse number of cities
        cities_match = re.search(r'Number of Cities:\s*(\d+)', output)
        if cities_match:
            metrics['num_cities'] = int(cities_match.group(1))

        # Parse solution generation
        sol_gen_match = re.search(r'Best Fitness of Generation (\d+):', output)
        if sol_gen_match:
            metrics['solution_generation'] = int(sol_gen_match.group(1))

        # Parse time per generation
        time_gen_match = re.search(r'Time per generation[:\s]+([0-9.]+)', output)
        if time_gen_match:
            metrics['time_per_generation'] = float(time_gen_match.group(1))

        # Determine status
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

def print_usage():
    print("Usage: python3 run_all_benchmarks.py [OPTIONS]")
    print("")
    print("Options:")
    print("  --medium, -m      Run only medium problems (TSPJ_*M)")
    print("  --large, -l       Run only large problems (TSPJ_*L)")
    print("  --all, -a         Run all problems (default)")
    print("  --test, -t        Run first 5 problems of each type (test mode)")
    print("  <number>          Run first N problems")
    print("  --help, -h        Show this help message")
    print("")
    print("Configuration options:")
    print("  --pop <N>         Population size (default: 3000)")
    print("  --gen <N>         Max generations (default: 2000)")
    print("  --stag <N>        Stagnation limit (default: 150)")
    print("  --ls <N>          Local search intensity 0-3 (default: 0)")
    print("  --timeout <N>     Timeout per problem in seconds (default: 600)")
    print("")
    print("Examples:")
    print("  python3 run_all_benchmarks.py                    # Run all problems")
    print("  python3 run_all_benchmarks.py --medium           # Run only medium")
    print("  python3 run_all_benchmarks.py --large            # Run only large")
    print("  python3 run_all_benchmarks.py --test             # Test mode (5 each)")
    print("  python3 run_all_benchmarks.py --medium 10        # First 10 medium")
    print("  python3 run_all_benchmarks.py --pop 5000 --gen 3000  # Custom settings")

def main():
    # Default configuration
    problem_type = "all"
    limit = None
    population_size = 3000
    generations = 2000
    stagnation = 150
    local_search = 0
    timeout = 600

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ['--help', '-h']:
            print_usage()
            return
        elif arg in ['--medium', '-m']:
            problem_type = "medium"
        elif arg in ['--large', '-l']:
            problem_type = "large"
        elif arg in ['--all', '-a']:
            problem_type = "all"
        elif arg in ['--test', '-t']:
            limit = 5
        elif arg == '--pop' and i + 1 < len(args):
            population_size = int(args[i + 1])
            i += 1
        elif arg == '--gen' and i + 1 < len(args):
            generations = int(args[i + 1])
            i += 1
        elif arg == '--stag' and i + 1 < len(args):
            stagnation = int(args[i + 1])
            i += 1
        elif arg == '--ls' and i + 1 < len(args):
            local_search = int(args[i + 1])
            i += 1
        elif arg == '--timeout' and i + 1 < len(args):
            timeout = int(args[i + 1])
            i += 1
        elif arg.isdigit():
            limit = int(arg)

        i += 1

    # Get problems
    problems = get_problems(problem_type, limit)

    if not problems:
        print(f"No {problem_type} problems found in data directory!")
        return

    # Count by type
    medium_count = sum(1 for p in problems if p[3] == "medium")
    large_count = sum(1 for p in problems if p[3] == "large")

    print(f"\n{'='*60}")
    print("TSPJ GA SOLVER BENCHMARK")
    print(f"{'='*60}")
    print(f"Problem type: {problem_type}")
    print(f"Total problems: {len(problems)} (Medium: {medium_count}, Large: {large_count})")
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Stagnation: {stagnation}, Local Search: {local_search}")
    print(f"Timeout: {timeout}s per problem")
    print(f"{'='*60}")

    # Prepare results storage
    results = []
    all_outputs = []

    total_start = time.time()

    # Run each problem
    for idx, (name, tt_file, jt_file, ptype) in enumerate(problems, 1):
        print(f"\n[{idx}/{len(problems)}] ({ptype}) ", end="")
        metrics, output = run_solver(
            name, tt_file, jt_file,
            mode=0,
            local_search=local_search,
            stagnation=stagnation,
            generations=generations,
            population_size=population_size,
            timeout=timeout
        )
        metrics['problem_type'] = ptype
        results.append(metrics)
        all_outputs.append(f"\n{'='*80}\n{name} ({ptype})\n{'='*80}\n{output}")

    total_time = time.time() - total_start

    # Write CSV results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark_{problem_type}_{timestamp}.csv"

    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['name', 'problem_type', 'status', 'best_fitness', 'num_cities',
                      'elapsed_time', 'time_per_generation', 'solution_generation', 'return_code']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Write full output log
    log_file = f"benchmark_{problem_type}_{timestamp}.log"
    with open(log_file, 'w') as f:
        f.write('\n'.join(all_outputs))

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for ptype in ["medium", "large"]:
        type_results = [r for r in results if r.get('problem_type') == ptype]
        if not type_results:
            continue

        successful = [r for r in type_results if r['status'] == 'success' and r['best_fitness'] is not None]
        failed = [r for r in type_results if r['status'] != 'success']

        print(f"\n{ptype.upper()} PROBLEMS:")
        print(f"  Total:           {len(type_results)}")
        print(f"  Successful:      {len(successful)}")
        print(f"  Failed/Timeout:  {len(failed)}")

        if successful:
            fitness_values = [r['best_fitness'] for r in successful]
            times = [r['elapsed_time'] for r in successful]

            print(f"  Fitness - Min: {min(fitness_values):.2f}, Max: {max(fitness_values):.2f}, Avg: {sum(fitness_values)/len(fitness_values):.2f}")
            print(f"  Time    - Min: {min(times):.2f}s, Max: {max(times):.2f}s, Avg: {sum(times)/len(times):.2f}s")

    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"\nResults saved to: {csv_file}")
    print(f"Full log saved to: {log_file}")

    # Print results table
    print(f"\n{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Problem':<15} {'Type':<8} {'Status':<10} {'Fitness':<12} {'Cities':<8} {'Time(s)':<10}")
    print("-" * 70)
    for r in results:
        fitness_str = f"{r['best_fitness']:.2f}" if r['best_fitness'] else "N/A"
        cities_str = str(r['num_cities']) if r['num_cities'] else "N/A"
        print(f"{r['name']:<15} {r.get('problem_type', 'N/A'):<8} {r['status']:<10} {fitness_str:<12} {cities_str:<8} {r['elapsed_time']:.2f}")

if __name__ == "__main__":
    main()
