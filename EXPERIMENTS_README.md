# GA Experiments System

Automated system for running genetic algorithm experiments on multiple TSP problems and collecting comprehensive statistics.

## Overview

The experiment system consists of:
1. **ga_runner** (C++/CUDA) - Runs GA on a single problem with parseable output
2. **run_experiments.py** (Python) - Orchestrates multiple experiments and collects results

## Features

- ✅ Automated execution on all medium and large problems
- ✅ Comprehensive statistics collection (time, fitness, convergence)
- ✅ Solution storage for medium problems
- ✅ Results exported to JSON and CSV formats
- ✅ Summary statistics and comparisons
- ✅ Configurable GA parameters

## Quick Start

### 1. Build the GA Runner

```bash
cd build_test
cmake ..
make ga_runner
```

### 2. Run Experiments

Run on all problems:
```bash
python3 run_experiments.py
```

Run only on medium problems:
```bash
python3 run_experiments.py --filter medium
```

Run only on large problems:
```bash
python3 run_experiments.py --filter large
```

### 3. View Results

Results are saved in the `experiment_results/` directory:
- `experiment_TIMESTAMP.json` - Full results including solutions
- `experiment_TIMESTAMP.csv` - Tabular results (without solutions)
- `experiment_TIMESTAMP_solutions.json` - Solutions only (for medium problems)

## Command-Line Options

```bash
python3 run_experiments.py [OPTIONS]

Options:
  --filter {medium,large,all}   Filter problems by size (default: all)
  --generations INT             Number of generations (default: 100)
  --population INT              Population size (default: 500)
  --build-dir DIR              Build directory (default: build_test)
  --data-dir DIR               Data directory (default: data)
  --no-solutions               Do not save solutions
  --output-prefix PREFIX       Output file prefix (default: experiment)
```

## Examples

### Run quick test with 50 generations on medium problems
```bash
python3 run_experiments.py --filter medium --generations 50
```

### Run comprehensive experiment with large population
```bash
python3 run_experiments.py --generations 200 --population 1000
```

### Run experiment without saving solutions (faster)
```bash
python3 run_experiments.py --no-solutions --output-prefix quick_test
```

## Collected Statistics

For each problem run, the following statistics are collected:

### Basic Info
- `problem_name` - Name of the TSP problem
- `problem_type` - "medium" or "large"
- `num_cities` - Number of cities in the problem
- `timestamp` - When the experiment was run

### GA Configuration
- `kernel_type` - "Original" or "Optimized" (based on problem size)
- `population_size` - Population size used
- `num_generations` - Number of generations run

### Performance Metrics
- `total_time` - Total execution time (ms)
- `avg_time_per_gen` - Average time per generation (ms)
- `init_time` - Initialization time (ms)
- `fitness_time` - Total fitness evaluation time (ms)
- `crossover_time` - Total crossover time (ms)
- `mutation_time` - Total mutation time (ms)

### Solution Quality
- `initial_fitness` - Best fitness at generation 0
- `final_fitness` - Best fitness at final generation
- `improvement` - Percentage improvement (negative means better for TSP)
- `converged_at_generation` - When solution converged (if applicable)

### Solution (Medium Problems Only)
- `solution` - Array of city indices representing the best tour

## Output Formats

### JSON Format
```json
{
  "problem_name": "TSPJ_78M",
  "problem_type": "medium",
  "num_cities": 404,
  "kernel_type": "Original",
  "population_size": 500,
  "num_generations": 100,
  "total_time": 10131.0,
  "avg_time_per_gen": 101.31,
  "initial_fitness": 36276.8,
  "final_fitness": 16794.4,
  "improvement": -53.70,
  "solution": [0, 145, 23, 67, ...]
}
```

### CSV Format
All fields except `solution` are exported to CSV for easy analysis in spreadsheets.

## Analyzing Results

### Using Python
```python
import json
import pandas as pd

# Load JSON results
with open('experiment_results/experiment_20241207_123456.json') as f:
    results = json.load(f)

# Load CSV results
df = pd.read_csv('experiment_results/experiment_20241207_123456.csv')

# Calculate average time by problem type
print(df.groupby('problem_type')['avg_time_per_gen'].mean())

# Find best improvement
best = df.loc[df['improvement'].idxmin()]
print(f"Best improvement: {best['problem_name']} ({best['improvement']}%)")
```

### Using Excel/LibreOffice
Simply open the CSV file and create pivot tables or charts.

## Manual Run (Single Problem)

You can also run a single problem manually:

```bash
cd build_test
./ga_runner \
    ../data/Medium_problems/Batch_04/TSPJ_78M_nodes_table_by_coordinates.csv \
    TSPJ_78M \
    100 \
    500 \
    1
```

Parameters:
1. Problem file path
2. Problem name
3. Number of generations
4. Population size
5. Save solution (1=yes, 0=no)

## Performance Comparison Example

After running experiments, you can compare performance:

```bash
# Run medium problems
python3 run_experiments.py --filter medium --output-prefix medium_test

# Run large problems
python3 run_experiments.py --filter large --output-prefix large_test

# Compare the CSV files
```

## Directory Structure

```
GA_Solver/
├── run_experiments.py          # Main experiment script
├── tests/
│   └── ga_runner.cu           # Single problem runner
├── build_test/
│   └── ga_runner              # Compiled runner
├── data/
│   ├── Medium_problems/
│   └── Large_problems/
└── experiment_results/         # Auto-created for results
    ├── experiment_*.json
    ├── experiment_*.csv
    └── experiment_*_solutions.json
```

## Tips

1. **Start Small**: Test with `--generations 50` first
2. **Filter by Size**: Use `--filter medium` for faster initial tests
3. **Save Solutions**: Only save solutions for problems you'll analyze later
4. **Custom Prefix**: Use meaningful prefixes like `--output-prefix baseline_run`
5. **Compare Runs**: Run multiple experiments with different parameters and compare

## Troubleshooting

### "No such file or directory"
Make sure you're running from the GA_Solver directory:
```bash
cd /home/harish/large_problem_tspj/GA_Solver
python3 run_experiments.py
```

### "ga_runner not found"
Build the runner first:
```bash
cd build_test && cmake .. && make ga_runner
```

### "No results collected"
Check that data files exist:
```bash
ls data/Medium_problems/Batch_*/
ls data/Large_problems/Batch_*/
```

## Advanced Usage

### Custom Analysis Script

```python
#!/usr/bin/env python3
import json
import sys

# Load results
with open(sys.argv[1]) as f:
    results = json.load(f)

# Find problems that didn't converge
no_converge = [
    r for r in results
    if 'converged_at_generation' not in r
]

print(f"{len(no_converge)} problems didn't converge:")
for r in no_converge:
    print(f"  {r['problem_name']}: final improvement {r['improvement']:.2f}%")
```

### Batch Comparison

Run the same problems with different parameters:
```bash
# Baseline
python3 run_experiments.py --generations 100 --population 500 --output-prefix baseline

# Large population
python3 run_experiments.py --generations 100 --population 1000 --output-prefix large_pop

# More generations
python3 run_experiments.py --generations 200 --population 500 --output-prefix long_run
```

Then compare the results to see which configuration works best!
