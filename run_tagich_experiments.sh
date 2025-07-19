#!/bin/bash
# run_taguchi_experiments.sh - Run Taguchi DOE experiments for GA solver

# Check if we're in the right directory
if [ ! -f "build/ga_solver" ]; then
    echo "Error: ga_solver not found in build/"
    echo "Please run this script from the GA_Solver root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Parse command line arguments
CLEAN_START=false
if [ "$1" == "--clean" ]; then
    CLEAN_START=true
fi

# Create results directory structure
if [ "$CLEAN_START" = true ]; then
    echo "Cleaning previous results..."
    rm -rf taguchi_results
fi

mkdir -p taguchi_results
mkdir -p taguchi_results/logs

# Add timestamp to results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting Taguchi experiments at $TIMESTAMP"

# Datasets to test
DATASETS="bays29,berlin52,eil101,eil51,eil76,fri26,gr17,gr21,gr24,gr48"

# L18 Orthogonal Array Parameters
# Format: "pop_size mutation_rate tournament_size diversity_% stagnation_limit cost_aware"
EXPERIMENTS=(
    "1000 0.2 3 15 1000 0"
    "1000 0.2 5 20 1500 1"
    "1000 0.2 7 30 2000 0"
    "1000 0.3 3 20 2000 1"
    "1000 0.3 5 30 1000 0"
    "1000 0.3 7 15 1500 1"
    "1000 0.4 3 30 1500 0"
    "1000 0.4 5 15 2000 1"
    "1000 0.4 7 20 1000 0"
    "2000 0.2 3 30 2000 1"
    "2000 0.2 5 15 1000 0"
    "2000 0.2 7 20 1500 1"
    "2000 0.3 3 15 1500 0"
    "2000 0.3 5 20 2000 1"
    "2000 0.3 7 30 1000 0"
    "2000 0.4 3 20 1000 1"
    "2000 0.4 5 30 1500 0"
    "2000 0.4 7 15 2000 1"
    "3000 0.2 3 20 1500 0"
    "3000 0.2 5 30 2000 1"
    "3000 0.2 7 15 1000 0"
    "3000 0.3 3 30 1000 1"
    "3000 0.3 5 15 1500 0"
    "3000 0.3 7 20 2000 1"
    "3000 0.4 3 15 2000 0"
    "3000 0.4 5 20 1000 1"
    "3000 0.4 7 30 1500 0"
)

# Number of runs per experiment configuration
RUNS_PER_CONFIG=3

# Results summary file
SUMMARY_FILE="taguchi_results/experiment_summary.csv"

# Create or clean summary file
if [ "$CLEAN_START" = true ] || [ ! -f "$SUMMARY_FILE" ]; then
    echo "Creating new summary file..."
    echo "Experiment,Run,Pop_Size,Mut_Rate,Tour_Size,Div_Perc,Stag_Lim,Cost_Aware,Dataset,Best_Fitness,Solution_Gen,Total_Gens,Total_Time,Time_Per_Gen" > $SUMMARY_FILE
else
    echo "Appending to existing summary file..."
fi

# Function to extract results from log file
extract_results() {
    local log_file=$1
    local dataset=$2
    
    # Extract relevant metrics from the log file
    # Adjust these based on your actual log format
    grep -A 5 "Dataset: $dataset" "$log_file" | tail -n +2
}

# Main experiment loop
total_experiments=${#EXPERIMENTS[@]}
current_exp=1

for exp_idx in "${!EXPERIMENTS[@]}"; do
    IFS=' ' read -r pop mut tour div stag cost <<< "${EXPERIMENTS[$exp_idx]}"
    exp_num=$((exp_idx + 1))
    
    echo "=================================================="
    echo "Running Experiment $exp_num of $total_experiments"
    echo "Parameters: Pop=$pop, Mut=$mut, Tour=$tour, Div=$div, Stag=$stag, Cost=$cost"
    echo "=================================================="
    
    # Create experiment-specific directory
    exp_dir="taguchi_results/exp_${exp_num}"
    mkdir -p "$exp_dir"
    
    # Run multiple times for statistical significance
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        echo "  Run $run of $RUNS_PER_CONFIG..."
        
        # Clean up any existing run directory
        run_dir="$exp_dir/run_${run}"
        if [ -d "$run_dir" ]; then
            echo "  Cleaning up existing directory: $run_dir"
            rm -rf "$run_dir"
        fi
        mkdir -p "$run_dir"
        
        # Ensure the directory exists and is writable
        chmod 755 "$run_dir"
        
        # Construct command - note we're using build/ga_solver
        cmd="./build/ga_solver"
        cmd="$cmd -d $DATASETS"
        cmd="$cmd -p $pop"
        cmd="$cmd -m $mut"
        cmd="$cmd -t $tour"
        cmd="$cmd -v $div"
        cmd="$cmd -s $stag"
        cmd="$cmd -o 0"  # Mode fixed at 0
        cmd="$cmd -g 5000"  # Limit generations for faster experiments
        cmd="$cmd -l $run_dir"
        
        if [ "$cost" -eq 1 ]; then
            cmd="$cmd -c"
        fi
        
        # Run experiment
        log_file="$exp_dir/run_${run}_output.txt"
        echo "  Command: $cmd"
        echo "  Logging to: $log_file"
        
        # Execute with timeout (30 minutes max per run)
        timeout 1800 $cmd > "$log_file" 2>&1
        
        if [ $? -eq 124 ]; then
            echo "  WARNING: Run $run timed out after 30 minutes"
        fi
        
        # Extract results for each dataset
        echo "  Extracting results..."
        
        # Parse the results log CSV file
        results_csv="$exp_dir/run_${run}/results_log.csv"
        echo "  Looking for results in: $results_csv"
        
        if [ -f "$results_csv" ]; then
            echo "  Found results file, parsing..."
            # Skip header and process each line
            tail -n +2 "$results_csv" | while IFS=',' read -r dataset best_fit sol_gen total_gen total_time time_per_gen mode extra; do
                echo "$exp_num,$run,$pop,$mut,$tour,$div,$stag,$cost,$dataset,$best_fit,$sol_gen,$total_gen,$total_time,$time_per_gen" >> $SUMMARY_FILE
            done
        else
            echo "  WARNING: No results CSV found at $results_csv"
            # Check if the solver ran successfully
            if grep -q "Best solution found" "$log_file"; then
                echo "  Solver completed but no CSV generated"
            else
                echo "  ERROR: Solver may have failed - check $log_file"
            fi
        fi
        
        echo "  Run $run completed."
    done
    
    echo "Experiment $exp_num completed."
    echo ""
done

echo "=================================================="
echo "