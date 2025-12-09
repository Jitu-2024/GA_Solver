#!/bin/bash
# Benchmark script for TSPJ GA Solver
# Runs all medium (TSPJ_*M) and/or large (TSPJ_*L) datasets
# Stores results for later aggregation

# Configuration - modify these as needed
POPULATION=3000
GENERATIONS=2000
STAGNATION=150
LOCAL_SEARCH=0
MODE=0
TIMEOUT=600  # 10 minutes per problem

# Output directories
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/run_${TIMESTAMP}"

# Create output directories
mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/raw_output"

# Results CSV file
RESULTS_CSV="${RUN_DIR}/results.csv"

# Write CSV header
echo "problem,type,status,fitness,cities,elapsed_time,time_per_gen,solution_gen,return_code" > "${RESULTS_CSV}"

# Function to run a single problem and extract results
run_problem() {
    local name=$1
    local ptype=$2

    echo "============================================================"
    echo "Running: ${name} (${ptype})"
    echo "============================================================"

    local start_time=$(date +%s.%N)
    local output_file="${RUN_DIR}/raw_output/${name}.txt"

    # Run the solver with timeout
    timeout ${TIMEOUT} ./ga_solver -d "${name}" -o ${MODE} -p ${POPULATION} \
        -g ${GENERATIONS} -s ${STAGNATION} --local-search ${LOCAL_SEARCH} \
        > "${output_file}" 2>&1

    local return_code=$?
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    # Determine status
    local status="success"
    if [ $return_code -eq 124 ]; then
        status="timeout"
    elif [ $return_code -ne 0 ]; then
        status="failed"
    fi

    # Extract fitness (line starting with "Fitness:")
    local fitness=$(grep -m1 "^Fitness:" "${output_file}" | grep -oP '[0-9]+\.?[0-9]*' | head -1)

    # Fallback: get last "Best Fitness of Generation" if no final fitness
    if [ -z "$fitness" ]; then
        fitness=$(grep "Best Fitness of Generation" "${output_file}" | tail -1 | grep -oP '[0-9]+\.?[0-9]*$')
    fi

    # If we got fitness, mark as success even if return code was non-zero
    if [ -n "$fitness" ]; then
        status="success"
    fi

    # Extract number of cities
    local cities=$(grep "Number of Cities:" "${output_file}" | grep -oP '[0-9]+' | head -1)

    # Extract time per generation
    local time_per_gen=$(grep "Time per generation" "${output_file}" | grep -oP '[0-9]+\.?[0-9]*' | head -1)

    # Extract solution generation (last improvement)
    local solution_gen=$(grep "Best Fitness of Generation" "${output_file}" | tail -1 | grep -oP 'Generation \K[0-9]+')

    # Default empty values
    fitness=${fitness:-"NA"}
    cities=${cities:-"NA"}
    time_per_gen=${time_per_gen:-"NA"}
    solution_gen=${solution_gen:-"NA"}

    # Print summary
    echo "Status: ${status}"
    echo "Fitness: ${fitness}"
    echo "Cities: ${cities}"
    echo "Time: ${elapsed}s"
    echo ""

    # Append to CSV
    echo "${name},${ptype},${status},${fitness},${cities},${elapsed},${time_per_gen},${solution_gen},${return_code}" >> "${RESULTS_CSV}"
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --medium, -m      Run only medium problems (TSPJ_*M)"
    echo "  --large, -l       Run only large problems (TSPJ_*L)"
    echo "  --all, -a         Run all problems (default)"
    echo "  --test, -t        Run first 5 problems of each type"
    echo "  --count N         Run first N problems of each type"
    echo "  --pop N           Population size (default: ${POPULATION})"
    echo "  --gen N           Max generations (default: ${GENERATIONS})"
    echo "  --stag N          Stagnation limit (default: ${STAGNATION})"
    echo "  --ls N            Local search intensity 0-3 (default: ${LOCAL_SEARCH})"
    echo "  --timeout N       Timeout per problem in seconds (default: ${TIMEOUT})"
    echo "  --help, -h        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                         # Run all problems"
    echo "  $0 --medium                # Run only medium problems"
    echo "  $0 --large                 # Run only large problems"
    echo "  $0 --test                  # Test mode (5 each)"
    echo "  $0 --medium --count 10     # First 10 medium problems"
    echo "  $0 --pop 5000 --gen 3000   # Custom settings"
}

# Parse arguments
PROBLEM_TYPE="all"
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --medium|-m)
            PROBLEM_TYPE="medium"
            shift
            ;;
        --large|-l)
            PROBLEM_TYPE="large"
            shift
            ;;
        --all|-a)
            PROBLEM_TYPE="all"
            shift
            ;;
        --test|-t)
            LIMIT=5
            shift
            ;;
        --count)
            LIMIT=$2
            shift 2
            ;;
        --pop)
            POPULATION=$2
            shift 2
            ;;
        --gen)
            GENERATIONS=$2
            shift 2
            ;;
        --stag)
            STAGNATION=$2
            shift 2
            ;;
        --ls)
            LOCAL_SEARCH=$2
            shift 2
            ;;
        --timeout)
            TIMEOUT=$2
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Save configuration
CONFIG_FILE="${RUN_DIR}/config.txt"
cat > "${CONFIG_FILE}" << EOF
Benchmark Configuration
=======================
Timestamp: ${TIMESTAMP}
Problem Type: ${PROBLEM_TYPE}
Limit: ${LIMIT:-"all"}
Population: ${POPULATION}
Generations: ${GENERATIONS}
Stagnation: ${STAGNATION}
Local Search: ${LOCAL_SEARCH}
Mode: ${MODE}
Timeout: ${TIMEOUT}s
EOF

echo ""
echo "============================================================"
echo "TSPJ GA SOLVER BENCHMARK"
echo "============================================================"
echo "Problem type: ${PROBLEM_TYPE}"
echo "Limit: ${LIMIT:-all}"
echo "Population: ${POPULATION}, Generations: ${GENERATIONS}"
echo "Stagnation: ${STAGNATION}, Local Search: ${LOCAL_SEARCH}"
echo "Timeout: ${TIMEOUT}s per problem"
echo "Results will be saved to: ${RUN_DIR}"
echo "============================================================"
echo ""

# Track timing
TOTAL_START=$(date +%s)

# Count problems
MEDIUM_COUNT=0
LARGE_COUNT=0
MEDIUM_SUCCESS=0
LARGE_SUCCESS=0

# Run medium problems
if [[ "$PROBLEM_TYPE" == "medium" || "$PROBLEM_TYPE" == "all" ]]; then
    count=0
    for i in $(seq 1 100); do
        if [[ -n "$LIMIT" && $count -ge $LIMIT ]]; then
            break
        fi

        name="TSPJ_${i}M"
        tt_file="data/${name}_TSPJ_TT.csv"
        jt_file="data/${name}_TSPJ_JT.csv"

        if [[ -f "$tt_file" && -f "$jt_file" ]]; then
            ((MEDIUM_COUNT++))
            ((count++))
            echo "[Medium ${count}] "
            run_problem "${name}" "medium"

            # Check if successful
            last_status=$(tail -1 "${RESULTS_CSV}" | cut -d',' -f3)
            if [[ "$last_status" == "success" ]]; then
                ((MEDIUM_SUCCESS++))
            fi
        fi
    done
fi

# Run large problems
if [[ "$PROBLEM_TYPE" == "large" || "$PROBLEM_TYPE" == "all" ]]; then
    count=0
    for i in $(seq 1 100); do
        if [[ -n "$LIMIT" && $count -ge $LIMIT ]]; then
            break
        fi

        name="TSPJ_${i}L"
        tt_file="data/${name}_TSPJ_TT.csv"
        jt_file="data/${name}_TSPJ_JT.csv"

        if [[ -f "$tt_file" && -f "$jt_file" ]]; then
            ((LARGE_COUNT++))
            ((count++))
            echo "[Large ${count}] "
            run_problem "${name}" "large"

            # Check if successful
            last_status=$(tail -1 "${RESULTS_CSV}" | cut -d',' -f3)
            if [[ "$last_status" == "success" ]]; then
                ((LARGE_SUCCESS++))
            fi
        fi
    done
fi

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_TIME / 60))

# Print summary
echo ""
echo "============================================================"
echo "BENCHMARK SUMMARY"
echo "============================================================"
echo "Medium problems: ${MEDIUM_SUCCESS}/${MEDIUM_COUNT} successful"
echo "Large problems: ${LARGE_SUCCESS}/${LARGE_COUNT} successful"
echo "Total time: ${TOTAL_TIME}s (${TOTAL_MINUTES} min)"
echo ""
echo "Results saved to: ${RESULTS_CSV}"
echo "Raw output logs: ${RUN_DIR}/raw_output/"
echo "Configuration: ${CONFIG_FILE}"
echo "============================================================"

# Generate quick stats if we have successful runs
if [[ -f "${RESULTS_CSV}" ]]; then
    echo ""
    echo "Quick Statistics:"
    echo "-----------------"

    # Medium stats
    if [[ $MEDIUM_SUCCESS -gt 0 ]]; then
        echo "Medium problems fitness:"
        grep ",medium,success," "${RESULTS_CSV}" | cut -d',' -f4 | \
            awk 'BEGIN{min=999999;max=0;sum=0;n=0}
                 {if($1!="NA"){if($1<min)min=$1; if($1>max)max=$1; sum+=$1; n++}}
                 END{if(n>0)printf "  Min: %.2f, Max: %.2f, Avg: %.2f\n", min, max, sum/n}'
    fi

    # Large stats
    if [[ $LARGE_SUCCESS -gt 0 ]]; then
        echo "Large problems fitness:"
        grep ",large,success," "${RESULTS_CSV}" | cut -d',' -f4 | \
            awk 'BEGIN{min=999999;max=0;sum=0;n=0}
                 {if($1!="NA"){if($1<min)min=$1; if($1>max)max=$1; sum+=$1; n++}}
                 END{if(n>0)printf "  Min: %.2f, Max: %.2f, Avg: %.2f\n", min, max, sum/n}'
    fi
fi

echo ""
echo "To view results: cat ${RESULTS_CSV}"
echo "To aggregate multiple runs, use: ./aggregate_results.sh"
