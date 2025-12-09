#!/bin/bash
# Aggregate results from multiple benchmark runs
# Combines all results.csv files and computes statistics

RESULTS_DIR="benchmark_results"
OUTPUT_FILE="aggregated_results.csv"
STATS_FILE="aggregated_stats.txt"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dir DIR         Results directory (default: ${RESULTS_DIR})"
    echo "  --output FILE     Output CSV file (default: ${OUTPUT_FILE})"
    echo "  --best            Show only best fitness per problem"
    echo "  --compare         Compare runs side by side"
    echo "  --help, -h        Show this help"
}

MODE="aggregate"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            RESULTS_DIR=$2
            shift 2
            ;;
        --output)
            OUTPUT_FILE=$2
            shift 2
            ;;
        --best)
            MODE="best"
            shift
            ;;
        --compare)
            MODE="compare"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

# Find all result files
RESULT_FILES=$(find "${RESULTS_DIR}" -name "results.csv" -type f | sort)

if [[ -z "$RESULT_FILES" ]]; then
    echo "No results.csv files found in ${RESULTS_DIR}"
    exit 1
fi

NUM_RUNS=$(echo "$RESULT_FILES" | wc -l)
echo "Found ${NUM_RUNS} benchmark runs"
echo ""

case $MODE in
    aggregate)
        # Combine all results into one file with run identifier
        echo "run_id,problem,type,status,fitness,cities,elapsed_time,time_per_gen,solution_gen,return_code" > "${OUTPUT_FILE}"

        for file in $RESULT_FILES; do
            run_id=$(dirname "$file" | xargs basename)
            # Skip header and add run_id column
            tail -n +2 "$file" | while read line; do
                echo "${run_id},${line}" >> "${OUTPUT_FILE}"
            done
        done

        echo "Aggregated results saved to: ${OUTPUT_FILE}"
        echo ""
        echo "Summary Statistics:" | tee "${STATS_FILE}"
        echo "===================" | tee -a "${STATS_FILE}"

        # Count by status
        echo "" | tee -a "${STATS_FILE}"
        echo "Status counts:" | tee -a "${STATS_FILE}"
        tail -n +2 "${OUTPUT_FILE}" | cut -d',' -f4 | sort | uniq -c | tee -a "${STATS_FILE}"

        # Medium problem stats
        echo "" | tee -a "${STATS_FILE}"
        echo "Medium Problems:" | tee -a "${STATS_FILE}"
        grep ",medium,success," "${OUTPUT_FILE}" | cut -d',' -f5 | \
            awk 'BEGIN{min=999999;max=0;sum=0;n=0}
                 {if($1!="NA" && $1+0==$1){if($1<min)min=$1; if($1>max)max=$1; sum+=$1; n++}}
                 END{if(n>0)printf "  Count: %d, Min: %.2f, Max: %.2f, Avg: %.2f\n", n, min, max, sum/n; else print "  No successful runs"}' | tee -a "${STATS_FILE}"

        # Large problem stats
        echo "" | tee -a "${STATS_FILE}"
        echo "Large Problems:" | tee -a "${STATS_FILE}"
        grep ",large,success," "${OUTPUT_FILE}" | cut -d',' -f5 | \
            awk 'BEGIN{min=999999;max=0;sum=0;n=0}
                 {if($1!="NA" && $1+0==$1){if($1<min)min=$1; if($1>max)max=$1; sum+=$1; n++}}
                 END{if(n>0)printf "  Count: %d, Min: %.2f, Max: %.2f, Avg: %.2f\n", n, min, max, sum/n; else print "  No successful runs"}' | tee -a "${STATS_FILE}"

        echo ""
        echo "Stats saved to: ${STATS_FILE}"
        ;;

    best)
        # Find best fitness for each problem across all runs
        BEST_FILE="best_results.csv"
        echo "problem,type,best_fitness,best_run,num_runs,avg_fitness,std_dev" > "${BEST_FILE}"

        # Get unique problems
        problems=$(tail -n +2 ${RESULT_FILES} | cut -d',' -f1 | sort -u)

        for prob in $problems; do
            # Get all fitness values for this problem
            data=$(grep "^${prob}," ${RESULT_FILES} | grep ",success," | cut -d',' -f4)

            if [[ -n "$data" ]]; then
                # Find best (minimum) fitness
                best=$(echo "$data" | sort -n | head -1)

                # Find which run had the best
                best_run=$(grep "^${prob}," ${RESULT_FILES} | grep ",${best}," | head -1)
                best_run_file=$(grep -l "^${prob},.*,${best}," ${RESULT_FILES} | head -1)
                best_run_id=$(dirname "$best_run_file" | xargs basename)

                # Get problem type
                ptype=$(echo "$best_run" | cut -d',' -f2)

                # Calculate stats
                stats=$(echo "$data" | awk '
                    BEGIN{sum=0;sumsq=0;n=0}
                    {sum+=$1; sumsq+=$1*$1; n++}
                    END{
                        avg=sum/n;
                        if(n>1) std=sqrt((sumsq-sum*sum/n)/(n-1));
                        else std=0;
                        printf "%d,%.2f,%.2f", n, avg, std
                    }')

                num_runs=$(echo "$stats" | cut -d',' -f1)
                avg=$(echo "$stats" | cut -d',' -f2)
                std=$(echo "$stats" | cut -d',' -f3)

                echo "${prob},${ptype},${best},${best_run_id},${num_runs},${avg},${std}" >> "${BEST_FILE}"
            fi
        done

        echo "Best results saved to: ${BEST_FILE}"
        echo ""
        echo "Top 10 best results:"
        head -1 "${BEST_FILE}"
        tail -n +2 "${BEST_FILE}" | sort -t',' -k3 -n | head -10
        ;;

    compare)
        # Side-by-side comparison of runs
        echo "Run Comparison:"
        echo "==============="
        echo ""

        for file in $RESULT_FILES; do
            run_id=$(dirname "$file" | xargs basename)
            config_file=$(dirname "$file")/config.txt

            echo "--- ${run_id} ---"
            if [[ -f "$config_file" ]]; then
                grep -E "^(Population|Generations|Local Search):" "$config_file"
            fi

            # Count successes
            total=$(tail -n +2 "$file" | wc -l)
            success=$(grep ",success," "$file" | wc -l)
            echo "Results: ${success}/${total} successful"

            # Average fitness
            avg=$(grep ",success," "$file" | cut -d',' -f4 | \
                awk 'BEGIN{sum=0;n=0}{if($1!="NA"){sum+=$1;n++}}END{if(n>0)printf "%.2f",sum/n; else print "N/A"}')
            echo "Avg Fitness: ${avg}"

            # Average time
            avg_time=$(grep ",success," "$file" | cut -d',' -f6 | \
                awk 'BEGIN{sum=0;n=0}{sum+=$1;n++}END{if(n>0)printf "%.2f",sum/n; else print "N/A"}')
            echo "Avg Time: ${avg_time}s"
            echo ""
        done
        ;;
esac
