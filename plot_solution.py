#!/usr/bin/env python3
"""
Plot TSPJ solution from experiment results
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_coordinates(coords_file):
    """Load city coordinates from CSV file"""
    coords = []
    with open(coords_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                coords.append((float(row[0]), float(row[1])))
    return coords

def plot_tspj_solution(problem_name, solution_file, coords_file, output_file=None):
    """Plot the TSPJ solution"""

    # Load solution from JSON
    with open(solution_file, 'r') as f:
        data = json.load(f)

    if problem_name not in data:
        print(f"Problem {problem_name} not found in solution file")
        return

    problem_data = data[problem_name]
    city_sequence = problem_data['solution']['city_sequence']
    job_sequence = problem_data['solution']['job_sequence']
    fitness = problem_data['final_fitness']

    # Load coordinates
    coords = load_coordinates(coords_file)
    num_cities = len(coords)

    print(f"Problem: {problem_name}")
    print(f"Number of cities: {num_cities}")
    print(f"Tour length: {len(city_sequence)}")
    print(f"Fitness: {fitness}")

    # Add depot at position 0 (use first coordinate as depot or center)
    # The city indices in solution are 1-indexed, coordinates are 0-indexed
    # Depot is at index 0 in coordinates

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Tour visualization
    ax1 = axes[0]

    # Extract coordinates for the tour
    # City sequence uses 1-indexed cities, coords are 0-indexed
    tour_x = []
    tour_y = []

    # Start from depot (index 0)
    depot_x, depot_y = coords[0]
    tour_x.append(depot_x)
    tour_y.append(depot_y)

    # Add cities in tour order
    for city_idx in city_sequence:
        # city_idx is 1-indexed, coords[0] is depot, coords[1] is city 1, etc.
        if city_idx <= len(coords):
            x, y = coords[city_idx - 1]  # Convert to 0-indexed
            tour_x.append(x)
            tour_y.append(y)

    # Return to depot
    tour_x.append(depot_x)
    tour_y.append(depot_y)

    # Plot the tour path
    ax1.plot(tour_x, tour_y, 'b-', linewidth=0.5, alpha=0.7, label='Tour')

    # Plot all cities
    all_x = [c[0] for c in coords]
    all_y = [c[1] for c in coords]
    ax1.scatter(all_x, all_y, c='lightblue', s=20, zorder=2, alpha=0.7)

    # Highlight depot
    ax1.scatter([depot_x], [depot_y], c='red', s=100, marker='s', zorder=5, label='Depot')

    # Highlight first and last few cities in tour
    n_highlight = 5
    for i, city_idx in enumerate(city_sequence[:n_highlight]):
        if city_idx <= len(coords):
            x, y = coords[city_idx - 1]
            ax1.scatter([x], [y], c='green', s=50, zorder=4)
            ax1.annotate(f'{i+1}', (x, y), fontsize=8, ha='center', va='bottom')

    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title(f'{problem_name} Tour\nFitness: {fitness:.0f}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tour order visualization (shows tour structure)
    ax2 = axes[1]

    # Create a heatmap-like visualization of tour distances
    n = len(city_sequence)
    distances = []

    # Calculate distances between consecutive cities
    prev_x, prev_y = depot_x, depot_y
    for city_idx in city_sequence:
        if city_idx <= len(coords):
            x, y = coords[city_idx - 1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            distances.append(dist)
            prev_x, prev_y = x, y

    # Add return to depot
    dist_back = np.sqrt((prev_x - depot_x)**2 + (prev_y - depot_y)**2)
    distances.append(dist_back)

    # Plot distance profile
    ax2.bar(range(len(distances)), distances, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Step in Tour')
    ax2.set_ylabel('Distance to Next City')
    ax2.set_title('Distance Profile Along Tour')
    ax2.axhline(y=np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    # Print statistics
    print(f"\nTour Statistics:")
    print(f"  Total tour distance: {sum(distances):.1f}")
    print(f"  Mean edge distance: {np.mean(distances):.1f}")
    print(f"  Max edge distance: {max(distances):.1f}")
    print(f"  Min edge distance: {min(distances):.1f}")

    return fig

if __name__ == "__main__":
    # Default paths
    problem_name = sys.argv[1] if len(sys.argv) > 1 else "TSPJ_1M"

    solution_file = "/home/harish/large_problem_tspj/GA_Solver/experiment_results/all_problems_20251208_053617_solutions.json"
    coords_file = f"/home/harish/large_problem_tspj/GA_Solver/data/Medium_problems/Batch_01/{problem_name}_nodes_table_by_coordinates.csv"
    output_file = f"/home/harish/large_problem_tspj/GA_Solver/{problem_name}_solution_plot.png"

    if not os.path.exists(coords_file):
        print(f"Coordinates file not found: {coords_file}")
        sys.exit(1)

    if not os.path.exists(solution_file):
        print(f"Solution file not found: {solution_file}")
        sys.exit(1)

    plot_tspj_solution(problem_name, solution_file, coords_file, output_file)
