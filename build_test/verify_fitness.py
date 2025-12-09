#!/usr/bin/env python3
"""Verify TSPJ fitness calculation"""
import numpy as np
import sys

# Load data
travel_times = []
with open('data/TSPJ_1M_TSPJ_TT.csv', 'r') as f:
    for line in f:
        row = [float(x) if x.strip() else 0.0 for x in line.strip().split(',')]
        travel_times.append(row)
travel_times = np.array(travel_times)

job_times_raw = []
with open('data/TSPJ_1M_TSPJ_JT.csv', 'r') as f:
    for line in f:
        row = [float(x) if x.strip() else 0.0 for x in line.strip().split(',')]
        if len(row) > 0:
            job_times_raw.append(row)
job_times_raw = np.array(job_times_raw)

numCities = travel_times.shape[0]  # 423
chromosomeLength = numCities - 1  # 422
numJobs = job_times_raw.shape[1] - 1  # 422

# Create flat job times (skip depot row, skip first column)
flatJobTimes = np.zeros((numCities - 1) * numJobs)
for i in range(1, numCities):
    for k in range(numJobs):
        flatJobTimes[(i - 1) * numJobs + k] = job_times_raw[i, k + 1]

def calculate_fitness(city_seq, job_seq, mode=2):
    """Calculate fitness exactly like the GPU kernel"""
    maxCompletionTime = 0.0
    currentTime = 0.0
    prevCity = 0  # depot
    
    for i in range(chromosomeLength):
        city = city_seq[i]
        job = job_seq[i]
        
        # Add travel time
        currentTime += travel_times[prevCity, city]
        
        if city > 0:
            # Job completion
            jobTime = flatJobTimes[(city - 1) * numJobs + (job - 1)]
            jobCompletionTime = currentTime + jobTime
            maxCompletionTime = max(maxCompletionTime, jobCompletionTime)
        
        prevCity = city
    
    # Return to depot
    currentTime += travel_times[prevCity, 0]
    
    return max(maxCompletionTime, currentTime), currentTime, maxCompletionTime

# Generate a random tour and job sequence like the GA does
np.random.seed(42)
city_seq = list(range(1, numCities))  # 1 to 422
np.random.shuffle(city_seq)
job_seq = list(range(1, numJobs + 1))  # 1 to 422
np.random.shuffle(job_seq)

fitness, tour_time, max_job = calculate_fitness(city_seq, job_seq)

print(f"=== Verification Test ===")
print(f"numCities (including depot): {numCities}")
print(f"chromosomeLength: {chromosomeLength}")
print(f"numJobs: {numJobs}")
print(f"\nRandom tour fitness:")
print(f"  Tour return time: {tour_time:.0f}")
print(f"  Max job completion: {max_job:.0f}")
print(f"  Fitness: {fitness:.0f}")

# Calculate nearest neighbor tour fitness
def nearest_neighbor_tour():
    n = travel_times.shape[0]
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        best_next = -1
        best_dist = float('inf')
        for j in range(1, n):
            if not visited[j] and travel_times[current, j] < best_dist:
                best_dist = travel_times[current, j]
                best_next = j
        if best_next >= 0:
            visited[best_next] = True
            tour.append(best_next)
            current = best_next
    return tour

nn_tour = nearest_neighbor_tour()
nn_job_seq = list(range(1, numJobs + 1))  # Same jobs, sequential
nn_fitness, nn_tour_time, nn_max_job = calculate_fitness(nn_tour, nn_job_seq)

print(f"\nNearest Neighbor tour fitness:")
print(f"  Tour return time: {nn_tour_time:.0f}")
print(f"  Max job completion: {nn_max_job:.0f}")
print(f"  Fitness: {nn_fitness:.0f}")

# Show what job times look like
print(f"\n=== Job Time Stats ===")
print(f"  Min job time: {flatJobTimes.min():.0f}")
print(f"  Max job time: {flatJobTimes.max():.0f}")
print(f"  Mean job time: {flatJobTimes.mean():.0f}")
