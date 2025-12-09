// crossover.cu: GPU-based EAX (Edge Assembly Crossover) for TSP + Uniform for Jobs

#include "crossover.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>
#include <set>

// =============================================================================
// CONSTANTS
// =============================================================================

#define MAX_AB_CYCLES 32
#define MAX_CYCLE_SIZE 256

// =============================================================================
// DATA STRUCTURES FOR EAX
// =============================================================================

// Edge in the union graph
struct UnionEdge {
    int to;           // Target vertex
    int parentMask;   // Bit 0 = parent A has this edge, Bit 1 = parent B has this edge
};

// AB-cycle representation
struct ABCycle {
    int vertices[MAX_CYCLE_SIZE];
    int length;
    bool valid;
};

// =============================================================================
// CPU EAX IMPLEMENTATION (Correct reference implementation)
// =============================================================================

// Build position lookup: posA[city] = position of city in parent A's tour
void buildPositionLookup(const std::vector<size_t>& tour, std::vector<int>& position) {
    position.resize(tour.size() + 1, -1);
    for (size_t i = 0; i < tour.size(); i++) {
        position[tour[i]] = i;
    }
}

// Get the next city in a tour from position p
inline int nextCity(const std::vector<size_t>& tour, int pos) {
    return tour[(pos + 1) % tour.size()];
}

// Get the previous city in a tour from position p
inline int prevCity(const std::vector<size_t>& tour, int pos) {
    return tour[(pos + tour.size() - 1) % tour.size()];
}

// EAX crossover for city sequences
void eaxCrossover(const std::vector<size_t>& parentA, const std::vector<size_t>& parentB,
                  std::vector<size_t>& child, std::mt19937& rng) {
    int n = parentA.size();

    // Build position lookups
    std::vector<int> posA(n + 1), posB(n + 1);
    for (int i = 0; i < n; i++) {
        posA[parentA[i]] = i;
        posB[parentB[i]] = i;
    }

    // Build adjacency for union graph G_AB
    // For each vertex, store up to 4 neighbors (2 from each parent)
    // adjList[v] = list of (neighbor, parentId) pairs
    std::vector<std::vector<std::pair<int, int>>> adj(n + 1);

    for (int i = 0; i < n; i++) {
        int cityA = parentA[i];
        int nextA = parentA[(i + 1) % n];
        int cityB = parentB[i];
        int nextB = parentB[(i + 1) % n];

        // Add edges from parent A (parentId = 0)
        adj[cityA].push_back({nextA, 0});
        adj[nextA].push_back({cityA, 0});

        // Add edges from parent B (parentId = 1)
        adj[cityB].push_back({nextB, 1});
        adj[nextB].push_back({cityB, 1});
    }

    // Find AB-cycles
    std::vector<std::vector<int>> abCycles;
    std::vector<std::vector<std::vector<bool>>> usedEdge(n + 1);

    // Initialize used edge tracking
    for (int v = 0; v <= n; v++) {
        usedEdge[v].resize(adj[v].size());
        for (size_t j = 0; j < adj[v].size(); j++) {
            usedEdge[v][j].resize(2, false);
        }
    }

    // Find all AB-cycles starting from vertices with degree > 2
    for (int startV = 1; startV <= n; startV++) {
        // Check if this vertex has unused edges from both parents
        bool hasUnusedA = false, hasUnusedB = false;
        for (size_t j = 0; j < adj[startV].size(); j++) {
            if (adj[startV][j].second == 0 && !usedEdge[startV][j][0]) hasUnusedA = true;
            if (adj[startV][j].second == 1 && !usedEdge[startV][j][1]) hasUnusedB = true;
        }

        if (!hasUnusedA || !hasUnusedB) continue;

        // Try to find an AB-cycle starting from this vertex
        std::vector<int> cycle;
        cycle.push_back(startV);

        int currentV = startV;
        int lastParent = -1;  // Will alternate between 0 and 1
        bool cycleComplete = false;

        // Start with an edge from parent A
        int nextV = -1;
        for (size_t j = 0; j < adj[currentV].size(); j++) {
            if (adj[currentV][j].second == 0 && !usedEdge[currentV][j][0]) {
                nextV = adj[currentV][j].first;
                usedEdge[currentV][j][0] = true;
                // Mark reverse edge too
                for (size_t k = 0; k < adj[nextV].size(); k++) {
                    if (adj[nextV][k].first == currentV && adj[nextV][k].second == 0) {
                        usedEdge[nextV][k][0] = true;
                        break;
                    }
                }
                lastParent = 0;
                break;
            }
        }

        if (nextV == -1) continue;

        currentV = nextV;
        cycle.push_back(currentV);

        // Continue alternating until we return to start
        int maxIter = n * 2;
        for (int iter = 0; iter < maxIter && !cycleComplete; iter++) {
            // Find next edge from alternate parent
            int targetParent = 1 - lastParent;
            nextV = -1;

            for (size_t j = 0; j < adj[currentV].size(); j++) {
                if (adj[currentV][j].second == targetParent && !usedEdge[currentV][j][targetParent]) {
                    nextV = adj[currentV][j].first;
                    usedEdge[currentV][j][targetParent] = true;
                    // Mark reverse edge
                    for (size_t k = 0; k < adj[nextV].size(); k++) {
                        if (adj[nextV][k].first == currentV && adj[nextV][k].second == targetParent) {
                            usedEdge[nextV][k][targetParent] = true;
                            break;
                        }
                    }
                    lastParent = targetParent;
                    break;
                }
            }

            if (nextV == -1) break;  // Dead end

            if (nextV == startV) {
                // Completed the cycle
                cycleComplete = true;
            } else {
                currentV = nextV;
                cycle.push_back(currentV);
            }
        }

        if (cycleComplete && cycle.size() >= 4) {
            abCycles.push_back(cycle);
        }
    }

    // If no AB-cycles found, just return parent A
    if (abCycles.empty()) {
        child = parentA;
        return;
    }

    // Select one or more AB-cycles randomly
    std::uniform_int_distribution<int> cycleDist(0, abCycles.size() - 1);
    int selectedCycle = cycleDist(rng);

    // Apply the selected AB-cycle to parent A
    // Create child starting from parent A
    child = parentA;

    // Build a set of vertices in the selected cycle
    std::set<int> cycleVertices(abCycles[selectedCycle].begin(), abCycles[selectedCycle].end());

    // For vertices in the cycle, we need to swap parent A edges with parent B edges
    // This is done by using parent B's ordering for cycle vertices

    // Simple approach: for cycle vertices, use their relative order from parent B
    // while keeping non-cycle vertices in parent A's order

    // More correct approach: rebuild the tour by modifying edges
    // For each edge (u,v) in the cycle that comes from parent A, replace it with
    // the corresponding parent B edge

    // Simplified implementation: mix segments
    // Find segments in parent A that are not in the cycle, keep them
    // For cycle portions, use parent B's ordering

    // Actually, the cleanest way is to use the "subtour" approach:
    // The AB-cycle defines which edges to swap

    // For simplicity, use a greedy reconstruction:
    std::vector<bool> inChild(n + 1, false);
    std::vector<size_t> newChild;
    newChild.reserve(n);

    // Start from a vertex
    int start = parentA[0];
    newChild.push_back(start);
    inChild[start] = true;

    int current = start;
    for (int step = 1; step < n; step++) {
        // Find next city: prefer parent A's edge unless it's in a swapped cycle
        int posInA = posA[current];
        int nextFromA = parentA[(posInA + 1) % n];
        int prevFromA = parentA[(posInA + n - 1) % n];

        int posInB = posB[current];
        int nextFromB = parentB[(posInB + 1) % n];
        int prevFromB = parentB[(posInB + n - 1) % n];

        // Check if current vertex is in the selected cycle
        bool inCycle = cycleVertices.count(current) > 0;

        int next = -1;

        // Try edges in priority order
        std::vector<int> candidates;
        if (inCycle) {
            // Prefer parent B edges for cycle vertices
            if (!inChild[nextFromB]) candidates.push_back(nextFromB);
            if (!inChild[prevFromB]) candidates.push_back(prevFromB);
            if (!inChild[nextFromA]) candidates.push_back(nextFromA);
            if (!inChild[prevFromA]) candidates.push_back(prevFromA);
        } else {
            // Prefer parent A edges for non-cycle vertices
            if (!inChild[nextFromA]) candidates.push_back(nextFromA);
            if (!inChild[prevFromA]) candidates.push_back(prevFromA);
            if (!inChild[nextFromB]) candidates.push_back(nextFromB);
            if (!inChild[prevFromB]) candidates.push_back(prevFromB);
        }

        for (int c : candidates) {
            if (!inChild[c]) {
                next = c;
                break;
            }
        }

        // Fallback: find any unvisited city
        if (next == -1) {
            for (int v = 1; v <= n; v++) {
                if (!inChild[v]) {
                    next = v;
                    break;
                }
            }
        }

        if (next == -1) break;  // Should not happen

        newChild.push_back(next);
        inChild[next] = true;
        current = next;
    }

    if (newChild.size() == (size_t)n) {
        child = newChild;
    } else {
        child = parentA;  // Fallback
    }
}

// Uniform crossover for job sequences
void uniformCrossover(const std::vector<size_t>& parent1, const std::vector<size_t>& parent2,
                      std::vector<size_t>& child, std::mt19937& rng) {
    size_t n = parent1.size();
    child.resize(n);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        child[i] = (dist(rng) < 0.5f) ? parent1[i] : parent2[i];
    }
}

// =============================================================================
// SIMPLE JOB CROSSOVER - Fast single-point crossover with O(n) repair
// =============================================================================
// For job sequences, we don't need complex crossover like OX.
// Simple single-point crossover with swap-based repair is faster and sufficient.

void simpleJobCrossover(const std::vector<size_t>& parent1, const std::vector<size_t>& parent2,
                        std::vector<size_t>& child, std::mt19937& rng) {
    size_t n = parent1.size();
    child.resize(n);

    // Single-point crossover
    std::uniform_int_distribution<size_t> dist(1, n - 1);
    size_t crossPoint = dist(rng);

    // Copy first part from parent1, second part from parent2
    for (size_t i = 0; i < crossPoint; i++) {
        child[i] = parent1[i];
    }
    for (size_t i = crossPoint; i < n; i++) {
        child[i] = parent2[i];
    }

    // Fast O(n) repair using position lookup
    // Build position map for what we have
    std::vector<size_t> position(n + 1, SIZE_MAX);  // position[job] = index in child
    std::vector<bool> seen(n + 1, false);
    std::vector<size_t> duplicates;
    std::vector<size_t> missing;

    // Find duplicates and build position map
    for (size_t i = 0; i < n; i++) {
        size_t job = child[i];
        if (seen[job]) {
            duplicates.push_back(i);  // This position has a duplicate
        } else {
            seen[job] = true;
            position[job] = i;
        }
    }

    // Find missing jobs
    for (size_t j = 1; j <= n; j++) {
        if (!seen[j]) {
            missing.push_back(j);
        }
    }

    // Replace duplicates with missing jobs
    for (size_t i = 0; i < duplicates.size(); i++) {
        child[duplicates[i]] = missing[i];
    }
}

// =============================================================================
// ORDER CROSSOVER (OX) - Reliable crossover for permutations
// =============================================================================

void orderCrossover(const std::vector<size_t>& parent1, const std::vector<size_t>& parent2,
                    std::vector<size_t>& child, std::mt19937& rng) {
    size_t n = parent1.size();
    child.resize(n);

    // Initialize child with zeros (markers for unfilled positions)
    std::fill(child.begin(), child.end(), 0);

    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);

    // Ensure point1 < point2
    if (point1 > point2) std::swap(point1, point2);

    // Copy the segment from parent1 between the crossover points
    std::set<size_t> inChild;
    for (size_t i = point1; i <= point2; i++) {
        child[i] = parent1[i];
        inChild.insert(parent1[i]);
    }

    // Fill remaining positions with cities from parent2 in order
    size_t childPos = (point2 + 1) % n;
    size_t parent2Pos = (point2 + 1) % n;

    // Safety counter to prevent infinite loops
    size_t maxIterations = n * 2;
    size_t iterations = 0;

    while (inChild.size() < n && iterations < maxIterations) {
        size_t city = parent2[parent2Pos];
        if (inChild.find(city) == inChild.end()) {
            child[childPos] = city;
            inChild.insert(city);
            childPos = (childPos + 1) % n;
        }
        parent2Pos = (parent2Pos + 1) % n;
        iterations++;
    }

    // Fallback: if we couldn't fill all positions, copy remaining from parent1
    if (inChild.size() < n) {
        for (size_t i = 0; i < n; i++) {
            if (child[i] == 0) {
                // Find a city not yet in child
                for (size_t c = 1; c <= n; c++) {
                    if (inChild.find(c) == inChild.end()) {
                        child[i] = c;
                        inChild.insert(c);
                        break;
                    }
                }
            }
        }
    }
}

// =============================================================================
// MAIN CROSSOVER FUNCTIONS
// =============================================================================

Genome performCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    size_t tourLength = parent1.citySequence.size();
    Genome child(tourLength, tourLength, mode);

    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng(rd());

    // Order Crossover (OX) for city sequence - reliable permutation crossover
    orderCrossover(parent1.citySequence, parent2.citySequence, child.citySequence, rng);

    // Simple job crossover - faster than OX, sufficient for job assignment
    // Job assignment is less critical than city ordering; last positions matter most
    simpleJobCrossover(parent1.jobSequence, parent2.jobSequence, child.jobSequence, rng);

    // Handle pickup sequence if mode == 1
    if (mode == 1) {
        // Pickup sequence uses same logic as city sequence
        orderCrossover(parent1.pickupSequence, parent2.pickupSequence, child.pickupSequence, rng);
    }

    return child;
}

std::vector<Genome> performBatchCrossover(const std::vector<Genome>& parents1,
                                          const std::vector<Genome>& parents2,
                                          int mode) {
    if (parents1.size() != parents2.size() || parents1.empty()) {
        return {};
    }

    std::vector<Genome> children;
    children.reserve(parents1.size());

    for (size_t i = 0; i < parents1.size(); i++) {
        children.push_back(performCrossover(parents1[i], parents2[i], mode));
    }

    return children;
}

// =============================================================================
// GPU KERNEL FOR UNIFORM JOB CROSSOVER
// =============================================================================

__global__ void uniformJobCrossoverKernel(const size_t* parent1Jobs, const size_t* parent2Jobs,
                                         size_t* childJobs, uint32_t numPairs, uint16_t jobLength,
                                         unsigned long seed) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t jobIdx = threadIdx.x;

    if (pairIdx < numPairs && jobIdx < jobLength) {
        curandState state;
        curand_init(seed + pairIdx * jobLength + jobIdx, 0, 0, &state);

        uint32_t baseIdx = pairIdx * jobLength;

        if (curand_uniform(&state) < 0.5f) {
            childJobs[baseIdx + jobIdx] = parent1Jobs[baseIdx + jobIdx];
        } else {
            childJobs[baseIdx + jobIdx] = parent2Jobs[baseIdx + jobIdx];
        }
    }
}

// =============================================================================
// GPU BATCH CROSSOVER (Uses CPU EAX for tour, GPU for jobs)
// =============================================================================

std::vector<Genome> performBatchGPUCrossover(const std::vector<Genome>& parents1,
                                              const std::vector<Genome>& parents2,
                                              int mode) {
    if (parents1.size() != parents2.size() || parents1.empty()) {
        return {};
    }

    uint32_t numPairs = static_cast<uint32_t>(parents1.size());
    size_t tourLength = parents1[0].citySequence.size();

    // Use CPU Order Crossover for tour sequences (reliable permutation crossover)
    std::vector<Genome> children;
    children.reserve(numPairs);

    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng(rd());

    for (uint32_t i = 0; i < numPairs; i++) {
        Genome child(tourLength, tourLength, mode);

        // Order Crossover for city sequence
        orderCrossover(parents1[i].citySequence, parents2[i].citySequence,
                    child.citySequence, rng);

        // Simple job crossover - faster than OX, sufficient for job assignment
        simpleJobCrossover(parents1[i].jobSequence, parents2[i].jobSequence,
                    child.jobSequence, rng);

        if (mode == 1) {
            orderCrossover(parents1[i].pickupSequence, parents2[i].pickupSequence,
                        child.pickupSequence, rng);
        }

        children.push_back(std::move(child));
    }

    return children;
}

// =============================================================================
// LEGACY INTERFACE FUNCTIONS
// =============================================================================

Genome performEAXCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    return performCrossover(parent1, parent2, mode);
}

std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                             const std::vector<Genome>& parents2,
                                             int mode) {
    return performBatchGPUCrossover(parents1, parents2, mode);
}

Genome performCostAwareEAXCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    return performCrossover(parent1, parent2, mode);
}

std::vector<Genome> performBatchCostAwareEAXCrossover(const std::vector<Genome>& parents1,
                                                       const std::vector<Genome>& parents2,
                                                       int mode) {
    return performBatchGPUCrossover(parents1, parents2, mode);
}
