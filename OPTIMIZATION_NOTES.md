# OX Crossover Optimization for Large TSP Problems

## Summary

The genetic operators in this project have been optimized to handle large TSP problems (512+ cities) efficiently.

## Changes Made

### 1. Removed MAX_CITIES Dependency
**Status: ✓ Confirmed**

All genetic operators (`Crossover` and `Mutation` classes) use the `numCities` parameter dynamically and do NOT rely on any hardcoded `MAX_CITIES` constant. This allows the system to scale to any problem size.

### 2. Optimized OX Crossover for 512+ Cities
**Location: `src/crossover.cu`**

#### Problem
The original OX crossover kernel had O(n²) complexity due to nested loops for duplicate checking (lines 46-77):
```cuda
for (int i = 0; i < numCities; ++i) {
    int city = parent2[(end + 1 + i) % numCities];
    bool isDuplicate = false;
    for (int j = start; j <= end; ++j) {  // Nested loop!
        if (child1[j] == city) {
            isDuplicate = true;
            break;
        }
    }
    if (!isDuplicate) {
        child1[child1Idx] = city;
        child1Idx = (child1Idx + 1) % numCities;
    }
}
```

For 512+ cities, this becomes very slow (262,144+ iterations per offspring pair).

#### Solution
Created an optimized kernel `orderCrossoverKernelOptimized()` that uses a lookup table (boolean array) instead of nested loops:

```cuda
bool* usedInChild1 = new bool[numCities];
// Mark cities used in the inherited segment
for (int i = start; i <= end; ++i) {
    usedInChild1[parent1[i]] = true;
}
// Fill remaining cities with O(1) lookup
for (int i = 0; i < numCities; ++i) {
    int city = parent2[(end + 1 + i) % numCities];
    if (!usedInChild1[city]) {  // O(1) lookup!
        child1[child1Idx] = city;
        child1Idx = (child1Idx + 1) % numCities;
    }
}
```

This reduces the complexity from **O(n²) to O(n)** for the duplicate checking operation.

#### Automatic Fallback
The `orderCrossover()` method now automatically selects the appropriate kernel based on problem size:

```cpp
if (numCities >= 512) {
    // Use optimized O(n) kernel for large problem sizes
    orderCrossoverKernelOptimized<<<blocks, threadsPerBlock>>>(...);
} else {
    // Use original kernel for small-medium problem sizes
    orderCrossoverKernel<<<blocks, threadsPerBlock>>>(...);
}
```

### 3. Performance Improvement

For a 1000-city TSP problem:
- **Original kernel**: ~1,000,000 iterations per offspring pair (worst case)
- **Optimized kernel**: ~2,000 iterations per offspring pair (2× numCities)

**Expected speedup: ~500x** for the duplicate checking operation alone.

## Testing

A comprehensive test has been added in `tests/test_large_crossover.cu` that:
1. Tests the original kernel with 100 cities (< 512 threshold)
2. Tests the optimized kernel with 1000 cities (> 512 threshold)
3. Validates that offspring are valid permutations
4. Verifies that the correct segments are inherited from parents

To build and run the test:
```bash
cd build
cmake ..
make test_large_crossover
./test_large_crossover
```

## Benefits

1. **Scalable**: No MAX_CITIES limit - can handle any problem size
2. **Fast for Large Problems**: O(n) complexity for 512+ cities
3. **Backward Compatible**: Original kernel preserved for smaller problems where overhead of lookup table allocation might not be beneficial
4. **GPU-Accelerated**: All crossover operations run on GPU for both small and large problems
5. **Automatic**: No user intervention required - the system automatically selects the best kernel

## Files Modified

- `src/crossover.cu`: Added optimized kernel and conditional logic
- `tests/test_large_crossover.cu`: New comprehensive test
- `CMakeLists.txt`: Added test target for large crossover
