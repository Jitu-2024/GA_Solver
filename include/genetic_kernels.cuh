#ifndef GENETIC_KERNELS_CUH
#define GENETIC_KERNELS_CUH

#include <cstdint>

void gpu_mutate(uint8_t* population, int total_bits, float mutation_rate);

#endif // GENETIC_KERNELS_CUH
