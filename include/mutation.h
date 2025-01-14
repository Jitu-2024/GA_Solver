// mutation.h: Header file for GPU-based mutation in TSPJ

#ifndef MUTATION_H
#define MUTATION_H

#include "genome.h"

// Host function to perform mutation for TSPJ
void performMutation(Genome& genome, float mutationRate);

#endif // MUTATION_H
