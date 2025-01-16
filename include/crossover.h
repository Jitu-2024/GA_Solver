// crossover.h: Header file for GPU-based order crossover in TSPJ

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "genome.h"
#include <utility>

// Host function to perform order crossover for TSPJ
std::pair<Genome, Genome> performCrossover(const Genome& parent1,
                                           const Genome& parent2, int mode);

#endif // CROSSOVER_H
