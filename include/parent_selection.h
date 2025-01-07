#ifndef PARENT_SELECTION_H
#define PARENT_SELECTION_H

#include <vector>
#include "genome.h"

class ParentSelection {
public:
    // Constructor: Initializes with the fitness values of the population
    ParentSelection(const std::vector<Genome>& population);

    // Destructor: Frees GPU memory
    ~ParentSelection();

    // Select parents using tournament selection
    void tournamentSelection(const std::vector<Genome>& population, std::vector<Genome>& parents, int numParents, int tournamentSize);


private:
    float* d_fitness; // Device fitness array
    int populationSize;
};

#endif // PARENT_SELECTION_H
