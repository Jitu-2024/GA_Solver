#ifndef GENOME_HPP
#define GENOME_HPP

#include <bitset>
#include <vector>
#include <cstdlib>
#include <ctime>

class Genome {
public:
    static const int gene_length = 8; // Bits per gene
    Genome(int num_genes);

    // Mutation function
    void mutate(float mutation_rate);

    // Utility function to print the genome (for debugging)
    void print() const;

private:
    std::vector<std::bitset<gene_length>> genes; // Vector of bitsets representing genes
};

#endif // GENOME_HPP
