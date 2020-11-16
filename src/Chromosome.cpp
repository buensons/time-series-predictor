#include "../include/Chromosome.h"

Chromosome::Chromosome() {
    fitness = 0.0;
    cumulativeProbability = 0.0;
}

auto Chromosome::fromWeights(std::vector<float> genes) -> Chromosome {
    Chromosome chromosome;
    chromosome.genes = genes;
    return chromosome;
}
