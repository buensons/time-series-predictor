#include "../include/Chromosome.h"

Chromosome::Chromosome() {
    fitness = 0.0;
    cumulativeProbability = 0.0;
}

auto Chromosome::random(int size) -> Chromosome {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    Chromosome chromosome;

    for(int i = 0; i < size; ++i) {
        chromosome.genes.push_back(distribution(generator));
    }
    return chromosome;
}

auto Chromosome::fromWeights(std::vector<float> genes) -> Chromosome {
    Chromosome chromosome;
    chromosome.genes = genes;
    return chromosome;
}