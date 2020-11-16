#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <vector>
#include <random>

class Chromosome {
public:
    float fitness;
    float cumulativeProbability;
    std::vector<float> genes;

    Chromosome();

    static auto fromWeights(std::vector<float> genes) -> Chromosome;
};
#endif
