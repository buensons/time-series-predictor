#include <vector>
#include <cmath>

class TimeSeriesTester {
public:
    TimeSeriesTester(std::vector<float> data, std::vector<float> weights, int nodes, int windowSize);

    auto test() -> float;

private:
    std::vector<float> data;
    std::vector<float> dataWeights;
    std::vector<float> mapWeights;

    int numberOfNodes;
    int windowSize;
};