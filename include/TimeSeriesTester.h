#include <vector>
#include <cmath>
#include <string>

class TimeSeriesTester {
public:
    TimeSeriesTester(std::vector<float> data, std::vector<float> weights, int nodes, int windowSize, int fitnessMode, std::string outFilename);

    auto test(std::ofstream &file) -> float;

private:
    std::vector<float> data;
    std::vector<float> dataWeights;
    std::vector<float> mapWeights;

    int numberOfNodes;
    int windowSize;
    int fitnessMode;
    std::string outFilename;
};