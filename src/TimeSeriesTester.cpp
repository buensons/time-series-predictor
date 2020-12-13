#include "../include/TimeSeriesTester.h"
#include <iostream>

TimeSeriesTester::TimeSeriesTester(std::vector<float> data, std::vector<float> weights, int nodes, int windowSize) 
    : dataWeights(weights.begin(), weights.begin() + nodes * windowSize), 
      mapWeights(weights.begin() + nodes * windowSize, weights.end()) 
{
    this->data = data;
    this->numberOfNodes = nodes;
    this->windowSize = windowSize;
}

auto TimeSeriesTester::test() -> float {
    float cumulativeError = 0.0;
    int i = 0;

    std::vector<float> mapInput(this->numberOfNodes, 0.0);

    while(i < data.size() - windowSize * numberOfNodes) {

        // data aggregation
        for(int j = 0; j < windowSize * numberOfNodes; ++j) {
            mapInput[j % numberOfNodes] += dataWeights[j] * data[i + j];
        }

        // squashing function for map input
        for(int j = 0; j < numberOfNodes; ++j) {
            mapInput[j] = 1.0 / (1.0 + std::exp(-5 * mapInput[j]));
        }
        
        // one step of cognitive map computation for each node
        for(int j = 0; j < numberOfNodes; ++j) {

            float x = 0.0f;
            for(int k = 0; k < numberOfNodes; ++k) {
                x += mapWeights[j * numberOfNodes + k] * mapInput[k];
            }

            float y = 1.0 / (1.0 + std::exp(-5 * x));
            float prediction_error = std::abs(y - data[windowSize * numberOfNodes + i + j]);

            // TODO: experiment with percentage error
            // TODO: experiment with max percentage error
            cumulativeError += prediction_error;
        }
        i += numberOfNodes;
    }

    float epsilon = cumulativeError / ((numberOfNodes * data.size()) - (numberOfNodes * windowSize));
    
    return 1.0 / (1.0 + epsilon);
}