#include "../include/TimeSeriesTester.h"
#include <iostream>
#include <string>
#include <fstream>

TimeSeriesTester::TimeSeriesTester(std::vector<float> data, std::vector<float> weights, int nodes, int windowSize, int fitnessMode, std::string outFilename) 
    : dataWeights(weights.begin(), weights.begin() + nodes * windowSize), 
      mapWeights(weights.begin() + nodes * windowSize, weights.end()) 
{
    this->data = data;
    this->numberOfNodes = nodes;
    this->windowSize = windowSize;
    this->fitnessMode = fitnessMode;
    this->outFilename = outFilename;
}

auto TimeSeriesTester::test(std::ofstream &file) -> float {
    float cumulativeError = 0.0;
    int i = 0;
    int count = 0;
    float rmseScore = 0.0f;
    std::vector<float> mapInput(this->numberOfNodes, 0.0);
    std::cout << "Testing started " << std::endl;
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
            rmseScore += (y - data[windowSize * numberOfNodes + i + j]) * (y - data[windowSize * numberOfNodes + i + j]);
            if(this->fitnessMode > 0)
                prediction_error = 2.0f * prediction_error / (abs(y) + abs(data[windowSize * numberOfNodes + i + j]));
            // TODO: experiment with percentage error
            // TODO: experiment with max percentage error
            if(this->fitnessMode != 2)
                cumulativeError += prediction_error;
            else 
                cumulativeError = std::max(cumulativeError, prediction_error);

            ++count;
        }
        i += numberOfNodes;
    }

    float epsilon = (this->fitnessMode == 2) ? cumulativeError : cumulativeError / count;
    file << std::string("Testing result: ") << 1.0f / (1.0f + epsilon) << std::endl;
    file << std::string("RMSE result: ") << std::sqrt(rmseScore / count) << std::endl;
    return 1.0 / (1.0 + epsilon);
}