#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include "include/TimeSeriesPredictor.cuh"

auto readDataToMemory(std::string path) -> std::vector<float>;
auto normalizeData(std::vector<float> &data) -> void;
auto usage(char * arg) -> void;

int main(int argc, char ** argv) {
    int nodes, populationSize, windowSize;
    std::string dataPath;

    if(argc != 5) {
        usage(argv[0]);
        return 1;
    }

    try {
        nodes = std::stoi(argv[1]);
        populationSize = std::stoi(argv[2]);
        windowSize = std::stoi(argv[3]);
        dataPath = std::string(argv[4]);
    } catch(std::exception const & e) {
        usage(argv[0]);
        return 1;
    }

    std::cout << "Reading input data from " << dataPath << std::endl;
    auto timeSeries = readDataToMemory(dataPath);
    normalizeData(timeSeries);

    TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize);
    std::vector<float> weights = predictor.train();

    // TODO: save the weights
    // TODO: add predict() method and use test data

    return 0;
}

auto usage(char * arg) -> void {
    std::cerr << "Usage: " << arg << " <num_of_nodes> <population_size> <window_size> <path_to_data>" << std::endl;
}

auto normalizeData(std::vector<float> &data) -> void {
    const auto [a, b] = std::minmax_element(begin(data), end(data));
    float min = *a;
    float max = *b;

    for(int i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - min) / (max - min);
    }
}

auto extractDataFromCsv(std::string path, std::vector<float> &dataVector) {
    std::ifstream infile(path);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
                
        while(iss.good()){
            std::string substr;
            getline(iss, substr, ',');
            dataVector.push_back(std::stof(substr));
        }
    }
}

auto readDataToMemory(std::string path) -> std::vector<float> {
    std::vector<float> dataVector;
    
    size_t pos = path.rfind('.');

    if (pos != std::string::npos) {
        std::string ext = path.substr(pos+1);

        if (ext == "csv") {
            extractDataFromCsv(path, dataVector);
            return dataVector;
        }
    }

    for(auto& p: std::filesystem::directory_iterator(path)) {
        const auto extension = p.path().extension();

        if(extension == ".csv") {
            extractDataFromCsv(p.path().string(), dataVector);
        }
    }

    return dataVector; 
}
