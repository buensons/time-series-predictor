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
#include "include/TimeSeriesTester.h"

auto readDataToMemory(const std::string &path) -> std::vector<float>;
void extractDataFromCsv(const std::string &path, std::vector<float> &data);
void saveWeights(const std::vector<float> &weights);
auto normalizeData(std::vector<float> &data) -> void;
auto usage(char * arg) -> void;

int main(int argc, char ** argv) {
    int nodes, populationSize, windowSize;
    std::string dataPath, mode, weightsFilePath;
    std::vector<float> weights;

    if(argc != 6) {
        usage(argv[0]);
        return 1;
    }

    try {
        mode = std::string(argv[1]);

        if(mode != "test" && mode != "train") {
            throw std::invalid_argument("Incorrect mode. Available options: test, train\n");
        }
        
        nodes = std::stoi(argv[2]);
        windowSize = std::stoi(argv[3]);
        dataPath = std::string(argv[4]);

        if(mode == "train") populationSize = std::stoi(argv[5]);
        else weightsFilePath = std::string(argv[5]);

    } catch(std::exception const & e) {
        usage(argv[0]);
        throw e;
    }

    std::cout << "Reading input data from " << dataPath << std::endl;
    auto timeSeries = readDataToMemory(dataPath);
    normalizeData(timeSeries);

    if(mode == "test") {
        extractDataFromCsv(weightsFilePath, weights);

        TimeSeriesTester tester(timeSeries, weights, nodes, windowSize);
        float result = tester.test();
        std::cout << "Result on test data: " << result << std::endl;
    } else {
        TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize);
        weights = predictor.train();
        saveWeights(weights);
    }

    return 0;
}

auto usage(char * arg) -> void {
    std::cerr << "Usage: " << arg << " (train | test) <num_of_nodes> <window_size> <path_to_data> "
        << "(<population_size> | <weights_file_path>)" << std::endl;
}

auto normalizeData(std::vector<float> &data) -> void {
    const auto [a, b] = std::minmax_element(begin(data), end(data));
    float min = *a;
    float max = *b;

    for(int i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - min) / (max - min);
    }
}

void extractDataFromCsv(const std::string &path, std::vector<float> &data) {
    std::ifstream infile(path);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
                
        while(iss.good()){
            std::string substr;
            getline(iss, substr, ',');
            data.push_back(std::stof(substr));
        }
    }
}

auto readDataToMemory(const std::string &path) -> std::vector<float> {
    std::vector<float> data;
    
    size_t pos = path.rfind('.');

    if (pos != std::string::npos) {
        std::string ext = path.substr(pos+1);

        if (ext == "csv") {
            extractDataFromCsv(path, data);
            return data;
        }
    }

    for(auto& p: std::filesystem::directory_iterator(path)) {
        const auto extension = p.path().extension();

        if(extension == ".csv") {
            extractDataFromCsv(p.path().string(), data);
        }
    }

    return data; 
}

void saveWeights(const std::vector<float> &weights) {
    std::ofstream file;
    file.open ("./weights.csv");
    
    for(auto weight: weights) {
        file << weight << std::endl;
    }

    file.close();

    std::cout << "Weights saved to weights.csv file" << std::endl;
}
