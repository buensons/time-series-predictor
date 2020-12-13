#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono> 
#include <algorithm>

#include "include/TimeSeriesPredictor.cuh"
#include "include/TimeSeriesTester.h"

auto readDataToMemory(const std::string &path) -> std::vector<float>;
void extractDataFromCsv(const std::string &path, std::vector<float> &data);
void saveWeights(const std::vector<float> &weights, std::string outFilename);
auto normalizeData(std::vector<float> &data) -> void;
auto usage(char * arg) -> void;


// Main function takes parameters in the following shape if training
// -> (string) mode (test or not test)
// -> (int) WinSize
// -> (string) data path
// -> (int) FFunction, where 0 = abs, 1 = avg %, 2 = max %
// -> (int) Population
// -> (float) Pmutation
// -> (float) Pcrossover
// -> (string with extension) Filename

// and if only testing 
// -> (string) mode
// -> (int) WinSize
// -> (string) data path
// -> (int) FFunction, where 0 = abs, 1 = avg %, 2 = max %
// -> (string) weightsFilePath
// -> (string) outFilename
int main(int argc, char ** argv) {
    int nodes, populationSize, windowSize, fitnessMode;
    float pMutation, pCrossover;
    std::string dataPath, mode, weightsFilePath, outFilename;
    std::vector<float> weights;

    try {
        mode = std::string(argv[1]);

        if(mode != "test" && mode != "train") {
            throw std::invalid_argument("Incorrect mode. Available options: test, train\n");
        }

        // nodes have been hardcoded as the data is 3 dimensional 
        nodes = 3;
        windowSize = std::stoi(argv[2]);
        dataPath = std::string(argv[3]);
        fitnessMode = std::stoi(argv[4]);
        
        if(mode == "train") {
            if(argc != 9) {
                usage(argv[0]);
                return 1;
            }
            pMutation = std::stof(argv[6]);
            pCrossover = std::stof(argv[7]);
            outFilename = std::string(argv[8]);
            populationSize = std::stoi(argv[5]);
        }
        else {
            if(argc != 7) {
                usage(argv[0]);
                return 1;
            }
            weightsFilePath = std::string(argv[5]);
            outFilename = std::string(argv[6]);
        }

    } catch(std::exception const & e) {
        usage(argv[0]);
        throw e;
    }

    std::cout << "Reading input data from " << dataPath << std::endl;
    auto timeSeries = readDataToMemory(dataPath);
    normalizeData(timeSeries);
    std::ofstream file;
    file.open (outFilename + ".results");
    if(mode == "test") {
        extractDataFromCsv(weightsFilePath, weights);
        file << "======================Testing phase results: ============================\n" << std::endl;
        TimeSeriesTester tester(timeSeries, weights, nodes, windowSize, fitnessMode, outFilename);
        float result = tester.test(file);
        file.close();
    } else {
        file << "======================Training phase results: ============================\n" << std::endl;
        auto start = std::chrono::high_resolution_clock::now(); 
        TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize, fitnessMode, pMutation, pCrossover);
        weights = predictor.train(file);
        auto stop = std::chrono::high_resolution_clock::now(); 
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        file << "Training time: " << duration.count() << std::endl;
        file.close();
        saveWeights(weights, outFilename);
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

void saveWeights(const std::vector<float> &weights, std::string outFilename) {
    std::ofstream file;
    file.open (outFilename + ".weights");
    
    for(auto weight: weights) {
        file << weight << std::endl;
    }

    file.close();

    std::cout << "Weights saved to " << outFilename << " file" << std::endl;
}
