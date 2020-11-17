#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include "include/ResultVerifier.hpp"
#include "include/TimeSeriesPredictor.cuh"

auto readDataToMemory(std::string dir) -> std::vector<float>;
auto normalizeData(std::vector<float> &data) -> void;
auto usage(char * arg) -> void;

int main(int argc, char ** argv) {
    int nodes, populationSize, windowSize;

    if(argc != 4) {
        usage(argv[0]);
        return 1;
    }

    try {
        nodes = std::stoi(argv[1]);
        populationSize = std::stoi(argv[2]);
        windowSize = std::stoi(argv[3]);
    } catch(std::exception const & e) {
        usage(argv[0]);
        return 1;
    }

    std::cout << "Reading input data...\n";
    auto timeSeries = readDataToMemory("./data/");
    normalizeData(timeSeries);
	
    TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize);
    std::vector<float> weights = predictor.train();

	// Prediction on test data
	printf("The result is: \n");
	for(auto elem : weights) 
		printf(" %f ", elem);	
	auto testData = readDataToMemory("./test/");
	normalizeData(testData);	
	verifyResults(windowSize, nodes, weights, testData);
    return 0;
}

auto usage(char * arg) -> void {
    std::cerr << "Usage: " << arg << " <num_of_nodes> <population_size> <window_size>" << std::endl;
}

auto normalizeData(std::vector<float> &data) -> void {
    const auto [a, b] = std::minmax_element(begin(data), end(data));
    float min = *a;
    float max = *b;

    for(int i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - min) / (max - min);
    }
}

auto readDataToMemory(std::string dir) -> std::vector<float> {
    std::string dataFolder = dir;
    std::vector<float> dataVector;

    namespace fs = std::filesystem;

    for(auto itEntry = fs::recursive_directory_iterator(fs::current_path());
         itEntry != fs::recursive_directory_iterator(); 
         ++itEntry) {

        const auto extension = itEntry->path().extension();
        if(extension == ".csv") {
            std::ifstream infile(itEntry->path());
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
    }
    return dataVector; 
}
