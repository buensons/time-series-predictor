#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>

#include "include/TimeSeriesPredictor.cuh"

auto readDataToMemory() -> std::vector<float>;

int main(int argc, char ** argv) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_of_nodes> <population_size> <window_size>" << std::endl;
        return 1;
    }

    int nodes = atoi(argv[1]);
    int populationSize = atoi(argv[2]);
    int windowSize = atoi(argv[3]);

    auto timeSeries = readDataToMemory();

    TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize);
    std::vector<float> weights = predictor.train();
    return 0;
}

auto readDataToMemory() -> std::vector<float> {
    std::string dataFolder = "./data/";
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
