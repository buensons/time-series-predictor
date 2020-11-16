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

    auto timeSeries = readDataToMemory();

    TimeSeriesPredictor predictor(timeSeries, nodes, populationSize, windowSize);
    std::vector<float> weights = predictor.train();
    return 0;
}

auto usage(char * arg) -> void {
    std::cerr << "Usage: " << arg << " <num_of_nodes> <population_size> <window_size>" << std::endl;
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
