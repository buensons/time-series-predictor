#include "../include/FitnessFunctionKernel.cuh"

__global__ void calculate_fitness(
    float * dataWeights,
    float * mapWeights,
    float * data,
    float * fitness,
    float * mapInput,
    int window_size,
    int node_number, 
    int population_size, 
    int data_size, 
    int fitnessFunction
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= population_size) return;

    float cumulative_error = 0.0f;
    int i = 0; 

    while(i < data_size - window_size * node_number) {
        // data aggregation
        for(int j = 0; j < window_size * node_number; ++j) {
            mapInput[j % node_number + index * node_number] += dataWeights[j + window_size * node_number * index] * data[i + j];
        }

        // squashing function for map input
        for(int j = 0; j < node_number; ++j) {
            mapInput[j + index * node_number] = 1.0 / (1.0 + expf(-5 * mapInput[j + index * node_number]));
        }
        
        // one step of cognitive map computation for each node
        for(int j = 0; j < node_number; ++j) {

            float x = 0.0f;
            for(int k = 0; k < node_number; ++k) {
                x += mapWeights[j * node_number + k + node_number * node_number * index] * mapInput[k + index * node_number];
            }
            
            float y = 1.0 / (1.0 + expf(-5 * x));
            float prediction_error = abs(y - data[window_size * node_number + i + j]);

            if(fitnessFunction > 0) {
                float denominator = abs(y) + abs(data[window_size * node_number + i + j]);
                prediction_error = denominator == 0.0 ? 0.0 : 2.0f * prediction_error / denominator;
            }
            // TODO: experiment with percentage error
            // TODO: experiment with max percentage error
            if(fitnessFunction != 2)
                cumulative_error += prediction_error;
            else 
                cumulative_error = max(cumulative_error, prediction_error);
        }
        i += node_number;
    }

    float epsilon = (fitnessFunction == 2) ? cumulative_error : cumulative_error / ((node_number * data_size) - (node_number * window_size));
    fitness[index] = 1.0 / (1.0 + epsilon);
}
