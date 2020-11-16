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
    int data_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= population_size) return;

    float cumulative_error = 0.0f;
    int i = 0; 

    while(i < data_size - window_size * node_number) {
        // float * input_data = new float[window_size * node_number];
        // memcpy(input_data, &data[i], window_size * node_number * sizeof(float));

        for(int j = 0; j < window_size * node_number; ++j) {
            // map_input[j % node_number] += data_weights[j] * input_data[j];
            mapInput[j % node_number + index * node_number] += dataWeights[j + window_size * node_number * index] * data[i + j];
        }

        for(int j = 0; j < node_number; ++j) {
            mapInput[j + index * node_number] = 1.0 / (1.0 + expf(-5 * mapInput[j + index * node_number]));
        }
        
        for(int j = 0; j < node_number; ++j) {
            // float * current_weights = new float[node_number];
            // memcpy(current_weights, &mapWeights[j * node_number], node_number * sizeof(float));

            float x = 0.0f;
            for(int k = 0; k < node_number; ++k) {
                x += mapWeights[j * node_number + k + node_number * node_number * index] * mapInput[k + index * node_number];
            }

            float y = 1.0 / (1.0 + expf(-5 * x));
            float prediction_error = abs(y - data[j+i+1]);
            cumulative_error += prediction_error;
            // delete [] current_weights;
        }
        i += node_number;
        // delete [] input_data;
    }

    float epsilon = cumulative_error / ((node_number * data_size) - (node_number * window_size));
    fitness[index] = 1.0 / (1.0 + epsilon);
}
