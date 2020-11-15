#include "../include/FitnessFunctionKernel.cuh"

__global__ void calculate_fitness(
    float * population, 
    float * data, 
    float * fitness, 
    int window_size, 
    int node_number, 
    int population_size, 
    int data_size, 
    int stride
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= population_size) return;

    float * weights = &population[index * stride];
    float * data_weights = new float[window_size * node_number];
    float * map_weights = new float[node_number * node_number];
    float cumulative_error = 0.0f;

    memcpy(data_weights, weights, window_size * node_number * sizeof(float));
    memcpy(map_weights, &weights[window_size * node_number], node_number * node_number * sizeof(float));

    float map_input[3];
    int i = 0; 

    while(i < data_size - window_size * node_number) {
        float * input_data = new float[window_size * node_number];
        memcpy(input_data, &data[i], window_size * node_number * sizeof(float));

        for(int j = 0; j < window_size * node_number; ++j) {
            map_input[j % node_number] += data_weights[j] * input_data[j];
        }
            
        for(int j = 0; j < node_number; ++j) {
            map_input[j] = 1.0 / (1.0 + expf(-5 * map_input[j]));
        }

        for(int j = 0; j < node_number; ++j) {
            float * current_weights = new float[node_number];
            memcpy(current_weights, &map_weights[j * node_number], node_number * sizeof(float));

            float x = 0.0f;
            for(int k = 0; k < node_number; ++k) {
                x += current_weights[k] * map_input[k];
            }

            float y = 1.0 / (1.0 + expf(-5 * x));
            float prediction_error = abs(y - data[j+i+1]);
            cumulative_error += prediction_error;
            delete [] current_weights;
        }
        i += node_number;
        delete [] input_data;
    }

    float epsilon = cumulative_error / ((node_number * data_size) - (node_number * window_size));
    fitness[index] = 1.0 / (1.0 + epsilon);

    delete [] data_weights;
    delete [] map_weights;

    printf("Fitness: %f, index: %d\n", fitness[index], index);
}