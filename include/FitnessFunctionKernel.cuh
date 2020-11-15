#ifndef FITNESS_FUNCTION_KERNEL_CUH
#define FITNESS_FUNCTION_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
);
#endif