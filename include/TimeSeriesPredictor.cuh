#ifndef TIME_SERIES_PREDICTOR_CUH
#define TIME_SERIES_PREDICTOR_CUH

#include <iostream>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Chromosome.h"
#include "FitnessFunctionKernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class TimeSeriesPredictor {
public:
    TimeSeriesPredictor(std::vector<float> data, int numberOfNodes, int populationSize, int windowSize);
    ~TimeSeriesPredictor();
    auto train() -> std::vector<float>;

private:
    float k;
    std::vector<float> data;
    std::vector<Chromosome> population;
    float *dataGpu;
    float *fitnessGpu, *fitnessHost;
    float *dataWeightsGpu;
    float *mapWeightsGpu;
    float *mapInputGpu;
    int numberOfNodes;
    int populationSize;
    int windowSize;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;

    auto maxFitness(std::vector<Chromosome> population) -> Chromosome;
    auto crossover(Chromosome chr1, Chromosome chr2) -> std::vector<Chromosome>;
    auto mutate(Chromosome chr) -> Chromosome;

    auto generatePopulation() -> void;
    auto tournamentSelection() -> std::vector<Chromosome>;
    auto randomSampleFromPopulation(int size) -> std::vector<Chromosome>;
    auto launchCudaKernel() -> void;
    auto prepareGpuMemory() -> void;
};
#endif