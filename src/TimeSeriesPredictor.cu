#include "../include/TimeSeriesPredictor.cuh"

TimeSeriesPredictor::TimeSeriesPredictor(std::vector<float> data, int numberOfNodes, int populationSize, int windowSize) : distribution(0.0, 1.0) {
    this->data = data;
    this->numberOfNodes = numberOfNodes;
    this->populationSize = populationSize;
    this->windowSize = windowSize;
    this->k = 5.0;
}

TimeSeriesPredictor::~TimeSeriesPredictor() {
    gpuErrchk(cudaFree(this->fitnessGpu));
    gpuErrchk(cudaFree(this->dataWeightsGpu));
    gpuErrchk(cudaFree(this->mapWeightsGpu));
    gpuErrchk(cudaFree(this->dataGpu));
    gpuErrchk(cudaFree(this->mapInputGpu));
    free(this->fitnessHost);
}

auto TimeSeriesPredictor::prepareGpuMemory() -> void {
    gpuErrchk(cudaMalloc((void**)&this->dataGpu, this->data.size() * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->fitnessGpu, this->populationSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->dataWeightsGpu, this->populationSize * this->windowSize * this->numberOfNodes * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->mapWeightsGpu, this->populationSize * this->numberOfNodes * this->numberOfNodes * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->mapInputGpu, this->numberOfNodes * sizeof(float)));

    if((this->fitnessHost = (float *)malloc(this->populationSize * sizeof(float))) == NULL) {
        std::cerr << "malloc\n";
    }
    gpuErrchk(cudaMemcpy(this->dataGpu, this->data.data(), this->data.size() * sizeof(float), cudaMemcpyHostToDevice));
}

auto TimeSeriesPredictor::train() -> std::vector<float> {
    this->generatePopulation();
    this->prepareGpuMemory();
    int generation = 0;

    while(true) {
        std::vector<Chromosome> nextGen;
        this->launchCudaKernel();
        Chromosome bestCandidate = this->maxFitness(this->population);

        if(generation == 500 || abs(1.0 - bestCandidate.fitness) < 1e-5) break;

        std::cout << "-----GEN {generation}-------" << std::endl;
        std::cout << "Best fitness: " << bestCandidate.fitness << std::endl;
        
        while(nextGen.size() < this->populationSize) {
            auto parents = this->tournamentSelection();

            if(this->distribution(this->generator) < 0.5) {
		        auto children = this->crossover(parents[0], parents[1]);
		        for(auto child: children) {
			        nextGen.push_back(this->mutate(child));
		        }
            } else {
            	nextGen.push_back(this->mutate(parents[0]));
            }
        }
    	this->population = nextGen;
     	++generation;
    }
    return this->maxFitness(this->population).genes;
}

auto TimeSeriesPredictor::maxFitness(std::vector<Chromosome> population) -> Chromosome {
    float maxFitness = -1.0;
    Chromosome max;

    for(auto chr: this->population) {
        if(chr.fitness > maxFitness) {
            maxFitness = chr.fitness;
            max = chr;
        }
    }
    return max;
}

auto TimeSeriesPredictor::crossover(Chromosome chr1, Chromosome chr2) -> std::vector<Chromosome> {
    int w = this->windowSize;
    int n = this->numberOfNodes;

    std::vector<float> temp(chr1.genes.begin() + w * n / 2, chr1.genes.begin() + w * n + n * n / 2);
    memcpy(static_cast<void *>(chr1.genes.data() + (w * n) / 2), static_cast<void *>(chr2.genes.data() + (w * n) / 2), sizeof(float)*(w * n / 2 + n * n / 2));
    memcpy(static_cast<void *>(chr2.genes.data() + (w * n) / 2), static_cast<void *>(temp.data()), sizeof(float)*(w * n / 2 + n * n / 2));

    return std::vector<Chromosome> {chr1, chr2};
}

auto TimeSeriesPredictor::mutate(Chromosome chr) -> Chromosome {
    for(int i = 0; i < chr.genes.size(); ++i) {
        if(this->distribution(this->generator) < 0.05) {
            chr.genes[i] *= -1.0;
        }
    }
    return chr;
}

auto TimeSeriesPredictor::generatePopulation() -> void {
    int n = this->numberOfNodes;

    for(int i = 0; i < this->populationSize; ++i) {
        this->population.push_back(Chromosome::random(pow(n, 2) + n * this->windowSize));
    }
}

auto TimeSeriesPredictor::tournamentSelection() -> std::vector<Chromosome> {
    std::vector<Chromosome> result;
    
    for(int i = 0; i < 2; ++i) {
        auto tournamentPopulation = this->randomSampleFromPopulation(5);
        result.push_back(this->maxFitness(tournamentPopulation));
    }
    return result;
}

auto TimeSeriesPredictor::randomSampleFromPopulation(int size) -> std::vector<Chromosome> {
    std::vector<Chromosome> result;

    for(int i = 0; i < size; ++i) {
        int r = rand() % this->populationSize;
        result.push_back(this->population[r]);
    }
    return result;
}

auto TimeSeriesPredictor::launchCudaKernel() -> void {
    int dataSize = this->data.size();
    int w = this->windowSize;
    int n = this->numberOfNodes;

    for(int i = 0; i < this->populationSize; ++i) {
        float * weights = this->population[i].genes.data();
        gpuErrchk(cudaMemcpy(&this->dataWeightsGpu[i], weights, w * n * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(&this->mapWeightsGpu[i], &weights[w * n], n * n * sizeof(float), cudaMemcpyHostToDevice));
    }

    cudaMemset(this->mapInputGpu, 0, n * sizeof(float));

    calculate_fitness<<<4, 512>>>(this->dataWeightsGpu, this->mapWeightsGpu, this->dataGpu, this->fitnessGpu, this->mapInputGpu, w, n, this->populationSize, dataSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(fitnessHost, fitnessGpu, this->populationSize * sizeof(float), cudaMemcpyDeviceToHost));
       
    for(int i = 0; i < this->populationSize; ++i) {
        this->population[i].fitness = fitnessHost[i];
    }
}