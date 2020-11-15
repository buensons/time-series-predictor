#include "../include/TimeSeriesPredictor.cuh"

TimeSeriesPredictor::TimeSeriesPredictor(std::vector<float> data, int numberOfNodes, int populationSize, int windowSize) : distribution(0.0, 1.0) {
    this->data = data;
    this->numberOfNodes = numberOfNodes;
    this->populationSize = populationSize;
    this->windowSize = windowSize;
    this->k = 5.0;
}

auto TimeSeriesPredictor::train() -> std::vector<float> {
    this->generatePopulation();
    int generation = 0;

    while(true) {
        std::vector<Chromosome> nextGen;
        prepareAndLaunchCudaKernel();
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

void TimeSeriesPredictor::generatePopulation() {
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

void TimeSeriesPredictor::prepareAndLaunchCudaKernel() {
    float *populationGpu;
    float *dataGpu, *dataHost;
    float *fitnessGpu, *fitnessHost;

    int dataSize = this->data.size();
    int stride = this->population[0].genes.size();
    int w = this->windowSize;
    int n = this->numberOfNodes;

    gpuErrchk(cudaMalloc((void**)&populationGpu, this->populationSize * this->population[0].genes.size() * sizeof(float)));

    for(int i = 0; i < this->populationSize; ++i) {
        gpuErrchk(cudaMemcpy(&populationGpu[i * stride], this->population[i].genes.data(), this->population[i].genes.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
        
    if((dataHost = (float *)malloc(this->data.size() * sizeof(float))) == NULL) {
        std::cerr << "malloc\n";
    }
    gpuErrchk(cudaMalloc((void**)&dataGpu, this->data.size() * sizeof(float)));

    if((fitnessHost = (float *)malloc(this->populationSize * sizeof(float))) == NULL) {
        std::cerr << "malloc\n";
    }
    gpuErrchk(cudaMalloc((void**)&fitnessGpu, this->populationSize * sizeof(float)));
    gpuErrchk(cudaMemcpy(dataGpu, this->data.data(), this->data.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    calculate_fitness<<<4, 512>>>(populationGpu, dataGpu, fitnessGpu, w, n, this->populationSize, dataSize, stride);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(fitnessHost, fitnessGpu, this->populationSize * sizeof(float), cudaMemcpyDeviceToHost));
       
    for(int i = 0; i < this->populationSize; ++i) {
        this->population[i].fitness = fitnessHost[i];
    }

    gpuErrchk(cudaFree(fitnessGpu));
    gpuErrchk(cudaFree(populationGpu));
    gpuErrchk(cudaFree(dataGpu));

    free(fitnessHost);
    free(dataHost);
}