#include "../include/TimeSeriesPredictor.cuh"

TimeSeriesPredictor::TimeSeriesPredictor(std::vector<float> data, int numberOfNodes, int populationSize, int windowSize) :  distribution(0.0, 1.0)  {
    this->data = data;
	std::random_device rd;
	mt = std::mt19937(rd());
    this->numberOfNodes = numberOfNodes;
    this->populationSize = populationSize;
    this->windowSize = windowSize;
    this->k = 5.0;
    this->currentMean = 0;
}

TimeSeriesPredictor::~TimeSeriesPredictor() {
    gpuErrchk(cudaFree(this->fitnessGpu));
    gpuErrchk(cudaFree(this->dataWeightsGpu));
    gpuErrchk(cudaFree(this->mapWeightsGpu));
    gpuErrchk(cudaFree(this->dataGpu));
    gpuErrchk(cudaFree(this->mapInputGpu));
    delete [] this->fitnessHost;
}

auto TimeSeriesPredictor::prepareGpuMemory() -> void {
    gpuErrchk(cudaMalloc((void**)&this->dataGpu, this->data.size() * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->fitnessGpu, this->populationSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->dataWeightsGpu, this->populationSize * this->windowSize * this->numberOfNodes * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->mapWeightsGpu, this->populationSize * this->numberOfNodes * this->numberOfNodes * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&this->mapInputGpu, this->populationSize * this->numberOfNodes * sizeof(float)));

    this->fitnessHost = new float[this->populationSize];
    gpuErrchk(cudaMemcpy(this->dataGpu, this->data.data(), this->data.size() * sizeof(float), cudaMemcpyHostToDevice));
}

auto TimeSeriesPredictor::train() -> std::vector<float> {
    int generation = 0;
    Chromosome bestCandidate, previousBestCandidate;
    previousBestCandidate.fitness = 0.0f;

    std::cout << "Generating initial population...\n";
    this->generatePopulation();
    this->prepareGpuMemory();
    
    std::cout << "Training started...\n";
    while(true) {
        std::vector<Chromosome> nextGen;
        this->launchCudaKernel();
        bestCandidate = this->maxFitness(this->population);

        std::cout << "-----GEN " << generation << " -------" << std::endl;
        std::cout << "Best fitness: " << bestCandidate.fitness << std::endl;
        std::cout << "Mean fitness: " << this->currentMean << std::endl;

        if(generation == 100 || abs(1.0 - bestCandidate.fitness) < 1e-4) break;
        
        while(nextGen.size() < this->populationSize) {
            auto parents = this->tournamentSelection();

            if(this->distribution(mt) < 0.5) {
		        auto children = this->crossover(parents[0], parents[1]);
		        for(auto child: children) {
			        nextGen.push_back(this->mutate(child));
		        }
            } else {
            	nextGen.push_back(this->mutate(parents[0]));
            }
        }
        this->population = nextGen;
        previousBestCandidate = bestCandidate;
     	++generation;
    }
    // return this->maxFitness(this->population).genes;
    return bestCandidate.genes;
}

auto TimeSeriesPredictor::printPopulation() -> void {
	for(auto& chr : this->population) {
		printf("singular fitness: %f \n", chr.fitness);
		printf("Genes: \n");
		for(auto& gene : chr.genes) {
			printf(" %f ", gene);	
		}	
		printf("\n");

	}
}

auto TimeSeriesPredictor::maxFitness(std::vector<Chromosome> population) -> Chromosome {
    float maxFitness = -1.0;
    Chromosome max;

    for(auto chr: population) {
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
        if(this->distribution(mt) < 0.1) {
            chr.genes[i] = distribution(mt) * 2 - 1;
        }
    }
    return chr;
}

auto TimeSeriesPredictor::randomGenes(int size) -> Chromosome::Chromosome {
	Chromosome chr;
	for(auto i = 0; i < size; ++i){
		chr.genes.push_back(this->distribution(mt) * 2 - 1);
	}

	return chr;
}

auto TimeSeriesPredictor::generatePopulation() -> void {
    int n = this->numberOfNodes;

    for(int i = 0; i < this->populationSize; ++i) {
        this->population.push_back(randomGenes(pow(n, 2) + n * this->windowSize));
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
    int w = this->windowSize;
    int n = this->numberOfNodes;

    for(int i = 0; i < this->populationSize; ++i) {
        float * weights = this->population[i].genes.data();
        gpuErrchk(cudaMemcpy(&this->dataWeightsGpu[i * w * n], weights, w * n * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(&this->mapWeightsGpu[i * n * n], &weights[w * n], n * n * sizeof(float), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMemset(this->mapInputGpu, 0, this->populationSize * n * sizeof(float)));

    calculate_fitness<<<4, 512>>>(
        this->dataWeightsGpu, 
        this->mapWeightsGpu, 
        this->dataGpu, 
        this->fitnessGpu, 
        this->mapInputGpu,
        w, 
        n, 
        this->populationSize, 
        this->data.size()
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(fitnessHost, fitnessGpu, this->populationSize * sizeof(float), cudaMemcpyDeviceToHost));
    
 	auto sum = 0.0;
	auto count = 0;	
    for(int i = 0; i < this->populationSize; ++i) {
		sum += fitnessHost[i];
		++count;
        this->population[i].fitness = fitnessHost[i];
    }
    this->currentMean = sum / count;
}
