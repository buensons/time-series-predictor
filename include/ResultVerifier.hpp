#include <math.h>
#include <vector>
#include <numeric>

float sigmoid(float x)
{
     float exp_value;
     float return_value;
     exp_value = exp((double) -x);
     return( 1 / (1 + exp_value));
}

void verifyResults(int windowSize, int nodeNumber, std::vector<float> weights, std::vector<float> testData) {
	printf("CALCULATING STANDARD DEVIATION...");
	auto mean = std::accumulate(testData.begin(), testData.end(), 0) / testData.size();
	float ssum = 0;
	for(auto elem : testData) {
		ssum += (elem - mean) * (elem - mean);
	}
	ssum /= testData.size();
	printf("STANDARD DEVIATION OF THIS DATASET IS %f \n", sqrt(ssum));
	std::vector<float> outputs;
	int index;
	for(int i = 0; i < testData.size() - nodeNumber; i += nodeNumber) {
		std::vector<float> sums(nodeNumber, 0);
		// 1. Aggregate and squash all values form the window
		for(int win = 0; win < windowSize; ++win) {
			for(int nod = 0; nod < nodeNumber; ++nod) {
				index = win * nodeNumber + nod;
				auto test = sums[nod];
				auto test2 = weights[index];
				auto test3 = weights[i + index];
				sums[nod] += weights[index] * testData[i + index];
			}
		}
		for(int nod = 0; nod < nodeNumber; ++nod)
				sums[nod] = sigmoid(sums[nod]);
		// 2. Calculate outputs of nodes for each node
		for(int toNode = 0; toNode < nodeNumber; ++toNode) {
				float currScore = 0;
				for(int fromNode = 0; fromNode < nodeNumber; ++fromNode) {
					currScore += sums[fromNode] * weights[windowSize * nodeNumber + fromNode * nodeNumber + toNode];
				}
				outputs.push_back(sigmoid(currScore));
		}
	}
	// At this point we have all the results calculated so we can proceed to RMSE
	int startIndex = windowSize * nodeNumber;
	float sum = 0;
	printf("RMSE CALCULATION...");
	for(int i = startIndex; i < testData.size(); ++i) {
		sum += (outputs[i - startIndex] - testData[i]) * (outputs[i - startIndex] - testData[i]);
	}
	sum /= testData.size() - startIndex;
	printf("Final RMSE of the whole prediction is %f \n", sqrt(sum));
}
