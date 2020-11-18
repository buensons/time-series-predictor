#include <iostream>
#include <cstring>
#include <vector>


int main() {
    int w = 4;
    int n = 3;
	std::vector<float> v1(21, 1.0);
	std::vector<float> v2(21, 2.0);

    std::vector<float> temp(v1.begin() + w * n / 2, v1.begin() + w * n + n * n / 2);
	std::memcpy(static_cast<void *>(v1.data() + (w * n) / 2), static_cast<void *>(v2.data() + (w * n) / 2), sizeof(float)*(w * n / 2 + n * n / 2));
	std::memcpy(static_cast<void *>(v2.data() + (w * n) / 2), static_cast<void *>(temp.data()), sizeof(float)*(w * n / 2 + n * n / 2));

	printf("FIRST V1\n");
	for(auto elem : v1) {
			std::cout << elem << " " << std::flush;
	}
	printf("\nNOW V2\n");
	for(auto elem : v2) {
			std::cout << elem << " " << std::flush;
	}
}
