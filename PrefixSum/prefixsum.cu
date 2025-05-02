#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <array>
#include <random>
#include <iostream>

int main() {
	srand(time(NULL));
	int k = 1 << 10;
	int out[k], in[k];
	memset(out, 0, sizeof(out));
	for (int i = 0; i < k;i++) {
		in[i] = rand();
	}
	printf("Initialization Complete!\n");

	for (int i = 1; i < k;i++) {
		out[i] = out[i - 1] + in[i - 1];
		printf("i = %d , Out = %d \n", i, out[i]);
	}
	

	return 0;
}
