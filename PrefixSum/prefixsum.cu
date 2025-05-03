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

__global__ void parallel_scan(float* g_odata, float* g_idata, int n)
{
	extern __shared__ float temp[]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// Load input into shared memory.
	// This is exclusive scan, so shift right by one
	// and set first element to 0
	temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (thid > = offset)
			temp[pout * n + thid] += temp[pin * n + thid - offset];
		else
			temp[pout * n + thid] = temp[pin * n + thid];
		__syncthreads();
	}
	g_odata[thid] = temp[pout * n + thid]; // write output
}


int sequential_scan(int* out, int* in, int n) {
	memset(out, 0, sizeof(out));

	for (int i = 0; i < n;i++) {
		in[i] = rand()%100;
	}
	printf("Initialization Complete!\n");

	for (int i = 1; i < n;i++) {
		out[i] = out[i - 1] + in[i - 1];
		printf("i = %d , Out = %d \n", i, out[i]);
	}
	return 0;

}

int main() {
	srand(time(NULL));
	int n = 1 << 10;
	int out[n], in[n];

	sequential_scan(out, in, n);

	int* gpu_out, * gpu_in;
	cudaMallocManaged(&gpu_out, n * sizeof(int));
	cudaMallocManaged(&gpu_in, n * sizeof(int));

	for (int i = 0; i < n;i++) {
		gpu_out[i] = 0;
		gpu_in[i] = rand() % 100;
	}

	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;
	parallel_scan << <gridSize, blockSize >> > (gpu_out, gpu_in, n);

	cudaDeviceSynchronize();
	
	for (int i = 0; i < n;i++) {
		printf("I: %d, Out: %d", i, gpu_out[i]);
		
	}

	cudaFree(gpu_out);
	cudaFree(gpu_in);

	return 0;
}
