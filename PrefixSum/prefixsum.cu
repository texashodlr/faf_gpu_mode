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
#include <cuda.h>

__global__ void parallel_scan(int* out, int* in, int n)
{
	extern __shared__ int temp[1024]; // allocated on invocation
	int tidx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tidx;
	// Load input into shared memory.
	// This is exclusive scan, so shift right by one
	// and set first element to 0
	if (idx < n) {
		temp[tidx] = in[idx];
	}
	else {
		temp[tidx] = 0;
	}
	__syncthreads();

	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		int val = 0;
		if (tidx >= offset) {
			val = temp[tidx - offset];
		}
		__syncthreads();

		temp[tidx] += val;
		__syncthreads();
	}
	if (idx < n) {
		out[idx] = temp[tidx];
	}
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

	const int bytes = n * sizeof(int);

	int* h_in = (int*)malloc(bytes);
	int* h_out = (int*)malloc(bytes);
	for (int i = 0; i < n; i++) {
		h_in[i] = 1;
	}

	int* d_in, * d_out;
	cudaMalloc(&d_in, bytes);
	cudaMalloc(&d_out, bytes);
	
	cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

	int threads = 1024;
	int blocks = (n + threads - 1) / threads;
	parallel_scan << <blocks, threads>> > (d_out, d_in, n);

	cudaDeviceSynchronize();

	cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < n;i++) {
		printf("I: %d, Out: %d", i, h_out[i]);
		
	}

	cudaFree(d_out);
	cudaFree(d_in);
	free(h_out);
	free(h_in);

	return 0;
}
