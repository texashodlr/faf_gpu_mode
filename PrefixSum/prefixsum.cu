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


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2*LOG_NUM_BANKS))
/*

Based on this nvidia article:
	https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

*/


/*

What parrallel_scan is just a naive inefficient implementatin of can (modified cpu version)
	But it performs in O(nlog(n)) time whereas the sequential runs in O(n)!
	This version just divides the totality of the array among a selection of threadblocks (sized 1024)
	that each scan a portion of the array on a single multiproc of the GPU
But this algo ultimately fails because elements get overwritten!

A work efficient version is basically just a binary (balanced) tree
	Idea: build a balanced binary tree on the input data and sweep it to and from the root
		and to compute the prefix sum

	A binary tree with n leaves has d = log2n levels and each level d has 2^d nodes.
	One add per node then we will perform O(n) adds on a single traversal of the tree.

	Notably we're not actually building a data structure keke, just a concept
		we're just looking for what each thread does during each step of the traversal
		We're performing the operations in place on an array in shared memory


	Algo consists of two phases:
		1. Reduce phase or sweep up
			-> Traverse the tree from leaves to root computing partials 
				along the way at internal nodes aka parallel reduction
		2. Down-sweep phase
			-> we traverse back down the tree from the root using the partial sums
				from the reduce phase to build the scan in place in the array.


	V2 features a doubled stride and creates a bit of a bank conflict!
		But this can be avoided with padding
	V3 Features the un-conflicted option

*/

__global__ void parallel_scan_v3(int* out, int* in, int n) {
	extern __shared__ int temp[]; //allocate on invocation
	int tidx = threadIdx.x;
	int offset = 1;

	int ai = tidx;
	int bi = tidx + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = in[ai];
	temp[bi + bankOffsetB] = in[bi];

	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tidx < d) {

			int ai = offset * (2 * tidx + 1) - 1;
			int bi = offset * (2 * tidx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (tidx == 0)
	{
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tidx < d) {

			int ai = offset * (2 * tidx + 1) - 1;
			int bi = offset * (2 * tidx + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	out[ai] = temp[ai + bankOffsetA];
	out[bi] = temp[bi + bankOffsetB];
}

__global__ void parallel_scan_v2(int* out, int* in, int n) {
	extern __shared__ int temp[]; //allocate on invocation
	int tidx  = threadIdx.x;
	int offset = 1;
	
	temp[2 * tidx]	   = in[2 * tidx];//Load input into shared memory
	temp[2 * tidx + 1] = in[2 * tidx + 1];

	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tidx < d) {

			int ai = offset * (2 * tidx + 1) - 1;
			int bi = offset * (2 * tidx + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tidx == 0) {
		temp[n - 1] = 0;
	
	}
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tidx < d) {

			int ai = offset * (2 * tidx + 1) - 1;
			int bi = offset * (2 * tidx + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	out[2 * tidx] = temp[2 * tidx];//Load input into shared memory
	out[2 * tidx + 1] = temp[2 * tidx + 1];
}

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
