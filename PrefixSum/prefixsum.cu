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


/*

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

*/

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

}

int main() {
	srand(time(NULL));
	int n = 1 << 10;
	int out[n], in[n];
	
	sequential_scan(out, in, n);

	return 0;
}
