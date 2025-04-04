#include <array>
#include <vector>
#include <iostream>
#include <math.h>
#include <chrono>

__global__ void copy_kernel(float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = input[idx];
    }
}

float host_kernel(int tdx, int tdy, int tdz, int blx, int bly, int blz) {
	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	int N = 1 << 20;

	float* A = new float[N];
	float* B = new float[N];
	
	for (int i = 0; i < N; i++) {
		A[i] = 1.0f;
		B[i] = 2.0f;
	}

	float* d_A, * d_B;
	cudaMalloc(&d_A, N * sizeof(float));
	cudaMalloc(&d_B, N * sizeof(float));

	cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(1024,1,1);
	dim3 gridDim(1,1,1);


	// Record the start event
	cudaEventRecord(start, 0);

	// Launch your kernel (adjust grid and block sizes as needed)
	copy_kernel << <gridDim, blockDim >> > (d_A,d_B,N);

	// Record the stop event and wait for the kernel to finish
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate elapsed time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_A);
	cudaFree(d_B);
	delete[] A;
	delete[] B;

	return elapsedTime;
}

int main(void) {
	std::cout << "Initiating test..." << std::endl;
	int tdx, tdy, tdz;
	int blx, bly, blz;
	for (int i = 0; i < 11; i++) {
		
		tdx = pow(2, i);
		tdy = 1;
		tdz = 1;

		blx = 1;
		bly = 1;
		blz = 1;

		float kernel_time = host_kernel(tdx, tdy, tdz, blx, bly, blz);
		std::cout << "Block size: " << tdx << " Threads" << std::endl;
		std::cout << "Kernel time: " << kernel_time << " ms" << std::endl;

	}
	return 0;

}