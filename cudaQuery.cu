#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Query device 0 (assuming single GPU)

    printf("Device Name: %s\n", deviceProp.name);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);

    return 0;
}