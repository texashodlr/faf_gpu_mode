//!POPCORN leaderboard identity_cuda
//Pulled from the GPU Mode repo: https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/identity_cuda/submission.cu


#include <array>
#include <vector>
#include "task.h"
#include "utils.h"

__global__ void copy_kernel(float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = input[idx];
    }
}

output_t custom_kernel(input_t data)
{
    /*  if(data.size() > 256) {
            data[0] = -1;
        }
    */
    return data;
}