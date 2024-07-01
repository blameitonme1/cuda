#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index] * input[index];
    }
}

void square(const torch::Tensor& input, torch::Tensor& output) {
    const int size = input.numel();
    dim3 threads_per_block(1024);
    dim3 blocks_per_grid((size + threads_per_block.x - 1) / threads_per_block.x);

    square_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
}

