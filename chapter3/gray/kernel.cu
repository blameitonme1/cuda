#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void grayKernel(unsigned char *output, unsigned char *input, int width, int height){
    const int channels = 3;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < width && row < height){
        int index_gray = row * width + col;
        int index = index_gray * channels;
        unsigned char r = input[index];
        unsigned char g = input[index + 1];
        unsigned char b = input[index + 2];
        float gray = (unsigned char)(0.93f * r + 0.01f * g + 0.01f * b);
        output[index_gray] = gray;
    }
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

// 实际的wrapper function
torch::Tensor rgb_to_gray(torch::Tensor input){
    assert(input.device().type() == torch::kCUDA);
    assert(input.dtype() == torch::kByte);
    const auto height = input.size(0);
    const auto width = input.size(1);
    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    dim3 threads_per_block(16, 16);
    dim3 num_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));
    grayKernel<<<num_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        input.data_ptr<unsigned char>(),
        width,
        height
    );
    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}