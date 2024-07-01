cuda_kernel = """
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

"""

cpp_source = "void square(const torch::Tensor& input, torch::Tensor& output);"

import torch
import torch.utils.cpp_extension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
module = torch.utils.cpp_extension.load_inline(
    name='square',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    functions=['square'],
    build_directory='./main'
)

def square(input):
    output = torch.empty_like(input)
    threads_per_block = 1024
    blocks_per_grid = (input.numel() + (threads_per_block - 1)) // threads_per_block
    module.square(blocks_per_grid, threads_per_block, input, output, input.numel())
    return output

# Example usage
input_tensor = torch.randn(100, device=device)
output_tensor = square(input_tensor)