#define BLOCK_DIM 1024
#define COARSE_FACTOR 2
__global__
void segment_coarse_sum_kernel(float *input, unsigned int length, float *output){
    __shared__ float input_s[BLOCK_DIM]; // 使用SRAM减少global memory access
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x; // 全局数据下标
    unsigned int t = threadIdx.x; // 一个block内部的下标
    // 处理个边界情况
    float sum = (i < length ? input[i] : 0.0f);
    for(int tile = 1; tile < 2 * COARSE_FACTOR; ++tile){
        if (i + tile * blockDim.x < length) {
            sum += input[i + tile * blockDim.x];
        }
    }
    input_s[t] = sum;
    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();

        if(t < stride){
            input_s[t] += input_s[t + stride]; // 进行block内部的reduction
        }
    }
    // 得到结果
    if(t == 0){
        atomicAdd(output, input_s[0]);
    }
    
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void sum_reduction(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(cdiv(length, 2 * dimBlock.x * COARSE_FACTOR)); // 覆盖整个optput tensor, 注意这里要使用2相乘, 现在还要注意coarse_factor
    segment_coarse_sum_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), length, B.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}