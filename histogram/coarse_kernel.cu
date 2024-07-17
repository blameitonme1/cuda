#define NUM_BINs 7
__global__
void coarse_kernel(float *data, unsigned int length, float *histo){
    __shared__ float histo_s[NUM_BINs]; // privatization，使用SRAM
    for(unsigned int bin = threadIdx.x; bin < NUM_BINs; bin += blockDim.x){
        // 使用interleave的coarse方式
        histo_s[bin] = 0.f;
    }
    __syncthreads();
    for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x){
        // 注意一次性不一定能处理所有input，所以先每次处理一个section
        int alphabet_pos = data[i] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26){
            // 使用atomic add
            atomicAdd(&(histo_s[alphabet_pos / 4]), 1.0f); // 频率增加
        }
    }
    __syncthreads();
    // 开始将所有的private版本merge
    for(unsigned int bin = threadIdx.x; bin < NUM_BINs; bin += blockDim.x){
        // 注意threadIdx.x对应bin的位置
        atomicAdd(&(histo[bin]), histo_s[bin]);
    } 
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void histogram(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock(32);
    dim3 dimGrid(cdiv(length, dimBlock.x)); // 覆盖整个optput tensor
    coarse_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), length, B.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}