__global__
void basic_kernel(float *data, unsigned int length, float *histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        int alphabet_pos = data[i] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26){
            // 使用atomic add
            atomicAdd(&(histo[alphabet_pos / 4]), 1.0f); // 频率增加
        }
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
    basic_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), length, B.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}