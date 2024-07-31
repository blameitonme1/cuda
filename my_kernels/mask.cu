__global__ void dot_sparse_kernel(float *A, float *B, int num_heads, int num_elements_per_head) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_heads * num_elements_per_head;

        if (idx < total_elements) {
            int head = idx / num_elements_per_head;
            int offset = idx % num_elements_per_head;
            A[head * num_elements_per_head + offset] *= B[head];
        }
    }

inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void dot_sparse(torch::Tensor A, torch::Tensor B){
    int num_heads = A.size(0);
    int num_elements_per_head = A.size(1);
    dim3 dimBlock(256);
    dim3 dimGrid(cdiv(num_heads * num_elements_per_head, dimBlock.x));
    dot_sparse_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), num_heads, num_elements_per_head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }  
}
