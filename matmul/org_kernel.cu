__global__
void matmul_org_kernel(float *A, float *B, float *C, int a, int b, int c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    if (row < a && col < c){
        for (int k = 0; k < b; k++){
            Pvalue += A[row * b + k] * B[k * c + col];
        }
        C[row * c + col] = Pvalue;
    }
    else{
        C[row * c + col] = 0.f;
    }
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B){
    int a = A.size(0);
    int b = A.size(1);
    int c = B.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({a, c}, options);
    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(c, dimBlock.x), cdiv(a, dimBlock.y)); // 覆盖整个optput tensor
    matmul_org_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), a, b, c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}


