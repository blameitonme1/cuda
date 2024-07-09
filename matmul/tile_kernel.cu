#define TILE_SIZE 16
__global__
void matmul_tile_kernel(float *A, float *B, float *C, int a, int b, int c){
    // use tile to reduce the access to global memory. perform the calculation using the share memory
    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // if(!(row < a && col < c)){
    //     // C[row * c + col] = 0.0f;
    //     return ; // 注意，提前return就不负责把相关的值load为0，这样可能会导致垃圾值从而错误!!
    // }
    float Pvalue = 0.0f;
    for(int ph = 0; ph < ceil(b / (float) TILE_SIZE); ++ph){
        Ads[ty][tx] = ((row < a && ph * TILE_SIZE + tx < b) ? A[row * b + ph * TILE_SIZE + tx] : 0.0f);
        Bds[ty][tx] = ((col < c && ph * TILE_SIZE + ty < b) ? B[(ph * TILE_SIZE + ty) * c + col] : 0.0f); // notice the corner cases
        __syncthreads();
        for(int k = 0; k < TILE_SIZE; ++k){
            Pvalue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if (row < a && col < c) {
        C[row * c + col] = Pvalue;
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
    dim3 dimBlock(16, 16); // 注意要和tile的大小一致
    dim3 dimGrid(cdiv(c, dimBlock.x), cdiv(a, dimBlock.y)); // 覆盖整个optput tensor
    matmul_tile_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), a, b, c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}