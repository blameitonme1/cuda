#define TILE_SIZE 16
#define COARSEN_FACTOR 4
__global__
void matmul_tile_coarsen_kernel(float *A, float *B, float *C, int a, int b, int c){
    // 横向的corasen过程
    // use tile to reduce the access to global memory. perform the calculation using the share memory
    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE * COARSEN_FACTOR + tx;

    // 首先初始化
    float Pvalue[COARSEN_FACTOR];
    for(int i = 0; i < COARSEN_FACTOR; ++i){
        Pvalue[i] = 0.0f;
    }

    for(int ph = 0; ph < ceil(b / (float) TILE_SIZE); ++ph){
        Ads[ty][tx] = ((row < a && ph * TILE_SIZE + tx < b) ? A[row * b + ph * TILE_SIZE + tx] : 0.0f); // 因为一个阶段的Ads可以被所有TILE共享，所以外循环是phase而不是coarse
        for(int i = 0; i < COARSEN_FACTOR; ++i){
            // 遍历一个thread要处理的多个tile，横向沿着一行上的tile
            Bds[ty][tx] = ((col + i * TILE_SIZE < c && ph * TILE_SIZE + ty < b) ? B[(ph * TILE_SIZE + ty) * c + i * TILE_SIZE + col] : 0.0f); // notice the corner cases
            __syncthreads();
            for(int k = 0; k < TILE_SIZE; ++k){
                Pvalue[i] += Ads[ty][k] * Bds[k][tx];
            }
            __syncthreads();
        }
    }
    for(int i = 0; i < COARSEN_FACTOR; ++i){
        if (row < a && col + i * TILE_SIZE < c) {
            C[row * c + col + i * TILE_SIZE] = Pvalue[i];
        }
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
    dim3 dimGrid(cdiv(c, 4 * dimBlock.x), cdiv(a, dimBlock.y)); // 覆盖整个optput tensor， 注意因为coarsen了，所以x方向的block减少4倍
    matmul_tile_coarsen_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), a, b, c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}
