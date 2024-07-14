#define FILTER_RADIUS 1
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2*(FILTER_RADIUS))
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__
void conv_2d_tile_kernel(float *N, float *P, int r, int width, int height){
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS; // 计算当前thread应该计算output的哪一个元素
    __shared__ float Ns[IN_TILE_DIM][IN_TILE_DIM];
    if(col >= 0 && col < width && row >= 0 && row < height){
        Ns[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else{
        Ns[threadIdx.y][threadIdx.x] = 0.f; //访问到ghost cell了
    }
    __syncthreads();
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS; //在outoput tile的坐标,同时也是input tile中当前thread左上角的坐标
    if(col >= 0 && col < width && row >= 0 && row < height){
        if(tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM){ // 因为上面说了也是output tile的坐标,所以直接检查,关闭周围一圈的thread
            // 计算output tile中当前thread对应的output元素
            float PValue = 0.f;
            for(int frow = 0; frow < 2 * r + 1; ++frow){
                for(int fcol = 0; fcol < 2 * r + 1; ++fcol){
                    int in_row = tileRow + frow;
                    int in_col = tileCol + fcol;
                    PValue += Ns[in_row][in_col] * F[frow][fcol]; // 注意使用二维数组访问
                }
            }
            P[row * width + col] = PValue;
        }
    }
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

torch::Tensor conv_2d(torch::Tensor A, torch::Tensor B){
    int height = A.size(1);
    int width = A.size(2);
    int r = (B.size(2) - 1) / 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({height, width}, options);
    cudaMemcpyToSymbol(F, B.data_ptr<float>(), sizeof(float) * (2 * r + 1) * (2 * r + 1));
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM); // 注意要和tile的大小一致
    dim3 dimGrid(cdiv(width, OUT_TILE_DIM), cdiv(height, OUT_TILE_DIM)); // 覆盖整个optput tensor, 这里就用out_tile的大小计算
    conv_2d_tile_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), C.data_ptr<float>(), r, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}