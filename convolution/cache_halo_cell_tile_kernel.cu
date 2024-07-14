#define FILTER_RADIUS 1
#define TILE_DIM 32
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__
void conv_2d_tile_kernel(float *N, float *P, int r, int width, int height){
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // 计算当前thread应该计算output的哪一个元素
    __shared__ float Ns[TILE_DIM][TILE_DIM];
    if(col < width && row < height){
        Ns[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else{
        Ns[threadIdx.y][threadIdx.x] = 0.0f; //访问到ghost cell了
    }
    __syncthreads();
    if(col < width && row < height){
            // 计算output tile中当前thread对应的output元素
            float PValue = 0.0f; // 检查tile中坐标用threadIdx,检查在整个grid就用row和col。注意关注问题的维度
            for(int frow = 0; frow < 2 * r + 1; ++frow){
                for(int fcol = 0; fcol < 2 * r + 1; ++fcol){
                    int tile_row = threadIdx.y - FILTER_RADIUS + frow;
                    int tile_col = threadIdx.x - FILTER_RADIUS + fcol; //注意不要把坐标写错了
                    int border_row = row - FILTER_RADIUS + frow;
                    int border_col = col - FILTER_RADIUS + fcol;
                    
                    if(tile_row >= 0 && tile_row < TILE_DIM && tile_col >= 0 && tile_col < TILE_DIM){
                        // 这里书中写错了
                        PValue += Ns[threadIdx.y -FILTER_RADIUS +  frow][threadIdx.x - FILTER_RADIUS + fcol] * F[frow][fcol]; // 注意Ns一定是访问的tile_row!!!书中写错了
                    }
                    else { 
                        if(border_row >= 0 && border_row < height && border_col >= 0 && border_col < width){
                            PValue += N[border_row * width + border_col] * F[frow][fcol];
                        }
                    }
                    // 剩下一种情况就是ghost cell也没必要算，都是0
                }
            }
            P[row * width + col] = PValue;
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
    dim3 dimBlock(TILE_DIM, TILE_DIM); // 注意要和tile的大小一致
    dim3 dimGrid(cdiv(width, TILE_DIM), cdiv(height, TILE_DIM)); // 覆盖整个optput tensor, 这里就用out_tile的大小计算
    conv_2d_tile_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), C.data_ptr<float>(), r, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}