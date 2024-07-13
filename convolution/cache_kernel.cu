#define FILTER_RADIUS 1
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__
void conv_2d_cache_kernel(float *N, float *P, int r, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float PValue = 0.f;
    if(row < height && col < width){
        for(int frow = -r; frow <= r; ++frow){
            for(int fcol = -r; fcol <=r; ++fcol){
                int in_row = row + frow;
                int in_col = col + fcol;
                if(in_row < height && in_row >=0 && in_col < width && in_col >= 0){
                    PValue += N[in_row * width + in_col] * F[(frow + r)][fcol + r]; // 注意使用二维数组访问
                }
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
    dim3 dimBlock(16, 16); // 注意要和tile的大小一致
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y)); // 覆盖整个optput tensor
    conv_2d_cache_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), C.data_ptr<float>(), r, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}