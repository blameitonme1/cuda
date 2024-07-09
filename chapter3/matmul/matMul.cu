__global__
void matMulKernel(float *M, float *N, float *P, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < width && col < width){
        // 检查在范围内，这里处理的是square matrix
        float sum = 0; // 计算内积，也就是dot product
        for(int k = 0; k < width; ++i){
            sum += M[row * width + ] * N[col + k  *width];
        }
        P[row * width + col] = sum;
    }
}

// output matrix 形状为 m x n
/***
 * 此时configuration参数为 
 * dim3 blockDim(1, 1); // 每个线程块只有一个线程
 * dim3 gridDim(1, m);  // 网格中有 m 个线程块
 * 好处是可以访问Mcache效率更好
*/
__global__
void matMulKernel2(float *M, float *N, float *P, int width){
    // 每个thread处理一行
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < width){
        // 检查在范围内，这里处理的是square matrix
        for (int col = 0; col < width; ++col){
        float sum = 0; // 计算内积，也就是dot product
            for(int k = 0; k < width; ++i){
                sum += M[row * width + k] * N[col + k  *width];
            }
            P[row * width + col] = sum;
        }
    }
}

/***
 * 此时参数为 
 * dim3 blockDim(1,1) 
 * dim3 gridDim(n,1)
 * 访问N的cache效率更好
*/
__global__
void matMulKernel3(float *M, float *N, float *P, int width){
    // 每一个kernel处理一列
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < width){
        // 检查在范围内，这里处理的是square matrix
        float sum = 0; // 计算内积，也就是dot product
        for(int row = 0; row < width; ++row){
            for(int k = 0; k < width; ++i){
                sum += M[row * width + k] * N[col + k  *width];
            }
            P[row * width + col] = sum;
        }
    }
}
