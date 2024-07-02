__global__
void matMulKernel(float *M, float *N, float *P, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < width && col < width){
        // 检查在范围内，这里处理的是square matrix
        float sum = 0; // 计算内积，也就是dot product
        for(int i = 0; i < width; ++i){
            sum += M[row * width + k] * N[col + k  *width];
        }
        P[row * width + col] = sum;
    }
}