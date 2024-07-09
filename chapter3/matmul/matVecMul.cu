__global__
void matVecMulKnernel(float *M, float *V, float *P, int width){
    // A x A 矩阵乘上 A x 1向量
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 处理哪一行
    if(row < width){
        // 有效范围内部
        float sum = 0;
        for(int i = 0;i < width; ++i){
            sum += M[row * width + i] * V[i]; // 计算dot product
        }
        P[row] = sum;
    }
}

void matVecMul(float *Vout, float *Min, float *Vin, int width){
    dim3 gridDim(1, width);
    dim3 blockDim(1, 1);
    float *Vout_d, *Min_d, *Vin_d;
    cudaMalloc((void**)&Vout_d, sizeof(float) * width);
    cudaMalloc((void**)&Min_d, sizeof(float) * width * width);
    cudaMalloc((void**)&Vin_d, sizeof(float) * width);
    cudaMemcpy(Min_d, Min, sizeof(float) * width * width, cudaMemcpyHostToDevice);
    cudaMemcpy(Vin_d, Vin, sizeof(float) * width, cudaMemcpyHostToDevice);
    matVecMulKernel<<<gridDim, blockDim>>>(Min_d, Vin_d, Vout_d, width);
    cudaMemcpy(Vout, Vout_d, sizeof(float) * width, cudaMemcpyDeviceToHost);
    cudaFree(Vout_d);
    cudaFree(Min_d);
    cudaFree(Vin_d);
}