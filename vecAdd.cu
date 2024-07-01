#include <iostream>
#include <vector>
__global__
void vecAddKnernel(float *A, float *B, float *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}
void vecAdd(float *A_h, float *B_h, float *C_h, int N){
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    int blocksPerGrid = (N + 255) / 256;
    vecAddKnernel<<<blocksPerGrid, 256>>>(A_d, B_d, C_d, N);
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    int N = 10; // 示例大小
    float *A_h = new float[N];
    float *B_h = new float[N];
    float *C_h = new float[N];
    for(int i=0; i < N; ++i){
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }
    vecAdd(A_h, B_h, C_h, N);
    for(int i=0; i < N; ++i){
        std::cout << C_h[i] << std::endl;
    }
    return 0;
}