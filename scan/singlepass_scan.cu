#include <stdio.h>
#define SECTION_SIZE 1024
__global__
void single_pass_scan_kernel(float *X, float *Y, int *flags, float *scan_value, unsigned int N, int *blockCounter) {
    // 预处理先使用dynamic block index assignment得到block id
    __shared__ int bid_s;
    if(threadIdx.x == 0){
        bid_s = atomicAdd(blockCounter, 1);
        // printf("block counter: %d\n", *blockCounter);
        // printf("bid s: %d\n", bid_s);
    }
    __syncthreads();
    unsigned int bid = bid_s; 

    // 第一步使用brent kung scan
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * bid * blockDim.x + threadIdx.x; // 注意这里是乘2倍, map的时候用bid来map!
    if(i < N) XY[threadIdx.x] = X[i];
    if(i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x]; // 集体将元素load到SRAM
    // reduction阶段，积累partial sum
    for(int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1; // 向高处积累partial sum
        if(index < SECTION_SIZE){
            XY[index] += XY[index - stride];
        }
    }
    // 开始reduction
    for(int stride = SECTION_SIZE / 4; stride >= 1; stride /= 2){
        // 初始化为SECTION_SIZE / 4 是因为最大的stride就这么大
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1; // map的坐标还是一样的，只不过上面加前面的，这个是作为前面的被加的
        if(index + stride < SECTION_SIZE){
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    float local_sum = XY[SECTION_SIZE - 1]; // 本段的sum
    // if(threadIdx.x == 0) {
    //     printf("local sum: %f block id %d\n", local_sum, blockIdx.x);
    // }  
    if(i < N) Y[i] = XY[threadIdx.x];
    if(i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    __syncthreads();
    __shared__ float previous_sum;
    if(threadIdx.x == 0){
        while(atomicAdd(&flags[bid], 0) == 0) {}
        previous_sum = scan_value[bid];
        // printf("previous sum: %f block id %d\n", previous_sum, blockIdx.x);
        scan_value[bid + 1] = previous_sum + local_sum;
        __threadfence(); // 确保scan_value内存已经被更新
        atomicAdd(&flags[bid + 1], 1);
    }
    __syncthreads();
    // 最后一步将previous sum加到所有元素上去
    if(i < N) Y[i] += previous_sum;
    if(i + blockDim.x < N) Y[i + blockDim.x] += previous_sum;
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void scan(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock(SECTION_SIZE / 2); // section的大小和thread的数量相同
    int aux_length = cdiv(length, SECTION_SIZE);
    dim3 dimGrid(aux_length);
    int *flags;
    cudaMalloc((void **)&flags, aux_length * sizeof(int));
    float *scan_value;
    cudaMalloc((void **)&scan_value, (aux_length + 1) * sizeof(float));
    int  *blockCounter;
    cudaMalloc((void **)&blockCounter, sizeof(int));
    cudaMemset(blockCounter, 0, sizeof(int));
    cudaMemset(flags, 0, aux_length * sizeof(int));
    cudaMemset(flags, 1, sizeof(int));
    cudaMemset(scan_value, 0, aux_length * sizeof(float));
    // printf("started launching the kernel!\n");
    single_pass_scan_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), flags, scan_value, length, blockCounter);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    cudaFree(flags);
    cudaFree(scan_value);
    cudaFree(blockCounter);   
}
