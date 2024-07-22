#define SECTION_SIZE 1024
__global__
void kogge_stone_scan(float *X, float *Y, unsigned int N) {
    // 使用kogge-stone scan算法,非常straright forward的算法
    // 输入X，输出Y, N是数组长度
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        XY[threadIdx.x] = X[i];
    }
    else{
        XY[threadIdx.x] = 0.f;
    }

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp = XY[threadIdx.x];
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads(); // 保证不覆盖原有的partial sum
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }
    if(i < N){
        Y[i] = XY[threadIdx.x]; // 写回结果
    }
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void scan_inblock(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock(SECTION_SIZE); // section的大小和thread的数量相同
    dim3 dimGrid(1); // 只用一个block
    kogge_stone_scan<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), length);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }   
}

