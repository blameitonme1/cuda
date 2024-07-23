
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

__global__
void kogge_stone_scan_seg(float *X, float *Y, float *S, unsigned int N) {
    // 使用kogge-stone scan算法, segemented
    // 输入X，输出Y, N是数组长度, S是辅助数组
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

    __syncthreads();
    if(threadIdx.x == blockDim.x - 1){
        S[blockIdx.x] = XY[SECTION_SIZE - 1]; // 储存结果到这个辅助数组里面
    }
}

__global__
void get_final_sum(float *S, float *Y, unsigned int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Y[i] += blockIdx.x >= 1 ? S[blockIdx.x - 1] : 0.f;
}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void scan(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock1(SECTION_SIZE); // section的大小和thread的数量相同
    int aux_length = cdiv(length, SECTION_SIZE);
    dim3 dimGrid1(aux_length);
    float *aux;
    cudaMalloc((void **)&aux, aux_length * sizeof(float));
    kogge_stone_scan_seg<<<dimGrid1, dimBlock1>>>(A.data_ptr<float>(), B.data_ptr<float>(), aux, length); // 完成了第一步
    dim3 dimBlock2(SECTION_SIZE);
    dim3 dimGrid2(1);
    kogge_stone_scan<<<dimGrid2, dimBlock2>>>(aux, aux, length); // 完成了第二步
    dim3 dimBlock3(SECTION_SIZE);
    dim3 dimGrid3(aux_length);
    get_final_sum<<<dimGrid3, dimBlock3>>>(aux, B.data_ptr<float>(), length); // 最后一步，加上aux的值得到最终结果
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }   
}



