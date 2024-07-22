#define SECTION_SIZE 1024
__global__
void brent_kung_scan(float *X, float *Y, unsigned int N) {
    // 使用brent-kung scan
    // 输入: X, 输出: Y, N为该段的长度，需小于SECTION_SIZE
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) XY[threadIdx.x] = X[i];
    if(i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x]; // 集体将元素load到SRAM
    // reduction阶段，积累partial sum
    for(int stride = 1; stride < blockDim.x; stride *= 2){
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
    if(i < N) Y[i] = XY[threadIdx.x];
    if(i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

}

// helper function
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

void scan_inblock(torch::Tensor A, torch::Tensor B){
    int length = A.size(0);
    dim3 dimBlock(SECTION_SIZE); // section的大小和thread的数量相同
    dim3 dimGrid(1); // 只用一个block
    brent_kung_scan<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), length);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }   
}

