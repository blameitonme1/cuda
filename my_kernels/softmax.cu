#define BLOCK_SIZE 256
__global__
void softmax_kernel(float *input, float *output, int nrows, int ncols){
    int row = blockIdx.x * blockDim.x; // 1D mapping, 一个block处理一行
    int col = threadIdx.x;
    __shared__ row_data[BLOCK_SIZE];
    __shared__ max_ele;
    __shared__ denominator;

    if (threadIdx.x == 0) {
        // 初始化两个全局变量在SRAM里面
        max_ele = 0.f;
        denominator = 0.f;  
    }
    __syncthreads();

    if(col < ncols){
        row_data[threadIdx.x] = input[row * ncols + col];
    }
    else{
        row_data[threadIdx.x] = 0.f;
    }

    __syncthreads();

    // 使用reduction计算最大值
    float local_max = row_data[col];
    for(int offset = BLOCK_SIZe / 2; offset > 0; offset /= 2){
        if(col < offset){
            float other = row_data[col + offset];
            if(other > local_max){
                local_max = other; // 更新最大值
            }
        }
    }

    if(threadIdx.x == 0){
        max_ele = local_max;
    }
    __syncthreads();
    
    if(col < ncols){
        row_data[col] = __expf(row_data[col] - max_ele);
        atomicAdd(&denominator, row_data[col]);
    }

    __syncthreads();

    if(col < ncols){
        output[row * ncols + col] = row_data[col] / denominator;
    }
}