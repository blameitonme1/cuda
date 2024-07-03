__global__
void colorToGrayScaleConversionKernel(unsigned char *Pin, unsigned char *Pout, int width, int height){
    // calculate the coordinates of the current pixel that thhis thread is processing
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width){
        int out_index = row * width + col;
        int in_index = out_index * 3; // get the actuall address in the Pin.
        // get all the color values
        unsigned char r = Pin[in_index]; // unsigned char is 8 bit unsigned integer
        unsigned char g = Pin[in_index + 1];
        unsigned char b = Pin[in_index + 2];
        Pout[out_index] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}