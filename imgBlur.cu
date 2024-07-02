__global__
void imgBlurKernel(unsigned char* Pin, unsigned char *Pout, int w, int h, int windowSize){
    // 为啥不取rgb？
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < h && col < w){
        int sumR = 0; // sumation of all the value in the convolution window for red
        int sumG = 0; // sumation of all the value in the convolution window for green
        int sumB = 0; // sumation of all the value in the convolution window for blue
        int pixels = 0; // number of pixels
        for(int i = -windowSize; i <= windowSize; ++i){
            for(int j = -windowSize; j <= windowSize; ++j){
                int newrow = row + i;
                int newcol = col + j;
                if(newrow >= 0 && newrow < h && newcol >= 0 && newcol < w){
                    int idx = (newrow * w + newcol) * 3;
                    sumR += Pin[idx];
                    sumG += Pin[idx + 1];
                    sumB += Pin[idx + 2];
                    pixels++;
                }
            }
        }
        int idx = (row * w + col) * 3;
        Pout[row * w + col] = (unsigned char) sumR / pixels;
        Pout[row * w + col + 1] = (unsigned char) sumG / pixels;
        Pout[row * w + col + 2] = (unsigned char) sumB / pixels;
    }
}