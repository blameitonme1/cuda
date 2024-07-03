from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = Path("kernel.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_gray(torch::Tensor input);"

    function = load_inline(
        name="function",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_gray"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():
    function = compile_extension()

    x = read_image("1.jpg").permute(1,2,0).cuda()
    y = function.rgb_to_gray(x)
    write_png(y.permute(2, 0, 1).cpu(), "/home/bo/cuda/chapter3/gray/2.png")


if __name__ == "__main__":
    main()

