import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

"""
直接使用使用范围最广最优的kernel, 前面的实现暂时不管
"""

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def compile_cuda_no_coarse():
    cuda_source = Path("seg_sum_kernel.cu").read_text()
    cpp_source = "void sum_reduction(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="sum_reduction",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["sum_reduction"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_coarse():
    cuda_source = Path("seg_coarse_sum_kernel.cu").read_text()
    cpp_source = "void sum_reduction(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="sum_reduction",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["sum_reduction"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():
    sum_reduction = compile_cuda_no_coarse()
    sum_reduction_coarse = compile_cuda_coarse()
    a = torch.ones(4100).cuda()
    b = torch.zeros(1).cuda() # 注意使用cuda!!!
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(100):
        sum_reduction.sum_reduction(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations basic: {elapsed_time_ms} ms")

    print(b)
    b = torch.zeros(1).cuda() # 注意使用cuda!!!

    start_event.record()
    for i in range(100):
        sum_reduction_coarse.sum_reduction(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations basic: {elapsed_time_ms} ms")

    print(b)

if __name__ == "__main__":
    main()
