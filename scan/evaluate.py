import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def compile_cuda_segmented_scan():
    cuda_source = Path("segmented_scan.cu").read_text()
    cpp_source = "void scan(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="scan",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["scan"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_singlepass_scan():
    cuda_source = Path("singlepass_scan.cu").read_text()
    cpp_source = "void scan(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="scan",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["scan"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():
    segmented_scan = compile_cuda_segmented_scan()
    singlepass_scan = compile_cuda_singlepass_scan()
    a = torch.ones(2000).cuda()
    b = torch.zeros(2000).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 测试segmented scan
    start_event.record()
    for i in range(100):
        segmented_scan.scan(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations segmented scan: {elapsed_time_ms} ms")
    # 测试singlepass scan
    start_event.record()
    for i in range(100):
        singlepass_scan.scan(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations singlepass scan: {elapsed_time_ms} ms")


if __name__ == "__main__":
    main()