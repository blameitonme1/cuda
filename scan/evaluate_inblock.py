import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def compile_cuda_kogge_stone_scan():
    cuda_source = Path("kogge_stone_scan.cu").read_text()
    cpp_source = "void scan_inblock(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="scan_inblock",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["scan_inblock"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_brent_kung_scan():
    cuda_source = Path("brent_kung_scan.cu").read_text()
    cpp_source = "void scan_inblock(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="scan_inblock",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["scan_inblock"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():
    kogge_stone_scan = compile_cuda_kogge_stone_scan()
    brent_kung_scan = compile_cuda_brent_kung_scan()
    a = torch.ones(1000).cuda()
    b = torch.zeros(1000).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 测试kogge-stone
    start_event.record()
    for i in range(100):
        kogge_stone_scan.scan_inblock(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations kogge stone scan: {elapsed_time_ms} ms")

    start_event.record()
    for i in range(100):
        brent_kung_scan.scan_inblock(a, b)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 100 iterations brent kung scan: {elapsed_time_ms} ms")


if __name__ == "__main__":
    main()