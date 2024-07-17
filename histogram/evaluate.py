import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def compile_cuda_basic():
    cuda_source = Path("basic_kernel.cu").read_text()
    cpp_source = "void histogram(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="histogram",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["histogram"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_coarse():
    cuda_source = Path("coarse_kernel.cu").read_text()
    cpp_source = "void histogram(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="histogram",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["histogram"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    histogram_basic = compile_cuda_basic()
    histogram_coarse = compile_cuda_coarse()
    file_path = './input.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        input_string = file.read()
    data = [ord(char) for char in input_string]
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()

    histo = torch.zeros(7, dtype=torch.float32, device='cuda')
    print(histo)
    start_event.record()
    for i in range(100):
        histogram_basic.histogram(data_tensor, histo)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 10 iterations basic: {elapsed_time_ms} ms")
    print(histo)
    histo = torch.zeros(7, dtype=torch.float32, device='cuda')
    start_event.record()
    for i in range(100):
        histogram_coarse.histogram(data_tensor, histo)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time for 10 iterations coarse: {elapsed_time_ms} ms")
    print(histo)

if __name__ == "__main__":
    main()