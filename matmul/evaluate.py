import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path

# def trace_handler(prof):
#     print(prof.key_averages().table(
#         sort_by="self_cuda_time_total", row_limit=-1))
#     prof.export_chrome_trace("test_trace_" + '23' + ".json")

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],

#     # In this example with wait=1, warmup=1, active=2, repeat=1,
#     # profiler will skip the first step/iteration,
#     # start warming up on the second, record
#     # the third and the forth iterations,
#     # after which the trace will become available
#     # and on_trace_ready (when set) is called;
#     # the cycle repeats starting with the next step

#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=2,
#         repeat=1),
#     on_trace_ready=trace_handler
#     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
#     # used when outputting for tensorboard
#     ) as p:
#         for iter in range(23):
#             # send a signal to the profiler that the next iteration has started
#             c = torch.matmul(a, b)
#             p.step()

def compile_cuda_org():
    cuda_source = Path("org_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="matmul",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_tile():
    cuda_source = Path("tile_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"
    function = load_inline(
        name="matmul",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).to(device)
    # b = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]], dtype=torch.float32).to(device)
    # a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
    # b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to(device)
    a = torch.randn(1000, 2000).to(device)    
    b = torch.randn(2000, 3000).to(device)    


    matmul_org = compile_cuda_org()
    matmul_tile = compile_cuda_tile()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c_ = matmul_org.matmul(a, b)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 1000 iterations org: {elapsed_time_ms} ms")
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
    

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c__ = matmul_tile.matmul(a, b)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 1000 iterations tile: {elapsed_time_ms} ms")
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = torch.matmul(a, b)
            
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 1000 iterations torch: {elapsed_time_ms} ms")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
    # assert torch.allclose(c, c_), "Results do not match"
    # assert torch.allclose(c, c__), "Results do not match"


if __name__ == "__main__":
    main()