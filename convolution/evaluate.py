import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_gray_image_to_tensor(image_path):
    # 定义一个转换操作，将图片转换为灰度图并转换为张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    # 打开图片并应用转换
    image = Image.open(image_path)
    tensor = transform(image)

    return tensor

def display_tensor_as_image(tensor, save_path=None):
    # 确保张量在 CPU 上
    tensor = tensor.cpu()
    
    # 如果张量有多个通道，只取第一个通道
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] > 1:
        tensor = tensor[0]
    
    # 将张量转换为 numpy 数组
    image_np = tensor.numpy()
    
    # 显示图像
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    
    # 如果提供了保存路径，则保存图像
    if save_path is not None:
        plt.imsave(save_path, image_np, cmap='gray')
        print(f"picture saved to {save_path}")

def compile_cuda_cache():
    cuda_source = Path("cache_kernel.cu").read_text()
    cpp_source = "torch::Tensor conv_2d(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="conv_2d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["conv_2d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_tile():
    cuda_source = Path("tile_kernel.cu").read_text()
    cpp_source = "torch::Tensor conv_2d(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="conv_2d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["conv_2d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_tile_cache_halo():
    cuda_source = Path("cache_halo_cell_tile_kernel.cu").read_text()
    cpp_source = "torch::Tensor conv_2d(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="conv_2d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["conv_2d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def compile_cuda_basic():
    cuda_source = Path("basic_kernel.cu").read_text()
    cpp_source = "torch::Tensor conv_2d(torch::Tensor A, torch::Tensor B);"

    function = load_inline(
        name="conv_2d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["conv_2d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return function

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    a = load_gray_image_to_tensor("./yuri.jpg").to(device)
    print(a.shape)
    # b = torch.ones(3, 3, device=device, dtype=torch.float32) / 9.0
    b = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype=torch.float32, device=device)
    b = b.unsqueeze(0).unsqueeze(0)


    conv2d_basic = compile_cuda_basic()
    conv2d_cache = compile_cuda_cache()
    conv2d_tile = compile_cuda_tile()
    conv2d_tile_cache_halo = compile_cuda_tile_cache_halo()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = conv2d_basic.conv_2d(a, b)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 10 iterations basic: {elapsed_time_ms} ms")
    
    display_tensor_as_image(c, "./basic.jpg")
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
        
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = conv2d_cache.conv_2d(a, b)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 10 iterations cache: {elapsed_time_ms} ms")
    
    display_tensor_as_image(c, "./cache.jpg")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = conv2d_tile.conv_2d(a, b)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 10 iterations cache: {elapsed_time_ms} ms")
    
    display_tensor_as_image(c, "./tile.jpg")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = conv2d_tile_cache_halo.conv_2d(a, b)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 10 iterations cache: {elapsed_time_ms} ms")

    display_tensor_as_image(c, "./cache_halo.jpg")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start_event.record()
        for i in range(10):
            c = F.conv2d(a, b)
            
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time for 10 iterations torch: {elapsed_time_ms} ms")
    
    display_tensor_as_image(c)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
    # assert torch.allclose(c, c_, rtol=1e-03, atol=1e-04), "Results do not match"
    # assert torch.allclose(c, c__, rtol=1e-03, atol=1e-04), "Results do not match"
    # assert torch.allclose(c, c___, rtol=1e-03, atol=1e-04), "Results do not match"


if __name__ == "__main__":
    main()