
import psutil
import torch

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_gpu_mem(rank=None):
    return torch.cuda.memory_allocated(rank) / 1024**2

def get_peak_gpu_mem(rank=None):
    return torch.cuda.max_memory_allocated(rank) / 1024**2

def get_mem_info(rank=None):
    return get_cpu_mem(), get_gpu_mem(rank), get_peak_gpu_mem(rank)

def print_mem_info(prefix='', rank=None):
    cpu_m = get_cpu_mem()
    gpu_m = get_gpu_mem(rank)
    p_gpu_m = get_peak_gpu_mem(rank)
    return f'{prefix}CPU memory: {cpu_m:.2f} MB, GPU memory: {gpu_m:.2f} MB, Peak GPU memory: {p_gpu_m:.2f} MB'

def DeviceInfo(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.device, p.shape, p.grad.device, p.grad.shape)
        else:
            print(name, p.device, p.shape, None, None)
