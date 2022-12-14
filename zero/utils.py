
import psutil
import torch

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2

def get_peak_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**2

def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, Peak GPU memory usage: {get_peak_gpu_mem():.2f} MB'
    # return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'

def DeviceInfo(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.device, p.shape, p.grad.device, p.grad.shape)
        else:
            print(name, p.device, p.shape, None, None)
