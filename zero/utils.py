import csv
import gc
import os

import pandas as pd
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

def get_memory_usage(rank, to_round=True):
    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()
    if not isinstance(rank, int):
        rank = int(rank)

    MA = torch.cuda.memory_allocated(rank) / (1024**3) # real memory allocated
    Max_MA = torch.cuda.max_memory_allocated(rank) / (1024**3) # peak memory allocated
    CA = torch.cuda.memory_reserved(rank) / (1024**3) # memory reserved by torch, >= MA
    Max_CA = torch.cuda.max_memory_reserved(rank) / (1024**3) # peak memory reserved by torch, >= Max_MA

    vm_stats = psutil.virtual_memory()
    used_GB = (vm_stats.total - vm_stats.available) / (1024**3)
    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()
    if to_round:
        return round(MA, 2), round(Max_MA, 2), round(CA, 2), round(Max_CA, 2), round(used_GB, 2)
    return MA, Max_MA, CA, Max_CA, used_GB


class ExpLog:
    def __init__(self, args):
        self.args = args
        self.root_dir = self.args.root_dir
        self.data = []
        self.params = {}
        self.init()

    def init(self):
        headers = ['model', 'gpus', 'batch_size', 'img_size', 'use_zero', 'nof', 'placement', 'use_pipeline', 'use_fp16', 'use_ac', 'seed']
        for h in headers:
            self.params[h] = getattr(self.args, h)

    def add(self, bt, tp):
        lrank = os.environ['LOCAL_RANK']
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        self.data.append({
            'Batch Time': bt,
            'Throughput': tp,
            'Allocated GPU Mem': ag,
            'Max Allocated GPU Mem': mag,
            'Reserved GPU Mem': rg,
            'Max Reserved GPU Mem': mrg,
            'Used CPU Mem': cm
        })

    def save_as_csv(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.root_dir, 'exp_log.csv')
        df_data = pd.DataFrame(self.data)
        df_params = pd.DataFrame(self.params, index=[0])
        df_params = pd.concat([df_params]*df_data.shape[0], ignore_index=True)
        df = pd.concat([df_params, df_data], axis=1)
        df.to_csv(file_path, index=False)

    def stats_and_save(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.root_dir, 'stat_log.csv')
        df_data = pd.DataFrame(self.data)
        df_params = pd.DataFrame(self.params, index=[0])
        df_stats = {}
        for item in ['Batch Time', 'Throughput', 'Allocated GPU Mem', 'Max Allocated GPU Mem', 'Reserved GPU Mem', 'Max Reserved GPU Mem', 'Used CPU Mem']:
            df_stats[item] = [df_data[item].mean(), df_data[item].var(), df_data[item].max()]
        df_stats = pd.DataFrame(df_stats)
        df_params = pd.concat([df_params]*df_stats.shape[0], ignore_index=True)
        df = pd.concat([df_params, df_stats], axis=1)
        df.to_csv(file_path, index=False)
