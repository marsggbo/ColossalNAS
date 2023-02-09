import os

dist_backends = [
    # 'torch_ddp',
    # 'torch_fsdp',
    'colossalai'
]
models = [
    # 'vit',
    # 'vit_b',
    # 'vit_h',
    'vit_g',
    # 'darts',
    # 'ofa'
    # 'resnet152'
]
gpus = [
    # 1,
    # 2, 
    4
]
batch_sizes = [
    # 16,
    32,
    # 64,
    # 128,
    # 256,
    # 512,
    # 640,
    # 768,
    # 1024,
    # 2048,
    # 3200,
    # 4096,
    # 9216,
]
img_sizes = [32]
use_zeros = [
    0,
    # 1,
]
use_pipelines = [
    0,
    # 1,
]
use_fp16s = [
    0,
    1,
]
debug = 0
steps = 50
exp_name = '_'

command = '''
torchrun --nproc_per_node={gpus} benchmark.py \
--dist_backend={dist_backend} \
--gpus {gpus} \
--model {model} \
--use_zero {use_zero} \
--use_pipeline {use_pipeline} \
--use_fp16 {use_fp16} \
--steps {steps} \
--bs {bs} \
--img_size {img_size} \
--exp_name {exp_name} \
--debug {debug} 
'''

param_set = []
for model in models:
    for dist_backend in dist_backends:
        for gpu in gpus:
            for bs in batch_sizes:
                for img_size in img_sizes:
                    params = {
                        'model': model,
                        'dist_backend': dist_backend,
                        'gpus': gpu,
                        'bs': bs,
                        'img_size': img_size,
                        'use_zero': 0,
                        'steps': steps,
                        'exp_name': exp_name,
                        'use_pipeline': 0,
                        'use_fp16': 1,
                        'debug': debug,
                    }
                    if dist_backend == 'colossalai':
                        for use_zero in use_zeros:
                            params.update({'use_zero': use_zero})
                            for use_pipeline in use_pipelines:
                                params.update({'use_pipeline': use_pipeline})
                                for use_fp16 in use_fp16s:
                                    params.update({'use_fp16': use_fp16})
                                    param_set.append(params.copy())
                    else:
                        param_set.append(params.copy())

commands = []
for param in param_set:
    commands.append(command.format(**param))
for i, command in enumerate(commands):
    print(i, command)
    # os.system(command)
print(f"Total {len(commands)} commands")
with open('./scripts/batch_benchmark.sh', 'w') as f:
    for command in commands:
        f.write(command)
        # os.system(command)