import os

dist_backends = [
    # 'torch_ddp',
    # 'torch_fsdp',
    'colossalai'
]
models = [
    # 'vit_s',
    'vit_b',
    'vit_h',
    'vit_g',
    # 'vit_10b',
    'darts',
    'ofa',
    'mobilenet',
    # 'resnet152'
]
gpus = [
    # 1,
    # 2,
    # 4,
    8
]
batch_sizes = [
    # 8,
    # 16,
    # 32,
    # 64,
    # 128,
    256,
    512,
    # 640,
    # 768,
    # 832,
    1024,
    2048,
    # 3200,
    # 4096,
    # 8192,
]
img_sizes = [
    32,
    # 128,
    # 224,
]
use_zeros = [
    0,
    1,
]
use_pipelines = [
    0,
    1,
]
use_fp16s = [
    0,
    1,
]
nofs = [
    0,
    # 0.2,
    # 0.5,
    1
]
debug = 0
steps = 100
exp_name = '_'
# seed = random.randint(0, 1000000)
seed = 888

# torchrun --nproc_per_node={gpus} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py \
command = '''
torchrun --nproc_per_node={gpus} benchmark.py \
--dist_backend={dist_backend} \
--gpus {gpus} \
--model {model} \
--use_zero {use_zero} \
--use_pipeline {use_pipeline} \
--use_fp16 {use_fp16} \
--nof {nof} \
--steps {steps} \
--batch_size {batch_size} \
--img_size {img_size} \
--exp_name {exp_name} \
--debug {debug} \
--seed {seed}
'''

param_set = []
for model in models:
    for dist_backend in dist_backends:
        for gpu in gpus:
            for batch_size in batch_sizes:
                for img_size in img_sizes:
                    if model == 'vit_10b' and batch_size > 32:
                        batch_size = batch_size // 256
                    if model == 'vit_g' and batch_size > 32:
                        batch_size = batch_size // 4
                    params = {
                        'model': model,
                        'dist_backend': dist_backend,
                        'gpus': gpu,
                        'batch_size': batch_size,
                        'img_size': img_size,
                        'use_zero': 0,
                        'steps': steps,
                        'exp_name': exp_name,
                        'use_pipeline': 0,
                        'use_fp16': 1,
                        'debug': debug,
                        'seed': seed
                    }
                    if dist_backend == 'colossalai':
                        for use_zero in use_zeros:
                            if model in ['vit_10b'] and not use_zero:
                                continue
                            params.update({'use_zero': use_zero})
                            for use_pipeline in use_pipelines:
                                if model in ['darts', 'ofa', 'mobilenet'] and use_pipeline == 1:
                                    continue
                                if gpu <= 1 and use_pipeline == 1: # GPU数量>1时，pipeline才能使用
                                    continue
                                if use_zero == 1 and use_pipeline == 1:
                                    continue
                                params.update({'use_pipeline': use_pipeline})
                                for use_fp16 in use_fp16s:
                                    # if use_zero == 1 and use_fp16 == 1:
                                    #     continue
                                    params.update({'use_fp16': use_fp16})
                                    if use_zero == 1:
                                        use_fp16 = 0
                                        params.update({'use_fp16': use_fp16})
                                        for nof in nofs:
                                                params.update({'nof': nof})
                                                param_set.append(params.copy())
                                    else:
                                        params.update({'nof': 0, 'use_fp16': use_fp16})
                                        param_set.append(params.copy())
                    else:
                        param_set.append(params.copy())

commands = []
for param in param_set:
    gen_command = command.format(**param)
    if gen_command not in commands:
        commands.append(gen_command)
for i, command in enumerate(commands):
    print(i, command)
    # os.system(command)
print(f"Total {len(commands)} commands")
with open('./scripts/batch_benchmark.sh', 'w') as f:
    for command in commands:
        f.write(command)
        # os.system(command)
