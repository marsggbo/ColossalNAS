import os

steps = 50
gpus_ = [
    1,
    # 2,
    # 4,
    # 8
]
use_acs = [
    0,
    1
]
use_fp16s = [
    0,
    1
]
use_pipelines = [
    0,
    # 1
]
use_zeros = [
    0,
    1
]
nofs = [
    0,
    1
]
placements = [
    # 'null',
    # 'cpu',
    'auto',
    # 'nvme'
]
img_sizes = [
    # 32,
    224
]
batch_sizes = [
    # 4,
    # 8,
    16,
    # 32,
    # 64,
    # 96,
    # 384,
    # 128,
    # 256,
    # 512,
    # 1024,
    # 2048,
    # 4096
]
models = [
    # 'vit_s',
    # 'vit_b',
    # 'vit_h',
    # 'vit_g',
    # 'vit_10b',
    'darts',
    # 'ofa',
    # 'mobilenet',
    # 'resnet152'
]
debug = 0
exp_name = 'null'
seeds = [
    666,
    # 888
]

def cmd_gen(gpus, model, img_size, batch_size, steps, use_ac, use_fp16, use_pipeline, use_zero,
        nof, placement, seed, exp_name, debug):
    return f'''torchrun --nproc_per_node={gpus} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
    --gpus {gpus} --model {model} --img_size {img_size} --batch_size {batch_size} --steps {steps} \
    --use_ac {use_ac} --use_fp16 {use_fp16} --use_pipeline {use_pipeline} --use_zero {use_zero} \
    --nof {nof} --placement {placement} --seed {seed} --exp_name {exp_name} --debug {debug} 
    '''

commands = []
for seed in seeds:
    for model in models:
        for img_size in img_sizes:
            for batch_size in batch_sizes:
                for gpus in gpus_:
                    for use_ac in use_acs:
                        for use_fp16 in use_fp16s:
                            for use_pipeline in use_pipelines:
                                if use_pipeline and gpus <= 1:
                                    continue # pipeline is not supported for single gpu
                                if use_pipeline == 1 and use_fp16 == 1:
                                    continue # pipeline fp16 is not supported
                                for use_zero in use_zeros:
                                    if use_zero == 1 and use_pipeline == 1:
                                        continue # zero pipeline is not supported
                                    if use_zero and not use_fp16:
                                        continue # we only focus on zero fp16
                                    if not use_zero:
                                        nof = 0
                                        placement = 'auto'
                                        cmd = cmd_gen(gpus, model, img_size, batch_size, steps, use_ac, use_fp16,
                                                use_pipeline, use_zero, nof, placement, seed, exp_name, debug)
                                        if cmd not in commands:
                                            commands.append(cmd)
                                        continue
                                    for nof in nofs:
                                        for placement in placements:
                                            cmd = cmd_gen(gpus, model, img_size, batch_size, steps, use_ac, use_fp16,
                                                use_pipeline, use_zero, nof, placement, seed, exp_name, debug)
                                            if cmd not in commands:
                                                commands.append(cmd)


for i, command in enumerate(commands):
    print(i, command)
    # os.system(command)
print(f"Total {len(commands)} commands")
with open('./scripts/batch_benchmark.sh', 'w') as f:
    for command in commands:
        f.write(command)
        # os.system(command)
