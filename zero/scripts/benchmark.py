
dist_backends = [
    # 'torch_ddp',
    # 'torch_fsdp',
    'colossalai'
]
models = [
    'vit',
    # 'darts',
    # 'ofa'
]
gpus = [
    1,
    2, 
    4
]
batch_sizes = [
    # 64,
    # 128,
    256,
    # 512
]
img_sizes = [32]
use_zeros = [0, 1]
debug = 0
steps = 20
exp_name = '_'

command = '''
torchrun --nproc_per_node={gpus} benchmark.py \
--dist_backend={dist_backend} \
--gpus {gpus} \
--model {model} \
--use_zero {use_zero} \
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
                        'debug': debug,
                    }
                    if dist_backend == 'colossalai':
                        for use_zero in use_zeros:
                            params.update({'use_zero': use_zero})
                            param_set.append(params.copy())
                    else:
                        param_set.append(params.copy())

commands = []
for param in param_set:
    commands.append(command.format(**param))
print(commands[0])
print(f"Total {len(commands)} commands")
with open('./scripts/batch_benchmark.sh', 'w') as f:
    for command in commands:
        f.write(command)