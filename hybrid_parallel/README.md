# ColossalNAS
NAS framework based on ColossalAI

# Run the code
```
export DATA=~/datasets/cifar10
torchrun --nproc_per_node=2 hybrid_parallel.py --config config.py
```
