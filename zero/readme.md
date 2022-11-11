# Run

```
torchrun --nproc_per_node=1 zero.py --gpus 1 --useZero 1 --debug 1
```

# Gemini Zero

```
torchrun --nproc_per_node=1 gemeniZero.py --placement cpu  --debug 0
```