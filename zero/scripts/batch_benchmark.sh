
torchrun --nproc_per_node=1 benchmark.py --dist_backend=colossalai --gpus 1 --model vit --use_zero 0 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=1 benchmark.py --dist_backend=colossalai --gpus 1 --model vit --use_zero 1 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=2 benchmark.py --dist_backend=colossalai --gpus 2 --model vit --use_zero 0 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=2 benchmark.py --dist_backend=colossalai --gpus 2 --model vit --use_zero 1 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit --use_zero 0 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit --use_zero 1 --steps 20 --bs 256 --img_size 32 --exp_name _ --debug 0 
