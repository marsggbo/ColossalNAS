
torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 0 --use_fp16 0 --steps 50 --bs 64 --img_size 32 --exp_name _test --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 0 --use_fp16 1 --steps 50 --bs 32 --img_size 32 --exp_name _test --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 0 --use_fp16 0 --steps 50 --bs 128 --img_size 32 --exp_name _test --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 0 --use_fp16 1 --steps 50 --bs 128 --img_size 32 --exp_name _test --debug 0 
