
# torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 0 --steps 50 --batch_size 64 --img_size 128 --exp_name _ --debug 0 

# torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 1 --steps 50 --batch_size 64 --img_size 128 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 1 --use_fp16 0 --steps 50 --batch_size 64 --img_size 128 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 1 --use_pipeline 0 --use_fp16 0 --steps 50 --batch_size 64 --img_size 128 --exp_name _ --debug 0 
