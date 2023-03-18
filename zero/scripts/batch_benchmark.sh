
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 50 --batch_size 512 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 50 --batch_size 512 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 50 --batch_size 512 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 50 --batch_size 512 --img_size 32 --exp_name _ --debug 0 

torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 benchmark.py --dist_backend=colossalai --gpus 1 --model darts --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 50 --batch_size 32 --img_size 32 --exp_name _ --debug 0 

