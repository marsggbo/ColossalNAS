
torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 1 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 1 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_b --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 1 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 1 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_h --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 128 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 128 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 1 --use_fp16 0 --nof 0 --steps 100 --batch_size 128 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 0 --use_pipeline 1 --use_fp16 1 --nof 0 --steps 100 --batch_size 128 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model vit_g --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 128 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model darts --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model ofa --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 512 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 0 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 0 --use_pipeline 0 --use_fp16 1 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888

torchrun --nproc_per_node=4 benchmark.py --dist_backend=colossalai --gpus 4 --model mobilenet --use_zero 1 --use_pipeline 0 --use_fp16 0 --nof 0 --steps 100 --batch_size 1024 --img_size 32 --exp_name _ --debug 0 --seed 888
