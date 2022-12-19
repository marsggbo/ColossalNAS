node_rank=$1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
torchrun \
--nproc_per_node=4 \
--nnodes=3 \
--node_rank=${node_rank} \
--rdzv_id=666 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.31.229.85:6688 \
benchmark.py \
--dist_backend=colossalai --gpus 12 --model vit_g --use_zero 1 --steps 30 --bs 1024 --img_size 32 --exp_name _ --debug 0 --placement cuda