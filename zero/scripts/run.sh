dist_backend=$1
gpus=$2
model=$3
use_zero=$4
steps=$5
bs=$6
img_size=$7
exp_name=$8
debug=$9

torchrun --nproc_per_node=${gpus} benchmark.py \
--dist_backend=${dist_backend:=colossalai} \
--gpus ${gpus} \
--model ${model:=darts} \
--use_zero ${use_zero:=1} \
--steps ${steps:=20} \
--bs ${bs:=64} \
--img_size ${img_size:=128} \
--exp_name ${exp_name:=''} \
--debug ${debug:=0} 

