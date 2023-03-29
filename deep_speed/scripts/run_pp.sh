#!/bin/bash
echo "所有传入的参数是：$@"
deepspeed  --master_port 1234 train_pp.py \
--deepspeed_config=./configs/pp_config.json -p 4 --steps=50 $@
