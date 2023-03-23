#!/bin/bash

deepspeed train_zero.py --deepspeed --deepspeed_config ./configs/zero_config.json \
--with_cuda 
$@
