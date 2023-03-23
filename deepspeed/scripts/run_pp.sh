#!/bin/bash

deepspeed  --master_port 1234 train_pp.py \
--deepspeed_config=./configs/pp_config.json -p 2 --steps=200
