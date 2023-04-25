run1="torchrun --nproc_per_node=1 train.py --gpus 1 --model"
run4="torchrun --nproc_per_node=4 train.py --gpus 4 --model"

MODEL=darts
# ${run1} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
${run1} ${MODEL} --img_size 224 --batch_size 4 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 0
${run1} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 0
# ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1

# ${run4} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1

MODEL=resnet152
# ${run1} ${MODEL} --img_size 224 --batch_size 48 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name full_vit_l --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 96 --steps 15    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name full_vit_l --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 256 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name full_vit_l --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 256 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name full_vit_l --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1

MODEL=vit_l
# ${run1} ${MODEL} --img_size 224 --batch_size 48 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name full_vit_l --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 48 --steps 15    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name full_vit_l --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 96 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name full_vit_l --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 128 --steps 15     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name full_vit_l --debug 0 --use_gemini 1
# # ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run1} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1


# # ${run4} ${MODEL} --img_size 224 --batch_size 360 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name full_vit_l --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 360 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 96 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# # # ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0
# # # ${run4} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 96 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1


MODEL=vit_h
# # ${run1} ${MODEL} --img_size 224 --batch_size 24 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 48 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 72 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run1} ${MODEL} --img_size 224 --batch_size 24 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 48 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 72 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1


# # ${run4} ${MODEL} --img_size 224 --batch_size 256 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 256 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 24 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 72 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run4} ${MODEL} --img_size 224 --batch_size 24 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0
# ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 64 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 72 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1


MODEL=vit_g
# ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run1} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run1} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run1} ${MODEL} --img_size 224 --batch_size 8 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1

# # ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 128 --steps 50    --use_ac 1 --use_fp16 1 --use_pipeline 1 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 

# ${run4} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # # ${run4} ${MODEL} --img_size 224 --batch_size 2 --steps 50    --use_ac 0 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run4} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
# ${run4} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # # ${run4} ${MODEL} --img_size 224 --batch_size 2 --steps 50     --use_ac 1 --use_fp16 1 --use_pipeline 0 --use_zero 0     --nof 0 --placement auto --seed 666 --exp_name null --debug 0 
# # ${run4} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cuda --seed 666 --exp_name null --debug 0 --use_gemini 1
# # ${run4} ${MODEL} --img_size 224 --batch_size 16 --steps 50     --use_ac 1 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1


# # torchrun --nproc_per_node=2 train.py --gpus 2 --model toy --img_size 224 --batch_size 8 --steps 50     --use_ac 0 --use_fp16 0 --use_pipeline 0 --use_zero 1     --nof 0 --placement cpu --seed 666 --exp_name null --debug 0 --use_gemini 1
