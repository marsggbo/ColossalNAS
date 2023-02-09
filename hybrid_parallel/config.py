from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 16
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 1
WARMUP_EPOCHS = 3

# large model config
IMG_SIZE = 32
PATCH_SIZE = 16
HIDDEN_SIZE = 2048
DEPTH = 32
NUM_HEADS = 64
MLP_RATIO = 4
NUM_CLASSES = 1000
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token

# # small model config
# IMG_SIZE = 224
# PATCH_SIZE = 16
# HIDDEN_SIZE = 512
# DEPTH = 4
# NUM_HEADS = 4
# MLP_RATIO = 2
# NUM_CLASSES = 1000
# CHECKPOINT = False
# SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE)**2 + 1    # add 1 for cls token

# parallel setting
ddp = 1
TENSOR_PARALLEL_SIZE = 1
TENSOR_PARALLEL_MODE = '1d'
# TENSOR_PARALLEL_MODE = '2d'
torch_ddp = dict(
    find_unused_parameters=True
)
parallel = dict(
    data=ddp,
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

# fp16 = dict(mode=AMP_TYPE.NAIVE)
# clip_grad_norm = 1.0

# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
