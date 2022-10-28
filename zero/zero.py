
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel
from time import time
from functools import partial

from ipdb import set_trace
from hyperbox_app.distributed.networks.ofa import OFAMobileNetV3
from hyperbox.mutator import RandomMutator


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def get_ofa(width_mult=4):
    return OFAMobileNetV3(
        width_mult=width_mult, depth_list=[4,5], expand_ratio_list=[4,6],
        base_stage_width=[16, 32, 64, 128, 256, 320, 480, 512, 960]
        # base_stage_width=[32, 64, 128, 256, 512, 512, 512, 960, 1024]
    )


def main():
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 5
    use_zero = True
    # use_zero = False
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    if use_zero:
        shard_strategy = TensorShardStrategy()
        set_trace()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
            # model = gpt2_medium(checkpoint=True)
            model = get_ofa(4)
            # model = OFAMobileNetV3(
            #     width_mult=4, depth_list=[4,5], expand_ratio_list=[4,6],
            #     base_stage_width=[32, 64, 128, 256, 512, 1024, 1024, 2048, 2048]
            # )
        numel = ctx.model_numel_tensor.item()
        logger.info(f'Model numel: {numel}', ranks=[0])
        get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
        # Set tensor_placement_policy='cpu', which will offload params, grads and os
        model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cpu', reuse_fp16_shard=True)
        logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    else:
        model = get_ofa(4).to(torch.cuda.current_device())
        numel = sum([p.numel() for p in model.parameters()])
        get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
    rm = RandomMutator(model)
    rm.reset()

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    if use_zero:
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        # input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        x = torch.rand(2,3,64,64).to(torch.cuda.current_device())
        y = torch.rand(2, 1000).to(torch.cuda.current_device())
        if not use_zero:
            rm.reset()
        elif n <= 3:
            rm.reset()
        optimizer.zero_grad()
        start = time()
        # outputs = model(input_ids, attn_mask)
        # loss = criterion(outputs, input_ids)
        outputs = model(x)
        loss = (outputs-y).sum()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        if use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}', ranks=[0])


if __name__ == '__main__':
    main()
