"""
Training entrypoint. Detects DDP/device, builds the model + MuonAdamW
optimizer + data loaders, then runs the training loop with periodic
validation, sampling, and HellaSwag eval.

Run with:  python train.py                                  (single GPU/CPU)
       or: torchrun --standalone --nproc_per_node=N train.py (multi-GPU DDP)
"""

import os
import math
import time
from functools import partial

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from Model import GPT_config, GPT
from optimizer import MuonAdamW
from DataLoader import DL_lite
from HellaSwag import render_example, iterate_examples, get_most_likely_row


###_________________________ DISTRIBUTED TRAINING _______________________________________

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), f'There is not cuda for the device to run ddp'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')


###_________________________________ LR SCHED _______________________________________________

# ------ Ratios --------
def get_lr_ratio(step, config):
    """Warmup-then-cosine-decay ratio in [min_lr_ratio, 1], multiplied onto each
    optimizer group's own base lr every step."""
    if step < config.warmup_steps: return (step + 1) / config.warmup_steps
    if step > config.max_steps: return config.min_lr_ratio
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1, f'There is something wrong with max steps:{config.max_steps}, step:{step}, warmup_steps:{config.warmup_steps}'
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return config.min_lr_ratio + coeff * (1 - config.min_lr_ratio)


# ------ Get LR ---------
def get_lr(step, config):
    """Absolute matrix/muon lr, derived from the ratio -- kept only for the printed log line.
    Since every group now has its own base lr, this number is representative, not literal."""
    return config.matrix_lr * get_lr_ratio(step, config)


###_____________________________________  INSTANCES  ______________________________________

# It a PyTorch function that speeds up float32 matrix multiplications on compatible NVIDIA GPUs by trading off a small amount of numerical precision for significant performance gains.
torch.set_float32_matmul_precision('high')

# config
config = GPT_config()
# gpt.py ships generic placeholders for these three fields since it doesn't know
# about DDP -- fill them in now with what we actually detected above
config.device = device
config.process_rank = ddp_rank
config.num_processes = ddp_world_size

# partial for ratio getter
lr_ratio_getter = partial(get_lr_ratio, config=config)

# make device_type
device_type = "cuda" if config.device.startswith("cuda") else "cpu"  # just to use it at the autocast, etc... and make 'cuda:3' -> 'cuda', 'cuda' -> 'cuda' ,etc...

# encoder
enc = tiktoken.get_encoding(config.tokenizer)

# model
model = GPT(config)
model.to(config.device)
if config.use_compile: model = torch.compile(model)  # compiles the model and makes kernel fusion for the operations
if ddp: model = DDP(model, device_ids=[ddp_local_rank])  # Forward pass / training step → use model (the DDP wrapper) — this is what makes multi-GPU synchronization work
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model / -> Anything else (custom methods, saving checkpoints, accessing .config) → use raw_model — because DDP's wrapper doesn't expose your class's custom stuff directly

# lr getter fn
lr_getter = partial(get_lr, config=config)

# data
train_dl = DL_lite(config, 'Train')
val_dl = DL_lite(config, 'Val')

# optim
optimizer = raw_model.optimizers_config(device_type, MuonAdamW)

# make a logging file
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')
with open(log_file, 'w') as f:  # open for writing to clear the file
    pass


##_____________________________________  TRAINING  ______________________________________

# Training Loop
for step in range(config.training_steps):
    start = time.time()
    last_step = (step == config.training_steps - 1)

    # Validation
    if (step % config.val_after_step == 0 or last_step) and (config.validation):
        model.eval()
        with torch.no_grad():
            val_accum_loss = 0.
            for _ in range(config.val_loss_accum_steps):
                x, y = val_dl.after_batch()
                x, y = x.to(config.device), y.to(config.device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= config.val_loss_accum_steps
                val_accum_loss += loss.detach()
        if ddp: dist.all_reduce(val_accum_loss, op=dist.ReduceOp.AVG)
        val_dl.reset()
        if master_process:
            print(f"Validation loss: {val_accum_loss.item():.4f}")
            with open(log_file, "a") as f: f.write(f"{step} val {val_accum_loss.item():.4f}\n")

            # Save checkpoints for the model
            if (step > 0) and (step % config.checkpoint_after_steps == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f'model_checkpoint_step_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,  # that is the stored config in the model object as an attr
                    'step': step,
                    'val_loss': val_accum_loss.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # Model Sampling
    if step % config.val_after_step == 0 and step > 0 and config.model_sampling and (not config.use_compile):
        model.eval()
        tokens = enc.encode('I am crazy man,')
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.repeat(config.num_sequence, 1)
        x_gen = tokens.to(config.device)

        while x_gen.shape[1] < config.max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x_gen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # take the highest 50 probs
                ix = torch.multinomial(topk_probs, num_samples=1)  # take one sample
                ids = torch.gather(topk_indices, dim=-1, index=ix)
                x_gen = torch.cat((x_gen, ids), dim=1)

        for i in range(config.num_sequence):
            decoded = enc.decode(x_gen[i, :config.max_length].tolist())
            print(f'Rank: {config.process_rank} | Sample{i + 1}: {decoded} ')

    # HellaSwag
    # once in a while evaluate hellaswag
    if (step % config.val_after_step == 0 or last_step) and (not config.use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank: continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f: f.write(f"{step} hella {acc_norm:.4f}\n")

    # Training
    model.train()
    accum_loss = 0
    for mini_step in range(train_dl.tot_mini_batches):
        x, y = train_dl.after_batch()
        x, y = x.to(config.device), y.to(config.device)
        if ddp: model.require_backward_grad_sync = (mini_step == train_dl.tot_mini_batches - 1)  # So that we avoid unnecessary communication during backward () unless it is the last step
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= train_dl.tot_mini_batches
            accum_loss += loss.detach()
        loss.backward()
    if ddp: dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)  # Averaging the Loss Across all the Processes
    lr = get_lr(step, config)  # For the printing, log
    lr_mult = lr_ratio_getter(step)
    # for param_group in optimizer.param_groups:  -----> Because we are using the MuonAdamW optimizer, we don't need to set the lr for each param group, we just pass it to the step function
    #     param_group['lr'] = lr
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # This Makes Normalization for the Norm of the Gradients proportionally, so that -> root(sum(grads**2)) <= 1
    optimizer.step(lr_mult)
    optimizer.zero_grad()
    if device_type == 'cuda': torch.cuda.synchronize()  # so that the cpu don't run the next command while the GPU still hasn't Finished

    # Printings
    end = time.time()
    time_taken = end - start
    tokens_count = train_dl.block_size * train_dl.bs * train_dl.tot_mini_batches
    if master_process:
        print(f'Step: {step:5d} | Loss: {accum_loss.item():.4f} | lr = {lr:.6f} | Grad_Norm = {grad_norm:6f} | Time: {time_taken:.4f}sec | Token/sec: {(tokens_count / time_taken):.3f}')
        with open(log_file, 'a') as f: f.write(f"{step} train {accum_loss.item():.6f}\n")

if ddp: destroy_process_group()  # Clean After the Multi-GPU Process