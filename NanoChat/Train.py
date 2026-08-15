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
from dataclasses import asdict
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from Config import GPT_config
from Model import  GPT
from optimizer import MuonAdamW
from DataLoader import CustomDataLoader, PrefetchLoader
from Tokenizer import RustTokenizer
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

# 1. ------ Config & Device -----------
config = GPT_config()
assert config.training_steps >= config.max_steps, f'training_steps: {config.training_steps} must be >= max_steps: {config.max_steps}'
config.device = device                                                 # gpt.py ships generic placeholders for these three fields since it doesn't know, about DDP -- fill them in now with what we actually detected above
config.process_rank = ddp_rank
config.num_processes = ddp_world_size
device_type = "cuda" if config.device.startswith("cuda") else "cpu"    # just to use it at the autocast, etc... and make 'cuda:3' -> 'cuda', 'cuda' -> 'cuda' ,etc...


# 2. ------ Precision --------
torch.set_float32_matmul_precision('high')                             # It a PyTorch function that speeds up float32 matrix multiplications on compatible NVIDIA GPUs by trading off a small amount of numerical precision for significant performance gains.

# 3. ------ Encoder ---------
enc = RustTokenizer.from_directory(config.tokenizer_dir)
config.vocab_size = enc.get_vocab_size()    # same placeholder pattern as device/process_rank/num_processes above -- the real trained tokenizer decides the real vocab size

# 4. ------ Model ----------
model = GPT(config)
model.to(config.device)
if config.use_compile: model = torch.compile(model)      # compiles the model and makes kernel fusion for the operations
if ddp: model = DDP(model, device_ids=[ddp_local_rank])  # Forward pass / training step → use model (the DDP wrapper) — this is what makes multi-GPU synchronization work
raw_model = model.module if ddp else model               # always contains the "raw" unwrapped model / -> Anything else (custom methods, saving checkpoints, accessing .config) → use raw_model — because DDP's wrapper doesn't expose your class's custom stuff directly

# 5. ------ Optimizer --------
optimizer = raw_model.optimizers_config(device_type, MuonAdamW)

# 6. ------ LR ---------
lr_getter = partial(get_lr, config=config)
lr_ratio_getter = partial(get_lr_ratio, config=config)


# 7. ------- DataLoaders ----------
train_dl = CustomDataLoader(config, 'Train')
val_dl = CustomDataLoader(config, 'Val')


# 8. ------ Create Logging File / Resume ---------
log_dir = os.environ.get('LOG_DIR', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')

start_step = 0
if config.resume_from is not None:
    ckpt = torch.load(config.resume_from, map_location=config.device)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    train_dl.load_state_dict(ckpt['train_dl'])
    start_step = ckpt['step'] + 1
    if master_process: print(f"Resumed from {config.resume_from} at step {start_step}")
else:
    with open(log_file, 'w') as f:              # open for writing to clear the file
        pass

train_dl = PrefetchLoader(train_dl, config.device)

# 9. ------ Weights & Biases ---------
if config.use_wandb and master_process:
    import wandb
    wandb.init(project=config.wandb_project, entity=config.wandb_entity,
               name=config.wandb_run_name, mode=config.wandb_mode, config=asdict(config))
    if config.wandb_watch_model: wandb.watch(raw_model, log='gradients', log_freq=config.wandb_watch_model_steps)


##_____________________________________ TRAINING HELPER FUNCTIONS _____________________________________

# 1. -------- Validate ----------
def validate(model, val_dl, device, device_type, val_loss_accum_steps, ddp, master_process, step, log_file, print_flag=True, log=True):
    """Run val_loss_accum_steps validation mini-batches and return the (DDP-averaged) mean loss."""
    model.eval()
    with torch.no_grad():
        val_accum_loss = 0.
        for _ in range(val_loss_accum_steps):
            x, y = val_dl.get_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= val_loss_accum_steps
            val_accum_loss += loss.detach()
    if ddp: dist.all_reduce(val_accum_loss, op=dist.ReduceOp.AVG)
    val_dl.reset()
    if master_process:
        if print_flag: print(f"Validation loss: {val_accum_loss.item():.4f}")
        if log:
            with open(log_file, "a") as f: f.write(f"{step} val {val_accum_loss.item():.4f}\n")
    return val_accum_loss


# 2. -------- Save Checkpoint ----------
def save_checkpoint(log_dir, step, **kwargs):
        """Save model/optimizer/dataloader state (plus any extra kwargs) to a step-numbered checkpoint file."""
        checkpoint_path = os.path.join(log_dir, f'model_checkpoint_step_{step:05d}.pt')
        checkpoint = {'step': step, **kwargs}
        torch.save(checkpoint, checkpoint_path)

# 3. --------- Sampling -----------
def sample(model, enc, num_sequence, max_length, device, device_type, process_rank, beginning_str='I am crazy man,'):
    """Autoregressively sample num_sequence continuations of beginning_str, print and return them."""
    model.eval()
    tokens = enc.encode(beginning_str)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.repeat(num_sequence, 1)
    x_gen = tokens.to(device)

    while x_gen.shape[1] < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x_gen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # take the highest 50 probs
            ix = torch.multinomial(topk_probs, num_samples=1)  # take one sample
            ids = torch.gather(topk_indices, dim=-1, index=ix)
            x_gen = torch.cat((x_gen, ids), dim=1)

    decoded_samples = []
    for i in range(num_sequence):
        decoded = enc.decode(x_gen[i, :max_length].tolist())
        decoded_samples.append(decoded)
        print(f'Rank: {process_rank} | Sample{i + 1}: {decoded} ')
    return decoded_samples


# 4. ------- Hellaswag Validation ---------
def validation_hellaswag(model, enc, device, device_type, ddp, ddp_world_size, ddp_rank, master_process, step, log_file):
    """Score the model on HellaSwag val, DDP-reduce the counts, print/log and return accuracy."""
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank: continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example, enc)
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
    return acc_norm

# 5. ------ Train -------
def train_model(model, train_dl, step, device, device_type, optimizer, lr_ratio_getter, clip_grad_norm_value, ddp):
    """Run one full grad-accum training step (all mini-batches) and return (loss, lr_mult, grad_norm)."""
    model.train()
    accum_loss = 0
    for mini_step in range(train_dl.tot_mini_batches):
        x, y = train_dl.get_batch()
        x, y = x.to(device), y.to(device)
        if ddp: model.require_backward_grad_sync = (mini_step == train_dl.tot_mini_batches - 1)  # So that we avoid unnecessary communication during backward () unless it is the last step
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= train_dl.tot_mini_batches
            accum_loss += loss.detach()
        loss.backward()
    if ddp: dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)  # Averaging the Loss Across all the Processes
    lr_mult = lr_ratio_getter(step)
    # for param_group in optimizer.param_groups:  -----> Because we are using the MuonAdamW optimizer, we don't need to set the lr for each param group, we just pass it to the step function
    #     param_group['lr'] = lr
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)  # This Makes Normalization for the Norm of the Gradients proportionally, so that -> root(sum(grads**2)) <= 1
    optimizer.step(lr_mult)
    optimizer.zero_grad()
    if device_type == 'cuda': torch.cuda.synchronize()  # so that the cpu don't run the next command while the GPU still hasn't Finished
    return accum_loss, lr_mult, grad_norm

##_____________________________________  TRAINING  ______________________________________

# Training Loop
for step in range(start_step, config.training_steps):
    start = time.time()
    last_step = (step == config.training_steps - 1)

    # 1. ---- Validation ----
    val_accum_loss = None
    if (step % config.val_after_step == 0 or last_step) and (config.validation):
        val_accum_loss = validate(model, val_dl, config.device, device_type, config.val_loss_accum_steps,
                                  ddp, master_process, step, log_file)

        # W&B
        if master_process and config.use_wandb:  wandb.log({'val/loss': val_accum_loss.item()}, step=step)

    # Save checkpoints for the model
    if master_process and (step > 0) and (step % config.checkpoint_after_steps == 0 or last_step):
        save_checkpoint(log_dir, step,
                        model=raw_model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        train_dl=train_dl.state_dict(),
                        config=raw_model.config,     # that is the stored config in the model object as an attr
                        val_loss=(val_accum_loss.item() if val_accum_loss is not None else None))

    # 2. ---- Model Sampling ----
    if step % config.val_after_step == 0 and step > 0 and config.model_sampling and (not config.use_compile):
        samples = sample(model, enc, config.num_sequence, config.max_length, config.device, device_type, config.process_rank)
        # W&B
        if master_process and config.use_wandb and config.wandb_log_samples:
            table = wandb.Table(columns=['step', 'sample_idx', 'text'])
            for i, s in enumerate(samples): table.add_data(step, i, s)
            wandb.log({'samples_generations': table}, step=step)

    # 3. ---- HellaSwag ----
    # once in a while evaluate hellaswag
    if (step % config.val_after_step == 0 or last_step) and (not config.use_compile):
        acc_norm = validation_hellaswag(model, enc, config.device, device_type, ddp, ddp_world_size, ddp_rank,
                                         master_process, step, log_file)
        # W&B
        if master_process and config.use_wandb:
            wandb.log({'eval/hellaswag_acc': acc_norm}, step=step)

    # 4. ---- Training ----
    accum_loss, lr_mult, grad_norm = train_model(model, train_dl, step, config.device, device_type,
                                                  optimizer, lr_ratio_getter, config.clip_grad_norm_value, ddp)

    # 5. ---- Printings ----
    end = time.time()
    time_taken = end - start
    tokens_count = train_dl.block_size * train_dl.bs * train_dl.tot_mini_batches
    lr = lr_getter(step)
    if master_process:
        # Prints
        print(f'Step: {step:5d} | Loss: {accum_loss.item():.4f} | lr = {lr:.6f} | Grad_Norm = {grad_norm:6f} | Time: {time_taken:.4f}sec | Token/sec: {(tokens_count / time_taken):.3f}')
        with open(log_file, 'a') as f: f.write(f"{step} train {accum_loss.item():.6f}\n")
        # W&B
        if config.use_wandb:
            wandb.log({'train/loss': accum_loss.item(),     'train/lr': lr, 'train/lr_mult': lr_mult,
                       'train/grad_norm': grad_norm.item(), 'train/tokens_per_sec': tokens_count / time_taken,
                       'train/step_time_sec': time_taken,   'train/epoch': train_dl.epoch}, step=step)


if master_process and config.use_wandb: wandb.finish()
if ddp: destroy_process_group()  # Clean After the Multi-GPU Process