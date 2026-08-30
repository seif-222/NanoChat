"""
Training entrypoint. Detects DDP/device, builds the model + MuonAdamW
optimizer + data loaders, then runs the training loop with periodic
validation, sampling, and HellaSwag eval.

Run with:  python train.py                                  (single GPU/CPU)
       or: torchrun --standalone --nproc_per_node=N train.py (multi-GPU DDP)
"""

import os
import time
import glob

from dataclasses import asdict
from functools import partial

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from Config import GPT_config
from Model import  GPT
from optimizer import MuonAdamW
from DataLoader import CustomDataLoader, PrefetchLoader
from Tokenizer import RustTokenizer
from train_utils import get_lr_ratio, get_lr, validate, validation_hellaswag, save_checkpoint, sample, train_model


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

# Helper_fn
def try_resume(path):
    """Try loading a full checkpoint. Returns start_step on success, None if it fails (so we can try an older one)."""
    try:
        ckpt = torch.load(path, map_location=config.device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_dl.load_state_dict(ckpt['train_dl'])
        return ckpt['step'] + 1
    except Exception as e:
        if master_process: print(f"Skipping unloadable checkpoint {path}: {e}")
        return None

# Variables
start_step = 0
resumed = False
resume_path = config.resume_from

# If / Else
if resume_path is not None:
    start_step = try_resume(resume_path)
    assert start_step is not None, f"Explicit resume_from={resume_path} failed to load"
    resumed = True
else:     # Self-heal: try newest → oldest checkpoints until one loads (handles preemption / bad leftovers)
    for candidate in sorted(glob.glob(os.path.join(log_dir, 'model_checkpoint_step_*.pt')), reverse=True):
        result = try_resume(candidate)
        if result is not None:
            start_step, resume_path, resumed = result, candidate, True
            break

# If / Else
if resumed:
    if master_process: print(f"Resumed from {resume_path} at step {start_step}")
else:
    with open(log_file, 'w') as f:  pass

train_dl = PrefetchLoader(train_dl, config.device)

# 9. ------ Weights & Biases ---------
if config.use_wandb and master_process:
    import wandb
    wandb.init(project=config.wandb_project, entity=config.wandb_entity,
               name=config.wandb_run_name, mode=config.wandb_mode, config=asdict(config),
               id=config.wandb_run_id, resume="allow")
    if config.wandb_watch_model: wandb.watch(raw_model, log='gradients', log_freq=config.wandb_watch_model_steps)

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
                                         master_process, step, log_file, max_examples=config.hellaswag_max_examples,
                                         batch_size=config.hellaswag_eval_batch_size)
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