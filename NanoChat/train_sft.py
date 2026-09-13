"""
SFT entrypoint. Loads the pretrained checkpoint, builds an SFTDataLoader from a
conversations .jsonl, derives max_steps/warmup_steps from the actual dataset size,
then reuses train_utils.py's train_model/validate/save_checkpoint verbatim.

Mirrors train.py's main loop structure
"""
import os
import copy
import glob
import time

from dataclasses import asdict
from functools import partial

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from Config import GPT_config
from Model import GPT
from optimizer import MuonAdamW
from DataLoader import SFTDataLoader
from Tokenizer import RustTokenizer
from train_utils import get_lr_ratio, get_lr, validate, save_checkpoint, train_model, validation_hellaswag, sample_chat

###_________________________ DISTRIBUTED TRAINING _______________________________________

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), 'There is no cuda for the device to run ddp'
    init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device         = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank       = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if master_process: print(f'[INFO] Using device: {device}')

device_type = 'cuda' if device.startswith('cuda') else 'cpu'
torch.set_float32_matmul_precision('high')

###_____________________________________  INSTANCES  ______________________________________

# 1. ---- Load Pretrained Checkpoint's Config + Override for SFT ----
_base_cfg = GPT_config()
pretrain_ckpt_path = os.environ.get('SFT_PRETRAIN_CKPT', _base_cfg.sft_pretrained_ckpt)
assert os.path.exists(pretrain_ckpt_path), f"no pretrained checkpoint at {pretrain_ckpt_path} -- set config.sft_pretrained_ckpt"
pretrain_ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)  # weights_only=False -> there are other things in there not just weights
config = copy.deepcopy(pretrain_ckpt['config'])       # start from the exact config the checkpoint was trained with
config.device        = device
config.process_rank  = ddp_rank
config.num_processes = ddp_world_size
config.wandb_run_id  = None                           # this is a new run, not a resume of the pretrain wandb run

# 2. ---- Add SFT attributes to Config ----
attrs = ('sft_data_path', 'sft_pretrained_ckpt', 'sft_log_dir', 'sft_bs', 'sft_grad_accum_mini_batches',
         'sft_val_fraction', 'sft_split_seed', 'sft_desired_epochs', 'sft_warmup_frac', 'sft_lr_scale',
         'sft_val_after_frac', 'sft_checkpoint_after_frac', 'sft_best_ckpt_min_delta')
for f in attrs:
    setattr(config, f, getattr(_base_cfg, f))
config.sft_data_path = os.environ.get('SFT_DATA_PATH', config.sft_data_path)  # Modal override, falls back to Config.py default

# 3. ---- Scale LRs down for FineTuning --
config.matrix_lr      *= config.sft_lr_scale
config.unembedding_lr *= config.sft_lr_scale
config.embedding_lr   *= config.sft_lr_scale
config.scalar_lr      *= config.sft_lr_scale

# 4. ---- Load Tokenizer ----
enc = RustTokenizer.from_directory(config.tokenizer_dir)
assert enc.get_vocab_size() == config.vocab_size, "tokenizer/checkpoint vocab size mismatch"

# 5.---- DataLoaders ----
train_dl = SFTDataLoader(config, 'Train', enc)
val_dl   = SFTDataLoader(config, 'Val',   enc)

# 6. ---- From Data: Training (Steps, Epochs, etc..) ----
steps_per_epoch = max(1, len(train_dl.examples) // (train_dl.bs * train_dl.tot_mini_batches * ddp_world_size))
config.max_steps = max(1, round(steps_per_epoch * config.sft_desired_epochs)) # round to the nearest integer
config.training_steps = config.max_steps
config.warmup_steps = max(1, round(config.max_steps * config.sft_warmup_frac))
config.val_after_step = max(1, round(config.max_steps * config.sft_val_after_frac))
config.checkpoint_after_steps = max(1, round(config.max_steps * config.sft_checkpoint_after_frac))
config.val_loss_accum_steps = max(1, len(val_dl.examples) // (val_dl.bs * ddp_world_size))
if master_process:
    print(f"[INFO] SFT schedule: {len(train_dl.examples)} | Train examples -> {steps_per_epoch} steps/epoch x {config.sft_desired_epochs} epochs  = max_steps = {config.max_steps} | warmup_steps={config.warmup_steps}")
    print(f"[INFO] SFT val_after_step={config.val_after_step} | checkpoint_after_steps={config.checkpoint_after_steps}")

# 7. ---- Model ----
model = GPT(config)
model.load_state_dict(pretrain_ckpt['model'])
model.to(device)
if config.use_compile: model = torch.compile(model)
if ddp: model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# 8. ---- Optimizer + LR ----
optimizer       = raw_model.optimizers_config(device_type, MuonAdamW)
lr_getter       = partial(get_lr, config=config)
lr_ratio_getter = partial(get_lr_ratio, config=config)

# 9. ---- Logging ----
log_dir = os.environ.get('SFT_LOG_DIR', config.sft_log_dir)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')

# self-healing resume: only ever resumes a previously-interrupted SFT run.
def try_resume(path):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_dl.load_state_dict(ckpt['train_dl'])
        return ckpt['step'] + 1
    except Exception as e:
        if master_process: print(f"[INFO] Skipping unloadable checkpoint {path}: {e}")
        return None

start_step = 0
resumed = False
for candidate in sorted(glob.glob(os.path.join(log_dir, 'model_checkpoint_step_*.pt')), reverse=True):
    result = try_resume(candidate)
    if result is not None:
        start_step, resumed = result, True
        break

if resumed:
    if master_process: print(f"[INFO] Resumed SFT at step {start_step}")
else:
    if master_process:
        with open(log_file, 'w') as f: pass

# -- Weights & Biases --
if config.use_wandb and master_process:
    import wandb
    wandb.init(project=config.wandb_project, entity=config.wandb_entity,
               name=(f"{config.wandb_run_name}-sft" if config.wandb_run_name else None),
               mode=config.wandb_mode, config=asdict(config), resume="allow",
               job_type="sft", tags=["sft"])
    # Add some extra useful info
    wandb.config.update({
        "pretrain_ckpt_step": pretrain_ckpt.get('step'),
        "pretrain_val_loss": pretrain_ckpt.get('val_loss'),
        "sft_train_examples": len(train_dl.examples),
        "sft_val_examples": len(val_dl.examples),
    })

##_____________________________________  TRAINING  ______________________________________

best_val_loss = float('inf')
best_step = None

for step in range(start_step, config.training_steps + 1):
    last_step = (step == config.training_steps)

    val_accum_loss = None
    if step % config.val_after_step == 0 or last_step:
        val_accum_loss = validate(model, val_dl, device, device_type, config.val_loss_accum_steps,
                                   ddp, master_process, step, log_file)
        if master_process and config.use_wandb: wandb.log({'val/loss': val_accum_loss.item()}, step=step)

        # separate best-checkpoint file, only overwritten on a real improvement (not val-loss noise)
        if master_process and val_accum_loss.item() < best_val_loss - config.sft_best_ckpt_min_delta:
            best_val_loss = val_accum_loss.item()
            best_step = step
            torch.save({'step': step, 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'train_dl': train_dl.state_dict(), 'config': raw_model.config, 'val_loss': best_val_loss},
                       os.path.join(log_dir, 'model_checkpoint_best.pt'))
            print(f'[INFO] Saving best model at STEP: {step} | VAL_LOSS: {best_val_loss}')
            if config.use_wandb: wandb.run.summary['best_val_loss'] = best_val_loss

        # cheap forgetting-check only (raw text, no chat formatting) -- start/end, not every eval
        if step == 0 or last_step:
            hella_acc = validation_hellaswag(model, enc, device, device_type, ddp, ddp_world_size,
                                              ddp_rank, master_process, step, log_file,
                                              max_examples=200, batch_size=config.hellaswag_eval_batch_size)
            if master_process and config.use_wandb: wandb.log({'val/hellaswag_acc': hella_acc}, step=step)

    if master_process and step > 0 and (step % config.checkpoint_after_steps == 0 or last_step):
        save_checkpoint(log_dir, step,
                         model=raw_model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         train_dl=train_dl.state_dict(),
                         config=raw_model.config,
                         val_loss=(val_accum_loss.item() if val_accum_loss is not None else None))

    # actual chat-format samples -- this is the metric that reflects what SFT trains for
    if master_process and config.model_sampling and (step % config.checkpoint_after_steps == 0 or last_step):
        eval_prompts = ["What's the capital of France?",
                        "Can you help me plan a birthday party?",
                        "Explain photosynthesis in one sentence."]
        samples = sample_chat(model, enc, eval_prompts, max_new_tokens=80, device=device,
                               device_type=device_type, process_rank=ddp_rank)
        if config.use_wandb and config.wandb_log_samples:
            table = wandb.Table(columns=["step", "prompt", "response"])
            for prompt, response in samples: table.add_data(step, prompt, response)
            wandb.log({"train/sft_samples": table}, step=step)

    step_start = time.time()
    accum_loss, lr_mult, grad_norm = train_model(model, train_dl, step, device, device_type,
                                                  optimizer, lr_ratio_getter, config.clip_grad_norm_value, ddp)
    step_time = time.time() - step_start

    if master_process:
        lr = lr_getter(step)
        print(f'Step: {step:5d} | Loss: {accum_loss.item():.4f} | lr = {lr:.6f} | Grad_Norm = {grad_norm:6f} | step_time = {step_time:.2f}s')
        with open(log_file, 'a') as f: f.write(f"{step} train {accum_loss.item():.6f}\n")
        if config.use_wandb:
            wandb.log({'train/loss': accum_loss.item(), 'train/lr': lr, 'train/lr_mult': lr_mult,
                       'train/grad_norm': grad_norm.item(), 'train/epoch': train_dl.epoch,
                       'train/step_time_sec': step_time}, step=step)

if master_process and config.use_wandb: wandb.finish()
if master_process:
    print(f"[INFO] SFT done. Checkpoints in {log_dir}")
    print(f'[INFO] Best_checkpoint is at STEP: {best_step}')
if ddp: destroy_process_group()