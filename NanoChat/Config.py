""" Contains All of the HYPERPARAMETERS for Model, Training, etc... """

from dataclasses import dataclass
from typing import Optional
import torch

###_________________________ HYPERPARAMETERS _______________________________________

@dataclass
class GPT_config:
    """All hyperparameters and flags for the model, data, optimizer and training."""

    # -------------------- model architecture --------------------
    n_embd: int = 896
    n_head: int = 7
    n_layer: int = 14
    vocab_size: int = 50304                 # Just a number that is divisible by 2 more times than the standard 50257
    block_size: int = 1024
    mpl_expantion_term: int = 4             # That is at the MLP linear layers   layer1 : (n_embd -> mpl_expantion_term * n_embd), ....

    # -------------------- data --------------------
    bs: int = 16                            # per-GPU micro batch
    tokenizer_dir: str = '/data/tokenizer'   # directory holding rustbpe_tokenizer.pkl (RustTokenizer.from_directory)
    tot_bs_for_grad_accum: int = 524288
    data_root: str = "/data/fineweb-edu_tokenized"   # <- point at your actual shards

    # -------------------- optimizer schedule (shape only, independent of absolute lr) --------------------
    min_lr_ratio: float = 0.1
    warmup_steps: int = 150
    max_steps: int = 2200

    # -------------------- optimizer: per-group base learning rates --------------------
    matrix_lr: float = 0.02              # Muon lr for all transformer-block matrices (q/k/v/proj/mlp/ve/ve_gate)
    unembedding_lr: float = 0.004        # lm_head (only used when untied from wte)
    embedding_lr: float = 0.2            # token embedding (wte), ve_te and any per-layer ve tables use half this
    scalar_lr: float = 0.5               # base rate for x0_lambdas, resid_lambdas uses 1% of this

    # -------------------- optimizer: betas / eps / decay -----(DEFAULTS)---------------
    adamw_beta1: float = 0.8
    adamw_beta2: float = 0.95
    muon_beta1: float = 0.95
    muon_beta2: float = 0.9
    eps: float = 1e-7
    weight_decay: float = 0.0            # applies to the Muon (matrix) group only
    adamw_weight_decay: float = 0.0           # For AdamW (default 0)

    # -------------------- Muon internals --------------------
    red_dim: int = -1
    steps: int = 5
    Nesterov: bool = True

    # -------------------- training & validation --------------------
    training_steps: int = 2200
    val_after_step: int = 500
    val_loss_accum_steps: int = 5
    hellaswag_max_examples: Optional[int] = 1000
    hellaswag_eval_batch_size: int = 32        # examples per forward pass during hellaswag eval (=4x rows/pass); higher = fewer host-device syncs
    checkpoint_after_steps: int = 200
    clip_grad_norm_value: float = 1.0     # clip grad_norm during training

    # -------------------- checkpoint / resume --------------------
    resume_from: Optional[str] = None       # path to a model_checkpoint_step_*.pt file to resume training from

    # -------------------- device / multi-GPU --------------------
    # placeholders -- the training script fills these in from the real DDP/device
    # detection once it runs, so this file never has to know about torchrun/env vars
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_rank: int = 0
    num_processes: int = 1

    # -------------------- options / feature flags --------------------
    validation: bool = True
    model_sampling: bool = True
    num_sequence: int = 3
    max_length: int = 40
    use_compile: bool = False               # note: compile breaks HellaSwag/gen

    # RoPE
    base: int = 100000

    # GQA
    n_kv_head: int = 7

    # Sliding window
    mask_pattern: str = 'SSSL'
    window_size: int = 256

    # Value Residual Injection
    n_embd_ve_gate: int = 12
    Table_embds_per_layer: bool = False
    ve_per_n_layers: int = 2                # how much do we want to apply it  (per n layers)

    # per-layer scalars
    num_per_layer_scalars: int = 1          # 1 is the default

    # SmearGate
    smear_gate_flag: bool = True
    n_embd_smear_gate: int = 24

    # Backout  -> Removing the Intermediate or Middle of output of Transformer from the final layer
    backout_flag: bool = True
    backout_layer: Optional[int] = None     # the default if it was none -> n_layer//2     - NOTE: first_layer_idx = 1
    backout_proj_flag: bool = False         # This create a linear_proj before backout_step:  x = x - self.backout_lambda * (x_backout) <-add proj to this

    # Logit Softcap
    logit_softcap: int = 15

    # Blending the EmbeddingTable weights with FinalLinearLayer weights
    blend_lm_head_wte_weights: bool = False

    # Use Flash Attention
    use_flash_attn_func_flag: bool = False


    # -------------------- SFT --------------------
    sft_data_path: str = '/data/sft_conversations.jsonl'
    sft_pretrained_ckpt: str = '/data/checkpoints/model_checkpoint_step_02199.pt'
    sft_log_dir: str = 'log_sft'
    sft_bs: int = 16                             # micro batch, in conversations not tokens
    sft_grad_accum_mini_batches: int = 1         # bump if you want gradient accumulation
    sft_val_fraction: float = 0.02
    sft_split_seed: int = 1337
    sft_desired_epochs: float = 3
    sft_warmup_frac: float = 0.075          # 7.5% of max_steps, middle of the 5-10% rule of thumb
    sft_lr_scale: float = 0.1               # applied to matrix/unembedding/embedding/scalar lr
    sft_val_after_frac: float = 0.05        # validate every 5% of total SFT steps (pretrain's val_after_step=500 is way off scale for a ~4-5k step SFT run)
    sft_checkpoint_after_frac: float = 0.10 # periodic checkpoint every 10% of total SFT steps
    sft_best_ckpt_min_delta: float = 0.001  # only overwrite model_checkpoint_best.pt if val loss improves by at least this much -- val loss on a small eval slice is noisy, this stops it re-saving on noise
    ignore_index: int = -100                     # in the F.cross_entropy in GPT.forward

    # -------------------- Weights & Biases --------------------
    use_wandb: bool = True
    wandb_project: str = 'Nanochat'
    wandb_entity: Optional[str] = 'seif-222-student'
    wandb_run_name: Optional[str] = None    # None -> let wandb auto-generate a name
    wandb_run_id: Optional[str] = None
    wandb_mode: str = 'online'              # 'online' | 'offline' | 'disabled'
    wandb_log_samples: bool = True          # log generated text samples as a wandb.Table
    wandb_watch_model: bool = False         # log gradient/param histograms via wandb.watch (slower, opt-in)
    wandb_watch_model_steps: int = 100