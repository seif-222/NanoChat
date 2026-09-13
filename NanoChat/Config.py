""" Contains All of the HYPERPARAMETERS for Model, Training, etc... """

from dataclasses import dataclass
from typing import Optional
import torch

###_________________________ HYPERPARAMETERS _______________________________________

@dataclass
class GPT_config:
    """All hyperparameters and flags for the model, data, optimizer and training."""

    # -------------------- model architecture --------------------
    n_embd: int = 896                        # hidden/embedding dim
    n_head: int = 7                          # query heads
    n_layer: int = 14                        # transformer blocks
    vocab_size: int = 50304                  # placeholder -- Train.py sets it from the tokenizer (real: 65536)
    block_size: int = 1024                   # context length
    mpl_expantion_term: int = 4              # MLP hidden = n_embd * term

    # -------------------- data --------------------
    bs: int = 16     # MINI-batch            # sequences per GPU mini-batch -> one grad-accum mini-batch = bs x block_size = 16,384 tokens (for current hyper params) [NAMING IS A BIT CONFUSING]
    tokenizer_dir: str = '/data/tokenizer'   # dir holding rustbpe_tokenizer.pkl (RustTokenizer.from_directory)
    tot_bs_for_grad_accum: int = 524288      # tokens before ONE real gradient update (all GPUs x all mini-batches) = 524,288; full training total = this x max_steps
    data_root: str = "/data/fineweb-edu_tokenized"   # pretrain shard dir

    # -------------------- optimizer schedule (shape only, independent of absolute lr) --------------------
    min_lr_ratio: float = 0.1           # cosine floor, as fraction of max
    warmup_steps: int = 150                  # linear warmup length
    max_steps: int = 2200                    # schedule horizon: cosine fully annealed by 2200; steps beyond run at the min_lr floor (0.1 ratio) on purpose

    # -------------------- optimizer: per-group base learning rates --------------------
    matrix_lr: float = 0.02                  # Muon lr for all transformer-block matrices (q/k/v/proj/mlp/ve/ve_gate)
    unembedding_lr: float = 0.004            # lm_head (only used when untied from wte)
    embedding_lr: float = 0.2                # token embedding (wte), ve_te and any per-layer ve tables use half this
    scalar_lr: float = 0.5                   # base rate for x0_lambdas, resid_lambdas uses 1% of this

    # -------------------- optimizer: betas / eps / decay -----(DEFAULTS)---------------
    adamw_beta1: float = 0.8                 # AdamW betas (embeddings/scalars)
    adamw_beta2: float = 0.95                #
    muon_beta1: float = 0.95                 # Muon momentum / EMA betas
    muon_beta2: float = 0.9                  #
    eps: float = 1e-7                        # optimizer epsilon
    weight_decay: float = 0.0                # Muon (matrix) group only
    adamw_weight_decay: float = 0.0          # AdamW groups

    # -------------------- Muon internals --------------------
    red_dim: int = -1                        # rows for the online orthogonalization (-1 = full)
    steps: int = 5                           # Newton-Schulz iterations
    Nesterov: bool = True                    # Nesterov momentum

    # -------------------- training & validation --------------------
    training_steps: int = 3750               # final step index (loop inclusive: range(start, training_steps + 1)) -> resume 2199 runs 2200..3750 at floor LR, final ckpt ...03750.pt
    val_after_step: int = 500                # validate every n steps
    val_loss_accum_steps: int = 5            # val mini-batches averaged
    hellaswag_max_examples: Optional[int] = 1000   # cap for in-training HellaSwag
    hellaswag_eval_batch_size: int = 32      # examples per forward (4x rows)
    checkpoint_after_steps: int = 200        # checkpoint cadence (~37 min at ~11s/step -- max ~$1.85 lost on a crash)
    clip_grad_norm_value: float = 1.0        # grad norm clip

    # -------------------- checkpoint / resume --------------------
    resume_from: Optional[str] = None        # path to a model_checkpoint_step_*.pt to resume from

    # -------------------- device / multi-GPU --------------------
    # placeholders -- the training script fills these in from the real DDP/device
    # detection once it runs, so this file never has to know about torchrun/env vars
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # filled by the training scripts
    process_rank: int = 0                    # DDP rank (0 = master)
    num_processes: int = 1                   # DDP world size

    # -------------------- options / feature flags --------------------
    validation: bool = True                  # run the val loop
    model_sampling: bool = True              # log sample generations
    num_sequence: int = 3                    # samples per sampling call
    max_length: int = 40                     # max sampled tokens
    use_compile: bool = False                # note: breaks HellaSwag/sampling

    # RoPE
    base: int = 100000                       # rotary base frequency

    # GQA
    n_kv_head: int = 7                       # KV heads (=n_head -> MHA)

    # Sliding window
    mask_pattern: str = 'SSSL'               # S=windowed, L=full, tiled over layers
    window_size: int = 256                   # sliding span

    # Value Residual Injection
    n_embd_ve_gate: int = 12                 # gate features from x
    Table_embds_per_layer: bool = False      # per-layer ve tables (else global ve_te)
    ve_per_n_layers: int = 2                 # apply VE every n layers (+last)

    # per-layer scalars
    num_per_layer_scalars: int = 1           # scalar groups per layer

    # SmearGate
    smear_gate_flag: bool = True             # add gated prev-token embedding
    n_embd_smear_gate: int = 24              # gate input features

    # Backout  -> Removing the Intermediate or Middle of output of Transformer from the final layer
    backout_flag: bool = True                # subtract mid-network state at the end
    backout_layer: Optional[int] = None      # default: n_layer//2
    backout_proj_flag: bool = False          # optional linear proj before subtraction

    # Logit Softcap
    logit_softcap: int = 15                  # 15*tanh(x/15)

    # Blending the EmbeddingTable weights with FinalLinearLayer weights
    blend_lm_head_wte_weights: bool = False  # tie lm_head to wte

    # Use Flash Attention
    use_flash_attn_func_flag: bool = False   # flash_attn (lazy import)

    # -------------------- SFT --------------------
    sft_data_path: str = '/data/sft_conversations.jsonl'                 # JSONL of {"messages": [...]}
    sft_pretrained_ckpt: str = '/data/checkpoints/model_checkpoint_step_03750.pt'   # base model checkpoint (the finished 2199->3750 floor-LR continuation)
    sft_log_dir: str = 'log_sft'                                          # SFT logs/checkpoints
    sft_bs: int = 16                                                      # micro batch (conversations)
    sft_grad_accum_mini_batches: int = 1                                  # grad accumulation
    sft_val_fraction: float = 0.01                                        # share of data -> val
    sft_split_seed: int = 1337                                            # deterministic split
    sft_desired_epochs: float = 1                                         # epochs over the train set
    sft_warmup_frac: float = 0.075                                        # warmup as fraction of max_steps
    sft_lr_scale: float = 0.1                                             # scale all base lrs down
    sft_val_after_frac: float = 0.05                                      # validate every x% of steps
    sft_checkpoint_after_frac: float = 0.10                               # save every x% of steps
    sft_best_ckpt_min_delta: float = 0.005                                # min val improvement to overwrite best
    ignore_index: int = -100                                              # CE ignore label (padding/non-assistant)

    # -------------------- RL --------------------
    rl_sft_ckpt: str = '/data/checkpoints_sft_v2/model_checkpoint_best.pt'  # SFT checkpoint RL loads (policy + frozen reference)
    rl_gsm8k_train_path: str = '/data/RL/train.jsonl'                     # {question, answer: float} train pool
    rl_gsm8k_test_path: str = '/data/RL/test.jsonl'                       # held-out questions used as RL val
    rl_log_dir: str = '/data/checkpoints_rl'                              # RL checkpoints + log.txt
    rl_prompts_per_step: int = 4                                          # distinct questions per optimizer step
    rl_k_samples: int = 8                                                 # completions per question (the GRPO group)
    rl_max_new_tokens: int = 256                                          # generation budget per completion
    rl_temperature: float = 0.8                                           # training sampling; val is greedy (temp 0)
    rl_top_p: float = 0.95                                                # training nucleus; unused at val
    rl_lr_scale: float = 0.02                                             # extra scale on the SFT checkpoint's LRs (those already include sft_lr_scale)
    rl_kl_beta: float = 0.02                                              # weight on KL to the frozen SFT reference
    rl_clip_grad_norm: float = 1.0                                        # grad-norm clip
    rl_max_steps: int = 300                                               # final step index (loop inclusive)
    rl_checkpoint_every: int = 25                                         # save every n steps
    rl_val_every: int = 25                                                # greedy val every n steps
    rl_val_questions: int = 32                                            # first n test questions

    # -------------------- Weights & Biases --------------------
    use_wandb: bool = True                    # log to W&B
    wandb_project: str = 'Nanochat'           # project name
    wandb_entity: Optional[str] = 'seif-222-student'   # account/team
    wandb_run_name: Optional[str] = 'Nanochat_SFT_Run'   # cont -> floor-LR continuation of the pretrain run (2199 -> 3750), new wandb run id
    wandb_run_id: Optional[str] = None        # resume id
    wandb_mode: str = 'online'                # 'online' | 'offline' | 'disabled'
    wandb_log_samples: bool = True            # log samples as a table
    wandb_watch_model: bool = False           # grad/param histograms (slower)
    wandb_watch_model_steps: int = 100        # watch log cadence
