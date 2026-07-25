import os
import math
import time
import inspect
import tiktoken
import numpy as np

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn import flash_attn_func
import torch.distributed as dist

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from HellaSwag import render_example, iterate_examples

###_________________________ DISTRIBUTED TRAINING _______________________________________

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), f'There is not cude for the device to run ddp'
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


###_________________________ HYPERPARAMETERS _______________________________________

@dataclass
class GPT_config:
    ### for model
    n_embd: int = 1024
    n_head: int = 8
    n_layer: int = 8
    vocab_size: int = 50304
    block_size: int = 512  # Just a number that is divisible by 2 more times that the standard 50257
    ### for data
    bs: int = 16                          # per-GPU micro batch
    tokenizer: str = 'gpt2'
    tot_bs_for_grad_accum: int = 131072
    # for the training shards
    data_root: str = "/kaggle/input/datasets/seif222/gpt-train-kaggle-zero-to-hero"   # <- point at your actual shards
    ### for optim sched
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10
    max_steps: int = 900
    ### for optim
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    # if using the MuonAdamW
    red_dim: int = -1
    steps: int = 5
    Nesterov: bool = True
    ### for training & Validation
    training_steps: int = 1000                 # = max_steps
    val_after_step: int = 100
    val_loss_accum_steps: int = 5
    ### CheckPoint
    checkpoint_after_steps: int = 50
    ### device
    device: str = device
    ### Multi-Device Training
    process_rank: int = ddp_rank
    num_processes: int = ddp_world_size
    ### Options
    # validaiton
    validation: bool = True
    # sampling
    model_sampling: bool = True
    num_sequence: int = 3
    max_length: int = 40
    #compile
    use_compile: bool = False                # note: compile breaks HellaSwag/gen
    ### RoPE
    base: int = 10000
    ## For GQA
    n_kv_head: int = 2
    ### Sliding window
    mask_pattern: str = 'SSL'
    window_size: int = 32
    ### Value Residual Injection
    n_embd_ve_gate:int = 24
    Table_embds_per_layer: bool = True
    ve_per_n_layers: int = 2    # how much do we want to apply it  (per n layers)
    ### per layer scalars
    num_per_layer_scalars: int = 1  # 1 is the default
    ### SmearGate
    smear_gate_flag: bool = True
    n_embd_smear_gate: int = 24
    ### Backout (Removing the Intermediate or Middle of output of Transformer from the final layer)
    backout_flag: bool = True
    backout_layer: Optional[int] = None  # the default if it was none -> n_layer//2     - NOTE: first_layer_idx = 1
    ### Logit Softcap
    logit_softcap: int = 15

##_____________________________________ OPTIMIZER __________________________________

class MuonAdamW:
    def __init__(self, params, lr, beta2, red_dim, steps=5 , beta1=0.9, Nesterov=True, eps=1e-7):
        self.params = params
        self.lr = lr
        self.beta2 = beta2
        self.red_dim = red_dim
        self.steps = steps
        self.beta1 = beta1
        self.Nesterov = Nesterov
        self.eps = eps
        self.i = 0

    def step(self, lr=None):
        if lr is None: lr = self.lr
        with torch.no_grad():
            for g in self.params:
                use_muon = g.get('use_muon', None)
                for p in g['params']:
                    self.opt_step(p, g['weight_decay'], lr, use_muon)
        self.i += 1

    def zero_grad(self):
        for g in self.params :
            for p in g['params'] :
                if p.grad is not None : p.grad.data.zero_()

    def opt_step(self, p, wd, lr, use_muon=None):
        if use_muon is None:
            use_muon = (p.dim() >= 2)   # fallback for old-style groups

        if not use_muon:
            if not hasattr(p,'grad_avg'): p.grad_avg = torch.zeros_like(p.grad.data)
            if not hasattr(p, 'grad_sqr_avg'): p.grad_sqr_avg = torch.zeros_like(p.grad.data)
            p.grad_avg.lerp_(p.grad, 1 - self.beta1)
            p.grad_sqr_avg.lerp_(p.grad.square(), 1 - self.beta2)
            unbiased_grad_avg = p.grad_avg / (1 - self.beta1 ** (self.i+1) )
            unbiased_grad_sqr_avg = p.grad_sqr_avg / (1 - self.beta2 ** (self.i+1) )
            update = unbiased_grad_avg / (unbiased_grad_sqr_avg.sqrt() + self.eps)
            if wd : update += wd * p.data
            p.data.sub_(lr * update)
        else:
            g = p.grad.data

            # Nesterov momentum
            if not hasattr(p,'grad_avg'): p.grad_avg = torch.zeros_like(g)
            p.grad_avg.lerp_(g, 1 - self.beta1)
            unbiased_grad_avg = p.grad_avg  / (1 - self.beta1 ** (self.i+1) )
            g = g.lerp(unbiased_grad_avg, self.beta1) if self.Nesterov else unbiased_grad_avg

            # Normalization using Norm
            target = g.norm(dim=(-1,-2), keepdim=True) * (g.size(-2)**-0.5)
            row_norm = g.norm(dim=(-1), keepdim=True)
            g = g * (target / (row_norm+self.eps))

            # Frobenius norm
            g /= (g.norm(dim=(-1,-2), keepdim=True) * 1.01 + self.eps)  # 1.01 is just safety so that everything is <1 and <-1 and not =

            # Newton sched
            a, b, c = 3.4445, -4.7750, 2.0315
            for _ in range(self.steps):
                A = g @ g.mT                # mT transposes the last 2 dims
                B = b * A + c * (A @ A)
                g = a * g + B @ g

            # Muon+ normalization
            targ_norm = min(g.size(-2), g.size(-1))  ** 0.5
            current_norm = g.norm(dim=(-1,-2), keepdim=True)
            g = g * (targ_norm / (current_norm+self.eps))

            # Variance Reduction
            v_mean = g.square().mean(dim=self.red_dim, keepdim=True)
            red_dim_sz = g.size(self.red_dim)
            v_norm_sq = v_mean.sum(dim=(-1,-2), keepdim=True) * red_dim_sz
            v_norm = v_norm_sq.sqrt()
            if not hasattr(p, 'v_mean_avg'): p.v_mean_avg = torch.zeros_like(v_mean)
            p.v_mean_avg.lerp_(v_mean, 1 - self.beta2)
            unbiased_v_mean_avg = p.v_mean_avg  / (1 - self.beta2 ** (self.i+1))
            step_sz = (unbiased_v_mean_avg+self.eps).rsqrt()
            scaled_sq_sum = (v_mean * red_dim_sz) * step_sz.square()
            v_norm_new = scaled_sq_sum.sum(dim=(-1,-2), keepdim=True).sqrt()
            final_scale = step_sz * (v_norm / v_norm_new)
            g = g * final_scale

            # Update
            mask = (g * unbiased_grad_avg) >= 0
            update = g + wd * p.data * mask if wd else g
            p.data.sub_(lr * update) # Make it in place


###________________________ CREATING THE RoPE POSITIONAL ENCODING ______________________

class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _create_inv_freq(self, head_dim, base=10000):
        dim = head_dim // 2
        x = torch.arange(dim, dtype=torch.float32)
        inv_freq = base ** (- x / dim )
        return inv_freq

    def _create_angles(self, inv_freq, sequence_length=100):
        positions = torch.arange(sequence_length,device=inv_freq.device,dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        return angles

    def _rotation(self, x, angles):
        x1 = x[...,:x.shape[-1]//2]
        x2 = x[...,x.shape[-1]//2:]

        sin_angles = angles.sin()
        cos_angles = angles.cos()

        # Applying the rotation matrix
        y1 = x1 * cos_angles - x2 * sin_angles
        y2 = x1 * sin_angles + x2 * cos_angles

        return torch.cat([y1, y2], dim=-1)

    def forward(self, x):
        # Get the dims & device
        n_head_dim = x.shape[-1]
        seq_len = x.shape[-2]
        # create angles
        angles = self._create_angles(self._create_inv_freq(n_head_dim, self.config.base).to(x.device),  seq_len)
        angles = angles.unsqueeze(0)
        return self._rotation(x, angles)


####_________________________ NORMALIZATION FUNC & (VE_FLAG BASED ON IDX,ETC... FUNC )___________________________

def norm(x, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight=None,  eps=eps)  # RMS normalization with epsilon for numerical stability

def compute_ve_flag(layer_idx, n_layers, ve_per_n_layers):
    is_alternating =   layer_idx % ve_per_n_layers == (n_layers - 1) % ve_per_n_layers
    return is_alternating or layer_idx == (n_layers - 1)  # is alternating or the last layer


###_________________________ CREATING THE MODEL _______________________________________

class CausalMultiHeadAttention(nn.Module):
  def __init__(self, config, idx):
    super().__init__()
    # Save attr
    self.rope = RoPE(config)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.n_kv_head = config.n_kv_head
    self.window_size = config.window_size
    self.n_embd_ve_gate =  config.n_embd_ve_gate
    self.ATTN_IDX = idx
    self.ve_flag = False
    # assert errors
    assert self.n_head % self.n_kv_head == 0, f"n_head: {self.n_head}, n_kv_head: {self.n_kv_head} Aren't Divisible"
    assert self.n_embd % self.n_head == 0, f"Number of Embeddings: {self.n_embd},  must be a multiple of n_head: {self.n_head}"
    assert self.window_size <= config.block_size , f'Window size: {self.window_size} should be <= block_size: {config.block_size}'
    assert config.ve_per_n_layers <= config.n_layer, f"ve_per_n_layers: {config.ve_per_n_layers} can't be > {config.n_layer}"
    if not config.Table_embds_per_layer : assert self.n_embd_ve_gate <= self.n_embd, f'Number of Embeddings in ve_Gate Residual: {self.n_embd_ve_gate}  should be <= Number of Embeddings: {self.n_embd}'
    # Group and Head size
    self.head_sz = self.n_embd // self.n_head
    self.group_sz = self.n_head // self.n_kv_head
    # qkv
    self.q_proj = nn.Linear(self.n_embd, self.n_embd)
    self.kv_proj = nn.Linear(self.n_embd, 2 * self.head_sz * self.n_kv_head)
    # Projection
    self.c_proj = nn.Linear(self.n_embd,self.n_embd)
    self.c_proj.INIT_SPECIAL_STD = 1
    # Gate Linear & Embedding Table (if True) -> For the Value Embeddings / depending on n chosen for the frequency of ve
    if compute_ve_flag(self.ATTN_IDX, config.n_layer, config.ve_per_n_layers):
        self.ve_flag = True
        if not config.Table_embds_per_layer : self.ve = nn.Linear(self.n_embd_ve_gate, self.head_sz * self.n_kv_head)
        if config.Table_embds_per_layer : self.ve = nn.Embedding(config.vocab_size, self.head_sz * self.n_kv_head)
        self.ve_gate = nn.Linear(self.n_embd_ve_gate, self.n_kv_head)

  def forward(self, x, char, x_ve_input): # x -> (B, T, E) | x_ve_input -> (token_indicies: in case of there is a ve_embd_table_per_layer),
    # shapes                                                            -> (ve_embeddings: in case of there is a Global ve_embd_table)
    b,t,c = x.shape
    # get qkv
    q = self.q_proj(x)
    kv = self.kv_proj(x)
    k, v = torch.chunk(kv, 2, dim=-1)
    # Make them 4D shaped
    q = q.reshape(b, t, self.n_head,    self.head_sz).transpose(1, 2)  # -> (B, n_head, T, head_sz)
    k = k.reshape(b, t, self.n_kv_head, self.head_sz).transpose(1, 2)  # -> (B, n_kv_head, T, head_sz)
    v = v.reshape(b, t, self.n_kv_head, self.head_sz)   # -> (B, T, n_kv_head, head_sz), We are not going to transpose now to make VE work, then we transpose
    # Add the Gate,ve to V
    if self.ve_flag:
        gate = 3 * torch.sigmoid(self.ve_gate(x[:,:,:self.n_embd_ve_gate])) # -> (B,T,C[:n]) -> (B,T,n_kv_head)
        if x_ve_input is not None:  ve = self.ve(x_ve_input).view(b, t, self.n_kv_head, self.head_sz) # (B,T, head_sz * n_kv_head) -> (B,T, n_kv_head, head_sz)
        else: raise ValueError("Missing required inputs: You must provide either 'x_ve' or 'x_ve_indicies' For ValueEmbeddings to Work")
        v = v + gate.unsqueeze(-1) * ve   # unsqueeze -> ( B, T, n_kv_head, 1)
    # Transpose V
    v = v.transpose(1, 2) # -> (B, n_kv_head, T, head_sz)
    # apply RoPE
    q,k = self.rope(q), self.rope(k)
    # Normalization & Rescaling
    q,k = norm(q), norm(k)
    q,k = q * 1.2 , k * 1.2
    # Expanding the k,v to match the number of heads
    k = k.unsqueeze(2).expand(-1, -1, self.group_sz, -1, -1).reshape(b, self.n_head, t, -1)   #  -> (B, n_head, T, head_sz)
    v = v.unsqueeze(2).expand(-1, -1, self.group_sz, -1, -1).reshape(b, self.n_head, t, -1)

    # Run FlashAttention (IF char == L)
    if char == 'L':
        wei = F.scaled_dot_product_attention(q, k, v, is_causal=True) # is_casual=True -> causal mask is the triangular mask we built manually with torch.tril + masked_fill.
        wei = wei.transpose(1, 2).reshape(b, t, c)                    # Combine the heads back

    # Run FlashAttention (IF char == S)
    if char == 'S' :
        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)  # Because the flash_attn_func expects -> (B, T, n_head, head_sz) not (B, n_head, T, head_sz)
        wei = flash_attn_func(q, k, v, causal=True, window_size=(self.window_size,0)) #  (left=window, right=0) -> only look back window tokens, never forward | causal=True is already what blocks future tokens but Both together are redundant but harmless.
        wei = wei.reshape(b, t, c)                                    # Combine the heads back

    # return the projection
    return self.c_proj(wei)


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.INIT_SPECIAL_STD = 1
  def forward(self, x):
    return self.c_proj(F.relu(self.c_fc(x)).square())   # Made the Avtivation as the ReLU^2



class Block(nn.Module):
  def __init__(self, config, idx):
    super().__init__()
    self.attn = CausalMultiHeadAttention(config, idx)
    self.mlp = MLP(config)
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.ln_2 = nn.LayerNorm(config.n_embd)
  def forward(self, x, char, x_ve_input=None):
    x = x + self.attn(self.ln_1(x), char, x_ve_input)
    return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    # save attr
    self.config = config
    # Expand the attention masking pattern & assert & Make mask UpperCase
    config.mask_pattern = config.mask_pattern.upper()
    assert config.n_layer >= len(config.mask_pattern), f'n_layer:{config.n_layer}  must be >= len(mask_pattern):{len(config.mask_pattern)}'
    assert all(c in 'SL' for c in config.mask_pattern), f'All chars in mask pattern: {config.mask_pattern} should be -> S or L'
    self.att_mask_patt = (config.n_layer * config.mask_pattern)[:config.n_layer - 1] + 'L'
    # Per_Layer_Scalars -> assert & repetition_attr, etc...
    self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer, config.num_per_layer_scalars))
    self.x0_lambdas =  nn.Parameter(torch.zeros(config.n_layer, config.num_per_layer_scalars))
    assert config.n_embd % config.num_per_layer_scalars == 0, f"num_per_layer_scalars: {config.num_per_layer_scalars}  Must be <=  n_emb: {config.n_embd}  && n_emb: {config.n_embd} should be divisible by num_per_layer_scalars: {config.num_per_layer_scalars}"
    self.repetition = config.n_embd // config.num_per_layer_scalars
    # Smear_Gate & Assert
    if config.smear_gate_flag:
        assert config.n_embd_smear_gate <= config.n_embd, f'Number of Embeddings in Smear_Gate: {config.n_embd_smear_gate}  should be <= Number of Embeddings: {config.n_embd}'
        self.smear_gate = nn.Linear(config.n_embd_smear_gate, 1) # That is for like Based on who I'm Now, How much do I depend on the token Before me
        self.smear_lambda = nn.Parameter(torch.zeros(1)) # Scaling Factor
    # backout
    if config.backout_flag :
        self.backout_lambda = nn.Parameter(torch.zeros(1))
        self.backout_layer = config.n_layer//2 if config.backout_layer is None else config.backout_layer
        assert 0 < self.backout_layer < config.n_layer, f"backout_layer: {self.backout_layer} must be between 1 and n_layer: {config.n_layer}"
    # Make the Transformer
    transformer_modules = {
        'wte': nn.Embedding(config.vocab_size, config.n_embd),
        'h': nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
        'ln_f': nn.LayerNorm(config.n_embd)
    }
    # add ve_TableEmbedding if True
    if not config.Table_embds_per_layer:
        transformer_modules['ve_te'] = nn.Embedding(config.vocab_size, config.n_embd_ve_gate)
    # make the module
    self.transformer = nn.ModuleDict(transformer_modules)
    # Make the head
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # Make the last Linear later the same as the embedding layer
    self.lm_head.weight = self.transformer.wte.weight
    # Apply Initialization
    self.apply(self._initalization)


  ## Make Initialization function
  def _initalization(self, module):
      if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, 'INIT_SPECIAL_STD'):  std *= (2 * self.config.n_layer) ** -0.5
          nn.init.normal_(module.weight,mean=0., std=std) # torch.nn
          if module.bias is not None:  nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          nn.init.normal_(module.weight, mean=0., std=0.02)


  def forward(self, x, targets=None): # x -> (B, Tokens)
    # Shape & Assert
    B, T = x.shape
    assert T <= self.config.block_size, f'The Context Exceeds the block size {T} > {self.config.block_size}'
    # get the VE_input
    x_ve_input = self.transformer.ve_te(x) if 've_te' in self.transformer else x
    # Tokens -> Embeddings
    x =  self.transformer.wte(x) # x -> (B, tokens, embs)
    # Smear Gate
    if self.config.smear_gate_flag:
        gate = torch.sigmoid(self.smear_gate(x[:,:-1,:self.config.n_embd_smear_gate]))
        prev = F.pad(self.smear_lambda * gate * x[:,:-1], (0,0,1,0)) # can't do x[:,1:] += ... directly — in-place ops corrupt autograd's tape, backward would read mutated values instead of originals → wrong gradients
        x = x + prev
    # Pass Input in Transformer Layers
    x0 = x.clone() # Save x0
    for i, (layer, char) in enumerate(zip(self.transformer.h, self.att_mask_patt)):
        resid_lambdas = self.resid_lambdas[i].repeat(self.repetition)
        x0_lambdas = self.x0_lambdas[i].repeat(self.repetition)
        x = resid_lambdas * x + x0_lambdas * x0
        x = layer(x, char, x_ve_input)
        if self.config.backout_flag:
            if i == (self.backout_layer - 1) : x_backout =  x.clone()
    if self.config.backout_flag: x = x - self.backout_lambda * x_backout
    # Apply LayerNorm
    x = self.transformer.ln_f(x)
    # lm_head -> Get Logits
    logits = self.lm_head(x)
    # Apply logits softcap
    logits = self.config.logit_softcap * torch.tanh(logits/self.config.logit_softcap)
    # Return Loss, Logits
    loss = None
    if targets is not None:  loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    return logits, loss


  def optimizers_config(self, device_type, optimizer=None):
    params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    # Embedding / unembedding must NOT go through Muon's Newton-Schulz step
    no_muon_names = {'transformer.wte.weight', 'lm_head.weight'}

    if optimizer is None:
        decay_params = [p for pn, p in params_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in params_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fuse = fused_available and device_type == 'cuda'
        return torch.optim.AdamW(optim_group, lr=self.config.max_lr,
                                  betas=(self.config.beta1, self.config.beta2),
                                  eps=self.config.eps, fused=use_fuse)
    else:
        muon_params = [p for pn, p in params_dict.items()
                        if p.dim() >= 2 and pn not in no_muon_names]
        adam_params = [p for pn, p in params_dict.items()
                        if p.dim() < 2 or pn in no_muon_names]
        optim_group = [
            {'params': muon_params, 'weight_decay': self.config.weight_decay, 'use_muon': True},
            {'params': adam_params, 'weight_decay': 0.,                       'use_muon': False},
        ]
        return optimizer(optim_group, lr=self.config.max_lr, beta1=self.config.beta1,
                          beta2=self.config.beta2, eps=self.config.eps,
                          steps=self.config.steps, red_dim=self.config.red_dim,
                          Nesterov=self.config.Nesterov)

###______________________________________ MAKE A DATALOADER ________________________________

def load_tokens(filename):
    np_loaded = np.load(filename)
    np_loaded = np_loaded.astype(np.int32) # convert uint16 to int32 before cast to long, otherwise pytorch doesn't like it
    return torch.tensor(np_loaded, dtype=torch.long)




# Make DL lite
class DL_lite:
    def __init__(self, config, split):
        assert split in {'Train', 'Val'}, 'Split must be one of "Train" or "Val"'
        # store attr
        self.block_size= config.block_size
        self.process_rank = config.process_rank
        self.num_processes = config.num_processes
        self.bs = config.bs
        assert config.tot_bs_for_grad_accum % (self.bs * config.num_processes) == 0, f' Total_Batch_size: {config.tot_bs_for_grad_accum} is not divisible by Batch_size: {self.bs} * Num_processes: {self.num_processes}'
        grad_accum = bool(config.tot_bs_for_grad_accum)
        self.tot_mini_batches = config.tot_bs_for_grad_accum // (self.bs * self.block_size * self.num_processes) if grad_accum else 1
        # Loading the data
        data_root = config.data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process: print(f'Found Shards =  {len(self.shards)} | Split = {split} | Total Batch Size = {config.tot_bs_for_grad_accum} | Grad Accum = {grad_accum} | Number Mini Batches = {self.tot_mini_batches} | Num_processes: {self.num_processes}')
        # Make a trackers
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.tr = self.bs * self.block_size * self.process_rank

    def after_batch(self):
        tokens = self.tokens[self.tr : self.tr + self.bs * self.block_size + 1 ]
        self.tr += self.bs * self.block_size * self.num_processes
        x = tokens[:-1].view(self.bs, -1)
        y = tokens[1:].view(self.bs, -1)
        if (self.bs * self.block_size * self.num_processes  + 1 + self.tr) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) # So that we advance to the next shard and if we finish the shards we loop again because of -> %
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.tr = self.bs * self.block_size * self.process_rank
        return x, y

###_________________________________ HELLASWAG FUNCTION _______________written in HellaSwag.py________________________________


def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[:, :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_tokens = shift_tokens.view(-1)
    loss = F.cross_entropy(shift_logits, shift_tokens, reduction='none')
    reshaped_loss = loss.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_loss = reshaped_loss * shift_mask
    sum_loss = torch.sum(masked_shift_loss, dim=-1)
    avg_loss = sum_loss / shift_mask.sum(-1)
    norm_preds = torch.argmin(avg_loss).item()

    return norm_preds

###_________________________________ LR SCHED _______________________________________________

# Creating a LR sched
def get_lr(step, config):
    min_lr = config.max_lr * config.min_lr_ratio
    if step < config.warmup_steps: return config.max_lr/config.warmup_steps * (step+1)
    if step > config.max_steps : return min_lr
    decay_ratio = (step-config.warmup_steps)/(config.max_steps-config.warmup_steps) # make a ratio between 0 and 1 that represent the current section in the cos
    assert 0 <= decay_ratio <= 1, f'There is something wrong with max steps:{config.max_steps}, step:{step}, warmup_steps:{config.warmup_steps}'
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.max_lr - min_lr)


###_____________________________________  INSTANCES  ______________________________________

# It a PyTorch function that speeds up float32 matrix multiplications on compatible NVIDIA GPUs by trading off a small amount of numerical precision for significant performance gains.
torch.set_float32_matmul_precision('high')

# config
config = GPT_config()

# make device_type
device_type = "cuda" if config.device.startswith("cuda") else "cpu" # just to use it at the autocast, etc... and make 'cuda:3' -> 'cuda', 'cuda' -> 'cuda' ,etc...

# encoder
enc = tiktoken.get_encoding(config.tokenizer)

# model
model = GPT(config)
model.to(config.device)
if config.use_compile: model = torch.compile(model) # compiles the model and makes kernel fusion for the operations
if ddp :  model = DDP(model, device_ids=[ddp_local_rank]) # Forward pass / training step → use model (the DDP wrapper) — this is what makes multi-GPU synchronization work
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
with open(log_file, 'w') as f: # open for writing to clear the file
    pass

##_____________________________________  TRAINING  ______________________________________


# Training Loop
for step in range(config.training_steps):
    start = time.time()
    last_step = (step == config.training_steps - 1)

    # Validation
    if (step % config.val_after_step == 0 or last_step) and (config.validation) :
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
                    'config': raw_model.config, # that is the stored config in the model object as an attr
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
            if i % ddp_world_size != ddp_rank:  continue
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
            with open(log_file, "a") as f:  f.write(f"{step} hella {acc_norm:.4f}\n")


    # Training
    model.train()
    accum_loss = 0
    for mini_step in range(train_dl.tot_mini_batches):
        x,y = train_dl.after_batch()
        x,y = x.to(config.device), y.to(config.device)
        if ddp:  model.require_backward_grad_sync = (mini_step == train_dl.tot_mini_batches - 1) # So that we avoid unnecessary communication during backward () unless it is the last step
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= train_dl.tot_mini_batches
            accum_loss += loss.detach()
        loss.backward()
    if ddp:  dist.all_reduce(accum_loss, op= dist.ReduceOp.AVG) # Averaging the Loss Across all the Processes
    lr = lr_getter(step)
    # for param_group in optimizer.param_groups:  -----> Becasue we are using the MuonAdamW optimizer, we don't need to set the lr for each param group, we just pass it to the step function
    #     param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) # This Makes Normalization for the Norm of the Gradients proportionally, so that -> root(sum(grads**2)) <= 1
    optimizer.step(lr)
    optimizer.zero_grad()
    if device_type == 'cuda' : torch.cuda.synchronize() # so that the cpu don't run the next command while the GPU still hasn't Finished

    # Printings
    end = time.time()
    time_taken = end - start
    tokens_count = train_dl.block_size * train_dl.bs * train_dl.tot_mini_batches
    if master_process:
        print(f'Step: {step:5d} | Loss: {accum_loss.item():.4f} | lr = {lr:.6f} | Norm = {norm:6f} | Time: {time_taken:.4f}sec | Token/sec: {(tokens_count / time_taken):.3f}')
        with open(log_file, 'a') as f: f.write(f"{step} train {accum_loss.item():.6f}\n")

if ddp: destroy_process_group() # Clean After the Multi-GPU Process