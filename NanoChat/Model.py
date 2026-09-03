"""
GPT model definition: config, RoPE, attention (GQA + sliding window + value
residual), MLP, transformer block, and the full GPT module.

This file is intentionally self-contained -- it does not know about DDP,
data loading, or the training loop. `device`, `process_rank` and
`num_processes` below are placeholders; the training script overwrites
them on the instantiated config once it has actually detected the
distributed setup (torchrun env vars, GPU availability, etc.).
"""

import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


###________________________ CREATING THE RoPE POSITIONAL ENCODING ______________________

class RoPE(nn.Module):
    """Rotary Positional Embeddings (half-dimension rotation style)."""

    def __init__(self, config):
        super().__init__()
        inv_freq = self._create_inv_freq(config.n_embd // config.n_head, config.base)
        self.sequence_length = config.block_size
        angles = self._create_angles(inv_freq, self.sequence_length).unsqueeze(0)
        self.register_buffer('sin_angles', angles.sin())
        self.register_buffer('cos_angles', angles.cos())

    def _create_inv_freq(self, head_dim, base=10000):
        """Geometric spread of frequencies from 1 down to ~1/base, one per rotation pair."""
        dim = head_dim // 2
        x = torch.arange(dim, dtype=torch.float32)
        inv_freq = base ** (-x / dim)
        return inv_freq

    def _create_angles(self, inv_freq, sequence_length=100):
        """Outer product of positions x frequencies -> one angle per (position, freq-pair)."""
        positions = torch.arange(sequence_length, device=inv_freq.device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        return angles

    def _rotation(self, x, sin_angles, cos_angles):
        """Rotate each (first-half, second-half) channel pair by its precomputed angle."""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]

        # Applying the rotation matrix
        y1 = x1 * cos_angles - x2 * sin_angles
        y2 = x1 * sin_angles + x2 * cos_angles

        return torch.cat([y1, y2], dim=-1)

    def forward(self, x, start_seq=0):
        """Apply RoPE to x, slicing the precomputed angle table to the current seq len."""
        # Get the dims & device
        seq_len = x.shape[-2]
        assert start_seq + seq_len <= self.sequence_length, f"Input Sequence Length for RoPE: (start_seq:{start_seq} + seq_len:{seq_len}){start_seq + seq_len} should be <= The initialized Sequence Length: {self.sequence_length}"
        # create angles
        sin_angles = self.sin_angles[:, start_seq:start_seq+seq_len].to(device=x.device, dtype=x.dtype)   # [:,:seq_len] not [:seq_len] because we unsqueezed so, dim: (1, max_seq_len, dim)
        cos_angles = self.cos_angles[:, start_seq:start_seq+seq_len].to(device=x.device, dtype=x.dtype)
        return self._rotation(x, sin_angles, cos_angles)


###_________________________ HELPER FUNCTIONS ___________________________

# 1. ------- Normalization -------
def norm(x, eps=1e-6):
    """RMSNorm without learnable weight."""
    return F.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)  # RMS normalization with epsilon for numerical stability


# 2. ------- VE Flag Helper --------
def compute_ve_flag(layer_idx, n_layers, ve_per_n_layers):
    """Decide whether this layer should receive Value Residual injection."""
    is_alternating = layer_idx % ve_per_n_layers == (n_layers - 1) % ve_per_n_layers
    return is_alternating or layer_idx == (n_layers - 1)  # is alternating or the last layer


# 3. ------- Sliding Window Attention ---------
def sliding_window_attn(q, k, v, group_size, window_mask):
    """Create Sliding Window Attention where, make the mask then use F.scaled_dot_product_attention()"""
    # Manually expand K,V from n_kv_head -> n_head
    k_exp = k.repeat_interleave(group_size, dim=1)  # (B, n_head, T, head_sz)
    v_exp = v.repeat_interleave(group_size, dim=1)  # (B, n_head, T, head_sz)

    wei = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=window_mask)
    return wei


###_________________________ CREATING THE MODEL _______________________________________


# 1. ---------- Attention ------------

class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head attention with GQA, optional sliding window, RoPE and Value Residual."""

    def __init__(self, config, idx, rope):
        super().__init__()
        # Save attr
        self.rope = rope
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.window_size = config.window_size
        self.n_embd_ve_gate = config.n_embd_ve_gate
        self.ATTN_IDX = idx
        self.ve_flag = False
        self.use_flash_attn_func_flag = config.use_flash_attn_func_flag

        # assert errors
        assert self.n_head % self.n_kv_head == 0, f"n_head: {self.n_head}, n_kv_head: {self.n_kv_head} Aren't Divisible"
        assert self.n_embd % self.n_head == 0, f"Number of Embeddings: {self.n_embd},  must be a multiple of n_head: {self.n_head}"
        assert self.window_size <= config.block_size, f'Window size: {self.window_size} should be <= block_size: {config.block_size}'
        assert config.ve_per_n_layers <= config.n_layer, f"ve_per_n_layers: {config.ve_per_n_layers} can't be > {config.n_layer}"
        if not config.Table_embds_per_layer: assert self.n_embd_ve_gate <= self.n_embd, f'Number of Embeddings in ve_Gate Residual: {self.n_embd_ve_gate}  should be <= Number of Embeddings: {self.n_embd}'

        # Precomputed sliding-window mask, sliced to T at forward time
        rows = torch.arange(config.block_size).unsqueeze(1)
        cols = torch.arange(config.block_size).unsqueeze(0)
        window_mask = (cols <= rows) & ((rows - cols) <= self.window_size)
        self.register_buffer('window_mask', window_mask)

        # Group and Head size
        self.head_sz = self.n_embd // self.n_head
        self.group_sz = self.n_head // self.n_kv_head

        # qkv
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.head_sz * self.n_kv_head, bias=False)

        # Projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.INIT_SPECIAL_STD = 1

        # Gate Linear & Embedding Table (if True) -> For the Value Embeddings / depending on n chosen for the frequency of ve
        if compute_ve_flag(self.ATTN_IDX, config.n_layer, config.ve_per_n_layers):
            self.ve_flag = True
            if not config.Table_embds_per_layer: self.ve = nn.Linear(self.n_embd_ve_gate, self.head_sz * self.n_kv_head, bias=False)
            if config.Table_embds_per_layer: self.ve = nn.Embedding(config.vocab_size, self.head_sz * self.n_kv_head)
            self.ve_gate = nn.Linear(self.n_embd_ve_gate, self.n_kv_head, bias=False)

    def forward(self, x, char, x_ve_input, kv_cache=None):  # x -> (B, T, E)     | x_ve_input -> (token_indicies: in case of there is a ve_embd_table_per_layer),
        """Run one attention block: QKV projection, value-residual injection, RoPE, then
        either full (char='L') or sliding-window (char='S') causal attention."""
        # shapes                                                               -> (ve_embeddings: in case of there is a Global ve_embd_table)
        b, t, c = x.shape

        # get qkv
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = torch.chunk(kv, 2, dim=-1)

        # Make them 4D shaped
        q = q.reshape(b, t, self.n_head, self.head_sz).transpose(1, 2)                      # -> (B, n_head, T, head_sz)
        k = k.reshape(b, t, self.n_kv_head, self.head_sz).transpose(1, 2)                   # -> (B, n_kv_head, T, head_sz)
        v = v.reshape(b, t, self.n_kv_head, self.head_sz)                                   # -> (B, T, n_kv_head, head_sz), We are not going to transpose now to make VE work, then we transpose

        # Add the Gate,ve to V
        if self.ve_flag:
            gate = 3 * torch.sigmoid(self.ve_gate(x[:, :, :self.n_embd_ve_gate]))                         # -> (B,T,C[:n]) -> (B,T,n_kv_head)
            if x_ve_input is not None: ve = self.ve(x_ve_input).view(b, t, self.n_kv_head, self.head_sz)  # (B,T, head_sz * n_kv_head) -> (B,T, n_kv_head, head_sz)
            else: raise ValueError("Missing required inputs: You must provide either 'x_ve' or 'x_ve_indicies' For ValueEmbeddings to Work")
            v = v + gate.unsqueeze(-1) * ve   # unsqueeze -> ( B, T, n_kv_head, 1)

        # Transpose V
        v = v.transpose(1, 2)                 # -> (B, n_kv_head, T, head_sz)

        # apply RoPE
        if kv_cache is not None: q, k = self.rope(q, kv_cache.n_tokens), self.rope(k, kv_cache.n_tokens)
        else:   q, k = self.rope(q), self.rope(k)

        # Normalization & Rescaling
        q, k = norm(q), norm(k)
        q, k = q * 1.2, k * 1.2

        # KV Cache
        if kv_cache is not None:
            k, v = kv_cache.insert(k, v, char, t)
            kv_len = k.shape[2]
            if char == 'L' and kv_len == t:
                wei = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # fused fast path — safe for L, length-independent
            else:
                i = torch.arange(t, device=q.device).unsqueeze(1) + (kv_len - t)
                j = torch.arange(kv_len, device=q.device).unsqueeze(0)
                mask = (j <= i) & (i - j <= self.window_size) if char == 'S' else j <= i
                wei = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            wei = wei.transpose(1, 2).reshape(b, t, c)
        else:
            # Run FlashAttention (IF char == L)
            if char == 'L':
                wei = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)      # enable_gqa=True -> is more efficient that reshaping and expanding ourselves
                wei = wei.transpose(1, 2).reshape(b, t, c)                                          # Combine the heads back

            # Run FlashAttention (IF char == S)
            if char == 'S':
                if self.use_flash_attn_func_flag:
                    from flash_attn import flash_attn_func  # imported lazily: only required if this flag is actually turned on
                    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)               # Because the flash_attn_func expects -> (B, T, n_head, head_sz) not (B, n_head, T, head_sz)
                    wei = flash_attn_func(q, k, v, causal=True, window_size=(self.window_size, 0))  # (left=window, right=0) -> only look back window tokens, never forward | causal=True is already what blocks future tokens but Both together are redundant but harmless.
                    wei = wei.reshape(b, t, c)
                else:
                    wei = sliding_window_attn(q, k, v, self.group_sz, self.window_mask[:t, :t])
                wei = wei.transpose(1, 2).reshape(b, t, c)                                       # Combine the heads back

        # return the projection
        return self.c_proj(wei)


# 2. ---------- MLP ------------

class MLP(nn.Module):
    """Simple MLP with ReLU² activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mpl_expantion_term * config.n_embd,   bias=False)
        self.c_proj = nn.Linear(config.mpl_expantion_term * config.n_embd, config.n_embd, bias=False)
        self.c_proj.INIT_SPECIAL_STD = 1

    def forward(self, x):
        """Up-project, ReLU², down-project."""
        return self.c_proj(F.relu(self.c_fc(x)).square())   # Made the Activation as the ReLU^2


# 3. ---------- Transformer Block ------------

class Block(nn.Module):
    """Transformer block = Attention + MLP (both with pre-norm)."""

    def __init__(self, config, idx, rope):
        super().__init__()
        self.attn = CausalMultiHeadAttention(config, idx, rope)
        self.mlp = MLP(config)

    def forward(self, x, char, x_ve_input=None, kv_cache=None):
        """Pre-norm residual attention, then pre-norm residual MLP."""
        x = x + self.attn(norm(x), char, x_ve_input, kv_cache)
        return x + self.mlp(norm(x))


# ----- KV cache -----------
class KVCache:
    """Pre-allocated K/V buffers for every layer: sliding (S) buffers keep the last window_size tokens,
    full (L) buffers keep up to max_seq_len. One buffer slot per layer, indexed by call order."""
    def __init__(self, batch_size, patterns, window_size, max_seq_len, n_kv_head, n_embd, n_head, device, dtype):
        """Allocate zeroed K/V buffers for each S/L slot and reset all positions/counters."""
        n_s, n_l = patterns.count('S'), patterns.count('L')
        head_dim = n_embd // n_head
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        shape_s = (n_s, batch_size, n_kv_head, window_size + 1, head_dim) # (window_size + 1) --bec-> mask is (rows - cols) <= window_size
        shape_l = (n_l, batch_size, n_kv_head, max_seq_len, head_dim)
        self.k_s = torch.zeros(shape_s, device=device, dtype=dtype)
        self.v_s = torch.zeros(shape_s, device=device, dtype=dtype)
        self.k_l = torch.zeros(shape_l, device=device, dtype=dtype)
        self.v_l = torch.zeros(shape_l, device=device, dtype=dtype)
        self.pos_s = 0
        self.pos_l = 0
        self.n_tokens = 0
        self.smear_cache = None     # The lastest token -> to apply smear gate at token by token generation

    def insert(self, added_k, added_v, pattern_char, n_added_tokens):
        """Write one layer's new K/V into its buffer slot. S buffers roll (only the last
        window_size tokens survive); L buffers just fill up. added_k/added_v -> (B, n_kv_head, t, head_dim)."""
        if pattern_char == 'S':
            k_buf, v_buf, pos, cap = self.k_s, self.v_s, self.pos_s, self.window_size + 1
        else:  # 'L'
            k_buf, v_buf, pos, cap = self.k_l, self.v_l, self.pos_l, self.max_seq_len
            assert (min(self.n_tokens,cap) + n_added_tokens) <= cap, "L-type buffer overflowed — full-attention layer tried to evict, which should never happen"

        valid = min(self.n_tokens, cap)          # tokens actually stored so far
        # old KV
        old_k = k_buf[pos, :, :, :valid]
        old_v = v_buf[pos, :, :, :valid]
        # returned KV
        returned_k = torch.cat([old_k, added_k], dim=-2)
        returned_v = torch.cat([old_v, added_v], dim=-2)
        # new KV
        new_k = returned_k[:, :, -cap:]
        new_v = returned_v[:, :, -cap:]
        k_buf[pos, :, :, :new_k.shape[-2]] = new_k
        v_buf[pos, :, :, :new_v.shape[-2]] = new_v
        # advance position
        if pattern_char == 'S': self.pos_s += 1
        else: self.pos_l += 1
        # Return
        return returned_k, returned_v

    def advance(self, n_added_tokens):
        """Move the global token counter after one full forward pass; buffer positions reset for the next pass."""
        self.n_tokens += n_added_tokens
        self.pos_s = 0
        self.pos_l = 0

# 4. ---------- GPT ------------

class GPT(nn.Module):
    """Main GPT model with optional SmearGate, Backout, Value Residual, GQA, etc."""

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
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer, config.num_per_layer_scalars))
        assert config.n_embd % config.num_per_layer_scalars == 0, f"num_per_layer_scalars: {config.num_per_layer_scalars}  Must be <=  n_emb: {config.n_embd}  && n_emb: {config.n_embd} should be divisible by num_per_layer_scalars: {config.num_per_layer_scalars}"
        self.repetition = config.n_embd // config.num_per_layer_scalars

        # Smear_Gate & Assert
        if config.smear_gate_flag:
            assert config.n_embd_smear_gate <= config.n_embd, f'Number of Embeddings in Smear_Gate: {config.n_embd_smear_gate}  should be <= Number of Embeddings: {config.n_embd}'
            self.smear_gate = nn.Linear(config.n_embd_smear_gate, 1, bias=False)  # That is for like Based on who I'm Now, How much do I depend on the token Before me
            self.smear_lambda = nn.Parameter(torch.zeros(1))                      # Scaling Factor

        # backout
        if config.backout_flag:
            self.backout_lambda = nn.Parameter(torch.zeros(1))
            self.backout_layer = config.n_layer // 2 if config.backout_layer is None else config.backout_layer
            assert 0 < self.backout_layer < config.n_layer, f"backout_layer: {self.backout_layer} must be between 1 and n_layer: {config.n_layer}"
            if config.backout_proj_flag: self.backout_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Make the Transformer
        rope = RoPE(config)    # Make the RoPE
        transformer_modules = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([Block(config, idx, rope) for idx in range(config.n_layer)]),
        }
        # add ve_TableEmbedding if True
        if not config.Table_embds_per_layer:
            transformer_modules['ve_te'] = nn.Embedding(config.vocab_size, config.n_embd_ve_gate)
        # make the module
        self.transformer = nn.ModuleDict(transformer_modules)

        # Make the head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Make the last Linear later the same as the embedding layer
        if config.blend_lm_head_wte_weights: self.lm_head.weight = self.transformer.wte.weight

        # Apply Initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize every parameter group with its own scheme (uniform for QKV/MLP-in,
        zero for projections, high-std normal for wte, near-zero for lm_head, etc.)."""
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # uniform bound giving the same std as Normal(0, n_embd^-0.5)

        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        if not self.config.blend_lm_head_wte_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            nn.init.uniform_(block.attn.q_proj.weight, -s, s)
            nn.init.uniform_(block.attn.kv_proj.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_flag:
                nn.init.uniform_(block.attn.ve.weight, -s, s)
                nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        if 've_te' in self.transformer:
            nn.init.normal_(self.transformer.ve_te.weight, mean=0.0, std=0.02)

        if self.config.smear_gate_flag:
            nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
            nn.init.zeros_(self.smear_lambda)
        if self.config.backout_flag:
            nn.init.constant_(self.backout_lambda, 0.2)
            if self.config.backout_proj_flag:
                nn.init.eye_(self.backout_proj.weight)

        n_layer = self.config.n_layer
        with torch.no_grad():
            for i in range(n_layer):
                self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
                self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

    def get_model_device(self):
        """Return the device the model parameters live on."""
        return next(self.parameters()).device

    def get_model_dtype(self):
        """Return the dtype the model parameters use."""
        return next(self.parameters()).dtype

    def forward(self, x, targets=None, kv_cache=None):  # x -> (B, Tokens)
        """Full forward pass: embed -> smear -> N transformer blocks (with per-layer
        resid/x0 scalars and optional backout) -> final norm -> lm_head -> softcap ->
        optional loss if targets are given."""
        # Shape & Assert
        B, T = x.shape
        assert T <= self.config.block_size, f'The Context Exceeds the block size {T} > {self.config.block_size}'

        # get the VE_input
        x_ve_input = self.transformer.ve_te(x) if 've_te' in self.transformer else x

        # Tokens -> Embeddings
        x = self.transformer.wte(x)  # x -> (B, tokens, embs)
        x = norm(x)

        # Smear Gate
        if self.config.smear_gate_flag:
            smear_cache = None   # This is the cache for the current pass
            if kv_cache is not None:
                smear_cache = kv_cache.smear_cache
                kv_cache.smear_cache = x[:, -1:].clone()   # save the next cache
            if smear_cache is None:
                gate = torch.sigmoid(self.smear_gate(x[:, 1:, :self.config.n_embd_smear_gate]))
                prev = F.pad(self.smear_lambda * gate * x[:, :-1], (0, 0, 1, 0))  # can't do x[:,1:] += ... directly — in-place ops corrupt autograd's tape, backward would read mutated values instead of originals → wrong gradients
                x = x + prev
            else:
                x_prev = torch.cat([smear_cache, x[:,:-1]], dim=1) if T > 1  else smear_cache
                gate = torch.sigmoid(self.smear_gate(x[:, :, :self.config.n_embd_smear_gate]))
                prev = self.smear_lambda * gate * x_prev
                x = x + prev

        # Pass Input in Transformer Layers
        x0 = x.clone()  # Save x0
        for i, (layer, char) in enumerate(zip(self.transformer.h, self.att_mask_patt)):
            # resid_lambdas & x0_lambdas
            resid_lambdas = self.resid_lambdas[i].repeat_interleave(self.repetition)
            x0_lambdas = self.x0_lambdas[i].repeat_interleave(self.repetition)
            x = resid_lambdas * x + x0_lambdas * x0
            # Pass through layer
            x = layer(x, char, x_ve_input, kv_cache)
            # Backout
            if self.config.backout_flag:
                if i == (self.backout_layer - 1): x_backout = x.clone()

        if kv_cache is not None: kv_cache.advance(T)
        # Backout
        if self.config.backout_flag:
            if self.config.backout_proj_flag: x_backout = self.backout_proj(x_backout)
            x = x - self.backout_lambda * x_backout

        # Apply RMSNorm
        x = norm(x)

        # lm_head -> Get Logits
        logits = self.lm_head(x)

        # Apply logits softcap
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        logits = logits.float()

        # Return Loss, Logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=self.config.ignore_index) if targets is not None else None
        return logits, loss

    def optimizers_config(self, device_type, optimizer=None):
        """Build either plain AdamW (single global group, fallback) or MuonAdamW
        (one param group per role, each with its own tuned lr/betas)."""
        named = dict(self.named_parameters())
        cfg = self.config
        dmodel_lr_scale = (cfg.n_embd / 768) ** -0.5

        # anything that must NEVER go through Muon: embeddings, per-layer scalars, lm_head
        embedding_names = {f"{m}.weight" for m, mod in self.named_modules() if isinstance(mod, nn.Embedding)}
        scalar_names = {'resid_lambdas', 'x0_lambdas'}
        if cfg.smear_gate_flag: scalar_names |= {'smear_gate.weight', 'smear_lambda'}
        if cfg.backout_flag:    scalar_names |= {'backout_lambda'}
        no_muon_names = embedding_names | scalar_names | {'lm_head.weight'}

        if optimizer is None:
            decay = [p for pn, p in named.items() if p.dim() >= 2 and pn not in no_muon_names]
            no_decay = [p for pn, p in named.items() if p.dim() < 2 or pn in no_muon_names]
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            return torch.optim.AdamW(
                [{'params': decay, 'weight_decay': cfg.weight_decay},
                 {'params': no_decay, 'weight_decay': 0.}],
                lr=cfg.matrix_lr, betas=(cfg.adamw_beta1, cfg.adamw_beta2),
                eps=cfg.eps, fused=fused_available and device_type == 'cuda')

        groups, assigned = [], set()

        def add_group(names, **kwargs):
            names = sorted(names)   # deterministic (alphabetical) order: the positional momentum list must not depend on set/hash iteration order across processes
            params = [named[n] for n in names if n in named]
            if params:
                groups.append(dict(params=params, **kwargs))
                assigned.update(n for n in names if n in named)

        add_group({'lm_head.weight'}, lr=cfg.unembedding_lr * dmodel_lr_scale, weight_decay=0.01,
                  beta1=0.8, beta2=0.96, eps=1e-10, use_muon=False)
        add_group({'transformer.wte.weight'}, lr=cfg.embedding_lr * dmodel_lr_scale, weight_decay=0.001,
                  beta1=0.8, beta2=0.995, eps=1e-10, use_muon=False)
        add_group({'transformer.ve_te.weight'}, lr=cfg.embedding_lr * dmodel_lr_scale * 0.5, weight_decay=0.01,
                  beta1=0.8, beta2=0.995, eps=1e-10, use_muon=False)
        add_group({'resid_lambdas'}, lr=cfg.scalar_lr * 0.01, weight_decay=0.05,
                  beta1=0.8, beta2=0.95, eps=1e-10, use_muon=False)
        add_group({'x0_lambdas'}, lr=cfg.scalar_lr, weight_decay=0.0,
                  beta1=0.96, beta2=0.95, eps=1e-10, use_muon=False)
        add_group({'smear_gate.weight', 'smear_lambda', 'backout_lambda'}, lr=0.2, weight_decay=0.0,
                  beta1=0.8, beta2=0.95, eps=1e-10, use_muon=False)

        # catch-all: any embedding param not already placed above -- this only fires when
        # Table_embds_per_layer=True, giving each per-layer `ve` embedding table its own
        # AdamW-with-embedding-lr group instead of silently being left out of optimization
        add_group(embedding_names - assigned, lr=cfg.embedding_lr * dmodel_lr_scale * 0.5, weight_decay=0.01,
                  beta1=0.8, beta2=0.995, eps=1e-10, use_muon=False)

        matrix_names = {pn for pn, p in named.items() if p.dim() >= 2 and pn not in no_muon_names}
        add_group(matrix_names, lr=cfg.matrix_lr, weight_decay=cfg.weight_decay, use_muon=True)

        # catches any parameter that silently never gets optimized
        missing = set(named) - assigned - matrix_names
        assert not missing, f"Parameters not assigned to any optimizer group: {missing}"

        return optimizer(groups, adamw_lr=cfg.embedding_lr * dmodel_lr_scale, adamw_wd=cfg.adamw_weight_decay,
                                 muon_lr=cfg.matrix_lr, muon_wd=cfg.weight_decay,
                                 red_dim=cfg.red_dim, steps=cfg.steps,
                                 adamw_beta1=cfg.adamw_beta1, adamw_beta2=cfg.adamw_beta2,
                                 muon_beta1=cfg.muon_beta1, muon_beta2=cfg.muon_beta2,
                                 Nesterov=cfg.Nesterov, eps=cfg.eps)