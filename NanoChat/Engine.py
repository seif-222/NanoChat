"""
Generation engine for inference on top of the GPT class.

Owns the whole inference loop: session setup (KV cache), prefill,
token-by-token decoding, sampling (temperature, top-k, top-p,
repetition penalty), and a streaming variant of the same loop.
"""

import torch
import torch.nn.functional as F

from Model import KVCache


class InferenceEngine:
    """Runs autoregressive generation on a GPT model, managing one KV-cache session at a time."""
    def __init__(self, model):
        self.model       = model
        self.device      = model.get_model_device()
        self.dtype       = model.get_model_dtype()
        self.seq_length  = model.config.block_size
        self.window_size = model.config.window_size
        self.n_kv_head   = model.config.n_kv_head
        self.n_embd      = model.config.n_embd
        self.n_head      = model.config.n_head
        self.pattern     = model.att_mask_patt
        self.kv_cache    = None    # as a starting point

    def new_session(self, batch_size):
        """Allocate a fresh KV cache sized for the given batch, replacing any previous one."""
        self.kv_cache = KVCache(batch_size, self.pattern, self.window_size, self.seq_length,
                                self.n_kv_head, self.n_embd, self.n_head, self.device, self.dtype)
    @torch.no_grad()
    def prefill(self, tokens):
        """Adding a bunch of tokens at once (prefill)"""
        assert self.kv_cache is not None, "call new_session() before prefill()"
        logits, _ = self.model(tokens, kv_cache=self.kv_cache)
        return logits[:, -1] # last one logits

    @torch.no_grad()
    def decode_step(self, token):
        """The Normal Token by token predictions"""
        logits, _ = self.model(token, kv_cache=self.kv_cache)
        return logits[:, -1]


    def sample(self, logits, temperature=1.0, repetition_penalty=1.0, top_k=None, top_p=None, recent_tokens=None):
        """Turn logits into one token per row: repetition penalty -> temperature -> top-k -> top-p -> multinomial."""
        logits = logits.clone() # so that we don't modify the main one

        # -- Penalty of repetitions --
        if repetition_penalty != 1.0 and recent_tokens is not None:
            for b in range(logits.shape[0]):
                seen = recent_tokens[b].unique()
                logits[b, seen] = torch.where(logits[b,seen] > 0,                   # Condition
                                              logits[b,seen] / repetition_penalty,  # if True
                                              logits[b,seen] * repetition_penalty)  # if False

        # -- Temperature --
        if temperature == 0.0:  return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature                        # it Temp < 1 -> bigger gap (more deterministic), Temp < 1 -> smaller gap (less deterministic)


        # -- Top K --
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)                                      # v -> biggest_values (e.g -> [3.8, 2.9, 2.1]), _ -> indicies
            logits = logits.masked_fill(logits < v[:, [-1]], value=float('-inf')) # [-1] instead of -1 is to just keep dim (as (B,1) | (B,) ) both are fine but just being defensive,  v[:, [-1]] this takes from v the last smallest value (e.g [20,15,.....,2]) it takes the 2, and then we make all of what is smaller than that (-inf)


        # -- Top P --
        if top_p is not None:
            assert 0 <= top_p <= 1, f"Top P: {top_p}, can't be negative or more than 1"
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            removed = cum_probs > top_p                   # True -> removed
            removed[..., 1:] = removed[..., :-1].clone()  # So that we can include the one that exceeded top_p, so also that means repeating the first sign (e.g. F,T,T,T -> F,F,T,T), .clone() is for memory safety
            removed[..., 0]  = False                      # So that at least the first is taken (False)
            sorted_logits = sorted_logits.masked_fill(removed, value=float('-inf'))
            logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_indices, sorted_logits) # Make a new tensor full of -inf (logits shape) -> Scatter the kept values into their original positions

        # -- Sample --
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


    @torch.no_grad()
    def generate(self, promt_tokens, continue_conversation=False, eos_token=None, pad_token_id=0, **sample_kwargs): # eos -or- bos
        """Generate continuations: prefill the prompt, then sample tokens until EOS or context is full."""
        # -- How many promts --
        B = promt_tokens.shape[0]
        # -- create KV-cache --
        if not continue_conversation: self.new_session(B)
        else:
            assert self.kv_cache is not None, "continue_conversation=True but no session exists — call generate() with continue_conversation=False first"
            assert self.kv_cache.n_tokens + promt_tokens.shape[1] <= self.seq_length, f"chunk of {promt_tokens.shape[1]} tokens won't fit: {self.kv_cache.n_tokens}/{self.seq_length} used"
        # -- get logits --
        logits = self.prefill(promt_tokens)
        # -- output --
        out = promt_tokens    # as the beginning
        # -- finish flag --
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        # ---- Loop ----
        for _ in range(self.seq_length):
            # Stop if exceed context
            if self.kv_cache.n_tokens >= self.seq_length:
                print(f"warning: context full at {self.kv_cache.n_tokens}/{self.seq_length}, stopping early")
                break
            # Rest of loop
            next_token = self.sample(logits, recent_tokens=out,  **sample_kwargs)
            if eos_token is not None:
                next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, pad_token_id), next_token)   # So that if we have reached the end for one sequence we pad
            out = torch.cat([out, next_token], dim=-1)
            if eos_token is not None:
                finished |= (next_token.squeeze(-1) == eos_token)  # |= -> OR
                if finished.all():   break            # if all of them are done break

            logits = self.decode_step(next_token)     # so that wwe have logits for the next iteration

        # -- return --
        return out


    @torch.no_grad()
    def stream_generation(self, promt_tokens, continue_conversation=False, eos_token=None, pad_token_id=None, **sample_kwargs):
        """To stream the response while it is made so user don't have to wait"""
        B = promt_tokens.shape[0]
        if not continue_conversation:  self.new_session(B)
        else:
            assert self.kv_cache is not None, "continue_conversation=True but no session exists — call generate() with continue_conversation=False first"
            assert self.kv_cache.n_tokens + promt_tokens.shape[1] <= self.seq_length, f"chunk of {promt_tokens.shape[1]} tokens won't fit: {self.kv_cache.n_tokens}/{self.seq_length} used"
        if eos_token is not None and pad_token_id is None: pad_token_id = eos_token
        logits = self.prefill(promt_tokens)
        recent = promt_tokens
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.seq_length):
            # Stop if exceed context
            if self.kv_cache.n_tokens >= self.seq_length:
                print(f"warning: context full at {self.kv_cache.n_tokens}/{self.seq_length}, stopping early")
                break
            # Rest of loop
            next_token = self.sample(logits, recent_tokens=recent, **sample_kwargs)
            if eos_token is not None:
                next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, pad_token_id), next_token)
            recent = torch.cat([recent, next_token], dim=-1)
            yield next_token

            if eos_token is not None:
                finished |= (next_token.squeeze(-1) == eos_token)
                if finished.all(): break

            logits = self.decode_step(next_token)

