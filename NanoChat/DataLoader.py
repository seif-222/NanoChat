"""
Data loaders for pretraining and SFT.

- CustomDataLoader: streams token ids from sharded .npy files on disk (memmap).
  Each DDP rank reads a different offset, advances through shards, and wraps
  around when exhausted.

- PrefetchLoader: thin wrapper that prefetches + pins the next batch on a
  background thread for the pretraining loader.

- SFTDataLoader: loads a JSONL of chat conversations into memory, tokenizes
  each with render_conversation(), and yields padded (x, y) batches with
  ignore_index on every non-assistant / padding position. Same public
  interface as CustomDataLoader so train_model()/validate() work unchanged.

SFT data is small enough to keep in RAM as a list of tensors; a flat memmap
does not help when you need per-batch padding and a supervision mask.
Both loaders are DDP-aware: every rank sees the same shuffle order and takes
a disjoint slice of it.
"""

import os
import json
import random
import threading
import queue
import numpy as np
import torch


# 1. -------- Load_Token Helper Function ---------
def load_tokens(filename):
    """Memory-map one .npy token shard from disk (lazy paging, avoids loading the whole shard into RAM)."""
    return np.load(filename, mmap_mode='r')


# 2. --------- Make DL lite ------------
class CustomDataLoader:
    """Minimal sharded token dataloader: yields (x, y) mini-batches for language
    modeling, advancing through shards on disk and wrapping around at the end."""

    def __init__(self, config, split):
        assert split in {'Train', 'Val'}, 'Split must be one of "Train" or "Val"'
        # store attr
        self.block_size = config.block_size
        self.process_rank = config.process_rank
        self.num_processes = config.num_processes
        self.bs = config.bs
        assert config.tot_bs_for_grad_accum % (self.bs * self.block_size * config.num_processes) == 0, f'Total_Batch_size: {config.tot_bs_for_grad_accum} is not divisible by bs: {self.bs} * block_size: {self.block_size} * num_processes: {config.num_processes}'
        grad_accum = bool(config.tot_bs_for_grad_accum)
        self.tot_mini_batches = config.tot_bs_for_grad_accum // (self.bs * self.block_size * self.num_processes) if grad_accum else 1

        # Loading the data
        data_root = config.data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.process_rank == 0:  # only the master process prints, same condition as the DDP setup's master_process
            print(f'[INFO] Found Shards =  {len(self.shards)} | Split = {split} | Total Batch Size = {config.tot_bs_for_grad_accum} | Grad Accum = {grad_accum} | Number Mini Batches = {self.tot_mini_batches} | Num_processes: {self.num_processes}')

        # Make a trackers
        self.reset()

    def reset(self, epoch=1, shard_idx=0, token_count=None):
        """Rewind to shard 0 and this process's starting offset within it."""
        self.epoch = epoch
        self.current_shard = shard_idx
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.token_count = self.bs * self.block_size * self.process_rank if token_count is None else token_count

    def state_dict(self):
        """Everything needed to resume training """
        return {'epoch': self.epoch, 'shard_idx': self.current_shard, 'token_count': self.token_count}

    def load_state_dict(self, state_dict):
        self.reset(state_dict['epoch'], state_dict['shard_idx'], state_dict['token_count'])


    def get_batch(self):
        """Return the next (x, y) mini-batch, advancing the read pointer and
        rolling over to the next shard (wrapping to shard 0 at the end) once
        the next read would run past the end of the current one."""
        tokens = self.tokens[self.token_count: self.token_count + self.bs * self.block_size + 1]
        tokens = tokens.astype(np.int32)  # convert uint16 to int32 before cast to long, otherwise pytorch doesn't like it
        tokens = torch.tensor(tokens, dtype=torch.long)
        self.token_count += self.bs * self.block_size * self.num_processes
        x = tokens[:-1].view(self.bs, -1)
        y = tokens[1:].view(self.bs, -1)
        if (self.bs * self.block_size * self.num_processes + 1 + self.token_count) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)  # So that we advance to the next shard and if we finish the shards we loop again because of -> %
            if self.current_shard == 0:  self.epoch += 1
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.token_count = self.bs * self.block_size * self.process_rank
        return x, y


# 3. --------- Prefetching Wrapper ------------
class PrefetchLoader:
    """Wraps a CustomDataLoader, prefetching/pinning the next batch on a background thread. Proxies other attrs through to the wrapped loader."""

    def __init__(self, dataloader, device, num_prefetch=2):
        self.dl = dataloader
        self.device = device
        self.q = queue.Queue(maxsize=num_prefetch)
        self.stop_flag = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_flag:
            x, y = self.dl.get_batch()
            x, y = x.pin_memory(), y.pin_memory()
            self.q.put((x, y))

    def get_batch(self):
        x, y = self.q.get()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def __getattr__(self, name):
        return getattr(self.dl, name)


# 4. --------- SFT DataLoader ------------
class SFTDataLoader:
    """Batches whole conversations (not a flat token stream) for SFT."""

    def __init__(self, config, split, tokenizer):
        assert split in {'Train', 'Val'}, 'Split must be one of "Train" or "Val"'
        self.block_size       = config.block_size
        self.bs               = config.sft_bs              # per-rank micro batch, same convention as CustomDataLoader.bs
        self.process_rank     = config.process_rank
        self.num_processes    = config.num_processes
        self.tot_mini_batches = config.sft_grad_accum_mini_batches
        self.ignore_index     = config.ignore_index        # Used to Ignore User tokens
        self.pad_token_id     = 0                          # Fills empty positions so the batch has equal length (for Input)

        self.examples, n_read, n_truncated = self._load_examples(config, split, tokenizer)
        assert len(self.examples) >= self.bs * self.num_processes, f"only {len(self.examples)} examples for split={split}, need >= bs*num_processes={self.bs * self.num_processes}"

        if self.process_rank == 0:  # Master prints
            if n_truncated: print(f"[INFO] SFTDataLoader[{split}]: {n_truncated}/{n_read} conversations truncated to block_size={self.block_size}")
            print(f"[INFO] SFTDataLoader[{split}]: {len(self.examples)} examples | BS={self.bs} x {self.num_processes} RANKS | tot_mini_batches={self.tot_mini_batches}")

        self.reset()

    def _load_examples(self, config, split, tokenizer):
        """Read the jsonl, split it, tokenize every conversation into a shifted (x, y) pair
        with ignore_index everywhere except assistant tokens. Returns (examples, n_read, n_truncated)."""
        with open(config.sft_data_path, 'r') as f:                     # File objects iterate line-by-line (not char-by-char like strings).
            lines = [json.loads(line) for line in f if line.strip()]   # Each `line` is a full JSON string, e.g.:  '{"messages": [{"role": "user", "content": "Hi"}, ...]}\n'
                                                                       # json.loads then turns that string into a Python dict.

        rng = random.Random(config.sft_split_seed)
        rng.shuffle(lines)                                             # deterministic shuffle before splitting -> fixed random partition, not first-N/last-N
        n_val = max(1, int(len(lines) * config.sft_val_fraction))      # How much lines (examples) for the validation
        split_lines = lines[:n_val] if split == 'Val' else lines[n_val:]
        assert len(split_lines) > 0, f"No examples left for split={split} -- check sft_val_fraction / dataset size"

        examples, n_truncated = [], 0
        for convo in split_lines:
            ids, mask = tokenizer.render_conversation(convo, max_tokens=self.block_size + 1)
            if len(ids) > self.block_size: n_truncated += 1
            if len(ids) < 2: continue                          # degenerate, nothing to predict
            y_mask = torch.tensor(mask[1:], dtype=torch.bool)  # mask shifted the same way y is shifted vs x
            if not y_mask.any(): continue                      # truncation ate every assistant token -- zero supervision, wastes a batch slot
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            y = torch.where(y_mask, y, torch.full_like(y, self.ignore_index))
            examples.append((x, y))

        return examples, len(split_lines), n_truncated

    def reset(self, epoch=1, round=0):
        """Rewind to the start of an epoch. Re-derives the shuffle order from `epoch` alone
        (shared across ranks -- same seed, same order), then seeks this rank's offset within
        it. (epoch, round) fully determines position, so that's all state_dict needs."""
        self.epoch = epoch
        self._shuffle()
        self.round = round

    def _shuffle(self):
        rng = random.Random(1000 + self.epoch)  # same seed on every rank -> identical order across ranks
        self.order = list(range(len(self.examples)))
        rng.shuffle(self.order)

    def state_dict(self): return {'epoch': self.epoch, 'round': self.round}

    def load_state_dict(self, state_dict): self.reset(state_dict['epoch'], state_dict['round'])

    def get_batch(self):
        """Return this rank's next (x, y) mini-batch of `bs` conversations, right-padded to
        the longest one in the batch."""
        stride = self.bs * self.num_processes
        steps_per_epoch = len(self.order) // stride  # floor -- drop the ragged remainder
        if self.round >= steps_per_epoch:
            self.epoch += 1
            self._shuffle()
            self.round = 0

        start = self.round * stride + self.bs * self.process_rank
        batch_idxs = self.order[start: start + self.bs]
        self.round += 1
        batch = [self.examples[i] for i in batch_idxs]

        max_len = max(x.shape[0] for x, _ in batch)
        x_out = torch.full((self.bs, max_len), self.pad_token_id, dtype=torch.long)
        y_out = torch.full((self.bs, max_len), self.ignore_index, dtype=torch.long)
        for row, (x, y) in enumerate(batch):
            L = x.shape[0]
            x_out[row, :L] = x
            y_out[row, :L] = y

        return x_out, y_out