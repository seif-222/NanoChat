"""
Sharded token data loader. Each shard is a .npy file of token ids; CustomDataLoader
walks through them sequentially per-process (offset by process_rank so each
DDP rank reads a different slice), wrapping back to shard 0 once exhausted.
"""

import os
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
            print(f'Found Shards =  {len(self.shards)} | Split = {split} | Total Batch Size = {config.tot_bs_for_grad_accum} | Grad Accum = {grad_accum} | Number Mini Batches = {self.tot_mini_batches} | Num_processes: {self.num_processes}')

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