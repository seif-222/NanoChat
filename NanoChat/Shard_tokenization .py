"""
shard_tokenization.py

Stream FineWeb-Edu, tokenize each document with a previously-trained
RustTokenizer, and write fixed-size uint16 .npy shards to disk until a
target total token count is reached. CPU-bound work.
"""
import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from Tokenizer import RustTokenizer


# Numbers
SHARD_SIZE =  int(1e8)    # 100M Tokens
TARGET_TOKENS = int(4e9)  # 4B tokens
# Out_Directory
LOCAL_OUT_DIR = './output/data/fineweb-edu_tokenized'
OUT_DIR = os.path.join(os.path.dirname(__file__), LOCAL_OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
# Dataset
DS_NAME = "HuggingFaceFW/fineweb-edu"
DS_REMOTE_NAME = "sample-10BT"

# Tokenizer
tok = RustTokenizer.from_directory('./output/tokenizer')
bos = tok.get_bos_token_id()
# Load DS
hf_data = load_dataset(DS_NAME, name=DS_REMOTE_NAME, split='train', streaming=True)


def tokenize(doc):
    """Tokenize one HF dataset row into a uint16 token id array."""
    tokens = tok.encode(doc['text'], prepend=bos)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'Dictionary is too large for uint16'
    return tokens_np.astype(np.uint16)

def write_shards(filename, token_np):
    """Save one shard's token ids to disk as a .npy file."""
    np.save(filename, token_np)


def main():
    """Tokenize FineWeb-Edu in parallel and write it out as fixed-size shards."""
    # nprocs
    nprocs = max(1, os.cpu_count()//2)                           # mp.Pool(nprocs) — creates a pool of nprocs worker processes, each a full independent Python interpreter
    # numbers
    shard_idx = 0
    token_count = 0
    tot_token_count = 0
    # empty array
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)  # allocates a 1D array of 100M uint16 slots WITHOUT initializing the values (garbage values)
    # progress_bar
    progress_bar = None

    with mp.Pool(processes=nprocs) as pool:                      # context manager, automatically terminates and cleans up at the end -even if there was error-

        for tokens in pool.imap(tokenize, hf_data, chunksize=16):
            # break if reached target token count
            if tot_token_count >= TARGET_TOKENS: break

            tot_token_count += len(tokens)
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count: token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None: progress_bar = tqdm(total=SHARD_SIZE, unit='Tokens', desc=f'shard {shard_idx}')
                progress_bar.update(len(tokens))
            else:
                # Path
                split = 'Val' if shard_idx == 0 else 'Train'
                filename = os.path.join(OUT_DIR, f'Shard_{split}-{shard_idx:05d}')
                # Remainder
                remainder = SHARD_SIZE - token_count
                all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
                progress_bar.update(remainder)
                # save
                write_shards(filename, all_tokens_np)
                # reset
                shard_idx += 1
                progress_bar = None
                # Populate the next shard with the leftovers
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remainings as last shard
        if token_count != 0:
            split = "Val" if shard_idx == 0 else "Train"
            filename = os.path.join(OUT_DIR, f"Shard_{split}-{shard_idx:05d}")
            write_shards(filename, all_tokens_np[:token_count])



if __name__ == '__main__':  main()