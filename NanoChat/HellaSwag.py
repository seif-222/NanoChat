"""
HellaSwag benchmark: download the dataset, render each example into
(context + ending) token/mask tensors, and score which of the 4 candidate
endings the model finds most likely.
"""

import os
import json
import requests
from tqdm import tqdm
import torch
from torch.nn import functional as F


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "HellaSwag")


def pick_device():
    """Auto-detect cuda > mps > cpu."""
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def download_file(url, fname, chunk_size=1024):
    """Stream a file to disk with a progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))         # resp.headers -> dictionary of HTTP response headers sent by the server / .get("content-length", 0) -> gets the file size in bytes from the header, returns 0 if header is missing (some servers don't send it)
    with open(fname, 'wb') as f, tqdm(desc=fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024,) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):  # resp.iter_content(chunk_size=1024) -> yields raw bytes in pieces of 1024 bytes at a time
            size = f.write(data)                               # > writes those raw bytes to disk & returns the number of bytes actually written
            bar.update(size)                                   # update how many bytes were just written


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split):
    """Download the given HellaSwag split into DATA_CACHE_DIR if not already present."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {split} data from {data_url} to {data_filename}")
        download_file(data_url, data_filename)


def render_example(example, enc):
    """Turn one HellaSwag example into (4, max_len) token/mask tensors."""
    # example is a dict from the jsonl file, looks like:
    # {
    #   "ctx": "The woman picked up the ball and",
    #   "label": 2,           <- index of the correct ending (0,1,2, or 3)
    #   "endings": [          <- always exactly 4 possible completions
    #       "threw it away",
    #       "sat down quietly",
    #       "threw it to her friend",
    #       "read a book"
    #   ]
    # }

    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {'label': label, 'ctx_tokens': None, 'endings_tokens': [],}

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows  = []       # -> will hold 4 lists, each being [ctx_tokens + one_ending_tokens]
    mask_rows = []       # -> will hold 4 lists, each being [0,0,0,...,1,1,1],  zeros over the context, ones over the ending,  so we can later evaluate loss ONLY on the ending part

    for ending in endings:
        end_tokens = enc.encode(' ' + ending)             # prepends a space before the ending before tokenizing, that would naturally follow the context
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append(len(ctx_tokens)*[0] + len(end_tokens)*[1])
        data['endings_tokens'].append(end_tokens)         # store each ending's tokens for debugging

    max_len = max(len(row) for row in tok_rows)           # needed because the 4 endings have different numbers of tokens, and we need all rows to be the same length to form a tensor

    tokens = torch.zeros((4, max_len), dtype=torch.long)  # we have 4 choices in each question
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)  # tokens[i, :len(tok_row)] -> selects row i, columns 0 to len(tok_row) |   = torch.tensor(tok_row) -> fills those positions with this row's token ids
        mask[i, :len(mask_row)]   = torch.tensor(mask_row) # columns beyond len(tok_row) stay as 0 (padding from torch.zeros above)

    return data, tokens, mask, label                      # data   -> debug dict
                                                          # tokens -> (4, max_len) tensor of token ids, one row per candidate ending
                                                          # mask   -> (4, max_len) tensor of 0s and 1s marking ending positions
                                                          # label  -> integer 0-3, which row is the correct ending


def render_examples_batch(examples, enc):
    """Same idea as render_example, but for a *list* of examples at once: all of their
    (ctx + ending) rows get packed into one padded (4*len(examples), max_len) tensor pair,
    so a whole group of examples can be scored in a single forward pass instead of one
    forward pass per example."""
    tok_rows, mask_rows, labels = [], [], []

    for example in examples:
        ctx_tokens = enc.encode(example["ctx"])
        for ending in example["endings"]:
            end_tokens = enc.encode(' ' + ending)
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append(len(ctx_tokens) * [0] + len(end_tokens) * [1])
        labels.append(example["label"])

    max_len = max(len(row) for row in tok_rows)              # padding target is now the max over the WHOLE group, not just one example
    tokens = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
    mask = torch.zeros((len(tok_rows), max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, labels        # tokens/mask -> (4*len(examples), max_len) | labels -> list of len(examples) ints, in example order


def get_most_likely_rows_batch(tokens, mask, logits, num_examples):
    """Batched version of get_most_likely_row: scores all 4*num_examples rows in one shot,
    then reshapes to (num_examples, 4) and returns the argmin candidate index per example."""
    shift_logits = (logits[:, :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_tokens = shift_tokens.view(-1)
    loss = F.cross_entropy(shift_logits, shift_tokens, reduction='none')
    reshaped_loss = loss.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_loss = reshaped_loss * shift_mask
    avg_loss = masked_shift_loss.sum(-1) / shift_mask.sum(-1)   # -> (4*num_examples,)
    avg_loss = avg_loss.view(num_examples, 4)                    # -> (num_examples, 4), row order matches render_examples_batch
    return torch.argmin(avg_loss, dim=1)                         # -> (num_examples,) predicted candidate per example, still on-device


def iterate_examples(split):
    """Yield HellaSwag examples one at a time (downloads the split first if needed)."""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:    # iterate over (e.g '{"ctx": "...", "label": 2, "endings": [...]}\n')
            example = json.loads(line)
            yield example


def get_most_likely_row(tokens, mask, logits):
    """Return the index (0-3) of the ending with the lowest average per-token loss."""
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


@torch.no_grad()
def evaluate(model, enc, device=None, device_type=None, batch_size=32):
    """Score `model` on the full HellaSwag val split and print running accuracy.
    Batched (batch_size examples -> one forward pass) instead of one forward pass per
    example -- the unbatched version paid a host-device sync on every single example,
    which is what made this loop so slow."""
    device = device if device is not None else pick_device()
    device_type = device_type if device_type is not None else ('cuda' if device.startswith('cuda') else 'cpu')
    model.to(device)
    torch.set_float32_matmul_precision('high')     # tells PyTorch to use TF32 precision on Ampere GPUs (A100, RTX 3090, etc.) | TF32 is faster than full FP32 with negligible accuracy loss | 'high' enables TF32, 'highest' forces full FP32
    num_correct_norm = 0                           # counts correct predictions using length-normalized loss
    num_total = 0                                  # counts total examples seen
    buffer = []                                    # examples waiting to be batched together

    def score_buffer():
        """Run one forward pass over everything currently sitting in `buffer`, update the
        running totals, then clear it."""
        nonlocal num_correct_norm, num_total
        tokens, mask, labels = render_examples_batch(buffer, enc)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):   # was missing before -- ran this whole eval in full fp32
            logits, _ = model(tokens)
        preds = get_most_likely_rows_batch(tokens, mask, logits, len(buffer))
        labels_t = torch.tensor(labels, device=preds.device)
        num_correct_norm += (preds == labels_t).sum().item()   # one sync for the whole batch, not one per example
        num_total += len(buffer)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")
        buffer.clear()

    for example in iterate_examples("val"):
        buffer.append(example)
        if len(buffer) == batch_size: score_buffer()
    if buffer: score_buffer()   # score the leftover partial batch (fewer than batch_size examples)

    return num_correct_norm / num_total


if __name__ == "__main__":
    import argparse
    from Tokenizer import RustTokenizer
    from Model import GPT

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="path to a model_checkpoint_step_*.pt file")
    parser.add_argument("-t", "--tokenizer_dir", type=str, default="./output/tokenizer", help="directory containing rustbpe_tokenizer.pkl")
    parser.add_argument("-d", "--device", type=str, default=None, help="device to use (default: auto-detect)")
    args = parser.parse_args()

    device = args.device if args.device is not None else pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device)  # dict saved by train.py: {'model':, 'config':, 'step':, 'val_loss':}
    model = GPT(ckpt['config'])
    model.load_state_dict(ckpt['model'])

    enc = RustTokenizer.from_directory(args.tokenizer_dir)

    evaluate(model, enc, device=device)