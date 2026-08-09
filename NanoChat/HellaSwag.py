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
        mask[i, :len(tok_row)]   = torch.tensor(mask_row) # columns beyond len(tok_row) stay as 0 (padding from torch.zeros above)

    return data, tokens, mask, label                      # data   -> debug dict
                                                          # tokens -> (4, max_len) tensor of token ids, one row per candidate ending
                                                          # mask   -> (4, max_len) tensor of 0s and 1s marking ending positions
                                                          # label  -> integer 0-3, which row is the correct ending


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
def evaluate(model, enc, device=None):
    """Score `model` on the full HellaSwag val split and print running accuracy."""
    device = device if device is not None else pick_device()
    model.to(device)
    torch.set_float32_matmul_precision('high')     # tells PyTorch to use TF32 precision on Ampere GPUs (A100, RTX 3090, etc.) | TF32 is faster than full FP32 with negligible accuracy loss | 'high' enables TF32, 'highest' forces full FP32
    num_correct_norm = 0                           # counts correct predictions using length-normalized loss
    num_correct = 0                                # counts correct predictions using raw total loss
    num_total = 0                                  # counts total examples seen

    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example, enc)
        tokens = tokens.to(device)
        mask   = mask.to(device)
        logits, _ = model(tokens)  # shape: (4, max_len, vocab_size)

        shift_logits = (logits[:, :-1, :]).contiguous()   #  removing the last position means we drop the prediction AFTER the sequence ends
        shift_tokens = (tokens[..., 1:]).contiguous()     #  tokens[..., 1:] -> removes the FIRST token, keeps everything from position 1 onward ->  aligns with shift_logits so that position i's logits are compared against token i+1
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_tokens = shift_tokens.view(-1)

        loss = F.cross_entropy(shift_logits, shift_tokens, reduction='none')  # reduction='none' -> returns one loss value per position instead of averaging

        reshaped_loss = loss.view(tokens.size(0), -1)
        shift_mask = mask[..., 1:].contiguous()
        masked_shift_loss = reshaped_loss * shift_mask

        sum_loss = torch.sum(masked_shift_loss, dim=-1)
        avg_loss = sum_loss / shift_mask.sum(-1)   # shift_mask.sum(dim=1) -> counts how many 1s are in each row,  i.e. how many ending tokens each candidate has, without this, longer endings would always have higher total loss

        preds = torch.argmin(sum_loss).item()
        norm_preds = torch.argmin(avg_loss).item()

        num_total += 1
        num_correct += int(preds == label)
        num_correct_norm += int(norm_preds == label)

        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")

        if num_total < 10:
            # only prints detailed debug info for the first 9 examples
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            # \n inside the string is a newline character
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                # avg_loss[i] -> indexes into the 4-element tensor to get loss for ending i
                # .item() -> converts tensor element to plain Python float
            print(f"predicted: {norm_preds}, actual: {label}")

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