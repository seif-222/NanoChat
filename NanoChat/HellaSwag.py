import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "HellaSwag")


def download_file(url, fname, chunk_size=1024):
    resp  =  requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0)) #  resp.headers -> dictionary of HTTP response headers sent by the server / .get("content-length", 0) -> gets the file size in bytes from the header, returns 0 if header is missing (some servers don't send it)
    with open(fname, 'wb') as f, tqdm(desc=fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024,) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):  #  resp.iter_content(chunk_size=1024) -> yields raw bytes in pieces of 1024 bytes at a time
            size = f.write(data)  # > writes those raw bytes to disk & returns the number of bytes actually written
            bar.update(size) # update how many bytes were just written


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


enc = tiktoken.get_encoding("gpt2")


def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {split} data from {data_url} to {data_filename}")
        download_file(data_url, data_filename)


def render_example(example):
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

    data = {
        'label': label,
        'ctx_tokens' : None,
        'endings_tokens' : [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows = []
    mask_rows = []
    # tok_rows  -> will hold 4 lists, each being [ctx_tokens + one_ending_tokens]
    # mask_rows -> will hold 4 lists, each being [0,0,0,...,1,1,1]
    #              zeros over the context, ones over the ending,  so we can later evaluate loss ONLY on the ending part



    for ending in endings:
        end_tokens = enc.encode(' ' + ending)
        # prepends a space before the ending before tokenizing
        # this is because GPT-2's tokenizer was trained on text where
        # words after a space get different token ids than words at the start
        # e.g. "friend" and " friend" are different tokens in GPT-2's vocab
        # so prepending the space gives the correct tokenization for a word
        # that would naturally follow the context

        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append(len(ctx_tokens)*[0] + len(end_tokens)*[1])

        data['endings_tokens'].append(end_tokens)       # store each ending's tokens for debugging


    max_len = max(len(row) for row in tok_rows)     # needed because the 4 endings have different numbers of tokens, and we need all rows to be the same length to form a tensor

    tokens = torch.zeros((4, max_len), dtype=torch.long) # we have 4 choices in each question
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row) # tokens[i, :len(tok_row)] -> selects row i, columns 0 to len(tok_row) |   = torch.tensor(tok_row) -> fills those positions with this row's token ids
        mask[i, :len(tok_row)] = torch.tensor(mask_row)      # columns beyond len(tok_row) stay as 0 (padding from torch.zeros above)


    return data, tokens, mask, label
    # data   -> debug dict
    # tokens -> (4, max_len) tensor of token ids, one row per candidate ending
    # mask   -> (4, max_len) tensor of 0s and 1s marking ending positions
    # label  -> integer 0-3, which row is the correct ending


def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            # iterates over the file one line at a time
            # each `line` is a string like '{"ctx": "...", "label": 2, "endings": [...]}\n'
            example = json.loads(line)
            yield example
            # yield makes this a GENERATOR function instead of a regular function
            # instead of returning all examples at once (which would load everything into RAM),
            # it pauses here and gives back one example at a time
            # the caller gets one example, processes it, then this resumes for the next one
            # memory efficient for large datasets


@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision('high')     # tells PyTorch to use TF32 precision on Ampere GPUs (A100, RTX 3090, etc.) | TF32 is faster than full FP32 with negligible accuracy loss | 'high' enables TF32, 'highest' forces full FP32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # loads HuggingFace's pretrained GPT-2 and moves to GPU/CPU
    num_correct_norm = 0   # counts correct predictions using length-normalized loss
    num_correct = 0        # counts correct predictions using raw total loss
    num_total = 0          # counts total examples seen


    for example in iterate_examples("val"):
        # calls our generator above, gets one example dict at a time
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        logits = model(tokens).logits # shape: (4, max_len, vocab_size)

        shift_logits = (logits[:,:-1,:]).contiguous()    # .contiguous() -> ensures the tensor's memory layout is sequential after slicing some PyTorch operations require this
                                                         #  removing the last position means we drop the prediction AFTER the sequence ends
        shift_tokens = (tokens[..., 1:]).contiguous()    #  tokens[..., 1:] -> removes the FIRST token, keeps everything from position 1 onward ->  aligns with shift_logits so that position i's logits are compared against token i+1

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))

        shift_tokens = shift_tokens.view(-1)

        loss = F.cross_entropy(shift_logits, shift_tokens, reduction='none')     # reduction='none' -> returns one loss value per position instead of averaging

        reshaped_loss = loss.view(tokens.size(0), -1)

        shift_mask = mask[...,1:].contiguous()

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



if __name__ == "__main__":
# this block ONLY runs if you execute this file directly: python hellaswag.py
# it does NOT run if another file imports this file as a module
# standard Python pattern to separate "runnable script" from "importable module"

    import argparse
    parser = argparse.ArgumentParser()
    # argparse lets you pass command line arguments when running the script
    # e.g. python hellaswag.py -m gpt2-medium -d cuda

    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    # -m is the short flag, --model_type is the long flag, both do the same thing
    # type=str -> convert the input to string
    # default="gpt2" -> if you don't pass -m, it uses "gpt2"
    # help="..." -> shown when you run python hellaswag.py --help

    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")

    args = parser.parse_args()
    # actually reads sys.argv (the command line you typed) and populates args
    # args.model_type -> whatever you passed with -m
    # args.device -> whatever you passed with -d

    evaluate(args.model_type, args.device)
    # calls the main function with the parsed arguments
















