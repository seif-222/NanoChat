"""
prepare_sft_data.py

Downloads and merges the SFT conversation datasets, writing a single JSONL file --
one {"messages": [...]} object per line, exactly the shape render_conversation()
expects.

Sources:
  1. HuggingFaceTB/everyday-conversations-llama3.1-2k   (~2.3k, MIT-ish/permissive)
  2. HuggingFaceH4/no_robots                            (~9.5k, CC-BY-NC-4.0 -- non-commercial)
  3. databricks/databricks-dolly-15k                    (~15k,  CC-BY-SA-3.0 -- commercial ok)
"""
import os
import json
import random
from datasets import load_dataset

# Paths
LOCAL_OUT_FILE = './output/data/sft_conversations.jsonl'
OUT_FILE = os.path.join(os.path.dirname(__file__), LOCAL_OUT_FILE)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

MAX_CONVO_CHARS = 3200    # So that it doesn't exceed the max_sequence_len for the model
SEED = 42

everyday_conversations_split = 'train_sft'
no_robots_split = 'train'
dolly_split = 'train'


def validate(messages, source, row_idx):
    if len(messages) < 2:
        return False, f"{source} row {row_idx}: fewer than 2 messages"
    if messages[0]['role'] != 'user':
        return False, f"{source} row {row_idx}: must start with 'user'"
    for i, m in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        if m['role'] != expected:
            return False, f"{source} row {row_idx}, msg {i}: role '{m['role']}', expected '{expected}'"
        if not isinstance(m['content'], str) or not m['content'].strip():  return False, f"{source} row {row_idx}, msg {i}: empty content"
    return True, None


def too_long(messages):
    return sum(len(m['content']) for m in messages) > MAX_CONVO_CHARS


def process(rows_iter, clean_fn, source):
    """Run clean_fn over raw rows from one dataset, validate, filter by length.
    clean_fn should return a list of {'role', 'content'} dicts, or None to skip
    the row outright (e.g. missing required fields). Returns list of
    {'messages': [...]}."""
    kept, dropped_invalid, dropped_long = [], 0, 0
    for i, row in enumerate(rows_iter):
        messages = clean_fn(row)
        if messages is None:
            dropped_invalid += 1
            continue
        ok, reason = validate(messages, source, i)
        if not ok:
            dropped_invalid += 1
            continue
        if too_long(messages):
            dropped_long += 1
            continue
        kept.append({'messages': messages})
    print(f'[INFO] {source}: kept {len(kept)}, dropped {dropped_invalid} invalid, {dropped_long} too long')
    return kept



def load_everyday_conversations(max_rows=None, split=everyday_conversations_split):
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split=split)
    if max_rows:
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_rows)))
    def clean(row):
        return [{'role': m['role'], 'content': m['content']} for m in row['messages']]
    return process(ds, clean, "everyday-conversations")


def load_no_robots(max_rows=None, split=no_robots_split):
    ds = load_dataset("HuggingFaceH4/no_robots", split=split)
    if max_rows:
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_rows)))
    def clean(row):
        return [{'role': m['role'], 'content': m['content']} for m in row['messages'] if m['role'] != 'system']
    return process(ds, clean, "no_robots")



def load_dolly(max_rows=None, split=dolly_split):
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    if max_rows:
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_rows)))
    def clean(row):
        instruction = (row.get('instruction') or '').strip()
        context = (row.get('context') or '').strip()
        response = (row.get('response') or '').strip()
        if not instruction or not response: return None
        user_turn = f"{instruction}\n\nContext:\n{context}" if context else instruction
        return [{'role': 'user', 'content': user_turn}, {'role': 'assistant', 'content': response}]
    return process(ds, clean, "dolly-15k")


def main():
    """Download + clean each source, merge, shuffle, write one JSON line per conversation."""
    all_conversations = []
    all_conversations += load_everyday_conversations()   # ~2.3k
    all_conversations += load_no_robots()                # ~9.5k
    all_conversations += load_dolly()                    # ~15k
    # Shuffle
    random.seed(SEED)
    random.shuffle(all_conversations)
    # create file
    with open(OUT_FILE, 'w') as f:
        for convo in all_conversations:
            f.write(json.dumps(convo) + '\n')
    # print
    print(f'\n [INFO] Wrote {len(all_conversations)} total conversations to {OUT_FILE}')


if __name__ == '__main__':
    main()