"""
gsm8k_data.py

Downloads GSM8K and writes a JSONL of {"question": str, "answer": float}
pairs -- the RL prompt pool. Not the SFT conversation format: RL only needs
a question and a gold number.

GSM8K answers are free-text that end in "#### <number>". The reasoning is
dropped (RL scores 0/1 on the number, it does not imitate the writeup).
"""
import os
import re
import json
from datasets import load_dataset

DS_NAME = "openai/gsm8k"
DS_CONFIG = "main"

# Paths
LOCAL_OUT_DIR = './output/data/gsm8k'
OUT_DIR = os.path.join(os.path.dirname(__file__), LOCAL_OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


def extract_final_answer(answer_text):
    """GSM8K answers always end with '#### <number>' -- pull that number out as a float."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    assert match, f"couldn't find '#### <number>' in: {answer_text!r}"
    return float(match.group(1).replace(',', ''))


def write_split(hf_split, out_filename):
    n_written = 0
    with open(out_filename, 'w') as f:
        for row in hf_split:
            answer = extract_final_answer(row['answer'])
            f.write(json.dumps({'question': row['question'], 'answer': answer}) + '\n')
            n_written += 1
    print(f'[INFO] Wrote {n_written} problems to {out_filename}')


def main():
    hf_data = load_dataset(DS_NAME, DS_CONFIG)
    write_split(hf_data['train'], os.path.join(OUT_DIR, 'train.jsonl'))
    write_split(hf_data['test'],  os.path.join(OUT_DIR, 'test.jsonl'))  # used as the RL val split


if __name__ == '__main__': main()
