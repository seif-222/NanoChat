"""
prepare_sft_data.py

Downloads the SFT conversations dataset from Hugging Face and writes it out as a
single JSONL file -- one {"messages": [...]} object per line, exactly the shape
render_conversation() expects.
Dataset: HuggingFaceTB/everyday-conversations-llama3.1-2k
"""
import os
import json
from datasets import load_dataset

# Names
DS_NAME = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
DS_SPLIT = "train_sft"   # this dataset's splits are named train_sft / test_sft, not train / test
# Directories
LOCAL_OUT_FILE = './output/data/sft_conversations.jsonl'
OUT_FILE = os.path.join(os.path.dirname(__file__), LOCAL_OUT_FILE)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)


def clean(row):
    """Keep only role/content per turn -- drop dataset-specific extra fields like 'full_topic', etc..."""
    return [{'role': m['role'], 'content': m['content']} for m in row['messages']]


def validate(messages, row_idx):
    """Catch malformed rows here, with a row number, instead of hundreds of examples deep
    into SFTDataLoader with a bare assert and no idea which conversation broke it."""
    assert len(messages) >= 2, f"row {row_idx}: conversation has fewer than 2 messages"
    assert messages[0]['role'] == 'user', f"row {row_idx}: conversation must start with 'user'"
    for i, m in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        assert m['role'] == expected, f"row {row_idx}, message {i}: role is '{m['role']}', expected '{expected}'"
        assert isinstance(m['content'], str) and m['content'].strip(), f"row {row_idx}, message {i}: empty content"  # Avoid lists,etc... for now and keep it strings


def main():
    """Download the dataset, validate + clean each row, write one JSON line per conversation."""
    hf_data = load_dataset(DS_NAME, split=DS_SPLIT)
    n_written = 0
    with open(OUT_FILE, 'w') as f:
        for i, row in enumerate(hf_data):
            messages = clean(row)
            validate(messages, i)
            f.write(json.dumps({'messages': messages}) + '\n')
            n_written += 1

    print(f'Wrote {n_written} conversations to {OUT_FILE}')


if __name__ == '__main__': main()