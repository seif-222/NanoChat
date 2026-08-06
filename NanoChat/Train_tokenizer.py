"""
train_tokenizer.py

Train a custom BPE tokenizer on a fixed CHARACTER budget streamed from
FineWeb-Edu.
"""
import os
from datasets import load_dataset
from Tokenizer import RustTokenizer
from tqdm import tqdm


# Numbers
MAX_CHARS = int(2e9)   # Train on 2B chars
VOCAB_SIZE = 65536
# Bar
progress_bar = tqdm(total=MAX_CHARS, unit='Chars', unit_scale=True, desc='Tokenizer_training')
# Out_Directory
LOCAL_OUT_DIR = './output/tokenizer'
OUT_DIR = os.path.join(os.path.dirname(__file__), LOCAL_OUT_DIR)
# Dataset
DS_NAME = "HuggingFaceFW/fineweb-edu"
DS_REMOTE_NAME = "sample-10BT"



def data_char_iterator(dataset, max_chars):
    """Yield doc text from a streaming dataset until max_chars is reached."""
    tot_chars = 0
    for doc in dataset:
        txt = doc['text']
        yield txt
        tot_chars += len(txt)
        progress_bar.update(len(txt))
        if tot_chars >= max_chars:
            progress_bar.close()
            print(f"Reached the character budget: {tot_chars:,} chars")
            return
    progress_bar.close()

def main():
    """Stream FineWeb-Edu, train the tokenizer on the char budget, and save it."""
    # get DS
    ds = load_dataset(DS_NAME, DS_REMOTE_NAME, split='train', streaming=True)
    # create Generator (initialization)
    char_iter = data_char_iterator(ds, MAX_CHARS)
    # create and train tokenizer
    tok = RustTokenizer.train_from_iterator(char_iter, VOCAB_SIZE)
    # save it
    tok.save(OUT_DIR)
    print(f'Vocab size = {tok.get_vocab_size()}')

if __name__ == "__main__":  main()