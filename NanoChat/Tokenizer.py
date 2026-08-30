"""
Tokenizer.py

Thin wrapper that trains a BPE tokenizer with rustbpe and serves it for
fast inference via tiktoken. a regex pre-tokenization pattern, a set of special tokens for chat
rendering, and simple save/load to a single pickle file.
"""
import os
from functools import lru_cache
import pickle
import tiktoken
import copy
# special tokens
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

# regex split pattern
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RustTokenizer:
    """Light wrapper: train with rustbpe, encode/decode via tiktoken."""

    def __init__(self, encoder, bos_token):
        """Store the tiktoken encoder and resolve the BOS token id."""
        self.enc = encoder
        self.bos_token_id = self.encode_special(bos_token)   # encode_special defined below

    @classmethod
    def train_from_iterator(cls, txt_iterator, n_vocab):
        """Train a new vocab from a text iterator and wrap it in a tiktoken Encoding."""
        import rustbpe
        # RustBPE tokenizer
        tokenizer = rustbpe.Tokenizer()
        # len & assert
        vocab_size_no_special = n_vocab - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256,  f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        # Train
        tokenizer.train_from_iterator(txt_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # Get from tokenizer:
        pattern = tokenizer.get_pattern()   # regex pattern
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list} # Converts the list of (k, v) pairs into a dict
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset+i for i, name in enumerate(SPECIAL_TOKENS)}
        # Tiktoken tokenizer
        enc = tiktoken.Encoding(name='rustbpe', pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, directory):
        """Load a previously saved tokenizer from disk."""
        pkl_path = os.path.join(directory, "rustbpe_tokenizer.pkl")
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """Load one of tiktoken's built-in pretrained encodings."""
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")   # openai uses "<|endoftext|>" instead of "<|bos|>", same thing but the openai naming is confusing

    def get_vocab_size(self):
        """Return the total vocab size."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """Return the set of special token strings."""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """Decode a single token id to its string form."""
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, txt):
        """Return the token id for a special token string."""
        return self.enc.encode_single_token(txt)

    def get_bos_token_id(self):
        """Return the BOS token id."""
        return self.bos_token_id

    def encode(self, txt,  prepend=None, append=None, num_threads=8):   # [FOR SPECIAL TOKENS] -prepend -> add to beginning of txt, -append -> add to end of txt
        """Encode a string or list of strings into token ids."""
        # Prepend & Append
        if prepend is not None: prepend_id = prepend if isinstance(prepend, int) else  self.encode_special(prepend)
        if append  is not None: append_id  = append  if isinstance(append, int)  else  self.encode_special(append)
        # Encode txt (str)
        if isinstance(txt, str):
            ids = self.enc.encode_ordinary(txt)
            if prepend is not None: ids.insert(0, prepend_id)
            if append  is not None: ids.append(append_id)
        # Encode txt (list)
        elif isinstance(txt, list):
            ids = self.enc.encode_ordinary_batch(txt, num_threads=num_threads)  # if list of strings
            if prepend is not None:
                for single_ids in ids: single_ids.insert(0, prepend_id)
            if append is not None:
                for single_ids in ids: single_ids.append(append_id)

        else: raise ValueError(f"Invalid input type: {type(txt)}, txt must be str or list")

        return ids

    def __call__(self, *args, **kwargs):
        """Shortcut for calling encode() directly on the instance."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        """Return the raw bytes for a single token id."""
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, directory):
        """Pickle the tiktoken encoding to disk."""
        os.makedirs(directory, exist_ok=True)
        pkl_path = os.path.join(directory, "rustbpe_tokenizer.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.enc, f)
            print(f"Saved tokenizer encoding to {pkl_path}")
            
    def render_conversation(self, conversation, max_tokens=None):

        ids, mask = [], []

        def add_tokens(tokens, mask_val):
            if isinstance(tokens, int):
                tokens = [tokens]
            ids.extend(tokens)
            mask.extend([mask_val] * len(tokens))

        # IF first message is a system promt
        if conversation['messages'][0]['role'] == 'system':
            conversation = copy.deepcopy(conversation)  # So that this doesn't happen layer when we dont want it to
            messages = conversation['messages']
            assert messages[1]['role'] == 'user', "System message must be followed by a user message"
            messages[1]['content'] = messages[0]['content'] + "\n\n" + messages[1]['content']
            messages = messages[1:]

        else: messages = conversation['messages']
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # get all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i , message in enumerate(messages):
            # Safety
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            content = message['content']

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # Cut at the max tokens (None = no cap)
        if max_tokens is None:
            return ids, mask
        return ids[:max_tokens], mask[:max_tokens]   # The beginning not [-max_tokens:] because at SFT the beginning is most important

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):

        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation (uncapped; this is just the prompt for priming)
        ids, _ = self.render_conversation(conversation, max_tokens=None)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids









