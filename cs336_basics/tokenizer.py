from __future__ import annotations  # allow forward references in type hints
import json
import sys
from typing import Iterable
from cs336_basics.pretokenization_example import PAT, Node
import regex
import numpy as np


from cs336_basics.common import gpt2_bytes_to_unicode
from tqdm import tqdm 

class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.vocab_inverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_inverse:
                self.vocab_inverse[token_bytes] = len(self.vocab_inverse)
                self.vocab[len(self.vocab_inverse)] = token_bytes
        self.pretokens_to_vocab = {}

    
    @classmethod
    def from_file(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] = None) -> tokenizer:
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # Convert the merges to bytes using the gpt2_byte_decoder
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_index, gpt2_vocab_item in gpt2_vocab.items()
        }

        merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_bpe_merges
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encodes the input text into a list of integers based on the vocabulary.
        """
        pretokens = []
        pretokens_to_tokens = {}
        pattern = f"({'|'.join(map(regex.escape, self.special_tokens))})"
        if not self.special_tokens:
            parts = [text]
        else:
            parts = regex.split(pattern, text)
        for part in parts:
            if part in self.special_tokens:
                pretokens.append(part)
                continue
            for match in regex.finditer(PAT, part):
                token = match.group()
                pretokens.append(token)

        for pretoken in set(pretokens):
            if pretoken in self.pretokens_to_vocab:
                continue
            pretoken_bytes = pretoken.encode("utf-8")
            if pretoken_bytes in self.vocab_inverse:
                self.pretokens_to_vocab[pretoken] = [self.vocab_inverse[pretoken_bytes]]
                # self.pretokens_to_tokens[pretoken] = Node(pretoken_bytes, pretoken, None, None)
                continue
            for byte in pretoken_bytes:
                if pretoken not in pretokens_to_tokens:
                    pretokens_to_tokens[pretoken] = Node(bytes([byte]), pretoken, None, None)
                    prev = pretokens_to_tokens[pretoken]
                else:
                    prev.next = Node(bytes([byte]), pretoken, None, None)
                    prev = prev.next
        # Merge the pretoken nodes based on the merges
        for merge_a, merge_b in self.merges:
            for pretoken in pretokens_to_tokens.keys():
                node = pretokens_to_tokens[pretoken]
                while node and node.next:
                    a,b = node.byte, node.next.byte
                    if a == merge_a and b == merge_b:
                        node.byte = a + b
                        node.next = node.next.next
                    node = node.next
        for pretoken in pretokens_to_tokens:
            node = pretokens_to_tokens[pretoken]
            self.pretokens_to_vocab[pretoken] = []
            while node:
                self.pretokens_to_vocab[pretoken].append(self.vocab_inverse[node.byte])
                node = node.next
        encoding = []
        for pretoken in pretokens:
            if pretoken in self.pretokens_to_vocab:
                encoding.extend(self.pretokens_to_vocab[pretoken])
        return encoding
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for line in tqdm(iterable):
            for token_id in self.encode(line):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of integers back into a string based on the vocabulary.
        """
        tokens = [self.vocab[id] for id in ids]
        return b''.join(tokens).decode("utf-8", errors='replace')
    
if __name__ == "__main__":
    vocab_filepath = sys.argv[1]
    merges_filepath = sys.argv[2]
    special_tokens = sys.argv[3:] if len(sys.argv) > 3 else ["<|endoftext|>"]
    file_path = sys.argv[4] if len(sys.argv) > 4 else None
    tokenizer_instance = tokenizer.from_file(vocab_filepath, merges_filepath, special_tokens)
    if file_path:
        output_path = f"results/{file_path.split('/')[-1].split('.')[0]}_encoded.txt"
        with open(file_path, 'r') as fin, open(output_path, 'w') as fout:
            # text = f.read()
            # original_length = len(text)
            # encoded = tokenizer_instance.encode(text)
            # print(f"Original length: {original_length}, Encoded length: {len(encoded)}. Ratio: {len(encoded) / original_length:.2f}")
            buffer = []
            chunk_size = 100000  # tweak this based on your memory/IO needs

            for token in tokenizer_instance.encode_iterable(fin):
                buffer.append(token)
                if len(buffer) >= chunk_size:
                    fout.write("\n".join(map(str, np.array(buffer).astype(np.uint16))))
                    fout.write("\n")
                    buffer.clear()

            # Write any remaining tokens
            if buffer:
                fout.write("\n".join(map(str, np.array(buffer).astype(np.uint16))))
                fout.write("\n")

