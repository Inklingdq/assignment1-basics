from __future__ import annotations  # allow forward references in type hints
from functools import reduce
import os
from typing import BinaryIO, Counter, Optional
import regex
import sys
from typing import Optional
import heapq


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Node:
    """
    A node in a doubly linked list.

    Attributes
    ----------
    byte : bytes
        The raw byte sequence held by the node.
    pretoken : str
        The pre-tokenised string corresponding to the byte span.
    prev : Optional[Node]
        Pointer to the previous node.
    next : Optional[Node]
        Pointer to the next node.
    """

    def __init__(
        self,
        byte: bytes,
        pretoken: str,
        prev: Optional[Node] = None,
        next: Optional[Node] = None,
    ) -> None:
        self.byte: bytes = byte
        self.pretoken: str = pretoken
        self.prev: Optional[Node] = prev
        self.next: Optional[Node] = next

    # ── mutators ────────────────────────────────────────────────────────────
    def update_prev(self, prev: Optional[Node]) -> None:
        self.prev = prev

    def update_next(self, next: Optional[Node]) -> None:
        self.next = next

    # ── convenience dunder ──────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Node(byte={self.byte!r}, pretoken={self.pretoken!r})"
    
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(arg: tuple):
    start, end, filename, special_tokens = arg
    local_counter = Counter()
    with open(filename, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for text in regex.split("|".join(map(regex.escape, special_tokens)), chunk):
                for match in regex.finditer(PAT, text):
                    token = match.group()
                    local_counter[token] += 1
    return local_counter

def train_bpe(filename: str, max_vocab:int, special_tokens: list[str])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = 4
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.

    with Pool(num_processes) as p:
        counters = p.map(process_chunk, [(0 if i == 0 else boundaries[i-1], boundaries[i], filename, special_tokens) for i in range(len(boundaries))])
    total = reduce(lambda x, y: x + y, counters)

    pairs = Counter()
    vocab = set()
    pair_index = {}
    for word, frequencey in total.items():
        bs = word.encode("utf-8")
        prev = None
        for i in range(len(bs) - 1):
            a,b = bytes([bs[i]]), bytes([bs[i+1]])
            node = Node(a, word, prev, None)
            if prev:
                prev.update_next(node)
            prev = node
            vocab.add(a)
            pair = (a, b)
            pairs[pair] += frequencey
            if pair not in pair_index:
                pair_index[pair] = [node]
            else:
                pair_index[pair].append(node)
        vocab.add(bytes([bs[-1]]))
        if prev:
            prev.update_next(Node(bytes([bs[-1]]), word, prev, None))
    for special_token in special_tokens:
        vocab.add(special_token)
    merge = []
    heap = [(-f, p) for p, f in pairs.items()]
    heapq.heapify(heap)
    while len(vocab) < max_vocab:
         if len(heap) == 0:
             break
         neg_freq, (a, b) = heapq.heappop(heap)
         if neg_freq == 0:
             break
         if -neg_freq != pairs[(a,b)]:  # ← count mismatched ⇒ stale
             continue # discard and keep popping
         vocab.add(a+b)
         merge.append((a,b))
         if (a,b) not in pair_index:
             continue
         for node in pair_index[(a,b)]:
             if not(node.next) or node.next.byte != b:
                 continue
             node.byte = a+b
             prev = node.prev
             if prev:
                 pairs[(prev.byte, a)] -= total[node.pretoken]
                 if pairs[(prev.byte,a)] == 0:
                     del pairs[(prev.byte,a)]
                 else:
                     heapq.heappush(heap, (-pairs[(prev.byte, a)], (prev.byte, a)))
                 pairs[(prev.byte, a + b)] = total[node.pretoken]
                 heapq.heappush(heap, (-pairs[(prev.byte, a + b)], (prev.byte, a + b)))
                 new_pair = (prev.byte, a + b) 
                 if new_pair not in pair_index:
                     pair_index[new_pair] = [prev]
                 else:
                     pair_index[new_pair].append(prev)
             if node.next.byte != b:
                 print(f"Error finding pairs {a}, {b}")
                 break
             node.next = node.next.next
             if node.next:
                pairs[(b, node.next.byte)] -= total[node.pretoken]
                if pairs[(b, node.next.byte)] == 0:
                    del pairs[(b, node.next.byte)]
                else:
                    heapq.heappush(heap, (-pairs[(b, node.next.byte)], (b, node.next.byte)))
                pairs[(a+b, node.next.byte)] = total[node.pretoken]
                heapq.heappush(heap, (-pairs[(a+b, node.next.byte)], (a+b, node.next.byte)))
                new_pair = (a+b, node.next.byte)
                if new_pair not in pair_index:
                    pair_index[new_pair] = [node]
                else:
                    pair_index[new_pair].append(node)
         del pairs[(a,b)]
         del pair_index[(a,b)]
    vocab_dict = {}
    for word in vocab:
        vocab_dict[len(vocab_dict)] = word
    return (word, merge)

## Usage
if __name__ == "__main__":
    from multiprocessing import Pool
    filename = sys.argv[1] if len(sys.argv) > 1 else "test.txt"
    max_vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    special_tokens = list(sys.argv[3]) if len(sys.argv) > 3 else ["<|endoftext|>"]
    train_bpe(filename, max_vocab, special_tokens)



