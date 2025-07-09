from functools import reduce
import os
from typing import BinaryIO, Counter
import regex
import sys

special_tokens = ["<|endoftext|>"]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    start, end, filename = arg
    local_counter = Counter()
    with open(filename, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for text in regex.split("|".join(map(regex.escape, special_tokens)), chunk):
                for match in regex.finditer(PAT, text):
                    token = match.group()
                    local_counter[token] += 1
    return local_counter
## Usage
if __name__ == "__main__":
    from multiprocessing import Pool
    filename = sys.argv[1] if len(sys.argv) > 1 else "test.txt"
    max_vocab = sys.argv[2] if len(sys.argv) > 2 else 1000
    num_processes = 4
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.

    with Pool(num_processes) as p:
        counters = p.map(process_chunk, [(0 if i == 0 else boundaries[i-1], boundaries[i], filename) for i in range(len(boundaries))])
    total = reduce(lambda x, y: x + y, counters)
        
    pairs = Counter()
    vocab = set()
    for word, frequencey in total:
        bytes = list(word.encode("utf-8"))
        for i in range(len(bytes) - 1):
            vocab.add(bytes[i])
            pairs[(bytes[i], bytes[i + 1])] += frequencey
        vocab.add(bytes[-1])

    for special_token in special_tokens:
         if special_token not in vocab:
              vocab[special_token] = len(vocab)
    merge = []
    while len(vocab) < max_vocab:
         a,b = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
         vocab.add(a+b)
         merge.append((a,b))
         del pairs (a,b)

