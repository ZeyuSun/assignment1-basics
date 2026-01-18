from typing import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Simple text tokenizer that splits text into tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []
        # Add special tokens to vocab so we don't split them
        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = token.encode('utf-8')

        # Create a reverse mapping from token to ID for encoding
        self.token_to_id = {token: id for id, token in self.vocab.items()}

        # Build a mapping from byte pairs to their order for quick lookup during merging
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        # Load vocab and merges from files
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            for line in f:
                token, idx = line.strip().split(b'\t')
                vocab[int(idx)] = token

        merges = []
        with open(merges_filepath, 'rb') as f:
            for line in f:
                a, b = line.strip().split(b'\t')
                merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    def _chunk_at_special_tokens(self, text: str) -> Iterator[str]:
        """Yield chunks of text split at special tokens."""
        if not self.special_tokens:
            yield text
        else:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(map(re.escape, sorted_special_tokens))
            chunks = re.split(f"({pattern})", text)
            for chunk in chunks:
                if chunk:
                    yield chunk

    def _get_bpe_tokens(self, pretoken: bytes) -> Iterator[int]:
        """Apply BPE merges to a pretoken and yield token IDs.

        Tokens are always in vocab. We are merging to get shorter token sequences.
        """
        tokens = [bytes([b]) for b in pretoken]  # TODO: why bytes([b])?
        while len(tokens) > 1:
            # Find the all mergeable pairs
            pairs = set()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    pairs.add(pair)

            if not pairs:
                break

            # Find the best pair to merge
            best_pair = min(pairs, key=lambda pair: self.merge_ranks[pair])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        pretokenize + BPE merge steps
        - No merging for special tokens
        """
        token_ids = []
        chunks = self._chunk_at_special_tokens(text)
        for chunk in chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.token_to_id[chunk.encode('utf-8')])
            else:  # normal text
                for pretoken_match in re.finditer(PAT, chunk):
                    pretoken = pretoken_match.group(0).encode('utf-8')
                    tokens = self._get_bpe_tokens(pretoken)
                    token_ids.extend(self.token_to_id[token] for token in tokens)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into a flat iterator of token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        tokens = [self.vocab[id] for id in ids]
        return b''.join(tokens).decode('utf-8', errors='ignore')