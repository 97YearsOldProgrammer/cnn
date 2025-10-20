import gzip
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

#####################
## UTILITY SECTION ##
#####################

def get_filepointer(filename):
	fp = None
	if   filename.endswith('.gz'): fp = gzip.open(filename, 'rt')
	elif filename == '-':          fp = sys.stdin
	else:                          fp = open(filename)
	return fp

def read_fasta(filename):

	name = None
	seqs = []

	fp = get_filepointer(filename)

	for line in fp:
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()
     
############################
#### Glove Tokenisation ####
############################

BASE_PAIR = ("A", "C", "G", "T")
BASE2IDX = {base: idx for idx, base in enumerate(BASE_PAIR)}

def apkmer(k: int, bps):
    "Generate All Possible K Kmer"

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not bps:
        raise ValueError("throw valid array of bps")

    if k == 1:
        return list(bps)

    next = apkmer(k - 1, bps)
    return [prefix + base for prefix in next for base in bps]

@dataclass
class KMerTokenizer:
    """DNA seq to kmer ids by sliding window algo"""

    k           : int
    stride      : int = 1
    vocabulary  : Sequence[str]
    unk_token   : None

    def __post_init__(self):
        # map allkmer with a int
        self.token2id = {token: idx for idx, token in enumerate(self.vocabulary)}

        # map the unknown bps as last element
        if self.unk_token is not None and self.unk_token not in self.token2id:
            self.token2id[self.unk_token] = len(self.token2id)

    def __call__(self, seq) -> torch.Tensor:
        seq     = seq.upper()
        tokens  = []

        # sliding window algo
        for t in range(0, max(len(seq) - self.k + 1, 0), self.stride):
            token = seq[t:t+self.k]
            tokens.append(self.token2id.get(token, self.token2id.get(self.unk_token, 0)))
        if not tokens:
            tokens.append(self.token2id.get(self.unk_token, 0))
        return torch.tensor(tokens, dtype=torch.long)

# ---------------------------------------------------------------------------
# Embedding and sequence encoders
# ---------------------------------------------------------------------------


class KMerEmbedding(nn.Module):
    """Embedding layer specialised for k-mer tokens."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq_len)
        return self.embedding(tokens)


class ConvolutionalFeatureExtractor(nn.Module):
    """Stack of temporal convolution layers used before the recurrent encoder."""

    def __init__(
        self,
        input_dim: int,
        num_filters: int,
        kernel_size: int,
        num_layers: int = 1,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:

        layers: List[nn.Module] = []
        in_channels = input_dim
        act_module = activation if activation is not None else nn.ReLU()
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(act_module)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = num_filters
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (batch, seq_len, embed_dim)
        x = embeddings.transpose(1, 2)  # -> (batch, embed_dim, seq_len)
        features = self.network(x)
        return features.transpose(1, 2)  # -> (batch, seq_len, num_filters)


class BidirectionalLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder that produces contextual representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (batch, seq_len, feature_dim)
        outputs, _ = self.lstm(features)
        return outputs  # (batch, seq_len, 2 * hidden_size)


class SequenceGlobalPool(nn.Module):
    """Global pooling across the sequence dimension."""

    def __init__(self, mode: str = "max") -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in {"max", "mean"}:
            raise ValueError("mode must be either 'max' or 'mean'")
        self.mode = mode

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.mode == "max":
            return sequence.max(dim=1).values
        return sequence.mean(dim=1)


class ChromatinAccessibilityCNNBiLSTM(nn.Module):
    """Full architecture mirroring the ISMB 2017 CNN-BiLSTM model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        conv_filters: int = 320,
        conv_kernel_size: int = 26,
        conv_layers: int = 1,
        lstm_hidden_size: int = 320,
        lstm_layers: int = 1,
        pooling: str = "max",
        dense_units: int = 925,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = KMerEmbedding(vocab_size, embedding_dim)
        self.conv = ConvolutionalFeatureExtractor(
            embedding_dim,
            conv_filters,
            conv_kernel_size,
            num_layers=conv_layers,
            dropout=dropout,
        )
        self.encoder = BidirectionalLSTMEncoder(conv_filters, lstm_hidden_size, num_layers=lstm_layers, dropout=dropout)
        self.pool = SequenceGlobalPool(pooling)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, dense_units),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        conv_features = self.conv(embedded)
        encoded = self.encoder(conv_features)
        pooled = self.pool(encoded)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class KMerSequenceDataset(Dataset):
    """Dataset that converts raw sequences to k-mer token ids on-the-fly."""

    def __init__(self, sequences: Iterable[str], labels: Sequence[float], tokenizer: KMerTokenizer):
        self.sequences = list(sequences)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if len(self.sequences) != len(self.labels):
            raise ValueError("sequences and labels must have the same length")
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        tokens = self.tokenizer(self.sequences[index])
        return tokens, self.labels[index]