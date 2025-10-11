import numpy as np
import torch
import torch.nn as nn
from troch.utils.data import Dataset, DataLoader

BASE2IDX = {'A':0, 'C':1, 'G':2, 'T':3}

def one_hot_dna(seq):
    len = len(seq)
    x   = np.zeros((4, len), dtype=np.float32)
    for i, b in enumerate(seq):
        j = BASE2IDX.get(b.upper(), None)
        if j is not None:
            x[j, i] = 1.0
    return x

class DNADataset(Dataset):

    def __init__(self, sequences, labels):
        self.x = [one_hot_dna(s) for s in sequences]         # list of (4, L)
        self.y = np.asarray(labels, dtype=np.float32)        # binary: 0/1
        
        # pad to same length if needed (simplest: right-pad with zeros)
        maxL = max(x.shape[1] for x in self.x)
        self.x = [np.pad(x, ((0,0),(0, maxL - x.shape[1])), mode='constant') for x in self.x]
        self.x = np.stack(self.x)                            # (N, 4, Lmax)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])
    
class DeepBindLike(nn.Module):

    def __init__(
            self, 
            in_channels=4, 
            motif_filters=128, 
            motif_width=19, 
            p_drop=0.2
        ):
        
        super().__init__()
        self.conv = nn.Conv1d(in_channels, motif_filters, kernel_size=motif_width)
        self.act  = nn.ReLU()
        # Global max pool → shape becomes (batch, motif_filters, 1)
        self.gmp  = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc   = nn.Linear(motif_filters, 1)  # binary classification

    def forward(self, x):              # x: (B, 4, L)
        z = self.conv(x)               # (B, F, L')
        z = self.act(z)
        z = self.gmp(z).squeeze(-1)    # (B, F)
        z = self.drop(z)
        logit = self.fc(z).squeeze(-1) # (B,)
        return logit