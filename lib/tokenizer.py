import gzip
import sys
import torch

import typing as tp
from contextlib     import closing
from dataclasses    import dataclass

from __future__ import annotations



def anti(seq):
	comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
	anti = seq.translate(comp)[::-1]
	return anti



#####################
## UTILITY SECTION ##
#####################

def getfp(filename):
    
	if   filename.endswith('.gz'):
		return gzip.open(filename, 'rt', encoding='ISO-8859-1')
	elif filename == '-':
		return sys.stdin
	else:
		return open(filename)

def read_fasta(filename):

	name = None
	seqs = []

	fp = getfp(filename)

	for line in fp:
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:].split()[0]
				seqs = []
			else:
				name = line[1:].split()[0]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

@dataclass
class Feature:
    """ Parse Single GFF line"""

    seqid:  str
    source: str
    typ:    str
    beg:    int
    end:    int
    score:  tp.Optional[float]
    strand: str
    phase:  tp.Optional[int]
    att:    tp.Dict[str, str]
    
def parse_gff(filename):
    
    fp          = getfp(filename)
    features    = []

    with closing(fp):
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("\t")
            if len(line) != 9:
                raise ValueError(f"Malformed GFF line: {line.rstrip()}")
            seqid, source, typ, beg, end, score, strand, phase, att = line
            score = None if score == "." else float(score)
            phase = None if phase == "." else int(phase)
            feature = Feature(
                seqid=  seqid,
                source= source,
                typ=    typ.lower(),
                beg=    int(beg),
                end=    int(end),
                score=  score,
                strand= strand,
                phase=  phase
                att=    att,
            )
            features.append(feature)

    return features

def choose_parent_id(feature):
    if "Parent" in feature.att:
        return feature.att["Parent"].split(",")[0]
    if "ID" in feature.att:
        return feature.att["ID"]
    return feature.seqid

@dataclass
class Word:
    typ:       str
    bps:       str
    strand:    str

def filter_gff(features, seqs):

    gff = {}
    filter  = {"exon", "intron"}

    for feature in features:
        if feature.type not in filter: continue
        parent_id = choose_parent_id(feature)
        gff.setdefault(parent_id, []).append(feature)

    return gff

def build_line(features, seqid, strand, seq):

    line = []

    # iterate through all feature under a parent ID
    for feature in features:

        if feature.seqid != seqid:
            raise ValueError(
                f"Transcript {seqid} has exons on multiple sequences "
                f"({seqid} vs {feature.seqid})"
            )
            
        if feature.strand != strand:
            raise ValueError(
                f"Transcript {seqid} mixes strands ({strand} vs {feature.strand})"
            )
        
        word = seq[feature.beg-1 : feature.end-1]
        if strand == "-":
            word = anti(word)
        line.append(word)

    return line

def collect_segments(features, sequences: tp.Dict[str, str]):
    grouped: tp.Dict[str, tp.List[Feature]] = {}

    transcript= {}

    for parent_id, feature in grouped.items():
        if not feature:
            continue
        seqid   = feature[0].seqid
        strand  = feature[0].strand

        if seqid not in sequences:
            raise KeyError(f"Sequence '{seqid}' referenced in GFF but absent from FASTA")
        
        seq     = sequences[seqid]
        feature = sorted(feature, key=lambda x: x.beg)
        line    = build_line(feature, seqid, strand, seq)
        transcript.append(line)
    
    return transcript


def build_segment_ids(
    segments: tp.Dict[str, tp.List[Segment]],
    tokenizer: KMerTokenizer,
) -> tp.Tuple[tp.Dict[str, tp.List[int]], tp.Dict[tp.Tuple[str, ...], int]]:
    sentences: tp.Dict[str, tp.List[int]] = {}
    segment_vocab: tp.Dict[tp.Tuple[str, ...], int] = {}

    for parent_id, items in segments.items():
        words: tp.List[int] = []
        for segment in items:
            token_tensor = tokenizer(segment.sequence)
            token_ids = tuple(str(x) for x in token_tensor.tolist())
            key = (segment.kind,) + token_ids
            if key not in segment_vocab:
                segment_vocab[key] = len(segment_vocab)
            words.append(segment_vocab[key])
        if words:
            sentences[parent_id] = words

    return sentences, segment_vocab
    
######################
#### Tokenisation ####
######################

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