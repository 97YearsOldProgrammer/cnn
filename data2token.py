import argparse
import logging
import sys
from lib import tokenizer as tk

parser = argparse.ArgumentParser(
    description="Tokenise FASTA/GFF annotations into a corpus for GloVe.")
parser.add_argument("fasta", type=str,
    help="Path to the FASTA file.")
parser.add_argument("gff", type=str,
    help="Path to the GFF/GFF3 file.")
parser.add_argument("output", type=str,
    help="Destination file for the generated corpus (plain text).")
parser.add_argument("--kmer", type=int, default=8,
    help="Size of the sliding k-mer window used for tokenisation [%(default)i].")
parser.add_argument("--stride", type=int, default=3,
    help="Stride of the sliding window used in the tokenizer [%(default)i].")
parser.add_argument("--unk", default="<UNK>", type=str,
    help="Unknown token for unseen k-mers [%(default)s].")
parser.add_argument("--utr", action="store_true",
    help="Include UTR regions (5' and 3' UTRs) in transcripts.")

arg = parser.parse_args()

# Determine which feature types to include
if arg.utr:
    filter = {"exon", "intron", "three_prime_utr", "five_prime_utr"}
else:
    filter = {"exon", "intron"}

# loading all sequences
sequences = {}
for name, seq in tk.read_fasta(arg.fasta):
    if name is None:
        continue
    sequences[name] = seq
logging.info(f"Finished parsing FASTA: {len(sequences)} sequences loaded")

# parsing all gff file
features    = tk.parse_gff(arg.gff)
grouped     = tk.group_features(features, filter)
logging.info(f"Grouped into {len(grouped)} transcripts")

# Build transcripts
transcripts = tk.build_transcript(grouped, sequences)
logging.info(f"Built {len(transcripts)} transcripts")

if not transcripts:
    logging.error("No transcripts were built; corpus is empty.")
    sys.exit(1)

# Create tokenizer
vocabulary  = tk.apkmer(arg.kmer)
logging.info(f"Finished vocabulary generation: {len(vocabulary)} k-mers")

tokenizer   = tk.KmerTokenizer(
    k=arg.kmer, 
    stride=arg.stride, 
    vocabulary=vocabulary, 
    unk_token=arg.unk
)

# Tokenize transcripts
tokenized = tk.tokenize_transcripts(transcripts, tokenizer)

# Write corpus
tk.write_tokenized_corpus(arg.output, tokenized)