import argparse

from lib import tokenizer as tk

parser = argparse.ArgumentParser(
    description="Tokenise FASTA/GFF annotations into a corpus for GloVe.")

parser.add_argument("fasta", required=True, type=str,
    help="Path to the FASTA file.")
parser.add_argument("gff", required=True, type=str,
    help="Path to the GFF/GFF3 file.")
parser.add_argument("--output", required=True, type=str,
    help="Destination file for the generated corpus (plain text).")
parser.add_argument("--kmer", type=int, default=3,
    help="Size of the sliding k-mer window used for tokenisation [%(default)i].")
parser.add_argument("--stride", type=int, default=1,
    help="Stride of the sliding window used in the tokenizer [%(default)i].")
parser.add_argument("--unk-token", default="<UNK>", type=str,
    help="Unknown token for unseen k-mers [%(default)s].")
parser.add_argument("--include-utr", action="store_true",
    help="Include UTR regions (5' and 3' UTRs) in transcripts.")
parser.add_argument("--feature-types", type=str, default=None,
    help="Comma-separated feature types to include (e.g., 'exon,cds'). Overrides --include-utr.")
parser.add_argument("--min-exon", type=int, default=0,
    help="Minimum exon length to include [%(default)i].")
parser.add_argument("--min-intron", type=int, default=0,
    help="Minimum intron length to include [%(default)i].")

arg = parser.parse_args()



# Determine which feature types to include
if arg.feature_types:
    feature_filter = set(ft.strip().lower() for ft in arg.feature_types.split(","))
elif arg.include_utr:
    feature_filter = {"exon", "cds", "three_prime_utr", "five_prime_utr"}
else:
    feature_filter = {"exon", "cds"}

sequences = {}
for name, seq in tk.read_fasta(arg.fasta):
    if name is None:
        continue
    sequences[name] = seq

features    = tk.parse_gff(arg.gff)
grouped     = tk.group_features(features, feature_filter)

# Build transcripts from sequences
transcripts = tk.build_transcript(grouped, sequences)

if not transcripts:
    raise RuntimeError("No transcripts were built; corpus is empty.")

# Create tokenizer
vocabulary  = tk.apkmer(arg.kmer)
tokenizer   = tk.KmerTokenizer(
    k=arg.kmer, 
    stride=arg.stride, 
    vocabulary=vocabulary, 
    unk_token=arg.unk_token
)

# Tokenize transcripts
tokenized = tk.tokenize_transcripts(transcripts, tokenizer)

# Write corpus
tk.write_tokenized_corpus(arg.output, tokenized)