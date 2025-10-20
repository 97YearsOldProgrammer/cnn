"""Build a GloVe-ready corpus from FASTA and GFF annotations."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

from lib.cnn import KMerTokenizer, apkmer, get_filepointer, read_fasta





def write_corpus(output_path: str, sentences: tp.Dict[str, tp.List[int]]) -> None:
    with open(output_path, "w", encoding="utf-8") as fp:
        for parent_id in sorted(sentences):
            tokens = " ".join(str(token) for token in sentences[parent_id])
            fp.write(tokens + "\n")


def write_segment_vocab(
    vocab_path: str,
    segment_vocab: tp.Dict[tp.Tuple[str, ...], int],
) -> None:
    reverse_vocab = {value: list(key) for key, value in segment_vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as fp:
        json.dump(reverse_vocab, fp, indent=2)


def build_tokenizer(args: argparse.Namespace) -> KMerTokenizer:
    alphabet = tuple(args.alphabet.upper())
    vocabulary = apkmer(args.kmer, alphabet)
    return KMerTokenizer(
        k=args.kmer,
        stride=args.stride,
        vocabulary=vocabulary,
        unk_token=args.unk_token,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenise FASTA/GFF annotations into a GloVe-ready corpus "
            "of exon and intron segments."
        )
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Identifier for the data model; used for logging only.",
    )
    parser.add_argument("--fasta", required=True, help="Path to the FASTA file.")
    parser.add_argument("--gff", required=True, help="Path to the GFF/GFF3 file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination file for the generated corpus (plain text).",
    )
    parser.add_argument(
        "--kmer",
        type=int,
        required=True,
        help="Size of the sliding k-mer window used for tokenisation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride of the sliding window used in the tokenizer.",
    )
    parser.add_argument(
        "--alphabet",
        default="ACGT",
        help="Alphabet used to generate the k-mer vocabulary (default: ACGT).",
    )
    parser.add_argument(
        "--unk-token",
        default=None,
        help="Optional unknown token for unseen k-mers.",
    )
    parser.add_argument(
        "--segment-vocab",
        help=(
            "Optional path to a JSON file describing how each segment id maps "
            "to the underlying k-mer tokens."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Preparing corpus for model '%s'", args.model_name)

    sequences = load_sequences(args.fasta)
    features = list(read_gff(args.gff))
    segments = collect_segments(features, sequences)

    tokenizer = build_tokenizer(args)
    sentences, segment_vocab = build_segment_ids(segments, tokenizer)

    if not sentences:
        raise RuntimeError("No exon annotations were found; corpus is empty.")

    write_corpus(args.output, sentences)
    logging.info("Wrote %d sentences to %s", len(sentences), args.output)

    if args.segment_vocab:
        write_segment_vocab(args.segment_vocab, segment_vocab)
        logging.info("Saved segment vocabulary to %s", args.segment_vocab)


if __name__ == "__main__":
    main()
