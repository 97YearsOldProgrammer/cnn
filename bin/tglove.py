#!/usr/bin/env python3
"""
Train GloVe word embeddings with configurable parameters.
This script replaces the hardcoded parameters in the bash training script.
"""

import argparse
import subprocess
import os
import sys



parser = argparse.ArgumentParser(
    description="Train GloVe word embeddings on a text corpus."
)
    
parser.add_argument("corpus", type=str,
    help="Path to input corpus text file.")
parser.add_argument("save_file", type=str,
    help="Output filename prefix for saving the trained vectors.")
parser.add_argument("--vocab-min-count", type=int, default=2,
    help="Minimum word frequency to include in vocabulary [%(default)s].")
parser.add_argument("--vector-size", type=int, default=100,
    help="Dimensionality of word embeddings [%(default)s].")
parser.add_argument("--max-iter", type=int, default=15,
    help="Number of training iterations [%(default)s].")
parser.add_argument("--window-size", type=int, default=15,
    help="Context window size (smaller=syntax, larger=semantics) [%(default)s].")
parser.add_argument("--memory", type=float, default=4.0,
    help="Memory limit in GB for cooccurrence and shuffle [%(default)s].")
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of threads for parallel training [%(default)s]."
    )
    
    parser.add_argument(
        "--x-max",
        type=int,
        default=10,
        help="Cutoff in weighting function [%(default)s]."
    )
    
    parser.add_argument(
        "--binary",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Save output in binary format (0=text, 1=binary, 2=both) [%(default)s]."
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Verbosity level [%(default)s]."
    )
    
    parser.add_argument(
        "--builddir",
        type=str,
        default="build",
        help="Directory containing GloVe executables [%(default)s]."
    )
    
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files (vocab, cooccurrence, shuffled)."
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.corpus):
        print(f"Error: Corpus file '{args.corpus}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.builddir):
        print(f"Error: Build directory '{args.builddir}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Define intermediate file names
    vocab_file = "vocab.txt"
    cooccurrence_file = "cooccurrence.bin"
    cooccurrence_shuf_file = "cooccurrence.shuf.bin"
    
    try:
        # Step 1: Build vocabulary
        print("\n" + "="*70)
        print("STEP 1: Building vocabulary")
        print("="*70)
        cmd = [
            f"{args.builddir}/vocab_count",
            "-min-count", str(args.vocab_min_count),
            "-verbose", str(args.verbose)
        ]
        print(f"$ {' '.join(cmd)} < {args.corpus} > {vocab_file}")
        
        with open(args.corpus, 'r') as infile, open(vocab_file, 'w') as outfile:
            subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)
        
        # Step 2: Build cooccurrence matrix
        print("\n" + "="*70)
        print("STEP 2: Building cooccurrence matrix")
        print("="*70)
        cmd = [
            f"{args.builddir}/cooccur",
            "-memory", str(args.memory),
            "-vocab-file", vocab_file,
            "-verbose", str(args.verbose),
            "-window-size", str(args.window_size)
        ]
        print(f"$ {' '.join(cmd)} < {args.corpus} > {cooccurrence_file}")
        
        with open(args.corpus, 'r') as infile, open(cooccurrence_file, 'wb') as outfile:
            subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)
        
        # Step 3: Shuffle cooccurrence matrix
        print("\n" + "="*70)
        print("STEP 3: Shuffling cooccurrence matrix")
        print("="*70)
        cmd = [
            f"{args.builddir}/shuffle",
            "-memory", str(args.memory),
            "-verbose", str(args.verbose)
        ]
        print(f"$ {' '.join(cmd)} < {cooccurrence_file} > {cooccurrence_shuf_file}")
        
        with open(cooccurrence_file, 'rb') as infile, open(cooccurrence_shuf_file, 'wb') as outfile:
            subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)
        
        # Step 4: Train GloVe model
        print("\n" + "="*70)
        print("STEP 4: Training GloVe model")
        print("="*70)
        cmd = [
            f"{args.builddir}/glove",
            "-save-file", args.save_file,
            "-threads", str(args.num_threads),
            "-input-file", cooccurrence_shuf_file,
            "-x-max", str(args.x_max),
            "-iter", str(args.max_iter),
            "-vector-size", str(args.vector_size),
            "-binary", str(args.binary),
            "-vocab-file", vocab_file,
            "-verbose", str(args.verbose)
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*70)
        print("SUCCESS: GloVe training completed!")
        print("="*70)
        print(f"Output files: {args.save_file}.*")
        
        # Cleanup intermediate files unless requested to keep
        if not args.keep_intermediate:
            print("\nCleaning up intermediate files...")
            for f in [vocab_file, cooccurrence_file, cooccurrence_shuf_file]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed: {f}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
