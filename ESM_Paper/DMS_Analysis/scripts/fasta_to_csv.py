#!/usr/bin/env python3
"""
Convert FASTA file to CSV format for ESM log-likelihood calculation.
Creates a simple CSV with sequence ID and sequence columns.
"""

import argparse
import pandas as pd
from Bio import SeqIO
from pathlib import Path


def fasta_to_csv(fasta_path: str, output_path: str):
    """
    Convert FASTA file to CSV with sequence data.

    Args:
        fasta_path: Path to input FASTA file
        output_path: Path to output CSV file
    """
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append({
            "node": record.id,
            "sequence": str(record.seq)
        })

    df = pd.DataFrame(records)

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Converted {len(records)} sequences from {fasta_path} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FASTA to CSV for ESM analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    fasta_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
