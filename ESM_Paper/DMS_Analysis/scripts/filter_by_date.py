#!/usr/bin/env python3
"""
Filter H3N2 sequences by date cutoff from Nextstrain tree JSON files.
Extracts sequences that have num_date <= cutoff_year.
"""

import argparse
import json
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def extract_nodes_before_date(tree_json_path: str, cutoff_date: float):
    """
    Extract all node names (terminal and internal) that have num_date <= cutoff_date.

    Args:
        tree_json_path: Path to Nextstrain tree JSON file
        cutoff_date: Cutoff date (decimal year, e.g., 2009.264)

    Returns:
        Set of node names that pass the date filter
    """
    with open(tree_json_path) as f:
        data = json.load(f)

    valid_nodes = set()

    def traverse(node):
        """Recursively traverse tree and collect nodes before cutoff."""
        name = node.get('name')
        num_date = node.get('node_attrs', {}).get('num_date', {}).get('value')

        if name and num_date is not None and num_date <= cutoff_date:
            valid_nodes.add(name)

        # Recurse to children
        for child in node.get('children', []):
            traverse(child)

    traverse(data['tree'])
    return valid_nodes


def filter_fasta_by_nodes(input_fasta: str, valid_nodes: set, output_fasta: str):
    """
    Filter FASTA file to only include sequences in valid_nodes set.

    Args:
        input_fasta: Path to input FASTA file
        valid_nodes: Set of valid node names
        output_fasta: Path to output FASTA file
    """
    filtered_records = []
    total_count = 0
    kept_count = 0

    for record in SeqIO.parse(input_fasta, "fasta"):
        total_count += 1
        if record.id in valid_nodes:
            filtered_records.append(record)
            kept_count += 1

    SeqIO.write(filtered_records, output_fasta, "fasta")

    print(f"Segment: {Path(input_fasta).stem}")
    print(f"  Total sequences: {total_count}")
    print(f"  Kept (≤ cutoff): {kept_count}")
    print(f"  Filtered out: {total_count - kept_count}")

    return kept_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter H3N2 sequences by date from Nextstrain tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--tree", required=True, help="Path to tree JSON file")
    parser.add_argument("--input-fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-fasta", required=True, help="Output filtered FASTA file")
    parser.add_argument("--cutoff-date", type=float, required=True,
                        help="Cutoff date (decimal year, e.g., 2009.264)")

    args = parser.parse_args()

    print(f"Filtering sequences with date ≤ {args.cutoff_date}")
    print(f"Tree: {args.tree}")

    # Extract valid node names from tree
    valid_nodes = extract_nodes_before_date(args.tree, args.cutoff_date)
    print(f"Valid nodes from tree: {len(valid_nodes)}")

    # Filter FASTA
    kept = filter_fasta_by_nodes(args.input_fasta, valid_nodes, args.output_fasta)

    if kept == 0:
        print("WARNING: No sequences passed the filter!")
    else:
        print(f"\nWrote {kept} sequences to {args.output_fasta}")


if __name__ == "__main__":
    main()
