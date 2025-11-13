#!/usr/bin/env python3
"""
Generate a deep mutational scanning (DMS) library from a reference sequence.
Creates a multi-FASTA file with all possible single amino acid mutations.
"""

from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse


def generate_dms_library(input_fasta: Path, output_fasta: Path, include_wildtype: bool = True):
    """
    Generate all possible single amino acid mutations from a reference sequence.

    Args:
        input_fasta: Path to input FASTA file with reference sequence
        output_fasta: Path to output multi-FASTA file with all variants
        include_wildtype: Whether to include the wild-type sequence in output
    """
    # Standard 20 amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # Read the reference sequence
    record = SeqIO.read(input_fasta, "fasta")
    ref_seq = str(record.seq)
    ref_id = record.id

    print(f"Reference sequence: {ref_id}")
    print(f"Length: {len(ref_seq)} amino acids")
    print(f"Generating {len(ref_seq) * 19} single mutants (19 mutations per position)...")

    variants = []

    # Optionally add wild-type sequence
    if include_wildtype:
        wt_record = SeqRecord(
            Seq(ref_seq),
            id=f"{ref_id}_WT",
            description="Wild-type reference sequence"
        )
        variants.append(wt_record)

    # Generate all single mutations
    mutation_count = 0
    for pos in range(len(ref_seq)):
        wt_aa = ref_seq[pos]

        for mut_aa in amino_acids:
            # Skip if mutation is same as wild-type
            if mut_aa == wt_aa:
                continue

            # Create mutant sequence
            mutant_seq = ref_seq[:pos] + mut_aa + ref_seq[pos+1:]

            # Create mutation label (1-indexed position)
            mutation_label = f"{wt_aa}{pos+1}{mut_aa}"

            # Create SeqRecord
            variant_record = SeqRecord(
                Seq(mutant_seq),
                id=f"{ref_id}_{mutation_label}",
                description=f"Single mutant: {mutation_label}"
            )
            variants.append(variant_record)
            mutation_count += 1

    # Write all variants to output file
    SeqIO.write(variants, output_fasta, "fasta")

    print(f"\nGenerated {mutation_count} single mutants")
    if include_wildtype:
        print(f"Total sequences in output: {mutation_count + 1} (including WT)")
    else:
        print(f"Total sequences in output: {mutation_count}")
    print(f"Output written to: {output_fasta}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deep mutational scanning library with all single amino acid mutations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Dataframes/estimated_sequence.fasta"),
        help="Input FASTA file with reference sequence"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Dataframes/dms_library_all_mutations.fasta"),
        help="Output multi-FASTA file with all single mutants"
    )

    parser.add_argument(
        "--no-wildtype",
        action="store_true",
        help="Exclude wild-type sequence from output"
    )

    args = parser.parse_args()

    # Check input exists
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Generate library
    generate_dms_library(
        input_fasta=args.input,
        output_fasta=args.output,
        include_wildtype=not args.no_wildtype
    )


if __name__ == "__main__":
    main()
