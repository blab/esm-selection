# translate nucleotide to protein

import argparse
import json
import requests
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
import os


def convert_to_prot(tree_file, root_file, segment):
    tree_file_annotations = open(tree_file, "r")
    tree_file_annotations = json.load(tree_file_annotations)

    root_file_sequence = open(root_file, "r")
    root_file_sequence = json.load(root_file_sequence)

    if segment == "ns":
        start_post = tree_file_annotations["meta"]["genome_annotations"]["NS1"]["start"]
        end_post = tree_file_annotations["meta"]["genome_annotations"]["NS1"]["end"]
    elif segment == "mp":
        start_post = tree_file_annotations["meta"]["genome_annotations"]["M1"]["start"]
        end_post = tree_file_annotations["meta"]["genome_annotations"]["M1"]["end"]
    else:
        start_post = tree_file_annotations["meta"]["genome_annotations"][
            segment.upper()
        ]["start"]
        end_post = tree_file_annotations["meta"]["genome_annotations"][segment.upper()][
            "end"
        ]

    sequence = Seq(root_file_sequence["nuc"])

    sub_seq = sequence[(start_post - 1) : (end_post)]

    protein_seq = sub_seq.translate()

    if segment == "mp":
        root_file_sequence["M1"] = str(protein_seq)
    
    elif segment == "ns":
        root_file_sequence["NS1"] = str(protein_seq)
    
    else:
        root_file_sequence[segment.upper()] = str(protein_seq)

    root_json_file_name = os.path.basename(root_file)

    with open(root_json_file_name, "w") as f:
        json.dump(root_file_sequence, f, indent=2)


def combine_ha_segments(root_file):

    root_file_sequence = open(root_file, "r")
    root_file_sequence = json.load(root_file_sequence)

    root_file_sequence["HA"] = (
        root_file_sequence["SigPep"]
        + root_file_sequence["HA1"]
        + root_file_sequence["HA2"]
    )

    root_json_file_name = os.path.basename(root_file)

    with open(root_json_file_name, "w") as f:
        json.dump(root_file_sequence, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--tree")
    parser.add_argument("--root")
    parser.add_argument("--segment")

    args = parser.parse_args()

    if args.segment == "ha":
        combine_ha_segments(args.root)
    else:
        convert_to_prot(args.tree, args.root, args.segment)
