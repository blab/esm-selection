# From Katie Kistler

import argparse
import json
import requests
from augur.utils import json_to_tree
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import os

# Extract fasta for each node


def apply_muts_to_root(root_seq, list_of_muts):
    """
    Apply a list of mutations to the root sequence
    to find the sequence at a given node. The list of mutations
    is ordered from root to node, so multiple mutations at the
    same site will correctly overwrite each other
    """

    # make the root sequence mutatable
    root_plus_muts = MutableSeq(root_seq)

    # apply all mutations to root sequence
    for mut in list_of_muts:
        # subtract 1 to deal with biological numbering vs python
        mut_site = int(mut[1:-1]) - 1
        # get the nuc that the site was mutated TO
        mutation = mut[-1]
        # apply mutation
        root_plus_muts[mut_site] = mutation

    return root_plus_muts


def getNodeSequences(gene, local_files, tree_file, root_file, output_file):
    """
    Get the sequence at each node in the given tree and
    save them as a FASTA file
    """
    # if we are fetching the JSONs from a URL
    if local_files == "False":
        # fetch the tree JSON from URL
        tree_json = requests.get(
            tree_file, headers={"accept": "application/json"}
        ).json()
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)
        # fetch the root JSON from URL
        root_json = requests.get(
            root_file, headers={"accept": "application/json"}
        ).json()
        # get the nucleotide sequence of root
        root_seq_nuc = root_json[gene]

    # if we are using paths to local JSONs
    elif local_files == "True":
        # load tree
        with open(tree_file, "r") as f:
            tree_json = json.load(f)
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)
        # load root sequence file
        with open(root_file, "r") as f:
            root_json = json.load(f)
        # get the nucleotide sequence of root
        root_seq_nuc = root_json[gene]

    ## Now find the node sequences

    # initialize list to store sequence records for each node
    sequence_records = []

    # find sequence at each node in the tree (includes internal nodes and terminal nodes)
    for node in tree.find_clades():

        # get path back to the root
        path = tree.get_path(node)

        # get all  mutations relative to root
        muts = [branch.branch_attrs["mutations"].get(gene, []) for branch in path]
        # flatten the list of nucleotide mutations
        muts = [item for sublist in muts for item in sublist]
        # get sequence at node
        node_seq = apply_muts_to_root(root_seq_nuc, muts)

        sequence_records.append(SeqRecord(node_seq, node.name, "", ""))

        # if gene == "M1":
        #    SeqIO.write(sequence_records, f"nodeSeqs_mp.fasta", "fasta")
        # elif gene == "NS1":
        #    SeqIO.write(sequence_records, f"nodeSeqs_ns.fasta", "fasta")
        # else:
        #    SeqIO.write(sequence_records, f"nodeSeqs_{gene.lower()}.fasta", "fasta")
        SeqIO.write(sequence_records, output_file, "fasta")


def read_fasta(file):
    seqs = {}
    with open(file) as f:
        header = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line
                seqs[header] = ""
            else:
                seqs[header] += line
    return seqs


def merge_fastas(files, output):
    merged = defaultdict(str)

    for file in files:
        seqs = read_fasta(file)
        for header, seq in seqs.items():
            merged[header] += seq

    with open(output, "w") as out:
        for header, seq in merged.items():
            out.write(f"{header}\n{seq}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--gene"
        # default="nuc",
        # help="Name of gene to return AA sequences for. 'nuc' will return full geneome nucleotide seq",
    )
    parser.add_argument(
        "--local-files"
        # default="False",
        # help="Toggle this on if you are supplying local JSON files for the tree and root sequence."
        # + "Default is to fetch them from a URL",
    )
    parser.add_argument(
        "--tree"
        # default="https://data.nextstrain.org/ncov_gisaid_global_all-time.json",
        # help="URL for the tree.json file, or path to the local JSON file if --local-files=True",
    )
    parser.add_argument(
        "--root"
        # default="https://data.nextstrain.org/ncov_gisaid_global_all-time_root-sequence.json",
        # help="URL for the root-sequence.json file, or path to the local JSON file if --local-files=True",
    )
    parser.add_argument("--output", default="", help="Custom output file name")

    args = parser.parse_args()

    if args.gene == "mp":
        gene = "M1"
        getNodeSequences(gene, args.local_files, args.tree, args.root, args.output)

    elif args.gene == "ns":
        gene = "NS1"
        getNodeSequences(gene, args.local_files, args.tree, args.root, args.output)

    elif args.gene == "ha":
        getNodeSequences(
            "HA1", args.local_files, args.tree, args.root, "nodeSeqs_ha1.fasta"
        )
        getNodeSequences(
            "HA2", args.local_files, args.tree, args.root, "nodeSeqs_ha2.fasta"
        )
        getNodeSequences(
            "SigPep", args.local_files, args.tree, args.root, "nodeSeqs_sigpep.fasta"
        )
        files = ["nodeSeqs_sigpep.fasta", "nodeSeqs_ha1.fasta", "nodeSeqs_ha2.fasta"]

        merge_fastas(files, args.output)
        for file in files:
            os.remove(file)

    else:
        gene = args.gene.upper()
        getNodeSequences(gene, args.local_files, args.tree, args.root, args.output)
