import argparse
import json
import requests
from augur.utils import json_to_tree
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
import pandas as pd

# Get terminals nodes for each internal node in tree


def collect_terminal_nodes(node):
    """
    Recursively collect terminal nodes (nodes without children) for each node in the tree.
    """
    name = node.get("name", "(unnamed)")
    children = node.get("children", [])

    # If the node has no children, it's a terminal node
    if not children:
        return [name]

    # Otherwise, collect terminal nodes from all children
    terminal_nodes = []
    for child in children:
        terminal_nodes.extend(collect_terminal_nodes(child))

    return terminal_nodes


def map_terminal_nodes(node):
    """
    Create a mapping of each node to its terminal nodes.
    """
    name = node.get("name", "(unnamed)")
    children = node.get("children", [])

    # Collect terminal nodes for this node
    terminal_nodes = collect_terminal_nodes(node)
    node_terminal_map[name] = terminal_nodes

    # Recurse into each child
    for child in children:
        map_terminal_nodes(child)


def get_freq_sum(
    tip_freq_file, node_terminal_map, node_fasta_file, segment, output_file
):
    # import and clean up the frequency JSON

    json_fh_frequency = open(tip_freq_file, "r")
    json_dict_frequency = json.load(json_fh_frequency)

    del json_dict_frequency["pivots"]
    del json_dict_frequency["generated_by"]

    # Get frequency sums for all nodes

    from collections import defaultdict

    # Output dictionary
    summed_frequencies = {}

    # Sum frequencies
    for node, terminals in node_terminal_map.items():
        summed = None
        for terminal in terminals:
            if terminal in json_dict_frequency:
                freqs = json_dict_frequency[terminal]["frequencies"]
                if summed is None:
                    summed = freqs.copy()
                else:
                    summed = [x + y for x, y in zip(summed, freqs)]
        if summed is not None:
            summed_frequencies[node] = summed

    # Filter out terminal nodes

    """
    filtered_dict = {
        key: value
        for key, value in summed_frequencies.items()
        if key.startswith("NODE_")
    }
    """

    # Get max frequency for each internal node

    max_values = {key: max(values) for key, values in summed_frequencies.items()}
    # max_values = {key: max(values) for key, values in filtered_dict.items()}

    # Path to your multi-FASTA file
    fasta_path = node_fasta_file

    # Create a new dictionary to hold scores and sequences
    node_data = {}

    # Load FASTA sequences into a dictionary
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

    # Add sequences to the node_data
    for node, score in max_values.items():
        if node in fasta_dict:
            node_data[node] = {
                "max_frequency": score,
                "sequence": str(fasta_dict[node].seq),
            }
        else:
            node_data[node] = {
                "max_frequency": score,
                "sequence": None,  # or handle missing sequences as needed
            }

    df = pd.DataFrame.from_dict(node_data, orient="index")

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "node"})

    df_to_csv = df.to_csv(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--tree")
    parser.add_argument("--segment")
    parser.add_argument("--tip-freq")
    parser.add_argument("--node-fasta")
    parser.add_argument("--output", default="", help="Custom output file name")

    args = parser.parse_args()

    # Load tree JSON
    with open(args.tree, "r") as f:
        data = json.load(f)

    # Entry point (usually under 'tree' or 'nodes')
    tree_root = data.get("tree", data)  # Adjust depending on JSON structure

    # Dictionary to store the mapping of nodes to their terminal nodes
    node_terminal_map = {}

    # Map terminal nodes for each node
    map_terminal_nodes(tree_root)

    get_freq_sum(
        args.tip_freq, node_terminal_map, args.node_fasta, args.segment, args.output
    )
