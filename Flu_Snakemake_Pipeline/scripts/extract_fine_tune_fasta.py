import os
import json
import argparse
from Bio import SeqIO


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--tree")
parser.add_argument("--gene")
parser.add_argument("--node-fasta")

args = parser.parse_args()

def extract_node_times(node):
    name = node.get('name')
    num_date = node.get('node_attrs', {}).get('num_date', {}).get('value')
    
    if name and num_date is not None:
        node_times[name] = num_date
    
    for child in node.get('children', []):
        extract_node_times(child)

with open(args.tree, "r") as f:
    data = json.load(f)

tree_root = data.get("tree", data)  # Adjust depending on JSON structure

node_terminal_map = {}

node_times = {}

extract_node_times(data['tree'])

node_times_trim = {key:val for key, val in node_times.items() if val < 1991}

record_dict = SeqIO.to_dict(SeqIO.parse(args.node_fasta, "fasta"))

record_dict_filtered = {k: v for k, v in record_dict.items() if k in node_times_trim}

SeqIO.write(record_dict_filtered.values(), f"fine_tune_fasta_{args.gene}.fasta", "fasta")