import os
import json
import argparse
from Bio import SeqIO
import pandas as pd

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--tree")
parser.add_argument("--gene")
parser.add_argument("--node-fasta")
parser.add_argument("--max-freq")

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

# Fileter for max freq abouve 0.75 and remove duplicates

#max_freq_df = pd.read_csv(args.max_freq)

#max_freq_df = max_freq_df[max_freq_df['max_frequency'] > 0.75]

#max_freq_df = max_freq_df.drop(columns=['sequence'])

#max_freq_df = max_freq_df.set_index(max_freq_df.columns[0])

#max_freq_dict = max_freq_df.to_dict()['max_frequency']

#record_dict_filtered_max = {k: v for k, v in record_dict_filtered.items() if k in max_freq_dict}

unique_seq_record_dict = {}
seen_sequences = set()

for key, record in record_dict_filtered.items():
    if str(record.seq) not in seen_sequences:
        unique_seq_record_dict[key] = record
        seen_sequences.add(str(record.seq))

trimmed_record_dict = {}

#rmv stop codon and following sequence

for key, record in unique_seq_record_dict.items():
    sequence = str(record.seq)
    stop_index = sequence.find("*")
    if stop_index != -1:
        sequence = sequence[:stop_index]  
    trimmed_record = record
    trimmed_record.seq = trimmed_record.seq.__class__(sequence)
    trimmed_record_dict[key] = trimmed_record

SeqIO.write(trimmed_record_dict.values(), f"fine_tune_fasta_{args.gene}.fasta", "fasta")