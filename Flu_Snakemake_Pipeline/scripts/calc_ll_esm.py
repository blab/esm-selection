import torch  # type: ignore
import esm  # type: ignore
import argparse
from Bio import SeqIO  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse
import time
import os

start_time = time.time()

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--max_freq")
parser.add_argument("--segment")
parser.add_argument("--model", default="esm2_t33_650M_UR50D", help="ESM model to use")
parser.add_argument("--fine_tune_model", default="")
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs for fine-tuning."
)
parser.add_argument("--output_file", default="", help="Custom output file name")

args = parser.parse_args()

max_freq_df = pd.read_csv(args.max_freq)
max_freq_df["log_likelihood"] = 0

# max_freq_df["sequence"] = max_freq_df["sequence"].str.rstrip('*')


# Function to remove stop codon and following codons
def remove_stop_codon(seq):
    stop_pos = seq.find("*")
    if stop_pos != -1:
        return seq[:stop_pos]
    return seq


# Apply the function to the 'sequence' column
max_freq_df["sequence"] = max_freq_df["sequence"].apply(remove_stop_codon)

max_freq_df_unique = max_freq_df.drop_duplicates(subset="sequence", keep="first")

max_freq_df_unique = max_freq_df_unique.reset_index(drop=True)

# 1. Load ESM-2 model

if args.fine_tune_model and args.model == "esm2_t33_650M_UR50D":
    from esm.pretrained import esm2_t33_650M_UR50D

    model, alphabet = esm2_t33_650M_UR50D()
    model.load_state_dict(torch.load(args.fine_tune_model, map_location="cpu"))
elif args.fine_tune_model and args.model == "esm2_t36_3B_UR50D":
    from esm.pretrained import esm2_t36_3B_UR50D

    model, alphabet = esm2_t36_3B_UR50D()
    model.load_state_dict(torch.load(args.fine_tune_model, map_location="cpu"))
elif args.fine_tune_model and args.model == "esm2_t48_15B_UR50D":
    from esm.pretrained import esm2_t48_15B_UR50D

    model, alphabet = esm2_t48_15B_UR50D()
    model.load_state_dict(torch.load(args.fine_tune_model, map_location="cpu"))
else:
    if args.model == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif args.model == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif args.model == "esm2_t48_15B_UR50D":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()


batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

repr_layer = model.num_layers

for index, sequence in enumerate(max_freq_df_unique["sequence"]):

    data = [(max_freq_df_unique["node"][index], sequence)]

    # 3. Tokenize
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens = batch_tokens.to(device)

    # 4. Compute log-likelihoods
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        logits = results["logits"]
        log_probs = torch.log_softmax(results["logits"], dim=-1)
        log_likelihood = log_probs.gather(2, batch_tokens.unsqueeze(-1)).sum().item()

        # print(f"Log-Likelihood: {log_likelihood:.2f}")
        max_freq_df_unique.at[index, "log_likelihood"] = log_likelihood

max_freq_df_unique = max_freq_df_unique.drop(columns=["node", "max_frequency"])

merged = max_freq_df.merge(max_freq_df_unique, on="sequence", how="left")

# Remove log_likelihood_x and rename log_likelihood_y
merged = merged.drop(columns=["log_likelihood_x"]).rename(
    columns={"log_likelihood_y": "log_likelyhood"}
)

end_time = time.time()
runtime = round(end_time - start_time, 3)  # seconds with milliseconds

merged["runtime"] = runtime  # add runtime as a column to all rows

# Ensure the output directory exists
if args.output_file:
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Save the merged DataFrame to the specified output file or default file
merged.to_csv(args.output_file, index=False)
