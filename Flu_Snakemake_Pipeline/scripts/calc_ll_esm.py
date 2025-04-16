import torch  # type: ignore
import esm  # type: ignore
import argparse
from Bio import SeqIO  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--max_freq")
parser.add_argument("--segment")

args = parser.parse_args()

max_freq_df = pd.read_csv(args.max_freq)
max_freq_df["log_likelihood"] = 0

max_freq_df["sequence"] = max_freq_df["sequence"].str.rstrip('*')

max_freq_df_unique = max_freq_df.drop_duplicates(subset="sequence", keep="first")

max_freq_df_unique = max_freq_df_unique.reset_index(drop=True)

# 1. Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for evaluation

for index, sequence in enumerate(max_freq_df_unique["sequence"]):

    data = [(max_freq_df_unique["node"][index], sequence)]

    # 3. Tokenize
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # 4. Compute log-likelihoods
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[], return_contacts=False)
        log_probs = torch.log_softmax(results["logits"], dim=-1)
        log_likelihood = log_probs.gather(2, batch_tokens.unsqueeze(-1)).sum().item()

        #print(f"Log-Likelihood: {log_likelihood:.2f}")
        max_freq_df_unique.at[index, "log_likelihood"] = log_likelihood

max_freq_df_unique = max_freq_df_unique.drop(columns=["node", "max_frequency"])

merged = max_freq_df.merge(max_freq_df_unique, on="sequence", how="left")

merged.to_csv(f"Max_Freq_Fasta_LL_{args.segment}.csv", index=False)
