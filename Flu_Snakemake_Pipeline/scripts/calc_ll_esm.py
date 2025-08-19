import torch  # type: ignore
import esm  # type: ignore
import argparse
from Bio import SeqIO  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
import time
import os
import json

# Try to import PEFT for pointer/adapters loading
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None

start_time = time.time()

parser = argparse.ArgumentParser(
    description="Compute log-likelihoods with ESM (supports LoRA pointer JSON or raw state_dict).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--max_freq", required=True)
parser.add_argument("--segment", required=True)
parser.add_argument("--model", default="esm2_t33_650M_UR50D", help="ESM model to use")
parser.add_argument("--fine_tune_model", default="", help="Path to FT artifact: pointer JSON or raw state_dict")
parser.add_argument("--epochs", type=int, default=1, help="(unused here; kept for metadata parity)")
parser.add_argument("--output", required=True, help="Output CSV path")

args = parser.parse_args()

max_freq_df = pd.read_csv(args.max_freq)
max_freq_df["log_likelihood"] = 0.0


# Function to remove stop codon and following codons
def remove_stop_codon(seq):
    stop_pos = seq.find("*")
    if stop_pos != -1:
        return seq[:stop_pos]
    return seq


# Apply the function to the 'sequence' column
max_freq_df["sequence"] = max_freq_df["sequence"].apply(remove_stop_codon)

max_freq_df_unique = max_freq_df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)


# --------- Model loading helpers ---------
def _load_base(model_name: str):
    """
    Load ESM base model and alphabet from esm.pretrained.<model_name>().
    """
    return getattr(esm.pretrained, model_name)()


def _try_load_pointer(path: str):
    """
    If path is a pointer JSON (created by the training script), return its dict.
    Otherwise return None.
    """
    if not path:
        return None
    # If it's clearly not JSON (e.g., binary), skip quickly
    # But still try: torch.load error earlier came from trying to load JSON as weights
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict) and meta.get("format") == "peft_lora_pointer_v1":
            return meta
    except Exception:
        pass
    return None


def load_model_with_optional_ft(model_name: str, ft_path: str):
    """
    Load base model, then:
      - If ft_path is a pointer JSON: apply PEFT adapters from the referenced directory.
      - Else if ft_path is a raw state dict: torch.load(..., weights_only=False) then load_state_dict(strict=False).
      - Else: return base model.
    """
    # Case 1: pointer JSON (LoRA adapters)
    meta = _try_load_pointer(ft_path)
    if meta is not None:
        if PeftModel is None:
            raise RuntimeError(
                "Pointer JSON provided, but 'peft' is not installed in this environment."
            )
        base_name = meta.get("base_model", model_name)
        base_model, alphabet = _load_base(base_name)
        adapters_dir = meta["adapters_path"]
        model = PeftModel.from_pretrained(base_model, adapters_dir)
        return model, alphabet

    # Case 2: raw state_dict (non-LoRA or merged LoRA)
    model, alphabet = _load_base(model_name)
    if ft_path:
        # PyTorch 2.6: weights_only=True by default; many older files require False.
        # This is safe if you trust the file (you do—it's your pipeline).
        state = torch.load(ft_path, map_location="cpu", weights_only=False)
        # Allow partial loads (e.g., heads or adapters merged)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return model, alphabet
# -----------------------------------------


# 1. Load ESM-2 model (with optional fine-tuned artifact)
model, alphabet = load_model_with_optional_ft(args.model, args.fine_tune_model)

batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Choose top (final) layer by default
repr_layer = getattr(model, "num_layers", None)
if repr_layer is None:
    # Fallback for any custom wrappers; most ESM models expose num_layers
    repr_layer = 33 if args.model == "esm2_t33_650M_UR50D" else (36 if args.model == "esm2_t36_3B_UR50D" else 48)

# Compute log-likelihoods sequence-by-sequence
for index, sequence in enumerate(max_freq_df_unique["sequence"]):
    data = [(max_freq_df_unique["node"][index], sequence)]

    # 3. Tokenize
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 4. Compute log-likelihoods
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        # logits = results["logits"]  # not needed separately
        log_probs = torch.log_softmax(results["logits"], dim=-1)
        # Sum token log-probs at the ground-truth token indices
        log_likelihood = log_probs.gather(2, batch_tokens.unsqueeze(-1)).sum().item()
        max_freq_df_unique.at[index, "log_likelihood"] = log_likelihood

# Keep only per-sequence score and merge back
max_freq_df_unique = max_freq_df_unique.drop(columns=["node", "max_frequency"], errors="ignore")
merged = max_freq_df.merge(max_freq_df_unique, on="sequence", how="left")

# Remove log_likelihood_x and rename log_likelihood_y if needed
if "log_likelihood_x" in merged.columns and "log_likelihood_y" in merged.columns:
    merged = merged.drop(columns=["log_likelihood_x"]).rename(columns={"log_likelihood_y": "log_likelihood"})

end_time = time.time()
runtime = round(end_time - start_time, 3)  # seconds with milliseconds
merged["runtime"] = runtime  # add runtime as a column to all rows

# Ensure the output directory exists
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save result
merged.to_csv(args.output, index=False)
print(f"[done] wrote {args.output} in {runtime}s")