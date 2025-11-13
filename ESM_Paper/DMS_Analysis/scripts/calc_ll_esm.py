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

# -------- utilities --------
def remove_stop_codon(seq: str) -> str:
    pos = seq.find("*")
    return seq if pos == -1 else seq[:pos]

max_freq_df["sequence"] = max_freq_df["sequence"].apply(remove_stop_codon)
max_freq_df_unique = max_freq_df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)

def _load_base(model_name: str):
    """Load ESM base model and alphabet from esm.pretrained.<model_name>()"""
    return getattr(esm.pretrained, model_name)()

def _try_load_pointer(path: str):
    """If path is a pointer JSON (created by the fine-tune script), return its dict; else None."""
    if not path:
        return None
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
    Load base model, then optionally apply:
      - LoRA adapters from pointer JSON (PEFT)
      - raw state_dict (non-LoRA or merged adapters)
    Returns: (model, alphabet, meta)
      meta = {
        "ft_kind": "lora" | "state_dict" | "none",
        "adapters_path": str|None,
        "base_model_loaded": str,
      }
    """
    # Case 1: pointer JSON (LoRA adapters)
    meta_ptr = _try_load_pointer(ft_path)
    if meta_ptr is not None:
        if PeftModel is None:
            raise RuntimeError("Pointer JSON provided, but 'peft' is not installed.")
        base_name = meta_ptr.get("base_model", model_name)
        base_model, alphabet = _load_base(base_name)
        adapters_dir = meta_ptr["adapters_path"]
        model = PeftModel.from_pretrained(base_model, adapters_dir)
        meta = {
            "ft_kind": "lora",
            "adapters_path": adapters_dir,
            "base_model_loaded": base_name,
        }
        return model, alphabet, meta

    # Case 2: raw state_dict (non-LoRA or merged LoRA)
    model, alphabet = _load_base(model_name)
    meta = {"ft_kind": "none", "adapters_path": None, "base_model_loaded": model_name}
    if ft_path:
        state = torch.load(ft_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        meta["ft_kind"] = "state_dict"
    return model, alphabet, meta
# ---------------------------

# 1) Load model (base or FT)
model, alphabet, meta = load_model_with_optional_ft(args.model, args.fine_tune_model)

batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# repr layer
repr_layer = getattr(model, "num_layers", None)
if repr_layer is None:
    repr_layer = 33 if args.model == "esm2_t33_650M_UR50D" else (36 if args.model == "esm2_t36_3B_UR50D" else 48)

# Identify special tokens to exclude from the log-likelihood sum
pad_idx = getattr(alphabet, "pad_idx", getattr(alphabet, "padding_idx", None))
special_idxs = {
    x for x in (
        getattr(alphabet, "cls_idx", None),
        getattr(alphabet, "eos_idx", None),
        getattr(alphabet, "bos_idx", None),
        getattr(alphabet, "mask_idx", None),
        pad_idx,
    ) if x is not None
}

# 2) Compute log-likelihoods per unique sequence
from torch.nn.functional import log_softmax  # type: ignore

for index, sequence in tqdm(enumerate(max_freq_df_unique["sequence"]), total=len(max_freq_df_unique), desc="LL"):
    data = [(max_freq_df_unique.at[index, "node"] if "node" in max_freq_df_unique.columns else f"seq{index}", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        logits = out["logits"]  # [B, L, V]
        log_probs = log_softmax(logits, dim=-1)  # [B, L, V]

        # Gather GT token log-probs
        gathered = log_probs.gather(2, batch_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L]

        # Mask out special tokens
        if special_idxs:
            mask = torch.ones_like(batch_tokens, dtype=torch.bool, device=device)
            for si in special_idxs:
                mask &= (batch_tokens != si)
            token_ll = gathered.masked_select(mask).sum()
        else:
            token_ll = gathered.sum()

        max_freq_df_unique.at[index, "log_likelihood"] = float(token_ll.item())

# Keep only per-sequence score and merge back
max_freq_df_unique = max_freq_df_unique.drop(columns=["node", "max_frequency"], errors="ignore")
merged = max_freq_df.merge(max_freq_df_unique, on="sequence", how="left")

# Cleanup potential duplicate columns
if "log_likelihood_x" in merged.columns and "log_likelihood_y" in merged.columns:
    merged = merged.drop(columns=["log_likelihood_x"]).rename(columns={"log_likelihood_y": "log_likelihood"})

# 3) Attach metadata columns for downstream analysis
end_time = time.time()
runtime = round(end_time - start_time, 3)

merged["runtime"] = runtime
merged["segment"] = args.segment
merged["epochs"] = args.epochs
merged["model_requested"] = args.model
merged["base_model_loaded"] = meta.get("base_model_loaded")
merged["ft_kind"] = meta.get("ft_kind")               # "lora" | "state_dict" | "none"
merged["adapters_path"] = meta.get("adapters_path")   # path or None
merged["repr_layer_used"] = repr_layer

# Ensure the output directory exists
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

merged.to_csv(args.output, index=False)
print(f"[done] wrote {args.output} in {runtime}s | ft_kind={merged['ft_kind'].iloc[0]} base={merged['base_model_loaded'].iloc[0]}")