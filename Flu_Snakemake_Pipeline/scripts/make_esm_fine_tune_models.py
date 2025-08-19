import torch
import esm
import argparse
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
import json  # <-- added

# ---- NEW: PEFT / LoRA ----
#try:
from peft import LoraConfig, get_peft_model, PeftModel
#except Exception as e:
#    get_peft_model = None  # we'll sanity-check later

class ProteinDataset(Dataset):
    """Dataset class for protein sequences."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

randseed = 12
torch.backends.cudnn.deterministic = True
random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed(randseed)
np.random.seed(randseed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune ESM-2 on protein sequences (with LoRA for 15B).")
    parser.add_argument("--input", type=str, default="alignment.fasta",
                        help="Input FASTA file containing training sequences.")
    parser.add_argument("--output", type=str, default="models/esm.bin",
                        help="File path to save the full fine-tuned model (non-LoRA path) or pointer JSON (LoRA).")
    parser.add_argument("--peft-output", type=str, default="adapters/esm2_15b_lora",
                        help="Directory to save LoRA adapters if 15B model is used.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fine-tuning.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--model", type=str,
                        choices=["esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"],
                        default="esm2_t33_650M_UR50D", help="Specify which ESM-2 model to use.")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use mixed precision (recommended on CUDA for less memory).")
    # LoRA hyperparams (used only for 15B)
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (15B only).")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (15B only).")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (15B only).")
    return parser.parse_args()

def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def mask_tokens(tokens, mask_token_idx, vocab_size, device, special_token_idxs=None):
    """Apply masking to input tokens (avoid masking special tokens if provided)."""
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, 0.15, device=device)

    if special_token_idxs is not None:
        for idx in special_token_idxs:
            probability_matrix = probability_matrix * (tokens != idx)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80%: replace with [MASK]
    mask_indices = masked_indices & (torch.rand(labels.shape, device=device) < 0.8)
    tokens[mask_indices] = mask_token_idx

    # 10%: random tokens
    random_indices = masked_indices & (torch.rand(labels.shape, device=device) < 0.1)
    random_tokens = torch.randint(0, vocab_size, labels.shape, device=device)
    tokens[random_indices] = random_tokens[random_indices]

    # 10%: keep original
    return tokens, labels

def train_model(model, dataloader, batch_converter, optimizer, device, epochs,
                mask_token_idx, vocab_size, repr_layer, use_amp=False, special_token_idxs=None):
    """Train the model on the given dataset."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)

            # Mask tokens (avoid special tokens if provided)
            masked_tokens, labels = mask_tokens(
                batch_tokens, mask_token_idx, vocab_size, device, special_token_idxs
            )

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    results = model(masked_tokens, repr_layers=[repr_layer])
                    logits = results["logits"]
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                results = model(masked_tokens, repr_layers=[repr_layer])
                logits = results["logits"]
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 1 == 0:
                print(f"Step {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg:.4f}")

def save_model(model, output_file, used_lora=False, peft_output_dir=None, base_model_name=None, extra_meta=None):
    """
    Save model or adapters.

    - Non-LoRA: writes a full state_dict to `output_file`.
    - LoRA: saves adapters to `peft_output_dir` AND writes a small JSON pointer to `output_file`
      so workflows (Snakemake) have a concrete file to depend on.
    """
    if used_lora:
        assert hasattr(model, "save_pretrained"), "PEFT model expected when used_lora=True."
        outdir = peft_output_dir or "adapters/esm_lora"
        os.makedirs(outdir, exist_ok=True)
        # Save adapters (PEFT will create adapter_model.* and config)
        try:
            model.save_pretrained(outdir, safe_serialization=False)
        except TypeError:
            model.save_pretrained(outdir)
        print(f"LoRA adapters saved to {outdir}")

        # Write pointer JSON at output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pointer = {
            "format": "peft_lora_pointer_v1",
            "base_model": base_model_name,
            "adapters_path": os.path.abspath(outdir),
        }
        if extra_meta:
            pointer.update(extra_meta)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pointer, f, indent=2)
        print(f"Wrote LoRA pointer to {output_file}")
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(model.state_dict(), output_file)
        print(f"Model saved to {output_file}")

def maybe_wrap_with_lora(model, args):
    """
    If using the 15B model, wrap attention and MLP Linear layers with LoRA.
    Targets are matched by module name substrings.
    """
    is_15b = args.model == "esm2_t48_15B_UR50D"
    if not is_15b:
        return model, False

    if get_peft_model is None:
        raise RuntimeError(
            "peft is not installed, but LoRA is required for the 15B model. "
            "Install with: pip install peft"
        )

    # ESM2 uses q_proj, k_proj, v_proj, out_proj in attention and fc1/fc2 in MLPs.
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=None  # non-HF forward; PEFT will wrap Linear layers it finds
    )

    peft_model = get_peft_model(model, lora_config)

    # Print a small summary so you can confirm only adapters will train
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        trainables = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"Trainable params: {trainables:,} / {total:,}")

    return peft_model, True

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Map model names to their corresponding layer numbers
    model_layer_map = {
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48
    }
    repr_layer = model_layer_map[args.model]

    # Load model and tokenizer
    print(f"Loading {args.model} model...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    batch_converter = alphabet.get_batch_converter()
    mask_token_idx = getattr(alphabet, "mask_idx", None)
    vocab_size = len(alphabet)

    # --- PATCH: support both 'pad_idx' (older esm) and 'padding_idx' (newer fair-esm)
    pad_idx = getattr(alphabet, "pad_idx", getattr(alphabet, "padding_idx", None))

    special_token_idxs = {
        x for x in (
            getattr(alphabet, "cls_idx", None),
            getattr(alphabet, "eos_idx", None),
            getattr(alphabet, "bos_idx", None),
            pad_idx,
        ) if x is not None
    }

    model = model.to(device)

    # ---- NEW: Wrap with LoRA for 15B (massive memory reduction) ----
    model, used_lora = maybe_wrap_with_lora(model, args)

    model.train()

    # Load training sequences
    print(f"Loading training sequences from {args.input}...")
    sequences = load_sequences(args.input)

    # Heuristic batch sizes (you can still tune these)
    if args.model == "esm2_t33_650M_UR50D":
        batches = 8
    elif args.model == "esm2_t36_3B_UR50D":
        batches = 2
    else:  # 15B
        batches = 1  # with LoRA + AMP you might push this up depending on GPU

    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batches, shuffle=True, collate_fn=lambda x: x)

    # Optimizer only over trainable (i.e., LoRA) params if used_lora
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    # Train
    print("Starting fine-tuning...")
    train_model(
        model, dataloader, batch_converter, optimizer, device, args.epochs,
        mask_token_idx, vocab_size, repr_layer, use_amp=args.use_amp, special_token_idxs=special_token_idxs
    )

    # Save
    if used_lora:
        print(f"Saving LoRA adapters to {args.peft_output}...")
        save_model(
            model, args.output,
            used_lora=True,
            peft_output_dir=args.peft_output,
            base_model_name=args.model,
            extra_meta={"epochs": args.epochs, "learning_rate": args.learning_rate}
        )
    else:
        print(f"Saving full fine-tuned model to {args.output}...")
        save_model(model, args.output, used_lora=False)

if __name__ == "__main__":
    main()