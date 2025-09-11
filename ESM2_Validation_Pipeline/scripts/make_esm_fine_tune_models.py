#!/usr/bin/env python3

import torch
import esm
import argparse
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
import math


class ProteinDataset(Dataset):
    """Dataset class for protein sequences."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ---- Reproducibility ----
randseed = 12
torch.backends.cudnn.deterministic = True
random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed(randseed)
np.random.seed(randseed)


# -----------------------------
#           ARGS
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 on protein sequences."
    )
    parser.add_argument("--input", type=str, default="alignment.fasta",
                        help="Input FASTA file containing training sequences.")
    parser.add_argument("--output", type=str, default="models/esm.bin",
                        help="Path to save the fine-tuned model state_dict.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fine-tuning.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup proportion of total steps.")
    parser.add_argument("--schedule", choices=["cosine", "linear", "none"], default="cosine",
                        help="LR schedule.")
    parser.add_argument("--model", type=str,
                        choices=["esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"],
                        default="esm2_t33_650M_UR50D", help="Specify which ESM-2 model to use.")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use mixed precision (recommended on CUDA for less memory).")
    parser.add_argument("--segment", type=str, default=None,
                        help="Optional segment name for identification.")

    return parser.parse_args()


# -----------------------------
#         DATA UTILS
# -----------------------------
def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Replace J with X in protein sequences
        seq_str = str(record.seq).replace("J", "X")
        sequences.append((record.id, seq_str))
    return sequences


# -----------------------------
#      MASKING / OBJECTIVE
# -----------------------------
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


# -----------------------------
#       TRAIN / EVAL
# -----------------------------


def train_model(model, dataloader, batch_converter, optimizer, scheduler, device, epochs,
                mask_token_idx, vocab_size, repr_layer, use_amp=False, special_token_idxs=None):
    """Train the model."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        steps_this_epoch = 0

        for i, batch in enumerate(dataloader):
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)

            masked_tokens, labels = mask_tokens(
                batch_tokens, mask_token_idx, vocab_size, device, special_token_idxs
            )

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    results = model(masked_tokens, repr_layers=[repr_layer])
                    logits = results["logits"]
                    loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                results = model(masked_tokens, repr_layers=[repr_layer])
                logits = results["logits"]
                loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            lr_now = optimizer.param_groups[0]['lr']
            steps_this_epoch += 1
            total_loss += loss.item()

            print(f"Step {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f} - LR: {lr_now:.2e}")

        avg_train = total_loss / max(1, steps_this_epoch)
        print(f"Epoch {epoch + 1} completed. Average Train Loss: {avg_train:.4f}")


# -----------------------------
#        SAVE HELPERS
# -----------------------------
def save_model(model, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(model.state_dict(), output_file)
    print(f"Model saved to {output_file}")


# -----------------------------
#        LR SCHEDULER
# -----------------------------
def build_scheduler(optimizer, total_steps, warmup_ratio=0.05, schedule="cosine"):
    if schedule == "none":
        return None
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
#            MAIN
# -----------------------------
def main():
    args = parse_arguments()

    # --- derive segment name early ---
    segment_name = (args.segment or os.path.splitext(os.path.basename(args.input))[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_layer_map = {
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48
    }
    repr_layer = model_layer_map[args.model]

    print(f"Loading {args.model} model...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    batch_converter = alphabet.get_batch_converter()
    mask_token_idx = getattr(alphabet, "mask_idx", None)
    vocab_size = len(alphabet)

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
    model.train()

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    train_sequences = load_sequences(args.input)

    # Heuristic batch sizes
    if args.model == "esm2_t33_650M_UR50D":
        batches = 8
    elif args.model == "esm2_t36_3B_UR50D":
        batches = 2
    else:
        batches = 1

    dataset = ProteinDataset(train_sequences)
    collate = lambda x: x
    dataloader = DataLoader(dataset, batch_size=batches, shuffle=True, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(dataloader))
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio, args.schedule)

    print("Starting fine-tuning...")
    train_model(
        model, dataloader, batch_converter, optimizer, scheduler, device, args.epochs,
        mask_token_idx, vocab_size, repr_layer, use_amp=args.use_amp, special_token_idxs=special_token_idxs
    )

    # Save model
    print(f"Saving fine-tuned model to {args.output}...")
    save_model(model, args.output)


if __name__ == "__main__":
    main()
