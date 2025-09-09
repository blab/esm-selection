#!/usr/bin/env python3
"""
Run a Learning Rate (LR) range test for ESM fine-tuning and write chosen LR to JSON.

Example:
  python scripts/run_lr_finder.py \
    --input results/fine_tune_fastas/<ps>/fine_tune_fasta_ha.fasta \
    --model esm2_t33_650M_UR50D --segment ha \
    --lora on --steps 100 --lr-min 1e-7 --lr-max 5e-4 \
    --out results/lr_finder/<ps>/lr_ha.json \
    --plot results/lr_finder/<ps>/plots/ha_lr_finder.png
"""

from __future__ import annotations
import argparse, os, json, math, random
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from Bio import SeqIO
import matplotlib.pyplot as plt
import esm

# ---- Optional LoRA ----
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


# ----------------------------- utils
def set_seed(seed=12):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ProteinDataset(Dataset):
    def __init__(self, sequences: list[tuple[str,str]]):
        self.sequences = sequences
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i]


def load_sequences(fasta_file: str) -> list[tuple[str,str]]:
    seqs = []
    for r in SeqIO.parse(fasta_file, "fasta"):
        seqs.append((r.id, str(r.seq)))
    return seqs


# ----------------------------- LR finder core
def _next_lr(cur, lr_min, lr_max, step_idx, total_steps, mode):
    if total_steps <= 1: return lr_max
    if mode == "exp":
        mult = (lr_max / lr_min) ** (1.0 / max(1, total_steps - 1))
        return cur * mult
    elif mode == "linear":
        return lr_min + (lr_max - lr_min) * (step_idx + 1) / max(1, total_steps - 1)
    else:
        raise ValueError("mode must be 'exp' or 'linear'")

def lr_range_test(model, dataloader, batch_converter, device, mask_token_idx, vocab_size, repr_layer,
                  steps=100, lr_min=1e-7, lr_max=1e-3, mode="exp",
                  weight_decay=0.01, use_amp=False, smoothing=0.98, stop_divergence=10.0):
    model_was_training = model.training
    model.train()
    with torch.no_grad():
        base_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr_min, weight_decay=weight_decay, betas=(0.9,0.999))
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    lrs, losses = [], []
    beta, avg_loss, best = smoothing, 0.0, float("inf")
    data_iter = iter(dataloader)
    lr = lr_min
    reason, stopped = "", False

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        _, _, toks = batch_converter(batch)
        toks = toks.to(device)

        # MLM-style masking
        labels = toks.clone()
        prob = torch.full(labels.shape, 0.15, device=device)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        # 80% mask
        mask_idx = masked & (torch.rand(labels.shape, device=device) < 0.8)
        toks[mask_idx] = mask_token_idx
        # 10% random
        rnd_idx = masked & (torch.rand(labels.shape, device=device) < 0.1)
        rnd = torch.randint(0, vocab_size, labels.shape, device=device)
        toks[rnd_idx] = rnd[rnd_idx]

        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(toks, repr_layers=[repr_layer])["logits"]
                loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
            torch.cuda.synchronize()
            loss.backward()
            optimizer.step()
        else:
            logits = model(toks, repr_layers=[repr_layer])["logits"]
            loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

        # smooth loss
        loss_val = float(loss.item())
        avg_loss = beta*avg_loss + (1-beta)*loss_val
        smoothed = avg_loss / (1 - (beta ** (step+1)))

        lrs.append(lr); losses.append(smoothed)
        if step > 5 and smoothed > stop_divergence * best:
            reason = f"Stopped early: {smoothed:.3f} > {stop_divergence}×{best:.3f}"
            stopped = True
            break
        if smoothed < best: best = smoothed
        lr = _next_lr(lr, lr_min, lr_max, step, steps, mode)

    # restore
    model.load_state_dict(base_state, strict=True)
    if model_was_training: model.train()
    else: model.eval()
    return {"lrs": lrs, "losses": losses, "stopped_early": stopped, "reason": reason}

def pick_lr_from_curve(lrs: list[float], losses: list[float], strategy="slope", factor=3.0, lr_floor=1e-8) -> float:
    lrs = np.asarray(lrs); losses = np.asarray(losses)
    if strategy == "slope":
        x = np.log10(lrs)
        slope = np.gradient(losses, x)
        idx = int(np.argmin(slope))
        base = float(lrs[idx])
        chosen = base / max(1.0, factor)
    else:
        idx = int(np.argmin(losses))
        base = float(lrs[idx])
        chosen = base / 10.0
    return float(max(lr_floor, chosen))

def plot_curve(lrs, losses, out_png=None, title="LR range test"):
    plt.figure()
    plt.plot(lrs, losses, marker="o", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Learning rate"); plt.ylabel("Smoothed training loss")
    plt.title(title); plt.grid(True)
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close()


# ----------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="training FASTA")
    ap.add_argument("--model", required=True, choices=["esm2_t33_650M_UR50D","esm2_t36_3B_UR50D","esm2_t48_15B_UR50D"])
    ap.add_argument("--segment", required=True)
    ap.add_argument("--lora", choices=["on","off","auto"], default="off")
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.0)

    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--min-steps-per-epoch", type=int, default=None)

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--lr-max", type=float, default=5e-4)
    ap.add_argument("--mode", choices=["exp","linear"], default="exp")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--use-amp", action="store_true")

    ap.add_argument("--strategy", choices=["slope","min/10"], default="slope")
    ap.add_argument("--factor", type=float, default=3.0)

    ap.add_argument("--out", required=True, help="JSON path to write chosen LR")
    ap.add_argument("--plot", required=False, help="PNG plot path")

    args = ap.parse_args()
    set_seed(12)

    # load model
    repr_map = {"esm2_t33_650M_UR50D":33, "esm2_t36_3B_UR50D":36, "esm2_t48_15B_UR50D":48}
    repr_layer = repr_map[args.model]
    model, alphabet = getattr(esm.pretrained, args.model)()
    batch_converter = alphabet.get_batch_converter()
    mask_token_idx = getattr(alphabet, "mask_idx", None)
    vocab_size = len(alphabet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # LoRA (if requested)
    used_lora = False
    if args.lora in ("on","auto"):
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("LoRA requested but 'peft' is not installed.")
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         bias="none", target_modules=target_modules, task_type=None)
        model = get_peft_model(model, cfg)
        used_lora = True

    # data
    seqs = load_sequences(args.input)
    ds = ProteinDataset(seqs)
    bs = args.batch_size if args.batch_size is not None else (8 if args.model=="esm2_t33_650M_UR50D" else (2 if args.model=="esm2_t36_3B_UR50D" else 1))
    sampler = None
    if args.min_steps_per_epoch and len(ds)>0:
        num_samples = max(len(ds), args.min_steps_per_epoch * bs)
        sampler = RandomSampler(ds, replacement=True, num_samples=num_samples)
    dl = DataLoader(ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler, drop_last=False, collate_fn=lambda x: x)

    # run finder
    res = lr_range_test(model, dl, batch_converter, device, mask_token_idx, vocab_size, repr_layer,
                        steps=args.steps, lr_min=args.lr_min, lr_max=args.lr_max, mode=args.mode,
                        weight_decay=(0.0 if used_lora else args.weight_decay), use_amp=args.use_amp)
    if args.plot:
        title = f"{args.segment} • {args.model} • {'LoRA' if used_lora else 'FT'}"
        plot_curve(res["lrs"], res["losses"], out_png=args.plot, title=title)

    chosen = pick_lr_from_curve(res["lrs"], res["losses"], strategy=args.strategy, factor=args.factor, lr_floor=max(1e-8, args.lr_min))
    out_payload = {
        "segment": args.segment,
        "model": args.model,
        "used_lora": used_lora,
        "steps": args.steps,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "strategy": args.strategy,
        "factor": args.factor,
        "chosen_learning_rate": chosen,
        "stopped_early": res["stopped_early"],
        "reason": res["reason"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[LR-FINDER] Chosen LR: {chosen:.3e} → wrote {args.out}")
    if args.plot:
        print(f"[LR-FINDER] Plot: {args.plot}")


if __name__ == "__main__":
    main()

