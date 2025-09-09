#!/usr/bin/env python3
"""
Fine-tune ESM-2 (with optional LoRA), log train/val losses, and generate plots.

Adds per-step validation (streaming):
- Records train_loss_per_step
- Records val_loss_per_step at regular intervals (and always at step 1 + end of epoch)
- Plots per-step and epoch-level curves

Reliability improvements:
- Metrics file is written atomically with fsync
- Periodic snapshots during training via --metrics-write-every-steps
- Snapshots on SIGINT/SIGTERM and interpreter exit
- Optional --run-id appended to metrics filename for per-run uniqueness

Requirements:
  pip install matplotlib
"""

import torch
import esm
import argparse
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
import json
import math
import glob
import time
import matplotlib.pyplot as plt
import re
import signal
import atexit

# ---- PEFT / LoRA (optional) ----
try:
    from peft import LoraConfig, get_peft_model, PeftModel  # noqa: F401
except Exception:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


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
        description="Fine-tune ESM-2 on protein sequences with optional LoRA adapters, "
                    "log metrics, and plot train/val losses."
    )
    parser.add_argument("--input", type=str, default="alignment.fasta",
                        help="Input FASTA file containing training+val sequences (if --val-input not provided).")
    parser.add_argument("--val-input", type=str, default=None,
                        help="Optional FASTA for validation; if omitted, use --val-ratio split from --input.")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="If --val-input not given, fraction of sequences used for validation.")
    parser.add_argument("--output", type=str, default="models/esm.bin",
                        help="If LoRA: JSON pointer path. If non-LoRA: full model state_dict path.")
    parser.add_argument(
        "--peft-output",
        type=str,
        default=None,
        help="Directory to save LoRA adapters (when LoRA is enabled). If omitted and --lora=auto, LoRA stays off."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fine-tuning.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (non-LoRA only).")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup proportion of total steps.")
    parser.add_argument("--schedule", choices=["cosine", "linear", "none"], default="cosine",
                        help="LR schedule.")
    parser.add_argument("--model", type=str,
                        choices=["esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"],
                        default="esm2_t33_650M_UR50D", help="Specify which ESM-2 model to use.")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use mixed precision (recommended on CUDA for less memory).")
    parser.add_argument("--segment", type=str, default=None,
                        help="Optional segment name; used in metrics and plot titles.")
    parser.add_argument(
        "--lora",
        choices=["on", "off", "auto"],
        default="auto",
        help="Enable LoRA. 'auto' enables LoRA iff --peft-output is provided."
    )
    parser.add_argument("--lora-r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (≈ rank).")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    # Logging / plotting
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory to write metrics JSON for plotting. "
                             "Default: <dirname(--output)>/metrics")
    parser.add_argument("--plots-out", type=str, default=None,
                        help="Combined grid figure path (epoch-level). "
                             "Default: <dirname(--output)>/plots/loss_by_segment.png")
    parser.add_argument("--plots-per-segment-dir", type=str, default=None,
                        help="Directory for one PNG per segment (epoch-level). "
                             "Default: <dirname(--output)>/plots/by_segment")
    parser.add_argument("--plot-after", action="store_true",
                        help="If set, generate plots right after training finishes.")

    # Per-step validation sampling frequency (DEFAULT: every step)
    parser.add_argument("--eval-every-steps", type=int, default=1,
                        help="How often to compute validation loss during training (in steps).")
    parser.add_argument("--val-steps-per-eval", type=int, default=1,
                        help="How many validation batches to average per evaluation tick.")

    # Reliability / uniqueness
    parser.add_argument("--metrics-write-every-steps", type=int, default=0,
                        help="If >0, also write metrics snapshot every N train steps (atomic).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional run identifier to make metrics filename unique (e.g. 20250902-113045).")

    return parser.parse_args()


# -----------------------------
#         DATA UTILS
# -----------------------------
def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
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
def evaluate_model(model, dataloader, batch_converter, device, mask_token_idx, vocab_size,
                   repr_layer, use_amp=False, special_token_idxs=None):
    """Compute average MLM loss on a validation dataloader."""
    model_was_training = model.training
    model.eval()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_batches = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)

            masked_tokens, labels = mask_tokens(
                batch_tokens.clone(), mask_token_idx, vocab_size, device, special_token_idxs
            )

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(masked_tokens, repr_layers=[repr_layer])["logits"]
                    loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                logits = model(masked_tokens, repr_layers=[repr_layer])["logits"]
                loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            total_batches += 1

    if model_was_training:
        model.train()
    return total_loss / max(1, total_batches)


def train_model(model, dataloader, val_dataloader, batch_converter, optimizer, scheduler, device, epochs,
                mask_token_idx, vocab_size, repr_layer, use_amp=False, special_token_idxs=None,
                eval_every_steps=1, val_steps_per_eval=1, snapshot_cb=None, snapshot_every_steps=0):
    """
    Train the model and return a history dict with per-step and per-epoch losses.

    Streams per-step validation loss every `eval_every_steps` steps (default: 1),
    and ALWAYS at step 1 and the last step of each epoch.

    If snapshot_cb is provided, snapshots are written:
      - every `snapshot_every_steps` steps (>0),
      - at the end of each epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    history = {
        "train_loss_per_step": [],
        "lr_per_step": [],
        "train_loss_per_epoch": [],
        "val_loss_per_epoch": [],
        "steps_per_epoch": [],
        "val_loss_per_step": [],
        "val_loss_steps_at": [],
    }

    val_iter = iter(val_dataloader) if val_dataloader is not None else None

    global_step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        steps_this_epoch = 0
        steps_in_epoch = len(dataloader)

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
            global_step += 1
            steps_this_epoch += 1
            total_loss += loss.item()

            history["train_loss_per_step"].append(loss.item())
            history["lr_per_step"].append(lr_now)

            # ---- Streaming validation trigger logic ----
            is_eval_tick = False
            if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                is_eval_tick = True
            if global_step == 1:  # always sample at very first step
                is_eval_tick = True
            if i == steps_in_epoch - 1:  # always sample at end of each epoch
                is_eval_tick = True

            if is_eval_tick and val_dataloader is not None:
                model_was_training = model.training
                model.eval()
                with torch.no_grad():
                    vals = []
                    for _ in range(max(1, val_steps_per_eval)):
                        try:
                            vbatch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_dataloader)
                            vbatch = next(val_iter)
                        vlabels, vstrs, vtokens = batch_converter(vbatch)
                        vtokens = vtokens.to(device)
                        vmasked, vlabels2 = mask_tokens(
                            vtokens.clone(), mask_token_idx, vocab_size, device, special_token_idxs
                        )
                        if use_amp and device.type == "cuda":
                            with torch.cuda.amp.autocast():
                                vlogits = model(vmasked, repr_layers=[repr_layer])["logits"]
                                vloss = ce(vlogits.view(-1, vlogits.size(-1)), vlabels2.view(-1))
                        else:
                            vlogits = model(vmasked, repr_layers=[repr_layer])["logits"]
                            vloss = ce(vlogits.view(-1, vlogits.size(-1)), vlabels2.view(-1))
                        vals.append(vloss.item())
                if model_was_training:
                    model.train()

                history["val_loss_per_step"].append(float(np.mean(vals)))
                history["val_loss_steps_at"].append(global_step)

            # Periodic snapshot
            if snapshot_cb and snapshot_every_steps > 0 and (global_step % snapshot_every_steps == 0):
                snapshot_cb()

            print(f"Step {i + 1}/{steps_in_epoch} - Loss: {loss.item():.4f} - LR: {lr_now:.2e}")

        avg_train = total_loss / max(1, steps_in_epoch)
        history["train_loss_per_epoch"].append(avg_train)
        history["steps_per_epoch"].append(steps_this_epoch)

        # Full-epoch validation (kept for epoch metrics)
        if val_dataloader is not None:
            val_loss = evaluate_model(
                model, val_dataloader, batch_converter, device, mask_token_idx, vocab_size,
                repr_layer, use_amp=use_amp, special_token_idxs=special_token_idxs
            )
            history["val_loss_per_epoch"].append(val_loss)
            print(f"Epoch {epoch + 1} completed. Train: {avg_train:.4f}  |  Val: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1} completed. Average Train Loss: {avg_train:.4f}")

        # Snapshot at end of epoch
        if snapshot_cb:
            snapshot_cb()

    return history


# -----------------------------
#        SAVE HELPERS
# -----------------------------
def save_model(model, output_file, used_lora=False, peft_output_dir=None, base_model_name=None, extra_meta=None):
    if used_lora:
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError("PEFT model expected when used_lora=True (missing save_pretrained).")
        outdir = peft_output_dir or "adapters/esm_lora"
        os.makedirs(outdir, exist_ok=True)
        try:
            model.save_pretrained(outdir, safe_serialization=False)
        except TypeError:
            model.save_pretrained(outdir)
        print(f"LoRA adapters saved to {outdir}")
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


def save_metrics(metrics: dict, path: str):
    """Atomically write JSON with fsync, replacing the target path."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[metrics] wrote {path}")


# -----------------------------
#       LORA TOGGLING
# -----------------------------
def decide_lora_enabled(args):
    if args.lora == "on":
        return True
    if args.lora == "off":
        return False
    return args.peft_output is not None and str(args.peft_output).strip() != ""


def maybe_wrap_with_lora(model, args, enable_lora):
    if not enable_lora:
        return model, False
    if get_peft_model is None or LoraConfig is None:
        raise RuntimeError("peft is not installed, but LoRA was requested. Install with: pip install peft")
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=target_modules, task_type=None
    )
    peft_model = get_peft_model(model, lora_config)
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        trainables = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"Trainable params: {trainables:,} / {total:,}")
    return peft_model, True


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
#           PLOTTING
# -----------------------------
def _latest_runs_by_segment(metrics_glob):
    by_seg = {}
    for path in glob.glob(metrics_glob):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        segment = payload.get("segment", "default")
        tag = "lora" if payload.get("used_lora") else "ft"
        cur = by_seg.setdefault(segment, {})
        prev = cur.get(tag)
        mtime = os.path.getmtime(path)
        if prev is None or mtime > os.path.getmtime(prev[0]):
            cur[tag] = (path, payload)
    return by_seg


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _plot_segment(ax, segment, runs):
    ax.set_title(segment)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    def plot_run(run, label_prefix):
        hist = run.get("history", {})
        xs = list(range(1, len(hist.get("train_loss_per_epoch", [])) + 1))
        if hist.get("train_loss_per_epoch"):
            ax.plot(xs, hist["train_loss_per_epoch"], marker="o", label=f"{label_prefix} train (epoch)")
        if hist.get("val_loss_per_epoch"):
            ax.plot(xs, hist["val_loss_per_epoch"], linestyle="--", marker="o", label=f"{label_prefix} val (epoch)")

    if "ft" in runs:
        plot_run(runs["ft"][1], "FT")
    if "lora" in runs:
        plot_run(runs["lora"][1], "LoRA")

    if ax.lines:
        ax.legend(fontsize=8)


def make_plots(logdir, combined_out, per_segment_dir):
    metrics_glob = os.path.join(logdir, "*_metrics.json")
    groups = _latest_runs_by_segment(metrics_glob)

    if not groups:
        print("[plot] No metrics found; skipping epoch-level plot generation.")
        return

    n = len(groups)
    cols = 3 if n >= 3 else n
    rows = math.ceil(n / max(1, cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4), squeeze=False)

    for ax, (segment, runs) in zip(axes.flat, groups.items()):
        _plot_segment(ax, segment, runs)

    used = len(groups)
    for j in range(used, rows * cols):
        fig.delaxes(axes.flat[j])

    _ensure_dir(combined_out)
    plt.tight_layout()
    plt.savefig(combined_out, dpi=180)
    print(f"[plot] wrote {combined_out}")

    os.makedirs(per_segment_dir, exist_ok=True)
    for segment, runs in groups.items():
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 4))
        _plot_segment(ax_s, segment, runs)
        out_seg = os.path.join(per_segment_dir, f"{segment}_loss_epoch.png")
        plt.tight_layout()
        plt.savefig(out_seg, dpi=180)
        plt.close(fig_s)
        print(f"[plot] wrote {out_seg}")


def plot_per_step_curves(history, out_path, title="Per-step Training & Validation Loss"):
    train_y = history.get("train_loss_per_step", []) or []
    val_y   = history.get("val_loss_per_step", []) or []
    val_x   = history.get("val_loss_steps_at", []) or list(range(1, len(val_y)+1))

    if not train_y and not val_y:
        print("[plot] No per-step arrays; skipping.")
        return

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(10,5))
    if train_y:
        ax.plot(range(1, len(train_y)+1), train_y, label="Train loss (per step)")
    if val_y:
        ax.plot(val_x, val_y, linestyle="--", marker="o", label="Val loss (per step)")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"[plot] wrote {out_path}")


def _derive_default_paths(output_path, segment, tag, model, logdir_arg, plots_out_arg, plots_seg_dir_arg):
    """
    Compute run-rooted paths (next to --output) unless user provided explicit paths.
    Returns: (logdir, combined_out, per_segment_dir, per_step_plot_path)
    """
    run_root = os.path.dirname(os.path.abspath(output_path))
    # defaults rooted at run directory
    default_logdir = os.path.join(run_root, "metrics")
    default_plots_root = os.path.join(run_root, "plots")
    default_combined = os.path.join(default_plots_root, "loss_by_segment.png")
    default_seg_dir = os.path.join(default_plots_root, "by_segment")
    default_per_step = os.path.join(default_plots_root, f"{segment}_{tag}_loss_per_step.png")

    logdir = logdir_arg or default_logdir
    combined_out = plots_out_arg or default_combined
    per_seg_dir = plots_seg_dir_arg or default_seg_dir
    return logdir, combined_out, per_seg_dir, default_per_step


# -----------------------------
#       UTIL: SAFE NAMES
# -----------------------------
def _safe_name(s: str) -> str:
    """Sanitize strings for filenames/paths: keep alnum, dot, underscore, dash; replace others with '_'."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', (s or 'default'))


# -----------------------------
#            MAIN
# -----------------------------
def main():
    args = parse_arguments()

    # --- derive segment name early and sanitize ---
    segment_name = (args.segment or os.path.splitext(os.path.basename(args.input))[0])
    segment_name = _safe_name(segment_name)

    # --- decide LoRA ahead of time so we can compute default paths & tag ---
    enable_lora = decide_lora_enabled(args)
    tag = "lora" if enable_lora else "ft"

    # --- derive & announce where metrics/plots will land ---
    logdir, combined_out, per_segment_dir, per_step_plot_path = _derive_default_paths(
        args.output, segment_name, tag, args.model, args.logdir, args.plots_out, args.plots_per_segment_dir
    )
    os.makedirs(logdir, exist_ok=True)

    # Per-run uniqueness for metrics file
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    metrics_path = os.path.join(logdir, f"{segment_name}_{tag}_{args.model}_{run_id}_metrics.json")
    print(f"[metrics] will write to: {os.path.abspath(metrics_path)}")

    # State captured for snapshots/handlers
    history = {}
    used_lora = False

    def _current_metrics_payload():
        run_ts = int(time.time())
        return {
            "segment": segment_name,
            "model": args.model,
            "used_lora": used_lora,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay if not used_lora else 0.0,
            "schedule": args.schedule,
            "warmup_ratio": args.warmup_ratio,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            } if used_lora else None,
            "history": history,  # may be partial
            "timestamp": run_ts,
        }

    def _snapshot():
        try:
            save_metrics(_current_metrics_payload(), metrics_path)
        except Exception as e:
            print(f"[metrics] snapshot failed: {e}")

    # Ensure we write on normal interpreter shutdown, SIGINT, SIGTERM
    atexit.register(_snapshot)

    def _sig_handler(signum, frame):
        print(f"[metrics] caught signal {signum}, writing snapshot...")
        _snapshot()
        if signum == signal.SIGTERM:
            raise SystemExit(143)  # 128+15

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass  # Some environments may not allow setting handlers
    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    try:
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

        print(f"LoRA enabled: {enable_lora}")
        model, used_lora = maybe_wrap_with_lora(model, args, enable_lora)
        model.train()

        # Load sequences
        if args.val_input:
            print(f"Loading training sequences from {args.input}...")
            train_sequences = load_sequences(args.input)
            print(f"Loading validation sequences from {args.val_input}...")
            val_sequences = load_sequences(args.val_input)
        else:
            print(f"Loading sequences (will split) from {args.input}...")
            all_sequences = load_sequences(args.input)
            n_total = len(all_sequences)
            n_val = max(1, int(round(args.val_ratio * n_total)))
            random.shuffle(all_sequences)
            val_sequences = all_sequences[:n_val]
            train_sequences = all_sequences[n_val:]
            print(f"Split {n_total} seqs -> train={len(train_sequences)}  val={len(val_sequences)}")

        # Heuristic batch sizes
        if args.model == "esm2_t33_650M_UR50D":
            batches = 8
        elif args.model == "esm2_t36_3B_UR50D":
            batches = 2
        else:
            batches = 1

        dataset = ProteinDataset(train_sequences)
        valset = ProteinDataset(val_sequences)
        collate = lambda x: x
        dataloader = DataLoader(dataset, batch_size=batches, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(valset, batch_size=batches, shuffle=False, collate_fn=collate)

        params = [p for p in model.parameters() if p.requires_grad]
        if used_lora:
            optimizer = torch.optim.AdamW([{"params": params, "weight_decay": 0.0}],
                                          lr=args.learning_rate, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                          betas=(0.9, 0.999), weight_decay=args.weight_decay)

        total_steps = args.epochs * max(1, len(dataloader))
        scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio, args.schedule)

        print("Starting fine-tuning...")
        history = train_model(
            model, dataloader, val_loader, batch_converter, optimizer, scheduler, device, args.epochs,
            mask_token_idx, vocab_size, repr_layer, use_amp=args.use_amp, special_token_idxs=special_token_idxs,
            eval_every_steps=args.eval_every_steps, val_steps_per_eval=args.val_steps_per_eval,
            snapshot_cb=_snapshot, snapshot_every_steps=args.metrics_write_every_steps
        )

        # Save model
        if used_lora:
            outdir = args.peft_output or f"adapters/{args.model}_lora"
            print(f"Saving LoRA adapters to {outdir}...")
            save_model(
                model, args.output,
                used_lora=True,
                peft_output_dir=outdir,
                base_model_name=args.model,
                extra_meta={"epochs": args.epochs, "learning_rate": args.learning_rate,
                            "schedule": args.schedule, "warmup_ratio": args.warmup_ratio,
                            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                            "segment": segment_name}
            )
        else:
            print(f"Saving full fine-tuned model to {args.output}...")
            save_model(model, args.output, used_lora=False)

    finally:
        # Write final metrics (may be identical to last snapshot)
        save_metrics(_current_metrics_payload(), metrics_path)

        # Optional plotting after training if requested and we have some history
        if args.plot_after and history:
            plot_per_step_curves(
                history,
                out_path=per_step_plot_path,
                title=f"{segment_name} — {'LoRA' if used_lora else 'FT'} — {args.model} (per step)"
            )
            make_plots(logdir, combined_out, per_segment_dir)


if __name__ == "__main__":
    main()
