#!/usr/bin/env python3
"""
Build a 3,000-sequence Influenza A multi-FASTA from NCBI, with:
  - Equal parts human and avian overall (1,500 each)
  - Equal number of FASTAs per segment (375 per each of the 8 segments)
  - Excluding H3N2
  - Outputs a single multi-FASTA and a summary CSV

Key change vs v1:
  Instead of relying on [Host]/[Gene] search fields (spotty on Nucleotide),
  this script grabs a broad candidate set per segment using organism + gene keywords,
  then fetches GenBank records to read /host and 'gene' qualifiers reliably.

Requires:
  - Python 3.8+
  - Biopython (`pip install biopython`)

Usage:
  python build_influenza_a_dataset.py --email you@uni.edu [--api-key YOUR_NCBI_API_KEY] [--out-prefix influenzaA_3000_excl_H3N2]
"""

import argparse
import csv
import random
import sys
import time
import re
from collections import defaultdict, Counter

from Bio import Entrez, SeqIO

# ---- Configuration you can tweak if needed ----

SEGMENTS = [
    ("PB2", 188, 187),  # segment 1
    ("PB1", 188, 187),  # segment 2
    ("PA", 188, 187),   # segment 3
    ("HA", 188, 187),   # segment 4
    ("NP", 187, 188),   # segment 5
    ("NA", 187, 188),   # segment 6
    ("M",  187, 188),   # segment 7
    ("NS", 187, 188),   # segment 8
]

DB = "nucleotide"
RETMAX_PER_QUERY = 50000   # broad but reasonable
FETCH_BATCH = 200
REQUEST_DELAY = 0.34        # ~3 req/sec
RANDOM_SEED = 42

# Host classification keywords (case-insensitive)
HUMAN_MARKERS = [
    "homo sapiens", "human"
]

AVIAN_MARKERS = [
    "avian", "bird", "aves", "gallus gallus", "anas platyrhynchos", "anser anser",
    "chicken", "duck", "turkey", "goose", "mallard", "quail", "gull", "pigeon",
    "sparrow", "teal", "wigeon", "swan"
]

# Regex to detect H3N2 in titles/notes/subtype text (handles variants like "H 3 N 2")
H3N2_REGEX = re.compile(r'h\s*3\s*n\s*2', re.IGNORECASE)

# -----------------------------------------------

def esearch_ids(term: str, retmax: int) -> list[str]:
    handle = Entrez.esearch(db=DB, term=term, retmax=retmax, usehistory="n")
    rec = Entrez.read(handle)
    handle.close()
    return rec.get("IdList", [])

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def extract_host(rec) -> str | None:
    """Return 'human', 'avian', or None based on source /host and any text fields."""
    # 1) From source feature qualifiers
    for feat in rec.features:
        if feat.type == "source":
            host_vals = feat.qualifiers.get("host", [])
            for h in host_vals:
                hl = h.lower()
                if any(m in hl for m in HUMAN_MARKERS):
                    return "human"
                if any(m in hl for m in AVIAN_MARKERS):
                    return "avian"
    # 2) Fallback to scanning annotations / description
    blob = " ".join([rec.description or ""] + [str(rec.annotations.get("comment", ""))]).lower()
    if any(m in blob for m in HUMAN_MARKERS):
        return "human"
    if any(m in blob for m in AVIAN_MARKERS):
        return "avian"
    return None

def is_h3n2(rec) -> bool:
    """Detect H3N2 subtype by scanning source and free text."""
    # Look at features (source and CDS notes) and description
    texts = [rec.description or ""]
    for feat in rec.features:
        # subtypes sometimes appear in 'note' or 'isolate' or 'subtype' qualifiers
        for key in ("note", "isolate", "subtype", "serotype", "strain"):
            vals = feat.qualifiers.get(key, [])
            for v in vals:
                texts.append(v)
    blob = " ".join(texts)
    return bool(H3N2_REGEX.search(blob))

def record_has_gene(rec, gene_symbol: str) -> bool:
    """Confirm record contains the target gene (PB2, PB1, etc.) via feature 'gene' or CDS product."""
    g = gene_symbol.lower()
    for feat in rec.features:
        if feat.type in ("gene", "CDS"):
            if "gene" in feat.qualifiers and any(g == x.lower() for x in feat.qualifiers["gene"]):
                return True
            # Some CDS use product like "polymerase PB2"
            if "product" in feat.qualifiers and any(g in x.lower() for x in feat.qualifiers["product"]):
                return True
    # Fallback: look in description
    if gene_symbol.lower() in (rec.description or "").lower():
        return True
    return False

def search_candidates_for_gene(gene: str) -> list[str]:
    """
    Broad query: organism + gene keywords + exclude H3N2 by text if present.
    We do NOT filter host here; host filtering is done by reading GenBank records.
    """
    # txid11320 = Influenza A virus
    # Use gene keyword in title/all fields to capture segment-specific records.
    term = f'(txid11320[Organism:exp]) AND ({gene}[All Fields] OR {gene}[Title])'
    ids = esearch_ids(term, retmax=RETMAX_PER_QUERY)
    return ids

def fetch_gb_records(ids: list[str]):
    """Yield SeqRecord objects fetched in GenBank format for given nucleotide IDs."""
    for chunk in batched(ids, FETCH_BATCH):
        h = Entrez.efetch(db=DB, id=",".join(chunk), rettype="gb", retmode="text")
        for rec in SeqIO.parse(h, "gb"):
            yield rec
        h.close()
        time.sleep(REQUEST_DELAY)

def main():
    parser = argparse.ArgumentParser(description="Build a 3,000-sequence Influenza A multi-FASTA (equal human/avian, equal per-segment, exclude H3N2).")
    parser.add_argument("--email", required=True, help="Your email (required by NCBI).")
    parser.add_argument("--api-key", default=None, help="NCBI API key (optional but recommended).")
    parser.add_argument("--out-prefix", default="influenzaA_3000_excl_H3N2", help="Output file prefix (default: influenzaA_3000_excl_H3N2).")
    parser.add_argument("--max-candidates-per-segment", type=int, default=20000, help="Cap on candidate records scanned per segment (default 20000).")
    parser.add_argument("--dry-run", action="store_true", help="Build queries and show counts without downloading FASTA.")
    args = parser.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    random.seed(RANDOM_SEED)

    out_fasta = f"{args.out_prefix}.fasta"
    out_summary = f"{args.out_prefix}__summary.csv"

    summary_rows = []
    selected_ids = []

    for gene, human_quota, avian_quota in SEGMENTS:
        print(f"[SEGMENT] {gene} - target human {human_quota}, avian {avian_quota}", file=sys.stderr)
        ids = search_candidates_for_gene(gene)
        if not ids:
            print(f"[WARN] No IDs from esearch for {gene}", file=sys.stderr)
            continue
        # Limit candidates to a cap for speed
        ids = ids[:args.max_candidates_per_segment]

        human_pool = []
        avian_pool = []

        scanned = 0
        for rec in fetch_gb_records(ids):
            scanned += 1
            # Ensure record actually corresponds to the intended gene
            if not record_has_gene(rec, gene):
                continue
            # Exclude H3N2
            if is_h3n2(rec):
                continue
            # Host classify
            host = extract_host(rec)
            if host == "human":
                human_pool.append(rec.id)
            elif host == "avian":
                avian_pool.append(rec.id)

        # Randomly sample quotas
        random.shuffle(human_pool)
        random.shuffle(avian_pool)
        human_picked = human_pool[:human_quota]
        avian_picked = avian_pool[:avian_quota]

        selected_ids.extend(human_picked + avian_picked)

        summary_rows.append({
            "segment": gene,
            "host_group": "human",
            "requested": human_quota,
            "selected": len(human_picked),
            "available_hits": len(human_pool),
            "scanned_candidates": scanned
        })
        summary_rows.append({
            "segment": gene,
            "host_group": "avian",
            "requested": avian_quota,
            "selected": len(avian_picked),
            "available_hits": len(avian_pool),
            "scanned_candidates": scanned
        })

        print(f"[INFO] {gene}: scanned {scanned} recs | human pool={len(human_pool)} | avian pool={len(avian_pool)}", file=sys.stderr)

    # Deduplicate IDs
    selected_ids = list(dict.fromkeys(selected_ids))
    total_selected = len(selected_ids)
    target_total = sum(h + a for _, h, a in SEGMENTS)

    if total_selected < target_total:
        print(f"[WARN] Selected {total_selected} sequences vs target {target_total}. The final FASTA may be short.", file=sys.stderr)
    elif total_selected > target_total:
        # Shouldn't happen, but trim if duplicates got removed unevenly
        selected_ids = selected_ids[:target_total]
        total_selected = len(selected_ids)

    # Write summary
    with open(out_summary, "w", newline="") as f:
        fieldnames = ["segment", "host_group", "requested", "selected", "available_hits", "scanned_candidates"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    if args.dry_run:
        print(f"[DRY RUN] Wrote summary to {out_summary}. No FASTA downloaded.")
        return

    # Fetch FASTA for selected IDs
    fasta_written = 0
    with open(out_fasta, "w") as out:
        for chunk in batched(selected_ids, 400):
            h = Entrez.efetch(db=DB, id=",".join(chunk), rettype="fasta", retmode="text")
            out.write(h.read())
            h.close()
            time.sleep(REQUEST_DELAY)
            fasta_written += len(chunk)

    print(f"[DONE] Wrote multi-FASTA: {out_fasta} ({fasta_written} sequences)", file=sys.stderr)
    print(f"[DONE] Wrote summary CSV: {out_summary}", file=sys.stderr)
    print("Tip: Use `seqkit stats` to verify counts and lengths. If a bucket is short, increase --max-candidates-per-segment or relax filters.", file=sys.stderr)

if __name__ == "__main__":
    main()
