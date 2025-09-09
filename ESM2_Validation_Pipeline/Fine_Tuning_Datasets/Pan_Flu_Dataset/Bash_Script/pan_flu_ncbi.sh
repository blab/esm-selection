#!/usr/bin/env bash
set -euo pipefail

segments=(PB2 PB1 PA HA NP NA M NS)
per_seg=375  # 3000 / 8
outdir="flu_out"
mkdir -p "$outdir"

# If you're on macOS without GNU coreutils, install coreutils, then alias shuf -> gshuf
shuf_cmd="shuf"; command -v shuf >/dev/null 2>&1 || shuf_cmd="gshuf"

for seg in "${segments[@]}"; do
  query="\"Influenza A virus\"[Organism] AND ${seg}[Gene] \
AND (\"Homo sapiens\"[All Fields] OR Aves[All Fields]) \
AND NOT H3N2[All Fields]"

  # 1) get a list of accessions
  esearch -db nuccore -query "$query" -usehistory y \
  | efetch -format acc \
  | $shuf_cmd -n "$per_seg" > "$outdir/${seg}.acc"

  # 2) fetch the fasta for those accessions (using epost for robustness)
  epost -db nuccore -input "$outdir/${seg}.acc" \
  | efetch -format fasta > "$outdir/${seg}.fasta"
done

# combine into one FASTA
cat "$outdir"/*.fasta > flu_3000_equal_per_segment_noH3N2.fasta
echo "Wrote flu_3000_equal_per_segment_noH3N2.fasta"

