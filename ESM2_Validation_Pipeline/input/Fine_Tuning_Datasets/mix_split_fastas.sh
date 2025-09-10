#!/usr/bin/env bash
set -euo pipefail

A="ESM_1965_Full.fasta"
B="H3N2_Dataset_Full.fasta"
C="pan_flu_1965_Full.fasta"

SEED=${SEED:-42}
TOTAL_PAIRS=${TOTAL_PAIRS:-0}
TOTAL_TRIPLE=${TOTAL_TRIPLE:-0}

min2(){ if [ "$1" -le "$2" ]; then echo "$1"; else echo "$2"; fi; }
min3(){ echo "$(min2 "$1" "$(min2 "$2" "$3")")"; }
count(){ seqkit stats -T "$1" | awk 'NR==2{print $4}'; }

nA=$(count "$A"); nB=$(count "$B"); nC=$(count "$C") >&2
echo "Seq counts → A=$nA  B=$nB  C=$nC" >&2

if [ "$TOTAL_PAIRS" -gt 0 ]; then half=$((TOTAL_PAIRS/2)); else half=0; fi
pickAB=$((1965/2))
pickBC=$((1965/2))
pickAC=$((1965/2))

if [ "$TOTAL_TRIPLE" -gt 0 ]; then third=$((TOTAL_TRIPLE/3)); else third=0; fi
pickABC=$((1965/3))

# AB 50/50
seqkit sample -s "$((SEED+1))" -n "$pickAB" "$A" > __A_AB.fasta
seqkit sample -s "$((SEED+2))" -n "$pickAB" "$B" > __B_AB.fasta
cat __A_AB.fasta __B_AB.fasta | seqkit shuffle -s "$((SEED+3))" > mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_Full.fasta

# BC 50/50
seqkit sample -s "$((SEED+4))" -n "$pickBC" "$B" > __B_BC.fasta
seqkit sample -s "$((SEED+5))" -n "$pickBC" "$C" > __C_BC.fasta
cat __B_BC.fasta __C_BC.fasta | seqkit shuffle -s "$((SEED+6))" > mix_BC_50_H3N2_Dataset_Full_50_pan_flu_1965_Full.fasta

# AC 50/50
seqkit sample -s "$((SEED+7))" -n "$pickAC" "$A" > __A_AC.fasta
seqkit sample -s "$((SEED+8))" -n "$pickAC" "$C" > __C_AC.fasta
cat __A_AC.fasta __C_AC.fasta | seqkit shuffle -s "$((SEED+9))" > mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full.fasta

# ABC 1/3 each
seqkit sample -s "$((SEED+10))" -n "$pickABC" "$A" > __A_ABC.fasta
seqkit sample -s "$((SEED+11))" -n "$pickABC" "$B" > __B_ABC.fasta
seqkit sample -s "$((SEED+12))" -n "$pickABC" "$C" > __C_ABC.fasta
cat __A_ABC.fasta __B_ABC.fasta __C_ABC.fasta | seqkit shuffle -s "$((SEED+13))" > mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_Full_33_pan_flu_1965_Full.fasta

rm __*_AB.fasta __*_BC.fasta __*_AC.fasta __*_ABC.fasta
echo "Done."
