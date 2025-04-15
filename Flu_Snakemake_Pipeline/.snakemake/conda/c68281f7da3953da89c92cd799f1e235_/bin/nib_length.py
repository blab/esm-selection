#!/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/.snakemake/conda/c68281f7da3953da89c92cd799f1e235_/bin/python

"""
Print the number of bases in a nib file.

usage: %prog nib_file
"""

import sys

from bx.seq import nib as seq_nib

with open(sys.argv[1]) as f:
    nib = seq_nib.NibFile(f)
print(nib.length)
