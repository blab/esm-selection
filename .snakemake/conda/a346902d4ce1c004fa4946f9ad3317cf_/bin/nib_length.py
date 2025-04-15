#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a346902d4ce1c004fa4946f9ad3317cf_/bin/python

"""
Print the number of bases in a nib file.

usage: %prog nib_file
"""

import sys

from bx.seq import nib as seq_nib

with open(sys.argv[1]) as f:
    nib = seq_nib.NibFile(f)
print(nib.length)
