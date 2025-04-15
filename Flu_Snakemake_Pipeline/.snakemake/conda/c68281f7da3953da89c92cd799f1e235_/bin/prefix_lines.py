#!/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/.snakemake/conda/c68281f7da3953da89c92cd799f1e235_/bin/python

"""
Simple script to add a prefix to every line in a file.
"""

import sys

for line in sys.stdin:
    print(sys.argv[1] + line, end=" ")
