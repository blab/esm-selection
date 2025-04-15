#!/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/.snakemake/conda/c68281f7da3953da89c92cd799f1e235_/bin/python

"""
Script to select random lines from a file. Reads entire file into
memory!

TODO: Replace this with a more elegant implementation.
"""

import random
import sys

ndesired = int(sys.argv[1])

for line in random.sample(sys.stdin.readlines(), ndesired):
    print(line, end=" ")
