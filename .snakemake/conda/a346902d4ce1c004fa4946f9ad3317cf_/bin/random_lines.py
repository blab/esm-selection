#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a346902d4ce1c004fa4946f9ad3317cf_/bin/python

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
