#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a08bd16c1f38b56d7a9f1570b29cae19_/bin/python

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
