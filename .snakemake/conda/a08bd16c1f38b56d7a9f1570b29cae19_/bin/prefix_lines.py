#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a08bd16c1f38b56d7a9f1570b29cae19_/bin/python

"""
Simple script to add a prefix to every line in a file.
"""

import sys

for line in sys.stdin:
    print(sys.argv[1] + line, end=" ")
