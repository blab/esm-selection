#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a346902d4ce1c004fa4946f9ad3317cf_/bin/python

"""
Simple script to add a prefix to every line in a file.
"""

import sys

for line in sys.stdin:
    print(sys.argv[1] + line, end=" ")
