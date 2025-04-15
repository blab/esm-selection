#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a346902d4ce1c004fa4946f9ad3317cf_/bin/python

"""
Read a file from stdin, split each line and write fields one per line to
stdout.

TODO: is this really that useful?
"""

import sys

for line in sys.stdin:
    for field in line.split():
        print(field)
