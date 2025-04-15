#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a346902d4ce1c004fa4946f9ad3317cf_/bin/python

"""
Print number of bases covered by all intervals in a bed file (bases covered by
more than one interval are counted only once). Multiple bed files can be
provided on the command line or to stdin.

usage: %prog bed files ...
"""

import fileinput
import sys

from bx.bitset_builders import binned_bitsets_from_file

bed_filenames = sys.argv[1:]
if bed_filenames:
    input = fileinput.input(bed_filenames)
else:
    input = sys.stdin

bitsets = binned_bitsets_from_file(input)

total = 0
for chrom in bitsets:
    total += bitsets[chrom].count_range(0, bitsets[chrom].size)

print(total)
