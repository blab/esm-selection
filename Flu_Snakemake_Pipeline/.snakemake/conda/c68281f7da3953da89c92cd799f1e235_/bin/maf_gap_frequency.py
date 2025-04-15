#!/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/.snakemake/conda/c68281f7da3953da89c92cd799f1e235_/bin/python

"""
Read a MAF from standard input and print the fraction of gap columns in
each block.

usage: %prog < maf > out
"""

import sys

import bx.align.maf


def main():
    for m in bx.align.maf.Reader(sys.stdin):
        gaps = 0
        for col in m.column_iter():
            if "-" in col:
                gaps += 1
        print(gaps / m.text_size)


if __name__ == "__main__":
    main()
