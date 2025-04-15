#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a08bd16c1f38b56d7a9f1570b29cae19_/bin/python

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
