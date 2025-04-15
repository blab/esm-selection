#!/Users/Carlos/Desktop/Bedford/esm-selection/.snakemake/conda/a08bd16c1f38b56d7a9f1570b29cae19_/bin/python

"""
Translate lists of space separated integers (magnitude less than 62) and print
as strings of alphanumeric characters. This is useful mainly for some machine
learning algorithms that only take string input.

usage: %prog < int_seqs > strings
"""

import sys

table = "012345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def main():
    for line in sys.stdin:
        ints = [int(f) for f in line.split()]
        if max(ints) > len(table):
            raise ValueError("Alphabet size too large!")
        print(str.join("", [table[i] for i in ints]))


if __name__ == "__main__":
    main()
