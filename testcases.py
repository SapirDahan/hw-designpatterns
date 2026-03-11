"""
Functions for parsing and running testcases from the testcases.txt file.

Written by Claude Code.
"""

def parse_testcases(filename:str):
    cases = []
    current = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("case="):
                current = {"name": line[5:]}
            elif line.startswith("input="):
                current["input"] = eval(line[6:])
            elif line.startswith("output="):
                current["output"] = eval(line[7:])
                cases.append(current)
    return cases
