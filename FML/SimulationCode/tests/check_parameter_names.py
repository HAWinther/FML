#!/usr/bin/env python3
"""Check that every parameter name the SimulationCode source reads is in the parameter table (src/ParameterSchema.h).

The code reads parameters by name (param.get<T>("name"), SIMCODE::get_parameter<T>(param, "name"),
r.get("name", ...), r.get_choice("name", ...)). A misspelled name is only caught at run time when that code path
is reached (and an optional parameter could silently take its default), so this checks all of them.
Also reports parameters in the table that the code never reads.

Usage: python3 check_parameter_names.py [path to SimulationCode]   (exit code 1 on unknown names)
"""
import glob
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = os.path.join(root, "src")

schema = open(os.path.join(src, "ParameterSchema.h")).read()
table = set(re.findall(r'(?:b\.optional|b\.required|always_optional)\(\s*"([A-Za-z0-9_]+)"', schema))

# Names read by the code (also through ParameterMap::contains and the conditions of the table itself)
read_patterns = [
    r'\.get<(?:[^<>()]|<[^<>]*>)*>\(\s*"([A-Za-z0-9_]+)"',
    r'get_(?:required_)?parameter<(?:[^<>()]|<[^<>]*>)*>\(\s*param\s*,\s*"([A-Za-z0-9_]+)"',
    r'\br\.get(?:_optional|_choice)?\(\s*"([A-Za-z0-9_]+)"',
    r'\.contains\(\s*"([A-Za-z0-9_]+)"',
    r'\b(?:is_true|is_one_of|if_true)\(\s*"([A-Za-z0-9_]+)"',
    r'spec\.name == "([A-Za-z0-9_]+)"',
]
used = {}
files = glob.glob(os.path.join(src, "*.h")) + glob.glob(os.path.join(src, "*", "*.h")) + glob.glob(
    os.path.join(src, "*.cpp"))
for f in files:
    text = open(f).read()
    for pat in read_patterns:
        for m in re.finditer(pat, text):
            used.setdefault(m.group(1), set()).add(os.path.relpath(f, root))

unknown = sorted(set(used) - table)
unread = sorted(table - set(used))
print(f"{len(table)} parameters in the table, {len(used)} names read by the code")
for name in unknown:
    print(f"ERROR: [{name}] is read in {', '.join(sorted(used[name]))} but is not in the parameter table")
for name in unread:
    print(f"Note: [{name}] is in the parameter table but never read by name in src/")
if unknown:
    sys.exit(1)
print("OK: every parameter read by the code is in the parameter table")
