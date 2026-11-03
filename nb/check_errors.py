"""List every cell that raised an error in an executed notebook: python nb/check_errors.py path.ipynb"""
import re
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
n_err = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            n_err += 1
            first = c.source.splitlines()[0][:90] if c.source else ""
            tb = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(o.get("traceback", [])))
            lines = [l for l in tb.splitlines() if l.strip()]
            print(f"--- cell {i}: {first}\n    {o['ename']}: {o['evalue'][:300]}")
            for l in lines[-8:-1]:
                print("      ", l[:160])
print(f"\n{n_err} error cell(s)")
