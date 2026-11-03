"""Assemble the assignment notebook from nb/s*.py section files (single source of truth for code = ra/*.py).

Usage:  python build_notebook.py            -> Assignment2_PS1_ResearchAssistant.ipynb
Inference / conclusion text written after execution lives in nb/inferences.md (sections delimited by
'## <!--KEY-->' lines) and replaces the matching <!--KEY--> placeholders.
"""
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "nb"))
import s0_setup, s1_task1, s2_task2, s3_task3, s4_task4, s5_task5, s6_task6, s7_task7, s8_end  # noqa: E402

OUT = ROOT / "Assignment2_PS1_ResearchAssistant.ipynb"


def load_inferences() -> dict:
    p = ROOT / "nb" / "inferences.md"
    if not p.exists():
        return {}
    parts = re.split(r"^## (<!--[A-Z0-9_]+-->)\s*$", p.read_text(encoding="utf-8"), flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def main():
    cells = []
    for mod in (s0_setup, s1_task1, s2_task2, s3_task3, s4_task4, s5_task5, s6_task6, s7_task7, s8_end):
        cells += mod.CELLS
    inf = load_inferences()
    for c in cells:
        if c.cell_type == "markdown":
            for key in re.findall(r"<!--[A-Z0-9_]+-->", c.source):
                c.source = c.source.replace(key, inf.get(key, "_(Written after executing the notebook.)_"))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, OUT)
    print(f"wrote {OUT.name}: {len(cells)} cells ({sum(c.cell_type=='code' for c in cells)} code)")


if __name__ == "__main__":
    main()
