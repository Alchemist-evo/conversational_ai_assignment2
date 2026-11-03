from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parent.parent


def md(s: str):
    return nbformat.v4.new_markdown_cell(dedent(s).strip("\n"))


def code(s: str):
    return nbformat.v4.new_code_cell(dedent(s).strip("\n"))


def module(name: str):
    src = (ROOT / "ra" / name).read_text(encoding="utf-8")
    return nbformat.v4.new_code_cell(f"%%writefile ra/{name}\n{src}")
