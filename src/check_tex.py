"""Sanity-check the braces and environments of a LaTeX file.

Usage: python -m src.check_tex path/to/section.tex
"""
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("usage: python -m src.check_tex path/to/file.tex")

text = Path(sys.argv[1]).read_text(encoding="utf-8")
opens = text.count("{")
closes = text.count("}")
print("open=%d close=%d diff=%d" % (opens, closes, opens - closes))
for env in ["tabular", "table", "figure", "align", "equation"]:
    b = text.count("begin{" + env + "}")
    e = text.count("end{" + env + "}")
    print("  %s: begin=%d end=%d ok=%s" % (env, b, e, b == e))
