#!/usr/bin/env python3
"""
Refresh embedded source snippets inside notebooks/main.ipynb.

It updates the Markdown <details> blocks with code fences so they reflect
the current contents of files in src/hcr/.

Usage: python3 scripts/refresh_snippets.py
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("notebooks/main.ipynb")

FILES = {
    "src-config-py": Path("src/hcr/config.py"),
    "src-ingest-py": Path("src/hcr/ingest.py"),
    "src-preprocess-py": Path("src/hcr/preprocess.py"),
    "src-features-py": Path("src/hcr/features.py"),
    "src-labels-py": Path("src/hcr/labels.py"),
    "src-models-py": Path("src/hcr/models.py"),
    "src-train-py": Path("src/hcr/train.py"),
    "src-eval-py": Path("src/hcr/eval.py"),
}


def build_details_block(path: Path, code: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"<details><summary><code>{path.as_posix()}</code></summary>\n")
    lines.append("\n")
    lines.append("```python\n")
    for l in code.splitlines():
        lines.append(l + ("\n" if not l.endswith("\n") else ""))
    lines.append("```\n")
    lines.append("</details>\n")
    return lines


def main() -> None:
    nb = json.loads(NB.read_text())
    cells = nb.get("cells", [])
    # Build new blocks
    blocks = {cid: build_details_block(p, p.read_text()) for cid, p in FILES.items() if p.exists()}
    # Replace cells whose id matches
    for cell in cells:
        cid = cell.get("id")
        if cid in blocks and cell.get("cell_type") == "markdown":
            cell["source"] = blocks[cid]
    NB.write_text(json.dumps(nb, indent=1))
    print("Refreshed embedded snippets in", NB)


if __name__ == "__main__":
    main()
