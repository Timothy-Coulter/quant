#!/usr/bin/env python3
"""Lightweight Markdown sanity checker for docs/ and README."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((?!http)(?!#)([^)]+)\)")


def iter_markdown_files() -> Iterable[Path]:
    """Yield Markdown files that we lint."""
    for candidate in [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]:
        if candidate.exists():
            yield candidate
    docs_dir = ROOT / "docs"
    if docs_dir.exists():
        yield from docs_dir.glob("*.md")


def lint_file(path: Path) -> list[str]:
    """Return a list of lint errors for a Markdown file."""
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    first_content = next((line for line in lines if line.strip()), "")
    if not first_content.startswith("#"):
        errors.append("missing top-level heading")

    for idx, line in enumerate(lines, start=1):
        if line.rstrip() != line:
            errors.append(f"line {idx}: trailing whitespace")

        for match in MARKDOWN_LINK_RE.finditer(line):
            target = match.group(1).split("#")[0]
            target_path = (path.parent / target).resolve()
            if not target_path.exists():
                errors.append(f"line {idx}: broken relative link -> {target}")
    return errors


def main() -> int:
    """Run the lint checks."""
    overall_errors = 0
    for md_file in iter_markdown_files():
        errors = lint_file(md_file)
        if errors:
            overall_errors += len(errors)
            for error in errors:
                print(f"[check_docs] {md_file.relative_to(ROOT)}: {error}")

    if overall_errors:
        print(f"[check_docs] Found {overall_errors} documentation issue(s).", file=sys.stderr)
        return 1

    print("[check_docs] Documentation looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
