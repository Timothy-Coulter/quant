#!/usr/bin/env python3
"""Run formatting, linting, type-checking, and tests in a single pass."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence

COMMAND_GROUPS: list[tuple[str, list[str]]] = [
    ("ruff format", ["uv", "run", "ruff", "format", "."]),
    ("black", ["uv", "run", "black", "."]),
    ("isort", ["uv", "run", "isort", "."]),
    ("ruff check --fix", ["uv", "run", "ruff", "check", "--fix", "."]),
    ("mypy", ["uv", "run", "mypy", "."]),
]


def run_command(label: str, command: Sequence[str]) -> None:
    """Execute a command and stream output to the console."""
    print(f"[run_ci_checks] {label}: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess failure path
        print(f"[run_ci_checks] {label} failed with exit code {exc.returncode}", file=sys.stderr)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Run formatters and static analysis only.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to pytest.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    """Entry-point for the helper script."""
    args = parse_args()

    for label, command in COMMAND_GROUPS:
        try:
            run_command(label, command)
        except subprocess.CalledProcessError as exc:
            return exc.returncode

    if not args.skip_tests:
        pytest_cmd = ["uv", "run", "pytest"]
        if args.pytest_args:
            pytest_cmd.extend(args.pytest_args)
        try:
            run_command("pytest", pytest_cmd)
        except subprocess.CalledProcessError as exc:
            return exc.returncode

    print("[run_ci_checks] All checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
