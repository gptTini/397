from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


QUALITY_ROOTS = (
    "src/cspm397",
    "tests/unit",
    "tests/integration",
    "tests/contract",
    "tests/adversarial",
    "tests/regression",
)


def collect(repo: Path, *, source_only: bool = False) -> list[str]:
    roots = ("src/cspm397",) if source_only else QUALITY_ROOTS
    files: list[str] = []
    for text in roots:
        root = repo / text
        if root.is_dir():
            files.extend(str(path) for path in sorted(root.rglob("*.py")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("format", "lint", "type"))
    args = parser.parse_args()
    repo = Path.cwd()
    files = collect(repo, source_only=args.mode == "type")
    if not files:
        print("QUALITY_FILES=0")
        return 0
    if args.mode == "format":
        command = ["ruff", "format", "--check", *files]
    elif args.mode == "lint":
        command = ["ruff", "check", *files]
    else:
        command = ["mypy", "--ignore-missing-imports", "--check-untyped-defs", *files]
    print("QUALITY_FILES=" + str(len(files)))
    return subprocess.run(command, cwd=repo, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
