from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    repo = Path.cwd()
    files: list[Path] = []
    for raw in args.roots:
        root = repo / raw
        if root.is_file() and root.name.startswith("test_") and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("test_*.py"))
    unique = sorted({path.resolve() for path in files})
    if not unique:
        print("NO_TEST_FILES")
        return 0
    print(f"TEST_FILES={len(unique)}")
    failures: list[str] = []
    for path in unique:
        relative = path.relative_to(repo.resolve())
        print(f"RUN={relative}")
        completed = subprocess.run([sys.executable, str(path)], cwd=repo, check=False)
        if completed.returncode != 0:
            failures.append(relative.as_posix())
    if failures:
        print("FAILED=" + ",".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
