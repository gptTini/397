from __future__ import annotations

import argparse
import fnmatch
import subprocess


OWNERSHIP = {
    "s1/": (
        "src/cspm397/adapters/**",
        "src/cspm397/trace/**",
        "tests/unit/trace/**",
        "tests/integration/trace/**",
    ),
    "s2/": (
        "src/cspm397/artifacts/**",
        "src/cspm397/profiling/**",
        "tests/unit/artifacts/**",
        "tests/integration/artifacts/**",
    ),
    "s3/": (
        "src/cspm397/predictability/**",
        "tests/unit/predictability/**",
        "tests/integration/predictability/**",
    ),
    "s4/": (
        "src/cspm397/trajectory/**",
        "tests/unit/trajectory/**",
        "tests/integration/trajectory/**",
    ),
    "s5/": (
        "src/cspm397/experiment/**",
        "configs/**",
        "tests/integration/experiment/**",
    ),
    "s6/": (
        "tests/contract/**",
        "tests/adversarial/**",
        "tests/regression/**",
        "scripts/check_scope.py",
        "scripts/validate_artifact.py",
        ".github/workflows/**",
    ),
    "s7/": (
        "review/science/**",
        "tests/science/**",
        "decision_policy.json",
    ),
}


def changed_files(base: str, head: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()]


def allowed(branch: str, path: str) -> bool:
    for prefix, patterns in OWNERSHIP.items():
        if branch.startswith(prefix):
            return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
    if branch.startswith("cspm/") or branch.startswith("s0/"):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    changed = changed_files(args.base, args.head)
    violations = [path for path in changed if not allowed(args.branch, path)]
    print("CHANGED=" + ",".join(changed))
    if violations:
        print("SCOPE_VIOLATIONS=" + ",".join(violations))
        return 1
    print("SCOPE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
