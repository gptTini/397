from __future__ import annotations

import ast
from pathlib import Path


RULES = {
    "src/cspm397/adapters": ("cspm397.predictability", "cspm397.trajectory"),
    "src/cspm397/trace": ("cspm397.predictability", "cspm397.trajectory"),
    "src/cspm397/predictability": ("transformers", "cspm397.adapters"),
    "src/cspm397/trajectory": ("transformers", "cspm397.adapters"),
    "src/cspm397/experiment": (
        "transformers",
        "cspm397.adapters.huggingface",
    ),
}


def imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def main() -> int:
    repo = Path.cwd()
    errors: list[str] = []
    for root_text, forbidden in RULES.items():
        root = repo / root_text
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            for imported in imports(path):
                for prefix in forbidden:
                    if imported == prefix or imported.startswith(prefix + "."):
                        errors.append(f"{path.relative_to(repo)} imports forbidden {imported}")
    if errors:
        print("\n".join(errors))
        return 1
    print("IMPORT_BOUNDARIES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
