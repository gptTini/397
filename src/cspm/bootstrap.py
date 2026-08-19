from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_canonical_json(path: str | Path, value: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value) + b"\n"
    p.write_bytes(data)
    return sha256_bytes(data)


def build_manifest(
    *,
    run_id: str,
    experiment_id: str,
    code_sha: str,
    config: dict[str, Any],
    dataset_identity: dict[str, Any],
    seed: int,
    started_at: str,
    completed_at: str,
    artifacts: list[dict[str, str]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "code_sha": code_sha,
        "config_sha256": sha256_json(config),
        "dataset_sha256": sha256_json(dataset_identity),
        "seed": seed,
        "started_at": started_at,
        "completed_at": completed_at,
        "artifacts": artifacts,
        "metrics": metrics,
    }


def validate_manifest_shape(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "run_id",
        "experiment_id",
        "code_sha",
        "config_sha256",
        "dataset_sha256",
        "seed",
        "started_at",
        "completed_at",
        "artifacts",
        "metrics",
    }
    missing = sorted(required - set(manifest))
    if missing:
        errors.append(f"missing_fields={missing}")
    if manifest.get("schema_version") != "1":
        errors.append("schema_version_must_equal_1")
    for field in ("config_sha256", "dataset_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{field}_must_be_64_hex_chars")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts_must_be_list")
    else:
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                errors.append(f"artifact_{index}_must_be_object")
                continue
            if set(artifact) != {"path", "sha256"}:
                errors.append(f"artifact_{index}_fields_invalid")
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"artifact_{index}_sha256_invalid")
    return errors
