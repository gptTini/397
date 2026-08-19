from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from cspm.contracts_reference import (
    validate_run_trace_composition,
    validate_trace_manifest_semantics,
    validate_trace_payload_summary,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT40 = re.compile(r"^[0-9a-f]{40}$")
_EXPERIMENT_ID = re.compile(r"^EXP[0-9]{3}$")
_PORTABLE_PATH_SEGMENT = re.compile(r"^[a-z0-9_-](?:[a-z0-9._-]*[a-z0-9_-])?$")
_WINDOWS_RESERVED_STEM = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$")
_ROOT_FIELDS = {
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
_ARTIFACT_FIELDS = {"path", "sha256"}


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical on-disk JSON representation used for both writes and hashes."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
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
    data = canonical_json_bytes(value)
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


def _is_safe_run_relative_path(value: str) -> bool:
    """Accept only the canonical cross-platform artifact path subset."""
    if not value:
        return False
    segments = value.split("/")
    if any(_PORTABLE_PATH_SEGMENT.fullmatch(segment) is None for segment in segments):
        return False
    for segment in segments:
        stem = segment.split(".", 1)[0]
        if _WINDOWS_RESERVED_STEM.fullmatch(stem) is not None:
            return False
    return True


def _is_json_integer(value: Any) -> bool:
    """Match JSON Schema integer semantics, including 397.0 but excluding bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def validate_run_manifest(manifest: Any) -> list[str]:
    """Mandatory RunManifest v1 reference validation pipeline.

    This function mirrors every structural constraint in
    contracts/run_manifest_v1.schema.json and then applies semantic constraints
    that JSON Schema cannot express cleanly, currently duplicate artifact paths.
    A caller must not treat JSON-Schema-only validation or a partial helper as a
    substitute for this pipeline.
    """
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["root_manifest_must_be_object"]

    keys = set(manifest)
    missing = sorted(_ROOT_FIELDS - keys)
    unknown = sorted(keys - _ROOT_FIELDS)
    if missing:
        errors.append(f"missing_fields={missing}")
    if unknown:
        errors.append(f"unknown_fields={unknown}")

    if manifest.get("schema_version") != "1":
        errors.append("schema_version_must_equal_1")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        errors.append("run_id_must_be_nonempty_string")

    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        errors.append("experiment_id_must_match_EXPddd")

    code_sha = manifest.get("code_sha")
    if not isinstance(code_sha, str) or _GIT40.fullmatch(code_sha) is None:
        errors.append("code_sha_must_be_full_40_hex")

    for field in ("config_sha256", "dataset_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            errors.append(f"{field}_must_be_64_hex_chars")

    if not _is_json_integer(manifest.get("seed")):
        errors.append("seed_must_be_integer")

    for field in ("started_at", "completed_at"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{field}_must_be_nonempty_string")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts_must_be_nonempty_list")
    else:
        seen_paths: set[str] = set()
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                errors.append(f"artifact_{index}_must_be_object")
                continue
            if set(artifact) != _ARTIFACT_FIELDS:
                errors.append(f"artifact_{index}_fields_invalid")

            path = artifact.get("path")
            if not isinstance(path, str) or not _is_safe_run_relative_path(path):
                errors.append(f"artifact_{index}_path_invalid")
            elif path in seen_paths:
                errors.append("duplicate_artifact_path")
            else:
                seen_paths.add(path)

            digest = artifact.get("sha256")
            if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
                errors.append(f"artifact_{index}_sha256_invalid")

    if not isinstance(manifest.get("metrics"), dict):
        errors.append("metrics_must_be_object")

    return errors


def validate_manifest_shape(manifest: Any) -> list[str]:
    """Backward-compatible alias for the full mandatory RunManifest pipeline."""
    return validate_run_manifest(manifest)


__all__ = [
    "build_manifest",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_manifest_shape",
    "validate_run_manifest",
    "validate_run_trace_composition",
    "validate_trace_manifest_semantics",
    "validate_trace_payload_summary",
    "write_canonical_json",
]
