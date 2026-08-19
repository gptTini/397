from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT40 = re.compile(r"^[0-9a-f]{40}$")


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
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
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
            if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
                errors.append(f"artifact_{index}_sha256_invalid")
    return errors


def validate_trace_manifest_semantics(manifest: dict[str, Any]) -> list[str]:
    """Cross-field TraceArtifact v1 rules that JSON Schema cannot fully express.

    JSON-schema validation is still required separately. This function freezes the
    semantic checks Wave-1 producers/consumers must enforce.
    """
    errors: list[str] = []

    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or _GIT40.fullmatch(git_sha) is None:
        errors.append("git_sha_must_be_full_40_hex")

    signature_dim = manifest.get("signature_dim")
    projection = manifest.get("projection")
    if not isinstance(projection, dict):
        errors.append("projection_must_be_object")
    else:
        if projection.get("output_dim") != signature_dim:
            errors.append("projection_output_dim_must_equal_signature_dim")

    num_layers = manifest.get("num_layers")
    selected_layers = manifest.get("selected_layers")
    if not isinstance(selected_layers, list) or not selected_layers:
        errors.append("selected_layers_must_be_nonempty")
    elif isinstance(num_layers, int) and num_layers > 0:
        if len(selected_layers) != len(set(selected_layers)):
            errors.append("selected_layers_must_be_unique")
        if any(not isinstance(x, int) or x < 0 or x >= num_layers for x in selected_layers):
            errors.append("selected_layers_out_of_range")

    num_tokens = manifest.get("num_tokens")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        errors.append("chunks_must_be_list")
        chunks = []
    if isinstance(num_tokens, int):
        if num_tokens > 0 and not chunks:
            errors.append("nonempty_trace_requires_chunks")
        if num_tokens == 0 and chunks:
            errors.append("zero_token_trace_must_not_have_chunks")

    seen_paths: set[str] = set()
    token_sum = 0
    for expected_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"chunk_{expected_index}_must_be_object")
            continue
        index = chunk.get("index")
        path = chunk.get("path")
        tokens = chunk.get("tokens")
        if index != expected_index:
            errors.append("chunk_indices_must_be_contiguous_from_zero")
        expected_path = f"trace/chunk_{expected_index:05d}.safetensors"
        if path != expected_path:
            errors.append(f"chunk_path_index_mismatch:{expected_index}")
        if isinstance(path, str):
            if path in seen_paths:
                errors.append("duplicate_chunk_path")
            seen_paths.add(path)
        if not isinstance(tokens, int) or tokens <= 0:
            errors.append(f"chunk_tokens_must_be_positive:{expected_index}")
        else:
            token_sum += tokens
    if isinstance(num_tokens, int) and token_sum != num_tokens:
        errors.append("chunk_token_sum_must_equal_num_tokens")

    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        arrays = {}
    has_router = manifest.get("has_router_trace")
    router = manifest.get("router")
    if has_router is True:
        if "expert_ids" not in arrays or "expert_weights" not in arrays:
            errors.append("router_trace_requires_expert_arrays")
        if not isinstance(router, dict):
            errors.append("router_trace_requires_router_metadata")
        else:
            num_experts = router.get("num_experts")
            top_k = router.get("top_k")
            tolerance = router.get("weight_sum_tolerance")
            if not isinstance(num_experts, int) or num_experts < 1:
                errors.append("router_num_experts_invalid")
            if not isinstance(top_k, int) or top_k < 1:
                errors.append("router_top_k_invalid")
            elif isinstance(num_experts, int) and top_k > num_experts:
                errors.append("router_top_k_exceeds_num_experts")
            if not isinstance(tolerance, (int, float)) or tolerance <= 0 or tolerance > 0.1:
                errors.append("router_weight_sum_tolerance_invalid")
    elif has_router is False:
        if router is not None:
            errors.append("router_metadata_forbidden_without_router_trace")
        if "expert_ids" in arrays or "expert_weights" in arrays or "router_entropy" in arrays:
            errors.append("router_arrays_forbidden_without_router_trace")

    return errors
