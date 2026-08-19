from __future__ import annotations

import re
from typing import Any

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT40 = re.compile(r"^[0-9a-f]{40}$")
_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}
_INT_DTYPES = {"int32", "int64"}
_REQUIRED_ARRAYS = {"token_ids", "sequence_ids", "positions", "state_signatures"}
_ROUTER_REQUIRED_ARRAYS = {"expert_ids", "expert_weights"}
_ROUTER_OPTIONAL_ARRAYS = {"router_entropy"}


def _shape_matches(value: Any, expected: list[int]) -> bool:
    return isinstance(value, list) and value == expected and all(
        isinstance(x, int) and x >= 0 for x in value
    )


def _descriptor_names(manifest: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    arrays = manifest.get("arrays")
    shapes = manifest.get("array_shapes")
    dtypes = manifest.get("array_dtypes")
    return (
        set(arrays) if isinstance(arrays, dict) else set(),
        set(shapes) if isinstance(shapes, dict) else set(),
        set(dtypes) if isinstance(dtypes, dict) else set(),
    )


def validate_trace_manifest_semantics(manifest: dict[str, Any]) -> list[str]:
    """Authoritative TraceArtifact-v1 cross-field validation for G0/Wave-1.

    JSON Schema validation is required separately. This validator freezes relations
    that JSON Schema cannot conveniently express, including exact tensor inventory,
    axis cardinality, dtype semantics, provenance, router descriptors, and chunks.
    """
    errors: list[str] = []

    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or _GIT40.fullmatch(git_sha) is None:
        errors.append("git_sha_must_be_full_40_hex")

    signature_dim = manifest.get("signature_dim")
    projection = manifest.get("projection")
    if not isinstance(projection, dict):
        errors.append("projection_must_be_object")
    elif projection.get("output_dim") != signature_dim:
        errors.append("projection_output_dim_must_equal_signature_dim")

    selected_layers = manifest.get("selected_layers")
    num_layers = manifest.get("num_layers")
    if not isinstance(selected_layers, list) or not selected_layers:
        errors.append("selected_layers_must_be_nonempty")
        layer_count: int | None = None
    else:
        layer_count = len(selected_layers)
        if len(set(selected_layers)) != len(selected_layers):
            errors.append("selected_layers_must_be_unique")
        if isinstance(num_layers, int) and num_layers > 0 and any(
            not isinstance(x, int) or x < 0 or x >= num_layers for x in selected_layers
        ):
            errors.append("selected_layers_out_of_range")

    num_tokens = manifest.get("num_tokens")
    num_sequences = manifest.get("num_sequences")
    if isinstance(num_tokens, int) and isinstance(num_sequences, int):
        if num_tokens == 0 and num_sequences != 0:
            errors.append("zero_token_trace_requires_zero_sequences")
        if num_tokens > 0 and num_sequences < 1:
            errors.append("nonempty_trace_requires_at_least_one_sequence")
        if num_sequences > num_tokens:
            errors.append("num_sequences_must_not_exceed_num_tokens")

    arrays = manifest.get("arrays")
    array_shapes = manifest.get("array_shapes")
    array_dtypes = manifest.get("array_dtypes")
    if not isinstance(arrays, dict):
        errors.append("arrays_must_be_object")
        arrays = {}
    if not isinstance(array_shapes, dict):
        errors.append("array_shapes_must_be_object")
        array_shapes = {}
    if not isinstance(array_dtypes, dict):
        errors.append("array_dtypes_must_be_object")
        array_dtypes = {}

    array_names, shape_names, dtype_names = _descriptor_names(manifest)
    if array_names != shape_names or array_names != dtype_names:
        errors.append("array_descriptor_name_sets_must_match_exactly")
    if not _REQUIRED_ARRAYS.issubset(array_names):
        errors.append("required_array_descriptors_missing")

    state_dtype = manifest.get("dtype")
    if state_dtype not in _FLOAT_DTYPES:
        errors.append("state_signature_dtype_invalid")
    if array_dtypes.get("state_signatures") != state_dtype:
        errors.append("state_signature_dtype_must_match_array_dtypes")
    for name in ("token_ids", "sequence_ids", "positions"):
        if array_dtypes.get(name) not in _INT_DTYPES:
            errors.append(f"{name}_dtype_must_be_integer")

    if isinstance(num_tokens, int):
        for name in ("token_ids", "sequence_ids", "positions"):
            if not _shape_matches(array_shapes.get(name), [num_tokens]):
                errors.append(f"{name}_shape_mismatch")
        if isinstance(layer_count, int) and isinstance(signature_dim, int):
            if not _shape_matches(
                array_shapes.get("state_signatures"),
                [num_tokens, layer_count, signature_dim],
            ):
                errors.append("state_signatures_shape_mismatch")

    sequence_stats = manifest.get("sequence_stats")
    if not isinstance(sequence_stats, dict):
        errors.append("sequence_stats_must_be_object")
    elif isinstance(num_sequences, int):
        if sequence_stats.get("unique_count") != num_sequences:
            errors.append("sequence_unique_count_must_equal_num_sequences")
        if num_sequences == 0:
            if sequence_stats.get("min_id") is not None or sequence_stats.get("max_id") is not None:
                errors.append("empty_sequence_ids_require_null_min_max")
        else:
            if sequence_stats.get("min_id") != 0:
                errors.append("sequence_min_id_must_equal_zero")
            if sequence_stats.get("max_id") != num_sequences - 1:
                errors.append("sequence_max_id_must_equal_num_sequences_minus_one")
        if sequence_stats.get("ids_contiguous_zero_based") is not True:
            errors.append("sequence_ids_must_be_contiguous_zero_based")
        if sequence_stats.get("positions_zero_based_contiguous") is not True:
            errors.append("positions_must_be_zero_based_contiguous")
        if sequence_stats.get("sequences_interleaved") is not False:
            errors.append("sequences_must_not_interleave")

    has_router = manifest.get("has_router_trace")
    router = manifest.get("router")
    router_names = _ROUTER_REQUIRED_ARRAYS | _ROUTER_OPTIONAL_ARRAYS
    if has_router is True:
        if not _ROUTER_REQUIRED_ARRAYS.issubset(array_names):
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
            if array_dtypes.get("expert_ids") not in _INT_DTYPES:
                errors.append("expert_ids_dtype_must_be_integer")
            if array_dtypes.get("expert_weights") not in _FLOAT_DTYPES:
                errors.append("expert_weights_dtype_must_be_float")
            if "router_entropy" in array_names and array_dtypes.get("router_entropy") not in _FLOAT_DTYPES:
                errors.append("router_entropy_dtype_must_be_float")
            if (
                isinstance(num_tokens, int)
                and isinstance(layer_count, int)
                and isinstance(top_k, int)
                and top_k >= 1
            ):
                expected = [num_tokens, layer_count, top_k]
                if not _shape_matches(array_shapes.get("expert_ids"), expected):
                    errors.append("expert_ids_shape_mismatch")
                if not _shape_matches(array_shapes.get("expert_weights"), expected):
                    errors.append("expert_weights_shape_mismatch")
                if "router_entropy" in array_names and not _shape_matches(
                    array_shapes.get("router_entropy"), [num_tokens, layer_count]
                ):
                    errors.append("router_entropy_shape_mismatch")
    elif has_router is False:
        if router is not None:
            errors.append("router_metadata_forbidden_without_router_trace")
        if array_names & router_names:
            errors.append("router_arrays_forbidden_without_router_trace")

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
        if path != f"trace/chunk_{expected_index:05d}.safetensors":
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

    return errors


def validate_trace_payload_summary(
    manifest: dict[str, Any],
    *,
    observed_array_shapes: dict[str, list[int]],
    observed_array_dtypes: dict[str, str],
    observed_sequence_stats: dict[str, Any],
) -> list[str]:
    """Compare independently recomputed payload inventory against the manifest."""
    errors: list[str] = []
    declared_arrays = manifest.get("arrays")
    declared_names = set(declared_arrays) if isinstance(declared_arrays, dict) else set()
    if set(observed_array_shapes) != declared_names:
        errors.append("payload_tensor_names_do_not_match_manifest")
    if set(observed_array_dtypes) != declared_names:
        errors.append("payload_tensor_names_do_not_match_manifest")
    if observed_array_shapes != manifest.get("array_shapes"):
        errors.append("payload_array_shapes_do_not_match_manifest")
    if observed_array_dtypes != manifest.get("array_dtypes"):
        errors.append("payload_array_dtypes_do_not_match_manifest")
    if observed_sequence_stats != manifest.get("sequence_stats"):
        errors.append("payload_sequence_stats_do_not_match_manifest")
    return list(dict.fromkeys(errors))


def validate_run_trace_composition(
    root_manifest: dict[str, Any],
    trace_manifest: dict[str, Any],
    *,
    trace_manifest_sha256: str | None = None,
) -> list[str]:
    """Bind duplicated provenance across root RunManifest and TraceArtifact child."""
    errors: list[str] = []
    equality = (
        ("run_id", "run_id", "run_id"),
        ("code_sha", "git_sha", "code_sha_git_sha"),
        ("config_sha256", "config_hash", "config_sha256_config_hash"),
        ("seed", "seed", "seed"),
    )
    for root_key, trace_key, label in equality:
        if root_manifest.get(root_key) != trace_manifest.get(trace_key):
            errors.append(f"root_trace_{label}_mismatch")

    artifacts = root_manifest.get("artifacts")
    refs = []
    if isinstance(artifacts, list):
        refs = [a for a in artifacts if isinstance(a, dict) and a.get("path") == "trace/manifest.json"]
    if len(refs) != 1:
        errors.append("root_must_reference_exactly_one_trace_manifest")
    elif trace_manifest_sha256 is not None:
        if not isinstance(trace_manifest_sha256, str) or _HEX64.fullmatch(trace_manifest_sha256) is None:
            errors.append("trace_manifest_sha256_invalid")
        elif refs[0].get("sha256") != trace_manifest_sha256:
            errors.append("root_trace_manifest_hash_mismatch")
    return errors
