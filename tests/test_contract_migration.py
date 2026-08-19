from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cspm.bootstrap import (
    validate_run_trace_composition,
    validate_trace_manifest_semantics,
    validate_trace_payload_summary,
)


def valid_trace_manifest() -> dict:
    return {
        "schema_version": "trace-v1",
        "run_id": "fixture",
        "model_id": "mock",
        "model_revision": "fixture-v1",
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "seed": 397,
        "projection": {
            "algorithm": "fixed_seeded_random_projection",
            "seed": 397,
            "input_dim": 16,
            "output_dim": 8,
        },
        "dtype": "float32",
        "signature_dim": 8,
        "num_sequences": 1,
        "num_tokens": 4,
        "num_layers": 2,
        "selected_layers": [0, 1],
        "has_router_trace": True,
        "router": {"num_experts": 8, "top_k": 2, "weight_sum_tolerance": 1e-6},
        "created_at": "2026-08-20T00:00:00Z",
        "arrays": {
            "token_ids": "[T]",
            "sequence_ids": "[T]",
            "positions": "[T]",
            "state_signatures": "[T,L,D]",
            "expert_ids": "[T,L,K]",
            "expert_weights": "[T,L,K]",
        },
        "array_shapes": {
            "token_ids": [4],
            "sequence_ids": [4],
            "positions": [4],
            "state_signatures": [4, 2, 8],
            "expert_ids": [4, 2, 2],
            "expert_weights": [4, 2, 2],
        },
        "array_dtypes": {
            "token_ids": "int64",
            "sequence_ids": "int64",
            "positions": "int64",
            "state_signatures": "float32",
            "expert_ids": "int64",
            "expert_weights": "float32",
        },
        "sequence_stats": {
            "unique_count": 1,
            "min_id": 0,
            "max_id": 0,
            "ids_contiguous_zero_based": True,
            "positions_zero_based_contiguous": True,
            "sequences_interleaved": False,
        },
        "chunks": [
            {
                "index": 0,
                "path": "trace/chunk_00000.safetensors",
                "sha256": "c" * 64,
                "tokens": 4,
            }
        ],
    }


def valid_root_manifest(trace_sha: str = "d" * 64) -> dict:
    return {
        "schema_version": "1",
        "run_id": "fixture",
        "experiment_id": "EXP001",
        "code_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "dataset_sha256": "e" * 64,
        "seed": 397,
        "started_at": "2026-08-20T00:00:00Z",
        "completed_at": "2026-08-20T00:01:00Z",
        "artifacts": [{"path": "trace/manifest.json", "sha256": trace_sha}],
        "metrics": {},
    }


class TraceSemanticContractTests(unittest.TestCase):
    def test_valid_manifest_has_no_cross_field_errors(self) -> None:
        self.assertEqual(validate_trace_manifest_semantics(valid_trace_manifest()), [])

    def test_rejects_unknown_git_sha(self) -> None:
        value = valid_trace_manifest()
        value["git_sha"] = "UNKNOWN"
        self.assertIn("git_sha_must_be_full_40_hex", validate_trace_manifest_semantics(value))

    def test_rejects_projection_signature_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["projection"]["output_dim"] = 7
        self.assertIn("projection_output_dim_must_equal_signature_dim", validate_trace_manifest_semantics(value))

    def test_rejects_nonzero_tokens_with_no_sequences(self) -> None:
        value = valid_trace_manifest()
        value["num_sequences"] = 0
        value["sequence_stats"].update({"unique_count": 0, "min_id": None, "max_id": None})
        self.assertIn("nonempty_trace_requires_at_least_one_sequence", validate_trace_manifest_semantics(value))

    def test_rejects_state_axis_cardinality_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["selected_layers"] = [0]
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("state_signatures_shape_mismatch", errors)
        self.assertIn("expert_ids_shape_mismatch", errors)
        self.assertIn("expert_weights_shape_mismatch", errors)

    def test_rejects_router_k_axis_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["router"]["top_k"] = 3
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("expert_ids_shape_mismatch", errors)
        self.assertIn("expert_weights_shape_mismatch", errors)

    def test_rejects_array_descriptor_inventory_asymmetry(self) -> None:
        value = valid_trace_manifest()
        value["array_shapes"]["router_entropy"] = [4, 2]
        self.assertIn("array_descriptor_name_sets_must_match_exactly", validate_trace_manifest_semantics(value))

    def test_rejects_state_signature_dtype_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["array_dtypes"]["state_signatures"] = "float16"
        self.assertIn("state_signature_dtype_must_match_array_dtypes", validate_trace_manifest_semantics(value))

    def test_rejects_wrong_integer_dtype(self) -> None:
        value = valid_trace_manifest()
        value["array_dtypes"]["token_ids"] = "float32"
        self.assertIn("token_ids_dtype_must_be_integer", validate_trace_manifest_semantics(value))

    def test_rejects_nonzero_tokens_with_no_chunks(self) -> None:
        value = valid_trace_manifest()
        value["chunks"] = []
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("nonempty_trace_requires_chunks", errors)
        self.assertIn("chunk_token_sum_must_equal_num_tokens", errors)

    def test_payload_summary_recomputes_names_shapes_dtypes_and_stats(self) -> None:
        value = valid_trace_manifest()
        good_shapes = dict(value["array_shapes"])
        good_dtypes = dict(value["array_dtypes"])
        good_stats = dict(value["sequence_stats"])
        self.assertEqual(
            validate_trace_payload_summary(
                value,
                observed_array_shapes=good_shapes,
                observed_array_dtypes=good_dtypes,
                observed_sequence_stats=good_stats,
            ),
            [],
        )
        bad_dtypes = dict(good_dtypes)
        bad_dtypes["state_signatures"] = "float16"
        self.assertIn(
            "payload_array_dtypes_do_not_match_manifest",
            validate_trace_payload_summary(
                value,
                observed_array_shapes=good_shapes,
                observed_array_dtypes=bad_dtypes,
                observed_sequence_stats=good_stats,
            ),
        )
        missing = dict(good_shapes)
        missing.pop("expert_weights")
        self.assertIn(
            "payload_tensor_names_do_not_match_manifest",
            validate_trace_payload_summary(
                value,
                observed_array_shapes=missing,
                observed_array_dtypes=good_dtypes,
                observed_sequence_stats=good_stats,
            ),
        )


class CompositionContractTests(unittest.TestCase):
    def test_matching_root_and_trace_provenance_passes(self) -> None:
        self.assertEqual(
            validate_run_trace_composition(
                valid_root_manifest(), valid_trace_manifest(), trace_manifest_sha256="d" * 64
            ),
            [],
        )

    def test_rejects_all_duplicated_identity_mismatches(self) -> None:
        root = valid_root_manifest()
        trace = valid_trace_manifest()
        root["run_id"] = "other"
        root["code_sha"] = "1" * 40
        root["config_sha256"] = "2" * 64
        root["seed"] = 999
        errors = validate_run_trace_composition(root, trace, trace_manifest_sha256="d" * 64)
        self.assertIn("root_trace_run_id_mismatch", errors)
        self.assertIn("root_trace_code_sha_git_sha_mismatch", errors)
        self.assertIn("root_trace_config_sha256_config_hash_mismatch", errors)
        self.assertIn("root_trace_seed_mismatch", errors)

    def test_rejects_missing_duplicate_or_wrong_child_hash_reference(self) -> None:
        root = valid_root_manifest()
        trace = valid_trace_manifest()
        root["artifacts"] = []
        self.assertIn(
            "root_must_reference_exactly_one_trace_manifest",
            validate_run_trace_composition(root, trace, trace_manifest_sha256="d" * 64),
        )
        root = valid_root_manifest("f" * 64)
        self.assertIn(
            "root_trace_manifest_hash_mismatch",
            validate_run_trace_composition(root, trace, trace_manifest_sha256="d" * 64),
        )


class FrozenContractFileTests(unittest.TestCase):
    def test_trace_schema_requires_dtype_inventory(self) -> None:
        schema = json.loads((REPO_ROOT / "schemas/trace-v1.schema.json").read_text("utf-8"))
        required = set(schema["required"])
        self.assertIn("array_dtypes", required)
        self.assertIn("int_dtype", schema["$defs"])
        self.assertIn("float_dtype", schema["$defs"])

    def test_manifest_authority_freezes_cross_manifest_identity(self) -> None:
        authority = (REPO_ROOT / "docs/contracts/MANIFEST_AUTHORITY.md").read_text("utf-8")
        for text in ("root.run_id == trace.run_id", "root.code_sha == trace.git_sha", "root.config_sha256 == trace.config_hash", "root.seed == trace.seed"):
            self.assertIn(text, authority)

    def test_invalid_fixture_specs_include_round3_cases(self) -> None:
        spec = json.loads((REPO_ROOT / "fixtures/spec/invalid_trace_v1_cases.json").read_text("utf-8"))
        ids = {case["id"] for case in spec["cases"]}
        for required in {
            "cross_manifest_provenance_mismatch",
            "payload_dtype_mismatch",
            "tensor_inventory_mismatch",
            "router_entropy_descriptor_asymmetry",
        }:
            self.assertIn(required, ids)

    def test_golden_fixture_has_exact_dtype_inventory(self) -> None:
        spec = json.loads((REPO_ROOT / "fixtures/spec/tiny_trace_v1.json").read_text("utf-8"))
        req = spec["manifest_requirements"]
        self.assertEqual(set(req["arrays"]), set(req["array_shapes"]))
        self.assertEqual(set(req["arrays"]), set(req["array_dtypes"]))
        self.assertEqual(req["dtype"], req["array_dtypes"]["state_signatures"])

    def test_sot_assigns_all_legacy_g0_paths_and_points_to_authoritative_validator(self) -> None:
        sot = (REPO_ROOT / "CSPM_SOT.yaml").read_text("utf-8")
        self.assertIn('schema_version: "2.3"', sot)
        for path in (
            "src/cspm/bootstrap.py",
            "src/cspm/contracts_reference.py",
            "experiments/exp000_bootstrap/**",
            "tests/test_bootstrap.py",
            "tests/test_contract_migration.py",
            "docs/ERROR_TAXONOMY.md",
        ):
            self.assertIn(path, sot)
        self.assertIn("validate_run_trace_composition", sot)
        self.assertIn("array_dtypes", sot)


if __name__ == "__main__":
    unittest.main()
