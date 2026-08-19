from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cspm.bootstrap import validate_trace_manifest_semantics


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
        "chunks": [
            {
                "index": 0,
                "path": "trace/chunk_00000.safetensors",
                "sha256": "c" * 64,
                "tokens": 4,
            }
        ],
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
        self.assertIn(
            "projection_output_dim_must_equal_signature_dim",
            validate_trace_manifest_semantics(value),
        )

    def test_rejects_nonzero_tokens_with_no_chunks(self) -> None:
        value = valid_trace_manifest()
        value["chunks"] = []
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("nonempty_trace_requires_chunks", errors)
        self.assertIn("chunk_token_sum_must_equal_num_tokens", errors)

    def test_rejects_chunk_index_path_and_token_sum_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["chunks"] = [
            {
                "index": 1,
                "path": "trace/chunk_00002.safetensors",
                "sha256": "c" * 64,
                "tokens": 3,
            }
        ]
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("chunk_indices_must_be_contiguous_from_zero", errors)
        self.assertIn("chunk_path_index_mismatch:0", errors)
        self.assertIn("chunk_token_sum_must_equal_num_tokens", errors)

    def test_rejects_router_trace_without_arrays_or_metadata(self) -> None:
        value = valid_trace_manifest()
        value.pop("router")
        value["arrays"].pop("expert_ids")
        value["arrays"].pop("expert_weights")
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("router_trace_requires_expert_arrays", errors)
        self.assertIn("router_trace_requires_router_metadata", errors)

    def test_rejects_router_arrays_when_router_trace_false(self) -> None:
        value = valid_trace_manifest()
        value["has_router_trace"] = False
        value.pop("router")
        self.assertIn(
            "router_arrays_forbidden_without_router_trace",
            validate_trace_manifest_semantics(value),
        )


class FrozenContractFileTests(unittest.TestCase):
    def test_trace_schema_freezes_nested_projection_and_router_conditionals(self) -> None:
        schema = json.loads((REPO_ROOT / "schemas/trace-v1.schema.json").read_text("utf-8"))
        required = set(schema["required"])
        self.assertIn("projection", required)
        self.assertNotIn("projection_algorithm", required)
        self.assertEqual(schema["properties"]["git_sha"]["pattern"], "^[0-9a-f]{40}$")
        self.assertEqual(schema["properties"]["selected_layers"]["minItems"], 1)
        self.assertIn("router", schema["properties"])
        self.assertGreaterEqual(len(schema["allOf"]), 4)

    def test_run_manifest_and_trace_manifest_have_distinct_authority(self) -> None:
        run_schema = json.loads(
            (REPO_ROOT / "contracts/run_manifest_v1.schema.json").read_text("utf-8")
        )
        trace_schema = json.loads(
            (REPO_ROOT / "schemas/trace-v1.schema.json").read_text("utf-8")
        )
        self.assertIn("root run manifest", run_schema["description"].lower())
        self.assertIn("trace/manifest.json", trace_schema["description"])
        authority = (REPO_ROOT / "docs/contracts/MANIFEST_AUTHORITY.md").read_text("utf-8")
        self.assertIn("<run>/manifest.json", authority)
        self.assertIn("<run>/trace/manifest.json", authority)
        self.assertIn("checksum index only", authority)

    def test_invalid_fixture_specs_are_actionable(self) -> None:
        spec = json.loads(
            (REPO_ROOT / "fixtures/spec/invalid_trace_v1_cases.json").read_text("utf-8")
        )
        ids = set()
        for case in spec["cases"]:
            self.assertTrue(case["id"])
            self.assertIn(case["stage"], {"READ", "WRITE", "VALIDATE"})
            self.assertTrue(case["mutation"])
            self.assertRegex(case["expected_error"], r"^E[0-9]{3}$")
            self.assertNotIn(case["id"], ids)
            ids.add(case["id"])
        for required in {
            "missing_manifest",
            "projection_dim_mismatch",
            "unknown_git_sha",
            "router_missing_arrays",
            "chunk_index_path_mismatch",
        }:
            self.assertIn(required, ids)

    def test_golden_fixture_freezes_expert_and_position_semantics(self) -> None:
        spec = json.loads((REPO_ROOT / "fixtures/spec/tiny_trace_v1.json").read_text("utf-8"))
        router = spec["manifest_requirements"]["router"]
        self.assertGreater(router["num_experts"], router["top_k"])
        self.assertGreater(router["weight_sum_tolerance"], 0)
        self.assertIn("zero-based contiguous", spec["array_semantics"]["positions"])
        self.assertIn("0 <= id < router.num_experts", spec["array_semantics"]["expert_ids"])


if __name__ == "__main__":
    unittest.main()
