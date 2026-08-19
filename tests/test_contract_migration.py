from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cspm.bootstrap import validate_trace_manifest_semantics, validate_trace_payload_summary


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

    def test_rejects_nonzero_tokens_with_no_sequences(self) -> None:
        value = valid_trace_manifest()
        value["num_sequences"] = 0
        value["sequence_stats"].update({"unique_count": 0, "min_id": None, "max_id": None})
        self.assertIn(
            "nonempty_trace_requires_at_least_one_sequence",
            validate_trace_manifest_semantics(value),
        )

    def test_rejects_num_sequences_above_num_tokens(self) -> None:
        value = valid_trace_manifest()
        value["num_sequences"] = 5
        value["sequence_stats"].update({"unique_count": 5, "max_id": 4})
        self.assertIn(
            "num_sequences_must_not_exceed_num_tokens",
            validate_trace_manifest_semantics(value),
        )

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

    def test_rejects_sequence_stats_mismatch(self) -> None:
        value = valid_trace_manifest()
        value["sequence_stats"]["unique_count"] = 2
        value["sequence_stats"]["ids_contiguous_zero_based"] = False
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("sequence_unique_count_must_equal_num_sequences", errors)
        self.assertIn("sequence_ids_must_be_contiguous_zero_based", errors)

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
        value["array_shapes"].pop("expert_ids")
        value["array_shapes"].pop("expert_weights")
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("router_trace_requires_expert_arrays", errors)
        self.assertIn("router_trace_requires_router_metadata", errors)

    def test_rejects_router_arrays_and_shapes_when_router_trace_false(self) -> None:
        value = valid_trace_manifest()
        value["has_router_trace"] = False
        value.pop("router")
        errors = validate_trace_manifest_semantics(value)
        self.assertIn("router_arrays_forbidden_without_router_trace", errors)
        self.assertIn("router_array_shapes_forbidden_without_router_trace", errors)

    def test_payload_summary_must_be_recomputed_and_match_manifest(self) -> None:
        value = valid_trace_manifest()
        good_shapes = dict(value["array_shapes"])
        good_stats = dict(value["sequence_stats"])
        self.assertEqual(
            validate_trace_payload_summary(
                value,
                observed_array_shapes=good_shapes,
                observed_sequence_stats=good_stats,
            ),
            [],
        )
        bad_shapes = dict(good_shapes)
        bad_shapes["state_signatures"] = [4, 1, 8]
        self.assertIn(
            "payload_array_shapes_do_not_match_manifest",
            validate_trace_payload_summary(
                value,
                observed_array_shapes=bad_shapes,
                observed_sequence_stats=good_stats,
            ),
        )


class FrozenContractFileTests(unittest.TestCase):
    def test_trace_schema_freezes_axis_and_router_contracts(self) -> None:
        schema = json.loads((REPO_ROOT / "schemas/trace-v1.schema.json").read_text("utf-8"))
        required = set(schema["required"])
        self.assertIn("projection", required)
        self.assertIn("array_shapes", required)
        self.assertIn("sequence_stats", required)
        self.assertNotIn("projection_algorithm", required)
        self.assertEqual(schema["properties"]["git_sha"]["pattern"], "^[0-9a-f]{40}$")
        self.assertEqual(schema["properties"]["selected_layers"]["minItems"], 1)
        self.assertEqual(schema["$defs"]["shape3"]["minItems"], 3)
        self.assertEqual(schema["$defs"]["shape3"]["maxItems"], 3)
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
            "state_axis_cardinality_mismatch",
            "router_k_axis_mismatch",
            "sequence_cardinality_mismatch",
        }:
            self.assertIn(required, ids)

    def test_golden_fixture_freezes_expert_position_and_axis_semantics(self) -> None:
        spec = json.loads((REPO_ROOT / "fixtures/spec/tiny_trace_v1.json").read_text("utf-8"))
        router = spec["manifest_requirements"]["router"]
        self.assertGreater(router["num_experts"], router["top_k"])
        self.assertGreater(router["weight_sum_tolerance"], 0)
        self.assertIn("zero-based contiguous", spec["array_semantics"]["positions"])
        self.assertIn("0 <= id < router.num_experts", spec["array_semantics"]["expert_ids"])
        shapes = spec["manifest_requirements"]["array_shapes"]
        self.assertEqual(shapes["state_signatures"], [64, 4, 8])
        self.assertEqual(shapes["expert_ids"], [64, 4, 2])

    def test_sot_records_migration_exception_and_raw_handoff_contract(self) -> None:
        sot = (REPO_ROOT / "CSPM_SOT.yaml").read_text("utf-8")
        self.assertIn('schema_version: "2.2"', sot)
        self.assertIn('id: "G0_PR5_S0_CI_REWORK"', sot)
        self.assertIn('paths: [".github/workflows/ci.yml"]', sot)
        self.assertIn('before_go_policy: "NO_IMPLEMENTATION_NO_PREPARATION_NO_BRANCH_TASKS"', sot)
        self.assertIn("allowed_decisions_base:", sot)
        self.assertIn("template: |-", sot)

    def test_bootstrap_decision_requires_s6_review_before_merge(self) -> None:
        decision = (
            REPO_ROOT / "docs/decisions/0000-bootstrap-ci-probe.md"
        ).read_text("utf-8")
        self.assertIn("SOFTWARE_REVIEWER=S6", decision)
        self.assertIn("S6 independently re-reviews PR #5", decision)
        self.assertNotIn("S7 independently reproduces and CI passes", decision)


if __name__ == "__main__":
    unittest.main()
