from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cspm.bootstrap import (
    canonical_json_bytes,
    sha256_file,
    validate_manifest_shape,
    validate_run_manifest,
)


def _load_exp000_module():
    path = REPO_ROOT / "experiments/exp000_bootstrap/run.py"
    spec = importlib.util.spec_from_file_location("cspm_exp000_run_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load EXP000 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_root_manifest() -> dict:
    return {
        "schema_version": "1",
        "run_id": "x",
        "experiment_id": "EXP000",
        "code_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "seed": 397,
        "started_at": "x",
        "completed_at": "x",
        "artifacts": [{"path": "result.json", "sha256": "d" * 64}],
        "metrics": {},
    }


def _root_manifest_with_path(path: str) -> dict:
    manifest = _valid_root_manifest()
    manifest["artifacts"] = [{"path": path, "sha256": "d" * 64}]
    return manifest


class BootstrapUnitTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self) -> None:
        left = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
        right = {"nested": {"x": 1, "y": 2}, "a": 1, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertTrue(canonical_json_bytes(left).endswith(b"\n"))

    def test_manifest_validator_rejects_missing_fields(self) -> None:
        errors = validate_manifest_shape({"schema_version": "1"})
        self.assertTrue(any(error.startswith("missing_fields=") for error in errors))

    def test_manifest_validator_rejects_unknown_sha_and_unsafe_path(self) -> None:
        manifest = _valid_root_manifest()
        manifest["code_sha"] = "UNKNOWN"
        manifest["artifacts"] = [{"path": "../result.json", "sha256": "2" * 64}]
        errors = validate_manifest_shape(manifest)
        self.assertIn("code_sha_must_be_full_40_hex", errors)
        self.assertIn("artifact_0_path_invalid", errors)

    def test_root_manifest_schema_and_helper_agree_on_cross_platform_paths(self) -> None:
        schema = json.loads((REPO_ROOT / "contracts/run_manifest_v1.schema.json").read_text("utf-8"))
        pattern_text = schema["properties"]["artifacts"]["items"]["properties"]["path"]["pattern"]
        pattern = re.compile(pattern_text)

        valid_paths = (
            "result.json",
            "trace/manifest.json",
            "nested/a-b_1.json",
            "com10.json",
            "nulx",
            "_meta/result-1.json",
        )
        invalid_paths = (
            "/outside/result.json",
            "../outside/result.json",
            "nested/../outside.json",
            "./result.json",
            "nested/./result.json",
            "C:\\outside\\result.json",
            "\\\\server\\share\\result.json",
            "..\\outside\\result.json",
            "C:/outside/result.json",
            "C:relative-result.json",
            "nested\\result.json",
            "NUL",
            "CON",
            "AUX.txt",
            "COM1",
            "LPT1.txt",
            "nul",
            "con",
            "aux.txt",
            "com1",
            "lpt1.txt",
            "con.foo.bar",
            "result.json:stream",
            "nested/result.json:stream",
            "trailing.",
            "trailing ",
            "nested/name.",
            "nested/name ",
            "UPPER.json",
            "unicode/한글.json",
            "double//slash",
        )

        for path in valid_paths:
            with self.subTest(path=path):
                self.assertIsNotNone(pattern.fullmatch(path))
                self.assertNotIn("artifact_0_path_invalid", validate_run_manifest(_root_manifest_with_path(path)))

        for path in invalid_paths:
            with self.subTest(path=path):
                self.assertIsNone(pattern.fullmatch(path))
                self.assertIn("artifact_0_path_invalid", validate_run_manifest(_root_manifest_with_path(path)))

    def test_manifest_authority_documents_portable_path_contract(self) -> None:
        authority = (REPO_ROOT / "docs/contracts/MANIFEST_AUTHORITY.md").read_text("utf-8")
        for required in (
            "lowercase ASCII",
            "Windows reserved device",
            "alternate data stream",
            "trailing dot or space",
        ):
            self.assertIn(required, authority)

    def test_root_manifest_reference_pipeline_rejects_schema_invalid_cases(self) -> None:
        cases = []

        value = _valid_root_manifest()
        value["run_id"] = ""
        cases.append(("empty_run_id", value, "run_id_must_be_nonempty_string"))

        value = _valid_root_manifest()
        value["experiment_id"] = "oops"
        cases.append(("bad_experiment_id", value, "experiment_id_must_match_EXPddd"))

        value = _valid_root_manifest()
        value["seed"] = "397"
        cases.append(("string_seed", value, "seed_must_be_integer"))

        value = _valid_root_manifest()
        value["seed"] = True
        cases.append(("boolean_seed", value, "seed_must_be_integer"))

        value = _valid_root_manifest()
        value["started_at"] = ""
        cases.append(("empty_started_at", value, "started_at_must_be_nonempty_string"))

        value = _valid_root_manifest()
        value["completed_at"] = ""
        cases.append(("empty_completed_at", value, "completed_at_must_be_nonempty_string"))

        value = _valid_root_manifest()
        value["metrics"] = []
        cases.append(("metrics_not_object", value, "metrics_must_be_object"))

        value = _valid_root_manifest()
        value["unknown"] = 1
        cases.append(("unknown_top_level", value, "unknown_fields=['unknown']"))

        value = _valid_root_manifest()
        value["artifacts"][0]["unknown"] = 1
        cases.append(("unknown_artifact_field", value, "artifact_0_fields_invalid"))

        for name, manifest, expected in cases:
            with self.subTest(name=name):
                errors = validate_run_manifest(manifest)
                self.assertIn(expected, errors)
                self.assertEqual(errors, validate_manifest_shape(manifest))

    def test_root_manifest_pipeline_schema_field_surface_is_frozen(self) -> None:
        schema = json.loads((REPO_ROOT / "contracts/run_manifest_v1.schema.json").read_text("utf-8"))
        valid = _valid_root_manifest()
        self.assertEqual(set(schema["required"]), set(valid))
        self.assertEqual(set(schema["properties"]), set(valid))
        self.assertFalse(schema["additionalProperties"])
        artifact_schema = schema["properties"]["artifacts"]["items"]
        self.assertEqual(set(artifact_schema["required"]), {"path", "sha256"})
        self.assertEqual(set(artifact_schema["properties"]), {"path", "sha256"})
        self.assertFalse(artifact_schema["additionalProperties"])
        self.assertEqual(validate_run_manifest(valid), [])

    def test_duplicate_artifact_path_is_explicit_semantic_rejection(self) -> None:
        value = _valid_root_manifest()
        value["artifacts"] = [
            {"path": "result.json", "sha256": "d" * 64},
            {"path": "result.json", "sha256": "e" * 64},
        ]
        self.assertIn("duplicate_artifact_path", validate_run_manifest(value))
        authority = (REPO_ROOT / "docs/contracts/MANIFEST_AUTHORITY.md").read_text("utf-8")
        self.assertIn("duplicate artifact paths", authority)
        self.assertIn("validate_run_manifest", authority)

    def test_non_object_root_manifest_fails_closed(self) -> None:
        self.assertEqual(validate_run_manifest([]), ["root_manifest_must_be_object"])


class Exp000IntegrationTests(unittest.TestCase):
    def _run(self, out: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments/exp000_bootstrap/run.py"),
                "--out",
                str(out),
                "--seed",
                "397",
                "--sample-count",
                "32",
            ],
            check=False,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )

    def test_clean_reproduction_has_identical_raw_result_and_identity_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"

            proc_a = self._run(run_a)
            proc_b = self._run(run_b)
            self.assertEqual(proc_a.returncode, 0, proc_a.stderr)
            self.assertEqual(proc_b.returncode, 0, proc_b.stderr)

            self.assertEqual((run_a / "result.json").read_bytes(), (run_b / "result.json").read_bytes())
            self.assertEqual((run_a / "config.json").read_bytes(), (run_b / "config.json").read_bytes())
            self.assertEqual(
                (run_a / "dataset_identity.json").read_bytes(),
                (run_b / "dataset_identity.json").read_bytes(),
            )

            manifest_a = json.loads((run_a / "manifest.json").read_text("utf-8"))
            manifest_b = json.loads((run_b / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest_a["config_sha256"], manifest_b["config_sha256"])
            self.assertEqual(manifest_a["dataset_sha256"], manifest_b["dataset_sha256"])
            self.assertEqual(manifest_a["seed"], manifest_b["seed"])
            self.assertEqual(manifest_a["metrics"], manifest_b["metrics"])
            self.assertEqual(manifest_a["artifacts"], manifest_b["artifacts"])
            self.assertEqual(manifest_a["artifacts"][0]["path"], "result.json")

            self.assertEqual(manifest_a["config_sha256"], sha256_file(run_a / "config.json"))
            self.assertEqual(manifest_a["dataset_sha256"], sha256_file(run_a / "dataset_identity.json"))
            self.assertEqual(manifest_a["artifacts"][0]["sha256"], sha256_file(run_a / "result.json"))

            verify_a = json.loads((run_a / "verification.json").read_text("utf-8"))
            verify_b = json.loads((run_b / "verification.json").read_text("utf-8"))
            self.assertTrue(verify_a["match"])
            self.assertTrue(verify_b["match"])
            self.assertEqual(verify_a["config_sha256_recorded"], verify_a["config_sha256_actual"])
            self.assertEqual(verify_a["dataset_sha256_recorded"], verify_a["dataset_sha256_actual"])

            self.assertEqual((run_a / "COMPLETE").read_bytes(), b"CSPM_RUN_COMPLETE_V1\n")
            self.assertEqual((run_b / "COMPLETE").read_bytes(), b"CSPM_RUN_COMPLETE_V1\n")
            self.assertFalse((run_a / ".COMPLETE.tmp").exists())
            self.assertFalse((run_b / ".COMPLETE.tmp").exists())

    def test_failure_before_final_commit_never_writes_complete(self) -> None:
        module = _load_exp000_module()
        original_write = module.write_canonical_json

        def fail_on_verification(path, value):
            if Path(path).name == "verification.json":
                raise RuntimeError("injected failure before completion")
            return original_write(path, value)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "partial_run"
            with mock.patch.object(module, "write_canonical_json", side_effect=fail_on_verification):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    module.run(out, 397, 32)
            self.assertTrue(out.exists())
            self.assertFalse((out / "COMPLETE").exists())
            self.assertFalse((out / ".COMPLETE.tmp").exists())

    def test_replace_failure_cleans_partial_complete_tmp(self) -> None:
        module = _load_exp000_module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "replace_failure"
            with mock.patch.object(module.os, "replace", side_effect=OSError("injected replace failure")):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    module.run(out, 397, 32)
            self.assertFalse((out / "COMPLETE").exists())
            self.assertFalse((out / ".COMPLETE.tmp").exists())

    def test_complete_tmp_fsync_failure_cleans_partial_tmp(self) -> None:
        module = _load_exp000_module()
        original_fsync = module.os.fsync
        call_count = 0

        def fail_on_complete_tmp_fsync(fd):
            nonlocal call_count
            call_count += 1
            if call_count == 6:
                raise OSError("injected COMPLETE tmp fsync failure")
            return original_fsync(fd)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tmp_fsync_failure"
            with mock.patch.object(module.os, "fsync", side_effect=fail_on_complete_tmp_fsync):
                with self.assertRaisesRegex(OSError, "injected COMPLETE tmp fsync failure"):
                    module.run(out, 397, 32)
            self.assertFalse((out / "COMPLETE").exists())
            self.assertFalse((out / ".COMPLETE.tmp").exists())

    def test_existing_run_directory_is_never_overwritten_or_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "same_run"
            first = self._run(out)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((out / "COMPLETE").exists())
            before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

            second = self._run(out)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)
            after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
