from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cspm.bootstrap import canonical_json_bytes, sha256_file, validate_manifest_shape


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
        manifest = {
            "schema_version": "1",
            "run_id": "x",
            "experiment_id": "EXP000",
            "code_sha": "UNKNOWN",
            "config_sha256": "0" * 64,
            "dataset_sha256": "1" * 64,
            "seed": 397,
            "started_at": "x",
            "completed_at": "x",
            "artifacts": [{"path": "../result.json", "sha256": "2" * 64}],
            "metrics": {},
        }
        errors = validate_manifest_shape(manifest)
        self.assertIn("code_sha_must_be_full_40_hex", errors)
        self.assertIn("artifact_0_path_invalid", errors)


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

    def test_existing_run_directory_is_never_overwritten_or_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "same_run"
            first = self._run(out)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

            second = self._run(out)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)
            after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
