from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Wave1UnlockGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sot = (REPO_ROOT / "CSPM_SOT.yaml").read_text("utf-8")

    def test_g0_pass_and_wave1_baseline_are_frozen(self) -> None:
        self.assertIn('status: "G0_PASS_WAVE1_READY"', self.sot)
        self.assertIn('g0_merged_main_sha: "ae23913fceb33690037b3146bd5358894a7bd569"', self.sot)
        self.assertIn('reviewed_head_sha: "32175f194294c838de5f12627e0226cdb156f0c9"', self.sot)
        self.assertIn('s6_decision: "G0_PASS_RECOMMENDED"', self.sot)
        self.assertIn('ci_run: 32614073453', self.sot)
        self.assertIn('status: "UNLOCKED"', self.sot)
        self.assertIn('worker_start_base_sha: "ae23913fceb33690037b3146bd5358894a7bd569"', self.sot)

    def test_shared_package_root_has_explicit_s0_ownership(self) -> None:
        self.assertIn('- "src/cspm397/__init__.py"', self.sot)
        self.assertIn('path: "src/cspm397/__init__.py"', self.sot)
        self.assertIn('owner: "S0"', self.sot)
        self.assertTrue((REPO_ROOT / "src/cspm397/__init__.py").is_file())

    def test_unowned_root_errors_module_is_forbidden_and_absent(self) -> None:
        self.assertIn('path: "src/cspm397/errors.py"', self.sot)
        self.assertIn('status: "FORBIDDEN"', self.sot)
        self.assertFalse((REPO_ROOT / "src/cspm397/errors.py").exists())

    def test_package_root_exports_no_worker_api(self) -> None:
        import cspm397

        self.assertEqual(cspm397.__all__, ())
        for worker_name in ("adapters", "artifacts", "profiling", "predictability", "trace", "trajectory"):
            self.assertFalse(hasattr(cspm397, worker_name))


if __name__ == "__main__":
    unittest.main()
