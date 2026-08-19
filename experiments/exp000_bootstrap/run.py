from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cspm.bootstrap import build_manifest, sha256_file, validate_manifest_shape, write_canonical_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def current_code_sha() -> str:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def deterministic_result(seed: int, sample_count: int) -> dict:
    rng = random.Random(seed)
    values = [rng.randrange(0, 1_000_000) for _ in range(sample_count)]
    checksum = sum((i + 1) * value for i, value in enumerate(values))
    return {
        "schema_version": "1",
        "experiment_id": "EXP000",
        "seed": seed,
        "sample_count": sample_count,
        "values": values,
        "weighted_checksum": checksum,
    }


def run(out_dir: Path, seed: int, sample_count: int) -> dict:
    started = utc_now()
    out_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment_id": "EXP000",
        "seed": seed,
        "sample_count": sample_count,
        "generator": "python.random.Random",
        "generator_contract": "MT deterministic for identical supported Python runtime",
    }
    dataset_identity = {
        "kind": "synthetic_bootstrap_fixture",
        "version": "1",
        "rows": sample_count,
    }

    result = deterministic_result(seed=seed, sample_count=sample_count)
    result_path = out_dir / "result.json"
    result_sha = write_canonical_json(result_path, result)

    config_path = out_dir / "config.json"
    write_canonical_json(config_path, config)

    completed = utc_now()
    manifest = build_manifest(
        run_id=out_dir.name,
        experiment_id="EXP000",
        code_sha=current_code_sha(),
        config=config,
        dataset_identity=dataset_identity,
        seed=seed,
        started_at=started,
        completed_at=completed,
        artifacts=[{"path": str(result_path), "sha256": result_sha}],
        metrics={
            "sample_count": sample_count,
            "weighted_checksum": result["weighted_checksum"],
            "result_bytes": result_path.stat().st_size,
        },
    )
    errors = validate_manifest_shape(manifest)
    if errors:
        raise RuntimeError("invalid manifest: " + ";".join(errors))

    manifest_path = out_dir / "manifest.json"
    write_canonical_json(manifest_path, manifest)

    verification = {
        "result_sha256_recorded": result_sha,
        "result_sha256_actual": sha256_file(result_path),
        "match": result_sha == sha256_file(result_path),
    }
    write_canonical_json(out_dir / "verification.json", verification)
    if not verification["match"]:
        raise RuntimeError("artifact hash mismatch")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSPM EXP000 bootstrap pipeline")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=397)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args()

    manifest = run(args.out, args.seed, args.sample_count)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
