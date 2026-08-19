# CSPM397 Session Entry Contract

This repository is a multi-session research project. `CSPM_SOT.yaml` is the authoritative project contract.

## Mandatory startup sequence

Every session MUST do this before editing code:

1. Read `CSPM_SOT.yaml` completely.
2. Record latest `main` SHA as `BASE_SHA`.
3. Confirm session ID, branch, phase, owned paths, and activation gate.
4. Refuse cross-session edits unless S0 first updates the SOT or issues an explicit contract-change task.
5. Use GitHub main + committed manifests/contracts as project state. Chat history is not source of truth.
6. If a gate is LOCKED, implementation may only be prepared when explicitly authorized; no scientific claim may be made.

## Phase-A boundary

Before `SCIENCE_GO=GO_CSPM`, sessions build and validate the measurement/research stack only:

- S1 MODEL + TRACE
- S2 ARTIFACT + PROFILER
- S3 PREDICTABILITY
- S4 TRAJECTORY + MDL
- S5 EXPERIMENT RUNNER
- S6 SOFTWARE QA / CI / RED TEAM
- S7 SCIENTIFIC AUDITOR / REPRODUCIBILITY

Synaptic Memory, Macro Compiler, ChronoRouter, Residual NeuroCore, and the final runtime are Phase B and MUST NOT be treated as active implementation work before scientific GO.

## Module boundary

`TraceArtifact v1` is the only cross-module trace data contract.

- S3 must not import `transformers` or model adapters.
- S4 must not import Qwen/model objects.
- S5 orchestrates public interfaces and must not reimplement analyzer internals.
- S1 must not implement predictability or trajectory analysis.
- Contract/schema files are S0-owned.

## Artifact safety

Runs are immutable.

- Never overwrite a completed or partial run directory.
- Tensor chunks use `.safetensors`; metadata uses JSON; event logs use JSONL.
- Chunk write protocol: `.tmp` -> fsync -> SHA-256 -> validate -> atomic rename.
- A run without `COMPLETE` is incomplete and is rejected by default readers.
- Resume starts from the last validated chunk.
- Every run records code SHA, config hash, data/model identity, seed, and artifact hashes.

Never use pickle/`allow_pickle=True`/arbitrary Python-object serialization for trace artifacts.

## Scientific constraints

- No result without declared simple baselines.
- Final train/test split is sequence/prompt-level, not random-token split.
- Normalization, discretization, clustering, and predictors fit train only.
- No speed claim from FLOPs alone.
- No success claim from latent cosine/MSE alone.
- Compression accounting includes encoded data + dictionary + index + metadata.
- Development seed may be 397; final conclusions use the predeclared multi-seed policy in SOT.
- Unexpectedly strong results are treated as possible leakage/measurement bugs until audited.
- Result lifecycle: RAW -> VALIDATED_SOFTWARE -> VALIDATED_SCIENCE -> ACCEPTED.
- Only ACCEPTED results may support project conclusions.
- S6 validates software correctness; S7 separately validates scientific interpretation.
- No post-hoc relaxation of gates after observing results.

## Bug protocol

Do not patch first. For every meaningful bug:

1. Record exact commit/environment/command.
2. Create minimal reproduction.
3. Add a failing regression test where feasible.
4. Identify owner session and root cause.
5. Owner fixes.
6. Re-run failing test and full relevant regression suite.
7. Keep regression tests permanently.

Use error taxonomy from `CSPM_SOT.yaml` / `docs/ERROR_TAXONOMY.md`.

## Required worker finish sequence

1. Run assigned tests.
2. Preserve raw artifacts and hashes.
3. Commit only owned paths.
4. Open one-responsibility PR to `main`.
5. Do NOT self-merge.
6. Emit exact `RAW-HANDOFF v1` fields from SOT.

## Raw mode

RAW_MODE applies only to ChatSol/CSPM. Worker sessions should not spend time generating charts, HTML, polished prose, or duplicate summaries unless S0 explicitly requests them.

## Bootstrap commands

Existing EXP000 bootstrap remains standard-library-only:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python experiments/exp000_bootstrap/run.py --out runs/exp000/bootstrap
```

Re-running with the same output directory MUST fail instead of overwrite.
