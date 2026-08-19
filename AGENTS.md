# CSPM397 Session Entry Contract

`CSPM_SOT.yaml` is the authoritative project contract. Chat history is not project state.

## Mandatory startup sequence

Every session MUST:

1. Read `CSPM_SOT.yaml` completely.
2. Record latest `main` SHA as `BASE_SHA`.
3. Confirm session ID, branch, phase, owned paths, and activation condition.
4. Refuse cross-session edits unless S0 issued an explicit contract-change task first.
5. Check the task gate before implementation or scientific claims.
6. Use GitHub main + committed contracts/manifests as source of truth.

## Phase boundary — strict

Before `SCIENCE_GO=GO_CSPM`, only Phase-A measurement/research-stack work is allowed:

- S1 MODEL + TRACE
- S2 ARTIFACT + PROFILER
- S3 PREDICTABILITY
- S4 TRAJECTORY + MDL
- S5 EXPERIMENT RUNNER
- S6 SOFTWARE QA / CI / RED TEAM
- S7 SCIENTIFIC AUDITOR / REPRODUCIBILITY

**Phase-B work is not merely LOCKED; it must not be implemented, prepared, scaffolded, assigned, or opened on a worker branch before GO_CSPM.** This includes Synaptic Memory, Macro Compiler, ChronoRouter, Residual NeuroCore, and final CSPM runtime work. S0 freezes Phase-B session IDs/ownership only after scientific GO.

## Module boundary

`TraceArtifact v1` is the only cross-module trace data contract.

- S3 must not import `transformers` or model adapters.
- S4 must not import Qwen/model objects.
- S5 orchestrates public interfaces and must not reimplement analyzer internals.
- S1 must not implement predictability or trajectory analysis.
- Contract/schema files are S0-owned and require S6 contract review.

TraceArtifact manifest authority is `<run>/trace/manifest.json`. Root experiment authority is `<run>/manifest.json`. `hashes/sha256.json` is only a checksum index. See `docs/contracts/MANIFEST_AUTHORITY.md`.

## Artifact safety

Runs are immutable.

- Never overwrite a completed or partial run directory.
- Tensor chunks use `.safetensors`; metadata uses JSON; event logs use JSONL.
- Chunk write protocol: `.tmp` -> fsync -> SHA-256 -> validate -> atomic rename.
- A run without `COMPLETE` is incomplete and rejected by default readers.
- Resume begins after the last fully validated chunk.
- Every run records code SHA, config hash, data/model identity, seed, and artifact hashes.
- Manifest paths are run-relative POSIX paths; absolute paths and `..` traversal are invalid.

Never use pickle, `allow_pickle=True`, or arbitrary Python-object serialization for trace artifacts.

## TraceArtifact semantic checks

JSON schema validation alone is insufficient. Producers/readers must also enforce the cross-field rules frozen in SOT, including:

- full 40-char git SHA; `UNKNOWN` is invalid
- `projection.output_dim == signature_dim`
- non-empty valid `selected_layers`
- nonzero token count requires chunks
- chunk indices contiguous from zero and paths matching indices
- sum of chunk token counts equals `num_tokens`
- router trace requires router metadata + expert arrays
- expert IDs and weights follow declared `num_experts` and weight-sum tolerance
- positions are zero-based contiguous inside each sequence and restart at sequence boundaries

## Error classification

Use `docs/ERROR_TAXONOMY.md`. Specific leaf errors win over generic parents: OOM -> E410, NaN -> E510, Inf -> E520. Missing required artifact at read stage -> E310; present but schema/semantic-invalid metadata -> E110; hash/truncation corruption -> E320.

## Scientific constraints

- No result without declared simple baselines.
- Final train/test split is sequence/prompt-level, not random-token split.
- Normalization, discretization, clustering, and predictors fit train only.
- No speed claim from FLOPs alone.
- No success claim from latent cosine/MSE alone.
- Compression accounting includes encoded data + dictionary + index + metadata.
- Development seed may be 397; final conclusions use SOT's predeclared multi-seed policy.
- Unexpectedly strong results are treated as possible leakage/measurement bugs until audited.
- Result lifecycle: RAW -> VALIDATED_SOFTWARE -> VALIDATED_SCIENCE -> ACCEPTED.
- Only ACCEPTED results may support conclusions.
- S6 validates software correctness; S7 separately validates scientific interpretation.
- No post-hoc relaxation of gates after results are observed.

## Bug protocol

For every meaningful bug:

1. Record exact commit/environment/command.
2. Create minimal reproduction.
3. Add a failing regression test where feasible.
4. Identify owner/root cause.
5. Owner fixes.
6. Re-run failing test and relevant regression suite.
7. Keep regression tests permanently.

## Review / merge order

Implementation -> self-test -> PR -> S6 software review -> S7 review when scientific semantics/results are involved -> S0 integration decision -> merge.

For G0 migration specifically: **S6 re-review happens before PR #5 merge.** S0 does not self-merge. Green CI is necessary but never substitutes for S6 review.

## Required worker finish sequence

1. Run assigned tests.
2. Preserve raw artifacts and hashes.
3. Commit only owned paths.
4. Open one-responsibility PR to `main`.
5. Do NOT self-merge.
6. Emit exact `RAW-HANDOFF v1` fields from SOT.

## Raw mode

RAW_MODE applies only to ChatSol/CSPM. Do not generate charts, HTML, polished prose, or duplicate summaries unless S0 explicitly requests them.

## Bootstrap commands

EXP000 remains standard-library-only:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python experiments/exp000_bootstrap/run.py --out runs/exp000/bootstrap
```

Re-running with the same output directory MUST fail without mutating existing files.
