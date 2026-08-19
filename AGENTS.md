# CSPM397 Session Entry Contract

`CSPM_SOT.yaml` is the authoritative project contract. Chat history is not project state.

## Mandatory startup sequence

Every session MUST:

1. Read `CSPM_SOT.yaml` completely.
2. Record latest `main` SHA as `BASE_SHA`.
3. Confirm session ID, branch, phase, owned paths, and activation condition.
4. Refuse cross-session edits unless S0 has already recorded an explicit task/exception in the SOT.
5. Check the task gate before implementation or scientific claims.
6. Use GitHub main + committed contracts/manifests as source of truth.

## Phase boundary — strict

Before `SCIENCE_GO=GO_CSPM`, only Phase-A measurement/research-stack work is allowed: S1 MODEL+TRACE, S2 ARTIFACT+PROFILER, S3 PREDICTABILITY, S4 TRAJECTORY+MDL, S5 EXPERIMENT RUNNER, S6 SOFTWARE QA/CI/RED TEAM, and S7 SCIENTIFIC AUDITOR.

**Phase-B work must not be implemented, prepared, scaffolded, assigned, or opened on a worker branch before GO_CSPM.** S0 freezes Phase-B session IDs/ownership only after scientific GO.

## Module boundary

`TraceArtifact v1` is the only cross-module trace data contract.

- S3 must not import `transformers` or model adapters.
- S4 must not import Qwen/model objects.
- S5 orchestrates public interfaces and must not reimplement analyzer internals.
- S1 must not implement predictability or trajectory analysis.
- Contract/schema files are S0-owned and require S6 contract review.
- `src/cspm/contracts_reference.py` is the G0 authoritative reference for trace semantic, payload-summary, and root↔trace composition validation.
- `src/cspm/bootstrap.py` only re-exports those trace validators and must never define a divergent trace contract.

## Manifest composition

Trace authority is `<run>/trace/manifest.json`; root run authority is `<run>/manifest.json`; `hashes/sha256.json` is checksum index only.

For any run containing TraceArtifact v1, all of these are mandatory:

- root contains exactly one `trace/manifest.json` artifact reference with exact child SHA-256
- `root.run_id == trace.run_id`
- `root.code_sha == trace.git_sha`
- `root.config_sha256 == trace.config_hash`
- `root.seed == trace.seed`

Provenance mismatch is E110; child-manifest byte/hash mismatch is E320. See `docs/contracts/MANIFEST_AUTHORITY.md`.

## Artifact safety

Runs are immutable. Never overwrite a completed or partial run directory. Tensor chunks use `.safetensors`; metadata uses JSON; event logs use JSONL. Chunk write protocol is `.tmp -> fsync -> SHA-256 -> validate -> atomic rename`. A run without `COMPLETE` is incomplete and rejected by default readers. Resume begins after the last fully validated chunk. Manifest paths are run-relative POSIX paths; absolute paths and `..` traversal are invalid. Never use pickle, `allow_pickle=True`, or arbitrary Python-object serialization.

## TraceArtifact semantic checks

JSON schema validation alone is insufficient.

Axis contract:

- `T == num_tokens`
- `L == len(selected_layers)`, not `num_layers`
- `D == signature_dim == projection.output_dim`
- when router trace exists, `K == router.top_k`

Tensor descriptor contract:

- the key sets of `arrays`, `array_shapes`, `array_dtypes`, and loaded payload tensor names are exactly equal
- `dtype` means **state-signature dtype only** and equals `array_dtypes.state_signatures`
- token/sequence/position/expert-id tensors use an allowed integer dtype
- state signatures/expert weights/router entropy use allowed floating dtypes
- producers derive names/shapes/dtypes from actual tensors
- S2/S6 readers independently recompute names/shapes/dtypes from loaded bytes before analysis

Sequence/cardinality contract:

- `num_tokens=0` iff `num_sequences=0`
- non-empty trace has `1 <= num_sequences <= num_tokens`
- sequence IDs use every contiguous integer id `0..num_sequences-1`
- positions are zero-based contiguous per sequence and sequence blocks do not interleave
- consumers independently recompute sequence statistics

Any payload-vs-manifest descriptor/cardinality mismatch is E110 and analysis stops.

Other required checks include full 40-char git SHA, valid `selected_layers`, contiguous chunk indices/paths, sum(chunk.tokens)==num_tokens, router metadata/descriptors when router trace exists, expert ID range, finite/nonnegative weights, and weight-sum tolerance.

## Error classification

Use `docs/ERROR_TAXONOMY.md`. Specific leaf errors win over generic parents: OOM -> E410, NaN -> E510, Inf -> E520. Missing required artifact at read stage -> E310; schema/semantic/provenance/payload mismatch -> E110; hash/truncation corruption -> E320.

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
- No post-hoc relaxation of gates.

## Bug protocol

For every meaningful bug: record exact commit/environment/command; create minimal reproduction; add a failing regression test where feasible; identify owner/root cause; owner fixes; rerun failing test and relevant regression suite; keep regression tests permanently.

## Ownership

Commit only owned paths. S0 explicitly owns the legacy G0 reference paths listed in SOT, including `src/cspm/bootstrap.py`, `src/cspm/contracts_reference.py`, EXP000, root bootstrap/contract tests, and `docs/ERROR_TAXONOMY.md`. Cross-session exceptions are valid only when narrowly recorded in SOT with PR/path/reason/reviewer/expiry. The PR #5 CI workflow exception expires when PR #5 is merged or closed.

## Review / merge order

Implementation -> self-test -> PR -> S6 software review -> S7 review when scientific semantics/results are involved -> S0 integration decision -> merge.

For G0 migration specifically, S6 rereview happens before PR #5 merge. S0 does not self-merge. Green CI is necessary but never substitutes for S6 review.

## Required worker finish sequence

1. Run assigned tests.
2. Preserve raw artifacts and hashes.
3. Commit only owned paths.
4. Open one-responsibility PR to `main`.
5. Do NOT self-merge.
6. Emit exact `RAW-HANDOFF v1` fields and allowed decision values from SOT.

## Raw mode

RAW_MODE applies only to ChatSol/CSPM. Do not generate charts, HTML, polished prose, or duplicate summaries unless S0 explicitly requests them.

## Bootstrap commands

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python experiments/exp000_bootstrap/run.py --out runs/exp000/bootstrap
```

Re-running with the same output directory MUST fail without mutating existing files.
