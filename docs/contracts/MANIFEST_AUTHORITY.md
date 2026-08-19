# Manifest Authority v1

This document removes ambiguity between the root experiment manifest and the TraceArtifact child manifest.

## Authoritative paths

- `<run>/manifest.json` = **RunManifest v1** governed by `contracts/run_manifest_v1.schema.json`.
- `<run>/trace/manifest.json` = **TraceArtifact v1 manifest** governed by `schemas/trace-v1.schema.json` plus the cross-field semantic rules in `CSPM_SOT.yaml` and `src/cspm/contracts_reference.py`.
- `<run>/hashes/sha256.json` = checksum index only. It never overrides either manifest.
- `<run>/COMPLETE` = completion sentinel. Absence makes the run incomplete by default.

## Composition and duplicated provenance

When a run contains TraceArtifact v1, the root RunManifest `artifacts` list MUST contain exactly one `trace/manifest.json` entry with the exact SHA-256 of the child manifest.

The duplicated identity fields are not independent metadata. They are equality invariants:

- `root.run_id == trace.run_id`
- `root.code_sha == trace.git_sha`
- `root.config_sha256 == trace.config_hash`
- `root.seed == trace.seed`

Any mismatch is a composition/semantic contract failure (`E110`). A child-manifest hash mismatch is byte-integrity failure (`E320`). `validate_run_trace_composition` is the bootstrap reference validator; S2/S6 production validators must preserve these invariants.

The actual child-manifest SHA-256 is a **mandatory validator input**. Readers MUST compute it from the loaded `trace/manifest.json` bytes and pass it to composition validation. There is no permitted API path that skips child-hash verification; omission is a caller/programming error rather than successful validation.

The child trace manifest never supplies experiment-level metrics or run state. The root manifest never redefines trace tensor descriptors, projection/router semantics, chunk ordering, or sequence statistics.

## Trace payload descriptors

TraceArtifact v1 has four views of the same tensor inventory:

1. `arrays` — symbolic shape contract per tensor name.
2. `array_shapes` — exact payload-derived integer shape per tensor name.
3. `array_dtypes` — exact payload-derived dtype per tensor name.
4. Loaded payload tensors — independently observed tensor names/shapes/dtypes.

The key sets of all four views MUST be identical. A tensor declared in only one or two descriptor maps is invalid. Readers recompute tensor names, shapes, dtypes, and sequence statistics from loaded payload bytes before analysis and reject mismatches as `E110`.

The root `dtype` field in TraceArtifact v1 means **state-signature dtype only** and MUST equal `array_dtypes.state_signatures`. Integer identity/index arrays use an allowed integer dtype; state signatures and router numeric weights/entropy use allowed floating dtypes.

## Path rule

All RunManifest artifact paths use one canonical cross-platform subset rather than host-native path syntax.

- Paths are run-relative, use `/` separators, and consist of lowercase ASCII segments only.
- Segment characters are limited to `a-z`, `0-9`, `_`, `-`, and `.`; a segment cannot start or end with `.`.
- Empty segments, repeated separators, raw `.`/`..` segments, absolute paths, drive-prefixed paths, backslashes, spaces, Unicode aliases, and uppercase/case-fold aliases are invalid.
- `:` is invalid, so Windows alternate data stream forms such as `result.json:stream` are impossible.
- A segment whose basename before the first `.` is a Windows reserved device name is invalid, case-independently by canonical lowercase rule: `con`, `prn`, `aux`, `nul`, `com1` through `com9`, and `lpt1` through `lpt9`. Extensions do not make a reserved device name valid.
- A trailing dot or space is invalid and cannot create a Windows-normalized alias.

`contracts/run_manifest_v1.schema.json` and `src/cspm/bootstrap.py::_is_safe_run_relative_path` MUST accept and reject the same path set. Any disagreement is a contract defect (`E110`).

## Validation order

1. Resolve run directory and required files. Missing files -> E310.
2. Validate root RunManifest shape/schema -> E110 on violation.
3. Load `trace/manifest.json`, compute SHA-256 from its exact bytes, and verify the root artifact reference -> E320 on mismatch.
4. Validate `trace/manifest.json` JSON schema -> E110.
5. Apply TraceArtifact semantic validator -> E110.
6. Apply root↔trace composition validator with the mandatory computed child SHA -> E110 for provenance mismatch, E320 for child-manifest hash mismatch.
7. Load chunks and independently recompute tensor inventory, shapes, dtypes, sequence stats, plus tensor-level semantic/numeric checks -> E110/E510/E520 as applicable.
8. Validate chunk hashes -> E320 for byte corruption.
9. Require `COMPLETE` for default read acceptance.
