# Manifest Authority v1

This document removes ambiguity between the root experiment manifest and the TraceArtifact child manifest.

## Authoritative paths

- `<run>/manifest.json` = **RunManifest v1** governed by `contracts/run_manifest_v1.schema.json`.
- `<run>/trace/manifest.json` = **TraceArtifact v1 manifest** governed by `schemas/trace-v1.schema.json` plus the cross-field semantic rules frozen in `CSPM_SOT.yaml` and `validate_trace_manifest_semantics`.
- `<run>/hashes/sha256.json` = checksum index only. It is never a manifest and never overrides either manifest.
- `<run>/COMPLETE` = completion sentinel. Its presence means the run reached committed completion; absence makes the run incomplete by default.

## Composition

RunManifest v1 describes the experiment run as a whole. If a run contains a TraceArtifact, the root RunManifest `artifacts` list MUST contain an entry for `trace/manifest.json` with the exact SHA-256 of that child manifest. Individual trace chunk hashes live in the child trace manifest and may also be duplicated in `hashes/sha256.json` for fast integrity scanning. Any duplicate hash record must agree byte-for-byte; disagreement is E320.

The child trace manifest never supplies experiment-level metrics, run state, or configuration identity authority. The root manifest never redefines tensor shapes, router semantics, chunk ordering, or trace-specific metadata.

## Path rule

All manifest artifact paths are run-relative POSIX-style paths. Absolute paths and `..` traversal are forbidden.

## Validation order

1. Resolve run directory and required files. Missing files -> E310.
2. Validate root RunManifest shape/schema -> E110 on violation.
3. Validate referenced artifact hashes -> E320 on mismatch.
4. If TraceArtifact exists, validate `trace/manifest.json` JSON schema -> E110.
5. Apply TraceArtifact cross-field semantic validator -> E110.
6. Validate chunk hashes and tensor-level semantics -> E320 for byte corruption; E110/E510/E520 for semantic/numeric violations.
7. Require `COMPLETE` for default read acceptance.
