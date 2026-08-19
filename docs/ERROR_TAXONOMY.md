# CSPM397 Error Taxonomy v1.1

Machine code is authoritative; prose is explanatory. When several codes appear plausible, use the most specific leaf code below. Generic parent codes MUST NOT replace a defined specific code.

| Code | Class | Default recoverability | Precedence / usage |
|---|---|---|---|
| E100 | CONTRACT | fatal | semantic contract violation not covered by E110 |
| E110 | SCHEMA | fatal | manifest/metadata/shape/value violates frozen schema/semantic validator |
| E200 | MODEL_LOAD | context-dependent | model load only |
| E210 | TOKENIZER | context-dependent | tokenizer only |
| E220 | TRACE | context-dependent | trace collection failure not covered by schema/numeric codes |
| E230 | ROUTER | context-dependent | router extraction/contract failure not covered by E110 |
| E300 | ARTIFACT_WRITE | retryable when destination is safe | write/fsync/rename failure; includes leftover tmp from failed write |
| E310 | ARTIFACT_READ | context-dependent | required artifact/sentinel/file absent or unreadable after path resolution |
| E320 | ARTIFACT_CORRUPT | fatal for immutable input | hash mismatch, truncation, corruption |
| E400 | CUDA | retryable/context-dependent | generic CUDA failure only; E410 wins for OOM |
| E410 | OOM | retryable after declared resource/config change | **wins over E400** for any out-of-memory condition |
| E420 | DEVICE | context-dependent | non-CUDA or device-selection failure |
| E500 | NUMERIC | fatal until explained | generic numeric failure only; E510/E520 win when applicable |
| E510 | NAN | fatal until explained | **wins over E500** when NaN is detected |
| E520 | INF | fatal until explained | **wins over E500** when +/-Inf is detected |
| E600 | PREDICTABILITY | context-dependent | analysis implementation failure |
| E610 | TRAJECTORY | context-dependent | trajectory/MDL implementation failure |
| E700 | EXPERIMENT | context-dependent | orchestration/state-machine failure not more specifically classified |
| E800 | DETERMINISM | fatal for deterministic scope | same frozen identity produces disallowed nondeterminism |
| E900 | INTERNAL | fatal until classified | last resort; must be reclassified when root cause becomes known |

## Stage rule

Classification is based on the stage at which the defect is observed:

- missing `manifest.json`, `COMPLETE`, or another required file while reading an artifact -> **E310**
- present manifest with missing/unknown fields, illegal shapes, invalid positions, illegal expert metadata, or cross-field inconsistency -> **E110**
- readable bytes whose SHA-256 does not match, or truncated immutable tensor -> **E320**
- `.tmp` left by an interrupted atomic write -> **E300** during writer recovery/cleanup

## Router numeric semantics

For TraceArtifact v1 with `has_router_trace=true`:

- manifest declares `router.num_experts >= 1`, `router.top_k >= 1`, `top_k <= num_experts`
- every expert id must satisfy `0 <= id < num_experts`
- expert weights must be finite and nonnegative
- per-token/per-layer selected expert weights must sum to 1 within `router.weight_sum_tolerance`
- `0 < weight_sum_tolerance <= 0.1`

## Position semantics

Within each `sequence_id`, positions are zero-based and contiguous: `0,1,2,...`. A new sequence restarts at 0. Sequence records must not interleave after a later sequence has begun. Violations are E110.

## Failure record

Every structured failure preserves at least:

```json
{
  "code": "E410",
  "stage": "TRACE",
  "recoverable": true,
  "run_id": "...",
  "git_sha": "...",
  "last_completed_chunk": 17,
  "exception": "...",
  "traceback": "..."
}
```

`recoverable=true` never permits silent fallback. The retry/config change must be recorded.
