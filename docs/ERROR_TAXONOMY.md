# CSPM397 Error Taxonomy v1

Machine code is authoritative; prose is explanatory.

| Code | Class | Default recoverability |
|---|---|---|
| E100 | CONTRACT | fatal |
| E110 | SCHEMA | fatal |
| E200 | MODEL_LOAD | context-dependent |
| E210 | TOKENIZER | context-dependent |
| E220 | TRACE | context-dependent |
| E230 | ROUTER | context-dependent |
| E300 | ARTIFACT_WRITE | retryable when destination is safe |
| E310 | ARTIFACT_READ | context-dependent |
| E320 | ARTIFACT_CORRUPT | fatal for immutable input |
| E400 | CUDA | retryable/context-dependent |
| E410 | OOM | retryable after declared resource/config change |
| E420 | DEVICE | context-dependent |
| E500 | NUMERIC | fatal until reproduced and explained |
| E510 | NAN | fatal until reproduced and explained |
| E520 | INF | fatal until reproduced and explained |
| E600 | PREDICTABILITY | context-dependent |
| E610 | TRAJECTORY | context-dependent |
| E700 | EXPERIMENT | context-dependent |
| E800 | DETERMINISM | fatal for deterministic scope |
| E900 | INTERNAL | fatal until classified |

## Failure record

Every structured failure should preserve at least:

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
