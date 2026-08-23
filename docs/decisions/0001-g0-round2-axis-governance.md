# Decision 0001 — G0 Round-2 Trace Axis + Governance Rework

STATUS=PENDING_S6_REREVIEW
PR=5
SOT_SCHEMA=2.2

SOURCE=S6 G0 rereview findings B008..B011

B008_TRACE_AXIS_CARDINALITY:
- T = num_tokens
- L = len(selected_layers)
- D = signature_dim = projection.output_dim
- K = router.top_k when router trace exists
- Trace manifest now carries exact integer `array_shapes`.
- Trace manifest now carries derived `sequence_stats`.
- Producers derive shapes/stats from actual payload.
- Consumers must recompute them from loaded tensors and reject mismatches before analysis.
- `num_tokens=0` iff `num_sequences=0`; non-empty trace requires `1 <= num_sequences <= num_tokens`.
- sequence IDs are exactly contiguous zero-based IDs and every declared sequence appears.

B009_TASK_SOURCE_VERSION:
- Issue #1 must point to current SOT schema 2.2/current PR head before S6 rereview.

B010_OWNERSHIP_EXCEPTION:
- Normal owner of `.github/workflows/**` is S6.
- Existing PR #5 receives one narrow exception `G0_PR5_S0_CI_REWORK` because S6's first review explicitly delegated B007 rework to S0.
- Exception applies only to `.github/workflows/ci.yml` on PR #5 and expires when the PR merges/closes.
- S6 rereview is mandatory before merge.

B011_RAW_HANDOFF:
- Base allowed DECISION values and G0/science extensions are frozen in SOT 2.2.
- RAW-HANDOFF v1 template is restored.

MERGE_GATE:
- current PR head CI green on Windows + Ubuntu
- S6 rereview current head
- S6 DECISION=G0_PASS_RECOMMENDED
- S0 does not self-merge
