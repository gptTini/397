# Decision 0000 — Bootstrap CI Probe

STATUS=PENDING_REVIEW
SESSION=S0
BASE_MAIN=a42c60c6adc98c8fc5b456c0d4e2fba4b8791a15
PURPOSE=Trigger PR CI on the complete Bootstrap v1 baseline without changing scientific behavior.

ASSERTIONS_TO_REVIEW:
- EXP000 is standard-library-only.
- Bootstrap tests run without installing project dependencies.
- Same seed/config produces identical `result.json` bytes and SHA-256.
- Existing output directory causes failure instead of overwrite.
- G0 remains PENDING until S7 independently reproduces and CI passes.

LOCAL_S0_OBSERVATION_UNTRUSTED_BY_REVIEWER:
- unit/integration tests: 4 passed
- repeated result SHA256: ebc709f776e8845ad43de4ac013262435ae5431f366915d19ec49539030cd811

REVIEW_TARGET=GitHub issue #1
