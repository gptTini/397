# Decision 0002 — G0 Round-3 Contract Rework

STATUS=PENDING_S6_REREVIEW
SESSION=S0
TARGET_PR=5
TRIGGER=S6 findings B012,B013,B014

## B012 ownership

S0 permanently owns the legacy G0 reference surface until a later explicit migration decision:

- `src/cspm/bootstrap.py`
- `src/cspm/contracts_reference.py`
- `experiments/exp000_bootstrap/**`
- `tests/test_bootstrap.py`
- `tests/test_contract_migration.py`
- `docs/ERROR_TAXONOMY.md`

The existing PR #5 exception for `.github/workflows/ci.yml` remains narrow and expires when PR #5 merges/closes.

## B013 root/trace provenance

For a run with TraceArtifact v1:

- root.run_id == trace.run_id
- root.code_sha == trace.git_sha
- root.config_sha256 == trace.config_hash
- root.seed == trace.seed
- root has exactly one `trace/manifest.json` artifact with exact child SHA-256

Semantic provenance mismatch = E110. Child manifest hash mismatch = E320.

## B014 tensor descriptors

`arrays`, `array_shapes`, `array_dtypes`, and loaded payload tensor names form an exact bijection. `dtype` is explicitly state-signature dtype and must equal `array_dtypes.state_signatures`. Readers recompute tensor names/shapes/dtypes plus sequence statistics from payload bytes before analysis.

REFERENCE_VALIDATORS=`src/cspm/contracts_reference.py`
BOOTSTRAP_BEHAVIOR=`src/cspm/bootstrap.py` re-exports reference validators; no divergent trace validator allowed.

MERGE_GATE=S6 G0_PASS_RECOMMENDED on current head + green Windows/Ubuntu CI
NO_SELF_MERGE=true
