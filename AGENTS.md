# CSPM397 Session Entry Contract

This repository is a multi-session research project.

## Mandatory startup sequence

Every session MUST do this before editing code:

1. Read `CSPM_SOT.yaml` completely.
2. Read the latest `main` SHA.
3. Confirm its assigned session ID and owned paths in `CSPM_SOT.yaml`.
4. Refuse to edit paths owned by another session unless S0 first changes the SOT.
5. Check the gate required for the task. A LOCKED gate means implementation may be prepared only when explicitly allowed, but no scientific success claim may be made.
6. Use GitHub `main` + committed manifests as source of truth. Do not rely on chat history as project state.

## Required worker finish sequence

1. Run the task's tests.
2. Preserve raw outputs; do not overwrite previous run artifacts.
3. Record code/config/data/artifact identity.
4. Commit only owned files.
5. Open a PR to `main`.
6. Do NOT self-merge.
7. Emit `RAW-HANDOFF v1` using the exact fields in `CSPM_SOT.yaml`.

## Scientific constraints

- No result without a declared baseline.
- No speed claim from FLOPs alone; report real wall-clock metrics when the gate requires speed.
- No success claim from latent cosine/MSE alone; behavioral preservation is required.
- No silent fallback. Every fallback must be observable in runtime events.
- No post-hoc relaxation of a scientific gate after seeing results.
- S7 independently reproduces and audits claims.

## Raw mode

This project uses compact raw handoffs by default. Do not spend time generating charts, HTML, polished reports, or duplicate prose summaries unless S0 explicitly requests them.

## Current bootstrap command

Until replaced by a later SOT revision:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python experiments/exp000_bootstrap/run.py --out runs/exp000/bootstrap
```

The second command writes raw runtime artifacts under `runs/`, which is not committed.
