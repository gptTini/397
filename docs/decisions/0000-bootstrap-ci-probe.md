# Decision 0000 — Bootstrap Migration / G0 Review

STATUS=REWORK_PENDING_S6_REREVIEW
SESSION=S0
BASE_MAIN=a42c60c6adc98c8fc5b456c0d4e2fba4b8791a15
TARGET_PR=#5
SOFTWARE_REVIEWER=S6
SCIENTIFIC_REVIEWER=S7_NOT_REQUIRED_FOR_G0_SOFTWARE_BOOTSTRAP

PURPOSE=Adopt the feasibility-first plan while preserving deterministic EXP000 and establish contracts strong enough for Wave-1.

S6_INITIAL_DECISION=REWORK
S6_FINDINGS=B001,B002,B003,B004,B005,B006,B007

REWORK_SCOPE:
- align SOT TraceArtifact metadata with nested schema projection object
- make config/data hashes equal emitted canonical file bytes
- freeze cross-field TraceArtifact semantic validator rules
- separate root RunManifest and child TraceArtifact manifest authority
- define error-code precedence and actionable invalid fixtures
- remove stale S7 G0 role and enforce S6 pre-merge re-review
- align actual G0 CI policy with Windows+Ubuntu contract/bootstrap tests
- explicitly prohibit Phase-B preparation before GO_CSPM

PASS_SEQUENCE:
1. S0 rework on PR #5.
2. PR CI green on Ubuntu + Windows.
3. S6 independently re-reviews PR #5 and issue #1.
4. S6 emits G0_PASS_RECOMMENDED or returns further REWORK.
5. S0 does not self-merge; merge requires external reviewed action/process.
6. After merge, freeze new main SHA and create Wave-1 tasks/branches.

REMOTE_BRANCH_PROTECTION:
- desired but current connected GitHub capability cannot configure repository rulesets/status-check protection
- this limitation is explicit rather than silently treated as enforced
- process fallback until repository-setting capability exists: NO_SELF_MERGE + green PR CI + S6 review

REVIEW_TARGET=GitHub issue #1
