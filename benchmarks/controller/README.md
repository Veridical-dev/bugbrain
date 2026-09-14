# Frozen controller benchmark

This directory is the public evidence bundle for BugBrain's trained-controller
experiment. It asks whether the biological FlyWire graph improves high-level
action selection for a repository-capable Codex worker.

## Contents

- `manifest.json`: the frozen base commit, split, mutation, goal, and verifier contract;
- `mutations/`: 16 deterministic regressions, each of which makes the complete verifier fail;
- `trajectories.json`: 16 real direct-Codex episodes with actual tool receipts and binary verifier rewards;
- `checkpoints/`: the validation-selected biological, shuffled-weight, degree-rewired, and observation-only policies.

The corpus has 8 training episodes, 2 validation episodes, and 6 held-out test
episodes. Its 153 actions were observed from `codex exec --json`; they were not
invented as a synthetic success path. Fourteen teacher runs passed the
independent verifier and two reward-zero failures were retained.

The mutation corpus deliberately stays within one small public Python project.
That makes every task cheap, inspectable, and independently testable, but it
does not establish cross-repository generalization. Raw trajectory receipts
include public source excerpts, command failures, and the original runner's
local temporary paths so the exact trained input remains checksum-reproducible.

## Frozen live protocol

Thirty paired seeds trained every arm. One checkpoint per arm was selected by
highest validation action accuracy, with the lowest seed breaking ties. Each
selected policy then received one fresh clone of every held-out mutation and a
12-action maximum. A fresh low-reasoning `gpt-5.6-luna` worker executed each
selected action with the complete repository available. Direct Codex received
the same repository, goal, model, reasoning effort, and verifier, but normal
single-call autonomy.

The independent verifier—not the agent's prose—assigned success. The compact
result and all per-task measurements are in
[`results/controller-benchmark.json`](../../results/controller-benchmark.json),
the immutable run receipts are in
[`results/controller-runs/`](../../results/controller-runs/),
and the complete 30-seed offline report is in
[`results/controller-policy-report.json`](../../results/controller-policy-report.json).

## Result

Biological topology did not improve offline held-out action imitation. In live
repair, biological BugBrain and direct Codex each passed 5/6 tasks, on different
subsets. Biological execution required 1.19× the input tokens and 2.39× the
worker time. The iterative scaffold changed a failure mode; the fly-specific
graph did not earn causal credit.

See [`docs/REPRODUCING.md`](../../docs/REPRODUCING.md) for commands.
