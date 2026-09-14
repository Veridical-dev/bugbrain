# Results, without the lab-coat fog machine

## Bottom line

BugBrain produced one negative result and one intriguing case study:

1. The literal biological topology did not measurably improve routing over matched graph controls.
2. Once every fly-routed reviewer received full agentic repository access, the swarm found two verified defect roots on Formbricks #8588 that CodeRabbit did not report. CodeRabbit found one different verified root that BugBrain missed.

These statements answer different questions. Neither licenses a “fly brain beats code review” headline.

## Mechanism experiment

Fifty consecutive seeds were run across four leave-one-project-out folds: 200 paired held-out trials.

| Arm | Mean heuristic micro-F1 | Population SD |
|---|---:|---:|
| Biological FlyWire | 0.376117 | 0.105737 |
| Random topology | 0.374884 | 0.103651 |
| Weight-shuffled | 0.371842 | 0.101364 |
| Code-only semantic | 0.378743 | 0.107016 |

Paired biological deltas:

| Comparison | Mean delta | Seed-block bootstrap 95% interval |
|---|---:|---:|
| Biological − random topology | +0.001234 | [−0.006679, +0.009133] |
| Biological − weight-shuffled | +0.004276 | [−0.003432, +0.012000] |
| Biological − code-only | −0.002626 | [−0.010447, +0.005011] |

Every interval crosses zero. The fly-specific wiring advantage is indistinguishable from noise on this corpus.

## Agentic same-commit case study

Target: [formbricks/formbricks PR #8588](https://github.com/formbricks/formbricks/pull/8588), base `0ed9568c5c5ddbba116387bba85599e216fb9e73`, head `6a2646be5dd22b81726a05ab5e756db2b5c2dbf9`.

| System | Raw comments | Credited comments | Unique registered roots | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| BugBrain, 8 agent circuits | 8 | 4 | 2/7 | 0.500 | 0.286 | 0.364 |
| CodeRabbit | 2 | 1 | 1/7 | 0.500 | 0.143 | 0.222 |

BugBrain found:

- A mixed-AND/OR precedence mismatch between the new in-memory evaluator and the Prisma path. Three circuits independently reported this one root.
- Missing server-side enforcement of the UI's circular survey-reference exclusion.

CodeRabbit found:

- A per-segment exception-isolation failure that could erase memberships computed for earlier segments.

There was no overlap. Duplicates count against BugBrain's precision and never increase unique-root recall.

## The invalid run we do not count

An earlier prompt-only precursor placed agents in an empty workspace and restricted findings to routed excerpts. It generated two source-refuted findings and made CodeRabbit appear to win 1–0. That was not the intended agentic reviewer, so it is superseded—not quietly deleted from the research history.

## Claim boundary

The truth register predates the agentic BugBrain run but was informed by historical audit work and includes CodeRabbit's author-confirmed root. The comparison is neither model-matched nor budget-matched: CodeRabbit's internal model, tools, prompt, cost, and latency are unknown, while BugBrain used eight `gpt-5.6-sol` low-reasoning sessions. One PR cannot establish product superiority.

Machine-readable numbers and finding grades live in [`results/`](results/).
