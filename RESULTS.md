# Results, without the lab-coat fog machine

## Bottom line

BugBrain produced one negative result and one intriguing case study:

1. The literal biological topology did not measurably improve routing over matched graph controls.
2. Once every fly-routed reviewer received full agentic repository access, the swarm found two verified defect roots on Formbricks #8588 that CodeRabbit did not report. CodeRabbit found one different verified root that BugBrain missed.

These statements answer different questions. Neither licenses a “fly brain beats code review” headline.

The trained-controller follow-up is now complete on a frozen real-repository
corpus. It produced another useful negative result: the controller sometimes
changes which task Codex solves, but biological topology did not earn causal
credit and the extra orchestration was less efficient.

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

## Trained-controller experiment

We recorded 16 real, independently verified Codex repair episodes against
deterministic mutations of BugBrain itself: 8 train, 2 validation, and 6
held-out test tasks. The corpus contains 153 actual tool actions, including 19
patches, 50 test actions, and 31 verification actions. Direct Codex passed 14
of the 16 source tasks; failed runs remain reward-zero examples.

The 2,048-neuron FlyWire core has 75,803 directed edges. Four policies received
identical observations and training settings over 30 paired seeds:

| Offline arm | Mean held-out action accuracy | Population SD |
|---|---:|---:|
| Biological FlyWire | 0.497175 | 0.041823 |
| Weight-shuffled | 0.475141 | 0.044946 |
| Degree-preserving rewired | **0.519774** | 0.036961 |
| Observation-only | 0.491525 | 0.000000 |

The biological-minus-rewired macro delta was −0.0225 with a paired task/seed
bootstrap 95% interval of [−0.0468, +0.0017]. Biological-minus-observation-only
was +0.0145 with [−0.0892, +0.0996]. The literal FlyWire topology did not
improve imitation of held-out actions.

For the operational test, we selected each arm's checkpoint by validation
accuracy only, then gave its worker a complete disposable checkout and up to 12
high-level actions on each held-out mutation. An independent 30-test command
assigned binary reward. The direct baseline used the same Codex model and full
repository access with normal autonomy.

| Live arm | Verifiers passed | Mean input tokens | Mean worker time | Action compliance |
|---|---:|---:|---:|---:|
| Biological FlyWire | **5/6** | 289,925 | 154.2 s | 84.6% |
| Weight-shuffled | 4/6 | 300,694 | 149.0 s | 85.2% |
| Degree-preserving rewired | 4/6 | 324,568 | 152.5 s | 85.5% |
| Observation-only | 3/6 | 308,777 | 151.6 s | 90.9% |
| Direct Codex | **5/6** | **242,889** | **64.6 s** | n/a |

Biological BugBrain and direct Codex succeeded on different five-task subsets:
the controller rescued `terminal-message`, which direct missed, while direct
solved `target-shuffle-copy`, which every controller missed. Their paired pass
delta is exactly zero (one win, one loss, four ties; two-sided exact sign
test p=1.0). Biological versus each graph null had one win and no losses, but
with six tasks the two-sided exact p-value is also 1.0. The biological arm used
1.19× the input tokens, 1.71× the output tokens, and 2.39× the worker time of
direct execution.

The worker obeyed the selected action contract on 84.6% of biological steps.
Repository snapshots confirmed zero mutations outside a selected `patch` step;
v0.3 additionally enforces read-only sandboxes for every non-patch action. This
means the useful artifact is a measurable iterative-agent scaffold, not evidence
that a fly learned software engineering.

The exact corpus, selected checkpoints, mutations, 30-seed policy report,
per-task scores, confidence intervals, and receipts are under
[`benchmarks/controller/`](benchmarks/controller/) and
[`results/controller-benchmark.json`](results/controller-benchmark.json).

## The invalid run we do not count

An earlier prompt-only precursor placed agents in an empty workspace and restricted findings to routed excerpts. It generated two source-refuted findings and made CodeRabbit appear to win 1–0. That was not the intended agentic reviewer, so it is superseded—not quietly deleted from the research history.

## Claim boundary

The truth register predates the agentic BugBrain run but was informed by historical audit work and includes CodeRabbit's author-confirmed root. The comparison is neither model-matched nor budget-matched: CodeRabbit's internal model, tools, prompt, cost, and latency are unknown, while BugBrain used eight `gpt-5.6-sol` low-reasoning sessions. One PR cannot establish product superiority.

Machine-readable numbers and finding grades live in [`results/`](results/).
