# Architecture

BugBrain separates the joke from the evidence.

```text
                         ┌──────────────────────────────┐
                         │ FlyWire v783 directed graph │
                         └──────────────┬───────────────┘
                                        │ select sparse core
repository observation ─► signed hashing├─► biological graph ──────┐
                                        ├─► shuffled weights      ├─► recurrent policy
                                        ├─► degree rewiring       ┘        │
                                        └─► observation-only null          ▼
                                                                next high-level action
                                                                         │
                                                                         ▼
                                                               repository-capable worker
```

The original file-routing reservoir remains available. The diagram shows the
trained-controller experiment, where the connectome chooses an action and
Codex supplies language reasoning, concrete paths/commands, and patches.

## Modules

| Module | Responsibility |
|---|---|
| `connectome.py` | Downloaded-table parsing, thresholding, compact CSR cache, sparse propagation |
| `diff_parser.py` | Deterministic conversion of unified diffs or frozen JSON into code units |
| `reservoir.py` | Feature hashing, graph controls, echo-state recurrence, ridge readout, paired benchmarks |
| `policy.py` | Trajectory schema, sparse recurrent policy, BPTT, graph nulls, checkpoints, Codex trace import |
| `controller.py` | Live policy/direct runners, bounded Codex workers, action receipts, repository snapshots, independent verifier reward |
| `router.py` | Legacy sparse-walk routing retained for comparison and small demos |
| `standalone.py` | Direct Codex CLI sessions, repository receipts, response parsing, competitor scoring |
| `cli.py` | Public command-line surface |

## Trust boundaries

### Truth never enters routing

Truth evidence is accepted only as an optional post-routing colocation grade. The feature encoder, reservoir state, readout, and clustering do not read it. Tests enforce this separation.

### An action target never enters its own observation

The trajectory importer maps completed Codex events to the small action
vocabulary. For step *t*, the encoded observation contains the goal and the
receipt from step *t − 1*. The command or tool call selected at step *t* is the
training target and is excluded from its input. Observation hashing statistics
are fit on training episodes only.

### A focus bundle is not a context boundary

The standalone runner prepends an agentic-access contract to each routed prompt. The agent starts inside the exact repository checkout and can inspect any source file. Findings are limited to files changed by the PR, but supporting evidence may come from anywhere in the repository.

### Repository receipts

Before launching reviewers, BugBrain verifies:

- the checkout's exact `HEAD`;
- existence of the declared base commit;
- a clean tracked working tree;
- the full changed-file list and its SHA-256 digest.

It repeats the receipt after the run and fails if anything changed.

### Persist-once calls

Each model response is stored with a contract containing the prompt digest, routing digest, model, reasoning effort, base, head, and configuration mode. A cached response is reused only when the complete contract matches. This avoids accidental cherry-picking through selective reruns.

## Biological and null arms

The biological core uses target-by-source orientation and raw retained synapse counts. Every graph arm is independently scaled to the same spectral radius. The two graph nulls preserve size and weight statistics while ablating either weights-on-topology or topology itself.

The code-only semantic baseline is intentionally smaller. It answers “does the reservoir expansion help?” only loosely; the matched graph arms answer the stronger topology question.

The trained policy uses a separate sparse null design. `weight_shuffled` keeps
the biological edges but permutes normalized weights. `degree_rewired` shuffles
target/weight pairs across fixed source slots, preserving every source
out-degree, target in-degree, node identity, and the complete weight multiset.
The biological and graph-null arms share feature mappings, action mappings,
initial parameters, optimizer settings, trajectories, and paired seeds.

Only flow-conditioned recurrent gains and the action decoder are learned. Edge
existence and direction remain fixed. Checkpoints contain plain NumPy arrays;
loading rebuilds the graph and checks its topology digest and neuron IDs.

## Security model

Agentic review uses `codex exec --sandbox read-only`. BugBrain does not grant write access to the reviewed checkout and does not post findings to a forge. Treat every repository as untrusted input anyway: review prompts, source comments, filenames, and generated outputs may be adversarial. Run experiments in disposable checkouts and inspect findings before using them.

Implementation trajectories require write access by definition. Record those
only in disposable worktrees with the narrowest useful sandbox, execute project
verifiers separately, and attach their result as reward. A model's claim that
tests passed is not a verifier.

In live implementation mode, only a policy-selected `patch` step receives a
workspace-write sandbox. Search, inspection, testing, reasoning, verification,
and stopping are process-enforced read-only. Every step records repository
snapshots; the frozen v0.3 benchmark observed zero changes outside `patch`
steps. Action-label compliance is measured separately because a shell command
can cross semantic categories even when it cannot write.
