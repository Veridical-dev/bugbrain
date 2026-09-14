# Training BugBrain

BugBrain's trained mode asks a narrower, testable question than “can a fly write
software?”:

> Does a fixed connectome topology provide a useful inductive bias for choosing
> the next action of a repository-capable coding agent?

Codex remains the language-and-tool worker. It can search the whole checkout,
inspect callers, run commands, edit in a disposable workspace, and explain its
result. BugBrain observes a compact history and chooses one high-level action:
`search`, `inspect`, `test`, `reason`, `patch`, `verify`, or `stop`.

## What is actually trained

The policy is a sparse recurrent network whose edges come from the directed,
weighted FlyWire graph. Each observation is encoded without a language-model
call and injected into annotated afferent neurons. Activity propagates over the
fixed graph and annotated efferent neurons are pooled into action logits.

Training uses full-episode backpropagation through time and reward-weighted
behavioral cloning from recorded Codex trajectories. It updates:

- three flow-conditioned self gains;
- three flow-conditioned message gains;
- one input gain and three flow biases;
- the small action decoder.

Neuron identities, graph edges, edge directions, input mapping, and output
mapping remain fixed. This is a trained connectome-constrained policy, not a
biophysical fly simulation and not a fine-tuned language model.

## Why this follows the game-controller idea

The useful pattern in [DOOMFLY](https://github.com/nftechie/doomfly) and the
[FlyGM locomotion model](https://arxiv.org/abs/2602.17997) is an interface:
environment observations enter sensory nodes, a connectome-constrained dynamic
system produces motor choices, and environment outcomes provide feedback.
BugBrain substitutes repository state for pixels, software actions for motor
commands, and tests or review grading for game reward.

The viral game videos are inspiration, not validation. A visible avatar moving
does not establish that biological topology helped. BugBrain therefore runs the
same trajectories, feature map, initialization, optimizer, and seeds against
explicit nulls.

## Record real Codex demonstrations

For review-only traces, run Codex in a clean checkout with read-only access:

```bash
codex exec --json --sandbox read-only -C /path/to/repo \
  "Review this change. Inspect the full repository and support every finding." \
  > trace.jsonl
```

For implementation traces, use a disposable worktree and an appropriate
workspace-write sandbox. Run the repository's verifier afterwards. Convert the
trace with its measured terminal reward; for a simple binary scheme use `1` for
a passing verifier and `0` for a failed one:

```bash
uv run bugbrain trace-import \
  --jsonl trace.jsonl \
  --key project-task-001 \
  --split train \
  --goal "Fix the retry race and preserve cancellation semantics" \
  --reward 1 \
  --out trajectories.json
```

Use `--append` for subsequent episodes. Split by repository or task family,
not by individual step, so near-identical observations cannot cross the train
and test boundary. Keep failed trajectories: negative or zero rewards are part
of the experiment.

`trace-import` records each action from `codex exec --json` but exposes only the
previous action result to the policy. The current command is the target and is
never copied into its own observation. The final reward is deliberately an
operator-supplied verifier outcome because Codex output is not ground truth.

Trace receipts retain up to 2,000 characters of the previous tool output and a
local source path. Treat raw and imported traces as potentially sensitive;
remove credentials, private source, customer data, and identifying paths before
sharing or committing a corpus.

## Train matched experimental arms

Fetch the public graph once, build its cache, then train:

```bash
uv run bugbrain download
uv run bugbrain build
uv run bugbrain policy-train \
  --trajectories trajectories.json \
  --neurons 2048 \
  --seeds 10 \
  --checkpoint-dir out/checkpoints \
  --out out/policy-report.json
```

`--neurons 0` retains every neuron in the cached graph; begin with 2,048 while
developing. Every seed trains four arms:

| Arm | Purpose |
|---|---|
| `biological` | FlyWire directions and normalized signed weights |
| `weight_shuffled` | Same topology and weight multiset, wrong weight placement |
| `degree_rewired` | Same node IDs, per-node in/out degree, and weight multiset, wrong topology |
| `observation_only` | No recurrent graph; checks whether observations already solve the task |

The directed rewiring is a configuration multigraph. It can change self loops
and create parallel edges; the report discloses this rather than silently
repairing the null.

## Load the learned controller

Pass the complete observation history so recurrent state is reconstructed:

```bash
uv run bugbrain policy-predict \
  --checkpoint out/checkpoints/biological-seed-2305.npz \
  --history examples/observation-history.json \
  --out out/next-action.json
```

The checkpoint contains no executable objects. Loading it rebuilds the graph
from the supplied cache and rejects mismatched neuron IDs or topology hashes.

## Run the repository-capable controller

`policy-run` reconstructs the recurrent state from the complete compact history,
asks the policy for one action, and launches a bounded Codex worker in the full
checkout. Codex retains normal repository search and reasoning capacity and
chooses all paths, commands, and patch content. Only a selected `patch` action
receives a workspace-write sandbox; all other actions are process-enforced
read-only. The independent verifier runs after the policy stops or exhausts its
budget.

```bash
uv run bugbrain policy-run \
  --checkpoint benchmarks/controller/checkpoints/biological-seed-2323.npz \
  --repo /tmp/disposable-bugbrain-task/repo \
  --goal "Fix the failing regression" \
  --mode implement --max-steps 12 \
  --verifier-json /tmp/disposable-bugbrain-task/verifier.json \
  --out-dir out/controller-run
```

Use `direct-run` on an independently prepared clone for the same-model baseline.
It gives Codex normal autonomy in a single call and can export its measured tool
trace as a training episode.

## Bundled smoke fixture

`examples/trajectories.json` is synthetic and exists only to test mechanics.
It is not evidence that biological topology works. A fast local smoke run uses
the eight-neuron test graph:

```bash
uv run bugbrain build \
  --connections tests/fixtures/connections.csv \
  --annotations tests/fixtures/annotations.tsv \
  --cache /tmp/bugbrain-mini.csr
uv run bugbrain policy-train \
  --trajectories examples/trajectories.json \
  --cache /tmp/bugbrain-mini.csr \
  --annotations tests/fixtures/annotations.tsv \
  --neurons 0 --features 48 --readout-width 12 --epochs 40 --seeds 3 \
  --checkpoint-dir /tmp/bugbrain-checkpoints \
  --out /tmp/bugbrain-policy-report.json
```

## Frozen real-repository result

The public controller corpus under `benchmarks/controller/` contains 16 real
Codex repair episodes and deterministic verifier-backed mutations: 8 train, 2
validation, and 6 test. It has 153 measured actions and a terminal reward sum of
14. Thirty paired seeds were trained without fitting features on holdouts.

Biological held-out imitation accuracy averaged 49.7%; the degree-preserving
rewiring averaged 52.0%. In validation-selected live rollouts, biological
BugBrain and direct Codex each passed 5/6 verifiers, on different task subsets.
The biological controller used 1.19× the input tokens and 2.39× the worker time.
This is a completed negative topology result and a promising iterative-agent
scaffold—not evidence that a fruit fly learned to program.

Held-out action accuracy remains an imitation metric, not code-review quality.
A broader positive claim still needs multiple repositories, hidden or external
verifiers, repeated live rollouts, a budget-matched hand-written scheduler, and
confidence intervals over biological-minus-control deltas. If the biological
arm does not beat those nulls, publish that too.
