# Reproducing the experiments

## 1. Install

```bash
git clone https://github.com/Veridical-dev/bugbrain.git
cd bugbrain
uv sync
uv run python -m unittest discover -s tests -v
```

Equivalent pip setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 2. Fetch the connectome

```bash
uv run bugbrain download
uv run bugbrain build
```

The build retains edges with `syn_count >= 5` and writes a compact cache under `cache/`. Both inputs are checksum-verified. Expect roughly 179 MB of downloads and a 23 MB cache.

## 3. Smoke-test the reservoir

```bash
uv run bugbrain reservoir-poc \
  --corpus examples/corpus.json \
  --holdout tiny_retry_change \
  --out out/tiny-fly-brain.json
```

This checks mechanics only. The included two-case corpus is much too small for inference.

## 4. Rebuild the four-project public corpus

The original benchmark consists of public diffs from `redhat-developer/mapt`, `formbricks/formbricks`, `grafana/grafana`, and `openclaw/openclaw`. We do not vendor their source patches. Recreate them from the pinned public commits:

```bash
uv run python scripts/fetch_benchmark.py --out-dir benchmarks
```

Then run 50 consecutive seeds starting at 1714:

```bash
uv run bugbrain reservoir-benchmark \
  --corpus benchmarks/corpus.json \
  --seeds 50 \
  --out out/reservoir-50-seeds.json
```

For one held-out routing artifact:

```bash
uv run bugbrain reservoir-poc \
  --corpus benchmarks/corpus.json \
  --holdout formbricks_8588_interaction_parity \
  --seed 1714 \
  --bundles 8 \
  --out out/formbricks-routing.json
```

## 5. Run the agentic Formbricks review

Create a disposable exact-head checkout with both commits available:

```bash
git clone https://github.com/formbricks/formbricks.git /tmp/bugbrain-formbricks
git -C /tmp/bugbrain-formbricks checkout --detach 6a2646be5dd22b81726a05ab5e756db2b5c2dbf9
git -C /tmp/bugbrain-formbricks cat-file -e 0ed9568c5c5ddbba116387bba85599e216fb9e73^{commit}
```

Run eight independent reviewers:

```bash
uv run bugbrain standalone-review \
  --routing out/formbricks-routing.json \
  --arm biological \
  --out-dir out/formbricks-agentic \
  --repo /tmp/bugbrain-formbricks \
  --base 0ed9568c5c5ddbba116387bba85599e216fb9e73 \
  --head 6a2646be5dd22b81726a05ab5e756db2b5c2dbf9 \
  --model gpt-5.6-sol \
  --reasoning-effort low \
  --use-user-config
```

Model availability changes over time. If that model is unavailable, record the replacement and treat the run as a replication rather than an exact replay.

The published grading artifacts can be checked without making model calls:

```bash
uv run bugbrain standalone-compare \
  --review results/formbricks-8588-agentic-review.json \
  --truth-register results/formbricks-8588-truth-register.json \
  --competitor-evidence results/formbricks-8588-coderabbit-evidence.json \
  --grades results/formbricks-8588-grades.json \
  --out out/formbricks-8588-comparison.json
```

## 6. Interpretation rules

- Predeclare the corpus, seed range, controls, model, reasoning effort, reviewer count, and grading rubric.
- Count repeated reports against precision and only once toward root recall.
- Validate findings against the exact source head.
- Never use truth descriptions in routing or prompts.
- Do not compare product-level scores as though model and token budgets were matched when they were not.
- Keep failed and null-result runs. That is where most of the science lives.

## 7. Reproduce the trained-controller mechanics

The included trajectory corpus is synthetic. Use it to verify training,
checkpointing, inference, and null-arm bookkeeping—not to make a performance
claim:

```bash
uv run bugbrain build \
  --connections tests/fixtures/connections.csv \
  --annotations tests/fixtures/annotations.tsv \
  --cache /tmp/bugbrain-policy-mini.csr
uv run bugbrain policy-train \
  --trajectories examples/trajectories.json \
  --cache /tmp/bugbrain-policy-mini.csr \
  --annotations tests/fixtures/annotations.tsv \
  --neurons 0 --features 48 --readout-width 12 --epochs 40 --seeds 3 \
  --checkpoint-dir /tmp/bugbrain-policy-checkpoints \
  --out /tmp/bugbrain-policy-report.json
```

For a real experiment, capture complete `codex exec --json` runs and import
them one episode at a time:

```bash
uv run bugbrain trace-import \
  --jsonl trace.jsonl --key project-task-001 --split train \
  --goal "Review the exact base-to-head change" --reward 1 \
  --out trajectories.json --append
```

Do not use `--append` for the first episode. The dataset must contain at least
one train and one test episode before training. Keep each repository or closely
related task family in only one split. See [Training BugBrain](TRAINING.md) for
the action contract, reward rules, checkpoint inference, and matched-control
design.
