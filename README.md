<p align="center">
  <img src="assets/bugbrain-logo.png" width="220" alt="BugBrain logo: a fruit fly whose compound eyes are neural graphs">
</p>

<h1 align="center">BugBrain</h1>

<p align="center"><strong>Peer review, but the peer has 139,255 neurons.</strong></p>

<p align="center">
  <a href="https://github.com/Veridical-dev/bugbrain/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Veridical-dev/bugbrain/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.21549559"><img alt="Data DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21549559.svg"></a>
</p>

BugBrain is an open experiment that routes agentic code reviewers using the wiring diagram of a real adult fruit-fly brain. Code changes become signals, signals pass through a fixed FlyWire connectome reservoir, and the resulting state helps assign files and review lenses to independent Codex agents.

The connectome does **not** understand TypeScript. It is routing metadata, not evidence. The agents still have to inspect the repository and justify every finding from code.

## The delightfully awkward result

We ran two different experiments, and honesty requires keeping both:

| Question | Result |
|---|---|
| Does biological fly wiring beat carefully matched random reservoirs at routing code? | **No measurable advantage.** All paired bootstrap intervals crossed zero. |
| Can a fly-routed agent swarm find real bugs when every agent gets full repository access? | On one Formbricks PR, **BugBrain found 2 verified defect roots CodeRabbit missed; CodeRabbit found 1 BugBrain missed.** |

That is not proof that flies are better code reviewers. It is proof that weird experiments become useful when they have controls, receipts, and the courage to publish the embarrassing part.

Read the full [paper](PAPER.md), the compact [results](RESULTS.md), and the [architecture](docs/ARCHITECTURE.md).

## What happens under the hood

```text
PR diff ──► code features ──► fruit-fly reservoir ──► focus bundles + review lenses
                                                              │
                                                              ▼
                                  independent Codex agents with full read-only repo access
                                                              │
                                                              ▼
                                                evidence-backed review findings
```

The biological arm uses a 512-neuron, 6,421-edge core selected from the 139,255-neuron FlyWire FAFB v783 graph. An echo-state-network adaptation applies three independent recurrent passes and a ridge readout trained on other repositories. Matched controls preserve graph size, edge count, weight multiset, self-loop count, input projection, readout dimension, and target spectral radius.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/) or pip, Git, and enough curiosity to download about 179 MB of neuroscience data.

```bash
git clone https://github.com/Veridical-dev/bugbrain.git
cd bugbrain
uv sync
uv run bugbrain download
uv run bugbrain build
uv run bugbrain reservoir-poc \
  --corpus examples/corpus.json \
  --holdout tiny_retry_change \
  --out out/tiny-fly-brain.json
```

The bundled corpus is deliberately tiny: it verifies that the full pipeline works, not that the fly deserves tenure.

### Run an agentic review

The standalone runner calls the Codex CLI directly—there is no proprietary review engine in the loop. Build a reservoir routing artifact whose holdout is the target PR, then point BugBrain at a clean checkout of its exact head:

```bash
uv run bugbrain standalone-review \
  --routing out/my-pr-routing.json \
  --arm biological \
  --out-dir out/my-pr-review \
  --repo /path/to/clean/checkout \
  --base <exact-base-sha> \
  --head <exact-head-sha> \
  --model <your-codex-model> \
  --reasoning-effort low \
  --use-user-config
```

Each reviewer may search every repository file, inspect the complete base-to-head diff, follow definitions and callers, and run read-only shell commands. The connectome-selected bundle is a focus assignment—not a context prison. The runner refuses a dirty checkout, a mismatched head, or a changed repository receipt.

See [Reproducing the experiments](docs/REPRODUCING.md) for the full protocol.

## Things BugBrain is not

- A production-ready GitHub bot
- A claim that neural anatomy magically understands source code
- A benchmark proving superiority over CodeRabbit, Codex, or anyone else
- Affiliated with or endorsed by the FlyWire Consortium, the cited researchers, Formbricks, CodeRabbit, or ESA
- Medical advice for flies

## Why open-source a negative result?

Because “we tried something strange and measured it properly” is more interesting than another architecture diagram with no falsifiable claim. The controls are the product here. If someone finds a representation, task, or learning rule where the biological graph consistently beats its nulls, we want to know.

Good first contributions include new preregistered PR corpora, stronger graph nulls, alternative neuron selections, routing visualizations, and attempts to reproduce or falsify the current result. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Data, attribution, and license

BugBrain code and original project assets are Apache-2.0 licensed. FlyWire-derived data is downloaded separately and remains under its source license. See [NOTICE](NOTICE) and [data/README.md](data/README.md) before redistributing derived data.

Built by [Veridical](https://veridical.dev/) with respect for fruit flies and mild concern for software engineering.
