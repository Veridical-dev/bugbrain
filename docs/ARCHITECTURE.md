# Architecture

BugBrain separates the joke from the evidence.

```text
                         ┌──────────────────────────────┐
                         │ FlyWire v783 directed graph │
                         └──────────────┬───────────────┘
                                        │ select 512-neuron core
PR diff ─► code units ─► signed hashing ├─► biological reservoir ─┐
                                        ├─► shuffled weights     ├─► ridge readout
                                        ├─► random topology      ┘        │
                                        └─► semantic control              ▼
                                                                  file-atomic bundles
                                                                         │
                                                                         ▼
                                                               independent agent sessions
                                                               full read-only repository
                                                                         │
                                                                         ▼
                                                                   JSON findings
```

## Modules

| Module | Responsibility |
|---|---|
| `connectome.py` | Downloaded-table parsing, thresholding, compact CSR cache, sparse propagation |
| `diff_parser.py` | Deterministic conversion of unified diffs or frozen JSON into code units |
| `reservoir.py` | Feature hashing, graph controls, echo-state recurrence, ridge readout, paired benchmarks |
| `router.py` | Legacy sparse-walk routing retained for comparison and small demos |
| `standalone.py` | Direct Codex CLI sessions, repository receipts, response parsing, competitor scoring |
| `cli.py` | Public command-line surface |

## Trust boundaries

### Truth never enters routing

Truth evidence is accepted only as an optional post-routing colocation grade. The feature encoder, reservoir state, readout, and clustering do not read it. Tests enforce this separation.

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

## Security model

Agentic review uses `codex exec --sandbox read-only`. BugBrain does not grant write access to the reviewed checkout and does not post findings to a forge. Treat every repository as untrusted input anyway: review prompts, source comments, filenames, and generated outputs may be adversarial. Run experiments in disposable checkouts and inspect findings before using them.
