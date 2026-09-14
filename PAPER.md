# Can a Fruit Fly Review Code?

## A controlled experiment in connectome-routed agentic code review

**Veridical Research · September 2026**

> A fruit fly entered a pull request. It found two bugs, repeated one of them three times, and refused to elaborate on whether this was consciousness.

## Abstract

We investigate whether the directed, weighted wiring diagram of an adult *Drosophila melanogaster* brain can provide a useful inductive bias for routing code-review work. We encode code hunks into a fixed FlyWire-derived reservoir, train a small ridge readout on other repositories, cluster files into reviewer assignments, and send those assignments to independent agentic code reviewers with full read-only repository access.

The experiment has two results. First, across 200 paired leave-one-project-out trials, the biological topology does not measurably outperform matched random-topology, weight-shuffled, or code-only controls. Second, in a same-commit case study on Formbricks pull request #8588, eight biological-arm reviewer agents identify two registered correctness roots missed by CodeRabbit; CodeRabbit identifies a different registered root missed by the swarm. The case study is intriguing but is neither model-matched nor budget-matched and cannot establish general superiority.

Our conclusion is less cinematic than “flies can code”: connectome routing is currently an entertaining, falsifiable diversity mechanism whose biological specificity remains unsupported. Publishing the controls, negative result, and failed precursor is the point.

## 1. The question

Modern code review benefits from diverse passes: one reviewer follows state transitions, another traces trust boundaries, another looks for error recovery, and so on. Most systems create that diversity with prompts, randomness, or hand-authored orchestration.

We asked a needlessly literal question:

> Could the topology of a real brain act as a fixed routing prior for a swarm of code-review agents?

This is not a claim that a static connectome simulates cognition. A connectome is a structural graph, not a behaving brain. In BugBrain, it never decides whether code is correct. It influences which code neighborhoods and review lenses are assigned together; language-model agents perform the actual repository analysis.

## 2. Data

The source graph derives from FlyWire FAFB version 783. The published reconstruction contains 139,255 proofread neurons and roughly 54.5 million synapses between those neurons [1]. BugBrain downloads the processed directed graph and neuron annotations deposited by Correig-Fraga, Guimerà, and Sales-Pardo [2], then retains edges with at least five synapses.

The resulting local cache contains:

- 139,255 neurons;
- 2,710,038 directed neuron-pair edges;
- 31,578,726 represented synapses;
- annotation-derived flow, class, type, and neurotransmitter metadata.

The large source files are not vendored. Checksums and provenance are pinned in the downloader.

## 3. Method

### 3.1 Code representation

Each changed hunk becomes a code unit with its file path, hunk heading, lexical tokens, and diff excerpt. A 96-dimensional signed hashing encoder is fit with inverse-document-frequency statistics from the training repositories only. Ground-truth defects are never exposed to feature construction, reservoir training, or routing.

### 3.2 The fly reservoir

The biological core contains the 512 neurons with highest total directed edge degree and the 6,421 directed edges among them. Raw synapse counts become recurrent weights in target-by-source orientation.

For each code unit, the reservoir applies the leaky recurrence

```text
x(t+1) = (1 − leak)x(t) + leak·tanh(Wx(t) + Win·u + bias)
```

with leak `0.9`, input scale `0.7`, three independent recurrent passes, and spectral radius `0.99`. Each hunk begins at zero state, preventing order leakage between samples. The formulation is an adaptation of standard echo-state-network mechanics and the public ESA `fly_connectome` reference implementation [3].

A ridge readout trained on other repositories predicts review-lens targets derived from deterministic code-risk heuristics. These labels test representation behavior; they are not bug truth. Files remain atomic during clustering. Reservoir similarity contributes 25% of the final routing similarity and code-native similarity contributes 75%.

### 3.3 Controls

The central causal question is not whether a 512-dimensional transformation helps. It is whether **this biological topology** helps. The graph controls therefore share:

- neuron count;
- directed edge count;
- raw nonzero weight multiset;
- self-loop count;
- input projection;
- readout dimension;
- hyperparameters;
- target spectral radius.

The `weight_shuffled` arm preserves topology while permuting nonzero weights. The `random_topology` arm replaces directed topology while preserving the weight multiset and self-loop count. A smaller `semantic` arm provides a code-only ridge baseline and is reported separately because it is not parameter-matched.

### 3.4 Agentic review

Routing produces eight focus bundles with learned review lenses. Each bundle launches a fresh Codex CLI session in a clean, read-only checkout of the exact pull-request head. Every agent can:

- inspect the complete base-to-head Git diff;
- read and search every repository file;
- follow declarations, callers, tests, and neighboring changes;
- run read-only shell commands;
- report a finding in any changed PR file.

The focus bundle is not an information boundary. Repository receipts verify the base, head, changed-file list, and clean tracked state before and after the run. Agent responses are persisted once under prompt/model/repository contracts. No finding is rerun selectively.

## 4. Experiment A: does the biological wiring matter?

We ran 50 consecutive random seeds over four leave-one-project-out folds, yielding 200 paired held-out trials.

| Arm | Mean heuristic micro-F1 | Population SD |
|---|---:|---:|
| Biological FlyWire | 0.376117 | 0.105737 |
| Random topology | 0.374884 | 0.103651 |
| Weight-shuffled | 0.371842 | 0.101364 |
| Code-only semantic | 0.378743 | 0.107016 |

| Paired delta | Mean | Seed-block bootstrap 95% interval |
|---|---:|---:|
| Biological − random topology | +0.001234 | [−0.006679, +0.009133] |
| Biological − weight-shuffled | +0.004276 | [−0.003432, +0.012000] |
| Biological − code-only | −0.002626 | [−0.010447, +0.005011] |

Every interval crosses zero. On this corpus and objective, there is no measurable biological-topology advantage.

This is the scientifically important result. Without the matched graph nulls, almost any interesting-looking routing could be misattributed to “the fly brain” rather than dimensional expansion, random projection, or ordinary code similarity.

## 5. Experiment B: BugBrain versus CodeRabbit

We compared the biological routing arm with the complete saved set of CodeRabbit's substantive initial comments on [Formbricks PR #8588](https://github.com/formbricks/formbricks/pull/8588). Both sides are bound to head `6a2646be5dd22b81726a05ab5e756db2b5c2dbf9`. Seven defect roots were registered before the BugBrain run, although the register was informed by historical audit evidence and is not competitor-blind.

| System | Comments | Credited comments | Unique roots | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| BugBrain biological arm | 8 | 4 | 2/7 | 0.500 | 0.286 | 0.364 |
| CodeRabbit | 2 | 1 | 1/7 | 0.500 | 0.143 | 0.222 |

Three BugBrain circuits independently found a mixed-AND/OR precedence mismatch between an in-memory evaluator and the existing Prisma path. A fourth found that client-side exclusion of circular survey references was not enforced by the server-side write path.

CodeRabbit found a per-segment exception-isolation failure that could discard memberships already computed for other segments. BugBrain missed it. CodeRabbit's other comment was useful maintainability advice without a demonstrated present behavioral defect.

There was no overlap among credited roots. The systems were complementary. Repeated BugBrain reports count against precision but do not increase recall.

## 6. The scrambled-brain chapter

The first prototype used an arbitrary hash-to-neuron diffusion and weak controls. It was not a faithful reservoir implementation. We replaced it.

The first direct competitor run made a second mistake: it placed reviewers in an empty directory and treated routed excerpts as their entire world. Both reported findings were refuted by repository context that the agents could not inspect. We also replaced that run.

These failures are not omitted because they are the reusable lesson:

> A weird mechanism does not excuse an ordinary evaluation bug.

If a reviewer is supposed to be agentic, give it the repository. If a biological graph is supposed to matter, compare it with graph nulls that preserve everything except the biological property under test.

## 7. Limitations

This work is a small research prototype.

- Four projects are insufficient for broad conclusions.
- The mechanism labels are heuristics, not validated defect truth.
- The direct comparison covers one pull request.
- BugBrain used eight reviewer sessions; CodeRabbit produced two substantive comments. Budgets are not matched.
- CodeRabbit's internal models, prompts, tools, context construction, latency, and cost are unknown.
- The truth register predates the agentic run but incorporates historical audit evidence, including CodeRabbit's confirmed root.
- The selected 512-neuron core and recurrence are engineering choices, not claims of biological fidelity.
- A static connectome omits neural dynamics, plasticity, embodiment, neuromodulation, and the inconvenient fact that the fly did not consent to reviewing TypeScript.

## 8. What would change our mind?

A positive biological-topology claim should require a larger preregistered corpus, repeated end-to-end trials, exact-source validation, and a biological arm that consistently beats both matched graph nulls—not merely a code-only baseline.

Promising follow-ups include:

1. larger, competitor-blind truth sets;
2. randomized reviewer budgets and blinded manual grading;
3. alternative biologically motivated neuron selections;
4. sparse local circuit routing instead of one global core;
5. learned but constrained input mappings;
6. ablations of recurrence, lens assignment, file atomicity, and swarm size;
7. comparison against ordinary randomized agent swarms with identical prompts and cost.

## 9. Conclusion

The fruit-fly connectome did not beat randomness as a routing representation. The fly-routed agent swarm nevertheless found two verified defects missed by another review tool on one PR. Those facts can coexist.

The defensible public line is:

> We gave a fruit-fly-brain-routed swarm full agentic access and put it against CodeRabbit on the same PR. The fly found two verified defects the rabbit missed; the rabbit found one the fly missed.

The more important line is:

> We ran the controls, and the literal fly wiring has not earned the credit—yet.

## References

1. Dorkenwald, S. et al. “Neuronal wiring diagram of an adult brain.” *Nature* 634, 124–138 (2024). [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)
2. Correig-Fraga, E., Guimerà, R. & Sales-Pardo, M. “Data and source data for ‘Structure alone supports efficient visual computation in the Drosophila visual system’.” Zenodo, version 1.0.0 (2026). [doi:10.5281/zenodo.21549559](https://doi.org/10.5281/zenodo.21549559)
3. European Space Agency. `fly_connectome`, reference commit `90679d672cd25905758b3abc6674fa6a28525d52`. [GitLab](https://gitlab.com/EuropeanSpaceAgency/fly_connectome/-/tree/90679d672cd25905758b3abc6674fa6a28525d52)
4. FlyWire Consortium. “FlyWire Whole-brain Connectome Connectivity Data v783.” Zenodo (2024). [doi:10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)
5. Schlegel, P. et al. “Whole-brain annotation and multi-connectome cell typing of *Drosophila*.” *Nature* 634, 139–152 (2024). [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)
