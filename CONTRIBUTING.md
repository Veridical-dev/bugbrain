# Contributing to BugBrain

Thank you for helping a fruit fly survive peer review.

## Development setup

```bash
git clone https://github.com/Veridical-dev/bugbrain.git
cd bugbrain
uv sync
uv run python -m unittest discover -s tests -v
```

Keep changes focused, add tests for behavioral changes, and run the full suite before opening a pull request.

## Scientific integrity rules

Changes to experiments must preserve the parts that make the joke defensible:

- preregister the corpus, seed range, model, budget, and grading rubric;
- keep defect truth out of features, training, routing, and reviewer prompts;
- split learned-policy trajectories by repository or task family, never by step;
- derive policy rewards from independent verifiers, not model self-reports;
- include matched topology and weight controls for biological claims;
- persist failed calls and null results;
- count duplicate findings against precision but only once toward root recall;
- validate findings against the exact source head;
- label exploratory results as exploratory;
- never turn one favorable pull request into a general product claim.

If a contribution changes an experimental contract, document the old and new contract rather than overwriting history.

## Good contribution areas

- competitor-blind, preregistered public PR corpora;
- stronger graph nulls and statistical analysis;
- alternative biologically motivated circuit selections;
- reviewer-budget-matched baselines;
- interactive routing visualizations;
- platform-neutral agent backends;
- real, permission-safe Codex trajectory datasets with hidden verifiers;
- documentation that makes the project funnier without making the claims worse.

## Pull requests

Explain the motivation, describe the verification performed, and call out any result or artifact that becomes non-comparable. Small pull requests are easier for both humans and insects.

By contributing, you agree that your contribution is licensed under Apache-2.0 according to section 5 of the license.
