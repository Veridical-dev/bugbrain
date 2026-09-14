# Security policy

BugBrain is an experimental research tool, not a hardened production review service.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose credentials, execute untrusted code outside the declared sandbox, corrupt a checkout, or leak private repository content. Email `contact@veridical.dev` with:

- the affected version or commit;
- reproduction steps;
- expected impact;
- any suggested mitigation.

We will acknowledge useful reports as soon as practical and coordinate disclosure when a fix is available.

## Important operating assumptions

- Source repositories, diffs, filenames, and comments are untrusted input.
- Model output is untrusted and requires human review.
- The standalone runner requests a read-only Codex sandbox and verifies the Git checkout before and after execution.
- User-enabled Codex configuration or MCP servers can expand capabilities beyond BugBrain's own code. Review those permissions separately.
- Run experiments in disposable checkouts without secrets.
- BugBrain never posts comments, merges changes, or modifies the target repository by itself.

Only the latest release is supported for security fixes.
