from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_review_response(
    response: str,
    *,
    reviewer_id: str,
    valid_files: set[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Parse the deliberately small standalone reviewer contract."""
    candidate = _FENCE.sub("", response.strip())
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, stop = candidate.find("["), candidate.rfind("]")
        if start < 0 or stop < start:
            return [], "response did not contain a JSON array"
        try:
            payload = json.loads(candidate[start : stop + 1])
        except json.JSONDecodeError as exc:
            return [], f"invalid JSON array: {exc.msg}"
    if not isinstance(payload, list):
        return [], "response root was not an array"

    findings = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            return [], f"finding {index + 1} was not an object"
        path = str(row.get("file") or "").strip()
        comment = str(row.get("comment") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        if path not in valid_files:
            return [], f"finding {index + 1} cited a file outside the changed PR files"
        if not comment or not evidence:
            return [], f"finding {index + 1} lacked comment or evidence"
        try:
            line = int(row.get("line") or 1)
        except (TypeError, ValueError):
            return [], f"finding {index + 1} had an invalid line"
        findings.append(
            {
                "reviewer_id": reviewer_id,
                "file": path,
                "line": max(1, line),
                "severity": str(row.get("severity") or "medium").lower(),
                "comment": comment,
                "evidence": evidence,
            }
        )
    return findings, None


def _run_process(command: Sequence[str], prompt: str, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise TimeoutError(f"review call exceeded {timeout} seconds")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def call_codex(
    prompt: str,
    *,
    cli: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    repository: Path,
    use_user_config: bool,
) -> tuple[str, dict[str, Any]]:
    """One direct, agentic Codex CLI call with no external review engine."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="flywire-standalone-output-") as output_dir:
        output = Path(output_dir) / "last-message.txt"
        command = [
            cli,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "-C",
            str(repository),
            "-o",
            str(output),
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if not use_user_config:
            command.append("--ignore-user-config")
        command.append("-")
        completed = _run_process(command, prompt, str(repository), timeout)
        elapsed = round(time.monotonic() - started, 3)
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "").strip().splitlines()
            tail = diagnostic[-1][:240] if diagnostic else "no diagnostic"
            raise RuntimeError(f"Codex CLI exited {completed.returncode}: {tail}")
        if not output.exists():
            raise RuntimeError("Codex CLI did not write its final-message file")
        response = output.read_text(encoding="utf-8", errors="replace")
        if not response.strip():
            raise RuntimeError("Codex CLI returned an empty final message")
        return response, {
            "status": "ok",
            "wall_seconds": elapsed,
            "returncode": completed.returncode,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "repository": str(repository),
            "sandbox": "read-only",
            "user_config_enabled": use_user_config,
        }


def repository_receipt(repository: Path, *, base: str, head: str) -> dict[str, Any]:
    repository = repository.resolve()
    if not repository.is_dir():
        raise ValueError(f"Repository checkout does not exist: {repository}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            raise ValueError(f"Git checkout validation failed: {detail[-1] if detail else arguments}")
        return completed.stdout.strip()

    actual_head = git("rev-parse", "HEAD")
    if actual_head != head:
        raise ValueError(f"Checkout HEAD is {actual_head}, expected {head}")
    git("cat-file", "-e", f"{base}^{{commit}}")
    status = git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise ValueError("Repository checkout has tracked working-tree changes")
    diff_names = [
        row for row in git("diff", "--name-only", f"{base}..{head}").splitlines() if row
    ]
    return {
        "path": str(repository),
        "base": base,
        "head": head,
        "tracked_worktree_clean": True,
        "changed_files": len(diff_names),
        "changed_file_paths": diff_names,
        "changed_files_sha256": _sha256("\n".join(diff_names).encode()),
    }


def agentic_review_prompt(prompt: str, *, base: str, head: str) -> str:
    scoped = prompt.replace(
        "Review only the supplied code and return a JSON array",
        "Review the supplied changes and return a JSON array",
    )
    instructions = f"""AGENTIC SOURCE ACCESS
You are running inside a read-only checkout of the complete repository at exact PR head {head}.
The comparison base is {base}. The routed diff below is your focus, not a context boundary.

Before reporting a finding, use repository tools to inspect the complete file, relevant declarations,
all callers/consumers, tests, and other PR hunks. You may run `git diff {base}..{head}` and
read/search any repository file. Do not infer that a migration, field, guard, or test is absent merely because it
is absent from the routed excerpt. Report only regressions introduced between the stated base and head.
Do not modify the checkout. Your final answer must obey the JSON-array contract in the routed prompt.

ROUTED FOCUS
"""
    return instructions + scoped


def run_standalone_review(args) -> dict[str, Any]:
    routing_bytes = args.routing.read_bytes()
    routing = json.loads(routing_bytes)
    if args.arm not in routing.get("arms", {}):
        raise ValueError(f"Routing artifact has no {args.arm!r} arm")
    bundles = list(routing["arms"][args.arm]["bundles"])
    if args.max_reviewers:
        bundles = bundles[: args.max_reviewers]
    repo_receipt = repository_receipt(args.repo, base=args.base, head=args.head)

    records = []
    findings: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles, 1):
        prompt = agentic_review_prompt(
            str(bundle["prompt"]), base=args.base, head=args.head
        )
        contract = {
            "schema": "bugbrain.standalone-review-call/1",
            "routing_sha256": _sha256(routing_bytes),
            "arm": args.arm,
            "reviewer_id": bundle["reviewer_id"],
            "prompt_sha256": _sha256(prompt.encode()),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "repository_head": args.head,
            "repository_base": args.base,
            "agentic_repository_access": True,
            "user_config_enabled": args.use_user_config,
        }
        call_path = args.out_dir / "calls" / f"{bundle['reviewer_id']}.json"
        if call_path.exists():
            record = json.loads(call_path.read_text(encoding="utf-8"))
            if record.get("contract") != contract:
                raise ValueError(f"Persisted call contract mismatch at {call_path}")
            state = "cached"
        else:
            response, receipt = call_codex(
                prompt,
                cli=args.codex_cli,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout_seconds,
                repository=args.repo.resolve(),
                use_user_config=args.use_user_config,
            )
            record = {"contract": contract, "response": response, "receipt": receipt}
            _atomic_json(call_path, record)
            state = "ok"
        # The circuit bundle is a focus, not an information or reporting boundary. Agentic
        # reviewers may follow evidence anywhere in the repository and report on any PR file.
        valid_files = set(repo_receipt["changed_file_paths"])
        parsed, error = parse_review_response(
            str(record.get("response") or ""),
            reviewer_id=str(bundle["reviewer_id"]),
            valid_files=valid_files,
        )
        records.append(
            {
                "reviewer_id": bundle["reviewer_id"],
                "state": state,
                "parse_error": error,
                "finding_count": len(parsed),
                "receipt": record.get("receipt") or {},
            }
        )
        if error is None:
            findings.extend(parsed)
        print(f"{index}/{len(bundles)} {bundle['reviewer_id']}: {state}, findings={len(parsed)}", flush=True)

    # Only byte-level duplicate removal. There is no model ranker or semantic deduplication.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for finding in findings:
        key = (finding["file"], finding["line"], finding["comment"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append({"id": f"F{len(unique) + 1:03d}", **finding})

    post_repo_receipt = repository_receipt(args.repo, base=args.base, head=args.head)
    if post_repo_receipt != repo_receipt:
        raise RuntimeError("Repository receipt changed during the supposedly read-only review")
    summary = {
        "schema": "bugbrain.standalone-review/1",
        "status": "ungraded_shadow_findings",
        "contract": {
            "routing": str(args.routing),
            "routing_sha256": _sha256(routing_bytes),
            "holdout": routing.get("holdout"),
            "arm": args.arm,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "reviewers": len(bundles),
            "external_review_engine_used": False,
            "repository_checkout_exposed_to_model": True,
            "agentic_shell_and_file_tools_enabled": True,
            "repository_sandbox": "read-only",
            "user_config_enabled": args.use_user_config,
        },
        "repository_receipt": repo_receipt,
        "repository_receipt_after": post_repo_receipt,
        "calls": records,
        "findings": unique,
    }
    _atomic_json(args.out_dir / "review.json", summary)
    return summary


def _truth_roots(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case = next((row for row in payload.get("prs", []) if row.get("key") == key), None)
    if case is None:
        raise ValueError(f"Truth register has no case {key!r}")
    return [row for row in case.get("truth", []) if row.get("truth_register") == "pre"]


def _competitor_case(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case = next((row for row in payload.get("pull_requests", []) if row.get("key") == key), None)
    if case is None:
        raise ValueError(f"Competitor evidence has no case {key!r}")
    return case


def _score(ids: set[str], grades: list[dict[str, Any]], truth_ids: set[str]) -> dict[str, Any]:
    grade_ids = [str(row.get("id")) for row in grades]
    if len(grade_ids) != len(set(grade_ids)) or set(grade_ids) != ids:
        raise ValueError("Grades must cover every system finding exactly once")
    matched = []
    for row in grades:
        truth_id = row.get("truth_id")
        if truth_id is not None and truth_id not in truth_ids:
            raise ValueError(f"Grade cites unknown truth root: {truth_id!r}")
        if truth_id:
            matched.append(str(truth_id))
    roots = sorted(set(matched))
    precision = len(matched) / len(ids) if ids else 0.0
    recall = len(roots) / len(truth_ids) if truth_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "comments": len(ids),
        "credited_comments": len(matched),
        "credited_truth_roots": roots,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def compare_with_competitor(args) -> dict[str, Any]:
    review_bytes = args.review.read_bytes()
    review = json.loads(review_bytes)
    key = str(review.get("contract", {}).get("holdout") or "")
    truths = _truth_roots(args.truth_register, key)
    truth_ids = {str(row["truth_id"]) for row in truths}
    competitor = _competitor_case(args.competitor_evidence, key)
    grades = json.loads(args.grades.read_text(encoding="utf-8"))
    if grades.get("key") != key:
        raise ValueError("Grading artifact case differs from standalone review")
    if competitor.get("comparison_relation") != "same_commit":
        raise ValueError("Direct comparison requires the competitor to have reviewed the same commit")
    review_head = str(review.get("repository_receipt", {}).get("head") or "")
    competitor_head = str(competitor.get("reviewed_head") or "")
    grades_head = str(grades.get("reviewed_head") or "")
    if not review_head or review_head != competitor_head:
        raise ValueError("Standalone reviewer and competitor must have reviewed the exact same head")
    if grades_head != review_head:
        raise ValueError("Grading artifact must identify the exact reviewed head")

    bugbrain_ids = {str(row["id"]) for row in review.get("findings", [])}
    competitor_findings = competitor.get("competitor_findings", [])
    competitor_ids = {str(row["id"]) for row in competitor_findings}
    result = {
        "schema": "bugbrain.competitor-comparison/1",
        "status": "source_graded_same_commit_shadow_comparison",
        "case": key,
        "repo": competitor.get("repo"),
        "pull": competitor.get("pull"),
        "reviewed_head": review_head,
        "truth_roots": len(truth_ids),
        "bugbrain": _score(bugbrain_ids, list(grades.get("bugbrain") or []), truth_ids),
        "competitor": {
            "name": competitor.get("competitor"),
            **_score(competitor_ids, list(grades.get("competitor") or []), truth_ids),
        },
        "contract": {
            "standalone_review": str(args.review),
            "standalone_review_sha256": _sha256(review_bytes),
            "truth_register": str(args.truth_register),
            "competitor_evidence": str(args.competitor_evidence),
            "grades": str(args.grades),
            "external_review_engine_used": False,
            "grading_is_manual_exact_source": True,
            "budgets_are_not_model_matched": True,
        },
        "grades": grades,
        "competitor_findings": competitor_findings,
    }
    _atomic_json(args.out, result)
    return result
