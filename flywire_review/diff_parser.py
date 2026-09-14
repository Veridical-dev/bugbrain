from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .model import CodeUnit


_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@\s?(.*)$")
_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$-]{1,}")
_CAMEL_EDGE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STOP = {
    "const", "else", "false", "from", "function", "import", "none", "null",
    "return", "self", "true", "undefined", "var", "with", "this", "that",
}

_RISK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("security/trust-boundaries", ("auth", "token", "permission", "role", "admin", "secret", "session", "cookie", "csrf", "sql", "sanitize")),
    ("state/invariants", ("state", "cache", "transaction", "commit", "rollback", "mutable", "lifecycle", "status")),
    ("concurrency/ordering", ("async", "await", "lock", "mutex", "race", "thread", "queue", "atomic", "concurrent")),
    ("errors/recovery", ("error", "exception", "catch", "except", "retry", "timeout", "fallback", "abort")),
    ("api/contracts", ("api", "request", "response", "schema", "contract", "public", "route", "endpoint", "validate")),
    ("data/serialization", ("json", "serialize", "parse", "database", "query", "migration", "model", "dto")),
    ("tests/observability", ("test", "assert", "log", "metric", "trace", "telemetry", "mock")),
    ("dependencies/supply-chain", ("package", "lockfile", "dependency", "version", "docker", "workflow", "requirements")),
)


def _tokenize(text: str) -> tuple[str, ...]:
    tokens: set[str] = set()
    for raw in _TOKEN.findall(text):
        for piece in re.split(r"[_$\-]+", raw):
            for camel in _CAMEL_EDGE.split(piece):
                token = camel.lower()
                if len(token) >= 2 and token not in _STOP:
                    tokens.add(token)
    return tuple(sorted(tokens))


def _risk_tags(tokens: tuple[str, ...], path: str) -> tuple[str, ...]:
    haystack = set(tokens) | set(_tokenize(path))
    tags = [label for label, words in _RISK_PATTERNS if haystack.intersection(words)]
    return tuple(tags)


def _unit_key(path: str, line: int, heading: str, excerpt: str) -> str:
    digest = hashlib.blake2b(
        f"{path}\0{line}\0{heading}\0{excerpt}".encode("utf-8"), digest_size=6
    ).hexdigest()
    return f"{path}:{line}:{digest}"


def parse_unified_diff(text: str, *, max_excerpt_chars: int = 6_000) -> list[CodeUnit]:
    """Turn a unified diff into stable per-hunk routing units."""
    units: list[CodeUnit] = []
    path = "unknown"
    start_line = 0
    heading = ""
    hunk_lines: list[str] = []

    def flush() -> None:
        nonlocal hunk_lines
        if not hunk_lines:
            return
        excerpt = "\n".join(hunk_lines)
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[: max_excerpt_chars - 24] + "\n... [hunk truncated]"
        token_text = f"{path}\n{heading}\n{excerpt}"
        tokens = _tokenize(token_text)
        units.append(
            CodeUnit(
                key=_unit_key(path, start_line, heading, excerpt),
                path=path,
                new_line=start_line,
                heading=heading or "changed hunk",
                excerpt=excerpt,
                tokens=tokens,
                risk_tags=_risk_tags(tokens, path),
            )
        )
        hunk_lines = []

    in_hunk = False
    for line in text.splitlines():
        match = _DIFF_HEADER.match(line)
        if match:
            flush()
            path = match.group(2)
            in_hunk = False
            continue
        match = _NEW_FILE.match(line)
        if match and not in_hunk and match.group(1) != "/dev/null":
            path = match.group(1)
            continue
        match = _HUNK.match(line)
        if match:
            flush()
            start_line = int(match.group(1))
            heading = match.group(3).strip()
            in_hunk = True
            continue
        if in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                hunk_lines.append(line)
            elif line.startswith("-") and not line.startswith("---"):
                hunk_lines.append(line)
            elif line.startswith(" ") and len(hunk_lines) < 80:
                hunk_lines.append(line)

    flush()
    return units


def parse_diff_file(path: str | Path) -> list[CodeUnit]:
    diff_path = Path(path)
    return parse_unified_diff(diff_path.read_text(encoding="utf-8", errors="replace"))


def parse_review_input(path: str | Path) -> list[CodeUnit]:
    """Read either a unified diff or a frozen-input JSON artifact."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8", errors="replace")
    if input_path.suffix.lower() != ".json":
        return parse_unified_diff(text)
    payload = json.loads(text)
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValueError(f"JSON input does not contain a files list: {input_path}")
    sections: list[str] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("patch"), str):
            continue
        path_value = str(item.get("filename") or "unknown")
        previous = str(item.get("previous_filename") or path_value)
        sections.extend(
            (
                f"diff --git a/{previous} b/{path_value}",
                f"--- a/{previous}",
                f"+++ b/{path_value}",
                item["patch"],
            )
        )
    return parse_unified_diff("\n".join(sections))
