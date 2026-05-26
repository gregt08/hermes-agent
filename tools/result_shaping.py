"""Shared helpers for compacting large model-facing tool outputs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Pattern


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_DISABLE_RESULT_COMPACTION_ENV = "HERMES_DISABLE_RESULT_COMPACTION"
_DEFAULT_RELEVANCE_RE = re.compile(
    r"\b("
    r"error|warning|failed|failure|exception|traceback|denied|blocked|captcha|"
    r"verify|verification|login|log in|sign in|signin|"
    r"submit|continue|next|action|required|security|forbidden|unauthorized"
    r")\b"
    r"|(?:@e\d+|\[ref=e\d+\])",
    re.IGNORECASE,
)
_RELEVANT_SNIPPETS_HEADER = "[RELEVANT OMITTED SNIPPETS]\n"
_DEFAULT_SECRET_TERM_RE = re.compile(r"\b(password|token)\b", re.IGNORECASE)


def normalize_result_mode(value: str | None) -> str:
    if isinstance(value, str) and value.lower() in {"auto", "full", "preview"}:
        return value.lower()
    return "auto"


def result_compaction_disabled() -> bool:
    return os.getenv(_DISABLE_RESULT_COMPACTION_ENV, "").strip().lower() in _TRUE_ENV_VALUES


@dataclass(frozen=True)
class CompactedText:
    text: str
    metadata: dict


def compact_text_output(
    text: str,
    *,
    result_mode: str = "auto",
    field_name: str = "output",
    threshold_chars: int = 20_000,
    preview_chars: int = 8_000,
    head_ratio: float = 0.45,
    hint: str = "",
    relevance_patterns: list[str | Pattern[str]] | None = None,
    relevance_context_lines: int = 1,
    max_relevance_snippets: int = 8,
) -> CompactedText:
    """Return head/tail text plus metadata when output should be compacted.

    The input text is never stored in metadata. Counts and a continuation hint
    are safe to expose even when the omitted content may contain sensitive data.
    """
    result_mode = normalize_result_mode(result_mode)
    if result_compaction_disabled() or result_mode == "full" or not isinstance(text, str):
        return CompactedText(text=text, metadata={})

    if preview_chars < 200:
        preview_chars = 200
    should_compact = result_mode == "preview" or len(text) > threshold_chars
    if not should_compact or len(text) <= preview_chars:
        return CompactedText(text=text, metadata={})

    head_chars = max(1, int(preview_chars * head_ratio))
    tail_chars = max(1, preview_chars - head_chars)
    relevance_text = _relevance_snippets(
        text,
        head_chars=head_chars,
        tail_chars=tail_chars,
        patterns=relevance_patterns,
        context_lines=relevance_context_lines,
        max_snippets=max_relevance_snippets,
    )
    snippet_budget = max(0, preview_chars // 4)
    if relevance_text and len(relevance_text) > snippet_budget:
        relevance_text = relevance_text[:snippet_budget].rstrip() + "\n... [relevance snippets truncated]"
    if relevance_text:
        middle_chars = len(_RELEVANT_SNIPPETS_HEADER) + len(relevance_text) + 2
        head_chars = max(1, head_chars - middle_chars // 2)
        tail_chars = max(1, tail_chars - (middle_chars - middle_chars // 2))

    omitted_chars = max(len(text) - head_chars - tail_chars, 0)
    if omitted_chars <= 0:
        return CompactedText(text=text, metadata={})

    total_lines = text.count("\n") + (1 if text else 0)
    omitted_text = text[head_chars:-tail_chars]
    omitted_lines = omitted_text.count("\n")
    notice = (
        f"\n\n... [{field_name.upper()} COMPACTED - {omitted_chars} chars"
        f" omitted from {len(text)} total"
    )
    if total_lines:
        notice += f"; {omitted_lines} lines omitted from {total_lines} total"
    notice += "] ...\n\n"

    metadata = {
        "compacted": True,
        f"{field_name}_original_chars": len(text),
        f"{field_name}_preview_chars": len(text[:head_chars] + notice + text[-tail_chars:]),
        f"{field_name}_omitted_chars": omitted_chars,
        f"{field_name}_total_lines": total_lines,
        f"{field_name}_omitted_lines": omitted_lines,
    }
    if hint:
        metadata["_hint"] = hint

    compacted_text = text[:head_chars] + notice
    if relevance_text:
        compacted_text += _RELEVANT_SNIPPETS_HEADER + relevance_text + "\n\n"
        metadata[f"{field_name}_relevance_snippets"] = relevance_text.count("\n---\n") + 1
    compacted_text += text[-tail_chars:]
    metadata[f"{field_name}_preview_chars"] = len(compacted_text)

    return CompactedText(text=compacted_text, metadata=metadata)


def _compile_patterns(patterns: list[str | Pattern[str]] | None) -> list[Pattern[str]]:
    if not patterns:
        return [_DEFAULT_RELEVANCE_RE]
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        if isinstance(pattern, str):
            compiled.append(re.compile(pattern, re.IGNORECASE))
        else:
            compiled.append(pattern)
    return compiled


def _relevance_snippets(
    text: str,
    *,
    head_chars: int,
    tail_chars: int,
    patterns: list[str | Pattern[str]] | None,
    context_lines: int,
    max_snippets: int,
) -> str:
    """Extract deterministic middle snippets that would be hidden by head/tail."""
    if max_snippets <= 0:
        return ""
    compiled = _compile_patterns(patterns)
    lines = text.splitlines()
    snippets: list[str] = []
    occupied: set[int] = set()
    cursor = 0
    for idx, line in enumerate(lines):
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end + 1
        if line_end <= head_chars or line_start >= len(text) - tail_chars:
            continue
        if patterns is None and _DEFAULT_SECRET_TERM_RE.search(line):
            continue
        if not any(pattern.search(line) for pattern in compiled):
            continue
        start = max(0, idx - max(context_lines, 0))
        end = min(len(lines), idx + max(context_lines, 0) + 1)
        if any(line_no in occupied for line_no in range(start, end)):
            continue
        occupied.update(range(start, end))
        snippets.append("\n".join(lines[start:end]))
        if len(snippets) >= max_snippets:
            break
    return "\n---\n".join(snippets)
