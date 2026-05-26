#!/usr/bin/env python3
"""
Reproducible static-context and tool-schema measurement helper.

Reports:
  - Full/all tool schema token estimate (all loadable tools)
  - Current hermes-cli toolset estimate
  - Candidate hermes-root-* mode estimates
  - Per-tool top-N schema costs

Does NOT enable telemetry and does NOT print secrets or raw memory content.
Gracefully degrades if tiktoken is unavailable (uses rough char/4 fallback).

Usage:
    python scripts/measure_static_context.py
    python scripts/measure_static_context.py --top 10
    python scripts/measure_static_context.py --candidate hermes-root-ops-core
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Setup logging (no telemetry)
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def _tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


def _rough_token_estimate(text: str) -> int:
    """Fallback when tiktoken unavailable: ~chars/4."""
    return max(1, len(text) // 4)


def _serialize_for_token_count(schema: dict) -> str:
    """Mirror the OpenAI tool schema serialization used in _estimate_tool_tokens."""
    return json.dumps({"type": "function", "function": schema}, separators=(",", ":"))


def _get_tool_token_counts() -> Dict[str, int]:
    """Return {tool_name: token_count} using tiktoken or rough fallback."""
    counts: Dict[str, int] = {}

    if _tiktoken_available():
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokenize = lambda text: len(enc.encode(text))
    else:
        tokenize = _rough_token_estimate

    class OptionalToolImportFilter(logging.Filter):
        """Suppress expected optional dependency warnings for this report helper."""

        EXPECTED_IMPORT_WARNINGS = {
            ("tools.browser_dialog_tool", "No module named 'websockets'"),
        }

        def filter(self, record: logging.LogRecord) -> bool:
            if record.getMessage().startswith("Could not import tool module "):
                message = record.getMessage()
                return not any(
                    module in message and reason in message
                    for module, reason in self.EXPECTED_IMPORT_WARNINGS
                )
            return True

    registry_logger = logging.getLogger("tools.registry")
    optional_import_filter = OptionalToolImportFilter()
    registry_logger.addFilter(optional_import_filter)
    try:
        # Trigger tool discovery
        import model_tools  # noqa: F401
        from tools.registry import registry
    except Exception as e:
        logger.warning("Tool registry unavailable (%s); using rough estimate.", e)
        return {}
    finally:
        registry_logger.removeFilter(optional_import_filter)

    for name in registry.get_all_tool_names():
        schema = registry.get_schema(name)
        if schema:
            text = _serialize_for_token_count(schema)
            counts[name] = tokenize(text)

    return counts


def _resolve_toolset_tools(name: str) -> List[str]:
    """Resolve toolset name to sorted list of tool names."""
    from toolsets import resolve_toolset
    return resolve_toolset(name)


def _token_total(tool_names: List[str], counts: Dict[str, int]) -> int:
    """Sum token counts for a list of tool names."""
    return sum(counts.get(name, 0) for name in tool_names)


def _print_table(rows: List[Dict], columns: List[str], title: str):
    """Print a compact aligned table."""
    print(f"\n{title}")
    print("-" * 60)
    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(col_widths[c]) for c in columns)
    print(header)
    print("-" * 60)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in columns))
    print()


def _percent_smaller(base: int, value: int) -> str:
    """Return a compact human-readable delta against a token baseline."""
    if not base:
        return "n/a"
    pct = round((1 - value / base) * 100)
    if pct > 0:
        return f"{pct}% smaller"
    if pct == 0:
        return "same"
    return f"{abs(pct)}% larger"


def main():
    parser = argparse.ArgumentParser(description="Measure static context / tool schema costs")
    parser.add_argument("--top", type=int, default=10, help="Show top N tools by token cost")
    parser.add_argument("--candidate", action="append", dest="candidates", default=[],
                       help="Also show estimate for this toolset name")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--all-tools", action="store_true",
                        help="Also show the full/all toolset estimate")
    args = parser.parse_args()

    counts = _get_tool_token_counts()

    if not counts:
        print("ERROR: Could not load tool registry or tiktoken (tried tiktoken and rough fallback).")
        print("Install tiktoken or run from the hermes-agent project root.")
        sys.exit(1)

    # Candidate modes to report
    candidate_modes = [
        "hermes-root-ops-core",
        "hermes-root-chat-min",
        "hermes-root-research",
        "hermes-root-coding-coord",
    ] + (args.candidates or [])

    # hermes-cli (the default)
    hermes_cli_tools = _resolve_toolset_tools("hermes-cli")
    hermes_cli_total = _token_total(hermes_cli_tools, counts)
    hermes_cli_count = len(hermes_cli_tools)

    # All tools
    all_tools_list = list(counts.keys())
    all_tools_total = sum(counts.values())
    all_tools_count = len(all_tools_list)

    # Per-tool sorted
    top_tools = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:args.top]

    if args.json:
        data = {
            "all_tools": {"count": all_tools_count, "total_tokens": all_tools_total},
            "hermes_cli": {"count": hermes_cli_count, "total_tokens": hermes_cli_total,
                           "tools": hermes_cli_tools},
            "top_tools": [{"name": n, "tokens": t} for n, t in top_tools],
            "candidate_modes": {},
        }
        for mode in candidate_modes:
            mode_tools = _resolve_toolset_tools(mode)
            data["candidate_modes"][mode] = {
                "count": len(mode_tools),
                "total_tokens": _token_total(mode_tools, counts),
                "tools": mode_tools,
            }
        print(json.dumps(data, indent=2))
        return

    # Human-readable output
    print("=" * 60)
    print("Hermes Static Context Measurement")
    print("=" * 60)

    has_tiktoken = _tiktoken_available()
    print(f"\n  Token estimator: {'tiktoken (cl100k_base)' if has_tiktoken else 'rough (chars/4 fallback)'}")
    print(f"  Tools measured : {all_tools_count}")

    # Summary row
    summary = [
        {"mode": "hermes-cli (default)", "tools": hermes_cli_count, "tokens": hermes_cli_total,
         "vs_all": _percent_smaller(all_tools_total, hermes_cli_total),
         "vs_hermes_cli": "baseline"},
    ]
    for mode in candidate_modes:
        mode_tools = _resolve_toolset_tools(mode)
        mode_total = _token_total(mode_tools, counts)
        mode_count = len(mode_tools)
        summary.append({
            "mode": mode,
            "tools": mode_count,
            "tokens": mode_total,
            "vs_all": _percent_smaller(all_tools_total, mode_total),
            "vs_hermes_cli": _percent_smaller(hermes_cli_total, mode_total),
        })

    if args.all_tools:
        summary.insert(0, {"mode": "ALL TOOLS", "tools": all_tools_count,
                           "tokens": all_tools_total, "vs_all": "baseline",
                           "vs_hermes_cli": "larger"})

    _print_table(summary, ["mode", "tools", "tokens", "vs_all", "vs_hermes_cli"], "SUMMARY")

    # Top N tools
    print(f"TOP {args.top} TOOLS BY SCHEMA COST")
    print("-" * 60)
    for i, (name, tokens) in enumerate(top_tools, 1):
        print(f"  {i:2}. {name:<35} {tokens:>6}t")
    print()

    print("NOTE: No telemetry enabled. No secrets or raw memory content printed.")
    print("Rollback: git revert schema changes; hermes-cli remains unchanged.")


if __name__ == "__main__":
    main()
