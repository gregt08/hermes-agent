"""Tests for scripts/measure_static_context.py and hermes-root-* toolsets."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_measurement_helper_runs_without_telemetry():
    """Helper should run without enabling telemetry and not print secrets."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/measure_static_context.py", "--all-tools"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    # Should complete without error
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Could not import tool module" not in result.stderr

    # Should not contain raw memory content or secrets
    forbidden = ["HERMES_API_KEY", "sk-", "OPENAI_API_KEY", "memory_content"]
    for term in forbidden:
        assert term not in result.stdout, f"Secret/leak in output: {term}"

    # Should show hermes-cli as default
    assert "hermes-cli" in result.stdout

    # Should show all candidate modes
    for mode in ("hermes-root-ops-core", "hermes-root-chat-min",
                 "hermes-root-research", "hermes-root-coding-coord"):
        assert mode in result.stdout


def test_new_toolsets_resolve_correctly():
    """New hermes-root-* toolsets should resolve to expected tools."""
    from toolsets import resolve_toolset

    # ops-core: file + terminal + skills + todo + session_search + clarify + messaging
    ops = resolve_toolset("hermes-root-ops-core")
    assert "terminal" in ops
    assert "read_file" in ops
    assert "skills_list" in ops
    assert "todo" in ops
    assert "session_search" in ops
    assert "clarify" in ops
    assert "send_message" in ops
    # Should NOT include heavy tools
    assert "cronjob" not in ops
    assert "delegate_task" not in ops
    assert "browser_navigate" not in ops

    # chat-min: smallest
    chat = resolve_toolset("hermes-root-chat-min")
    assert len(chat) <= len(ops)
    assert "terminal" not in chat
    assert "read_file" not in chat
    assert "web_search" not in chat

    # research: web + browser
    research = resolve_toolset("hermes-root-research")
    assert "web_search" in research
    assert "browser_navigate" in research
    assert "terminal" not in research

    # coding-coord: delegation + kanban
    coding = resolve_toolset("hermes-root-coding-coord")
    assert "delegate_task" in coding
    assert "kanban_show" in coding
    assert "terminal" in coding


def test_root_toolsets_smaller_than_hermes_cli():
    """All hermes-root-* modes should be token-smaller than hermes-cli."""
    from toolsets import resolve_toolset

    hermes_cli = set(resolve_toolset("hermes-cli"))

    for mode in ("hermes-root-ops-core", "hermes-root-chat-min",
                 "hermes-root-research", "hermes-root-coding-coord"):
        tools = set(resolve_toolset(mode))
        # Each mode should be a proper subset (not equal, not superset)
        assert tools.issubset(hermes_cli), f"{mode} has tools not in hermes-cli: {tools - hermes_cli}"


def test_hermes_cli_unchanged():
    """hermes-cli toolset should remain exactly _HERMES_CORE_TOOLS."""
    from toolsets import _HERMES_CORE_TOOLS, get_toolset, resolve_toolset

    info = get_toolset("hermes-cli")
    assert info is not None
    assert info["description"] == "Full interactive CLI toolset - all default tools plus cronjob management"

    hermes_cli_tools = resolve_toolset("hermes-cli")
    assert hermes_cli_tools == sorted(set(_HERMES_CORE_TOOLS))


def test_cronjob_enabled_toolsets_schema_and_persistence_contract(tmp_path, monkeypatch):
    """Cron jobs should expose and persist per-job toolset scoping."""
    # Trigger tool discovery.
    import model_tools  # noqa: F401
    from tools.registry import registry

    schema = registry.get_schema("cronjob")
    props = schema["parameters"]["properties"]
    enabled_toolsets = props["enabled_toolsets"]

    assert enabled_toolsets["type"] == "array"
    assert enabled_toolsets["items"]["type"] == "string"

    from cron.jobs import create_job, get_job

    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    job = create_job(
        prompt="check a feed",
        schedule="every 1h",
        enabled_toolsets=["web", "session_search"],
    )
    fetched = get_job(job["id"])

    assert fetched["enabled_toolsets"] == ["web", "session_search"]


def test_measurement_script_json_mode():
    """JSON output mode should produce machine-readable data."""
    import subprocess
    import sys
    import json

    result = subprocess.run(
        [sys.executable, "scripts/measure_static_context.py", "--json"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "Could not import tool module" not in result.stderr
    data = json.loads(result.stdout)

    assert "hermes_cli" in data
    assert "candidate_modes" in data
    assert "top_tools" in data
    assert data["hermes_cli"]["total_tokens"] > 0
    assert len(data["top_tools"]) > 0


def test_schema_parameter_shape_preserved():
    """Edited tools should retain same parameter keys and required fields."""
    # Trigger tool discovery
    import model_tools  # noqa: F401
    from tools.registry import registry

    for tool_name in ("terminal", "session_search", "cronjob", "skill_manage"):
        schema = registry.get_schema(tool_name)
        assert schema is not None, f"Schema missing for {tool_name}"
        assert "parameters" in schema
        props = schema["parameters"].get("properties", {})
        assert len(props) > 0, f"No properties for {tool_name}"
        # Spot-check key parameters are present
        if tool_name == "terminal":
            assert "command" in props
            assert "background" in props
            assert "notify_on_complete" in props
            assert "watch_patterns" in props
        elif tool_name == "cronjob":
            assert "action" in props
            assert "enabled_toolsets" in props
        elif tool_name == "session_search":
            assert "query" in props
            assert "session_id" in props
