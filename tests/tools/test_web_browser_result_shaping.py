import json

from tools import browser_tool
from tools.web_tools import _shape_web_extract_result


def test_web_extract_small_content_preserves_metadata():
    result = {
        "url": "https://example.com",
        "title": "Example",
        "content": "short content",
        "error": None,
    }

    shaped = _shape_web_extract_result(result, "auto")

    assert shaped == result


def test_web_extract_large_auto_compacts_content():
    content = "HEAD\n" + ("middle\n" * 4000) + "TAIL"
    result = {
        "url": "https://example.com/long",
        "title": "Long",
        "content": content,
        "error": None,
    }

    shaped = _shape_web_extract_result(result, "auto")

    assert shaped["url"] == result["url"]
    assert shaped["title"] == result["title"]
    assert shaped["compacted"] is True
    assert shaped["content_original_chars"] == len(content)
    assert "CONTENT COMPACTED" in shaped["content"]
    assert "HEAD" in shaped["content"]
    assert "TAIL" in shaped["content"]
    assert "result_mode='full'" in shaped["_hint"]
    assert len(shaped["content"]) < len(content)


def test_web_extract_preserves_relevant_middle_snippets():
    content = (
        "HEAD\n"
        + ("ordinary content\n" * 1800)
        + "Security verification failed: CAPTCHA required before continue.\n"
        + "Use the login submit action after verification.\n"
        + ("ordinary tail content\n" * 1800)
        + "TAIL"
    )
    result = {
        "url": "https://example.com/security",
        "title": "Security",
        "content": content,
        "error": None,
        "blocked": False,
    }

    shaped = _shape_web_extract_result(result, "auto")

    assert shaped["url"] == result["url"]
    assert shaped["title"] == result["title"]
    assert shaped["blocked"] is False
    assert shaped["compacted"] is True
    assert "[RELEVANT OMITTED SNIPPETS]" in shaped["content"]
    assert "Security verification failed" in shaped["content"]
    assert "login submit action" in shaped["content"]
    assert shaped["content_relevance_snippets"] >= 1


def test_web_extract_does_not_preserve_standalone_sensitive_terms():
    content = (
        "HEAD\n"
        + ("ordinary content\n" * 1800)
        + "Glossary words only: password token credential placeholders.\n"
        + ("ordinary tail content\n" * 1800)
        + "TAIL"
    )
    result = {
        "url": "https://example.com/glossary",
        "title": "Glossary",
        "content": content,
        "error": None,
    }

    shaped = _shape_web_extract_result(result, "auto")

    assert shaped["compacted"] is True
    assert "Glossary words only" not in shaped["content"]
    assert "content_relevance_snippets" not in shaped


def test_web_extract_does_not_boost_contextual_sensitive_terms():
    content = (
        "HEAD\n"
        + ("ordinary content\n" * 1800)
        + "Authentication error: password reset token is required before submit.\n"
        + ("ordinary tail content\n" * 1800)
        + "TAIL"
    )
    result = {
        "url": "https://example.com/auth",
        "title": "Auth",
        "content": content,
        "error": None,
    }

    shaped = _shape_web_extract_result(result, "auto")

    assert shaped["compacted"] is True
    assert "Authentication error: password reset token" not in shaped["content"]


def test_compaction_relevance_budget_accounts_for_snippet_wrapper():
    content = (
        "HEAD\n"
        + ("ordinary content\n" * 500)
        + "Security verification failed: CAPTCHA required before continue.\n"
        + ("ordinary tail content\n" * 500)
        + "TAIL"
    )
    result = {
        "url": "https://example.com/security-small-preview",
        "title": "Security",
        "content": content,
        "error": None,
    }

    shaped = _shape_web_extract_result(result, "preview")

    assert shaped["compacted"] is True
    assert "[RELEVANT OMITTED SNIPPETS]" in shaped["content"]
    assert shaped["content_preview_chars"] == len(shaped["content"])


def test_web_extract_full_and_env_optout_preserve_content(monkeypatch):
    content = "HEAD\n" + ("middle\n" * 4000) + "TAIL"
    result = {
        "url": "https://example.com/long",
        "title": "Long",
        "content": content,
        "error": None,
    }

    assert _shape_web_extract_result(result, "full") == result

    monkeypatch.setenv("HERMES_DISABLE_RESULT_COMPACTION", "true")
    assert _shape_web_extract_result(result, "preview") == result


def test_web_extract_error_result_unchanged():
    result = {
        "url": "https://example.com/fail",
        "title": "",
        "content": "x" * 50000,
        "error": "failed",
    }

    assert _shape_web_extract_result(result, "auto") == result


def _large_snapshot() -> str:
    lines = ["heading"]
    lines.extend(f"static text line {i}" for i in range(1200))
    lines.append('button "Buy now" [ref=e12]')
    lines.append("dialog Verification required")
    lines.append("link Continue @e34")
    lines.extend(f"footer text line {i}" for i in range(100))
    return "\n".join(lines)


def test_browser_snapshot_large_default_preserves_refs_and_actions():
    snapshot = _large_snapshot()

    shaped = browser_tool.shape_browser_snapshot(snapshot, result_mode="auto")

    assert len(shaped) < len(snapshot)
    assert "[SNAPSHOT COMPACTED:" in shaped
    assert "[ref=e12]" in shaped
    assert "@e34" in shaped
    assert "Verification required" in shaped
    assert "result_mode='full'" in shaped


def test_browser_snapshot_relevant_middle_lines_win_budget():
    lines = ["heading"]
    lines.extend(f"static header line {i} " + ("x" * 160) for i in range(80))
    lines.extend(f"static middle line {i}" for i in range(1000))
    lines.append('button "Log in" [ref=e77]')
    lines.append("alert CAPTCHA verify required")
    lines.append("link Continue @e88")
    lines.extend(f"static footer line {i} " + ("y" * 160) for i in range(80))
    snapshot = "\n".join(lines)

    shaped = browser_tool.shape_browser_snapshot(snapshot, result_mode="auto", max_chars=2200)

    assert len(shaped) < len(snapshot)
    assert "[SNAPSHOT COMPACTED:" in shaped
    assert "[ref=e77]" in shaped
    assert "@e88" in shaped
    assert "CAPTCHA verify required" in shaped


def test_browser_snapshot_full_and_env_optout_preserve_exact(monkeypatch):
    snapshot = _large_snapshot()

    assert browser_tool.shape_browser_snapshot(snapshot, result_mode="full") == snapshot

    monkeypatch.setenv("HERMES_DISABLE_RESULT_COMPACTION", "true")
    assert browser_tool.shape_browser_snapshot(snapshot, result_mode="preview") == snapshot


def test_browser_snapshot_tool_preserves_element_count(monkeypatch):
    snapshot = _large_snapshot()

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"snapshot": snapshot, "refs": {"e12": {}, "e34": {}}},
        },
    )

    result = json.loads(browser_tool.browser_snapshot(task_id="shape-test"))

    assert result["success"] is True
    assert result["element_count"] == 2
    assert "[SNAPSHOT COMPACTED:" in result["snapshot"]
    assert "[ref=e12]" in result["snapshot"]
    assert "@e34" in result["snapshot"]


def test_browser_snapshot_result_mode_full_uses_full_backend_snapshot(monkeypatch):
    snapshot = _large_snapshot()
    calls = []

    def fake_run_browser_command(task_id, command, args=None, timeout=None):
        calls.append((command, args))
        return {
            "success": True,
            "data": {"snapshot": snapshot, "refs": {"e12": {}, "e34": {}}},
        }

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run_browser_command)

    result = json.loads(browser_tool.browser_snapshot(task_id="shape-test", result_mode="full"))

    assert result["success"] is True
    assert result["snapshot"] == snapshot
    assert ("snapshot", []) in calls


def test_browser_navigate_auto_snapshot_uses_result_mode(monkeypatch):
    snapshot = _large_snapshot()
    calls = []

    def fake_run_browser_command(task_id, command, args=None, timeout=None):
        calls.append((command, args))
        if command == "open":
            return {
                "success": True,
                "data": {"url": "https://example.com", "title": "Example"},
            }
        if command == "snapshot":
            return {
                "success": True,
                "data": {"snapshot": snapshot, "refs": {"e12": {}, "e34": {}}},
            }
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda task_id: {"_first_nav": False})
    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run_browser_command)

    result = json.loads(
        browser_tool.browser_navigate(
            "https://example.com",
            task_id="shape-nav",
            result_mode="preview",
        )
    )

    assert result["success"] is True
    assert result["url"] == "https://example.com"
    assert result["title"] == "Example"
    assert result["element_count"] == 2
    assert "[SNAPSHOT COMPACTED:" in result["snapshot"]
    assert "[ref=e12]" in result["snapshot"]
    assert "@e34" in result["snapshot"]
    assert ("snapshot", ["-c"]) in calls
