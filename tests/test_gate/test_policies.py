"""Tests for Gate static policies."""

import pytest
from alchemy.gate.policies import PolicyDecision, check_static_policy


class TestSafeTools:
    """Tools that should always be accepted without inference."""

    @pytest.mark.parametrize("tool", [
        "Read", "Glob", "Grep", "WebFetch", "WebSearch",
        "TodoWrite", "AskUserQuestion",
    ])
    def test_safe_tools_accepted(self, tool):
        decision, reason = check_static_policy(tool, {})
        assert decision == PolicyDecision.ACCEPT
        assert tool in reason

    def test_unknown_tool_goes_to_review(self):
        decision, _ = check_static_policy("SomeNewTool", {})
        assert decision == PolicyDecision.REVIEW


class TestBashSafe:
    """Bash commands that should be auto-accepted."""

    @pytest.mark.parametrize("cmd", [
        "git status",
        "git diff --staged",
        "git log --oneline -10",
        "git branch -a",
        "ls -la",
        "pwd",
        "echo hello",
        "cat package.json",
        "head -20 README.md",
        "npm test",
        "npm run build",
        "pytest tests/",
        "python -m pytest -x",
        "pip list",
        "node --version",
        "which python",
        "grep -r pattern .",
        "find . -name '*.py'",
    ])
    def test_safe_bash_accepted(self, cmd):
        decision, reason = check_static_policy("Bash", {"command": cmd})
        assert decision == PolicyDecision.ACCEPT, f"{cmd} should be accepted: {reason}"

    def test_empty_command_accepted(self):
        decision, _ = check_static_policy("Bash", {"command": ""})
        assert decision == PolicyDecision.ACCEPT


class TestBashDangerous:
    """Bash commands that should always be denied."""

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf .",
        "rm -r /home",
        "git push --force origin main",
        "git push -f origin master",
        "DROP TABLE users",
        "DROP DATABASE production",
        "TRUNCATE TABLE logs",
        "git push --no-verify",
    ])
    def test_dangerous_bash_denied(self, cmd):
        decision, reason = check_static_policy("Bash", {"command": cmd})
        assert decision == PolicyDecision.DENY, f"{cmd} should be denied: {reason}"


class TestBashAmbiguous:
    """Bash commands that should go to LLM review."""

    @pytest.mark.parametrize("cmd", [
        "npm install express",
        "pip install flask",
        "docker run hello-world",
        "curl https://example.com",
        "ssh user@host",
        "chmod 755 script.sh",
    ])
    def test_ambiguous_bash_reviewed(self, cmd):
        decision, _ = check_static_policy("Bash", {"command": cmd})
        assert decision == PolicyDecision.REVIEW, f"{cmd} should go to review"


class TestFileOperations:
    """Write/Edit operations on sensitive vs normal files."""

    @pytest.mark.parametrize("path", [
        ".env",
        ".env.local",
        "config/credentials.json",
        "secrets.yaml",
        "server.key",
        "id_rsa",
        "/home/user/.ssh/config",
        "cert.pem",
    ])
    def test_sensitive_file_denied(self, path):
        decision, _ = check_static_policy("Write", {"file_path": path})
        assert decision == PolicyDecision.DENY, f"Write to {path} should be denied"

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "README.md",
        "package.json",
        "tests/test_app.py",
    ])
    def test_normal_file_reviewed(self, path):
        """Normal files should go to review (not auto-accepted)."""
        decision, _ = check_static_policy("Write", {"file_path": path})
        assert decision == PolicyDecision.REVIEW

    def test_edit_sensitive_denied(self):
        decision, _ = check_static_policy("Edit", {"file_path": ".env"})
        assert decision == PolicyDecision.DENY
