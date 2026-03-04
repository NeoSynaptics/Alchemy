"""Tests for Playwright agent prompts and formatting."""

from alchemy.agent.pw_prompts import (
    SYSTEM_PROMPT,
    format_user_prompt,
    format_action_log_entry,
)


class TestSystemPrompt:
    def test_contains_action_format(self):
        assert "click @REF" in SYSTEM_PROMPT
        assert "type @REF" in SYSTEM_PROMPT
        assert "scroll down" in SYSTEM_PROMPT
        assert "key KEYNAME" in SYSTEM_PROMPT
        assert "done" in SYSTEM_PROMPT

    def test_contains_response_format(self):
        assert "Thought:" in SYSTEM_PROMPT
        assert "Action:" in SYSTEM_PROMPT

    def test_one_action_rule(self):
        assert "exactly ONE" in SYSTEM_PROMPT


class TestFormatUserPrompt:
    def test_basic(self):
        prompt = format_user_prompt(
            task="Search for pole vault",
            snapshot_text="- textbox 'Search' [ref=e1]",
            action_log=[],
            step=1,
        )

        assert "Task: Search for pole vault" in prompt
        assert "Step: 1" in prompt
        assert "Accessibility tree" in prompt
        assert "[ref=e1]" in prompt
        assert "Thought:" in prompt

    def test_with_action_log(self):
        prompt = format_user_prompt(
            task="Do something",
            snapshot_text="- button 'OK' [ref=e1]",
            action_log=["Step 1: click @e2 → OK", "Step 2: type @e3 \"hello\" → OK"],
            step=3,
        )

        assert "Previous actions:" in prompt
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "Step: 3" in prompt

    def test_action_log_truncation(self):
        log = [f"Step {i}: click @e{i} → OK" for i in range(1, 21)]
        prompt = format_user_prompt(
            task="Task",
            snapshot_text="text",
            action_log=log,
            step=21,
            max_log_entries=5,
        )

        # Should only have the last 5 entries (Step 16-20)
        assert "Step 16:" in prompt
        assert "Step 20:" in prompt
        assert "Step 1:" not in prompt

    def test_empty_snapshot(self):
        prompt = format_user_prompt(
            task="Task",
            snapshot_text="[Empty page]",
            action_log=[],
            step=1,
        )
        assert "[Empty page]" in prompt


class TestFormatActionLogEntry:
    def test_click_success(self):
        entry = format_action_log_entry(step=1, action_type="click", ref="e5")
        assert entry == "Step 1: click @e5 → OK"

    def test_type_success(self):
        entry = format_action_log_entry(step=2, action_type="type", ref="e3", text="hello")
        assert entry == 'Step 2: type @e3 "hello" → OK'

    def test_scroll_success(self):
        entry = format_action_log_entry(step=3, action_type="scroll")
        assert entry == "Step 3: scroll → OK"

    def test_failure(self):
        entry = format_action_log_entry(
            step=4, action_type="click", ref="e7", success=False, error="Element not found"
        )
        assert "FAILED" in entry
        assert "Element not found" in entry

    def test_long_text_truncated(self):
        entry = format_action_log_entry(step=1, action_type="type", ref="e1", text="a" * 100)
        # Should truncate to 50 chars
        assert "a" * 50 in entry
        assert "a" * 51 not in entry
