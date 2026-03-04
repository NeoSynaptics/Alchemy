"""Tests for approval gate — irreversible action detection."""

import pytest

from alchemy.agent.pw_action_parser import PlaywrightAction
from alchemy.approval.gate import ApprovalGate, is_irreversible


class TestApprovalGate:
    def setup_method(self):
        self.gate = ApprovalGate(enabled=True)

    def test_scroll_never_needs_approval(self):
        action = PlaywrightAction(type="scroll", direction="down")
        assert self.gate.needs_approval(action) is False

    def test_wait_never_needs_approval(self):
        action = PlaywrightAction(type="wait")
        assert self.gate.needs_approval(action) is False

    def test_done_never_needs_approval(self):
        action = PlaywrightAction(type="done")
        assert self.gate.needs_approval(action) is False

    def test_click_submit_needs_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Submit") is True

    def test_click_send_needs_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Send Email") is True

    def test_click_delete_needs_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Delete File") is True

    def test_click_purchase_needs_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Purchase Now") is True

    def test_click_buy_needs_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Buy") is True

    def test_click_safe_button_no_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Next") is False

    def test_click_search_no_approval(self):
        action = PlaywrightAction(type="click", ref="e5")
        assert self.gate.needs_approval(action, element_name="Search") is False

    def test_click_cancel_overrides_keywords(self):
        action = PlaywrightAction(type="click", ref="e5")
        # "cancel" is a safe override even though "cancel subscription" has "cancel"
        assert self.gate.needs_approval(action, element_name="Cancel") is False

    def test_disabled_gate(self):
        gate = ApprovalGate(enabled=False)
        action = PlaywrightAction(type="click", ref="e5")
        assert gate.needs_approval(action, element_name="Delete Everything") is False

    def test_thought_triggers_approval(self):
        action = PlaywrightAction(
            type="click", ref="e5",
            thought="I need to submit the form to complete the purchase"
        )
        assert self.gate.needs_approval(action) is True

    def test_extra_keywords(self):
        gate = ApprovalGate(enabled=True, extra_keywords={"deploy", "merge"})
        action = PlaywrightAction(type="click", ref="e5")
        assert gate.needs_approval(action, element_name="Deploy to Production") is True
        assert gate.needs_approval(action, element_name="Merge PR") is True

    def test_type_action_with_irreversible_context(self):
        action = PlaywrightAction(type="type", ref="e3", text="confirm delete")
        assert self.gate.needs_approval(action) is True


class TestIsIrreversible:
    def test_simple_check(self):
        action = PlaywrightAction(type="click", ref="e1")
        assert is_irreversible(action, "Submit Order") is True
        assert is_irreversible(action, "Open Menu") is False
