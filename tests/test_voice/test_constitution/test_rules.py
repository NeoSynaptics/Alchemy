"""Tests for constitutional rules — individual rule evaluation."""

from alchemy.voice.schemas import ActionTier, VisionAction
from alchemy.voice.constitution.rules import (
    AuthenticationRule,
    CommunicationRule,
    DestructiveActionRule,
    FinancialActionRule,
    SystemModificationRule,
)


class TestDestructiveActionRule:
    def setup_method(self):
        self.rule = DestructiveActionRule()

    def test_delete_triggers(self):
        action = VisionAction(action="click", reasoning="delete the file")
        verdict = self.rule.evaluate(action, "Clean up downloads")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_uninstall_triggers(self):
        action = VisionAction(action="click", reasoning="uninstall the app")
        verdict = self.rule.evaluate(action, "Remove Chrome")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_shutdown_triggers(self):
        action = VisionAction(action="click", reasoning="Click shutdown")
        verdict = self.rule.evaluate(action, "Restart PC")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_safe_action_passes(self):
        action = VisionAction(action="click", reasoning="Open the browser")
        verdict = self.rule.evaluate(action, "Open Chrome")
        assert verdict is None

    def test_wipe_in_goal_triggers(self):
        action = VisionAction(action="click", reasoning="Confirm")
        verdict = self.rule.evaluate(action, "Wipe all data from disk")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE


class TestFinancialActionRule:
    def setup_method(self):
        self.rule = FinancialActionRule()

    def test_purchase_triggers(self):
        action = VisionAction(action="click", reasoning="Click buy now")
        verdict = self.rule.evaluate(action, "Purchase a subscription")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_dollar_amount_triggers(self):
        action = VisionAction(action="click", reasoning="Confirm $49.99")
        verdict = self.rule.evaluate(action, "Complete order")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_transfer_triggers(self):
        action = VisionAction(action="click", reasoning="Send money")
        verdict = self.rule.evaluate(action, "Transfer funds")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_browsing_passes(self):
        action = VisionAction(action="click", reasoning="Open product page")
        verdict = self.rule.evaluate(action, "Look at this product")
        assert verdict is None


class TestCommunicationRule:
    def setup_method(self):
        self.rule = CommunicationRule()

    def test_send_email_high_risk(self):
        action = VisionAction(action="click", reasoning="Send to alice@test.com")
        verdict = self.rule.evaluate(action, "Email to someone")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_post_is_notify(self):
        action = VisionAction(action="click", reasoning="Post the message")
        verdict = self.rule.evaluate(action, "Update status")
        assert verdict is not None
        assert verdict.tier == ActionTier.NOTIFY

    def test_broadcast_is_approve(self):
        action = VisionAction(action="click", reasoning="Send to all contacts")
        verdict = self.rule.evaluate(action, "Broadcast message")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_reading_passes(self):
        action = VisionAction(action="click", reasoning="Open inbox")
        verdict = self.rule.evaluate(action, "Check email")
        assert verdict is None


class TestAuthenticationRule:
    def setup_method(self):
        self.rule = AuthenticationRule()

    def test_typing_password_triggers(self):
        action = VisionAction(action="type", text="hunter2", reasoning="Enter password")
        verdict = self.rule.evaluate(action, "Log in to account")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_click_login_ignored(self):
        # Only type actions trigger auth rule
        action = VisionAction(action="click", reasoning="Click sign in button")
        verdict = self.rule.evaluate(action, "Log in to account")
        assert verdict is None

    def test_typing_search_passes(self):
        action = VisionAction(action="type", text="hello world", reasoning="Type in search box")
        verdict = self.rule.evaluate(action, "Search for something")
        assert verdict is None


class TestSystemModificationRule:
    def setup_method(self):
        self.rule = SystemModificationRule()

    def test_registry_triggers(self):
        action = VisionAction(action="click", reasoning="Open regedit")
        verdict = self.rule.evaluate(action, "Edit registry")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_firewall_triggers(self):
        action = VisionAction(action="click", reasoning="Change firewall rule")
        verdict = self.rule.evaluate(action, "Modify settings")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_normal_app_passes(self):
        action = VisionAction(action="click", reasoning="Open Notepad")
        verdict = self.rule.evaluate(action, "Write notes")
        assert verdict is None
