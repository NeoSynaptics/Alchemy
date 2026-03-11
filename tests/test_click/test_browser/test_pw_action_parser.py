"""Tests for Playwright agent action parser."""

import pytest

from alchemy.click.pw_action_parser import (
    PlaywrightAction,
    ParseError,
    parse_playwright_response,
    _extract_ref,
    _extract_quoted_text,
)


class TestExtractRef:
    def test_simple(self):
        assert _extract_ref("@e5") == "e5"

    def test_in_context(self):
        assert _extract_ref("click @e12 and stuff") == "e12"

    def test_no_ref(self):
        assert _extract_ref("no ref here") is None

    def test_multiple_refs_first_wins(self):
        assert _extract_ref("@e1 then @e2") == "e1"


class TestExtractQuotedText:
    def test_simple(self):
        assert _extract_quoted_text('"hello world"') == "hello world"

    def test_in_context(self):
        assert _extract_quoted_text('@e5 "search query"') == "search query"

    def test_no_quotes(self):
        assert _extract_quoted_text("no quotes") is None

    def test_empty_quotes(self):
        assert _extract_quoted_text('""') == ""


class TestParsePlaywrightResponse:
    def test_click(self):
        raw = "Thought: I see a submit button.\nAction: click @e4"
        action = parse_playwright_response(raw)

        assert action.type == "click"
        assert action.ref == "e4"
        assert action.thought == "I see a submit button."

    def test_type(self):
        raw = 'Thought: Need to search.\nAction: type @e3 "pole vault"'
        action = parse_playwright_response(raw)

        assert action.type == "type"
        assert action.ref == "e3"
        assert action.text == "pole vault"

    def test_scroll_down(self):
        raw = "Thought: Need to see more.\nAction: scroll down"
        action = parse_playwright_response(raw)

        assert action.type == "scroll"
        assert action.direction == "down"

    def test_scroll_up(self):
        raw = "Thought: Go back up.\nAction: scroll up"
        action = parse_playwright_response(raw)

        assert action.type == "scroll"
        assert action.direction == "up"

    def test_key_enter(self):
        raw = "Thought: Submit the form.\nAction: key Enter"
        action = parse_playwright_response(raw)

        assert action.type == "key"
        assert action.key_name == "Enter"

    def test_key_tab(self):
        raw = "Thought: Next field.\nAction: key Tab"
        action = parse_playwright_response(raw)

        assert action.type == "key"
        assert action.key_name == "Tab"

    def test_select(self):
        raw = 'Thought: Choose option.\nAction: select @e7 "United States"'
        action = parse_playwright_response(raw)

        assert action.type == "select"
        assert action.ref == "e7"
        assert action.text == "United States"

    def test_wait(self):
        raw = "Thought: Page is loading.\nAction: wait"
        action = parse_playwright_response(raw)

        assert action.type == "wait"

    def test_done(self):
        raw = "Thought: Task complete.\nAction: done"
        action = parse_playwright_response(raw)

        assert action.type == "done"

    def test_no_thought(self):
        raw = "Action: click @e1"
        action = parse_playwright_response(raw)

        assert action.type == "click"
        assert action.ref == "e1"
        assert action.thought == ""

    def test_multiline_thought(self):
        raw = "Thought: I see a form with several fields.\nThe search box is at the top.\nAction: click @e3"
        action = parse_playwright_response(raw)

        assert action.type == "click"
        assert "form" in action.thought

    def test_no_action_line_raises(self):
        with pytest.raises(ParseError, match="No 'Action:' line"):
            parse_playwright_response("Just some text without action")

    def test_click_no_ref_raises(self):
        with pytest.raises(ParseError, match="requires a ref"):
            parse_playwright_response("Action: click somewhere")

    def test_type_no_ref_raises(self):
        with pytest.raises(ParseError, match="requires a ref"):
            parse_playwright_response('Action: type "hello"')

    def test_type_no_text_raises(self):
        with pytest.raises(ParseError, match="requires quoted text"):
            parse_playwright_response("Action: type @e3 hello")

    def test_key_no_name_raises(self):
        with pytest.raises(ParseError, match="requires a key name"):
            parse_playwright_response("Action: key")

    def test_unknown_action_raises(self):
        with pytest.raises(ParseError, match="Unknown action"):
            parse_playwright_response("Action: dance @e1")

    def test_fill_alias(self):
        raw = 'Action: fill @e5 "text"'
        action = parse_playwright_response(raw)
        assert action.type == "type"

    def test_scroll_default_down(self):
        raw = "Action: scroll"
        action = parse_playwright_response(raw)
        assert action.direction == "down"

    def test_extra_whitespace(self):
        raw = "Thought:  spaces everywhere  \nAction:  click  @e1  "
        action = parse_playwright_response(raw)
        assert action.type == "click"
        assert action.ref == "e1"
