"""Conversation manager — context window and history for 14B chat.

Includes rolling summarization: after SUMMARIZE_THRESHOLD exchanges the
oldest messages are distilled into bullet-point summaries by the 14B,
keeping the context window compact while preserving key facts/decisions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import UUID

from alchemy.voice.models.schemas import ChatMessage

logger = logging.getLogger(__name__)

# -- Summarization constants --------------------------------------------------
SUMMARIZE_THRESHOLD = 8  # user+assistant pairs before triggering summarization
KEEP_RECENT = 6  # messages to keep verbatim (3 exchanges) after summarizing

SUMMARIZE_PROMPT = (
    "Distill the following conversation into concise bullet points. "
    "Capture: key facts, user preferences, decisions made, questions asked, "
    "and any commitments or promises. Do NOT add commentary — just the bullets.\n\n"
    "{conversation}\n\n"
    "Bullet-point summary:"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are Neo, a local AI assistant. You talk like a sharp, knowledgeable friend — "
    "not a textbook. The user is speaking to you by VOICE, so your responses will be "
    "read aloud by a text-to-speech engine.\n\n"
    "CRITICAL RULE: Respond in 2-3 sentences MAXIMUM. This is non-negotiable. "
    "Every word costs time to speak aloud. Be concise.\n\n"
    "Rules:\n"
    "- Be direct and opinionated. Say what YOU think.\n"
    "- Sound natural — like talking, not writing.\n"
    "- No bullet points, no numbered lists, no markdown, no emojis.\n"
    "- If asked something complex, give ONE key insight and offer to go deeper.\n"
    "- Never say 'As an AI' or 'I don't have feelings'. Just be real."
)


class ConversationManager:
    """Manages conversation history and context windows.

    In-memory storage keyed by conversation_id. Uses a sliding window
    to stay within the 14B's context limit. After SUMMARIZE_THRESHOLD
    exchanges, older messages are distilled into a rolling summary
    that gets injected as context — keeping the prompt compact while
    preserving key facts, decisions, and preferences.
    """

    def __init__(
        self,
        max_history: int = 50,
        max_tokens_estimate: int = 24000,
        system_prompt: str | None = None,
    ) -> None:
        self._conversations: dict[UUID, list[ChatMessage]] = defaultdict(list)
        self._summaries: dict[UUID, str] = {}
        self._turn_counts: dict[UUID, int] = defaultdict(int)
        self._summarizing: set[UUID] = set()  # guard against concurrent runs
        self._max_history = max_history
        self._max_tokens = max_tokens_estimate
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_messages(
        self,
        conversation_id: UUID,
        knowledge_context: list[str] | None = None,
    ) -> list[ChatMessage]:
        """Get full message list for Ollama including system prompt.

        Layout: [system, (summary)?, ...recent_history]

        The rolling summary (if present) is injected into the system
        prompt so the model sees it as grounding context, not as a
        prior assistant turn.

        Args:
            conversation_id: The conversation to fetch messages for.
            knowledge_context: Optional list of distilled knowledge docs
                to inject into the system prompt (from NEO-RX Nightwatch).

        Returns [system, ...history] trimmed to fit context window.
        """
        history = self._conversations[conversation_id]

        if len(history) > self._max_history:
            history = history[-self._max_history :]
            self._conversations[conversation_id] = history

        # Build system prompt with optional knowledge injection
        prompt = self._system_prompt

        # Inject rolling summary if we have one
        summary = self._summaries.get(conversation_id)
        if summary:
            prompt += (
                "\n\n---\nConversation so far (summarized):\n" + summary
            )

        if knowledge_context:
            knowledge_block = "\n\n---\nRelevant knowledge:\n" + "\n\n".join(
                knowledge_context
            )
            prompt += knowledge_block

        system_msg = ChatMessage(role="system", content=prompt)
        messages = [system_msg] + list(history)

        # Estimate tokens (~4 chars per token) and trim oldest if over budget
        total_chars = sum(len(m.content) for m in messages)
        while total_chars > self._max_tokens * 4 and len(messages) > 2:
            removed = messages.pop(1)
            total_chars -= len(removed.content)

        return messages

    # -- Summarization --------------------------------------------------------

    def needs_summarization(self, conversation_id: UUID) -> bool:
        """Check if this conversation has enough turns to warrant summarization."""
        if conversation_id in self._summarizing:
            return False  # already running
        return self._turn_counts[conversation_id] >= SUMMARIZE_THRESHOLD

    def get_messages_to_summarize(self, conversation_id: UUID) -> list[ChatMessage]:
        """Return the older messages that should be compressed into a summary.

        Keeps the most recent KEEP_RECENT messages verbatim; everything
        older (plus the existing summary if any) is fodder for the new summary.
        """
        history = self._conversations[conversation_id]
        if len(history) <= KEEP_RECENT:
            return []
        return list(history[:-KEEP_RECENT])

    def build_summarize_request(self, conversation_id: UUID) -> list[ChatMessage]:
        """Build the prompt messages for the summarization LLM call."""
        old_messages = self.get_messages_to_summarize(conversation_id)
        if not old_messages:
            return []

        # Format the conversation block for the summarizer
        lines: list[str] = []
        existing = self._summaries.get(conversation_id)
        if existing:
            lines.append(f"Previous summary:\n{existing}\n")
        for msg in old_messages:
            label = "User" if msg.role == "user" else "Neo"
            lines.append(f"{label}: {msg.content}")

        conversation_text = "\n".join(lines)
        prompt = SUMMARIZE_PROMPT.format(conversation=conversation_text)

        return [ChatMessage(role="user", content=prompt)]

    def apply_summary(self, conversation_id: UUID, summary_text: str) -> None:
        """Store the new rolling summary and trim the old messages."""
        self._summaries[conversation_id] = summary_text.strip()

        # Remove the old messages that were just summarized
        history = self._conversations[conversation_id]
        if len(history) > KEEP_RECENT:
            self._conversations[conversation_id] = history[-KEEP_RECENT:]

        # Reset turn counter so next summarization fires after another threshold
        self._turn_counts[conversation_id] = len(
            self._conversations[conversation_id]
        ) // 2

        self._summarizing.discard(conversation_id)
        logger.info(
            "Summary applied conv=%s, kept=%d recent messages, summary=%d chars",
            str(conversation_id)[:8],
            len(self._conversations[conversation_id]),
            len(summary_text),
        )

    def mark_summarizing(self, conversation_id: UUID) -> bool:
        """Mark conversation as currently being summarized. Returns False if already running."""
        if conversation_id in self._summarizing:
            return False
        self._summarizing.add(conversation_id)
        return True

    def cancel_summarizing(self, conversation_id: UUID) -> None:
        """Clear the summarizing flag (on error)."""
        self._summarizing.discard(conversation_id)

    def add_user_message(self, conversation_id: UUID, content: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=content)
        self._conversations[conversation_id].append(msg)
        self._turn_counts[conversation_id] += 1
        return msg

    def add_assistant_message(self, conversation_id: UUID, content: str) -> ChatMessage:
        msg = ChatMessage(role="assistant", content=content)
        self._conversations[conversation_id].append(msg)
        return msg

    def clear(self, conversation_id: UUID) -> None:
        self._conversations.pop(conversation_id, None)
        self._summaries.pop(conversation_id, None)
        self._turn_counts.pop(conversation_id, None)
        self._summarizing.discard(conversation_id)

    def active_conversations(self) -> int:
        return len(self._conversations)
