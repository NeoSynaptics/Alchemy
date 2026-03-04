"""Core protocol contracts — interfaces that outer layers must implement.

The core never imports concrete implementations. It depends only on these
Protocol definitions. Adapters (e.g., OllamaClient) implement them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Contract: anything that can do LLM inference."""

    async def chat(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
        options: dict | None = None,
    ) -> dict: ...

    async def chat_think(
        self,
        model: str,
        messages: list[dict],
        think: bool = True,
        options: dict | None = None,
        seed: int | None = None,
    ) -> dict: ...

    async def ping(self) -> bool: ...


@runtime_checkable
class BrowserProvider(Protocol):
    """Contract: anything that can provide browser pages."""

    async def start(self) -> None: ...

    async def new_page(self, url: str | None = None): ...

    async def close(self) -> None: ...


@runtime_checkable
class ApprovalChecker(Protocol):
    """Contract: decides if an action needs human approval."""

    def needs_approval(
        self,
        action,
        element_name: str = "",
        page_url: str = "",
    ) -> bool: ...
