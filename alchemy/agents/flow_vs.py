"""AlchemyFlowVS — internal VS Code automation agent.

Uses AlchemyFlowAgent to click buttons and relay text inside VS Code.
Internal-only. Not user-callable. Touches Cloud AI — cannot be Tier 2.

Toggle: settings.agents.flow_vs.enabled (default: False)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from alchemy.click.flow.flow_agent import FlowAgent, StepResult

logger = logging.getLogger(__name__)

# VS Code-specific goals for common operations
_VSCODE_GOALS = {
    "click_button": "Click the button labeled '{label}' in VS Code",
    "relay_text": "Type the following text into the {target} in VS Code: {text}",
    "accept_suggestion": "Click the 'Accept' button on the inline suggestion in VS Code",
    "open_terminal": "Click on the Terminal tab or open a new terminal in VS Code",
    "run_command": "Open the command palette and type: {command}",
}


class FlowVSAgent:
    """Internal agent: automates VS Code via AlchemyFlowAgent.

    Lifecycle:
      - Created when agents.flow_vs.enabled is True
      - Starts a background loop waiting for commands
      - Uses FlowAgent.step() for each visual interaction
      - Stops when toggled off or VS Code closes
    """

    def __init__(
        self,
        *,
        flow_agent: FlowAgent | None = None,
        ollama: Any = None,
        screen: Any = None,
        executor: Any = None,
    ) -> None:
        self._flow_agent = flow_agent
        self._ollama = ollama
        self._screen = screen
        self._executor = executor
        self._running = False
        self._task: asyncio.Task | None = None
        self._command_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    @property
    def running(self) -> bool:
        return self._running

    def _ensure_flow_agent(self) -> FlowAgent:
        if self._flow_agent is None:
            if self._ollama is None or self._screen is None:
                raise RuntimeError(
                    "FlowVSAgent: needs either a FlowAgent or ollama+screen"
                )
            self._flow_agent = FlowAgent(
                ollama=self._ollama,
                screen=self._screen,
                executor=self._executor,
            )
        return self._flow_agent

    async def start(self) -> None:
        if self._running:
            return
        self._ensure_flow_agent()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("FlowVSAgent: started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("FlowVSAgent: stopped")

    async def click_button(self, label: str) -> StepResult | None:
        """Click a button in VS Code by label. Returns step result."""
        agent = self._ensure_flow_agent()
        goal = _VSCODE_GOALS["click_button"].format(label=label)
        try:
            return await agent.step(goal)
        except Exception as e:
            logger.error("FlowVSAgent: click_button(%r) failed: %s", label, e)
            return None

    async def relay_text(self, text: str, target: str = "editor") -> StepResult | None:
        """Type text into a VS Code element. Returns step result."""
        agent = self._ensure_flow_agent()
        goal = _VSCODE_GOALS["relay_text"].format(text=text, target=target)
        try:
            return await agent.step(goal)
        except Exception as e:
            logger.error("FlowVSAgent: relay_text failed: %s", e)
            return None

    async def run_command(self, command: str) -> StepResult | None:
        """Open VS Code command palette and run a command."""
        agent = self._ensure_flow_agent()
        goal = _VSCODE_GOALS["run_command"].format(command=command)
        try:
            return await agent.step(goal)
        except Exception as e:
            logger.error("FlowVSAgent: run_command(%r) failed: %s", command, e)
            return None

    async def enqueue(self, action: str, **kwargs: Any) -> None:
        """Queue a command for the background loop."""
        await self._command_queue.put((action, kwargs))

    async def _loop(self) -> None:
        """Background loop — processes queued commands."""
        while self._running:
            try:
                action, kwargs = await asyncio.wait_for(
                    self._command_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if action == "click_button":
                    await self.click_button(kwargs.get("label", ""))
                elif action == "relay_text":
                    await self.relay_text(
                        kwargs.get("text", ""),
                        kwargs.get("target", "editor"),
                    )
                elif action == "run_command":
                    await self.run_command(kwargs.get("command", ""))
                else:
                    logger.warning("FlowVSAgent: unknown action %r", action)
            except Exception as e:
                logger.error("FlowVSAgent: error processing %s: %s", action, e)
