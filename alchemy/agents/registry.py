"""AlchemyAgents registry — tracks running internal agents."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for internal agents. Manages lifecycle (start/stop/status)."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        self._agents[name] = agent
        logger.info("AlchemyAgents: registered %s", name)

    def get(self, name: str) -> Any | None:
        return self._agents.get(name)

    def all_agents(self) -> dict[str, Any]:
        return dict(self._agents)

    async def start(self, name: str) -> bool:
        agent = self._agents.get(name)
        if agent and hasattr(agent, "start"):
            await agent.start()
            return True
        return False

    async def stop(self, name: str) -> bool:
        agent = self._agents.get(name)
        if agent and hasattr(agent, "stop"):
            await agent.stop()
            return True
        return False

    async def stop_all(self) -> None:
        for name in list(self._agents):
            await self.stop(name)
