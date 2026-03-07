"""ConnectAgent ABC + AgentRouter — routes tunnel messages to agents.

Each agent that wants to be callable through the tunnel implements
ConnectAgent and registers with the AgentRouter. Messages arrive
with an `agent` field that maps to a registered agent_id.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from alchemy.connect.protocol import AlchemyMessage

logger = logging.getLogger(__name__)


class ConnectAgent(ABC):
    """Base class for anything callable through the AlchemyConnect tunnel.

    Subclasses implement handle() to process inbound messages and
    yield response messages back to the device.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique agent name (e.g. 'chat', 'browser', 'voice')."""
        ...

    @abstractmethod
    async def handle(
        self,
        msg: AlchemyMessage,
        device_id: str,
    ) -> AsyncIterator[AlchemyMessage]:
        """Process an inbound message and yield zero or more responses.

        Args:
            msg: The incoming message from the device.
            device_id: The authenticated device that sent it.

        Yields:
            AlchemyMessage responses to send back to the device.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Return agent capabilities for discovery. Override for richer info."""
        return {
            "agent_id": self.agent_id,
            "description": self.__class__.__doc__ or "",
        }


class AgentRouter:
    """Routes incoming tunnel messages to registered ConnectAgents."""

    def __init__(self) -> None:
        self._agents: dict[str, ConnectAgent] = {}

    def register(self, agent: ConnectAgent) -> None:
        """Register an agent. Overwrites if same agent_id exists."""
        self._agents[agent.agent_id] = agent
        logger.info("ConnectAgent registered: %s", agent.agent_id)

    def get(self, agent_id: str) -> ConnectAgent | None:
        return self._agents.get(agent_id)

    @property
    def available_agents(self) -> list[str]:
        return list(self._agents.keys())

    def describe_all(self) -> list[dict[str, Any]]:
        """Return descriptions of all registered agents."""
        return [a.describe() for a in self._agents.values()]
