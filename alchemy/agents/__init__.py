"""AlchemyAgents — internal agent orchestration layer.

Core module. Houses all internal agents that use AlchemyFlowAgent
and other Alchemy primitives to automate specific targets.

Agents here are INTERNAL — not user-callable, not exposed via API.
They are toggled on/off via settings and run silently.

Current agents:
  - FlowVS: VS Code automation (clicks buttons, relays text)
  - [future]: more target-specific agents

Similar target-specific agents for Discord, Slack, etc. will be
Tier 2 apps — NOT in this module. This module is reserved for
agents that touch Cloud AI or other core internals.
"""

from alchemy.agents.manifest import MANIFEST
from alchemy.agents.registry import AgentRegistry

__all__ = [
    "MANIFEST",
    "AgentRegistry",
]
