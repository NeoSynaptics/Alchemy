"""Cloud AI Bridge — core primitive #4.

Manages cloud AI provider setup (credentials, VS Code extensions, validation).
Designed to be called by a future setup wizard or CLI.
"""

from alchemy.cloud.providers import CloudProvider, get_provider, list_providers
from alchemy.cloud.setup import CloudSetup

__all__ = ["CloudProvider", "CloudSetup", "get_provider", "list_providers"]
