"""NEO-N — Device file tunnel for phones and tablets.

Receives files over HTTP from paired devices and stages them for
BaratzaMemory batch ingest. No AI, no GPU — pure receive and store.

Reuses AlchemyConnect's device pairing. Any device already paired via
Connect can upload files through NEO-N.

Architecture:
    Phone App (AlchemyApps)
        | Binary frame via AlchemyConnect  OR  POST /v1/neo-n/upload
    NEO-N (this module)
        | Saves to staging folder
    BaratzaMemory batch ingest (automatic pickup)
"""

from alchemy.neo_n.manifest import MANIFEST
from alchemy.neo_n.receiver import FileReceiver

__all__ = ["MANIFEST", "FileReceiver"]
