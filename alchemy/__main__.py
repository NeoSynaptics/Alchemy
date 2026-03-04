"""Run Alchemy server: python -m alchemy"""

import asyncio
import sys
import uvicorn
from config.settings import settings

if __name__ == "__main__":
    # Playwright needs ProactorEventLoop on Windows, but uvicorn reload
    # spawns a subprocess that breaks it. Disable reload for production.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(
        "alchemy.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
