"""Run Alchemy server: python -m alchemy"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "alchemy.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
