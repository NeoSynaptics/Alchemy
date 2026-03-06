"""Live test: Desktop agent clicks the blue Yes button using Qwen2.5-VL 7B.

Usage:
    cd C:/Users/info/GitHub/Alchemy
    python scripts/click_test.py
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

from alchemy.adapters.ollama import OllamaClient
from alchemy.desktop.controller import DesktopController
from alchemy.desktop.agent import DesktopAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def main():
    # 1. Start Ollama client
    ollama = OllamaClient(host="http://localhost:11434", timeout=300.0, keep_alive="10m")
    await ollama.start()

    # 2. Create desktop controller in ghost mode (orange cursor visible)
    controller = DesktopController(
        screenshot_width=1280,
        screenshot_height=720,
        screenshot_quality=75,
        mode="ghost",
    )

    # 3. Create desktop agent with Qwen2.5-VL 7B
    agent = DesktopAgent(
        ollama_client=ollama,
        controller=controller,
        model="qwen2.5vl:7b",
        max_steps=5,
        temperature=0.0,
        max_tokens=384,
    )

    print("\n=== LIVE TEST: Click the blue Yes button (AlchemyWord.bat) ===")
    print("Agent mode: GHOST (orange cursor visible)")
    print("Model: qwen2.5vl:7b")
    print("Starting in 2 seconds...\n")

    await asyncio.sleep(2)

    # 4. Run the task
    result = await agent.run("Click the blue Yes button for AlchemyWord.bat")

    # 5. Report
    print(f"\n=== RESULT: {result.status.value} ({result.total_ms:.0f}ms) ===")
    for step in result.steps:
        coords = f"({step.x}, {step.y})" if step.x is not None else ""
        print(f"  Step {step.step}: {step.action_type} {coords} [{step.inference_ms:.0f}ms] — {step.thought[:80]}")
    if result.error:
        print(f"  Error: {result.error}")

    await ollama.close()


if __name__ == "__main__":
    asyncio.run(main())
