"""Single-shot test: send screenshot to Qwen2.5-VL and see raw coordinate output."""

import asyncio
import sys
sys.path.insert(0, ".")

from alchemy.adapters.ollama import OllamaClient
from alchemy.desktop.controller import DesktopController


async def main():
    ollama = OllamaClient(host="http://localhost:11434", timeout=300.0)
    await ollama.start()

    ctrl = DesktopController(screenshot_width=1280, screenshot_height=720, screenshot_quality=85, mode="shadow")
    screenshot = await ctrl.screenshot()
    print(f"Screenshot: {len(screenshot)} bytes, image space: {ctrl.image_width}x{ctrl.image_height}")
    print(f"Screen: {ctrl.screen.width}x{ctrl.screen.height}")

    # Save for reference
    with open("debug_click_target.jpg", "wb") as f:
        f.write(screenshot)

    prompt = """Look at this screenshot. Find the blue "Yes" button and output its exact pixel coordinates.
The image is 1280x720 pixels. Output ONLY in this format:
Thought: [what you see]
Action: click(start_box="(X,Y)")
Where X and Y are pixel positions in the 1280x720 image."""

    response = await ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{"role": "user", "content": prompt}],
        images=[screenshot],
        options={"temperature": 0.0, "num_predict": 256, "num_ctx": 2048},
    )

    raw = response.get("message", {}).get("content", "")
    print(f"\n=== RAW MODEL OUTPUT ===\n{raw}\n")

    await ollama.close()


if __name__ == "__main__":
    asyncio.run(main())
