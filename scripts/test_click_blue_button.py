"""Quick test: AlchemyFlow clicks a blue button on screen.

Usage:
    cd C:/Users/info/GitHub/Alchemy
    python scripts/test_click_blue_button.py

Make sure the blue button is visible on screen before running!
"""
import asyncio
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

async def main():
    from alchemy.adapters.ollama import OllamaClient
    from alchemy.desktop.controller import DesktopController
    from alchemy.desktop.agent import DesktopAgent

    print("\n=== AlchemyFlow Test: Click the Blue Button ===")
    print("You have 3 seconds to make sure the blue button is visible...\n")
    await asyncio.sleep(3)

    # 1. Connect to Ollama
    ollama = OllamaClient(host="http://localhost:11434", timeout=300.0, keep_alive="10m")
    await ollama.start()

    # 2. Desktop controller in ghost mode
    #    Lower-res screenshot (1280x720) for fast inference,
    #    but coordinate scaling uses real screen size (1920x1080) automatically.
    controller = DesktopController(
        screenshot_width=1280,
        screenshot_height=720,
        screenshot_quality=85,
        mode="ghost",
    )

    # 3. DesktopAgent with Qwen2.5-VL 7B (proven model) + native point_2d format
    agent = DesktopAgent(
        ollama_client=ollama,
        controller=controller,
        model="qwen2.5vl:7b",
        max_steps=3,
        temperature=0.0,
        max_tokens=512,
        num_ctx=8192,
    )

    # 4. Save a debug screenshot so we can see what the model sees
    screenshot_bytes = await controller.screenshot()
    with open("debug_screenshot.jpg", "wb") as f:
        f.write(screenshot_bytes)
    print(f"Saved debug_screenshot.jpg ({len(screenshot_bytes)} bytes, {controller.image_width}x{controller.image_height})")

    # 5. Run: click the blue button
    print("Taking screenshot and asking Qwen2.5-VL 7B to find the blue button...")
    result = await agent.run("Click the blue 'Yes' button")

    # 6. Report
    print(f"\n--- Result: {result.status.value} ({result.total_ms:.0f}ms) ---")
    print(f"    Screen: {controller.screen.width}x{controller.screen.height}")
    print(f"    Image sent: {controller.image_width}x{controller.image_height}")
    for step in result.steps:
        coords = f"({step.x}, {step.y})" if step.x is not None else ""
        print(f"  Step {step.step}: {step.action_type} {coords} "
              f"[{step.inference_ms:.0f}ms] -- {step.thought[:120]}")
    if result.error:
        print(f"  Error: {result.error}")

    await ollama.close()
    controller.park_cursor()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
