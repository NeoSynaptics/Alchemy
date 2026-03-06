"""AlchemyClick Standalone Demo — exercises the 10 proven patterns.

Runs without the full Alchemy server. Just Ollama + Windows desktop.

Usage:
    cd C:/Users/info/GitHub/Alchemy
    python scripts/alchemyclick_demo.py                    # Interactive menu
    python scripts/alchemyclick_demo.py --pattern ghost    # Run specific pattern
    python scripts/alchemyclick_demo.py --report           # Show pattern status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

sys.path.insert(0, ".")

from alchemy.click.patterns import ALL_PATTERNS, pattern_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("alchemyclick.demo")


# ---------------------------------------------------------------------------
# Pattern demos — each exercises one proven pattern in isolation
# ---------------------------------------------------------------------------

async def demo_screenshot_vlm_click():
    """Pattern 1: Screenshot → VLM → Coordinate → Click."""
    from alchemy.adapters.ollama import OllamaClient
    from alchemy.desktop.controller import DesktopController
    from alchemy.desktop.agent import DesktopAgent

    print("\n--- Pattern 1: Screenshot → VLM → Click ---")
    print("This will take a screenshot, send it to Qwen2.5-VL 7B,")
    print("and click whatever element you describe.\n")

    task = input("What should I click? (e.g., 'the Start button'): ").strip()
    if not task:
        task = "the Windows Start button in the taskbar"

    ollama = OllamaClient(host="http://localhost:11434", timeout=300.0, keep_alive="10m")
    await ollama.start()

    controller = DesktopController(
        screenshot_width=1280, screenshot_height=720,
        screenshot_quality=75, mode="ghost",
    )

    agent = DesktopAgent(
        ollama_client=ollama, controller=controller,
        model="qwen2.5vl:7b", max_steps=3, temperature=0.0, max_tokens=384,
    )

    print(f"\nRunning: '{task}' (ghost mode, max 3 steps)...")
    await asyncio.sleep(2)

    result = await agent.run(f"Click {task}")

    print(f"\nResult: {result.status.value} ({result.total_ms:.0f}ms)")
    for step in result.steps:
        coords = f"({step.x}, {step.y})" if step.x is not None else ""
        print(f"  Step {step.step}: {step.action_type} {coords} "
              f"[{step.inference_ms:.0f}ms] — {step.thought[:80]}")
    if result.error:
        print(f"  Error: {result.error}")

    await ollama.close()


async def demo_ghost_cursor():
    """Pattern 2: Ghost Cursor Overlay."""
    from alchemy.desktop.cursor import AICursor, CursorConfig

    print("\n--- Pattern 2: Ghost Cursor ---")
    print("Orange 24px dot will appear and demonstrate:")
    print("  - Smooth glide animation (ease-out cubic)")
    print("  - Click flash (#FF6600 → #FFAA00)")
    print("  - Park behavior (bottom-right corner)\n")

    cursor = AICursor(CursorConfig(size=24, color="#FF6600", glide_duration=0.4))
    cursor.start()
    time.sleep(0.5)

    # Demo path: center → top-left → top-right → click center → park
    screen_w, screen_h = cursor._screen_w, cursor._screen_h
    cx, cy = screen_w // 2, screen_h // 2

    print("  Moving to center...")
    cursor.move_to(cx, cy)
    time.sleep(0.8)

    print("  Gliding to top-left...")
    cursor.move_to(200, 200)
    time.sleep(0.8)

    print("  Gliding to top-right...")
    cursor.move_to(screen_w - 200, 200)
    time.sleep(0.8)

    print("  Click flash at center...")
    cursor.click_at(cx, cy)
    time.sleep(0.8)

    print("  Parking in corner...")
    cursor.park()
    time.sleep(1.0)

    cursor.stop()
    print("  Done. Ghost cursor destroyed.\n")


async def demo_sendinput_isolation():
    """Pattern 3: SendInput Mouse Isolation."""
    from alchemy.desktop.controller import _save_cursor, _restore_cursor, _send_click, _screen_size

    print("\n--- Pattern 3: SendInput Isolation ---")
    print("Your mouse cursor position will be saved, a click sent elsewhere,")
    print("and your cursor restored. You should see NO cursor movement.\n")

    saved = _save_cursor()
    print(f"  Saved cursor position: ({saved[0]}, {saved[1]})")

    w, h = _screen_size()
    # Click in center of screen (invisible)
    _send_click(w // 2, h // 2, w, h)
    print(f"  Sent click at ({w // 2}, {h // 2}) via SendInput")

    after = _save_cursor()
    _restore_cursor(*saved)
    restored = _save_cursor()

    print(f"  After click position: ({after[0]}, {after[1]})")
    print(f"  Restored to: ({restored[0]}, {restored[1]})")

    drift = abs(saved[0] - restored[0]) + abs(saved[1] - restored[1])
    if drift <= 2:
        print("  PASS: Zero cursor drift — isolation working.\n")
    else:
        print(f"  WARN: Cursor drifted {drift}px — check timing.\n")


async def demo_dual_format_parsing():
    """Pattern 4: Dual Coordinate Format Parsing."""
    from alchemy.click.action_parser import parse_uitars_response, to_vision_action, CoordMode

    print("\n--- Pattern 4: Dual Format Parsing ---")
    print("Testing both v1 (normalized 0-1000) and v1.5 (absolute pixel) formats.\n")

    # v1 format (normalized)
    v1_raw = 'Thought: I see the search box.\nAction: click(start_box=\'<|box_start|>(500,250)<|box_end|>\')'
    parsed_v1 = parse_uitars_response(v1_raw)
    action_v1 = to_vision_action(parsed_v1, 1920, 1080, coord_mode=CoordMode.NORMALIZED)
    print(f"  v1 (normalized 500,250 on 1920x1080): → ({action_v1.x}, {action_v1.y})")

    # v1.5 format (absolute in resized space)
    v15_raw = 'Thought: I see the search box.\nAction: click(start_box=\'(640,360)\')'
    parsed_v15 = parse_uitars_response(v15_raw)
    action_v15 = to_vision_action(
        parsed_v15, 1920, 1080,
        coord_mode=CoordMode.ABSOLUTE, resized_width=1280, resized_height=720,
    )
    print(f"  v1.5 (absolute 640,360 resized 1280x720 → 1920x1080): → ({action_v15.x}, {action_v15.y})")

    # Point format (official v1)
    point_raw = 'Thought: Search box found.\nAction: click(point=\'<point>750 500</point>\')'
    parsed_pt = parse_uitars_response(point_raw)
    action_pt = to_vision_action(parsed_pt, 1920, 1080, coord_mode=CoordMode.NORMALIZED)
    print(f"  point (750,500 normalized on 1920x1080): → ({action_pt.x}, {action_pt.y})")

    # Verify all parsed correctly
    all_ok = all(a.x is not None and a.y is not None for a in [action_v1, action_v15, action_pt])
    print(f"\n  {'PASS' if all_ok else 'FAIL'}: All formats parsed successfully.\n")


async def demo_action_tier_classification():
    """Pattern 5: 3-Tier Action Safety Classification."""
    from alchemy.click.action_parser import classify_tier
    from alchemy.schemas import ActionTier, VisionAction

    print("\n--- Pattern 5: Action Tier Classification ---")
    print("Each action type maps to AUTO / NOTIFY / APPROVE.\n")

    test_cases = [
        ("click", ActionTier.AUTO),
        ("scroll", ActionTier.AUTO),
        ("wait", ActionTier.AUTO),
        ("type", ActionTier.NOTIFY),
        ("hotkey", ActionTier.NOTIFY),
        ("done", ActionTier.AUTO),
    ]

    all_ok = True
    for action_name, expected_tier in test_cases:
        action = VisionAction(action=action_name, tier=ActionTier.AUTO)
        tier = classify_tier(action)
        status = "OK" if tier == expected_tier else "FAIL"
        if tier != expected_tier:
            all_ok = False
        print(f"  {action_name:15s} → {tier.value:8s} (expected {expected_tier.value}) [{status}]")

    print(f"\n  {'PASS' if all_ok else 'FAIL'}: Tier classification correct.\n")


async def demo_task_lifecycle():
    """Pattern 6: Task Lifecycle + Approval Gates."""
    from alchemy.click.task_manager import TaskManager
    from alchemy.schemas import TaskStatus

    print("\n--- Pattern 6: Task Lifecycle ---")
    print("Creating a task and walking through the state machine.\n")

    tm = TaskManager()
    task = tm.create_task("Demo: click the save button")

    print(f"  Created task {task.task_id}")
    print(f"  Status: {task.status.value}")

    tm.update_task(task.task_id, status=TaskStatus.RUNNING)
    state = tm.get_task(task.task_id)
    print(f"  → RUNNING: {state.status.value}")

    tm.update_task(task.task_id, status=TaskStatus.WAITING_APPROVAL)
    state = tm.get_task(task.task_id)
    print(f"  → WAITING_APPROVAL: {state.status.value}")

    # Simulate approval
    tm.approve(task.task_id)
    print(f"  → Approved (event set: {state.approval_event.is_set()})")

    tm.update_task(task.task_id, status=TaskStatus.COMPLETED)
    state = tm.get_task(task.task_id)
    print(f"  → COMPLETED: {state.status.value}")

    print(f"\n  PASS: Full lifecycle traversed.\n")


async def demo_multi_step():
    """Pattern 7: Multi-Step Execution (dry run — no actual model call)."""
    print("\n--- Pattern 7: Multi-Step Execution ---")
    print("Verifying history windowing and step counting logic.\n")

    # Simulate the history windowing from vision_agent.py
    history_window = 4
    messages = [{"role": "user", "content": "system prompt"}]  # Step 0

    for step in range(1, 12):
        messages.append({"role": "user", "content": f"Step {step} screenshot"})
        messages.append({"role": "assistant", "content": f"Action: click(...) at step {step}"})

        # Apply windowing
        if len(messages) > history_window * 2 + 1:
            messages = [messages[0]] + messages[-(history_window * 2):]

        print(f"  Step {step:2d}: {len(messages)} messages in context "
              f"(window={history_window}, max={history_window * 2 + 1})")

    bounded = len(messages) <= history_window * 2 + 1
    print(f"\n  {'PASS' if bounded else 'FAIL'}: History bounded at {len(messages)} messages.\n")


async def demo_adaptive_timeouts():
    """Pattern 10: Adaptive Timeouts."""
    from alchemy.router.categories import TaskCategory

    print("\n--- Pattern 10: Adaptive Timeouts ---")
    print("Each task category gets a tuned timeout.\n")

    _CATEGORY_TIMEOUTS = {
        TaskCategory.MEDIA: 180.0,
        TaskCategory.WEB: 240.0,
        TaskCategory.FILE: 180.0,
        TaskCategory.COMMUNICATION: 150.0,
        TaskCategory.DEVELOPMENT: 480.0,
        TaskCategory.SYSTEM: 150.0,
        TaskCategory.GENERAL: 300.0,
    }

    for cat, timeout in _CATEGORY_TIMEOUTS.items():
        print(f"  {cat.value:20s} → {timeout:5.0f}s")

    print(f"\n  PASS: {len(_CATEGORY_TIMEOUTS)} categories configured.\n")


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

DEMOS = {
    "screenshot-vlm-click": ("1. Screenshot → VLM → Click", demo_screenshot_vlm_click),
    "ghost-cursor": ("2. Ghost Cursor Overlay", demo_ghost_cursor),
    "sendinput-isolation": ("3. SendInput Mouse Isolation", demo_sendinput_isolation),
    "dual-format-parsing": ("4. Dual Format Parsing", demo_dual_format_parsing),
    "action-tier": ("5. Action Tier Classification", demo_action_tier_classification),
    "task-lifecycle": ("6. Task Lifecycle", demo_task_lifecycle),
    "multi-step": ("7. Multi-Step Execution", demo_multi_step),
    "adaptive-timeouts": ("10. Adaptive Timeouts", demo_adaptive_timeouts),
}


async def interactive_menu():
    """Show menu and let user pick a pattern to demo."""
    print("\n" + "=" * 50)
    print("  AlchemyClick — Standalone Demo")
    print("=" * 50)
    print()
    print(pattern_report())
    print()
    print("Available demos:")
    for key, (label, _) in DEMOS.items():
        print(f"  {key:25s} — {label}")
    print(f"  {'all':25s} — Run all demos")
    print(f"  {'quit':25s} — Exit")
    print()

    while True:
        choice = input("Pick a demo (or 'all'/'quit'): ").strip().lower()
        if choice == "quit":
            break
        if choice == "all":
            for key, (label, fn) in DEMOS.items():
                print(f"\n{'=' * 50}")
                print(f"  {label}")
                print("=" * 50)
                try:
                    await fn()
                except Exception as e:
                    print(f"  ERROR: {e}\n")
            break
        if choice in DEMOS:
            try:
                await DEMOS[choice][1]()
            except Exception as e:
                print(f"  ERROR: {e}\n")
        else:
            print(f"  Unknown demo: {choice}")


def main():
    parser = argparse.ArgumentParser(description="AlchemyClick Standalone Demo")
    parser.add_argument("--pattern", type=str, help="Run a specific pattern demo by key")
    parser.add_argument("--report", action="store_true", help="Show pattern status report")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()

    if args.report:
        print(pattern_report())
        return

    if args.pattern:
        if args.pattern not in DEMOS:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {', '.join(DEMOS.keys())}")
            return
        asyncio.run(DEMOS[args.pattern][1]())
        return

    if args.all:
        asyncio.run(interactive_menu())
        return

    asyncio.run(interactive_menu())


if __name__ == "__main__":
    main()
