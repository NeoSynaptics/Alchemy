"""AlchemyFlowVS — Auto-approve Claude Code permission dialogs in VS Code.

Background daemon that monitors for blue permission buttons in VS Code,
reads the dialog context, asks Qwen3 14B if it's safe, then clicks.

Architecture:
    1. Screenshot loop (every 1s) — PIL color scan for blue button (~1ms)
    2. Blue detected → Qwen2.5-VL reads dialog context (what command/query)
    3. Qwen3 14B decides: approve / deny / skip (almost always approve)
    4. AlchemyFlow clicks Yes or No via ghost cursor + SendInput

Usage:
    cd C:/Users/info/GitHub/Alchemy
    python -m alchemy.desktop.flowvs
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from enum import Enum

from alchemy.adapters.ollama import OllamaClient
from alchemy.desktop.controller import DesktopController

logger = logging.getLogger(__name__)


# --- Configuration ---

# VS Code blue button color range (RGB).
# The "1 Yes" button is a saturated blue: roughly #0078D4 → #007ACC.
# We detect any pixel cluster within this range.
_BLUE_MIN = (0, 90, 170)
_BLUE_MAX = (60, 160, 255)

# Minimum number of blue pixels in a cluster to count as a button
_BLUE_PIXEL_THRESHOLD = 500

# Only scan the left half of the screen (VS Code panel is there)
_SCAN_REGION_X_FRAC = 0.55  # scan left 55% of screen
# Only scan bottom portion vertically (dialog appears in bottom half)
_SCAN_REGION_Y_FRAC_START = 0.35
_SCAN_REGION_Y_FRAC_END = 0.95  # stop before taskbar

# Cooldown after clicking — don't re-detect immediately
_POST_CLICK_COOLDOWN = 4.0

# How often to scan when idle
_SCAN_INTERVAL = 1.0


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    SKIP = "skip"  # not a permission dialog


@dataclass
class DialogContext:
    """Parsed permission dialog from VS Code."""
    dialog_type: str  # "search", "bash", "read", "write", "edit", etc.
    context_text: str  # the command or query being requested
    button_x: int  # image-space X of the blue button center
    button_y: int  # image-space Y of the blue button center


# --- Blue Button Detection (fast, no VLM) ---

def _detect_blue_button(screenshot_bytes: bytes, img_w: int, img_h: int) -> tuple[int, int] | None:
    """Scan screenshot for a blue button cluster. Returns (center_x, center_y) or None.

    Uses PIL to check pixel colors. Runs in ~2ms for 1280x720 images.
    Only scans the expected region (left panel, bottom half).
    """
    from PIL import Image

    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    pixels = img.load()

    # Define scan region
    scan_x_end = int(img_w * _SCAN_REGION_X_FRAC)
    scan_y_start = int(img_h * _SCAN_REGION_Y_FRAC_START)
    scan_y_end = int(img_h * _SCAN_REGION_Y_FRAC_END)

    blue_pixels: list[tuple[int, int]] = []

    # Sample every 2nd pixel for speed
    for y in range(scan_y_start, scan_y_end, 2):
        for x in range(0, scan_x_end, 2):
            r, g, b = pixels[x, y]
            if (
                _BLUE_MIN[0] <= r <= _BLUE_MAX[0]
                and _BLUE_MIN[1] <= g <= _BLUE_MAX[1]
                and _BLUE_MIN[2] <= b <= _BLUE_MAX[2]
            ):
                blue_pixels.append((x, y))

    if len(blue_pixels) < _BLUE_PIXEL_THRESHOLD:
        return None

    # Return center of the blue cluster
    avg_x = sum(p[0] for p in blue_pixels) // len(blue_pixels)
    avg_y = sum(p[1] for p in blue_pixels) // len(blue_pixels)

    logger.info(
        "Blue button detected: %d blue pixels, center (%d, %d)",
        len(blue_pixels), avg_x, avg_y,
    )
    return (avg_x, avg_y)


# --- Context Reading (Qwen2.5-VL) ---

_READ_CONTEXT_PROMPT = """\
Look at this VS Code screenshot. There is a permission dialog with a blue button.
Read the dialog and tell me:
1. What type of action is being requested (search, bash command, read file, write file, edit, etc.)
2. The exact text/command being requested

Reply in this exact format:
TYPE: <action_type>
CONTEXT: <the exact text or command>"""


async def _read_dialog_context(
    ollama: OllamaClient,
    screenshot_bytes: bytes,
    button_x: int,
    button_y: int,
) -> DialogContext | None:
    """Use Qwen2.5-VL to read the permission dialog text."""
    try:
        response = await ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{"role": "user", "content": _READ_CONTEXT_PROMPT}],
            images=[screenshot_bytes],
            options={
                "temperature": 0.0,
                "num_predict": 256,
                "num_ctx": 8192,
            },
        )
        raw = response.get("message", {}).get("content", "")
        logger.info("Dialog context response: %s", raw[:200])

        # Parse TYPE and CONTEXT
        dialog_type = "unknown"
        context_text = ""
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                dialog_type = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("CONTEXT:"):
                context_text = line.split(":", 1)[1].strip()

        if not context_text:
            context_text = raw[:200]

        return DialogContext(
            dialog_type=dialog_type,
            context_text=context_text,
            button_x=button_x,
            button_y=button_y,
        )
    except Exception as e:
        logger.error("Failed to read dialog context: %s", e)
        return None


# --- Decision Making (Qwen3 14B) ---

_DECISION_PROMPT = """\
You are a safety gate for an AI coding assistant (Claude Code) running in VS Code.
A permission dialog appeared asking to perform an action. Decide: approve or deny.

Rules:
- Web searches: ALWAYS approve (harmless information gathering)
- File reads: ALWAYS approve (reading is non-destructive)
- Bash commands: approve UNLESS the command is destructive (rm -rf, drop database, etc.)
- File writes/edits: approve if the path looks like a normal project file
- DENY only if the action could cause irreversible damage

Action type: {dialog_type}
Command/query: {context_text}

Reply with EXACTLY one word: APPROVE or DENY"""


async def _decide(
    ollama: OllamaClient,
    dialog: DialogContext,
) -> Decision:
    """Ask Qwen3 14B whether to approve this action."""
    try:
        prompt = _DECISION_PROMPT.format(
            dialog_type=dialog.dialog_type,
            context_text=dialog.context_text,
        )
        response = await ollama.chat_think(
            model="qwen3:14b",
            messages=[{"role": "user", "content": prompt}],
            think=False,  # fast mode, no reasoning needed
            options={
                "temperature": 0.0,
                "num_predict": 10,
            },
        )
        answer = response.get("content", "").strip().upper()
        logger.info("Decision for '%s': %s", dialog.context_text[:60], answer)

        if "APPROVE" in answer:
            return Decision.APPROVE
        return Decision.DENY
    except Exception as e:
        logger.error("Decision failed: %s — defaulting to DENY (fail-closed)", e)
        return Decision.DENY  # fail-closed: deny when uncertain


# --- Click Execution (AlchemyFlow) ---

async def _click_button(
    controller: DesktopController,
    img_x: int,
    img_y: int,
    approve: bool,
) -> str:
    """Click Yes (approve) or No (deny) using AlchemyFlow.

    img_x, img_y are in image pixel space — we scale to screen.
    If denying, we click lower (the "No" option is below "Yes").
    """
    screen = controller.screen
    img_w = controller.image_width
    img_h = controller.image_height

    if approve:
        # Click the blue Yes button directly
        screen_x = round(img_x / img_w * screen.width)
        screen_y = round(img_y / img_h * screen.height)
    else:
        # Click "No" — it's roughly 50px below "Yes" in image space
        screen_x = round(img_x / img_w * screen.width)
        screen_y = round((img_y + 50) / img_h * screen.height)

    screen_x = min(max(screen_x, 0), screen.width)
    screen_y = min(max(screen_y, 0), screen.height)

    result = await controller.click(screen_x, screen_y)
    logger.info("Clicked %s at (%d, %d)", "Yes" if approve else "No", screen_x, screen_y)
    return result


# --- Main Loop ---

class AlchemyFlowVS:
    """Background daemon that auto-approves VS Code permission dialogs."""

    def __init__(
        self,
        ollama: OllamaClient,
        controller: DesktopController,
        scan_interval: float = _SCAN_INTERVAL,
        post_click_cooldown: float = _POST_CLICK_COOLDOWN,
    ):
        self._ollama = ollama
        self._controller = controller
        self._scan_interval = scan_interval
        self._cooldown = post_click_cooldown
        self._running = False
        self._stats = {"scans": 0, "detections": 0, "approvals": 0, "denials": 0}

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def run(self):
        """Main monitoring loop. Runs until stop() is called."""
        self._running = True
        logger.info("AlchemyFlowVS started — monitoring for permission dialogs...")

        while self._running:
            try:
                await self._scan_once()
            except Exception as e:
                logger.error("Scan error: %s", e)

            await asyncio.sleep(self._scan_interval)

        logger.info("AlchemyFlowVS stopped. Stats: %s", self._stats)

    async def run_once(self) -> bool:
        """Single scan cycle. Returns True if a button was clicked."""
        return await self._scan_once()

    def stop(self):
        """Signal the monitoring loop to stop."""
        self._running = False

    async def _scan_once(self) -> bool:
        """One scan cycle: screenshot → detect → read → decide → click."""
        self._stats["scans"] += 1

        # 1. Take screenshot
        screenshot = await self._controller.screenshot()
        img_w = self._controller.image_width
        img_h = self._controller.image_height

        # 2. Fast blue button detection (~2ms)
        t0 = time.monotonic()
        button = await asyncio.to_thread(
            _detect_blue_button, screenshot, img_w, img_h,
        )
        detect_ms = (time.monotonic() - t0) * 1000

        if button is None:
            return False

        self._stats["detections"] += 1
        btn_x, btn_y = button
        logger.info("Blue button found in %.1fms at (%d, %d)", detect_ms, btn_x, btn_y)

        # 3. Read dialog context with Qwen2.5-VL
        t1 = time.monotonic()
        dialog = await _read_dialog_context(
            self._ollama, screenshot, btn_x, btn_y,
        )
        read_ms = (time.monotonic() - t1) * 1000

        if dialog is None:
            logger.warning("Could not read dialog context — skipping")
            return False

        logger.info(
            "Dialog [%.0fms]: type=%s, context='%s'",
            read_ms, dialog.dialog_type, dialog.context_text[:80],
        )

        # 4. Decide with Qwen3 14B
        t2 = time.monotonic()
        decision = await _decide(self._ollama, dialog)
        decide_ms = (time.monotonic() - t2) * 1000
        logger.info("Decision [%.0fms]: %s", decide_ms, decision.value)

        # 5. Click
        if decision == Decision.APPROVE:
            self._stats["approvals"] += 1
            await _click_button(self._controller, btn_x, btn_y, approve=True)
        elif decision == Decision.DENY:
            self._stats["denials"] += 1
            await _click_button(self._controller, btn_x, btn_y, approve=False)
        else:
            return False

        # 6. Cooldown — wait before resuming scanning
        await asyncio.sleep(self._cooldown)
        return True


# --- CLI Entry Point ---

async def main():
    """Run AlchemyFlowVS as a standalone daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("\n=== AlchemyFlowVS — Auto-Approve Daemon ===")
    print("Monitoring VS Code for permission dialogs...")
    print("Press Ctrl+C to stop.\n")

    ollama = OllamaClient(host="http://localhost:11434", timeout=300.0, keep_alive="10m")
    await ollama.start()

    controller = DesktopController(
        screenshot_width=1280,
        screenshot_height=720,
        screenshot_quality=85,
        mode="ghost",
    )

    daemon = AlchemyFlowVS(ollama=ollama, controller=controller)

    try:
        await daemon.run()
    except KeyboardInterrupt:
        daemon.stop()
        print(f"\nStopped. Stats: {daemon.stats}")
    finally:
        await ollama.close()
        controller.park_cursor()


if __name__ == "__main__":
    asyncio.run(main())
