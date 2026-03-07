"""AlchemySynteticTest — Desktop testing environment for the Alchemy ecosystem.

Follows the Alchemy App Contract:
  - Uses alchemy_boot.pyw to auto-start Alchemy if not running
  - Checks module contracts before running tests
  - Never loads models directly — Alchemy Core handles that
"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path

from runners.alchemy_boot import start_alchemy, start_voice, check_contracts, is_alchemy_running, is_voice_running
from runners.voice_runner import VoiceTestRunner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOICE_URL = "http://localhost:8100"
ALCHEMY_URL = "http://localhost:8000"

WINDOW_TITLE = "Alchemy Syntetic Test"
WINDOW_SIZE = "720x700"

BG = "#1a1a2e"
CARD_BG = "#16213e"
TEXT_FG = "#e0e0e0"
ACCENT = "#0f3460"
GREEN = "#2ecc71"
RED = "#e74c3c"
YELLOW = "#f39c12"
BLUE = "#3498db"
DIM = "#7f8c8d"


class TestCard:
    """A single test panel with button, status label, and log area."""

    def __init__(self, parent: tk.Frame, title: str, subtitle: str, enabled: bool = True):
        self.frame = tk.Frame(parent, bg=CARD_BG, padx=12, pady=10,
                              highlightbackground=ACCENT, highlightthickness=1)
        self.frame.pack(fill="x", padx=10, pady=5)

        # Header row
        header = tk.Frame(self.frame, bg=CARD_BG)
        header.pack(fill="x")

        self.title_label = tk.Label(header, text=title, font=("Segoe UI", 13, "bold"),
                                    fg=TEXT_FG, bg=CARD_BG, anchor="w")
        self.title_label.pack(side="left")

        self.status_label = tk.Label(header, text="READY" if enabled else "NOT WIRED",
                                     font=("Segoe UI", 10),
                                     fg=GREEN if enabled else DIM, bg=CARD_BG)
        self.status_label.pack(side="right")

        # Subtitle
        tk.Label(self.frame, text=subtitle, font=("Segoe UI", 9),
                 fg=DIM, bg=CARD_BG, anchor="w").pack(fill="x")

        # Log area
        self.log_text = tk.Text(self.frame, height=5, bg="#0d1117", fg=TEXT_FG,
                                font=("Consolas", 9), relief="flat", wrap="word",
                                state="disabled", padx=6, pady=4)
        self.log_text.pack(fill="x", pady=(6, 4))

        # Tag colors
        self.log_text.tag_configure("pass", foreground=GREEN)
        self.log_text.tag_configure("fail", foreground=RED)
        self.log_text.tag_configure("warn", foreground=YELLOW)
        self.log_text.tag_configure("info", foreground=BLUE)
        self.log_text.tag_configure("dim", foreground=DIM)

        # Button
        btn_text = f"Run {title}" if enabled else f"{title} (Not Wired)"
        self.run_btn = tk.Button(self.frame, text=btn_text,
                                 font=("Segoe UI", 10, "bold"),
                                 bg=ACCENT if enabled else "#2c3e50",
                                 fg=TEXT_FG, relief="flat", padx=16, pady=4,
                                 activebackground=BLUE,
                                 state="normal" if enabled else "disabled")
        self.run_btn.pack(pady=(2, 0))

    def log(self, msg: str, tag: str = ""):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def set_status(self, text: str, color: str = TEXT_FG):
        self.status_label.config(text=text, fg=color)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        ico = Path(__file__).parent / "assets" / "alchemytest.ico"
        if ico.exists():
            self.root.iconbitmap(str(ico))

        # Title
        title_frame = tk.Frame(self.root, bg=BG, pady=8)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="Alchemy Syntetic Test",
                 font=("Segoe UI", 18, "bold"), fg=TEXT_FG, bg=BG).pack()

        # Status bar (Alchemy connection)
        self.status_bar = tk.Label(title_frame, text="Checking Alchemy...",
                                   font=("Segoe UI", 10), fg=YELLOW, bg=BG)
        self.status_bar.pack()

        # Container
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True)

        # Test cards
        self.voice_card = TestCard(
            container, "AlchemyVoice",
            "Cold start, mic check, synthetic input, response timing, second message",
            enabled=True,
        )
        self.click_card = TestCard(
            container, "AlchemyClick",
            "Browser a11y click + Flow vision coordinate tests",
            enabled=True,
        )
        self.gpu_card = TestCard(
            container, "AlchemyGPU",
            "VRAM status, model loading, GPU health, timeout diagnostics",
            enabled=True,
        )
        self.memory_card = TestCard(
            container, "AlchemyMemory",
            "Memory persistence and recall validation",
            enabled=False,
        )

        # Wire buttons
        self.voice_card.run_btn.config(command=self._run_voice)
        self.click_card.run_btn.config(command=lambda: self._not_yet(self.click_card))
        self.gpu_card.run_btn.config(command=lambda: self._not_yet(self.gpu_card))

        # Disable all test buttons until Alchemy is confirmed running
        self._set_all_buttons("disabled")

        # Async loop for runners
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # Kick off Alchemy boot check on startup
        self._boot_thread = threading.Thread(target=self._ensure_alchemy, daemon=True)
        self._boot_thread.start()

    # ------------------------------------------------------------------
    # Alchemy boot (runs on startup)
    # ------------------------------------------------------------------
    def _ensure_alchemy(self):
        """Ensure Alchemy is running. Auto-starts via alchemy_boot.pyw if needed."""

        def log(msg, tag=""):
            self.root.after(0, self._boot_log, msg, tag)

        try:
            # 1. Start Alchemy Core (:8000)
            log("Checking if Alchemy is running...", "info")
            ok = start_alchemy(log)
            if not ok:
                self.root.after(0, self._boot_failed)
                return

            # 2. Start AlchemyVoice (:8100)
            log("Checking if AlchemyVoice is running...", "info")
            start_voice(log)  # Non-fatal — some tests don't need voice

            # 3. Check contracts
            log("Checking module contracts...", "info")
            result = check_contracts(log)

            if result["all_satisfied"]:
                self.root.after(0, self._boot_ready, result)
            else:
                self.root.after(0, self._boot_partial, result)

        except Exception as e:
            # Never leave buttons locked — enable them even on error
            log(f"Boot error: {e}", "fail")
            self.root.after(0, self._boot_error, str(e))

    def _boot_log(self, msg: str, tag: str = ""):
        # Show boot messages in the status bar (last line only)
        self.status_bar.config(text=msg, fg=BLUE)

    def _boot_ready(self, contracts: dict):
        n = len(contracts["modules"])
        self.status_bar.config(
            text=f"Alchemy running  |  {n} modules  |  All contracts OK",
            fg=GREEN,
        )
        self._set_all_buttons("normal")

    def _boot_partial(self, contracts: dict):
        n = len(contracts["modules"])
        bad = ", ".join(contracts["unsatisfied"])
        self.status_bar.config(
            text=f"Alchemy running  |  {n} modules  |  Unsatisfied: {bad}",
            fg=YELLOW,
        )
        # Enable buttons — tests will handle contract failures gracefully
        self._set_all_buttons("normal")

    def _boot_failed(self):
        self.status_bar.config(
            text="Alchemy NOT running  |  Auto-start failed  |  Tests unavailable",
            fg=RED,
        )
        # Still enable buttons so user can retry after manually starting Alchemy
        self._set_all_buttons("normal")

    def _boot_error(self, msg: str):
        self.status_bar.config(
            text=f"Boot error: {msg}",
            fg=RED,
        )
        self._set_all_buttons("normal")

    def _set_all_buttons(self, state: str):
        for card in [self.voice_card, self.click_card, self.gpu_card]:
            card.run_btn.config(state=state)

    def _not_yet(self, card: TestCard):
        card.clear_log()
        card.log("Test not implemented yet. Coming soon.", "warn")

    # ------------------------------------------------------------------
    # Voice test
    # ------------------------------------------------------------------
    def _run_voice(self):
        self.voice_card.run_btn.config(state="disabled")
        self.voice_card.clear_log()
        self.voice_card.set_status("RUNNING...", YELLOW)
        self.voice_card.log("Starting AlchemyVoice test...", "info")

        runner = VoiceTestRunner(
            voice_url=VOICE_URL,
            log_fn=self._voice_log,
            done_fn=self._voice_done,
            root=self.root,
        )
        asyncio.run_coroutine_threadsafe(runner.run(), self._loop)

    def _voice_log(self, msg: str, tag: str = ""):
        self.root.after(0, self.voice_card.log, msg, tag)

    def _voice_done(self, passed: bool, should_route_gpu: bool = False):
        def _update():
            if should_route_gpu:
                self.voice_card.set_status("TIMEOUT -> GPU TEST", RED)
                self.voice_card.log(
                    "Response too slow or timed out. Routing to AlchemyGPU test.", "warn"
                )
                self.gpu_card.set_status("QUEUED (from Voice)", YELLOW)
                self.gpu_card.log("Voice test flagged slow response. GPU check queued.", "warn")
            elif passed:
                self.voice_card.set_status("PASSED", GREEN)
            else:
                self.voice_card.set_status("FAILED", RED)
            self.voice_card.run_btn.config(state="normal")
        self.root.after(0, _update)

    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()
        self._loop.call_soon_threadsafe(self._loop.stop)


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
