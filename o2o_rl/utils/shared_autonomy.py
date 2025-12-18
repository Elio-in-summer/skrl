from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any, Optional

import torch


class ModeController:
    """Keyboard-based mode switcher (autonomous vs external control)."""

    def __init__(self) -> None:
        self.mode = "autonomous"
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while True:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                break
            if not ch:
                break
            self._queue.put(ch.lower())

    def poll(self) -> bool:
        changed = False
        while True:
            try:
                key = self._queue.get_nowait()
            except queue.Empty:
                break
            if key == "w" and self.mode != "external":
                self.mode = "external"
                changed = True
            elif key == "s" and self.mode != "autonomous":
                self.mode = "autonomous"
                changed = True
        return changed


class SharedAutonomyController:
    """Generic shared-autonomy loop that mediates between human input and the trainer."""

    def __init__(
        self,
        trainer,
        listener: Any,
        processor: Any,
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self.trainer = trainer
        self.listener = listener
        self.processor = processor
        self.mode_controller = ModeController()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.last_action: Optional[torch.Tensor] = None
        self._poll_interval = poll_interval

        if not getattr(self.listener, "ready", False):
            print("[WARN]: Shared autonomy listener not ready; external takeover disabled")

    def start(self) -> None:
        self.thread.start()
        self._print_mode()

    def stop(self) -> None:
        self.stop_event.set()
        if hasattr(self.listener, "stop"):
            try:
                self.listener.stop()
            except Exception:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        print()

    def _print_mode(self) -> None:
        label = "External" if self.mode_controller.mode == "external" else "Autonomous"
        print(f"\r[MODE] {label:<20}", end="", flush=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if self.mode_controller.poll():
                self._print_mode()
                if self.mode_controller.mode == "autonomous":
                    self.trainer.end_human_intervene()

            if self.mode_controller.mode == "external" and getattr(self.listener, "ready", False):
                payload = self.listener.get_latest()
                tensor = self.processor.convert(payload)
                if tensor is not None:
                    self.last_action = tensor
                    self.trainer.update_external_action(tensor)
                    self.trainer.activate_human_intervene()
                elif self.last_action is not None:
                    self.trainer.update_external_action(self.last_action)
                    self.trainer.activate_human_intervene()
                else:
                    self.trainer.end_human_intervene()
            else:
                self.trainer.end_human_intervene()

            time.sleep(self._poll_interval)

