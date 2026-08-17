from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any


class SimulationState:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.running = False
        self.starting = False
        self.simulation_id: str | None = None
        self.mode: str | None = None
        self.frame_number = 0
        self.latest: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=500)
        self.current_image_path: Path | None = None
        self.last_error: str | None = None

    def reset(self):
        with self.lock:
            self.stop_event.clear()
            self.frame_number = 0
            self.latest = None
            self.history.clear()
            self.current_image_path = None
            self.last_error = None
