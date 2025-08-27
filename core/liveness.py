from __future__ import annotations
from collections import deque
import time

class BlinkLiveness:
    """Valida liveness por piscadas (EAR abaixo do limiar por N frames).
    Conta blinks em uma janela móvel e exige um mínimo de eventos.
    """
    def __init__(self, ear_thresh: float, consec: int, window_sec: int, required: int):
        self.ear_thresh = ear_thresh
        self.consec = consec
        self.window_sec = window_sec
        self.required = required
        self._below = 0
        self._events = deque()

    def update(self, ear: float, now: float | None = None):
        now = now or time.time()
        if ear < self.ear_thresh:
            self._below += 1
        else:
            if self._below >= self.consec:
                self._events.append(now)
            self._below = 0
        while self._events and now - self._events[0] > self.window_sec:
            self._events.popleft()

    def ok(self) -> bool:
        return len(self._events) >= self.required

    def count(self) -> int:
        return len(self._events)
